use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nixe_loader_storage::{FileStorage, FormatLoader, LoadError, StorageError, StorageRef};
use sha2::{Digest, Sha256};

use crate::ncz::{reblock_solid, verify_reblocked};
use crate::{NczCompressionKind, NszLoader};

const COPY_BUFFER_SIZE: usize = 1024 * 1024;

/// Converts solid NCZ entries in an NSZ to independent 1 MiB blocks.
///
/// Returns `false` for an already suitable package. With `keep_backup`, the
/// original is kept at `<path>.solid`; existing backups are never overwritten. Listing titles does
/// not call this function. Anonymous, on-demand NCA caches remain unchanged.
pub fn prepare_nsz_for_random_access(
    path: impl AsRef<Path>,
    keep_backup: bool,
) -> Result<bool, LoadError> {
    let path = fs::canonicalize(path.as_ref()).map_err(StorageError::from)?;
    // The lock is released on every exit, including errors and process death.
    // Reopen the path after acquiring it: a previous converter may have already
    // replaced the file whose inode we just locked.
    let source_lock = File::open(&path).map_err(StorageError::from)?;
    source_lock.lock().map_err(StorageError::from)?;
    let source: StorageRef = Arc::new(FileStorage::open(&path)?);
    let archive = NszLoader::load(source)?;
    if !archive.entries().iter().any(|entry| {
        entry
            .ncz()
            .is_some_and(|ncz| ncz.compression_kind() == NczCompressionKind::Solid)
    }) {
        return Ok(false);
    }
    let mut backup_name = path.as_os_str().to_os_string();
    backup_name.push(".solid");
    let backup = std::path::PathBuf::from(backup_name);
    if keep_backup && backup.try_exists().map_err(StorageError::from)? {
        return Err(LoadError::invalid(
            "NSZ",
            format!(
                "cannot convert {}: backup {} already exists; move the backup before retrying",
                path.display(),
                backup.display()
            ),
        ));
    }
    let directory = path.parent().expect("canonical path has a parent");
    let mut temporary = tempfile::NamedTempFile::new_in(directory).map_err(StorageError::from)?;
    log::info!("converting solid NSZ to 1 MiB blocks: {}", path.display());
    let started = Instant::now();
    let mut metadata = pfs0_metadata(&archive)?;
    temporary.write_all(&metadata).map_err(StorageError::from)?;
    let data_offset = metadata.len() as u64;
    let mut expected = Vec::with_capacity(archive.entries().len());
    let mut copy_buffer = vec![0; COPY_BUFFER_SIZE];
    for (index, (stored, entry)) in archive
        .pfs0
        .entries()
        .iter()
        .zip(archive.entries())
        .enumerate()
    {
        let input = archive.pfs0.open_entry(stored)?;
        let begin = temporary.stream_position().map_err(StorageError::from)?;
        let digest = if let Some(ncz) = entry
            .ncz()
            .filter(|ncz| ncz.compression_kind() == NczCompressionKind::Solid)
        {
            let mut progress = Progress::new(
                "converting",
                entry.stored_name(),
                ncz.logical_size() - 0x4000,
            );
            let hash = reblock_solid(input, ncz, temporary.as_file_mut(), &mut |bytes| {
                progress.update(bytes)
            })?;
            Expected::Reblocked(hash)
        } else {
            Expected::Copied(copy_entry(
                input,
                temporary.as_file_mut(),
                &mut copy_buffer,
            )?)
        };
        let end = temporary.stream_position().map_err(StorageError::from)?;
        let slot = 0x10 + index * 0x18;
        metadata[slot..slot + 8].copy_from_slice(&(begin - data_offset).to_le_bytes());
        metadata[slot + 8..slot + 16].copy_from_slice(&(end - begin).to_le_bytes());
        expected.push(digest);
    }
    temporary
        .seek(SeekFrom::Start(0))
        .map_err(StorageError::from)?;
    temporary.write_all(&metadata).map_err(StorageError::from)?;
    let converted = NszLoader::load(Arc::new(FileStorage::open(temporary.path())?))?;
    log::info!("verifying converted NSZ: {}", path.display());
    for ((stored, entry), expected) in converted
        .pfs0
        .entries()
        .iter()
        .zip(converted.entries())
        .zip(expected)
    {
        let input = converted.pfs0.open_entry(stored)?;
        match expected {
            Expected::Reblocked(hash) => {
                let ncz = entry
                    .ncz()
                    .ok_or_else(|| LoadError::invalid("NSZ", "converted NCZ entry is missing"))?;
                let mut progress = Progress::new(
                    "verifying",
                    entry.stored_name(),
                    ncz.logical_size() - 0x4000,
                );
                verify_reblocked(input, ncz, hash, &mut |bytes| progress.update(bytes))?;
            }
            Expected::Copied(hash) => {
                if hash_entry(input, &mut copy_buffer)? != hash {
                    return Err(LoadError::invalid(
                        "NSZ",
                        format!(
                            "copied entry {:?} does not match the original",
                            entry.stored_name()
                        ),
                    ));
                }
            }
        }
    }
    drop(converted);
    drop(archive);
    temporary
        .as_file()
        .set_permissions(
            source_lock
                .metadata()
                .map_err(StorageError::from)?
                .permissions(),
        )
        .map_err(StorageError::from)?;
    temporary.as_file().sync_all().map_err(StorageError::from)?;
    // Linking the original to its backup before the atomic replacement avoids
    // a crash window with no .nsz file. hard_link also refuses existing backups.
    if keep_backup {
        fs::hard_link(&path, &backup).map_err(StorageError::from)?;
        sync_directory(directory)?;
    }
    temporary
        .persist(&path)
        .map_err(|error| StorageError::from(error.error))?;
    sync_directory(directory)?;
    if keep_backup {
        log::info!(
            "NSZ conversion finished in {:.1}s; original preserved at {}",
            started.elapsed().as_secs_f64(),
            backup.display()
        );
    } else {
        log::info!(
            "NSZ conversion finished in {:.1}s; original replaced without a backup",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(true)
}

enum Expected {
    Reblocked([u8; 32]),
    Copied([u8; 32]),
}

fn pfs0_metadata(archive: &super::NszArchive) -> Result<Vec<u8>, LoadError> {
    let mut strings = Vec::new();
    let mut slots = vec![0_u8; archive.entries().len() * 0x18];
    for (slot, entry) in slots.chunks_exact_mut(0x18).zip(archive.entries()) {
        let offset = u32::try_from(strings.len())
            .map_err(|_| LoadError::invalid("NSZ", "converted string table is too large"))?;
        slot[16..20].copy_from_slice(&offset.to_le_bytes());
        strings.extend_from_slice(entry.stored_name().as_bytes());
        strings.push(0);
    }
    // Keep the data area aligned without changing any entry's stored name.
    let metadata_size = (0x10 + slots.len() + strings.len()).next_multiple_of(0x10);
    if metadata_size > 64 * 1024 * 1024 {
        return Err(LoadError::invalid(
            "NSZ",
            "converted metadata exceeds the safety limit",
        ));
    }
    let mut metadata = Vec::with_capacity(metadata_size);
    metadata.extend_from_slice(b"PFS0");
    metadata.extend_from_slice(&(archive.entries().len() as u32).to_le_bytes());
    metadata.extend_from_slice(&((metadata_size - 0x10 - slots.len()) as u32).to_le_bytes());
    metadata.extend_from_slice(&0_u32.to_le_bytes());
    metadata.extend_from_slice(&slots);
    metadata.extend_from_slice(&strings);
    metadata.resize(metadata_size, 0);
    Ok(metadata)
}

fn copy_entry(
    source: StorageRef,
    output: &mut File,
    buffer: &mut [u8],
) -> Result<[u8; 32], LoadError> {
    let mut digest = Sha256::new();
    let size = source.len()?;
    let mut offset = 0;
    while offset < size {
        let count = (size - offset).min(buffer.len() as u64) as usize;
        source.read_at(offset, &mut buffer[..count])?;
        output
            .write_all(&buffer[..count])
            .map_err(StorageError::from)?;
        digest.update(&buffer[..count]);
        offset += count as u64;
    }
    Ok(digest.finalize().into())
}

fn hash_entry(source: StorageRef, buffer: &mut [u8]) -> Result<[u8; 32], LoadError> {
    let mut digest = Sha256::new();
    let size = source.len()?;
    let mut offset = 0;
    while offset < size {
        let count = (size - offset).min(buffer.len() as u64) as usize;
        source.read_at(offset, &mut buffer[..count])?;
        digest.update(&buffer[..count]);
        offset += count as u64;
    }
    Ok(digest.finalize().into())
}

fn sync_directory(path: &Path) -> Result<(), StorageError> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

struct Progress<'a> {
    phase: &'static str,
    name: &'a str,
    total: u64,
    last: Instant,
}

impl<'a> Progress<'a> {
    fn new(phase: &'static str, name: &'a str, total: u64) -> Self {
        log::info!(
            "{phase} NCZ entry {name}: {} MiB",
            total.div_ceil(1024 * 1024)
        );
        Self {
            phase,
            name,
            total,
            last: Instant::now(),
        }
    }

    fn update(&mut self, bytes: u64) {
        if bytes == self.total || self.last.elapsed() >= Duration::from_secs(2) {
            log::info!(
                "{} {}: {:.1}%",
                self.phase,
                self.name,
                bytes as f64 * 100.0 / self.total as f64
            );
            self.last = Instant::now();
        }
    }
}
