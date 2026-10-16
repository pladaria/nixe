use std::collections::HashSet;

use nixe_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::compressed_package::logical_nca_name;
use crate::{CompressedPackageEntry, NczLoader, Pfs0Archive};

mod reblock;
pub use reblock::prepare_nsz_for_random_access;

/// Loads an NSZ by reusing the PFS0 parser and wrapping its NCZ entries.
#[derive(Debug)]
pub struct NszLoader;

impl FormatLoader for NszLoader {
    type Output = NszArchive;

    const FORMAT_NAME: &'static str = "NSZ";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        let pfs0 = Pfs0Archive::parse_nsz(storage, Self::FORMAT_NAME)?;
        let mut entries = Vec::with_capacity(pfs0.entries().len());
        let mut logical_names = HashSet::with_capacity(pfs0.entries().len());
        for stored in pfs0.entries() {
            let stored_storage = pfs0.open_entry(stored)?;
            let entry = match logical_nca_name(stored.name()) {
                Some(logical_name) => {
                    let ncz = NczLoader::load(stored_storage).map_err(|error| {
                        LoadError::invalid(
                            Self::FORMAT_NAME,
                            format!("entry {:?}: {error}", stored.name()),
                        )
                    })?;
                    CompressedPackageEntry::compressed(
                        stored.name().to_owned(),
                        logical_name,
                        stored.offset(),
                        stored.size(),
                        ncz,
                    )
                }
                None => CompressedPackageEntry::ordinary(
                    stored.name().to_owned(),
                    stored.offset(),
                    stored.size(),
                    stored_storage,
                ),
            };
            if !logical_names.insert(entry.logical_name().to_ascii_lowercase()) {
                return Err(LoadError::invalid(
                    Self::FORMAT_NAME,
                    format!(
                        "logical entry name {:?} is duplicated or ambiguous",
                        entry.logical_name()
                    ),
                ));
            }
            entries.push(entry);
        }
        Ok(NszArchive { pfs0, entries })
    }
}

/// Logical package view of an NSZ.
#[derive(Debug)]
pub struct NszArchive {
    pfs0: Pfs0Archive,
    entries: Vec<CompressedPackageEntry>,
}

impl NszArchive {
    pub fn entries(&self) -> &[CompressedPackageEntry] {
        &self.entries
    }

    pub fn entry(&self, logical_name: &str) -> Option<&CompressedPackageEntry> {
        self.entries
            .iter()
            .find(|entry| entry.logical_name() == logical_name)
    }

    pub fn open_entry(&self, entry: &CompressedPackageEntry) -> Result<StorageRef, LoadError> {
        entry.open()
    }

    pub fn open(&self, logical_name: &str) -> Result<Option<StorageRef>, LoadError> {
        self.entry(logical_name)
            .map(CompressedPackageEntry::open)
            .transpose()
    }

    pub const fn data_offset(&self) -> u64 {
        self.pfs0.data_offset()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nixe_loader_storage::{Storage, StorageError};

    use super::*;

    #[derive(Debug)]
    struct Bytes(Vec<u8>);

    impl Storage for Bytes {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.0.len() as u64)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
            Ok(())
        }
    }

    fn ncz(tail: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0x33; 0x4000];
        bytes.extend_from_slice(b"NCZSECTN");
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0x4000_u64.to_le_bytes());
        bytes.extend_from_slice(&(tail.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(&zstd::stream::encode_all(tail, 1).unwrap());
        bytes
    }

    fn pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(strings.len() as u32);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PFS0");
        bytes.extend_from_slice(&(files.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        let mut offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(name_offsets) {
            bytes.extend_from_slice(&offset.to_le_bytes());
            bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&name_offset.to_le_bytes());
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            offset += data.len() as u64;
        }
        bytes.extend_from_slice(&strings);
        for (_, data) in files {
            bytes.extend_from_slice(data);
        }
        bytes
    }

    #[test]
    fn exposes_mixed_entries_with_logical_names_and_sizes() {
        let tail = b"logical tail";
        let compressed = ncz(tail);
        let package = pfs0(&[("content.NCZ", &compressed), ("ticket.tik", b"ticket")]);
        let archive = NszLoader::load(Arc::new(Bytes(package))).unwrap();
        assert_eq!(archive.entries()[0].stored_name(), "content.NCZ");
        assert_eq!(archive.entries()[0].logical_name(), "content.nca");
        assert_eq!(archive.entries()[0].stored_size(), compressed.len() as u64);
        assert_eq!(
            archive.entries()[0].logical_size(),
            0x4000 + tail.len() as u64
        );
        assert!(archive.entries()[0].ncz().is_some());
        assert!(archive.entries()[1].ncz().is_none());

        let storage = archive.open("content.nca").unwrap().unwrap();
        let mut actual = [0_u8; 12];
        storage.read_at(0x4000, &mut actual).unwrap();
        assert_eq!(&actual, tail);
    }

    #[test]
    fn rejects_ambiguous_logical_names() {
        let compressed = ncz(b"tail");
        let package = pfs0(&[("same.nca", b"ordinary"), ("same.ncz", &compressed)]);
        assert!(NszLoader::load(Arc::new(Bytes(package))).is_err());
    }

    #[test]
    fn accepts_final_name_terminated_in_alignment_padding() {
        let compressed = ncz(b"tail");
        let mut package = pfs0(&[("content.ncz", &compressed)]);
        package[8..12].copy_from_slice(&11_u32.to_le_bytes());
        package[0x10..0x18].copy_from_slice(&1_u64.to_le_bytes());

        let archive = NszLoader::load(Arc::new(Bytes(package))).unwrap();
        assert_eq!(archive.entries()[0].logical_name(), "content.nca");
    }

    #[test]
    fn converts_solid_entries_preserves_other_entries_and_keeps_the_original() {
        use nixe_loader_storage::FileStorage;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("game.NSZ");
        let backup = directory.path().join("game.NSZ.solid");
        let tail = vec![0x5a; 2 * 1024 * 1024 + 19];
        let solid = ncz(&tail);
        let original = pfs0(&[
            ("content.NCZ", &solid),
            ("ticket.tik", b"ticket"),
            ("empty", b""),
        ]);
        std::fs::write(&path, &original).unwrap();
        assert!(prepare_nsz_for_random_access(&path, true).unwrap());
        assert_eq!(std::fs::read(&backup).unwrap(), original);
        let archive = NszLoader::load(Arc::new(FileStorage::open(&path).unwrap())).unwrap();
        let info = archive.entries()[0].ncz().unwrap().block_info().unwrap();
        assert_eq!(info.block_size(), 1024 * 1024);
        assert_eq!(info.block_count(), 3);
        let mut output = vec![0; tail.len()];
        archive
            .open("content.nca")
            .unwrap()
            .unwrap()
            .read_at(0x4000, &mut output)
            .unwrap();
        assert_eq!(output, tail);
        let mut ticket = [0; 6];
        archive
            .open("ticket.tik")
            .unwrap()
            .unwrap()
            .read_at(0, &mut ticket)
            .unwrap();
        assert_eq!(&ticket, b"ticket");
        assert!(archive.open("empty").unwrap().unwrap().is_empty().unwrap());
        let converted = std::fs::read(&path).unwrap();
        assert!(!prepare_nsz_for_random_access(&path, true).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), converted);
        assert_eq!(std::fs::read(&backup).unwrap(), original);

        // A package can mix already blocked and solid entries. Re-conversion
        // must copy the blocked entry byte-for-byte.
        let blocked = archive.pfs0.open("content.NCZ").unwrap().unwrap();
        let mut blocked_bytes = vec![0; blocked.len().unwrap() as usize];
        blocked.read_at(0, &mut blocked_bytes).unwrap();
        let mixed_path = directory.path().join("mixed.nsz");
        let mixed = pfs0(&[("blocked.ncz", &blocked_bytes), ("solid.ncz", &solid)]);
        std::fs::write(&mixed_path, mixed).unwrap();
        assert!(prepare_nsz_for_random_access(&mixed_path, false).unwrap());
        assert!(!directory.path().join("mixed.nsz.solid").exists());
        let mixed = NszLoader::load(Arc::new(FileStorage::open(&mixed_path).unwrap())).unwrap();
        let mut copied = vec![0; blocked_bytes.len()];
        mixed
            .pfs0
            .open("blocked.ncz")
            .unwrap()
            .unwrap()
            .read_at(0, &mut copied)
            .unwrap();
        assert_eq!(copied, blocked_bytes);
        assert!(mixed.entries().iter().all(
            |entry| entry.ncz().unwrap().compression_kind() == crate::NczCompressionKind::Block
        ));
    }

    #[test]
    fn conversion_failure_leaves_the_original_and_existing_backups_untouched() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("game.nsz");
        let backup = directory.path().join("game.nsz.solid");
        let original = pfs0(&[("content.ncz", &ncz(b"tail"))]);
        std::fs::write(&path, &original).unwrap();
        std::fs::write(&backup, b"existing backup").unwrap();
        assert!(
            prepare_nsz_for_random_access(&path, true)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert_eq!(std::fs::read(&backup).unwrap(), b"existing backup");
        std::fs::remove_file(&backup).unwrap();
        let solid = ncz(&vec![7; 20_000]);
        for malformed in [
            &solid[..solid.len() - 5],
            &[solid.as_slice(), b"trailing"].concat(),
        ] {
            let original = pfs0(&[("content.ncz", malformed)]);
            std::fs::write(&path, &original).unwrap();
            assert!(prepare_nsz_for_random_access(&path, true).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), original);
            assert!(!backup.exists());
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        }
    }

    #[test]
    fn concurrent_converters_publish_one_verified_package_and_one_backup() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("game.nsz");
        let original = pfs0(&[("content.ncz", &ncz(&vec![3; 1024 * 1024 + 7]))]);
        std::fs::write(&path, &original).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let threads = (0..4)
            .map(|_| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    prepare_nsz_for_random_access(path, true).unwrap()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            threads
                .into_iter()
                .filter_map(|thread| thread.join().unwrap().then_some(()))
                .count(),
            1
        );
        assert_eq!(
            std::fs::read(directory.path().join("game.nsz.solid")).unwrap(),
            original
        );
        assert!(!prepare_nsz_for_random_access(path, true).unwrap());
    }
}
