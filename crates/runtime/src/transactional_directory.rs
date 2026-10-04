//! Host directory snapshots published atomically at explicit commit boundaries.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// One private working directory and its durable committed generation.
/// Unchanged files are hard-linked between immutable generations; modified
/// files are copied. Open file handles always refer to the working directory.
#[derive(Debug)]
pub struct TransactionalDirectory {
    base: PathBuf,
    working: PathBuf,
    committed: PathBuf,
    dirty: BTreeSet<PathBuf>,
    capacity: u64,
    used: u64,
    _lock: File,
    writers: Arc<AtomicUsize>,
}

impl TransactionalDirectory {
    pub fn open(base: PathBuf, capacity: u64) -> io::Result<Self> {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(base.join("lock"))?;
        lock.try_lock().map_err(io::Error::from)?;
        let name = match fs::read_to_string(base.join("current")) {
            Ok(name)
                if name.starts_with("generation-")
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase() || c == b'-') =>
            {
                name
            }
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid directory generation",
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => "data".to_owned(),
            Err(error) => return Err(error),
        };
        let committed = base.join(name);
        let working = base.join(unique_name("working"));
        fs::create_dir(&working)?;
        let used = match copy_tree(&committed, &working, None) {
            Ok(used) => used,
            Err(error) => {
                let _ = fs::remove_dir_all(&working);
                return Err(error);
            }
        };
        Ok(Self {
            base,
            working,
            committed,
            dirty: BTreeSet::new(),
            capacity,
            used,
            _lock: lock,
            writers: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[must_use]
    pub fn working_directory(&self) -> &Path {
        &self.working
    }

    #[must_use]
    pub fn can_resize(&self, old: u64, new: u64) -> bool {
        self.used
            .checked_sub(old)
            .and_then(|used| used.checked_add(new))
            .is_some_and(|used| used <= self.capacity)
    }

    pub fn record_resize(&mut self, path: &str, old: u64, new: u64) {
        self.used = self.used - old + new;
        self.mark_changed(path);
    }

    pub fn mark_changed(&mut self, path: &str) {
        self.dirty
            .insert(PathBuf::from(path.trim_start_matches('/')));
    }

    pub fn open_writer(&self) -> DirectoryWriteLease {
        self.writers.fetch_add(1, Ordering::Relaxed);
        DirectoryWriteLease(Arc::new(WriterLease {
            writers: self.writers.clone(),
        }))
    }

    pub fn has_open_writers(&self) -> bool {
        self.writers.load(Ordering::Relaxed) != 0
    }

    pub fn commit(&mut self) -> io::Result<()> {
        if self.has_open_writers() {
            return Err(io::Error::new(
                io::ErrorKind::ResourceBusy,
                "writable files remain open",
            ));
        }
        let name = unique_name("generation");
        let next = self.base.join(&name);
        fs::create_dir(&next)?;
        if let Err(error) = copy_tree(&self.working, &next, Some((&self.committed, &self.dirty))) {
            let _ = fs::remove_dir_all(next);
            return Err(error);
        }
        let marker = self.base.join(unique_name("publishing"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker)?;
        file.write_all(name.as_bytes())?;
        file.sync_all()?;
        fs::rename(&marker, self.base.join("current"))?;
        sync_directory(&self.base)?;
        let previous = std::mem::replace(&mut self.committed, next);
        self.dirty.clear();
        // The new committed generation is already durable. Reclamation cannot
        // undo that commit; failure only leaves an unused generation on disk.
        let _ = fs::remove_dir_all(previous);
        Ok(())
    }
}

/// Shared lifetime of one writable file, including cloned handles.
#[derive(Clone, Debug)]
pub struct DirectoryWriteLease(#[allow(dead_code)] Arc<WriterLease>);
#[derive(Debug)]
struct WriterLease {
    writers: Arc<AtomicUsize>,
}
impl Drop for WriterLease {
    fn drop(&mut self) {
        self.writers.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Drop for TransactionalDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.working);
    }
}

fn unique_name(prefix: &str) -> String {
    format!(
        "{prefix}-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

fn copy_tree(
    source: &Path,
    target: &Path,
    reuse: Option<(&Path, &BTreeSet<PathBuf>)>,
) -> io::Result<u64> {
    fn copy(
        source: &Path,
        target: &Path,
        relative: &Path,
        reuse: Option<(&Path, &BTreeSet<PathBuf>)>,
    ) -> io::Result<u64> {
        let mut size = 0_u64;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            let path = relative.join(entry.file_name());
            let destination = target.join(entry.file_name());
            if kind.is_dir() {
                fs::create_dir(&destination)?;
                size = size
                    .checked_add(copy(&entry.path(), &destination, &path, reuse)?)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "directory size overflow")
                    })?;
            } else if kind.is_file() {
                let linked = if let Some((previous, dirty)) = reuse {
                    !dirty.contains(&path) && previous.join(&path).is_file()
                } else {
                    false
                };
                if linked {
                    fs::hard_link(reuse.unwrap().0.join(&path), &destination)?;
                } else {
                    fs::copy(entry.path(), &destination)?;
                    File::open(&destination)?.sync_all()?;
                }
                size = size.checked_add(entry.metadata()?.len()).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "directory size overflow")
                })?;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "transactional directories do not expose links or special files",
                ));
            }
        }
        sync_directory(target)?;
        Ok(size)
    }
    copy(source, target, Path::new(""), reuse)
}

pub(crate) fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_rejects_writers_until_the_last_clone_is_closed() {
        let base = tempfile::tempdir().unwrap();
        fs::create_dir(base.path().join("data")).unwrap();
        let mut volume = TransactionalDirectory::open(base.path().to_owned(), 4096).unwrap();
        let writer = volume.open_writer();
        let cloned = writer.clone();
        assert_eq!(
            volume.commit().unwrap_err().kind(),
            io::ErrorKind::ResourceBusy
        );
        drop(writer);
        assert!(volume.has_open_writers());
        drop(cloned);
        volume.commit().unwrap();
    }

    #[test]
    fn uncommitted_changes_are_discarded_and_commits_preserve_other_files() {
        let base = tempfile::tempdir().unwrap();
        fs::create_dir(base.path().join("data")).unwrap();
        fs::write(base.path().join("data/unchanged"), b"original").unwrap();
        {
            let mut volume = TransactionalDirectory::open(base.path().to_owned(), 4096).unwrap();
            fs::write(volume.working_directory().join("new"), b"committed").unwrap();
            volume.record_resize("/new", 0, 9);
            volume.commit().unwrap();
            fs::write(volume.working_directory().join("new"), b"uncommitted").unwrap();
        }
        let volume = TransactionalDirectory::open(base.path().to_owned(), 4096).unwrap();
        assert_eq!(
            fs::read(volume.working_directory().join("new")).unwrap(),
            b"committed"
        );
        assert_eq!(
            fs::read(volume.working_directory().join("unchanged")).unwrap(),
            b"original"
        );
        assert!(TransactionalDirectory::open(base.path().to_owned(), 4096).is_err());
        assert!(!volume.can_resize(9, 4096));
    }
}
