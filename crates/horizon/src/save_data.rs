//! Application save-data identity and creation from NACP declarations.

use nixe_runtime::TransactionalDirectory;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use nixe_loader_title::ApplicationControlProperty;

static CREATION_SEQUENCE: AtomicU64 = AtomicU64::new(1);
type OpenVolumes = BTreeMap<(String, u128), Weak<Mutex<TransactionalDirectory>>>;

const METADATA_MAGIC: &[u8; 8] = b"NIXESAV1";

/// Persistent save-data namespace for one launched application.
#[derive(Clone, Debug)]
pub struct SaveDataSystem {
    root: PathBuf,
    program_id: u64,
    account_size: i64,
    account_journal: i64,
    device_size: i64,
    device_journal: i64,
    bcat_size: i64,
    cache_size: i64,
    volumes: Arc<Mutex<OpenVolumes>>,
}

#[derive(Debug)]
pub(crate) enum SaveDataError {
    Unsupported(&'static str),
    Io(io::Error),
}

impl From<io::Error> for SaveDataError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl SaveDataSystem {
    #[must_use]
    pub fn new(root: PathBuf, program_id: u64, nacp: &ApplicationControlProperty) -> Self {
        Self {
            root,
            program_id,
            account_size: nacp.user_account_save_data_size,
            account_journal: nacp.user_account_save_data_journal_size,
            device_size: nacp.device_save_data_size,
            device_journal: nacp.device_save_data_journal_size,
            bcat_size: nacp.bcat_delivery_cache_storage_size,
            cache_size: nacp.cache_storage_size,
            volumes: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub(crate) fn path(&self, kind: &str, user: u128) -> PathBuf {
        self.root
            .join(format!("{:016x}", self.program_id))
            .join(kind)
            .join(format!("{user:032x}"))
    }

    pub(crate) const fn program_id(&self) -> u64 {
        self.program_id
    }

    pub(crate) fn open(
        &self,
        kind: &str,
        user: u128,
    ) -> io::Result<Arc<Mutex<TransactionalDirectory>>> {
        let key = (kind.to_owned(), user);
        let mut volumes = self
            .volumes
            .lock()
            .map_err(|_| io::Error::other("save-data namespace lock poisoned"))?;
        if let Some(volume) = volumes.get(&key).and_then(Weak::upgrade) {
            return Ok(volume);
        }
        let path = self.path(kind, user);
        let metadata = fs::read(path.join("metadata"))?;
        if metadata.len() != 24 || &metadata[..8] != METADATA_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid save-data metadata",
            ));
        }
        let capacity = u64::from_le_bytes(metadata[8..16].try_into().unwrap());
        let volume = Arc::new(Mutex::new(TransactionalDirectory::open(path, capacity)?));
        volumes.insert(key, Arc::downgrade(&volume));
        Ok(volume)
    }

    // EnsureApplicationSaveData creates only types declared by the NACP, and
    // leaves existing saves intact. Success reports no additional space needed.
    // https://switchbrew.org/wiki/AM_services#EnsureSaveData
    // https://switchbrew.org/wiki/NACP
    pub(crate) fn ensure(&self, user: u128) -> Result<u64, SaveDataError> {
        if self.bcat_size != 0 || self.cache_size != 0 {
            return Err(SaveDataError::Unsupported(
                "BCAT/cache save creation is not implemented",
            ));
        }
        for (kind, uid, size, journal) in [
            ("account", user, self.account_size, self.account_journal),
            ("device", 0, self.device_size, self.device_journal),
        ] {
            if size < 0 || journal < 0 {
                return Err(SaveDataError::Unsupported("negative NACP save-data size"));
            }
            if size == 0 || (kind == "account" && uid == 0) {
                continue;
            }
            ensure_volume(&self.path(kind, uid), size as u64, journal as u64)?;
        }
        Ok(0)
    }
}

fn ensure_volume(path: &Path, size: u64, journal: u64) -> io::Result<()> {
    match fs::read(path.join("metadata")) {
        Ok(bytes) => {
            if bytes.len() != 24 || &bytes[..8] != METADATA_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid save-data metadata",
                ));
            }
            // Ensure does not resize an existing save or overwrite its contents.
            return Ok(());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound && !path.exists() => {}
        Err(error) => return Err(error),
    }
    let parent = path.parent().expect("a save path has a namespace parent");
    fs::create_dir_all(parent)?;
    let sequence = CREATION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".creating-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let result = (|| {
        fs::create_dir(staging.join("data"))?;
        let mut metadata = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(staging.join("metadata"))?;
        metadata.write_all(METADATA_MAGIC)?;
        metadata.write_all(&size.to_le_bytes())?;
        metadata.write_all(&journal.to_le_bytes())?;
        metadata.sync_all()?;
        File::open(&staging)?.sync_all()?;
        fs::rename(&staging, path)?;
        File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_is_persistent_and_ensure_preserves_existing_contents_and_sizes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("save");
        ensure_volume(&path, 4096, 1024).unwrap();
        let before = fs::read(path.join("metadata")).unwrap();
        fs::write(path.join("data/progress"), b"existing progress").unwrap();
        ensure_volume(&path, 8192, 2048).unwrap();
        assert_eq!(fs::read(path.join("metadata")).unwrap(), before);
        assert_eq!(
            fs::read(path.join("data/progress")).unwrap(),
            b"existing progress"
        );
    }
}
