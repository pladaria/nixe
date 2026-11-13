//! Save-data access by application/user identity and selected creation sizes.

use nixe_runtime::TransactionalDirectory;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use nixe_loader_title::ApplicationControlProperty;

static CREATION_SEQUENCE: AtomicU64 = AtomicU64::new(1);
type OpenVolumes = BTreeMap<(u64, String, u128), Weak<Mutex<TransactionalDirectory>>>;

const METADATA_MAGIC: &[u8; 8] = b"NIXESAV1";

/// Persistent save-data access with the launch application's creation settings.
#[derive(Clone, Debug)]
pub struct SaveDataSystem {
    root: PathBuf,
    program_id: u64,
    account_size: i64,
    account_journal: i64,
    device_size: i64,
    device_journal: i64,
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
    /// Uses NACP declarations when available, otherwise explicit fallback sizes
    /// for account/device saves. Sizes are logical capacities, not preallocation.
    #[must_use]
    pub fn new(root: PathBuf, program_id: u64, nacp: Option<&ApplicationControlProperty>) -> Self {
        let (account_size, account_journal, device_size, device_journal) = match nacp {
            Some(nacp) => (
                nacp.user_account_save_data_size,
                nacp.user_account_save_data_journal_size,
                nacp.device_save_data_size,
                nacp.device_save_data_journal_size,
            ),
            None => {
                // @SPECULATIVE: without control.nacp, save requirements are
                // unknown. Allow account/device saves with 3.75 GiB for data
                // and journal each; these are emulator policy, not title metadata.
                // Existing volumes keep their persisted sizes and contents.
                // Missing-metadata size policy reference:
                // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/filesystem/save_data_controller.cpp
                const FALLBACK_SIZE: i64 = 0xF000_0000;
                log::warn!(
                    "application save-data fallback: account/device data and journal sizes are each {FALLBACK_SIZE} bytes (logical capacity, no preallocation); reason: control.nacp is missing and the title's save requirements are unknown"
                );
                (FALLBACK_SIZE, FALLBACK_SIZE, FALLBACK_SIZE, FALLBACK_SIZE)
            }
        };
        Self {
            root,
            program_id,
            account_size,
            account_journal,
            device_size,
            device_journal,
            volumes: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    fn path(&self, program_id: u64, kind: &str, user: u128) -> PathBuf {
        self.root
            .join(format!("{program_id:016x}"))
            .join(kind)
            .join(format!("{user:032x}"))
    }

    pub(crate) fn bcat_directory(&self) -> PathBuf {
        self.root
            .join(format!("{:016x}", self.program_id))
            .join("bcat")
    }

    /// Prepares account/device saves using the selected creation sizes.
    /// The system launcher also ensures application saves; applications need
    /// not issue IApplicationFunctions::EnsureSaveData before mounting them.
    /// https://switchbrew.org/wiki/AM_services#EnsureSaveData
    pub fn prepare_for_launch(&self) -> io::Result<()> {
        let user = u128::from_le_bytes(crate::user::DEFAULT_USER.id().encode());
        self.ensure_user_volumes(user).map_err(|error| match error {
            SaveDataError::Unsupported(detail) => {
                io::Error::new(io::ErrorKind::Unsupported, detail)
            }
            SaveDataError::Io(error) => error,
        })
    }

    pub(crate) fn open(
        &self,
        program_id: u64,
        kind: &str,
        user: u128,
    ) -> io::Result<Arc<Mutex<TransactionalDirectory>>> {
        // Zero refers to the caller's application; an explicit ID selects that
        // application's existing save. Opening never creates a missing save.
        // https://switchbrew.org/wiki/Filesystem_services#SaveDataAttribute
        let program_id = if program_id == 0 {
            self.program_id
        } else {
            program_id
        };
        let key = (program_id, kind.to_owned(), user);
        let mut volumes = self
            .volumes
            .lock()
            .map_err(|_| io::Error::other("save-data namespace lock poisoned"))?;
        if let Some(volume) = volumes.get(&key).and_then(Weak::upgrade) {
            return Ok(volume);
        }
        let path = self.path(program_id, kind, user);
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

    // EnsureSaveData creates the account save for the requested user, independently
    // of BCAT/cache declarations. It preserves existing contents and returns
    // success with zero additional space only after ensuring the volume exists.
    // https://switchbrew.org/wiki/AM_services#EnsureSaveData
    pub(crate) fn ensure(&self, user: u128) -> Result<u64, SaveDataError> {
        if self.account_size < 0 || self.account_journal < 0 {
            return Err(SaveDataError::Unsupported("negative NACP save-data size"));
        }
        ensure_volume(
            &self.path(self.program_id, "account", user),
            self.account_size as u64,
            self.account_journal as u64,
        )?;
        Ok(0)
    }

    fn ensure_user_volumes(&self, user: u128) -> Result<(), SaveDataError> {
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
            ensure_volume(
                &self.path(self.program_id, kind, uid),
                size as u64,
                journal as u64,
            )?;
        }
        Ok(())
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
