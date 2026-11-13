//! Host-generated, read-only system data archives exposed through fsp-srv.
mod ng_word;

#[cfg(test)]
use nixe_loader_storage::StorageError;
use nixe_loader_storage::{BytesStorage, StorageRef};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

pub const NG_WORD_2_ID: u64 = 0x0100_0000_0000_0823;

/// Per-run system resources. A dictionary edit is picked up by the next launch;
/// existing IStorage objects retain an immutable image for their whole lifetime.
#[derive(Default)]
pub struct SystemArchives {
    dictionary: Option<PathBuf>,
    ng_word: OnceLock<Result<StorageRef, String>>,
}
impl std::fmt::Debug for SystemArchives {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemArchives")
            .field("dictionary", &self.dictionary)
            .field("loaded", &self.ng_word.get().is_some())
            .finish()
    }
}
impl SystemArchives {
    pub fn new(dictionary: Option<PathBuf>) -> Self {
        Self {
            dictionary,
            ng_word: OnceLock::new(),
        }
    }

    /// Returns a generated resource, or None for an archive we do not implement.
    pub fn open(&self, title_id: u64) -> Result<Option<StorageRef>, String> {
        if title_id != NG_WORD_2_ID {
            return Ok(None);
        }
        let image = self
            .ng_word
            .get_or_init(|| {
                let path = self.dictionary.as_ref().ok_or(
                    "NgWord2 requires system.dictionary pointing to a word-list TOML file",
                )?;
                let source = std::fs::read_to_string(path)
                    .map_err(|e| format!("dictionary {}: {e}", path.display()))?;
                let files = ng_word::files(&source)
                    .map_err(|e| format!("dictionary {}: {e}", path.display()))?;
                let mut tree = nixe_loader_content::ReadOnlyTree::default();
                for (name, bytes) in files {
                    tree.insert_file(&format!("/{name}"), Arc::new(BytesStorage(bytes.into())))
                        .map_err(|error| error.to_string())?;
                }
                let image = nixe_loader_content::RomFsArchive::from_tree(tree)
                    .map_err(|error| error.to_string())?
                    .storage();
                log::info!(
                    "generated NgWord2 system archive from {}: {} bytes",
                    path.display(),
                    image.len().map_err(|error| error.to_string())?
                );
                Ok(image)
            })
            .as_ref()
            .map_err(Clone::clone)?
            .clone();
        Ok(Some(image))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use nixe_loader_storage::FormatLoader;
    #[test]
    fn generated_archive_is_a_readable_romfs_and_retains_its_dictionary_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dict.toml");
        std::fs::write(&path, "[categories.test]\nwords = [\"bad\", \"rude\"]").unwrap();
        let archives = SystemArchives::new(Some(path.clone()));
        assert!(archives.open(0x999).unwrap().is_none());
        let storage = archives.open(NG_WORD_2_ID).unwrap().unwrap();
        std::fs::remove_file(path).unwrap();
        let romfs = nixe_loader_content::RomFsLoader::load(storage.clone()).unwrap();
        assert_eq!(romfs.files().len(), 52);
        assert!(romfs.file("/ac_common_not_b_nx").is_some());
        let second = archives.open(NG_WORD_2_ID).unwrap().unwrap();
        let mut a = vec![0; storage.len().unwrap() as usize];
        let mut b = vec![0; second.len().unwrap() as usize];
        storage.read_at(0, &mut a).unwrap();
        second.read_at(0, &mut b).unwrap();
        assert_eq!(a, b);
        assert!(matches!(
            storage.read_at(storage.len().unwrap(), &mut [0]),
            Err(StorageError::OutOfBounds)
        ));
    }
    #[test]
    fn missing_or_invalid_dictionary_is_an_actionable_error() {
        assert!(
            SystemArchives::default()
                .open(NG_WORD_2_ID)
                .err()
                .unwrap()
                .contains("system.dictionary")
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dict.toml");
        std::fs::write(&path, "[categories.test]\nwords = [\"\"]").unwrap();
        assert!(
            SystemArchives::new(Some(path))
                .open(NG_WORD_2_ID)
                .err()
                .unwrap()
                .contains("category test")
        );
    }
}
