use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::Pfs0Entry;
use crate::pfs0::Pfs0Archive;

/// Loads Nintendo Submission Package (NSP) files.
///
/// NSP is the package commonly used for digitally distributed titles, updates,
/// and downloadable content. It groups NCAs and related installation metadata
/// in a package, unlike XCI, which models the partitions and metadata of a
/// physical game card.
#[derive(Debug)]
pub struct NspLoader;

impl FormatLoader for NspLoader {
    type Output = NspArchive;

    const FORMAT_NAME: &'static str = "NSP";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        Ok(NspArchive {
            pfs0: Pfs0Archive::parse(storage, Self::FORMAT_NAME)?,
        })
    }
}

/// Parsed view of an NSP package.
///
/// The archive retains the original storage and exposes each contained file as
/// a bounded storage view, so large NCA files are never copied into memory.
#[derive(Debug)]
pub struct NspArchive {
    pfs0: Pfs0Archive,
}

impl NspArchive {
    /// Returns package entries in their on-disk order.
    pub fn entries(&self) -> &[Pfs0Entry] {
        self.pfs0.entries()
    }

    /// Finds an entry by its exact file name.
    pub fn entry(&self, name: &str) -> Option<&Pfs0Entry> {
        self.pfs0.entry(name)
    }

    /// Opens an entry as an independent bounded storage view.
    pub fn open_entry(&self, entry: &Pfs0Entry) -> Result<StorageRef, LoadError> {
        self.pfs0.open_entry(entry)
    }

    /// Finds and opens an entry by name.
    pub fn open(&self, name: &str) -> Result<Option<StorageRef>, LoadError> {
        self.pfs0.open(name)
    }

    /// Returns the byte offset at which file data begins.
    pub const fn data_offset(&self) -> u64 {
        self.pfs0.data_offset()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_storage::{Storage, StorageError};

    use super::*;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            u64::try_from(self.0.len()).map_err(|_| StorageError::OutOfBounds)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            let source = self.0.get(start..end).ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(source);
            Ok(())
        }
    }

    fn build_nsp(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut names = Vec::new();
        for (name, _) in files {
            names.push(u32::try_from(strings.len()).unwrap());
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }

        let mut result = Vec::new();
        result.extend_from_slice(b"PFS0");
        result.extend_from_slice(&u32::try_from(files.len()).unwrap().to_le_bytes());
        result.extend_from_slice(&u32::try_from(strings.len()).unwrap().to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());

        let mut file_offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(names) {
            result.extend_from_slice(&file_offset.to_le_bytes());
            result.extend_from_slice(&u64::try_from(data.len()).unwrap().to_le_bytes());
            result.extend_from_slice(&name_offset.to_le_bytes());
            result.extend_from_slice(&0_u32.to_le_bytes());
            file_offset += u64::try_from(data.len()).unwrap();
        }

        result.extend_from_slice(&strings);
        for (_, data) in files {
            result.extend_from_slice(data);
        }
        result
    }

    fn load_bytes(bytes: Vec<u8>) -> Result<NspArchive, LoadError> {
        NspLoader::load(Arc::new(VecStorage(bytes)))
    }

    #[test]
    fn lists_and_opens_entries() {
        let archive =
            load_bytes(build_nsp(&[("first.nca", b"abc"), ("title.tik", b"de")])).unwrap();

        assert_eq!(archive.entries().len(), 2);
        assert_eq!(archive.entries()[0].name(), "first.nca");
        assert_eq!(archive.entries()[0].size(), 3);

        let storage = archive.open("title.tik").unwrap().unwrap();
        let mut data = [0_u8; 2];
        storage.read_at(0, &mut data).unwrap();
        assert_eq!(&data, b"de");
    }

    #[test]
    fn rejects_incorrect_magic() {
        let mut data = build_nsp(&[]);
        data[..4].copy_from_slice(b"NOPE");

        assert!(matches!(
            load_bytes(data),
            Err(LoadError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(matches!(
            load_bytes(b"PFS0".to_vec()),
            Err(LoadError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn rejects_unterminated_name() {
        let mut data = build_nsp(&[("a", b"")]);
        let string_offset = 16 + 24;
        data[string_offset + 1] = b'x';

        assert!(matches!(
            load_bytes(data),
            Err(LoadError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn rejects_entry_outside_package() {
        let mut data = build_nsp(&[("a", b"")]);
        data[16..24].copy_from_slice(&u64::MAX.to_le_bytes());

        assert!(matches!(
            load_bytes(data),
            Err(LoadError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_names() {
        let data = build_nsp(&[("same", b"a"), ("same", b"b")]);

        assert!(matches!(
            load_bytes(data),
            Err(LoadError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn rejects_excessive_metadata() {
        let mut data = build_nsp(&[]);
        data[4..8].copy_from_slice(&u32::MAX.to_le_bytes());

        assert!(matches!(
            load_bytes(data),
            Err(LoadError::InvalidFormat { .. })
        ));
    }
}
