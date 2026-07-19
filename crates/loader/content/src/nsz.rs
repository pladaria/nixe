use std::collections::HashSet;

use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::compressed_package::logical_nca_name;
use crate::{CompressedPackageEntry, NczLoader, Pfs0Archive};

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

    use swiitx_loader_storage::{Storage, StorageError};

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
}
