use std::collections::HashSet;

use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::compressed_package::logical_nca_name;
use crate::xci::load_compressed_xci;
use crate::{CompressedPackageEntry, NczLoader, XciArchive, XciPartitionKind};

/// Loads an XCZ by reusing XCI/HFS0 parsing and wrapping stored NCZ files.
#[derive(Debug)]
pub struct XczLoader;

impl FormatLoader for XczLoader {
    type Output = XczArchive;

    const FORMAT_NAME: &'static str = "XCZ";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        let xci = load_compressed_xci(storage)?;
        let mut partitions = Vec::with_capacity(xci.partitions().len());
        for partition in xci.partitions() {
            let archive = partition.archive();
            let mut entries = Vec::with_capacity(archive.entries().len());
            let mut logical_names = HashSet::with_capacity(archive.entries().len());
            for stored in archive.entries() {
                if stored.has_advertised_hash() {
                    let integrity = archive.validate_entry(stored)?;
                    if !integrity.is_valid() {
                        return Err(LoadError::invalid(
                            Self::FORMAT_NAME,
                            format!(
                                "partition {:?} entry {:?} failed its stored HFS0 hash",
                                partition.name(),
                                stored.name()
                            ),
                        ));
                    }
                }
                let stored_storage = archive.open_entry(stored)?;
                let entry = match logical_nca_name(stored.name()) {
                    Some(logical_name) => {
                        let ncz = NczLoader::load(stored_storage).map_err(|error| {
                            LoadError::invalid(
                                Self::FORMAT_NAME,
                                format!(
                                    "partition {:?} entry {:?}: {error}",
                                    partition.name(),
                                    stored.name()
                                ),
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
                            "partition {:?} logical entry name {:?} is duplicated or ambiguous",
                            partition.name(),
                            entry.logical_name()
                        ),
                    ));
                }
                entries.push(entry);
            }
            partitions.push(XczPartition {
                kind: partition.kind().clone(),
                entries,
            });
        }
        Ok(XczArchive { xci, partitions })
    }
}

/// Logical entries exposed by one XCZ HFS0 partition.
#[derive(Debug)]
pub struct XczPartition {
    kind: XciPartitionKind,
    entries: Vec<CompressedPackageEntry>,
}

impl XczPartition {
    pub fn kind(&self) -> &XciPartitionKind {
        &self.kind
    }

    pub fn name(&self) -> &str {
        self.kind.name()
    }

    pub fn entries(&self) -> &[CompressedPackageEntry] {
        &self.entries
    }

    pub fn open_entry(&self, entry: &CompressedPackageEntry) -> Result<StorageRef, LoadError> {
        entry.open()
    }
}

/// Parsed XCZ with both stored XCI metadata and logical partition entries.
#[derive(Debug)]
pub struct XczArchive {
    xci: XciArchive,
    partitions: Vec<XczPartition>,
}

impl XczArchive {
    pub fn xci(&self) -> &XciArchive {
        &self.xci
    }

    pub fn partitions(&self) -> &[XczPartition] {
        &self.partitions
    }

    pub fn partition(&self, kind: &XciPartitionKind) -> Option<&XczPartition> {
        self.partitions
            .iter()
            .find(|partition| partition.kind() == kind)
    }

    pub fn secure_partition(&self) -> Result<&XczPartition, LoadError> {
        self.partition(&XciPartitionKind::Secure)
            .ok_or_else(|| LoadError::invalid("XCZ", "title loading requires a secure partition"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sha2::{Digest, Sha256};
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
        let mut bytes = vec![0x44; 0x4000];
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

    fn hfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut names = Vec::new();
        for (name, _) in files {
            names.push(strings.len() as u32);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HFS0");
        bytes.extend_from_slice(&(files.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        let mut offset = 0_u64;
        for ((_, data), name) in files.iter().zip(names) {
            bytes.extend_from_slice(&offset.to_le_bytes());
            bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&name.to_le_bytes());
            bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&[0; 8]);
            bytes.extend_from_slice(&Sha256::digest(data));
            offset += data.len() as u64;
        }
        bytes.extend_from_slice(&strings);
        for (_, data) in files {
            bytes.extend_from_slice(data);
        }
        bytes
    }

    fn xci(secure: &[u8]) -> Vec<u8> {
        let root = hfs0(&[("secure", secure)]);
        let root_header_size = 0x10 + 0x40 + "secure".len() + 1;
        let image_size = 0x200 + root.len();
        let pages = image_size.div_ceil(0x200);
        let mut bytes = vec![0_u8; pages * 0x200];
        bytes[0x100..0x104].copy_from_slice(b"HEAD");
        bytes[0x118..0x11c].copy_from_slice(&((pages - 1) as u32).to_le_bytes());
        bytes[0x130..0x138].copy_from_slice(&0x200_u64.to_le_bytes());
        bytes[0x138..0x140].copy_from_slice(&(root_header_size as u64).to_le_bytes());
        bytes[0x140..0x160].copy_from_slice(&Sha256::digest(&root[..root_header_size]));
        bytes[0x200..0x200 + root.len()].copy_from_slice(&root);
        bytes
    }

    fn clear_hfs0_hashes(bytes: &mut [u8]) {
        let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        for index in 0..count {
            let hash_offset = 0x10 + index * 0x40 + 0x20;
            bytes[hash_offset..hash_offset + 0x20].fill(0);
        }
    }

    #[test]
    fn validates_stored_hfs0_then_exposes_logical_secure_entries() {
        let tail = b"xcz logical data";
        let compressed = ncz(tail);
        let secure = hfs0(&[("program.ncz", &compressed), ("cert.cert", b"cert")]);
        let archive = XczLoader::load(Arc::new(Bytes(xci(&secure)))).unwrap();
        let partition = archive.secure_partition().unwrap();
        assert_eq!(partition.entries()[0].stored_name(), "program.ncz");
        assert_eq!(partition.entries()[0].logical_name(), "program.nca");
        let mut actual = vec![0_u8; tail.len()];
        partition.entries()[0]
            .open()
            .unwrap()
            .read_at(0x4000, &mut actual)
            .unwrap();
        assert_eq!(actual, tail);
    }

    #[test]
    fn rejects_corrupt_stored_ncz_before_reconstruction() {
        let compressed = ncz(b"data");
        let mut secure = hfs0(&[("program.ncz", &compressed)]);
        *secure.last_mut().unwrap() ^= 1;
        assert!(XczLoader::load(Arc::new(Bytes(xci(&secure)))).is_err());
    }

    #[test]
    fn accepts_rebuilt_hfs0_with_original_xci_geometry_and_unadvertised_hashes() {
        let compressed = ncz(b"data");
        let mut secure = hfs0(&[("program.ncz", &compressed)]);
        secure[8..12].copy_from_slice(&11_u32.to_le_bytes());
        secure[0x10..0x18].copy_from_slice(&1_u64.to_le_bytes());
        clear_hfs0_hashes(&mut secure);
        let mut image = xci(&secure);

        let root_offset = 0x200;
        image[root_offset + 8..root_offset + 12].copy_from_slice(&6_u32.to_le_bytes());
        image[root_offset + 0x10..root_offset + 0x18].copy_from_slice(&1_u64.to_le_bytes());
        clear_hfs0_hashes(&mut image[root_offset..]);
        image[0x118..0x11c].copy_from_slice(&u32::MAX.to_le_bytes());
        image[0x138..0x140].copy_from_slice(&0x200_u64.to_le_bytes());
        image[0x140..0x160].fill(0x55);

        let archive = XczLoader::load(Arc::new(Bytes(image))).unwrap();
        assert!(archive.xci().root_header_integrity().is_none());
        assert!(
            archive.xci().partitions()[0]
                .root_entry_integrity()
                .is_none()
        );
        assert_eq!(
            archive.secure_partition().unwrap().entries()[0].logical_name(),
            "program.nca"
        );
    }
}
