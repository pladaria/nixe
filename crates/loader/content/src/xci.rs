use std::collections::HashSet;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef, SubStorage};

use crate::{Hfs0Archive, Hfs0Entry, Hfs0HashResult, Hfs0Loader};

const PUBLIC_HEADER_SIZE: u64 = 0x200;
const MEDIA_UNIT_SIZE: u64 = 0x200;
const HASH_BUFFER_SIZE: usize = 1024 * 1024;

/// Public, unencrypted fields from an XCI game-card header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XciHeader {
    pub rsa_signature: [u8; 0x100],
    pub secure_area_start_page: u32,
    pub backup_area_start_page: u32,
    pub title_key_index_raw: u8,
    pub title_key_decryption_index: u8,
    pub key_index: u8,
    pub card_size_code: u8,
    pub header_version: u8,
    pub flags: u8,
    pub package_id: u64,
    pub valid_data_end_page: u32,
    pub secondary_header_fields: [u8; 4],
    pub iv: [u8; 16],
    pub root_hfs0_offset: u64,
    pub root_hfs0_header_size: u64,
    pub root_hfs0_header_hash: [u8; 32],
    pub initial_data_hash: [u8; 32],
    pub secure_mode: u32,
    pub title_key_flag: u32,
    pub key_flag: u32,
    pub area_limit_page: u32,
    pub encrypted_gamecard_info: [u8; 0x70],
}

impl XciHeader {
    pub fn secure_area_start_offset(&self) -> Option<u64> {
        u64::from(self.secure_area_start_page).checked_mul(MEDIA_UNIT_SIZE)
    }

    pub fn backup_area_start_offset(&self) -> Option<u64> {
        u64::from(self.backup_area_start_page).checked_mul(MEDIA_UNIT_SIZE)
    }

    pub fn valid_data_end_offset(&self) -> Option<u64> {
        u64::from(self.valid_data_end_page)
            .checked_add(1)?
            .checked_mul(MEDIA_UNIT_SIZE)
    }

    pub fn area_limit_offset(&self) -> Option<u64> {
        u64::from(self.area_limit_page)
            .checked_add(1)?
            .checked_mul(MEDIA_UNIT_SIZE)
    }
}

/// Integrity result for the root HFS0 header advertised by the XCI header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XciRootHeaderIntegrity {
    pub expected: [u8; 32],
    pub actual: [u8; 32],
}

impl XciRootHeaderIntegrity {
    pub fn is_valid(&self) -> bool {
        self.expected == self.actual
    }
}

/// Role assigned to a root XCI HFS0 entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum XciPartitionKind {
    Update,
    Normal,
    Secure,
    Logo,
    Unknown(String),
}

impl XciPartitionKind {
    fn from_name(name: &str) -> Result<Self, LoadError> {
        let lower = name.to_ascii_lowercase();
        let known = match lower.as_str() {
            "update" => Some(Self::Update),
            "normal" => Some(Self::Normal),
            "secure" => Some(Self::Secure),
            "logo" => Some(Self::Logo),
            _ => None,
        };
        if known.is_some() && name != lower {
            return Err(LoadError::invalid(
                "XCI",
                format!("partition name {name:?} is an ambiguous case variant"),
            ));
        }
        Ok(known.unwrap_or_else(|| Self::Unknown(name.to_owned())))
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Update => "update",
            Self::Normal => "normal",
            Self::Secure => "secure",
            Self::Logo => "logo",
            Self::Unknown(name) => name,
        }
    }
}

/// One nested HFS0 partition from an XCI root filesystem.
#[derive(Debug)]
pub struct XciPartition {
    kind: XciPartitionKind,
    root_entry: Hfs0Entry,
    root_entry_integrity: Option<Hfs0HashResult>,
    archive: Hfs0Archive,
}

impl XciPartition {
    pub fn kind(&self) -> &XciPartitionKind {
        &self.kind
    }

    pub fn name(&self) -> &str {
        self.kind.name()
    }

    pub fn root_entry(&self) -> &Hfs0Entry {
        &self.root_entry
    }

    pub fn root_entry_integrity(&self) -> Option<&Hfs0HashResult> {
        self.root_entry_integrity.as_ref()
    }

    pub fn archive(&self) -> &Hfs0Archive {
        &self.archive
    }

    pub fn open(&self, name: &str) -> Result<Option<StorageRef>, LoadError> {
        self.archive.open(name)
    }

    pub fn validate_entries(&self) -> Result<Vec<Hfs0HashResult>, LoadError> {
        self.archive.validate_all()
    }
}

/// Parsed, bounded view of an XCI image and its HFS0 partitions.
#[derive(Debug)]
pub struct XciArchive {
    header: XciHeader,
    root_header_integrity: Option<XciRootHeaderIntegrity>,
    root: Hfs0Archive,
    partitions: Vec<XciPartition>,
}

impl XciArchive {
    pub fn header(&self) -> &XciHeader {
        &self.header
    }

    pub fn root_header_integrity(&self) -> Option<&XciRootHeaderIntegrity> {
        self.root_header_integrity.as_ref()
    }

    pub fn root(&self) -> &Hfs0Archive {
        &self.root
    }

    pub fn partitions(&self) -> &[XciPartition] {
        &self.partitions
    }

    pub fn partition(&self, kind: &XciPartitionKind) -> Option<&XciPartition> {
        self.partitions
            .iter()
            .find(|partition| partition.kind() == kind)
    }

    pub fn partition_by_name(&self, name: &str) -> Option<&XciPartition> {
        self.partitions
            .iter()
            .find(|partition| partition.name() == name)
    }

    pub fn secure_partition(&self) -> Result<&XciPartition, LoadError> {
        self.partition(&XciPartitionKind::Secure)
            .ok_or_else(|| LoadError::invalid("XCI", "title loading requires a secure partition"))
    }

    pub fn open_partition_file(
        &self,
        partition: &XciPartitionKind,
        name: &str,
    ) -> Result<Option<StorageRef>, LoadError> {
        self.partition(partition)
            .map(|partition| partition.open(name))
            .transpose()
            .map(Option::flatten)
    }
}

/// Loads NX Card Image (XCI) files.
#[derive(Debug)]
pub struct XciLoader;

impl FormatLoader for XciLoader {
    type Output = XciArchive;

    const FORMAT_NAME: &'static str = "XCI";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        load_xci(storage, XciValidation::Strict)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XciValidation {
    Strict,
    Compressed,
}

pub(crate) fn load_compressed_xci(storage: StorageRef) -> Result<XciArchive, LoadError> {
    load_xci(storage, XciValidation::Compressed)
}

fn load_xci(storage: StorageRef, validation: XciValidation) -> Result<XciArchive, LoadError> {
    let storage_len = storage.len()?;
    if storage_len < PUBLIC_HEADER_SIZE {
        return Err(LoadError::invalid(
            XciLoader::FORMAT_NAME,
            "header is truncated",
        ));
    }
    let mut bytes = [0_u8; PUBLIC_HEADER_SIZE as usize];
    storage.read_at(0, &mut bytes)?;
    if &bytes[0x100..0x104] != b"HEAD" {
        return Err(LoadError::invalid(
            XciLoader::FORMAT_NAME,
            "expected HEAD magic",
        ));
    }

    let title_key_index_raw = bytes[0x10c];
    let header = XciHeader {
        rsa_signature: bytes[..0x100].try_into().expect("validated XCI header"),
        secure_area_start_page: read_u32(&bytes, 0x104),
        backup_area_start_page: read_u32(&bytes, 0x108),
        title_key_index_raw,
        title_key_decryption_index: title_key_index_raw >> 4,
        key_index: title_key_index_raw & 0x0f,
        card_size_code: bytes[0x10d],
        header_version: bytes[0x10e],
        flags: bytes[0x10f],
        package_id: read_u64(&bytes, 0x110),
        valid_data_end_page: read_u32(&bytes, 0x118),
        secondary_header_fields: bytes[0x11c..0x120]
            .try_into()
            .expect("validated XCI header"),
        iv: bytes[0x120..0x130]
            .try_into()
            .expect("validated XCI header"),
        root_hfs0_offset: read_u64(&bytes, 0x130),
        root_hfs0_header_size: read_u64(&bytes, 0x138),
        root_hfs0_header_hash: bytes[0x140..0x160]
            .try_into()
            .expect("validated XCI header"),
        initial_data_hash: bytes[0x160..0x180]
            .try_into()
            .expect("validated XCI header"),
        secure_mode: read_u32(&bytes, 0x180),
        title_key_flag: read_u32(&bytes, 0x184),
        key_flag: read_u32(&bytes, 0x188),
        area_limit_page: read_u32(&bytes, 0x18c),
        encrypted_gamecard_info: bytes[0x190..0x200]
            .try_into()
            .expect("validated XCI header"),
    };

    header.secure_area_start_offset().ok_or_else(|| {
        LoadError::invalid(XciLoader::FORMAT_NAME, "secure-area page address overflows")
    })?;
    header.backup_area_start_offset().ok_or_else(|| {
        LoadError::invalid(XciLoader::FORMAT_NAME, "backup-area page address overflows")
    })?;
    header.area_limit_offset().ok_or_else(|| {
        LoadError::invalid(XciLoader::FORMAT_NAME, "area-limit page address overflows")
    })?;
    let valid_data_end = header.valid_data_end_offset().ok_or_else(|| {
        LoadError::invalid(XciLoader::FORMAT_NAME, "valid-data-end address overflows")
    })?;
    if validation == XciValidation::Strict && valid_data_end > storage_len {
        return Err(LoadError::invalid(
            XciLoader::FORMAT_NAME,
            "declared valid-data range is outside the image",
        ));
    }
    let root_header_end = header
        .root_hfs0_offset
        .checked_add(header.root_hfs0_header_size)
        .ok_or_else(|| LoadError::invalid(XciLoader::FORMAT_NAME, "root HFS0 header overflows"))?;
    let physical_data_end = match validation {
        XciValidation::Strict => valid_data_end,
        XciValidation::Compressed => storage_len,
    };
    if root_header_end > physical_data_end {
        return Err(LoadError::invalid(
            XciLoader::FORMAT_NAME,
            "root HFS0 header is outside the image data range",
        ));
    }
    let root_extent = physical_data_end
        .checked_sub(header.root_hfs0_offset)
        .ok_or_else(|| LoadError::invalid(XciLoader::FORMAT_NAME, "root HFS0 offset is invalid"))?;
    let root_storage: StorageRef = Arc::new(SubStorage::new(
        storage,
        header.root_hfs0_offset,
        root_extent,
    )?);
    let root_header_integrity = if validation == XciValidation::Strict {
        let actual_hash = hash_prefix(&root_storage, header.root_hfs0_header_size)?;
        let integrity = XciRootHeaderIntegrity {
            expected: header.root_hfs0_header_hash,
            actual: actual_hash,
        };
        if !integrity.is_valid() {
            return Err(LoadError::invalid(
                XciLoader::FORMAT_NAME,
                "root HFS0 header hash does not match",
            ));
        }
        Some(integrity)
    } else {
        None
    };

    let root = match validation {
        XciValidation::Strict => Hfs0Loader::load(root_storage)?,
        XciValidation::Compressed => Hfs0Archive::parse_xcz(root_storage, "XCZ root HFS0")?,
    };
    if validation == XciValidation::Strict && root.data_offset() != header.root_hfs0_header_size {
        return Err(LoadError::invalid(
            XciLoader::FORMAT_NAME,
            "root HFS0 metadata size does not match the header declaration",
        ));
    }
    let mut partitions = Vec::with_capacity(root.entries().len());
    let mut names = HashSet::with_capacity(root.entries().len());
    for entry in root.entries() {
        let folded = entry.name().to_ascii_lowercase();
        if !names.insert(folded) {
            return Err(LoadError::invalid(
                XciLoader::FORMAT_NAME,
                "root partition names are ambiguous when compared case-insensitively",
            ));
        }
        let kind = XciPartitionKind::from_name(entry.name())?;
        let integrity = if validation == XciValidation::Strict || entry.has_advertised_hash() {
            let integrity = root.validate_entry(entry)?;
            if !integrity.is_valid() {
                return Err(LoadError::invalid(
                    XciLoader::FORMAT_NAME,
                    format!(
                        "root partition {} failed HFS0 hash validation",
                        entry.name()
                    ),
                ));
            }
            Some(integrity)
        } else {
            None
        };
        let partition_storage = root.open_entry(entry)?;
        let archive = match validation {
            XciValidation::Strict => Hfs0Archive::parse(partition_storage, "XCI partition HFS0"),
            XciValidation::Compressed => {
                Hfs0Archive::parse_xcz(partition_storage, "XCZ partition HFS0")
            }
        }
        .map_err(|error| contextual_partition_error(entry.name(), error))?;
        partitions.push(XciPartition {
            kind,
            root_entry: entry.clone(),
            root_entry_integrity: integrity,
            archive,
        });
    }

    Ok(XciArchive {
        header,
        root_header_integrity,
        root,
        partitions,
    })
}

fn contextual_partition_error(name: &str, error: LoadError) -> LoadError {
    LoadError::invalid("XCI", format!("partition {name:?}: {error}"))
}

fn hash_prefix(storage: &StorageRef, size: u64) -> Result<[u8; 32], LoadError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_SIZE];
    let mut offset = 0_u64;
    while offset < size {
        let read_size = usize::try_from((size - offset).min(HASH_BUFFER_SIZE as u64))
            .expect("hash buffer size fits usize");
        storage.read_at(offset, &mut buffer[..read_size])?;
        hasher.update(&buffer[..read_size]);
        offset += read_size as u64;
    }
    Ok(hasher.finalize().into())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated range"),
    )
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

    fn hfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut offsets = Vec::new();
        for (name, _) in files {
            offsets.push(strings.len() as u32);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut result = Vec::new();
        result.extend_from_slice(b"HFS0");
        result.extend_from_slice(&(files.len() as u32).to_le_bytes());
        result.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());
        let mut offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(offsets) {
            result.extend_from_slice(&offset.to_le_bytes());
            result.extend_from_slice(&(data.len() as u64).to_le_bytes());
            result.extend_from_slice(&name_offset.to_le_bytes());
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(&[0; 8]);
            result.extend_from_slice(&Sha256::digest(data));
            offset += data.len() as u64;
        }
        result.extend_from_slice(&strings);
        for (_, data) in files {
            result.extend_from_slice(data);
        }
        result
    }

    fn xci(partitions: &[(&str, &[u8])]) -> Vec<u8> {
        let root = hfs0(partitions);
        let root_offset = 0x200_u64;
        let root_header_size = 0x10
            + partitions.len() * 0x40
            + partitions
                .iter()
                .map(|(name, _)| name.len() + 1)
                .sum::<usize>();
        let image_size = root_offset as usize + root.len();
        let pages = image_size.div_ceil(MEDIA_UNIT_SIZE as usize);
        let mut bytes = vec![0_u8; pages * MEDIA_UNIT_SIZE as usize];
        bytes[0x100..0x104].copy_from_slice(b"HEAD");
        bytes[0x10c] = 0xab;
        bytes[0x118..0x11c].copy_from_slice(&((pages - 1) as u32).to_le_bytes());
        bytes[0x130..0x138].copy_from_slice(&root_offset.to_le_bytes());
        bytes[0x138..0x140].copy_from_slice(&(root_header_size as u64).to_le_bytes());
        bytes[0x140..0x160].copy_from_slice(&Sha256::digest(&root[..root_header_size]));
        bytes[root_offset as usize..root_offset as usize + root.len()].copy_from_slice(&root);
        bytes
    }

    fn load(bytes: Vec<u8>) -> Result<XciArchive, LoadError> {
        XciLoader::load(Arc::new(Bytes(bytes)))
    }

    #[test]
    fn loads_secure_optional_and_unknown_partitions() {
        let secure = hfs0(&[("content.nca", b"abc")]);
        let logo = hfs0(&[]);
        let unknown = hfs0(&[]);
        let archive = load(xci(&[
            ("secure", &secure),
            ("logo", &logo),
            ("future", &unknown),
        ]))
        .unwrap();
        assert!(archive.root_header_integrity().unwrap().is_valid());
        assert_eq!(archive.header().title_key_index_raw, 0xab);
        assert_eq!(archive.header().title_key_decryption_index, 0x0a);
        assert_eq!(archive.header().key_index, 0x0b);
        assert_eq!(archive.partitions().len(), 3);
        assert_eq!(
            archive
                .secure_partition()
                .unwrap()
                .archive()
                .entries()
                .len(),
            1
        );
        assert_eq!(
            archive.partitions()[2].kind(),
            &XciPartitionKind::Unknown("future".to_owned())
        );
    }

    #[test]
    fn rejects_bad_magic_hash_and_partition() {
        let secure = hfs0(&[]);
        let mut bad_magic = xci(&[("secure", &secure)]);
        bad_magic[0x100..0x104].copy_from_slice(b"NOPE");
        assert!(load(bad_magic).is_err());

        let mut bad_hash = xci(&[("secure", &secure)]);
        bad_hash[0x140] ^= 1;
        assert!(load(bad_hash).is_err());

        let invalid = b"not hfs0";
        assert!(load(xci(&[("secure", invalid)])).is_err());
    }

    #[test]
    fn rejects_truncation_and_case_variant() {
        assert!(load(vec![0; 0x100]).is_err());
        let empty = hfs0(&[]);
        assert!(load(xci(&[("Secure", &empty)])).is_err());

        let mut truncated = xci(&[("secure", &empty)]);
        truncated.truncate(truncated.len() - 1);
        assert!(load(truncated).is_err());
    }

    #[test]
    fn generic_inspection_does_not_require_a_secure_partition() {
        let normal = hfs0(&[]);
        let archive = load(xci(&[("normal", &normal)])).unwrap();

        assert!(archive.secure_partition().is_err());
        assert!(archive.partition(&XciPartitionKind::Normal).is_some());
    }
}
