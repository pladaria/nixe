use std::fmt::{Display, Formatter};

use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

const HEADER_SIZE: u64 = 0x20;
const CONTENT_INFO_SIZE: u64 = 0x38;
const CONTENT_META_INFO_SIZE: u64 = 0x10;
const DIGEST_SIZE: u64 = 0x20;
const MAX_TABLE_SIZE: u64 = 16 * 1024 * 1024;

/// Role of packaged content metadata. Unknown values remain distinguishable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CnmtMetaType {
    Unknown,
    SystemProgram,
    SystemData,
    SystemUpdate,
    BootImagePackage,
    BootImagePackageSafe,
    Application,
    Patch,
    AddOnContent,
    Delta,
    DataPatch,
    Unrecognized(u8),
}

impl From<u8> for CnmtMetaType {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Unknown,
            0x01 => Self::SystemProgram,
            0x02 => Self::SystemData,
            0x03 => Self::SystemUpdate,
            0x04 => Self::BootImagePackage,
            0x05 => Self::BootImagePackageSafe,
            0x80 => Self::Application,
            0x81 => Self::Patch,
            0x82 => Self::AddOnContent,
            0x83 => Self::Delta,
            0x84 => Self::DataPatch,
            value => Self::Unrecognized(value),
        }
    }
}

impl Display for CnmtMetaType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => formatter.write_str("Unknown"),
            Self::SystemProgram => formatter.write_str("SystemProgram"),
            Self::SystemData => formatter.write_str("SystemData"),
            Self::SystemUpdate => formatter.write_str("SystemUpdate"),
            Self::BootImagePackage => formatter.write_str("BootImagePackage"),
            Self::BootImagePackageSafe => formatter.write_str("BootImagePackageSafe"),
            Self::Application => formatter.write_str("Application"),
            Self::Patch => formatter.write_str("Patch"),
            Self::AddOnContent => formatter.write_str("AddOnContent"),
            Self::Delta => formatter.write_str("Delta"),
            Self::DataPatch => formatter.write_str("DataPatch"),
            Self::Unrecognized(value) => write!(formatter, "Unknown({value:#04X})"),
        }
    }
}

/// Platform encoded in the packaged CNMT header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CnmtPlatform {
    Nx,
    Unknown(u8),
}

impl From<u8> for CnmtPlatform {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Nx,
            value => Self::Unknown(value),
        }
    }
}

impl Display for CnmtPlatform {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nx => formatter.write_str("NX"),
            Self::Unknown(value) => write!(formatter, "Unknown({value:#04X})"),
        }
    }
}

/// Installation form encoded in the packaged CNMT header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CnmtInstallType {
    Full,
    FragmentOnly,
    Unknown(u8),
}

impl From<u8> for CnmtInstallType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Full,
            1 => Self::FragmentOnly,
            value => Self::Unknown(value),
        }
    }
}

impl Display for CnmtInstallType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => formatter.write_str("Full"),
            Self::FragmentOnly => formatter.write_str("FragmentOnly"),
            Self::Unknown(value) => write!(formatter, "Unknown({value:#04X})"),
        }
    }
}

/// Role of an NCA referenced by one CNMT content record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CnmtContentType {
    Meta,
    Program,
    Data,
    Control,
    HtmlDocument,
    LegalInformation,
    DeltaFragment,
    Unknown(u8),
}

impl From<u8> for CnmtContentType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Meta,
            1 => Self::Program,
            2 => Self::Data,
            3 => Self::Control,
            4 => Self::HtmlDocument,
            5 => Self::LegalInformation,
            6 => Self::DeltaFragment,
            value => Self::Unknown(value),
        }
    }
}

impl Display for CnmtContentType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Meta => formatter.write_str("Meta"),
            Self::Program => formatter.write_str("Program"),
            Self::Data => formatter.write_str("Data"),
            Self::Control => formatter.write_str("Control"),
            Self::HtmlDocument => formatter.write_str("HtmlDocument"),
            Self::LegalInformation => formatter.write_str("LegalInformation"),
            Self::DeltaFragment => formatter.write_str("DeltaFragment"),
            Self::Unknown(value) => write!(formatter, "Unknown({value:#04X})"),
        }
    }
}

/// Type-specific data immediately following the packaged CNMT header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CnmtExtendedHeader {
    None,
    Application {
        patch_id: u64,
        required_system_version: u32,
        required_application_version: u32,
    },
    Patch {
        application_id: u64,
        required_system_version: u32,
        extended_data_size: u32,
        reserved: [u8; 8],
    },
    AddOnContent {
        application_id: u64,
        required_application_version: u32,
        content_accessibilities: u8,
        padding: [u8; 3],
        data_patch_id: u64,
    },
    LegacyAddOnContent {
        application_id: u64,
        required_application_version: u32,
        padding: u32,
    },
    Delta {
        application_id: u64,
        extended_data_size: u32,
        padding: u32,
    },
    SystemUpdate {
        extended_data_size: u32,
    },
    Unknown(Vec<u8>),
}

impl CnmtExtendedHeader {
    fn declared_extended_data_size(&self) -> Option<u64> {
        match self {
            Self::Patch {
                extended_data_size, ..
            }
            | Self::Delta {
                extended_data_size, ..
            }
            | Self::SystemUpdate { extended_data_size } => Some(u64::from(*extended_data_size)),
            Self::None
            | Self::Application { .. }
            | Self::AddOnContent { .. }
            | Self::LegacyAddOnContent { .. } => Some(0),
            Self::Unknown(_) => None,
        }
    }
}

/// One packaged content-info record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CnmtContentInfo {
    pub hash: [u8; 32],
    pub content_id: [u8; 16],
    pub size: u64,
    pub attributes: u8,
    pub content_type: CnmtContentType,
    pub id_offset: u8,
}

/// One reference to another content-meta record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CnmtContentMetaInfo {
    pub title_id: u64,
    pub version: u32,
    pub content_meta_type: CnmtMetaType,
    pub attributes: u8,
    pub padding: [u8; 2],
}

/// Canonical packaged content metadata stored inside a meta NCA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CnmtContentMeta {
    pub title_id: u64,
    pub version: u32,
    pub content_meta_type: CnmtMetaType,
    pub platform: CnmtPlatform,
    pub extended_header_size: u16,
    pub attributes: u8,
    pub storage_id: u8,
    pub install_type: CnmtInstallType,
    pub committed: bool,
    pub required_download_system_version: u32,
    pub reserved: [u8; 4],
    pub extended_header: CnmtExtendedHeader,
    pub contents: Vec<CnmtContentInfo>,
    pub content_meta: Vec<CnmtContentMetaInfo>,
    /// Size of type-specific data after both record tables. The parser does
    /// not copy this potentially large region into memory.
    pub extended_data_size: u64,
    pub digest: [u8; 32],
}

/// Loads canonical binary packaged content metadata (`*.cnmt`).
#[derive(Debug)]
pub struct CnmtLoader;

impl FormatLoader for CnmtLoader {
    type Output = CnmtContentMeta;

    const FORMAT_NAME: &'static str = "CNMT";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        parse_cnmt(storage)
    }
}

fn parse_cnmt(storage: StorageRef) -> Result<CnmtContentMeta, LoadError> {
    let storage_len = storage.len()?;
    if storage_len < HEADER_SIZE {
        return Err(LoadError::invalid("CNMT", "main header is truncated"));
    }

    let mut header = [0_u8; HEADER_SIZE as usize];
    storage.read_at(0, &mut header)?;
    let content_meta_type = CnmtMetaType::from(header[0x0C]);
    let extended_header_size = read_u16(&header, 0x0E);
    let content_count = u64::from(read_u16(&header, 0x10));
    let content_meta_count = u64::from(read_u16(&header, 0x12));

    let content_table_size = content_count
        .checked_mul(CONTENT_INFO_SIZE)
        .ok_or_else(|| LoadError::invalid("CNMT", "content table size overflows"))?;
    let content_meta_table_size = content_meta_count
        .checked_mul(CONTENT_META_INFO_SIZE)
        .ok_or_else(|| LoadError::invalid("CNMT", "content-meta table size overflows"))?;
    let table_size = content_table_size
        .checked_add(content_meta_table_size)
        .ok_or_else(|| LoadError::invalid("CNMT", "record table size overflows"))?;
    if table_size > MAX_TABLE_SIZE {
        return Err(LoadError::invalid(
            "CNMT",
            "record tables exceed the 16 MiB safety limit",
        ));
    }

    let extended_header_offset = HEADER_SIZE;
    let content_offset = checked_end(
        extended_header_offset,
        u64::from(extended_header_size),
        "extended header",
    )?;
    let content_meta_offset = checked_end(content_offset, content_table_size, "content table")?;
    let extended_data_offset = checked_end(
        content_meta_offset,
        content_meta_table_size,
        "content-meta table",
    )?;
    let minimum_size = checked_end(extended_data_offset, DIGEST_SIZE, "CNMT digest")?;
    if minimum_size > storage_len {
        return Err(LoadError::invalid(
            "CNMT",
            "header or record tables are truncated",
        ));
    }

    let extended_header_len = usize::from(extended_header_size);
    let mut extended_header_bytes = vec![0_u8; extended_header_len];
    storage.read_at(extended_header_offset, &mut extended_header_bytes)?;
    let extended_header = parse_extended_header(content_meta_type, &extended_header_bytes)?;

    let inferred_extended_data_size = storage_len - minimum_size;
    let extended_data_size = match extended_header.declared_extended_data_size() {
        Some(declared) if declared != inferred_extended_data_size => {
            return Err(LoadError::invalid(
                "CNMT",
                format!(
                    "declared extended-data size {declared:#x} does not match available size {inferred_extended_data_size:#x}"
                ),
            ));
        }
        Some(declared) => declared,
        None => inferred_extended_data_size,
    };
    let digest_offset = checked_end(extended_data_offset, extended_data_size, "extended data")?;

    let content_capacity = usize::try_from(content_count)
        .map_err(|_| LoadError::invalid("CNMT", "content count does not fit in memory"))?;
    let mut contents = Vec::with_capacity(content_capacity);
    for index in 0..content_count {
        let offset = record_offset(content_offset, index, CONTENT_INFO_SIZE, "content record")?;
        let mut bytes = [0_u8; CONTENT_INFO_SIZE as usize];
        storage.read_at(offset, &mut bytes)?;
        let mut size_bytes = [0_u8; 8];
        // CNMT stores the declared content size as a packed, little-endian
        // unsigned 40-bit integer at bytes 0x30..0x35.
        size_bytes[..5].copy_from_slice(&bytes[0x30..0x35]);
        contents.push(CnmtContentInfo {
            hash: bytes[..0x20].try_into().expect("fixed CNMT hash range"),
            content_id: bytes[0x20..0x30]
                .try_into()
                .expect("fixed CNMT content-ID range"),
            size: u64::from_le_bytes(size_bytes),
            attributes: bytes[0x35],
            content_type: CnmtContentType::from(bytes[0x36]),
            id_offset: bytes[0x37],
        });
    }

    let content_meta_capacity = usize::try_from(content_meta_count)
        .map_err(|_| LoadError::invalid("CNMT", "content-meta count does not fit in memory"))?;
    let mut content_meta = Vec::with_capacity(content_meta_capacity);
    for index in 0..content_meta_count {
        let offset = record_offset(
            content_meta_offset,
            index,
            CONTENT_META_INFO_SIZE,
            "content-meta record",
        )?;
        let mut bytes = [0_u8; CONTENT_META_INFO_SIZE as usize];
        storage.read_at(offset, &mut bytes)?;
        content_meta.push(CnmtContentMetaInfo {
            title_id: read_u64(&bytes, 0),
            version: read_u32(&bytes, 8),
            content_meta_type: CnmtMetaType::from(bytes[0x0C]),
            attributes: bytes[0x0D],
            padding: bytes[0x0E..0x10]
                .try_into()
                .expect("fixed CNMT reference padding"),
        });
    }

    let mut digest = [0_u8; DIGEST_SIZE as usize];
    storage.read_at(digest_offset, &mut digest)?;
    let committed = match header[0x17] {
        0 => false,
        1 => true,
        value => {
            return Err(LoadError::invalid(
                "CNMT",
                format!("committed flag has invalid value {value}"),
            ));
        }
    };

    Ok(CnmtContentMeta {
        title_id: read_u64(&header, 0),
        version: read_u32(&header, 8),
        content_meta_type,
        platform: CnmtPlatform::from(header[0x0D]),
        extended_header_size,
        attributes: header[0x14],
        storage_id: header[0x15],
        install_type: CnmtInstallType::from(header[0x16]),
        committed,
        required_download_system_version: read_u32(&header, 0x18),
        reserved: header[0x1C..0x20]
            .try_into()
            .expect("fixed CNMT reserved range"),
        extended_header,
        contents,
        content_meta,
        extended_data_size,
        digest,
    })
}

fn parse_extended_header(
    content_meta_type: CnmtMetaType,
    bytes: &[u8],
) -> Result<CnmtExtendedHeader, LoadError> {
    let invalid_size = |expected: &str| {
        LoadError::invalid(
            "CNMT",
            format!(
                "{:?} extended header has size {:#x}, expected {expected}",
                content_meta_type,
                bytes.len()
            ),
        )
    };
    match content_meta_type {
        CnmtMetaType::Application => {
            if bytes.len() != 0x10 {
                return Err(invalid_size("0x10"));
            }
            Ok(CnmtExtendedHeader::Application {
                patch_id: read_u64(bytes, 0),
                required_system_version: read_u32(bytes, 8),
                required_application_version: read_u32(bytes, 0x0C),
            })
        }
        CnmtMetaType::Patch => {
            if bytes.len() != 0x18 {
                return Err(invalid_size("0x18"));
            }
            Ok(CnmtExtendedHeader::Patch {
                application_id: read_u64(bytes, 0),
                required_system_version: read_u32(bytes, 8),
                extended_data_size: read_u32(bytes, 0x0C),
                reserved: bytes[0x10..0x18]
                    .try_into()
                    .expect("fixed patch reserved range"),
            })
        }
        CnmtMetaType::AddOnContent => {
            if bytes.len() != 0x10 && bytes.len() != 0x18 {
                return Err(invalid_size("0x10 (legacy) or 0x18"));
            }
            if bytes.len() == 0x18 {
                Ok(CnmtExtendedHeader::AddOnContent {
                    application_id: read_u64(bytes, 0),
                    required_application_version: read_u32(bytes, 8),
                    content_accessibilities: bytes[0x0C],
                    padding: bytes[0x0D..0x10]
                        .try_into()
                        .expect("fixed add-on-content padding"),
                    data_patch_id: read_u64(bytes, 0x10),
                })
            } else {
                Ok(CnmtExtendedHeader::LegacyAddOnContent {
                    application_id: read_u64(bytes, 0),
                    required_application_version: read_u32(bytes, 8),
                    padding: read_u32(bytes, 0x0C),
                })
            }
        }
        CnmtMetaType::Delta => {
            if bytes.len() != 0x10 {
                return Err(invalid_size("0x10"));
            }
            Ok(CnmtExtendedHeader::Delta {
                application_id: read_u64(bytes, 0),
                extended_data_size: read_u32(bytes, 8),
                padding: read_u32(bytes, 0x0C),
            })
        }
        CnmtMetaType::SystemUpdate => {
            if bytes.is_empty() {
                Ok(CnmtExtendedHeader::None)
            } else if bytes.len() == 4 {
                Ok(CnmtExtendedHeader::SystemUpdate {
                    extended_data_size: read_u32(bytes, 0),
                })
            } else {
                Err(invalid_size("0 or 0x4"))
            }
        }
        CnmtMetaType::Unknown
        | CnmtMetaType::SystemProgram
        | CnmtMetaType::SystemData
        | CnmtMetaType::BootImagePackage
        | CnmtMetaType::BootImagePackageSafe
        | CnmtMetaType::DataPatch => {
            if bytes.is_empty() {
                Ok(CnmtExtendedHeader::None)
            } else {
                Err(invalid_size("0"))
            }
        }
        CnmtMetaType::Unrecognized(_) => Ok(CnmtExtendedHeader::Unknown(bytes.to_vec())),
    }
}

fn checked_end(offset: u64, size: u64, name: &str) -> Result<u64, LoadError> {
    offset
        .checked_add(size)
        .ok_or_else(|| LoadError::invalid("CNMT", format!("{name} range overflows")))
}

fn record_offset(base: u64, index: u64, size: u64, name: &str) -> Result<u64, LoadError> {
    let relative = index
        .checked_mul(size)
        .ok_or_else(|| LoadError::invalid("CNMT", format!("{name} offset overflows")))?;
    checked_end(base, relative, name)
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("validated CNMT u16 range"),
    )
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated CNMT u32 range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated CNMT u64 range"),
    )
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
            buffer.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
            Ok(())
        }
    }

    fn load(bytes: Vec<u8>) -> Result<CnmtContentMeta, LoadError> {
        CnmtLoader::load(Arc::new(VecStorage(bytes)))
    }

    fn build_cnmt(meta_type: u8, extended_header: &[u8], contents: usize, refs: usize) -> Vec<u8> {
        let mut bytes = vec![0_u8; HEADER_SIZE as usize];
        put_u64(&mut bytes, 0, 0x0100_1234_5678_9000);
        put_u32(&mut bytes, 8, 42);
        bytes[0x0C] = meta_type;
        put_u16(
            &mut bytes,
            0x0E,
            u16::try_from(extended_header.len()).unwrap(),
        );
        put_u16(&mut bytes, 0x10, u16::try_from(contents).unwrap());
        put_u16(&mut bytes, 0x12, u16::try_from(refs).unwrap());
        bytes[0x17] = 1;
        bytes.extend_from_slice(extended_header);
        for index in 0..contents {
            let mut record = [0_u8; CONTENT_INFO_SIZE as usize];
            record[..0x20].fill(u8::try_from(index + 1).unwrap());
            record[0x20..0x30].fill(u8::try_from(index + 0x10).unwrap());
            record[0x30..0x35].copy_from_slice(&[0x05, 0x04, 0x03, 0x02, 0x01]);
            record[0x35] = 9;
            record[0x36] = if index == 0 { 1 } else { 0xEE };
            record[0x37] = u8::try_from(index).unwrap();
            bytes.extend_from_slice(&record);
        }
        for index in 0..refs {
            let mut record = [0_u8; CONTENT_META_INFO_SIZE as usize];
            put_u64(&mut record, 0, 0x0100_0000_0000_0800 + index as u64);
            put_u32(&mut record, 8, 7);
            record[0x0C] = 0x81;
            record[0x0D] = 3;
            bytes.extend_from_slice(&record);
        }
        bytes.extend_from_slice(&[0xA5; DIGEST_SIZE as usize]);
        bytes
    }

    #[test]
    fn parses_application_content_and_meta_records() {
        let mut extended = [0_u8; 0x10];
        put_u64(&mut extended, 0, 0x0100_1234_5678_9800);
        put_u32(&mut extended, 8, 100);
        put_u32(&mut extended, 0x0C, 200);
        let metadata = load(build_cnmt(0x80, &extended, 2, 1)).unwrap();

        assert_eq!(metadata.content_meta_type, CnmtMetaType::Application);
        assert_eq!(metadata.contents.len(), 2);
        assert_eq!(metadata.contents[0].size, 0x01_0203_0405);
        assert_eq!(metadata.contents[0].content_type, CnmtContentType::Program);
        assert_eq!(
            metadata.contents[1].content_type,
            CnmtContentType::Unknown(0xEE)
        );
        assert_eq!(metadata.content_meta.len(), 1);
        assert_eq!(
            metadata.content_meta[0].content_meta_type,
            CnmtMetaType::Patch
        );
        assert_eq!(metadata.digest, [0xA5; 32]);
    }

    #[test]
    fn parses_patch_application_relationship_and_extended_data() {
        let mut extended = [0_u8; 0x18];
        put_u64(&mut extended, 0, 0x0100_1234_5678_9000);
        put_u32(&mut extended, 8, 300);
        put_u32(&mut extended, 0x0C, 3);
        let mut bytes = build_cnmt(0x81, &extended, 0, 0);
        let digest = bytes.split_off(bytes.len() - 32);
        bytes.extend_from_slice(&[1, 2, 3]);
        bytes.extend_from_slice(&digest);

        let metadata = load(bytes).unwrap();
        assert_eq!(metadata.extended_data_size, 3);
        assert!(matches!(
            metadata.extended_header,
            CnmtExtendedHeader::Patch {
                application_id: 0x0100_1234_5678_9000,
                required_system_version: 300,
                extended_data_size: 3,
                ..
            }
        ));
    }

    #[test]
    fn preserves_unknown_meta_type_and_extended_header() {
        let metadata = load(build_cnmt(0xF2, &[1, 2, 3], 0, 0)).unwrap();
        assert_eq!(metadata.content_meta_type, CnmtMetaType::Unrecognized(0xF2));
        assert_eq!(
            metadata.extended_header,
            CnmtExtendedHeader::Unknown(vec![1, 2, 3])
        );
    }

    #[test]
    fn rejects_truncated_main_and_extended_headers() {
        assert!(load(vec![0; 0x1F]).is_err());
        let mut bytes = build_cnmt(0x80, &[0; 0x10], 0, 0);
        put_u16(&mut bytes, 0x0E, 0x1000);
        assert!(load(bytes).is_err());
    }

    #[test]
    fn rejects_wrong_known_extended_header_size() {
        assert!(load(build_cnmt(0x81, &[0; 0x10], 0, 0)).is_err());
    }

    #[test]
    fn rejects_oversized_counts_before_allocating() {
        let mut bytes = build_cnmt(0x80, &[0; 0x10], 0, 0);
        put_u16(&mut bytes, 0x10, u16::MAX);
        put_u16(&mut bytes, 0x12, u16::MAX);
        assert!(load(bytes).is_err());
    }

    #[test]
    fn rejects_truncated_content_record() {
        let mut bytes = build_cnmt(0x80, &[0; 0x10], 1, 0);
        bytes.truncate(bytes.len() - 33);
        assert!(load(bytes).is_err());
    }

    #[test]
    fn checked_record_offsets_reject_overflow() {
        assert!(record_offset(u64::MAX, 1, CONTENT_INFO_SIZE, "test").is_err());
        assert!(record_offset(0, u64::MAX, CONTENT_INFO_SIZE, "test").is_err());
    }

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}
