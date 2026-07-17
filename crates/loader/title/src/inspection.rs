use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use swiitx_loader_content::{
    NcaContentType, NcaDistributionType, NcaEncryptionType, NcaFormatVersion, NcaKeySet, NcaLoader,
    NcaSectionType, NspLoader,
};
use swiitx_loader_storage::{FileStorage, FormatLoader, LoadError, Storage, StorageRef};

const MAX_AUXILIARY_METADATA_SIZE: u64 = 1024 * 1024;

/// Container format recognized while inspecting a title path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageFormat {
    /// Nintendo Submission Package backed by PFS0.
    Nsp,
}

impl Display for PackageFormat {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nsp => formatter.write_str("NSP (PFS0)"),
        }
    }
}

/// Best-effort classification based on a package entry's extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntryKind {
    /// NCA containing packaged content metadata.
    MetaContentArchive,
    /// Nintendo Content Archive.
    ContentArchive,
    /// Title ticket.
    Ticket,
    /// Certificate associated with a ticket.
    Certificate,
    /// XML metadata supplied alongside content.
    XmlMetadata,
    /// Entry not recognized from its name.
    Other,
}

impl Display for EntryKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.pad(match self {
            Self::MetaContentArchive => "meta NCA",
            Self::ContentArchive => "NCA",
            Self::Ticket => "ticket",
            Self::Certificate => "certificate",
            Self::XmlMetadata => "XML metadata",
            Self::Other => "other",
        })
    }
}

/// Information available from one NSP PFS0 entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryInspection {
    /// Entry name from the PFS0 string table.
    pub name: String,
    /// Best-effort type inferred from the entry name.
    pub kind: EntryKind,
    /// Absolute byte offset inside the package.
    pub offset: u64,
    /// Entry size in bytes.
    pub size: u64,
    /// Parsed NCA header and section layout for content-archive entries.
    pub nca: Option<NcaInspection>,
    /// Explanation when an NCA entry could not be inspected.
    pub nca_warning: Option<String>,
}

/// Header and section information parsed from one NCA entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcaInspection {
    pub format_version: NcaFormatVersion,
    pub distribution_type: NcaDistributionType,
    pub content_type: NcaContentType,
    pub size: u64,
    pub title_id: u64,
    pub sdk_version: u32,
    pub key_generation: u8,
    pub key_area_key_index: u8,
    pub rights_id: Option<[u8; 16]>,
    pub source_is_decrypted: bool,
    pub sections: Vec<NcaSectionInspection>,
}

/// Physical section discovered in an NCA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcaSectionInspection {
    pub index: u8,
    pub offset: u64,
    pub size: u64,
    pub section_type: NcaSectionType,
    pub encryption_type: NcaEncryptionType,
    pub fs_header_hash_valid: bool,
}

/// One content record described by auxiliary CNMT XML metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentRecordInspection {
    /// Content role, such as Program, Control, or Meta.
    pub content_type: String,
    /// 128-bit content identifier encoded as hexadecimal text.
    pub id: String,
    /// Declared content size in bytes.
    pub size: u64,
    /// SHA-256 digest encoded as hexadecimal text, when present.
    pub hash: Option<String>,
    /// Key generation required by this content, when present.
    pub key_generation: Option<u32>,
}

/// Information parsed from optional `.cnmt.xml` dump metadata.
///
/// This XML is a convenient auxiliary representation and is not a substitute
/// for validating the canonical CNMT stored inside its meta NCA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentMetaInspection {
    /// Content-meta role, such as Application, Patch, or AddOnContent.
    pub content_type: String,
    /// Title identifier declared by the metadata.
    pub title_id: u64,
    /// Raw title version.
    pub version: u32,
    /// Base title associated with a patch, when declared.
    pub original_id: Option<u64>,
    /// Application associated with add-on content, when declared.
    pub application_id: Option<u64>,
    /// Minimum key generation among the package contents, when declared.
    pub minimum_key_generation: Option<u32>,
    /// Required system version, when declared.
    pub required_system_version: Option<u32>,
    /// Required application version, when declared.
    pub required_application_version: Option<u32>,
    /// Content records listed by the metadata.
    pub contents: Vec<ContentRecordInspection>,
    /// Overall metadata digest, when present.
    pub digest: Option<String>,
}

/// Information obtained from one package without parsing its encrypted NCAs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageInspection {
    /// Local path from which the package was opened.
    pub path: PathBuf,
    /// Recognized outer package format.
    pub format: PackageFormat,
    /// Total package size in bytes.
    pub size: u64,
    /// Offset at which the PFS0 file-data area starts.
    pub data_offset: u64,
    /// Files stored in the package.
    pub entries: Vec<EntryInspection>,
    /// Optional auxiliary content metadata found alongside the NCAs.
    pub content_meta: Option<ContentMetaInspection>,
    /// Explanation when auxiliary metadata existed but could not be read.
    pub metadata_warning: Option<String>,
    /// Problems encountered while importing title keys from package tickets.
    pub ticket_warnings: Vec<String>,
}

impl PackageInspection {
    /// Returns the sum of the sizes declared by all entries.
    pub fn payload_size(&self) -> u64 {
        self.entries
            .iter()
            .fold(0_u64, |total, entry| total.saturating_add(entry.size))
    }

    /// Returns bytes occupied by the PFS0 header, tables, and any padding.
    pub fn container_overhead(&self) -> u64 {
        self.size.saturating_sub(self.payload_size())
    }
}

/// Best-effort inspection report for a file or directory supplied by the user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TitleInspection {
    /// Original path supplied for inspection.
    pub path: PathBuf,
    /// Supported packages successfully inspected below the path.
    pub packages: Vec<PackageInspection>,
    /// Regular files skipped because their format is not supported yet.
    pub ignored_files: Vec<PathBuf>,
}

/// Inspects title packages without loading their payloads into memory.
#[derive(Debug)]
pub struct TitleInspector;

impl TitleInspector {
    /// Inspects one NSP file or every direct child file of a directory.
    pub fn inspect(path: impl AsRef<Path>) -> Result<TitleInspection, InspectError> {
        Self::inspect_impl(path.as_ref(), None)
    }

    /// Inspects title packages and decrypts their NCAs with caller-owned keys.
    /// Encrypted title keys present in NSP tickets are imported into the key set.
    pub fn inspect_with_key_set(
        path: impl AsRef<Path>,
        keys: &mut NcaKeySet,
    ) -> Result<TitleInspection, InspectError> {
        Self::inspect_impl(path.as_ref(), Some(keys))
    }

    fn inspect_impl(
        path: &Path,
        mut keys: Option<&mut NcaKeySet>,
    ) -> Result<TitleInspection, InspectError> {
        let metadata = fs::metadata(path).map_err(|source| InspectError::Io {
            path: path.to_owned(),
            source,
        })?;

        let candidates = if metadata.is_file() {
            vec![path.to_owned()]
        } else if metadata.is_dir() {
            directory_files(path)?
        } else {
            return Err(InspectError::UnsupportedPath(path.to_owned()));
        };

        let mut packages = Vec::new();
        let mut ignored_files = Vec::new();
        for candidate in candidates {
            match package_format(&candidate) {
                Some(PackageFormat::Nsp) => {
                    packages.push(inspect_nsp(&candidate, keys.as_deref_mut())?)
                }
                None => ignored_files.push(candidate),
            }
        }

        if packages.is_empty() {
            return Err(InspectError::NoSupportedPackages(path.to_owned()));
        }

        Ok(TitleInspection {
            path: path.to_owned(),
            packages,
            ignored_files,
        })
    }
}

/// Errors produced while inspecting a local title path.
#[derive(Debug)]
pub enum InspectError {
    /// A local file-system operation failed.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A package could not be parsed.
    Package { path: PathBuf, source: LoadError },
    /// The supplied path is neither a regular file nor a directory.
    UnsupportedPath(PathBuf),
    /// No supported package was found at the supplied path.
    NoSupportedPackages(PathBuf),
}

impl Display for InspectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "cannot access {}: {source}", path.display())
            }
            Self::Package { path, source } => {
                write!(formatter, "cannot inspect {}: {source}", path.display())
            }
            Self::UnsupportedPath(path) => {
                write!(formatter, "unsupported path type: {}", path.display())
            }
            Self::NoSupportedPackages(path) => write!(
                formatter,
                "no supported title packages found at {}",
                path.display()
            ),
        }
    }
}

impl Error for InspectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Package { source, .. } => Some(source),
            Self::UnsupportedPath(_) | Self::NoSupportedPackages(_) => None,
        }
    }
}

fn directory_files(path: &Path) -> Result<Vec<PathBuf>, InspectError> {
    let entries = fs::read_dir(path).map_err(|source| InspectError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| InspectError::Io {
            path: path.to_owned(),
            source,
        })?;
        let file_type = entry.file_type().map_err(|source| InspectError::Io {
            path: entry.path(),
            source,
        })?;
        if file_type.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

fn package_format(path: &Path) -> Option<PackageFormat> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| extension.eq_ignore_ascii_case("nsp"))
        .map(|_| PackageFormat::Nsp)
}

fn inspect_nsp(
    path: &Path,
    mut keys: Option<&mut NcaKeySet>,
) -> Result<PackageInspection, InspectError> {
    let storage = FileStorage::open(path).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source: LoadError::Storage(source),
    })?;
    let size = storage.len().map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source: LoadError::Storage(source),
    })?;
    let storage: StorageRef = Arc::new(storage);
    let archive = NspLoader::load(storage).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source,
    })?;
    let ticket_warnings = keys
        .as_deref_mut()
        .map_or_else(Vec::new, |keys| import_ticket_keys(&archive, keys));
    let mut entries = Vec::with_capacity(archive.entries().len());
    for entry in archive.entries() {
        let kind = entry_kind(entry.name());
        let (nca, nca_warning) = if matches!(
            kind,
            EntryKind::MetaContentArchive | EntryKind::ContentArchive
        ) {
            let result = archive
                .open_entry(entry)
                .and_then(|storage| match keys.as_deref() {
                    Some(keys) => NcaLoader::load_with_key_provider(storage, keys),
                    None => NcaLoader::load(storage),
                });
            match result {
                Ok(archive) => (Some(inspect_nca(&archive)), None),
                Err(error) => (None, Some(error.to_string())),
            }
        } else {
            (None, None)
        };
        entries.push(EntryInspection {
            name: entry.name().to_owned(),
            kind,
            offset: entry.offset(),
            size: entry.size(),
            nca,
            nca_warning,
        });
    }
    let (content_meta, metadata_warning) = inspect_auxiliary_metadata(&archive);

    Ok(PackageInspection {
        path: path.to_owned(),
        format: PackageFormat::Nsp,
        size,
        data_offset: archive.data_offset(),
        entries,
        content_meta,
        metadata_warning,
        ticket_warnings,
    })
}

fn import_ticket_keys(
    archive: &swiitx_loader_content::NspArchive,
    keys: &mut NcaKeySet,
) -> Vec<String> {
    const ENCRYPTED_TITLE_KEY_OFFSET: u64 = 0x180;
    const RIGHTS_ID_OFFSET: u64 = 0x2A0;
    const REQUIRED_TICKET_SIZE: u64 = RIGHTS_ID_OFFSET + 16;

    let mut warnings = Vec::new();
    for entry in archive
        .entries()
        .iter()
        .filter(|entry| entry_kind(entry.name()) == EntryKind::Ticket)
    {
        let result = (|| {
            if entry.size() < REQUIRED_TICKET_SIZE {
                return Err(LoadError::invalid("ticket", "ticket is truncated"));
            }
            let storage = archive.open_entry(entry)?;
            let mut encrypted_title_key = [0_u8; 16];
            let mut rights_id = [0_u8; 16];
            storage.read_at(ENCRYPTED_TITLE_KEY_OFFSET, &mut encrypted_title_key)?;
            storage.read_at(RIGHTS_ID_OFFSET, &mut rights_id)?;
            keys.insert_encrypted_title_key(rights_id, encrypted_title_key);
            Ok::<_, LoadError>(())
        })();
        if let Err(error) = result {
            warnings.push(format!("{}: {error}", entry.name()));
        }
    }
    warnings
}

fn inspect_nca(archive: &swiitx_loader_content::NcaArchive) -> NcaInspection {
    let header = archive.header();
    NcaInspection {
        format_version: header.version(),
        distribution_type: header.distribution_type(),
        content_type: header.content_type(),
        size: header.size(),
        title_id: header.title_id(),
        sdk_version: header.sdk_version(),
        key_generation: header.key_generation(),
        key_area_key_index: header.key_area_key_index(),
        rights_id: header.rights_id().copied(),
        source_is_decrypted: header.source_is_decrypted(),
        sections: archive
            .sections()
            .iter()
            .map(|section| NcaSectionInspection {
                index: section.index(),
                offset: section.offset(),
                size: section.size(),
                section_type: section.section_type(),
                encryption_type: section.encryption_type(),
                fs_header_hash_valid: section.fs_header_hash_valid(),
            })
            .collect(),
    }
}

fn inspect_auxiliary_metadata(
    archive: &swiitx_loader_content::NspArchive,
) -> (Option<ContentMetaInspection>, Option<String>) {
    let Some(entry) = archive
        .entries()
        .iter()
        .find(|entry| entry.name().to_ascii_lowercase().ends_with(".cnmt.xml"))
    else {
        return (None, None);
    };

    if entry.size() > MAX_AUXILIARY_METADATA_SIZE {
        return (
            None,
            Some("auxiliary CNMT XML exceeds the 1 MiB safety limit".to_owned()),
        );
    }

    let result = (|| {
        let storage = archive.open_entry(entry)?;
        let length = usize::try_from(entry.size())
            .map_err(|_| LoadError::invalid("CNMT XML", "size does not fit in memory"))?;
        let mut bytes = vec![0_u8; length];
        storage.read_at(0, &mut bytes)?;
        let xml = std::str::from_utf8(&bytes)
            .map_err(|_| LoadError::invalid("CNMT XML", "document is not valid UTF-8"))?;
        parse_content_meta_xml(xml)
            .ok_or_else(|| LoadError::invalid("CNMT XML", "required fields are missing"))
    })();

    match result {
        Ok(metadata) => (Some(metadata), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

fn parse_content_meta_xml(xml: &str) -> Option<ContentMetaInspection> {
    let content_type = tag_value(xml, "Type")?.to_owned();
    let title_id = parse_hex_u64(tag_value(xml, "Id")?)?;
    let version = tag_value(xml, "Version")?.parse().ok()?;
    let original_id = tag_value(xml, "OriginalId").and_then(parse_hex_u64);
    let application_id = tag_value(xml, "ApplicationId").and_then(parse_hex_u64);
    let minimum_key_generation =
        tag_value(xml, "KeyGenerationMin").and_then(|value| value.parse().ok());
    let required_system_version =
        tag_value(xml, "RequiredSystemVersion").and_then(|value| value.parse().ok());
    let required_application_version =
        tag_value(xml, "RequiredApplicationVersion").and_then(|value| value.parse().ok());
    let digest = tag_value(xml, "Digest").map(str::to_owned);
    let contents = element_values(xml, "Content")
        .filter_map(|content| {
            Some(ContentRecordInspection {
                content_type: tag_value(content, "Type")?.to_owned(),
                id: tag_value(content, "Id")?.to_owned(),
                size: tag_value(content, "Size")?.parse().ok()?,
                hash: tag_value(content, "Hash").map(str::to_owned),
                key_generation: tag_value(content, "KeyGeneration")
                    .and_then(|value| value.parse().ok()),
            })
        })
        .collect();

    Some(ContentMetaInspection {
        content_type,
        title_id,
        version,
        original_id,
        application_id,
        minimum_key_generation,
        required_system_version,
        required_application_version,
        contents,
        digest,
    })
}

fn tag_value<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?.checked_add(open.len())?;
    let end = start.checked_add(xml[start..].find(&close)?)?;
    Some(xml[start..end].trim())
}

fn element_values<'a>(xml: &'a str, tag: &'a str) -> impl Iterator<Item = &'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut remaining = xml;

    std::iter::from_fn(move || {
        let start = remaining.find(&open)?.checked_add(open.len())?;
        let end = start.checked_add(remaining[start..].find(&close)?)?;
        let value = &remaining[start..end];
        remaining = &remaining[end + close.len()..];
        Some(value)
    })
}

fn parse_hex_u64(value: &str) -> Option<u64> {
    u64::from_str_radix(value.trim_start_matches("0x"), 16).ok()
}

fn entry_kind(name: &str) -> EntryKind {
    let name = name.to_ascii_lowercase();
    if name.ends_with(".cnmt.nca") {
        EntryKind::MetaContentArchive
    } else if name.ends_with(".nca") {
        EntryKind::ContentArchive
    } else if name.ends_with(".tik") {
        EntryKind::Ticket
    } else if name.ends_with(".cert") {
        EntryKind::Certificate
    } else if name.ends_with(".xml") {
        EntryKind::XmlMetadata
    } else {
        EntryKind::Other
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_storage::{StorageError, StorageRef};

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

    #[test]
    fn classifies_common_nsp_entries() {
        assert_eq!(entry_kind("meta.cnmt.nca"), EntryKind::MetaContentArchive);
        assert_eq!(entry_kind("program.nca"), EntryKind::ContentArchive);
        assert_eq!(entry_kind("title.tik"), EntryKind::Ticket);
        assert_eq!(entry_kind("title.cert"), EntryKind::Certificate);
        assert_eq!(entry_kind("info.xml"), EntryKind::XmlMetadata);
        assert_eq!(entry_kind("readme.txt"), EntryKind::Other);
    }

    #[test]
    fn parses_auxiliary_content_meta_xml() {
        let xml = r#"
            <ContentMeta>
              <Type>Patch</Type>
              <Id>0x01002cd00a51c800</Id>
              <Version>786432</Version>
              <Content>
                <Type>Program</Type>
                <Id>0123456789abcdef0123456789abcdef</Id>
                <Size>42</Size>
                <Hash>abcdef</Hash>
                <KeyGeneration>7</KeyGeneration>
              </Content>
              <KeyGenerationMin>7</KeyGenerationMin>
              <RequiredSystemVersion>123</RequiredSystemVersion>
              <OriginalId>0x01002cd00a51c000</OriginalId>
            </ContentMeta>
        "#;

        let metadata = parse_content_meta_xml(xml).unwrap();

        assert_eq!(metadata.content_type, "Patch");
        assert_eq!(metadata.title_id, 0x0100_2cd0_0a51_c800);
        assert_eq!(metadata.version, 786_432);
        assert_eq!(metadata.original_id, Some(0x0100_2cd0_0a51_c000));
        assert_eq!(metadata.contents.len(), 1);
        assert_eq!(metadata.contents[0].size, 42);
    }

    #[test]
    fn captures_parsed_nca_header_information() {
        let mut bytes = vec![0_u8; 0xC00];
        bytes[0x200..0x204].copy_from_slice(b"NCA3");
        bytes[0x204] = 1;
        bytes[0x205] = 2;
        bytes[0x206] = 8;
        bytes[0x207] = 1;
        let archive_size = bytes.len() as u64;
        bytes[0x208..0x210].copy_from_slice(&archive_size.to_le_bytes());
        bytes[0x210..0x218].copy_from_slice(&0x0100_1234_5678_9000_u64.to_le_bytes());
        bytes[0x21C..0x220].copy_from_slice(&0x0012_0304_u32.to_le_bytes());
        let storage: StorageRef = Arc::new(VecStorage(bytes));

        let archive = NcaLoader::load(storage).unwrap();
        let inspection = inspect_nca(&archive);

        assert_eq!(inspection.format_version, NcaFormatVersion::Nca3);
        assert_eq!(inspection.distribution_type, NcaDistributionType::GameCard);
        assert_eq!(inspection.content_type, NcaContentType::Control);
        assert_eq!(inspection.title_id, 0x0100_1234_5678_9000);
        assert_eq!(inspection.key_generation, 7);
        assert_eq!(inspection.key_area_key_index, 1);
        assert!(inspection.source_is_decrypted);
        assert!(inspection.sections.is_empty());
    }
}
