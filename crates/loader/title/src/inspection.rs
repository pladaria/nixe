use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use swiitx_loader_content::{
    ApplicationVersion, CnmtContentMeta, CnmtMetaType, ContentMetaVersion,
    DecodedContentMetaVersion, NcaContentType, NcaDistributionType, NcaEncryptionType,
    NcaFormatVersion, NcaKeySet, NcaLoader, NcaSectionType, NczCompressionKind, NczLoader,
    NspLoader, NszLoader, SystemVersion, XciHeader, XciLoader, XciPartitionKind, XczLoader,
};
use swiitx_loader_executable::{NroLoader, NroRange};
use swiitx_loader_storage::{FileStorage, FormatLoader, LoadError, Storage, StorageRef};

const MAX_AUXILIARY_METADATA_SIZE: u64 = 1024 * 1024;

use crate::discovery::{directory_files, package_format};
use crate::package_content::{
    PackageContent, import_ticket_keys, load_canonical_content_meta, load_canonical_content_metas,
    load_control_metadata,
};
use crate::{ControlMetadata, DirectoryScanOptions, PackageFormat};

impl Display for PackageFormat {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nsp => formatter.write_str("NSP (PFS0)"),
            Self::Nsz => formatter.write_str("NSZ (PFS0 + NCZ)"),
            Self::Xci => formatter.write_str("XCI (HFS0)"),
            Self::Xcz => formatter.write_str("XCZ (HFS0 + NCZ)"),
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
    /// Stored name when it differs from the logical entry name.
    pub stored_name: Option<String>,
    /// Stored compressed size when it differs from the logical size.
    pub stored_size: Option<u64>,
    /// NCZ compression and reconstruction metadata, when applicable.
    pub ncz: Option<NczInspection>,
    /// Advertised HFS0 hashed prefix, when the entry comes from XCI.
    pub hashed_region_size: Option<u64>,
    /// Result of validating the advertised HFS0 prefix, when applicable.
    pub hash_valid: Option<bool>,
    /// Parsed NCA header and section layout for content-archive entries.
    pub nca: Option<NcaInspection>,
    /// Explanation when an NCA entry could not be inspected.
    pub nca_warning: Option<String>,
}

/// Non-sensitive metadata parsed from an NCZ entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NczInspection {
    pub compression: NczCompressionKind,
    pub logical_size: u64,
    pub block_size: Option<u64>,
    pub block_count: Option<u32>,
    pub sections: Vec<NczSectionInspection>,
}

/// Non-sensitive reconstruction range from an NCZ section table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NczSectionInspection {
    pub offset: u64,
    pub size: u64,
    pub crypto_type: u64,
}

/// Inspection result for an explicitly supplied standalone NCZ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneNczInspection {
    pub path: PathBuf,
    pub stored_size: u64,
    pub ncz: NczInspection,
    pub nca: Option<NcaInspection>,
    pub nca_warning: Option<String>,
}

/// Semantic role of a validated NRO segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NroSegmentKind {
    Text,
    ReadOnly,
    Data,
}

impl Display for NroSegmentKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Text => "text",
            Self::ReadOnly => "read-only",
            Self::Data => "data",
        })
    }
}

/// One validated loadable segment from a standalone NRO executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NroSegmentInspection {
    pub kind: NroSegmentKind,
    pub memory_offset: u64,
    pub file_size: u64,
    pub memory_size: u64,
    pub mapping_size: u64,
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

/// Optional homebrew assets appended to a standalone NRO executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NroAssetsInspection {
    pub version: u32,
    pub icon_size: Option<u64>,
    pub nacp_size: Option<u64>,
    pub romfs_size: Option<u64>,
}

/// Validated metadata from a standalone NRO executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NroInspection {
    pub path: PathBuf,
    pub size: u64,
    pub version: u32,
    pub flags: u32,
    pub executable_size: u64,
    pub entry_offset: u64,
    pub module_id: [u8; 32],
    pub module_header_offset: u64,
    pub dso_handle_offset: u64,
    pub embedded_api_info: (u64, u64),
    pub dynamic_string_table: (u64, u64),
    pub dynamic_symbol_table: (u64, u64),
    pub segments: Vec<NroSegmentInspection>,
    pub assets: Option<NroAssetsInspection>,
}

/// One nested partition reported while inspecting an XCI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XciPartitionInspection {
    pub name: String,
    pub kind: XciPartitionKind,
    pub offset: u64,
    pub size: u64,
    pub hashed_region_size: u64,
    pub hash_valid: Option<bool>,
    pub data_offset: u64,
    pub entries: Vec<EntryInspection>,
}

/// Header and partition information specific to an XCI container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XciInspection {
    pub header: XciHeader,
    pub root_header_hash_valid: Option<bool>,
    pub root_data_offset: u64,
    pub partitions: Vec<XciPartitionInspection>,
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
    pub version: ContentMetaVersion,
    /// Base title associated with a patch, when declared.
    pub original_id: Option<u64>,
    /// Application associated with add-on content, when declared.
    pub application_id: Option<u64>,
    /// Minimum key generation among the package contents, when declared.
    pub minimum_key_generation: Option<u32>,
    /// Required system version, when declared.
    pub required_system_version: Option<SystemVersion>,
    /// Required application version, when declared.
    pub required_application_version: Option<ApplicationVersion>,
    /// Content records listed by the metadata.
    pub contents: Vec<ContentRecordInspection>,
    /// Overall metadata digest, when present.
    pub digest: Option<String>,
}

impl ContentMetaInspection {
    /// Decodes the auxiliary version only when its textual type is recognized.
    pub fn decoded_version(&self) -> DecodedContentMetaVersion {
        match auxiliary_content_meta_type(&self.content_type) {
            Some(content_meta_type) => self.version.decode(content_meta_type),
            None => DecodedContentMetaVersion::Unknown(self.version),
        }
    }
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
    /// XCI-specific header and nested partition information.
    pub xci: Option<XciInspection>,
    /// Files stored in the package.
    pub entries: Vec<EntryInspection>,
    /// Canonical binary content metadata read from the package's meta NCA.
    pub canonical_content_meta: Option<CnmtContentMeta>,
    /// Every canonical CNMT found in deterministic container order.
    pub canonical_content_metas: Vec<CnmtContentMeta>,
    /// Explanation when the meta NCA or its binary CNMT could not be read and
    /// validated.
    pub canonical_metadata_warning: Option<String>,
    /// Localized titles, icons, and application properties from the Control NCA.
    pub control_metadata: Option<ControlMetadata>,
    /// Control metadata corresponding to every readable canonical CNMT.
    pub control_metadatas: Vec<ControlMetadata>,
    /// Explanation when canonical CNMT declares Control data that could not be read.
    pub control_metadata_warning: Option<String>,
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
        if let Some(xci) = &self.xci {
            return xci.partitions.iter().fold(0_u64, |total, partition| {
                total.saturating_add(partition.size)
            });
        }
        self.entries.iter().fold(0_u64, |total, entry| {
            total.saturating_add(entry.stored_size.unwrap_or(entry.size))
        })
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
    /// Standalone NRO executables successfully inspected below the path.
    pub nros: Vec<NroInspection>,
    /// Regular files skipped because their format is not supported yet.
    pub ignored_files: Vec<PathBuf>,
    /// Standalone NCZ information when the explicit input is an `.ncz` file.
    pub standalone_ncz: Option<StandaloneNczInspection>,
}

/// Inspects title packages and standalone NRO executables lazily.
#[derive(Debug)]
pub struct TitleInspector;

impl TitleInspector {
    /// Inspects multiple package files or directories in caller-provided order.
    pub fn inspect_paths<P, I>(paths: I) -> Result<Vec<TitleInspection>, InspectError>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        Self::inspect_paths_with_options(paths, DirectoryScanOptions::default())
    }

    /// Inspects multiple paths using the supplied directory options.
    pub fn inspect_paths_with_options<P, I>(
        paths: I,
        options: DirectoryScanOptions,
    ) -> Result<Vec<TitleInspection>, InspectError>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        let paths: Vec<PathBuf> = paths
            .into_iter()
            .map(|path| path.as_ref().to_owned())
            .collect();
        Self::inspect_paths_impl(paths, None, options)
    }

    /// Inspects multiple paths with one shared caller-owned key set.
    pub fn inspect_paths_with_key_set<P, I>(
        paths: I,
        keys: &mut NcaKeySet,
    ) -> Result<Vec<TitleInspection>, InspectError>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        Self::inspect_paths_with_key_set_and_options(paths, keys, DirectoryScanOptions::default())
    }

    /// Inspects multiple paths with one shared caller-owned key set.
    ///
    /// Ticket keys imported while inspecting an earlier path remain available
    /// to every later path in the caller-provided order.
    pub fn inspect_paths_with_key_set_and_options<P, I>(
        paths: I,
        keys: &mut NcaKeySet,
        options: DirectoryScanOptions,
    ) -> Result<Vec<TitleInspection>, InspectError>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        let paths: Vec<PathBuf> = paths
            .into_iter()
            .map(|path| path.as_ref().to_owned())
            .collect();
        Self::inspect_paths_impl(paths, Some(keys), options)
    }

    /// Inspects one package file or recursively scans a directory.
    pub fn inspect(path: impl AsRef<Path>) -> Result<TitleInspection, InspectError> {
        Self::inspect_with_options(path, DirectoryScanOptions::default())
    }

    /// Inspects a package path using the supplied directory options.
    pub fn inspect_with_options(
        path: impl AsRef<Path>,
        options: DirectoryScanOptions,
    ) -> Result<TitleInspection, InspectError> {
        Self::inspect_impl(path.as_ref(), None, options)
    }

    /// Inspects title packages and decrypts their NCAs with caller-owned keys.
    /// Encrypted title keys present in NSP tickets are imported into the key set.
    pub fn inspect_with_key_set(
        path: impl AsRef<Path>,
        keys: &mut NcaKeySet,
    ) -> Result<TitleInspection, InspectError> {
        Self::inspect_with_key_set_and_options(path, keys, DirectoryScanOptions::default())
    }

    /// Inspects a package path with keys and the supplied directory options.
    pub fn inspect_with_key_set_and_options(
        path: impl AsRef<Path>,
        keys: &mut NcaKeySet,
        options: DirectoryScanOptions,
    ) -> Result<TitleInspection, InspectError> {
        Self::inspect_impl(path.as_ref(), Some(keys), options)
    }

    /// Finds and validates standalone NRO executables without inspecting title packages.
    pub fn inspect_nros_with_options(
        path: impl AsRef<Path>,
        options: DirectoryScanOptions,
    ) -> Result<Vec<NroInspection>, InspectError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|source| InspectError::Io {
            path: path.to_owned(),
            source,
        })?;
        let candidates = if metadata.is_file() {
            vec![path.to_owned()]
        } else if metadata.is_dir() {
            directory_files(path, options).map_err(|error| InspectError::Io {
                path: error.path,
                source: error.source,
            })?
        } else {
            return Err(InspectError::UnsupportedPath(path.to_owned()));
        };

        candidates
            .into_iter()
            .filter(|candidate| has_extension(candidate, "nro"))
            .map(|candidate| inspect_nro(&candidate))
            .collect()
    }

    fn inspect_paths_impl(
        paths: Vec<PathBuf>,
        mut keys: Option<&mut NcaKeySet>,
        options: DirectoryScanOptions,
    ) -> Result<Vec<TitleInspection>, InspectError> {
        if paths.is_empty() {
            return Err(InspectError::NoPaths);
        }
        let mut inspections = Vec::with_capacity(paths.len());
        for path in paths {
            inspections.push(Self::inspect_impl(&path, keys.as_deref_mut(), options)?);
        }
        deduplicate_inspections(&mut inspections)?;
        Ok(inspections)
    }

    fn inspect_impl(
        path: &Path,
        mut keys: Option<&mut NcaKeySet>,
        options: DirectoryScanOptions,
    ) -> Result<TitleInspection, InspectError> {
        let metadata = fs::metadata(path).map_err(|source| InspectError::Io {
            path: path.to_owned(),
            source,
        })?;

        if metadata.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("ncz"))
        {
            return Ok(TitleInspection {
                path: path.to_owned(),
                packages: Vec::new(),
                nros: Vec::new(),
                ignored_files: Vec::new(),
                standalone_ncz: Some(inspect_standalone_ncz(path, keys.as_deref())?),
            });
        }

        let candidates = if metadata.is_file() {
            vec![path.to_owned()]
        } else if metadata.is_dir() {
            directory_files(path, options).map_err(|error| InspectError::Io {
                path: error.path,
                source: error.source,
            })?
        } else {
            return Err(InspectError::UnsupportedPath(path.to_owned()));
        };

        let mut packages = Vec::new();
        let mut nros = Vec::new();
        let mut ignored_files = Vec::new();
        for candidate in candidates {
            match package_format(&candidate) {
                Some(PackageFormat::Nsp) => {
                    packages.push(inspect_nsp(&candidate, keys.as_deref_mut())?)
                }
                Some(PackageFormat::Nsz) => {
                    packages.push(inspect_nsz(&candidate, keys.as_deref_mut())?)
                }
                Some(PackageFormat::Xci) => {
                    packages.push(inspect_xci(&candidate, keys.as_deref_mut())?)
                }
                Some(PackageFormat::Xcz) => {
                    packages.push(inspect_xcz(&candidate, keys.as_deref_mut())?)
                }
                None if has_extension(&candidate, "nro") => nros.push(inspect_nro(&candidate)?),
                None => ignored_files.push(candidate),
            }
        }

        if packages.is_empty() && nros.is_empty() {
            return Err(InspectError::NoSupportedPackages(path.to_owned()));
        }

        Ok(TitleInspection {
            path: path.to_owned(),
            packages,
            nros,
            ignored_files,
            standalone_ncz: None,
        })
    }
}

fn deduplicate_inspections(inspections: &mut [TitleInspection]) -> Result<(), InspectError> {
    let mut seen = BTreeSet::new();
    for inspection in inspections {
        let mut packages = Vec::with_capacity(inspection.packages.len());
        for package in std::mem::take(&mut inspection.packages) {
            if insert_unique_path(&mut seen, &package.path)? {
                packages.push(package);
            }
        }
        inspection.packages = packages;

        let mut nros = Vec::with_capacity(inspection.nros.len());
        for nro in std::mem::take(&mut inspection.nros) {
            if insert_unique_path(&mut seen, &nro.path)? {
                nros.push(nro);
            }
        }
        inspection.nros = nros;

        let mut ignored = Vec::with_capacity(inspection.ignored_files.len());
        for path in std::mem::take(&mut inspection.ignored_files) {
            if insert_unique_path(&mut seen, &path)? {
                ignored.push(path);
            }
        }
        inspection.ignored_files = ignored;

        let keep_standalone = inspection
            .standalone_ncz
            .as_ref()
            .map(|ncz| insert_unique_path(&mut seen, &ncz.path))
            .transpose()?;
        if keep_standalone == Some(false) {
            inspection.standalone_ncz = None;
        }
    }
    Ok(())
}

fn insert_unique_path(seen: &mut BTreeSet<PathBuf>, path: &Path) -> Result<bool, InspectError> {
    let identity = fs::canonicalize(path).map_err(|source| InspectError::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(seen.insert(identity))
}

/// Errors produced while inspecting a local title path.
#[derive(Debug)]
pub enum InspectError {
    /// The caller supplied no inspection roots.
    NoPaths,
    /// A local file-system operation failed.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A supported input could not be parsed.
    Package { path: PathBuf, source: LoadError },
    /// The supplied path is neither a regular file nor a directory.
    UnsupportedPath(PathBuf),
    /// No supported package or executable was found at the supplied path.
    NoSupportedPackages(PathBuf),
}

impl Display for InspectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPaths => formatter.write_str("no inspection paths were supplied"),
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
                "no supported title packages or NRO executables found at {}",
                path.display()
            ),
        }
    }
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

fn inspect_nro(path: &Path) -> Result<NroInspection, InspectError> {
    let storage = FileStorage::open(path).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source: LoadError::Storage(source),
    })?;
    let size = storage.len().map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source: LoadError::Storage(source),
    })?;
    let image = NroLoader::load(Arc::new(storage)).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source,
    })?;
    let executable = image.executable();
    let metadata = image.metadata();
    let segments = executable
        .segments()
        .iter()
        .map(|segment| {
            let permissions = segment.permissions();
            NroSegmentInspection {
                kind: match segment.kind() {
                    swiitx_loader_executable::ExecutableSegmentKind::Text => NroSegmentKind::Text,
                    swiitx_loader_executable::ExecutableSegmentKind::ReadOnly => {
                        NroSegmentKind::ReadOnly
                    }
                    swiitx_loader_executable::ExecutableSegmentKind::Data => NroSegmentKind::Data,
                },
                memory_offset: segment.memory_offset(),
                file_size: segment.file_size(),
                memory_size: segment.memory_size(),
                mapping_size: segment.mapping_size(),
                readable: permissions.is_readable(),
                writable: permissions.is_writable(),
                executable: permissions.is_executable(),
            }
        })
        .collect();
    let assets = image
        .assets()
        .map(|assets| -> Result<_, LoadError> {
            Ok(NroAssetsInspection {
                version: assets.version(),
                icon_size: asset_size(assets.icon())?,
                nacp_size: asset_size(assets.nacp())?,
                romfs_size: asset_size(assets.romfs())?,
            })
        })
        .transpose()
        .map_err(|source| InspectError::Package {
            path: path.to_owned(),
            source,
        })?;

    Ok(NroInspection {
        path: path.to_owned(),
        size,
        version: metadata.version(),
        flags: metadata.flags(),
        executable_size: metadata.executable_size(),
        entry_offset: executable.entry_offset(),
        module_id: *executable.module_id(),
        module_header_offset: metadata.module_header_offset(),
        dso_handle_offset: metadata.dso_handle_offset(),
        embedded_api_info: range_tuple(metadata.embedded_api_info()),
        dynamic_string_table: range_tuple(metadata.dynamic_string_table()),
        dynamic_symbol_table: range_tuple(metadata.dynamic_symbol_table()),
        segments,
        assets,
    })
}

fn range_tuple(range: NroRange) -> (u64, u64) {
    (range.offset(), range.size())
}

fn asset_size(asset: Option<&StorageRef>) -> Result<Option<u64>, LoadError> {
    asset
        .map(|storage| storage.len().map_err(LoadError::Storage))
        .transpose()
}

impl Error for InspectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Package { source, .. } => Some(source),
            Self::NoPaths | Self::UnsupportedPath(_) | Self::NoSupportedPackages(_) => None,
        }
    }
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
            stored_name: None,
            stored_size: None,
            ncz: None,
            hashed_region_size: None,
            hash_valid: None,
            nca,
            nca_warning,
        });
    }
    let (canonical_content_meta, canonical_metadata_warning) =
        inspect_canonical_metadata(&archive, keys.as_deref());
    let (control_metadata, control_metadata_warning) =
        canonical_content_meta
            .as_ref()
            .map_or((None, None), |content_meta| {
                match load_control_metadata(
                    &archive,
                    content_meta,
                    keys.as_deref().map(|keys| keys as _),
                ) {
                    Ok(metadata) => (metadata, None),
                    Err(error) => (None, Some(error.to_string())),
                }
            });
    let (content_meta, metadata_warning) = inspect_auxiliary_metadata(&archive);
    let control_metadatas = control_metadata.clone().into_iter().collect();

    Ok(PackageInspection {
        path: path.to_owned(),
        format: PackageFormat::Nsp,
        size,
        data_offset: archive.data_offset(),
        xci: None,
        entries,
        canonical_content_metas: canonical_content_meta.clone().into_iter().collect(),
        canonical_content_meta,
        canonical_metadata_warning,
        control_metadata,
        control_metadatas,
        control_metadata_warning,
        content_meta,
        metadata_warning,
        ticket_warnings,
    })
}

fn inspect_nsz(
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
    let archive = NszLoader::load(storage).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source,
    })?;
    let ticket_warnings = keys
        .as_deref_mut()
        .map_or_else(Vec::new, |keys| import_ticket_keys(&archive, keys));
    let mut entries = Vec::with_capacity(archive.entries().len());
    for entry in archive.entries() {
        let kind = entry_kind(entry.logical_name());
        let (nca, nca_warning) =
            inspect_entry_nca(entry.open(), kind, keys.as_deref().map(|keys| keys as _));
        entries.push(EntryInspection {
            name: entry.logical_name().to_owned(),
            kind,
            offset: entry.stored_offset(),
            size: entry.logical_size(),
            stored_name: (entry.stored_name() != entry.logical_name())
                .then(|| entry.stored_name().to_owned()),
            stored_size: (entry.stored_size() != entry.logical_size())
                .then_some(entry.stored_size()),
            ncz: entry.ncz().map(inspect_ncz_metadata),
            hashed_region_size: None,
            hash_valid: None,
            nca,
            nca_warning,
        });
    }
    let (canonical_content_meta, canonical_metadata_warning) =
        inspect_canonical_metadata(&archive, keys.as_deref());
    let (control_metadata, control_metadata_warning) =
        canonical_content_meta
            .as_ref()
            .map_or((None, None), |content_meta| {
                match load_control_metadata(
                    &archive,
                    content_meta,
                    keys.as_deref().map(|keys| keys as _),
                ) {
                    Ok(metadata) => (metadata, None),
                    Err(error) => (None, Some(error.to_string())),
                }
            });
    let (content_meta, metadata_warning) = inspect_auxiliary_metadata(&archive);
    let control_metadatas = control_metadata.clone().into_iter().collect();

    Ok(PackageInspection {
        path: path.to_owned(),
        format: PackageFormat::Nsz,
        size,
        data_offset: archive.data_offset(),
        xci: None,
        entries,
        canonical_content_metas: canonical_content_meta.clone().into_iter().collect(),
        canonical_content_meta,
        canonical_metadata_warning,
        control_metadata,
        control_metadatas,
        control_metadata_warning,
        content_meta,
        metadata_warning,
        ticket_warnings,
    })
}

fn inspect_xci(
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
    let archive = XciLoader::load(storage).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source,
    })?;

    let mut partition_inspections = Vec::with_capacity(archive.partitions().len());
    for partition in archive.partitions() {
        let mut entries = Vec::with_capacity(partition.archive().entries().len());
        for entry in partition.archive().entries() {
            let integrity = partition
                .archive()
                .validate_entry(entry)
                .map_err(|source| InspectError::Package {
                    path: path.to_owned(),
                    source,
                })?;
            let kind = entry_kind(entry.name());
            let (nca, nca_warning) = if matches!(
                kind,
                EntryKind::MetaContentArchive | EntryKind::ContentArchive
            ) {
                let result = partition.archive().open_entry(entry).and_then(|storage| {
                    match keys.as_deref() {
                        Some(keys) => NcaLoader::load_with_key_provider(storage, keys),
                        None => NcaLoader::load(storage),
                    }
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
                stored_name: None,
                stored_size: None,
                ncz: None,
                hashed_region_size: Some(entry.hashed_region_size()),
                hash_valid: Some(integrity.is_valid()),
                nca,
                nca_warning,
            });
        }
        partition_inspections.push(XciPartitionInspection {
            name: partition.name().to_owned(),
            kind: partition.kind().clone(),
            offset: partition.root_entry().offset(),
            size: partition.root_entry().size(),
            hashed_region_size: partition.root_entry().hashed_region_size(),
            hash_valid: partition
                .root_entry_integrity()
                .map(|integrity| integrity.is_valid()),
            data_offset: partition.archive().data_offset(),
            entries,
        });
    }

    let secure = archive
        .partition(&XciPartitionKind::Secure)
        .map(|partition| partition.archive());
    let ticket_warnings = match (secure, keys.as_deref_mut()) {
        (Some(secure), Some(keys)) => import_ticket_keys(secure, keys),
        _ => Vec::new(),
    };
    let key_provider = keys.as_deref().map(|keys| keys as _);
    let (canonical_content_metas, canonical_metadata_warning) = match secure {
        Some(secure) => match load_canonical_content_metas(secure, key_provider) {
            Ok(metadata) => (metadata, None),
            Err(error) => (Vec::new(), Some(error.to_string())),
        },
        None => (
            Vec::new(),
            Some("XCI has no secure partition; no title metadata was loaded".to_owned()),
        ),
    };
    let canonical_content_meta = canonical_content_metas.first().cloned();
    let mut control_metadatas = Vec::new();
    let mut control_warnings = Vec::new();
    if let Some(secure) = secure {
        for content_meta in &canonical_content_metas {
            match load_control_metadata(secure, content_meta, key_provider) {
                Ok(Some(metadata)) => control_metadatas.push(metadata),
                Ok(None) => {}
                Err(error) => {
                    control_warnings.push(format!("{:016X}: {error}", content_meta.title_id));
                }
            }
        }
    }
    let control_metadata = control_metadatas.first().cloned();
    let control_metadata_warning =
        (!control_warnings.is_empty()).then(|| control_warnings.join("; "));
    let entries = partition_inspections
        .iter()
        .find(|partition| partition.kind == XciPartitionKind::Secure)
        .map_or_else(Vec::new, |partition| partition.entries.clone());
    let xci = XciInspection {
        header: archive.header().clone(),
        root_header_hash_valid: archive
            .root_header_integrity()
            .map(|integrity| integrity.is_valid()),
        root_data_offset: archive.root().data_offset(),
        partitions: partition_inspections,
    };

    Ok(PackageInspection {
        path: path.to_owned(),
        format: PackageFormat::Xci,
        size,
        data_offset: secure.map_or(archive.root().data_offset(), |secure| secure.data_offset()),
        xci: Some(xci),
        entries,
        canonical_content_meta,
        canonical_content_metas,
        canonical_metadata_warning,
        control_metadata,
        control_metadatas,
        control_metadata_warning,
        content_meta: None,
        metadata_warning: None,
        ticket_warnings,
    })
}

fn inspect_xcz(
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
    let archive = XczLoader::load(storage).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source,
    })?;
    let xci_archive = archive.xci();

    let mut partition_inspections = Vec::with_capacity(archive.partitions().len());
    for (partition, stored_partition) in archive.partitions().iter().zip(xci_archive.partitions()) {
        let mut entries = Vec::with_capacity(partition.entries().len());
        for (entry, stored) in partition
            .entries()
            .iter()
            .zip(stored_partition.archive().entries())
        {
            let hash_valid = if stored.has_advertised_hash() {
                let integrity =
                    stored_partition
                        .archive()
                        .validate_entry(stored)
                        .map_err(|source| InspectError::Package {
                            path: path.to_owned(),
                            source,
                        })?;
                Some(integrity.is_valid())
            } else {
                None
            };
            let kind = entry_kind(entry.logical_name());
            let (nca, nca_warning) =
                inspect_entry_nca(entry.open(), kind, keys.as_deref().map(|keys| keys as _));
            entries.push(EntryInspection {
                name: entry.logical_name().to_owned(),
                kind,
                offset: entry.stored_offset(),
                size: entry.logical_size(),
                stored_name: (entry.stored_name() != entry.logical_name())
                    .then(|| entry.stored_name().to_owned()),
                stored_size: (entry.stored_size() != entry.logical_size())
                    .then_some(entry.stored_size()),
                ncz: entry.ncz().map(inspect_ncz_metadata),
                hashed_region_size: Some(stored.hashed_region_size()),
                hash_valid,
                nca,
                nca_warning,
            });
        }
        partition_inspections.push(XciPartitionInspection {
            name: stored_partition.name().to_owned(),
            kind: stored_partition.kind().clone(),
            offset: stored_partition.root_entry().offset(),
            size: stored_partition.root_entry().size(),
            hashed_region_size: stored_partition.root_entry().hashed_region_size(),
            hash_valid: stored_partition
                .root_entry_integrity()
                .map(|integrity| integrity.is_valid()),
            data_offset: stored_partition.archive().data_offset(),
            entries,
        });
    }

    let secure = archive.partition(&XciPartitionKind::Secure);
    let ticket_warnings = match (secure, keys.as_deref_mut()) {
        (Some(secure), Some(keys)) => import_ticket_keys(secure, keys),
        _ => Vec::new(),
    };
    let key_provider = keys.as_deref().map(|keys| keys as _);
    let (canonical_content_metas, canonical_metadata_warning) = match secure {
        Some(secure) => match load_canonical_content_metas(secure, key_provider) {
            Ok(metadata) => (metadata, None),
            Err(error) => (Vec::new(), Some(error.to_string())),
        },
        None => (
            Vec::new(),
            Some("XCZ has no secure partition; no title metadata was loaded".to_owned()),
        ),
    };
    let canonical_content_meta = canonical_content_metas.first().cloned();
    let mut control_metadatas = Vec::new();
    let mut control_warnings = Vec::new();
    if let Some(secure) = secure {
        for content_meta in &canonical_content_metas {
            match load_control_metadata(secure, content_meta, key_provider) {
                Ok(Some(metadata)) => control_metadatas.push(metadata),
                Ok(None) => {}
                Err(error) => {
                    control_warnings.push(format!("{:016X}: {error}", content_meta.title_id));
                }
            }
        }
    }
    let control_metadata = control_metadatas.first().cloned();
    let control_metadata_warning =
        (!control_warnings.is_empty()).then(|| control_warnings.join("; "));
    let entries = partition_inspections
        .iter()
        .find(|partition| partition.kind == XciPartitionKind::Secure)
        .map_or_else(Vec::new, |partition| partition.entries.clone());
    let xci = XciInspection {
        header: xci_archive.header().clone(),
        root_header_hash_valid: xci_archive
            .root_header_integrity()
            .map(|integrity| integrity.is_valid()),
        root_data_offset: xci_archive.root().data_offset(),
        partitions: partition_inspections,
    };
    let data_offset = xci_archive
        .partition(&XciPartitionKind::Secure)
        .map_or(xci_archive.root().data_offset(), |partition| {
            partition.archive().data_offset()
        });

    Ok(PackageInspection {
        path: path.to_owned(),
        format: PackageFormat::Xcz,
        size,
        data_offset,
        xci: Some(xci),
        entries,
        canonical_content_meta,
        canonical_content_metas,
        canonical_metadata_warning,
        control_metadata,
        control_metadatas,
        control_metadata_warning,
        content_meta: None,
        metadata_warning: None,
        ticket_warnings,
    })
}

fn inspect_standalone_ncz(
    path: &Path,
    keys: Option<&NcaKeySet>,
) -> Result<StandaloneNczInspection, InspectError> {
    let storage = FileStorage::open(path).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source: LoadError::Storage(source),
    })?;
    let stored_size = storage.len().map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source: LoadError::Storage(source),
    })?;
    let archive = NczLoader::load(Arc::new(storage)).map_err(|source| InspectError::Package {
        path: path.to_owned(),
        source,
    })?;
    let ncz = inspect_ncz_metadata(&archive);
    let result = match keys {
        Some(keys) => NcaLoader::load_with_key_provider(archive.nca_storage(), keys),
        None => NcaLoader::load(archive.nca_storage()),
    };
    let (nca, nca_warning) = match result {
        Ok(archive) => (Some(inspect_nca(&archive)), None),
        Err(error) => (None, Some(error.to_string())),
    };
    Ok(StandaloneNczInspection {
        path: path.to_owned(),
        stored_size,
        ncz,
        nca,
        nca_warning,
    })
}

fn inspect_entry_nca(
    storage: Result<StorageRef, LoadError>,
    kind: EntryKind,
    keys: Option<&dyn swiitx_loader_content::NcaKeyProvider>,
) -> (Option<NcaInspection>, Option<String>) {
    if !matches!(
        kind,
        EntryKind::MetaContentArchive | EntryKind::ContentArchive
    ) {
        return (None, None);
    }
    let result = storage.and_then(|storage| match keys {
        Some(keys) => NcaLoader::load_with_key_provider(storage, keys),
        None => NcaLoader::load(storage),
    });
    match result {
        Ok(archive) => (Some(inspect_nca(&archive)), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

fn inspect_ncz_metadata(archive: &swiitx_loader_content::NczArchive) -> NczInspection {
    NczInspection {
        compression: archive.compression_kind(),
        logical_size: archive.logical_size(),
        block_size: archive.block_info().map(|info| info.block_size()),
        block_count: archive.block_info().map(|info| info.block_count()),
        sections: archive
            .sections()
            .iter()
            .map(|section| NczSectionInspection {
                offset: section.offset(),
                size: section.size(),
                crypto_type: section.crypto_type(),
            })
            .collect(),
    }
}

fn inspect_canonical_metadata<C: PackageContent + ?Sized>(
    archive: &C,
    keys: Option<&NcaKeySet>,
) -> (Option<CnmtContentMeta>, Option<String>) {
    match load_canonical_content_meta(archive, keys.map(|keys| keys as _)) {
        Ok(metadata) => (Some(metadata), None),
        Err(error) => (None, Some(error.to_string())),
    }
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

fn inspect_auxiliary_metadata<C: PackageContent + ?Sized>(
    archive: &C,
) -> (Option<ContentMetaInspection>, Option<String>) {
    let Some(index) = (0..archive.entry_count()).find(|index| {
        archive
            .entry_name(*index)
            .to_ascii_lowercase()
            .ends_with(".cnmt.xml")
    }) else {
        return (None, None);
    };

    if archive.entry_size(index) > MAX_AUXILIARY_METADATA_SIZE {
        return (
            None,
            Some("auxiliary CNMT XML exceeds the 1 MiB safety limit".to_owned()),
        );
    }

    let result = (|| {
        let storage = archive.open_entry_at(index)?;
        let length = usize::try_from(archive.entry_size(index))
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
    let version = ContentMetaVersion::from_raw(tag_value(xml, "Version")?.parse().ok()?);
    let original_id = tag_value(xml, "OriginalId").and_then(parse_hex_u64);
    let application_id = tag_value(xml, "ApplicationId").and_then(parse_hex_u64);
    let minimum_key_generation =
        tag_value(xml, "KeyGenerationMin").and_then(|value| value.parse().ok());
    let required_system_version = tag_value(xml, "RequiredSystemVersion")
        .and_then(|value| value.parse().ok())
        .map(SystemVersion::from_raw);
    let required_application_version = tag_value(xml, "RequiredApplicationVersion")
        .and_then(|value| value.parse().ok())
        .map(ApplicationVersion::from_raw);
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

fn auxiliary_content_meta_type(content_type: &str) -> Option<CnmtMetaType> {
    match content_type {
        "SystemProgram" => Some(CnmtMetaType::SystemProgram),
        "SystemData" => Some(CnmtMetaType::SystemData),
        "SystemUpdate" => Some(CnmtMetaType::SystemUpdate),
        "BootImagePackage" => Some(CnmtMetaType::BootImagePackage),
        "BootImagePackageSafe" => Some(CnmtMetaType::BootImagePackageSafe),
        "Application" => Some(CnmtMetaType::Application),
        "Patch" => Some(CnmtMetaType::Patch),
        "AddOnContent" => Some(CnmtMetaType::AddOnContent),
        "Delta" => Some(CnmtMetaType::Delta),
        "DataPatch" => Some(CnmtMetaType::DataPatch),
        _ => None,
    }
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
    use std::io::Write;
    use std::sync::Arc;

    use sha2::{Digest, Sha256};
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

    fn synthetic_nro() -> Vec<u8> {
        let mut bytes = vec![0_u8; 0x2800];
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        put_u32(&mut bytes, 0x04, 0x100);
        put_u32(&mut bytes, 0x14, 7);
        put_u32(&mut bytes, 0x18, 0x2800);
        put_u32(&mut bytes, 0x1c, 0x55aa);
        put_u32(&mut bytes, 0x20, 0);
        put_u32(&mut bytes, 0x24, 0x1000);
        put_u32(&mut bytes, 0x28, 0x1000);
        put_u32(&mut bytes, 0x2c, 0x1000);
        put_u32(&mut bytes, 0x30, 0x2000);
        put_u32(&mut bytes, 0x34, 0x800);
        put_u32(&mut bytes, 0x38, 0x800);
        for (index, byte) in bytes[0x40..0x60].iter_mut().enumerate() {
            *byte = u8::try_from(index).unwrap();
        }
        put_u32(&mut bytes, 0x60, 0x2040);
        put_u32(&mut bytes, 0x68, 0x1100);
        put_u32(&mut bytes, 0x6c, 0x20);
        put_u32(&mut bytes, 0x70, 0x1200);
        put_u32(&mut bytes, 0x74, 0x30);
        put_u32(&mut bytes, 0x78, 0x1300);
        put_u32(&mut bytes, 0x7c, 0x48);
        bytes
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
    fn explicitly_inspects_a_standalone_ncz_without_cataloguing_it_as_a_package() {
        let tail = vec![0x5a; 512];
        let mut bytes = vec![0_u8; 0x4000];
        bytes.extend_from_slice(b"NCZSECTN");
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0x4000_u64.to_le_bytes());
        bytes.extend_from_slice(&(tail.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(&zstd_bytes(&tail));
        let mut file = tempfile::Builder::new().suffix(".ncz").tempfile().unwrap();
        file.write_all(&bytes).unwrap();

        let inspection = TitleInspector::inspect(file.path()).unwrap();
        assert!(inspection.packages.is_empty());
        assert!(inspection.ignored_files.is_empty());
        let standalone = inspection.standalone_ncz.unwrap();
        assert_eq!(standalone.ncz.compression, NczCompressionKind::Solid);
        assert_eq!(standalone.ncz.logical_size, 0x4000 + tail.len() as u64);
        assert!(standalone.nca.is_none());
        assert!(standalone.nca_warning.is_some());
    }

    #[test]
    fn inspects_a_standalone_nro_as_a_supported_executable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("homebrew.NRO");
        fs::write(&path, synthetic_nro()).unwrap();

        let inspection = TitleInspector::inspect(&path).unwrap();

        assert!(inspection.packages.is_empty());
        assert!(inspection.ignored_files.is_empty());
        assert!(inspection.standalone_ncz.is_none());
        assert_eq!(inspection.nros.len(), 1);
        let nro = &inspection.nros[0];
        assert_eq!(nro.path, path);
        assert_eq!(nro.size, 0x2800);
        assert_eq!(nro.executable_size, 0x2800);
        assert_eq!(nro.module_header_offset, 0x100);
        assert_eq!(nro.segments.len(), 3);
        assert_eq!(nro.segments[0].kind, NroSegmentKind::Text);
        assert!(nro.segments[0].executable);
        assert_eq!(nro.segments[2].kind, NroSegmentKind::Data);
        assert!(nro.segments[2].writable);
    }

    #[test]
    fn directory_inspection_includes_nros_instead_of_ignoring_them() {
        let directory = tempfile::tempdir().unwrap();
        let nro_path = directory.path().join("homebrew.nro");
        let ignored_path = directory.path().join("readme.txt");
        fs::write(&nro_path, synthetic_nro()).unwrap();
        fs::write(&ignored_path, b"unsupported").unwrap();

        let inspection = TitleInspector::inspect(directory.path()).unwrap();

        assert!(inspection.packages.is_empty());
        assert_eq!(inspection.nros.len(), 1);
        assert_eq!(inspection.nros[0].path, nro_path);
        assert_eq!(inspection.ignored_files, vec![ignored_path]);

        let nros = TitleInspector::inspect_nros_with_options(
            directory.path(),
            DirectoryScanOptions::default(),
        )
        .unwrap();
        assert_eq!(nros.len(), 1);
        assert_eq!(nros[0].path, inspection.nros[0].path);
    }

    #[test]
    fn multipath_inspection_preserves_root_order_and_uses_the_same_pipeline() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first.nro");
        let second = directory.path().join("second.NRO");
        fs::write(&first, synthetic_nro()).unwrap();
        fs::write(&second, synthetic_nro()).unwrap();

        let inspections = TitleInspector::inspect_paths([&first, &second]).unwrap();

        assert_eq!(inspections.len(), 2);
        assert_eq!(inspections[0].path, first);
        assert_eq!(inspections[0].nros[0].path, inspections[0].path);
        assert_eq!(inspections[1].path, second);
        assert_eq!(inspections[1].nros[0].path, inspections[1].path);
    }

    #[test]
    fn multipath_inspection_rejects_an_empty_root_set() {
        assert!(matches!(
            TitleInspector::inspect_paths(Vec::<PathBuf>::new()),
            Err(InspectError::NoPaths)
        ));
    }

    #[test]
    fn multipath_inspection_keeps_only_the_first_overlapping_file() {
        let directory = tempfile::tempdir().unwrap();
        let nro = directory.path().join("overlap.nro");
        fs::write(&nro, synthetic_nro()).unwrap();

        let inspections = TitleInspector::inspect_paths([directory.path(), nro.as_path()]).unwrap();

        assert_eq!(inspections.len(), 2);
        assert_eq!(inspections[0].nros.len(), 1);
        assert!(inspections[1].nros.is_empty());
    }

    fn zstd_bytes(bytes: &[u8]) -> Vec<u8> {
        zstd::stream::encode_all(bytes, 1).unwrap()
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
        assert_eq!(metadata.version.raw(), 786_432);
        assert_eq!(metadata.decoded_version().to_string(), "12.0");
        assert_eq!(metadata.required_system_version.unwrap().raw(), 123);
        assert_eq!(metadata.original_id, Some(0x0100_2cd0_0a51_c000));
        assert_eq!(metadata.contents.len(), 1);
        assert_eq!(metadata.contents[0].size, 42);
    }

    #[test]
    fn leaves_unrecognized_auxiliary_content_meta_version_raw() {
        let xml = r#"
            <ContentMeta>
              <Type>FutureContent</Type>
              <Id>0x01002cd00a51c800</Id>
              <Version>786432</Version>
            </ContentMeta>
        "#;

        let metadata = parse_content_meta_xml(xml).unwrap();

        assert_eq!(
            metadata.decoded_version(),
            DecodedContentMetaVersion::Unknown(ContentMetaVersion::from_raw(786_432))
        );
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

    #[test]
    fn parses_canonical_cnmt_without_auxiliary_xml() {
        let inner_pfs0 = build_pfs0(&[("Application_0100123456789000.cnmt", &application_cnmt())]);
        let nsp = load_synthetic_nsp(build_meta_nca(&inner_pfs0));

        let (metadata, warning) = inspect_canonical_metadata(&nsp, None);

        assert!(warning.is_none());
        let metadata = metadata.unwrap();
        assert_eq!(metadata.title_id, 0x0100_1234_5678_9000);
        assert_eq!(
            metadata.content_meta_type,
            swiitx_loader_content::CnmtMetaType::Application
        );
        assert_eq!(metadata.contents.len(), 1);
    }

    #[test]
    fn warns_when_meta_pfs0_has_no_cnmt_entry() {
        let inner_pfs0 = build_pfs0(&[("readme.txt", b"not metadata")]);
        let nsp = load_synthetic_nsp(build_meta_nca(&inner_pfs0));

        let (metadata, warning) = inspect_canonical_metadata(&nsp, None);

        assert!(metadata.is_none());
        assert!(warning.unwrap().contains("contains 0 .cnmt entries"));
    }

    #[test]
    fn warns_when_meta_section_payload_is_not_pfs0() {
        let nsp = load_synthetic_nsp(build_meta_nca(b"not a PFS0 file!"));

        let (metadata, warning) = inspect_canonical_metadata(&nsp, None);

        assert!(metadata.is_none());
        assert!(warning.unwrap().contains("expected PFS0 magic"));
    }

    #[test]
    fn rejects_cnmt_from_meta_section_with_invalid_integrity() {
        let inner_pfs0 = build_pfs0(&[("Application.cnmt", &application_cnmt())]);
        let mut nca = build_meta_nca(&inner_pfs0);
        nca[0xE00] ^= 0x80;
        let nsp = load_synthetic_nsp(nca);

        let (metadata, warning) = inspect_canonical_metadata(&nsp, None);

        assert!(metadata.is_none());
        assert!(warning.unwrap().contains("failed integrity validation"));
    }

    #[test]
    fn reads_control_nca_metadata_by_canonical_content_id() {
        let content_id = [0x22_u8; 16];
        let control_nca = build_control_nca(0x0100_1234_5678_9000);
        let cnmt = application_cnmt_with_control(content_id, control_nca.len() as u64);
        let inner_pfs0 = build_pfs0(&[("Application.cnmt", &cnmt)]);
        let meta_nca = build_meta_nca(&inner_pfs0);
        let control_name = format!("{}.nca", "22".repeat(16));
        let nsp_bytes = build_pfs0(&[("meta.cnmt.nca", &meta_nca), (&control_name, &control_nca)]);
        let storage: StorageRef = Arc::new(VecStorage(nsp_bytes));
        let nsp = NspLoader::load(storage).unwrap();
        let content_meta = load_canonical_content_meta(&nsp, None).unwrap();

        let control = load_control_metadata(&nsp, &content_meta, None)
            .unwrap()
            .unwrap();

        let title = control
            .nacp
            .title(swiitx_loader_content::NacpLanguage::AmericanEnglish);
        assert_eq!(title.name, "Synthetic title");
        assert_eq!(title.publisher, "Synthetic publisher");
        assert_eq!(control.nacp.display_version, "1.2.3");
        assert_eq!(control.icons().len(), 1);
        assert_eq!(
            control.icons()[0].language,
            swiitx_loader_content::NacpLanguage::AmericanEnglish
        );
    }

    #[test]
    fn accepts_patch_control_nca_with_application_title_id() {
        let application_id = 0x0100_1234_5678_9000;
        let content_id = [0x44_u8; 16];
        let control_nca = build_control_nca(application_id);
        let cnmt = patch_cnmt_with_control(
            application_id,
            content_id,
            u64::try_from(control_nca.len()).unwrap(),
        );
        let inner_pfs0 = build_pfs0(&[("Patch.cnmt", &cnmt)]);
        let meta_nca = build_meta_nca(&inner_pfs0);
        let control_name = format!("{}.nca", "44".repeat(16));
        let nsp_bytes = build_pfs0(&[("meta.cnmt.nca", &meta_nca), (&control_name, &control_nca)]);
        let storage: StorageRef = Arc::new(VecStorage(nsp_bytes));
        let nsp = NspLoader::load(storage).unwrap();
        let content_meta = load_canonical_content_meta(&nsp, None).unwrap();

        let control = load_control_metadata(&nsp, &content_meta, None)
            .unwrap()
            .unwrap();

        assert_eq!(
            control
                .nacp
                .title(swiitx_loader_content::NacpLanguage::AmericanEnglish)
                .name,
            "Synthetic title"
        );
    }

    fn application_cnmt() -> Vec<u8> {
        let mut cnmt = vec![0_u8; 0x20];
        put_u64(&mut cnmt, 0, 0x0100_1234_5678_9000);
        put_u32(&mut cnmt, 8, 7);
        cnmt[0x0C] = 0x80;
        put_u16(&mut cnmt, 0x0E, 0x10);
        put_u16(&mut cnmt, 0x10, 1);
        cnmt[0x17] = 1;
        let mut extended = [0_u8; 0x10];
        put_u64(&mut extended, 0, 0x0100_1234_5678_9800);
        cnmt.extend_from_slice(&extended);
        let mut content = [0_u8; 0x38];
        content[..0x20].fill(0x11);
        content[0x20..0x30].fill(0x22);
        content[0x30] = 0x34;
        content[0x31] = 0x12;
        content[0x36] = 1;
        cnmt.extend_from_slice(&content);
        cnmt.extend_from_slice(&[0x33; 0x20]);
        cnmt
    }

    fn application_cnmt_with_control(content_id: [u8; 16], content_size: u64) -> Vec<u8> {
        let mut cnmt = application_cnmt();
        cnmt[0x50..0x60].copy_from_slice(&content_id);
        cnmt[0x60..0x66].copy_from_slice(&content_size.to_le_bytes()[..6]);
        cnmt[0x66] = 3;
        cnmt
    }

    fn patch_cnmt_with_control(
        application_id: u64,
        content_id: [u8; 16],
        content_size: u64,
    ) -> Vec<u8> {
        let mut cnmt = vec![0_u8; 0x20];
        put_u64(&mut cnmt, 0, application_id + 0x800);
        put_u32(&mut cnmt, 8, 1);
        cnmt[0x0C] = 0x81;
        put_u16(&mut cnmt, 0x0E, 0x18);
        put_u16(&mut cnmt, 0x10, 1);
        cnmt[0x17] = 1;
        let mut extended = [0_u8; 0x18];
        put_u64(&mut extended, 0, application_id);
        cnmt.extend_from_slice(&extended);
        let mut content = [0_u8; 0x38];
        content[..0x20].fill(0x55);
        content[0x20..0x30].copy_from_slice(&content_id);
        content[0x30..0x36].copy_from_slice(&content_size.to_le_bytes()[..6]);
        content[0x36] = 3;
        cnmt.extend_from_slice(&content);
        cnmt.extend_from_slice(&[0x66; 0x20]);
        cnmt
    }

    fn build_pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _) in files {
            name_offsets.push(u32::try_from(strings.len()).unwrap());
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }

        let mut pfs0 = Vec::new();
        pfs0.extend_from_slice(b"PFS0");
        pfs0.extend_from_slice(&u32::try_from(files.len()).unwrap().to_le_bytes());
        pfs0.extend_from_slice(&u32::try_from(strings.len()).unwrap().to_le_bytes());
        pfs0.extend_from_slice(&0_u32.to_le_bytes());
        let mut relative_offset = 0_u64;
        for ((_, data), name_offset) in files.iter().zip(name_offsets) {
            pfs0.extend_from_slice(&relative_offset.to_le_bytes());
            pfs0.extend_from_slice(&u64::try_from(data.len()).unwrap().to_le_bytes());
            pfs0.extend_from_slice(&name_offset.to_le_bytes());
            pfs0.extend_from_slice(&0_u32.to_le_bytes());
            relative_offset += u64::try_from(data.len()).unwrap();
        }
        pfs0.extend_from_slice(&strings);
        for (_, data) in files {
            pfs0.extend_from_slice(data);
        }
        pfs0
    }

    fn build_control_nca(title_id: u64) -> Vec<u8> {
        const SECTION_OFFSET: usize = 0xC00;
        const BLOCK_SIZE: usize = 0x10000;
        let mut nacp = vec![0_u8; swiitx_loader_content::NACP_SIZE];
        nacp[.."Synthetic title".len()].copy_from_slice(b"Synthetic title");
        nacp[0x200..0x200 + "Synthetic publisher".len()].copy_from_slice(b"Synthetic publisher");
        nacp[0x302C..0x3030].copy_from_slice(&1_u32.to_le_bytes());
        nacp[0x3060..0x3065].copy_from_slice(b"1.2.3");
        let romfs = build_romfs(&[
            ("control.nacp", &nacp),
            ("icon_AmericanEnglish.dat", &[0xFF, 0xD8, 0xFF, 0xD9]),
        ]);
        assert!(romfs.len() <= BLOCK_SIZE);
        let section_size = romfs.len().next_multiple_of(0x200);
        let mut nca = vec![0_u8; SECTION_OFFSET + section_size];
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x205] = 2;
        nca[0x206] = 1;
        let nca_size = nca.len() as u64;
        put_u64(&mut nca, 0x208, nca_size);
        put_u64(&mut nca, 0x210, title_id);
        put_u32(&mut nca, 0x240, (SECTION_OFFSET / 0x200) as u32);
        put_u32(
            &mut nca,
            0x244,
            ((SECTION_OFFSET + section_size) / 0x200) as u32,
        );
        nca[SECTION_OFFSET..SECTION_OFFSET + romfs.len()].copy_from_slice(&romfs);

        let mut padded = vec![0_u8; BLOCK_SIZE];
        padded[..romfs.len()].copy_from_slice(&romfs);
        let master_hash: [u8; 32] = Sha256::digest(&padded).into();
        let fs = &mut nca[0x400..0x600];
        fs[2] = 0;
        fs[3] = 3;
        fs[4] = 1;
        fs[0x08..0x0C].copy_from_slice(b"IVFC");
        put_u32(fs, 0x10, 0x20);
        put_u32(fs, 0x14, 2);
        put_u64(fs, 0x18, 0);
        put_u64(fs, 0x20, romfs.len() as u64);
        put_u32(fs, 0x28, 16);
        fs[0xC8..0xE8].copy_from_slice(&master_hash);
        let fs_hash: [u8; 32] = Sha256::digest(&nca[0x400..0x600]).into();
        nca[0x280..0x2A0].copy_from_slice(&fs_hash);
        nca
    }

    fn build_romfs(files: &[(&str, &[u8])]) -> Vec<u8> {
        const DIRECTORY_META_OFFSET: usize = 0x54;
        const FILE_META_OFFSET: usize = 0x70;
        let mut file_meta = Vec::new();
        let mut data_offset = 0_u64;
        for (index, (name, data)) in files.iter().enumerate() {
            let next = if index + 1 == files.len() {
                u32::MAX
            } else {
                (file_meta.len() + 0x20 + name.len().next_multiple_of(4)) as u32
            };
            file_meta.extend_from_slice(&0_u32.to_le_bytes());
            file_meta.extend_from_slice(&next.to_le_bytes());
            file_meta.extend_from_slice(&data_offset.to_le_bytes());
            file_meta.extend_from_slice(&(data.len() as u64).to_le_bytes());
            file_meta.extend_from_slice(&u32::MAX.to_le_bytes());
            file_meta.extend_from_slice(&(name.len() as u32).to_le_bytes());
            file_meta.extend_from_slice(name.as_bytes());
            file_meta.resize(file_meta.len().next_multiple_of(4), 0);
            data_offset += data.len() as u64;
        }

        let file_data_offset = (FILE_META_OFFSET + file_meta.len()).next_multiple_of(0x10);
        let mut bytes = vec![0_u8; file_data_offset];
        put_u64(&mut bytes, 0, 0x50);
        put_u64(&mut bytes, 0x08, 0x50);
        put_u64(&mut bytes, 0x10, 4);
        put_u64(&mut bytes, 0x18, DIRECTORY_META_OFFSET as u64);
        put_u64(&mut bytes, 0x20, 0x18);
        put_u64(&mut bytes, 0x28, 0x6C);
        put_u64(&mut bytes, 0x30, 4);
        put_u64(&mut bytes, 0x38, FILE_META_OFFSET as u64);
        put_u64(&mut bytes, 0x40, file_meta.len() as u64);
        put_u64(&mut bytes, 0x48, file_data_offset as u64);
        put_u32(&mut bytes, DIRECTORY_META_OFFSET, 0);
        put_u32(&mut bytes, DIRECTORY_META_OFFSET + 4, u32::MAX);
        put_u32(&mut bytes, DIRECTORY_META_OFFSET + 8, u32::MAX);
        put_u32(&mut bytes, DIRECTORY_META_OFFSET + 0x0C, 0);
        put_u32(&mut bytes, DIRECTORY_META_OFFSET + 0x10, u32::MAX);
        bytes[FILE_META_OFFSET..FILE_META_OFFSET + file_meta.len()].copy_from_slice(&file_meta);
        for (_, data) in files {
            bytes.extend_from_slice(data);
        }
        bytes
    }

    fn build_meta_nca(payload: &[u8]) -> Vec<u8> {
        const SECTION_OFFSET: usize = 0xC00;
        const SECTION_SIZE: usize = 0x400;
        const DATA_OFFSET: usize = 0x200;
        const BLOCK_SIZE: usize = 0x200;
        assert!(payload.len() <= BLOCK_SIZE);

        let mut nca = vec![0_u8; SECTION_OFFSET + SECTION_SIZE];
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x205] = 1;
        nca[0x206] = 1;
        put_u64(&mut nca, 0x208, (SECTION_OFFSET + SECTION_SIZE) as u64);
        put_u64(&mut nca, 0x210, 0x0100_1234_5678_9000);
        put_u32(&mut nca, 0x240, (SECTION_OFFSET / 0x200) as u32);
        put_u32(
            &mut nca,
            0x244,
            ((SECTION_OFFSET + SECTION_SIZE) / 0x200) as u32,
        );

        let data_start = SECTION_OFFSET + DATA_OFFSET;
        nca[data_start..data_start + payload.len()].copy_from_slice(payload);
        let data_hash: [u8; 32] = Sha256::digest(payload).into();
        nca[SECTION_OFFSET..SECTION_OFFSET + 0x20].copy_from_slice(&data_hash);
        let master_hash: [u8; 32] =
            Sha256::digest(&nca[SECTION_OFFSET..SECTION_OFFSET + 0x20]).into();

        let fs = &mut nca[0x400..0x600];
        fs[2] = 1;
        fs[3] = 2;
        fs[4] = 1;
        fs[0x08..0x28].copy_from_slice(&master_hash);
        put_u32(fs, 0x28, BLOCK_SIZE as u32);
        put_u64(fs, 0x30, 0);
        put_u64(fs, 0x38, 0x20);
        put_u64(fs, 0x40, DATA_OFFSET as u64);
        put_u64(fs, 0x48, payload.len() as u64);
        let fs_hash: [u8; 32] = Sha256::digest(&nca[0x400..0x600]).into();
        nca[0x280..0x2A0].copy_from_slice(&fs_hash);
        nca
    }

    fn load_synthetic_nsp(meta_nca: Vec<u8>) -> swiitx_loader_content::NspArchive {
        let nsp_bytes = build_pfs0(&[("meta.cnmt.nca", &meta_nca)]);
        let storage: StorageRef = Arc::new(VecStorage(nsp_bytes));
        NspLoader::load(storage).unwrap()
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
