use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nixe_loader_content::{
    ApplicationControlProperty, ApplicationVersion, CnmtContentInfo, CnmtContentMeta,
    CnmtExtendedHeader, CnmtMetaType, NacpLanguage, NcaArchive, NcaKeyProvider, NupArchive,
    NupPackageKind,
};
use nixe_loader_storage::StorageRef;

use crate::package_content::open_canonical_content;
use crate::{ControlMetadata, PackageFormat};

/// Identifies an application to which base, patch, and add-on content belongs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApplicationId(u64);

impl ApplicationId {
    /// Creates an application identifier from its raw value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw identifier value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the conventional add-on base ID for this application family.
    /// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/file_sys/common_funcs.h
    pub const fn add_on_content_base_id(self) -> TitleId {
        TitleId((self.0 & !0x1FFF) + 0x1000)
    }
}

impl Display for ApplicationId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:016X}", self.0)
    }
}

/// Identifies one concrete application, patch, or add-on title.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TitleId(u64);

impl TitleId {
    /// Creates a title identifier from its raw value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw identifier value.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the application family encoded in a program, patch or add-on ID.
    /// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/file_sys/common_funcs.h
    pub const fn base_application_id(self) -> ApplicationId {
        ApplicationId(self.0 & !0x1FFF)
    }

    /// Returns the AOC index encoded in an add-on title ID.
    pub const fn add_on_content_index(self) -> u32 {
        (self.0 & 0x7FF) as u32
    }
}

impl Display for TitleId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:016X}", self.0)
    }
}

/// Describes the role of a package within a complete title.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContentType {
    /// Base game or application.
    Application,
    /// Update associated with an application.
    Patch,
    /// Downloadable content associated with an application.
    AddOnContent,
    /// Incremental data used while constructing a patch.
    Delta,
}

/// Errors produced while converting canonical CNMT into title-domain metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageMetadataError {
    /// The parsed content-meta role cannot be represented in the title catalog.
    UnsupportedContentMetaType { content_meta_type: CnmtMetaType },
    /// The content-meta extended header does not match its declared role.
    IncompatibleContentMetaHeader { content_meta_type: CnmtMetaType },
}

impl Display for PackageMetadataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedContentMetaType { content_meta_type } => write!(
                formatter,
                "content-meta type {content_meta_type} is not supported by the title catalog"
            ),
            Self::IncompatibleContentMetaHeader { content_meta_type } => write!(
                formatter,
                "content-meta type {content_meta_type} has an incompatible extended header"
            ),
        }
    }
}

impl Error for PackageMetadataError {}

/// Metadata extracted from one NSP, NSZ, XCI, or equivalent package.
#[derive(Clone)]
pub struct PackageMetadata {
    /// Identifier of this concrete content title.
    pub title_id: TitleId,
    /// Identifier of the application to which this package belongs.
    pub application_id: ApplicationId,
    /// Package version obtained from its content metadata.
    pub version: ApplicationVersion,
    /// Role of this package in the resolved title.
    pub content_type: ContentType,
    /// Random-access source containing the package.
    pub source: StorageRef,
    /// Parsed Control NCA metadata, when this package declares it and keys were available.
    control_metadata: Option<ControlMetadata>,
    /// Canonical binary metadata used to compare and resolve package revisions.
    contents: PackageContents,
    source_format: Option<PackageFormat>,
    source_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
enum PackageContents {
    Nca(CnmtContentMeta),
    Nup {
        archive: Arc<NupArchive>,
        index: usize,
        nacp: Option<Box<ApplicationControlProperty>>,
    },
}

impl PackageMetadata {
    /// Creates title-domain package metadata from canonical binary CNMT.
    pub fn from_content_meta(
        content_meta: &CnmtContentMeta,
        source: StorageRef,
    ) -> Result<Self, PackageMetadataError> {
        let (content_type, application_id) = match (
            content_meta.content_meta_type,
            &content_meta.extended_header,
        ) {
            (CnmtMetaType::Application, CnmtExtendedHeader::Application { .. }) => (
                ContentType::Application,
                ApplicationId::new(content_meta.title_id),
            ),
            (CnmtMetaType::Patch, CnmtExtendedHeader::Patch { application_id, .. }) => {
                (ContentType::Patch, ApplicationId::new(*application_id))
            }
            (
                CnmtMetaType::AddOnContent,
                CnmtExtendedHeader::AddOnContent { application_id, .. }
                | CnmtExtendedHeader::LegacyAddOnContent { application_id, .. },
            ) => (
                ContentType::AddOnContent,
                ApplicationId::new(*application_id),
            ),
            (CnmtMetaType::Delta, CnmtExtendedHeader::Delta { application_id, .. }) => {
                (ContentType::Delta, ApplicationId::new(*application_id))
            }
            (
                CnmtMetaType::Application
                | CnmtMetaType::Patch
                | CnmtMetaType::AddOnContent
                | CnmtMetaType::Delta,
                _,
            ) => {
                return Err(PackageMetadataError::IncompatibleContentMetaHeader {
                    content_meta_type: content_meta.content_meta_type,
                });
            }
            (content_meta_type, _) => {
                return Err(PackageMetadataError::UnsupportedContentMetaType { content_meta_type });
            }
        };

        Ok(Self {
            title_id: TitleId::new(content_meta.title_id),
            application_id,
            version: ApplicationVersion::from_raw(content_meta.version.raw()),
            content_type,
            source,
            control_metadata: None,
            contents: PackageContents::Nca(content_meta.clone()),
            source_format: None,
            source_path: None,
        })
    }

    /// Returns the canonical binary content metadata retained for resolution.
    pub fn canonical_content_meta(&self) -> Option<&CnmtContentMeta> {
        match &self.contents {
            PackageContents::Nca(meta) => Some(meta),
            PackageContents::Nup { .. } => None,
        }
    }

    pub(crate) fn from_nup(
        archive: Arc<NupArchive>,
        index: usize,
        application_id: ApplicationId,
        source: StorageRef,
        nacp: Option<ApplicationControlProperty>,
    ) -> Self {
        let package = &archive.packages()[index];
        Self {
            title_id: TitleId::new(package.title_id),
            application_id,
            version: ApplicationVersion::from_raw(package.version),
            content_type: match package.kind {
                NupPackageKind::Base => ContentType::Application,
                NupPackageKind::Update => ContentType::Patch,
                NupPackageKind::AddOn => ContentType::AddOnContent,
            },
            source,
            control_metadata: None,
            contents: PackageContents::Nup {
                archive,
                index,
                nacp: nacp.map(Box::new),
            },
            source_format: None,
            source_path: None,
        }
    }

    /// Native tree container and package index; never represented as synthetic CNMT.
    pub fn nup(&self) -> Option<(&Arc<NupArchive>, usize)> {
        match &self.contents {
            PackageContents::Nup { archive, index, .. } => Some((archive, *index)),
            _ => None,
        }
    }

    pub fn is_standalone_update(&self) -> bool {
        self.nup()
            .is_some_and(|(archive, index)| archive.packages()[index].standalone_update)
    }

    pub fn display_name(&self, preferences: &[NacpLanguage]) -> Option<&str> {
        match &self.contents {
            PackageContents::Nca(_) => self
                .control_metadata()
                .and_then(|control| control.nacp.preferred_title(preferences))
                .map(|(_, title)| title.name.as_str()),
            PackageContents::Nup {
                archive,
                index,
                nacp,
            } => {
                let meta = &archive.packages()[*index].metadata;
                for language in preferences {
                    let tag = language.language_tag();
                    if let Some((_, title)) = meta
                        .localized_titles
                        .iter()
                        .find(|(lang, _)| lang == tag)
                        .or_else(|| {
                            meta.localized_titles
                                .iter()
                                .find(|(lang, _)| lang == tag.split('-').next().unwrap_or(tag))
                        })
                    {
                        return Some(title);
                    }
                }
                meta.title
                    .as_deref()
                    .filter(|name| !name.is_empty())
                    .or_else(|| {
                        nacp.as_ref()
                            .and_then(|nacp| nacp.preferred_title(preferences))
                            .map(|(_, title)| title.name.as_str())
                    })
                    .or_else(|| {
                        meta.localized_titles
                            .first()
                            .map(|(_, title)| title.as_str())
                    })
            }
        }
    }

    pub fn display_version(&self) -> Option<&str> {
        match &self.contents {
            PackageContents::Nca(_) => self
                .control_metadata()
                .map(|c| c.nacp.display_version.as_str()),
            PackageContents::Nup {
                archive,
                index,
                nacp,
            } => archive.packages()[*index]
                .metadata
                .display_version
                .as_deref()
                .or_else(|| nacp.as_ref().map(|n| n.display_version.as_str())),
        }
        .filter(|version| !version.is_empty())
    }

    /// Application control fields, independent of their container provenance.
    pub fn application_control(&self) -> Option<&ApplicationControlProperty> {
        match &self.contents {
            PackageContents::Nca(_) => self.control_metadata.as_ref().map(|control| &control.nacp),
            PackageContents::Nup { nacp, .. } => nacp.as_deref(),
        }
    }

    /// Opens one NCA selected by this package's canonical CNMT record.
    ///
    /// Metadata constructed directly with [`Self::from_content_meta`] has no
    /// container locator and cannot be reopened. Catalog discovery attaches
    /// that locator without retaining a mutable package parser.
    pub fn open_content(
        &self,
        content: &CnmtContentInfo,
        keys: Option<&dyn NcaKeyProvider>,
    ) -> Result<NcaArchive, nixe_loader_storage::LoadError> {
        let format = self.source_format.ok_or_else(|| {
            nixe_loader_storage::LoadError::invalid(
                "canonical package content",
                "package metadata has no container locator",
            )
        })?;
        open_canonical_content(self, content, format, keys)
    }

    /// Returns parsed Control NCA metadata attached while loading the package.
    pub fn control_metadata(&self) -> Option<&ControlMetadata> {
        self.control_metadata.as_ref()
    }

    pub(crate) fn set_control_metadata(&mut self, metadata: Option<ControlMetadata>) {
        self.control_metadata = metadata;
    }

    /// Returns the host package path, when loaded from a local file.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    pub(crate) fn set_source(&mut self, format: PackageFormat, path: &Path) {
        self.source_format = Some(format);
        self.source_path = Some(path.to_owned());
    }

    /// Returns the patch title declared by an application package.
    pub fn patch_id(&self) -> Option<TitleId> {
        match &self.canonical_content_meta()?.extended_header {
            CnmtExtendedHeader::Application { patch_id, .. } => Some(TitleId::new(*patch_id)),
            _ => None,
        }
    }

    /// Returns the minimum effective application version required by this package.
    pub fn required_application_version(&self) -> Option<ApplicationVersion> {
        match &self.canonical_content_meta()?.extended_header {
            CnmtExtendedHeader::Application {
                required_application_version,
                ..
            }
            | CnmtExtendedHeader::AddOnContent {
                required_application_version,
                ..
            }
            | CnmtExtendedHeader::LegacyAddOnContent {
                required_application_version,
                ..
            } => Some(*required_application_version),
            _ => None,
        }
    }

    pub(crate) fn has_same_canonical_metadata(&self, other: &Self) -> bool {
        match (&self.contents, &other.contents) {
            (PackageContents::Nca(a), PackageContents::Nca(b)) => a == b,
            (
                PackageContents::Nup {
                    archive: a,
                    index: ai,
                    ..
                },
                PackageContents::Nup {
                    archive: b,
                    index: bi,
                    ..
                },
            ) => {
                let (a, b) = (&a.packages()[*ai], &b.packages()[*bi]);
                a.content_hash == b.content_hash
                    && a.metadata == b.metadata
                    && a.standalone_update == b.standalone_update
            }
            _ => false,
        }
    }
}

impl Debug for PackageMetadata {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PackageMetadata")
            .field("title_id", &self.title_id)
            .field("application_id", &self.application_id)
            .field("version", &self.version)
            .field("content_type", &self.content_type)
            .field("control_metadata", &self.control_metadata)
            .field("contents", &self.contents)
            .field("source_format", &self.source_format)
            .field("source", &"<storage>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nixe_loader_content::{CnmtInstallType, CnmtPlatform};
    use nixe_loader_storage::{Storage, StorageError};

    use super::*;

    const TITLE_ID: u64 = 0x0100_1234_5678_9800;
    const APPLICATION_ID: u64 = 0x0100_1234_5678_9000;

    #[derive(Debug)]
    struct EmptyStorage;

    impl Storage for EmptyStorage {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(0)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            if offset == 0 && buffer.is_empty() {
                Ok(())
            } else {
                Err(StorageError::OutOfBounds)
            }
        }
    }

    fn content_meta(
        content_meta_type: CnmtMetaType,
        extended_header: CnmtExtendedHeader,
    ) -> CnmtContentMeta {
        CnmtContentMeta {
            title_id: TITLE_ID,
            version: 42.into(),
            content_meta_type,
            platform: CnmtPlatform::Nx,
            extended_header_size: 0,
            attributes: 0,
            storage_id: 0,
            install_type: CnmtInstallType::Full,
            committed: true,
            required_download_system_version: 0.into(),
            reserved: [0; 4],
            extended_header,
            contents: Vec::new(),
            content_meta: Vec::new(),
            extended_data_size: 0,
            digest: [0; 32],
        }
    }

    #[test]
    fn exposes_canonical_application_version_requirements() {
        let source: StorageRef = Arc::new(EmptyStorage);
        let application = PackageMetadata::from_content_meta(
            &content_meta(
                CnmtMetaType::Application,
                CnmtExtendedHeader::Application {
                    patch_id: TITLE_ID + 0x800,
                    required_system_version: 0.into(),
                    required_application_version: 12.into(),
                },
            ),
            source.clone(),
        )
        .unwrap();
        let add_on = PackageMetadata::from_content_meta(
            &content_meta(
                CnmtMetaType::AddOnContent,
                CnmtExtendedHeader::LegacyAddOnContent {
                    application_id: APPLICATION_ID,
                    required_application_version: 7.into(),
                    padding: 0,
                },
            ),
            source,
        )
        .unwrap();

        assert_eq!(application.patch_id(), Some(TitleId::new(TITLE_ID + 0x800)));
        assert_eq!(
            application
                .required_application_version()
                .map(|version| version.raw()),
            Some(12)
        );
        assert_eq!(add_on.patch_id(), None);
        assert_eq!(
            add_on
                .required_application_version()
                .map(|version| version.raw()),
            Some(7)
        );
    }

    #[test]
    fn rejects_unsupported_content_meta_type() {
        let source: StorageRef = Arc::new(EmptyStorage);
        let metadata = content_meta(CnmtMetaType::SystemProgram, CnmtExtendedHeader::None);

        assert_eq!(
            PackageMetadata::from_content_meta(&metadata, source).unwrap_err(),
            PackageMetadataError::UnsupportedContentMetaType {
                content_meta_type: CnmtMetaType::SystemProgram,
            }
        );
    }

    #[test]
    fn rejects_incompatible_extended_header() {
        let source: StorageRef = Arc::new(EmptyStorage);
        let metadata = content_meta(
            CnmtMetaType::Patch,
            CnmtExtendedHeader::Application {
                patch_id: 0,
                required_system_version: 0.into(),
                required_application_version: 0.into(),
            },
        );

        assert_eq!(
            PackageMetadata::from_content_meta(&metadata, source).unwrap_err(),
            PackageMetadataError::IncompatibleContentMetaHeader {
                content_meta_type: CnmtMetaType::Patch,
            }
        );
    }
}
