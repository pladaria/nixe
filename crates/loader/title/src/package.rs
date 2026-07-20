use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

use swiitx_loader_content::{
    ApplicationVersion, CnmtContentInfo, CnmtContentMeta, CnmtExtendedHeader, CnmtMetaType,
    NcaArchive, NcaKeyProvider,
};
use swiitx_loader_storage::StorageRef;

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
    canonical_content_meta: CnmtContentMeta,
    source_format: Option<PackageFormat>,
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
            canonical_content_meta: content_meta.clone(),
            source_format: None,
        })
    }

    /// Returns the canonical binary content metadata retained for resolution.
    pub fn canonical_content_meta(&self) -> &CnmtContentMeta {
        &self.canonical_content_meta
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
    ) -> Result<NcaArchive, swiitx_loader_storage::LoadError> {
        let format = self.source_format.ok_or_else(|| {
            swiitx_loader_storage::LoadError::invalid(
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

    pub(crate) fn set_source_format(&mut self, format: PackageFormat) {
        self.source_format = Some(format);
    }

    /// Returns the patch title declared by an application package.
    pub fn patch_id(&self) -> Option<TitleId> {
        match &self.canonical_content_meta.extended_header {
            CnmtExtendedHeader::Application { patch_id, .. } => Some(TitleId::new(*patch_id)),
            _ => None,
        }
    }

    /// Returns the minimum effective application version required by this package.
    pub fn required_application_version(&self) -> Option<ApplicationVersion> {
        match &self.canonical_content_meta.extended_header {
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
        self.canonical_content_meta == other.canonical_content_meta
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
            .field("canonical_content_meta", &self.canonical_content_meta)
            .field("source_format", &self.source_format)
            .field("source", &"<storage>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_content::{CnmtInstallType, CnmtPlatform};
    use swiitx_loader_storage::{Storage, StorageError};

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
    fn converts_supported_content_meta_types() {
        let cases = [
            (
                CnmtMetaType::Application,
                CnmtExtendedHeader::Application {
                    patch_id: TITLE_ID + 0x800,
                    required_system_version: 0.into(),
                    required_application_version: 0.into(),
                },
                ContentType::Application,
                TITLE_ID,
            ),
            (
                CnmtMetaType::Patch,
                CnmtExtendedHeader::Patch {
                    application_id: APPLICATION_ID,
                    required_system_version: 0.into(),
                    extended_data_size: 0,
                    reserved: [0; 8],
                },
                ContentType::Patch,
                APPLICATION_ID,
            ),
            (
                CnmtMetaType::AddOnContent,
                CnmtExtendedHeader::AddOnContent {
                    application_id: APPLICATION_ID,
                    required_application_version: 0.into(),
                    content_accessibilities: 0,
                    padding: [0; 3],
                    data_patch_id: 0,
                },
                ContentType::AddOnContent,
                APPLICATION_ID,
            ),
            (
                CnmtMetaType::AddOnContent,
                CnmtExtendedHeader::LegacyAddOnContent {
                    application_id: APPLICATION_ID,
                    required_application_version: 0.into(),
                    padding: 0,
                },
                ContentType::AddOnContent,
                APPLICATION_ID,
            ),
            (
                CnmtMetaType::Delta,
                CnmtExtendedHeader::Delta {
                    application_id: APPLICATION_ID,
                    extended_data_size: 0,
                    padding: 0,
                },
                ContentType::Delta,
                APPLICATION_ID,
            ),
        ];

        for (content_meta_type, extended_header, expected_type, expected_application_id) in cases {
            let source: StorageRef = Arc::new(EmptyStorage);
            let metadata = content_meta(content_meta_type, extended_header);
            let package = PackageMetadata::from_content_meta(&metadata, source.clone()).unwrap();

            assert_eq!(package.title_id, TitleId::new(TITLE_ID));
            assert_eq!(
                package.application_id,
                ApplicationId::new(expected_application_id)
            );
            assert_eq!(package.version.raw(), 42);
            assert_eq!(package.content_type, expected_type);
            assert_eq!(package.canonical_content_meta(), &metadata);
            assert!(Arc::ptr_eq(&package.source, &source));
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
