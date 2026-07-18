use std::path::Path;

use swiitx_loader_content::CnmtContentMeta;
use swiitx_loader_storage::StorageRef;

use crate::{ApplicationId, PackageMetadata, TitleError};

/// Collection of package metadata discovered in one or more locations.
#[derive(Debug, Default)]
pub struct TitleCatalog {
    packages: Vec<PackageMetadata>,
}

impl TitleCatalog {
    /// Creates an empty title catalog.
    pub const fn new() -> Self {
        Self {
            packages: Vec::new(),
        }
    }

    /// Creates a catalog from metadata produced by content loaders.
    pub fn from_packages(packages: Vec<PackageMetadata>) -> Self {
        Self { packages }
    }

    /// Scans a directory for supported content packages.
    ///
    /// Directory scanning depends on NSP, NSZ, XCI, NCA, and CNMT parsing and
    /// will be implemented once those content loaders expose package metadata.
    pub fn scan_directory(_path: impl AsRef<Path>) -> Result<Self, TitleError> {
        Err(TitleError::NotImplemented {
            operation: "directory title scanning",
        })
    }

    /// Adds package metadata to the catalog.
    pub fn add(&mut self, package: PackageMetadata) {
        self.packages.push(package);
    }

    /// Converts canonical binary CNMT into package metadata and adds it.
    pub fn add_content_meta(
        &mut self,
        content_meta: &CnmtContentMeta,
        source: StorageRef,
    ) -> Result<(), TitleError> {
        self.add(PackageMetadata::from_content_meta(content_meta, source)?);
        Ok(())
    }

    /// Returns every package in discovery order.
    pub fn packages(&self) -> &[PackageMetadata] {
        &self.packages
    }

    /// Returns packages associated with one application.
    pub fn packages_for(
        &self,
        application_id: ApplicationId,
    ) -> impl Iterator<Item = &PackageMetadata> {
        self.packages
            .iter()
            .filter(move |package| package.application_id == application_id)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_content::{CnmtExtendedHeader, CnmtInstallType, CnmtMetaType, CnmtPlatform};
    use swiitx_loader_storage::{Storage, StorageError};

    use super::*;

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

    fn application_content_meta(title_id: u64, version: u32) -> CnmtContentMeta {
        CnmtContentMeta {
            title_id,
            version,
            content_meta_type: CnmtMetaType::Application,
            platform: CnmtPlatform::Nx,
            extended_header_size: 0x10,
            attributes: 0,
            storage_id: 0,
            install_type: CnmtInstallType::Full,
            committed: true,
            required_download_system_version: 0,
            reserved: [0; 4],
            extended_header: CnmtExtendedHeader::Application {
                patch_id: title_id + 0x800,
                required_system_version: 0,
                required_application_version: 0,
            },
            contents: Vec::new(),
            content_meta: Vec::new(),
            extended_data_size: 0,
            digest: [0; 32],
        }
    }

    #[test]
    fn adds_canonical_content_meta_in_discovery_order() {
        let mut catalog = TitleCatalog::new();
        let first_source: StorageRef = Arc::new(EmptyStorage);
        let second_source: StorageRef = Arc::new(EmptyStorage);

        catalog
            .add_content_meta(
                &application_content_meta(0x0100_1234_0000_0000, 1),
                first_source.clone(),
            )
            .unwrap();
        catalog
            .add_content_meta(
                &application_content_meta(0x0100_5678_0000_0000, 2),
                second_source.clone(),
            )
            .unwrap();

        assert_eq!(catalog.packages().len(), 2);
        assert_eq!(
            catalog.packages()[0].title_id,
            crate::TitleId::new(0x0100_1234_0000_0000)
        );
        assert_eq!(
            catalog.packages()[1].title_id,
            crate::TitleId::new(0x0100_5678_0000_0000)
        );
        assert!(Arc::ptr_eq(&catalog.packages()[0].source, &first_source));
        assert!(Arc::ptr_eq(&catalog.packages()[1].source, &second_source));
    }

    #[test]
    fn does_not_add_content_meta_when_conversion_fails() {
        let mut catalog = TitleCatalog::new();
        let source: StorageRef = Arc::new(EmptyStorage);
        let mut metadata = application_content_meta(0x0100_1234_0000_0000, 1);
        metadata.content_meta_type = CnmtMetaType::SystemData;
        metadata.extended_header = CnmtExtendedHeader::None;

        assert!(catalog.add_content_meta(&metadata, source).is_err());
        assert!(catalog.packages().is_empty());
    }
}
