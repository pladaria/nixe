use std::path::Path;

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
