use crate::{ApplicationId, ControlMetadata, PackageMetadata};

/// Coherent view of a base application and its selected optional content.
#[derive(Clone, Debug)]
pub struct ResolvedTitle {
    /// Application represented by this title.
    pub application_id: ApplicationId,
    /// Base application package.
    pub base: PackageMetadata,
    /// Highest compatible patch version found for the application, if any.
    pub patch: Option<PackageMetadata>,
    /// Newest compatible revision of each add-on title, in title-ID order.
    pub add_ons: Vec<PackageMetadata>,
}

impl ResolvedTitle {
    /// Returns the selected patch's Control metadata, falling back to the base.
    pub fn control_metadata(&self) -> Option<&ControlMetadata> {
        self.patch
            .as_ref()
            .and_then(PackageMetadata::control_metadata)
            .or_else(|| self.base.control_metadata())
    }
}
