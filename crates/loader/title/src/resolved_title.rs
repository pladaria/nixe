use crate::{ApplicationId, PackageMetadata};

/// Coherent view of a base application and its selected optional content.
#[derive(Clone, Debug)]
pub struct ResolvedTitle {
    /// Application represented by this title.
    pub application_id: ApplicationId,
    /// Base application package.
    pub base: PackageMetadata,
    /// Highest-version patch found for the application, if any.
    pub patch: Option<PackageMetadata>,
    /// Add-on packages associated with the application.
    pub add_ons: Vec<PackageMetadata>,
}
