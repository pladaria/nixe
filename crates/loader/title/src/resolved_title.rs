use crate::{ApplicationId, ControlMetadata, PackageMetadata};

/// Selected application content, including explicitly standalone update containers.
#[derive(Clone, Debug)]
pub struct ResolvedTitle {
    /// Application represented by this title.
    pub application_id: ApplicationId,
    /// Base application, absent for an explicitly standalone update.
    pub base: Option<PackageMetadata>,
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
            .or_else(|| {
                self.base
                    .as_ref()
                    .and_then(PackageMetadata::control_metadata)
            })
    }
}

impl ResolvedTitle {
    pub fn packages(&self) -> impl Iterator<Item = &PackageMetadata> {
        self.base
            .iter()
            .chain(self.patch.iter())
            .chain(self.add_ons.iter())
    }

    pub fn display_name(&self, languages: &[crate::NacpLanguage]) -> Option<&str> {
        self.patch
            .as_ref()
            .and_then(|p| p.display_name(languages))
            .or_else(|| self.base.as_ref().and_then(|p| p.display_name(languages)))
    }

    pub fn display_version(&self) -> Option<&str> {
        self.patch
            .as_ref()
            .and_then(PackageMetadata::display_version)
            .or_else(|| {
                self.base
                    .as_ref()
                    .and_then(PackageMetadata::display_version)
            })
    }
}
