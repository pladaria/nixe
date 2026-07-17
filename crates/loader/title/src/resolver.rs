use crate::{ApplicationId, ContentType, ResolvedTitle, TitleCatalog, TitleError};

/// Resolves catalogued packages into a complete launchable title description.
#[derive(Debug)]
pub struct TitleResolver;

impl TitleResolver {
    /// Selects one base application, the newest patch, and all associated DLC.
    pub fn resolve(
        catalog: &TitleCatalog,
        application_id: ApplicationId,
    ) -> Result<ResolvedTitle, TitleError> {
        let packages: Vec<_> = catalog.packages_for(application_id).collect();
        let bases: Vec<_> = packages
            .iter()
            .copied()
            .filter(|package| package.content_type == ContentType::Application)
            .collect();

        let base = match bases.as_slice() {
            [] => return Err(TitleError::MissingBase { application_id }),
            [base] => (*base).clone(),
            bases => {
                return Err(TitleError::ConflictingBases {
                    application_id,
                    count: bases.len(),
                });
            }
        };

        let patch = packages
            .iter()
            .copied()
            .filter(|package| package.content_type == ContentType::Patch)
            .max_by_key(|package| package.version)
            .cloned();

        let mut add_ons: Vec<_> = packages
            .iter()
            .copied()
            .filter(|package| package.content_type == ContentType::AddOnContent)
            .cloned()
            .collect();
        add_ons.sort_by_key(|package| (package.title_id, package.version));

        Ok(ResolvedTitle {
            application_id,
            base,
            patch,
            add_ons,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_storage::{Storage, StorageError, StorageRef};

    use super::*;
    use crate::{PackageMetadata, TitleId};

    const APPLICATION_ID: ApplicationId = ApplicationId::new(0x0100_1234_1234_0000);

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

    fn package(title_id: u64, version: u32, content_type: ContentType) -> PackageMetadata {
        let source: StorageRef = Arc::new(EmptyStorage);

        PackageMetadata {
            title_id: TitleId::new(title_id),
            application_id: APPLICATION_ID,
            version,
            content_type,
            source,
        }
    }

    #[test]
    fn resolves_base_latest_patch_and_add_ons() {
        let catalog = TitleCatalog::from_packages(vec![
            package(APPLICATION_ID.get(), 0, ContentType::Application),
            package(APPLICATION_ID.get() + 0x800, 1, ContentType::Patch),
            package(APPLICATION_ID.get() + 0x800, 3, ContentType::Patch),
            package(APPLICATION_ID.get() + 0x1001, 0, ContentType::AddOnContent),
        ]);

        let title = TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap();

        assert_eq!(title.base.content_type, ContentType::Application);
        assert_eq!(title.patch.unwrap().version, 3);
        assert_eq!(title.add_ons.len(), 1);
    }

    #[test]
    fn rejects_missing_base() {
        let catalog = TitleCatalog::new();

        assert_eq!(
            TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap_err(),
            TitleError::MissingBase {
                application_id: APPLICATION_ID,
            }
        );
    }

    #[test]
    fn rejects_multiple_bases() {
        let catalog = TitleCatalog::from_packages(vec![
            package(APPLICATION_ID.get(), 0, ContentType::Application),
            package(APPLICATION_ID.get(), 0, ContentType::Application),
        ]);

        assert_eq!(
            TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap_err(),
            TitleError::ConflictingBases {
                application_id: APPLICATION_ID,
                count: 2,
            }
        );
    }
}
