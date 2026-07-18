use std::collections::BTreeMap;

use crate::{
    ApplicationId, ContentType, PackageMetadata, ResolvedTitle, TitleCatalog, TitleError, TitleId,
};

/// Resolves catalogued packages into a complete launchable title description.
#[derive(Debug)]
pub struct TitleResolver;

impl TitleResolver {
    /// Resolves every application relationship in catalog discovery order.
    pub fn resolve_all(catalog: &TitleCatalog) -> Result<Vec<ResolvedTitle>, TitleError> {
        catalog
            .application_ids()
            .map(|application_id| Self::resolve(catalog, application_id))
            .collect()
    }

    /// Selects one base application, the newest compatible patch, and compatible DLC.
    pub fn resolve(
        catalog: &TitleCatalog,
        application_id: ApplicationId,
    ) -> Result<ResolvedTitle, TitleError> {
        let packages = deduplicate(catalog.packages_for(application_id), application_id)?;
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

        let expected_patch_id = base
            .patch_id()
            .expect("validated application metadata must declare its patch title");
        let patches: Vec<_> = packages
            .iter()
            .copied()
            .filter(|package| package.content_type == ContentType::Patch)
            .collect();
        if let Some(patch) = patches
            .iter()
            .find(|package| package.title_id != expected_patch_id)
        {
            return Err(TitleError::IncompatiblePatchRelationship {
                application_id,
                expected_patch_id,
                patch_id: patch.title_id,
            });
        }

        let required_patch_version = base.required_application_version().unwrap_or(0);
        let patch = patches
            .iter()
            .copied()
            .filter(|package| package.version >= required_patch_version)
            .max_by_key(|package| package.version)
            .cloned();
        if patch.is_none() && base.version < required_patch_version {
            return Err(TitleError::MissingCompatiblePatch {
                application_id,
                required_application_version: required_patch_version,
                newest_available_version: patches.iter().map(|package| package.version).max(),
            });
        }
        let effective_application_version = patch
            .as_ref()
            .map_or(base.version, |package| package.version);

        let mut add_on_revisions = BTreeMap::<TitleId, Vec<&PackageMetadata>>::new();
        for package in packages
            .iter()
            .copied()
            .filter(|package| package.content_type == ContentType::AddOnContent)
        {
            add_on_revisions
                .entry(package.title_id)
                .or_default()
                .push(package);
        }

        let mut add_ons = Vec::with_capacity(add_on_revisions.len());
        for (title_id, revisions) in add_on_revisions {
            let compatible = revisions
                .iter()
                .copied()
                .filter(|package| {
                    package.required_application_version().unwrap_or(0)
                        <= effective_application_version
                })
                .max_by_key(|package| package.version);
            let Some(compatible) = compatible else {
                let required_application_version = revisions
                    .iter()
                    .filter_map(|package| package.required_application_version())
                    .min()
                    .unwrap_or(0);
                return Err(TitleError::IncompatibleAddOnContent {
                    application_id,
                    title_id,
                    required_application_version,
                    actual_application_version: effective_application_version,
                });
            };
            add_ons.push(compatible.clone());
        }

        Ok(ResolvedTitle {
            application_id,
            base,
            patch,
            add_ons,
        })
    }
}

type LogicalCoordinate = (ContentType, TitleId, u32);

fn deduplicate<'a>(
    packages: impl Iterator<Item = &'a PackageMetadata>,
    application_id: ApplicationId,
) -> Result<Vec<&'a PackageMetadata>, TitleError> {
    let mut coordinates = BTreeMap::<LogicalCoordinate, Vec<&PackageMetadata>>::new();
    for package in packages {
        coordinates
            .entry((package.content_type, package.title_id, package.version))
            .or_default()
            .push(package);
    }

    let mut deduplicated = Vec::with_capacity(coordinates.len());
    for ((content_type, title_id, version), packages) in coordinates {
        let mut canonical_variants = Vec::new();
        for package in packages {
            if !canonical_variants
                .iter()
                .any(|candidate: &&PackageMetadata| candidate.has_same_canonical_metadata(package))
            {
                canonical_variants.push(package);
            }
        }
        if canonical_variants.len() > 1 {
            return Err(TitleError::ConflictingPackages {
                application_id,
                content_type,
                title_id,
                version,
                count: canonical_variants.len(),
            });
        }
        deduplicated.push(canonical_variants[0]);
    }
    Ok(deduplicated)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_content::{
        CnmtContentMeta, CnmtExtendedHeader, CnmtInstallType, CnmtMetaType, CnmtPlatform,
    };
    use swiitx_loader_storage::{Storage, StorageError, StorageRef};

    use super::*;

    const APPLICATION_ID: ApplicationId = ApplicationId::new(0x0100_1234_1234_0000);
    const PATCH_ID: u64 = APPLICATION_ID.get() + 0x800;
    const FIRST_ADD_ON_ID: u64 = APPLICATION_ID.get() + 0x1001;
    const SECOND_ADD_ON_ID: u64 = APPLICATION_ID.get() + 0x1002;

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

    fn package(
        title_id: u64,
        version: u32,
        content_meta_type: CnmtMetaType,
        extended_header: CnmtExtendedHeader,
        fingerprint: u8,
    ) -> PackageMetadata {
        package_with_source(
            title_id,
            version,
            content_meta_type,
            extended_header,
            fingerprint,
            Arc::new(EmptyStorage),
        )
    }

    fn package_with_source(
        title_id: u64,
        version: u32,
        content_meta_type: CnmtMetaType,
        extended_header: CnmtExtendedHeader,
        fingerprint: u8,
        source: StorageRef,
    ) -> PackageMetadata {
        let metadata = CnmtContentMeta {
            title_id,
            version,
            content_meta_type,
            platform: CnmtPlatform::Nx,
            extended_header_size: 0,
            attributes: 0,
            storage_id: 0,
            install_type: CnmtInstallType::Full,
            committed: true,
            required_download_system_version: 0,
            reserved: [0; 4],
            extended_header,
            contents: Vec::new(),
            content_meta: Vec::new(),
            extended_data_size: 0,
            digest: [fingerprint; 32],
        };
        PackageMetadata::from_content_meta(&metadata, source).unwrap()
    }

    fn base(version: u32, required_patch_version: u32, fingerprint: u8) -> PackageMetadata {
        base_for(APPLICATION_ID, version, required_patch_version, fingerprint)
    }

    fn base_for(
        application_id: ApplicationId,
        version: u32,
        required_patch_version: u32,
        fingerprint: u8,
    ) -> PackageMetadata {
        package(
            application_id.get(),
            version,
            CnmtMetaType::Application,
            CnmtExtendedHeader::Application {
                patch_id: application_id.get() + 0x800,
                required_system_version: 0,
                required_application_version: required_patch_version,
            },
            fingerprint,
        )
    }

    fn patch(version: u32, fingerprint: u8) -> PackageMetadata {
        patch_with_title(PATCH_ID, version, fingerprint)
    }

    fn patch_with_title(title_id: u64, version: u32, fingerprint: u8) -> PackageMetadata {
        package(
            title_id,
            version,
            CnmtMetaType::Patch,
            CnmtExtendedHeader::Patch {
                application_id: APPLICATION_ID.get(),
                required_system_version: 0,
                extended_data_size: 0,
                reserved: [0; 8],
            },
            fingerprint,
        )
    }

    fn add_on(
        title_id: u64,
        version: u32,
        required_application_version: u32,
        fingerprint: u8,
    ) -> PackageMetadata {
        package(
            title_id,
            version,
            CnmtMetaType::AddOnContent,
            CnmtExtendedHeader::AddOnContent {
                application_id: APPLICATION_ID.get(),
                required_application_version,
                content_accessibilities: 0,
                padding: [0; 3],
                data_patch_id: 0,
            },
            fingerprint,
        )
    }

    fn delta(version: u32) -> PackageMetadata {
        package(
            PATCH_ID,
            version,
            CnmtMetaType::Delta,
            CnmtExtendedHeader::Delta {
                application_id: APPLICATION_ID.get(),
                extended_data_size: 0,
                padding: 0,
            },
            0,
        )
    }

    #[test]
    fn resolves_latest_compatible_patch_and_add_on_revisions_in_canonical_order() {
        let catalog = TitleCatalog::from_packages(vec![
            add_on(SECOND_ADD_ON_ID, 1, 0, 1),
            patch(1, 1),
            add_on(FIRST_ADD_ON_ID, 3, 4, 3),
            base(0, 2, 0),
            add_on(FIRST_ADD_ON_ID, 2, 3, 2),
            patch(3, 3),
        ]);

        let title = TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap();

        assert_eq!(title.base.content_type, ContentType::Application);
        assert_eq!(title.patch.unwrap().version, 3);
        assert_eq!(title.add_ons.len(), 2);
        assert_eq!(title.add_ons[0].title_id, TitleId::new(FIRST_ADD_ON_ID));
        assert_eq!(title.add_ons[0].version, 2);
        assert_eq!(title.add_ons[1].title_id, TitleId::new(SECOND_ADD_ON_ID));
    }

    #[test]
    fn rejects_missing_base() {
        let catalog = TitleCatalog::new();

        assert!(matches!(
            TitleResolver::resolve(&catalog, APPLICATION_ID),
            Err(TitleError::MissingBase {
                application_id: APPLICATION_ID
            })
        ));
    }

    #[test]
    fn collapses_duplicates_and_preserves_the_first_source() {
        let first_source: StorageRef = Arc::new(EmptyStorage);
        let second_source: StorageRef = Arc::new(EmptyStorage);
        let first = package_with_source(
            APPLICATION_ID.get(),
            0,
            CnmtMetaType::Application,
            CnmtExtendedHeader::Application {
                patch_id: PATCH_ID,
                required_system_version: 0,
                required_application_version: 0,
            },
            7,
            first_source.clone(),
        );
        let duplicate = package_with_source(
            APPLICATION_ID.get(),
            0,
            CnmtMetaType::Application,
            CnmtExtendedHeader::Application {
                patch_id: PATCH_ID,
                required_system_version: 0,
                required_application_version: 0,
            },
            7,
            second_source,
        );
        let catalog = TitleCatalog::from_packages(vec![first, duplicate]);

        let title = TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap();

        assert!(Arc::ptr_eq(&title.base.source, &first_source));
    }

    #[test]
    fn collapses_duplicate_patch_and_add_on_packages() {
        let catalog = TitleCatalog::from_packages(vec![
            base(0, 0, 0),
            patch(2, 2),
            patch(2, 2),
            add_on(FIRST_ADD_ON_ID, 1, 0, 1),
            add_on(FIRST_ADD_ON_ID, 1, 0, 1),
        ]);

        let title = TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap();

        assert_eq!(title.patch.unwrap().version, 2);
        assert_eq!(title.add_ons.len(), 1);
    }

    #[test]
    fn rejects_conflicting_canonical_packages_at_one_coordinate() {
        let catalog = TitleCatalog::from_packages(vec![base(0, 0, 1), base(0, 0, 2)]);

        assert!(matches!(
            TitleResolver::resolve(&catalog, APPLICATION_ID),
            Err(TitleError::ConflictingPackages {
                application_id: APPLICATION_ID,
                content_type: ContentType::Application,
                title_id,
                version: 0,
                count: 2,
            }) if title_id == TitleId::new(APPLICATION_ID.get())
        ));
    }

    #[test]
    fn rejects_multiple_distinct_base_revisions() {
        let catalog = TitleCatalog::from_packages(vec![base(0, 0, 0), base(1, 0, 1)]);

        assert!(matches!(
            TitleResolver::resolve(&catalog, APPLICATION_ID),
            Err(TitleError::ConflictingBases {
                application_id: APPLICATION_ID,
                count: 2
            })
        ));
    }

    #[test]
    fn rejects_conflicting_patch_and_add_on_revisions() {
        let patch_catalog =
            TitleCatalog::from_packages(vec![base(0, 0, 0), patch(2, 1), patch(2, 2)]);
        assert!(matches!(
            TitleResolver::resolve(&patch_catalog, APPLICATION_ID),
            Err(TitleError::ConflictingPackages {
                content_type: ContentType::Patch,
                version: 2,
                ..
            })
        ));

        let add_on_catalog = TitleCatalog::from_packages(vec![
            base(0, 0, 0),
            add_on(FIRST_ADD_ON_ID, 1, 0, 1),
            add_on(FIRST_ADD_ON_ID, 1, 0, 2),
        ]);
        assert!(matches!(
            TitleResolver::resolve(&add_on_catalog, APPLICATION_ID),
            Err(TitleError::ConflictingPackages {
                content_type: ContentType::AddOnContent,
                version: 1,
                ..
            })
        ));
    }

    #[test]
    fn reports_missing_compatible_patch() {
        let catalog = TitleCatalog::from_packages(vec![base(0, 4, 0), patch(3, 3)]);

        assert!(matches!(
            TitleResolver::resolve(&catalog, APPLICATION_ID),
            Err(TitleError::MissingCompatiblePatch {
                application_id: APPLICATION_ID,
                required_application_version: 4,
                newest_available_version: Some(3),
            })
        ));
    }

    #[test]
    fn rejects_a_patch_that_does_not_match_the_base_relationship() {
        let wrong_patch_id = PATCH_ID + 1;
        let catalog = TitleCatalog::from_packages(vec![
            base(0, 0, 0),
            patch_with_title(wrong_patch_id, 1, 1),
        ]);

        assert!(matches!(
            TitleResolver::resolve(&catalog, APPLICATION_ID),
            Err(TitleError::IncompatiblePatchRelationship {
                expected_patch_id,
                patch_id,
                ..
            }) if expected_patch_id == TitleId::new(PATCH_ID)
                && patch_id == TitleId::new(wrong_patch_id)
        ));
    }

    #[test]
    fn reports_add_on_without_a_compatible_revision() {
        let catalog = TitleCatalog::from_packages(vec![
            base(0, 0, 0),
            patch(2, 2),
            add_on(FIRST_ADD_ON_ID, 1, 3, 1),
            add_on(FIRST_ADD_ON_ID, 2, 4, 2),
        ]);

        assert!(matches!(
            TitleResolver::resolve(&catalog, APPLICATION_ID),
            Err(TitleError::IncompatibleAddOnContent {
                application_id: APPLICATION_ID,
                title_id,
                required_application_version: 3,
                actual_application_version: 2,
            }) if title_id == TitleId::new(FIRST_ADD_ON_ID)
        ));
    }

    #[test]
    fn ignores_delta_packages_in_the_resolved_title() {
        let catalog = TitleCatalog::from_packages(vec![base(0, 0, 0), delta(1)]);

        let title = TitleResolver::resolve(&catalog, APPLICATION_ID).unwrap();

        assert!(title.patch.is_none());
        assert!(title.add_ons.is_empty());
    }

    #[test]
    fn resolves_all_application_groups_in_discovery_order() {
        let second_application_id = ApplicationId::new(0x0100_5678_1234_0000);
        let catalog = TitleCatalog::from_packages(vec![
            base(0, 0, 0),
            base_for(second_application_id, 0, 0, 0),
        ]);

        let titles = TitleResolver::resolve_all(&catalog).unwrap();

        assert_eq!(titles.len(), 2);
        assert_eq!(titles[0].application_id, APPLICATION_ID);
        assert_eq!(titles[1].application_id, second_application_id);
    }

    #[test]
    fn resolve_all_reports_an_orphan_package_group() {
        let catalog = TitleCatalog::from_packages(vec![patch(1, 1)]);

        assert!(matches!(
            TitleResolver::resolve_all(&catalog),
            Err(TitleError::MissingBase {
                application_id: APPLICATION_ID
            })
        ));
    }
}
