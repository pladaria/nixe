//! Package relationships and presentation metadata for native unencrypted trees.

use std::path::Path;
use std::sync::Arc;

use nixe_loader_content::{NacpLoader, NupLoader, NupPackageKind};
use nixe_loader_executable::NpdmLoader;
use nixe_loader_storage::{BytesStorage, FormatLoader, LoadError, StorageRef};

use crate::{ApplicationId, PackageFormat, PackageMetadata, TitleCatalog};

pub(crate) fn add(
    catalog: &mut TitleCatalog,
    path: &Path,
    source: StorageRef,
) -> Result<(), LoadError> {
    let archive = Arc::new(NupLoader::load(source.clone())?);
    let mut application = None;
    for package in archive
        .packages()
        .iter()
        .filter(|p| p.kind != NupPackageKind::AddOn)
    {
        let program_id = match &package.metadata.npdm {
            Some(bytes) => NpdmLoader::program_id_from_metadata(bytes)?,
            None if package.kind == NupPackageKind::Base => package.title_id,
            None => continue,
        };
        if program_id == 0
            || application.is_some_and(|id| id != program_id)
            || (package.kind == NupPackageKind::Base && package.title_id != program_id)
        {
            return Err(LoadError::invalid(
                "NUP",
                "conflicting application identities in package metadata",
            ));
        }
        application = Some(program_id);
    }
    let application = ApplicationId::new(application.ok_or_else(|| {
        LoadError::invalid(
            "NUP",
            "no base or program metadata identifies the owning application",
        )
    })?);
    for (index, package) in archive.packages().iter().enumerate() {
        if package
            .exefs
            .is_some_and(|root| archive.find_file(root, "main.npdm").is_some())
            && package.metadata.npdm.is_none()
        {
            return Err(LoadError::invalid(
                "NUP",
                "ExeFS main.npdm is missing its mandatory metadata record",
            ));
        }
        let nacp = package
            .metadata
            .nacp
            .as_ref()
            .map(|bytes| NacpLoader::load(Arc::new(BytesStorage(bytes.clone()))))
            .transpose()?;
        let mut metadata =
            PackageMetadata::from_nup(archive.clone(), index, application, source.clone(), nacp);
        metadata.set_source(PackageFormat::Nup, path);
        catalog.add(metadata);
    }
    Ok(())
}
