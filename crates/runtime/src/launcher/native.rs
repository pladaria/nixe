use super::*;
use nixe_loader_content::{ExeFsArchive, NupPackage, ReadOnlyTree, RomFsArchive};

pub(super) fn build_native(
    path: &Path,
    resolved: ResolvedTitle,
    keys: Option<&dyn NcaKeyProvider>,
) -> Result<LaunchPlan, LaunchError> {
    let effective = resolved
        .patch
        .as_ref()
        .or(resolved.base.as_ref())
        .expect("resolved native application");
    // Executable modules belong to one program revision. Only RomFS uses the
    // directory overlay; retaining an old subsdk beside a new program is invalid.
    let program = resolved
        .patch
        .iter()
        .chain(resolved.base.iter())
        .find(|p| {
            p.nup()
                .is_some_and(|(archive, index)| archive.packages()[index].exefs.is_some())
        })
        .ok_or_else(|| {
            LaunchError::invalid(
                LaunchStage::ProgramContent,
                path,
                "native application has no ExeFS",
            )
        })?;
    let (archive, index) = program.nup().expect("native program");
    let package = &archive.packages()[index];
    let exefs = ExeFsArchive::from_tree(
        archive
            .tree(package.exefs.expect("selected ExeFS"))
            .map_err(|error| LaunchError::load(LaunchStage::ProgramContent, path, error))?,
    )
    .map_err(|error| LaunchError::load(LaunchStage::ProgramContent, path, error))?;
    let npdm = load_npdm(
        path,
        &exefs,
        resolved.application_id,
        package.metadata.npdm.as_ref(),
    )?;
    let (modules, entry) = load_modules(path, &exefs)?;
    let mut tree: Option<ReadOnlyTree> = None;
    let mut contributors = Vec::new();
    for (package, role) in resolved
        .base
        .iter()
        .map(|p| (p, MountProvenance::Base))
        .chain(resolved.patch.iter().map(|p| (p, MountProvenance::Patch)))
    {
        let (archive, index) = package.nup().expect("native application package");
        if let Some(root) = archive.packages()[index].romfs {
            let next = archive
                .tree(root)
                .map_err(|error| LaunchError::load(LaunchStage::PrimaryFileSystem, path, error))?;
            if let Some(tree) = tree.as_mut() {
                tree.overlay(next);
            } else {
                tree = Some(next);
            }
            contributors.push((package, role));
        }
    }
    let primary = tree
        .map(|tree| {
            let provenance = match contributors.as_slice() {
                [_, _] => MountProvenance::BaseAndPatch,
                [(_, role)] => *role,
                _ => unreachable!("RomFS has one or two contributors"),
            };
            let hashes: Vec<_> = contributors
                .iter()
                .map(|(package, _)| {
                    let (archive, index) = package.nup().expect("native RomFS package");
                    archive.packages()[index].content_hash
                })
                .collect();
            let content = match hashes.as_slice() {
                [base, update] => ContentIdentity::NupOverlay {
                    base: *base,
                    update: *update,
                },
                [hash] => ContentIdentity::NupTree(*hash),
                _ => unreachable!("RomFS has one or two contributors"),
            };
            RomFsArchive::from_tree(tree)
                .map(|romfs| ReadOnlyMount::new(provenance, Some(content), romfs))
                .map_err(|error| LaunchError::load(LaunchStage::PrimaryFileSystem, path, error))
        })
        .transpose()?;
    let add_ons = load_add_ons(path, &resolved, keys)?;
    Ok(LaunchPlan::new(
        LaunchKind::Packaged(Box::new(PackagedIdentity::new(
            resolved.application_id,
            effective.title_id,
            effective.version,
            ContentIdentity::NupTree(package.content_hash),
            npdm,
        ))),
        modules,
        entry,
        primary,
        add_ons,
        resolved.application_control().cloned(),
    )
    .with_display_version(resolved.display_version()))
}

pub(super) fn addon_mount(
    path: &Path,
    archive: &nixe_loader_content::NupArchive,
    package: &NupPackage,
) -> Result<ReadOnlyMount, LaunchError> {
    let root = package.romfs.ok_or_else(|| {
        LaunchError::invalid(
            LaunchStage::AddOnContent,
            path,
            "native add-on has no RomFS",
        )
    })?;
    archive
        .tree(root)
        .and_then(RomFsArchive::from_tree)
        .map(|romfs| {
            ReadOnlyMount::new(
                MountProvenance::AddOn,
                Some(ContentIdentity::NupTree(package.content_hash)),
                romfs,
            )
        })
        .map_err(|error| LaunchError::load(LaunchStage::AddOnContent, path, error))
}
