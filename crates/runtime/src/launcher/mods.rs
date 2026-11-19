//! Local mods layer over the effective (updated) application.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/file_sys/patch_manager.cpp
use super::*;
use nixe_loader_content::{ExeFsArchive, ReadOnlyTree};
use nixe_loader_executable::NsoPatch;
use std::{collections::BTreeMap, fs, io};

#[derive(Default)]
pub(super) struct TitleMods {
    replacements: BTreeMap<String, PathBuf>,
    patches: Vec<PathBuf>,
    romfs: Vec<PathBuf>,
}

fn failure(path: &Path, msg: impl Display) -> LaunchError {
    LaunchError::invalid(LaunchStage::Modifications, path, msg.to_string())
}

fn entries(path: &Path) -> Result<Vec<PathBuf>, io::Error> {
    let mut files = fs::read_dir(path)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.sort_unstable();
    Ok(files)
}

impl TitleMods {
    pub(super) fn discover(root: Option<&Path>, title: ApplicationId) -> Result<Self, LaunchError> {
        let mut mods = Self::default();
        let Some(root) = root else { return Ok(mods) };
        let title_id = title.to_string();
        let title_dir = root.join(&title_id);
        // Exact casing is the usual layout. Also accept directories copied from
        // case-insensitive filesystems without descending into unrelated titles.
        let dirs = match entries(&title_dir) {
            Ok(dirs) => dirs,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let roots = match entries(root) {
                    Ok(paths) => paths,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(mods),
                    Err(e) => return Err(failure(root, e)),
                };
                let Some(other) = roots.iter().find(|p| {
                    p.file_name()
                        .is_some_and(|n| n.eq_ignore_ascii_case(&title_id))
                }) else {
                    return Ok(mods);
                };
                entries(other).map_err(|e| failure(other, e))?
            }
            Err(e) => return Err(failure(&title_dir, e)),
        };
        for package in dirs {
            if !package.is_dir() {
                continue;
            }
            for layer in entries(&package).map_err(|e| failure(&package, e))? {
                let name = layer.file_name().unwrap();
                if name.eq_ignore_ascii_case("romfs") && layer.is_dir() {
                    mods.romfs.push(layer);
                } else if name.eq_ignore_ascii_case("exefs") && layer.is_dir() {
                    for file in entries(&layer).map_err(|e| failure(&layer, e))? {
                        if !file.is_file() {
                            continue;
                        }
                        let ext = file.extension().unwrap_or_default();
                        if ext.eq_ignore_ascii_case("pchtxt") || ext.eq_ignore_ascii_case("ips") {
                            mods.patches.push(file);
                        } else if let Some(name) = file.file_name().and_then(|s| s.to_str())
                            && (name == "main.npdm" || module_role(name).is_some())
                        {
                            mods.replacements.entry(name.to_owned()).or_insert(file);
                        }
                    }
                }
            }
        }
        Ok(mods)
    }

    pub(super) fn exefs(&self, mut exefs: ExeFsArchive) -> Result<ExeFsArchive, LaunchError> {
        for (name, path) in &self.replacements {
            let source = FileStorage::open(path).map_err(|e| failure(path, e))?;
            exefs.replace_file(name, Arc::new(source));
            log::info!("applied ExeFS replacement {}", path.display());
        }
        Ok(exefs)
    }

    pub(super) fn attach_patches(&self, modules: &mut [LaunchModule]) -> Result<(), LaunchError> {
        if self.patches.is_empty() {
            return Ok(());
        }
        let ids: Vec<_> = modules
            .iter()
            .map(|module| {
                let id = module.image().executable().module_id();
                log::debug!(
                    "ExeFS {} build ID {}",
                    module.name(),
                    std::fmt::from_fn(|f| id.iter().try_for_each(|b| write!(f, "{b:02X}")))
                );
                *id
            })
            .collect();
        for path in &self.patches {
            let Some(patch) = NsoPatch::load(path, &ids).map_err(|e| failure(path, e))? else {
                continue;
            };
            let patch = Arc::new(patch);
            for (m, id) in modules.iter_mut().zip(&ids) {
                if id == &patch.build_id {
                    m.patches.push(Arc::clone(&patch));
                }
            }
        }
        Ok(())
    }

    pub(super) fn has_romfs(&self) -> bool {
        !self.romfs.is_empty()
    }

    pub(super) fn overlay_romfs(
        &self,
        mut tree: ReadOnlyTree,
    ) -> Result<ReadOnlyTree, LaunchError> {
        // The first named mod has the highest priority for replacement files.
        for dir in self.romfs.iter().rev() {
            let mut overlay = ReadOnlyTree::default();
            add_directory(dir, "", &mut overlay, &mut Vec::new())?;
            tree.overlay(overlay);
            log::info!("applied RomFS layer {}", dir.display());
        }
        Ok(tree)
    }
}

fn add_directory(
    dir: &Path,
    prefix: &str,
    tree: &mut ReadOnlyTree,
    parents: &mut Vec<PathBuf>,
) -> Result<(), LaunchError> {
    let real = dir.canonicalize().map_err(|e| failure(dir, e))?;
    if parents.contains(&real) {
        return Err(failure(dir, "cyclic directory link"));
    }
    parents.push(real);
    for path in entries(dir).map_err(|e| failure(dir, e))? {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| failure(&path, "invalid UTF-8 filename"))?;
        let guest = format!("{prefix}/{name}");
        let ty = fs::metadata(&path).map_err(|e| failure(&path, e))?;
        if ty.is_dir() {
            tree.insert_directory(&guest)
                .map_err(|e| failure(&path, e))?;
            add_directory(&path, &guest, tree, parents)?;
        } else if ty.is_file() {
            let file = FileStorage::open(&path).map_err(|e| failure(&path, e))?;
            tree.insert_file(&guest, Arc::new(file))
                .map_err(|e| failure(&path, e))?;
        }
    }
    parents.pop();
    Ok(())
}
