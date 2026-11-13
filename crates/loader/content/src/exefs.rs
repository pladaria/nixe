use crate::nca::{NcaSection, NcaSectionType};
use crate::pfs0::Pfs0Archive;
use crate::tree::{ReadOnlyTree, TreeEntry};
use nixe_loader_storage::{FormatLoader, LoadError, StorageRef};
use std::collections::BTreeMap;

/// Loads the PFS0 encoding of an executable filesystem.
#[derive(Debug)]
pub struct ExeFsLoader;

impl ExeFsLoader {
    pub fn load_nca_section(section: &NcaSection) -> Result<ExeFsArchive, LoadError> {
        if section.section_type() != NcaSectionType::Pfs0 {
            return Err(LoadError::invalid("ExeFS", "NCA section is not PFS0"));
        }
        Self::load(section.payload_storage()?)
    }
}

impl FormatLoader for ExeFsLoader {
    type Output = ExeFsArchive;
    const FORMAT_NAME: &'static str = "ExeFS";
    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        let pfs0 = Pfs0Archive::parse(storage, Self::FORMAT_NAME)?;
        let mut tree = ReadOnlyTree::default();
        for entry in pfs0.entries() {
            tree.insert_file(&format!("/{}", entry.name()), pfs0.open_entry(entry)?)?;
        }
        ExeFsArchive::from_tree(tree)
    }
}

/// Executable files shared by native package trees and PFS0 images.
#[derive(Clone)]
pub struct ExeFsArchive {
    files: BTreeMap<String, StorageRef>,
}

impl ExeFsArchive {
    pub fn from_tree(tree: ReadOnlyTree) -> Result<Self, LoadError> {
        let mut files = BTreeMap::new();
        for (path, entry) in tree.entries {
            if path == "/" {
                continue;
            }
            let TreeEntry::File(source) = entry else {
                return Err(LoadError::invalid(
                    "ExeFS",
                    "executable tree contains a directory",
                ));
            };
            let name = path.strip_prefix('/').expect("normalized tree path");
            if name.contains('/') {
                return Err(LoadError::invalid(
                    "ExeFS",
                    "executable file is not at the root",
                ));
            }
            files.insert(name.into(), source);
        }
        Ok(Self { files })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    pub fn open(&self, name: &str) -> Option<StorageRef> {
        self.files.get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.files.contains_key(name)
    }
}

impl std::fmt::Debug for ExeFsArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExeFsArchive")
            .field("files", &self.files.keys())
            .finish()
    }
}
