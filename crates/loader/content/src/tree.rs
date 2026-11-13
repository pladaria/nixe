//! Immutable file sources and directory overlays, independent of container encoding.

use std::collections::BTreeMap;

use nixe_loader_storage::{LoadError, StorageRef};

#[derive(Clone)]
pub(crate) enum TreeEntry {
    Directory,
    File(StorageRef),
}

/// A normalized absolute-path namespace retaining bounded, lazy file sources.
#[derive(Clone)]
pub struct ReadOnlyTree {
    pub(crate) entries: BTreeMap<String, TreeEntry>,
}

impl Default for ReadOnlyTree {
    fn default() -> Self {
        Self {
            entries: BTreeMap::from([("/".into(), TreeEntry::Directory)]),
        }
    }
}

impl ReadOnlyTree {
    pub fn insert_directory(&mut self, path: &str) -> Result<(), LoadError> {
        self.insert(path, TreeEntry::Directory)
    }

    pub fn insert_file(&mut self, path: &str, source: StorageRef) -> Result<(), LoadError> {
        self.insert(path, TreeEntry::File(source))
    }

    fn insert(&mut self, path: &str, entry: TreeEntry) -> Result<(), LoadError> {
        let (parent, name) = path
            .rsplit_once('/')
            .ok_or_else(|| invalid("path is not absolute"))?;
        if !path.starts_with('/')
            || path[1..]
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
            || name.is_empty()
            || name == "."
            || name == ".."
            || name.contains('\0')
            || name.len() > 0x1000
        {
            return Err(invalid("invalid path component"));
        }
        let parent = if parent.is_empty() { "/" } else { parent };
        if !matches!(self.entries.get(parent), Some(TreeEntry::Directory)) {
            return Err(invalid(format!("parent directory is missing: {parent}")));
        }
        if self.entries.contains_key(path) {
            return Err(invalid(format!("duplicate entry: {path}")));
        }
        self.entries.insert(path.into(), entry);
        Ok(())
    }

    /// Later files replace earlier files; directories merge recursively. Replacing
    /// a directory with a file removes its children from the effective namespace.
    pub fn overlay(&mut self, update: Self) {
        for (path, entry) in update.entries {
            if matches!(entry, TreeEntry::File(_))
                && matches!(self.entries.get(&path), Some(TreeEntry::Directory))
            {
                let prefix = format!("{path}/");
                let children: Vec<_> = self
                    .entries
                    .range(prefix.clone()..)
                    .take_while(|(name, _)| name.starts_with(&prefix))
                    .map(|(name, _)| name.clone())
                    .collect();
                for child in children {
                    self.entries.remove(&child);
                }
            }
            self.entries.insert(path, entry);
        }
    }
}

fn invalid(reason: impl Into<String>) -> LoadError {
    LoadError::invalid("file tree", reason)
}
