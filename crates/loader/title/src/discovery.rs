use std::fs;
use std::path::{Path, PathBuf};

/// Controls how package files are discovered below a directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryScanOptions {
    /// Whether files in nested directories are included.
    pub recursive: bool,
}

impl DirectoryScanOptions {
    /// Creates directory scan options with recursive discovery enabled.
    pub const fn new() -> Self {
        Self { recursive: true }
    }

    /// Sets whether files in nested directories are included.
    pub const fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

impl Default for DirectoryScanOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Container format recognized while discovering title packages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageFormat {
    /// Nintendo Submission Package backed by PFS0.
    Nsp,
    /// NX Card Image backed by nested HFS0 partitions.
    Xci,
}

pub(crate) struct DirectoryDiscoveryError {
    pub path: PathBuf,
    pub source: std::io::Error,
}

pub(crate) fn directory_files(
    path: &Path,
    options: DirectoryScanOptions,
) -> Result<Vec<PathBuf>, DirectoryDiscoveryError> {
    let mut directories = vec![path.to_owned()];
    let mut files = Vec::new();

    while let Some(directory) = directories.pop() {
        let entries = fs::read_dir(&directory).map_err(|source| DirectoryDiscoveryError {
            path: directory.clone(),
            source,
        })?;
        let mut nested_directories = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|source| DirectoryDiscoveryError {
                path: directory.clone(),
                source,
            })?;
            let entry_path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|source| DirectoryDiscoveryError {
                    path: entry_path.clone(),
                    source,
                })?;
            if file_type.is_file() {
                files.push(entry_path);
            } else if options.recursive && file_type.is_dir() {
                nested_directories.push(entry_path);
            }
        }

        // Reverse sorting makes the lexically first directory the next one
        // visited by the LIFO stack.
        nested_directories.sort_by(|left, right| right.cmp(left));
        directories.extend(nested_directories);
    }

    // A global sort defines discovery order independently of directory entry
    // enumeration and traversal order.
    files.sort();
    Ok(files)
}

pub(crate) fn package_format(path: &Path) -> Option<PackageFormat> {
    let extension = path.extension()?.to_str()?;
    if extension.eq_ignore_ascii_case("nsp") {
        Some(PackageFormat::Nsp)
    } else if extension.eq_ignore_ascii_case("xci") {
        Some(PackageFormat::Xci)
    } else {
        None
    }
}
