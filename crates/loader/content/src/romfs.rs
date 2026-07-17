use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::LoadedContent;

/// Loads Read-Only File System (RomFS) images.
///
/// RomFS stores a title's read-only files, such as assets, data, and resources.
/// Unlike ExeFS, it is a hierarchical data file system and does not primarily
/// contain executable modules.
#[derive(Debug)]
pub struct RomFsLoader;

impl FormatLoader for RomFsLoader {
    type Output = LoadedContent;

    const FORMAT_NAME: &'static str = "RomFS";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
