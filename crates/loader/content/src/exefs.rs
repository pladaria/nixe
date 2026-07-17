use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::LoadedContent;

/// Loads Executable File System (ExeFS) images.
///
/// ExeFS is the file-system section that normally contains a title's NSO
/// modules and process metadata. It differs from RomFS because it describes the
/// executable side of a title rather than its read-only assets and resources.
#[derive(Debug)]
pub struct ExeFsLoader;

impl FormatLoader for ExeFsLoader {
    type Output = LoadedContent;

    const FORMAT_NAME: &'static str = "ExeFS";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
