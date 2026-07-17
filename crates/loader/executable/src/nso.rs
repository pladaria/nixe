use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::ExecutableImage;

/// Loads Nintendo Shared Object (NSO) files.
///
/// NSO is the executable format used by official Switch software and is
/// commonly stored inside ExeFS. Its segments may be compressed and it carries
/// module metadata, unlike the homebrew-oriented NRO format.
#[derive(Debug)]
pub struct NsoLoader;

impl FormatLoader for NsoLoader {
    type Output = ExecutableImage;

    const FORMAT_NAME: &'static str = "NSO";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
