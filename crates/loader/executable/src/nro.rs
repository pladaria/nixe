use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::ExecutableImage;

/// Loads Nintendo Relocatable Object (NRO) files.
///
/// NRO is the executable format normally used by Nintendo Switch homebrew. It
/// carries its code and data segments in one directly loadable file, unlike an
/// NSO, which is normally retrieved from an official title's ExeFS partition.
#[derive(Debug)]
pub struct NroLoader;

impl FormatLoader for NroLoader {
    type Output = ExecutableImage;

    const FORMAT_NAME: &'static str = "NRO";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
