use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::LoadedContent;

/// Loads Nintendo Content Archive (NCA) files.
///
/// NCA is the core archive used for Switch programs, data, control information,
/// and other content. Unlike NSP and XCI, it is an individual content unit and
/// can add encryption, integrity checks, and several internal file-system
/// sections rather than packaging a complete distribution by itself.
#[derive(Debug)]
pub struct NcaLoader;

impl FormatLoader for NcaLoader {
    type Output = LoadedContent;

    const FORMAT_NAME: &'static str = "NCA";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
