use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::LoadedContent;

/// Loads NX Card Image (XCI) files.
///
/// XCI represents the contents and partition layout of a physical Nintendo
/// Switch game card. Unlike the installation-oriented NSP package, it includes
/// game-card-specific structure and may contain secure, update, normal, and
/// logo partitions.
#[derive(Debug)]
pub struct XciLoader;

impl FormatLoader for XciLoader {
    type Output = LoadedContent;

    const FORMAT_NAME: &'static str = "XCI";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
