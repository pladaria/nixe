use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef};

use crate::LoadedContent;

/// Loads Nintendo Submission Package (NSP) files.
///
/// NSP is the package commonly used for digitally distributed titles, updates,
/// and downloadable content. It groups NCAs and related installation metadata
/// in a package, unlike XCI, which models the partitions and metadata of a
/// physical game card.
#[derive(Debug)]
pub struct NspLoader;

impl FormatLoader for NspLoader {
    type Output = LoadedContent;

    const FORMAT_NAME: &'static str = "NSP";

    fn load(_storage: StorageRef) -> Result<Self::Output, LoadError> {
        Err(LoadError::NotImplemented {
            format: Self::FORMAT_NAME,
        })
    }
}
