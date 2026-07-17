//! Loaders for Nintendo Switch content containers and file-system images.

mod exefs;
mod nca;
mod nsp;
mod romfs;
mod xci;

pub use exefs::ExeFsLoader;
pub use nca::NcaLoader;
pub use nsp::NspLoader;
pub use romfs::RomFsLoader;
pub use xci::XciLoader;

/// Placeholder representation of content exposed by a parsed container.
///
/// Named entries, nested storage regions, integrity metadata, and content type
/// information will be added as the individual formats are implemented.
#[derive(Debug)]
pub struct LoadedContent;
