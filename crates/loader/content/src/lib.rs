//! Loaders for Nintendo Switch content containers and file-system images.

mod crypto;
mod exefs;
mod integrity;
mod keys;
mod nca;
mod nsp;
mod pfs0;
mod romfs;
mod xci;

pub use exefs::ExeFsLoader;
pub use integrity::{IntegrityCheck, IntegrityCheckKind, IntegrityReport, IntegrityStatus};
pub use keys::{KeyAreaKeyIndex, KeySetError, NcaKeyProvider, NcaKeySet};
pub use nca::{
    NcaArchive, NcaContentType, NcaDistributionType, NcaEncryptionType, NcaFormatVersion,
    NcaHeader, NcaLoader, NcaSection, NcaSectionType,
};
pub use nsp::{NspArchive, NspLoader};
pub use pfs0::Pfs0Entry;
pub use romfs::RomFsLoader;
pub use xci::XciLoader;

/// Placeholder representation of content exposed by a parsed container.
///
/// Named entries, nested storage regions, integrity metadata, and content type
/// information will be added as the individual formats are implemented.
#[derive(Debug)]
pub struct LoadedContent;
