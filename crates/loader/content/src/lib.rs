//! Loaders for Nintendo Switch content containers and file-system images.

mod cnmt;
mod crypto;
mod exefs;
mod hfs0;
mod integrity;
mod keys;
mod nacp;
mod nca;
mod nsp;
mod pfs0;
mod romfs;
mod version;
mod xci;

pub use cnmt::{
    CnmtContentInfo, CnmtContentMeta, CnmtContentMetaInfo, CnmtContentType, CnmtExtendedHeader,
    CnmtInstallType, CnmtLoader, CnmtMetaType, CnmtPlatform,
};
pub use exefs::ExeFsLoader;
pub use hfs0::{Hfs0Archive, Hfs0Entry, Hfs0HashResult, Hfs0Loader};
pub use integrity::{IntegrityCheck, IntegrityCheckKind, IntegrityReport, IntegrityStatus};
pub use keys::{KeyAreaKeyIndex, KeySetError, NcaKeyProvider, NcaKeySet};
pub use nacp::{
    AddOnContentRegistrationType, ApplicationControlProperty, ApplicationTitle,
    AppropriateAgeForChina, CrashReportPolicy, DataLossConfirmation, HdcpPolicy, LogoHandling,
    LogoType, NACP_SIZE, NacpLanguage, NacpLoader, PlayLogPolicy, PlayLogQueryCapability,
    RuntimeAddOnContentInstall, RuntimeParameterDelivery, ScreenshotPolicy, StartupUserAccount,
    SupportedLanguages, UserAccountSwitchLock, VideoCapturePolicy,
};
pub use nca::{
    NcaArchive, NcaContentType, NcaDistributionType, NcaEncryptionType, NcaFormatVersion,
    NcaHeader, NcaLoader, NcaSection, NcaSectionType,
};
pub use nsp::{NspArchive, NspLoader};
pub use pfs0::{Pfs0Archive, Pfs0Entry, Pfs0Loader};
pub use romfs::{RomFsArchive, RomFsFile, RomFsLoader};
pub use version::{
    ApplicationVersion, ContentMetaVersion, DecodedContentMetaVersion, SystemVersion,
};
pub use xci::{
    XciArchive, XciHeader, XciLoader, XciPartition, XciPartitionKind, XciRootHeaderIntegrity,
};

/// Placeholder representation of content exposed by a parsed container.
///
/// Named entries, nested storage regions, integrity metadata, and content type
/// information will be added as the individual formats are implemented.
#[derive(Debug)]
pub struct LoadedContent;
