//! Loaders for Nintendo Switch content containers and file-system images.

mod binary;
mod bktr;
mod cnmt;
mod compressed_package;
mod crypto;
mod exefs;
mod hfs0;
mod integrity;
mod keys;
mod nacp;
mod nca;
mod ncz;
mod nsp;
mod nsz;
mod nup;
mod pfs0;
mod romfs;
mod tree;
mod version;
mod xci;
mod xcz;
mod zstd_stream;

pub use bktr::{BktrPatch, BucketTreeHeader};
pub use cnmt::{
    CnmtContentInfo, CnmtContentMeta, CnmtContentMetaInfo, CnmtContentType, CnmtExtendedHeader,
    CnmtInstallType, CnmtLoader, CnmtMetaType, CnmtPlatform,
};
pub use compressed_package::CompressedPackageEntry;
pub use exefs::{ExeFsArchive, ExeFsLoader};
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
    BktrPatchInfo, NcaArchive, NcaContentType, NcaDistributionType, NcaEncryptionType,
    NcaFormatVersion, NcaHeader, NcaLoader, NcaSection, NcaSectionType,
};
pub use ncz::{NczArchive, NczBlockInfo, NczCompressionKind, NczLoader, NczSection};
pub use nsp::{NspArchive, NspLoader};
pub use nsz::{NszArchive, NszLoader, SolidBackup, prepare_nsz_for_random_access};
pub use nup::{
    NupArchive, NupCodec, NupDirectory, NupFile, NupLoader, NupMetadata, NupPackage, NupPackageKind,
};
pub use pfs0::{Pfs0Archive, Pfs0Entry, Pfs0Loader};
pub use romfs::{RomFsArchive, RomFsFile, RomFsLoader};
pub use tree::ReadOnlyTree;
pub use version::{
    ApplicationVersion, ContentMetaVersion, DecodedContentMetaVersion, SystemVersion,
};
pub use xci::{
    XciArchive, XciHeader, XciLoader, XciPartition, XciPartitionKind, XciRootHeaderIntegrity,
};
pub use xcz::{XczArchive, XczLoader, XczPartition};
