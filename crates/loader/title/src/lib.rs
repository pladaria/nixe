//! Discovery and resolution of complete Nintendo Switch titles.

mod catalog;
mod control;
mod discovery;
mod error;
mod inspection;
mod package;
mod package_content;
mod resolved_title;
mod resolver;

pub use catalog::TitleCatalog;
pub use control::{ControlIcon, ControlMetadata};
pub use discovery::{DirectoryScanOptions, PackageFormat};
pub use error::TitleError;
pub use inspection::{
    ContentMetaInspection, ContentRecordInspection, EntryInspection, EntryKind, InspectError,
    NcaInspection, NcaSectionInspection, NczInspection, NczSectionInspection, NroAssetsInspection,
    NroInspection, NroSegmentInspection, NroSegmentKind, PackageInspection,
    StandaloneNczInspection, TitleInspection, TitleInspector, XciInspection,
    XciPartitionInspection,
};
pub use nixe_loader_content::{
    AddOnContentRegistrationType, ApplicationControlProperty, ApplicationTitle, ApplicationVersion,
    AppropriateAgeForChina, CnmtContentInfo, CnmtContentMeta, CnmtContentMetaInfo, CnmtContentType,
    CnmtExtendedHeader, CnmtInstallType, CnmtMetaType, CnmtPlatform, ContentMetaVersion,
    CrashReportPolicy, DataLossConfirmation, DecodedContentMetaVersion, HdcpPolicy, LogoHandling,
    LogoType, NacpLanguage, PlayLogPolicy, PlayLogQueryCapability, RuntimeAddOnContentInstall,
    RuntimeParameterDelivery, ScreenshotPolicy, StartupUserAccount, SupportedLanguages,
    SystemVersion, UserAccountSwitchLock, VideoCapturePolicy,
};
pub use package::{ApplicationId, ContentType, PackageMetadata, PackageMetadataError, TitleId};
pub use resolved_title::ResolvedTitle;
pub use resolver::TitleResolver;
