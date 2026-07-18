//! Discovery and resolution of complete Nintendo Switch titles.

mod catalog;
mod control;
mod discovery;
mod error;
mod inspection;
mod nsp_metadata;
mod package;
mod resolved_title;
mod resolver;

pub use catalog::TitleCatalog;
pub use control::{ControlIcon, ControlMetadata};
pub use discovery::{DirectoryScanOptions, PackageFormat};
pub use error::TitleError;
pub use inspection::{
    ContentMetaInspection, ContentRecordInspection, EntryInspection, EntryKind, InspectError,
    NcaInspection, NcaSectionInspection, PackageInspection, TitleInspection, TitleInspector,
};
pub use package::{ApplicationId, ContentType, PackageMetadata, PackageMetadataError, TitleId};
pub use resolved_title::ResolvedTitle;
pub use resolver::TitleResolver;
pub use swiitx_loader_content::{
    AddOnContentRegistrationType, ApplicationControlProperty, ApplicationTitle, ApplicationVersion,
    AppropriateAgeForChina, CnmtContentInfo, CnmtContentMeta, CnmtContentMetaInfo, CnmtContentType,
    CnmtExtendedHeader, CnmtInstallType, CnmtMetaType, CnmtPlatform, ContentMetaVersion,
    CrashReportPolicy, DataLossConfirmation, DecodedContentMetaVersion, HdcpPolicy, LogoHandling,
    LogoType, NacpLanguage, PlayLogPolicy, PlayLogQueryCapability, RuntimeAddOnContentInstall,
    RuntimeParameterDelivery, ScreenshotPolicy, StartupUserAccount, SupportedLanguages,
    SystemVersion, UserAccountSwitchLock, VideoCapturePolicy,
};
