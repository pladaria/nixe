//! Discovery and resolution of complete Nintendo Switch titles.

mod catalog;
mod error;
mod inspection;
mod package;
mod resolved_title;
mod resolver;

pub use catalog::TitleCatalog;
pub use error::TitleError;
pub use inspection::{
    ContentMetaInspection, ContentRecordInspection, EntryInspection, EntryKind, InspectError,
    NcaInspection, NcaSectionInspection, PackageFormat, PackageInspection, TitleInspection,
    TitleInspector,
};
pub use package::{ApplicationId, ContentType, PackageMetadata, TitleId};
pub use resolved_title::ResolvedTitle;
pub use resolver::TitleResolver;
pub use swiitx_loader_content::{
    CnmtContentInfo, CnmtContentMeta, CnmtContentMetaInfo, CnmtContentType, CnmtExtendedHeader,
    CnmtInstallType, CnmtMetaType, CnmtPlatform,
};
