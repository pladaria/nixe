use std::fmt::{Debug, Display, Formatter};

use swiitx_loader_storage::StorageRef;

/// Identifies an application to which base, patch, and add-on content belongs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApplicationId(u64);

impl ApplicationId {
    /// Creates an application identifier from its raw value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw identifier value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for ApplicationId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:016X}", self.0)
    }
}

/// Identifies one concrete application, patch, or add-on title.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TitleId(u64);

impl TitleId {
    /// Creates a title identifier from its raw value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw identifier value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for TitleId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:016X}", self.0)
    }
}

/// Describes the role of a package within a complete title.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentType {
    /// Base game or application.
    Application,
    /// Update associated with an application.
    Patch,
    /// Downloadable content associated with an application.
    AddOnContent,
    /// Incremental data used while constructing a patch.
    Delta,
}

/// Metadata extracted from one NSP, NSZ, XCI, or equivalent package.
#[derive(Clone)]
pub struct PackageMetadata {
    /// Identifier of this concrete content title.
    pub title_id: TitleId,
    /// Identifier of the application to which this package belongs.
    pub application_id: ApplicationId,
    /// Package version obtained from its content metadata.
    pub version: u32,
    /// Role of this package in the resolved title.
    pub content_type: ContentType,
    /// Random-access source containing the package.
    pub source: StorageRef,
}

impl Debug for PackageMetadata {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PackageMetadata")
            .field("title_id", &self.title_id)
            .field("application_id", &self.application_id)
            .field("version", &self.version)
            .field("content_type", &self.content_type)
            .field("source", &"<storage>")
            .finish()
    }
}
