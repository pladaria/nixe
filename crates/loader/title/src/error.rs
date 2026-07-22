use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use nixe_loader_storage::LoadError;

use crate::{ApplicationId, ApplicationVersion, ContentType, PackageMetadataError, TitleId};

/// Errors produced while discovering or resolving titles.
#[derive(Debug)]
pub enum TitleError {
    /// A local file-system operation failed while discovering packages.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The supplied catalog path is not a directory.
    NotDirectory { path: PathBuf },
    /// The directory contains no package format supported by the catalog.
    NoSupportedPackages { path: PathBuf },
    /// A supported package or its canonical metadata could not be loaded.
    Package { path: PathBuf, source: LoadError },
    /// Canonical content metadata could not be represented in the catalog.
    PackageMetadata {
        path: PathBuf,
        source: PackageMetadataError,
    },
    /// No base application was found for an application identifier.
    MissingBase { application_id: ApplicationId },
    /// More than one base application matched the same identifier.
    ConflictingBases {
        application_id: ApplicationId,
        count: usize,
    },
    /// Canonically different packages claim the same logical coordinate.
    ConflictingPackages {
        application_id: ApplicationId,
        content_type: ContentType,
        title_id: TitleId,
        version: ApplicationVersion,
        count: usize,
    },
    /// A patch relationship does not match the patch declared by its base.
    IncompatiblePatchRelationship {
        application_id: ApplicationId,
        expected_patch_id: TitleId,
        patch_id: TitleId,
    },
    /// The base requires a patch version that is not available.
    MissingCompatiblePatch {
        application_id: ApplicationId,
        required_application_version: ApplicationVersion,
        newest_available_version: Option<ApplicationVersion>,
    },
    /// No revision of one add-on supports the resolved application version.
    IncompatibleAddOnContent {
        application_id: ApplicationId,
        title_id: TitleId,
        required_application_version: ApplicationVersion,
        actual_application_version: ApplicationVersion,
    },
}

impl Display for TitleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "cannot access {}: {source}", path.display())
            }
            Self::NotDirectory { path } => {
                write!(
                    formatter,
                    "title catalog path is not a directory: {}",
                    path.display()
                )
            }
            Self::NoSupportedPackages { path } => write!(
                formatter,
                "no supported title packages found at {}",
                path.display()
            ),
            Self::Package { path, source } => {
                write!(
                    formatter,
                    "cannot load title package {}: {source}",
                    path.display()
                )
            }
            Self::PackageMetadata { path, source } => write!(
                formatter,
                "cannot catalog title package {}: {source}",
                path.display()
            ),
            Self::MissingBase { application_id } => {
                write!(formatter, "no base application found for {application_id}")
            }
            Self::ConflictingBases {
                application_id,
                count,
            } => write!(
                formatter,
                "found {count} base applications for {application_id}"
            ),
            Self::ConflictingPackages {
                application_id,
                content_type,
                title_id,
                version,
                count,
            } => write!(
                formatter,
                "found {count} conflicting {content_type:?} packages for {application_id} at title {title_id} version {version}"
            ),
            Self::IncompatiblePatchRelationship {
                application_id,
                expected_patch_id,
                patch_id,
            } => write!(
                formatter,
                "patch {patch_id} for {application_id} does not match declared patch {expected_patch_id}"
            ),
            Self::MissingCompatiblePatch {
                application_id,
                required_application_version,
                newest_available_version,
            } => match newest_available_version {
                Some(version) => write!(
                    formatter,
                    "application {application_id} requires patch version {required_application_version}, but newest available version is {version}"
                ),
                None => write!(
                    formatter,
                    "application {application_id} requires patch version {required_application_version}, but no patch is available"
                ),
            },
            Self::IncompatibleAddOnContent {
                application_id,
                title_id,
                required_application_version,
                actual_application_version,
            } => write!(
                formatter,
                "add-on {title_id} requires application version {required_application_version}, but {application_id} resolves to version {actual_application_version}"
            ),
        }
    }
}

impl Error for TitleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Package { source, .. } => Some(source),
            Self::PackageMetadata { source, .. } => Some(source),
            Self::NotDirectory { .. }
            | Self::NoSupportedPackages { .. }
            | Self::MissingBase { .. }
            | Self::ConflictingBases { .. }
            | Self::ConflictingPackages { .. }
            | Self::IncompatiblePatchRelationship { .. }
            | Self::MissingCompatiblePatch { .. }
            | Self::IncompatibleAddOnContent { .. } => None,
        }
    }
}
