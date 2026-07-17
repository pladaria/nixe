use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::ApplicationId;

/// Errors produced while discovering or resolving titles.
#[derive(Debug, PartialEq, Eq)]
pub enum TitleError {
    /// The requested operation depends on content parsing not yet available.
    NotImplemented { operation: &'static str },
    /// No base application was found for an application identifier.
    MissingBase { application_id: ApplicationId },
    /// More than one base application matched the same identifier.
    ConflictingBases {
        application_id: ApplicationId,
        count: usize,
    },
}

impl Display for TitleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented { operation } => {
                write!(formatter, "{operation} is not implemented")
            }
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
        }
    }
}

impl Error for TitleError {}
