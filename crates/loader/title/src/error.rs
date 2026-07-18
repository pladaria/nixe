use std::error::Error;
use std::fmt::{Display, Formatter};

use swiitx_loader_content::CnmtMetaType;

use crate::ApplicationId;

/// Errors produced while discovering or resolving titles.
#[derive(Debug, PartialEq, Eq)]
pub enum TitleError {
    /// The requested operation depends on content parsing not yet available.
    NotImplemented { operation: &'static str },
    /// The parsed content-meta role cannot be represented in the title catalog.
    UnsupportedContentMetaType { content_meta_type: CnmtMetaType },
    /// The content-meta extended header does not match its declared role.
    IncompatibleContentMetaHeader { content_meta_type: CnmtMetaType },
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
            Self::UnsupportedContentMetaType { content_meta_type } => write!(
                formatter,
                "content-meta type {content_meta_type} is not supported by the title catalog"
            ),
            Self::IncompatibleContentMetaHeader { content_meta_type } => write!(
                formatter,
                "content-meta type {content_meta_type} has an incompatible extended header"
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
        }
    }
}

impl Error for TitleError {}
