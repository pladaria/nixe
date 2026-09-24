#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    InvalidGuestCode,
    Unsupported,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    pub(crate) kind: Kind,
    pub(crate) detail: Box<str>,
}

impl Error {
    pub(crate) fn invalid(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: Kind::InvalidGuestCode,
            detail: detail.into(),
        }
    }

    pub(crate) fn unsupported(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: Kind::Unsupported,
            detail: detail.into(),
        }
    }

    pub(crate) fn internal(detail: impl Into<Box<str>>) -> Self {
        Self {
            kind: Kind::Internal,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for Error {}
