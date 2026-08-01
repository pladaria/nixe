//! Host-independent submission and completion identities.

use std::fmt::{Display, Formatter};

use nixe_memory::DeviceVisibilityPoint;

macro_rules! opaque_submission_id {
    ($(#[$meta:meta])* $name:ident, $label:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Creates an identity from a value assigned by its owner.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Returns the owner-assigned numeric representation.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, concat!($label, "=0x{:016x}"), self.0)
            }
        }
    };
}

opaque_submission_id!(
    /// Identity assigned to work accepted by a console GPU frontend.
    ///
    /// It is neither a guest fence value nor a concrete backend handle.
    FrontendSubmissionId,
    "frontend-submission"
);
opaque_submission_id!(
    /// Opaque token assigned when a backend accepts one neutral submission.
    ///
    /// The numeric representation has no guest meaning and is not a host
    /// pointer or graphics-API object.
    BackendSubmissionToken,
    "backend-submission"
);

/// Evidence that the host backend completed one accepted submission.
///
/// Completion does not imply that device writes have become visible to guest
/// memory or that a guest timeline point may be signaled.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HostCompletion {
    submission: BackendSubmissionToken,
}

impl HostCompletion {
    /// Records completion reported by the backend which owns the token.
    #[must_use]
    pub const fn new(submission: BackendSubmissionToken) -> Self {
        Self { submission }
    }

    /// Returns the backend submission which completed.
    #[must_use]
    pub const fn submission(self) -> BackendSubmissionToken {
        self.submission
    }
}

impl Display for HostCompletion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "host-completion[{}]", self.submission)
    }
}

/// Evidence that declared device writes reached a memory visibility point.
///
/// This remains distinct from host queue completion and from a guest fence.
/// A coordinator may establish it with a download and cache operations, or
/// prove that movement is unnecessary on a coherent shared-memory host.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VisibilityCompletion {
    point: DeviceVisibilityPoint,
}

impl VisibilityCompletion {
    /// Records a transition completed by the memory visibility coordinator.
    #[must_use]
    pub const fn new(point: DeviceVisibilityPoint) -> Self {
        Self { point }
    }

    /// Returns the completed device visibility point.
    #[must_use]
    pub const fn point(self) -> DeviceVisibilityPoint {
        self.point
    }
}

impl Display for VisibilityCompletion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "visibility-completion[{}]", self.point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_domains_are_typed_and_pointer_free() {
        let frontend = FrontendSubmissionId::new(3);
        let backend = BackendSubmissionToken::new(7);
        let host = HostCompletion::new(backend);
        let visibility = VisibilityCompletion::new(DeviceVisibilityPoint::new(11));

        assert_eq!(frontend.get(), 3);
        assert_eq!(backend.get(), 7);
        assert_eq!(host.submission(), backend);
        assert_eq!(visibility.point(), DeviceVisibilityPoint::new(11));
        assert_eq!(
            frontend.to_string(),
            "frontend-submission=0x0000000000000003"
        );
        assert_eq!(backend.to_string(), "backend-submission=0x0000000000000007");
        assert_eq!(
            host.to_string(),
            "host-completion[backend-submission=0x0000000000000007]"
        );
        assert_eq!(
            visibility.to_string(),
            "visibility-completion[visibility-point=0x000000000000000b]"
        );
    }
}
