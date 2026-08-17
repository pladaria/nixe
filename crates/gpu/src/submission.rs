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

/// Zero-based backend transaction within one logical frontend submission.
///
/// Most frontend submissions contain one final segment at index zero. A
/// frontend may use additional ordered segments when canonical-memory updates
/// must become visible between otherwise neutral GPU operations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct FrontendSubmissionSegment(u32);

impl FrontendSubmissionSegment {
    pub const FIRST: Self = Self(0);

    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// Identity of one initialized backend instance.
///
/// It is assigned by the composition root and is neither a host pointer nor a
/// graphics-API object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct BackendInstanceId(u64);

impl BackendInstanceId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for BackendInstanceId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "backend-instance=0x{:016x}", self.0)
    }
}

/// Pointer-free, backend-scoped token for one accepted neutral submission.
///
/// The generation changes whenever a released slot is reused. Consequently a
/// stale token cannot name later work, even within the same backend instance.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BackendSubmissionToken {
    instance: BackendInstanceId,
    slot: u64,
    generation: u32,
}

impl BackendSubmissionToken {
    #[must_use]
    pub const fn new(instance: BackendInstanceId, slot: u64, generation: u32) -> Self {
        Self {
            instance,
            slot,
            generation,
        }
    }

    #[must_use]
    pub const fn instance(self) -> BackendInstanceId {
        self.instance
    }

    #[must_use]
    pub const fn slot(self) -> u64 {
        self.slot
    }

    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl Display for BackendSubmissionToken {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "backend-submission[{} slot=0x{:016x} generation={}]",
            self.instance, self.slot, self.generation
        )
    }
}

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
        let instance = BackendInstanceId::new(5);
        let backend = BackendSubmissionToken::new(instance, 7, 2);
        let host = HostCompletion::new(backend);
        let visibility = VisibilityCompletion::new(DeviceVisibilityPoint::new(11));

        assert_eq!(frontend.get(), 3);
        assert_eq!(backend.instance(), instance);
        assert_eq!(backend.slot(), 7);
        assert_eq!(backend.generation(), 2);
        assert_eq!(host.submission(), backend);
        assert_eq!(visibility.point(), DeviceVisibilityPoint::new(11));
        assert_eq!(
            frontend.to_string(),
            "frontend-submission=0x0000000000000003"
        );
        assert_eq!(
            backend.to_string(),
            "backend-submission[backend-instance=0x0000000000000005 slot=0x0000000000000007 generation=2]"
        );
        assert_eq!(
            host.to_string(),
            "host-completion[backend-submission[backend-instance=0x0000000000000005 slot=0x0000000000000007 generation=2]]"
        );
        assert_eq!(
            visibility.to_string(),
            "visibility-completion[visibility-point=0x000000000000000b]"
        );
    }
}
