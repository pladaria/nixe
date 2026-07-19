//! IR operations and their architectural source metadata.

use crate::location::LocationDescriptor;

/// An IR operation that may fault, with mandatory precise guest location.
///
/// The operation payload remains generic until the typed IR operation set is
/// introduced. Any may-fault variant must use this envelope rather than carry a
/// bare payload.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FaultingOperation<T> {
    /// Guest instruction whose semantics the operation implements.
    pub location: LocationDescriptor,
    /// Typed IR operation payload.
    pub operation: T,
}

impl<T> FaultingOperation<T> {
    /// Attaches the source required for precise exception reconstruction.
    #[must_use]
    pub const fn new(location: LocationDescriptor, operation: T) -> Self {
        Self {
            location,
            operation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{address::GuestVirtualAddress, location::ExecutionState, profile::CpuProfileId};

    #[test]
    fn faulting_operation_requires_a_location() {
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x8000),
            ExecutionState::A64,
            CpuProfileId::new(3),
        );
        let operation = FaultingOperation::new(location, "load-i32");

        assert_eq!(operation.location, location);
        assert_eq!(operation.operation, "load-i32");
    }
}
