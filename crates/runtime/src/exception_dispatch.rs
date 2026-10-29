//! Backend-independent runtime contract for architectural exception dispatch.
//!
//! CPU backends report an [`ExceptionDispatchRequest`]. Runtime policy
//! handles it and returns an [`ExceptionDispatchOutcome`] which an outer
//! execution loop applies. Dispatch does not access native CPU internals.

use crate::GuestCpuState;
use nixe_cpu::location::LocationDescriptor;
use nixe_memory::GuestVirtualAddress;
use nixe_scheduler::{GuestThreadId, VirtualCpuId};
use std::sync::Arc;

use crate::ThreadObject;

/// Maximum number of guest bytes retained from a fatal break payload.
pub const MAX_GUEST_BREAK_PAYLOAD_BYTES: usize = 32;

/// Pointer-free snapshot of a small guest diagnostic payload.
///
/// Exception dispatch captures this while guest memory is still available so
/// later teardown and reporting never need to dereference a stale guest
/// address. Larger payloads remain represented by their original address and
/// size only.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct GuestBreakPayload {
    bytes: [u8; MAX_GUEST_BREAK_PAYLOAD_BYTES],
    len: u8,
}

impl GuestBreakPayload {
    #[must_use]
    pub fn new(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() || bytes.len() > MAX_GUEST_BREAK_PAYLOAD_BYTES {
            return None;
        }
        let mut snapshot = [0; MAX_GUEST_BREAK_PAYLOAD_BYTES];
        snapshot[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            bytes: snapshot,
            len: bytes.len() as u8,
        })
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }
}

impl std::fmt::Debug for GuestBreakPayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("0x")?;
        for byte in self.as_bytes() {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

pub use nixe_cpu::execution::ExceptionDispatchRequest;

/// Guest continuation selected after handling an exception.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionResume {
    /// Continue after the exception-raising instruction.
    ///
    /// For a supervisor call this advances by the four-byte A64 encoding width.
    Next,
    /// Execute the faulting instruction again after the blocking condition has
    /// been resolved.
    Retry,
}

/// Runtime object whose lifetime an exception handler requests to end.
///
/// A current-thread exit ends the process only when its last started thread
/// terminates. Paused or waiting started threads continue to keep it alive.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionTerminationScope {
    CurrentThread,
    Process,
}

/// Pointer-free architectural reason accompanying a termination request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionTerminationReason {
    /// An ordinary process or thread exit SVC.
    Requested,
    /// A fatal Horizon-style break carrying its guest diagnostic payload.
    Break {
        reason: u64,
        info: u64,
        size: u64,
        payload: Option<GuestBreakPayload>,
    },
}

/// Runtime decision produced by an exception dispatcher.
///
/// `Fault` is generic so each platform layer can retain a typed diagnostic
/// without forcing Horizon result-code policy into the CPU or generic runtime
/// contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionDispatchOutcome<F> {
    /// Handling completed and guest execution may continue immediately.
    Resume(ExceptionResume),
    /// The thread is suspended and retains an explicit eventual continuation.
    Suspend(ExceptionResume),
    /// The guest-visible operation was rejected, but execution may continue.
    ///
    /// Platform policy must install its stable guest ABI result before
    /// returning this outcome. The typed diagnostic is retained for host-side
    /// reporting and is never copied into guest state by the generic runtime.
    /// Rejection always continues after the exception-raising instruction;
    /// operations that can retry must use [`Self::Suspend`] explicitly.
    Reject { diagnostic: F },
    /// The selected runtime object requested deterministic termination.
    Terminate {
        scope: ExceptionTerminationScope,
        exit_code: u64,
        reason: ExceptionTerminationReason,
    },
    /// Dispatch failed and guest execution must not continue.
    Fault(F),
}

/// Observable lifecycle result after a dispatch outcome has been applied.
///
/// A fault value remains typed and host-side; it is never copied into guest
/// registers implicitly.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionHandlingResult<F> {
    /// The continuation was installed and the process is runnable.
    Resumed,
    /// The continuation was installed but an explicit later wakeup is needed.
    Suspended,
    /// A stable result was returned through the guest ABI and execution may
    /// continue; the richer diagnostic remains available to the host.
    Rejected(F),
    /// Termination was applied with its original scope and exit code.
    Terminated {
        scope: ExceptionTerminationScope,
        exit_code: u64,
        reason: ExceptionTerminationReason,
    },
    /// Dispatch failed and the process was moved to the faulted state.
    Fault(F),
}

/// Current guest thread visible while runtime policy handles an exception.
pub struct ExceptionThreadContext<'a> {
    tls_base: GuestVirtualAddress,
    id: GuestThreadId,
    vcpu: VirtualCpuId,
    object: &'a Arc<ThreadObject>,
    state: &'a mut GuestCpuState,
}

impl<'a> ExceptionThreadContext<'a> {
    pub(crate) const fn new(
        id: GuestThreadId,
        vcpu: VirtualCpuId,
        object: &'a Arc<ThreadObject>,
        state: &'a mut GuestCpuState,
        tls_base: GuestVirtualAddress,
    ) -> Self {
        Self {
            tls_base,
            id,
            vcpu,
            object,
            state,
        }
    }

    #[must_use]
    pub const fn id(&self) -> GuestThreadId {
        self.id
    }

    #[must_use]
    pub const fn vcpu(&self) -> VirtualCpuId {
        self.vcpu
    }

    pub const fn tls_base(&self) -> GuestVirtualAddress {
        self.tls_base
    }

    /// Borrows the current thread already retained by this dispatch. Operations
    /// that outlive the call explicitly retain their own reference.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_synchronization.cpp#L66-L67
    #[must_use]
    pub const fn object(&self) -> &'a Arc<ThreadObject> {
        self.object
    }

    /// Reads current registers through the exclusive stopped-context owner.
    #[must_use]
    pub const fn state(&self) -> &GuestCpuState {
        self.state
    }

    /// Writes syscall results directly into the current register owner.
    pub const fn state_mut(&mut self) -> &mut GuestCpuState {
        self.state
    }
}

/// Complete current process/thread view supplied to runtime exception policy.
pub struct ExceptionDispatchContext<'a> {
    source: LocationDescriptor,
    process: &'a Arc<crate::RunnableProcess>,
    thread: ExceptionThreadContext<'a>,
}

impl<'a> ExceptionDispatchContext<'a> {
    pub(crate) const fn new(
        source: LocationDescriptor,
        process: &'a Arc<crate::RunnableProcess>,
        thread: ExceptionThreadContext<'a>,
    ) -> Self {
        Self {
            source,
            process,
            thread,
        }
    }

    /// Original instruction location, retained across owned kernel phases.
    #[must_use]
    pub const fn source(&self) -> LocationDescriptor {
        self.source
    }

    #[must_use]
    pub const fn process(&self) -> &'a Arc<crate::RunnableProcess> {
        self.process
    }

    #[must_use]
    pub const fn thread(&self) -> &ExceptionThreadContext<'a> {
        &self.thread
    }

    pub const fn thread_mut(&mut self) -> &mut ExceptionThreadContext<'a> {
        &mut self.thread
    }
}

/// Why a stopped process could not route a supervisor call.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionRouteError {
    NotSupervisorCall,
    UnknownThread(GuestThreadId),
    ProcessNotSuspended {
        lifecycle: nixe_scheduler::ThreadLifecycle,
    },
    SourceMismatch {
        requested: LocationDescriptor,
        current: LocationDescriptor,
    },
}

impl std::fmt::Display for ExceptionRouteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSupervisorCall => {
                formatter.write_str("execution stop is not a supervisor call")
            }
            Self::UnknownThread(id) => write!(formatter, "guest thread {id} does not exist"),
            Self::ProcessNotSuspended { lifecycle } => {
                write!(
                    formatter,
                    "cannot route exception while thread is {lifecycle:?}"
                )
            }
            Self::SourceMismatch { requested, current } => write!(
                formatter,
                "exception source does not identify the current instruction: requested=[{requested}] current=[{current}]"
            ),
        }
    }
}

impl std::error::Error for ExceptionRouteError {}

/// Runtime-owned architectural exception router.
///
/// The borrowed context and associated fault type keep process/thread services
/// and platform error policy explicit. A CPU backend only constructs the
/// request and applies the returned control decision.
pub trait ExceptionDispatcher {
    type Fault;

    fn dispatch(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        request: ExceptionDispatchRequest,
    ) -> ExceptionDispatchOutcome<Self::Fault>;
}

#[cfg(test)]
mod tests {
    use nixe_cpu::{exception::ExceptionKind, location::LocationDescriptor, profile::CpuProfileId};
    use nixe_memory::GuestVirtualAddress;

    use super::*;

    fn source() -> LocationDescriptor {
        LocationDescriptor::new(GuestVirtualAddress::new(0x7100_1000), CpuProfileId::new(7))
    }

    #[test]
    fn request_preserves_backend_independent_architectural_context() {
        let request =
            ExceptionDispatchRequest::new(source(), ExceptionKind::SupervisorCall, Some(0x2a));

        assert_eq!(request.source(), source());
        assert_eq!(request.kind(), ExceptionKind::SupervisorCall);
        assert_eq!(request.syndrome(), Some(0x2a));
    }

    #[test]
    fn rejection_keeps_the_host_diagnostic_typed() {
        let rejected = ExceptionDispatchOutcome::Reject {
            diagnostic: "runtime service unavailable",
        };

        assert!(matches!(
            rejected,
            ExceptionDispatchOutcome::Reject {
                diagnostic: "runtime service unavailable",
            }
        ));
    }

    #[test]
    fn next_is_distinct_from_an_explicit_retry() {
        assert_ne!(ExceptionResume::Next, ExceptionResume::Retry);
    }
}
