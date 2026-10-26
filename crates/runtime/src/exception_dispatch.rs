//! Backend-independent runtime contract for architectural exception dispatch.
//!
//! CPU backends report an [`ExceptionDispatchRequest`]. Runtime policy
//! handles it and returns an [`ExceptionDispatchOutcome`] which an outer
//! execution loop applies. Dispatch does not access native CPU internals.

use crate::GuestCpuState;
use crate::memory::ProcessMemory;
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_memory::GuestVirtualAddress;
use nixe_scheduler::{GuestThreadId, VirtualCpuId};

use crate::{
    AddressWaitRegistry, HandleTable, ProcessMemoryLayout, ProcessMountNamespace, ThreadObject,
};

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

pub(crate) struct ExceptionProcessResources<'a> {
    pub memory: &'a std::sync::Arc<ProcessMemory>,
    pub mounts: &'a std::sync::Arc<ProcessMountNamespace>,
    pub handles: &'a std::sync::Arc<HandleTable>,
    pub address_waits: &'a std::sync::Arc<AddressWaitRegistry>,
}

/// Guest continuation selected after handling an exception.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionResume {
    /// Continue after the exception-raising instruction.
    ///
    /// For a supervisor call this advances by the four-byte A64 encoding width.
    Next,
    /// Continue at an explicit guest location selected by runtime policy.
    At(LocationDescriptor),
    /// Execute the faulting instruction again after the blocking condition has
    /// been resolved.
    Retry,
}

/// Runtime object whose lifetime an exception handler requests to end.
///
/// Keeping this distinction in the backend-independent contract lets a future
/// multi-thread scheduler remove only the calling thread. The current
/// single-thread process model converts `CurrentThread` into process exit when
/// that thread is the last runnable thread.
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

/// Process resources visible while runtime policy handles an exception.
///
/// This is a borrowed runtime view rather than a backend-owned object. It
/// exposes guest-domain identities and service boundaries without leaking raw
/// host pointers or coupling dispatch to native CPU internals.
pub struct ExceptionProcessContext<'a> {
    process_id: u64,
    cpu: ProcessCpuContext,
    address_space_limit: u64,
    memory_layout: ProcessMemoryLayout,
    process_local_region: GuestVirtualAddress,
    random_entropy: [u64; 4],
    thread_policy: Option<crate::process::ThreadPolicy>,
    memory: &'a std::sync::Arc<ProcessMemory>,
    mounts: &'a std::sync::Arc<ProcessMountNamespace>,
    handles: &'a std::sync::Arc<HandleTable>,
    address_waits: &'a std::sync::Arc<AddressWaitRegistry>,
}

#[derive(Clone, Copy)]
pub(crate) struct ExceptionProcessMetadata {
    pub process_id: u64,
    pub cpu: ProcessCpuContext,
    pub address_space_limit: u64,
    pub memory_layout: ProcessMemoryLayout,
    pub process_local_region: GuestVirtualAddress,
    pub random_entropy: [u64; 4],
    pub thread_policy: Option<crate::process::ThreadPolicy>,
}

impl<'a> ExceptionProcessContext<'a> {
    #[must_use]
    pub fn allowed_core_mask(&self) -> Option<u64> {
        self.thread_policy?.allowed_core_mask()
    }

    #[must_use]
    pub fn allowed_priority_mask(&self) -> Option<u64> {
        self.thread_policy?.allowed_priority_mask()
    }

    pub(crate) const fn new(
        metadata: ExceptionProcessMetadata,
        resources: ExceptionProcessResources<'a>,
    ) -> Self {
        Self {
            process_id: metadata.process_id,
            cpu: metadata.cpu,
            address_space_limit: metadata.address_space_limit,
            memory_layout: metadata.memory_layout,
            process_local_region: metadata.process_local_region,
            random_entropy: metadata.random_entropy,
            thread_policy: metadata.thread_policy,
            memory: resources.memory,
            mounts: resources.mounts,
            handles: resources.handles,
            address_waits: resources.address_waits,
        }
    }

    /// Returns the runtime-owned guest process identity.
    #[must_use]
    pub const fn process_id(&self) -> u64 {
        self.process_id
    }

    /// Returns the first address outside the process user address space.
    #[must_use]
    pub const fn address_space_limit(&self) -> u64 {
        self.address_space_limit
    }

    /// Returns the runtime-owned virtual regions reported by process-query APIs.
    #[must_use]
    pub const fn memory_layout(&self) -> ProcessMemoryLayout {
        self.memory_layout
    }

    #[must_use]
    pub const fn user_exception_context_address(&self) -> GuestVirtualAddress {
        self.process_local_region
    }

    /// Returns one of the four immutable entropy words assigned at process creation.
    #[must_use]
    pub const fn random_entropy(&self, index: usize) -> Option<u64> {
        if index < self.random_entropy.len() {
            Some(self.random_entropy[index])
        } else {
            None
        }
    }

    pub fn heap_size(&self) -> u64 {
        self.memory.heap_size()
    }
    pub fn total_user_physical_memory_size(&self) -> u64 {
        self.memory.accounting().total_user_physical_memory_size()
    }
    pub fn used_user_physical_memory_size(&self) -> u64 {
        self.memory.accounting().used_user_physical_memory_size()
    }
    pub fn total_non_system_user_physical_memory_size(&self) -> u64 {
        self.memory
            .accounting()
            .total_non_system_user_physical_memory_size()
    }
    pub fn used_non_system_user_physical_memory_size(&self) -> u64 {
        self.memory
            .accounting()
            .used_non_system_user_physical_memory_size()
    }
    pub fn total_system_resource_size(&self) -> u64 {
        self.memory.accounting().total_system_resource_size()
    }
    pub fn used_system_resource_size(&self) -> u64 {
        self.memory.accounting().used_system_resource_size()
    }

    /// Returns the immutable CPU profile and address-space identity of the
    /// current process.
    #[must_use]
    pub const fn cpu(&self) -> ProcessCpuContext {
        self.cpu
    }

    /// Returns the current process's canonical execution memory.
    #[must_use]
    pub const fn memory(&self) -> &std::sync::Arc<ProcessMemory> {
        self.memory
    }

    /// Returns the current process filesystem namespace.
    #[must_use]
    pub const fn mounts(&self) -> &std::sync::Arc<ProcessMountNamespace> {
        self.mounts
    }

    /// Returns the current process handle table.
    #[must_use]
    pub const fn handles(&self) -> &std::sync::Arc<HandleTable> {
        self.handles
    }

    #[must_use]
    pub const fn address_waits(&self) -> &std::sync::Arc<AddressWaitRegistry> {
        self.address_waits
    }

    /// Borrows the immutable mounts and internally synchronized handle table.
    ///
    /// Platform IPC adapters need both resources for an atomic service lookup
    /// without making the generic runtime depend on a console protocol.
    pub fn mounts_and_handles(&self) -> (&ProcessMountNamespace, &HandleTable) {
        (self.mounts, self.handles)
    }
}

/// Current guest thread visible while runtime policy handles an exception.
pub struct ExceptionThreadContext<'a> {
    id: GuestThreadId,
    vcpu: VirtualCpuId,
    object: ThreadObject,
    handle: u32,
    state: &'a mut GuestCpuState,
}

impl<'a> ExceptionThreadContext<'a> {
    pub(crate) const fn new(
        id: GuestThreadId,
        vcpu: VirtualCpuId,
        object: ThreadObject,
        handle: u32,
        state: &'a mut GuestCpuState,
    ) -> Self {
        Self {
            id,
            vcpu,
            object,
            handle,
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

    /// Returns the runtime-owned identity of the current thread.
    #[must_use]
    pub fn object(&self) -> ThreadObject {
        self.object.clone()
    }

    /// Returns the process-local handle installed for the current thread.
    #[must_use]
    pub const fn handle(&self) -> u32 {
        self.handle
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
    process: ExceptionProcessContext<'a>,
    thread: ExceptionThreadContext<'a>,
}

impl<'a> ExceptionDispatchContext<'a> {
    pub(crate) const fn new(
        source: LocationDescriptor,
        process: ExceptionProcessContext<'a>,
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
    pub const fn process(&self) -> &ExceptionProcessContext<'a> {
        &self.process
    }

    pub const fn process_mut(&mut self) -> &mut ExceptionProcessContext<'a> {
        &mut self.process
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
    CurrentThreadUnavailable {
        handle: u32,
    },
    ContinuationProfileMismatch {
        source: LocationDescriptor,
        target: LocationDescriptor,
    },
    InvalidContinuationTarget {
        target: LocationDescriptor,
    },
    ContinuationAddressOverflow {
        source: LocationDescriptor,
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
            Self::CurrentThreadUnavailable { handle } => write!(
                formatter,
                "current thread handle {handle:#x} is absent or has the wrong object type"
            ),
            Self::ContinuationProfileMismatch { source, target } => write!(
                formatter,
                "exception continuation changes the immutable CPU profile: source=[{source}] target=[{target}]"
            ),
            Self::InvalidContinuationTarget { target } => {
                write!(
                    formatter,
                    "invalid exception continuation target [{target}]"
                )
            }
            Self::ContinuationAddressOverflow { source } => write!(
                formatter,
                "exception continuation overflows the guest address after source [{source}]"
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
    fn suspended_dispatch_retains_retry_or_explicit_resume_intent() {
        let retry: ExceptionDispatchOutcome<()> =
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        let resume: ExceptionDispatchOutcome<()> =
            ExceptionDispatchOutcome::Suspend(ExceptionResume::At(source()));

        assert_ne!(retry, resume);
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
