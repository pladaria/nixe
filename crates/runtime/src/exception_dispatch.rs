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
pub type DispatchFault = Box<dyn std::error::Error + Send + Sync>;

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
    /// The guest-visible operation was rejected, but execution may continue.
    ///
    /// Platform policy must install its stable guest ABI result before
    /// returning this outcome. The typed diagnostic is retained for host-side
    /// reporting and is never copied into guest state by the generic runtime.
    /// The platform selects the host log level for the retained diagnostic.
    /// Rejection always continues after the exception-raising instruction;
    /// operations that retry retain their stack while the kernel waits.
    Reject { diagnostic: F, level: log::Level },
    /// The selected runtime object requested deterministic termination.
    Terminate {
        scope: ExceptionTerminationScope,
        exit_code: u64,
        reason: ExceptionTerminationReason,
    },
    /// Dispatch failed and guest execution must not continue.
    Fault(F),
}

impl<F> ExceptionDispatchOutcome<F> {
    pub fn map_fault<G>(self, map: impl FnOnce(F) -> G) -> ExceptionDispatchOutcome<G> {
        match self {
            Self::Resume(resume) => ExceptionDispatchOutcome::Resume(resume),
            Self::Reject { diagnostic, level } => ExceptionDispatchOutcome::Reject {
                diagnostic: map(diagnostic),
                level,
            },
            Self::Terminate {
                scope,
                exit_code,
                reason,
            } => ExceptionDispatchOutcome::Terminate {
                scope,
                exit_code,
                reason,
            },
            Self::Fault(fault) => ExceptionDispatchOutcome::Fault(map(fault)),
        }
    }
}

/// Observable lifecycle result after a dispatch outcome has been applied.
///
/// A fault value remains typed and host-side; it is never copied into guest
/// registers implicitly.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionHandlingResult<F> {
    /// The continuation was installed and the process is runnable.
    Resumed,
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
    pub const fn new(
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
    generation: u64,
}

impl<'a> ExceptionDispatchContext<'a> {
    pub const fn new(
        source: LocationDescriptor,
        process: &'a Arc<crate::RunnableProcess>,
        thread: ExceptionThreadContext<'a>,
        generation: u64,
    ) -> Self {
        Self {
            source,
            process,
            thread,
            generation,
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
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

/// A producer makes the actual kernel thread runnable. Its suspended stack
/// consumes the result on whichever physical core next selects it.
struct FiberWake {
    scheduler: std::sync::Weak<nixe_scheduler::SchedulerState>,
    token: nixe_scheduler::WakeToken,
    notified: std::sync::atomic::AtomicBool,
}
impl std::task::Wake for FiberWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.notified
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(scheduler) = self.scheduler.upgrade() {
            scheduler.wake_fiber(self.token);
        }
    }
}
impl ExceptionDispatchContext<'_> {
    pub fn retain_request(&self) -> crate::KernelRequest {
        crate::KernelRequest::new(self)
    }
    fn check_admission(&self) {
        if self.process.generation() != self.generation {
            crate::fiber::cancel_current();
        }
    }
    /// Wait on real asynchronous backend work from the current kernel stack.
    /// A ready operation returns inline. A pending operation parks the guest
    /// fiber directly; there is no future/continuation owned by the core loop.
    pub fn wait_future<F: std::future::Future + Send>(
        &mut self,
        future: F,
        wait: Option<Arc<nixe_scheduler::KernelWait>>,
    ) -> Result<F::Output, nixe_scheduler::SchedulerError> {
        use std::sync::atomic::Ordering;
        use std::task::{Context, Poll, Waker};
        self.check_admission();
        let mut future = std::pin::pin!(future);
        if let Poll::Ready(result) = future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            if wait.is_some() {
                self.process
                    .scheduler()
                    .resume_kernel(crate::machine::core::current_lease())?;
            }
            self.check_admission();
            return Ok(result);
        }
        let scheduler = self.process.scheduler().clone();
        let wait = match wait {
            Some(wait) => wait,
            None => scheduler.register_wait(self.thread.id())?,
        };
        wait.commit();
        let notified = Arc::new(FiberWake {
            scheduler: Arc::downgrade(&scheduler),
            token: wait.token(),
            notified: std::sync::atomic::AtomicBool::new(false),
        });
        let waker = Waker::from(notified.clone());
        loop {
            self.check_admission();
            notified.notified.store(false, Ordering::Release);
            match future.as_mut().poll(&mut Context::from_waker(&waker)) {
                Poll::Ready(result) => {
                    scheduler.resume_kernel(crate::machine::core::current_lease())?;
                    self.check_admission();
                    return Ok(result);
                }
                Poll::Pending => {
                    scheduler.park_fiber(wait.token())?;
                    if notified.notified.load(Ordering::Acquire) {
                        continue;
                    }
                    crate::fiber::suspend(
                        self.thread.object,
                        self.thread.state,
                        nixe_scheduler::Completion::Waiting,
                    );
                    self.thread.vcpu = crate::machine::core::current_lease().vcpu;
                }
            }
        }
    }
    pub fn wait_io<F: std::future::Future>(&mut self, future: F) -> F::Output {
        crate::machine::core::wait_io(future)
    }
    pub fn wait_access(
        &mut self,
        mut access: nixe_memory::PendingAccess,
    ) -> nixe_memory::AccessResultValue {
        let mut observed: Option<std::task::Waker> = None;
        self.wait_io(std::future::poll_fn(move |context| {
            if observed
                .as_ref()
                .is_none_or(|old| !old.will_wake(context.waker()))
            {
                let waker = context.waker().clone();
                observed = Some(waker.clone());
                access.set_waker(Arc::new(move || waker.wake_by_ref()));
            }
            access
                .poll()
                .map_or(std::task::Poll::Pending, std::task::Poll::Ready)
        }))
    }
}

/// Runtime-owned architectural exception router.
///
/// The borrowed context and associated fault type keep process/thread services
/// and platform error policy explicit. A CPU backend only constructs the
/// request and applies the returned control decision.
///
/// # Safety
/// Dispatch runs on a guest fiber which can resume on another host thread after
/// a kernel wait. All values retained on that stack across suspension must be
/// `Send`. In particular, no thread-local borrow, native-core reference, or
/// thread-affine lock guard may span a call that can suspend the guest fiber.
pub unsafe trait ExceptionDispatcher: Send + Sync + 'static {
    type Fault: std::error::Error + Send + Sync + 'static;

    fn dispatch(
        self: &Arc<Self>,
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
            level: log::Level::Warn,
        }
        .map_fault(str::to_owned);

        assert!(matches!(
            rejected,
            ExceptionDispatchOutcome::Reject {
                diagnostic,
                level: log::Level::Warn,
            } if diagnostic == "runtime service unavailable"
        ));
    }

    #[test]
    fn next_is_distinct_from_an_explicit_retry() {
        assert_ne!(ExceptionResume::Next, ExceptionResume::Retry);
    }
}
