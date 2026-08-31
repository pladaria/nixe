//! Small concrete execution contract shared by runtime and CPU backends.

use core::fmt::{self, Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use nixe_memory::{AddressSpaceId, CanonicalRangeTranslator, GuestVirtualAddress};

use crate::coverage::CoverageId;
use crate::error::{InstructionFetchFault, UnallocatedEncoding};
use crate::exception::ExceptionKind;
use crate::location::{InstructionEncoding, LocationDescriptor};
use crate::memory::{CpuMemory, DataAccessFault, ExecutionMemoryLease};
use crate::profile::ProcessCpuContext;
use crate::state::{RegisterContext, ThreadCpuState};

macro_rules! identity {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(u64);

        impl $name {
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

identity!(CpuProcessId);
identity!(CpuThreadId);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TimerSnapshot {
    pub counter: u64,
    pub frequency: u64,
}

pub trait ArchitecturalTimer {
    fn snapshot(&self) -> TimerSnapshot;
}

#[derive(Clone, Default)]
pub struct VcpuEventState {
    state: Arc<VcpuEventStateInner>,
}

#[derive(Default)]
struct VcpuEventStateInner {
    event_register: AtomicBool,
    pending_interrupts: AtomicU32,
}

impl VcpuEventState {
    pub fn signal_event(&self) {
        self.state.event_register.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn consume_event(&self) -> bool {
        self.state.event_register.swap(false, Ordering::AcqRel)
    }

    pub fn post_interrupts(&self, mask: u32) {
        if mask != 0 {
            self.state
                .pending_interrupts
                .fetch_or(mask, Ordering::Release);
            self.signal_event();
        }
    }

    #[must_use]
    pub fn interrupts_pending(&self) -> bool {
        self.state.pending_interrupts.load(Ordering::Acquire) != 0
    }

    #[must_use]
    pub fn take_pending_interrupts(&self) -> u32 {
        self.state.pending_interrupts.swap(0, Ordering::AcqRel)
    }

    #[must_use]
    pub fn pending_interrupts_address(&self) -> usize {
        std::ptr::from_ref(&self.state.pending_interrupts).addr()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SchedulerRequest {
    Yield,
    WaitForEvent,
    WaitForInterrupt,
    SendEvent,
}

pub struct RunRequest<'a> {
    pub cpu: ProcessCpuContext,
    pub memory: &'a dyn CpuMemory,
    /// Live mapping-stability proof required by a LinuxDirect backend.
    pub memory_lease: Option<ExecutionMemoryLease<'a>>,
    pub state: &'a mut ThreadCpuState,
    /// Exact instruction limit for the interpreter. The normal JIT uses
    /// control-driven entry and backedge synchronization instead.
    pub instruction_budget: u64,
    pub loader_return: Option<GuestVirtualAddress>,
    pub timer: &'a dyn ArchitecturalTimer,
    pub events: VcpuEventState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CpuExit {
    UnsupportedSemantics {
        source: LocationDescriptor,
        encoding: InstructionEncoding,
        disassembly: Box<str>,
        coverage_id: CoverageId,
    },
    UnallocatedEncoding {
        error: UnallocatedEncoding,
    },
    FetchFault {
        fault: InstructionFetchFault,
    },
    BudgetExhausted,
    Safepoint,
    PendingEvent {
        mask: u32,
    },
    Scheduled {
        source: LocationDescriptor,
        request: SchedulerRequest,
    },
    ArchitecturalException {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    SupervisorCall {
        source: LocationDescriptor,
        immediate: u32,
    },
    DataFault {
        source: LocationDescriptor,
        fault: DataAccessFault,
    },
    LoaderReturn {
        source: LocationDescriptor,
        result_code: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExceptionDispatchRequest {
    source: LocationDescriptor,
    kind: ExceptionKind,
    syndrome: Option<u64>,
}

impl ExceptionDispatchRequest {
    #[must_use]
    pub const fn new(
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    ) -> Self {
        Self {
            source,
            kind,
            syndrome,
        }
    }

    #[must_use]
    pub const fn source(self) -> LocationDescriptor {
        self.source
    }
    #[must_use]
    pub const fn kind(self) -> ExceptionKind {
        self.kind
    }
    #[must_use]
    pub const fn syndrome(self) -> Option<u64> {
        self.syndrome
    }
}

impl CpuExit {
    #[must_use]
    pub fn exception_dispatch_request(&self) -> Option<ExceptionDispatchRequest> {
        let (source, kind, syndrome) = match self {
            Self::ArchitecturalException {
                source,
                kind,
                syndrome,
            } => (*source, *kind, *syndrome),
            Self::SupervisorCall { source, immediate } => (
                *source,
                ExceptionKind::SupervisorCall,
                Some(u64::from(*immediate)),
            ),
            Self::DataFault { source, .. } => (*source, ExceptionKind::DataAbort, None),
            Self::UnallocatedEncoding { error } => (
                error.instruction.location,
                ExceptionKind::UndefinedInstruction,
                None,
            ),
            _ => return None,
        };
        Some(ExceptionDispatchRequest::new(source, kind, syndrome))
    }
}

impl Display for CpuExit {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSemantics {
                source,
                encoding,
                disassembly,
                coverage_id,
            } => write!(
                formatter,
                "unsupported-semantics source=[{source}] encoding={encoding} disassembly={disassembly} coverage={coverage_id}"
            ),
            Self::UnallocatedEncoding { error } => {
                write!(formatter, "unallocated-encoding {error}")
            }
            Self::FetchFault { fault } => write!(formatter, "fetch-fault {fault}"),
            Self::BudgetExhausted => formatter.write_str("budget-exhausted"),
            Self::Safepoint => formatter.write_str("safepoint"),
            Self::PendingEvent { mask } => write!(formatter, "pending-event mask=0x{mask:08x}"),
            Self::Scheduled { source, request } => {
                write!(formatter, "scheduled source=[{source}] request={request:?}")
            }
            Self::ArchitecturalException {
                source,
                kind,
                syndrome,
            } => write!(
                formatter,
                "architectural-exception source=[{source}] kind={kind:?} syndrome={syndrome:?}"
            ),
            Self::SupervisorCall { source, immediate } => write!(
                formatter,
                "supervisor-call source=[{source}] immediate={immediate:?}"
            ),
            Self::DataFault { source, fault } => {
                write!(formatter, "data-fault source=[{source}] fault={fault:?}")
            }
            Self::LoaderReturn {
                source,
                result_code,
            } => write!(
                formatter,
                "loader-return source=[{source}] result=0x{result_code:016x}"
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReport {
    /// Backend-defined coarse forward progress. The interpreter reports exact
    /// instructions; the normal JIT reports completed native boundaries.
    pub progress: u64,
    pub stop: CpuExit,
    pub context: RegisterContext,
}

impl Display for ExecutionReport {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "progress={} stop=[{}] registers=[{}]",
            self.progress, self.stop, self.context
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CpuFaultKind {
    InvalidRequest,
    Internal,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuFault {
    pub backend: &'static str,
    pub kind: CpuFaultKind,
    pub progress: u64,
    pub message: Box<str>,
    pub context: Box<RegisterContext>,
}

impl Display for CpuFault {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "backend={} kind={:?} progress={} message={} registers=[{}]",
            self.backend, self.kind, self.progress, self.message, self.context
        )
    }
}

impl std::error::Error for CpuFault {}

pub trait BoundMemory: CpuMemory + CanonicalRangeTranslator {}
impl<T> BoundMemory for T where T: CpuMemory + CanonicalRangeTranslator {}

#[derive(Clone, Copy)]
pub struct MemoryBinding<'a> {
    pub address_space: AddressSpaceId,
    pub end_exclusive: GuestVirtualAddress,
    pub memory: &'a dyn BoundMemory,
    pub mapping_epoch: u64,
    pub invalidation_cursor: nixe_memory::MemoryInvalidationCursor,
}

const CONTROL_PREEMPT: u32 = 1 << 0;
const CONTROL_CODE_INVALIDATION: u32 = 1 << 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ControlRequest {
    Preempt,
    CodeInvalidation,
}

impl ControlRequest {
    const fn bit(self) -> u32 {
        match self {
            Self::Preempt => CONTROL_PREEMPT,
            Self::CodeInvalidation => CONTROL_CODE_INVALIDATION,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ControlSnapshot {
    pub requests: u32,
    pub invalidation_epoch: u64,
}

impl ControlSnapshot {
    #[must_use]
    pub const fn contains(self, request: ControlRequest) -> bool {
        self.requests & request.bit() != 0
    }
}

struct CpuControlState {
    requests: AtomicU32,
    synchronization_counter: AtomicU32,
    invalidation_epoch: AtomicU64,
    acknowledged_invalidation_epoch: AtomicU64,
    active_executions: AtomicU32,
}

impl Default for CpuControlState {
    fn default() -> Self {
        Self {
            requests: AtomicU32::new(0),
            synchronization_counter: AtomicU32::new(CpuControl::SYNCHRONIZATION_INTERVAL),
            invalidation_epoch: AtomicU64::new(0),
            acknowledged_invalidation_epoch: AtomicU64::new(0),
            active_executions: AtomicU32::new(0),
        }
    }
}

#[derive(Clone, Default)]
pub struct CpuControl {
    state: Arc<CpuControlState>,
}

pub struct ExecutionGuard {
    state: Arc<CpuControlState>,
}

impl Drop for ExecutionGuard {
    fn drop(&mut self) {
        self.state.active_executions.fetch_sub(1, Ordering::AcqRel);
    }
}

impl CpuControl {
    pub const SYNCHRONIZATION_INTERVAL: u32 = 4000;

    #[must_use]
    pub fn enter_execution(&self) -> ExecutionGuard {
        self.state.active_executions.fetch_add(1, Ordering::AcqRel);
        ExecutionGuard {
            state: Arc::clone(&self.state),
        }
    }

    #[must_use]
    pub fn execution_active(&self) -> bool {
        self.state.active_executions.load(Ordering::Acquire) != 0
    }

    pub fn request(&self, request: ControlRequest) {
        self.state
            .requests
            .fetch_or(request.bit(), Ordering::Release);
        self.state
            .synchronization_counter
            .store(0, Ordering::Release);
    }

    pub fn request_invalidation(&self, epoch: u64) {
        self.state
            .invalidation_epoch
            .fetch_max(epoch, Ordering::AcqRel);
        self.request(ControlRequest::CodeInvalidation);
    }

    #[must_use]
    pub fn take_pending(&self) -> Option<ControlSnapshot> {
        let requests = self.state.requests.swap(0, Ordering::AcqRel);
        (requests != 0).then(|| ControlSnapshot {
            requests,
            invalidation_epoch: self.state.invalidation_epoch.load(Ordering::Acquire),
        })
    }

    #[must_use]
    pub fn pending_word_address(&self) -> usize {
        std::ptr::from_ref(&self.state.requests).addr()
    }

    #[must_use]
    pub fn synchronization_counter_address(&self) -> usize {
        std::ptr::from_ref(&self.state.synchronization_counter).addr()
    }

    pub fn acknowledge(&self, snapshot: ControlSnapshot) {
        if snapshot.requests & CONTROL_CODE_INVALIDATION != 0 {
            self.state
                .acknowledged_invalidation_epoch
                .fetch_max(snapshot.invalidation_epoch, Ordering::AcqRel);
        }
    }

    pub fn acknowledge_invalidation(&self, epoch: u64) {
        self.state
            .acknowledged_invalidation_epoch
            .fetch_max(epoch, Ordering::AcqRel);
    }

    #[must_use]
    pub fn acknowledged_invalidation(&self, epoch: u64) -> bool {
        self.state
            .acknowledged_invalidation_epoch
            .load(Ordering::Acquire)
            >= epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_pending_word_is_the_linearizable_request_set() {
        let control = CpuControl::default();
        let pending = unsafe { &*(control.pending_word_address() as *const AtomicU32) };
        let synchronization =
            unsafe { &*(control.synchronization_counter_address() as *const AtomicU32) };
        assert_eq!(
            synchronization.load(Ordering::Acquire),
            CpuControl::SYNCHRONIZATION_INTERVAL
        );

        control.request(ControlRequest::CodeInvalidation);
        assert_eq!(pending.load(Ordering::Acquire), CONTROL_CODE_INVALIDATION);
        assert_eq!(synchronization.load(Ordering::Acquire), 0);
        let snapshot = control.take_pending().unwrap();
        assert!(snapshot.contains(ControlRequest::CodeInvalidation));
        assert_eq!(pending.load(Ordering::Acquire), 0);

        control.request(ControlRequest::Preempt);
        assert_ne!(pending.load(Ordering::Acquire), 0);
        assert!(
            control
                .take_pending()
                .is_some_and(|snapshot| snapshot.contains(ControlRequest::Preempt))
        );
    }
}
