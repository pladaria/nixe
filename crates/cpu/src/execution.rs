//! Small concrete execution contract shared by runtime and CPU backends.

use core::fmt::{self, Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::coverage::CoverageId;
use crate::error::{InstructionFetchFault, UnallocatedEncoding};
use crate::exception::ExceptionKind;
use crate::location::{InstructionEncoding, LocationDescriptor};
use crate::state::RegisterContext;

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
    pending_interrupts: Arc<AtomicU32>,
}

impl VcpuEventState {
    pub fn post_interrupts(&self, mask: u32) {
        if mask != 0 {
            self.pending_interrupts.fetch_or(mask, Ordering::Release);
        }
    }

    #[must_use]
    pub fn take_pending_interrupts(&self) -> u32 {
        self.pending_interrupts.swap(0, Ordering::AcqRel)
    }
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
    Safepoint,
    PendingEvent {
        mask: u32,
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
    /// Runtime classification of a completed loader-stub SVC. CPU backends
    /// report the ordinary SupervisorCall and do not recognize loader addresses.
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
            Self::Safepoint => formatter.write_str("safepoint"),
            Self::PendingEvent { mask } => write!(formatter, "pending-event mask=0x{mask:08x}"),
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
    pub stop: CpuExit,
    /// Owned compact diagnostic snapshot when required by a fault. A normal native
    /// boundary may omit it while retaining complete state with its owner.
    pub context: Option<RegisterContext>,
}

impl Display for ExecutionReport {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "stop=[{}] registers=[", self.stop)?;
        match &self.context {
            Some(context) => Display::fmt(context, formatter)?,
            None => formatter.write_str("not captured")?,
        }
        formatter.write_str("]")
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
    pub message: Box<str>,
    /// Compact snapshot of exact synchronized state, if available. `Some`
    /// also guarantees that the caller's full architectural state is current.
    /// A terminal native failure may leave no coherent snapshot; the incoming
    /// state must not be substituted.
    pub context: Option<Box<RegisterContext>>,
}

impl Display for CpuFault {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "backend={} kind={:?} message={} registers=[",
            self.backend, self.kind, self.message
        )?;
        match &self.context {
            Some(context) => Display::fmt(context, formatter)?,
            None => formatter.write_str("unavailable")?,
        }
        formatter.write_str("]")
    }
}

impl std::error::Error for CpuFault {}

#[derive(Clone)]
pub struct CpuControl {
    state: Arc<CpuControlState>,
}

struct CpuControlState {
    requested: AtomicBool,
    interrupt: Arc<dyn Fn() + Send + Sync>,
}

impl CpuControl {
    pub fn with_interrupt(interrupt: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            state: Arc::new(CpuControlState {
                requested: AtomicBool::new(false),
                interrupt,
            }),
        }
    }

    pub fn request_preemption(&self) {
        self.state.requested.store(true, Ordering::Release);
        (self.state.interrupt)();
    }

    #[must_use]
    pub fn take_preemption(&self) -> bool {
        self.state.requested.swap(false, Ordering::AcqRel)
    }
}
