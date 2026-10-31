//! Runtime ownership of native CPU processes and per-core register contexts.
use crate::{ExceptionTerminationScope, GuestBreakPayload, GuestCpuState, VirtualClock};
use nixe_cpu::{
    execution::{ArchitecturalTimer, CpuFault, TimerSnapshot},
    location::LocationDescriptor,
    state::RegisterContext,
};
use nixe_cpu_jit::{JitRunRequest, JitThread};
use std::{
    error::Error,
    fmt::{Display, Formatter},
};

pub use nixe_cpu::execution::{CpuExit as ExecutionStop, ExecutionReport};
pub(crate) type CpuThread = Box<JitThread>;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProcessExitCause {
    HostRequested,
    ProcessRequested,
    LastThreadExited,
    LoaderReturned,
    GuestBreak {
        reason: u64,
        info: u64,
        size: u64,
        payload: Option<GuestBreakPayload>,
    },
}

/// One A64 frame-pointer record retained for post-teardown diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestStackFrame {
    /// Address of the frame record in the guest stack.
    pub frame_pointer: u64,
    /// Saved link register belonging to the caller.
    pub return_address: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProcessExit {
    pub cause: ProcessExitCause,
    pub exit_code: u64,
    pub source: Option<LocationDescriptor>,
    pub thread_id: u64,
    /// Architectural state captured at the terminating instruction, when the
    /// runtime owns a resident CPU state for the exiting thread.
    pub context: Option<Box<RegisterContext>>,
    /// Bounded A64 frame-pointer walk captured before guest memory is released.
    pub frames: Box<[GuestStackFrame]>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ThreadExit {
    pub requested_scope: ExceptionTerminationScope,
    pub exit_code: u64,
    pub source: Option<LocationDescriptor>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProcessTeardownReport {
    pub previous_lifecycle: nixe_scheduler::ProcessLifecycle,
    pub exit: Option<ProcessExit>,
    pub threads_released: usize,
    pub modules_released: usize,
    pub mappings_released: usize,
    pub physical_pages_released: usize,
    pub mounts_released: usize,
    pub handles_released: usize,
    pub address_waiters_released: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessTeardownFailure {
    pub report: Box<ProcessTeardownReport>,
    pub fault: Box<CpuFault>,
}

impl Display for ProcessTeardownFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "CPU backend failed to shut down during teardown: {}",
            self.fault
        )
    }
}

impl Error for ProcessTeardownFailure {}

struct RuntimeTimer<'a> {
    clock: &'a VirtualClock,
    frequency: u64,
}
pub(crate) struct CoreExecution {
    pub(crate) thread: GuestCpuState,
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) architectural_timer_frequency: u64,
}
impl CoreExecution {
    pub(crate) fn run(
        &mut self,
        worker: &mut nixe_cpu_direct_memory::NativeWorker,
        thread: &mut CpuThread,
    ) -> Result<ExecutionReport, CpuFault> {
        let timer = RuntimeTimer {
            clock: &self.virtual_clock,
            frequency: self.architectural_timer_frequency,
        };
        thread.run_slice(
            worker,
            JitRunRequest {
                state: &mut self.thread,
                timer: &timer,
            },
        )
    }
}
impl ArchitecturalTimer for RuntimeTimer<'_> {
    fn snapshot(&self) -> TimerSnapshot {
        let ticks = self
            .clock
            .elapsed()
            .as_nanos()
            .saturating_mul(u128::from(self.frequency))
            / 1_000_000_000;
        TimerSnapshot {
            counter: u64::try_from(ticks).unwrap_or(u64::MAX),
            frequency: self.frequency,
        }
    }
}
