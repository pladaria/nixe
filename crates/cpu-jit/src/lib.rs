//! A64 execution using the vendored Dynarmic recompiler.
#[cfg(all(test, target_arch = "x86_64"))]
mod canonical_memory_tests;
pub mod code;
#[cfg(target_arch = "x86_64")]
pub mod execution;
mod ffi;
#[cfg(target_arch = "x86_64")]
pub mod instruction;
pub mod metrics;
#[cfg(target_arch = "x86_64")]
pub mod process;
mod state;
#[cfg(target_arch = "x86_64")]
pub use process::{Process as JitProcess, Thread as JitThread};
pub use state::ThreadState;

use nixe_cpu::execution::{
    ArchitecturalTimer, CpuExit, CpuFault, CpuFaultKind, ExecutionReport, VcpuEventState,
};
use std::{
    ffi::c_void,
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Debug)]
pub struct JitError(Box<str>);
impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for JitError {}

struct Native {
    pointer: *mut c_void,
    space: Arc<nixe_memory::AddressSpace>,
    context_owned: AtomicBool,
}
impl Native {
    fn address_space(&self) -> &Arc<nixe_memory::AddressSpace> {
        &self.space
    }
}
// Only HaltExecution may touch live execution concurrently. ThreadState owns
// stopped registers exclusively; moving it into synchronous native execution
// transfers that authority to the worker until return. A different state cannot
// enter while context_owned is set. Callback borrows last only for that run.
unsafe impl Send for Native {}
unsafe impl Sync for Native {}
impl Drop for Native {
    fn drop(&mut self) {
        unsafe { ffi::nixe_dynarmic_destroy(self.pointer) }
    }
}

/// Native slices retain register ownership in ThreadState. Fault reports
/// capture a diagnostic context; normal exits use the resident state directly.
pub struct JitRunRequest<'a> {
    pub state: &'a mut ThreadState,
    pub instruction_budget: u64,
    pub timer: &'a dyn ArchitecturalTimer,
    pub events: &'a VcpuEventState,
}

fn fault(message: impl Into<Box<str>>, state: &ThreadState, progress: u64) -> CpuFault {
    CpuFault {
        backend: "dynarmic",
        kind: CpuFaultKind::Internal,
        progress,
        message: message.into(),
        context: Some(Box::new(state.register_context())),
    }
}
fn unavailable_state_fault(message: impl Into<Box<str>>, progress: u64) -> CpuFault {
    CpuFault {
        backend: "dynarmic",
        kind: CpuFaultKind::Internal,
        progress,
        message: message.into(),
        context: None,
    }
}
fn report(stop: CpuExit, progress: u64, state: &ThreadState) -> ExecutionReport {
    let needs_context = !matches!(
        stop,
        CpuExit::SupervisorCall { .. }
            | CpuExit::BudgetExhausted
            | CpuExit::Safepoint
            | CpuExit::PendingEvent { .. }
            | CpuExit::Scheduled { .. }
    );
    ExecutionReport {
        stop,
        progress,
        context: needs_context.then(|| state.register_context()),
    }
}
