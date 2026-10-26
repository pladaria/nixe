//! A64 execution using the vendored Dynarmic recompiler.
mod code;
#[cfg(target_arch = "x86_64")]
mod execution;
mod ffi;
#[cfg(all(test, target_arch = "x86_64"))]
mod memory_tests;
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
    pub timer: &'a dyn ArchitecturalTimer,
    pub events: &'a VcpuEventState,
}

fn fault(message: impl Into<Box<str>>, state: &ThreadState) -> CpuFault {
    CpuFault {
        backend: "dynarmic",
        kind: CpuFaultKind::Internal,
        message: message.into(),
        context: Some(Box::new(state.register_context())),
    }
}
fn unavailable_state_fault(message: impl Into<Box<str>>) -> CpuFault {
    CpuFault {
        backend: "dynarmic",
        kind: CpuFaultKind::Internal,
        message: message.into(),
        context: None,
    }
}
fn report(stop: CpuExit, state: &ThreadState) -> ExecutionReport {
    let needs_context = !matches!(
        stop,
        CpuExit::SupervisorCall { .. } | CpuExit::Safepoint | CpuExit::PendingEvent { .. }
    );
    ExecutionReport {
        stop,
        context: needs_context.then(|| state.register_context()),
    }
}
