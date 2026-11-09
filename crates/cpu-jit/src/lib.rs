//! A64 execution using the vendored Dynarmic recompiler.
mod code;
#[cfg(target_arch = "x86_64")]
mod execution;
mod ffi;
pub mod metrics;
#[cfg(target_arch = "x86_64")]
pub mod process;
mod state;
#[cfg(target_arch = "x86_64")]
pub use process::{Process as JitProcess, Thread as JitThread};
pub use state::ThreadState;

use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, CpuFault, CpuFaultKind, ExecutionReport};
use std::{
    ffi::c_void,
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Debug)]
pub enum JitError {
    Stopped,
    Failure(Box<str>),
}
impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => f.write_str("JIT process is stopping"),
            Self::Failure(message) => message.fmt(f),
        }
    }
}
impl std::error::Error for JitError {}

struct Native {
    pointer: *mut c_void,
    space: Arc<nixe_memory::AddressSpace>,
    context_owned: AtomicBool,
    interrupt: Arc<NativeInterrupt>,
    #[cfg(debug_assertions)]
    owner_thread: std::thread::ThreadId,
}
struct NativeInterrupt(std::sync::Mutex<Option<usize>>);
impl NativeInterrupt {
    fn halt(&self) {
        let pointer = self.0.lock().unwrap();
        if let Some(pointer) = *pointer {
            unsafe {
                ffi::nixe_dynarmic_halt(pointer as *mut c_void);
            }
        }
    }
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
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.owner_thread,
            std::thread::current().id(),
            "native context must retire on its creating worker"
        );
        // Exclude an interrupt before invalidating its target, as PhysicalCore's
        // context guard does. Interrupt handles never retain the native owner.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/physical_core.cpp#L233
        let mut interrupt = self.interrupt.0.lock().unwrap();
        *interrupt = None;
        unsafe { ffi::nixe_dynarmic_destroy(self.pointer) }
    }
}

/// Native slices retain register ownership in ThreadState. Fault reports
/// capture a diagnostic context; normal exits use the resident state directly.
pub struct JitRunRequest<'a> {
    pub state: &'a mut ThreadState,
    pub timer: &'a dyn ArchitecturalTimer,
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
    let needs_context = !matches!(stop, CpuExit::SupervisorCall { .. } | CpuExit::Safepoint);
    ExecutionReport {
        stop,
        context: needs_context.then(|| state.register_context()),
    }
}
