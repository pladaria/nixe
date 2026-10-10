//! Share nixe's signal capture with the interpreter. Fault attribution and all
//! Dynarmic/Rust callbacks run after sigreturn, with the caller's host FP state.
use crate::{callbacks, ffi};
use nixe_cpu_direct_memory::{CapturedFault, FaultDisposition, NativeInvocation, NativeWorker};
use std::ffi::c_void;

struct Invocation<'a> {
    core: *mut c_void,
    context: *mut c_void,
    budget: u64,
    exit: &'a mut ffi::Exit,
    ok: bool,
}
unsafe extern "C" fn gateway(context: *mut c_void, _entry: usize) {
    let call = unsafe { &mut *context.cast::<Invocation<'_>>() };
    call.ok = unsafe { ffi::nixe_dynarmic_run(call.core, call.context, call.budget, call.exit) };
}
unsafe extern "C" fn dispatch(
    _context: *mut c_void,
    fault: *mut CapturedFault<'_>,
) -> FaultDisposition {
    crate::metrics::record(crate::metrics::Counter::FaultDispatches, 1);
    let fault = unsafe { &mut *fault };
    let (mut entry, mut ret) = (0, 0);
    if !unsafe { ffi::nixe_dynarmic_resolve_fault(fault.native_pc() as u64, &mut entry, &mut ret) }
    {
        return FaultDisposition::FatalUnattributed;
    }
    unsafe { fault.call_on_resume(entry as usize, ret as usize) }
}

pub fn run(
    worker: &mut NativeWorker,
    arena: nixe_memory::DirectAddressSpaceView,
    core: *mut c_void,
    context: &mut callbacks::Context<'_>,
    budget: u64,
    exit: &mut ffi::Exit,
) -> Result<bool, Box<str>> {
    let mut call = Invocation {
        core,
        context: (context as *mut callbacks::Context<'_>).cast(),
        budget,
        exit,
        ok: false,
    };
    #[cfg(target_arch = "x86_64")]
    let fp = {
        let mut mxcsr = 0_u32;
        unsafe {
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
        }
        [u64::from(mxcsr), 0]
    };
    #[cfg(target_arch = "aarch64")]
    let fp = {
        let control: u64;
        let status: u64;
        unsafe {
            core::arch::asm!("mrs {},fpcr", "mrs {},fpsr", out(reg) control, out(reg) status, options(nostack, preserves_flags));
        }
        [control, status]
    };
    unsafe {
        worker
            .faults()
            .map_err(|e| e.to_string().into_boxed_str())?
            .invoke_captured(
                arena,
                fp,
                dispatch,
                std::ptr::null_mut(),
                NativeInvocation {
                    gateway,
                    context: (&mut call as *mut Invocation<'_>).cast(),
                    entry: 0,
                },
            )
    }
    .map_err(|e| e.to_string().into_boxed_str())?;
    Ok(call.ok)
}
