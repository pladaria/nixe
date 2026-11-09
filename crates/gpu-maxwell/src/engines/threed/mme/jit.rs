//! Cached native MME programs. The channel owner also owns their allocator.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/macro.cpp#L1340-L1435
use super::*;
use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
    rc::Rc,
};

type NativeProgram = unsafe extern "C" fn(*mut Invocation, *const u32, *const u32);
type Emit = unsafe extern "C" fn(*mut c_void, u32, u32) -> u32;

#[repr(C)]
struct Invocation {
    bank: *mut u32,
    host: *mut c_void,
    registers: [u32; 8],
    carry: u32,
    emit: Emit,
    next_parameter: *const u32,
    error: u32,
    error_pc: u32,
    error_value: u32,
}
#[repr(C)]
struct CompileError {
    pc: u32,
    message: [c_char; 256],
}
unsafe extern "C" {
    fn nixe_mme_runtime_new() -> *mut c_void;
    fn nixe_mme_runtime_delete(runtime: *mut c_void);
    fn nixe_mme_compile(
        runtime: *mut c_void,
        words: *const u32,
        count: usize,
        result: *mut Option<NativeProgram>,
        error: *mut CompileError,
    ) -> bool;
    fn nixe_mme_release(runtime: *mut c_void, program: NativeProgram);
}

#[derive(Debug)]
pub(super) struct Runtime(NonNull<c_void>);
impl Runtime {
    pub(super) fn new() -> Result<Rc<Self>, MaxwellThreeDMmeExecutionError> {
        // SAFETY: creates an allocator owned exclusively by this GPU channel.
        NonNull::new(unsafe { nixe_mme_runtime_new() })
            .map(|p| Rc::new(Self(p)))
            .ok_or_else(|| MaxwellThreeDMmeExecutionError::Compilation {
                address: MaxwellThreeDMmeRamAddress::new(0),
                diagnostic: "cannot allocate the native MME runtime".into(),
            })
    }
    pub(super) fn compile(
        self: &Rc<Self>,
        words: &[MaxwellThreeDMmeInstruction],
        start: u32,
    ) -> Result<Rc<Program>, MaxwellThreeDMmeExecutionError> {
        let mut function = None;
        let mut error = CompileError {
            pc: 0,
            message: [0; 256],
        };
        // SAFETY: instructions are repr(transparent) u32 words; compilation
        // borrows them synchronously and transfers executable ownership only.
        let ok = unsafe {
            nixe_mme_compile(
                self.0.as_ptr(),
                words.as_ptr().cast(),
                words.len(),
                &mut function,
                &mut error,
            )
        };
        if !ok {
            // SAFETY: native diagnostics always terminate the fixed buffer.
            let message = unsafe { CStr::from_ptr(error.message.as_ptr()) }.to_string_lossy();
            return Err(MaxwellThreeDMmeExecutionError::Compilation {
                address: MaxwellThreeDMmeRamAddress::new(start.wrapping_add(error.pc)),
                diagnostic: message.as_ref().into(),
            });
        }
        log::debug!(
            "MME JIT compiled macro at {start:#x}: {} instructions",
            words.len()
        );
        nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::MmeJitCompilations, 1);
        Ok(Rc::new(Program {
            function: function.expect("successful native MME compilation"),
            runtime: Rc::clone(self),
            start,
        }))
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: compiled functions retain this Rc until after their release.
        unsafe { nixe_mme_runtime_delete(self.0.as_ptr()) };
    }
}
#[derive(Debug)]
pub(in crate::engines::threed) struct Program {
    function: NativeProgram,
    runtime: Rc<Runtime>,
    start: u32,
}
impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: the channel allocator outlives this immutable function.
        unsafe { nixe_mme_release(self.runtime.0.as_ptr(), self.function) };
    }
}
struct Host<'a, H: MaxwellThreeDMmeHost> {
    engine: &'a mut H,
    error: Option<H::Error>,
}
unsafe extern "C" fn emit<H: MaxwellThreeDMmeHost>(
    host: *mut c_void,
    method: u32,
    value: u32,
) -> u32 {
    // SAFETY: execute installs this exact monomorphized callback and keeps the
    // exclusive host borrow live until the native program has returned.
    let host = unsafe { &mut *host.cast::<Host<'_, H>>() };
    match host
        .engine
        .emit_method((method & MME_METHOD_DWORD_MASK) as u16, value)
    {
        Ok(()) => 1,
        Err(error) => {
            host.error = Some(error);
            0
        }
    }
}
impl Program {
    pub(super) fn execute<H: MaxwellThreeDMmeHost>(
        &self,
        parameters: &[u32],
        host: &mut H,
    ) -> Result<(), MaxwellThreeDMmeRunError<H::Error>> {
        let bank = host.register_words_ptr();
        let mut host = Host {
            engine: host,
            error: None,
        };
        let mut invocation = Invocation {
            bank,
            host: (&mut host as *mut Host<'_, H>).cast(),
            registers: [0; 8],
            carry: 0,
            emit: emit::<H>,
            next_parameter: parameters.as_ptr(),
            error: 0,
            error_pc: 0,
            error_value: 0,
        };
        // SAFETY: the generated program uses this ABI, borrows the parameter
        // range, reads the canonical bank directly, and calls only emit::<H>.
        // The bank stays in place under the GPU owner's exclusive channel
        // borrow. Method callbacks complete before the next native instruction.
        unsafe {
            (self.function)(
                &mut invocation,
                parameters.as_ptr(),
                parameters.as_ptr().add(parameters.len()),
            )
        };
        if let Some(error) = host.error {
            return Err(MaxwellThreeDMmeRunError::Host(error));
        }
        // SAFETY: native parameter fetches advance only within this slice.
        let consumed =
            unsafe { invocation.next_parameter.offset_from(parameters.as_ptr()) } as usize;
        let error = match invocation.error {
            1 => Some(MaxwellThreeDMmeExecutionError::ParameterUnavailable { index: consumed }),
            2 => Some(MaxwellThreeDMmeExecutionError::RegisterReadUnavailable {
                method_dword: invocation.error_value as u16,
            }),
            3 => Some(MaxwellThreeDMmeExecutionError::MissingInstruction {
                address: MaxwellThreeDMmeRamAddress::new(
                    self.start.wrapping_add(invocation.error_pc),
                ),
            }),
            0 if consumed != parameters.len() => {
                Some(MaxwellThreeDMmeExecutionError::UnconsumedParameters {
                    consumed,
                    supplied: parameters.len(),
                })
            }
            0 => None,
            _ => unreachable!("native MME error encoding"),
        };
        error.map_or(Ok(()), |error| {
            Err(MaxwellThreeDMmeRunError::Execution(error))
        })
    }
}
