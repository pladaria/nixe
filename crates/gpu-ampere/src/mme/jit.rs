//! TU104 native programs share the channel's uploaded instruction RAM and allocator.
use super::*;
use std::{
    any::Any,
    ffi::{CStr, c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    rc::Rc,
};

type NativeProgram = unsafe extern "C" fn(*mut Invocation) -> u32;
#[repr(C)]
struct Invocation {
    regs: [u32; 24],
    carry: u32,
    method: u32,
    increment: u32,
    loops: u32,
    loop_start: u32,
    loop_end: u32,
    next_pc: u32,
    budget: u32,
    scratch_x: u32,
    scratch_y: u32,
    params: *const u32,
    params_size: usize,
    consumed: usize,
    dram: *mut u32,
    dram_size: usize,
    bank: *const u32,
    valid_bank: *const u32,
    host: *mut c_void,
    emit: unsafe extern "C" fn(*mut Invocation, u32, u32) -> u32,
    poll: unsafe extern "C" fn(*mut Invocation) -> u32,
    start: u32,
    error_pc: u32,
    error_value: u32,
    end_next: u8,
    stopped: u8,
    cancel_end: u8,
    padding: u8,
    owned: *const u64,
    resolve: unsafe extern "C" fn(*mut Invocation, u32) -> u32,
    state_value: u32,
}
const _: () = {
    assert!(size_of::<Instruction>() == 22);
    assert!(std::mem::offset_of!(Invocation, params) == 136);
    assert!(std::mem::offset_of!(Invocation, start) == 216);
    assert!(std::mem::offset_of!(Invocation, owned) == 232);
    assert!(size_of::<Invocation>() == 256);
};
#[repr(C)]
struct CompileError {
    pc: u32,
    message: [c_char; 256],
}
unsafe extern "C" {
    fn nixe_ampere_mme_compile(
        runtime: *mut c_void,
        instructions: *const Instruction,
        uploaded: *const u8,
        count: usize,
        track_ownership: bool,
        function: *mut Option<NativeProgram>,
        code_bytes: *mut usize,
        error: *mut CompileError,
    ) -> bool;
}

#[derive(Default)]
pub(super) struct Cache {
    runtime: Option<Rc<nixe_mme_jit::Runtime>>,
    program: Option<Program>,
}
impl Cache {
    pub fn invalidate(&mut self) {
        self.program = None;
    }
    pub fn execute<H: super::Host>(
        &mut self,
        instructions: &[Instruction],
        uploaded: &[u8],
        start: u32,
        fifo: &mut Fifo<'_>,
        dram: &mut [u32],
        host: &mut H,
    ) -> Result<()> {
        if host.stopped() {
            return Err("MME execution cancelled".into());
        }
        if self.program.is_none() {
            if self.runtime.is_none() {
                self.runtime = Some(Rc::new(
                    nixe_mme_jit::Runtime::new().ok_or("cannot allocate Ampere MME JIT runtime")?,
                ));
            }
            let runtime = self.runtime.as_ref().unwrap();
            let count = uploaded
                .iter()
                .rposition(|&mask| mask != 0)
                .map_or(0, |i| i + 1);
            let mut function = None;
            let mut code_bytes = 0;
            let mut error = CompileError {
                pc: 0,
                message: [0; 256],
            };
            // SAFETY: the C ABI consumes these immutable decoded instructions
            // synchronously. Code owns no pointers into either input slice.
            let ok = unsafe {
                nixe_ampere_mme_compile(
                    runtime.as_ptr(),
                    instructions.as_ptr(),
                    uploaded.as_ptr(),
                    count,
                    H::TRACK_OWNERSHIP,
                    &mut function,
                    &mut code_bytes,
                    &mut error,
                )
            };
            if !ok {
                // SAFETY: the compiler always terminates its diagnostic buffer.
                let message = unsafe { CStr::from_ptr(error.message.as_ptr()) }.to_string_lossy();
                return Err(
                    format!("Ampere MME JIT compilation pc={:#x}: {message}", error.pc).into(),
                );
            }
            self.program = Some(Program {
                function: function.expect("successful native MME compilation"),
                runtime: runtime.clone(),
            });
            log::debug!(
                "Ampere MME JIT compiled instruction RAM: {} instructions; {code_bytes} native bytes",
                uploaded[..count].iter().filter(|&&mask| mask == 7).count()
            );
            nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::MmeJitCompilations, 1);
        }
        self.program
            .as_ref()
            .unwrap()
            .execute(start, fifo, dram, host)
    }
}
struct Program {
    function: NativeProgram,
    runtime: Rc<nixe_mme_jit::Runtime>,
}
impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: immutable code is no longer executing; its allocator is retained.
        unsafe { self.runtime.release(self.function as *const c_void) };
    }
}
struct Host<'a, 'input, H> {
    engine: &'a mut H,
    fifo: &'a mut Fifo<'input>,
    error: Option<Box<str>>,
    panic: Option<Box<dyn Any + Send>>,
}
impl<H: super::Host> Host<'_, '_, H> {
    fn callback(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) -> u32 {
        // Unwinding cannot cross generated frames, which have no unwind metadata.
        // Resume internal panics only after returning to the Rust caller.
        match catch_unwind(AssertUnwindSafe(|| f(self))) {
            Ok(Ok(())) => 1,
            Ok(Err(error)) => {
                self.error = Some(error);
                0
            }
            Err(panic) => {
                self.panic = Some(panic);
                0
            }
        }
    }
}
unsafe extern "C" fn emit<H: super::Host>(state: *mut Invocation, method: u32, value: u32) -> u32 {
    // SAFETY: installed only for this invocation and this monomorphized host.
    let state = unsafe { &mut *state };
    let host = unsafe { &mut *state.host.cast::<Host<'_, '_, H>>() };
    host.callback(|host| {
        if host.engine.stopped() {
            return Err("MME execution cancelled".into());
        }
        if host.engine.emit(method, value)? == Emission::Dma {
            host.fifo.set_consumed(state.consumed);
            // SAFETY: only DMA borrows these mutable buffers while generated
            // code is suspended. Refresh pointers after reborrowing/reallocation.
            let dram = unsafe { std::slice::from_raw_parts_mut(state.dram, state.dram_size) };
            let result = host.engine.dma(method, dram, host.fifo);
            state.dram = dram.as_mut_ptr();
            state.params = host.fifo.data().as_ptr().cast();
            state.params_size = host.fifo.data().len() / 4;
            state.consumed = host.fifo.consumed();
            result
        } else {
            Ok(())
        }
    })
}

unsafe extern "C" fn resolve<H: super::Host>(state: *mut Invocation, index: u32) -> u32 {
    // SAFETY: native execution is suspended and this callback is paired with H.
    let state = unsafe { &mut *state };
    let host = unsafe { &mut *state.host.cast::<Host<'_, '_, H>>() };
    host.callback(|host| {
        state.state_value = host.engine.state(index * 4)?;
        Ok(())
    })
}
unsafe extern "C" fn poll<H: super::Host>(state: *mut Invocation) -> u32 {
    // SAFETY: as for emit; no reference is retained across the callback.
    let state = unsafe { &mut *state };
    let host = unsafe { &mut *state.host.cast::<Host<'_, '_, H>>() };
    host.callback(|host| {
        if host.engine.stopped() {
            Err("MME execution cancelled".into())
        } else {
            Ok(())
        }
    })
}
impl Program {
    fn execute<H: super::Host>(
        &self,
        start: u32,
        fifo: &mut Fifo<'_>,
        dram: &mut [u32],
        engine: &mut H,
    ) -> Result<()> {
        let (bank, valid_bank, owned) = engine.bank();
        let mut host = Host {
            engine,
            fifo,
            error: None,
            panic: None,
        };
        let mut state = Invocation {
            // @SIMPLIFICATION Initial reads return zero instead of faulting;
            // the compatibility assumption is documented on the interpreter's Run.
            regs: [0; 24],
            carry: 0,
            method: 0,
            increment: 0,
            loops: 0,
            loop_start: 0,
            loop_end: 0,
            next_pc: 0,
            budget: 1024,
            scratch_x: 0,
            scratch_y: 0,
            // AMD64 loads accept unaligned little-endian command bytes. Rust
            // never dereferences this pointer as u32 or creates a u32 slice.
            params: host.fifo.data().as_ptr().cast(),
            params_size: host.fifo.data().len() / 4,
            consumed: host.fifo.consumed(),
            dram: dram.as_mut_ptr(),
            dram_size: dram.len(),
            bank,
            valid_bank,
            host: (&mut host as *mut Host<'_, '_, H>).cast(),
            emit: emit::<H>,
            poll: poll::<H>,
            start,
            error_pc: start,
            error_value: 0,
            end_next: 0,
            stopped: 0,
            cancel_end: 0,
            padding: 0,
            owned,
            resolve: resolve::<H>,
            state_value: 0,
        };
        // SAFETY: the generated ABI is checked above. Slices and bank remain
        // alive and exclusively owned throughout execution and callbacks.
        let error = unsafe { (self.function)(&mut state) };
        host.fifo.set_consumed(state.consumed);
        if let Some(panic) = host.panic {
            resume_unwind(panic);
        }
        if error == 0 {
            return Ok(());
        }
        let detail = host.error.unwrap_or_else(|| match error {
            1 => "MME parameter FIFO underflow".into(),
            2 => "invalid MME source register".into(),
            3 => "invalid MME destination register".into(),
            4 => "unsupported MME extended operation".into(),
            5 => "MME MERGE destination outside word".into(),
            6 => "invalid MME LOOP".into(),
            7 => "MME DREAD outside data RAM".into(),
            8 => "MME DWRITE outside data RAM".into(),
            9 => "invalid MME output selector".into(),
            10 => "MME host callback failed".into(),
            11 => "MME fetch outside uploaded instruction RAM".into(),
            13 => format!(
                "MME STATE reads unknown method {:#x}",
                state.error_value * 4
            )
            .into(),
            14 => "MME ADDC/SUBB without matching ALU pair".into(),
            15 => "invalid MME control/data instruction encoding".into(),
            16 => "MME predicate uses lane-dependent immediate".into(),
            17 => "MME branch before code RAM".into(),
            _ => format!("unknown native MME error {error}").into(),
        });
        Err(format!(
            "MME start={start:#x} pc={:#x} remaining_parameters={}: {detail}",
            state.error_pc,
            host.fifo.len()
        )
        .into())
    }
}
