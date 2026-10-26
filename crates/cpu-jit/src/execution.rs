//! Native execution with synchronous memory callbacks on the process arena.
use crate::{
    JitError, Native, ThreadState,
    code::{CodeCache, CodeCapture, Lookup, ResolvedCode},
    ffi,
    metrics::{Counter, Timer, record},
};
use nixe_cpu::execution::{ArchitecturalTimer, CpuControl};
use nixe_cpu_direct_memory::{CapturedFault, FaultDisposition, NativeInvocation, NativeWorker};
use nixe_memory::{
    AccessOperation, AccessOrdering, AccessResult, AddressSpace, AtomicAccess, AtomicOperation,
    CoreContext, GuestVirtualAddress, MemoryError, MemoryPermissions, NativeAdmission,
    NativeReadArea,
};
use std::{
    ffi::c_void,
    sync::{Arc, atomic::AtomicBool},
};
pub struct Core {
    handle: Arc<Native>,
    context: CoreContext,
    code: CodeCache,
    read_area: Option<NativeReadArea>,
    control: CpuControl,
}
pub struct Configuration {
    pub cpu: nixe_cpu::profile::ProcessCpuContext,
    pub timer_frequency: u32,
}
/// Bridge event, decoded by the CPU execution loop after Run has returned.
pub struct Event {
    pub kind: u32,
    pub detail: u32,
    pub pc: u64,
    pub word: Option<u32>,
}
pub struct Boundary {
    pub event: Event,
    pub code: Vec<CodeCapture>,
    pub preempted: bool,
    pub fetch_fault: Option<nixe_cpu::error::InstructionFetchFault>,
}
struct Callbacks<'a> {
    space: &'a AddressSpace,
    admission: &'a NativeAdmission<'a>,
    timer: &'a dyn ArchitecturalTimer,
    cache: &'a mut CodeCache,
    read_area: &'a mut Option<NativeReadArea>,
    process: &'a crate::process::Process,
    core: *mut c_void,
    code: Vec<CodeCapture>,
    error: Option<MemoryError>,
}
struct Invocation {
    core: *mut c_void,
    context: *mut c_void,
    exit: ffi::Exit,
    ok: bool,
}
impl Core {
    pub fn new(space: Arc<AddressSpace>, configuration: Configuration) -> Result<Self, JitError> {
        let register = |key| match nixe_cpu::semantics::a64::runtime_register_read(
            configuration.cpu.platform(),
            key,
        ) {
            Some(nixe_cpu::semantics::a64::RuntimeRegisterRead::Constant(value)) => value as u32,
            _ => unreachable!("CPU profile defines CTR_EL0 and DCZID_EL0"),
        };
        let pointer = unsafe {
            ffi::nixe_dynarmic_create(
                ffi::Callbacks {
                    code,
                    memory,
                    counter,
                },
                space.native_data_base().as_ptr() as usize,
                space.native_page_table_base() as usize,
                space.width().bits() as usize,
                configuration.timer_frequency,
                register(0xd53b0020),
                register(0xd53b00e0),
            )
        };
        if pointer.is_null() {
            return Err(JitError(ffi::error()));
        }
        let handle = Arc::new(Native {
            pointer,
            space: Arc::clone(&space),
            context_owned: AtomicBool::new(false),
        });
        let weak = Arc::downgrade(&handle);
        let context = space.register_core(Arc::new(move || {
            if let Some(handle) = weak.upgrade() {
                unsafe {
                    ffi::nixe_dynarmic_halt(handle.pointer);
                }
            }
        }));
        let interrupt = Arc::downgrade(&handle);
        let control = CpuControl::with_interrupt(Arc::new(move || {
            if let Some(handle) = interrupt.upgrade() {
                unsafe {
                    ffi::nixe_dynarmic_halt(handle.pointer);
                }
            }
        }));
        Ok(Self {
            handle,
            context,
            code: CodeCache::default(),
            read_area: None,
            control,
        })
    }
    pub fn space(&self) -> &Arc<AddressSpace> {
        self.handle.address_space()
    }
    pub fn select_state(&mut self, state: &mut ThreadState) -> Result<(), JitError> {
        state.select(&self.handle)
    }
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }
    pub fn clear_exclusive(&mut self) {
        unsafe {
            ffi::nixe_dynarmic_clear_exclusive(self.handle.pointer);
        }
    }
    pub fn install_code(&mut self, code: ResolvedCode) -> Result<(), MemoryError> {
        if code.space() != self.space().identity() {
            return Err(MemoryError::Invalid(
                "resolved code belongs to another address space",
            ));
        }
        // A failed speculative fetch may already have generated a NoExecute
        // terminal. Retire that terminal before retrying the captured word.
        record(Counter::InvalidationCalls, 1);
        record(Counter::InvalidationBytes, 4);
        let _measurement = Timer::new(Counter::InvalidationNanoseconds);
        if !unsafe { ffi::nixe_dynarmic_invalidate(self.handle.pointer, code.address().get(), 4) } {
            return Err(self.context.fail(MemoryError::Execution(ffi::error())));
        }
        self.code.install(self.space().identity(), code)
    }
    pub fn run(
        &mut self,
        worker: &mut NativeWorker,
        timer: &dyn ArchitecturalTimer,
        process: &crate::process::Process,
    ) -> Result<Boundary, JitError> {
        let handle = &self.handle;
        unsafe {
            ffi::nixe_dynarmic_clear_halt(handle.pointer);
        }
        let admission_timer = Timer::new(Counter::AdmissionNanoseconds);
        let admission = self.context.enter_native().map_err(memory_error)?;
        drop(admission_timer);
        // A racing preemption is either consumed here or remains in the
        // native halt word. The caller must not clear Halt after this check.
        if self.control.take_preemption() {
            return Ok(Boundary {
                event: Event {
                    kind: 0,
                    detail: 0,
                    pc: 0,
                    word: None,
                },
                code: Vec::new(),
                preempted: true,
                fetch_fault: None,
            });
        }
        admission
            .synchronize_code(|range| {
                record(Counter::InvalidationCalls, 1);
                record(Counter::InvalidationBytes, range.bytes);
                let _measurement = Timer::new(Counter::InvalidationNanoseconds);
                if unsafe {
                    ffi::nixe_dynarmic_invalidate(
                        handle.pointer,
                        range.start.get(),
                        range.bytes as usize,
                    )
                } {
                    Ok(())
                } else {
                    Err(MemoryError::Execution(ffi::error()))
                }
            })
            .map_err(memory_error)?;
        let space = handle.address_space();
        let mut callbacks = Callbacks {
            space,
            admission: &admission,
            timer,
            cache: &mut self.code,
            read_area: &mut self.read_area,
            process,
            core: handle.pointer,
            code: Vec::new(),
            error: None,
        };
        let mut call = Invocation {
            core: handle.pointer,
            context: std::ptr::from_mut(&mut callbacks).cast(),
            exit: ffi::Exit::default(),
            ok: false,
        };
        let base = space.native_data_base().as_ptr() as usize;
        let mut mxcsr = 0u32;
        unsafe {
            core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
        }
        let opaque = std::ptr::from_mut(&mut call).cast();
        record(Counter::NativeEntries, 1);
        let measurement = Timer::new(Counter::NativeRunNanoseconds);
        unsafe {
            worker
                .faults()
                .map_err(|e| JitError(e.to_string().into()))?
                .invoke_captured(
                    base..base + space.width().bytes() as usize,
                    [u64::from(mxcsr), 0],
                    dispatch,
                    opaque,
                    NativeInvocation {
                        gateway,
                        context: opaque,
                        entry: 0,
                    },
                )
        }
        .map_err(|e| {
            memory_error(
                self.context
                    .fail(MemoryError::Execution(e.to_string().into())),
            )
        })?;
        drop(measurement);
        if !call.ok {
            return Err(memory_error(
                self.context.fail(MemoryError::Execution(ffi::error())),
            ));
        }
        if let Some(error) = callbacks.error.take() {
            return Err(memory_error(error));
        }
        let fetch_fault = if call.exit.kind == 3
            && call.exit.detail == 9
            && callbacks.code.is_empty()
        {
            let address = GuestVirtualAddress::new(call.exit.pc);
            match callbacks.cache.fetch(space, &admission, address) {
                Err(error) => {
                    use nixe_cpu::error::InstructionFetchFaultReason as Reason;
                    let reason = if !address.get().is_multiple_of(4) {
                        Reason::Misaligned
                    } else {
                        match error {
                            MemoryError::Unmapped(_) => Reason::Unmapped,
                            MemoryError::PermissionDenied { .. } => Reason::ExecutePermissionDenied,
                            MemoryError::AddressRange { .. } => Reason::AddressOverflow,
                            error => Reason::Memory(error.to_string().into()),
                        }
                    };
                    Some(nixe_cpu::error::InstructionFetchFault::new(
                        space.identity(),
                        address,
                        reason,
                    ))
                }
                Ok(Lookup::Capture(capture)) => {
                    callbacks.code.push(capture);
                    None
                }
                Ok(Lookup::Ready(_)) => {
                    return Err(JitError(
                        "native fetch fault for executable captured code".into(),
                    ));
                }
            }
        } else {
            None
        };
        let word = if call.exit.kind == 1
            || (call.exit.kind == 3 && !matches!(call.exit.detail, 3..=7 | 9))
        {
            match callbacks
                .cache
                .fetch(space, &admission, GuestVirtualAddress::new(call.exit.pc))
                .map_err(memory_error)?
            {
                Lookup::Ready(word) => Some(word),
                Lookup::Capture(_) => {
                    return Err(JitError(
                        "executed exception has no retained code input".into(),
                    ));
                }
            }
        } else {
            None
        };
        Ok(Boundary {
            event: Event {
                kind: call.exit.kind,
                detail: call.exit.detail,
                pc: call.exit.pc,
                word,
            },
            code: callbacks.code,
            preempted: false,
            fetch_fault,
        })
    }
}
fn memory_error(error: MemoryError) -> JitError {
    JitError(error.to_string().into())
}
unsafe extern "C" fn code(opaque: *mut c_void, address: u64, output: *mut u32) -> bool {
    record(Counter::CodeFetches, 1);
    let context = unsafe { &mut *opaque.cast::<Callbacks<'_>>() };
    match context.cache.fetch(
        context.space,
        context.admission,
        GuestVirtualAddress::new(address),
    ) {
        Ok(Lookup::Ready(word)) => {
            unsafe {
                *output = word;
            }
            true
        }
        Ok(Lookup::Capture(capture)) => {
            if !context
                .code
                .iter()
                .any(|pending| pending.address() == capture.address())
            {
                context.code.push(capture);
            }
            false
        }
        // The translator may speculatively look past the executed path. Its
        // NoExecute terminal reports an architectural fault only if reached.
        Err(_) => false,
    }
}
unsafe extern "C" fn counter(opaque: *mut c_void) -> u64 {
    unsafe { &*opaque.cast::<Callbacks<'_>>() }
        .timer
        .snapshot()
        .counter
}
unsafe extern "C" fn memory(
    opaque: *mut c_void,
    address: u64,
    bytes: u32,
    operation: u32,
    value: *mut u64,
    expected: *const u64,
) -> bool {
    record(
        match operation {
            0 => Counter::ReadCallbacks,
            1 => Counter::WriteCallbacks,
            _ => Counter::AtomicCallbacks,
        },
        1,
    );
    let context = unsafe { &mut *opaque.cast::<Callbacks<'_>>() };
    if !matches!(bytes, 1 | 2 | 4 | 8 | 16) || operation > 2 {
        context.error = Some(MemoryError::Invalid("invalid native memory callback"));
        return false;
    }
    if context.error.is_some() {
        return false;
    }
    let capture = (|| {
        if operation == 2 && !address.is_multiple_of(u64::from(bytes)) {
            unsafe {
                *value = 0;
            }
            return Ok(true);
        }
        let required = match operation {
            0 => MemoryPermissions::READ,
            _ => MemoryPermissions::WRITE,
        };
        let address = GuestVirtualAddress::new(address);
        let input = unsafe { u128::from(*value) | (u128::from(*value.add(1)) << 64) };
        if operation == 0
            && let Some(result) = context.space.read_native(
                context.admission,
                address,
                bytes as usize,
                context.read_area.as_ref(),
            )?
        {
            unsafe {
                *value = result as u64;
                *value.add(1) = (result >> 64) as u64;
            }
            return Ok(true);
        }
        if operation == 1
            && context.space.write_native(
                context.admission,
                address,
                &input.to_le_bytes()[..bytes as usize],
            )?
        {
            return Ok(true);
        }
        let ranges =
            context
                .space
                .capture_native(context.admission, address, u64::from(bytes), required)?;
        let access = match operation {
            0 => AccessOperation::ReadNative,
            1 => AccessOperation::Write(input.to_le_bytes()[..bytes as usize].into()),
            2 => {
                let expected =
                    unsafe { u128::from(*expected) | (u128::from(*expected.add(1)) << 64) };
                AccessOperation::Atomic(AtomicAccess {
                    bytes: bytes as u8,
                    ordering: AccessOrdering::SequentiallyConsistent,
                    operation: AtomicOperation::CompareExchange {
                        expected,
                        replacement: input,
                    },
                })
            }
            _ => unreachable!(),
        };
        context.admission.suspend();
        let completed = context
            .process
            .resolve_access(context.space.machine().access_captured(&ranges, access));
        context.admission.resume()?;
        // Dynarmic queues invalidations while Run is active. Its generated code
        // is retired only after return, so no suspended native frame is freed.
        context.admission.synchronize_code(|range| {
            if unsafe {
                ffi::nixe_dynarmic_invalidate(context.core, range.start.get(), range.bytes as usize)
            } {
                Ok(())
            } else {
                Err(MemoryError::Execution(ffi::error()))
            }
        })?;
        match completed {
            Ok(AccessResult::NativeRead {
                bytes: result,
                area,
            }) => {
                let mut data = [0; 16];
                data[..result.len()].copy_from_slice(&result);
                let result = u128::from_le_bytes(data);
                unsafe {
                    *value = result as u64;
                    *value.add(1) = (result >> 64) as u64;
                }
                *context.read_area = area;
            }
            Ok(AccessResult::Written) => {}
            Ok(AccessResult::Atomic { written, .. }) => unsafe {
                *value = u64::from(written);
            },
            Ok(_) => return Err(MemoryError::Invalid("unexpected native access completion")),
            Err(error) => {
                context.error = Some(MemoryError::Execution(error.to_string().into()));
                return Ok(false);
            }
        }
        Ok(true)
    })();
    match capture {
        Ok(done) => done,
        Err(error) if error.is_access_fault() => {
            // The normal callback policy does not abort or reconstruct an
            // invalid guest instruction. Match the reference default values.
            unsafe {
                *value = u64::from(operation == 2);
                *value.add(1) = 0;
            }
            true
        }
        Err(error) => {
            let error = context.process.fail(Arc::new(error));
            context.error = Some(MemoryError::Execution(error.to_string().into()));
            false
        }
    }
}
unsafe extern "C" fn gateway(opaque: *mut c_void, _: usize) {
    let call = unsafe { &mut *opaque.cast::<Invocation>() };
    call.ok = unsafe { ffi::nixe_dynarmic_run(call.core, call.context, &mut call.exit) };
}
unsafe extern "C" fn dispatch(
    _opaque: *mut c_void,
    fault: *mut CapturedFault<'_>,
) -> FaultDisposition {
    record(Counter::FaultDispatches, 1);
    let fault = unsafe { &mut *fault };
    // Faulted sites use the canonical page table on subsequent executions.
    // Dynamic permissions restore direct RAM loads/stores when pages reopen;
    // device accesses do not keep generating host protection signals.
    let (mut entry, mut ret) = (0, 0);
    if !unsafe { ffi::nixe_dynarmic_resolve_fault(fault.native_pc() as u64, &mut entry, &mut ret) }
    {
        return FaultDisposition::FatalUnattributed;
    }
    unsafe { fault.call_on_resume(entry as usize, ret as usize) }
}
