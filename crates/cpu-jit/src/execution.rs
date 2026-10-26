//! Native execution on one canonical process arena. Cold operations own their
//! original physical inputs before the native admission is released.
use crate::{
    JitError, Native, ThreadState,
    code::{CodeCache, CodeCapture, Lookup, ResolvedCode},
    ffi,
    instruction::{CompletedInstruction, ExclusiveUpdate, InstructionContinuation},
    metrics::{Counter, Timer, record},
};
use nixe_cpu::{
    execution::{ArchitecturalTimer, ControlSnapshot, CpuControl, CpuExit},
    state::a64::A64State,
};
use nixe_cpu_direct_memory::{CapturedFault, FaultDisposition, NativeInvocation, NativeWorker};
use nixe_memory::{
    AccessOperation, AccessOrdering, AddressSpace, AllocationRange, AtomicAccess, AtomicOperation,
    CoreContext, GuestVirtualAddress, MemoryError, MemoryPermissions, NativeAdmission,
    NativeReadArea, NativeReservation,
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
    cpu: nixe_cpu::profile::ProcessCpuContext,
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
    pub value: u64,
    pub word: Option<u32>,
}
pub struct Boundary {
    pub ticks: u64,
    pub event: Event,
    pub code: Vec<CodeCapture>,
    pub memory: Option<InstructionContinuation>,
    pub control: Option<ControlSnapshot>,
    pub fault: Option<DataFault>,
    pub invalidate_code: Option<GuestVirtualAddress>,
    pub fetch_fault: Option<nixe_cpu::error::InstructionFetchFault>,
}
pub struct DataFault {
    pub pc: GuestVirtualAddress,
    pub address: GuestVirtualAddress,
    pub bytes: u8,
    pub kind: nixe_cpu::memory::DataAccessKind,
    pub cause: MemoryError,
}
/// The callback's exact subaccess, captured under native admission.
/// A pair/structure continuation uses the fault address to exclude committed
/// stores; neither this operation nor its remaining suffix may retranslate VA.
pub(crate) struct CapturedAccess {
    pub address: GuestVirtualAddress,
    pub bytes: u8,
    pub operation: AccessOperation,
    pub ranges: Vec<AllocationRange>,
}
struct Subaccess {
    address: GuestVirtualAddress,
    bytes: u8,
    operation: AccessOperation,
    ranges: Vec<AllocationRange>,
}
struct Callbacks<'a> {
    space: &'a AddressSpace,
    admission: &'a NativeAdmission<'a>,
    timer: &'a dyn ArchitecturalTimer,
    cache: &'a mut CodeCache,
    read_area: &'a Option<NativeReadArea>,
    code: Vec<CodeCapture>,
    memory: Option<Subaccess>,
    error: Option<MemoryError>,
    fault: Option<DataFault>,
}
struct Invocation {
    core: *mut c_void,
    context: *mut c_void,
    budget: u64,
    exit: ffi::Exit,
    ok: bool,
}
impl Core {
    pub fn new(space: Arc<AddressSpace>, configuration: Configuration) -> Result<Self, JitError> {
        if configuration.cpu.address_space_id() != space.identity() {
            return Err(JitError(
                "CPU configuration belongs to another address space".into(),
            ));
        }
        if configuration.timer_frequency == 0 {
            return Err(JitError(
                "architectural timer frequency must be nonzero".into(),
            ));
        }
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
                    data_cache,
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
            cpu: configuration.cpu,
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
    pub fn load(&mut self, state: &A64State) {
        record(Counter::ContextLoads, 1);
        unsafe {
            ffi::nixe_dynarmic_load(self.handle.pointer, state);
        }
    }
    pub fn save(&mut self, state: &mut A64State) {
        record(Counter::ContextSaves, 1);
        unsafe {
            ffi::nixe_dynarmic_save(self.handle.pointer, state);
        }
    }
    pub fn read_register(&self, register: u32) -> u64 {
        unsafe { ffi::nixe_dynarmic_read(self.handle.pointer, register) }
    }
    pub fn write_register(&mut self, register: u32, value: u64) {
        unsafe {
            ffi::nixe_dynarmic_write(self.handle.pointer, register, value);
        }
    }
    pub fn clear_exclusive(&mut self) {
        unsafe {
            ffi::nixe_dynarmic_clear_exclusive(self.handle.pointer);
        }
    }
    pub fn reserve_exclusive(&mut self, reservation: &NativeReservation) {
        unsafe {
            ffi::nixe_dynarmic_reserve_exclusive(self.handle.pointer, reservation);
        }
    }
    pub fn complete(
        &mut self,
        completed: CompletedInstruction,
    ) -> Result<(Option<CpuExit>, u64), MemoryError> {
        if completed.space != self.space().identity() {
            return Err(MemoryError::Invalid(
                "instruction belongs to another address space",
            ));
        }
        record(Counter::GuestTicks, completed.ticks);
        if let Some(area) = completed.read_area {
            self.read_area = Some(area);
        }
        self.load(&completed.state);
        match completed.exclusive {
            ExclusiveUpdate::Keep => {}
            ExclusiveUpdate::Clear => self.clear_exclusive(),
            ExclusiveUpdate::Set(proof) => self.reserve_exclusive(&proof),
        }
        Ok((completed.stop, completed.ticks))
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
        budget: u64,
        timer: &dyn ArchitecturalTimer,
    ) -> Result<Boundary, JitError> {
        if budget == 0 || budget > i64::MAX as u64 {
            return Err(JitError("native budget must be in 1..=i64::MAX".into()));
        }
        let handle = &self.handle;
        unsafe {
            ffi::nixe_dynarmic_clear_halt(handle.pointer);
        }
        let admission_timer = Timer::new(Counter::AdmissionNanoseconds);
        let admission = self.context.enter_native().map_err(memory_error)?;
        drop(admission_timer);
        // A racing preemption is either consumed here or remains in the
        // native halt word. The caller must not clear Halt after this check.
        if let Some(control) = self.control.take_pending() {
            return Ok(Boundary {
                ticks: 0,
                event: Event {
                    kind: 0,
                    detail: 0,
                    pc: 0,
                    value: 0,
                    word: None,
                },
                code: Vec::new(),
                memory: None,
                control: Some(control),
                fault: None,
                invalidate_code: None,
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
            read_area: &self.read_area,
            code: Vec::new(),
            memory: None,
            error: None,
            fault: None,
        };
        let mut call = Invocation {
            core: handle.pointer,
            context: std::ptr::from_mut(&mut callbacks).cast(),
            budget,
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
        let executing = self.control.enter_execution();
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
        // Dynarmic charges its synthetic missing-fetch/unsupported terminal as
        // one cycle. Neither terminal executes the selected guest instruction.
        // Code preparation must not consume the caller's quantum; memory
        // completion charges the architectural operation itself.
        if call.exit.kind == 1 || (call.exit.kind == 3 && call.exit.detail == 9) {
            call.exit.ticks = call.exit.ticks.saturating_sub(1);
        }
        record(Counter::GuestTicks, call.exit.ticks);
        if !call.ok {
            return Err(memory_error(
                self.context.fail(MemoryError::Execution(ffi::error())),
            ));
        }
        drop(executing);
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
        if matches!(call.exit.kind, 4 | 5) {
            call.exit.pc = unsafe { ffi::nixe_dynarmic_read(handle.pointer, 32) }.wrapping_sub(4);
        }
        let invalidate_code = if call.exit.kind == 5 && call.exit.detail == 0 {
            let address = GuestVirtualAddress::new(call.exit.value);
            match space.check_native_access(&admission, address, MemoryPermissions::READ) {
                Ok(_) => Some(GuestVirtualAddress::new(call.exit.value & !63)),
                Err(cause) => {
                    callbacks.fault = Some(DataFault {
                        pc: GuestVirtualAddress::new(call.exit.pc),
                        address,
                        bytes: 1,
                        kind: nixe_cpu::memory::DataAccessKind::Read,
                        cause,
                    });
                    None
                }
            }
        } else {
            None
        };
        if let Some(fault) = &mut callbacks.fault {
            fault.pc = GuestVirtualAddress::new(if call.exit.kind == 4 || call.exit.kind == 5 {
                call.exit.pc
            } else {
                unsafe { ffi::nixe_dynarmic_read(handle.pointer, 32) }
            });
        }
        let word = if call.exit.kind == 3 && !matches!(call.exit.detail, 3..=7 | 9) {
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
        let memory = if callbacks.memory.is_some() || call.exit.kind == 1 {
            let pc = GuestVirtualAddress::new(if call.exit.kind == 1 {
                call.exit.pc
            } else {
                unsafe { ffi::nixe_dynarmic_read(handle.pointer, 32) }
            });
            let instruction = match callbacks
                .cache
                .fetch(space, &admission, pc)
                .map_err(memory_error)?
            {
                Lookup::Ready(word) => word,
                Lookup::Capture(_) => {
                    return Err(JitError(
                        "executed instruction has no retained code input".into(),
                    ));
                }
            };
            let captured = callbacks.memory.take().map(|input| CapturedAccess {
                address: input.address,
                bytes: input.bytes,
                operation: input.operation,
                ranges: input.ranges,
            });
            let mut state = A64State::default();
            record(Counter::ContextSaves, 1);
            unsafe {
                ffi::nixe_dynarmic_save(handle.pointer, &mut state);
            }
            state.set_pc(pc.get());
            Some(InstructionContinuation::capture(
                self.cpu,
                state,
                (captured, instruction),
                space,
                &admission,
            )?)
        } else {
            None
        };
        Ok(Boundary {
            ticks: call.exit.ticks,
            event: Event {
                kind: call.exit.kind,
                detail: call.exit.detail,
                pc: call.exit.pc,
                value: call.exit.value,
                word,
            },
            code: callbacks.code,
            memory,
            control: None,
            fault: callbacks.fault,
            invalidate_code,
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
unsafe extern "C" fn data_cache(opaque: *mut c_void, address: u64) -> bool {
    let context = unsafe { &mut *opaque.cast::<Callbacks<'_>>() };
    match context.space.check_native_access(
        context.admission,
        GuestVirtualAddress::new(address),
        MemoryPermissions::READ,
    ) {
        Ok(_) => true,
        Err(cause) => {
            context.fault = Some(DataFault {
                pc: GuestVirtualAddress::MIN,
                address: GuestVirtualAddress::new(address),
                bytes: 1,
                kind: nixe_cpu::memory::DataAccessKind::Read,
                cause,
            });
            false
        }
    }
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
    if !matches!(bytes, 1 | 2 | 4 | 8 | 16) || operation > 5 || operation == 4 {
        context.error = Some(MemoryError::Invalid("invalid native memory callback"));
        return false;
    }
    if context.memory.is_some() || context.fault.is_some() {
        context.error = Some(MemoryError::Invalid(
            "native instruction continued after memory abort",
        ));
        return false;
    }
    let capture = (|| {
        if matches!(operation, 2 | 3 | 5) && !address.is_multiple_of(u64::from(bytes)) {
            return Err(MemoryError::Invalid("unaligned native atomic access"));
        }
        let required = match operation {
            0 | 3 => MemoryPermissions::READ,
            1 | 5 => MemoryPermissions::WRITE,
            _ => MemoryPermissions::READ_WRITE,
        };
        let address = GuestVirtualAddress::new(address);
        // Ordinary RAM stores complete under the current admission. Cold STXR
        // needs physical identity to reject a mismatched proof immediately;
        // deferred effects are captured before this admission is released.
        let ranges = if operation == 5 {
            context
                .space
                .capture_native(context.admission, address, u64::from(bytes), required)?
        } else {
            Vec::new()
        };
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
        let operation = match operation {
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
            3 => AccessOperation::Atomic(AtomicAccess {
                bytes: bytes as u8,
                ordering: AccessOrdering::SequentiallyConsistent,
                operation: AtomicOperation::LoadExclusive,
            }),
            5 => {
                let proof = unsafe { expected.cast::<NativeReservation>().read_unaligned() };
                let Some(reservation) = context
                    .space
                    .machine()
                    .capture_native_reservation(&ranges[0], proof)?
                else {
                    unsafe {
                        *value = 0;
                    }
                    return Ok(true);
                };
                AccessOperation::Atomic(AtomicAccess {
                    bytes: bytes as u8,
                    ordering: AccessOrdering::SequentiallyConsistent,
                    operation: AtomicOperation::StoreExclusive {
                        reservation,
                        replacement: input,
                    },
                })
            }
            _ => unreachable!(),
        };
        context.memory = Some(Subaccess {
            address,
            bytes: bytes as u8,
            operation,
            ranges,
        });
        Ok(false)
    })();
    match capture {
        Ok(done) => done,
        Err(error) => {
            context.fault = Some(DataFault {
                pc: GuestVirtualAddress::MIN,
                address: GuestVirtualAddress::new(address),
                bytes: bytes as u8,
                kind: if matches!(operation, 0 | 3) {
                    nixe_cpu::memory::DataAccessKind::Read
                } else {
                    nixe_cpu::memory::DataAccessKind::Write
                },
                cause: error,
            });
            false
        }
    }
}
unsafe extern "C" fn gateway(opaque: *mut c_void, _: usize) {
    let call = unsafe { &mut *opaque.cast::<Invocation>() };
    call.ok =
        unsafe { ffi::nixe_dynarmic_run(call.core, call.context, call.budget, &mut call.exit) };
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
