//! JIT-private native execution-frame and helper ABI.

use core::mem::{align_of, offset_of, size_of};

use nixe_cpu::location::ExecutionState;
use nixe_cpu::state::a32::{A32GeneralRegister, Cpsr, DOUBLE_REGISTER_COUNT};
use nixe_cpu::state::a64::{
    A64GeneralRegister, A64Register, GENERAL_REGISTER_COUNT as A64_GPR_COUNT, Nzcv,
    VECTOR_REGISTER_COUNT,
};
use nixe_cpu::state::{A32State, A64State, ThreadCpuState};

pub(crate) const NATIVE_ABI_VERSION: u32 = 6;
pub(crate) const NO_LOADER_RETURN: u64 = u64::MAX;

pub(crate) const EXECUTION_STATE_A64: u32 = 0;
pub(crate) const EXECUTION_STATE_A32: u32 = 1;
pub(crate) const EXECUTION_STATE_T32: u32 = 2;

pub(crate) const EXIT_NONE: u32 = 0;
pub(crate) const EXIT_BUDGET_EXHAUSTED: u32 = 1;
pub(crate) const EXIT_SAFEPOINT: u32 = 2;
pub(crate) const EXIT_PENDING_EVENT: u32 = 3;
pub(crate) const EXIT_LOADER_RETURN: u32 = 4;
pub(crate) const EXIT_DISPATCH: u32 = 5;
pub(crate) const EXIT_ARCHITECTURAL: u32 = 6;
pub(crate) const EXIT_UNSUPPORTED: u32 = 7;
pub(crate) const EXIT_DATA_FAULT: u32 = 8;
pub(crate) const EXIT_SCHEDULED: u32 = 9;
pub(crate) const EXIT_INTERNAL: u32 = 10;

pub(crate) const SCHEDULE_YIELD: u32 = 1;
pub(crate) const SCHEDULE_WAIT_FOR_EVENT: u32 = 2;
pub(crate) const SCHEDULE_WAIT_FOR_INTERRUPT: u32 = 3;
pub(crate) const SCHEDULE_SEND_EVENT: u32 = 4;

pub(crate) const SYSTEM_POLL: u32 = 0;
pub(crate) const SYSTEM_CACHE_INSTRUCTION_INVALIDATE: u32 = 1;
pub(crate) const SYSTEM_CACHE_DATA_INVALIDATE: u32 = 2;
pub(crate) const SYSTEM_CACHE_DATA_CLEAN: u32 = 3;
pub(crate) const SYSTEM_CACHE_DATA_CLEAN_INVALIDATE: u32 = 4;
pub(crate) const SYSTEM_CACHE_INSTRUCTION_PREFETCH: u32 = 5;
pub(crate) const SYSTEM_READ_RUNTIME_REGISTER: u32 = 6;
pub(crate) const SYSTEM_WAIT_FOR_EVENT: u32 = 7;
pub(crate) const SYSTEM_WAIT_FOR_INTERRUPT: u32 = 8;
pub(crate) const SYSTEM_SEND_EVENT_LOCAL: u32 = 9;
pub(crate) const SYSTEM_HOTNESS_PROMOTION: u32 = 10;

pub(crate) const MAX_HELPER_ARGUMENTS: usize = 16;
pub(crate) const MAX_HELPER_RESULTS: usize = 4;

/// Stable two-limb representation used instead of a compiler-specific `u128`
/// ABI decision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct AbiU128 {
    pub(crate) low: u64,
    pub(crate) high: u64,
}

/// Fixed scratch storage used only while crossing one typed helper boundary.
/// Generated code never passes compiler-specific vector or 128-bit aggregate
/// conventions through the C ABI.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C, align(16))]
pub(crate) struct HelperScratch {
    pub(crate) arguments: [AbiU128; MAX_HELPER_ARGUMENTS],
    pub(crate) results: [AbiU128; MAX_HELPER_RESULTS],
}

impl From<u128> for AbiU128 {
    fn from(value: u128) -> Self {
        Self {
            low: value as u64,
            high: (value >> 64) as u64,
        }
    }
}

impl From<AbiU128> for u128 {
    fn from(value: AbiU128) -> Self {
        u128::from(value.low) | (u128::from(value.high) << 64)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct A64Frame {
    x: [u64; A64_GPR_COUNT],
    sp: u64,
    pc: u64,
    vector: [AbiU128; VECTOR_REGISTER_COUNT],
    tpidr_el0: u64,
    tpidrro_el0: u64,
    nzcv: u32,
    fpcr: u32,
    fpsr: u32,
    reserved: u32,
}

impl Default for A64Frame {
    fn default() -> Self {
        Self {
            x: [0; A64_GPR_COUNT],
            sp: 0,
            pc: 0,
            vector: [AbiU128::default(); VECTOR_REGISTER_COUNT],
            tpidr_el0: 0,
            tpidrro_el0: 0,
            nzcv: 0,
            fpcr: 0,
            fpsr: 0,
            reserved: 0,
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct A32Frame {
    r: [u32; nixe_cpu::state::a32::GENERAL_REGISTER_COUNT],
    pc: u32,
    cpsr: u32,
    d: [u64; DOUBLE_REGISTER_COUNT],
    fpscr: u32,
    tpidrurw: u32,
    tpidruro: u32,
    reserved: u32,
}

/// Linux fastmem addresses and the mapping epoch visible to native code.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct MemoryAcceleration {
    pub(crate) address_space: u64,
    pub(crate) mapping_epoch: u64,
    pub(crate) fastmem_base: usize,
    pub(crate) fastmem_entries: usize,
    pub(crate) fastmem_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct NativeControl {
    pub(crate) instruction_budget: u64,
    pub(crate) loader_return: u64,
    pub(crate) invalidation_epoch: u64,
    pub(crate) control_pending_address: usize,
    pub(crate) interrupt_pending_address: usize,
    pub(crate) request_flags: u32,
    pub(crate) event_mask: u32,
}

impl Default for NativeControl {
    fn default() -> Self {
        Self {
            instruction_budget: 0,
            loader_return: NO_LOADER_RETURN,
            invalidation_epoch: 0,
            control_pending_address: 0,
            interrupt_pending_address: 0,
            request_flags: 0,
            event_mask: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct NativeExit {
    pub(crate) kind: u32,
    pub(crate) detail: u32,
    pub(crate) source_pc: u64,
    pub(crate) payload0: u64,
    pub(crate) payload1: u64,
    pub(crate) instructions_executed: u64,
}

/// Link-chain state that is private to one executor invocation. Native regions
/// carry this state across tail calls; Rust resets the retired count whenever
/// the sole link resolver regains control.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct NativeDispatch {
    pub(crate) link_table: usize,
    pub(crate) metadata: usize,
    pub(crate) region_id: u64,
    pub(crate) retired: u64,
}

macro_rules! define_helper_abi {
    (
        $(
            $field:ident : $helper:ident(
                $($argument:ident : $argument_type:ty),* $(,)?
            ) -> $result:ty;
        )+
    ) => {
        $(
            pub(crate) type $helper = unsafe extern "C" fn(
                $($argument: $argument_type),*
            ) -> $result;
        )+

        /// Typed helper slots are versioned independently from Cranelift and
        /// the engine-neutral Rust API. Null slots cannot be published into
        /// generated code.
        #[derive(Clone, Copy)]
        #[repr(C)]
        pub(crate) struct HelperTable {
            pub(crate) abi_version: u32,
            pub(crate) byte_size: u32,
            $(pub(crate) $field: Option<$helper>,)+
        }
    };
}

define_helper_abi! {
    memory_read: MemoryReadHelper(
        frame: *mut ExecutionFrame,
        address: u64,
        descriptor: u64,
        result: *mut AbiU128,
    ) -> u32;
    memory_write: MemoryWriteHelper(
        frame: *mut ExecutionFrame,
        address: u64,
        descriptor: u64,
        value: *const AbiU128,
    ) -> u32;
    atomic: AtomicHelper(
        frame: *mut ExecutionFrame,
        address: u64,
        descriptor: u64,
        operand: *const AbiU128,
        result: *mut AbiU128,
    ) -> u32;
    exclusive: ExclusiveHelper(
        frame: *mut ExecutionFrame,
        address: u64,
        descriptor: u64,
        value: *const AbiU128,
        result: *mut AbiU128,
    ) -> u32;
    semantic: SemanticHelper(
        frame: *mut ExecutionFrame,
        operation: u32,
    ) -> u32;
    system: SystemHelper(
        frame: *mut ExecutionFrame,
        operation: u32,
        argument: u64,
    ) -> u32;
}

/// Tail-convention entries are never invoked directly by Rust.
pub(crate) type NativeEntryAddress = usize;

/// The Cranelift-generated gateway is the sole C-ABI entry into a native chain.
pub(crate) type NativeGateway =
    unsafe extern "C" fn(frame: *mut ExecutionFrame, entry: NativeEntryAddress);

pub(crate) static EMPTY_HELPER_TABLE: HelperTable = HelperTable {
    abi_version: NATIVE_ABI_VERSION,
    byte_size: size_of::<HelperTable>() as u32,
    memory_read: None,
    memory_write: None,
    atomic: None,
    exclusive: None,
    semantic: None,
    system: None,
};

/// The sole layout visible to generated code. Both architectural payloads are
/// present so no Rust enum or union layout becomes part of the native ABI.
#[repr(C, align(16))]
pub(crate) struct ExecutionFrame {
    pub(crate) abi_version: u32,
    pub(crate) byte_size: u32,
    pub(crate) execution_state: u32,
    pub(crate) reserved: u32,
    a64: A64Frame,
    a32: A32Frame,
    pub(crate) memory: MemoryAcceleration,
    pub(crate) control: NativeControl,
    pub(crate) dispatch: NativeDispatch,
    pub(crate) helpers: usize,
    pub(crate) host_context: usize,
    pub(crate) scratch: HelperScratch,
    pub(crate) exit: NativeExit,
}

impl Default for ExecutionFrame {
    fn default() -> Self {
        Self {
            abi_version: NATIVE_ABI_VERSION,
            byte_size: size_of::<Self>() as u32,
            execution_state: EXECUTION_STATE_A64,
            reserved: 0,
            a64: A64Frame::default(),
            a32: A32Frame::default(),
            memory: MemoryAcceleration::default(),
            control: NativeControl::default(),
            dispatch: NativeDispatch::default(),
            helpers: std::ptr::from_ref(&EMPTY_HELPER_TABLE).addr(),
            host_context: 0,
            scratch: HelperScratch::default(),
            exit: NativeExit::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameError {
    StateKindChanged,
    InconsistentA32ExecutionState,
    InvalidA32InstructionAddress,
}

impl ExecutionFrame {
    pub(crate) fn install_host_context(&mut self, helpers: &'static HelperTable, context: *mut ()) {
        self.helpers = std::ptr::from_ref(helpers).addr();
        self.host_context = context.addr();
    }

    pub(crate) fn clear_host_context(&mut self) {
        self.helpers = std::ptr::from_ref(&EMPTY_HELPER_TABLE).addr();
        self.host_context = 0;
    }

    pub(crate) fn import_state(&mut self, state: &ThreadCpuState) {
        self.exit = NativeExit::default();
        match state {
            ThreadCpuState::A64(state) => self.import_a64(state),
            ThreadCpuState::A32(state) => self.import_a32(state),
        }
    }

    pub(crate) fn commit_state(&self, state: &mut ThreadCpuState) -> Result<(), FrameError> {
        match (self.execution_state, state) {
            (EXECUTION_STATE_A64, ThreadCpuState::A64(state)) => {
                self.commit_a64(state);
                Ok(())
            }
            (EXECUTION_STATE_A32 | EXECUTION_STATE_T32, ThreadCpuState::A32(state)) => {
                self.commit_a32(state)
            }
            _ => Err(FrameError::StateKindChanged),
        }
    }

    pub(crate) fn current_pc(&self) -> u64 {
        match self.execution_state {
            EXECUTION_STATE_A64 => self.a64.pc,
            EXECUTION_STATE_A32 | EXECUTION_STATE_T32 => u64::from(self.a32.pc),
            _ => 0,
        }
    }

    pub(crate) fn execution_state(&self) -> Option<ExecutionState> {
        match self.execution_state {
            EXECUTION_STATE_A64 => Some(ExecutionState::A64),
            EXECUTION_STATE_A32 => Some(ExecutionState::A32),
            EXECUTION_STATE_T32 => Some(ExecutionState::T32),
            _ => None,
        }
    }

    pub(crate) fn a64_result_code(&self) -> Option<u64> {
        (self.execution_state == EXECUTION_STATE_A64).then_some(self.a64.x[0])
    }

    fn import_a64(&mut self, state: &A64State) {
        self.execution_state = EXECUTION_STATE_A64;
        for (index, target) in self.a64.x.iter_mut().enumerate() {
            let register = A64GeneralRegister::new(index as u8).expect("frame has X0-X30");
            *target = state.read_x(A64Register::General(register));
        }
        self.a64.sp = state.read_x(A64Register::StackPointer);
        self.a64.pc = state.pc();
        self.a64.nzcv = state.nzcv().bits();
        for (index, target) in self.a64.vector.iter_mut().enumerate() {
            *target = state.vector(index as u8).expect("frame has V0-V31").into();
        }
        self.a64.fpcr = state.fpcr();
        self.a64.fpsr = state.fpsr();
        self.a64.tpidr_el0 = state.tpidr_el0();
        self.a64.tpidrro_el0 = state.tpidrro_el0();
    }

    fn import_a32(&mut self, state: &A32State) {
        self.execution_state = match state.execution_state() {
            ExecutionState::A32 => EXECUTION_STATE_A32,
            ExecutionState::T32 => EXECUTION_STATE_T32,
            ExecutionState::A64 => unreachable!("A32 state cannot select A64"),
        };
        for (index, target) in self.a32.r.iter_mut().enumerate() {
            let register = A32GeneralRegister::new(index as u8).expect("frame has R0-R14");
            *target = state.read_r(register);
        }
        self.a32.pc = state.instruction_address();
        self.a32.cpsr = state.cpsr().bits();
        for (index, target) in self.a32.d.iter_mut().enumerate() {
            *target = state.read_d(index as u8).expect("frame has D0-D31");
        }
        self.a32.fpscr = state.fpscr();
        self.a32.tpidrurw = state.tpidrurw();
        self.a32.tpidruro = state.tpidruro();
    }

    fn commit_a64(&self, state: &mut A64State) {
        for (index, value) in self.a64.x.iter().copied().enumerate() {
            let register = A64GeneralRegister::new(index as u8).expect("frame has X0-X30");
            state.write_x(A64Register::General(register), value);
        }
        state.write_x(A64Register::StackPointer, self.a64.sp);
        state.set_pc(self.a64.pc);
        state.set_nzcv(Nzcv::from_bits(self.a64.nzcv));
        for (index, value) in self.a64.vector.iter().copied().enumerate() {
            assert!(state.set_vector(index as u8, value.into()));
        }
        state.set_fpcr(self.a64.fpcr);
        state.set_fpsr(self.a64.fpsr);
        state.set_tpidr_el0(self.a64.tpidr_el0);
        state.set_tpidrro_el0_from_runtime(self.a64.tpidrro_el0);
    }

    fn commit_a32(&self, state: &mut A32State) -> Result<(), FrameError> {
        for (index, value) in self.a32.r.iter().copied().enumerate() {
            let register = A32GeneralRegister::new(index as u8).expect("frame has R0-R14");
            state.write_r(register, value);
        }
        let cpsr = Cpsr::from_bits(self.a32.cpsr);
        let expected_state = match self.execution_state {
            EXECUTION_STATE_A32 => ExecutionState::A32,
            EXECUTION_STATE_T32 => ExecutionState::T32,
            _ => return Err(FrameError::StateKindChanged),
        };
        if cpsr.execution_state() != expected_state {
            return Err(FrameError::InconsistentA32ExecutionState);
        }
        state.set_cpsr(cpsr);
        state
            .set_instruction_address(self.a32.pc)
            .map_err(|_| FrameError::InvalidA32InstructionAddress)?;
        for (index, value) in self.a32.d.iter().copied().enumerate() {
            assert!(state.write_d(index as u8, value));
        }
        state.set_fpscr(self.a32.fpscr);
        state.set_tpidrurw(self.a32.tpidrurw);
        state.set_tpidruro_from_runtime(self.a32.tpidruro);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameOffsets {
    pub(crate) execution_state: usize,
    pub(crate) a64_x: usize,
    pub(crate) a64_sp: usize,
    pub(crate) a64_pc: usize,
    pub(crate) a64_vector: usize,
    pub(crate) a64_tpidr_el0: usize,
    pub(crate) a64_tpidrro_el0: usize,
    pub(crate) a64_nzcv: usize,
    pub(crate) a64_fpcr: usize,
    pub(crate) a64_fpsr: usize,
    pub(crate) a32_r: usize,
    pub(crate) a32_pc: usize,
    pub(crate) a32_cpsr: usize,
    pub(crate) a32_d: usize,
    pub(crate) a32_fpscr: usize,
    pub(crate) a32_tpidrurw: usize,
    pub(crate) a32_tpidruro: usize,
    pub(crate) memory_address_space: usize,
    pub(crate) memory_mapping_epoch: usize,
    pub(crate) memory_fastmem_base: usize,
    pub(crate) memory_fastmem_entries: usize,
    pub(crate) memory_fastmem_size: usize,
    pub(crate) control_instruction_budget: usize,
    pub(crate) control_loader_return: usize,
    pub(crate) control_invalidation_epoch: usize,
    pub(crate) control_pending_address: usize,
    pub(crate) interrupt_pending_address: usize,
    pub(crate) control_request_flags: usize,
    pub(crate) control_event_mask: usize,
    pub(crate) dispatch_link_table: usize,
    pub(crate) dispatch_metadata: usize,
    pub(crate) dispatch_region_id: usize,
    pub(crate) dispatch_retired: usize,
    pub(crate) helpers: usize,
    pub(crate) host_context: usize,
    pub(crate) scratch_arguments: usize,
    pub(crate) scratch_results: usize,
    pub(crate) exit_kind: usize,
    pub(crate) exit_detail: usize,
    pub(crate) exit_source_pc: usize,
    pub(crate) exit_payload0: usize,
    pub(crate) exit_payload1: usize,
    pub(crate) exit_instructions_executed: usize,
}

pub(crate) const FRAME_OFFSETS: FrameOffsets = FrameOffsets {
    execution_state: offset_of!(ExecutionFrame, execution_state),
    a64_x: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, x),
    a64_sp: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, sp),
    a64_pc: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, pc),
    a64_vector: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, vector),
    a64_tpidr_el0: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, tpidr_el0),
    a64_tpidrro_el0: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, tpidrro_el0),
    a64_nzcv: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, nzcv),
    a64_fpcr: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, fpcr),
    a64_fpsr: offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, fpsr),
    a32_r: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, r),
    a32_pc: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, pc),
    a32_cpsr: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, cpsr),
    a32_d: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, d),
    a32_fpscr: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, fpscr),
    a32_tpidrurw: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, tpidrurw),
    a32_tpidruro: offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, tpidruro),
    memory_address_space: offset_of!(ExecutionFrame, memory)
        + offset_of!(MemoryAcceleration, address_space),
    memory_mapping_epoch: offset_of!(ExecutionFrame, memory)
        + offset_of!(MemoryAcceleration, mapping_epoch),
    memory_fastmem_base: offset_of!(ExecutionFrame, memory)
        + offset_of!(MemoryAcceleration, fastmem_base),
    memory_fastmem_entries: offset_of!(ExecutionFrame, memory)
        + offset_of!(MemoryAcceleration, fastmem_entries),
    memory_fastmem_size: offset_of!(ExecutionFrame, memory)
        + offset_of!(MemoryAcceleration, fastmem_size),
    control_instruction_budget: offset_of!(ExecutionFrame, control)
        + offset_of!(NativeControl, instruction_budget),
    control_loader_return: offset_of!(ExecutionFrame, control)
        + offset_of!(NativeControl, loader_return),
    control_invalidation_epoch: offset_of!(ExecutionFrame, control)
        + offset_of!(NativeControl, invalidation_epoch),
    control_pending_address: offset_of!(ExecutionFrame, control)
        + offset_of!(NativeControl, control_pending_address),
    interrupt_pending_address: offset_of!(ExecutionFrame, control)
        + offset_of!(NativeControl, interrupt_pending_address),
    control_request_flags: offset_of!(ExecutionFrame, control)
        + offset_of!(NativeControl, request_flags),
    control_event_mask: offset_of!(ExecutionFrame, control) + offset_of!(NativeControl, event_mask),
    dispatch_link_table: offset_of!(ExecutionFrame, dispatch)
        + offset_of!(NativeDispatch, link_table),
    dispatch_metadata: offset_of!(ExecutionFrame, dispatch) + offset_of!(NativeDispatch, metadata),
    dispatch_region_id: offset_of!(ExecutionFrame, dispatch)
        + offset_of!(NativeDispatch, region_id),
    dispatch_retired: offset_of!(ExecutionFrame, dispatch) + offset_of!(NativeDispatch, retired),
    helpers: offset_of!(ExecutionFrame, helpers),
    host_context: offset_of!(ExecutionFrame, host_context),
    scratch_arguments: offset_of!(ExecutionFrame, scratch) + offset_of!(HelperScratch, arguments),
    scratch_results: offset_of!(ExecutionFrame, scratch) + offset_of!(HelperScratch, results),
    exit_kind: offset_of!(ExecutionFrame, exit) + offset_of!(NativeExit, kind),
    exit_detail: offset_of!(ExecutionFrame, exit) + offset_of!(NativeExit, detail),
    exit_source_pc: offset_of!(ExecutionFrame, exit) + offset_of!(NativeExit, source_pc),
    exit_payload0: offset_of!(ExecutionFrame, exit) + offset_of!(NativeExit, payload0),
    exit_payload1: offset_of!(ExecutionFrame, exit) + offset_of!(NativeExit, payload1),
    exit_instructions_executed: offset_of!(ExecutionFrame, exit)
        + offset_of!(NativeExit, instructions_executed),
};

impl FrameOffsets {
    pub(crate) const fn all(self) -> [usize; 43] {
        [
            self.execution_state,
            self.a64_x,
            self.a64_sp,
            self.a64_pc,
            self.a64_vector,
            self.a64_tpidr_el0,
            self.a64_tpidrro_el0,
            self.a64_nzcv,
            self.a64_fpcr,
            self.a64_fpsr,
            self.a32_r,
            self.a32_pc,
            self.a32_cpsr,
            self.a32_d,
            self.a32_fpscr,
            self.a32_tpidrurw,
            self.a32_tpidruro,
            self.memory_address_space,
            self.memory_mapping_epoch,
            self.memory_fastmem_base,
            self.memory_fastmem_entries,
            self.memory_fastmem_size,
            self.control_instruction_budget,
            self.control_loader_return,
            self.control_invalidation_epoch,
            self.control_pending_address,
            self.interrupt_pending_address,
            self.control_request_flags,
            self.control_event_mask,
            self.dispatch_link_table,
            self.dispatch_metadata,
            self.dispatch_region_id,
            self.dispatch_retired,
            self.helpers,
            self.host_context,
            self.scratch_arguments,
            self.scratch_results,
            self.exit_kind,
            self.exit_detail,
            self.exit_source_pc,
            self.exit_payload0,
            self.exit_payload1,
            self.exit_instructions_executed,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HelperOffsets {
    pub(crate) abi_version: usize,
    pub(crate) byte_size: usize,
    pub(crate) memory_read: usize,
    pub(crate) memory_write: usize,
    pub(crate) atomic: usize,
    pub(crate) exclusive: usize,
    pub(crate) semantic: usize,
    pub(crate) system: usize,
}

pub(crate) const HELPER_OFFSETS: HelperOffsets = HelperOffsets {
    abi_version: offset_of!(HelperTable, abi_version),
    byte_size: offset_of!(HelperTable, byte_size),
    memory_read: offset_of!(HelperTable, memory_read),
    memory_write: offset_of!(HelperTable, memory_write),
    atomic: offset_of!(HelperTable, atomic),
    exclusive: offset_of!(HelperTable, exclusive),
    semantic: offset_of!(HelperTable, semantic),
    system: offset_of!(HelperTable, system),
};

impl HelperOffsets {
    pub(crate) const fn all(self) -> [usize; 8] {
        [
            self.abi_version,
            self.byte_size,
            self.memory_read,
            self.memory_write,
            self.atomic,
            self.exclusive,
            self.semantic,
            self.system,
        ]
    }
}

const _: () = assert!(align_of::<ExecutionFrame>() == 16);
const _: () = assert!(size_of::<ExecutionFrame>() <= u32::MAX as usize);
const _: () = assert!(size_of::<HelperTable>() <= u32::MAX as usize);
const _: () = assert!(size_of::<NativeGateway>() == size_of::<usize>());

#[cfg(test)]
mod tests {
    use super::*;

    fn a64_register(index: u8) -> A64Register {
        A64Register::General(A64GeneralRegister::new(index).unwrap())
    }

    #[test]
    fn offsets_are_derived_from_the_private_frame_layout() {
        assert_eq!(
            FRAME_OFFSETS.a64_x,
            offset_of!(ExecutionFrame, a64) + offset_of!(A64Frame, x)
        );
        assert_eq!(
            FRAME_OFFSETS.a32_r,
            offset_of!(ExecutionFrame, a32) + offset_of!(A32Frame, r)
        );
        assert_eq!(
            FRAME_OFFSETS.memory_mapping_epoch,
            offset_of!(ExecutionFrame, memory) + offset_of!(MemoryAcceleration, mapping_epoch)
        );
        assert_eq!(
            FRAME_OFFSETS.control_instruction_budget,
            offset_of!(ExecutionFrame, control) + offset_of!(NativeControl, instruction_budget)
        );
        assert_eq!(
            FRAME_OFFSETS.dispatch_link_table,
            offset_of!(ExecutionFrame, dispatch) + offset_of!(NativeDispatch, link_table)
        );
        assert_eq!(
            FRAME_OFFSETS.dispatch_retired,
            offset_of!(ExecutionFrame, dispatch) + offset_of!(NativeDispatch, retired)
        );
        assert_eq!(FRAME_OFFSETS.helpers, offset_of!(ExecutionFrame, helpers));
        assert_eq!(
            FRAME_OFFSETS.host_context,
            offset_of!(ExecutionFrame, host_context)
        );
        assert_eq!(
            FRAME_OFFSETS.exit_kind,
            offset_of!(ExecutionFrame, exit) + offset_of!(NativeExit, kind)
        );
        assert!(
            FRAME_OFFSETS
                .all()
                .into_iter()
                .all(|offset| offset < size_of::<ExecutionFrame>())
        );
        assert!(
            HELPER_OFFSETS
                .all()
                .into_iter()
                .all(|offset| offset < size_of::<HelperTable>())
        );
        assert_eq!(align_of::<ExecutionFrame>(), 16);
        assert_eq!(
            ExecutionFrame::default().byte_size as usize,
            size_of::<ExecutionFrame>()
        );
    }

    #[test]
    fn a64_state_round_trips_every_architectural_field() {
        let mut source = A64State::default();
        for index in 0..A64_GPR_COUNT as u8 {
            source.write_x(
                a64_register(index),
                u64::from(index) * 0x0101_0101_0101_0101,
            );
        }
        source.write_x(A64Register::StackPointer, 0x7100_0080_0000);
        source.set_pc(0x7100_0000_1000);
        source.set_nzcv(Nzcv::from_bits(Nzcv::N | Nzcv::C));
        for index in 0..VECTOR_REGISTER_COUNT as u8 {
            assert!(source.set_vector(index, (u128::from(index) << 96) | (u128::from(index) + 9)));
        }
        source.set_fpcr(0x07c0_0000);
        source.set_fpsr(0x0800_009f);
        source.set_tpidr_el0(0x1234_5678_9abc_def0);
        source.set_tpidrro_el0_from_runtime(0xfedc_ba98_7654_3210);
        let source = ThreadCpuState::A64(Box::new(source));
        let mut destination = ThreadCpuState::A64(Box::default());

        let mut frame = ExecutionFrame::default();
        frame.import_state(&source);
        frame.commit_state(&mut destination).unwrap();

        assert_eq!(destination, source);
    }

    #[test]
    fn a32_and_t32_state_round_trip_every_architectural_field() {
        for thumb in [false, true] {
            let mut source = if thumb {
                A32State::t32()
            } else {
                A32State::a32()
            };
            for index in 0..nixe_cpu::state::a32::GENERAL_REGISTER_COUNT as u8 {
                source.write_r(
                    A32GeneralRegister::new(index).unwrap(),
                    u32::from(index) * 0x0101_0101,
                );
            }
            source
                .set_instruction_address(if thumb { 0x1002 } else { 0x1000 })
                .unwrap();
            source.set_cpsr(Cpsr::from_bits(
                Cpsr::USER_MODE | Cpsr::N | Cpsr::V | if thumb { Cpsr::T } else { 0 },
            ));
            for index in 0..DOUBLE_REGISTER_COUNT as u8 {
                assert!(source.write_d(index, u64::from(index) * 0x0101_0101_0101_0101));
            }
            source.set_fpscr(0x0800_009f);
            source.set_tpidrurw(0x1234_5678);
            source.set_tpidruro_from_runtime(0x9abc_def0);
            let source = ThreadCpuState::A32(Box::new(source));
            let mut destination = ThreadCpuState::A32(Box::new(if thumb {
                A32State::t32()
            } else {
                A32State::a32()
            }));

            let mut frame = ExecutionFrame::default();
            frame.import_state(&source);
            frame.commit_state(&mut destination).unwrap();

            assert_eq!(destination, source);
        }
    }

    #[test]
    fn canonical_state_kind_cannot_change_through_the_native_frame() {
        let source = ThreadCpuState::A64(Box::default());
        let mut destination = ThreadCpuState::A32(Box::default());
        let mut frame = ExecutionFrame::default();
        frame.import_state(&source);

        assert_eq!(
            frame.commit_state(&mut destination),
            Err(FrameError::StateKindChanged)
        );
    }

    #[test]
    fn detached_helper_table_is_versioned_and_exposes_no_entrypoints() {
        assert_eq!(EMPTY_HELPER_TABLE.abi_version, NATIVE_ABI_VERSION);
        assert_eq!(
            EMPTY_HELPER_TABLE.byte_size as usize,
            size_of::<HelperTable>()
        );
        assert!(EMPTY_HELPER_TABLE.memory_read.is_none());
        assert!(EMPTY_HELPER_TABLE.memory_write.is_none());
        assert!(EMPTY_HELPER_TABLE.atomic.is_none());
        assert!(EMPTY_HELPER_TABLE.exclusive.is_none());
        assert!(EMPTY_HELPER_TABLE.semantic.is_none());
        assert!(EMPTY_HELPER_TABLE.system.is_none());
    }
}
