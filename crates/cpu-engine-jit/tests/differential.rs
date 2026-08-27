use nixe_cpu::decode::{self, DecodeResult, DecodeSupport, InstructionPattern};
use nixe_cpu::location::{ExecutionState, InstructionSize, LocationDescriptor};
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions,
    SYNTHETIC_PAGE_SIZE, SyntheticMemory,
};
use nixe_cpu::profile::{CapabilityStatus, GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{
    ThreadCpuState,
    a32::{A32GeneralRegister, A32State, Cpsr, GENERAL_REGISTER_COUNT as A32_REGISTER_COUNT},
    a64::{
        A64GeneralRegister, A64Register, A64State, GENERAL_REGISTER_COUNT as A64_REGISTER_COUNT,
        Nzcv, VECTOR_REGISTER_COUNT,
    },
};
use nixe_cpu_engine::{
    DomainMemoryBinding, DomainRequest, EngineDomain, EngineDomainId, EngineExecutor,
    EngineExecutorId, EngineProvider, EngineTimer, RunRequest, TimerSnapshot,
};
use nixe_cpu_engine_interpreter::InterpreterProvider;
use nixe_cpu_engine_jit::JitProvider;
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidationSource,
};

const SPACE: AddressSpaceId = AddressSpaceId::new(0x004a_4954_3136);
const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const DATA: GuestVirtualAddress = GuestVirtualAddress::new(0x8000);

struct FixedTimer;

impl EngineTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0x1234_5678,
            frequency: 19_200_000,
        }
    }
}

struct EngineHarness {
    domain: Box<dyn EngineDomain>,
    executor: Option<Box<dyn EngineExecutor>>,
}

impl EngineHarness {
    fn new(provider: &dyn EngineProvider, domain_id: u64, executor_id: u64) -> Self {
        let cpu = cpu();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(domain_id),
                cpu,
            })
            .unwrap();
        let binding = ExecutionMemory::new();
        domain
            .bind_memory(DomainMemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
                memory: &binding,
                mapping_epoch: binding.mapping_epoch().get(),
                invalidation_cursor: binding.invalidation_cursor(),
            })
            .unwrap();
        let executor = domain
            .create_executor(EngineExecutorId::new(executor_id))
            .unwrap();
        Self {
            domain,
            executor: Some(executor),
        }
    }

    fn run(
        &mut self,
        memory: &SyntheticMemory,
        state: &mut ThreadCpuState,
    ) -> Result<nixe_cpu_engine::ExecutionReport, nixe_cpu_engine::EngineFault> {
        self.run_with_budget(memory, state, 1)
    }

    fn run_with_budget(
        &mut self,
        memory: &SyntheticMemory,
        state: &mut ThreadCpuState,
        instruction_budget: u64,
    ) -> Result<nixe_cpu_engine::ExecutionReport, nixe_cpu_engine::EngineFault> {
        self.executor
            .as_mut()
            .expect("the harness executor is live")
            .run_slice(RunRequest {
                cpu: cpu(),
                memory,
                state,
                instruction_budget,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
    }
}

impl Drop for EngineHarness {
    fn drop(&mut self) {
        drop(self.executor.take());
        self.domain.shutdown().unwrap();
    }
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn full_width_simd_extract_compiles_without_i128_shift_immediates() {
    const CODE_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(0x30_0010);
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(CODE_PAGE));
    assert!(memory.initialize_ram(CODE_PAGE, 0, &0x6e1f_43ff_u32.to_le_bytes()));
    assert!(memory.initialize_ram(CODE_PAGE, 4, &0x17ff_ffff_u32.to_le_bytes()));
    assert!(memory.map_page(SPACE, CODE, CODE_PAGE, MemoryPermissions::READ_EXECUTE));

    let mut initial = A64State::default();
    initial.set_pc(CODE.get());
    assert!(initial.set_vector(31, 0xfedc_ba98_7654_3210_0123_4567_89ab_cdef));
    let mut interpreter_state = ThreadCpuState::A64(Box::new(initial.clone()));
    let mut jit_state = ThreadCpuState::A64(Box::new(initial));
    let mut interpreter = EngineHarness::new(&InterpreterProvider, 0x50, 0x50);
    let mut jit = EngineHarness::new(&JitProvider::new(), 0x51, 0x51);

    let interpreter_report = interpreter
        .run_with_budget(&memory, &mut interpreter_state, 12)
        .unwrap();
    let jit_report = jit.run_with_budget(&memory, &mut jit_state, 12).unwrap();

    assert_eq!(jit_report, interpreter_report);
    assert_eq!(jit_state, interpreter_state);
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn every_supported_registry_fixture_matches_the_reference_interpreter() {
    for execution_state in [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ] {
        run_registry_state(execution_state);
    }
}

fn run_registry_state(execution_state: ExecutionState) {
    let interpreter = InterpreterProvider;
    let jit = JitProvider::new();
    let identity = state_index(execution_state) as u64;
    let mut interpreter = EngineHarness::new(&interpreter, identity * 2 + 1, identity * 2 + 1);
    let mut jit = EngineHarness::new(&jit, identity * 2 + 2, identity * 2 + 2);
    let profile = GuestCpuProfile::switch_1();
    let mut executed = 0_usize;

    for pattern in patterns().filter(|pattern| {
        pattern.execution_state == execution_state && pattern.decoder == DecodeSupport::Ready
    }) {
        if !profile
            .allowed_execution_states()
            .contains(pattern.execution_state)
            || pattern.required_features.iter().any(|feature| {
                profile.instruction_features().status(*feature) != CapabilityStatus::Enabled
            })
        {
            continue;
        }
        let fixture = pattern.regression_fixture.unwrap_or_else(|| {
            panic!(
                "{} {} has no differential fixture",
                pattern.execution_state, pattern.coverage_id
            )
        });
        let location = LocationDescriptor::new(CODE, pattern.execution_state, profile.id());
        match decode::decode(&profile, location, fixture.encoding) {
            DecodeResult::Decoded(decoded) => {
                assert_eq!(decoded.instruction.coverage_id(), pattern.coverage_id)
            }
            other => panic!(
                "{} {} differential fixture did not decode as supported: {other:?}",
                pattern.execution_state, pattern.coverage_id
            ),
        }

        let code_page = GuestPhysicalPageId::new(0x10_0000 + u64::from(pattern.coverage_id.get()));
        let interpreter_memory = fixture_memory(pattern, code_page);
        let jit_memory = fixture_memory(pattern, code_page);
        let initial = initial_state(pattern.execution_state, pattern.coverage_id.get());
        let mut interpreter_state = initial.clone();
        let mut jit_state = initial;
        let identity = format!(
            "{} {} ({})",
            pattern.execution_state, pattern.coverage_id, pattern.name
        );
        let interpreter_report = interpreter
            .run(&interpreter_memory, &mut interpreter_state)
            .unwrap_or_else(|error| panic!("{identity} interpreter failure: {error}"));
        let jit_report = jit
            .run(&jit_memory, &mut jit_state)
            .unwrap_or_else(|error| panic!("{identity} JIT failure: {error}"));

        assert_eq!(jit_report.stop, interpreter_report.stop, "{identity} exit");
        assert_eq!(
            jit_report.instructions_executed, interpreter_report.instructions_executed,
            "{identity} retired instruction count"
        );
        assert_eq!(
            jit_report.context, interpreter_report.context,
            "{identity} context"
        );
        assert_eq!(jit_state, interpreter_state, "{identity} canonical state");
        assert_eq!(
            snapshot_data(&jit_memory),
            snapshot_data(&interpreter_memory),
            "{identity} memory"
        );
        assert!(
            !matches!(
                jit_report.stop,
                nixe_cpu_engine::EngineExit::InterpretOne { .. }
                    | nixe_cpu_engine::EngineExit::UnsupportedSemantics { .. }
            ),
            "{identity} used semantic fallback"
        );
        executed += 1;
    }

    assert_ne!(executed, 0, "{execution_state} registry was not exercised");
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn ldpsw_offset_uses_the_displaced_address_in_compiled_code() {
    const CODE_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(0x30_0000);
    const DATA_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(0x30_0001);
    const FIRST: u32 = 0x0058_4af0;
    const SECOND: u32 = 0xff59_8a70;

    let memory = {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(CODE_PAGE));
        assert!(memory.add_ram_page(DATA_PAGE));
        assert!(memory.initialize_ram(CODE_PAGE, 0, &0x6941_0820_u32.to_le_bytes()));
        assert!(memory.initialize_ram(CODE_PAGE, 4, &0x1400_0000_u32.to_le_bytes()));
        assert!(memory.initialize_ram(DATA_PAGE, 0, b"MOD0\0\0\0\0"));
        assert!(memory.initialize_ram(DATA_PAGE, 8, &FIRST.to_le_bytes()));
        assert!(memory.initialize_ram(DATA_PAGE, 12, &SECOND.to_le_bytes()));
        assert!(memory.map_page(SPACE, CODE, CODE_PAGE, MemoryPermissions::READ_EXECUTE));
        assert!(memory.map_page(SPACE, DATA, DATA_PAGE, MemoryPermissions::READ_WRITE));
        memory
    };
    let initial = {
        let mut state = A64State::default();
        state.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            DATA.get(),
        );
        state.set_pc(CODE.get());
        ThreadCpuState::A64(Box::new(state))
    };
    let mut interpreter_state = initial.clone();
    let mut jit_state = initial;
    let mut interpreter = EngineHarness::new(&InterpreterProvider, 0x40, 0x40);
    let mut jit = EngineHarness::new(&JitProvider::new(), 0x41, 0x41);

    let interpreter_report = interpreter.run(&memory, &mut interpreter_state).unwrap();
    let jit_report = jit.run(&memory, &mut jit_state).unwrap();

    assert_eq!(jit_report, interpreter_report);
    assert_eq!(jit_state, interpreter_state);
    let ThreadCpuState::A64(state) = jit_state else {
        unreachable!();
    };
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
        u64::from(FIRST)
    );
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(2).unwrap())),
        SECOND as i32 as i64 as u64
    );
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(1).unwrap())),
        DATA.get()
    );
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn a64_lazy_arithmetic_flags_match_reference_edge_cases() {
    // ADDS/SUBS/ADCS/SBCS in W and X forms, including zero, unsigned
    // carry/borrow, signed overflow, and both incoming carry values.
    let cases = [
        (0x2b01_0002, 0_u64, 0_u64, false),
        (0x2b01_0002, u32::MAX as u64, 1, false),
        (0x2b01_0002, i32::MAX as u64, 1, false),
        (0x6b01_0002, 0, 1, false),
        (0x6b01_0002, i32::MIN as u32 as u64, 1, false),
        (0xab01_0002, 0, 0, false),
        (0xab01_0002, u64::MAX, 1, false),
        (0xab01_0002, i64::MAX as u64, 1, false),
        (0xeb01_0002, 0, 1, false),
        (0xeb01_0002, i64::MIN as u64, 1, false),
        (0x3a01_0002, u32::MAX as u64, 0, false),
        (0x3a01_0002, u32::MAX as u64, 0, true),
        (0x7a01_0002, 0, 0, false),
        (0x7a01_0002, 0, 0, true),
        (0xba01_0002, u64::MAX, 0, false),
        (0xba01_0002, u64::MAX, 0, true),
        (0xfa01_0002, 0, 0, false),
        (0xfa01_0002, 0, 0, true),
    ];
    let mut interpreter = EngineHarness::new(&InterpreterProvider, 0x60, 0x60);
    let mut jit = EngineHarness::new(&JitProvider::new(), 0x61, 0x61);

    for (index, (encoding, lhs, rhs, carry)) in cases.into_iter().enumerate() {
        let code_page = GuestPhysicalPageId::new(0x31_0000 + index as u64);
        let memory = raw_a64_memory(encoding, code_page);
        let mut initial = A64State::default();
        initial.set_pc(CODE.get());
        initial.write_x(
            A64Register::General(A64GeneralRegister::new(0).unwrap()),
            lhs,
        );
        initial.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            rhs,
        );
        initial.set_nzcv(Nzcv::from_bits(if carry { Nzcv::C } else { 0 }));
        let mut interpreter_state = ThreadCpuState::A64(Box::new(initial.clone()));
        let mut jit_state = ThreadCpuState::A64(Box::new(initial));

        let interpreter_report = interpreter.run(&memory, &mut interpreter_state).unwrap();
        let jit_report = jit.run(&memory, &mut jit_state).unwrap();
        assert_eq!(jit_report, interpreter_report, "encoding {encoding:#010x}");
        assert_eq!(jit_state, interpreter_state, "encoding {encoding:#010x}");
    }
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn a64_typed_scalar_ir_matches_reference_edge_cases() {
    // Representative encodings cover extended/shifted operands, every bitfield
    // canonical form, EXTR, Arm division policy, variable shifts, transformed
    // selects, multiply families, unary bit operations, TBZ/TBNZ, MOVK, ADR,
    // and ADRP. See Arm ARM DDI 0602, Base Instructions.
    let instructions = [
        0x8b25_c883, // add x3,x4,w5,sxtw #2
        0x8b05_1c83, // add x3,x4,x5,lsl #7
        0x8ae5_2483, // bic x3,x4,x5,ror #9
        0x9347_5c83, // sbfx x3,x4,#7,#17
        0x9377_1c83, // sbfiz x3,x4,#9,#8
        0xb347_5c83, // bfxil x3,x4,#7,#17
        0xb377_1c83, // bfi x3,x4,#9,#8
        0xd347_5c83, // ubfx x3,x4,#7,#17
        0xd377_1c83, // ubfiz x3,x4,#9,#8
        0x93c5_4483, // extr x3,x4,x5,#17
        0x9ac5_0883, // udiv x3,x4,x5
        0x9ac5_0c83, // sdiv x3,x4,x5
        0x9ac5_2083, // lslv x3,x4,x5
        0x9ac5_2483, // lsrv x3,x4,x5
        0x9ac5_2883, // asrv x3,x4,x5
        0x9ac5_2c83, // rorv x3,x4,x5
        0x9a85_0083, // csel x3,x4,x5,eq
        0x9a85_0483, // csinc x3,x4,x5,eq
        0xda85_0083, // csinv x3,x4,x5,eq
        0xda85_0483, // csneg x3,x4,x5,eq
        0x9b05_1883, // madd x3,x4,x5,x6
        0x9b05_9883, // msub x3,x4,x5,x6
        0x9b25_1883, // smaddl x3,w4,w5,x6
        0x9b25_9883, // smsubl x3,w4,w5,x6
        0x9ba5_1883, // umaddl x3,w4,w5,x6
        0x9ba5_9883, // umsubl x3,w4,w5,x6
        0x9b45_7c83, // smulh x3,x4,x5
        0x9bc5_7c83, // umulh x3,x4,x5
        0xdac0_0083, // rbit x3,x4
        0xdac0_0483, // rev16 x3,x4
        0xdac0_0883, // rev32 x3,x4
        0xdac0_0c83, // rev x3,x4
        0xdac0_1083, // clz x3,x4
        0xdac0_1483, // cls x3,x4
        0xb6f8_0004, // tbz x4,#63,.
        0x3738_0004, // tbnz w4,#7,.
        0xf2d5_79a3, // movk x3,#0xabcd,lsl #32
        0x1000_0003, // adr x3,.
        0x9000_0003, // adrp x3,.
        0x93c4_4483, // ror x3,x4,#17 (EXTR alias)
        0x93c5_0083, // extr x3,x4,x5,#0
        0x1ac5_0883, // udiv w3,w4,w5
        0x1ac5_0c83, // sdiv w3,w4,w5
        0x1ac5_2083, // lslv w3,w4,w5
        0x1ac5_2483, // lsrv w3,w4,w5
        0x1ac5_2883, // asrv w3,w4,w5
        0x1ac5_2c83, // rorv w3,w4,w5
        0x1b05_1883, // madd w3,w4,w5,w6
        0x1b05_9883, // msub w3,w4,w5,w6
        0x5ac0_0083, // rbit w3,w4
        0x5ac0_0483, // rev16 w3,w4
        0x5ac0_0883, // rev w3,w4
        0x5ac0_1083, // clz w3,w4
        0x5ac0_1483, // cls w3,w4
    ];
    let operands = [
        (0_u64, 0_u64, 0_u64, 0_u32),
        (u64::MAX, 0, 1, Nzcv::Z),
        (i64::MIN as u64, u64::MAX, 0x0123_4567_89ab_cdef, 0),
        (0x0123_4567_89ab_cdef, 65, 0xfedc_ba98_7654_3210, Nzcv::Z),
    ];
    let mut interpreter = EngineHarness::new(&InterpreterProvider, 0x70, 0x70);
    let mut jit = EngineHarness::new(&JitProvider::new(), 0x71, 0x71);

    for (instruction_index, encoding) in instructions.into_iter().enumerate() {
        for (operand_index, (lhs, rhs, addend, flags)) in operands.into_iter().enumerate() {
            let code_page = GuestPhysicalPageId::new(
                0x32_0000 + (instruction_index * operands.len() + operand_index) as u64,
            );
            let memory = raw_a64_memory(encoding, code_page);
            let mut initial = A64State::default();
            initial.set_pc(CODE.get());
            for (index, value) in [(3, 0xa5a5_5a5a_a5a5_5a5a), (4, lhs), (5, rhs), (6, addend)] {
                initial.write_x(
                    A64Register::General(A64GeneralRegister::new(index).unwrap()),
                    value,
                );
            }
            initial.set_nzcv(Nzcv::from_bits(flags));
            let mut interpreter_state = ThreadCpuState::A64(Box::new(initial.clone()));
            let mut jit_state = ThreadCpuState::A64(Box::new(initial));

            let interpreter_report = interpreter.run(&memory, &mut interpreter_state).unwrap();
            let jit_report = jit.run(&memory, &mut jit_state).unwrap();
            assert_eq!(
                jit_report, interpreter_report,
                "encoding {encoding:#010x}, operand set {operand_index}"
            );
            assert_eq!(
                jit_state, interpreter_state,
                "encoding {encoding:#010x}, operand set {operand_index}"
            );
        }
    }
}

fn raw_a64_memory(encoding: u32, code_page: GuestPhysicalPageId) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &encoding.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &0x1400_0000_u32.to_le_bytes()));
    assert!(memory.map_page(SPACE, CODE, code_page, MemoryPermissions::READ_EXECUTE));
    memory
}

fn patterns() -> impl Iterator<Item = &'static InstructionPattern> {
    decode::a64::patterns()
        .iter()
        .chain(decode::a32::patterns())
        .chain(decode::t32::patterns_16())
        .chain(decode::t32::patterns_32())
}

fn cpu() -> ProcessCpuContext {
    ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE)
}

fn fixture_memory(pattern: &InstructionPattern, code_page: GuestPhysicalPageId) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    let first_data_page = GuestPhysicalPageId::new(0x20_0000);
    let second_data_page = GuestPhysicalPageId::new(0x20_0001);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(first_data_page));
    assert!(memory.add_ram_page(second_data_page));

    let mut code = instruction_bytes(pattern);
    code.extend_from_slice(&terminator_bytes(pattern.execution_state));
    assert!(memory.initialize_ram(code_page, 0, &code));
    for (page_index, page) in [first_data_page, second_data_page].into_iter().enumerate() {
        let bytes: Vec<_> = (0..SYNTHETIC_PAGE_SIZE)
            .map(|offset| {
                (offset as u8)
                    .wrapping_mul(37)
                    .wrapping_add(page_index as u8 * 53)
            })
            .collect();
        assert!(memory.initialize_ram(page, 0, &bytes));
    }
    assert!(memory.map_page(SPACE, CODE, code_page, MemoryPermissions::READ_EXECUTE,));
    assert!(memory.map_page(SPACE, DATA, first_data_page, MemoryPermissions::READ_WRITE,));
    assert!(memory.map_page(
        SPACE,
        DATA.checked_add(SYNTHETIC_PAGE_SIZE as u64).unwrap(),
        second_data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
}

fn instruction_bytes(pattern: &InstructionPattern) -> Vec<u8> {
    let encoding = pattern
        .regression_fixture
        .expect("supported patterns have differential fixtures")
        .encoding;
    match (pattern.execution_state, encoding.size()) {
        (ExecutionState::T32, InstructionSize::Bits32) => {
            let bits = encoding.bits();
            [(bits >> 16) as u16, bits as u16]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect()
        }
        (_, InstructionSize::Bits16) => encoding.bits().to_le_bytes()[..2].to_vec(),
        (_, InstructionSize::Bits32) => encoding.bits().to_le_bytes().to_vec(),
    }
}

fn terminator_bytes(state: ExecutionState) -> Vec<u8> {
    match state {
        ExecutionState::A64 => 0x1400_0000_u32.to_le_bytes().to_vec(),
        ExecutionState::A32 => 0xeaff_fffe_u32.to_le_bytes().to_vec(),
        ExecutionState::T32 => 0xe7fe_u16.to_le_bytes().to_vec(),
    }
}

fn initial_state(execution_state: ExecutionState, seed: u32) -> ThreadCpuState {
    match execution_state {
        ExecutionState::A64 => {
            let mut state = A64State::default();
            for index in 0..A64_REGISTER_COUNT as u8 {
                let register = A64Register::General(A64GeneralRegister::new(index).unwrap());
                state.write_x(register, mixed(seed, index));
            }
            state.write_x(
                A64Register::General(A64GeneralRegister::new(0).unwrap()),
                DATA.get() + 0x800,
            );
            state.write_x(A64Register::General(A64GeneralRegister::new(1).unwrap()), 0);
            state.write_x(A64Register::StackPointer, DATA.get() + 0x900);
            state.set_pc(CODE.get());
            state.set_nzcv(Nzcv::from_bits(Nzcv::V));
            for index in 0..VECTOR_REGISTER_COUNT as u8 {
                let low = mixed(seed ^ 0xa5a5_5a5a, index);
                let high = mixed(seed ^ 0x5a5a_a5a5, index);
                assert!(state.set_vector(index, u128::from(low) | (u128::from(high) << 64)));
            }
            state.set_fpcr(0);
            state.set_fpsr(0x0800_0000);
            state.set_tpidr_el0(mixed(seed, 61));
            state.set_tpidrro_el0_from_runtime(mixed(seed, 62));
            ThreadCpuState::A64(Box::new(state))
        }
        ExecutionState::A32 | ExecutionState::T32 => {
            let mut state = if execution_state == ExecutionState::A32 {
                A32State::a32()
            } else {
                A32State::t32()
            };
            for index in 0..A32_REGISTER_COUNT as u8 {
                state.write_r(
                    A32GeneralRegister::new(index).unwrap(),
                    mixed(seed, index) as u32,
                );
            }
            state.write_r(
                A32GeneralRegister::new(0).unwrap(),
                (DATA.get() + 0x800) as u32,
            );
            state.write_r(A32GeneralRegister::new(1).unwrap(), 0);
            state.write_r(
                A32GeneralRegister::new(13).unwrap(),
                (DATA.get() + 0x900) as u32,
            );
            state.set_instruction_address(CODE.get() as u32).unwrap();
            let execution_bit = if execution_state == ExecutionState::T32 {
                Cpsr::T
            } else {
                0
            };
            state.set_cpsr(Cpsr::from_bits(
                Cpsr::USER_MODE | Cpsr::Z | Cpsr::C | execution_bit,
            ));
            for index in 0..32_u8 {
                assert!(state.write_d(index, mixed(seed ^ 0x1357_9bdf, index)));
            }
            state.set_fpscr(0x0800_0000);
            state.set_tpidrurw(mixed(seed, 61) as u32);
            state.set_tpidruro_from_runtime(mixed(seed, 62) as u32);
            ThreadCpuState::A32(Box::new(state))
        }
    }
}

fn mixed(seed: u32, index: u8) -> u64 {
    let mut value = u64::from(seed) ^ (u64::from(index) << 32) ^ 0x9e37_79b9_7f4a_7c15;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value.wrapping_mul(0x94d0_49bb_1331_11eb) ^ (value >> 31)
}

fn snapshot_data(memory: &SyntheticMemory) -> Vec<u8> {
    let mut snapshot = Vec::with_capacity(SYNTHETIC_PAGE_SIZE * 2);
    for offset in (0..SYNTHETIC_PAGE_SIZE * 2).step_by(16) {
        snapshot.extend_from_slice(
            &memory
                .read(
                    SPACE,
                    DATA.checked_add(offset as u64).unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Quadword),
                )
                .unwrap()
                .value
                .bits()
                .to_le_bytes(),
        );
    }
    snapshot
}

const fn state_index(state: ExecutionState) -> usize {
    match state {
        ExecutionState::A64 => 0,
        ExecutionState::A32 => 1,
        ExecutionState::T32 => 2,
    }
}
