mod support;

use std::num::NonZeroU32;

use nixe_cpu::{
    address::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress},
    interpreter::{InterpreterContext, InterpreterOutcome, execute_one_with_context},
    ir::terminator::ExceptionKind,
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue,
        SyntheticMemory,
    },
    profile::{GuestCpuProfile, ProcessCpuContext},
    state::{
        A32State, A64State, ThreadCpuState,
        a32::{A32GeneralRegister, Cpsr},
        a64::{A64GeneralRegister, A64Register, Nzcv},
    },
    translate::{BlockTranslationConfig, translate_block},
};
use support::ir_evaluator::{IrReferenceEvaluator, ReferenceOutcome};

const ADDRESS_SPACE: AddressSpaceId = AddressSpaceId::new(0x4449_4646);
const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x4000;
const DATA_BYTES: usize = 128;
const RANDOM_SEEDS: u64 = 48;
const RANDOM_SEQUENCE_LENGTH: usize = 24;

#[derive(Clone, Debug, Eq, PartialEq)]
enum NormalizedOutcome {
    Resume(LocationDescriptor),
    Exception {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    DataAbort {
        source: LocationDescriptor,
        fault: DataAccessFault,
    },
}

#[derive(Clone)]
struct DifferentialCase {
    name: String,
    profile: GuestCpuProfile,
    initial_state: ThreadCpuState,
    instructions: Vec<InstructionEncoding>,
    initial_data: [u8; DATA_BYTES],
}

#[test]
fn directed_sequences_compare_state_vectors_memory_pc_flags_and_exceptions() {
    for case in directed_cases() {
        run_case(&case);
    }
}

#[test]
fn bounded_generated_a64_sequences_match_the_reference_evaluator() {
    for seed in 0..RANDOM_SEEDS {
        let case = generated_a64_case(0x5eed_0000_0000_0000 | seed);
        run_case(&case);
    }
}

fn run_case(case: &DifferentialCase) {
    let program = encode_program(case.initial_state.execution_state(), &case.instructions);
    let interpreter_memory = fixture_memory(&program, &case.initial_data);
    let reference_memory = fixture_memory(&program, &case.initial_data);
    let process = ProcessCpuContext::new(case.profile, ADDRESS_SPACE);
    let interpreter_context = InterpreterContext::new(process).with_memory(&interpreter_memory);
    let mut interpreter_state = case.initial_state.clone();
    let mut reference_state = case.initial_state.clone();
    let mut evaluator = IrReferenceEvaluator::new(ADDRESS_SPACE, &reference_memory);

    for (index, encoding) in case.instructions.iter().copied().enumerate() {
        let start = location(case.profile, &reference_state);
        let interpreter =
            execute_one_with_context(interpreter_context, &mut interpreter_state, encoding)
                .unwrap_or_else(|error| {
                    panic!(
                        "{} instruction {index} interpreter failure: {error}",
                        case.name
                    )
                });
        let block = translate_block(
            BlockTranslationConfig {
                max_guest_instructions: NonZeroU32::new(1).unwrap(),
            },
            &case.profile,
            ADDRESS_SPACE,
            start,
            &reference_memory,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{} instruction {index} translation failure: {error}",
                case.name
            )
        });
        let reference = evaluator
            .execute(&mut reference_state, &block)
            .unwrap_or_else(|error| {
                panic!(
                    "{} instruction {index} IR evaluation failure: {error}",
                    case.name
                )
            });

        assert_eq!(
            normalize_interpreter(&interpreter),
            normalize_reference(&reference),
            "{} instruction {index} outcome mismatch",
            case.name
        );
        assert_eq!(
            interpreter_state, reference_state,
            "{} instruction {index} architectural state mismatch",
            case.name
        );
        assert_eq!(
            snapshot_memory(&interpreter_memory),
            snapshot_memory(&reference_memory),
            "{} instruction {index} memory mismatch",
            case.name
        );

        if !matches!(interpreter, InterpreterOutcome::Resume(_)) {
            break;
        }
    }
}

fn directed_cases() -> Vec<DifferentialCase> {
    let profile = GuestCpuProfile::switch_1();
    let data = initial_data(0x1234_5678_9abc_def0);

    let mut a64 = initial_a64_state(0x100);
    let ThreadCpuState::A64(state) = &mut a64 else {
        unreachable!()
    };
    state.write_x(x(0), u64::MAX);
    state.write_x(x(1), 1);
    state.write_x(x(20), DATA_BASE);

    let mut faulting_a64 = initial_a64_state(0x101);
    let ThreadCpuState::A64(state) = &mut faulting_a64 else {
        unreachable!()
    };
    state.write_x(x(0), 0xfeed_face_cafe_beef);
    state.write_x(x(20), 0x9000);

    vec![
        DifferentialCase {
            name: "a64-scalar-flags-memory".into(),
            profile,
            initial_state: a64,
            instructions: vec![
                0xab01_0002_u32.into(), // ADDS X2,X0,X1
                0xf900_0682_u32.into(), // STR X2,[X20,#8]
                0xf940_0683_u32.into(), // LDR X3,[X20,#8]
                0xca03_0044_u32.into(), // EOR X4,X2,X3
                0xd400_00e1_u32.into(), // SVC #7
            ],
            initial_data: data,
        },
        DifferentialCase {
            name: "a64-precise-data-abort".into(),
            profile,
            initial_state: faulting_a64,
            instructions: vec![0xf900_0280_u32.into()], // STR X0,[X20]
            initial_data: data,
        },
        DifferentialCase {
            name: "a64-direct-control".into(),
            profile,
            initial_state: initial_a64_state(0x102),
            instructions: vec![0x1400_0002_u32.into()], // B +8
            initial_data: data,
        },
        DifferentialCase {
            name: "a32-nop".into(),
            profile,
            initial_state: initial_a32_state(0x200, ExecutionState::A32),
            instructions: vec![0xe320_f000_u32.into()],
            initial_data: data,
        },
        DifferentialCase {
            name: "a32-direct-control".into(),
            profile,
            initial_state: initial_a32_state(0x201, ExecutionState::A32),
            instructions: vec![0xea00_0000_u32.into()], // B +8 (architectural PC bias)
            initial_data: data,
        },
        DifferentialCase {
            name: "t32-it-and-nop".into(),
            profile,
            initial_state: initial_a32_state(0x300, ExecutionState::T32),
            instructions: vec![
                InstructionEncoding::from_u16(0xbf18), // IT NE
                InstructionEncoding::from_u16(0xbf00), // NOP
            ],
            initial_data: data,
        },
        DifferentialCase {
            name: "t32-direct-control".into(),
            profile,
            initial_state: initial_a32_state(0x301, ExecutionState::T32),
            instructions: vec![InstructionEncoding::from_u16(0xe000)], // B +4
            initial_data: data,
        },
    ]
}

fn generated_a64_case(seed: u64) -> DifferentialCase {
    let mut random = DeterministicRandom::new(seed);
    let mut instructions = Vec::with_capacity(RANDOM_SEQUENCE_LENGTH);
    for _ in 0..RANDOM_SEQUENCE_LENGTH {
        let rd = random.register();
        let rn = random.register();
        let rm = random.register();
        let immediate = random.next_u32() & 0xfff;
        let encoding = match random.next_u32() % 10 {
            0 => 0xd280_0000 | ((random.next_u32() & 0xffff) << 5) | u32::from(rd),
            1 => 0x9100_0000 | (immediate << 10) | (u32::from(rn) << 5) | u32::from(rd),
            2 => 0xd100_0000 | (immediate << 10) | (u32::from(rn) << 5) | u32::from(rd),
            3 => 0xb100_0000 | (immediate << 10) | (u32::from(rn) << 5) | u32::from(rd),
            4 => 0xf100_0000 | (immediate << 10) | (u32::from(rn) << 5) | u32::from(rd),
            5 => 0x8a00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd),
            6 => 0xaa00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd),
            7 => 0xca00_0000 | (u32::from(rm) << 16) | (u32::from(rn) << 5) | u32::from(rd),
            8 => {
                let offset = random.next_u32() % 16;
                0xf900_0000 | (offset << 10) | (20 << 5) | u32::from(rd)
            }
            _ => {
                let offset = random.next_u32() % 16;
                0xf940_0000 | (offset << 10) | (20 << 5) | u32::from(rd)
            }
        };
        instructions.push(encoding.into());
    }
    DifferentialCase {
        name: format!("generated-a64-seed-{seed:016x}"),
        profile: GuestCpuProfile::switch_1(),
        initial_state: initial_a64_state(seed),
        instructions,
        initial_data: initial_data(seed ^ 0xd1ff_e2e3_a4a5_b6b7),
    }
}

fn initial_a64_state(seed: u64) -> ThreadCpuState {
    let mut random = DeterministicRandom::new(seed);
    let mut state = A64State::default();
    for index in 0..31 {
        state.write_x(x(index), random.next_u64());
    }
    state.write_x(A64Register::StackPointer, DATA_BASE + 0x80);
    state.write_x(x(20), DATA_BASE);
    state.set_pc(CODE_BASE);
    state.set_nzcv(Nzcv::from_bits((random.next_u32() & 0xf) << 28));
    for index in 0..32 {
        let value = u128::from(random.next_u64()) | (u128::from(random.next_u64()) << 64);
        assert!(state.set_vector(index, value));
    }
    state.set_fpcr(random.next_u32());
    state.set_fpsr(random.next_u32());
    state.set_tpidr_el0(random.next_u64());
    state.set_tpidrro_el0_from_runtime(random.next_u64());
    ThreadCpuState::A64(Box::new(state))
}

fn initial_a32_state(seed: u64, execution_state: ExecutionState) -> ThreadCpuState {
    let mut random = DeterministicRandom::new(seed);
    let mut state = if execution_state == ExecutionState::T32 {
        A32State::t32()
    } else {
        A32State::a32()
    };
    for index in 0..15 {
        state.write_r(r(index), random.next_u32());
    }
    for index in 0..32 {
        assert!(state.write_d(index, random.next_u64()));
    }
    let state_bit = if execution_state == ExecutionState::T32 {
        Cpsr::T
    } else {
        0
    };
    state.set_cpsr(Cpsr::from_bits(
        Cpsr::USER_MODE | state_bit | ((random.next_u32() & 0xf) << 28),
    ));
    state.set_fpscr(random.next_u32());
    state.set_tpidrurw(random.next_u32());
    state.set_tpidruro_from_runtime(random.next_u32());
    state.set_instruction_address(CODE_BASE as u32).unwrap();
    ThreadCpuState::A32(Box::new(state))
}

fn fixture_memory(program: &[u8], data: &[u8; DATA_BYTES]) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    assert!(memory.initialize_ram(code_page, 0, program));
    assert!(memory.initialize_ram(data_page, 0, data));
    assert!(memory.map_page(
        ADDRESS_SPACE,
        GuestVirtualAddress::new(CODE_BASE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        ADDRESS_SPACE,
        GuestVirtualAddress::new(DATA_BASE),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
}

fn encode_program(state: ExecutionState, instructions: &[InstructionEncoding]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for encoding in instructions {
        match (state, encoding.size().bytes()) {
            (ExecutionState::T32, 2) => {
                bytes.extend_from_slice(&(encoding.bits() as u16).to_le_bytes());
            }
            (ExecutionState::T32, 4) => {
                bytes.extend_from_slice(&((encoding.bits() >> 16) as u16).to_le_bytes());
                bytes.extend_from_slice(&(encoding.bits() as u16).to_le_bytes());
            }
            (_, 4) => bytes.extend_from_slice(&encoding.bits().to_le_bytes()),
            _ => panic!("invalid encoding width for {state}"),
        }
    }
    bytes
}

fn snapshot_memory(memory: &SyntheticMemory) -> [u8; DATA_BYTES] {
    let mut bytes = [0; DATA_BYTES];
    for (offset, byte) in bytes.iter_mut().enumerate() {
        let value = memory
            .read(
                ADDRESS_SPACE,
                GuestVirtualAddress::new(DATA_BASE + offset as u64),
                MemoryAccess::normal(MemoryAccessSize::Byte),
            )
            .unwrap()
            .value;
        let MemoryValue::U8(value) = value else {
            unreachable!()
        };
        *byte = value;
    }
    bytes
}

fn normalize_interpreter(outcome: &InterpreterOutcome) -> NormalizedOutcome {
    match outcome {
        InterpreterOutcome::Resume(location) => NormalizedOutcome::Resume(*location),
        InterpreterOutcome::Exception {
            source,
            kind,
            syndrome,
        } => NormalizedOutcome::Exception {
            source: *source,
            kind: *kind,
            syndrome: *syndrome,
        },
        InterpreterOutcome::DataAbort { source, fault } => NormalizedOutcome::DataAbort {
            source: *source,
            fault: fault.clone(),
        },
        InterpreterOutcome::Scheduled { .. } => panic!("scheduled instruction is outside MVP"),
        InterpreterOutcome::ProfileDisabled(_) | InterpreterOutcome::Unallocated(_) => {
            panic!("decode rejection is outside generated MVP")
        }
    }
}

fn normalize_reference(outcome: &ReferenceOutcome) -> NormalizedOutcome {
    match outcome {
        ReferenceOutcome::Resume(location) => NormalizedOutcome::Resume(*location),
        ReferenceOutcome::Exception {
            source,
            kind,
            syndrome,
        } => NormalizedOutcome::Exception {
            source: *source,
            kind: *kind,
            syndrome: *syndrome,
        },
        ReferenceOutcome::DataAbort { source, fault } => NormalizedOutcome::DataAbort {
            source: *source,
            fault: fault.clone(),
        },
    }
}

fn location(profile: GuestCpuProfile, state: &ThreadCpuState) -> LocationDescriptor {
    match state {
        ThreadCpuState::A64(state) => LocationDescriptor::new(
            GuestVirtualAddress::new(state.pc()),
            ExecutionState::A64,
            profile.id(),
        ),
        ThreadCpuState::A32(state) => LocationDescriptor::new(
            GuestVirtualAddress::new(u64::from(state.instruction_address())),
            state.execution_state(),
            profile.id(),
        ),
    }
}

fn initial_data(seed: u64) -> [u8; DATA_BYTES] {
    let mut random = DeterministicRandom::new(seed);
    let mut data = [0; DATA_BYTES];
    for byte in &mut data {
        *byte = random.next_u32() as u8;
    }
    data
}

fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

fn r(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).unwrap()
}

struct DeterministicRandom(u64);

impl DeterministicRandom {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn register(&mut self) -> u8 {
        (self.next_u32() % 16) as u8
    }
}
