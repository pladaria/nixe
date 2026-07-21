use crate::{
    address::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress},
    coverage::CoverageId,
    ir::terminator::{ExceptionKind, Terminator},
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::{
        CpuMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue, SyntheticMemory,
    },
    profile::{GuestCpuProfile, ProcessCpuContext},
    state::{
        ThreadCpuState,
        a32::A32GeneralRegister,
        a64::{A64GeneralRegister, A64Register},
    },
};

use super::{
    InstructionSupport, InterpreterContext, InterpreterError, InterpreterOutcome,
    InterpreterPolicy, execute_fallback, execute_one, execute_one_with_context,
    instruction_support,
};

fn source(
    profile: GuestCpuProfile,
    pc: u64,
    execution_state: ExecutionState,
) -> LocationDescriptor {
    LocationDescriptor::new(GuestVirtualAddress::new(pc), execution_state, profile.id())
}

#[test]
fn interpreter_only_t32_movs_executes_once_and_resumes_at_next_pc() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A32(Box::new(crate::state::A32State::t32()));
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.set_instruction_address(0x1000).unwrap();
    let terminator = Terminator::InterpretOne {
        source: source(profile, 0x1000, ExecutionState::T32),
        encoding: InstructionEncoding::from_u16(0x237f),
        coverage_id: 0x0002_0003,
    };

    let outcome = execute_fallback(
        InterpreterPolicy::default(),
        &profile,
        &mut state,
        &terminator,
    )
    .unwrap();

    assert_eq!(
        outcome,
        InterpreterOutcome::Resume(source(profile, 0x1002, ExecutionState::T32))
    );
    let ThreadCpuState::A32(a32) = state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(A32GeneralRegister::new(3).unwrap()), 127);
    assert_eq!(a32.instruction_address(), 0x1002);
}

#[test]
fn strict_mode_rejects_fallback_before_mutating_state() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A32(Box::new(crate::state::A32State::t32()));
    let terminator = Terminator::InterpretOne {
        source: source(profile, 0, ExecutionState::T32),
        encoding: InstructionEncoding::from_u16(0x2001),
        coverage_id: 0x0002_0003,
    };

    let error = execute_fallback(
        InterpreterPolicy {
            strict_fallback: true,
        },
        &profile,
        &mut state,
        &terminator,
    )
    .unwrap_err();

    assert!(matches!(error, InterpreterError::StrictFallback { .. }));
    assert_eq!(
        state,
        ThreadCpuState::A32(Box::new(crate::state::A32State::t32()))
    );
}

#[test]
fn unallocated_and_profile_disabled_encodings_take_exception_paths() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let outcome = execute_one(&profile, &mut state, 0_u32.into()).unwrap();

    assert_eq!(
        outcome,
        InterpreterOutcome::Exception {
            source: source(profile, 0, ExecutionState::A64),
            kind: ExceptionKind::UndefinedInstruction,
            syndrome: None,
        }
    );

    // The provisional Switch 2 profile keeps Advanced SIMD unknown, so this
    // recognized vector encoding must not become an implementation fallback.
    let profile = GuestCpuProfile::switch_2_native();
    let mut state = ThreadCpuState::A64(Box::default());
    let outcome = execute_one(&profile, &mut state, 0x4e22_1c20_u32.into()).unwrap();
    assert_eq!(
        outcome,
        InterpreterOutcome::Exception {
            source: source(profile, 0, ExecutionState::A64),
            kind: ExceptionKind::UndefinedInstruction,
            syndrome: None,
        }
    );
}

fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}

#[test]
fn a64_integer_reference_semantics_execute_without_ir() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(1), 10);

    execute_one(&profile, &mut state, 0xd280_0020_u32.into()).unwrap(); // MOVZ X0,#1
    execute_one(&profile, &mut state, 0x8b01_0000_u32.into()).unwrap(); // ADD X0,X0,X1
    execute_one(&profile, &mut state, 0xf100_041f_u32.into()).unwrap(); // CMP X0,#1

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), 11);
    assert!(a64.nzcv().carry());
    assert!(!a64.nzcv().zero());
    assert_eq!(a64.pc(), 12);
}

#[test]
fn every_a64_scalar_integer_family_has_a_reference_handler() {
    let profile = GuestCpuProfile::switch_1();
    let encodings: [u32; 17] = [
        0x9100_0400, // ADD X0,X0,#1
        0xd280_0020, // MOVZ X0,#1
        0x8b01_0000, // ADD X0,X0,X1
        0x8b21_4000, // ADD X0,X0,W1,UXTW
        0x9a01_0000, // ADC X0,X0,X1
        0x9240_0000, // AND X0,X0,#1
        0xaa01_0000, // ORR X0,X0,X1
        0xd340_fc00, // UBFM X0,X0,#0,#63
        0x93c1_0400, // EXTR X0,X0,X1,#1
        0x9ac1_2000, // LSLV X0,X0,X1
        0xfa41_0000, // CCMP X0,X1,#0,EQ
        0xfa41_0800, // CCMP X0,#1,#0,EQ
        0x9a81_0000, // CSEL X0,X0,X1,EQ
        0x9b01_0800, // MADD X0,X0,X1,X2
        0xdac0_1000, // CLZ X0,X0
        0x1000_0000, // ADR X0,#0
        0x9000_0000, // ADRP X0,#0
    ];

    for encoding in encodings {
        let mut state = ThreadCpuState::A64(Box::default());
        let outcome = execute_one(&profile, &mut state, encoding.into())
            .unwrap_or_else(|error| panic!("encoding {encoding:#010x}: {error}"));
        assert!(
            matches!(outcome, InterpreterOutcome::Resume(_)),
            "encoding {encoding:#010x}: {outcome:?}"
        );
    }
}

#[test]
fn a64_system_register_reference_semantics_preserve_thread_state() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_tpidr_el0(0x1234_5678_9abc_def0);

    execute_one(&profile, &mut state, 0xd53b_d043_u32.into()).unwrap(); // MRS X3,TPIDR_EL0

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(3)), 0x1234_5678_9abc_def0);
    assert_eq!(a64.pc(), 4);
}

#[test]
fn a64_basic_system_semantics_are_exact_and_runtime_hints_remain_explicit() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(3), 0xfeed_face_cafe_beef);

    execute_one(&profile, &mut state, 0xd51b_d043_u32.into()).unwrap(); // MSR TPIDR_EL0,X3
    execute_one(&profile, &mut state, 0xd503_3bbf_u32.into()).unwrap(); // DMB ISH
    execute_one(&profile, &mut state, 0xd503_3fdf_u32.into()).unwrap(); // ISB

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.tpidr_el0(), 0xfeed_face_cafe_beef);
    assert_eq!(a64.pc(), 12);

    let error = execute_one(&profile, &mut state, 0xd503_203f_u32.into()).unwrap_err(); // YIELD
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 12, "unsupported scheduler hint must not retire");
}

#[test]
fn a64_memory_reference_semantics_use_process_address_space_and_report_faults() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(44);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(91);
    let profile = GuestCpuProfile::switch_1();
    let process = ProcessCpuContext::new(profile, SPACE);
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x1000),
        PAGE,
        MemoryPermissions::READ_WRITE,
    ));
    let context = InterpreterContext::new(process).with_memory(&memory);
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(0), 0xab);
    a64.write_x(x(1), 0x1008);

    execute_one_with_context(context, &mut state, 0x3900_0020_u32.into()).unwrap(); // STRB W0,[X1]
    execute_one_with_context(context, &mut state, 0x3940_0022_u32.into()).unwrap(); // LDRB W2,[X1]

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(2)), 0xab);
    a64.write_x(x(1), 0x4000);
    let outcome = execute_one_with_context(context, &mut state, 0x3940_0022_u32.into()).unwrap();
    assert!(matches!(outcome, InterpreterOutcome::DataAbort { .. }));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 8, "faulting memory instruction must not retire");
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x1008),
                MemoryAccess::normal(MemoryAccessSize::Byte),
            )
            .unwrap()
            .value,
        MemoryValue::U8(0xab),
    );
}

#[test]
fn every_a64_ordinary_scalar_memory_family_has_a_reference_handler() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(45);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(92);
    let profile = GuestCpuProfile::switch_1();
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x1000),
        PAGE,
        MemoryPermissions::READ_WRITE,
    ));
    let context =
        InterpreterContext::new(ProcessCpuContext::new(profile, SPACE)).with_memory(&memory);
    let encodings: [u32; 9] = [
        0x5800_0000, // LDR X0,literal
        0xf940_0020, // LDR X0,[X1]
        0xf840_1083, // LDUR X3,[X4,#1]
        0xf840_8cc5, // LDR X5,[X6,#8]!
        0xf840_8507, // LDR X7,[X8],#8
        0xf861_6800, // LDR X0,[X0,X1]
        0xa940_0c82, // LDP X2,X3,[X4]
        0xc8df_fc20, // LDAR X0,[X1]
        0xc89f_fc20, // STLR X0,[X1]
    ];

    for encoding in encodings {
        let mut state = ThreadCpuState::A64(Box::default());
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        a64.set_pc(0x1000);
        for register in 0..=8 {
            a64.write_x(x(register), 0x1000);
        }
        if encoding == 0xf861_6800 {
            a64.write_x(x(1), 0);
        }
        let outcome = execute_one_with_context(context, &mut state, encoding.into())
            .unwrap_or_else(|error| panic!("encoding {encoding:#010x}: {error}"));
        assert!(
            matches!(outcome, InterpreterOutcome::Resume(_)),
            "encoding {encoding:#010x}: {outcome:?}"
        );
    }
}

#[test]
fn coverage_distinguishes_lifted_and_interpreter_only_instructions() {
    let profile = GuestCpuProfile::switch_1();
    let decoded = match crate::decode::decode(
        &profile,
        source(profile, 0, ExecutionState::T32),
        InstructionEncoding::from_u16(0x2001),
    ) {
        crate::decode::DecodeResult::Decoded(decoded) => decoded,
        other => panic!("expected decoded MOVS, got {other:?}"),
    };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );
    assert_eq!(
        decoded.instruction.coverage_id(),
        CoverageId::new(0x0002_0003)
    );
}

#[test]
fn a64_control_reference_semantics_update_link_and_pc() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let outcome = execute_one(
        &profile,
        &mut state,
        InstructionEncoding::from_u32(0x9400_0002),
    )
    .unwrap();

    assert_eq!(
        outcome,
        InterpreterOutcome::Resume(source(profile, 8, ExecutionState::A64))
    );
    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(
        a64.read_x(crate::state::a64::A64Register::General(
            crate::state::a64::A64GeneralRegister::new(30).unwrap()
        )),
        4
    );
}
