use crate::{
    address::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress},
    coverage::CoverageId,
    ir::terminator::Terminator,
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
    ArchitecturalTimerSnapshot, InstructionSupport, InterpreterContext, InterpreterError,
    InterpreterOutcome, InterpreterPolicy, execute_fallback, execute_one, execute_one_with_context,
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
fn a32_mvp_executes_predicated_integer_flags_and_interworking() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.set_instruction_address(0x1000).unwrap();

    execute_one(&profile, &mut state, 0xe3a0_0001_u32.into()).unwrap(); // MOV R0,#1
    execute_one(&profile, &mut state, 0xe280_1002_u32.into()).unwrap(); // ADD R1,R0,#2
    execute_one(&profile, &mut state, 0xe351_0003_u32.into()).unwrap(); // CMP R1,#3
    execute_one(&profile, &mut state, 0x13a0_2009_u32.into()).unwrap(); // MOVNE R2,#9 (skipped)

    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(A32GeneralRegister::new(1).unwrap()), 3);
    assert_eq!(a32.read_r(A32GeneralRegister::new(2).unwrap()), 0);
    assert!(a32.cpsr().zero());
    a32.write_r(A32GeneralRegister::new(3).unwrap(), 0x2001);
    execute_one(&profile, &mut state, 0xe12f_ff13_u32.into()).unwrap(); // BX R3
    let ThreadCpuState::A32(a32) = state else {
        unreachable!()
    };
    assert_eq!(a32.execution_state(), ExecutionState::T32);
    assert_eq!(a32.instruction_address(), 0x2000);
}

#[test]
fn a32_and_t32_mvp_memory_families_use_the_shared_process_context() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(47);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(94);
    let profile = GuestCpuProfile::switch_1();
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x1000),
        PAGE,
        MemoryPermissions::READ_WRITE
    ));
    let context =
        InterpreterContext::new(ProcessCpuContext::new(profile, SPACE)).with_memory(&memory);

    let mut a32_state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut a32_state else {
        unreachable!()
    };
    a32.set_instruction_address(0x2000).unwrap();
    a32.write_r(A32GeneralRegister::new(0).unwrap(), 0xfeed_beef);
    a32.write_r(A32GeneralRegister::new(1).unwrap(), 0x1000);
    execute_one_with_context(context, &mut a32_state, 0xe581_0004_u32.into()).unwrap(); // STR R0,[R1,#4]
    execute_one_with_context(context, &mut a32_state, 0xe591_2004_u32.into()).unwrap(); // LDR R2,[R1,#4]
    let ThreadCpuState::A32(a32) = &a32_state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(A32GeneralRegister::new(2).unwrap()), 0xfeed_beef);

    let mut t32_state = ThreadCpuState::A32(Box::new(crate::state::A32State::t32()));
    let ThreadCpuState::A32(t32) = &mut t32_state else {
        unreachable!()
    };
    t32.set_instruction_address(0x3000).unwrap();
    t32.write_r(A32GeneralRegister::new(0).unwrap(), 0x1234_5678);
    t32.write_r(A32GeneralRegister::new(1).unwrap(), 0x1000);
    execute_one_with_context(
        context,
        &mut t32_state,
        InstructionEncoding::from_u16(0x6048),
    )
    .unwrap(); // STR R0,[R1,#4]
    execute_one_with_context(
        context,
        &mut t32_state,
        InstructionEncoding::from_u16(0x684a),
    )
    .unwrap(); // LDR R2,[R1,#4]
    let ThreadCpuState::A32(t32) = t32_state else {
        unreachable!()
    };
    assert_eq!(t32.read_r(A32GeneralRegister::new(2).unwrap()), 0x1234_5678);
}

#[test]
fn t32_mvp_tracks_it_and_executes_wide_branch_link() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A32(Box::new(crate::state::A32State::t32()));
    let ThreadCpuState::A32(t32) = &mut state else {
        unreachable!()
    };
    t32.set_instruction_address(0x1000).unwrap();

    execute_one(&profile, &mut state, InstructionEncoding::from_u16(0x2000)).unwrap(); // MOVS R0,#0 (Z=1)
    execute_one(&profile, &mut state, InstructionEncoding::from_u16(0xbf18)).unwrap(); // IT NE
    execute_one(&profile, &mut state, InstructionEncoding::from_u16(0x2107)).unwrap(); // MOV R1,#7 (skipped)
    let ThreadCpuState::A32(t32) = &state else {
        unreachable!()
    };
    assert_eq!(t32.read_r(A32GeneralRegister::new(1).unwrap()), 0);
    assert!(!t32.cpsr().it_state().is_active());

    execute_one(
        &profile,
        &mut state,
        InstructionEncoding::from_u32(0xf000_f800),
    )
    .unwrap(); // BL +0
    let ThreadCpuState::A32(t32) = state else {
        unreachable!()
    };
    assert_eq!(t32.instruction_address(), 0x100a);
    assert_eq!(t32.read_r(A32GeneralRegister::new(14).unwrap()), 0x100b);
}

#[test]
fn a32_neon_aliases_execute_bitwise_and_lane_integer_operations() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.write_q(0, 0x0102_0304_0506_0708_1112_1314_1516_1718);

    execute_one(&profile, &mut state, 0xf200_0150_u32.into()).unwrap(); // VAND Q0,Q0,Q0
    execute_one(&profile, &mut state, 0xf200_0840_u32.into()).unwrap(); // VADD.I8 Q0,Q0,Q0

    let ThreadCpuState::A32(a32) = state else {
        unreachable!()
    };
    assert_eq!(
        a32.read_q(0).unwrap(),
        0x0204_0608_0a0c_0e10_2224_2628_2a2c_2e30
    );
    assert_eq!(a32.read_d(0).unwrap(), 0x2224_2628_2a2c_2e30);
}

#[test]
fn a32_neon_single_register_memory_transfer_round_trips_d_registers() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(48);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(95);
    let profile = GuestCpuProfile::switch_1();
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x1000),
        PAGE,
        MemoryPermissions::READ_WRITE
    ));
    let context =
        InterpreterContext::new(ProcessCpuContext::new(profile, SPACE)).with_memory(&memory);
    let mut state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.write_r(A32GeneralRegister::new(0).unwrap(), 0x1000);
    a32.write_d(0, 0x0123_4567_89ab_cdef);

    execute_one_with_context(context, &mut state, 0xf400_070f_u32.into()).unwrap(); // VST1.8 {D0},[R0]
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.write_d(0, 0);
    execute_one_with_context(context, &mut state, 0xf420_070f_u32.into()).unwrap(); // VLD1.8 {D0},[R0]

    let ThreadCpuState::A32(a32) = state else {
        unreachable!()
    };
    assert_eq!(a32.read_d(0), Some(0x0123_4567_89ab_cdef));
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
fn unallocated_and_profile_disabled_encodings_keep_distinct_undefined_paths() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let outcome = execute_one(&profile, &mut state, 0_u32.into()).unwrap();

    let InterpreterOutcome::Unallocated(error) = outcome else {
        panic!("unallocated encoding was not classified distinctly");
    };
    assert_eq!(
        error.instruction.location,
        source(profile, 0, ExecutionState::A64)
    );
    assert_eq!(error.instruction.encoding, 0_u32.into());

    // The provisional Switch 2 profile keeps Advanced SIMD unknown, so this
    // recognized vector encoding must not become an implementation fallback.
    let profile = GuestCpuProfile::switch_2_native();
    let mut state = ThreadCpuState::A64(Box::default());
    let outcome = execute_one(&profile, &mut state, 0x4e22_1c20_u32.into()).unwrap();
    let InterpreterOutcome::ProfileDisabled(error) = outcome else {
        panic!("profile-disabled encoding was not classified distinctly");
    };
    assert_eq!(
        error.instruction.location,
        source(profile, 0, ExecutionState::A64)
    );
    assert_eq!(error.instruction.encoding, 0x4e22_1c20_u32.into());
    assert_eq!(
        error.required_feature,
        crate::profile::InstructionFeature::AdvancedSimd
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
fn a64_high_dynamic_tag_comparison_takes_signed_greater_than_branch() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(16), 0x6fff_fff9);
    a64.write_x(x(13), 0x6fff_fff8);

    execute_one(&profile, &mut state, 0xeb0d_021f_u32.into()).unwrap(); // CMP X16,X13
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert!(!a64.nzcv().negative());
    assert!(!a64.nzcv().zero());
    assert!(!a64.nzcv().overflow());

    execute_one(&profile, &mut state, 0x5400_00ec_u32.into()).unwrap(); // B.GT +0x1c
    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 0x20);
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
    execute_one(&profile, &mut state, 0xd53b_00e4_u32.into()).unwrap(); // MRS X4,DCZID_EL0

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(3)), 0x1234_5678_9abc_def0);
    assert_eq!(a64.read_x(x(4)), 0x14, "DC ZVA is prohibited at EL0");
    assert_eq!(a64.pc(), 8);
}

#[test]
fn a64_architectural_timer_registers_use_the_runtime_snapshot() {
    let profile = GuestCpuProfile::switch_1();
    let context = InterpreterContext::new(ProcessCpuContext::new(profile, AddressSpaceId::new(0)))
        .with_architectural_timer(ArchitecturalTimerSnapshot {
            counter: 0x1234_5678_9abc_def0,
            frequency: 19_200_000,
        });
    let mut state = ThreadCpuState::A64(Box::default());

    execute_one_with_context(context, &mut state, 0xd53b_e001_u32.into()).unwrap(); // MRS X1,CNTFRQ_EL0
    execute_one_with_context(context, &mut state, 0xd53b_e022_u32.into()).unwrap(); // MRS X2,CNTVCT_EL0

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(1)), 19_200_000);
    assert_eq!(a64.read_x(x(2)), 0x1234_5678_9abc_def0);
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
    execute_one(&profile, &mut state, 0xd503_245f_u32.into()).unwrap(); // BTI C as HINT

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.tpidr_el0(), 0xfeed_face_cafe_beef);
    assert_eq!(a64.pc(), 16);

    let error = execute_one(&profile, &mut state, 0xd503_203f_u32.into()).unwrap_err(); // YIELD
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 16, "unsupported scheduler hint must not retire");
}

#[test]
fn a64_simd_duplicate_general_replicates_each_allocated_lane_width() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(1), 0x8877_6655_4433_2211);

    for (encoding, expected) in [
        (0x4e01_0c20_u32, 0x1111_1111_1111_1111_1111_1111_1111_1111),
        (0x4e02_0c20, 0x2211_2211_2211_2211_2211_2211_2211_2211),
        (0x4e04_0c20, 0x4433_2211_4433_2211_4433_2211_4433_2211),
        (0x4e08_0c20, 0x8877_6655_4433_2211_8877_6655_4433_2211),
    ] {
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(expected), "encoding={encoding:#010x}");
    }
}

#[test]
fn a64_simd_move_immediate_32_expands_lanes_and_clears_inactive_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::MAX));
    assert!(a64.set_vector(3, u128::MAX));

    execute_one(&profile, &mut state, 0x4f00_041f_u32.into()).unwrap(); // MOVI V31.4S,#0
    execute_one(&profile, &mut state, 0x0f05_4563_u32.into()).unwrap(); // MOVI V3.2S,#0xab,LSL #16

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(0));
    assert_eq!(
        a64.vector(3),
        Some(0x0000_0000_0000_0000_00ab_0000_00ab_0000)
    );
    assert_eq!(a64.pc(), 8);
}

#[test]
fn a64_simd_unsigned_move_extracts_each_lane_width_and_zero_extends() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, 0x8877_6655_4433_2211_fedc_ba98_7654_3210));

    for (encoding, register, expected) in [
        (0x0e01_3c01_u32, 1, 0x10),
        (0x0e1f_3c02, 2, 0x88),
        (0x0e02_3c03, 3, 0x3210),
        (0x0e1e_3c04, 4, 0x8877),
        (0x0e04_3c05, 5, 0x7654_3210),
        (0x0e1c_3c06, 6, 0x8877_6655),
        (0x4e08_3c07, 7, 0xfedc_ba98_7654_3210),
        (0x4e18_3c08, 8, 0x8877_6655_4433_2211),
    ] {
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.read_x(x(register)),
            expected,
            "encoding={encoding:#010x}"
        );
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 32);
}

#[test]
fn a64_simd_integer_add_sub_wrap_each_lane_and_clear_inactive_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let arrangements = [
        (0x0e22_8420_u32, 0x2e22_8420_u32, 8_u8, 64_u8),
        (0x4e22_8420, 0x6e22_8420, 8, 128),
        (0x0e62_8420, 0x2e62_8420, 16, 64),
        (0x4e62_8420, 0x6e62_8420, 16, 128),
        (0x0ea2_8420, 0x2ea2_8420, 32, 64),
        (0x4ea2_8420, 0x6ea2_8420, 32, 128),
        (0x4ee2_8420, 0x6ee2_8420, 64, 128),
    ];

    for (add, subtract, lane_bits, vector_bits) in arrangements {
        let lane_mask = (1_u128 << lane_bits) - 1;
        let lane_count = vector_bits / lane_bits;
        let active_mask = if vector_bits == 128 {
            u128::MAX
        } else {
            u128::from(u64::MAX)
        };
        let ones = (0..lane_count).fold(0_u128, |value, lane| {
            value | (1_u128 << (u32::from(lane) * u32::from(lane_bits)))
        });
        let expected_subtract = (0..lane_count).fold(0_u128, |value, lane| {
            value | ((lane_mask - 1) << (u32::from(lane) * u32::from(lane_bits)))
        });
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, u128::MAX));
        assert!(a64.set_vector(2, ones));

        execute_one(&profile, &mut state, add.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(0), "ADD encoding={add:#010x}");

        execute_one(&profile, &mut state, subtract.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(0),
            Some(expected_subtract & active_mask),
            "SUB encoding={subtract:#010x}"
        );
    }

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, 0x1234_5678_8000_0000_0000_0001_ffff_ffff));
    assert!(a64.set_vector(30, 0xedcb_a988_8000_0000_ffff_ffff_0000_0001));
    execute_one(&profile, &mut state, 0x4ebe_87fe_u32.into()).unwrap();
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(0));
}

#[test]
fn a64_simd_quadword_single_and_pair_memory_transfers_round_trip() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(49);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(96);
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
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let first = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00;
    let second = 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100;
    assert!(a64.set_vector(0, first));
    assert!(a64.set_vector(1, second));
    a64.write_x(x(4), 0x1000);

    execute_one_with_context(context, &mut state, 0x3d80_0080_u32.into()).unwrap(); // STR Q0,[X4]
    execute_one_with_context(context, &mut state, 0x3dc0_0082_u32.into()).unwrap(); // LDR Q2,[X4]
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(2), Some(first));

    execute_one_with_context(context, &mut state, 0xad01_0480_u32.into()).unwrap(); // STP Q0,Q1,[X4,#32]
    execute_one_with_context(context, &mut state, 0xad41_0c82_u32.into()).unwrap(); // LDP Q2,Q3,[X4,#32]
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(2), Some(first));
    assert_eq!(a64.vector(3), Some(second));
}

#[test]
fn a64_simd_pre_and_post_index_transfers_cover_sizes_writeback_and_faults() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(51);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(98);
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
    let mut state = ThreadCpuState::A64(Box::default());
    let value = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ffab_u128;

    // STR Q30,[X1],#16: the exact instruction observed during libnx startup.
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(30, value));
    a64.write_x(x(1), 0x1000);
    execute_one_with_context(context, &mut state, 0x3c81_043e_u32.into()).unwrap();
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(1)), 0x1010);
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                MemoryAccess::normal(MemoryAccessSize::Quadword),
            )
            .unwrap()
            .value,
        MemoryValue::U128(value),
    );

    for (index, (size_bits, access_size)) in [
        (0_u32, MemoryAccessSize::Byte),
        (1_u32 << 30, MemoryAccessSize::Halfword),
        (2_u32 << 30, MemoryAccessSize::Word),
        (3_u32 << 30, MemoryAccessSize::Doubleword),
        (1_u32 << 23, MemoryAccessSize::Quadword),
    ]
    .into_iter()
    .enumerate()
    {
        let base = 0x1100 + index as u64 * 0x40;
        let offset = if index.is_multiple_of(2) { 16_i16 } else { -16 };
        let immediate = u32::from((offset as u16) & 0x01ff) << 12;
        let expected = match access_size {
            MemoryAccessSize::Byte => value & u128::from(u8::MAX),
            MemoryAccessSize::Halfword => value & u128::from(u16::MAX),
            MemoryAccessSize::Word => value & u128::from(u32::MAX),
            MemoryAccessSize::Doubleword => value & u128::from(u64::MAX),
            MemoryAccessSize::Quadword => value,
        };

        for (mode_bits, pre_index) in [(0x0400_u32, false), (0x0c00, true)] {
            let store = 0x3c00_0000 | size_bits | immediate | mode_bits | (1 << 5);
            let load = store | (1 << 22);
            let transfer_address = if pre_index {
                base.wrapping_add_signed(i64::from(offset))
            } else {
                base
            };

            let ThreadCpuState::A64(a64) = &mut state else {
                unreachable!()
            };
            assert!(a64.set_vector(0, value));
            a64.write_x(x(1), base);
            execute_one_with_context(context, &mut state, store.into()).unwrap();
            let ThreadCpuState::A64(a64) = &mut state else {
                unreachable!()
            };
            assert_eq!(
                a64.read_x(x(1)),
                base.wrapping_add_signed(i64::from(offset)),
                "store encoding={store:#010x}"
            );
            assert!(a64.set_vector(0, u128::MAX));
            a64.write_x(x(1), base);
            execute_one_with_context(context, &mut state, load.into()).unwrap();

            let ThreadCpuState::A64(a64) = &state else {
                unreachable!()
            };
            assert_eq!(a64.vector(0), Some(expected), "load encoding={load:#010x}");
            assert_eq!(
                a64.read_x(x(1)),
                base.wrapping_add_signed(i64::from(offset)),
                "load encoding={load:#010x}"
            );
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(transfer_address),
                        MemoryAccess::normal(access_size),
                    )
                    .unwrap()
                    .value,
                match access_size {
                    MemoryAccessSize::Byte => MemoryValue::U8(value as u8),
                    MemoryAccessSize::Halfword => MemoryValue::U16(value as u16),
                    MemoryAccessSize::Word => MemoryValue::U32(value as u32),
                    MemoryAccessSize::Doubleword => MemoryValue::U64(value as u64),
                    MemoryAccessSize::Quadword => MemoryValue::U128(value),
                },
                "store encoding={store:#010x}"
            );
        }
    }

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(1), 0x4000);
    let pc = a64.pc();
    let outcome = execute_one_with_context(context, &mut state, 0x3c81_0420_u32.into()).unwrap();
    assert!(matches!(outcome, InterpreterOutcome::DataAbort { .. }));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(1)), 0x4000, "fault must suppress writeback");
    assert_eq!(a64.pc(), pc, "faulting instruction must not retire");
}

#[test]
fn a64_simd_register_offset_transfers_cover_extensions_scaling_and_sizes() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(50);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(97);
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
    let mut state = ThreadCpuState::A64(Box::default());
    let value = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ffab;

    for (store, load, vector, base, base_value, offset, offset_value, expected) in [
        (
            0x3c22_4820_u32,
            0x3c62_4820_u32,
            0,
            1,
            0x1000,
            2,
            0x10,
            0xab,
        ),
        (
            0x7c25_d883,
            0x7c65_d883,
            3,
            4,
            0x1040,
            5,
            0xffff_fff8,
            0xffab,
        ),
        (0xbc28_68e6, 0xbc68_68e6, 6, 7, 0x1080, 8, 0x20, 0xddee_ffab),
        (
            0xfc2b_7949,
            0xfc6b_7949,
            9,
            10,
            0x1100,
            11,
            2,
            0x99aa_bbcc_ddee_ffab,
        ),
        (0x3cae_59ac, 0x3cee_59ac, 12, 13, 0x1200, 14, 3, value),
        (
            0x3cb1_fa0f,
            0x3cf1_fa0f,
            15,
            16,
            0x1300,
            17,
            u64::MAX - 1,
            value,
        ),
        (0x3ca0_69be, 0x3ce0_69be, 30, 13, 0x1400, 0, 0, value),
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(vector, value));
        a64.write_x(x(base), base_value);
        a64.write_x(x(offset), offset_value);

        execute_one_with_context(context, &mut state, store.into()).unwrap();
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(vector, u128::MAX));
        execute_one_with_context(context, &mut state, load.into()).unwrap();

        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(vector),
            Some(expected),
            "load encoding={load:#010x}"
        );
    }
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
fn a64_pair_offset_mode_applies_its_scaled_immediate_without_writeback() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(45);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(92);
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
    assert!(memory.initialize_ram(PAGE, 8, &0x1122_3344_u32.to_le_bytes()));
    assert!(memory.initialize_ram(PAGE, 12, &0xffff_fffe_u32.to_le_bytes()));
    let context = InterpreterContext::new(process).with_memory(&memory);
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(1), 0x1000);

    // LDPSW X0, X2, [X1, #8]
    execute_one_with_context(context, &mut state, 0x6941_0820_u32.into()).unwrap();

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), 0x1122_3344);
    assert_eq!(a64.read_x(x(2)), u64::MAX - 1);
    assert_eq!(a64.read_x(x(1)), 0x1000);
}

#[test]
fn a64_exclusive_monitor_uses_physical_identity_and_generation() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(46);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(93);
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
    assert!(memory.initialize_ram(PAGE, 0, &7_u32.to_le_bytes()));
    let monitor = std::cell::RefCell::new(crate::vcpu::ExclusiveMonitorState::default());
    let context = InterpreterContext::new(process)
        .with_memory(&memory)
        .with_exclusive_monitor(&monitor);
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(3), 0x1000);

    execute_one_with_context(context, &mut state, 0x885f_fc60_u32.into()).unwrap(); // LDAXR W0,[X3]
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.read_w(x(0)), 7);
    a64.write_x(x(0), 9);
    execute_one_with_context(context, &mut state, 0x8801_fc60_u32.into()).unwrap(); // STLXR W1,W0,[X3]
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_w(x(1)), 0);
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(9),
    );

    execute_one_with_context(context, &mut state, 0x885f_fc60_u32.into()).unwrap();
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(11),
        )
        .unwrap();
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(0), 13);
    execute_one_with_context(context, &mut state, 0x8801_fc60_u32.into()).unwrap();
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_w(x(1)), 1);
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x1000),
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(11),
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
fn a64_unscaled_load_applies_a_negative_signed_offset_without_writeback() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(49);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(96);
    let profile = GuestCpuProfile::switch_1();
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x1000),
        PAGE,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            MemoryAccess::normal(MemoryAccessSize::Doubleword),
            MemoryValue::U64(0x1122_3344_5566_7788),
        )
        .unwrap();
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x1008),
            MemoryAccess::normal(MemoryAccessSize::Doubleword),
            MemoryValue::U64(0x8877_6655_4433_2211),
        )
        .unwrap();
    let context =
        InterpreterContext::new(ProcessCpuContext::new(profile, SPACE)).with_memory(&memory);
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(12), 0x1008);

    execute_one_with_context(context, &mut state, 0xf85f_8190_u32.into()).unwrap();

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(16)), 0x1122_3344_5566_7788);
    assert_eq!(a64.read_x(x(12)), 0x1008);
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
