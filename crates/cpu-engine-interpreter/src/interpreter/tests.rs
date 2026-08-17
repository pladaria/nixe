use core::cell::Cell;

use nixe_cpu::{
    address::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress},
    coverage::CoverageId,
    ir::terminator::Terminator,
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::{
        CpuMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue, SyntheticMemory,
    },
    profile::{CapabilityStatus, GuestCpuProfile, InstructionFeature, ProcessCpuContext},
    state::{
        ThreadCpuState,
        a32::A32GeneralRegister,
        a64::{A64GeneralRegister, A64Register, Nzcv},
    },
};

use super::{
    ArchitecturalTimer, ArchitecturalTimerSnapshot, InstructionSupport, InterpreterContext,
    InterpreterError, InterpreterOutcome, InterpreterPolicy, execute_fallback, execute_one,
    execute_one_with_context, has_semantics, instruction_support,
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
    let mut state = ThreadCpuState::A32(Box::new(nixe_cpu::state::A32State::t32()));
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

    let mut t32_state = ThreadCpuState::A32(Box::new(nixe_cpu::state::A32State::t32()));
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
    let mut state = ThreadCpuState::A32(Box::new(nixe_cpu::state::A32State::t32()));
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
    let mut state = ThreadCpuState::A32(Box::new(nixe_cpu::state::A32State::t32()));
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
        ThreadCpuState::A32(Box::new(nixe_cpu::state::A32State::t32()))
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
        nixe_cpu::profile::InstructionFeature::AdvancedSimd
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
    execute_one(&profile, &mut state, 0xd53b_0025_u32.into()).unwrap(); // MRS X5,CTR_EL0

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(3)), 0x1234_5678_9abc_def0);
    assert_eq!(a64.read_x(x(4)), 0x14, "DC ZVA is prohibited at EL0");
    assert_eq!(a64.read_x(x(5)), 0x0004_0004);
    assert_eq!(a64.pc(), 12);
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
fn a64_architectural_timer_provider_is_only_sampled_by_timer_reads() {
    struct CountingTimer {
        samples: Cell<u32>,
    }

    impl ArchitecturalTimer for CountingTimer {
        fn snapshot(&self) -> ArchitecturalTimerSnapshot {
            self.samples.set(self.samples.get() + 1);
            ArchitecturalTimerSnapshot {
                counter: 42,
                frequency: 19_200_000,
            }
        }
    }

    let profile = GuestCpuProfile::switch_1();
    let timer = CountingTimer {
        samples: Cell::new(0),
    };
    let context = InterpreterContext::new(ProcessCpuContext::new(profile, AddressSpaceId::new(0)))
        .with_architectural_timer_provider(&timer);
    let mut state = ThreadCpuState::A64(Box::default());

    execute_one_with_context(context, &mut state, 0xd503_201f_u32.into()).unwrap(); // NOP
    assert_eq!(timer.samples.get(), 0);
    execute_one_with_context(context, &mut state, 0xd53b_e020_u32.into()).unwrap(); // MRS X0,CNTVCT_EL0
    assert_eq!(timer.samples.get(), 1);
}

#[test]
fn a64_cache_maintenance_is_a_coherent_memory_no_op() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(8), 0x1234_5000);

    execute_one(&profile, &mut state, 0xd50b_7e28_u32.into()).unwrap(); // DC CIVAC,X8

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(8)), 0x1234_5000);
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
    execute_one(&profile, &mut state, 0xd503_245f_u32.into()).unwrap(); // BTI C as HINT

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.tpidr_el0(), 0xfeed_face_cafe_beef);
    assert_eq!(a64.pc(), 16);

    let outcome = execute_one(&profile, &mut state, 0xd503_203f_u32.into()).unwrap(); // YIELD
    assert!(matches!(outcome, InterpreterOutcome::Scheduled { .. }));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 20, "YIELD retires before scheduler handoff");
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
fn a64_simd_duplicate_element_covers_all_arrangements_and_captured_alias() {
    let profile = GuestCpuProfile::switch_1();
    let captured = InstructionEncoding::from_u32(0x0e04_07ff); // DUP V31.2S,V31.S[0]
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), captured)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded DUP element, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let source_value = 0x8877_6655_4433_2211_fedc_ba98_7654_3210_u128;
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, source_value));
    for register in [1, 3, 5, 7, 9, 11, 13] {
        assert!(a64.set_vector(register, source_value));
    }

    execute_one(&profile, &mut state, captured).unwrap();
    for encoding in [
        0x0e0f_0420_u32, // DUP V0.8B,V1.B[7]
        0x4e1f_0462,     // DUP V2.16B,V3.B[15]
        0x0e0e_04a4,     // DUP V4.4H,V5.H[3]
        0x4e1e_04e6,     // DUP V6.8H,V7.H[7]
        0x0e0c_0528,     // DUP V8.2S,V9.S[1]
        0x4e1c_056a,     // DUP V10.4S,V11.S[3]
        0x4e18_05ac,     // DUP V12.2D,V13.D[1]
    ] {
        execute_one(&profile, &mut state, encoding.into()).unwrap();
    }

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(0x7654_3210_7654_3210));
    assert_eq!(a64.vector(0), Some(0xfefe_fefe_fefe_fefe));
    assert_eq!(a64.vector(2), Some(u128::from_le_bytes([0x88; 16])));
    assert_eq!(a64.vector(4), Some(0xfedc_fedc_fedc_fedc));
    assert_eq!(
        a64.vector(6),
        Some(0x8877_8877_8877_8877_8877_8877_8877_8877)
    );
    assert_eq!(a64.vector(8), Some(0xfedc_ba98_fedc_ba98));
    assert_eq!(
        a64.vector(10),
        Some(0x8877_6655_8877_6655_8877_6655_8877_6655)
    );
    assert_eq!(
        a64.vector(12),
        Some(0x8877_6655_4433_2211_8877_6655_4433_2211)
    );
    assert_eq!(a64.pc(), 32);
}

#[test]
fn a64_simd_scalar_and_vector_immediate_right_shifts_cover_signedness_and_widths() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, 0xaaaa_bbbb_cccc_dddd_fedc_ba98_7654_3210));

    execute_one(&profile, &mut state, 0x7f60_07fe_u32.into()).unwrap(); // USHR D30,D31,#32
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(0x0000_0000_fedc_ba98));

    assert!(a64.set_vector(5, u128::from(u64::MAX - 1)));
    execute_one(&profile, &mut state, 0x5f7f_04a4_u32.into()).unwrap(); // SSHR D4,D5,#1
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(4), Some(u128::from(u64::MAX)));

    let byte_lanes = 0x80ff_7f01_0204_0810_80ff_7f01_0204_0810;
    assert!(a64.set_vector(1, byte_lanes));
    assert!(a64.set_vector(17, byte_lanes));
    execute_one(&profile, &mut state, 0x2f0f_0420_u32.into()).unwrap(); // USHR V0.8B,V1.8B,#1
    execute_one(&profile, &mut state, 0x4f0f_0630_u32.into()).unwrap(); // SSHR V16.16B,V17.16B,#1
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0x407f_3f00_0102_0408));
    assert_eq!(
        a64.vector(16),
        Some(0xc0ff_3f00_0102_0408_c0ff_3f00_0102_0408)
    );

    for (encoding, source_register, destination_register, expected) in [
        (0x2f1f_04a4_u32, 5, 4, 0x7fff_7fff_7fff_7fff), // USHR V4.4H,V5.4H,#1
        (0x6f10_04e6, 7, 6, 0),                         // USHR V6.8H,V7.8H,#16
        (0x2f3f_0528, 9, 8, 0x7fff_ffff_7fff_ffff),     // USHR V8.2S,V9.2S,#1
        (0x6f20_056a, 11, 10, 0),                       // USHR V10.4S,V11.4S,#32
        (
            0x6f7f_05ac,
            13,
            12,
            0x7fff_ffff_ffff_ffff_7fff_ffff_ffff_ffff,
        ), // USHR V12.2D,V13.2D,#1
        (0x6f40_05ee, 15, 14, 0),                       // USHR V14.2D,V15.2D,#64
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(source_register, u128::MAX));
        assert!(a64.set_vector(destination_register, u128::MAX));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(destination_register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }
    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 40);
}

#[test]
fn a64_simd_scalar_and_vector_immediate_left_shifts_cover_aliases_and_widths() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(30, 0xaaaa_bbbb_cccc_dddd_0123_4567_89ab_cdef));
    }

    // Captured in-place scalar form. The scalar write clears the upper half.
    execute_one(&profile, &mut state, 0x5f60_57de_u32.into()).unwrap(); // SHL D30,D30,#32
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(0x89ab_cdef_0000_0000));

    for (encoding, source_register, destination_register, source, expected) in [
        (
            0x0f08_5420_u32,
            1,
            0,
            0xaaaa_bbbb_cccc_dddd_8877_6655_4433_2211,
            0x8877_6655_4433_2211,
        ), // SHL V0.8B,V1.8B,#0
        (
            0x4f0f_5462,
            3,
            2,
            0x0101_0101_0101_0101_0101_0101_0101_0101,
            0x8080_8080_8080_8080_8080_8080_8080_8080,
        ), // SHL V2.16B,V3.16B,#7
        (
            0x4f1f_54e6,
            7,
            6,
            0x0001_0001_0001_0001_0001_0001_0001_0001,
            0x8000_8000_8000_8000_8000_8000_8000_8000,
        ), // SHL V6.8H,V7.8H,#15
        (
            0x4f3f_556a,
            11,
            10,
            0x0000_0001_0000_0001_0000_0001_0000_0001,
            0x8000_0000_8000_0000_8000_0000_8000_0000,
        ), // SHL V10.4S,V11.4S,#31
        (
            0x4f7f_55ee,
            15,
            14,
            0x0000_0000_0000_0001_0000_0000_0000_0001,
            0x8000_0000_0000_0000_8000_0000_0000_0000,
        ), // SHL V14.2D,V15.2D,#63
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(source_register, source));
        assert!(a64.set_vector(destination_register, u128::MAX));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(destination_register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 24);
}

#[test]
fn a64_simd_variable_shifts_cover_signedness_widths_and_captured_alias() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let byte_values = 0x8003_40ff_0181_7f80_u128;
    let byte_shifts = 0x8180_fe00_08f8_01ff_u128;
    {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, byte_values));
        assert!(a64.set_vector(2, byte_shifts));
        assert!(a64.set_vector(3, byte_values));
        assert!(a64.set_vector(4, byte_shifts));
    }

    execute_one(&profile, &mut state, 0x0e22_4420_u32.into()).unwrap(); // SSHL V0.8B,V1.8B,V2.8B
    execute_one(&profile, &mut state, 0x2e24_4462_u32.into()).unwrap(); // USHL V2.8B,V3.8B,V4.8B

    {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(0xff00_10ff_00ff_fec0));
        assert_eq!(a64.vector(2), Some(0x0000_10ff_0000_fe40));
        assert!(a64.set_vector(31, 3_u128 << 32 | 0x8000_0000));
        assert!(a64.set_vector(29, u128::from(u64::MAX) << 32 | 1));
    }
    execute_one(&profile, &mut state, 0x0ebd_47fd_u32.into()).unwrap(); // SSHL V29.2S,V31.2S,V29.2S

    for (encoding, value, shifts, expected) in [
        (
            0x0e62_4420_u32,
            0x8000_0001_7fff_ffff,
            0xffff_0001_0010_fff0,
            0xc000_0002_0000_ffff,
        ),
        (
            0x0ea2_4420,
            0x8000_0000_0000_0001,
            0xffff_ffff_0000_0020,
            0xc000_0000_0000_0000,
        ),
        (
            0x4ee2_4420,
            0xffff_ffff_ffff_ffff_8000_0000_0000_0000,
            0xffff_ffff_ffff_ffc0_0000_0000_0000_0001,
            0xffff_ffff_ffff_ffff_0000_0000_0000_0000,
        ),
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, value));
        assert!(a64.set_vector(2, shifts));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(expected), "encoding={encoding:#010x}");
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.vector(29), Some(1_u128 << 32));
    assert_eq!(a64.pc(), 24);
}

#[test]
fn a64_simd_count_bits_covers_both_vector_widths_and_captured_alias() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let source = 0xffff_ffff_ffff_ffff_ff80_7f55_0f03_0100_u128;
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(30, source));
    assert!(a64.set_vector(3, source));

    execute_one(&profile, &mut state, 0x0e20_5bde_u32.into()).unwrap(); // CNT V30.8B,V30.8B
    execute_one(&profile, &mut state, 0x4e20_5862_u32.into()).unwrap(); // CNT V2.16B,V3.16B

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(0x0801_0704_0402_0100));
    assert_eq!(
        a64.vector(2),
        Some(0x0808_0808_0808_0808_0801_0704_0402_0100)
    );
    assert_eq!(a64.pc(), 8);
}

#[test]
fn a64_simd_add_across_vector_covers_every_allocated_arrangement_and_alias() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(30, 0xaaaa_bbbb_cccc_dddd_0807_0605_0403_0201));
    assert!(a64.set_vector(3, u128::MAX));
    assert!(a64.set_vector(
        5,
        u128::from(u64::MAX) << 48 | 3_u128 << 32 | 2_u128 << 16 | 1
    ));
    assert!(a64.set_vector(7, u128::MAX));
    assert!(a64.set_vector(9, u128::MAX));

    for encoding in [
        0x0e31_bbde_u32, // ADDV B30,V30.8B
        0x4e31_b862,     // ADDV B2,V3.16B
        0x0e71_b8a4,     // ADDV H4,V5.4H
        0x4e71_b8e6,     // ADDV H6,V7.8H
        0x4eb1_b928,     // ADDV S8,V9.4S
    ] {
        execute_one(&profile, &mut state, encoding.into()).unwrap();
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(36));
    assert_eq!(a64.vector(2), Some(0xf0));
    assert_eq!(a64.vector(4), Some(5));
    assert_eq!(a64.vector(6), Some(0xfff8));
    assert_eq!(a64.vector(8), Some(0xffff_fffc));
    assert_eq!(a64.pc(), 20);
}

#[test]
fn a64_simd_modified_immediate_expands_lanes_and_clears_inactive_bits() {
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
fn a64_simd_modified_immediate_covers_move_negate_merge_and_bitmask_forms() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());

    for (encoding, register, initial, expected) in [
        (
            0x6f00_05fa_u32,
            26,
            0,
            0xffff_fff0_ffff_fff0_ffff_fff0_ffff_fff0,
        ), // MVNI V26.4S,#0xf
        (0x6f01_c681, 1, 0, 0xffff_cb00_ffff_cb00_ffff_cb00_ffff_cb00), // MVNI V1.4S,#0x34,MSL #8
        (0x4f02_a6c2, 2, 0, 0x5600_5600_5600_5600_5600_5600_5600_5600), // MOVI V2.8H,#0x56,LSL #8
        (
            0x2f03_8703,
            3,
            u128::MAX,
            0x0000_0000_0000_0000_ff87_ff87_ff87_ff87,
        ), // MVNI V3.4H,#0x78
        (
            0x4f04_5744,
            4,
            0x0000_0001_0000_0001_0000_0001_0000_0001,
            0x009a_0001_009a_0001_009a_0001_009a_0001,
        ), // ORR V4.4S,#0x9a,LSL #16
        (
            0x6f05_b785,
            5,
            u128::MAX,
            0x43ff_43ff_43ff_43ff_43ff_43ff_43ff_43ff,
        ), // BIC V5.8H,#0xbc,LSL #8
        (0x4f06_e7c6, 6, 0, 0xdede_dede_dede_dede_dede_dede_dede_dede), // MOVI V6.16B,#0xde
        (0x6f05_e547, 7, 0, 0xff00_ff00_ff00_ff00_ff00_ff00_ff00_ff00), // MOVI V7.2D,#0xff00ff00ff00ff00
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(register, initial));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }
}

#[test]
fn a64_simd_fmov_immediate_covers_two_and_four_single_lanes_and_two_double_lanes() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(0x07c0_0000);
    a64.set_fpsr(0x9f);

    for (encoding, register, expected) in [
        (
            0x0f03_f600_u32,
            0,
            0x0000_0000_0000_0000_3f80_0000_3f80_0000,
        ), // FMOV V0.2S,#1.0
        (0x4f03_f61e, 30, 0x3f80_0000_3f80_0000_3f80_0000_3f80_0000), // FMOV V30.4S,#1.0
        (0x6f03_f601, 1, 0x3ff0_0000_0000_0000_3ff0_0000_0000_0000),  // FMOV V1.2D,#1.0
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(register, u128::MAX));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
        assert_eq!(a64.fpcr(), 0x07c0_0000);
        assert_eq!(a64.fpsr(), 0x9f);
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 12);
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
fn a64_simd_insert_element_copies_each_lane_width_and_preserves_other_lanes() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());

    for (encoding, destination, source, expected) in [
        (
            0x6e03_07be_u32,
            0x1111_1111_1111_1111_1111_1111_1111_1111,
            0x8877_6655_4433_2211_fedc_ba98_7654_32ab,
            0x1111_1111_1111_1111_1111_1111_1111_ab11,
        ), // MOV V30.B[1],V29.B[0]
        (
            0x6e0e_7462,
            u128::MAX,
            0x1234_8877_6655_4433_2211_fedc_ba98_7654,
            0xffff_ffff_ffff_ffff_1234_ffff_ffff_ffff,
        ), // MOV V2.H[3],V3.H[7]
        (
            0x6e14_24a4,
            0,
            0x8877_6655_4433_2211_fedc_ba98_7654_3210,
            0x0000_0000_fedc_ba98_0000_0000_0000_0000,
        ), // MOV V4.S[2],V5.S[1]
        (
            0x6e18_04e6,
            u128::MAX,
            0x8877_6655_4433_2211_fedc_ba98_7654_3210,
            0xfedc_ba98_7654_3210_ffff_ffff_ffff_ffff,
        ), // MOV V6.D[1],V7.D[0]
    ] {
        let destination_register = (encoding & 0x1f) as u8;
        let source_register = ((encoding >> 5) & 0x1f) as u8;
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(destination_register, destination));
        assert!(a64.set_vector(source_register, source));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(destination_register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }
}

#[test]
fn a64_simd_insert_general_truncates_source_and_preserves_other_lanes() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(9), 0xfedc_ba98_7654_3210);
    a64.write_x(x(11), 0xfedc_ba98_7654_3210);
    a64.write_x(x(13), 0xfedc_ba98_7654_3210);
    a64.write_x(x(15), 0xfedc_ba98_7654_3210);

    for (encoding, register, expected) in [
        (
            0x4e03_1d28_u32,
            8,
            0xffff_ffff_ffff_ffff_ffff_ffff_ffff_10ff,
        ), // MOV V8.B[1],W9
        (0x4e0a_1d6a, 10, 0xffff_ffff_ffff_ffff_ffff_3210_ffff_ffff), // MOV V10.H[2],W11
        (0x4e1c_1dac, 12, 0x7654_3210_ffff_ffff_ffff_ffff_ffff_ffff), // MOV V12.S[3],W13
        (0x4e18_1dee, 14, 0xfedc_ba98_7654_3210_ffff_ffff_ffff_ffff), // MOV V14.D[1],X15
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(register, u128::MAX));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }
}

#[test]
fn a64_simd_two_source_permutations_cover_all_operations_and_arrangements() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let first = u128::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let second = u128::from_le_bytes([
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e,
        0x8f,
    ]);
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, first));
    assert!(a64.set_vector(2, second));

    for (encoding, destination, expected) in [
        (
            0x4e02_1823_u32, // UZP1 V3.16B,V1.16B,V2.16B
            3,
            [
                0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e,
            ],
        ),
        (
            0x4e02_5824, // UZP2 V4.16B,V1.16B,V2.16B
            4,
            [
                1, 3, 5, 7, 9, 11, 13, 15, 0x81, 0x83, 0x85, 0x87, 0x89, 0x8b, 0x8d, 0x8f,
            ],
        ),
        (
            0x4e02_2825, // TRN1 V5.16B,V1.16B,V2.16B
            5,
            [
                0, 0x80, 2, 0x82, 4, 0x84, 6, 0x86, 8, 0x88, 10, 0x8a, 12, 0x8c, 14, 0x8e,
            ],
        ),
        (
            0x4e02_6826, // TRN2 V6.16B,V1.16B,V2.16B
            6,
            [
                1, 0x81, 3, 0x83, 5, 0x85, 7, 0x87, 9, 0x89, 11, 0x8b, 13, 0x8d, 15, 0x8f,
            ],
        ),
        (
            0x0e02_3827, // ZIP1 V7.8B,V1.8B,V2.8B
            7,
            [0, 0x80, 1, 0x81, 2, 0x82, 3, 0x83, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            0x4e42_6828, // TRN2 V8.8H,V1.8H,V2.8H
            8,
            [
                2, 3, 0x82, 0x83, 6, 7, 0x86, 0x87, 10, 11, 0x8a, 0x8b, 14, 15, 0x8e, 0x8f,
            ],
        ),
        (
            0x4e82_5829, // UZP2 V9.4S,V1.4S,V2.4S
            9,
            [
                4, 5, 6, 7, 12, 13, 14, 15, 0x84, 0x85, 0x86, 0x87, 0x8c, 0x8d, 0x8e, 0x8f,
            ],
        ),
        (
            0x4ec2_782a, // ZIP2 V10.2D,V1.2D,V2.2D
            10,
            [
                8, 9, 10, 11, 12, 13, 14, 15, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
            ],
        ),
    ] {
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(destination), Some(u128::from_le_bytes(expected)));
    }
}

#[test]
fn a64_simd_zip1_handles_the_observed_overlapping_destination() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        30,
        u128::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ));
    assert!(a64.set_vector(
        29,
        u128::from_le_bytes([
            0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
            0x8e, 0x8f,
        ])
    ));

    execute_one(&profile, &mut state, 0x4e1d_3bde_u32.into()).unwrap();
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(30),
        Some(u128::from_le_bytes([
            0, 0x80, 1, 0x81, 2, 0x82, 3, 0x83, 4, 0x84, 5, 0x85, 6, 0x86, 7, 0x87,
        ]))
    );
}

#[test]
fn a64_simd_extract_supports_both_vector_widths_and_aliasing() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let first = u128::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let second = u128::from_le_bytes([
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e,
        0x8f,
    ]);
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, first));
    assert!(a64.set_vector(2, second));
    assert!(a64.set_vector(10, first));
    assert!(a64.set_vector(11, second));
    assert!(a64.set_vector(31, first));

    execute_one(&profile, &mut state, 0x6e02_4023_u32.into()).unwrap(); // EXT V3.16B,V1.16B,V2.16B,#8
    execute_one(&profile, &mut state, 0x2e0b_3949_u32.into()).unwrap(); // EXT V9.8B,V10.8B,V11.8B,#7
    execute_one(&profile, &mut state, 0x6e1f_43ff_u32.into()).unwrap(); // EXT V31.16B,V31.16B,V31.16B,#8

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(3),
        Some(u128::from_le_bytes([
            8, 9, 10, 11, 12, 13, 14, 15, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        ]))
    );
    assert_eq!(
        a64.vector(9),
        Some(u128::from(u64::from_le_bytes([
            7, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,
        ])))
    );
    assert_eq!(
        a64.vector(31),
        Some(u128::from_le_bytes([
            8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7
        ]))
    );
}

#[test]
fn a64_fmov_to_general_copies_scalar_and_upper_lane_bit_patterns() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, 0x8877_6655_4433_2211_fedc_ba98_7654_3210));

    for (encoding, register, expected) in [
        (0x1ee6_03e0_u32, 0, 0x3210),            // FMOV W0,H31
        (0x1e26_03e1, 1, 0x7654_3210),           // FMOV W1,S31
        (0x9e66_03e2, 2, 0xfedc_ba98_7654_3210), // FMOV X2,D31
        (0x9eae_03e3, 3, 0x8877_6655_4433_2211), // FMOV X3,V31.D[1]
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
    assert_eq!(a64.pc(), 16);
}

#[test]
fn a64_fmov_from_general_clears_scalar_upper_bits_and_preserves_other_lane() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(9), 0xfedc_ba98_7654_3210);
    for register in 10..=13 {
        assert!(a64.set_vector(register, u128::MAX));
    }

    for (encoding, register, expected) in [
        (0x1e27_012a_u32, 10, 0x0000_0000_7654_3210), // FMOV S10,W9
        (0x9e67_012b, 11, 0xfedc_ba98_7654_3210),     // FMOV D11,X9
        (0x9eaf_012c, 12, 0xfedc_ba98_7654_3210_ffff_ffff_ffff_ffff), // FMOV V12.D[1],X9
        (0x1ee7_012d, 13, 0x0000_0000_0000_3210),     // FMOV H13,W9
    ] {
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.pc(), 16);
}

#[test]
fn a64_scalar_fmov_register_copies_exact_bits_for_all_precisions() {
    let profile = GuestCpuProfile::switch_1()
        .with_instruction_feature(InstructionFeature::Fp16, CapabilityStatus::Enabled);
    let captured = InstructionEncoding::from_u32(0x1e20_41c3); // FMOV S3,S14
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), captured)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FMOV, got {other:?}"),
        };
    assert!(has_semantics(&decoded));

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(14, u128::from(0x7f80_0001_u32) | (u128::MAX << 32)));
    assert!(a64.set_vector(3, u128::MAX));
    assert!(a64.set_vector(
        15,
        u128::from(0x7ff0_0000_0000_0001_u64) | (u128::MAX << 64)
    ));
    assert!(a64.set_vector(4, u128::MAX));
    assert!(a64.set_vector(16, u128::from(0x8001_u16) | (u128::MAX << 16)));
    assert!(a64.set_vector(5, u128::MAX));
    a64.set_fpcr(u32::MAX);
    a64.set_fpsr(0x95);

    execute_one(&profile, &mut state, captured).unwrap();
    execute_one(&profile, &mut state, 0x1e60_41e4_u32.into()).unwrap(); // FMOV D4,D15
    execute_one(&profile, &mut state, 0x1ee0_4205_u32.into()).unwrap(); // FMOV H5,H16

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(3), Some(u128::from(0x7f80_0001_u32)));
    assert_eq!(a64.vector(4), Some(u128::from(0x7ff0_0000_0000_0001_u64)));
    assert_eq!(a64.vector(5), Some(0x8001));
    assert_eq!(a64.fpcr(), u32::MAX);
    assert_eq!(a64.fpsr(), 0x95);
    assert_eq!(a64.pc(), 12);
}

#[test]
fn a64_scalar_fabs_fneg_transform_only_the_sign_bit_for_all_precisions() {
    let profile = GuestCpuProfile::switch_1()
        .with_instruction_feature(InstructionFeature::Fp16, CapabilityStatus::Enabled);
    let captured = InstructionEncoding::from_u32(0x1e20_c3fe); // FABS S30,S31
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), captured)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FABS, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::from(0xff80_0001_u32) | (u128::MAX << 32)));
    assert!(a64.set_vector(9, u128::from(0x7ff0_0000_0000_0001_u64) | (u128::MAX << 64)));
    assert!(a64.set_vector(5, u128::from(0x8001_u16) | (u128::MAX << 16)));
    assert!(a64.set_vector(11, u128::from(0x0001_u16) | (u128::MAX << 16)));
    a64.set_fpcr(u32::MAX);
    a64.set_fpsr(0x95);

    execute_one(&profile, &mut state, captured).unwrap();
    execute_one(&profile, &mut state, 0x1e61_4128_u32.into()).unwrap(); // FNEG D8,D9
    execute_one(&profile, &mut state, 0x1ee0_c0a4_u32.into()).unwrap(); // FABS H4,H5
    execute_one(&profile, &mut state, 0x1ee1_416a_u32.into()).unwrap(); // FNEG H10,H11

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(u128::from(0x7f80_0001_u32)));
    assert_eq!(a64.vector(8), Some(u128::from(0xfff0_0000_0000_0001_u64)));
    assert_eq!(a64.vector(4), Some(1));
    assert_eq!(a64.vector(10), Some(0x8001));
    assert_eq!(a64.fpcr(), u32::MAX);
    assert_eq!(a64.fpsr(), 0x95);
    assert_eq!(a64.pc(), 16);
}

#[test]
fn a64_vector_fabs_fneg_cover_all_base_arrangements_and_captured_alias() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        a64.set_fpcr(u32::MAX);
        a64.set_fpsr(0x95);
    }

    for (encoding, source_register, destination_register, source, expected) in [
        (
            0x0ea0_f820_u32,
            1,
            0,
            (u128::from(u64::MAX) << 64) | 0x8000_0000_ff80_0001,
            0x0000_0000_7f80_0001,
        ), // FABS V0.2S,V1.2S
        (
            0x4ea0_f862,
            3,
            2,
            0xff80_0001_8000_0000_7fc0_0001_0000_0001,
            0x7f80_0001_0000_0000_7fc0_0001_0000_0001,
        ), // FABS V2.4S,V3.4S
        (
            0x4ee0_f8a4,
            5,
            4,
            0xfff0_0000_0000_0001_8000_0000_0000_0000,
            0x7ff0_0000_0000_0001_0000_0000_0000_0000,
        ), // FABS V4.2D,V5.2D
        (
            0x2ea0_fbde,
            30,
            30,
            (u128::from(u64::MAX) << 64) | 0x3f80_0000_bf80_0000,
            0xbf80_0000_3f80_0000,
        ), // FNEG V30.2S,V30.2S (captured)
        (
            0x6ea0_f928,
            9,
            8,
            0x7f80_0001_8000_0000_3f80_0000_bf80_0000,
            0xff80_0001_0000_0000_bf80_0000_3f80_0000,
        ), // FNEG V8.4S,V9.4S
        (
            0x6ee0_f96a,
            11,
            10,
            0x7ff0_0000_0000_0001_8000_0000_0000_0000,
            0xfff0_0000_0000_0001_0000_0000_0000_0000,
        ), // FNEG V10.2D,V11.2D
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(source_register, source));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(destination_register),
            Some(expected),
            "encoding={encoding:#010x}"
        );
    }

    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.fpcr(), u32::MAX);
    assert_eq!(a64.fpsr(), 0x95);
    assert_eq!(a64.pc(), 24);
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
fn a64_simd_bitwise_family_handles_logic_destination_masks_and_vector_width() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let first = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128;
    let second = 0x00ff_00ff_00ff_00ff_ff00_ff00_ff00_ff00_u128;
    let destination = 0xaaaa_5555_aaaa_5555_0f0f_f0f0_0f0f_f0f0_u128;
    let cases = [
        (0x4e22_1c20_u32, first & second), // AND V0.16B,V1.16B,V2.16B
        (0x4e62_1c20, first & !second),    // BIC V0.16B,V1.16B,V2.16B
        (0x4ea2_1c20, first | second),     // ORR V0.16B,V1.16B,V2.16B
        (0x4ee2_1c20, first | !second),    // ORN V0.16B,V1.16B,V2.16B
        (0x6e22_1c20, first ^ second),     // EOR V0.16B,V1.16B,V2.16B
        (
            0x6e62_1c20, // BSL V0.16B,V1.16B,V2.16B
            (destination & first) | (!destination & second),
        ),
        (
            0x6ea2_1c20, // BIT V0.16B,V1.16B,V2.16B
            (destination & !second) | (first & second),
        ),
        (
            0x6ee2_1c20, // BIF V0.16B,V1.16B,V2.16B
            (destination & second) | (first & !second),
        ),
    ];

    for (encoding, expected) in cases {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(0, destination));
        assert!(a64.set_vector(1, first));
        assert!(a64.set_vector(2, second));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(expected), "encoding={encoding:#010x}");
    }

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::MAX));
    assert!(a64.set_vector(1, first));
    assert!(a64.set_vector(2, second));
    execute_one(&profile, &mut state, 0x0e22_1c20_u32.into()).unwrap(); // AND V0.8B,V1.8B,V2.8B
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some((first & second) & u128::from(u64::MAX)));
}

#[test]
fn a64_simd_bitwise_executes_observed_libnx_orr_encoding() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let first = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210_u128;
    let second = 0xf000_0000_0000_000f_0000_ffff_0000_ffff_u128;
    assert!(a64.set_vector(3, first));
    assert!(a64.set_vector(4, second));

    execute_one(&profile, &mut state, 0x4ea4_1c71_u32.into()).unwrap(); // ORR V17.16B,V3.16B,V4.16B
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(17), Some(first | second));
}

#[test]
fn a64_simd_shift_right_narrow_executes_observed_libnx_encoding() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let source_lanes = [
        0x0010_u16, 0x0123, 0x0ff0, 0x1234, 0x8000, 0xabcd, 0xfff0, 0x000f,
    ];
    let source = source_lanes
        .into_iter()
        .enumerate()
        .fold(0_u128, |value, (lane, element)| {
            value | (u128::from(element) << (lane * 16))
        });
    let expected = source_lanes
        .into_iter()
        .enumerate()
        .fold(0_u64, |value, (lane, element)| {
            value | (u64::from((element >> 4) as u8) << (lane * 8))
        });
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, source));
    execute_one(&profile, &mut state, 0x0f0c_8400_u32.into()).unwrap(); // SHRN V0.8B,V0.8H,#4
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(expected)));

    assert!(a64.set_vector(0, u128::from(0x8877_6655_4433_2211_u64)));
    assert!(a64.set_vector(1, source));
    execute_one(&profile, &mut state, 0x4f0c_8420_u32.into()).unwrap(); // SHRN2 V0.16B,V1.8H,#4
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(0),
        Some((u128::from(expected) << 64) | 0x8877_6655_4433_2211_u128)
    );
}

#[test]
fn a64_simd_extract_narrow_covers_all_lane_widths_and_upper_half_forms() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let source = 0x8877_6655_4433_2211_fedc_ba98_7654_3210_u128;
    assert!(a64.set_vector(1, source));
    assert!(a64.set_vector(3, source));
    assert!(a64.set_vector(5, source));

    execute_one(&profile, &mut state, 0x0e21_2820_u32.into()).unwrap(); // XTN V0.8B,V1.8H
    execute_one(&profile, &mut state, 0x0e61_2862_u32.into()).unwrap(); // XTN V2.4H,V3.4S
    execute_one(&profile, &mut state, 0x0ea1_28a4_u32.into()).unwrap(); // XTN V4.2S,V5.2D

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0x7755_3311_dc98_5410));
    assert_eq!(a64.vector(2), Some(0x6655_2211_ba98_3210));
    assert_eq!(a64.vector(4), Some(0x4433_2211_7654_3210));

    let low = 0x0123_4567_89ab_cdef_u64;
    for register in [6, 8, 10] {
        assert!(a64.set_vector(register, u128::from(low)));
    }
    assert!(a64.set_vector(7, source));
    assert!(a64.set_vector(9, source));
    assert!(a64.set_vector(11, source));

    execute_one(&profile, &mut state, 0x4e21_28e6_u32.into()).unwrap(); // XTN2 V6.16B,V7.8H
    execute_one(&profile, &mut state, 0x4e61_2928_u32.into()).unwrap(); // XTN2 V8.8H,V9.4S
    execute_one(&profile, &mut state, 0x4ea1_296a_u32.into()).unwrap(); // XTN2 V10.4S,V11.2D

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(6),
        Some((0x7755_3311_dc98_5410_u128 << 64) | u128::from(low))
    );
    assert_eq!(
        a64.vector(8),
        Some((0x6655_2211_ba98_3210_u128 << 64) | u128::from(low))
    );
    assert_eq!(
        a64.vector(10),
        Some((0x4433_2211_7654_3210_u128 << 64) | u128::from(low))
    );
    assert_eq!(a64.pc(), 24);
}

#[test]
fn a64_simd_extract_narrow_executes_captured_aliasing_encoding() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x0ea1_2bde); // XTN V30.2S,V30.2D
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded XTN, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(30, 0x8877_6655_4433_2211_fedc_ba98_7654_3210));

    execute_one(&profile, &mut state, encoding).unwrap();

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(0x4433_2211_7654_3210));
    assert_eq!(a64.pc(), 4);
}

#[test]
fn a64_simd_pairwise_integer_family_reduces_adjacent_lanes_from_each_source() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let first = [
        0x80, 0x7f, 0xff, 0x00, 0x05, 0x04, 0xfe, 0xfd, 0x20, 0x10, 0x81, 0x82, 0x00, 0xff, 0x07,
        0x07,
    ];
    let second = [
        0x01, 0x02, 0xc8, 0x64, 0x00, 0xff, 0x7f, 0x80, 0x09, 0x03, 0xfe, 0x01, 0x08, 0x04, 0x06,
        0x0c,
    ];
    let cases = [
        (
            0x4e22_bc20_u32, // ADDP V0.16B,V1.16B,V2.16B
            [
                0xff, 0xff, 0x09, 0xfb, 0x30, 0x03, 0xff, 0x0e, 0x03, 0x2c, 0xff, 0xff, 0x0c, 0xff,
                0x0c, 0x12,
            ],
        ),
        (
            0x4e22_a420, // SMAXP V0.16B,V1.16B,V2.16B
            [
                0x7f, 0x00, 0x05, 0xfe, 0x20, 0x82, 0x00, 0x07, 0x02, 0x64, 0x00, 0x7f, 0x09, 0x01,
                0x08, 0x0c,
            ],
        ),
        (
            0x4e22_ac20, // SMINP V0.16B,V1.16B,V2.16B
            [
                0x80, 0xff, 0x04, 0xfd, 0x10, 0x81, 0xff, 0x07, 0x01, 0xc8, 0xff, 0x80, 0x03, 0xfe,
                0x04, 0x06,
            ],
        ),
        (
            0x6e22_a420, // UMAXP V0.16B,V1.16B,V2.16B
            [
                0x80, 0xff, 0x05, 0xfe, 0x20, 0x82, 0xff, 0x07, 0x02, 0xc8, 0xff, 0x80, 0x09, 0xfe,
                0x08, 0x0c,
            ],
        ),
        (
            0x6e22_ac20, // UMINP V0.16B,V1.16B,V2.16B
            [
                0x7f, 0x00, 0x04, 0xfd, 0x10, 0x81, 0x00, 0x07, 0x01, 0x64, 0x00, 0x7f, 0x03, 0x01,
                0x04, 0x06,
            ],
        ),
    ];

    for (encoding, expected) in cases {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, u128::from_le_bytes(first)));
        assert!(a64.set_vector(2, u128::from_le_bytes(second)));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(0),
            Some(u128::from_le_bytes(expected)),
            "encoding={encoding:#010x}"
        );
    }
}

#[test]
fn a64_simd_elementwise_min_max_handles_signed_lanes_and_register_aliasing() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let first = u128::from(10_u32) | (u128::from((-20_i32) as u32) << 32);
    let second = u128::from(5_u32) | (u128::from((-30_i32) as u32) << 32);
    assert!(a64.set_vector(30, first));
    assert!(a64.set_vector(31, second));

    execute_one(&profile, &mut state, 0x0ebf_6fdf_u32.into()).unwrap(); // SMIN V31.2S,V30.2S,V31.2S

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(second));
}

#[test]
fn a64_simd_pairwise_executes_observed_libnx_encodings_with_register_aliasing() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let first = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let second = [
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(17, u128::from_le_bytes(first)));
    assert!(a64.set_vector(18, u128::from_le_bytes(second)));
    execute_one(&profile, &mut state, 0x4e32_be31_u32.into()).unwrap(); // ADDP V17.16B,V17.16B,V18.16B
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(17),
        Some(u128::from_le_bytes([
            1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
        ]))
    );

    let source = [1, 9, 7, 3, 0, 255, 128, 127, 10, 11, 12, 2, 4, 8, 6, 5];
    assert!(a64.set_vector(17, u128::from_le_bytes(source)));
    execute_one(&profile, &mut state, 0x6e31_a631_u32.into()).unwrap(); // UMAXP V17.16B,V17.16B,V17.16B
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(17),
        Some(u128::from_le_bytes([
            9, 7, 255, 128, 11, 12, 8, 6, 9, 7, 255, 128, 11, 12, 8, 6,
        ]))
    );
}

#[test]
fn a64_simd_add_pairwise_supports_64_bit_lanes_and_clears_inactive_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::MAX));
    assert!(a64.set_vector(2, u128::from_le_bytes([1; 16])));
    execute_one(&profile, &mut state, 0x0e22_bc20_u32.into()).unwrap(); // ADDP V0.8B,V1.8B,V2.8B
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(0),
        Some(u128::from(u64::from_le_bytes([
            0xfe, 0xfe, 0xfe, 0xfe, 2, 2, 2, 2,
        ])))
    );

    assert!(a64.set_vector(1, u128::from(u64::MAX) | (u128::from(1_u64) << 64)));
    assert!(a64.set_vector(2, u128::from(2_u64) | (u128::from(3_u64) << 64)));
    execute_one(&profile, &mut state, 0x4ee2_bc20_u32.into()).unwrap(); // ADDP V0.2D,V1.2D,V2.2D
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(5_u64) << 64));
}

#[test]
fn a64_simd_pairwise_integer_supports_halfword_and_word_lanes() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        1,
        u128::from_le_bytes([1, 0, 2, 0, 0x2c, 1, 0x90, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    ));
    assert!(a64.set_vector(
        2,
        u128::from_le_bytes([0xf4, 1, 0x58, 2, 0xff, 0xff, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,])
    ));
    execute_one(&profile, &mut state, 0x0e62_bc20_u32.into()).unwrap(); // ADDP V0.4H,V1.4H,V2.4H
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(0),
        Some(u128::from(u64::from_le_bytes([
            3, 0, 0xbc, 2, 0x4c, 4, 1, 0,
        ])))
    );

    assert!(a64.set_vector(
        1,
        u128::from(1_u32)
            | (u128::from(u32::MAX) << 32)
            | (u128::from(4_u32) << 64)
            | (u128::from(3_u32) << 96)
    ));
    assert!(a64.set_vector(
        2,
        u128::from(5_u32)
            | (u128::from(6_u32) << 32)
            | (u128::from(9_u32) << 64)
            | (u128::from(8_u32) << 96)
    ));
    execute_one(&profile, &mut state, 0x6ea2_a420_u32.into()).unwrap(); // UMAXP V0.4S,V1.4S,V2.4S
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(0),
        Some(
            u128::from(u32::MAX)
                | (u128::from(4_u32) << 32)
                | (u128::from(6_u32) << 64)
                | (u128::from(9_u32) << 96)
        )
    );
}

#[test]
fn a64_simd_integer_register_comparisons_produce_per_lane_masks() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let lhs_half = [0x80, 0x7f, 5, 5, 0, 0xff, 0x55, 0xaa];
    let rhs_half = [0x7f, 0x80, 5, 6, 1, 0xfe, 0xaa, 0x55];
    let mut lhs = [0_u8; 16];
    let mut rhs = [0_u8; 16];
    lhs[..8].copy_from_slice(&lhs_half);
    lhs[8..].copy_from_slice(&lhs_half);
    rhs[..8].copy_from_slice(&rhs_half);
    rhs[8..].copy_from_slice(&rhs_half);

    let cases = [
        (
            0x4e21_34a3_u32, // CMGT V3.16B,V5.16B,V1.16B
            [0, 0xff, 0, 0, 0, 0xff, 0xff, 0],
        ),
        (
            0x6e21_34a3, // CMHI V3.16B,V5.16B,V1.16B
            [0xff, 0, 0, 0, 0, 0xff, 0, 0xff],
        ),
        (
            0x4e21_3ca3, // CMGE V3.16B,V5.16B,V1.16B
            [0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0],
        ),
        (
            0x6e21_3ca3, // CMHS V3.16B,V5.16B,V1.16B
            [0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff],
        ),
        (
            0x4e21_8ca3, // CMTST V3.16B,V5.16B,V1.16B
            [0, 0, 0xff, 0xff, 0, 0xff, 0, 0],
        ),
        (
            0x6e21_8ca3, // CMEQ V3.16B,V5.16B,V1.16B
            [0, 0, 0xff, 0, 0, 0, 0, 0],
        ),
    ];
    for (encoding, expected_half) in cases {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(5, u128::from_le_bytes(lhs)));
        assert!(a64.set_vector(1, u128::from_le_bytes(rhs)));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        let mut expected = [0_u8; 16];
        expected[..8].copy_from_slice(&expected_half);
        expected[8..].copy_from_slice(&expected_half);
        assert_eq!(
            a64.vector(3),
            Some(u128::from_le_bytes(expected)),
            "encoding={encoding:#010x}"
        );
    }
}

#[test]
fn a64_simd_integer_zero_comparisons_cover_all_relations() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let source_half = [0x80, 0, 1, 0xff, 0x7f, 0, 2, 0xfe];
    let mut source = [0_u8; 16];
    source[..8].copy_from_slice(&source_half);
    source[8..].copy_from_slice(&source_half);
    let cases = [
        (
            0x4e20_8823_u32, // CMGT V3.16B,V1.16B,#0
            [0, 0, 0xff, 0, 0xff, 0, 0xff, 0],
        ),
        (
            0x6e20_8823, // CMGE V3.16B,V1.16B,#0
            [0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0],
        ),
        (
            0x4e20_9823, // CMEQ V3.16B,V1.16B,#0
            [0, 0xff, 0, 0, 0, 0xff, 0, 0],
        ),
        (
            0x6e20_9823, // CMLE V3.16B,V1.16B,#0
            [0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff],
        ),
        (
            0x4e20_a823, // CMLT V3.16B,V1.16B,#0
            [0xff, 0, 0, 0xff, 0, 0, 0, 0xff],
        ),
    ];
    for (encoding, expected_half) in cases {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, u128::from_le_bytes(source)));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        let mut expected = [0_u8; 16];
        expected[..8].copy_from_slice(&expected_half);
        expected[8..].copy_from_slice(&expected_half);
        assert_eq!(
            a64.vector(3),
            Some(u128::from_le_bytes(expected)),
            "encoding={encoding:#010x}"
        );
    }
}

#[test]
fn a64_simd_integer_register_comparisons_cover_64_bit_lanes_and_clear_upper_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        5,
        (u128::from(i64::MIN as u64) << 64) | u128::from(i64::MAX as u64)
    ));
    assert!(a64.set_vector(1, u128::from(u64::MAX) << 64));
    execute_one(&profile, &mut state, 0x4ee1_34a3_u32.into()).unwrap(); // CMGT V3.2D,V5.2D,V1.2D
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(3), Some(u128::from(u64::MAX)));

    assert!(a64.set_vector(5, u128::MAX));
    assert!(a64.set_vector(1, 0));
    execute_one(&profile, &mut state, 0x2e21_3ca3_u32.into()).unwrap(); // CMHS V3.8B,V5.8B,V1.8B
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(3), Some(u128::from(u64::MAX)));
}

#[test]
fn a64_simd_integer_to_float_converts_signed_and_unsigned_vector_arrangements() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let source = u128::from(0_u32)
        | (u128::from((-1_i32) as u32) << 32)
        | (u128::from(16_777_217_u32) << 64)
        | (u128::from(i32::MIN as u32) << 96);
    assert!(a64.set_vector(31, source));
    a64.set_fpsr(1 << 1);

    execute_one(&profile, &mut state, 0x4e21_dbfc_u32.into()).unwrap(); // SCVTF V28.4S,V31.4S
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(28),
        Some(
            u128::from(0x0000_0000_u32)
                | (u128::from(0xbf80_0000_u32) << 32)
                | (u128::from(0x4b80_0000_u32) << 64)
                | (u128::from(0xcf00_0000_u32) << 96)
        )
    );
    assert_eq!(a64.fpsr(), (1 << 1) | (1 << 4));

    assert!(a64.set_vector(7, u128::MAX));
    assert!(a64.set_vector(6, u128::MAX));
    execute_one(&profile, &mut state, 0x2e21_d8e6_u32.into()).unwrap(); // UCVTF V6.2S,V7.2S
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(6),
        Some(u128::from(0x4f80_0000_u32) | (u128::from(0x4f80_0000_u32) << 32))
    );

    assert!(a64.set_vector(11, u128::MAX));
    execute_one(&profile, &mut state, 0x6e61_d96a_u32.into()).unwrap(); // UCVTF V10.2D,V11.2D
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(10),
        Some(u128::from(0x43f0_0000_0000_0000_u64) | (u128::from(0x43f0_0000_0000_0000_u64) << 64))
    );
}

#[test]
fn a64_simd_integer_to_float_obeys_fpcr_rounding_direction() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let source = u128::from(16_777_217_u32) | (u128::from((-16_777_217_i32) as u32) << 32);
    let cases = [
        (0_u32, 0x4b80_0000_u32, 0xcb80_0000_u32),
        (1_u32, 0x4b80_0001_u32, 0xcb80_0000_u32),
        (2_u32, 0x4b80_0000_u32, 0xcb80_0001_u32),
        (3_u32, 0x4b80_0000_u32, 0xcb80_0000_u32),
    ];
    for (mode, positive, negative) in cases {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, source));
        a64.set_fpcr(mode << 22);
        execute_one(&profile, &mut state, 0x0e21_d820_u32.into()).unwrap(); // SCVTF V0.2S,V1.2S
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(0),
            Some(u128::from(positive) | (u128::from(negative) << 32)),
            "FPCR.RMode={mode:#04b}"
        );
    }
}

#[test]
fn a64_simd_integer_to_float_trap_boundary_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let location = source(profile, 0, ExecutionState::A64);
    let encoding = InstructionEncoding::from_u32(0x4e21_dbfc);
    let decoded = match nixe_cpu::decode::decode(&profile, location, encoding) {
        nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
        other => panic!("expected decoded SCVTF vector, got {other:?}"),
    };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::from(16_777_217_u32)));
    assert!(a64.set_vector(28, u128::MAX));
    a64.set_fpcr(1 << 12);
    a64.set_fpsr(0x2);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(28), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x2);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_simd_integer_to_float_covers_signed_unsigned_and_captured_alias() {
    let profile = GuestCpuProfile::switch_1();
    let captured = InstructionEncoding::from_u32(0x7e21_d9ad); // UCVTF S13,S13
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), captured)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar SIMD UCVTF, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(13, u128::from(16_777_217_u32) | (u128::MAX << 32)));
    assert!(a64.set_vector(5, u128::from((-3_i64) as u64) | (u128::MAX << 64)));
    assert!(a64.set_vector(4, u128::MAX));
    a64.set_fpsr(1 << 1);

    execute_one(&profile, &mut state, captured).unwrap();
    execute_one(&profile, &mut state, 0x5e61_d8a4_u32.into()).unwrap(); // SCVTF D4,D5

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(13), Some(u128::from(0x4b80_0000_u32)));
    assert_eq!(a64.vector(4), Some(u128::from((-3.0_f64).to_bits())));
    assert_eq!(a64.fpsr(), (1 << 1) | (1 << 4));
    assert_eq!(a64.pc(), 8);
}

#[test]
fn a64_scalar_simd_integer_to_float_enabled_inexact_exception_is_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x7e21_d9ad); // UCVTF S13,S13
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let original = u128::from(16_777_217_u32) | (u128::MAX << 32);
    assert!(a64.set_vector(13, original));
    a64.set_fpcr(1 << 12); // IXE
    a64.set_fpsr(0x82);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(13), Some(original));
    assert_eq!(a64.fpsr(), 0x82);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_integer_to_float_converts_all_switch1_width_combinations() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    for register in [0_u8, 2, 4, 6] {
        assert!(a64.set_vector(register, u128::MAX));
    }
    a64.write_x(x(1), 1);
    a64.write_x(x(3), u64::MAX);
    a64.write_x(x(5), 0x8000_0000);
    a64.write_x(x(7), 16_777_217);
    a64.set_fpsr(1 << 1);

    execute_one(&profile, &mut state, 0x9e63_0020_u32.into()).unwrap(); // UCVTF D0,X1
    execute_one(&profile, &mut state, 0x1e22_0062_u32.into()).unwrap(); // SCVTF S2,W3
    execute_one(&profile, &mut state, 0x1e62_00a4_u32.into()).unwrap(); // SCVTF D4,W5
    execute_one(&profile, &mut state, 0x9e23_00e6_u32.into()).unwrap(); // UCVTF S6,X7

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(1.0_f64.to_bits())));
    assert_eq!(a64.vector(2), Some(u128::from((-1.0_f32).to_bits())));
    assert_eq!(
        a64.vector(4),
        Some(u128::from((-2_147_483_648.0_f64).to_bits()))
    );
    assert_eq!(a64.vector(6), Some(u128::from(0x4b80_0000_u32)));
    assert_eq!(a64.fpsr(), (1 << 1) | (1 << 4));
}

#[test]
fn a64_scalar_integer_to_float_obeys_fpcr_rounding_and_trap_is_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let cases = [
        (0_u32, 0x4b80_0000_u32),
        (1_u32, 0x4b80_0001_u32),
        (2_u32, 0x4b80_0000_u32),
        (3_u32, 0x4b80_0000_u32),
    ];
    for (mode, expected) in cases {
        let mut state = ThreadCpuState::A64(Box::default());
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        a64.write_x(x(1), 16_777_217);
        a64.set_fpcr(mode << 22);
        execute_one(&profile, &mut state, 0x9e23_0020_u32.into()).unwrap(); // UCVTF S0,X1
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(u128::from(expected)));
        assert_eq!(a64.fpsr(), 1 << 4);
    }

    let encoding = InstructionEncoding::from_u32(0x9e23_0020); // UCVTF S0,X1
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar UCVTF, got {other:?}"),
        };
    assert_eq!(instruction_support(&decoded), InstructionSupport::Lifted);

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(1), 16_777_217);
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpcr(1 << 12);
    a64.set_fpsr(0x82);
    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x82);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_float_to_integer_converts_all_switch1_width_combinations() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from((-3.75_f32).to_bits())));
    assert!(a64.set_vector(3, u128::from(17.75_f64.to_bits())));
    assert!(a64.set_vector(5, u128::from(123.75_f32.to_bits())));
    assert!(a64.set_vector(29, u128::from(42.75_f64.to_bits())));

    execute_one(&profile, &mut state, 0x1e38_0020_u32.into()).unwrap(); // FCVTZS W0,S1
    execute_one(&profile, &mut state, 0x1e79_0066_u32.into()).unwrap(); // FCVTZU W6,D3
    execute_one(&profile, &mut state, 0x9e39_00a4_u32.into()).unwrap(); // FCVTZU X4,S5
    execute_one(&profile, &mut state, 0x9e79_03a2_u32.into()).unwrap(); // FCVTZU X2,D29

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), u64::from((-3_i32) as u32));
    assert_eq!(a64.read_x(x(6)), 17);
    assert_eq!(a64.read_x(x(4)), 123);
    assert_eq!(a64.read_x(x(2)), 42);
    assert_eq!(a64.fpsr(), 1 << 4);
}

#[test]
fn a64_scalar_float_to_integer_saturates_and_handles_subnormal_inputs() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from(f64::INFINITY.to_bits())));
    assert!(a64.set_vector(3, u128::from(f64::NAN.to_bits())));
    assert!(a64.set_vector(5, u128::from((-0.5_f64).to_bits())));

    execute_one(&profile, &mut state, 0x9e79_0020_u32.into()).unwrap(); // FCVTZU X0,D1
    execute_one(&profile, &mut state, 0x9e78_0062_u32.into()).unwrap(); // FCVTZS X2,D3
    execute_one(&profile, &mut state, 0x9e79_00a4_u32.into()).unwrap(); // FCVTZU X4,D5

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), u64::MAX);
    assert_eq!(a64.read_x(x(2)), 0);
    assert_eq!(a64.read_x(x(4)), 0);
    assert_eq!(a64.fpsr(), (1 << 0) | (1 << 4));

    a64.set_fpsr(0);
    a64.set_fpcr(1 << 24);
    assert!(a64.set_vector(1, 1)); // Smallest positive D subnormal.
    execute_one(&profile, &mut state, 0x9e79_0020_u32.into()).unwrap(); // FCVTZU X0,D1
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), 0);
    assert_eq!(a64.fpsr(), 1 << 7);
}

#[test]
fn a64_scalar_float_to_integer_enabled_exception_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x9e79_03a2); // FCVTZU X2,D29
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FCVTZU, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(29, u128::from(f64::INFINITY.to_bits())));
    a64.write_x(x(2), 0xfeed_face_cafe_beef);
    a64.set_fpcr(1 << 8);
    a64.set_fpsr(0x82);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(2)), 0xfeed_face_cafe_beef);
    assert_eq!(a64.fpsr(), 0x82);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_float_to_integer_honors_all_encoded_rounding_directions() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from(2.5_f32.to_bits())));
    assert!(a64.set_vector(5, u128::from((-2.25_f32).to_bits())));
    assert!(a64.set_vector(28, u128::from(12.75_f64.to_bits())));

    execute_one(&profile, &mut state, 0x1e20_0020_u32.into()).unwrap(); // FCVTNS W0,S1
    execute_one(&profile, &mut state, 0x1e24_0022_u32.into()).unwrap(); // FCVTAS W2,S1
    execute_one(&profile, &mut state, 0x1e28_00a3_u32.into()).unwrap(); // FCVTPS W3,S5
    execute_one(&profile, &mut state, 0x1e30_00a4_u32.into()).unwrap(); // FCVTMS W4,S5
    execute_one(&profile, &mut state, 0x9e71_0381_u32.into()).unwrap(); // FCVTMU X1,D28

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), 2);
    assert_eq!(a64.read_x(x(2)), 3);
    assert_eq!(a64.read_x(x(3)), u64::from((-2_i32) as u32));
    assert_eq!(a64.read_x(x(4)), u64::from((-3_i32) as u32));
    assert_eq!(a64.read_x(x(1)), 12);
    assert_eq!(a64.fpsr(), 1 << 4);
}

#[test]
fn a64_scalar_unsigned_rounding_checks_range_after_rounding() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from((-0.25_f64).to_bits())));
    assert!(a64.set_vector(3, u128::from((-0.5_f64).to_bits())));

    execute_one(&profile, &mut state, 0x9e69_0020_u32.into()).unwrap(); // FCVTPU X0,D1
    execute_one(&profile, &mut state, 0x9e61_0022_u32.into()).unwrap(); // FCVTNU X2,D1
    execute_one(&profile, &mut state, 0x9e65_0064_u32.into()).unwrap(); // FCVTAU X4,D3

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(0)), 0);
    assert_eq!(a64.read_x(x(2)), 0);
    assert_eq!(a64.read_x(x(4)), 0);
    assert_eq!(a64.fpsr(), (1 << 0) | (1 << 4));
}

#[test]
fn a64_scalar_float_to_integer_directional_trap_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x9e71_0381); // FCVTMU X1,D28
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FCVTMU, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(28, u128::from((-0.25_f64).to_bits())));
    a64.write_x(x(1), 0xfeed_face_cafe_beef);
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x82);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(1)), 0xfeed_face_cafe_beef);
    assert_eq!(a64.fpsr(), 0x82);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_simd_float_divide_executes_captured_four_single_lane_form() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        28,
        u128::from(6.0_f32.to_bits())
            | (u128::from((-7.0_f32).to_bits()) << 32)
            | (u128::from(1.0_f32.to_bits()) << 64)
    ));
    assert!(a64.set_vector(
        30,
        u128::from(2.0_f32.to_bits()) | (u128::from(2.0_f32.to_bits()) << 32)
    ));
    a64.set_fpsr(1 << 4);

    execute_one(&profile, &mut state, 0x6e3e_ff9c_u32.into()).unwrap(); // FDIV V28.4S,V28.4S,V30.4S
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(28),
        Some(
            u128::from(3.0_f32.to_bits())
                | (u128::from((-3.5_f32).to_bits()) << 32)
                | (u128::from(f32::INFINITY.to_bits()) << 64)
                | (u128::from(0x7fc0_0000_u32) << 96)
        )
    );
    assert_eq!(a64.fpsr(), (1 << 4) | (1 << 1) | 1);
}

#[test]
fn a64_simd_float_divide_supports_all_arrangements_and_fpcr_rounding() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());

    let expected = [0x3eaa_aaab_u32, 0x3eaa_aaab, 0x3eaa_aaaa, 0x3eaa_aaaa];
    for (rounding, expected) in expected.into_iter().enumerate() {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(
            1,
            u128::from(1.0_f32.to_bits()) | (u128::from(1.0_f32.to_bits()) << 32)
        ));
        assert!(a64.set_vector(
            2,
            u128::from(3.0_f32.to_bits()) | (u128::from(3.0_f32.to_bits()) << 32)
        ));
        assert!(a64.set_vector(0, u128::MAX));
        a64.set_fpcr((rounding as u32) << 22);
        a64.set_fpsr(0);
        execute_one(&profile, &mut state, 0x2e22_fc20_u32.into()).unwrap(); // FDIV V0.2S,V1.2S,V2.2S
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(
            a64.vector(0),
            Some(u128::from(expected) | (u128::from(expected) << 32))
        );
        assert_eq!(a64.fpsr(), 1 << 4);
    }

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        7,
        u128::from(1.0_f64.to_bits()) | (u128::from((-6.0_f64).to_bits()) << 64)
    ));
    assert!(a64.set_vector(
        8,
        u128::from(3.0_f64.to_bits()) | (u128::from(2.0_f64.to_bits()) << 64)
    ));
    a64.set_fpcr(0);
    a64.set_fpsr(0);
    execute_one(&profile, &mut state, 0x6e68_fce6_u32.into()).unwrap(); // FDIV V6.2D,V7.2D,V8.2D
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(6),
        Some(u128::from(0x3fd5_5555_5555_5555_u64) | (u128::from((-3.0_f64).to_bits()) << 64))
    );
    assert_eq!(a64.fpsr(), 1 << 4);
}

#[test]
fn a64_simd_float_divide_nan_controls_and_trap_boundary_are_precise() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x2e22_fc20); // FDIV V0.2S,V1.2S,V2.2S
    let location = source(profile, 0, ExecutionState::A64);
    let decoded = match nixe_cpu::decode::decode(&profile, location, encoding) {
        nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
        other => panic!("expected decoded FDIV vector, got {other:?}"),
    };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from(1.0_f32.to_bits())));
    assert!(a64.set_vector(2, 0));
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpcr(1 << 9); // DZE
    a64.set_fpsr(0x80);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_simd_float_divide_applies_default_nan_and_flush_controls() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        28,
        u128::from(0x7f80_0001_u32)
            | (u128::from(1_u32) << 32)
            | (u128::from(0x0080_0000_u32) << 64)
            | (u128::from(0x7f7f_ffff_u32) << 96)
    ));
    assert!(a64.set_vector(
        30,
        u128::from(1.0_f32.to_bits())
            | (u128::from(1.0_f32.to_bits()) << 32)
            | (u128::from(2.0_f32.to_bits()) << 64)
            | (u128::from(0x0080_0000_u32) << 96)
    ));
    a64.set_fpcr((1 << 25) | (1 << 24)); // DN | FZ

    execute_one(&profile, &mut state, 0x6e3e_ff9c_u32.into()).unwrap(); // FDIV V28.4S,V28.4S,V30.4S
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(
        a64.vector(28),
        Some(
            u128::from(0x7fc0_0000_u32)
                | (u128::from(0_u32) << 32)
                | (u128::from(0_u32) << 64)
                | (u128::from(f32::INFINITY.to_bits()) << 96)
        )
    );
    assert_eq!(a64.fpsr(), (1 << 7) | (1 << 4) | (1 << 3) | (1 << 2) | 1);
}

#[test]
fn a64_scalar_fmov_immediate_executes_all_allocated_precisions() {
    let profile = GuestCpuProfile::switch_1()
        .with_instruction_feature(InstructionFeature::Fp16, CapabilityStatus::Enabled);
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::MAX));
    assert!(a64.set_vector(2, u128::MAX));
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpcr(u32::MAX);
    a64.set_fpsr(0x95);

    execute_one(&profile, &mut state, 0x1e2e_101f_u32.into()).unwrap(); // FMOV S31,#1.0
    execute_one(&profile, &mut state, 0x1e78_1002_u32.into()).unwrap(); // FMOV D2,#-0.125
    execute_one(&profile, &mut state, 0x1eee_1000_u32.into()).unwrap(); // FMOV H0,#1.0

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(u128::from(1.0_f32.to_bits())));
    assert_eq!(a64.vector(2), Some(u128::from((-0.125_f64).to_bits())));
    assert_eq!(a64.vector(0), Some(0x3c00));
    assert_eq!(a64.fpcr(), u32::MAX);
    assert_eq!(a64.fpsr(), 0x95);
}

#[test]
fn a64_scalar_fmov_immediate_is_interpreter_only_and_profile_gating_is_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e2e_101f);
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded FMOV immediate, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpsr(0x80);
    let outcome = execute_one(&profile, &mut state, 0x1eee_1000_u32.into()).unwrap();
    assert!(matches!(outcome, InterpreterOutcome::ProfileDisabled(_)));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fcvt_executes_single_double_family_and_clears_upper_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(
        30,
        u128::from(1.5_f32.to_bits()) | (u128::from(u64::MAX) << 64)
    ));
    assert!(a64.set_vector(1, u128::from((-2.25_f64).to_bits())));
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpsr(0x2);

    execute_one(&profile, &mut state, 0x1e22_c3de_u32.into()).unwrap(); // FCVT D30,S30
    execute_one(&profile, &mut state, 0x1e62_4020_u32.into()).unwrap(); // FCVT S0,D1

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(u128::from(1.5_f64.to_bits())));
    assert_eq!(a64.vector(0), Some(u128::from((-2.25_f32).to_bits())));
    assert_eq!(a64.fpsr(), 0x2);
}

#[test]
fn a64_scalar_fcvt_obeys_rounding_and_reports_special_value_status() {
    let profile = GuestCpuProfile::switch_1();
    let halfway_above_one = 0x3ff0_0000_1000_0000_u64;
    let cases = [
        (0_u32, 0x3f80_0000_u32),
        (1_u32, 0x3f80_0001_u32),
        (2_u32, 0x3f80_0000_u32),
        (3_u32, 0x3f80_0000_u32),
    ];
    for (mode, expected) in cases {
        let mut state = ThreadCpuState::A64(Box::default());
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, u128::from(halfway_above_one)));
        a64.set_fpcr(mode << 22);
        execute_one(&profile, &mut state, 0x1e62_4020_u32.into()).unwrap(); // FCVT S0,D1
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(u128::from(expected)));
        assert_eq!(a64.fpsr(), 1 << 4);
    }

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from(0x7f80_0001_u32))); // signaling NaN
    a64.set_fpcr(1 << 25); // DN
    execute_one(&profile, &mut state, 0x1e22_c020_u32.into()).unwrap(); // FCVT D0,S1
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(0x7ff8_0000_0000_0000_u64)));
    assert_eq!(a64.fpsr(), 1);

    assert!(a64.set_vector(1, u128::from(0x8000_0001_u32))); // negative subnormal
    a64.set_fpcr(1 << 24); // FZ
    a64.set_fpsr(0);
    execute_one(&profile, &mut state, 0x1e22_c020_u32.into()).unwrap(); // FCVT D0,S1
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(0x8000_0000_0000_0000_u64)));
    assert_eq!(a64.fpsr(), 1 << 7);

    assert!(a64.set_vector(1, 1)); // minimum positive double subnormal
    a64.set_fpcr(0);
    a64.set_fpsr(0);
    execute_one(&profile, &mut state, 0x1e62_4020_u32.into()).unwrap(); // FCVT S0,D1
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0));
    assert_eq!(a64.fpsr(), (1 << 3) | (1 << 4));
}

#[test]
fn a64_scalar_fcvt_enabled_exception_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e22_c020); // FCVT D0,S1
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FCVT, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, u128::from(0x7f80_0001_u32))); // signaling NaN
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fdiv_executes_single_double_family_and_clears_upper_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::from(6.0_f64.to_bits()) | (u128::MAX << 64)));
    assert!(a64.set_vector(30, u128::from(2.0_f64.to_bits())));
    assert!(a64.set_vector(1, u128::from(1.0_f32.to_bits())));
    assert!(a64.set_vector(2, 0));
    a64.set_fpsr(0x80);

    execute_one(&profile, &mut state, 0x1e7e_1bff_u32.into()).unwrap(); // FDIV D31,D31,D30
    execute_one(&profile, &mut state, 0x1e22_1820_u32.into()).unwrap(); // FDIV S0,S1,S2

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(u128::from(3.0_f64.to_bits())));
    assert_eq!(a64.vector(0), Some(u128::from(f32::INFINITY.to_bits())));
    assert_eq!(a64.fpsr(), 0x80 | (1 << 1));
}

#[test]
fn a64_scalar_fdiv_enabled_exception_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e62_1820); // FDIV D0,D1,D2
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FDIV, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(1, 0));
    assert!(a64.set_vector(2, 0));
    assert!(a64.set_vector(0, u128::MAX));
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fcmp_fcmpe_execute_register_and_zero_family() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::from(3.0_f64.to_bits())));
    assert!(a64.set_vector(31, u128::from(2.0_f64.to_bits())));

    execute_one(&profile, &mut state, 0x1e7f_2010_u32.into()).unwrap(); // FCMPE D0,D31
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), 1 << 29);

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::from((-1.0_f32).to_bits())));
    execute_one(&profile, &mut state, 0x1e20_2008_u32.into()).unwrap(); // FCMP S0,#0.0
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), 1 << 31);

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::from((-0.0_f64).to_bits())));
    execute_one(&profile, &mut state, 0x1e60_2018_u32.into()).unwrap(); // FCMPE D0,#0.0
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), (1 << 30) | (1 << 29));
}

#[test]
fn a64_scalar_fcmp_nan_signaling_and_enabled_exception_are_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::from(0x7ff8_0000_0000_0001_u64))); // qNaN
    assert!(a64.set_vector(1, u128::from(1.0_f64.to_bits())));

    execute_one(&profile, &mut state, 0x1e61_2000_u32.into()).unwrap(); // FCMP D0,D1
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), (1 << 29) | (1 << 28));
    assert_eq!(a64.fpsr(), 0);

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);
    a64.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(1 << 31));
    let error = execute_one(&profile, &mut state, 0x1e61_2010_u32.into()).unwrap_err(); // FCMPE D0,D1
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), 1 << 31);
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 4);
}

#[test]
fn a64_scalar_fccmp_fccmpe_execute_both_condition_paths_and_captured_form() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(0, u128::from(0x7fc0_0001_u32))); // qNaN
    assert!(a64.set_vector(1, u128::from(1.0_f32.to_bits())));
    a64.set_fpcr(1 << 8); // IOE must not matter on the false condition path.
    a64.set_fpsr(0x80);
    a64.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(1 << 30)); // Z=1

    execute_one(&profile, &mut state, 0x1e21_140a_u32.into()).unwrap(); // FCCMP S0,S1,#10,NE
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), 0xa000_0000);
    assert_eq!(a64.fpsr(), 0x80);

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::from(1.0_f64.to_bits())));
    assert!(a64.set_vector(30, u128::from(2.0_f64.to_bits())));
    a64.set_fpcr(0);
    a64.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(0)); // NE holds.
    execute_one(&profile, &mut state, 0x1e7e_17e4_u32.into()).unwrap(); // FCCMP D31,D30,#4,NE
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), 1 << 31);
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 8);
}

#[test]
fn a64_scalar_fccmpe_enabled_exception_is_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(2, u128::from(0x7fc0_0001_u32))); // qNaN
    assert!(a64.set_vector(3, u128::from(1.0_f32.to_bits())));
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);
    a64.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits(0));

    let error = execute_one(&profile, &mut state, 0x1e23_e45f_u32.into()).unwrap_err(); // FCCMPE S2,S3,#15,AL
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.nzcv().bits(), 0);
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_frint_executes_fixed_rounding_family_and_clears_upper_bits() {
    let profile = GuestCpuProfile::switch_1();
    let cases = [
        (0x1e24_4020_u32, 2.0_f32), // FRINTN S0,S1
        (0x1e24_c020_u32, 2.0_f32), // FRINTP S0,S1
        (0x1e25_4020_u32, 1.0_f32), // FRINTM S0,S1
        (0x1e25_c020_u32, 1.0_f32), // FRINTZ S0,S1
        (0x1e26_4020_u32, 2.0_f32), // FRINTA S0,S1
    ];
    for (encoding, expected) in cases {
        let mut state = ThreadCpuState::A64(Box::default());
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, u128::from(1.5_f32.to_bits())));
        assert!(a64.set_vector(0, u128::MAX));

        execute_one(&profile, &mut state, encoding.into()).unwrap();

        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(u128::from(expected.to_bits())));
        assert_eq!(a64.fpsr(), 0);
    }

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::from((-1.25_f64).to_bits())));
    execute_one(&profile, &mut state, 0x1e65_43ff_u32.into()).unwrap(); // FRINTM D31,D31
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(u128::from((-2.0_f64).to_bits())));
}

#[test]
fn a64_scalar_frint_current_rounding_distinguishes_exact_status() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(2 << 22); // Round toward negative infinity.
    assert!(a64.set_vector(1, u128::from(1.75_f64.to_bits())));

    execute_one(&profile, &mut state, 0x1e67_4020_u32.into()).unwrap(); // FRINTX D0,D1
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(1.0_f64.to_bits())));
    assert_eq!(a64.fpsr(), 1 << 4);

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpsr(0);
    execute_one(&profile, &mut state, 0x1e67_c020_u32.into()).unwrap(); // FRINTI D0,D1
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from(1.0_f64.to_bits())));
    assert_eq!(a64.fpsr(), 0);
}

#[test]
fn a64_scalar_frint_enabled_exceptions_are_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e67_4020); // FRINTX D0,D1
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FRINTX, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 12); // IXE
    a64.set_fpsr(0x80);
    assert!(a64.set_vector(1, u128::from(1.5_f64.to_bits())));
    assert!(a64.set_vector(0, u128::MAX));

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fadd_fsub_execute_single_double_family() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::from(2.5_f64.to_bits())));
    assert!(a64.set_vector(29, u128::from(1.25_f64.to_bits()) | (u128::MAX << 64)));
    assert!(a64.set_vector(1, u128::from(5.5_f32.to_bits())));
    assert!(a64.set_vector(2, u128::from(2.0_f32.to_bits())));

    execute_one(&profile, &mut state, 0x1e7d_2bfd_u32.into()).unwrap(); // FADD D29,D31,D29
    execute_one(&profile, &mut state, 0x1e22_3820_u32.into()).unwrap(); // FSUB S0,S1,S2

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(29), Some(u128::from(3.75_f64.to_bits())));
    assert_eq!(a64.vector(0), Some(u128::from(3.5_f32.to_bits())));
    assert_eq!(a64.fpsr(), 0);
}

#[test]
fn a64_scalar_fadd_obeys_rounding_and_signed_zero_rules() {
    let profile = GuestCpuProfile::switch_1();
    let halfway = (970_u64) << 52; // 2^-53, half an ulp at 1.0.
    for (mode, expected) in [(0_u32, 1.0_f64.to_bits()), (1_u32, 1.0_f64.to_bits() + 1)] {
        let mut state = ThreadCpuState::A64(Box::default());
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        a64.set_fpcr(mode << 22);
        assert!(a64.set_vector(1, u128::from(1.0_f64.to_bits())));
        assert!(a64.set_vector(2, u128::from(halfway)));

        execute_one(&profile, &mut state, 0x1e62_2820_u32.into()).unwrap(); // FADD D0,D1,D2

        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(u128::from(expected)));
        assert_eq!(a64.fpsr(), 1 << 4);
    }

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(2 << 22); // Round toward negative infinity.
    assert!(a64.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(a64.set_vector(2, u128::from((-1.0_f64).to_bits())));
    execute_one(&profile, &mut state, 0x1e62_2820_u32.into()).unwrap(); // FADD D0,D1,D2
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::from((-0.0_f64).to_bits())));
}

#[test]
fn a64_scalar_fadd_enabled_invalid_exception_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e62_2820); // FADD D0,D1,D2
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FADD, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);
    assert!(a64.set_vector(1, u128::from(f64::INFINITY.to_bits())));
    assert!(a64.set_vector(2, u128::from(f64::NEG_INFINITY.to_bits())));
    assert!(a64.set_vector(0, u128::MAX));

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fmul_fnmul_execute_single_double_family_and_clear_upper_bits() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(29, u128::from(1.5_f64.to_bits())));
    assert!(a64.set_vector(28, u128::from(2.0_f64.to_bits())));
    assert!(a64.set_vector(7, u128::from(2.0_f32.to_bits())));
    assert!(a64.set_vector(8, u128::from((-3.0_f32).to_bits())));
    assert!(a64.set_vector(6, u128::MAX));

    execute_one(&profile, &mut state, 0x1e7c_0bbc_u32.into()).unwrap(); // FMUL D28,D29,D28
    execute_one(&profile, &mut state, 0x1e28_88e6_u32.into()).unwrap(); // FNMUL S6,S7,S8

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(28), Some(u128::from(3.0_f64.to_bits())));
    assert_eq!(a64.vector(6), Some(u128::from(6.0_f32.to_bits())));
    assert_eq!(a64.fpsr(), 0);
}

#[test]
fn a64_scalar_fmul_obeys_rounding_and_flush_to_zero_controls() {
    let profile = GuestCpuProfile::switch_1();
    let operand = 1.0_f64.to_bits() + 1;
    for (mode, expected) in [
        (0_u32, 1.0_f64.to_bits() + 2),
        (1_u32, 1.0_f64.to_bits() + 3),
    ] {
        let mut state = ThreadCpuState::A64(Box::default());
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        a64.set_fpcr(mode << 22);
        assert!(a64.set_vector(1, u128::from(operand)));

        execute_one(&profile, &mut state, 0x1e61_0820_u32.into()).unwrap(); // FMUL D0,D1,D1

        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(u128::from(expected)));
        assert_eq!(a64.fpsr(), 1 << 4);
    }

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 24);
    assert!(a64.set_vector(1, 1)); // Smallest positive D subnormal.
    assert!(a64.set_vector(2, u128::from(1.0_f64.to_bits())));
    execute_one(&profile, &mut state, 0x1e62_0820_u32.into()).unwrap(); // FMUL D0,D1,D2
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0));
    assert_eq!(a64.fpsr(), 1 << 7);
}

#[test]
fn a64_scalar_fmul_enabled_invalid_exception_is_atomic_and_interpreter_only() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e62_0820); // FMUL D0,D1,D2
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), encoding)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FMUL, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);
    assert!(a64.set_vector(1, u128::from(f64::INFINITY.to_bits())));
    assert!(a64.set_vector(2, 0));
    assert!(a64.set_vector(0, u128::MAX));

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fused_multiply_add_family_is_single_rounded_and_alias_safe() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    for (encoding, expected) in [
        (0x1f02_0c20_u32, 10.0_f32), // FMADD S0,S1,S2,S3
        (0x1f02_8c20, -2.0_f32),     // FMSUB S0,S1,S2,S3
        (0x1f22_0c20, -10.0_f32),    // FNMADD S0,S1,S2,S3
        (0x1f22_8c20, 2.0_f32),      // FNMSUB S0,S1,S2,S3
    ] {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(1, u128::from(2.0_f32.to_bits())));
        assert!(a64.set_vector(2, u128::from(3.0_f32.to_bits())));
        assert!(a64.set_vector(3, u128::from(4.0_f32.to_bits())));
        execute_one(&profile, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.vector(0), Some(u128::from(expected.to_bits())));
    }

    {
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        assert!(a64.set_vector(29, u128::from(2.0_f64.to_bits())));
        assert!(a64.set_vector(0, u128::from(3.0_f64.to_bits())));
        assert!(a64.set_vector(30, u128::from(4.0_f64.to_bits())));
    }
    execute_one(&profile, &mut state, 0x1f40_7bbe_u32.into()).unwrap(); // FMADD D30,D29,D0,D30
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(u128::from(10.0_f64.to_bits())));

    assert!(a64.set_vector(1, 0x3f80_0001));
    assert!(a64.set_vector(2, 0x3f7f_fffe));
    assert!(a64.set_vector(3, u128::from(1.0_f32.to_bits())));
    execute_one(&profile, &mut state, 0x1f02_8c20_u32.into()).unwrap(); // FMSUB S0,S1,S2,S3
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0x2880_0000));
}

#[test]
fn a64_scalar_fused_multiply_add_enabled_inexact_exception_is_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1f02_0c20); // FMADD S0,S1,S2,S3
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 12); // IXE
    a64.set_fpsr(0x80);
    assert!(a64.set_vector(0, u128::MAX));
    assert!(a64.set_vector(1, u128::from(1.0_f32.to_bits())));
    assert!(a64.set_vector(2, u128::from(1.0_f32.to_bits())));
    assert!(a64.set_vector(3, 0x3380_0000)); // 2^-24, halfway below the next f32.

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_square_root_handles_precision_rounding_aliasing_and_status() {
    let profile = GuestCpuProfile::switch_1();
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(30, u128::from(4.0_f32.to_bits())));
    assert!(a64.set_vector(1, u128::from(4.0_f64.to_bits())));
    execute_one(&profile, &mut state, 0x1e21_c3de_u32.into()).unwrap(); // FSQRT S30,S30
    execute_one(&profile, &mut state, 0x1e61_c020_u32.into()).unwrap(); // FSQRT D0,D1
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(u128::from(2.0_f32.to_bits())));
    assert_eq!(a64.vector(0), Some(u128::from(2.0_f64.to_bits())));

    assert!(a64.set_vector(1, u128::from(2.0_f32.to_bits())));
    a64.set_fpcr(1 << 22); // Round toward positive infinity.
    a64.set_fpsr(0);
    execute_one(&profile, &mut state, 0x1e21_c020_u32.into()).unwrap(); // FSQRT S0,S1
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0x3fb5_04f4));
    assert_eq!(a64.fpsr(), 1 << 4);

    assert!(a64.set_vector(1, 1)); // Smallest positive Binary32 subnormal.
    a64.set_fpcr(1 << 24); // FZ
    a64.set_fpsr(0);
    execute_one(&profile, &mut state, 0x1e21_c020_u32.into()).unwrap();
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(0));
    assert_eq!(a64.fpsr(), 1 << 7);
}

#[test]
fn a64_scalar_square_root_enabled_invalid_exception_is_atomic() {
    let profile = GuestCpuProfile::switch_1();
    let encoding = InstructionEncoding::from_u32(0x1e21_c020); // FSQRT S0,S1
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.set_fpcr(1 << 8); // IOE
    a64.set_fpsr(0x80);
    assert!(a64.set_vector(0, u128::MAX));
    assert!(a64.set_vector(1, u128::from((-1.0_f32).to_bits())));

    let error = execute_one(&profile, &mut state, encoding).unwrap_err();
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
    let ThreadCpuState::A64(a64) = state else {
        unreachable!()
    };
    assert_eq!(a64.vector(0), Some(u128::MAX));
    assert_eq!(a64.fpsr(), 0x80);
    assert_eq!(a64.pc(), 0);
}

#[test]
fn a64_scalar_fcsel_selects_exact_single_and_double_bits() {
    let profile = GuestCpuProfile::switch_1();
    let captured = InstructionEncoding::from_u32(0x1e3e_cffe); // FCSEL S30,S31,S30,GT
    let decoded =
        match nixe_cpu::decode::decode(&profile, source(profile, 0, ExecutionState::A64), captured)
        {
            nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
            other => panic!("expected decoded scalar FCSEL, got {other:?}"),
        };
    assert_eq!(
        instruction_support(&decoded),
        InstructionSupport::InterpreterOnly
    );

    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    let single_false = 0x7f80_0001_u32; // Signaling NaN payload must be copied unchanged.
    assert!(a64.set_vector(31, u128::from(0x0123_4567_u32) | (u128::MAX << 32)));
    assert!(a64.set_vector(30, u128::from(single_false) | (u128::MAX << 32)));
    a64.set_nzcv(Nzcv::from_bits(0x6000_0000)); // GT is false because Z is set.
    a64.set_fpcr(u32::MAX);
    a64.set_fpsr(0x95);

    execute_one(&profile, &mut state, captured).unwrap();

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert_eq!(a64.vector(30), Some(u128::from(single_false)));
    assert!(a64.set_vector(4, u128::from(0x8000_0000_0000_0000_u64)));
    assert!(a64.set_vector(5, u128::from(0x7ff0_0000_0000_0001_u64)));
    assert!(a64.set_vector(3, u128::MAX));
    a64.set_nzcv(Nzcv::from_bits(Nzcv::Z));

    execute_one(&profile, &mut state, 0x1e65_0c83_u32.into()).unwrap(); // FCSEL D3,D4,D5,EQ

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(3), Some(u128::from(0x8000_0000_0000_0000_u64)));
    assert_eq!(a64.fpcr(), u32::MAX);
    assert_eq!(a64.fpsr(), 0x95);
    assert_eq!(a64.pc(), 8);
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
fn a64_simd_ld1_st1_single_structure_transfers_selected_lanes() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(55);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(102);
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

    for (address, access_size, value) in [
        (0x1000, MemoryAccessSize::Byte, MemoryValue::U8(0xab)),
        (0x1010, MemoryAccessSize::Halfword, MemoryValue::U16(0x1234)),
        (
            0x1020,
            MemoryAccessSize::Word,
            MemoryValue::U32(0x89ab_cdef),
        ),
        (
            0x1030,
            MemoryAccessSize::Doubleword,
            MemoryValue::U64(0x0123_4567_89ab_cdef),
        ),
    ] {
        memory
            .write(
                SPACE,
                GuestVirtualAddress::new(address),
                MemoryAccess::normal(access_size),
                value,
            )
            .unwrap();
    }

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    for (base, address) in [(1, 0x1000), (5, 0x1010), (9, 0x1020), (13, 0x1030)] {
        a64.write_x(x(base), address);
    }
    for register in [29, 4, 8, 12] {
        assert!(a64.set_vector(register, u128::MAX));
    }

    for encoding in [
        0x0d40_183d_u32, // LD1 {V29.B}[6],[X1]
        0x0d40_58a4,     // LD1 {V4.H}[3],[X5]
        0x4d40_8128,     // LD1 {V8.S}[2],[X9]
        0x4d40_85ac,     // LD1 {V12.D}[1],[X13]
    ] {
        execute_one_with_context(context, &mut state, encoding.into()).unwrap();
    }

    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(29), Some(!(0xff << 48) | (0xab << 48)));
    assert_eq!(a64.vector(4), Some(!(0xffff << 48) | (0x1234 << 48)));
    assert_eq!(
        a64.vector(8),
        Some(!(u128::from(u32::MAX) << 64) | (0x89ab_cdef << 64))
    );
    assert_eq!(
        a64.vector(12),
        Some(!(u128::from(u64::MAX) << 64) | (u128::from(0x0123_4567_89ab_cdef_u64) << 64))
    );

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(2, u128::from(0x5a_u8) << 120));
    a64.write_x(x(3), 0x1040);
    execute_one_with_context(context, &mut state, 0x4d00_1c62_u32.into()).unwrap(); // ST1 {V2.B}[15],[X3]
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x1040),
                MemoryAccess::normal(MemoryAccessSize::Byte),
            )
            .unwrap()
            .value,
        MemoryValue::U8(0x5a)
    );
}

#[test]
fn a64_simd_single_structure_post_index_uses_immediate_or_register_offset() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(56);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(103);
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
            MemoryAccess::normal(MemoryAccessSize::Byte),
            MemoryValue::U8(0x7b),
        )
        .unwrap();
    let context =
        InterpreterContext::new(ProcessCpuContext::new(profile, SPACE)).with_memory(&memory);
    let mut state = ThreadCpuState::A64(Box::default());
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(17), 0x1000);
    a64.write_x(x(21), 0x1010);
    a64.write_x(x(22), 0x30);
    assert!(a64.set_vector(24, u128::from(0x0123_4567_89ab_cdef_u64) << 64));
    a64.write_x(x(25), 0x1020);
    a64.write_x(x(26), 0x40);

    execute_one_with_context(context, &mut state, 0x0ddf_1e30_u32.into()).unwrap(); // LD1 {V16.B}[7],[X17],#1
    execute_one_with_context(context, &mut state, 0x4d9a_8738_u32.into()).unwrap(); // ST1 {V24.D}[1],[X25],X26
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(16), Some(u128::from(0x7b_u8) << 56));
    assert_eq!(a64.read_x(x(17)), 0x1001);
    assert_eq!(a64.read_x(x(25)), 0x1060);
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x1020),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
            )
            .unwrap()
            .value,
        MemoryValue::U64(0x0123_4567_89ab_cdef)
    );
}

#[test]
fn a64_simd_ld1_st1_multiple_structures_transfer_consecutive_registers() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(52);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(99);
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
    let first = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff_u128;
    let second = 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128;
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            MemoryAccess::normal(MemoryAccessSize::Quadword),
            MemoryValue::U128(first),
        )
        .unwrap();
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x1010),
            MemoryAccess::normal(MemoryAccessSize::Quadword),
            MemoryValue::U128(second),
        )
        .unwrap();

    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    a64.write_x(x(2), 0x1000);
    // LD1 {V1.16B,V2.16B},[X2],#32: the exact instruction observed in libnx.
    execute_one_with_context(context, &mut state, 0x4cdf_a041_u32.into()).unwrap();
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(1), Some(first));
    assert_eq!(a64.vector(2), Some(second));
    assert_eq!(a64.read_x(x(2)), 0x1020);

    let stored_low = [
        0x1111_1111_1111_1111_u64,
        0x2222_2222_2222_2222,
        0x3333_3333_3333_3333,
        0x4444_4444_4444_4444,
    ];
    for (register_count, encoding) in [
        (1_u64, 0x0c9f_7020_u32),
        (2, 0x0c9f_a020),
        (3, 0x0c9f_6020),
        (4, 0x0c9f_2020),
    ] {
        let base = 0x1200 + register_count * 0x40;
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        for (register, value) in stored_low.into_iter().enumerate() {
            assert!(a64.set_vector(register as u8, u128::from(value)));
        }
        a64.write_x(x(1), base);
        execute_one_with_context(context, &mut state, encoding.into()).unwrap();
        let ThreadCpuState::A64(a64) = &state else {
            unreachable!()
        };
        assert_eq!(a64.read_x(x(1)), base + register_count * 8);
        for (index, expected) in stored_low
            .into_iter()
            .take(register_count as usize)
            .enumerate()
        {
            assert_eq!(
                memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(base + index as u64 * 8),
                        MemoryAccess::normal(MemoryAccessSize::Doubleword),
                    )
                    .unwrap()
                    .value,
                MemoryValue::U64(expected),
            );
        }
    }

    let low_first = 0x0123_4567_89ab_cdef_u64;
    let low_second = 0xfedc_ba98_7654_3210_u64;
    for (address, value) in [(0x1080, low_first), (0x1088, low_second)] {
        memory
            .write(
                SPACE,
                GuestVirtualAddress::new(address),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
                MemoryValue::U64(value),
            )
            .unwrap();
    }
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    assert!(a64.set_vector(31, u128::MAX));
    assert!(a64.set_vector(0, u128::MAX));
    a64.write_x(x(3), 0x1080);
    execute_one_with_context(context, &mut state, 0x0c40_a07f_u32.into()).unwrap(); // LD1 {V31.8B,V0.8B},[X3]
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.vector(31), Some(u128::from(low_first)));
    assert_eq!(a64.vector(0), Some(u128::from(low_second)));
    assert_eq!(
        a64.read_x(x(3)),
        0x1080,
        "no-offset form must not write back"
    );

    let stored = [
        0x1111_1111_1111_1111_0000_0000_0000_0001_u128,
        0x2222_2222_2222_2222_0000_0000_0000_0002,
        0x3333_3333_3333_3333_0000_0000_0000_0003,
        0x4444_4444_4444_4444_0000_0000_0000_0004,
    ];
    let ThreadCpuState::A64(a64) = &mut state else {
        unreachable!()
    };
    for (register, value) in [
        (30_u8, stored[0]),
        (31, stored[1]),
        (0, stored[2]),
        (1, stored[3]),
    ] {
        assert!(a64.set_vector(register, value));
    }
    a64.write_x(x(4), 0x1100);
    a64.write_x(x(5), 0x40);
    execute_one_with_context(context, &mut state, 0x4c85_2c9e_u32.into()).unwrap(); // ST1 {V30.2D-V1.2D},[X4],X5
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(4)), 0x1140);
    for (index, expected) in stored.into_iter().enumerate() {
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(0x1100 + index as u64 * 16),
                    MemoryAccess::normal(MemoryAccessSize::Quadword),
                )
                .unwrap()
                .value,
            MemoryValue::U128(expected),
        );
    }

    let error = execute_one_with_context(context, &mut state, 0x4c40_8020_u32.into()).unwrap_err(); // LD2 {V0.16B,V1.16B},[X1]
    assert!(matches!(
        error,
        InterpreterError::UnsupportedInstruction { .. }
    ));
}

#[test]
fn a64_simd_ld1_post_index_suppresses_writeback_on_data_abort() {
    const SPACE: AddressSpaceId = AddressSpaceId::new(53);
    const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(100);
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
    a64.write_x(x(2), 0x1ff0);
    let pc = a64.pc();

    let outcome = execute_one_with_context(context, &mut state, 0x4cdf_a041_u32.into()).unwrap();
    assert!(matches!(outcome, InterpreterOutcome::DataAbort { .. }));
    let ThreadCpuState::A64(a64) = &state else {
        unreachable!()
    };
    assert_eq!(a64.read_x(x(2)), 0x1ff0);
    assert_eq!(a64.pc(), pc);
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
    let monitor = std::cell::RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
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
    let decoded = match nixe_cpu::decode::decode(
        &profile,
        source(profile, 0, ExecutionState::T32),
        InstructionEncoding::from_u16(0x2001),
    ) {
        nixe_cpu::decode::DecodeResult::Decoded(decoded) => decoded,
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
        a64.read_x(nixe_cpu::state::a64::A64Register::General(
            nixe_cpu::state::a64::A64GeneralRegister::new(30).unwrap()
        )),
        4
    );
}
