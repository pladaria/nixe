use crate::{
    address::GuestVirtualAddress,
    coverage::CoverageId,
    ir::terminator::{ExceptionKind, Terminator},
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    profile::GuestCpuProfile,
    state::{ThreadCpuState, a32::A32GeneralRegister},
};

use super::{
    InstructionSupport, InterpreterError, InterpreterOutcome, InterpreterPolicy, execute_fallback,
    execute_one, instruction_support,
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

    // Advanced SIMD is deliberately unknown in the built-in profile, so this
    // recognized vector encoding must not become an implementation fallback.
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
