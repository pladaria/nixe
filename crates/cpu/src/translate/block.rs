//! State-independent translation-block formation and cut policy.

use core::{fmt, num::NonZeroU32};

use crate::{
    address::{AddressSpaceId, GuestVirtualAddress},
    decode::{DecodeResult, DecodedOpcode, OperandId, OperandValue},
    error::{FrontendError, FrontendInternalError, InvalidIr},
    ir::{
        block::{BlockExit, BlockExitKind, BlockMetadata, InstructionSource, IrBlock},
        builder::{BuildError, IrBuilder},
        op::{OperationKind, StateRegister},
        terminator::{ControlTarget, Terminator},
        value::{Immediate, Operand},
    },
    location::{DecodedInstruction, ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::InstructionMemory,
    profile::GuestCpuProfile,
    state::{a32::A32GeneralRegister, a64::A64GeneralRegister},
};

/// Default maximum number of guest instructions in one translation block.
pub const DEFAULT_MAX_GUEST_INSTRUCTIONS: NonZeroU32 = NonZeroU32::new(64).unwrap();

/// Bounded policy used by the lazy block translator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlockTranslationConfig {
    pub max_guest_instructions: NonZeroU32,
}

impl Default for BlockTranslationConfig {
    fn default() -> Self {
        Self {
            max_guest_instructions: DEFAULT_MAX_GUEST_INSTRUCTIONS,
        }
    }
}

/// Failure to calculate an architectural instruction or branch address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AddressCalculationError {
    AddressOverflow,
    MissingBranchDisplacement,
    MisalignedTarget {
        target: GuestVirtualAddress,
        execution_state: ExecutionState,
    },
}

impl fmt::Display for AddressCalculationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddressOverflow => formatter.write_str("guest instruction address overflow"),
            Self::MissingBranchDisplacement => {
                formatter.write_str("decoded branch has no signed displacement")
            }
            Self::MisalignedTarget {
                target,
                execution_state,
            } => write!(
                formatter,
                "target {target} is misaligned for {execution_state}"
            ),
        }
    }
}

impl std::error::Error for AddressCalculationError {}

pub(crate) enum LiftOutcome {
    Continue,
    Terminate(Terminator),
    Interpret(crate::coverage::CoverageId),
}

/// Lazily translates one bounded block beginning at `start`.
///
/// Fetch provenance remains in guest domains, each instruction is dispatched
/// through the decoder and state-specific lifter, and the returned IR has
/// already passed the common verifier.
pub fn translate_block(
    config: BlockTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &impl InstructionMemory,
) -> Result<IrBlock, FrontendError> {
    validate_start(profile, start)?;
    let first_page = memory.code_page_span(address_space, start.pc)?;
    if !first_page.contains(start.pc) {
        return Err(internal(
            "instruction memory returned a page not containing the start PC",
        ));
    }

    let provisional = BlockMetadata::new(start, 0, 0, [], [], []);
    let mut builder = IrBuilder::new(provisional);
    let mut sources = Vec::new();
    let mut dependencies = Vec::new();
    let mut pc = start.pc;
    let terminator = loop {
        if !sources.is_empty() && !first_page.contains(pc) {
            break direct_branch(ControlTarget::Direct {
                pc,
                execution_state: start.execution_state,
            });
        }

        let location = LocationDescriptor::new(pc, start.execution_state, profile.id());
        let (encoding, fetched_dependencies) = fetch_instruction(memory, address_space, location)?;
        let next_pc = advance_pc(location, encoding)?;
        for dependency in fetched_dependencies.iter() {
            if !dependencies.contains(&dependency) {
                dependencies.push(dependency);
            }
        }
        sources.push(InstructionSource::new(
            location,
            encoding,
            fetched_dependencies,
        ));

        let outcome = match crate::decode::decode(profile, location, encoding) {
            DecodeResult::Decoded(decoded) => lift_decoded(&mut builder, &decoded),
            DecodeResult::RecognizedUnimplemented(decoded) => {
                if crate::interpreter::has_semantics(&decoded) {
                    LiftOutcome::Terminate(interpret_terminator(&decoded))
                } else {
                    LiftOutcome::Terminate(unsupported_terminator(
                        &decoded,
                        "neither the lifter nor interpreter implements this instruction",
                    ))
                }
            }
            DecodeResult::Unallocated { .. }
            | DecodeResult::Reserved { .. }
            | DecodeResult::ProfileDisabled { .. } => {
                LiftOutcome::Terminate(Terminator::Exception {
                    source: location,
                    kind: crate::ir::terminator::ExceptionKind::UndefinedInstruction,
                    syndrome: None,
                })
            }
        };

        match outcome {
            LiftOutcome::Continue => {
                pc = next_pc;
                if sources.len() == config.max_guest_instructions.get() as usize
                    || !first_page.contains(pc)
                {
                    break direct_branch(ControlTarget::Direct {
                        pc,
                        execution_state: start.execution_state,
                    });
                }
            }
            LiftOutcome::Terminate(terminator) => break terminator,
            LiftOutcome::Interpret(coverage_id) => {
                let decoded = match crate::decode::decode(profile, location, encoding) {
                    DecodeResult::Decoded(decoded)
                    | DecodeResult::RecognizedUnimplemented(decoded) => decoded,
                    _ => unreachable!("lifter outcome requires a decoded instruction"),
                };
                break if crate::interpreter::has_semantics(&decoded) {
                    Terminator::InterpretOne {
                        source: location,
                        encoding,
                        coverage_id: coverage_id.get(),
                    }
                } else {
                    unsupported_terminator(
                        &decoded,
                        "lifter requested fallback but interpreter semantics are unavailable",
                    )
                };
            }
        }
    };

    let guest_byte_count = sources.iter().try_fold(0_u32, |total, source| {
        total.checked_add(u32::from(source.encoding.size().bytes()))
    });
    let guest_instruction_count = u32::try_from(sources.len()).ok();
    let metadata = BlockMetadata::new(
        start,
        guest_byte_count.ok_or_else(|| internal("translated byte count overflow"))?,
        guest_instruction_count.ok_or_else(|| internal("translated instruction count overflow"))?,
        exits_for_terminator(&terminator),
        dependencies,
        sources,
    );
    builder.replace_metadata(metadata);
    builder.terminate(terminator).map_err(build_error)?;
    builder.finish().map_err(build_error)
}

fn validate_start(
    profile: &GuestCpuProfile,
    start: LocationDescriptor,
) -> Result<(), FrontendError> {
    if start.profile_id != profile.id() {
        return Err(internal(
            "block start profile does not match the selected profile",
        ));
    }
    if !profile
        .allowed_execution_states()
        .contains(start.execution_state)
    {
        return Err(internal(
            "block start execution state is disabled by the profile",
        ));
    }
    if !start.is_aligned() {
        return Err(internal("block start PC is not instruction aligned"));
    }
    if matches!(
        start.execution_state,
        ExecutionState::A32 | ExecutionState::T32
    ) && start.pc.get() > u64::from(u32::MAX)
    {
        return Err(internal(
            "A32/T32 block start lies outside the 32-bit address domain",
        ));
    }
    Ok(())
}

fn fetch_instruction(
    memory: &impl InstructionMemory,
    address_space: AddressSpaceId,
    location: LocationDescriptor,
) -> Result<(InstructionEncoding, crate::memory::CodeDependencies), FrontendError> {
    match location.execution_state {
        ExecutionState::A64 | ExecutionState::A32 => {
            let fetched = memory.fetch32(address_space, location.pc)?;
            Ok((
                InstructionEncoding::from_u32(fetched.bits),
                fetched.dependencies,
            ))
        }
        ExecutionState::T32 => {
            let first = memory.fetch16(address_space, location.pc)?;
            if location.execution_state.instruction_size(first.bits)
                == crate::location::InstructionSize::Bits16
            {
                Ok((
                    InstructionEncoding::from_u16(first.bits),
                    first.dependencies,
                ))
            } else {
                let fetched = memory.fetch_t32_32(address_space, location.pc)?;
                Ok((
                    InstructionEncoding::from_u32(fetched.bits),
                    fetched.dependencies,
                ))
            }
        }
    }
}

fn lift_decoded(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    match decoded.location.execution_state {
        ExecutionState::A64 => super::a64::lift(builder, decoded),
        ExecutionState::A32 => super::a32::lift(builder, decoded),
        ExecutionState::T32 => super::t32::lift(builder, decoded),
    }
}

fn interpret_terminator(decoded: &DecodedInstruction<DecodedOpcode>) -> Terminator {
    Terminator::InterpretOne {
        source: decoded.location,
        encoding: decoded.encoding,
        coverage_id: decoded.instruction.coverage_id().get(),
    }
}

fn unsupported_terminator(
    decoded: &DecodedInstruction<DecodedOpcode>,
    reason: impl Into<Box<str>>,
) -> Terminator {
    Terminator::UnsupportedInstruction {
        source: decoded.location,
        encoding: decoded.encoding,
        coverage_id: decoded.instruction.coverage_id().get(),
        disassembly: crate::decode::disassemble(&decoded.instruction)
            .to_string()
            .into(),
        reason: reason.into(),
    }
}

pub(crate) const fn direct_branch(target: ControlTarget) -> Terminator {
    Terminator::Direct { target }
}

fn exits_for_terminator(terminator: &Terminator) -> Vec<BlockExit> {
    let target = |kind, target: &ControlTarget| BlockExit {
        kind,
        target: match target {
            ControlTarget::Direct { pc, .. } => Some(*pc),
            ControlTarget::Indirect { .. } | ControlTarget::A32Interworking { .. } => None,
        },
    };
    match terminator {
        Terminator::Direct {
            target: destination,
        } => {
            vec![target(BlockExitKind::Direct, destination)]
        }
        Terminator::Conditional {
            taken, fallthrough, ..
        } => vec![
            target(BlockExitKind::ConditionalTaken, taken),
            target(BlockExitKind::ConditionalFallthrough, fallthrough),
        ],
        Terminator::Indirect {
            target: destination,
        } => {
            vec![target(BlockExitKind::Indirect, destination)]
        }
        Terminator::Call {
            target: destination,
            ..
        } => {
            vec![target(BlockExitKind::Call, destination)]
        }
        Terminator::Return {
            target: destination,
        } => {
            vec![target(BlockExitKind::Return, destination)]
        }
        Terminator::Exception { .. } => vec![BlockExit {
            kind: BlockExitKind::Exception,
            target: None,
        }],
        Terminator::InterpretOne { .. } => vec![BlockExit {
            kind: BlockExitKind::Interpreter,
            target: None,
        }],
        Terminator::UnsupportedInstruction { .. } => vec![BlockExit {
            kind: BlockExitKind::UnsupportedInstruction,
            target: None,
        }],
        Terminator::Stop { .. } => vec![BlockExit {
            kind: BlockExitKind::Stop,
            target: None,
        }],
    }
}

fn advance_pc(
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> Result<GuestVirtualAddress, FrontendError> {
    let increment = u64::from(encoding.size().bytes());
    match location.execution_state {
        ExecutionState::A64 => location
            .pc
            .checked_add(increment)
            .ok_or_else(|| internal(AddressCalculationError::AddressOverflow.to_string())),
        ExecutionState::A32 | ExecutionState::T32 => Ok(GuestVirtualAddress::new(u64::from(
            (location.pc.get() as u32).wrapping_add(increment as u32),
        ))),
    }
}

/// Calculates an immediate branch target using the current execution state's
/// PC bias, address width, scaling already applied by the decoder, and target
/// alignment.
pub fn direct_branch_target(
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<ControlTarget, AddressCalculationError> {
    let OperandValue::Signed(displacement) = decoded
        .instruction
        .operands()
        .get(OperandId::Immediate)
        .ok_or(AddressCalculationError::MissingBranchDisplacement)?
    else {
        return Err(AddressCalculationError::MissingBranchDisplacement);
    };
    let state = decoded.location.execution_state;
    let pc = decoded.location.pc;
    let target = match state {
        ExecutionState::A64 => pc.wrapping_offset(displacement),
        ExecutionState::A32 => GuestVirtualAddress::new(u64::from(
            (pc.get() as u32)
                .wrapping_add(8)
                .wrapping_add(displacement as u32),
        )),
        ExecutionState::T32 => GuestVirtualAddress::new(u64::from(
            (pc.get() as u32)
                .wrapping_add(4)
                .wrapping_add(displacement as u32),
        )),
    };
    if !state.is_instruction_address_aligned(target) {
        return Err(AddressCalculationError::MisalignedTarget {
            target,
            execution_state: state,
        });
    }
    Ok(ControlTarget::Direct {
        pc: target,
        execution_state: state,
    })
}

/// Converts an AArch32 BX-like raw address into an aligned A32 or T32 target.
pub fn a32_interworking_target(
    raw_address: GuestVirtualAddress,
) -> Result<ControlTarget, AddressCalculationError> {
    if raw_address.get() > u64::from(u32::MAX) {
        return Err(AddressCalculationError::AddressOverflow);
    }
    let raw = raw_address.get() as u32;
    let (address, execution_state) = if raw & 1 != 0 {
        (raw & !1, ExecutionState::T32)
    } else {
        (raw & !3, ExecutionState::A32)
    };
    Ok(ControlTarget::Direct {
        pc: GuestVirtualAddress::new(u64::from(address)),
        execution_state,
    })
}

/// Forms a host-independent computed target in the guest address domain.
#[must_use]
pub const fn indirect_target(address: Operand, execution_state: ExecutionState) -> ControlTarget {
    ControlTarget::Indirect {
        address,
        execution_state,
    }
}

/// Forms a computed A32/T32 interworking target. Its guest address bit zero is
/// deliberately interpreted by the backend, never converted to a host pointer.
#[must_use]
pub const fn indirect_interworking_target(address: Operand) -> ControlTarget {
    ControlTarget::A32Interworking { address }
}

/// Creates a conditional exit with both CFG successors retained explicitly.
#[must_use]
pub const fn conditional_terminator(
    condition: Operand,
    taken: ControlTarget,
    fallthrough: ControlTarget,
) -> Terminator {
    Terminator::Conditional {
        condition,
        taken,
        fallthrough,
    }
}

/// Emits the architectural link-register update and returns the corresponding
/// call terminator. Targets remain guest addresses and carry their destination
/// execution state.
pub fn emit_call(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    target: ControlTarget,
    return_address: GuestVirtualAddress,
) -> Result<Terminator, BuildError> {
    let (register, value) = match source.execution_state {
        ExecutionState::A64 => (
            StateRegister::A64X(A64GeneralRegister::new(30).unwrap()),
            Immediate::I64(return_address.get()),
        ),
        ExecutionState::A32 => (
            StateRegister::A32R(A32GeneralRegister::new(14).unwrap()),
            Immediate::I32(return_address.get() as u32),
        ),
        ExecutionState::T32 => (
            StateRegister::A32R(A32GeneralRegister::new(14).unwrap()),
            Immediate::I32((return_address.get() as u32) | 1),
        ),
    };
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register,
            value: value.into(),
        },
    )?;
    Ok(Terminator::Call {
        target,
        return_address,
    })
}

fn build_error(error: BuildError) -> FrontendError {
    FrontendError::InvalidIr(InvalidIr::new(None, error.to_string()))
}

fn internal(reason: impl Into<Box<str>>) -> FrontendError {
    FrontendError::Internal(FrontendInternalError::new(None, reason))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{CodeGeneration, GuestPhysicalPageId},
        ir::{
            op::{Condition, FlagOperation, OperationKind, StateRegister},
            terminator::ControlTarget,
        },
        memory::{
            CodeDependencies, CodePageDependency, MemoryPermissions, SYNTHETIC_PAGE_SIZE,
            SyntheticMemory,
        },
    };

    const SPACE: AddressSpaceId = AddressSpaceId::new(3);

    fn memory_with_pages(base: u64, pages: usize) -> SyntheticMemory {
        let mut memory = SyntheticMemory::new();
        for index in 0..pages {
            let page = GuestPhysicalPageId::new(index as u64 + 1);
            assert!(memory.add_ram_page(page));
            assert!(memory.map_page(
                SPACE,
                GuestVirtualAddress::new(base + index as u64 * SYNTHETIC_PAGE_SIZE as u64),
                page,
                MemoryPermissions::READ_EXECUTE,
            ));
        }
        memory
    }

    fn put(memory: &mut SyntheticMemory, page: u64, offset: usize, bytes: &[u8]) {
        assert!(memory.initialize_ram(GuestPhysicalPageId::new(page), offset, bytes));
    }

    fn start(profile: GuestCpuProfile, pc: u64, state: ExecutionState) -> LocationDescriptor {
        LocationDescriptor::new(GuestVirtualAddress::new(pc), state, profile.id())
    }

    fn operation_family(kind: &OperationKind) -> &'static str {
        match kind {
            OperationKind::Constant(_) => "constant",
            OperationKind::Scalar(_) => "scalar",
            OperationKind::Address(_) => "address",
            OperationKind::ReadState(_) => "read-state",
            OperationKind::WriteState { .. } => "write-state",
            OperationKind::Flags(_) => "flags",
            OperationKind::Memory(_) => "memory",
            OperationKind::Barrier(_) => "barrier",
            OperationKind::CacheMaintenance(_) => "cache-maintenance",
            OperationKind::Exclusive(_) => "exclusive",
            OperationKind::Atomic(_) => "atomic",
            OperationKind::Vector(_) => "vector",
            OperationKind::FloatingPoint(_) => "floating-point",
            OperationKind::Helper(_) => "helper",
        }
    }

    fn terminator_family(terminator: &Terminator) -> &'static str {
        match terminator {
            Terminator::Direct { .. } => "direct",
            Terminator::Conditional { .. } => "conditional",
            Terminator::Indirect { .. } => "indirect",
            Terminator::Call { .. } => "call",
            Terminator::Return { .. } => "return",
            Terminator::Exception { .. } => "exception",
            Terminator::InterpretOne { .. } => "interpret-one",
            Terminator::UnsupportedInstruction { .. } => "unsupported",
            Terminator::Stop { .. } => "stop",
        }
    }

    // The encodings and expected instruction names trace Arm DDI 0602 and
    // DDI 0597. See `crates/cpu/tests/README.md`. The projection intentionally
    // excludes temporary SSA IDs while retaining semantic operation families.
    #[test]
    fn disassembly_and_ir_goldens_cover_every_implemented_lifter_family() {
        struct Case {
            state: ExecutionState,
            bytes: &'static [u8],
            expected_sources: &'static [&'static str],
            required_operation: Option<&'static str>,
            expected_terminator: &'static str,
        }

        let cases = [
            Case {
                state: ExecutionState::A64,
                bytes: &[0x00, 0x00, 0x00, 0x14],
                expected_sources: &["b imm=#0"],
                required_operation: None,
                expected_terminator: "direct",
            },
            Case {
                state: ExecutionState::A64,
                bytes: &[
                    0xbf, 0x3b, 0x03, 0xd5, // dmb ish
                    0x01, 0x00, 0x00, 0xd4, // svc #0
                ],
                expected_sources: &["barrier", "svc"],
                required_operation: Some("barrier"),
                expected_terminator: "exception",
            },
            Case {
                state: ExecutionState::A64,
                bytes: &[
                    0x23, 0x44, 0x00, 0x91, // add x3,x1,#17
                    0x01, 0x00, 0x00, 0xd4,
                ],
                expected_sources: &["add-sub-immediate", "svc"],
                required_operation: Some("scalar"),
                expected_terminator: "exception",
            },
            Case {
                state: ExecutionState::A64,
                bytes: &[
                    0x20, 0x00, 0x40, 0xf9, // ldr x0,[x1]
                    0x01, 0x00, 0x00, 0xd4,
                ],
                expected_sources: &["load-store-unsigned", "svc"],
                required_operation: Some("memory"),
                expected_terminator: "exception",
            },
            Case {
                state: ExecutionState::A64,
                bytes: &[
                    0x00, 0x1c, 0x20, 0x4e, // and v0.16b,v0.16b,v0.16b
                    0x01, 0x00, 0x00, 0xd4,
                ],
                expected_sources: &["simd-bitwise", "svc"],
                required_operation: Some("helper"),
                expected_terminator: "exception",
            },
            Case {
                state: ExecutionState::A32,
                bytes: &[0xff, 0xff, 0xff, 0xea],
                expected_sources: &["b imm=#-4, cond=#14"],
                required_operation: None,
                expected_terminator: "direct",
            },
            Case {
                state: ExecutionState::A32,
                bytes: &[0x01, 0x00, 0xa0, 0xe3],
                expected_sources: &["data-processing"],
                required_operation: None,
                expected_terminator: "interpret-one",
            },
            Case {
                state: ExecutionState::A32,
                bytes: &[0x00, 0x10, 0x90, 0xe5],
                expected_sources: &["load-store-single"],
                required_operation: None,
                expected_terminator: "interpret-one",
            },
            Case {
                state: ExecutionState::A32,
                bytes: &[0x10, 0x01, 0x00, 0xf2],
                expected_sources: &["neon-bitwise"],
                required_operation: None,
                expected_terminator: "interpret-one",
            },
            Case {
                state: ExecutionState::T32,
                bytes: &[0xff, 0xe7],
                expected_sources: &["b imm=#-2"],
                required_operation: Some("flags"),
                expected_terminator: "conditional",
            },
            Case {
                state: ExecutionState::T32,
                bytes: &[0x7f, 0x23],
                expected_sources: &["movs dst=r3, imm=#127"],
                required_operation: None,
                expected_terminator: "interpret-one",
            },
            Case {
                state: ExecutionState::T32,
                bytes: &[0x01, 0x48],
                expected_sources: &["load-literal"],
                required_operation: None,
                expected_terminator: "interpret-one",
            },
        ];

        let profile = GuestCpuProfile::switch_1();
        for case in cases {
            let mut memory = memory_with_pages(0x1000, 1);
            put(&mut memory, 1, 0, case.bytes);
            let block = translate_block(
                BlockTranslationConfig::default(),
                &profile,
                SPACE,
                start(profile, 0x1000, case.state),
                &memory,
            )
            .unwrap();
            let sources = block
                .metadata
                .sources
                .iter()
                .map(|source| {
                    let decoded =
                        match crate::decode::decode(&profile, source.location, source.encoding) {
                            DecodeResult::Decoded(decoded)
                            | DecodeResult::RecognizedUnimplemented(decoded) => decoded,
                            result => panic!("golden source no longer decodes: {result:?}"),
                        };
                    crate::decode::disassemble(&decoded.instruction).to_string()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                sources.iter().map(String::as_str).collect::<Vec<_>>(),
                case.expected_sources,
                "state={}",
                case.state
            );
            assert_eq!(
                terminator_family(&block.terminator),
                case.expected_terminator,
                "state={} sources={sources:?}",
                case.state
            );
            if let Some(required) = case.required_operation {
                assert!(
                    block
                        .operations
                        .iter()
                        .any(|operation| operation_family(&operation.kind) == required),
                    "state={} sources={sources:?} lacks {required} IR",
                    case.state
                );
            } else {
                assert!(
                    block.operations.is_empty(),
                    "state={} sources={sources:?} unexpectedly emitted IR",
                    case.state
                );
            }

            let printed =
                crate::ir::print::print_block(&block, crate::ir::print::IrPrintOptions::default());
            assert_eq!(
                printed,
                crate::ir::print::print_block(&block, crate::ir::print::IrPrintOptions::default())
            );
        }
    }

    #[test]
    fn translated_instruction_terminator_classes_have_stable_boundaries() {
        let profile = GuestCpuProfile::switch_1();
        let cases = [
            (ExecutionState::A64, 0x1400_0000_u32, "direct"),
            (ExecutionState::A64, 0x5400_0000_u32, "conditional"),
            (ExecutionState::A64, 0xd61f_0000_u32, "indirect"),
            (ExecutionState::A64, 0x9400_0000_u32, "call"),
            (ExecutionState::A64, 0xd65f_03c0_u32, "return"),
            (ExecutionState::A64, 0xd400_0001_u32, "exception"),
        ];
        for (state, encoding, expected) in cases {
            let mut memory = memory_with_pages(0x1000, 1);
            put(&mut memory, 1, 0, &encoding.to_le_bytes());
            let block = translate_block(
                BlockTranslationConfig::default(),
                &profile,
                SPACE,
                start(profile, 0x1000, state),
                &memory,
            )
            .unwrap();
            assert_eq!(terminator_family(&block.terminator), expected);
            assert_eq!(block.metadata.guest_instruction_count, 1);
        }

        for (encoding, expected) in [(0x2001_u16, "interpret-one"), (0x0000, "interpret-one")] {
            let mut memory = memory_with_pages(0x1000, 1);
            put(&mut memory, 1, 0, &encoding.to_le_bytes());
            let block = translate_block(
                BlockTranslationConfig::default(),
                &profile,
                SPACE,
                start(profile, 0x1000, ExecutionState::T32),
                &memory,
            )
            .unwrap();
            assert_eq!(terminator_family(&block.terminator), expected);
            assert_eq!(block.metadata.guest_instruction_count, 1);
        }

        let mut unsupported = memory_with_pages(0x1000, 1);
        put(&mut unsupported, 1, 0, &0xd503_20df_u32.to_le_bytes());
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A64),
            &unsupported,
        )
        .unwrap();
        assert_eq!(terminator_family(&block.terminator), "unsupported");
        assert_eq!(block.metadata.guest_instruction_count, 1);

        // Stop is a dispatcher boundary rather than an instruction-produced
        // exit, but its block metadata classification is still part of the IR
        // contract.
        let stop = Terminator::Stop {
            source: start(profile, 0x1000, ExecutionState::A64),
            reason: crate::ir::terminator::StopReason::TranslationLimit,
        };
        assert_eq!(terminator_family(&stop), "stop");
        assert!(matches!(
            exits_for_terminator(&stop).as_slice(),
            [BlockExit {
                kind: BlockExitKind::Stop,
                target: None,
            }]
        ));
    }

    #[test]
    fn translates_each_execution_state_and_calculates_direct_targets() {
        let profile = GuestCpuProfile::switch_1();
        let cases = [
            (ExecutionState::A64, 0x1400_0002_u32, 0x1008),
            (ExecutionState::A32, 0xeaff_ffff_u32, 0x1004),
        ];
        for (state, encoding, expected) in cases {
            let mut memory = memory_with_pages(0x1000, 1);
            put(&mut memory, 1, 0, &encoding.to_le_bytes());
            let block = translate_block(
                BlockTranslationConfig::default(),
                &profile,
                SPACE,
                start(profile, 0x1000, state),
                &memory,
            )
            .unwrap();
            assert_eq!(block.metadata.guest_instruction_count, 1);
            assert!(matches!(
                block.terminator,
                Terminator::Direct {
                    target: ControlTarget::Direct { pc, execution_state }
                } if pc == GuestVirtualAddress::new(expected) && execution_state == state
            ));
        }

        let mut memory = memory_with_pages(0x1000, 1);
        put(&mut memory, 1, 0, &0xe001_u16.to_le_bytes());
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::T32),
            &memory,
        )
        .unwrap();
        assert!(matches!(
            block.terminator,
            Terminator::Conditional {
                taken: ControlTarget::Direct {
                    pc,
                    execution_state: ExecutionState::T32,
                },
                fallthrough: ControlTarget::Direct {
                    pc: fallthrough,
                    execution_state: ExecutionState::T32,
                },
                ..
            } if pc == GuestVirtualAddress::new(0x1006)
                && fallthrough == GuestVirtualAddress::new(0x1002)
        ));
    }

    #[test]
    fn a32_branch_condition_is_an_explicit_cpsr_consumer() {
        let profile = GuestCpuProfile::switch_1();
        let mut memory = memory_with_pages(0x1000, 1);
        put(&mut memory, 1, 0, &0x1a00_0000_u32.to_le_bytes()); // b.ne +0
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A32),
            &memory,
        )
        .unwrap();

        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::ReadState(StateRegister::A32Cpsr)
        )));
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Flags(FlagOperation::Evaluate {
                condition: Condition::Ne,
                ..
            })
        )));
        assert!(matches!(
            block.terminator,
            Terminator::Conditional {
                taken: ControlTarget::Direct { pc, .. },
                fallthrough: ControlTarget::Direct { pc: fallthrough, .. },
                ..
            } if pc == GuestVirtualAddress::new(0x1008)
                && fallthrough == GuestVirtualAddress::new(0x1004)
        ));
    }

    #[test]
    fn t32_itstate_is_explicitly_installed_consumed_and_advanced() {
        let profile = GuestCpuProfile::switch_1();
        let mut memory = memory_with_pages(0x1000, 1);
        put(&mut memory, 1, 0, &0xbf08_u16.to_le_bytes()); // it eq
        put(&mut memory, 1, 2, &0xe001_u16.to_le_bytes()); // b +2
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::T32),
            &memory,
        )
        .unwrap();

        assert_eq!(block.metadata.guest_instruction_count, 2);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(
                    operation.kind,
                    OperationKind::WriteState {
                        register: StateRegister::A32Cpsr,
                        ..
                    }
                ))
                .count(),
            2
        );
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Flags(FlagOperation::EvaluateEncoded {
                nv_is_unconditional: false,
                ..
            })
        )));
        assert!(matches!(block.terminator, Terminator::Conditional { .. }));
    }

    #[test]
    fn cuts_at_limit_and_annotates_budget_and_fallthrough() {
        let profile = GuestCpuProfile::switch_1();
        let mut memory = memory_with_pages(0x1000, 1);
        for offset in (0..12).step_by(4) {
            put(&mut memory, 1, offset, &0xd503_201f_u32.to_le_bytes());
        }
        let block = translate_block(
            BlockTranslationConfig {
                max_guest_instructions: NonZeroU32::new(2).unwrap(),
            },
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A64),
            &memory,
        )
        .unwrap();
        assert_eq!(block.metadata.guest_instruction_count, 2);
        assert_eq!(block.metadata.guest_byte_count, 8);
        assert_eq!(block.metadata.budget_safepoint.guest_instruction_cost, 2);
        assert_eq!(
            block.metadata.exits[0].target,
            Some(GuestVirtualAddress::new(0x1008))
        );
    }

    #[test]
    fn page_cut_records_dependencies_and_allows_cross_page_t32_completion() {
        let profile = GuestCpuProfile::switch_1();
        let base = 0x4000;
        let mut memory = memory_with_pages(base, 2);
        let offset = SYNTHETIC_PAGE_SIZE - 2;
        put(&mut memory, 1, offset, &0xf3af_u16.to_le_bytes());
        put(&mut memory, 2, 0, &0x8000_u16.to_le_bytes());
        put(&mut memory, 2, 2, &0xbf00_u16.to_le_bytes());
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, base + offset as u64, ExecutionState::T32),
            &memory,
        )
        .unwrap();
        assert_eq!(block.metadata.guest_instruction_count, 1);
        assert_eq!(block.metadata.guest_byte_count, 4);
        assert_eq!(block.metadata.code_dependencies.len(), 2);
        assert_eq!(
            block.metadata.exits[0].target,
            Some(GuestVirtualAddress::new(
                base + SYNTHETIC_PAGE_SIZE as u64 + 2
            ))
        );
    }

    #[test]
    fn exceptions_unsupported_and_interpreter_fallbacks_cut_immediately() {
        let profile = GuestCpuProfile::switch_1();
        let mut unallocated = memory_with_pages(0x1000, 1);
        put(&mut unallocated, 1, 0, &0_u32.to_le_bytes());
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A64),
            &unallocated,
        )
        .unwrap();
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: crate::ir::terminator::ExceptionKind::UndefinedInstruction,
                ..
            }
        ));

        let mut recognized = memory_with_pages(0x2000, 1);
        // A recognized architectural hint whose precise behavior is not yet
        // among the frontend's supported scheduling hints.
        put(&mut recognized, 1, 0, &0xd503_20df_u32.to_le_bytes());
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x2000, ExecutionState::A64),
            &recognized,
        )
        .unwrap();
        assert!(matches!(
            block.terminator,
            Terminator::UnsupportedInstruction {
                source,
                encoding,
                ref disassembly,
                ..
            } if source == start(profile, 0x2000, ExecutionState::A64)
                && encoding == InstructionEncoding::from_u32(0xd503_20df)
                && !disassembly.is_empty()
        ));

        let mut interpreter_only = memory_with_pages(0x3000, 1);
        put(&mut interpreter_only, 1, 0, &0x2001_u16.to_le_bytes());
        let block = translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            start(profile, 0x3000, ExecutionState::T32),
            &interpreter_only,
        )
        .unwrap();
        assert!(matches!(block.terminator, Terminator::InterpretOne { .. }));
    }

    #[test]
    fn aarch32_wrap_and_interworking_keep_guest_state_explicit() {
        let a32 = a32_interworking_target(GuestVirtualAddress::new(0x2002)).unwrap();
        let t32 = a32_interworking_target(GuestVirtualAddress::new(0x2003)).unwrap();
        assert_eq!(
            a32,
            ControlTarget::Direct {
                pc: GuestVirtualAddress::new(0x2000),
                execution_state: ExecutionState::A32,
            }
        );
        assert_eq!(
            t32,
            ControlTarget::Direct {
                pc: GuestVirtualAddress::new(0x2002),
                execution_state: ExecutionState::T32,
            }
        );

        let wrapped = advance_pc(
            start(
                GuestCpuProfile::switch_1(),
                0xffff_fffc,
                ExecutionState::A32,
            ),
            InstructionEncoding::from_u32(0xe320_f000),
        )
        .unwrap();
        assert_eq!(wrapped, GuestVirtualAddress::new(0));
    }

    #[test]
    fn indirect_targets_never_contain_host_pointers() {
        let address = crate::ir::value::Immediate::Address(GuestVirtualAddress::new(0x9876)).into();
        assert_eq!(
            indirect_target(address, ExecutionState::A64),
            ControlTarget::Indirect {
                address,
                execution_state: ExecutionState::A64,
            }
        );
        assert_eq!(
            indirect_interworking_target(address),
            ControlTarget::A32Interworking { address }
        );
    }

    #[test]
    fn calls_write_the_architectural_link_register_before_terminating() {
        let profile = GuestCpuProfile::switch_1();
        let source = start(profile, 0x1000, ExecutionState::T32);
        let dependency = CodePageDependency {
            page: GuestPhysicalPageId::new(1),
            generation: CodeGeneration::new(2),
        };
        let return_address = GuestVirtualAddress::new(0x1004);
        let target = ControlTarget::Direct {
            pc: GuestVirtualAddress::new(0x2000),
            execution_state: ExecutionState::A32,
        };
        let metadata = BlockMetadata::new(
            source,
            4,
            1,
            [BlockExit {
                kind: BlockExitKind::Call,
                target: Some(GuestVirtualAddress::new(0x2000)),
            }],
            [dependency],
            [InstructionSource::new(
                source,
                InstructionEncoding::from_u32(0xf000_f800),
                CodeDependencies::one(dependency),
            )],
        );
        let mut builder = IrBuilder::new(metadata);
        let terminator = emit_call(&mut builder, source, target, return_address).unwrap();
        builder.terminate(terminator).unwrap();
        let block = builder.finish().unwrap();
        assert!(matches!(
            block.operations[0].kind,
            OperationKind::WriteState {
                register: StateRegister::A32R(register),
                value: Operand::Immediate(Immediate::I32(0x1005)),
            } if register.index() == 14
        ));
        assert!(matches!(
            block.terminator,
            Terminator::Call {
                target: ControlTarget::Direct {
                    execution_state: ExecutionState::A32,
                    ..
                },
                return_address: address,
            } if address == return_address
        ));
    }
}
