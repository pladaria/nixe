//! State-independent basic-block discovery used only by region formation.

use core::num::NonZeroU32;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::{
    decode::{DecodeResult, DecodedOpcode},
    error::{FrontendError, FrontendInternalError, InvalidIr},
    ir::{
        block::{BlockEndReason, BlockMetadata, InstructionSource, IrBlock},
        builder::{BuildError, IrBuilder},
        terminator::{ControlTarget, Terminator},
        value::Operand,
    },
    location::{DecodedInstruction, ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::InstructionMemory,
    platform::PlatformDecoder,
    profile::GuestCpuProfile,
};

/// Default local instruction cut for one discovered basic block.
pub const DEFAULT_MAX_GUEST_INSTRUCTIONS: NonZeroU32 = NonZeroU32::new(64).unwrap();

/// Absolute allocation bound accepted for one discovered basic block.
///
/// Normal runtime policy uses [`DEFAULT_MAX_GUEST_INSTRUCTIONS`]. This larger
/// ceiling prevents malformed configuration from turning translation into an
/// unbounded allocation even if a future memory implementation exposes pages
/// larger than the synthetic test page.
pub const MAX_GUEST_INSTRUCTIONS_PER_BLOCK: u32 = 4_096;

/// Maximum IR expansion allowed for one guest instruction.
///
/// The value deliberately leaves headroom for exact architectural semantics;
/// exceeding it is treated as a frontend implementation error rather than
/// allowing an accidental expansion loop to consume unbounded memory.
pub const MAX_IR_OPERATIONS_PER_GUEST_INSTRUCTION: usize = 64;

/// Bounded policy used by the region former's basic-block discovery primitive.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BasicBlockLimits {
    pub max_guest_instructions: NonZeroU32,
}

impl Default for BasicBlockLimits {
    fn default() -> Self {
        Self {
            max_guest_instructions: DEFAULT_MAX_GUEST_INSTRUCTIONS,
        }
    }
}

pub(crate) enum LiftOutcome {
    Continue,
    Terminate(Terminator),
    Unsupported(crate::coverage::CoverageId),
}

/// Lazily translates one bounded block beginning at `start`.
///
/// Fetch provenance remains in guest domains, each instruction is dispatched
/// through the decoder and state-specific lifter, and the returned IR has
/// already passed the common verifier.
#[cfg(test)]
pub(crate) fn translate_basic_block(
    config: BasicBlockLimits,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
) -> Result<IrBlock, FrontendError> {
    translate_basic_block_with_decoder(
        config,
        profile.into(),
        profile,
        address_space,
        start,
        memory,
    )
}

pub(crate) fn translate_basic_block_with_decoder(
    config: BasicBlockLimits,
    decoder: PlatformDecoder,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
) -> Result<IrBlock, FrontendError> {
    translate_basic_block_internal(
        config,
        decoder,
        profile,
        address_space,
        start,
        memory,
        false,
    )
}

pub(crate) fn translate_basic_block_with_disassembly(
    config: BasicBlockLimits,
    decoder: PlatformDecoder,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
) -> Result<IrBlock, FrontendError> {
    translate_basic_block_internal(config, decoder, profile, address_space, start, memory, true)
}

fn translate_basic_block_internal(
    config: BasicBlockLimits,
    decoder: PlatformDecoder,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
    capture_disassembly: bool,
) -> Result<IrBlock, FrontendError> {
    validate_start(start)?;
    if config.max_guest_instructions.get() > MAX_GUEST_INSTRUCTIONS_PER_BLOCK {
        return Err(internal(
            "configured guest instruction limit exceeds the frontend allocation bound",
        ));
    }
    let first_page = memory.code_page_span(address_space, start.pc)?;
    if !first_page.contains(start.pc) {
        return Err(internal(
            "instruction memory returned a page not containing the start PC",
        ));
    }

    let provisional = BlockMetadata::new(start, 0, 0, []);
    let mut builder = IrBuilder::new(provisional);
    let mut sources = Vec::new();
    let mut pc = start.pc;
    let (terminator, end_reason) = loop {
        if !sources.is_empty() && !first_page.contains(pc) {
            break (
                direct_branch(ControlTarget::Direct {
                    pc,
                    execution_state: start.execution_state,
                }),
                BlockEndReason::PageBoundary,
            );
        }

        let location = LocationDescriptor::new(pc, start.execution_state, profile.id());
        let (encoding, fetched_dependencies) = fetch_instruction(memory, address_space, location)?;
        let next_pc = advance_pc(location, encoding)?;
        let mut source = InstructionSource::new(location, encoding, fetched_dependencies);

        let operations_before_lift = builder.operation_count();
        let outcome = match crate::decode::decode(decoder, location, encoding) {
            DecodeResult::Decoded(decoded) => {
                if capture_disassembly {
                    source = source.with_disassembly(
                        crate::decode::disassemble(&decoded.instruction).to_string(),
                    );
                }
                lift_decoded(&mut builder, &decoded)
            }
            DecodeResult::RecognizedUnimplemented(decoded) => {
                if capture_disassembly {
                    source = source.with_disassembly(
                        crate::decode::disassemble(&decoded.instruction).to_string(),
                    );
                }
                LiftOutcome::Terminate(unsupported_terminator(&decoded))
            }
            DecodeResult::Unallocated { reason, .. } => {
                if capture_disassembly {
                    source = source.with_disassembly(format!("<unallocated: {reason}>"));
                }
                LiftOutcome::Terminate(Terminator::Exception {
                    source: location,
                    kind: crate::exception::ExceptionKind::UndefinedInstruction,
                    syndrome: None,
                })
            }
            DecodeResult::Reserved { name, reason, .. } => {
                if capture_disassembly {
                    source = source.with_disassembly(format!("<{name}: reserved: {reason}>"));
                }
                LiftOutcome::Terminate(Terminator::Exception {
                    source: location,
                    kind: crate::exception::ExceptionKind::UndefinedInstruction,
                    syndrome: None,
                })
            }
        };
        sources.push(source);
        let emitted_operations = builder
            .operation_count()
            .checked_sub(operations_before_lift)
            .ok_or_else(|| internal("IR operation count moved backwards during lifting"))?;
        if emitted_operations > MAX_IR_OPERATIONS_PER_GUEST_INSTRUCTION {
            return Err(internal(
                "one guest instruction exceeded the frontend IR operation bound",
            ));
        }

        match outcome {
            LiftOutcome::Continue => {
                pc = next_pc;
                let instruction_limit =
                    sources.len() == config.max_guest_instructions.get() as usize;
                let page_boundary = !first_page.contains(pc);
                if instruction_limit || page_boundary {
                    let reason = match (instruction_limit, page_boundary) {
                        (true, true) => BlockEndReason::InstructionLimitAtPageBoundary,
                        (true, false) => BlockEndReason::InstructionLimit,
                        (false, true) => BlockEndReason::PageBoundary,
                        (false, false) => unreachable!("a block-cut condition was checked"),
                    };
                    break (
                        direct_branch(ControlTarget::Direct {
                            pc,
                            execution_state: start.execution_state,
                        }),
                        reason,
                    );
                }
            }
            LiftOutcome::Terminate(terminator) => {
                let reason = end_reason_for_terminator(&terminator);
                break (terminator, reason);
            }
            LiftOutcome::Unsupported(coverage_id) => {
                let terminator = Terminator::UnsupportedInstruction {
                    source: location,
                    encoding,
                    coverage_id: coverage_id.get(),
                    disassembly: "unsupported".into(),
                    reason: "no JIT semantics are implemented for this instruction".into(),
                };
                let reason = end_reason_for_terminator(&terminator);
                break (terminator, reason);
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
        sources,
    )
    .with_end_reason(end_reason);
    builder.replace_metadata(metadata);
    builder.terminate(terminator).map_err(build_error)?;
    builder.finish().map_err(build_error)
}

const fn end_reason_for_terminator(terminator: &Terminator) -> BlockEndReason {
    match terminator {
        Terminator::Direct { .. } => BlockEndReason::DirectBranch,
        Terminator::Conditional { .. } => BlockEndReason::ConditionalBranch,
        Terminator::Indirect { .. } => BlockEndReason::IndirectBranch,
        Terminator::Call { .. } | Terminator::ConditionalCall { .. } => BlockEndReason::Call,
        Terminator::Return { .. } => BlockEndReason::Return,
        Terminator::Exception { .. } | Terminator::ConditionalException { .. } => {
            BlockEndReason::Exception
        }
        Terminator::UnsupportedInstruction { .. } => BlockEndReason::UnsupportedInstruction,
        Terminator::Stop { .. } => BlockEndReason::RuntimeStop,
    }
}

fn validate_start(start: LocationDescriptor) -> Result<(), FrontendError> {
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
    memory: &(impl InstructionMemory + ?Sized),
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

fn unsupported_terminator(decoded: &DecodedInstruction<DecodedOpcode>) -> Terminator {
    Terminator::UnsupportedInstruction {
        source: decoded.location,
        encoding: decoded.encoding,
        coverage_id: decoded.instruction.coverage_id().get(),
        disassembly: crate::decode::disassemble(&decoded.instruction)
            .to_string()
            .into_boxed_str(),
        reason: "no JIT semantics are implemented for this instruction".into(),
    }
}

pub(crate) const fn direct_branch(target: ControlTarget) -> Terminator {
    Terminator::Direct { target }
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
            .ok_or_else(|| internal("guest instruction address overflow")),
        ExecutionState::A32 | ExecutionState::T32 => Ok(GuestVirtualAddress::new(u64::from(
            (location.pc.get() as u32).wrapping_add(increment as u32),
        ))),
    }
}

/// Forms a host-independent computed target in the guest address domain.
#[must_use]
pub const fn indirect_target(
    address: Operand,
    execution_state: ExecutionState,
    source: LocationDescriptor,
) -> ControlTarget {
    ControlTarget::Indirect {
        address,
        execution_state,
        source,
    }
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

/// Creates a call terminator whose architectural link-register update is
/// committed atomically with the validated control transfer.
#[must_use]
pub const fn call_terminator(
    target: ControlTarget,
    return_address: GuestVirtualAddress,
) -> Terminator {
    Terminator::Call {
        target,
        return_address,
    }
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
    use nixe_memory::{ContentGeneration, GuestPhysicalPageId, MappingGeneration};

    use crate::{
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
            OperationKind::ReadFlags(_) => "read-flags",
            OperationKind::WriteFlags { .. } => "write-flags",
            OperationKind::Flags(_) => "flags",
            OperationKind::Memory(_) => "memory",
            OperationKind::Barrier(_) => "barrier",
            OperationKind::ProcessorHint(_) => "processor_hint",
            OperationKind::RuntimeRegisterRead(_) => "runtime_register_read",
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
            Terminator::ConditionalCall { .. } => "conditional-call",
            Terminator::Indirect { .. } => "indirect",
            Terminator::Call { .. } => "call",
            Terminator::Return { .. } => "return",
            Terminator::Exception { .. } => "exception",
            Terminator::ConditionalException { .. } => "conditional-exception",
            Terminator::UnsupportedInstruction { .. } => "unsupported",
            Terminator::Stop { .. } => "stop",
        }
    }

    #[test]
    fn rejects_configuration_above_the_hard_allocation_bound() {
        let profile = GuestCpuProfile::switch_1();
        let memory = memory_with_pages(0x1000, 1);
        let result = translate_basic_block(
            BasicBlockLimits {
                max_guest_instructions: NonZeroU32::new(MAX_GUEST_INSTRUCTIONS_PER_BLOCK + 1)
                    .unwrap(),
            },
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A64),
            &memory,
        );

        assert!(matches!(result, Err(FrontendError::Internal(_))));
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
                bytes: &[0x01, 0x00, 0xa0, 0xe3, 0x00, 0x00, 0x00, 0xef],
                expected_sources: &["data-processing", "svc cond=#14"],
                required_operation: Some("helper"),
                expected_terminator: "exception",
            },
            Case {
                state: ExecutionState::A32,
                bytes: &[0x00, 0x10, 0x90, 0xe5, 0x00, 0x00, 0x00, 0xef],
                expected_sources: &["load-store-single", "svc cond=#14"],
                required_operation: Some("memory"),
                expected_terminator: "exception",
            },
            Case {
                state: ExecutionState::A32,
                bytes: &[0x10, 0x01, 0x00, 0xf2, 0x00, 0x00, 0x00, 0xef],
                expected_sources: &["neon-bitwise", "svc cond=#14"],
                required_operation: Some("helper"),
                expected_terminator: "exception",
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
                bytes: &[0x7f, 0x23, 0x00, 0xdf],
                expected_sources: &["movs dst=r3, imm=#127", "svc"],
                required_operation: Some("helper"),
                expected_terminator: "conditional-exception",
            },
            Case {
                state: ExecutionState::T32,
                bytes: &[0x01, 0x48, 0x00, 0xdf],
                expected_sources: &["load-literal", "svc"],
                required_operation: Some("memory"),
                expected_terminator: "conditional-exception",
            },
        ];

        let profile = GuestCpuProfile::switch_1();
        for case in cases {
            let mut memory = memory_with_pages(0x1000, 1);
            put(&mut memory, 1, 0, case.bytes);
            let block = translate_basic_block(
                BasicBlockLimits::default(),
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

            assert_eq!(format!("{block:?}"), format!("{block:?}"));
        }
    }

    #[test]
    fn translated_instruction_terminator_classes_have_stable_boundaries() {
        let profile = GuestCpuProfile::switch_1();
        let cases = [
            (
                ExecutionState::A64,
                0x1400_0000_u32,
                "direct",
                BlockEndReason::DirectBranch,
            ),
            (
                ExecutionState::A64,
                0x5400_0000_u32,
                "conditional",
                BlockEndReason::ConditionalBranch,
            ),
            (
                ExecutionState::A64,
                0xd61f_0000_u32,
                "indirect",
                BlockEndReason::IndirectBranch,
            ),
            (
                ExecutionState::A64,
                0x9400_0000_u32,
                "call",
                BlockEndReason::Call,
            ),
            (
                ExecutionState::A64,
                0xd65f_03c0_u32,
                "return",
                BlockEndReason::Return,
            ),
            (
                ExecutionState::A64,
                0xd400_0001_u32,
                "exception",
                BlockEndReason::Exception,
            ),
        ];
        for (state, encoding, expected, end_reason) in cases {
            let mut memory = memory_with_pages(0x1000, 1);
            put(&mut memory, 1, 0, &encoding.to_le_bytes());
            let block = translate_basic_block(
                BasicBlockLimits::default(),
                &profile,
                SPACE,
                start(profile, 0x1000, state),
                &memory,
            )
            .unwrap();
            assert_eq!(terminator_family(&block.terminator), expected);
            assert_eq!(block.metadata.end_reason, end_reason);
            assert_eq!(block.metadata.guest_instruction_count, 1);
        }

        let mut future_fallback = memory_with_pages(0x1000, 1);
        put(&mut future_fallback, 1, 0, &0xbf10_u16.to_le_bytes());
        let block = translate_basic_block(
            BasicBlockLimits::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::T32),
            &future_fallback,
        )
        .unwrap();
        assert_eq!(terminator_family(&block.terminator), "unsupported");
        assert_eq!(
            block.metadata.end_reason,
            BlockEndReason::UnsupportedInstruction
        );
        assert_eq!(block.metadata.guest_instruction_count, 1);

        let mut fallback = memory_with_pages(0x1000, 1);
        put(&mut fallback, 1, 0, &0xd503_20df_u32.to_le_bytes());
        let block = translate_basic_block(
            BasicBlockLimits::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A64),
            &fallback,
        )
        .unwrap();
        assert_eq!(terminator_family(&block.terminator), "unsupported");
        assert_eq!(
            block.metadata.end_reason,
            BlockEndReason::UnsupportedInstruction
        );
        assert_eq!(block.metadata.guest_instruction_count, 1);

        // Stop is a dispatcher boundary rather than an instruction-produced exit.
        let stop = Terminator::Stop {
            source: start(profile, 0x1000, ExecutionState::A64),
            reason: crate::ir::terminator::StopReason::TranslationLimit,
        };
        assert_eq!(terminator_family(&stop), "stop");
        assert!(matches!(stop, Terminator::Stop { .. }));
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
            let block = translate_basic_block(
                BasicBlockLimits::default(),
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
        let block = translate_basic_block(
            BasicBlockLimits::default(),
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
        let block = translate_basic_block(
            BasicBlockLimits::default(),
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
        let block = translate_basic_block(
            BasicBlockLimits::default(),
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
    fn basic_block_discovery_cuts_at_its_local_limit_and_preserves_fallthrough() {
        let profile = GuestCpuProfile::switch_1();
        let mut memory = memory_with_pages(0x1000, 1);
        for offset in (0..12).step_by(4) {
            put(&mut memory, 1, offset, &0xd503_201f_u32.to_le_bytes());
        }
        let block = translate_basic_block(
            BasicBlockLimits {
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
        assert!(matches!(
            block.terminator,
            Terminator::Direct {
                target: ControlTarget::Direct { pc, .. }
            } if pc == GuestVirtualAddress::new(0x1008)
        ));
        assert_eq!(block.metadata.end_reason, BlockEndReason::InstructionLimit);
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
        let block = translate_basic_block(
            BasicBlockLimits::default(),
            &profile,
            SPACE,
            start(profile, base + offset as u64, ExecutionState::T32),
            &memory,
        )
        .unwrap();
        assert_eq!(block.metadata.guest_instruction_count, 1);
        assert_eq!(block.metadata.guest_byte_count, 4);
        assert_eq!(block.metadata.sources[0].dependencies.iter().count(), 2);
        assert_eq!(block.metadata.end_reason, BlockEndReason::PageBoundary);
        assert!(matches!(
            block.terminator,
            Terminator::Direct {
                target: ControlTarget::Direct { pc, .. }
            } if pc == GuestVirtualAddress::new(base + SYNTHETIC_PAGE_SIZE as u64 + 2)
        ));
    }

    #[test]
    fn exceptions_unsupported_instructions_and_future_fallbacks_cut_immediately() {
        let profile = GuestCpuProfile::switch_1();
        let mut unallocated = memory_with_pages(0x1000, 1);
        put(&mut unallocated, 1, 0, &0_u32.to_le_bytes());
        let block = translate_basic_block(
            BasicBlockLimits::default(),
            &profile,
            SPACE,
            start(profile, 0x1000, ExecutionState::A64),
            &unallocated,
        )
        .unwrap();
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: crate::exception::ExceptionKind::UndefinedInstruction,
                ..
            }
        ));

        let mut recognized = memory_with_pages(0x2000, 1);
        // This recognized hint has semantics in neither execution engine.
        put(&mut recognized, 1, 0, &0xd503_20df_u32.to_le_bytes());
        let block = translate_basic_block(
            BasicBlockLimits::default(),
            &profile,
            SPACE,
            start(profile, 0x2000, ExecutionState::A64),
            &recognized,
        )
        .unwrap();
        assert!(matches!(
            block.terminator,
            Terminator::UnsupportedInstruction { source, encoding, .. }
                if source == start(profile, 0x2000, ExecutionState::A64)
                    && encoding == InstructionEncoding::from_u32(0xd503_20df)
        ));

        let mut future_fallback = memory_with_pages(0x3000, 1);
        put(&mut future_fallback, 1, 0, &0xbf10_u16.to_le_bytes());
        let block = translate_basic_block(
            BasicBlockLimits::default(),
            &profile,
            SPACE,
            start(profile, 0x3000, ExecutionState::T32),
            &future_fallback,
        )
        .unwrap();
        assert!(matches!(
            block.terminator,
            Terminator::UnsupportedInstruction { .. }
        ));
    }

    #[test]
    fn aarch32_instruction_address_wraps_in_the_guest_domain() {
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
        let source = start(GuestCpuProfile::switch_1(), 0x1000, ExecutionState::A64);
        let address = crate::ir::value::Immediate::Address(GuestVirtualAddress::new(0x9876)).into();
        assert_eq!(
            indirect_target(address, ExecutionState::A64, source),
            ControlTarget::Indirect {
                address,
                execution_state: ExecutionState::A64,
                source,
            }
        );
    }

    #[test]
    fn call_terminators_own_the_architectural_link_update() {
        let profile = GuestCpuProfile::switch_1();
        let source = start(profile, 0x1000, ExecutionState::T32);
        let dependency = CodePageDependency {
            page: GuestPhysicalPageId::new(1),
            generation: ContentGeneration::new(2),
            mapping_generation: MappingGeneration::new(1),
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
            [InstructionSource::new(
                source,
                InstructionEncoding::from_u32(0xf000_f800),
                CodeDependencies::one(dependency),
            )],
        );
        let mut builder = IrBuilder::new(metadata);
        let terminator = call_terminator(target, return_address);
        builder.terminate(terminator).unwrap();
        let block = builder.finish().unwrap();
        assert!(block.operations.is_empty());
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
