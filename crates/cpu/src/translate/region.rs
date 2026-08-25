//! Deterministic bounded multi-block region formation.

use core::num::NonZeroU32;
use std::collections::VecDeque;

use nixe_memory::AddressSpaceId;

use crate::{
    error::{FrontendError, FrontendInternalError},
    ir::{
        block::{BlockId, IrBlock},
        region::{
            IrRegion, RegionEntry, RegionExit, RegionExitKind, RegionMetadata, RegionSafepoint,
            RegionSafepointKind,
        },
        terminator::{ControlTarget, Terminator},
        verify::verify_region,
    },
    location::LocationDescriptor,
    memory::{CodePageDependency, InstructionMemory},
    profile::GuestCpuProfile,
};

use super::block::{
    BasicBlockLimits, MAX_GUEST_INSTRUCTIONS_PER_BLOCK, translate_basic_block,
    translate_basic_block_with_disassembly,
};

pub const DEFAULT_MAX_BLOCKS_PER_REGION: NonZeroU32 = NonZeroU32::new(32).unwrap();
pub const DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_REGION: NonZeroU32 = NonZeroU32::new(256).unwrap();
pub const DEFAULT_MAX_GUEST_BYTES_PER_REGION: NonZeroU32 = NonZeroU32::new(16 * 1024).unwrap();
pub const DEFAULT_MAX_IR_OPERATIONS_PER_REGION: NonZeroU32 = NonZeroU32::new(16 * 1024).unwrap();
pub const DEFAULT_MAX_CODE_DEPENDENCIES_PER_REGION: NonZeroU32 = NonZeroU32::new(64).unwrap();
pub const DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_BASIC_BLOCK: NonZeroU32 =
    super::block::DEFAULT_MAX_GUEST_INSTRUCTIONS;

pub const MAX_BLOCKS_PER_REGION: u32 = 256;
pub const MAX_GUEST_INSTRUCTIONS_PER_REGION: u32 = 4_096;
pub const MAX_GUEST_BYTES_PER_REGION: u32 = 1024 * 1024;
pub const MAX_IR_OPERATIONS_PER_REGION: u32 = 256 * 1024;
pub const MAX_CODE_DEPENDENCIES_PER_REGION: u32 = 4_096;
pub const MAX_GUEST_INSTRUCTIONS_PER_BASIC_BLOCK: u32 = MAX_GUEST_INSTRUCTIONS_PER_BLOCK;

/// All work and retained metadata limits for one region formation request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegionTranslationConfig {
    /// Maximum number of retained basic blocks.
    pub max_blocks: NonZeroU32,
    /// Maximum sum of guest instructions across retained blocks.
    pub max_guest_instructions: NonZeroU32,
    /// Maximum sum of translated guest bytes across retained blocks.
    pub max_guest_bytes: NonZeroU32,
    /// Maximum sum of IR operations across retained blocks.
    pub max_ir_operations: NonZeroU32,
    /// Maximum distinct physical code dependencies.
    pub max_code_dependencies: NonZeroU32,
    /// Local discovery cut which prevents one straight-line block monopolizing a region.
    pub max_guest_instructions_per_block: NonZeroU32,
}

impl Default for RegionTranslationConfig {
    fn default() -> Self {
        Self {
            max_blocks: DEFAULT_MAX_BLOCKS_PER_REGION,
            max_guest_instructions: DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_REGION,
            max_guest_bytes: DEFAULT_MAX_GUEST_BYTES_PER_REGION,
            max_ir_operations: DEFAULT_MAX_IR_OPERATIONS_PER_REGION,
            max_code_dependencies: DEFAULT_MAX_CODE_DEPENDENCIES_PER_REGION,
            max_guest_instructions_per_block: DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_BASIC_BLOCK,
        }
    }
}

/// Forms the sole production frontend compilation unit.
pub fn translate_region(
    config: RegionTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
) -> Result<IrRegion, FrontendError> {
    translate_region_internal(config, profile, address_space, start, memory, false)
}

pub(crate) fn translate_region_with_disassembly(
    config: RegionTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
) -> Result<IrRegion, FrontendError> {
    translate_region_internal(config, profile, address_space, start, memory, true)
}

fn translate_region_internal(
    config: RegionTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
    capture_disassembly: bool,
) -> Result<IrRegion, FrontendError> {
    validate_config(config)?;
    let mut blocks = Vec::new();
    let mut pending = VecDeque::from([start]);
    let mut rejected = Vec::new();

    while let Some(location) = pending.pop_front() {
        if block_for_location(&blocks, location).is_some() || rejected.contains(&location) {
            continue;
        }
        cut_blocks_at_entry(
            &mut blocks,
            config,
            profile,
            address_space,
            location,
            memory,
            capture_disassembly,
        )?;
        let (instruction_count, byte_count, operation_count, dependencies) =
            region_totals(&blocks)?;
        if blocks.len() >= config.max_blocks.get() as usize {
            rejected.push(location);
            continue;
        }
        let remaining_instructions = config
            .max_guest_instructions
            .get()
            .saturating_sub(instruction_count);
        if remaining_instructions == 0 {
            rejected.push(location);
            continue;
        }
        let per_block = config
            .max_guest_instructions_per_block
            .get()
            .min(remaining_instructions);
        let block_config = BasicBlockLimits {
            max_guest_instructions: NonZeroU32::new(per_block).unwrap(),
        };
        let candidate_result = if capture_disassembly {
            translate_basic_block_with_disassembly(
                block_config,
                profile,
                address_space,
                location,
                memory,
            )
        } else {
            translate_basic_block(block_config, profile, address_space, location, memory)
        };
        let candidate = match candidate_result {
            Ok(candidate) => candidate,
            Err(FrontendError::InstructionFetch(_)) if !blocks.is_empty() => {
                rejected.push(location);
                continue;
            }
            Err(error) => return Err(error),
        };
        let candidate_operations = u32::try_from(candidate.operations.len())
            .map_err(|_| internal("basic-block IR operation count overflow"))?;
        let mut candidate_dependencies = dependencies.clone();
        extend_dependencies(&mut candidate_dependencies, &candidate);
        let fits = instruction_count
            .checked_add(candidate.metadata.guest_instruction_count)
            .is_some_and(|value| value <= config.max_guest_instructions.get())
            && byte_count
                .checked_add(candidate.metadata.guest_byte_count)
                .is_some_and(|value| value <= config.max_guest_bytes.get())
            && operation_count
                .checked_add(candidate_operations)
                .is_some_and(|value| value <= config.max_ir_operations.get())
            && candidate_dependencies.len() <= config.max_code_dependencies.get() as usize;
        if !fits {
            if blocks.is_empty() {
                return Err(internal(
                    "configured region bounds cannot contain the entry basic block",
                ));
            }
            rejected.push(location);
            continue;
        }

        for successor in discoverable_successors(&candidate.terminator, profile) {
            if block_for_location(&blocks, successor).is_none() && !pending.contains(&successor) {
                pending.push_back(successor);
            }
        }
        blocks.push(candidate);
    }

    if blocks.is_empty() {
        return Err(internal("region formation produced no entry block"));
    }

    let (instruction_count, byte_count, operation_count, dependencies) = region_totals(&blocks)?;
    internalize_edges(&mut blocks, profile);
    let entries = entries(&blocks)?;
    let exits = external_exits(&blocks)?;
    let safepoints = safepoints(&blocks, &entries)?;
    let metadata = RegionMetadata {
        start,
        guest_byte_count: byte_count,
        guest_instruction_count: instruction_count,
        ir_operation_count: operation_count,
        entries: entries.into_boxed_slice(),
        exits: exits.into_boxed_slice(),
        code_dependencies: dependencies.into_boxed_slice(),
        safepoints: safepoints.into_boxed_slice(),
    };
    let region = IrRegion::new(metadata, blocks);
    verify_region(&region).map_err(|error| {
        FrontendError::InvalidIr(crate::error::InvalidIr::new(None, error.to_string()))
    })?;
    Ok(region)
}

fn cut_blocks_at_entry(
    blocks: &mut [IrBlock],
    config: RegionTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    entry: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
    capture_disassembly: bool,
) -> Result<(), FrontendError> {
    for block in blocks {
        let Some(instruction_index) = block
            .metadata
            .sources
            .iter()
            .position(|source| source.location == entry)
        else {
            continue;
        };
        if instruction_index == 0 {
            continue;
        }
        let max_guest_instructions = NonZeroU32::new(
            u32::try_from(instruction_index)
                .map_err(|_| internal("basic-block entry split index overflow"))?,
        )
        .ok_or_else(|| internal("basic-block entry split produced an empty prefix"))?;
        let block_config = BasicBlockLimits {
            max_guest_instructions: max_guest_instructions
                .min(config.max_guest_instructions_per_block),
        };
        let prefix = if capture_disassembly {
            translate_basic_block_with_disassembly(
                block_config,
                profile,
                address_space,
                block.metadata.start,
                memory,
            )?
        } else {
            translate_basic_block(
                block_config,
                profile,
                address_space,
                block.metadata.start,
                memory,
            )?
        };
        if !matches!(
            prefix.terminator,
            Terminator::Direct {
                target: ControlTarget::Direct { pc, execution_state }
            } if pc == entry.pc && execution_state == entry.execution_state
        ) {
            return Err(internal(
                "retranslated entry prefix did not end at the requested guest location",
            ));
        }
        *block = prefix;
    }
    Ok(())
}

fn region_totals(
    blocks: &[IrBlock],
) -> Result<(u32, u32, u32, Vec<CodePageDependency>), FrontendError> {
    let mut instructions = 0_u32;
    let mut bytes = 0_u32;
    let mut operations = 0_u32;
    let mut dependencies = Vec::new();
    for block in blocks {
        instructions = instructions
            .checked_add(block.metadata.guest_instruction_count)
            .ok_or_else(|| internal("region instruction count overflow"))?;
        bytes = bytes
            .checked_add(block.metadata.guest_byte_count)
            .ok_or_else(|| internal("region byte count overflow"))?;
        operations = operations
            .checked_add(
                u32::try_from(block.operations.len())
                    .map_err(|_| internal("region operation count overflow"))?,
            )
            .ok_or_else(|| internal("region operation count overflow"))?;
        extend_dependencies(&mut dependencies, block);
    }
    Ok((instructions, bytes, operations, dependencies))
}

fn validate_config(config: RegionTranslationConfig) -> Result<(), FrontendError> {
    let valid = config.max_blocks.get() <= MAX_BLOCKS_PER_REGION
        && config.max_guest_instructions.get() <= MAX_GUEST_INSTRUCTIONS_PER_REGION
        && config.max_guest_bytes.get() <= MAX_GUEST_BYTES_PER_REGION
        && config.max_ir_operations.get() <= MAX_IR_OPERATIONS_PER_REGION
        && config.max_code_dependencies.get() <= MAX_CODE_DEPENDENCIES_PER_REGION
        && config.max_guest_instructions_per_block.get() <= MAX_GUEST_INSTRUCTIONS_PER_BLOCK;
    if valid {
        Ok(())
    } else {
        Err(internal(
            "configured region limit exceeds a frontend allocation bound",
        ))
    }
}

fn block_for_location(blocks: &[IrBlock], location: LocationDescriptor) -> Option<BlockId> {
    blocks
        .iter()
        .position(|block| block.metadata.start == location)
        .and_then(|index| u32::try_from(index).ok())
        .map(BlockId::new)
}

fn direct_location(
    target: &ControlTarget,
    profile: &GuestCpuProfile,
) -> Option<LocationDescriptor> {
    let ControlTarget::Direct {
        pc,
        execution_state,
    } = target
    else {
        return None;
    };
    Some(LocationDescriptor::new(*pc, *execution_state, profile.id()))
}

fn discoverable_successors(
    terminator: &Terminator,
    profile: &GuestCpuProfile,
) -> Vec<LocationDescriptor> {
    let mut result = Vec::new();
    let mut push = |target: &ControlTarget| {
        if let Some(location) = direct_location(target, profile) {
            result.push(location);
        }
    };
    match terminator {
        Terminator::Direct { target } => push(target),
        Terminator::Conditional {
            taken, fallthrough, ..
        } => {
            push(fallthrough);
            push(taken);
        }
        Terminator::ConditionalCall { fallthrough, .. }
        | Terminator::ConditionalException { fallthrough, .. } => push(fallthrough),
        Terminator::Indirect { .. }
        | Terminator::Call { .. }
        | Terminator::Return { .. }
        | Terminator::Exception { .. }
        | Terminator::InterpretOne { .. }
        | Terminator::UnsupportedInstruction { .. }
        | Terminator::Stop { .. } => {}
    }
    result
}

fn internalize_target(
    target: &mut ControlTarget,
    starts: &[(LocationDescriptor, BlockId)],
    profile: &GuestCpuProfile,
) {
    let Some(location) = direct_location(target, profile) else {
        return;
    };
    if let Some((_, block)) = starts.iter().find(|(start, _)| *start == location) {
        *target = ControlTarget::Internal { block: *block };
    }
}

fn internalize_edges(blocks: &mut [IrBlock], profile: &GuestCpuProfile) {
    let starts: Vec<_> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            (
                block.metadata.start,
                BlockId::new(u32::try_from(index).expect("bounded region block index")),
            )
        })
        .collect();
    for block in blocks {
        match &mut block.terminator {
            Terminator::Direct { target } => internalize_target(target, &starts, profile),
            Terminator::Conditional {
                taken, fallthrough, ..
            } => {
                internalize_target(taken, &starts, profile);
                internalize_target(fallthrough, &starts, profile);
            }
            Terminator::ConditionalCall { fallthrough, .. }
            | Terminator::ConditionalException { fallthrough, .. } => {
                internalize_target(fallthrough, &starts, profile);
            }
            Terminator::Indirect { .. }
            | Terminator::Call { .. }
            | Terminator::Return { .. }
            | Terminator::Exception { .. }
            | Terminator::InterpretOne { .. }
            | Terminator::UnsupportedInstruction { .. }
            | Terminator::Stop { .. } => {}
        }
    }
}

fn entries(blocks: &[IrBlock]) -> Result<Vec<RegionEntry>, FrontendError> {
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            Ok(RegionEntry {
                location: block.metadata.start,
                block: BlockId::new(
                    u32::try_from(index).map_err(|_| internal("region block index overflow"))?,
                ),
            })
        })
        .collect()
}

fn external_exits(blocks: &[IrBlock]) -> Result<Vec<RegionExit>, FrontendError> {
    let mut exits = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let id = BlockId::new(
            u32::try_from(index).map_err(|_| internal("region block index overflow"))?,
        );
        let target = |exits: &mut Vec<RegionExit>, kind, target: &ControlTarget| {
            if !matches!(target, ControlTarget::Internal { .. }) {
                exits.push(RegionExit {
                    block: id,
                    kind,
                    target: Some(*target),
                });
            }
        };
        match &block.terminator {
            Terminator::Direct {
                target: destination,
            } => target(&mut exits, RegionExitKind::Direct, destination),
            Terminator::Conditional {
                taken, fallthrough, ..
            } => {
                target(&mut exits, RegionExitKind::ConditionalTaken, taken);
                target(
                    &mut exits,
                    RegionExitKind::ConditionalFallthrough,
                    fallthrough,
                );
            }
            Terminator::ConditionalCall {
                target: destination,
                fallthrough,
                ..
            } => {
                target(&mut exits, RegionExitKind::Call, destination);
                target(
                    &mut exits,
                    RegionExitKind::ConditionalFallthrough,
                    fallthrough,
                );
            }
            Terminator::Indirect {
                target: destination,
            } => target(&mut exits, RegionExitKind::Indirect, destination),
            Terminator::Call {
                target: destination,
                ..
            } => target(&mut exits, RegionExitKind::Call, destination),
            Terminator::Return {
                target: destination,
            } => target(&mut exits, RegionExitKind::Return, destination),
            Terminator::Exception { .. } => exits.push(exit(id, RegionExitKind::Exception)),
            Terminator::ConditionalException { fallthrough, .. } => {
                exits.push(exit(id, RegionExitKind::Exception));
                target(
                    &mut exits,
                    RegionExitKind::ConditionalFallthrough,
                    fallthrough,
                );
            }
            Terminator::InterpretOne { .. } => exits.push(exit(id, RegionExitKind::Interpreter)),
            Terminator::UnsupportedInstruction { .. } => {
                exits.push(exit(id, RegionExitKind::UnsupportedInstruction))
            }
            Terminator::Stop { .. } => exits.push(exit(id, RegionExitKind::Stop)),
        }
    }
    Ok(exits)
}

fn exit(block: BlockId, kind: RegionExitKind) -> RegionExit {
    RegionExit {
        block,
        kind,
        target: None,
    }
}

fn internal_targets(terminator: &Terminator) -> Vec<BlockId> {
    let mut result = Vec::new();
    let mut push = |target: &ControlTarget| {
        if let ControlTarget::Internal { block } = target {
            result.push(*block);
        }
    };
    match terminator {
        Terminator::Direct { target } => push(target),
        Terminator::Conditional {
            taken, fallthrough, ..
        } => {
            push(taken);
            push(fallthrough);
        }
        Terminator::ConditionalCall { fallthrough, .. }
        | Terminator::ConditionalException { fallthrough, .. } => push(fallthrough),
        _ => {}
    }
    result
}

fn safepoints(
    blocks: &[IrBlock],
    entries: &[RegionEntry],
) -> Result<Vec<RegionSafepoint>, FrontendError> {
    let mut result: Vec<_> = entries
        .iter()
        .map(|entry| RegionSafepoint {
            block: entry.block,
            target: None,
            kind: RegionSafepointKind::Entry,
        })
        .collect();
    for (index, block) in blocks.iter().enumerate() {
        let source = BlockId::new(
            u32::try_from(index).map_err(|_| internal("region block index overflow"))?,
        );
        for target in internal_targets(&block.terminator) {
            if target <= source {
                result.push(RegionSafepoint {
                    block: source,
                    target: Some(target),
                    kind: RegionSafepointKind::BackwardEdge,
                });
            }
        }
    }
    Ok(result)
}

fn extend_dependencies(dependencies: &mut Vec<CodePageDependency>, block: &IrBlock) {
    for source in &block.metadata.sources {
        for dependency in source.dependencies.iter() {
            if !dependencies.contains(&dependency) {
                dependencies.push(dependency);
            }
        }
    }
}

fn internal(message: impl Into<Box<str>>) -> FrontendError {
    FrontendError::Internal(FrontendInternalError::new(None, message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{GuestPhysicalPageId, GuestVirtualAddress};

    use crate::{
        ir::{
            region::{RegionExitKind, RegionSafepointKind},
            terminator::ControlTarget,
            verify::verify_region,
        },
        location::ExecutionState,
        memory::{MemoryPermissions, SyntheticMemory},
    };

    const SPACE: AddressSpaceId = AddressSpaceId::new(0x5245_4749_4f4e);

    fn location(profile: GuestCpuProfile, pc: u64) -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(pc),
            ExecutionState::A64,
            profile.id(),
        )
    }

    fn memory(words: &[u32]) -> SyntheticMemory {
        let mut memory = SyntheticMemory::new();
        let page = GuestPhysicalPageId::new(1);
        assert!(memory.add_ram_page(page));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            page,
            MemoryPermissions::READ_EXECUTE,
        ));
        for (index, word) in words.iter().enumerate() {
            assert!(memory.initialize_ram(page, index * 4, &word.to_le_bytes()));
        }
        memory
    }

    #[test]
    fn conditional_region_shares_an_interior_entry_without_duplicate_instructions() {
        let profile = GuestCpuProfile::switch_1();
        // b.eq +8; nop; ret
        let memory = memory(&[0x5400_0040, 0xd503_201f, 0xd65f_03c0]);
        let region = translate_region(
            RegionTranslationConfig::default(),
            &profile,
            SPACE,
            location(profile, 0x1000),
            &memory,
        )
        .unwrap();

        verify_region(&region).unwrap();
        assert_eq!(region.blocks.len(), 3);
        assert_eq!(region.metadata.guest_instruction_count, 3);
        assert_eq!(region.metadata.entries.len(), 3);
        assert_eq!(region.blocks[1].metadata.sources.len(), 1);
        assert!(matches!(
            region.blocks[0].terminator,
            Terminator::Conditional {
                taken: ControlTarget::Internal { block: taken },
                fallthrough: ControlTarget::Internal { block: fallthrough },
                ..
            } if taken == BlockId::new(2) && fallthrough == BlockId::new(1)
        ));
        assert!(matches!(
            region.blocks[1].terminator,
            Terminator::Direct {
                target: ControlTarget::Internal { block }
            } if block == BlockId::new(2)
        ));
        assert_eq!(region.metadata.exits.len(), 1);
        assert_eq!(region.metadata.exits[0].kind, RegionExitKind::Return);
    }

    #[test]
    fn backward_internal_edges_have_explicit_safepoints() {
        let profile = GuestCpuProfile::switch_1();
        // nop; b.eq -4; ret
        let memory = memory(&[0xd503_201f, 0x54ff_ffe0, 0xd65f_03c0]);
        let region = translate_region(
            RegionTranslationConfig::default(),
            &profile,
            SPACE,
            location(profile, 0x1000),
            &memory,
        )
        .unwrap();

        assert!(region.metadata.safepoints.iter().any(|safepoint| {
            safepoint.kind == RegionSafepointKind::BackwardEdge
                && safepoint.block == BlockId::new(0)
                && safepoint.target == Some(BlockId::new(0))
        }));
        assert!(region.metadata.entries.iter().all(|entry| {
            region.metadata.safepoints.iter().any(|safepoint| {
                safepoint.kind == RegionSafepointKind::Entry && safepoint.block == entry.block
            })
        }));
    }

    #[test]
    fn configured_region_bound_leaves_unformed_successors_external() {
        let profile = GuestCpuProfile::switch_1();
        let memory = memory(&[0x5400_0040, 0xd503_201f, 0xd65f_03c0]);
        let region = translate_region(
            RegionTranslationConfig {
                max_blocks: NonZeroU32::new(1).unwrap(),
                ..RegionTranslationConfig::default()
            },
            &profile,
            SPACE,
            location(profile, 0x1000),
            &memory,
        )
        .unwrap();

        assert_eq!(region.blocks.len(), 1);
        assert_eq!(region.metadata.entries.len(), 1);
        assert_eq!(region.metadata.exits.len(), 2);
        assert!(region.metadata.exits.iter().all(|exit| {
            exit.target
                .is_some_and(ControlTarget::requires_state_commit)
        }));
    }
}
