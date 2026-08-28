use std::collections::{HashSet, VecDeque};

use nixe_cpu::decode::a64::{A64Instruction, control, system};
use nixe_cpu::decode::{self, DecodeResult, DecodedOpcode};
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::{DecodedInstruction, InstructionEncoding, LocationDescriptor};
use nixe_cpu::memory::{CodePageDependency, CodePageSpan, InstructionMemory};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use super::DirectJitError;

pub(super) const DEFAULT_MAX_REGION_INSTRUCTIONS: usize = 256;
pub(super) const DEFAULT_MAX_REGION_BLOCKS: usize = 64;
pub(super) const DEFAULT_MAX_CODE_DEPENDENCIES: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct RegionKey {
    pub(super) address_space: AddressSpaceId,
    pub(super) start: GuestVirtualAddress,
    pub(super) platform: nixe_cpu::platform::TargetPlatform,
}

impl RegionKey {
    pub(super) const fn new(cpu: ProcessCpuContext, start: LocationDescriptor) -> Self {
        Self {
            address_space: cpu.address_space_id(),
            start: start.pc,
            platform: cpu.platform(),
        }
    }

    pub(super) const fn at(self, target: GuestVirtualAddress) -> Self {
        Self {
            start: target,
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RegionLimits {
    pub(super) instructions: usize,
    pub(super) blocks: usize,
    pub(super) dependencies: usize,
}

impl Default for RegionLimits {
    fn default() -> Self {
        Self {
            instructions: DEFAULT_MAX_REGION_INSTRUCTIONS,
            blocks: DEFAULT_MAX_REGION_BLOCKS,
            dependencies: DEFAULT_MAX_CODE_DEPENDENCIES,
        }
    }
}

pub(super) struct NativeRegion {
    pub(super) key: RegionKey,
    pub(super) blocks: Box<[BasicBlockRecord]>,
    pub(super) dependencies: Box<[CodePageDependency]>,
    pub(super) mapping_dependencies: Box<[CodePageSpan]>,
    pub(super) external_exits: Box<[ExternalExitRecord]>,
    #[cfg(test)]
    pub(super) instruction_count: usize,
}

pub(super) struct BasicBlockRecord {
    pub(super) start: LocationDescriptor,
    pub(super) instructions: Box<[DecodedInstruction<DecodedOpcode>]>,
    pub(super) terminator: BlockTerminator,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BlockTerminator {
    Direct {
        target: GuestVirtualAddress,
    },
    Conditional {
        taken: GuestVirtualAddress,
        not_taken: GuestVirtualAddress,
    },
    Call {
        target: GuestVirtualAddress,
        return_address: GuestVirtualAddress,
    },
    Indirect,
    Architectural {
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    Unsupported,
    Limit {
        continuation: GuestVirtualAddress,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExternalExitKind {
    Direct,
    ConditionalTaken,
    ConditionalNotTaken,
    Call,
    Indirect,
    Architectural,
    Unsupported,
    RegionLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ExternalExitRecord {
    pub(super) source: LocationDescriptor,
    pub(super) target: Option<GuestVirtualAddress>,
    pub(super) kind: ExternalExitKind,
}

pub(super) fn discover_region(
    cpu: ProcessCpuContext,
    memory: &(impl InstructionMemory + ?Sized),
    start: LocationDescriptor,
    limits: RegionLimits,
    published_entry: impl Fn(GuestVirtualAddress) -> bool,
) -> Result<NativeRegion, DirectJitError> {
    if limits.instructions == 0 || limits.blocks == 0 || limits.dependencies == 0 {
        return Err(DirectJitError::internal(
            "direct JIT region limits must be non-zero",
        ));
    }
    let key = RegionKey::new(cpu, start);
    let mut pending = VecDeque::from([start.pc]);
    let mut scheduled = HashSet::from([start.pc]);
    let mut blocks = Vec::new();
    let mut dependencies = Vec::new();
    let mut mapping_dependencies = Vec::new();
    let mut instruction_count = 0;

    while let Some(block_start) = pending.pop_front() {
        if blocks.len() == limits.blocks || instruction_count == limits.instructions {
            break;
        }
        let block_location = LocationDescriptor::new(block_start, cpu.profile_id());
        let mut pc = block_start;
        let mut instructions = Vec::new();
        let terminator = loop {
            if instruction_count == limits.instructions {
                break BlockTerminator::Limit { continuation: pc };
            }
            if pc != block_start && published_entry(pc) {
                break BlockTerminator::Direct { target: pc };
            }
            if pc != block_start && scheduled.contains(&pc) {
                break BlockTerminator::Direct { target: pc };
            }
            let location = LocationDescriptor::new(pc, cpu.profile_id());
            let (encoding, fetched_dependencies) = fetch(memory, key.address_space, location)?;
            for dependency in fetched_dependencies.iter() {
                if !dependencies.contains(&dependency) {
                    if dependencies.len() == limits.dependencies {
                        break;
                    }
                    dependencies.push(dependency);
                }
            }
            if fetched_dependencies
                .iter()
                .any(|dependency| !dependencies.contains(&dependency))
            {
                break BlockTerminator::Limit { continuation: pc };
            }
            if !mapping_dependencies
                .iter()
                .any(|span: &CodePageSpan| span.contains(pc))
            {
                mapping_dependencies.push(memory.code_page_span(key.address_space, pc).map_err(
                    |fault| {
                        DirectJitError::invalid(format!(
                            "direct JIT code-page query failed: {fault}"
                        ))
                    },
                )?);
            }
            let decoded = match decode::decode(cpu.decoder(), location, encoding) {
                DecodeResult::Decoded(decoded) => decoded,
                DecodeResult::RecognizedUnimplemented(decoded) => {
                    instructions.push(decoded);
                    instruction_count += 1;
                    break BlockTerminator::Unsupported;
                }
                DecodeResult::Unallocated { reason, .. } => {
                    return Err(DirectJitError::invalid(format!(
                        "invalid direct JIT instruction stream: {location} encoding={encoding} reason={reason}"
                    )));
                }
                DecodeResult::Reserved { name, reason, .. } => {
                    return Err(DirectJitError::invalid(format!(
                        "invalid direct JIT instruction stream: {location} encoding={encoding} reason={name}: reserved: {reason}"
                    )));
                }
            };
            instructions.push(decoded);
            instruction_count += 1;
            let decoded = instructions
                .last()
                .expect("just pushed decoded instruction");
            let next = GuestVirtualAddress::new(pc.get().wrapping_add(4));
            match decode::a64::normalize(&decoded.instruction, decoded.encoding) {
                A64Instruction::System(instruction)
                    if !system_instruction_supported(cpu.platform(), instruction) =>
                {
                    break BlockTerminator::Unsupported;
                }
                A64Instruction::Control(control::Instruction::Nop(_))
                | A64Instruction::Integer(_)
                | A64Instruction::Memory(_)
                | A64Instruction::System(_)
                | A64Instruction::FpSimd(_) => pc = next,
                A64Instruction::Control(control::Instruction::BranchImmediate(fields)) => {
                    let target = branch_target(pc, fields.immediate_26, 26);
                    schedule(
                        target,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    break BlockTerminator::Direct { target };
                }
                A64Instruction::Control(control::Instruction::ConditionalBranch(fields)) => {
                    let taken = branch_target(pc, fields.immediate_19, 19);
                    let not_taken = next;
                    schedule(
                        taken,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    schedule(
                        not_taken,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    break BlockTerminator::Conditional { taken, not_taken };
                }
                A64Instruction::Control(control::Instruction::CompareBranch(fields)) => {
                    let taken = branch_target(pc, fields.immediate_19, 19);
                    let not_taken = next;
                    schedule(
                        taken,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    schedule(
                        not_taken,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    break BlockTerminator::Conditional { taken, not_taken };
                }
                A64Instruction::Control(control::Instruction::TestBranch(fields)) => {
                    let taken = branch_target(pc, fields.immediate_14, 14);
                    let not_taken = next;
                    schedule(
                        taken,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    schedule(
                        not_taken,
                        limits.blocks,
                        &published_entry,
                        &mut scheduled,
                        &mut pending,
                    );
                    break BlockTerminator::Conditional { taken, not_taken };
                }
                A64Instruction::Control(control::Instruction::BranchLinkImmediate(fields)) => {
                    break BlockTerminator::Call {
                        target: branch_target(pc, fields.immediate_26, 26),
                        return_address: next,
                    };
                }
                A64Instruction::Control(control::Instruction::BranchRegister(_)) => {
                    break BlockTerminator::Indirect;
                }
                A64Instruction::Control(control::Instruction::SupervisorCall(fields)) => {
                    break BlockTerminator::Architectural {
                        kind: ExceptionKind::SupervisorCall,
                        syndrome: Some(u64::from(fields.immediate_16)),
                    };
                }
                A64Instruction::Control(control::Instruction::Breakpoint(fields)) => {
                    break BlockTerminator::Architectural {
                        kind: ExceptionKind::Breakpoint,
                        syndrome: Some(u64::from(fields.immediate_16)),
                    };
                }
                _ => break BlockTerminator::Unsupported,
            }
        };
        blocks.push(BasicBlockRecord {
            start: block_location,
            instructions: instructions.into_boxed_slice(),
            terminator,
        });
    }

    let starts: HashSet<_> = blocks.iter().map(|block| block.start.pc).collect();
    let mut external_exits = Vec::new();
    for block in &blocks {
        let source = block
            .instructions
            .last()
            .map_or(block.start, |instruction| instruction.location);
        match block.terminator {
            BlockTerminator::Direct { target } if !starts.contains(&target) => {
                external_exits.push(ExternalExitRecord {
                    source,
                    target: Some(target),
                    kind: ExternalExitKind::Direct,
                });
            }
            BlockTerminator::Conditional { taken, not_taken } => {
                if !starts.contains(&taken) {
                    external_exits.push(ExternalExitRecord {
                        source,
                        target: Some(taken),
                        kind: ExternalExitKind::ConditionalTaken,
                    });
                }
                if !starts.contains(&not_taken) {
                    external_exits.push(ExternalExitRecord {
                        source,
                        target: Some(not_taken),
                        kind: ExternalExitKind::ConditionalNotTaken,
                    });
                }
            }
            BlockTerminator::Call { target, .. } => external_exits.push(ExternalExitRecord {
                source,
                target: Some(target),
                kind: ExternalExitKind::Call,
            }),
            BlockTerminator::Indirect => external_exits.push(ExternalExitRecord {
                source,
                target: None,
                kind: ExternalExitKind::Indirect,
            }),
            BlockTerminator::Architectural { .. } => {
                external_exits.push(ExternalExitRecord {
                    source,
                    target: None,
                    kind: ExternalExitKind::Architectural,
                });
            }
            BlockTerminator::Unsupported => external_exits.push(ExternalExitRecord {
                source,
                target: None,
                kind: ExternalExitKind::Unsupported,
            }),
            BlockTerminator::Limit { continuation } => {
                external_exits.push(ExternalExitRecord {
                    source,
                    target: Some(continuation),
                    kind: ExternalExitKind::RegionLimit,
                });
            }
            BlockTerminator::Direct { .. } => {}
        }
    }

    Ok(NativeRegion {
        key,
        blocks: blocks.into_boxed_slice(),
        dependencies: dependencies.into_boxed_slice(),
        mapping_dependencies: mapping_dependencies.into_boxed_slice(),
        external_exits: external_exits.into_boxed_slice(),
        #[cfg(test)]
        instruction_count,
    })
}

fn system_instruction_supported(
    platform: nixe_cpu::platform::TargetPlatform,
    instruction: system::Instruction,
) -> bool {
    let fields = instruction.operands();
    match instruction {
        system::Instruction::Hint(_) => {
            nixe_cpu::semantics::a64::hint_operation(platform, fields.hint).is_some()
                || matches!(fields.hint, 32 | 34 | 36 | 38)
        }
        system::Instruction::ReadRegister(_) => {
            matches!(
                fields.system_key,
                0xd53b_4200 | 0xd53b_4400 | 0xd53b_4420 | 0xd53b_d040 | 0xd53b_d060
            ) || nixe_cpu::semantics::a64::runtime_register_read(platform, fields.system_key)
                .is_some()
        }
        system::Instruction::WriteRegister(_) => matches!(
            fields.system_key,
            0xd51b_4200 | 0xd51b_4400 | 0xd51b_4420 | 0xd51b_d040
        ),
        system::Instruction::Barrier(_) => nixe_cpu::semantics::a64::barrier_operation(
            fields.barrier_opcode,
            fields.barrier_option,
        )
        .is_some(),
        system::Instruction::System(_) => {
            nixe_cpu::semantics::a64::cache_maintenance_operation(fields.system_key).is_some()
        }
    }
}

fn schedule(
    target: GuestVirtualAddress,
    block_limit: usize,
    published_entry: &impl Fn(GuestVirtualAddress) -> bool,
    scheduled: &mut HashSet<GuestVirtualAddress>,
    pending: &mut VecDeque<GuestVirtualAddress>,
) {
    if !published_entry(target) && scheduled.len() < block_limit && scheduled.insert(target) {
        pending.push_back(target);
    }
}

fn fetch(
    memory: &(impl InstructionMemory + ?Sized),
    address_space: AddressSpaceId,
    location: LocationDescriptor,
) -> Result<(InstructionEncoding, nixe_cpu::memory::CodeDependencies), DirectJitError> {
    memory
        .fetch32(address_space, location.pc)
        .map(|fetched| {
            (
                InstructionEncoding::from_u32(fetched.bits),
                fetched.dependencies,
            )
        })
        .map_err(|fault| DirectJitError::invalid(format!("direct JIT fetch failed: {fault}")))
}

fn branch_target(
    pc: GuestVirtualAddress,
    immediate: impl Into<u64>,
    bits: u8,
) -> GuestVirtualAddress {
    GuestVirtualAddress::new(pc.get().wrapping_add_signed(
        nixe_cpu::semantics::a64::signed_immediate(immediate.into(), bits) << 2,
    ))
}
