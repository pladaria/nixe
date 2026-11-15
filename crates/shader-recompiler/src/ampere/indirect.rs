//! Finite native branch tables. Resolve targets from coherent constant data;
//! the same reads remain compilation dependencies in the environment snapshot.
use super::instruction::{Instruction, unsupported};
use crate::{
    Environment, Result,
    frontend::flow::BranchRegister,
    ir::{Condition, Reg},
    types::{ByteOffset, CbufIndex, ProgramOffset},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Table {
    pub register: BranchRegister,
    pub base: i32,
    pub targets: Vec<ProgramOffset>,
    definitions: Vec<ProgramOffset>,
}

#[derive(Clone, Copy, Debug)]
enum Register {
    General(u32),
    Uniform(u32),
    UniformPredicate(u32),
}

/// Instruction-level edges deliberately exclude the synthetic IR branches used
/// to predicate individual instructions. All reaching paths must prove the same
/// unconditional definition; a branch join alone does not clobber a register.
pub(super) struct Definitions<'a> {
    code: &'a BTreeMap<ProgramOffset, Instruction>,
    predecessors: BTreeMap<ProgramOffset, Vec<ProgramOffset>>,
    calls: &'a BTreeMap<ProgramOffset, crate::frontend::flow::FunctionId>,
    entry: ProgramOffset,
}
impl<'a> Definitions<'a> {
    pub fn new(
        code: &'a BTreeMap<ProgramOffset, Instruction>,
        calls: &'a BTreeMap<ProgramOffset, crate::frontend::flow::FunctionId>,
        tables: &BTreeMap<ProgramOffset, Table>,
        entry: ProgramOffset,
        exit_handler: Option<ProgramOffset>,
    ) -> Result<Self> {
        let mut predecessors: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (&pc, &i) in code {
            let next = ProgramOffset(pc.0 + 16);
            let mut fallthrough = true;
            if matches!(i.opcode(), 0x944 | 0x947) && !calls.contains_key(&pc) {
                let (guard, condition) = i.branch_predicates()?;
                if guard != Condition::always(false) && condition != Condition::always(false) {
                    predecessors
                        .entry(i.branch_target(pc)?)
                        .or_default()
                        .push(pc);
                }
                fallthrough =
                    guard != Condition::always(true) || condition != Condition::always(true);
            } else if matches!(i.opcode(), 0x949 | 0x958) {
                if let Some(table) = tables.get(&pc) {
                    for &target in &table.targets {
                        predecessors.entry(target).or_default().push(pc);
                    }
                }
                fallthrough = false;
            } else if matches!(i.opcode(), 0x94d | 0x950) {
                fallthrough = i.condition()? != Condition::always(true);
                if i.opcode() == 0x94d
                    && i.condition()? != Condition::always(false)
                    && let Some(handler) = exit_handler.filter(|&handler| pc < handler)
                {
                    predecessors.entry(handler).or_default().push(pc);
                }
            }
            if fallthrough && code.contains_key(&next) {
                predecessors.entry(next).or_default().push(pc);
            }
        }
        Ok(Self {
            code,
            predecessors,
            calls,
            entry,
        })
    }

    fn definition(
        &self,
        end: ProgramOffset,
        register: Register,
    ) -> Result<(ProgramOffset, Instruction)> {
        if matches!(
            register,
            Register::General(255) | Register::Uniform(63) | Register::UniformPredicate(7)
        ) {
            return Err(unsupported(
                "Native branch-table source is the zero register",
            ));
        }
        let mut pending = vec![end];
        let mut visited = BTreeSet::new();
        let mut found = None;
        while let Some(pc) = pending.pop() {
            if !visited.insert(pc) {
                continue;
            }
            if pc == self.entry || !self.predecessors.contains_key(&pc) {
                return Err(unsupported(format!(
                    "Native branch register {register:?} has no definition on a path to {:#x}",
                    pc.0
                )));
            }
            for &previous in &self.predecessors[&pc] {
                let i = self.code[&previous];
                if self.calls.contains_key(&previous) {
                    return Err(unsupported(
                        "Cannot prove native branch register across a call",
                    ));
                }
                if i.condition()? != Condition::always(false)
                    && writes_register(i, previous, register)?
                {
                    if i.condition()? != Condition::always(true) {
                        return Err(unsupported("Predicated native branch-table definition"));
                    }
                    if found.is_some_and(|(at, _)| at != previous) {
                        return Err(unsupported(format!(
                            "Native branch register {register:?} has conflicting reaching definitions"
                        )));
                    }
                    found = Some((previous, i));
                } else {
                    pending.push(previous);
                }
            }
        }
        found.ok_or_else(|| {
            unsupported(format!(
                "Native branch register {register:?} has no reaching definition"
            ))
        })
    }
}

fn writes_register(i: Instruction, pc: ProgramOffset, register: Register) -> Result<bool> {
    let uniform_words = match i.opcode() {
        0xc82 | 0x882 | 0x290 | 0x890 | 0x292 | 0x892 | 0x2a4 | 0x4a4 | 0x8a4 | 0x899 | 0x299
        | 0x499 | 0x287 | 0x887 | 0x2bd | 0x2be | 0x2bf | 0x3c2 | 0x3c4 | 0x9c3 | 0x886 => 1,
        0x2a5 => 2,
        0xab9 | 0xabb => {
            if i.bits(73, 3) == 5 {
                2
            } else {
                1
            }
        }
        0x28c | 0x88c | 0x958 => 0,
        _ => {
            // Classify recognized general instructions below as well: unknown
            // opcodes must never be silently crossed in any register namespace.
            let words = general_words(i, pc)?;
            return Ok(match register {
                Register::General(register) => {
                    (i.bits(16, 8)..i.bits(16, 8) + words).contains(&register)
                }
                _ => false,
            });
        }
    };
    Ok(match register {
        Register::General(_) => false,
        Register::Uniform(register) => {
            (i.bits(16, 8)..i.bits(16, 8) + uniform_words).contains(&register)
        }
        Register::UniformPredicate(register) => match i.opcode() {
            0x28c | 0x88c | 0x290 | 0x890 => [i.bits(81, 3), i.bits(84, 3)].contains(&register),
            0x292 | 0x892 | 0x886 => i.bits(81, 3) == register,
            _ => false,
        },
    })
}

fn general_words(i: Instruction, pc: ProgramOffset) -> Result<u32> {
    // Unknown writers cannot be crossed: an unrecognised instruction may
    // overwrite either word of the value whose definition is being tracked.
    let words = match i.opcode() {
        0x202 | 0x802 | 0xa02 | 0x210 | 0x810 | 0xa10 | 0x220 | 0x820 | 0xa20 | 0x221 | 0x421
        | 0x621 | 0x821 | 0xa21 | 0x223 | 0x423 | 0x623 | 0x823 | 0xa23 | 0x217 | 0x817 | 0xa17
        | 0x819 | 0x308 | 0x908 | 0xb08 | 0x326 | 0x526 | 0x209 | 0x809 | 0xa09 | 0xc09 | 0x212
        | 0x812 | 0xa12 | 0xc12 | 0x224 | 0x424 | 0x624 | 0x824 | 0xa24 | 0xc24 | 0xe24 => 1,
        0xb82 => {
            if i.bits(73, 3) == 5 {
                2
            } else {
                1
            }
        }
        0x321 => i.bits(74, 2) + 1,
        0x322 | 0x20b | 0x80b | 0xa0b | 0xc0b | 0x918 | 0x20c | 0x80c | 0xa0c | 0xc0c | 0x944
        | 0x947 | 0x949 | 0x94d | 0x95b => 0,
        // BSSY, BSYNC and BREAK update convergence barriers, not GPRs.
        // Flow analysis has already validated their modes and predication.
        // https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-reference
        0x941 | 0x942 | 0x945 => 0,
        _ => {
            return Err(unsupported(format!(
                "Cannot prove native branch register across {:#x} at {:#x}",
                i.opcode(),
                pc.0
            )));
        }
    };
    Ok(words)
}

fn definitions(
    flow: &Definitions<'_>,
    pc: ProgramOffset,
    branch: Instruction,
) -> Result<(Vec<ProgramOffset>, Instruction, u32)> {
    let uniform = branch.opcode() == 0x958;
    if branch.condition()? != Condition::always(true)
        || branch.bits(82, 14) != if uniform { 0x2e0 } else { 0xe0 }
    {
        return Err(unsupported(
            "Native computed branch predicate/convergence mode",
        ));
    }
    let register_file = if uniform {
        Register::Uniform
    } else {
        Register::General
    };
    let register = branch.bits(24, 8);
    let (load_pc, load) = flow.definition(pc, register_file(register))?;
    if load.opcode() != if uniform { 0xabb } else { 0xb82 }
        || load.bits(64, 32) != if uniform { 0x08000800 } else { 0x800 }
        || load.bits(16, 8) != register
    {
        return Err(unsupported(
            "Native branch requires a 32-bit direct constant table",
        ));
    }
    let (shift_pc, shift) = flow.definition(load_pc, register_file(load.bits(24, 8)))?;
    if shift.opcode() != if uniform { 0x899 } else { 0x819 }
        || shift.bits(32, 32) != 2
        || shift.bits(64, 32) != if uniform { 0x0800063f } else { 0x6ff }
    {
        return Err(unsupported(
            "Native branch-table index is not a word offset",
        ));
    }
    let (limit_pc, limit) = flow.definition(shift_pc, register_file(shift.bits(24, 8)))?;
    let mut definitions = vec![load_pc, shift_pc, limit_pc];
    if uniform {
        // UISETP.LT.U32 + USEL computes min(index, bound). Prove the
        // predicate and value through every reaching path, including writes
        // between the comparison and select; adjacency is not assumed.
        // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L1975-L2015
        let predicate = limit.bits(87, 3);
        if limit.opcode() != 0x887 || limit.bits(64, 32) != 0x08000000 | (predicate << 23) {
            return Err(unsupported(
                "Native uniform branch index lacks an unsigned select bound",
            ));
        }
        let (compare_pc, compare) =
            flow.definition(limit_pc, Register::UniformPredicate(predicate))?;
        let source = Register::Uniform(limit.bits(24, 8));
        if compare.opcode() != 0x88c
            || compare.bits(24, 8) != limit.bits(24, 8)
            || compare.bits(32, 32) != limit.bits(32, 32)
            || compare.bits(64, 32) != 0x0bf01070 | (predicate << 17)
        {
            return Err(unsupported(
                "Native uniform branch bound is not an unsigned minimum",
            ));
        }
        let (source_pc, _) = flow.definition(compare_pc, source)?;
        if flow.definition(limit_pc, source)?.0 != source_pc {
            return Err(unsupported(
                "Native uniform branch index changed after its comparison",
            ));
        }
        definitions.extend([compare_pc, source_pc]);
    } else if limit.opcode() != 0x817 || limit.bits(64, 32) != 0x0380_0000 {
        return Err(unsupported(
            "Native branch-table index lacks an unsigned minimum bound",
        ));
    }
    Ok((definitions, load, limit.bits(32, 32)))
}

pub(super) fn resolve(
    env: &mut impl Environment,
    flow: &Definitions<'_>,
    pc: ProgramOffset,
    branch: Instruction,
) -> Result<Table> {
    let (definitions, load, bound) = definitions(flow, pc, branch)?;
    let count = bound
        .checked_add(1)
        .filter(|&count| count <= 65536 / 4)
        .ok_or_else(|| unsupported("Native branch table exceeds constant-buffer size"))?;
    let base = branch.branch_target(pc)?.0;
    let offset = (load.bits(38, 16) as i16) as u32;
    let mut targets = Vec::with_capacity(count as usize);
    for index in 0..count {
        let value = env.read_cbuf_value(
            CbufIndex(load.bits(54, 5)),
            ByteOffset(offset.wrapping_add(index * 4)),
        )?;
        let address = base.wrapping_add(value);
        if address < env.metadata().entry_address()?.0 || address & 15 != 0 {
            return Err(unsupported(format!(
                "Invalid native branch-table destination {address:#x}"
            )));
        }
        targets.push(ProgramOffset(address));
    }
    targets.sort_unstable();
    targets.dedup();
    Ok(Table {
        register: if branch.opcode() == 0x958 {
            BranchRegister::Uniform(branch.bits(24, 8) as u8)
        } else {
            BranchRegister::General(Reg(branch.bits(24, 8) as u8))
        },
        base: base as i32,
        targets,
        definitions,
    })
}

impl Table {
    pub fn verify(
        &self,
        flow: &Definitions<'_>,
        pc: ProgramOffset,
        branch: Instruction,
    ) -> Result<()> {
        // Newly discovered indirect edges may introduce another predecessor.
        // Revalidate against the complete function before accepting its graph.
        if definitions(flow, pc, branch)?.0 != self.definitions {
            return Err(unsupported(
                "Native branch-table definition changed during flow discovery",
            ));
        }
        Ok(())
    }
}
