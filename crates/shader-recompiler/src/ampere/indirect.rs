//! Finite native branch tables. Resolve targets from coherent constant data;
//! the same reads remain compilation dependencies in the environment snapshot.
use super::instruction::{Instruction, unsupported};
use crate::{
    Environment, Result,
    ir::{Condition, Reg},
    types::{ByteOffset, CbufIndex, ProgramOffset},
};
use std::collections::BTreeMap;

pub(super) struct Table {
    pub register: Reg,
    pub base: i32,
    pub targets: Vec<ProgramOffset>,
}

fn definition(
    code: &BTreeMap<ProgramOffset, Instruction>,
    begin: ProgramOffset,
    end: ProgramOffset,
    register: u32,
) -> Result<(ProgramOffset, Instruction)> {
    for (&pc, &i) in code.range(begin..end).rev() {
        // Unknown writers cannot be crossed: an unrecognised instruction may
        // overwrite either word of the value whose definition is being tracked.
        let words = match i.opcode() {
            0x202 | 0x802 | 0xa02 | 0x210 | 0x810 | 0xa10 | 0x220 | 0x820 | 0xa20 | 0x221
            | 0x421 | 0x621 | 0x821 | 0xa21 | 0x223 | 0x423 | 0x623 | 0x823 | 0xa23 | 0x217
            | 0x817 | 0xa17 | 0x819 | 0x308 | 0x908 | 0xb08 | 0x326 | 0x526 | 0x209 | 0x809
            | 0xa09 | 0xc09 => 1,
            0xb82 => {
                if i.bits(73, 3) == 5 {
                    2
                } else {
                    1
                }
            }
            0x321 => i.bits(74, 2) + 1,
            0x322 | 0x20b | 0x80b | 0xa0b | 0x918 => 0,
            _ => {
                return Err(unsupported(format!(
                    "Cannot prove native branch register R{register} across {:#x} at {:#x}",
                    i.opcode(),
                    pc.0
                )));
            }
        };
        if (i.bits(16, 8)..i.bits(16, 8) + words).contains(&register) {
            if i.condition()? != Condition::always(true) {
                return Err(unsupported("Predicated native branch-table definition"));
            }
            return Ok((pc, i));
        }
    }
    Err(unsupported(
        "Native branch-table definition crosses a control-flow boundary",
    ))
}

pub(super) fn resolve(
    env: &mut impl Environment,
    code: &BTreeMap<ProgramOffset, Instruction>,
    begin: ProgramOffset,
    pc: ProgramOffset,
    branch: Instruction,
) -> Result<Table> {
    if branch.condition()? != Condition::always(true) || branch.bits(82, 14) != 0xe0 {
        return Err(unsupported(
            "Native computed branch predicate/convergence mode",
        ));
    }
    let register = branch.bits(24, 8);
    let (load_pc, load) = definition(code, begin, pc, register)?;
    if load.opcode() != 0xb82 || load.bits(64, 32) != 0x800 || load.bits(16, 8) != register {
        return Err(unsupported(
            "Native branch requires a 32-bit direct constant table",
        ));
    }
    let (shift_pc, shift) = definition(code, begin, load_pc, load.bits(24, 8))?;
    if shift.opcode() != 0x819 || shift.bits(32, 32) != 2 || shift.bits(64, 32) != 0x6ff {
        return Err(unsupported(
            "Native branch-table index is not a word offset",
        ));
    }
    let (_, limit) = definition(code, begin, shift_pc, shift.bits(24, 8))?;
    if limit.opcode() != 0x817 || limit.bits(64, 32) != 0x0380_0000 {
        return Err(unsupported(
            "Native branch-table index lacks an unsigned minimum bound",
        ));
    }
    let count = limit
        .bits(32, 32)
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
        register: Reg(register as u8),
        base: base as i32,
        targets,
    })
}
