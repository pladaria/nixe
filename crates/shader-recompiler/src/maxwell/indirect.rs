use super::{
    bits::{extract, extract_signed},
    decode::{Opcode, fetch},
    instruction::Instruction,
    location::Location,
};
use crate::{
    CompileError, Environment, Result,
    diagnostics::FailureCategory,
    ir::Reg,
    types::{ByteOffset, CbufIndex},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectTable {
    pub cbuf_index: CbufIndex,
    pub cbuf_offset: ByteOffset,
    pub num_entries: u32,
    pub branch_offset: i32,
    pub branch_reg: Reg,
}

fn track<E: Environment>(
    environment: &mut E,
    begin: Location,
    cursor: &mut Option<Location>,
    predicate: impl Fn(Instruction, Opcode) -> bool,
) -> Result<Option<Instruction>> {
    while let Some(pc) = *cursor {
        if pc < begin {
            break;
        }
        let decoded = fetch(environment, pc)?;
        let previous = pc.previous();
        // A wrapped byte address is outside the backwards search interval.
        *cursor = (previous < pc).then_some(previous);
        if predicate(decoded.instruction, decoded.opcode) {
            return Ok(Some(decoded.instruction));
        }
    }
    Ok(None)
}

pub fn track_indirect_table<E: Environment>(
    environment: &mut E,
    branch: Location,
    begin: Location,
) -> Result<Option<IndirectTable>> {
    let decoded = fetch(environment, branch)?;
    if !matches!(decoded.opcode, Opcode::BRX | Opcode::JMX) {
        return Err(CompileError::new(
            FailureCategory::Internal,
            "Indirect tracker requires BRX or JMX",
        ));
    }
    let branch_reg = decoded.instruction.reg8();
    let branch_offset = decoded.instruction.branch_offset();
    let mut cursor = Some(branch);
    let Some(ldc) = track(environment, begin, &mut cursor, |inst, op| {
        op == Opcode::LDC
            && inst.reg0() == branch_reg
            && extract::<48, 3>(inst.0) == 4
            && extract::<44, 2>(inst.0) == 0
    })?
    else {
        return Ok(None);
    };
    let ldc_reg = ldc.reg8();
    let Some(shl) = track(environment, begin, &mut cursor, |inst, op| {
        op == Opcode::SHL_imm && inst.reg0() == ldc_reg
    })?
    else {
        return Ok(None);
    };
    let shl_reg = shl.reg8();
    let Some(imnmx) = track(environment, begin, &mut cursor, |inst, op| {
        op == Opcode::IMNMX_imm && inst.reg0() == shl_reg
    })?
    else {
        return Ok(None);
    };
    if extract::<56, 1>(imnmx.0) != 0 {
        return Ok(None);
    }
    Ok(Some(IndirectTable {
        cbuf_index: CbufIndex(extract::<36, 5>(ldc.0) as u32),
        cbuf_offset: ByteOffset(extract_signed::<20, 16>(ldc.0) as u32),
        num_entries: extract::<20, 19>(imnmx.0) as u32 + 1,
        branch_offset,
        branch_reg,
    }))
}
