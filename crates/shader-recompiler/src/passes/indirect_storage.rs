//! Resolve bounded tables of guest pointers without changing their guest bytes.
use super::{
    fold::{arg, node},
    storage::low_address,
    storage_table::storage_opcode,
};
use crate::{
    CompileError, HostTranslateInfo, Result,
    artifact::PointerTableSpan,
    diagnostics::FailureCategory,
    ir::{Emitter, Ir, Opcode as O, Program, Value},
    types::Stage,
};
use std::collections::BTreeMap;

fn unsupported(message: &str) -> CompileError {
    CompileError::new(FailureCategory::Unsupported, message)
}

// Only scan elements proven to be pointers. An arbitrary 64-bit load can
// address a field in a mixed structure; its other words must not be interpreted
// as addresses by resource preparation.
fn dense_pointer_index(ir: &Ir, offset: Value, bank: u32, base: u32) -> Result<bool> {
    let Some(sub) = node(ir, offset, O::ISub32)? else {
        return Ok(false);
    };
    let Some(sum) = node(ir, arg(ir, sub, 0)?, O::IAdd32)? else {
        return Ok(false);
    };
    for (index, pointer) in [(0, 1), (1, 0)] {
        let Some(pointer) = node(ir, arg(ir, sum, pointer)?, O::GetCbufU32)? else {
            continue;
        };
        if arg(ir, pointer, 0)? != Value::U32(bank) || arg(ir, pointer, 1)? != Value::U32(base) {
            continue;
        }
        if let Some(shift) = node(ir, arg(ir, sum, index)?, O::ShiftLeftLogical32)? {
            return Ok(arg(ir, shift, 1)? == Value::U32(3));
        }
    }
    Ok(false)
}

pub fn lower_pointer_tables(
    ir: &mut Ir,
    program: &mut Program,
    host: &HostTranslateInfo,
) -> Result<()> {
    let mut accesses = Vec::new();
    let mut tables = BTreeMap::<u32, (PointerTableSpan, bool)>::new();
    for &block in &program.post_order {
        for &id in ir.block(block)?.instructions() {
            let inst = ir.inst(id)?;
            if storage_opcode(inst.opcode()).is_none() {
                continue;
            }
            let Some((low, high, offset)) = low_address(ir, id)? else {
                continue;
            };
            let (Some(low), Some(high)) = (
                node(ir, low, O::CompositeExtractU32x2)?,
                node(ir, high, O::CompositeExtractU32x2)?,
            ) else {
                continue;
            };
            if ir.resolve(arg(ir, low, 0)?)? != ir.resolve(arg(ir, high, 0)?)?
                || arg(ir, low, 1)? != Value::U32(0)
                || arg(ir, high, 1)? != Value::U32(1)
            {
                continue;
            }
            let Some(load) = node(ir, arg(ir, low, 0)?, O::LoadStorage64)? else {
                continue;
            };
            let Value::U32(table) = arg(ir, load, 0)? else {
                continue;
            };
            let source = &program.info.storage_buffers_descriptors[table as usize];
            if !dense_pointer_index(
                ir,
                arg(ir, load, 1)?,
                source.cbuf_index.0,
                source.cbuf_offset.0,
            )? {
                return Err(unsupported(
                    "Indirect pointer-table indexing is not a dense array of 64-bit pointers",
                ));
            }
            let (op, bytes) = match inst.opcode() {
                O::LoadGlobal32 => (O::LoadPhysical32, 4),
                O::LoadGlobal64 => (O::LoadPhysical64, 8),
                O::LoadGlobal128 => (O::LoadPhysical128, 16),
                O::WriteGlobal32 => (O::WritePhysical32, 4),
                O::WriteGlobal64 => (O::WritePhysical64, 8),
                O::WriteGlobal128 => (O::WritePhysical128, 16),
                _ => {
                    return Err(unsupported(
                        "Indirect pointer-table narrow accesses and atomics are not implemented",
                    ));
                }
            };
            if !host.support_int64 || !host.support_buffer_device_address {
                return Err(unsupported(
                    "Indirect pointer tables require host buffer device addresses and 64-bit arithmetic",
                ));
            }
            // Physical target bounds use a signed byte displacement. Do not
            // truncate an arbitrary 64-bit addend into an unrelated allocation.
            if let Some(add) = node(ir, arg(ir, id, 0)?, O::IAdd64)?
                && let Value::U64(value) = arg(ir, add, 1)?
                && value as i64 != value as i32 as i64
            {
                return Err(unsupported(
                    "Indirect pointer-table displacement exceeds signed 32 bits",
                ));
            }
            let start = offset as i32;
            if !offset.is_multiple_of(4) {
                return Err(unsupported("Unaligned indirect pointer-table access"));
            }
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| unsupported("Indirect pointer-table access extent overflow"))?;
            let span = tables
                .entry(table)
                .or_insert((PointerTableSpan { start, end }, false));
            span.0.start = span.0.start.min(start);
            span.0.end = span.0.end.max(end);
            span.1 |= inst.may_have_side_effects();
            accesses.push((block, id, load, table, arg(ir, load, 1)?, start, op));
        }
    }
    let nvn_base = match program.stage {
        Stage::VertexA | Stage::VertexB => 0x110,
        Stage::TessellationControl => 0x210,
        Stage::TessellationEval | Stage::Compute => 0x310,
        Stage::Geometry => 0x410,
        Stage::Fragment => 0x510,
    };
    let mut indices = BTreeMap::new();
    for (index, (span, written)) in tables {
        let mut descriptor = program.info.storage_buffers_descriptors[index as usize].clone();
        if descriptor.cbuf_index.0 != 0
            || descriptor
                .cbuf_offset
                .0
                .checked_sub(nvn_base)
                .is_none_or(|offset| offset >= 256 || !offset.is_multiple_of(16))
        {
            return Err(unsupported(
                "Indirect pointer table has no bounded NVN address/size descriptor",
            ));
        }
        if descriptor.is_written {
            return Err(unsupported(
                "Shader writes its indirect pointer table during the same invocation",
            ));
        }
        descriptor.is_written = written;
        descriptor.pointer_table = Some(span);
        indices.insert(index, program.info.storage_buffers_descriptors.len() as u32);
        program.info.storage_buffers_descriptors.push(descriptor)?;
    }
    let mut pointers = BTreeMap::new();
    for &(_, _, load, table, table_offset, _, _) in &accesses {
        if pointers.contains_key(&load) {
            continue;
        }
        let block = program
            .post_order
            .iter()
            .copied()
            .find(|&block| {
                ir.block(block)
                    .is_ok_and(|b| b.instructions().contains(&load))
            })
            .expect("pointer source belongs to the shader");
        let mut e = Emitter::before(ir, block, load)?;
        let pointer = e.emit(
            O::LoadStorage64,
            &[Value::U32(indices[&table]), table_offset],
            0,
        )?;
        let address = e.emit(O::PackUint2x32, &[pointer], 0)?;
        pointers.insert(load, address);
    }
    for (block, id, load, _, _, offset, op) in accesses {
        let input = if ir.inst(id)?.num_args() > 1 {
            arg(ir, id, 1)?
        } else {
            Value::None
        };
        let mut e = Emitter::before(ir, block, id)?;
        let mut address = pointers[&load];
        if offset != 0 {
            address = e.emit(O::IAdd64, &[address, Value::U64(offset as i64 as u64)], 0)?;
        }
        if op.signature().arguments.len() == 2 {
            e.emit(op, &[address, input], 0)?;
            e.ir.invalidate(id)?;
        } else {
            let result = e.emit(op, &[address], 0)?;
            e.ir.replace_uses_with(id, result)?;
        }
    }
    Ok(())
}

/// Remove bindings whose only consumers were replaced by native pointer loads.
pub fn compact_storage(ir: &mut Ir, program: &mut Program) -> Result<()> {
    use super::storage_table::is_storage;
    if !program
        .info
        .storage_buffers_descriptors
        .iter()
        .any(|d| d.pointer_table.is_some())
    {
        return Ok(());
    }
    let mut instructions = Vec::new();
    let mut indices = BTreeMap::new();
    for &block in &program.post_order {
        for &id in ir.block(block)?.instructions() {
            if is_storage(ir.inst(id)?.opcode()) {
                let Value::U32(index) = arg(ir, id, 0)? else {
                    return Err(unsupported("Dynamic storage descriptor index"));
                };
                indices.insert(index, 0);
                instructions.push((id, index));
            }
        }
    }
    let old = std::mem::take(&mut program.info.storage_buffers_descriptors);
    for (index, output) in &mut indices {
        *output = program.info.storage_buffers_descriptors.len() as u32;
        program
            .info
            .storage_buffers_descriptors
            .push(old[*index as usize].clone())?;
    }
    for (id, index) in instructions {
        ir.set_arg(id, 0, Value::U32(indices[&index]))?;
    }
    Ok(())
}
