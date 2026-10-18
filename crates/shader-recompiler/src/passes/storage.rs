use super::{
    fold::{arg, node},
    storage_table::storage_opcode,
};
use crate::{
    HostTranslateInfo, Result,
    arena::InstId,
    artifact::{ShaderInfo, StorageBufferDescriptor},
    ir::{Emitter, Ir, Opcode as O, Program, Value},
    types::{ByteOffset, CbufIndex},
};
use std::collections::BTreeMap;

fn low_address(ir: &Ir, id: InstId) -> Result<Option<(Value, u32)>> {
    let mut address = arg(ir, id, 0)?;
    let mut offset = 0;
    if let Some(add) = node(ir, address, O::IAdd64)? {
        let Value::U64(immediate) = arg(ir, add, 1)? else {
            return Ok(None);
        };
        offset = immediate as u32;
        address = arg(ir, add, 0)?;
    }
    if let Some(pack) = node(ir, address, O::PackUint2x32)? {
        address = arg(ir, pack, 0)?;
    }
    let Some(vector) = node(ir, address, O::CompositeConstructU32x2)? else {
        return Ok(None);
    };
    Ok(Some((ir.inst(vector)?.arg(0)?, offset)))
}
fn track(ir: &Ir, value: Value, bias: bool) -> Result<Option<(u32, u32)>> {
    ir.breadth_first_search(value, |id, inst| {
        if !matches!(inst.opcode(), O::GetCbufU32 | O::GetCbufU32x2) {
            return None;
        }
        let (Value::U32(bank), Value::U32(offset)) = (arg(ir, id, 0).ok()?, arg(ir, id, 1).ok()?)
        else {
            return None;
        };
        if !offset.is_multiple_of(if bias { 16 } else { 8 }) {
            return None;
        }
        if bias && (bank != 0 || !(0x100..0x700).contains(&offset)) {
            return None;
        }
        Some((bank, offset))
    })
}
pub fn global_memory_to_storage(
    ir: &mut Ir,
    program: &mut Program,
    host: &HostTranslateInfo,
) -> Result<()> {
    let mut descriptors = BTreeMap::new();
    let mut replacements = Vec::new();
    for &block in &program.post_order {
        for &id in ir.block(block)?.instructions() {
            let inst = ir.inst(id)?;
            let Some(op) = storage_opcode(inst.opcode()) else {
                continue;
            };
            let Some((low, _)) = low_address(ir, id)? else {
                continue;
            };
            let address = match track(ir, low, true)? {
                Some(address) => Some(address),
                None => track(ir, low, false)?,
            };
            let Some(address) = address else { continue };
            *descriptors.entry(address).or_insert(false) |= inst.may_have_side_effects();
            replacements.push((block, id, address, op));
        }
    }
    for (&(bank, offset), &is_written) in &descriptors {
        program
            .info
            .storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: CbufIndex(bank),
                cbuf_offset: ByteOffset(offset),
                count: 1,
                is_written,
            })?;
    }
    // The indices are local to this pass's sorted descriptor set.
    let indices: BTreeMap<_, _> = descriptors
        .keys()
        .enumerate()
        .map(|(index, &address)| (address, index as u32))
        .collect();
    for (block, id, (bank, base), op) in replacements {
        let (low, immediate) =
            low_address(ir, id)?.expect("Collected global address remains intact");
        let input = if ir.inst(id)?.num_args() > 1 {
            ir.inst(id)?.arg(1)?
        } else {
            Value::None
        };
        let store = op.signature().result == crate::ir::Type::Void;
        let mut e = Emitter::before(ir, block, id)?;
        let mut offset = low;
        if immediate != 0 {
            offset = e.i_add(offset, Value::U32(immediate))?;
        }
        let base_value = e.get_cbuf(Value::U32(bank), Value::U32(base), 32, false)?;
        let aligned = e.bitwise_and(
            base_value,
            Value::U32(!((host.min_ssbo_alignment as u32).wrapping_sub(1))),
        )?;
        let offset = e.i_sub(offset, aligned)?;
        let index = Value::U32(indices[&(bank, base)]);
        let result = if op.signature().arguments.len() == 2 {
            e.emit(op, &[index, offset], 0)?
        } else {
            e.emit(op, &[index, offset, input], 0)?
        };
        if store {
            e.ir.invalidate(id)?;
        } else {
            e.ir.replace_uses_with(id, result)?;
        }
    }
    Ok(())
}
pub fn join_storage_info(base: &mut ShaderInfo, source: &ShaderInfo) -> Result<()> {
    for descriptor in source.storage_buffers_descriptors.iter() {
        if let Some(existing) = base
            .storage_buffers_descriptors
            .iter_mut()
            .find(|existing| {
                existing.cbuf_index == descriptor.cbuf_index
                    && existing.cbuf_offset == descriptor.cbuf_offset
                    && existing.count == descriptor.count
            })
        {
            existing.is_written |= descriptor.is_written;
        } else {
            base.storage_buffers_descriptors.push(descriptor.clone())?;
        }
    }
    Ok(())
}

pub fn add_nvn_storage_buffers(program: &mut Program) -> Result<()> {
    use crate::types::Stage;
    if !program.info.uses_global_memory {
        return Ok(());
    }
    let base = match program.stage {
        Stage::VertexA | Stage::VertexB => 0x110,
        Stage::TessellationControl => 0x210,
        Stage::TessellationEval | Stage::Compute => 0x310,
        Stage::Geometry => 0x410,
        Stage::Fragment => 0x510,
    };
    for index in 0..16 {
        if program.info.nvn_buffer_used & (1 << index) == 0 {
            continue;
        }
        let offset = ByteOffset(base + index * 16);
        if let Some(existing) = program
            .info
            .storage_buffers_descriptors
            .iter_mut()
            .find(|d| d.cbuf_offset == offset)
        {
            existing.is_written |= program.info.stores_global_memory;
        } else {
            program
                .info
                .storage_buffers_descriptors
                .push(StorageBufferDescriptor {
                    cbuf_index: CbufIndex(0),
                    cbuf_offset: offset,
                    count: 1,
                    is_written: program.info.stores_global_memory,
                })?;
        }
    }
    Ok(())
}
