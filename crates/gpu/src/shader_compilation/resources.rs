//! Proven constant-buffer coverage for native resource preparation.
use nixe_shader_recompiler::{
    Result,
    arena::InstId,
    artifact::{MAX_CBUFS, ShaderInfo},
    ir::{Ir, Opcode as O, Program, Value},
};
use std::collections::HashMap;

/// Tightens resource coverage without modifying shader instructions or their ABI.
/// Unknown offsets retain the compiler's conservative coverage. In particular,
/// the declared buffer limit is not evidence that every byte is actually read.
pub fn resource_info(ir: &Ir, program: &Program) -> Result<ShaderInfo> {
    let mut info = program.info.clone();
    if info.uses_cbuf_indirect {
        return Ok(info);
    }
    let mut sizes = [0u32; MAX_CBUFS];
    let mut bounds = HashMap::new();
    for &block in &program.post_order {
        for &id in ir.block(block)?.instructions() {
            let inst = ir.inst(id)?;
            let width = match inst.opcode() {
                O::GetCbufU8 | O::GetCbufS8 => 1,
                O::GetCbufU16 | O::GetCbufS16 => 2,
                O::GetCbufU32 | O::GetCbufF32 => 4,
                O::GetCbufU32x2 => 8,
                _ => continue,
            };
            let Value::U32(index) = ir.resolve(inst.arg(0)?)? else {
                return Ok(program.info.clone());
            };
            let Some(size) = sizes.get_mut(index as usize) else {
                return Ok(program.info.clone());
            };
            let bound = upper_bound(ir, inst.arg(1)?, &mut bounds, 64)?;
            let end = bound
                .checked_add(width)
                .and_then(|end| end.checked_add(15))
                .map_or(u32::MAX, |end| end & !15);
            *size = (*size).max(end);
        }
    }
    for (size, bound) in info.constant_buffer_used_sizes.iter_mut().zip(sizes) {
        if bound != 0 {
            *size = (*size).min(bound);
        }
    }
    Ok(info)
}

fn upper_bound(ir: &Ir, value: Value, memo: &mut HashMap<InstId, u32>, depth: u32) -> Result<u32> {
    let id = match ir.resolve(value)? {
        Value::U32(value) => return Ok(value),
        Value::Inst(id) if depth != 0 => id,
        _ => return Ok(u32::MAX),
    };
    if let Some(&bound) = memo.get(&id) {
        return Ok(bound);
    }
    let inst = ir.inst(id)?;
    let mut arg = |index| upper_bound(ir, inst.arg(index)?, memo, depth - 1);
    let bound = match inst.opcode() {
        O::UMin32 | O::BitwiseAnd32 => arg(0)?.min(arg(1)?),
        O::IAdd32 => arg(0)?.saturating_add(arg(1)?),
        O::IMul32 => arg(0)?.saturating_mul(arg(1)?),
        O::ShiftLeftLogical32 => match ir.resolve(inst.arg(1)?)? {
            Value::U32(shift) if shift < 32 => arg(0)?.saturating_mul(1u32 << shift),
            _ => u32::MAX,
        },
        // Phi cycles, signed arithmetic and unchecked shifts are not proofs.
        _ => u32::MAX,
    };
    memo.insert(id, bound);
    Ok(bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_shader_recompiler::{
        Stage,
        arena::BlockId,
        environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
        passes::collect_shader_info,
    };

    fn emit(ir: &mut Ir, block: BlockId, opcode: O, args: &[Value]) -> Value {
        Value::Inst(ir.append(block, opcode, args, 0).unwrap())
    }

    #[test]
    fn bounded_jump_table_imports_only_its_proven_coverage() {
        let mut ir = Ir::new().unwrap();
        let block = ir.new_block().unwrap();
        let mut program = Program::new(Stage::VertexB);
        program.blocks.push(block);
        program.post_order.push(block);
        let input = emit(&mut ir, block, O::UndefU32, &[]);
        let index = emit(&mut ir, block, O::UMin32, &[input, Value::U32(12)]);
        let offset = emit(
            &mut ir,
            block,
            O::ShiftLeftLogical32,
            &[index, Value::U32(2)],
        );
        let offset = emit(&mut ir, block, O::Identity, &[offset]);
        emit(&mut ir, block, O::GetCbufU32, &[Value::U32(1), offset]);
        let env = SnapshotEnvironment::new(
            EnvironmentMetadata {
                stage: Stage::VertexB,
                ..Default::default()
            },
            Observations::default(),
        );
        collect_shader_info(&env, &ir, &mut program).unwrap();
        assert_eq!(program.info.constant_buffer_used_sizes[1], 0x10000);
        let info = resource_info(&ir, &program).unwrap();
        // Thirteen words occupy 52 bytes, rounded to the uniform ABI's 16 bytes.
        assert_eq!(info.constant_buffer_used_sizes[1], 64);
        assert_eq!(
            info.constant_buffer_descriptors,
            program.info.constant_buffer_descriptors
        );
        assert_eq!(program.info.constant_buffer_used_sizes[1], 0x10000);

        emit(&mut ir, block, O::GetCbufU32, &[Value::U32(1), input]);
        assert_eq!(
            resource_info(&ir, &program)
                .unwrap()
                .constant_buffer_used_sizes[1],
            0x10000
        );
    }

    #[test]
    fn overflow_and_unknown_values_never_become_small_unsigned_bounds() {
        let mut ir = Ir::new().unwrap();
        let block = ir.new_block().unwrap();
        let input = emit(&mut ir, block, O::UndefU32, &[]);
        let masked = emit(&mut ir, block, O::BitwiseAnd32, &[input, Value::U32(255)]);
        let scaled = emit(&mut ir, block, O::IMul32, &[masked, Value::U32(48)]);
        let offset = emit(&mut ir, block, O::IAdd32, &[scaled, Value::U32(12)]);
        let mut memo = HashMap::new();
        assert_eq!(
            upper_bound(&ir, offset, &mut memo, 64).unwrap(),
            255 * 48 + 12
        );
        let wrapped = emit(&mut ir, block, O::IAdd32, &[masked, Value::U32(u32::MAX)]);
        assert_eq!(upper_bound(&ir, wrapped, &mut memo, 64).unwrap(), u32::MAX);
        let shifted = emit(
            &mut ir,
            block,
            O::ShiftLeftLogical32,
            &[masked, Value::U32(31)],
        );
        assert_eq!(upper_bound(&ir, shifted, &mut memo, 64).unwrap(), u32::MAX);
        let unknown_shift = emit(&mut ir, block, O::ShiftLeftLogical32, &[masked, input]);
        assert_eq!(
            upper_bound(&ir, unknown_shift, &mut memo, 64).unwrap(),
            u32::MAX
        );
        assert_eq!(upper_bound(&ir, offset, &mut memo, 0).unwrap(), u32::MAX);
    }
}
