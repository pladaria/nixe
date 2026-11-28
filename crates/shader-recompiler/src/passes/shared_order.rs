use crate::{
    Result,
    ir::{Ir, Opcode, Program},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
}

fn shared_access(opcode: Opcode) -> Option<Access> {
    use Opcode::*;
    match opcode {
        LoadSharedU8 | LoadSharedS8 | LoadSharedU16 | LoadSharedS16 | LoadSharedU32
        | LoadSharedU64 | LoadSharedU128 => Some(Access::Read),
        WriteSharedU8 | WriteSharedU16 | WriteSharedU32 | WriteSharedU64 | WriteSharedU128 => {
            Some(Access::Write)
        }
        _ => None,
    }
}

pub(crate) fn preserve_workgroup_shared_order(ir: &mut Ir, program: &Program) -> Result<()> {
    let mut insertions = Vec::new();
    for &block in &program.blocks {
        let mut converged = false;
        let mut pending = None;
        insertions.clear();
        for (position, &id) in ir.block(block)?.instructions().iter().enumerate() {
            match ir.inst(id)?.opcode() {
                Opcode::Barrier => {
                    converged = true;
                    pending = None;
                }
                opcode => {
                    if let Some(access) = shared_access(opcode) {
                        if pending.is_some_and(|previous| previous != access) {
                            insertions.push(position);
                        }
                        pending = Some(access);
                    }
                }
            }
        }
        // Native lane-ordered shared accesses can implement a prefix scan with
        // a single BAR.SYNC per step. SPIR-V also needs ordering between the
        // stores of one lane and the following loads of its neighbors.
        // A BAR.SYNC in this block already requires every workgroup invocation
        // to execute the block, so the extra barriers keep that same scope.
        // https://docs.vulkan.org/spec/latest/appendices/memorymodel.html
        if converged {
            for &position in insertions.iter().rev() {
                ir.insert(block, position, Opcode::Barrier, &[], 0)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Stage, ir::Value};

    #[test]
    fn orders_shared_accesses_in_workgroup_barrier_blocks() {
        use Opcode::*;
        let mut ir = Ir::new().unwrap();
        let converged = ir.new_block().unwrap();
        let other = ir.new_block().unwrap();
        let mut program = Program::new(Stage::Compute);
        program.blocks.extend([converged, other]);
        let offset = Value::U32(0);
        let count = Value::U32(1);
        for block in [converged, other] {
            ir.append(block, WriteSharedU32, &[offset, count], 0)
                .unwrap();
            ir.append(block, WriteSharedU32, &[offset, count], 0)
                .unwrap();
            ir.append(block, LoadSharedU32, &[offset], 0).unwrap();
            if block == converged {
                ir.append(block, Barrier, &[], 0).unwrap();
            }
            ir.append(block, WriteSharedU32, &[offset, count], 0)
                .unwrap();
            ir.append(block, LoadSharedU32, &[offset], 0).unwrap();
        }

        preserve_workgroup_shared_order(&mut ir, &program).unwrap();
        let opcodes = |block| {
            ir.block(block)
                .unwrap()
                .instructions()
                .iter()
                .map(|&id| ir.inst(id).unwrap().opcode())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            opcodes(converged),
            [
                WriteSharedU32,
                WriteSharedU32,
                Barrier,
                LoadSharedU32,
                Barrier,
                WriteSharedU32,
                Barrier,
                LoadSharedU32,
            ]
        );
        assert_eq!(
            opcodes(other),
            [
                WriteSharedU32,
                WriteSharedU32,
                LoadSharedU32,
                WriteSharedU32,
                LoadSharedU32,
            ]
        );
    }
}
