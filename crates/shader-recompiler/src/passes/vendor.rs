use super::fold::{arg, node};
use crate::{
    Result,
    ir::{Emitter, Ir, Opcode as O, Program, Value},
};

pub fn vendor_workaround(ir: &mut Ir, p: &Program) -> Result<()> {
    for &block in &p.post_order {
        for id in ir.block(block)?.instructions().to_vec() {
            if ir.inst(id)?.opcode() != O::IAdd32 {
                continue;
            }
            let (Some(left), Some(right)) = (
                node(ir, arg(ir, id, 0)?, O::ShiftLeftLogical32)?,
                node(ir, arg(ir, id, 1)?, O::BitFieldUExtract)?,
            ) else {
                continue;
            };
            if arg(ir, left, 1)? != Value::U32(16)
                || arg(ir, right, 1)? != Value::U32(16)
                || arg(ir, right, 2)? != Value::U32(16)
            {
                continue;
            }
            let mut low = arg(ir, left, 0)?;
            if let Some(mul) = node(ir, low, O::IMul32)? {
                low = arg(ir, mul, 0)?;
            }
            let Some(low) = node(ir, low, O::BitFieldUExtract)? else {
                continue;
            };
            if arg(ir, low, 1)? != Value::U32(0) || arg(ir, low, 2)? != Value::U32(16) {
                continue;
            }
            let args = [ir.inst(id)?.arg(0)?, ir.inst(id)?.arg(1)?];
            let mut e = Emitter::before(ir, block, id)?;
            let result = e.emit(O::BitwiseOr32, &args, 0)?;
            e.ir.replace_uses_with(id, result)?;
        }
    }
    Ok(())
}
