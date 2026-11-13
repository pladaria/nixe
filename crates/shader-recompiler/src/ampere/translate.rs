use super::instruction::{Instruction, unsupported};
mod alu;
mod texture;
use crate::{
    Environment, Result,
    arena::BlockId,
    frontend::asl::CodeTranslator,
    ir::{Attribute, Emitter, FmzMode, FpControl, FpRounding, Ir, Pred, Reg, Type, Value},
    types::ProgramOffset,
};

pub struct AmpereTranslator;
impl<E: Environment> CodeTranslator<E> for AmpereTranslator {
    fn translate(
        &mut self,
        env: &mut E,
        ir: &mut Ir,
        block: BlockId,
        begin: ProgramOffset,
        end: ProgramOffset,
    ) -> Result<()> {
        for pc in (begin.0..end.0).step_by(16) {
            let instruction = Instruction::fetch(env, ProgramOffset(pc))?;
            let mut visitor = Visitor {
                metadata: env.metadata(),
                emit: Emitter::new(ir, block)?,
                instruction,
            };
            visitor.translate().map_err(|e| {
                let detail = format!("{e}; native instruction={:032x}", instruction.0);
                unsupported(detail).at(
                    env.metadata().stage,
                    ProgramOffset(pc),
                    Some(instruction.0 as u64),
                )
            })?;
        }
        Ok(())
    }
}
struct Visitor<'a> {
    metadata: &'a crate::environment::EnvironmentMetadata,
    emit: Emitter<'a>,
    instruction: Instruction,
}
impl Visitor<'_> {
    fn combine(&mut self, a: Value, b: Value, operation: u32) -> Result<Value> {
        match operation {
            0 => self.emit.logical_and(a, b),
            1 => self.emit.logical_or(a, b),
            2 => self.emit.logical_xor(a, b),
            _ => Err(unsupported("Native predicate combination")),
        }
    }
    fn predicate_pair(&mut self, comparison: Value) -> Result<()> {
        let i = self.instruction;
        let input = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let first = self.combine(comparison, input, i.bits(74, 2))?;
        let inverse = self.emit.logical_not(comparison)?;
        let second = self.combine(inverse, input, i.bits(74, 2))?;
        self.emit.set_pred(Pred::new(i.bits(81, 3) as u8)?, first)?;
        self.emit.set_pred(Pred::new(i.bits(84, 3) as u8)?, second)
    }
    fn reg(&mut self, index: u32) -> Result<Value> {
        self.emit.get_reg(Reg(index as u8))
    }
    fn raw_b(&mut self) -> Result<Value> {
        let i = self.instruction;
        match i.bits(9, 3) {
            1 => self.reg(i.bits(32, 8)),
            4 => Ok(Value::U32(i.bits(32, 32))),
            5 => self.constant(),
            _ => Err(unsupported("Native integer source form")),
        }
    }
    fn write(&mut self, value: Value) -> Result<()> {
        self.emit
            .set_reg(Reg(self.instruction.bits(16, 8) as u8), value)
            .map(|_| ())
    }
    fn constant(&mut self) -> Result<Value> {
        let i = self.instruction;
        let offset = i.bits(38, 16);
        if offset & 3 != 0 || i.bits(91, 1) != 0 {
            return Err(unsupported("Native constant-buffer indexing mode"));
        }
        self.emit
            .get_cbuf(Value::U32(i.bits(54, 5)), Value::U32(offset), 32, false)
    }
    fn operand(&mut self, which: u32) -> Result<Value> {
        let i = self.instruction;
        let form = i.bits(9, 3);
        let (value, abs, neg) = if which == 0 {
            (self.reg(i.bits(24, 8))?, i.bits(73, 1), i.bits(72, 1))
        } else {
            if !(1..=7).contains(&form) {
                return Err(unsupported("Native ALU source form"));
            }
            let wide = if which == 1 {
                matches!(form, 4..=6)
            } else {
                matches!(form, 2 | 3 | 7)
            };
            if wide {
                match form {
                    2 | 4 => (Value::U32(i.bits(32, 32)), 0, 0),
                    3 | 5 => (self.constant()?, i.bits(62, 1), i.bits(63, 1)),
                    _ => return Err(unsupported("Native uniform register source")),
                }
            } else {
                let upper = which != 1 || matches!(form, 2 | 3 | 7);
                (
                    self.reg(i.bits(if upper { 64 } else { 32 }, 8))?,
                    i.bits(if upper { 74 } else { 62 }, 1),
                    i.bits(if upper { 75 } else { 63 }, 1),
                )
            }
        };
        let value = self.emit.bit_cast(Type::F32, value)?;
        self.emit.fp_abs_neg(value, abs != 0, neg != 0)
    }
    fn write_float(&mut self, value: Value) -> Result<()> {
        let value = self.emit.bit_cast(Type::U32, value)?;
        self.emit
            .set_reg(Reg(self.instruction.bits(16, 8) as u8), value)
            .map(|_| ())
    }
    fn translate(&mut self) -> Result<()> {
        let i = self.instruction;
        match i.opcode() {
            0x207 | 0x807 | 0xa07 | 0x208 | 0x808 | 0xa08 => self.select(),
            0x210 | 0x810 | 0xa10 => self.iadd3(),
            0x245 | 0xa45 => self.i2fp(),
            0x308 | 0x908 | 0xb08 => self.mufu(),
            0x326 => self.interpolate(),
            0xb60 => self.texture_sample(),
            0x20b | 0x80b | 0xa0b => {
                if i.bits(64, 8) != 0 || i.bits(91, 1) != 0 {
                    return Err(unsupported("Native floating predicate mode"));
                }
                let a = self.operand(0)?;
                let b = self.operand(1)?;
                let control = FpControl {
                    fmz_mode: if i.bits(80, 1) != 0 {
                        FmzMode::FTZ
                    } else {
                        FmzMode::None
                    },
                    ..Default::default()
                };
                let comparison =
                    crate::frontend::float_compare(&mut self.emit, a, b, i.bits(76, 4), control)?;
                self.predicate_pair(comparison)
            }
            0x94d => crate::frontend::export_fragment(&mut self.emit, self.metadata),
            0x918 => {
                if i.bits(16, 32) != 0 || i.bits(48, 32) != 0 || i.bits(80, 12) != 0 {
                    return Err(unsupported("Native NOP operands"));
                }
                Ok(())
            }
            0x202 | 0x802 | 0xa02 => {
                if i.bits(72, 4) != 15 || i.bits(76, 16) != 0 {
                    return Err(unsupported("Native MOV modifiers"));
                }
                let value = match i.opcode() {
                    0x202 if i.bits(40, 24) == 0 => self.reg(i.bits(32, 8))?,
                    0x802 => Value::U32(i.bits(32, 32)),
                    0xa02 if i.bits(62, 2) == 0 => self.constant()?,
                    _ => return Err(unsupported("Native MOV source encoding")),
                };
                self.emit
                    .set_reg(Reg(i.bits(16, 8) as u8), value)
                    .map(|_| ())
            }
            0x220 | 0x820 | 0xa20 | 0x221 | 0x421 | 0x621 | 0x821 | 0xa21 | 0x223 | 0x423
            | 0x623 | 0x823 | 0xa23 => {
                let op = i.opcode() & 0x1ff;
                if op == 0x20 {
                    let scale = i.bits(84, 3);
                    if i.bits(76, 1) != 0
                        || scale == 0
                        || i.bits(81, 3) != 0
                        || i.bits(87, 5) != 0
                        || (scale != 4 && (i.bits(80, 1) == 0 || i.bits(78, 2) != 0))
                    {
                        return Err(unsupported("Native FMUL mode"));
                    }
                } else if i.bits(81, 11) != 0 || (op == 0x23 && i.bits(76, 1) != 0) {
                    return Err(unsupported("Native floating ALU mode"));
                }
                let control = FpControl {
                    no_contraction: true,
                    rounding: match i.bits(78, 2) {
                        0 => FpRounding::RN,
                        1 => FpRounding::RM,
                        2 => FpRounding::RP,
                        _ => FpRounding::RZ,
                    },
                    fmz_mode: if i.bits(80, 1) != 0 {
                        FmzMode::FTZ
                    } else {
                        FmzMode::None
                    },
                };
                let mut a = self.operand(0)?;
                let b_slot = if op == 0x21 && matches!(i.bits(9, 3), 2 | 3 | 7) {
                    2
                } else {
                    1
                };
                let b = self.operand(b_slot)?;
                let mut value = match op {
                    0x20 => {
                        let scale = i.bits(84, 3);
                        if scale != 4 {
                            let factor =
                                [0.0f32, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0][scale as usize];
                            a = self.emit.fp_mul(
                                a,
                                Value::F32(factor.to_bits()),
                                FpControl::default(),
                            )?;
                        }
                        self.emit.fp_mul(a, b, control)?
                    }
                    0x21 => self.emit.fp_add(a, b, control)?,
                    _ => {
                        let c = self.operand(2)?;
                        self.emit.fp_fma(a, b, c, control)?
                    }
                };
                if i.bits(77, 1) != 0 {
                    value = self.emit.fp_saturate(value)?;
                }
                self.write_float(value)
            }
            0x321 | 0x322 => {
                let address = i.bits(40, 10);
                let count = i.bits(74, 2) + 1;
                if i.bits(76, 4) != 0
                    || i.bits(24, 8) != 255
                    || address & 3 != 0
                    || address + count * 4 > 0x400
                {
                    return Err(unsupported("Native indexed/patch/output attribute access"));
                }
                let store = i.opcode() == 0x322;
                let reg = Reg(i.bits(if store { 32 } else { 16 }, 8) as u8);
                let vertex = self.reg(i.bits(if store { 64 } else { 32 }, 8))?;
                for component in 0..count {
                    let attribute = Attribute::from_index((address / 4 + component) as u16)
                        .ok_or_else(|| unsupported("Unknown native attribute"))?;
                    let reg = reg.offset(component as i32)?;
                    if store {
                        let value = self.emit.get_reg(reg)?;
                        let value = self.emit.bit_cast(Type::F32, value)?;
                        self.emit.set_attribute(attribute, value, vertex)?;
                    } else {
                        let value = self.emit.get_attribute_at_vertex(attribute, vertex)?;
                        let value = self.emit.bit_cast(Type::U32, value)?;
                        self.emit.set_reg(reg, value)?;
                    }
                }
                Ok(())
            }
            _ => Err(unsupported(format!(
                "Native shader opcode {:#x} is not implemented",
                i.opcode()
            ))),
        }
    }
}
