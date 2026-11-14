use super::instruction::{Instruction, unsupported};
mod alu;
mod geometry;
mod integer;
mod matrix;
mod memory;
mod special;
mod surface;
mod texture;
mod uniform;
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
            6 if i.bits(91, 1) != 0 && i.bits(38, 24) == 0 => self.ureg(i.bits(32, 6)),
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
                    6 | 7 if i.bits(91, 1) != 0 && i.bits(38, 24) == 0 => {
                        (self.ureg(i.bits(32, 6))?, i.bits(62, 1), i.bits(63, 1))
                    }
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
            // Hardware reconvergence tokens have no per-invocation data effect.
            0x945 | 0x941 | 0x942 => Ok(()),
            0xc82 | 0x882 => self.umov(),
            0x3c2 => self.r2ur(),
            0x919 | 0x9c3 => self.special_register(),
            0x88c | 0x28c => self.uisetp(),
            0x81c => self.plop3(),
            0x307 | 0xb07 => self.frnd(),
            0x209 | 0x809 | 0xa09 | 0xc09 => self.fmnmx(),
            0x381 | 0x981 => self.global_load(),
            0x386 | 0x986 => self.global_store(),
            0x388 | 0x984 | 0x988 => self.shared_memory(),
            0xb97 | 0xb9b | 0x998 | 0x99c => self.surface_memory(),
            0x83b => self.matrix_load(),
            0x240 | 0x840 => self.hmnmx(),
            0x23a => self.matrix_transpose(),
            0x23c => self.matrix_multiply(),
            0xb1d => {
                if i.bits(12, 4) != 7
                    || i.bits(16, 16) != 0
                    || i.bits(32, 32) != 0
                    || i.bits(64, 32) & !0x10000 != 0
                {
                    return Err(unsupported("Native workgroup barrier mode"));
                }
                self.emit.barrier().map(|_| ())
            }
            0x224 | 0x424 | 0x624 | 0x824 | 0xa24 | 0xc24 | 0xe24 | 0x2a4 | 0x4a4 | 0x8a4 => {
                self.imad()
            }
            0x2a5 => self.uniform_wide_multiply(),
            0x225 | 0x227 | 0x625 | 0x825 | 0x827 | 0xa27 => self.extended_multiply(),
            0x212 | 0x812 | 0xa12 | 0xc12 | 0x292 | 0x892 => self.lop3(),
            0x20c | 0x80c | 0xa0c | 0xc0c => self.isetp(),
            0x213 | 0xa13 | 0xc13 => {
                if i.bits(24, 8) != 0
                    || i.bits(64, 32) != if i.opcode() == 0xc13 { 0x08000000 } else { 0 }
                    || (i.opcode() == 0x213 && i.bits(32, 32) > 255)
                {
                    return Err(unsupported("Native integer absolute mode"));
                }
                let value = self.raw_b()?;
                let negative = self.emit.i_neg(value)?;
                let absolute = self.emit.s_max(value, negative)?;
                self.write(absolute)
            }
            0x217 | 0x817 | 0xa17 => self.imnmx(),
            0x219 | 0x419 | 0xa19 | 0xc19 => self.shift_variable(),
            0x819 | 0x899 => self.shf(),
            0x211 | 0x811 | 0xa11 | 0xc11 => self.lea(),
            0xb82 => self.ldc(),
            0xab9 | 0xabb => self.uldc(),
            0x803 => self.p2r(),
            0x804 => self.r2p(),
            0x805 => {
                if i.bits(24, 8) != 0
                    || i.bits(32, 32) != 0
                    || i.bits(64, 32) != 0x1ff00
                    || i.bits(16, 8) >= 254
                {
                    return Err(unsupported("Native CS2R source/width mode"));
                }
                // SRZ is the constant-zero special register; CS2R reads a pair.
                let register = Reg(i.bits(16, 8) as u8);
                self.emit.set_reg(register, Value::U32(0))?;
                self.emit.set_reg(register.offset(1)?, Value::U32(0))?;
                Ok(())
            }
            0x305 | 0xb05 | 0xd05 => self.f2i(),
            0x230 => self.hadd2_float(),
            0x23e | 0xa3e => self.pack_floats(),
            0x234 | 0x434 | 0x634 => self.hsetp(),
            0x207 | 0x807 | 0xa07 | 0x208 | 0x808 | 0xa08 => self.select(),
            0x210 | 0x810 | 0xa10 | 0xc10 | 0x290 | 0x890 => self.iadd3(),
            0x300 | 0xd00 | 0x309 | 0xd09 => self.bit_scan(),
            0x806 | 0x886 => self.vote(),
            0x3a1 => self.match_all(),
            0x389 | 0x589 | 0x989 | 0xf89 => self.shuffle(),
            0x822 => self.swizzled_add(),
            0x3a8 | 0x9a8 => self.global_atomic(),
            0x324 | 0x924 => self.geometry_output(),
            0x923 => self.geometry_vertex_index(),
            0x816 => self.permute_bytes(),
            0x245 | 0xa45 | 0xc45 => self.i2fp(),
            0x306 | 0xd06 => self.i2f_narrow(),
            0x308 | 0x908 | 0xb08 | 0xd08 => self.mufu(),
            0x326 | 0x526 => self.interpolate(),
            0xb60 => self.texture_sample(),
            0xb66 | 0xf66 | 0x367 => self.texture_fetch(),
            0xb6f | 0x370 => self.texture_query(),
            0xb69 | 0xf69 | 0x36a => self.texture_lod(),
            0xb63 | 0xf63 => self.texture_gather(),
            0x98e => self.global_reduction(),
            0x20a | 0x80a | 0xa0a | 0xc0a | 0x20b | 0x80b | 0xa0b | 0xc0b => self.fset(),
            0x94d => crate::frontend::export_fragment(&mut self.emit, self.metadata),
            0x91a => {
                if i.bits(0, 32) != 0x791a
                    || i.bits(32, 32) & !0x7fff != 0x8000
                    || i.bits(64, 32) != 0
                {
                    return Err(unsupported("Native dependency barrier mode"));
                }
                // Scoreboard completion is represented by the semantic IR's
                // data dependencies; this is not a memory or subgroup barrier.
                Ok(())
            }
            0x918 => {
                if i.bits(16, 32) != 0 || i.bits(48, 32) != 0 || i.bits(80, 12) != 0 {
                    return Err(unsupported("Native NOP operands"));
                }
                Ok(())
            }
            0x202 | 0x802 | 0xa02 | 0xc02 => {
                if i.bits(72, 4) != 15
                    || i.bits(76, 16) != if i.opcode() == 0xc02 { 0x8000 } else { 0 }
                {
                    return Err(unsupported("Native MOV modifiers"));
                }
                let value = match i.opcode() {
                    0x202 if i.bits(40, 24) == 0 => self.reg(i.bits(32, 8))?,
                    0x802 => Value::U32(i.bits(32, 32)),
                    0xa02 if i.bits(62, 2) == 0 => self.constant()?,
                    0xc02 if i.bits(32, 32) < 64 => self.ureg(i.bits(32, 6))?,
                    _ => return Err(unsupported("Native MOV source encoding")),
                };
                self.emit
                    .set_reg(Reg(i.bits(16, 8) as u8), value)
                    .map(|_| ())
            }
            0x220 | 0x820 | 0xa20 | 0xc20 | 0x221 | 0x421 | 0x621 | 0x821 | 0xa21 | 0xe21
            | 0x223 | 0x423 | 0x623 | 0x823 | 0xa23 | 0xc23 | 0xe23 => {
                let op = i.opcode() & 0x1ff;
                if op == 0x20 {
                    let scale = i.bits(84, 3);
                    if i.bits(76, 1) != 0
                        || scale == 0
                        || i.bits(81, 3) != 0
                        || i.bits(87, 4) != 0
                        || i.bits(91, 1) != u32::from(i.opcode() == 0xc20)
                        || (scale != 4 && (i.bits(80, 1) == 0 || i.bits(78, 2) != 0))
                    {
                        return Err(unsupported("Native FMUL mode"));
                    }
                } else if i.bits(81, 10) != 0
                    || i.bits(91, 1) != u32::from(matches!(i.opcode(), 0xc23 | 0xe23 | 0xe21))
                    || (op == 0x23 && i.bits(76, 1) != 0)
                {
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
