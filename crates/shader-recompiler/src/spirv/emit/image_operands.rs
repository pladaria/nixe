use super::*;
use spirv::ImageOperands as M;
#[derive(Default)]
pub(super) struct Operands {
    pub mask: u32,
    args: Vec<u32>,
}
impl Operands {
    pub fn add(&mut self, mask: M, values: &[Id]) {
        self.mask |= mask.bits();
        self.args.extend(values.iter().map(|id| id.0));
    }
    pub fn append(&self, words: &mut Vec<u32>, required: bool) {
        if required || self.mask != 0 {
            words.push(self.mask);
            words.extend_from_slice(&self.args);
        }
    }
    pub fn offset(
        &mut self,
        ctx: &mut EmitContext<'_>,
        ir: &Ir,
        value: Value,
        dynamic: bool,
    ) -> Result<()> {
        let value = ir.resolve(value)?;
        if value == Value::None {
            return Ok(());
        }
        if let Value::U32(bits) = value {
            let value = ctx.builder.constant(ctx.types.s32[0], &[bits])?;
            self.add(M::CONST_OFFSET, &[value]);
            return Ok(());
        }
        if let Value::Inst(id) = value {
            let inst = ir.inst(id)?;
            let n = match inst.opcode() {
                O::CompositeConstructU32x2 => 2,
                O::CompositeConstructU32x3 => 3,
                O::CompositeConstructU32x4 => 4,
                _ => 0,
            };
            if n > 0 && (0..n).all(|i| matches!(inst.arg(i), Ok(Value::U32(_)))) {
                let mut components = [Id::NONE; 4];
                for (i, component) in components.iter_mut().enumerate().take(n) {
                    *component = ctx
                        .builder
                        .constant(ctx.types.s32[0], &[literal(ir, inst.arg(i)?)?])?;
                }
                let constant = ctx
                    .builder
                    .constant_composite(ctx.types.s32[n - 1], &components[..n])?;
                self.add(M::CONST_OFFSET, &[constant]);
                return Ok(());
            }
        }
        if dynamic {
            self.add(M::OFFSET, &[ctx.def(ir, value)?]);
        }
        Ok(())
    }
    pub fn sample(
        ctx: &mut EmitContext<'_>,
        ir: &Ir,
        bias: bool,
        lod: bool,
        clamp: bool,
        value: Id,
        offset: Value,
    ) -> Result<Self> {
        let mut operands = Self::default();
        if bias || lod {
            let value = if clamp {
                ctx.extract(ctx.types.f32[0], value, 0)?
            } else {
                value
            };
            operands.add(if bias { M::BIAS } else { M::LOD }, &[value]);
        }
        operands.offset(ctx, ir, offset, false)?;
        if clamp {
            let value = if bias {
                ctx.extract(ctx.types.f32[0], value, 1)?
            } else {
                value
            };
            operands.add(M::MIN_LOD, &[value]);
        }
        Ok(operands)
    }
    pub fn gather(
        ctx: &mut EmitContext<'_>,
        ir: &Ir,
        offset: Value,
        offset2: Value,
    ) -> Result<Self> {
        let mut operands = Self::default();
        if offset2 == Value::None {
            operands.offset(ctx, ir, offset, true)?;
            return Ok(operands);
        }
        let mut words = [[0; 4]; 2];
        for (out, value) in words.iter_mut().zip([offset, offset2]) {
            let Value::Inst(id) = ir.resolve(value)? else {
                return Err(invariant("Gather PTP offset must be a composite"));
            };
            let inst = ir.inst(id)?;
            for i in 0..inst.num_args() {
                if !ir.is_immediate(inst.arg(i)?)? {
                    return Ok(operands);
                }
            }
            if inst.opcode() != O::CompositeConstructU32x4 {
                return Err(invariant(
                    "Gather PTP offset must contain four U32 components",
                ));
            }
            for (i, word) in out.iter_mut().enumerate() {
                *word = literal(ir, inst.arg(i)?)?;
            }
        }
        let mut pairs = [Id::NONE; 4];
        for (i, pair) in pairs.iter_mut().enumerate() {
            let a = ctx
                .builder
                .constant(ctx.types.s32[0], &[words[i / 2][i % 2 * 2]])?;
            let b = ctx
                .builder
                .constant(ctx.types.s32[0], &[words[i / 2][i % 2 * 2 + 1]])?;
            *pair = ctx.builder.constant_composite(ctx.types.s32[1], &[a, b])?;
        }
        let four = ctx.const_u32(4)?;
        let ty = ctx.builder.type_array(ctx.types.s32[1], four)?;
        let offsets = ctx.builder.constant_composite(ty, &pairs)?;
        operands.add(M::CONST_OFFSETS, &[offsets]);
        Ok(operands)
    }
}
