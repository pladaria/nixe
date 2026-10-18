use super::*;
use spirv::GlslStd450Op as G;
impl EmitContext<'_> {
    pub(super) fn float_or_convert(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        let comparison = match opcode {
            O::FPOrdNotEqual16 | O::FPOrdNotEqual32 | O::FPOrdNotEqual64 => Some(Op::FOrdNotEqual),
            O::FPUnordEqual16 | O::FPUnordEqual32 | O::FPUnordEqual64 => Some(Op::FUnordEqual),
            O::FPUnordLessThan16 | O::FPUnordLessThan32 | O::FPUnordLessThan64 => {
                Some(Op::FUnordLessThan)
            }
            O::FPUnordGreaterThan16 | O::FPUnordGreaterThan32 | O::FPUnordGreaterThan64 => {
                Some(Op::FUnordGreaterThan)
            }
            O::FPUnordLessThanEqual16 | O::FPUnordLessThanEqual32 | O::FPUnordLessThanEqual64 => {
                Some(Op::FUnordLessThanEqual)
            }
            O::FPUnordGreaterThanEqual16
            | O::FPUnordGreaterThanEqual32
            | O::FPUnordGreaterThanEqual64 => Some(Op::FUnordGreaterThanEqual),
            _ => None,
        };
        if let Some(op) = comparison {
            let lhs = self.def(ir, ir.inst(id)?.arg(0)?)?;
            let rhs = self.def(ir, ir.inst(id)?.arg(1)?)?;
            let ty = self.types.bool;
            let special = self.profile.ignore_nan_fp_comparisons;
            let ordered = op == Op::FOrdNotEqual;
            let mut result = self.op(
                if special && ordered {
                    Op::FOrdEqual
                } else {
                    op
                },
                ty,
                &[lhs, rhs],
            )?;
            if special {
                for arg in [lhs, rhs] {
                    let mut nan = self.op(Op::IsNan, ty, &[arg])?;
                    if ordered {
                        nan = self.op(Op::LogicalNot, ty, &[nan])?;
                    }
                    result = self.op(
                        if ordered {
                            Op::LogicalAnd
                        } else {
                            Op::LogicalOr
                        },
                        ty,
                        &[result, nan],
                    )?;
                }
            }
            return Ok(Some(result));
        }
        let width = match opcode {
            O::FPSaturate16 | O::FPClamp16 => 16,
            O::FPSaturate32 | O::FPClamp32 | O::FPRecip32 => 32,
            O::FPSaturate64 | O::FPClamp64 | O::FPRecip64 => 64,
            _ => return self.convert(ir, id),
        };
        let ty = match width {
            16 => self.types.f16[0],
            32 => self.types.f32[0],
            _ => self.types.f64[0],
        };
        let value = self.def(ir, ir.inst(id)?.arg(0)?)?;
        let one_bits: &[u32] = match width {
            16 => &[0x3c00],
            32 => &[1.0f32.to_bits()],
            _ => &[0, 0x3ff0_0000],
        };
        let result = if matches!(opcode, O::FPRecip32 | O::FPRecip64) {
            let one = self.builder.constant(ty, one_bits)?;
            self.op(Op::FDiv, ty, &[one, value])?
        } else {
            let (min, max) = if matches!(opcode, O::FPClamp16 | O::FPClamp32 | O::FPClamp64) {
                (
                    self.def(ir, ir.inst(id)?.arg(1)?)?,
                    self.def(ir, ir.inst(id)?.arg(2)?)?,
                )
            } else {
                (
                    self.builder
                        .constant(ty, if width == 64 { &[0, 0] } else { &[0] })?,
                    self.builder.constant(ty, one_bits)?,
                )
            };
            if self.profile.has_broken_spirv_clamp {
                let lower = self.ext(G::FMax, ty, &[value, min])?;
                self.ext(G::FMin, ty, &[lower, max])?
            } else {
                self.ext(G::FClamp, ty, &[value, min, max])?
            }
        };
        Ok(Some(result))
    }
}
