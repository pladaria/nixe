use super::*;
use spirv::GlslStd450Op as G;

impl EmitContext<'_> {
    fn integer_flags(&mut self, ir: &mut Ir, id: InstId, result: Id) -> Result<()> {
        for (pseudo, op) in [
            (O::GetZeroFromOp, Op::IEqual),
            (O::GetSignFromOp, Op::SLessThan),
        ] {
            if ir.associated_pseudo(id, pseudo)?.is_some() {
                let flag = self.op(op, self.types.bool, &[result, self.constants.u32_zero])?;
                self.pseudo(ir, id, pseudo, flag)?;
            }
        }
        Ok(())
    }
    pub(super) fn arithmetic(&mut self, ir: &mut Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        if !matches!(
            opcode,
            O::IAdd32
                | O::BitwiseAnd32
                | O::BitwiseOr32
                | O::BitwiseXor32
                | O::BitFieldSExtract
                | O::BitFieldUExtract
                | O::SMin32
                | O::SMax32
                | O::SClamp32
                | O::UClamp32
        ) {
            return self.float_or_convert(ir, id);
        }
        let n = ir.inst(id)?.num_args();
        let mut a = [Id::NONE; 3];
        for (i, slot) in a.iter_mut().enumerate().take(n) {
            *slot = self.def(ir, ir.inst(id)?.arg(i)?)?;
        }
        let u = self.types.u32[0];
        let b = self.types.bool;
        let zero = self.constants.u32_zero;
        let result = match opcode {
            O::IAdd32 => {
                let result = if ir.associated_pseudo(id, O::GetCarryFromOp)?.is_some() {
                    let ty = self.builder.type_struct(&[u, u])?;
                    let pair = self.op(Op::IAddCarry, ty, &a[..2])?;
                    let carry = self.extract(u, pair, 1)?;
                    let flag = self.op(Op::INotEqual, b, &[carry, zero])?;
                    self.pseudo(ir, id, O::GetCarryFromOp, flag)?;
                    self.extract(u, pair, 0)?
                } else {
                    self.op(Op::IAdd, u, &a[..2])?
                };
                if ir.associated_pseudo(id, O::GetOverflowFromOp)?.is_some() {
                    let positive = self.op(Op::SGreaterThanEqual, b, &[a[0], zero])?;
                    let max = self.const_u32(0x7fff_ffff)?;
                    let difference = self.op(Op::ISub, u, &[max, a[0]])?;
                    let positive_test = self.op(Op::SGreaterThan, b, &[a[1], difference])?;
                    let negative_test = self.op(Op::SLessThan, b, &[a[1], difference])?;
                    let flag = self.op(Op::Select, b, &[positive, positive_test, negative_test])?;
                    self.pseudo(ir, id, O::GetOverflowFromOp, flag)?;
                }
                result
            }
            O::SMin32 | O::SMax32 => {
                if self.profile.has_broken_signed_operations {
                    for value in &mut a[..2] {
                        *value = self.op(Op::Bitcast, self.types.s32[0], &[*value])?;
                    }
                }
                let result = self.ext(
                    if opcode == O::SMin32 {
                        G::SMin
                    } else {
                        G::SMax
                    },
                    u,
                    &a[..2],
                )?;
                return Ok(Some(if self.profile.has_broken_signed_operations {
                    self.op(Op::Bitcast, u, &[result])?
                } else {
                    result
                }));
            }
            O::SClamp32 | O::UClamp32 => {
                let signed = opcode == O::SClamp32;
                let cast = signed
                    && (self.profile.has_broken_signed_operations
                        || self.profile.has_broken_spirv_clamp);
                let ty = if cast { self.types.s32[0] } else { u };
                if cast {
                    for value in &mut a {
                        *value = self.op(Op::Bitcast, ty, &[*value])?;
                    }
                }
                let result = if self.profile.has_broken_spirv_clamp {
                    let upper =
                        self.ext(if signed { G::SMin } else { G::UMin }, ty, &[a[0], a[2]])?;
                    self.ext(if signed { G::SMax } else { G::UMax }, ty, &[upper, a[1]])?
                } else {
                    self.ext(if signed { G::SClamp } else { G::UClamp }, ty, &a)?
                };
                if cast {
                    self.op(Op::Bitcast, u, &[result])?
                } else {
                    result
                }
            }
            _ => {
                let op = match opcode {
                    O::BitwiseAnd32 => Op::BitwiseAnd,
                    O::BitwiseOr32 => Op::BitwiseOr,
                    O::BitwiseXor32 => Op::BitwiseXor,
                    O::BitFieldSExtract => Op::BitFieldSExtract,
                    O::BitFieldUExtract => Op::BitFieldUExtract,
                    _ => unreachable!(),
                };
                self.op(op, u, &a[..n])?
            }
        };
        self.integer_flags(ir, id, result)?;
        Ok(Some(result))
    }
}
