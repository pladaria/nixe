use super::*;
impl EmitContext<'_> {
    fn narrow_integer(&mut self, value: Id, width: u32, signed: bool) -> Result<Id> {
        let supported = if width == 8 {
            self.profile.support_int8
        } else {
            self.profile.support_int16
        };
        if supported {
            let ty = match (width, signed) {
                (8, false) => self.types.u8,
                (8, true) => self.types.s8,
                (_, false) => self.types.u16,
                (_, true) => self.types.s16,
            };
            self.op(
                if signed { Op::SConvert } else { Op::UConvert },
                ty,
                &[value],
            )
        } else {
            let count = self.const_u32(width)?;
            self.op(
                if signed {
                    Op::BitFieldSExtract
                } else {
                    Op::BitFieldUExtract
                },
                self.types.u32[0],
                &[value, self.constants.u32_zero, count],
            )
        }
    }
    pub(super) fn convert(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        let route = match opcode {
            O::ConvertS16F16 => (0, 16, true),
            O::ConvertU16F16 => (0, 16, false),
            O::ConvertF16S8 => (1, 8, true),
            O::ConvertF16U8 => (1, 8, false),
            O::ConvertF16S16 => (1, 16, true),
            O::ConvertF16U16 => (1, 16, false),
            O::ConvertS16F32 => (0, 16, true),
            O::ConvertU16F32 => (0, 16, false),
            O::ConvertF32S8 => (1, 8, true),
            O::ConvertF32U8 => (1, 8, false),
            O::ConvertF32S16 => (1, 16, true),
            O::ConvertF32U16 => (1, 16, false),
            O::ConvertS16F64 => (0, 16, true),
            O::ConvertU16F64 => (0, 16, false),
            O::ConvertF64S8 => (1, 8, true),
            O::ConvertF64U8 => (1, 8, false),
            O::ConvertF64S16 => (1, 16, true),
            O::ConvertF64U16 => (1, 16, false),
            O::ConvertU8U32 => (2, 8, false),
            O::ConvertU32U8 => (3, 8, false),
            O::ConvertS32S8 => (3, 8, true),
            O::ConvertU16U32 => (2, 16, false),
            O::ConvertU32U16 => (3, 16, false),
            O::ConvertS32S16 => (3, 16, true),
            O::ConvertS32F32 => (4, 32, true),
            O::ConvertF32S32 | O::ConvertF64S32 => (5, 32, true),
            O::ConvertF16F32 => (6, 16, false),
            _ => return Ok(None),
        };
        let (route, width, signed) = route;
        let value = self.def(ir, ir.inst(id)?.arg(0)?)?;
        let u = self.types.u32[0];
        let ty = if matches!(route, 1 | 5 | 6) {
            self.types.get(opcode.signature().result)?
        } else {
            u
        };
        let result = match route {
            0 => {
                let dest = if self.profile.support_int16 {
                    self.types.u16
                } else {
                    u
                };
                let value = self.op(
                    if signed {
                        Op::ConvertFToS
                    } else {
                        Op::ConvertFToU
                    },
                    dest,
                    &[value],
                )?;
                if self.profile.support_int16 {
                    self.op(
                        if signed { Op::SConvert } else { Op::UConvert },
                        u,
                        &[value],
                    )?
                } else {
                    self.narrow_integer(value, 16, signed)?
                }
            }
            1 => {
                let value = self.narrow_integer(value, width, signed)?;
                self.op(
                    if signed {
                        Op::ConvertSToF
                    } else {
                        Op::ConvertUToF
                    },
                    ty,
                    &[value],
                )?
            }
            2 => self.narrow_integer(value, width, false)?,
            3 => {
                let supported = if width == 8 {
                    self.profile.support_int8
                } else {
                    self.profile.support_int16
                };
                if supported {
                    self.op(
                        if signed { Op::SConvert } else { Op::UConvert },
                        u,
                        &[value],
                    )?
                } else {
                    self.narrow_integer(value, width, signed)?
                }
            }
            4 => {
                let signed_ty = if self.profile.has_broken_signed_operations {
                    self.types.s32[0]
                } else {
                    u
                };
                let converted = self.op(Op::ConvertFToS, signed_ty, &[value])?;
                if self.profile.has_broken_signed_operations {
                    self.op(Op::Bitcast, u, &[converted])?
                } else {
                    converted
                }
            }
            5 => {
                let value = if self.profile.has_broken_signed_operations {
                    self.op(Op::Bitcast, self.types.s32[0], &[value])?
                } else {
                    value
                };
                self.op(Op::ConvertSToF, ty, &[value])?
            }
            6 => {
                let converted = self.op(Op::FConvert, ty, &[value])?;
                if cfg!(target_os = "android") {
                    converted
                } else {
                    let nan = self.op(Op::IsNan, self.types.bool, &[converted])?;
                    let zero = self.builder.constant(ty, &[0])?;
                    self.op(Op::Select, ty, &[nan, zero, converted])?
                }
            }
            _ => unreachable!(),
        };
        Ok(Some(result))
    }
}
