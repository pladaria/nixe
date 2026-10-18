use super::*;
impl EmitContext<'_> {
    pub(super) fn shared_pointer(&mut self, offset: Id, width: u32, extra: u32) -> Result<Id> {
        let (pointer, object) = match width {
            8 => (self.objects.shared_u8, self.objects.shared_memory_u8),
            16 => (self.objects.shared_u16, self.objects.shared_memory_u16),
            32 => (self.objects.shared_u32, self.objects.shared_memory_u32),
            64 => (self.objects.shared_u32x2, self.objects.shared_memory_u32x2),
            128 => (self.objects.shared_u32x4, self.objects.shared_memory_u32x4),
            _ => return Err(invariant("Invalid shared memory width")),
        };
        let mut index = offset;
        if width > 8 {
            let shift = self.const_u32((width / 8).trailing_zeros())?;
            index = self.op(Op::ShiftRightArithmetic, self.types.u32[0], &[index, shift])?;
        }
        if extra != 0 {
            let extra = self.const_u32(extra)?;
            index = self.op(Op::IAdd, self.types.u32[0], &[index, extra])?;
        }
        if pointer == Id::NONE || object == Id::NONE {
            return Err(unsupported("Shared memory view is not declared"));
        }
        if self.uses_explicit_workgroup_layout {
            self.op(
                Op::AccessChain,
                pointer,
                &[object, self.constants.u32_zero, index],
            )
        } else {
            self.op(Op::AccessChain, pointer, &[object, index])
        }
    }
    pub(super) fn shared_instruction(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let (width, signed, store) = match ir.inst(id)?.opcode() {
            O::LoadSharedU8 => (8, false, false),
            O::LoadSharedS8 => (8, true, false),
            O::LoadSharedU16 => (16, false, false),
            O::LoadSharedS16 => (16, true, false),
            O::LoadSharedU32 => (32, false, false),
            O::LoadSharedU64 => (64, false, false),
            O::LoadSharedU128 => (128, false, false),
            O::WriteSharedU8 => (8, false, true),
            O::WriteSharedU16 => (16, false, true),
            O::WriteSharedU32 => (32, false, true),
            O::WriteSharedU64 => (64, false, true),
            O::WriteSharedU128 => (128, false, true),
            _ => return Ok(None),
        };
        let offset_value = ir.inst(id)?.arg(0)?;
        let offset = self.def(ir, offset_value)?;
        let u = self.types.u32[0];
        let ty = match width {
            8 => self.types.u8,
            16 => self.types.u16,
            32 => u,
            64 => self.types.u32[1],
            _ => self.types.u32[3],
        };
        if self.uses_explicit_workgroup_layout {
            let pointer = self.shared_pointer(offset, width, 0)?;
            if store {
                let value = self.def(ir, ir.inst(id)?.arg(1)?)?;
                let value = if width < 32 {
                    self.op(Op::UConvert, ty, &[value])?
                } else {
                    value
                };
                self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
                return Ok(Some(Id::NONE));
            }
            let value = self.op(Op::Load, ty, &[pointer])?;
            return Ok(Some(if width < 32 {
                self.op(
                    if signed { Op::SConvert } else { Op::UConvert },
                    u,
                    &[value],
                )?
            } else {
                value
            }));
        }
        if width < 32 {
            if store {
                let value = self.def(ir, ir.inst(id)?.arg(1)?)?;
                let func = if width == 8 {
                    self.objects.shared_store_u8_func
                } else {
                    self.objects.shared_store_u16_func
                };
                self.op(Op::FunctionCall, self.types.void, &[func, offset, value])?;
                return Ok(Some(Id::NONE));
            }
            let pointer = self.shared_pointer(offset, 32, 0)?;
            let word = self.op(Op::Load, u, &[pointer])?;
            let shift = self.const_u32(3)?;
            let shift = self.op(Op::ShiftLeftLogical, u, &[offset, shift])?;
            let mask = self.const_u32(if width == 8 { 24 } else { 16 })?;
            let bit = self.op(Op::BitwiseAnd, u, &[shift, mask])?;
            let count = self.const_u32(width)?;
            return Ok(Some(self.op(
                if signed {
                    Op::BitFieldSExtract
                } else {
                    Op::BitFieldUExtract
                },
                u,
                &[word, bit, count],
            )?));
        }
        let n = (width / 32) as usize;
        let mut values = [Id::NONE; 4];
        let stored = if store {
            self.def(ir, ir.inst(id)?.arg(1)?)?
        } else {
            Id::NONE
        };
        for (i, value) in values.iter_mut().enumerate().take(n) {
            let pointer = self.shared_pointer(offset, 32, i as u32)?;
            if store {
                let component = if n == 1 {
                    stored
                } else {
                    self.extract(u, stored, i as u32)?
                };
                self.builder
                    .instruction(Op::Store, &[pointer.0, component.0])?;
            } else {
                *value = self.op(Op::Load, u, &[pointer])?;
            }
        }
        Ok(Some(if store {
            Id::NONE
        } else if n == 1 {
            values[0]
        } else {
            self.op(Op::CompositeConstruct, ty, &values[..n])?
        }))
    }
}
