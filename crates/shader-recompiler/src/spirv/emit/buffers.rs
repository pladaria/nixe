use super::*;
use crate::spirv::BufferKind as K;
impl EmitContext<'_> {
    pub(super) fn buffer_index(
        &mut self,
        ir: &Ir,
        offset: Value,
        bytes: u32,
        extra: u32,
    ) -> Result<Id> {
        if let Value::U32(value) = ir.resolve(offset)? {
            return self.const_u32((value / bytes).wrapping_add(extra));
        }
        let mut index = self.def(ir, offset)?;
        if bytes > 1 {
            let shift = self.const_u32(bytes.trailing_zeros())?;
            index = self.op(Op::ShiftRightLogical, self.types.u32[0], &[index, shift])?;
        }
        if extra != 0 {
            let extra = self.const_u32(extra)?;
            index = self.op(Op::IAdd, self.types.u32[0], &[index, extra])?;
        }
        Ok(index)
    }
    pub(super) fn storage_pointer(
        &mut self,
        ir: &Ir,
        binding: Value,
        offset: Value,
        kind: K,
        extra: u32,
    ) -> Result<Id> {
        let binding = match ir.resolve(binding)? {
            Value::U32(v) => v as usize,
            _ => return Err(unsupported("Dynamic storage buffer indexing")),
        };
        let object = self
            .ssbos
            .get(binding)
            .ok_or_else(|| invariant("Storage buffer index out of range"))?[kind as usize];
        let (_, bytes) = self.buffer_element(kind);
        let index = self.buffer_index(ir, offset, bytes, extra)?;
        let pointer = self.storage_types[kind as usize].element;
        if object == Id::NONE || pointer == Id::NONE {
            return Err(unsupported("Storage buffer view is not declared"));
        }
        self.op(
            Op::AccessChain,
            pointer,
            &[object, self.constants.u32_zero, index],
        )
    }
    fn cbuf_load(&mut self, ir: &Ir, binding: Value, offset: Value, kind: K) -> Result<Id> {
        let (ty, bytes) = self.buffer_element(kind);
        let index = self.buffer_index(ir, offset, bytes, 0)?;
        if let Value::U32(binding) = ir.resolve(binding)? {
            let object = self
                .cbufs
                .get(binding as usize)
                .ok_or_else(|| invariant("Constant buffer index out of range"))?[kind as usize];
            let pointer_type = self.uniform_types[kind as usize];
            if object == Id::NONE || pointer_type == Id::NONE {
                return Err(unsupported("Constant buffer view is not declared"));
            }
            let pointer = self.op(
                Op::AccessChain,
                pointer_type,
                &[object, self.constants.u32_zero, index],
            )?;
            let value = self.op(Op::Load, ty, &[pointer])?;
            if !ir.is_immediate(offset)? && self.profile.has_broken_robust {
                let limit = self.const_u32(0xffff)?;
                let valid = self.op(Op::ULessThanEqual, self.types.bool, &[index, limit])?;
                let zero = self.builder.constant_null(ty)?;
                self.op(Op::Select, ty, &[valid, value, zero])
            } else {
                Ok(value)
            }
        } else {
            let func = match kind {
                K::U8 | K::S8 => self.objects.load_const_func_u8,
                K::U16 | K::S16 => self.objects.load_const_func_u16,
                K::U32 => self.objects.load_const_func_u32,
                K::F32 => self.objects.load_const_func_f32,
                K::U32x2 => self.objects.load_const_func_u32x2,
                K::U32x4 => self.objects.load_const_func_u32x4,
                K::U64 => return Err(unsupported("Indirect U64 constant buffer load")),
            };
            if func == Id::NONE {
                return Err(unsupported(
                    "Indirect constant buffer accessor is not declared",
                ));
            }
            let binding = self.def(ir, binding)?;
            self.op(Op::FunctionCall, ty, &[func, binding, index])
        }
    }
    fn cbuf_component(&mut self, ir: &Ir, vector: Id, offset: Value, extra: u32) -> Result<Id> {
        if let Value::U32(offset) = ir.resolve(offset)? {
            return self.extract(self.types.u32[0], vector, (offset / 4) % 4 + extra);
        }
        let shift = self.buffer_index(ir, offset, 4, 0)?;
        let three = self.const_u32(3)?;
        let mut index = self.op(Op::BitwiseAnd, self.types.u32[0], &[shift, three])?;
        if extra != 0 {
            let extra = self.const_u32(extra)?;
            index = self.op(Op::IAdd, self.types.u32[0], &[index, extra])?;
        }
        self.op(
            Op::VectorExtractDynamic,
            self.types.u32[0],
            &[vector, index],
        )
    }
    pub(super) fn buffer_instruction(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        // (address space: constant/storage/global/local, width, signed, store)
        let (space, width, signed, store) = match opcode {
            O::GetCbufU8 => (0, 8, false, false),
            O::GetCbufS8 => (0, 8, true, false),
            O::GetCbufU16 => (0, 16, false, false),
            O::GetCbufS16 => (0, 16, true, false),
            O::GetCbufU32 => (0, 32, false, false),
            O::GetCbufF32 => (0, 0, false, false),
            O::GetCbufU32x2 => (0, 64, false, false),
            O::LoadStorageU8 => (1, 8, false, false),
            O::LoadStorageS8 => (1, 8, true, false),
            O::LoadStorageU16 => (1, 16, false, false),
            O::LoadStorageS16 => (1, 16, true, false),
            O::LoadStorage32 => (1, 32, false, false),
            O::LoadStorage64 => (1, 64, false, false),
            O::LoadStorage128 => (1, 128, false, false),
            O::WriteStorageU8 => (1, 8, false, true),
            O::WriteStorageS8 => (1, 8, true, true),
            O::WriteStorageU16 => (1, 16, false, true),
            O::WriteStorageS16 => (1, 16, true, true),
            O::WriteStorage32 => (1, 32, false, true),
            O::WriteStorage64 => (1, 64, false, true),
            O::WriteStorage128 => (1, 128, false, true),
            O::LoadGlobal32 => (2, 32, false, false),
            O::WriteGlobal32 => (2, 32, false, true),
            O::LoadGlobal64 => (2, 64, false, false),
            O::WriteGlobal64 => (2, 64, false, true),
            O::LoadGlobal128 => (2, 128, false, false),
            O::WriteGlobal128 => (2, 128, false, true),
            O::LoadLocal => (3, 32, false, false),
            O::WriteLocal => (3, 32, false, true),
            _ => return Ok(None),
        };
        let binding = ir.inst(id)?.arg(0)?;
        let u = self.types.u32[0];
        if space == 2 {
            let n = width / 32;
            let ty = if store {
                self.types.void
            } else {
                self.types.u32[n as usize - 1]
            };
            if !self.profile.support_int64 {
                return Ok(Some(if store {
                    Id::NONE
                } else {
                    self.builder.constant_null(ty)?
                }));
            }
            let func = match (store, width) {
                (false, 32) => self.objects.load_global_func_u32,
                (false, 64) => self.objects.load_global_func_u32x2,
                (false, _) => self.objects.load_global_func_u32x4,
                (true, 32) => self.objects.write_global_func_u32,
                (true, 64) => self.objects.write_global_func_u32x2,
                (true, _) => self.objects.write_global_func_u32x4,
            };
            if func == Id::NONE {
                return Err(unsupported("Global memory accessor is not declared"));
            }
            let address = self.def(ir, binding)?;
            let value = if store {
                self.def(ir, ir.inst(id)?.arg(1)?)?
            } else {
                Id::NONE
            };
            let args = [func, address, value];
            let result = self.op(Op::FunctionCall, ty, &args[..if store { 3 } else { 2 }])?;
            return Ok(Some(if store { Id::NONE } else { result }));
        }
        if space == 3 {
            let offset = self.def(ir, binding)?;
            let pointer = self.op(
                Op::AccessChain,
                self.types.private_u32,
                &[self.objects.local_memory, offset, self.constants.u32_zero],
            )?;
            return Ok(Some(if store {
                let value = self.def(ir, ir.inst(id)?.arg(1)?)?;
                self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
                Id::NONE
            } else {
                self.op(Op::Load, u, &[pointer])?
            }));
        }
        let offset = ir.inst(id)?.arg(1)?;
        let alias = self.profile.support_descriptor_aliasing;
        let narrow_native = alias
            && match width {
                8 => {
                    self.profile.support_int8
                        && if space == 0 {
                            self.profile.support_uniform_and_storage_buffer_8bit
                        } else {
                            self.profile.support_storage_buffer_8bit
                        }
                }
                16 => {
                    self.profile.support_int16
                        && if space == 0 {
                            self.profile.support_uniform_and_storage_buffer_16bit
                        } else {
                            self.profile.support_storage_buffer_16bit
                        }
                }
                _ => false,
            };
        let kind = match (width, signed) {
            (8, false) if narrow_native => K::U8,
            (8, true) if narrow_native => K::S8,
            (16, false) if narrow_native => K::U16,
            (16, true) if narrow_native => K::S16,
            (64, _) if alias => K::U32x2,
            (128, _) if alias => K::U32x4,
            (0, _) if alias => K::F32,
            _ => K::U32,
        };
        let (ty, _) = self.buffer_element(kind);
        let narrow = width == 8 || width == 16;
        if store {
            let value = self.def(ir, ir.inst(id)?.arg(2)?)?;
            if narrow && !narrow_native {
                let pointer = self.storage_pointer(ir, binding, offset, K::U32, 0)?;
                let bit = self.bit_offset(ir, offset, width)?;
                let count = self.const_u32(width)?;
                let func = self.objects.write_storage_cas_loop_func;
                if func == Id::NONE {
                    return Err(unsupported("Narrow storage CAS helper is not declared"));
                }
                self.op(
                    Op::FunctionCall,
                    self.types.void,
                    &[func, pointer, value, bit, count],
                )?;
            } else if width > 32 && !alias {
                for i in 0..width / 32 {
                    let component = self.extract(u, value, i)?;
                    let pointer = self.storage_pointer(ir, binding, offset, K::U32, i)?;
                    self.builder
                        .instruction(Op::Store, &[pointer.0, component.0])?;
                }
            } else {
                let value = if narrow {
                    self.op(Op::SConvert, ty, &[value])?
                } else {
                    value
                };
                let pointer = self.storage_pointer(ir, binding, offset, kind, 0)?;
                self.builder.instruction(Op::Store, &[pointer.0, value.0])?;
            }
            return Ok(Some(Id::NONE));
        }
        let value = if space == 0 && !alias {
            let vector = self.cbuf_load(ir, binding, offset, K::U32x4)?;
            let first = self.cbuf_component(ir, vector, offset, 0)?;
            if width == 64 {
                let second = self.cbuf_component(ir, vector, offset, 1)?;
                self.op(Op::CompositeConstruct, self.types.u32[1], &[first, second])?
            } else if width == 0 {
                self.op(Op::Bitcast, self.types.f32[0], &[first])?
            } else {
                first
            }
        } else if space == 0 {
            self.cbuf_load(ir, binding, offset, kind)?
        } else if width > 32 && !alias {
            let n = (width / 32) as usize;
            let mut values = [Id::NONE; 4];
            for (i, value) in values.iter_mut().enumerate().take(n) {
                let pointer = self.storage_pointer(ir, binding, offset, K::U32, i as u32)?;
                *value = self.op(Op::Load, u, &[pointer])?;
            }
            self.op(Op::CompositeConstruct, self.types.u32[n - 1], &values[..n])?
        } else {
            let pointer = self.storage_pointer(ir, binding, offset, kind, 0)?;
            self.op(Op::Load, ty, &[pointer])?
        };
        let result = if narrow_native {
            self.op(
                if signed { Op::SConvert } else { Op::UConvert },
                u,
                &[value],
            )?
        } else if narrow {
            let bit = self.bit_offset(ir, offset, width)?;
            let count = self.const_u32(width)?;
            self.op(
                if signed {
                    Op::BitFieldSExtract
                } else {
                    Op::BitFieldUExtract
                },
                u,
                &[value, bit, count],
            )?
        } else {
            value
        };
        Ok(Some(result))
    }
}
