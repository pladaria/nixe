use super::*;
use crate::spirv::BufferKind as K;
use spirv::{GlslStd450Op as G, Scope};
impl EmitContext<'_> {
    pub(super) fn atomic_instruction(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        let (shared, width, atomic) = match opcode {
            O::SharedAtomicIAdd32 => (true, 32, Op::AtomicIAdd),
            O::SharedAtomicSMin32 => (true, 32, Op::AtomicSMin),
            O::SharedAtomicUMin32 => (true, 32, Op::AtomicUMin),
            O::SharedAtomicSMax32 => (true, 32, Op::AtomicSMax),
            O::SharedAtomicUMax32 => (true, 32, Op::AtomicUMax),
            O::SharedAtomicAnd32 => (true, 32, Op::AtomicAnd),
            O::SharedAtomicOr32 => (true, 32, Op::AtomicOr),
            O::SharedAtomicXor32 => (true, 32, Op::AtomicXor),
            O::SharedAtomicExchange32 => (true, 32, Op::AtomicExchange),
            O::SharedAtomicExchange64 => (true, 64, Op::AtomicExchange),
            O::SharedAtomicExchange32x2 => (true, 2, Op::AtomicExchange),
            O::StorageAtomicIAdd32 => (false, 32, Op::AtomicIAdd),
            O::StorageAtomicSMin32 => (false, 32, Op::AtomicSMin),
            O::StorageAtomicUMin32 => (false, 32, Op::AtomicUMin),
            O::StorageAtomicSMax32 => (false, 32, Op::AtomicSMax),
            O::StorageAtomicUMax32 => (false, 32, Op::AtomicUMax),
            O::StorageAtomicAnd32 => (false, 32, Op::AtomicAnd),
            O::StorageAtomicOr32 => (false, 32, Op::AtomicOr),
            O::StorageAtomicXor32 => (false, 32, Op::AtomicXor),
            O::StorageAtomicExchange32 => (false, 32, Op::AtomicExchange),
            O::StorageAtomicIAdd64 => (false, 64, Op::AtomicIAdd),
            O::StorageAtomicSMin64 => (false, 64, Op::AtomicSMin),
            O::StorageAtomicUMin64 => (false, 64, Op::AtomicUMin),
            O::StorageAtomicSMax64 => (false, 64, Op::AtomicSMax),
            O::StorageAtomicUMax64 => (false, 64, Op::AtomicUMax),
            O::StorageAtomicAnd64 => (false, 64, Op::AtomicAnd),
            O::StorageAtomicOr64 => (false, 64, Op::AtomicOr),
            O::StorageAtomicXor64 => (false, 64, Op::AtomicXor),
            O::StorageAtomicExchange64 => (false, 64, Op::AtomicExchange),
            O::StorageAtomicIAdd32x2 => (false, 2, Op::AtomicIAdd),
            O::StorageAtomicSMin32x2 => (false, 2, Op::AtomicSMin),
            O::StorageAtomicUMin32x2 => (false, 2, Op::AtomicUMin),
            O::StorageAtomicSMax32x2 => (false, 2, Op::AtomicSMax),
            O::StorageAtomicUMax32x2 => (false, 2, Op::AtomicUMax),
            O::StorageAtomicAnd32x2 => (false, 2, Op::AtomicAnd),
            O::StorageAtomicOr32x2 => (false, 2, Op::AtomicOr),
            O::StorageAtomicXor32x2 => (false, 2, Op::AtomicXor),
            O::StorageAtomicExchange32x2 => (false, 2, Op::AtomicExchange),
            _ => return self.atomic_helper(ir, id),
        };
        let u = self.types.u32[0];
        let ty = match width {
            32 => u,
            64 => self.types.u64,
            _ => self.types.u32[1],
        };
        let value = self.def(ir, ir.inst(id)?.arg(if shared { 1 } else { 2 })?)?;
        let offset_value = ir.inst(id)?.arg(if shared { 0 } else { 1 })?;
        let binding = if shared {
            Value::None
        } else {
            ir.inst(id)?.arg(0)?
        };
        let native = width == 32
            || (width == 64
                && if shared {
                    self.profile.support_shared_int64_atomics && self.uses_explicit_workgroup_layout
                } else {
                    self.profile.support_int64_atomics
                });
        if !shared
            && width != 32
            && atomic != Op::AtomicExchange
            && !self.profile.support_descriptor_aliasing
        {
            return Ok(Some(self.builder.constant_null(ty)?));
        }
        if native {
            let pointer = if shared {
                let offset = self.def(ir, offset_value)?;
                if width == 32 {
                    self.shared_pointer(offset, 32, 0)?
                } else {
                    let shift = self.const_u32(3)?;
                    let index = self.op(Op::ShiftRightArithmetic, u, &[offset, shift])?;
                    self.op(
                        Op::AccessChain,
                        self.objects.shared_u64,
                        &[
                            self.objects.shared_memory_u64,
                            self.constants.u32_zero,
                            index,
                        ],
                    )?
                }
            } else {
                self.storage_pointer(
                    ir,
                    binding,
                    offset_value,
                    if width == 32 { K::U32 } else { K::U64 },
                    0,
                )?
            };
            let scope = self.const_u32(Scope::Device as u32)?;
            return Ok(Some(self.op(
                atomic,
                ty,
                &[pointer, scope, self.constants.u32_zero, value],
            )?));
        }
        if shared {
            let offset = self.def(ir, offset_value)?;
            let p0 = self.shared_pointer(offset, 32, 0)?;
            let p1 = self.shared_pointer(offset, 32, 1)?;
            let a = self.op(Op::Load, u, &[p0])?;
            let b = self.op(Op::Load, u, &[p1])?;
            let vector = self.op(Op::Bitcast, self.types.u32[1], &[value])?;
            for (i, pointer) in [p0, p1].into_iter().enumerate() {
                let component = self.extract(u, vector, i as u32)?;
                self.builder
                    .instruction(Op::Store, &[pointer.0, component.0])?;
            }
            let old = self.op(Op::CompositeConstruct, self.types.u32[1], &[a, b])?;
            return Ok(Some(if width == 64 {
                self.op(Op::Bitcast, ty, &[old])?
            } else {
                old
            }));
        }
        let pointer = self.storage_pointer(ir, binding, offset_value, K::U32x2, 0)?;
        let loaded = self.op(Op::Load, self.types.u32[1], &[pointer])?;
        let original = if width == 64 {
            self.op(Op::Bitcast, ty, &[loaded])?
        } else {
            loaded
        };
        let mut result = match atomic {
            Op::AtomicIAdd => self.op(Op::IAdd, ty, &[value, original])?,
            Op::AtomicAnd => self.op(Op::BitwiseAnd, ty, &[value, original])?,
            Op::AtomicOr => self.op(Op::BitwiseOr, ty, &[value, original])?,
            Op::AtomicXor => self.op(Op::BitwiseXor, ty, &[value, original])?,
            Op::AtomicSMin => self.ext(G::SMin, ty, &[value, original])?,
            Op::AtomicUMin => self.ext(G::UMin, ty, &[value, original])?,
            Op::AtomicSMax => self.ext(G::SMax, ty, &[value, original])?,
            Op::AtomicUMax => self.ext(G::UMax, ty, &[value, original])?,
            Op::AtomicExchange => value,
            _ => unreachable!(),
        };
        if width == 64 && atomic != Op::AtomicExchange {
            result = self.op(Op::Bitcast, self.types.u32[1], &[result])?;
        }
        self.builder
            .instruction(Op::Store, &[pointer.0, result.0])?;
        Ok(Some(original))
    }
    fn atomic_helper(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let opcode = ir.inst(id)?.opcode();
        let (func, ty, shared, pack) = match opcode {
            O::SharedAtomicInc32 => (
                self.objects.increment_cas_shared,
                self.types.u32[0],
                true,
                0,
            ),
            O::SharedAtomicDec32 => (
                self.objects.decrement_cas_shared,
                self.types.u32[0],
                true,
                0,
            ),
            O::StorageAtomicInc32 => (self.objects.increment_cas_ssbo, self.types.u32[0], false, 0),
            O::StorageAtomicDec32 => (self.objects.decrement_cas_ssbo, self.types.u32[0], false, 0),
            O::StorageAtomicAddF32 => (self.objects.f32_add_cas, self.types.f32[0], false, 0),
            O::StorageAtomicAddF16x2 => (self.objects.f16x2_add_cas, self.types.f16[1], false, 1),
            O::StorageAtomicMinF16x2 => (self.objects.f16x2_min_cas, self.types.f16[1], false, 1),
            O::StorageAtomicMaxF16x2 => (self.objects.f16x2_max_cas, self.types.f16[1], false, 1),
            O::StorageAtomicAddF32x2 => (self.objects.f32x2_add_cas, self.types.f32[1], false, 2),
            O::StorageAtomicMinF32x2 => (self.objects.f32x2_min_cas, self.types.f32[1], false, 2),
            O::StorageAtomicMaxF32x2 => (self.objects.f32x2_max_cas, self.types.f32[1], false, 2),
            _ => return Ok(None),
        };
        if func == Id::NONE {
            return Err(unsupported("Atomic CAS helper is not declared"));
        }
        let value = self.def(ir, ir.inst(id)?.arg(if shared { 1 } else { 2 })?)?;
        let offset = ir.inst(id)?.arg(if shared { 0 } else { 1 })?;
        let result = if shared {
            let offset = self.def(ir, offset)?;
            let shift = self.const_u32(2)?;
            let index = self.op(
                Op::ShiftRightArithmetic,
                self.types.u32[0],
                &[offset, shift],
            )?;
            self.op(Op::FunctionCall, ty, &[func, index, value])?
        } else {
            let binding = literal(ir, ir.inst(id)?.arg(0)?)? as usize;
            let object = self
                .ssbos
                .get(binding)
                .ok_or_else(|| invariant("Atomic storage buffer index out of range"))?
                [K::U32 as usize];
            let index = self.buffer_index(ir, offset, 4, 0)?;
            self.op(Op::FunctionCall, ty, &[func, index, value, object])?
        };
        Ok(Some(match pack {
            0 => result,
            1 => self.op(Op::Bitcast, self.types.u32[0], &[result])?,
            _ => self.ext(G::PackHalf2x16, self.types.u32[0], &[result])?,
        }))
    }
}
