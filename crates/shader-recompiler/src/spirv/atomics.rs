use super::{BufferKind as K, EmitContext, Id};
use crate::{Result, artifact::ShaderInfo, ir::Program};
use spirv::{Capability, GlslStd450Op as G, Op, Scope, StorageClass};
#[derive(Clone, Copy)]
enum Operation {
    Increment,
    Decrement,
    Add,
    Min,
    Max,
}
impl EmitContext<'_> {
    fn atomic_calculation(&mut self, operation: Operation, ty: Id) -> Result<Id> {
        let ft = self.builder.type_function(ty, &[ty, ty])?;
        let function = self.builder.begin_function(ty, ft)?;
        let a = self.builder.parameter(ty)?;
        let b = self.builder.parameter(ty)?;
        self.builder.new_label()?;
        let result = match operation {
            Operation::Increment => {
                let predicate =
                    self.builder
                        .result(Op::UGreaterThanEqual, self.types.bool, &[a.0, b.0])?;
                let one = self.builder.constant(ty, &[1])?;
                let increment = self.builder.result(Op::IAdd, ty, &[a.0, one.0])?;
                self.builder.result(
                    Op::Select,
                    ty,
                    &[predicate.0, self.constants.u32_zero.0, increment.0],
                )?
            }
            Operation::Decrement => {
                let zero = self.builder.constant(ty, &[0])?;
                let one = self.builder.constant(ty, &[1])?;
                let left = self
                    .builder
                    .result(Op::IEqual, self.types.bool, &[a.0, zero.0])?;
                let right = self
                    .builder
                    .result(Op::UGreaterThan, self.types.bool, &[a.0, b.0])?;
                let predicate =
                    self.builder
                        .result(Op::LogicalOr, self.types.bool, &[left.0, right.0])?;
                let decrement = self.builder.result(Op::ISub, ty, &[a.0, one.0])?;
                self.builder
                    .result(Op::Select, ty, &[predicate.0, b.0, decrement.0])?
            }
            Operation::Add => self.builder.result(Op::FAdd, ty, &[a.0, b.0])?,
            Operation::Min => self.builder.glsl(ty, G::FMin, &[a, b])?,
            Operation::Max => self.builder.glsl(ty, G::FMax, &[a, b])?,
        };
        self.builder.instruction(Op::ReturnValue, &[result.0])?;
        self.builder.end_function()?;
        Ok(function)
    }
    fn atomic_loop(
        &mut self,
        operation: Operation,
        array: Id,
        element: Id,
        value_type: Id,
        memory_type: Id,
        scope: Scope,
    ) -> Result<Id> {
        let shared = scope == Scope::Workgroup;
        let structure = !shared || self.uses_explicit_workgroup_layout;
        let calculate = self.atomic_calculation(operation, value_type)?;
        let z = self.constants.u32_zero;
        let u = self.types.u32[0];
        let scope = self.const_u32(scope as u32)?;
        let header = self.builder.reserve_id()?;
        let next = self.builder.reserve_id()?;
        let merge = self.builder.reserve_id()?;
        let params = if shared {
            vec![u, value_type]
        } else {
            vec![u, value_type, array]
        };
        let ft = self.builder.type_function(value_type, &params)?;
        let function = self.builder.begin_function(value_type, ft)?;
        let index = self.builder.parameter(u)?;
        let b = self.builder.parameter(value_type)?;
        let base = if shared {
            self.objects.shared_memory_u32
        } else {
            self.builder.parameter(array)?
        };
        self.builder.new_label()?;
        self.builder.instruction(Op::Branch, &[header.0])?;
        self.builder.label(header)?;
        self.builder
            .instruction(Op::LoopMerge, &[merge.0, next.0, 0])?;
        self.builder.instruction(Op::Branch, &[next.0])?;
        self.builder.label(next)?;
        let indices = if structure {
            vec![base.0, z.0, index.0]
        } else {
            vec![base.0, index.0]
        };
        let pointer = self.builder.result(Op::AccessChain, element, &indices)?;
        let (returned, success) = if value_type == self.types.f32[1] {
            let old = self.builder.result(Op::Load, u, &[pointer.0])?;
            let unpacked = self.builder.glsl(value_type, G::UnpackHalf2x16, &[old])?;
            let calculated = self.builder.result(
                Op::FunctionCall,
                value_type,
                &[calculate.0, unpacked.0, b.0],
            )?;
            let packed = self.builder.glsl(u, G::PackHalf2x16, &[calculated])?;
            let returned = self.builder.result(
                Op::AtomicCompareExchange,
                u,
                &[pointer.0, scope.0, z.0, z.0, packed.0, old.0],
            )?;
            let success = self
                .builder
                .result(Op::IEqual, self.types.bool, &[returned.0, old.0])?;
            (returned, success)
        } else {
            let old = self.builder.result(Op::Load, memory_type, &[pointer.0])?;
            let value = if value_type == memory_type {
                old
            } else {
                self.builder.result(Op::Bitcast, value_type, &[old.0])?
            };
            let calculated =
                self.builder
                    .result(Op::FunctionCall, value_type, &[calculate.0, value.0, b.0])?;
            let new = if value_type == memory_type {
                calculated
            } else {
                self.builder
                    .result(Op::Bitcast, memory_type, &[calculated.0])?
            };
            let returned = self.builder.result(
                Op::AtomicCompareExchange,
                u,
                &[pointer.0, scope.0, z.0, z.0, new.0, old.0],
            )?;
            let success = self
                .builder
                .result(Op::IEqual, self.types.bool, &[returned.0, old.0])?;
            (returned, success)
        };
        self.builder
            .instruction(Op::BranchConditional, &[success.0, merge.0, header.0])?;
        self.builder.label(merge)?;
        let result = if value_type == self.types.f32[1] {
            self.builder
                .glsl(value_type, G::UnpackHalf2x16, &[returned])?
        } else {
            self.builder
                .result(Op::Bitcast, value_type, &[returned.0])?
        };
        self.builder.instruction(Op::ReturnValue, &[result.0])?;
        self.builder.end_function()?;
        Ok(function)
    }
    pub(super) fn define_shared_atomics(&mut self, p: &Program) -> Result<()> {
        if p.info.uses_shared_increment {
            self.objects.increment_cas_shared = self.atomic_loop(
                Operation::Increment,
                self.objects.shared_memory_u32_type,
                self.objects.shared_u32,
                self.types.u32[0],
                self.types.u32[0],
                Scope::Workgroup,
            )?;
        }
        if p.info.uses_shared_decrement {
            self.objects.decrement_cas_shared = self.atomic_loop(
                Operation::Decrement,
                self.objects.shared_memory_u32_type,
                self.objects.shared_u32,
                self.types.u32[0],
                self.types.u32[0],
                Scope::Workgroup,
            )?;
        }
        Ok(())
    }
    pub(super) fn define_storage_atomics(&mut self, info: &ShaderInfo) -> Result<()> {
        if info.storage_buffers_descriptors.is_empty() {
            return Ok(());
        }
        let ptr = self.storage_types[K::U32 as usize];
        let u = self.types.u32[0];
        macro_rules! cas {
            ($enabled:expr,$field:ident,$operation:ident,$value:expr,$memory:expr) => {
                if $enabled {
                    self.builder
                        .capability(Capability::VariablePointersStorageBuffer)?;
                    self.objects.$field = self.atomic_loop(
                        Operation::$operation,
                        ptr.array,
                        ptr.element,
                        $value,
                        $memory,
                        Scope::Device,
                    )?;
                }
            };
        }
        cas!(
            info.uses_global_increment,
            increment_cas_ssbo,
            Increment,
            u,
            u
        );
        cas!(
            info.uses_global_decrement,
            decrement_cas_ssbo,
            Decrement,
            u,
            u
        );
        cas!(
            info.uses_atomic_f32_add,
            f32_add_cas,
            Add,
            self.types.f32[0],
            u
        );
        cas!(
            info.uses_atomic_f16x2_add,
            f16x2_add_cas,
            Add,
            self.types.f16[1],
            self.types.f16[1]
        );
        cas!(
            info.uses_atomic_f16x2_min,
            f16x2_min_cas,
            Min,
            self.types.f16[1],
            self.types.f16[1]
        );
        cas!(
            info.uses_atomic_f16x2_max,
            f16x2_max_cas,
            Max,
            self.types.f16[1],
            self.types.f16[1]
        );
        cas!(
            info.uses_atomic_f32x2_add,
            f32x2_add_cas,
            Add,
            self.types.f32[1],
            self.types.f32[1]
        );
        cas!(
            info.uses_atomic_f32x2_min,
            f32x2_min_cas,
            Min,
            self.types.f32[1],
            self.types.f32[1]
        );
        cas!(
            info.uses_atomic_f32x2_max,
            f32x2_max_cas,
            Max,
            self.types.f32[1],
            self.types.f32[1]
        );
        Ok(())
    }
    pub(super) fn define_narrow_storage_write(&mut self, info: &ShaderInfo) -> Result<()> {
        if self.profile.support_int8 && self.profile.support_int16
            || !info.uses_int8 && !info.uses_int16
        {
            return Ok(());
        }
        self.builder
            .capability(Capability::VariablePointersStorageBuffer)?;
        let u = self.types.u32[0];
        let z = self.constants.u32_zero;
        let ptr = self.builder.type_pointer(StorageClass::StorageBuffer, u)?;
        let ft = self
            .builder
            .type_function(self.types.void, &[ptr, u, u, u])?;
        let function = self.builder.begin_function(self.types.void, ft)?;
        let pointer = self.builder.parameter(ptr)?;
        let value = self.builder.parameter(u)?;
        let offset = self.builder.parameter(u)?;
        let count = self.builder.parameter(u)?;
        self.builder.new_label()?;
        let scope = self.const_u32(1)?;
        let body = self.builder.reserve_id()?;
        let next = self.builder.reserve_id()?;
        let end = self.builder.reserve_id()?;
        let header = self.builder.reserve_id()?;
        self.builder.instruction(Op::Branch, &[header.0])?;
        self.builder.label(header)?;
        self.builder
            .instruction(Op::LoopMerge, &[end.0, next.0, 0])?;
        self.builder.instruction(Op::Branch, &[body.0])?;
        self.builder.label(body)?;
        let expected = self.builder.result(Op::Load, u, &[pointer.0])?;
        let desired = self.builder.result(
            Op::BitFieldInsert,
            u,
            &[expected.0, value.0, offset.0, count.0],
        )?;
        let actual = self.builder.result(
            Op::AtomicCompareExchange,
            u,
            &[pointer.0, scope.0, z.0, z.0, desired.0, expected.0],
        )?;
        let success = self
            .builder
            .result(Op::IEqual, self.types.bool, &[expected.0, actual.0])?;
        self.builder
            .instruction(Op::BranchConditional, &[success.0, end.0, next.0])?;
        self.builder.label(end)?;
        self.builder.instruction(Op::Return, &[])?;
        self.builder.label(next)?;
        self.builder.instruction(Op::Branch, &[header.0])?;
        self.builder.end_function()?;
        self.objects.write_storage_cas_loop_func = function;
        Ok(())
    }
}
