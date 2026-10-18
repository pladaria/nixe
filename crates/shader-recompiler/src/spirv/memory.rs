use super::{EmitContext, Id};
use crate::{
    Result,
    artifact::{HostUniformLayout, ShaderInfo},
    ir::Program,
    types::Stage,
};
use spirv::{Capability as C, Decoration as D, Op, StorageClass as S};
impl EmitContext<'_> {
    pub(super) fn global_interface(&mut self, id: Id) {
        if self.profile.supported_spirv >= 0x00010400 {
            self.interfaces.push(id);
        }
    }
    pub(super) fn define_local_memory(&mut self, p: &Program) -> Result<()> {
        if p.execution.local_memory_size == 0 {
            return Ok(());
        }
        let count = self.const_u32(p.execution.local_memory_size.div_ceil(4))?;
        let element = self.builder.type_struct(&[self.types.u32[0]])?;
        let array = self.builder.type_array(element, count)?;
        let pointer = self.builder.type_pointer(S::Private, array)?;
        let id = self.builder.variable(pointer, S::Private, None)?;
        self.objects.local_memory = id;
        self.global_interface(id);
        Ok(())
    }
    fn shared_alias(&mut self, element: Id, stride: u32, bytes: u32) -> Result<(Id, Id, Id)> {
        let count = self.const_u32(bytes.div_ceil(stride))?;
        let array = self.builder.type_array(element, count)?;
        self.builder.decorate(array, D::ArrayStride, &[stride])?;
        let structure = self.builder.type_struct(&[array])?;
        self.builder
            .member_decorate(structure, 0, D::Offset, &[0])?;
        self.builder.decorate(structure, D::Block, &[])?;
        let pointer = self.builder.type_pointer(S::Workgroup, structure)?;
        let element_pointer = self.builder.type_pointer(S::Workgroup, element)?;
        let id = self.builder.variable(pointer, S::Workgroup, None)?;
        self.builder.decorate(id, D::Aliased, &[])?;
        self.interfaces.push(id);
        Ok((id, element_pointer, pointer))
    }
    pub(super) fn define_shared_memory(&mut self, p: &Program) -> Result<()> {
        self.uses_explicit_workgroup_layout = self.profile.support_explicit_workgroup_layout
            && (!p.info.uses_int8 || self.profile.support_workgroup_layout_8bit_access)
            && (!p.info.uses_int16 || self.profile.support_workgroup_layout_16bit_access);
        let bytes = p.execution.shared_memory_size;
        if bytes == 0 {
            return Ok(());
        }
        if self.uses_explicit_workgroup_layout {
            self.builder
                .extension("SPV_KHR_workgroup_memory_explicit_layout")?;
            self.builder
                .capability(C::WorkgroupMemoryExplicitLayoutKHR)?;
            if p.info.uses_int8 && self.profile.support_int8 {
                self.builder
                    .capability(C::WorkgroupMemoryExplicitLayout8BitAccessKHR)?;
                let (id, ptr, _) = self.shared_alias(self.types.u8, 1, bytes)?;
                self.objects.shared_memory_u8 = id;
                self.objects.shared_u8 = ptr;
            }
            if p.info.uses_int16 && self.profile.support_int16 {
                self.builder
                    .capability(C::WorkgroupMemoryExplicitLayout16BitAccessKHR)?;
                let (id, ptr, _) = self.shared_alias(self.types.u16, 2, bytes)?;
                self.objects.shared_memory_u16 = id;
                self.objects.shared_u16 = ptr;
            }
            if p.info.uses_int64 && self.profile.support_int64 {
                let (id, ptr, _) = self.shared_alias(self.types.u64, 8, bytes)?;
                self.objects.shared_memory_u64 = id;
                self.objects.shared_u64 = ptr;
            }
            let (id, ptr, array) = self.shared_alias(self.types.u32[0], 4, bytes)?;
            self.objects.shared_memory_u32 = id;
            self.objects.shared_u32 = ptr;
            self.objects.shared_memory_u32_type = array;
            let (id, ptr, _) = self.shared_alias(self.types.u32[1], 8, bytes)?;
            self.objects.shared_memory_u32x2 = id;
            self.objects.shared_u32x2 = ptr;
            let (id, ptr, _) = self.shared_alias(self.types.u32[3], 16, bytes)?;
            self.objects.shared_memory_u32x4 = id;
            self.objects.shared_u32x4 = ptr;
        } else {
            let count = self.const_u32(bytes.div_ceil(4))?;
            let array = self.builder.type_array(self.types.u32[0], count)?;
            let pointer = self.builder.type_pointer(S::Workgroup, array)?;
            self.objects.shared_memory_u32_type = pointer;
            self.objects.shared_u32 = self.builder.type_pointer(S::Workgroup, self.types.u32[0])?;
            self.objects.shared_memory_u32 = self.builder.variable(pointer, S::Workgroup, None)?;
            self.interfaces.push(self.objects.shared_memory_u32);
            if p.info.uses_int8 {
                self.objects.shared_store_u8_func = self.shared_narrow_store(24, 8)?;
            }
            if p.info.uses_int16 {
                self.objects.shared_store_u16_func = self.shared_narrow_store(16, 16)?;
            }
        }
        Ok(())
    }
    fn shared_narrow_store(&mut self, mask: u32, width: u32) -> Result<Id> {
        let u = self.types.u32[0];
        let z = self.constants.u32_zero;
        let two = self.const_u32(2)?;
        let three = self.const_u32(3)?;
        let one = self.const_u32(1)?;
        let mask = self.const_u32(mask)?;
        let width = self.const_u32(width)?;
        let ty = self.builder.type_function(self.types.void, &[u, u])?;
        let header = self.builder.reserve_id()?;
        let next = self.builder.reserve_id()?;
        let merge = self.builder.reserve_id()?;
        let function = self.builder.begin_function(self.types.void, ty)?;
        let offset = self.builder.parameter(u)?;
        let value = self.builder.parameter(u)?;
        self.builder.new_label()?;
        self.builder.instruction(Op::Branch, &[header.0])?;
        self.builder.label(header)?;
        let word = self
            .builder
            .result(Op::ShiftRightArithmetic, u, &[offset.0, two.0])?;
        let shift = self
            .builder
            .result(Op::ShiftLeftLogical, u, &[offset.0, three.0])?;
        let bit = self.builder.result(Op::BitwiseAnd, u, &[shift.0, mask.0])?;
        self.builder
            .instruction(Op::LoopMerge, &[merge.0, next.0, 0])?;
        self.builder.instruction(Op::Branch, &[next.0])?;
        self.builder.label(next)?;
        let pointer = self.builder.result(
            Op::AccessChain,
            self.objects.shared_u32,
            &[self.objects.shared_memory_u32.0, word.0],
        )?;
        let old = self.builder.result(Op::Load, u, &[pointer.0])?;
        let new = self
            .builder
            .result(Op::BitFieldInsert, u, &[old.0, value.0, bit.0, width.0])?;
        let returned = self.builder.result(
            Op::AtomicCompareExchange,
            u,
            &[pointer.0, one.0, z.0, z.0, new.0, old.0],
        )?;
        let success = self
            .builder
            .result(Op::IEqual, self.types.bool, &[returned.0, old.0])?;
        self.builder
            .instruction(Op::BranchConditional, &[success.0, merge.0, header.0])?;
        self.builder.label(merge)?;
        self.builder.instruction(Op::Return, &[])?;
        self.builder.end_function()?;
        Ok(function)
    }
    pub(super) fn define_host_uniforms(&mut self, info: &ShaderInfo) -> Result<()> {
        if info.uses_rescaling_uniform {
            if self.profile.unified_descriptor_binding {
                let four = self.const_u32(4)?;
                let two = self.const_u32(2)?;
                let textures = self.builder.type_array(self.types.u32[0], four)?;
                let images = self.builder.type_array(self.types.u32[0], two)?;
                self.builder.decorate(textures, D::ArrayStride, &[4])?;
                self.builder.decorate(images, D::ArrayStride, &[4])?;
                self.objects.rescaling_textures_type = textures;
                self.objects.rescaling_images_type = images;
                let mut members = vec![textures, images];
                if self.stage != Stage::Compute {
                    members.push(self.types.f32[0]);
                }
                let structure = self.builder.type_struct(&members)?;
                self.builder.decorate(structure, D::Block, &[])?;
                for (i, offset) in [0, 16, 24].into_iter().take(members.len()).enumerate() {
                    self.builder
                        .member_decorate(structure, i as u32, D::Offset, &[offset])?;
                }
                let pointer = self.builder.type_pointer(S::PushConstant, structure)?;
                let id = self.builder.variable(pointer, S::PushConstant, None)?;
                self.objects.rescaling_push_constants = id;
                self.global_interface(id);
                self.host_uniforms
                    .push(HostUniformLayout::RescalingPushConstants {
                        texture_mask_offset: 0,
                        texture_mask_words: 4,
                        image_mask_offset: 16,
                        image_mask_words: 2,
                        down_factor_offset: (self.stage != Stage::Compute).then_some(24),
                    });
            } else {
                let pointer = self
                    .builder
                    .type_pointer(S::UniformConstant, self.types.f32[3])?;
                let id = self.builder.variable(pointer, S::UniformConstant, None)?;
                self.builder.decorate(id, D::Location, &[0])?;
                self.objects.rescaling_uniform_constant = id;
                self.global_interface(id);
                self.host_uniforms
                    .push(HostUniformLayout::RescalingUniformConstant { location: 0 });
            }
        }
        if info.uses_render_area && self.profile.unified_descriptor_binding {
            let structure = self.builder.type_struct(&[self.types.f32[3]])?;
            self.builder.decorate(structure, D::Block, &[])?;
            self.builder
                .member_decorate(structure, 0, D::Offset, &[0])?;
            let pointer = self.builder.type_pointer(S::PushConstant, structure)?;
            let id = self.builder.variable(pointer, S::PushConstant, None)?;
            self.objects.render_area_push_constant = id;
            self.global_interface(id);
            self.host_uniforms
                .push(HostUniformLayout::RenderAreaPushConstants {
                    offset: 0,
                    size: 16,
                });
        }
        Ok(())
    }
}
