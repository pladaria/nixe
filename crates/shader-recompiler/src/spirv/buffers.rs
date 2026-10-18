use super::{BufferKind as K, EmitContext, Id, StoragePointers};
use crate::{
    CompileError, Result,
    artifact::{ResourceAccess, ResourceBinding, ResourceSource, ShaderInfo},
    diagnostics::FailureCategory,
    ir::Type,
    types::{DescriptorBinding, DescriptorSet, ResourceIndex},
};
use spirv::{Decoration as D, StorageClass as S};
impl EmitContext<'_> {
    pub(super) fn buffer_element(&self, kind: K) -> (Id, u32) {
        match kind {
            K::U8 => (self.types.u8, 1),
            K::S8 => (self.types.s8, 1),
            K::U16 => (self.types.u16, 2),
            K::S16 => (self.types.s16, 2),
            K::U32 => (self.types.u32[0], 4),
            K::F32 => (self.types.f32[0], 4),
            K::U64 => (self.types.u64, 8),
            K::U32x2 => (self.types.u32[1], 8),
            K::U32x4 => (self.types.u32[3], 16),
        }
    }
    pub(super) fn bind(&mut self, id: Id, binding: u32) -> Result<()> {
        self.builder.decorate(id, D::Binding, &[binding])?;
        self.builder.decorate(id, D::DescriptorSet, &[0])?;
        self.global_interface(id);
        Ok(())
    }
    fn uniform_view(&mut self, info: &ShaderInfo, kind: K, mut binding: u32) -> Result<()> {
        let (element, stride) = self.buffer_element(kind);
        let count = self.const_u32(65536 / stride)?;
        let array = self.builder.type_array(element, count)?;
        self.builder.decorate(array, D::ArrayStride, &[stride])?;
        let structure = self.builder.type_struct(&[array])?;
        self.builder.decorate(structure, D::Block, &[])?;
        self.builder
            .member_decorate(structure, 0, D::Offset, &[0])?;
        let pointer = self.builder.type_pointer(S::Uniform, structure)?;
        self.uniform_types[kind as usize] = self.builder.type_pointer(S::Uniform, element)?;
        for d in info.constant_buffer_descriptors.iter() {
            let id = self.builder.variable(pointer, S::Uniform, None)?;
            self.bind(id, binding)?;
            let end = d
                .index
                .0
                .checked_add(d.count)
                .filter(|&end| end <= 18)
                .ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::ResourceLimit,
                        "Constant buffer descriptor range exceeds 18 banks",
                    )
                })?;
            for bank in d.index.0..end {
                self.cbufs[bank as usize][kind as usize] = id;
            }
            binding = binding.wrapping_add(d.count);
        }
        Ok(())
    }
    pub(super) fn define_constant_buffers(
        &mut self,
        info: &ShaderInfo,
        binding: &mut u32,
    ) -> Result<()> {
        if info.constant_buffer_descriptors.is_empty() {
            return Ok(());
        }
        let mut emitted_binding = *binding;
        for (index, d) in info.constant_buffer_descriptors.iter().enumerate() {
            self.resource_bindings.push(ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(emitted_binding),
                count: 1,
                resource_index: ResourceIndex(index as u32),
                source: ResourceSource::ConstantBuffer(d.clone()),
                access: ResourceAccess::Read,
            });
            emitted_binding = emitted_binding.wrapping_add(d.count);
        }
        if !self.profile.support_descriptor_aliasing {
            self.uniform_view(info, K::U32x4, *binding)?;
            *binding = emitted_binding;
            return Ok(());
        }
        let mut types = info.used_constant_buffer_types | info.used_indirect_cbuf_types;
        if (types & Type::U8).0 != 0 {
            if self.profile.support_int8 && self.profile.support_uniform_and_storage_buffer_8bit {
                self.uniform_view(info, K::U8, *binding)?;
                self.uniform_view(info, K::S8, *binding)?;
            } else {
                types |= Type::U32;
            }
        }
        if (types & Type::U16).0 != 0 {
            if self.profile.support_int16 && self.profile.support_uniform_and_storage_buffer_16bit {
                self.uniform_view(info, K::U16, *binding)?;
                self.uniform_view(info, K::S16, *binding)?;
            } else {
                types |= Type::U32;
            }
        }
        for (ty, kind) in [
            (Type::U32, K::U32),
            (Type::F32, K::F32),
            (Type::U32x2, K::U32x2),
        ] {
            if (types & ty).0 != 0 {
                self.uniform_view(info, kind, *binding)?;
            }
        }
        *binding = binding.wrapping_add(info.constant_buffer_descriptors.len() as u32);
        Ok(())
    }
    fn storage_view(&mut self, info: &ShaderInfo, kind: K, mut binding: u32) -> Result<()> {
        let (element, stride) = self.buffer_element(kind);
        let array = self.builder.type_runtime_array(element)?;
        self.builder.decorate(array, D::ArrayStride, &[stride])?;
        let structure = self.builder.type_struct(&[array])?;
        self.builder.decorate(structure, D::Block, &[])?;
        self.builder
            .member_decorate(structure, 0, D::Offset, &[0])?;
        let pointer = self.builder.type_pointer(S::StorageBuffer, structure)?;
        let element = self.builder.type_pointer(S::StorageBuffer, element)?;
        self.storage_types[kind as usize] = StoragePointers {
            array: pointer,
            element,
        };
        let mut index = 0u32;
        for d in info.storage_buffers_descriptors.iter() {
            let id = self.builder.variable(pointer, S::StorageBuffer, None)?;
            self.bind(id, binding)?;
            let end = index
                .checked_add(d.count)
                .filter(|&end| end <= 32)
                .ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::ResourceLimit,
                        "Storage descriptor range exceeds 32 slots",
                    )
                })?;
            for slot in index..end {
                self.ssbos[slot as usize][kind as usize] = id;
            }
            index = end;
            binding = binding.wrapping_add(d.count);
        }
        Ok(())
    }
    pub(super) fn define_storage_buffers(
        &mut self,
        info: &ShaderInfo,
        binding: &mut u32,
    ) -> Result<()> {
        if info.storage_buffers_descriptors.is_empty() {
            return Ok(());
        }
        self.builder
            .extension("SPV_KHR_storage_buffer_storage_class")?;
        let types = if self.profile.support_descriptor_aliasing {
            info.used_storage_buffer_types | Type::U32
        } else {
            Type::U32
        };
        if self.profile.support_int8
            && self.profile.support_storage_buffer_8bit
            && (types & Type::U8).0 != 0
        {
            self.storage_view(info, K::U8, *binding)?;
            self.storage_view(info, K::S8, *binding)?;
        }
        if self.profile.support_int16
            && self.profile.support_storage_buffer_16bit
            && (types & Type::U16).0 != 0
        {
            self.storage_view(info, K::U16, *binding)?;
            self.storage_view(info, K::S16, *binding)?;
        }
        for (ty, kind) in [
            (Type::U32, K::U32),
            (Type::F32, K::F32),
            (Type::U64, K::U64),
            (Type::U32x2, K::U32x2),
            (Type::U32x4, K::U32x4),
        ] {
            if (types & ty).0 != 0 {
                self.storage_view(info, kind, *binding)?;
            }
        }
        for (index, d) in info.storage_buffers_descriptors.iter().enumerate() {
            self.resource_bindings.push(ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(*binding),
                count: 1,
                resource_index: ResourceIndex(index as u32),
                source: ResourceSource::StorageBuffer(d.clone()),
                access: if d.is_written {
                    ResourceAccess::ReadWrite
                } else {
                    ResourceAccess::Read
                },
            });
            *binding = binding.wrapping_add(d.count);
        }
        Ok(())
    }
}
