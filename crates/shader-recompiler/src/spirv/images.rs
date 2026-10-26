use super::{EmitContext, Id, ImageDefinition};
use crate::{
    CompileError, Result,
    artifact::{ResourceAccess, ResourceBinding, ResourceSource, ShaderInfo},
    diagnostics::FailureCategory,
    types::{DescriptorBinding, DescriptorSet, ImageFormat, ResourceIndex, TextureType as T},
};
use spirv::{Dim, ImageFormat as F, StorageClass as S};
fn format(format: ImageFormat) -> F {
    match format {
        ImageFormat::Typeless => F::Unknown,
        ImageFormat::R8_UINT => F::R8ui,
        ImageFormat::R8_SINT => F::R8i,
        ImageFormat::R16_UINT => F::R16ui,
        ImageFormat::R16_SINT => F::R16i,
        ImageFormat::R32_UINT => F::R32ui,
        ImageFormat::R32G32_UINT => F::Rg32ui,
        ImageFormat::R32G32B32A32_UINT => F::Rgba32ui,
    }
}
fn access(read: bool, write: bool) -> ResourceAccess {
    if read && write {
        ResourceAccess::ReadWrite
    } else if write {
        ResourceAccess::Write
    } else {
        ResourceAccess::Read
    }
}
impl EmitContext<'_> {
    fn sampled_image_type(
        &mut self,
        ty: T,
        integer: bool,
        depth: bool,
        multisample: bool,
    ) -> Result<Id> {
        let (dim, array, ms) = match ty {
            T::Color1D => (Dim::Dim1D, false, false),
            T::ColorArray1D => (Dim::Dim1D, true, false),
            T::Color2D | T::Color2DRect => (Dim::Dim2D, false, multisample),
            T::ColorArray2D => (Dim::Dim2D, true, multisample),
            T::Color3D => (Dim::Dim3D, false, false),
            T::ColorCube => (Dim::DimCube, false, false),
            T::ColorArrayCube => (Dim::DimCube, true, false),
            T::Buffer => {
                return Err(CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Buffer is not a sampled texture dimension",
                ));
            }
        };
        if dim == Dim::DimCube && array {
            self.builder
                .capability(spirv::Capability::SampledCubeArray)?;
        }
        self.builder.type_image(
            if integer {
                self.types.u32[0]
            } else {
                self.types.f32[0]
            },
            dim,
            depth,
            array,
            ms,
            1,
            F::Unknown,
        )
    }
    fn storage_image_type(
        &mut self,
        ty: T,
        integer: bool,
        pixel_format: ImageFormat,
    ) -> Result<Id> {
        let (dim, array) = match ty {
            T::Color1D => (Dim::Dim1D, false),
            T::ColorArray1D => (Dim::Dim1D, true),
            T::Color2D => (Dim::Dim2D, false),
            T::ColorArray2D => (Dim::Dim2D, true),
            T::Color3D => (Dim::Dim3D, false),
            T::Buffer => {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Storage image buffer in image descriptor list",
                ));
            }
            _ => {
                return Err(CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Unsupported storage image dimension",
                ));
            }
        };
        self.builder.type_image(
            if integer {
                self.types.u32[0]
            } else {
                self.types.f32[0]
            },
            dim,
            false,
            array,
            false,
            2,
            format(pixel_format),
        )
    }
    pub(super) fn define_texture_buffers(
        &mut self,
        info: &ShaderInfo,
        binding: &mut u32,
    ) -> Result<()> {
        if info.texture_buffer_descriptors.is_empty() {
            return Ok(());
        }
        let image = self.builder.type_image(
            self.types.f32[0],
            Dim::DimBuffer,
            false,
            false,
            false,
            1,
            F::Unknown,
        )?;
        self.objects.image_buffer_type = image;
        let pointer = self.builder.type_pointer(S::UniformConstant, image)?;
        for (index, d) in info.texture_buffer_descriptors.iter().enumerate() {
            if d.count != 1 {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Array of texture buffers",
                ));
            }
            let id = self.builder.variable(pointer, S::UniformConstant, None)?;
            self.bind(id, *binding)?;
            self.texture_buffers.push(ImageDefinition {
                id,
                image_type: image,
                pointer_type: pointer,
                count: d.count,
                ..Default::default()
            });
            self.resource_bindings.push(ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(*binding),
                count: 1,
                resource_index: ResourceIndex(index as u32),
                source: ResourceSource::TextureBuffer(d.clone()),
                access: ResourceAccess::Read,
            });
            *binding = binding.wrapping_add(1);
        }
        Ok(())
    }
    pub(super) fn define_image_buffers(
        &mut self,
        info: &ShaderInfo,
        binding: &mut u32,
    ) -> Result<()> {
        for (index, d) in info.image_buffer_descriptors.iter().enumerate() {
            let sampled = if d.is_integer {
                self.types.u32[0]
            } else {
                self.types.f32[0]
            };
            let image = self.builder.type_image(
                sampled,
                Dim::DimBuffer,
                false,
                false,
                false,
                2,
                format(d.format),
            )?;
            let pointer = self.builder.type_pointer(S::UniformConstant, image)?;
            let id = self.builder.variable(pointer, S::UniformConstant, None)?;
            self.bind(id, *binding)?;
            self.image_buffers.push(ImageDefinition {
                id,
                image_type: image,
                pointer_type: pointer,
                count: d.count,
                is_integer: d.is_integer,
                ..Default::default()
            });
            self.resource_bindings.push(ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(*binding),
                count: 1,
                resource_index: ResourceIndex(index as u32),
                source: ResourceSource::ImageBuffer(d.clone()),
                access: access(d.is_read, d.is_written),
            });
            *binding = binding.wrapping_add(1);
        }
        Ok(())
    }
    pub(super) fn define_textures(
        &mut self,
        info: &ShaderInfo,
        binding: &mut u32,
        scaling: &mut u32,
    ) -> Result<()> {
        for (index, d) in info.texture_descriptors.iter().enumerate() {
            let image = self.sampled_image_type(
                d.texture_type,
                d.is_integer,
                d.is_depth,
                d.is_multisample,
            )?;
            let sampled = self.builder.type_sampled_image(image)?;
            let pointer = self.builder.type_pointer(S::UniformConstant, sampled)?;
            let declaration_pointer = if d.count > 1 {
                let count = self.const_u32(d.count)?;
                let array = self.builder.type_array(sampled, count)?;
                self.builder.type_pointer(S::UniformConstant, array)?
            } else {
                pointer
            };
            let id = self
                .builder
                .variable(declaration_pointer, S::UniformConstant, None)?;
            self.bind(id, *binding)?;
            self.textures.push(ImageDefinition {
                id,
                image_type: image,
                sampled_type: sampled,
                pointer_type: pointer,
                count: d.count,
                is_integer: d.is_integer,
                is_multisample: d.is_multisample,
            });
            self.resource_bindings.push(ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(*binding),
                count: d.count.max(1),
                resource_index: ResourceIndex(index as u32),
                source: ResourceSource::Texture(d.clone()),
                access: ResourceAccess::Read,
            });
            *binding = binding.wrapping_add(1);
            *scaling = scaling.wrapping_add(1);
        }
        if info.uses_atomic_image_u32 {
            self.objects.image_u32 = self.builder.type_pointer(S::Image, self.types.u32[0])?;
        }
        Ok(())
    }
    pub(super) fn define_images(
        &mut self,
        info: &ShaderInfo,
        binding: &mut u32,
        scaling: &mut u32,
    ) -> Result<()> {
        for (index, d) in info.image_descriptors.iter().enumerate() {
            let image = self.storage_image_type(d.texture_type, d.is_integer, d.format)?;
            let pointer = self.builder.type_pointer(S::UniformConstant, image)?;
            let id = self.builder.variable(pointer, S::UniformConstant, None)?;
            self.bind(id, *binding)?;
            self.images.push(ImageDefinition {
                id,
                image_type: image,
                pointer_type: pointer,
                count: d.count,
                is_integer: d.is_integer,
                ..Default::default()
            });
            self.resource_bindings.push(ResourceBinding {
                set: DescriptorSet(0),
                binding: DescriptorBinding(*binding),
                count: 1,
                resource_index: ResourceIndex(index as u32),
                source: ResourceSource::Image(d.clone()),
                access: access(d.is_read, d.is_written),
            });
            *binding = binding.wrapping_add(1);
            *scaling = scaling.wrapping_add(1);
        }
        Ok(())
    }
}
