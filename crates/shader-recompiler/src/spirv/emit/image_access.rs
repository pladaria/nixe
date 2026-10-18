use super::*;
use crate::{ir::TextureInstInfo, spirv::ImageDefinition, types::TextureType};
use spirv::Decoration;
impl EmitContext<'_> {
    pub(super) fn image_definition(
        &self,
        info: TextureInstInfo,
        storage: bool,
    ) -> Result<ImageDefinition> {
        let array = match (storage, info.texture_type() == TextureType::Buffer) {
            (false, false) => &self.textures,
            (false, true) => &self.texture_buffers,
            (true, false) => &self.images,
            (true, true) => &self.image_buffers,
        };
        array
            .get(info.descriptor_index() as usize)
            .copied()
            .ok_or_else(|| invariant("Image descriptor index out of range"))
    }
    fn nonuniform(&mut self, id: Id) -> Result<()> {
        if self.non_uniform_ids.insert(id) {
            self.builder.decorate(id, Decoration::NonUniform, &[])?;
        }
        Ok(())
    }
    pub(super) fn image_object(
        &mut self,
        ir: &Ir,
        info: TextureInstInfo,
        index: Value,
        storage: bool,
        raw: bool,
    ) -> Result<(Id, bool)> {
        let def = self.image_definition(info, storage)?;
        let buffer = info.texture_type() == TextureType::Buffer;
        let kind = match (storage, buffer) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        };
        let support = [
            self.profile.support_sampled_image_array_nonuniform_indexing,
            self.profile.support_storage_image_array_nonuniform_indexing,
            self.profile
                .support_uniform_texel_buffer_array_nonuniform_indexing,
            self.profile
                .support_storage_texel_buffer_array_nonuniform_indexing,
        ][kind];
        let marked = def.count > 1 && !ir.is_immediate(index)? && support;
        let mut pointer = def.id;
        if def.count > 1 {
            let index = self.def(ir, index)?;
            if marked {
                self.nonuniform(index)?;
                self.uses_nonuniform[kind] = true;
            }
            pointer = self.op(
                Op::AccessChain,
                if !storage && buffer {
                    self.objects.image_buffer_type
                } else {
                    def.pointer_type
                },
                &[pointer, index],
            )?;
        }
        let ty = if storage || buffer {
            def.image_type
        } else {
            def.sampled_type
        };
        let object = self.op(Op::Load, ty, &[pointer])?;
        if marked {
            self.nonuniform(pointer)?;
            self.nonuniform(object)?;
        }
        let object = if raw && !storage && !buffer {
            let image = self.op(Op::Image, def.image_type, &[object])?;
            if marked {
                self.nonuniform(image)?;
            }
            image
        } else {
            object
        };
        Ok((
            object,
            if buffer && !storage {
                false
            } else {
                def.is_integer
            },
        ))
    }
    pub(super) fn image_result(
        &mut self,
        ir: &mut Ir,
        id: InstId,
        op: Op,
        ty: Id,
        words: &[u32],
    ) -> Result<Id> {
        let sparse = ir.associated_pseudo(id, O::GetSparseFromOp)?.is_some();
        let result_ty = if sparse {
            self.builder.type_struct(&[self.types.u32[0], ty])?
        } else {
            ty
        };
        let op = if sparse {
            match op {
                Op::ImageSampleImplicitLod => Op::ImageSparseSampleImplicitLod,
                Op::ImageSampleExplicitLod => Op::ImageSparseSampleExplicitLod,
                Op::ImageSampleDrefImplicitLod => Op::ImageSparseSampleDrefImplicitLod,
                Op::ImageSampleDrefExplicitLod => Op::ImageSparseSampleDrefExplicitLod,
                Op::ImageGather => Op::ImageSparseGather,
                Op::ImageDrefGather => Op::ImageSparseDrefGather,
                Op::ImageFetch => Op::ImageSparseFetch,
                Op::ImageRead => Op::ImageSparseRead,
                _ => return Err(invariant("Operation cannot produce sparse residency")),
            }
        } else {
            op
        };
        let result = self.builder.result(op, result_ty, words)?;
        if sparse {
            let code = self.extract(self.types.u32[0], result, 0)?;
            let resident = self.op(Op::ImageSparseTexelsResident, self.types.bool, &[code])?;
            self.pseudo(ir, id, O::GetSparseFromOp, resident)?;
        }
        if TextureInstInfo(ir.inst(id)?.flags).relaxed_precision() {
            self.builder
                .decorate(result, Decoration::RelaxedPrecision, &[])?;
        }
        if sparse {
            self.extract(ty, result, 1)
        } else {
            Ok(result)
        }
    }
    pub(super) fn gather_coords(
        &mut self,
        info: TextureInstInfo,
        image: Id,
        coords: Id,
    ) -> Result<Id> {
        let n = match info.texture_type() {
            TextureType::Color2D | TextureType::Color2DRect => 2,
            TextureType::ColorArray2D | TextureType::ColorCube => 3,
            _ => return Ok(coords),
        };
        let size = self.op(
            Op::ImageQuerySizeLod,
            self.types.u32[n - 1],
            &[image, self.constants.u32_zero],
        )?;
        let nudge = self.const_f32(1.0 / 512.0)?;
        let values = [nudge, nudge, self.constants.f32_zero];
        let ty = self.types.f32[n - 1];
        let offset = self.builder.constant_composite(ty, &values[..n])?;
        let size = self.op(Op::ConvertUToF, ty, &[size])?;
        let offset = self.op(Op::FDiv, ty, &[offset, size])?;
        self.op(Op::FAdd, ty, &[coords, offset])
    }
    pub(super) fn fetch_coords(
        &mut self,
        info: TextureInstInfo,
        coords: Id,
        mut offset: Id,
    ) -> Result<Id> {
        if offset == Id::NONE {
            return Ok(coords);
        }
        let n = match info.texture_type() {
            TextureType::Buffer | TextureType::Color1D => 1,
            TextureType::ColorArray1D => {
                offset = self.op(
                    Op::CompositeConstruct,
                    self.types.u32[1],
                    &[offset, self.constants.u32_zero],
                )?;
                2
            }
            TextureType::Color2D | TextureType::Color2DRect => 2,
            TextureType::ColorArray2D => {
                let x = self.extract(self.types.u32[0], offset, 0)?;
                let y = self.extract(self.types.u32[0], offset, 1)?;
                offset = self.op(
                    Op::CompositeConstruct,
                    self.types.u32[2],
                    &[x, y, self.constants.u32_zero],
                )?;
                3
            }
            TextureType::Color3D => 3,
            TextureType::ColorCube | TextureType::ColorArrayCube => return Ok(coords),
        };
        self.op(Op::IAdd, self.types.u32[n - 1], &[coords, offset])
    }
}
