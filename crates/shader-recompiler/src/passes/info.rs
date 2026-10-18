use super::{fold::arg, info_usage::simple_usages, storage_table::storage_opcode};
use crate::{
    CompileError, Environment, Result,
    arena::InstId,
    artifact::{ConstantBufferDescriptor, MAX_CBUFS, MAX_INDIRECT_CBUFS, ShaderInfo},
    diagnostics::FailureCategory,
    ir::{Attribute, FmzMode, FpControl, Ir, Opcode as O, Program, TextureInstInfo, Type, Value},
    maxwell::header::{ProgramHeader, SystemValuesB, SystemValuesC},
    types::{CbufIndex, ImageFormat, PixelImap, Stage, TextureType},
};
fn add_cbuf(info: &mut ShaderInfo, index: u32) -> Result<()> {
    if index >= MAX_CBUFS as u32 {
        return Err(CompileError::new(
            FailureCategory::ResourceLimit,
            "Constant buffer index exceeds compiler limit",
        ));
    }
    if info.constant_buffer_mask & (1 << index) != 0 {
        return Ok(());
    }
    info.constant_buffer_mask |= 1 << index;
    info.constant_buffer_descriptors
        .push(ConstantBufferDescriptor {
            index: CbufIndex(index),
            count: 1,
        })?;
    info.constant_buffer_descriptors
        .sort_unstable_by_key(|descriptor| descriptor.index);
    Ok(())
}
fn cbuf(info: &mut ShaderInfo, ir: &Ir, id: InstId) -> Result<()> {
    let (ty, size) = match ir.inst(id)?.opcode() {
        O::GetCbufU8 | O::GetCbufS8 => (Type::U8, 1),
        O::GetCbufU16 | O::GetCbufS16 => (Type::U16, 2),
        O::GetCbufU32 => (Type::U32, 4),
        O::GetCbufF32 => (Type::F32, 4),
        O::GetCbufU32x2 => (Type::U32x2, 8),
        _ => return Ok(()),
    };
    let index = arg(ir, id, 0)?;
    let offset = arg(ir, id, 1)?;
    if let Value::U32(index) = index {
        add_cbuf(info, index)?;
        info.used_constant_buffer_types |= ty;
        let used = &mut info.constant_buffer_used_sizes[index as usize];
        *used = if let Value::U32(offset) = offset {
            (*used).max(offset.wrapping_add(size)).wrapping_add(15) & !15
        } else {
            0x10000
        };
    } else {
        info.uses_cbuf_indirect = true;
        info.used_indirect_cbuf_types |= ty;
        for index in 0..MAX_INDIRECT_CBUFS {
            add_cbuf(info, index as u32)?;
            info.constant_buffer_used_sizes[index] = 0x10000;
        }
    }
    match (index, offset) {
        (Value::U32(0), Value::U32(offset))
            if (info.nvn_buffer_base..info.nvn_buffer_base + 16 * 16).contains(&offset) =>
        {
            info.nvn_buffer_used |= 1 << ((offset - info.nvn_buffer_base) / 16)
        }
        (Value::U32(0), Value::Inst(_)) | (Value::Inst(_), _) => info.nvn_buffer_used = u16::MAX,
        _ => (),
    }
    Ok(())
}
fn fp_control(info: &mut ShaderInfo, ir: &Ir, id: InstId) -> Result<()> {
    let width = match ir.inst(id)?.opcode() {
        O::FPAdd16
        | O::FPFma16
        | O::FPMul16
        | O::FPRoundEven16
        | O::FPFloor16
        | O::FPCeil16
        | O::FPTrunc16 => 16,
        O::FPAdd32
        | O::FPFma32
        | O::FPMul32
        | O::FPRoundEven32
        | O::FPFloor32
        | O::FPCeil32
        | O::FPTrunc32
        | O::FPOrdEqual32
        | O::FPUnordEqual32
        | O::FPOrdNotEqual32
        | O::FPUnordNotEqual32
        | O::FPOrdLessThan32
        | O::FPUnordLessThan32
        | O::FPOrdGreaterThan32
        | O::FPUnordGreaterThan32
        | O::FPOrdLessThanEqual32
        | O::FPUnordLessThanEqual32
        | O::FPOrdGreaterThanEqual32
        | O::FPUnordGreaterThanEqual32
        | O::ConvertF16F32
        | O::ConvertF64F32 => 32,
        _ => return Ok(()),
    };
    let mode = FpControl::from_bits(ir.inst(id)?.flags)?.fmz_mode;
    let flush = matches!(mode, FmzMode::FTZ | FmzMode::FMZ);
    let preserve = mode == FmzMode::None;
    if width == 16 {
        info.uses_fp16_denorms_flush |= flush;
        info.uses_fp16_denorms_preserve |= preserve;
    } else {
        info.uses_fp32_denorms_flush |= flush;
        info.uses_fp32_denorms_preserve |= preserve;
    }
    Ok(())
}
fn usages(info: &mut ShaderInfo, ir: &Ir, id: InstId) -> Result<()> {
    let inst = ir.inst(id)?;
    let op = inst.opcode();
    simple_usages(info, op);
    if storage_opcode(op).is_some() {
        info.stores_global_memory |= inst.may_have_side_effects();
        info.uses_global_memory = true;
        info.uses_int64 = true;
        info.used_constant_buffer_types |= Type::U32 | Type::U32x2;
        info.used_storage_buffer_types |= Type::U32 | Type::U32x2 | Type::U32x4;
    }
    match op {
        O::GetAttribute | O::GetAttributeU32 | O::SetAttribute => {
            let Value::Attribute(attribute) = arg(ir, id, 0)? else {
                return Err(malformed("Attribute metadata"));
            };
            if op == O::SetAttribute {
                info.stores.set(attribute, true);
            } else {
                info.loads.set(attribute, true);
            }
        }
        O::GetPatch | O::SetPatch => {
            let Value::Patch(patch) = arg(ir, id, 0)? else {
                return Err(malformed("Patch metadata"));
            };
            if patch.is_generic() {
                info.uses_patches[patch.generic_index()? as usize] = true;
            } else if op == O::GetPatch {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Reading non-generic patch",
                ));
            } else if patch.index() < 4 {
                info.stores_tess_level_outer = true;
            } else {
                info.stores_tess_level_inner = true;
            }
        }
        O::SetFragColor => {
            let Value::U32(target) = arg(ir, id, 0)? else {
                return Err(malformed("Fragment output index"));
            };
            *info
                .stores_frag_color
                .get_mut(target as usize)
                .ok_or_else(|| malformed("Fragment output index out of range"))? = true;
        }
        O::BindlessImageSampleImplicitLod
        | O::BindlessImageSampleExplicitLod
        | O::BindlessImageSampleDrefImplicitLod
        | O::BindlessImageSampleDrefExplicitLod
        | O::BindlessImageGather
        | O::BindlessImageGatherDref
        | O::BindlessImageFetch
        | O::BindlessImageQueryDimensions
        | O::BindlessImageQueryLod
        | O::BindlessImageGradient
        | O::BoundImageSampleImplicitLod
        | O::BoundImageSampleExplicitLod
        | O::BoundImageSampleDrefImplicitLod
        | O::BoundImageSampleDrefExplicitLod
        | O::BoundImageGather
        | O::BoundImageGatherDref
        | O::BoundImageFetch
        | O::BoundImageQueryDimensions
        | O::BoundImageQueryLod
        | O::BoundImageGradient
        | O::ImageGather
        | O::ImageGatherDref
        | O::ImageFetch
        | O::ImageQueryDimensions
        | O::ImageGradient
        | O::ImageSampleImplicitLod
        | O::ImageSampleExplicitLod
        | O::ImageSampleDrefImplicitLod
        | O::ImageSampleDrefExplicitLod
        | O::ImageQueryLod => {
            let flags = TextureInstInfo(inst.flags);
            info.uses_sampled_1d |= matches!(
                flags.texture_type(),
                TextureType::Color1D | TextureType::ColorArray1D
            );
            info.uses_sparse_residency |= ir.associated_pseudo(id, O::GetSparseFromOp)?.is_some();
            if matches!(
                op,
                O::ImageSampleImplicitLod
                    | O::ImageSampleExplicitLod
                    | O::ImageSampleDrefImplicitLod
                    | O::ImageSampleDrefExplicitLod
                    | O::ImageQueryLod
            ) {
                info.uses_shadow_lod |= flags.is_depth();
            }
        }
        O::ImageRead | O::ImageWrite => {
            let flags = TextureInstInfo(inst.flags);
            let typeless = flags.image_format() == ImageFormat::Typeless;
            info.uses_image_1d |= matches!(
                flags.texture_type(),
                TextureType::Color1D | TextureType::ColorArray1D
            );
            if op == O::ImageRead {
                info.uses_typeless_image_reads |= typeless;
                info.uses_sparse_residency |=
                    ir.associated_pseudo(id, O::GetSparseFromOp)?.is_some();
            } else {
                info.uses_typeless_image_writes |= typeless;
                info.uses_image_buffers |= flags.texture_type() == TextureType::Buffer;
            }
        }
        O::BindlessImageAtomicIAdd32
        | O::BindlessImageAtomicSMin32
        | O::BindlessImageAtomicUMin32
        | O::BindlessImageAtomicSMax32
        | O::BindlessImageAtomicUMax32
        | O::BindlessImageAtomicInc32
        | O::BindlessImageAtomicDec32
        | O::BindlessImageAtomicAnd32
        | O::BindlessImageAtomicOr32
        | O::BindlessImageAtomicXor32
        | O::BindlessImageAtomicExchange32
        | O::BoundImageAtomicIAdd32
        | O::BoundImageAtomicSMin32
        | O::BoundImageAtomicUMin32
        | O::BoundImageAtomicSMax32
        | O::BoundImageAtomicUMax32
        | O::BoundImageAtomicInc32
        | O::BoundImageAtomicDec32
        | O::BoundImageAtomicAnd32
        | O::BoundImageAtomicOr32
        | O::BoundImageAtomicXor32
        | O::BoundImageAtomicExchange32
        | O::ImageAtomicIAdd32
        | O::ImageAtomicSMin32
        | O::ImageAtomicUMin32
        | O::ImageAtomicSMax32
        | O::ImageAtomicUMax32
        | O::ImageAtomicInc32
        | O::ImageAtomicDec32
        | O::ImageAtomicAnd32
        | O::ImageAtomicOr32
        | O::ImageAtomicXor32
        | O::ImageAtomicExchange32 => {
            info.uses_atomic_image_u32 = true;
            info.uses_image_1d |= matches!(
                TextureInstInfo(inst.flags).texture_type(),
                TextureType::Color1D | TextureType::ColorArray1D
            );
        }
        _ => (),
    }
    fp_control(info, ir, id)?;
    cbuf(info, ir, id)
}
fn malformed(message: &str) -> CompileError {
    CompileError::new(FailureCategory::Internal, message)
}
fn system_varyings(mask: &mut crate::artifact::VaryingState, b: SystemValuesB, c: SystemValuesC) {
    for (bit, attribute) in [
        Attribute::PrimitiveId,
        Attribute::Layer,
        Attribute::ViewportIndex,
        Attribute::PointSize,
        Attribute::PositionX,
        Attribute::PositionY,
        Attribute::PositionZ,
        Attribute::PositionW,
    ]
    .into_iter()
    .enumerate()
    {
        mask.set(attribute, b.0 & (1 << bit) != 0);
    }
    for (bit, attribute) in [
        (8, Attribute::PointSpriteS),
        (9, Attribute::PointSpriteT),
        (10, Attribute::FogCoordinate),
        (12, Attribute::TessellationEvaluationPointU),
        (13, Attribute::TessellationEvaluationPointV),
        (14, Attribute::InstanceId),
        (15, Attribute::VertexId),
    ] {
        mask.set(attribute, c.0 & (1 << bit) != 0);
    }
    for index in 0..8 {
        mask.set(
            Attribute::from_index((Attribute::ClipDistance0.index() + index) as u16).unwrap(),
            c.clip_distances() & (1 << index) != 0,
        );
    }
}
fn header(info: &mut ShaderInfo, stage: Stage, sph: &ProgramHeader) {
    if stage == Stage::Compute {
        return;
    }
    if stage == Stage::Fragment {
        if info.loads_indexed_attributes {
            for index in 0..32 {
                for (component, mode) in sph.generic_input_map(index).into_iter().enumerate() {
                    info.loads.set(
                        Attribute::generic(index, component).unwrap(),
                        mode != PixelImap::Unused,
                    );
                }
            }
        }
        return;
    }
    if info.loads_indexed_attributes {
        for index in 0..32 {
            for (component, enabled) in sph.input_generic(index).into_iter().enumerate() {
                info.loads
                    .set(Attribute::generic(index, component).unwrap(), enabled);
            }
        }
        system_varyings(
            &mut info.loads,
            sph.vtg_input_system_b(),
            sph.vtg_input_system_c(),
        );
    }
    if info.stores_indexed_attributes {
        for index in 0..32 {
            for (component, enabled) in sph.output_generic(index).into_iter().enumerate() {
                info.stores
                    .set(Attribute::generic(index, component).unwrap(), enabled);
            }
        }
        system_varyings(
            &mut info.stores,
            sph.vtg_output_system_b(),
            sph.vtg_output_system_c(),
        );
        for index in 0..8 {
            if sph.vtg_output_system_c().clip_distances() & (1 << index) != 0 {
                info.used_clip_distances = index + 1;
            }
        }
    }
}
pub fn collect_shader_info<E: Environment>(env: &E, ir: &Ir, program: &mut Program) -> Result<()> {
    program.info.nvn_buffer_base = match program.stage {
        Stage::VertexA | Stage::VertexB => 0x110,
        Stage::TessellationControl => 0x210,
        Stage::TessellationEval | Stage::Compute => 0x310,
        Stage::Geometry => 0x410,
        Stage::Fragment => 0x510,
    };
    for &block in &program.post_order {
        for &id in ir.block(block)?.instructions() {
            usages(&mut program.info, ir, id)?;
        }
    }
    header(&mut program.info, env.metadata().stage, &env.metadata().sph);
    Ok(())
}
