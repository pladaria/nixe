use super::{
    fold::arg,
    texture_descriptors as descriptors,
    texture_table::{indexed, storage},
    texture_tracking::{Address, Tracker, direct, sampled_limit, size_shift},
};
use crate::{
    CompileError, HostTranslateInfo, Result,
    arena::{BlockId, InstId},
    artifact::{
        DescriptorCount, ImageBufferDescriptor, ImageDescriptor, ShaderInfo,
        TextureBufferDescriptor, TextureDescriptor, num_descriptors,
    },
    diagnostics::FailureCategory,
    environment::{Environment, EnvironmentCaches},
    ir::{Emitter, Ir, Opcode as O, Program, TextureInstInfo, Value},
    types::{ByteOffset, TexturePixelFormat as P, TextureType as T},
};

fn static_count<T: DescriptorCount>(descriptors: &[T]) -> u32 {
    descriptors
        .iter()
        .filter(|d| d.count() <= 1)
        .fold(0u32, |n, d| n.wrapping_add(d.count()))
}
fn dynamic_cap(info: &ShaderInfo, host: &HostTranslateInfo, arrays: u32) -> u32 {
    let sampled = sampled_limit(host);
    let resources = host.max_per_stage_resources.max(1);
    if arrays == 0 {
        return 1024.min(sampled).min(resources);
    }
    let static_sampled = static_count(&info.texture_buffer_descriptors)
        .wrapping_add(static_count(&info.texture_descriptors));
    let static_resources = num_descriptors(&info.constant_buffer_descriptors)
        .wrapping_add(num_descriptors(&info.storage_buffers_descriptors))
        .wrapping_add(static_sampled)
        .wrapping_add(num_descriptors(&info.image_buffer_descriptors))
        .wrapping_add(num_descriptors(&info.image_descriptors));
    (sampled.saturating_sub(static_sampled) / arrays)
        .min(resources.saturating_sub(static_resources) / arrays)
        .max(1)
}
fn rectangle(ir: &mut Ir, block: BlockId, id: InstId) -> Result<()> {
    let flags = ir.inst(id)?.flags;
    let coord = ir.inst(id)?.arg(1)?;
    let mut e = Emitter::before(ir, block, id)?;
    // This query is introduced after resource collection and shares descriptor zero.
    let dimensions = e.emit(
        O::ImageQueryDimensions,
        &[Value::U32(0), Value::U32(0), Value::U1(true)],
        flags,
    )?;
    let mut normalized = [Value::None; 2];
    for (i, value) in normalized.iter_mut().enumerate() {
        let size = e.emit(
            O::CompositeExtractU32x4,
            &[dimensions, Value::U32(i as u32)],
            0,
        )?;
        let size = e.emit(O::ConvertF32U32, &[size], 0)?;
        let reciprocal = e.emit(O::FPRecip32, &[size], 0)?;
        let coordinate = e.emit(O::CompositeExtractF32x2, &[coord, Value::U32(i as u32)], 0)?;
        *value = e.emit(O::FPMul32, &[coordinate, reciprocal], 0)?;
    }
    let normalized = e.emit(O::CompositeConstructF32x2, &normalized, 0)?;
    e.ir.set_arg(id, 1, normalized)
}
fn snorm(ir: &mut Ir, block: BlockId, id: InstId, format: P) -> Result<()> {
    let scale: f32 = match format {
        P::A8B8G8R8_SNORM | P::R8G8_SNORM | P::R8_SNORM => 1.0 / 127.0,
        P::R16G16B16A16_SNORM | P::R16G16_SNORM | P::R16_SNORM => 1.0 / 32767.0,
        _ => return Ok(()),
    };
    let inst = ir.inst(id)?;
    let flags = inst.flags;
    let args = (0..inst.num_args())
        .map(|i| inst.arg(i))
        .collect::<Result<Vec<_>>>()?;
    let mut e = Emitter::before(ir, block, id)?;
    let fetch = e.emit(O::ImageFetch, &args, flags)?;
    let mut components = [Value::None; 4];
    for (i, component) in components.iter_mut().enumerate() {
        let value = e.emit(O::CompositeExtractF32x4, &[fetch, Value::U32(i as u32)], 0)?;
        let value = e.emit(O::BitCastU32F32, &[value], 0)?;
        let value = e.emit(O::ConvertF32S32, &[value], 0)?;
        *component = e.emit(O::FPMul32, &[value, Value::F32(scale.to_bits())], 0)?;
    }
    let converted = e.emit(O::CompositeConstructF32x4, &components, 0)?;
    e.ir.replace_uses_with(id, converted)
}

pub fn texture_pass(
    ir: &mut Ir,
    program: &mut Program,
    env: &mut dyn Environment,
    host: &HostTranslateInfo,
    caches: &mut EnvironmentCaches<InstId, Value>,
) -> Result<()> {
    let mut tracker = Tracker::new(env, caches, host);
    let mut replacements = Vec::<(BlockId, InstId, O, Address)>::new();
    for &block in &program.post_order {
        for &id in ir.block(block)?.instructions() {
            let Some((op, bindless)) = indexed(ir.inst(id)?.opcode()) else {
                continue;
            };
            let address = if bindless {
                tracker.track(ir, arg(ir, id, 0)?)?.ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::Unsupported,
                        "Failed to track bindless texture constant buffer",
                    )
                })?
            } else {
                let Value::U32(offset) = arg(ir, id, 0)? else {
                    return Err(CompileError::new(
                        FailureCategory::InvalidArgument,
                        "Bound texture offset must be immediate",
                    ));
                };
                direct(
                    tracker.env.metadata().texture_bound_buffer,
                    ByteOffset(offset),
                )
            };
            replacements.push((block, id, op, address));
        }
    }
    replacements.sort_unstable_by_key(|(_, _, _, address)| (address.index, address.offset));
    let arrays = replacements
        .iter()
        .filter(|(_, id, op, address)| {
            address.count > 1
                && !storage(*op)
                && TextureInstInfo(ir.inst(*id).expect("Collected instruction exists").flags)
                    .texture_type()
                    != T::Buffer
        })
        .count() as u32;
    let cap = dynamic_cap(&program.info, host, arrays);
    let mut last_integer = None;
    for (block, id, op, a) in replacements {
        ir.replace_opcode(id, op)?;
        let mut flags = TextureInstInfo(ir.inst(id)?.flags);
        let mut multisample = false;
        match op {
            O::ImageQueryDimensions => {
                let handle = tracker.handle(&a)?;
                flags.set_texture_type(tracker.env.read_texture_type(handle)?);
                ir.inst_mut(id)?.flags = flags.0;
            }
            O::ImageSampleImplicitLod if flags.texture_type() == T::Color2D => {
                let handle = tracker.handle(&a)?;
                if tracker.env.read_texture_type(handle)? == T::Color2DRect {
                    rectangle(ir, block, id)?;
                }
            }
            O::ImageFetch => {
                if matches!(
                    flags.texture_type(),
                    T::Color2D | T::Color2DRect | T::ColorArray2D
                ) {
                    multisample = arg(ir, id, 4)? != Value::None;
                } else {
                    ir.set_arg(id, 4, Value::None)?;
                }
                if flags.texture_type() == T::Color1D {
                    let handle = tracker.handle(&a)?;
                    if tracker.env.read_texture_type(handle)? == T::Buffer {
                        flags.set_texture_type(T::Buffer);
                    }
                }
            }
            _ => (),
        }
        let shift = if a.count > 1 {
            size_shift(ir, a.dynamic_offset)?
        } else {
            3
        };
        let mut count = a.count;
        let is_storage = storage(op);
        if is_storage && a.has_secondary {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                "Storage image has a separate sampler",
            ));
        }
        let integer = if is_storage || flags.texture_type() != T::Buffer {
            if let Some((index, offset, integer)) = last_integer
                && index == a.index
                && offset == a.offset
            {
                integer
            } else {
                let handle = tracker.handle(&a)?;
                let integer = tracker.env.is_texture_pixel_format_integer(handle)?;
                last_integer = Some((a.index, a.offset, integer));
                integer
            }
        } else {
            false
        };
        let index = if is_storage {
            let written = op != O::ImageRead;
            let read = op != O::ImageWrite;
            if flags.texture_type() == T::Buffer {
                descriptors::image_buffer(
                    &mut program.info,
                    ImageBufferDescriptor {
                        format: flags.image_format(),
                        is_written: written,
                        is_read: read,
                        is_integer: integer,
                        cbuf_index: a.index,
                        cbuf_offset: a.offset,
                        count,
                        size_shift: shift,
                    },
                )
            } else {
                descriptors::image(
                    &mut program.info,
                    ImageDescriptor {
                        texture_type: flags.texture_type(),
                        format: flags.image_format(),
                        is_written: written,
                        is_read: read,
                        is_integer: integer,
                        cbuf_index: a.index,
                        cbuf_offset: a.offset,
                        count,
                        size_shift: shift,
                    },
                )
            }
        } else if flags.texture_type() == T::Buffer {
            descriptors::texture_buffer(
                &mut program.info,
                TextureBufferDescriptor {
                    has_secondary: a.has_secondary,
                    cbuf_index: a.index,
                    cbuf_offset: a.offset,
                    shift_left: a.shift_left,
                    secondary_cbuf_index: a.secondary_index,
                    secondary_cbuf_offset: a.secondary_offset,
                    secondary_shift_left: a.secondary_shift_left,
                    count,
                    size_shift: shift,
                },
            )
        } else {
            count = count.min(cap);
            flags.set_is_integer(integer);
            descriptors::texture(
                &mut program.info,
                TextureDescriptor {
                    texture_type: flags.texture_type(),
                    is_depth: flags.is_depth(),
                    is_multisample: multisample,
                    is_integer: integer,
                    has_secondary: a.has_secondary,
                    cbuf_index: a.index,
                    cbuf_offset: a.offset,
                    shift_left: a.shift_left,
                    secondary_cbuf_index: a.secondary_index,
                    secondary_cbuf_offset: a.secondary_offset,
                    secondary_shift_left: a.secondary_shift_left,
                    count,
                    size_shift: shift,
                },
            )
        };
        flags.set_descriptor_index(index);
        ir.inst_mut(id)?.flags = flags.0;
        let array_index = if count > 1 {
            let mut e = Emitter::before(ir, block, id)?;
            let index = e.emit(
                O::ShiftRightLogical32,
                &[a.dynamic_offset, Value::U32(shift)],
                0,
            )?;
            e.emit(O::UMin32, &[index, Value::U32(count - 1)], 0)?
        } else {
            Value::None
        };
        ir.set_arg(id, 0, array_index)?;
        if !host.support_snorm_render_buffer
            && op == O::ImageFetch
            && flags.texture_type() == T::Buffer
        {
            let handle = tracker.handle(&a)?;
            let format = tracker.env.read_texture_pixel_format(handle)?;
            snorm(ir, block, id, format)?;
        }
    }
    Ok(())
}
