use crate::artifact::{
    ImageBufferDescriptor, ImageDescriptor, ShaderInfo, TextureBufferDescriptor, TextureDescriptor,
};

pub(super) fn texture_buffer(info: &mut ShaderInfo, desc: TextureBufferDescriptor) -> u32 {
    if let Some(index) = info
        .texture_buffer_descriptors
        .iter()
        .position(|d| *d == desc)
    {
        return index as u32;
    }
    let index = info.texture_buffer_descriptors.len() as u32;
    info.texture_buffer_descriptors.push(desc);
    index
}
pub(super) fn image_buffer(info: &mut ShaderInfo, desc: ImageBufferDescriptor) -> u32 {
    let key = |d: &ImageBufferDescriptor| {
        (
            d.format,
            d.cbuf_index,
            d.cbuf_offset,
            d.count,
            d.size_shift,
            d.handle_program,
        )
    };
    if let Some(index) = info
        .image_buffer_descriptors
        .iter()
        .position(|d| key(d) == key(&desc))
    {
        let d = &mut info.image_buffer_descriptors[index];
        d.is_written |= desc.is_written;
        d.is_read |= desc.is_read;
        d.is_integer |= desc.is_integer;
        return index as u32;
    }
    let index = info.image_buffer_descriptors.len() as u32;
    info.image_buffer_descriptors.push(desc);
    index
}
pub(super) fn texture(info: &mut ShaderInfo, desc: TextureDescriptor) -> u32 {
    let key = |d: &TextureDescriptor| {
        (
            d.texture_type,
            d.is_depth,
            d.has_secondary,
            d.cbuf_index,
            d.cbuf_offset,
            d.shift_left,
            d.secondary_cbuf_index,
            d.secondary_cbuf_offset,
            d.secondary_shift_left,
            d.count,
            d.size_shift,
            d.handle_program,
        )
    };
    if let Some(index) = info
        .texture_descriptors
        .iter()
        .position(|d| key(d) == key(&desc))
    {
        let d = &mut info.texture_descriptors[index];
        d.is_multisample |= desc.is_multisample;
        d.is_integer |= desc.is_integer;
        return index as u32;
    }
    let index = info.texture_descriptors.len() as u32;
    info.texture_descriptors.push(desc);
    index
}
pub(super) fn image(info: &mut ShaderInfo, desc: ImageDescriptor) -> u32 {
    let key = |d: &ImageDescriptor| {
        (
            d.texture_type,
            d.format,
            d.cbuf_index,
            d.cbuf_offset,
            d.count,
            d.size_shift,
            d.handle_program,
        )
    };
    if let Some(index) = info
        .image_descriptors
        .iter()
        .position(|d| key(d) == key(&desc))
    {
        let d = &mut info.image_descriptors[index];
        d.is_written |= desc.is_written;
        d.is_read |= desc.is_read;
        d.is_integer |= desc.is_integer;
        return index as u32;
    }
    let index = info.image_descriptors.len() as u32;
    info.image_descriptors.push(desc);
    index
}
pub(super) fn join_texture_info(
    base: &mut ShaderInfo,
    source: &ShaderInfo,
    storage: &[u32],
) -> [Vec<u32>; 4] {
    let mut programs = Vec::with_capacity(source.handle_programs.len());
    for code in &source.handle_programs {
        let mut code = code.clone();
        code.remap_storage(storage);
        let index = if let Some(i) = base.handle_programs.iter().position(|p| *p == code) {
            i
        } else {
            let i = base.handle_programs.len();
            base.handle_programs.push(code);
            i
        };
        programs.push(index as u32);
    }
    let mut indices = std::array::from_fn(|_| Vec::new());
    macro_rules! join {
        ($kind:ident,$list:ident,$index:literal) => {
            for d in &source.$list {
                let mut d = d.clone();
                d.handle_program = d.handle_program.map(|p| programs[p as usize]);
                indices[$index].push($kind(base, d));
            }
        };
    }
    join!(texture_buffer, texture_buffer_descriptors, 0);
    join!(image_buffer, image_buffer_descriptors, 1);
    join!(texture, texture_descriptors, 2);
    join!(image, image_descriptors, 3);
    indices
}

pub(super) fn remap(
    ir: &mut crate::ir::Ir,
    blocks: &[crate::arena::BlockId],
    textures: &[Vec<u32>; 4],
    storage: &[u32],
) -> crate::Result<()> {
    use crate::ir::{Opcode as O, TextureInstInfo, Value};
    use crate::types::TextureType;
    for &block in blocks {
        let count = ir.block(block)?.instructions().len();
        for i in 0..count {
            let id = ir.block(block)?.instructions()[i];
            let inst = ir.inst(id)?;
            let op = inst.opcode();
            if super::storage_table::is_storage(op) {
                if let Value::U32(binding) = inst.arg(0)? {
                    ir.set_arg(id, 0, Value::U32(storage[binding as usize]))?;
                }
            } else if matches!(op, O::IsTextureScaled | O::IsImageScaled) {
                if let Value::U32(index) = inst.arg(0)? {
                    ir.set_arg(
                        id,
                        0,
                        Value::U32(
                            textures[if op == O::IsTextureScaled { 2 } else { 3 }][index as usize],
                        ),
                    )?;
                }
            } else if super::texture_table::sampled(op) || super::texture_table::storage(op) {
                let mut flags = TextureInstInfo(inst.flags);
                let list = usize::from(flags.texture_type() != TextureType::Buffer) * 2
                    + usize::from(super::texture_table::storage(op));
                flags.set_descriptor_index(textures[list][flags.descriptor_index() as usize]);
                ir.inst_mut(id)?.flags = flags.0;
            }
        }
    }
    Ok(())
}
