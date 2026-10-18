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
    let key =
        |d: &ImageBufferDescriptor| (d.format, d.cbuf_index, d.cbuf_offset, d.count, d.size_shift);
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
pub fn join_texture_info(base: &mut ShaderInfo, source: &ShaderInfo) {
    for d in &source.texture_buffer_descriptors {
        texture_buffer(base, d.clone());
    }
    for d in &source.image_buffer_descriptors {
        image_buffer(base, d.clone());
    }
    for d in &source.texture_descriptors {
        texture(base, d.clone());
    }
    for d in &source.image_descriptors {
        image(base, d.clone());
    }
}
