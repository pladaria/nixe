use super::*;
#[derive(Clone, Copy, Debug)]
pub enum AttachmentLoad {
    Load,
    Clear([f32; 4]),
    Discard,
}
#[derive(Clone, Copy, Debug)]
pub enum AttachmentStore {
    Store,
    Discard,
}
pub(super) fn plane_range(p: ImagePlane) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(p.aspect)
        .base_mip_level(p.mip_level)
        .level_count(1)
        .base_array_layer(p.array_layer)
        .layer_count(1)
}
