//! Atomic resource resolution for immutable `MAXWELL_B` state snapshots.
//!
//! Maxwell GPU virtual addresses and PTE kinds are interpreted here, while
//! canonical storage and neutral resource views retain their own identities.
//! No host layout, swizzle operation, or backend object crosses this boundary.

#[cfg(test)]
mod benchmark;
mod compute;
pub(crate) use compute::ComputeTextures;

use std::{
    cell::Cell,
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{
    AddressMode, BackingView, BlockLinearLayout, BufferDescription, BufferId, BufferView,
    ComponentSwizzle, FilterMode, GpuAllocationDescription, GpuAllocationId, GpuVirtualAddress,
    ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind,
    ImageMemoryLayout, ImageSubresourceRange, ImageView, SampleCount, SamplerDescription, Swizzle,
};
use nixe_memory::MemoryPermissions;

use crate::address_space::MaxwellGpuReadBatch;
use crate::engines::MaxwellShaderBufferSize;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    MaxwellAllocationId, MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellMappingId,
    MaxwellResolvedRange,
};

use super::state::MaxwellThreeDResourceGroupIdentity;
use super::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MaxwellThreeDAttachmentReadiness,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetState, MaxwellThreeDDepthArrayControl,
    MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState,
    MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionValue, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDResourceStateIdentity, MaxwellThreeDSampleMode,
    MaxwellThreeDSamplerBindingMode, MaxwellThreeDState, MaxwellThreeDUnresolvedAddress,
};

// Public Switch NVIDIA memory kinds. Only the verified compression families
// listed below are accepted; accepting a kind does not make compressed guest
// bytes directly importable as generic block-linear storage.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h#L18-L250
const MAXWELL_PITCH_KIND: u8 = 0x00;
const MAXWELL_PITCH_NO_SWIZZLE_KIND: u8 = 0xfd;
const MAXWELL_GENERIC_BLOCK_LINEAR_KIND: u8 = 0xfe;
// Deko3d's verified single-sample compressed color kinds. These retain the
// same block-linear address equation as the corresponding generic surface,
// but identify the color-compression family selected for each texel width.
// https://github.com/devkitPro/deko3d/blob/6ee80db52aac0168303fc2f6417232997e464999/source/maxwell/image_formats.cpp#L95-L128
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h#L210-L243
const MAXWELL_C32_2CRA_KIND: u8 = 0xdb;
const MAXWELL_C64_2CRA_KIND: u8 = 0xe9;
const MAXWELL_C128_2CR_KIND: u8 = 0xf5;
// Single-sample 2CZ depth kinds preserve distinct guest depth/stencil packing.
// They require materialization, not a generic block-linear byte import.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h#L35-L100
// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.cpp#L35-L73
const MAXWELL_S8Z24_2CZ_KIND: u8 = 0x17;
const MAXWELL_Z24S8_2CZ_KIND: u8 = 0x51;
// Single-sample Z16 depth compression. Render-target descriptors still define
// its block-linear geometry; native depth clears/draws materialize the pixels.
// https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/volta/gv100/dev_mmu.ref.txt
const MAXWELL_Z16_2Z_KIND: u8 = 0x07;
// Four-sample kinds selected by deko3d. Their bytes are never imported as
// single-sample texels, even when compression of subsequent writes is off.
// https://github.com/devkitPro/deko3d/blob/master/source/maxwell/image_formats.cpp
// https://github.com/switchbrew/libnx/blob/master/nx/include/switch/nvidia/types.h
const MAXWELL_C32_MS4_2CBR_KIND: u8 = 0xe0;
const MAXWELL_S8Z24_MS4_2CZ_KIND: u8 = 0x19;
const MAXWELL_Z24S8_MS4_2CZ_KIND: u8 = 0x53;
const MAXWELL_DESCRIPTOR_SIZE: u64 = 32;

/// Frontend role of one completely resolved resource.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellThreeDResourceRole {
    VertexStream(u8),
    IndexBuffer,
    ConstantBuffer {
        group: u8,
        slot: u8,
    },
    StorageBuffer {
        group: u8,
        slot: u8,
        byte_offset: u32,
    },
    TextureHeaders,
    Samplers,
    SampledImage {
        texture: MaxwellThreeDTextureReference,
        dimension: MaxwellThreeDTextureDimension,
    },
    Sampler(MaxwellThreeDTextureReference),
    ColorTarget(u8),
    DepthStencilTarget,
    BlitSource,
    BlitDestination,
}

/// Shader-visible dimensionality required from a Maxwell sampled-image TIC.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellThreeDTextureDimension {
    Two,
    Three,
    TwoArray,
    Cube,
    CubeArray,
}

/// Draw-local location of a raw Maxwell TIC/TSC handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaxwellThreeDTextureReference {
    pub(super) group: u8,
    pub(super) constant_buffer_slot: u8,
    byte_offset: u32,
    shift_left: u8,
    pub(super) secondary: Option<(u8, u32, u8)>,
}

impl MaxwellThreeDTextureReference {
    #[must_use]
    pub const fn new(group: u8, constant_buffer_slot: u8, byte_offset: u32) -> Self {
        Self {
            group,
            constant_buffer_slot,
            byte_offset,
            shift_left: 0,
            secondary: None,
        }
    }

    pub fn expression(
        group: u8,
        primary: (u8, u32, u8),
        secondary: Option<(u8, u32, u8)>,
    ) -> Option<Self> {
        let valid = |(slot, _, shift): (u8, u32, u8)| {
            usize::from(slot) < MAXWELL_CONSTANT_BUFFER_SLOT_COUNT && shift < 32
        };
        if usize::from(group) >= MAXWELL_BIND_GROUP_COUNT
            || !valid(primary)
            || secondary.is_some_and(|part| !valid(part))
        {
            return None;
        }
        Some(Self {
            group,
            constant_buffer_slot: primary.0,
            byte_offset: primary.1,
            shift_left: primary.2,
            secondary,
        })
    }
}

/// Access implied by the state reference before neutral command lowering.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDResourceAccess {
    Read,
    Write,
}

/// Exact mapping lifetime retained by one resource range.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDMappingReference {
    mapping: MaxwellMappingId,
    generation: nixe_gpu::MappingGeneration,
    allocation: MaxwellAllocationId,
    backing_offset: u64,
    size: u64,
    kind: u8,
}

impl MaxwellThreeDMappingReference {
    #[must_use]
    pub const fn mapping(self) -> MaxwellMappingId {
        self.mapping
    }
    #[must_use]
    pub const fn generation(self) -> nixe_gpu::MappingGeneration {
        self.generation
    }
    #[must_use]
    pub const fn allocation(self) -> MaxwellAllocationId {
        self.allocation
    }
    #[must_use]
    pub const fn backing_offset(self) -> u64 {
        self.backing_offset
    }
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }
    #[must_use]
    pub const fn kind(self) -> u8 {
        self.kind
    }
}

/// Buffer interpretation published only after complete snapshot validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedBuffer {
    role: MaxwellThreeDResourceRole,
    access: MaxwellThreeDResourceAccess,
    description: BufferDescription,
    allocation_description: GpuAllocationDescription,
    view: BufferView,
    source: MaxwellResolvedRange,
    mappings: Arc<[MaxwellThreeDMappingReference]>,
}

impl MaxwellThreeDResolvedBuffer {
    #[must_use]
    pub const fn role(&self) -> MaxwellThreeDResourceRole {
        self.role
    }
    #[must_use]
    pub const fn access(&self) -> MaxwellThreeDResourceAccess {
        self.access
    }
    #[must_use]
    pub const fn description(&self) -> BufferDescription {
        self.description
    }
    #[must_use]
    pub const fn allocation_description(&self) -> GpuAllocationDescription {
        self.allocation_description
    }
    #[must_use]
    pub const fn view(&self) -> &BufferView {
        &self.view
    }
    #[must_use]
    pub const fn source(&self) -> &MaxwellResolvedRange {
        &self.source
    }
    #[must_use]
    pub fn mappings(&self) -> &[MaxwellThreeDMappingReference] {
        &self.mappings
    }
}

/// Guest image layout retained without conversion to a host representation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDPreservedImageLayout {
    layout: ImageMemoryLayout,
    pte_kind: u8,
    compression_enabled: bool,
}

/// Original Maxwell texel encoding retained independently of neutral component semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDGuestImageFormat {
    /// Canonical texels authored by another device frontend (for example VIC).
    Canonical(ImageFormat),
    Color(MaxwellThreeDColorTargetFormat),
    DepthStencil(MaxwellThreeDDepthStencilFormat),
    Texture(u32),
}

/// Source-preserving neutral interpretation of one Maxwell TSC entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedSampler {
    role: MaxwellThreeDResourceRole,
    compare: Option<nixe_gpu::DepthCompareOperation>,
    min_filter: FilterMode,
    mag_filter: FilterMode,
    mip_filter: FilterMode,
    address_modes: [AddressMode; 3],
    border_color: Option<[u32; 4]>,
    lod_bias_fixed: i16,
    lod_min_fixed: u16,
    lod_max_fixed: u16,
    max_anisotropy: u8,
}

impl MaxwellThreeDResolvedSampler {
    #[must_use]
    pub const fn role(self) -> MaxwellThreeDResourceRole {
        self.role
    }

    pub fn description(self) -> Result<SamplerDescription, MaxwellThreeDResourceError> {
        SamplerDescription::new(
            self.min_filter,
            self.mag_filter,
            self.mip_filter,
            self.address_modes,
            f32::from(self.lod_min_fixed) / 256.0,
            f32::from(self.lod_max_fixed) / 256.0,
            f32::from(self.max_anisotropy),
        )
        .map(|mut description| {
            description.border_color = self.border_color;
            description.compare = self.compare;
            description.lod_bias = f32::from(self.lod_bias_fixed) / 256.0;
            description
        })
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role: self.role })
    }
}

impl MaxwellThreeDPreservedImageLayout {
    #[must_use]
    pub const fn layout(self) -> ImageMemoryLayout {
        self.layout
    }
    #[must_use]
    pub const fn pte_kind(self) -> u8 {
        self.pte_kind
    }

    /// Returns whether guest bytes require Maxwell compression materialization.
    #[must_use]
    pub const fn requires_materialization(self) -> bool {
        // Attachment compression selectors control subsequent writes, not
        // whether a compressed PTE kind contains compressed color/depth data.
        // deko3d selects these kinds at image allocation time independently
        // of binding the attachment (see the format-selection reference above).
        self.compression_enabled
            || matches!(
                self.pte_kind,
                MAXWELL_C32_2CRA_KIND
                    | MAXWELL_C64_2CRA_KIND
                    | MAXWELL_C128_2CR_KIND
                    | MAXWELL_S8Z24_2CZ_KIND
                    | MAXWELL_Z24S8_2CZ_KIND
                    | MAXWELL_Z16_2Z_KIND
                    | MAXWELL_C32_MS4_2CBR_KIND
                    | MAXWELL_S8Z24_MS4_2CZ_KIND
                    | MAXWELL_Z24S8_MS4_2CZ_KIND
            )
    }

    /// Returns whether the guest bytes already have a direct canonical
    /// representation. Compressed color/depth kinds are not directly importable:
    /// disabling future compression does not decompress their existing bytes.
    /// Generic 16Bx2 mappings are already canonical.
    #[must_use]
    pub const fn has_direct_canonical_representation(self) -> bool {
        !self.requires_materialization() || self.pte_kind == MAXWELL_GENERIC_BLOCK_LINEAR_KIND
    }

    /// Whether the emulated host image can retain canonical texels. Single-sample
    /// color compression is represented by resident pixels, with demanded CPU
    /// visibility exporting those pixels through the ordinary block-linear
    /// swizzle. It does not require reproducing the hardware's compression tags.
    /// This is separate from importing previously unmaterialized guest storage.
    /// Depth/stencil packing and multisample storage retain their own boundaries.
    /// https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/util.cpp#L920-L1020
    #[must_use]
    pub const fn supports_canonical_texel_storage(self) -> bool {
        self.has_direct_canonical_representation()
            || matches!(
                self.pte_kind,
                MAXWELL_C32_2CRA_KIND | MAXWELL_C64_2CRA_KIND | MAXWELL_C128_2CR_KIND
            )
    }
}

/// Image interpretation published only after complete snapshot validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedImage {
    role: MaxwellThreeDResourceRole,
    access: MaxwellThreeDResourceAccess,
    description: ImageDescription,
    allocation_description: GpuAllocationDescription,
    view: ImageView,
    source: MaxwellResolvedRange,
    mappings: Arc<[MaxwellThreeDMappingReference]>,
    guest_layout: MaxwellThreeDPreservedImageLayout,
    guest_format: MaxwellThreeDGuestImageFormat,
}

impl MaxwellThreeDResolvedImage {
    #[must_use]
    pub const fn role(&self) -> MaxwellThreeDResourceRole {
        self.role
    }
    #[must_use]
    pub const fn access(&self) -> MaxwellThreeDResourceAccess {
        self.access
    }
    #[must_use]
    pub const fn description(&self) -> ImageDescription {
        self.description
    }
    #[must_use]
    pub const fn allocation_description(&self) -> GpuAllocationDescription {
        self.allocation_description
    }
    #[must_use]
    pub const fn view(&self) -> &ImageView {
        &self.view
    }
    #[must_use]
    pub const fn source(&self) -> &MaxwellResolvedRange {
        &self.source
    }
    #[must_use]
    pub fn mappings(&self) -> &[MaxwellThreeDMappingReference] {
        &self.mappings
    }

    #[must_use]
    pub const fn guest_layout(&self) -> MaxwellThreeDPreservedImageLayout {
        self.guest_layout
    }

    #[must_use]
    pub const fn guest_format(&self) -> MaxwellThreeDGuestImageFormat {
        self.guest_format
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDResolvedResource {
    Buffer(MaxwellThreeDResolvedBuffer),
    Image(MaxwellThreeDResolvedImage),
}

impl MaxwellThreeDResolvedResource {
    #[must_use]
    pub const fn role(&self) -> MaxwellThreeDResourceRole {
        match self {
            Self::Buffer(value) => value.role,
            Self::Image(value) => value.role,
        }
    }

    fn source(&self) -> &MaxwellResolvedRange {
        match self {
            Self::Buffer(value) => &value.source,
            Self::Image(value) => &value.source,
        }
    }
}

/// Immutable consumed groups; unchanged groups share their resource records.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedResources {
    resources: [Option<Arc<[Arc<MaxwellThreeDResolvedResource>]>>; 4],
    samplers: Option<Arc<[MaxwellThreeDResolvedSampler]>>,
}
impl MaxwellThreeDResolvedResources {
    pub fn resources(&self) -> impl Iterator<Item = &MaxwellThreeDResolvedResource> {
        self.resources
            .iter()
            .flatten()
            .flat_map(|group| group.iter().map(Arc::as_ref))
    }
    pub fn resource(
        &self,
        role: MaxwellThreeDResourceRole,
    ) -> Option<&MaxwellThreeDResolvedResource> {
        self.resource_arc(role).map(Arc::as_ref)
    }
    fn resource_arc(
        &self,
        role: MaxwellThreeDResourceRole,
    ) -> Option<&Arc<MaxwellThreeDResolvedResource>> {
        let group = self.resources[role.binding_group()].as_ref()?;
        let index = group.binary_search_by_key(&role, |r| r.role()).ok()?;
        Some(&group[index])
    }
    pub fn sampler(
        &self,
        role: MaxwellThreeDResourceRole,
    ) -> Option<&MaxwellThreeDResolvedSampler> {
        let samplers = self.samplers.as_ref()?;
        let index = samplers
            .binary_search_by_key(&role, |sampler| sampler.role())
            .ok()?;
        Some(&samplers[index])
    }
    pub(crate) fn same_resource(&self, other: &Self, role: MaxwellThreeDResourceRole) -> bool {
        match (self.resource_arc(role), other.resource_arc(role)) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
    pub(crate) fn same_group(&self, other: &Self, group: usize) -> bool {
        let resources = match (&self.resources[group], &other.resources[group]) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        resources
            && (group != 2
                || match (&self.samplers, &other.samplers) {
                    (None, None) => true,
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    _ => false,
                })
    }
    pub(crate) fn same_bindings(&self, other: &Self) -> bool {
        (0..4).all(|group| self.same_group(other, group))
    }
    pub fn validate_mappings(
        &self,
        address_space: &MaxwellGpuMemory,
    ) -> Result<(), MaxwellThreeDResourceError> {
        for resource in self.resources() {
            for segment in resource.source().segments() {
                if !address_space.retained_mapping_is_current(segment.mapping()) {
                    return Err(MaxwellThreeDResourceError::StaleMapping {
                        mapping: segment.mapping().id(),
                        generation: segment.mapping().generation(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_binding(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuMemory,
    role: MaxwellThreeDResourceRole,
    size: Option<MaxwellShaderBufferSize>,
    storage_alignment: u64,
    retained_backings: &mut MaxwellThreeDRetainedBackingCache,
    access: &mut nixe_gpu::MemoryAccess<'_>,
    sampled: bool,
) -> Result<RetainedBinding, MaxwellThreeDResourceError> {
    use MaxwellThreeDResourceRole as R;
    let sample_mode = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::SampleMode)
        .value()
        .and_then(|value| match value {
            MaxwellThreeDFixedFunctionValue::SampleMode(mode) => Some(*mode),
            _ => None,
        });
    let mut builder = ResourceBuilder::new(address_space, sample_mode, retained_backings, access);
    let bindings = state.shader_bindings();
    let texture_cbuf = |texture: MaxwellThreeDTextureReference, slot: u8| {
        bindings.groups()[usize::from(texture.group)].constant_buffers()[usize::from(slot)]
            .filter(|b| b.enabled())
            .and_then(|b| Some((b.address()?.get(), u64::from(b.size()?))))
    };
    match role {
        R::VertexStream(slot) => {
            let stream = &state.vertex_input().streams(slot as usize);
            if stream.format().value().is_some_and(|f| f.enabled()) {
                let address = stream
                    .address()
                    .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
                let limit = stream
                    .limit()
                    .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
                builder.buffer(role, address, inclusive_size(address, limit, role)?)?;
            }
        }
        R::IndexBuffer => {
            let index = state.vertex_input().index();
            let address = unresolved(index.address_upper().value(), index.address_lower().value())
                .ok_or(MaxwellThreeDResourceError::IncompleteState {
                    role: MaxwellThreeDResourceRole::IndexBuffer,
                })?;
            let role = MaxwellThreeDResourceRole::IndexBuffer;
            let limit = unresolved(index.limit_upper().value(), index.limit_lower().value());
            if limit.is_none()
                && (index.limit_upper().raw().is_some() || index.limit_lower().raw().is_some())
            {
                return Err(MaxwellThreeDResourceError::IncompleteState { role });
            }
            // DRAW_INDEX_BUFFER consumes [first, first+count). deko3d does not
            // program C/D limits, so resolve the demanded bytes instead of assuming
            // an allocation-sized limit or reading/scanning index values on CPU.
            // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/draw.mme
            let count = index
                .count()
                .value()
                .copied()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
            let size = {
                let first = index
                    .first()
                    .value()
                    .copied()
                    .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
                let element_size = index
                    .element_size()
                    .value()
                    .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?
                    .bytes();
                if count == 0 || !address.get().is_multiple_of(element_size) {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                let end = first
                    .checked_add(count)
                    .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
                let size = u64::from(end) * element_size;
                if let Some(limit) = limit
                    && size > inclusive_size(address, limit, role)?
                {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                size
            };
            builder.buffer(role, address, size)?;
        }
        R::ConstantBuffer { group, slot } => {
            builder.constant_buffer(
                &bindings,
                group,
                slot,
                match size {
                    Some(MaxwellShaderBufferSize::Proven(end)) => Some(end),
                    _ => None,
                },
            )?;
        }
        R::StorageBuffer {
            group,
            slot,
            byte_offset,
        } => {
            builder.storage_buffer(
                state,
                role,
                group,
                slot,
                byte_offset,
                size.ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
                storage_alignment,
            )?;
        }
        R::SampledImage { texture, dimension } => {
            let handle = builder.texture_handle(
                texture,
                texture_cbuf(texture, texture.constant_buffer_slot),
                texture
                    .secondary
                    .and_then(|(slot, _, _)| texture_cbuf(texture, slot)),
            )?;
            builder.descriptor_pool(
                R::TextureHeaders,
                bindings.texture_headers().address().map(|a| a.get()),
                bindings.texture_headers().maximum_index().value().copied(),
            )?;
            builder.sampled_texture(
                bindings.maxwell_texture_headers().value().copied(),
                bindings.sampler_binding().value().copied(),
                texture,
                handle,
                dimension,
                sampled,
            )?;
        }
        R::Sampler(texture) => {
            let handle = builder.texture_handle(
                texture,
                texture_cbuf(texture, texture.constant_buffer_slot),
                texture
                    .secondary
                    .and_then(|(slot, _, _)| texture_cbuf(texture, slot)),
            )?;
            builder.sampler_pool(&bindings)?;
            builder.sampler(bindings.sampler_binding().value().copied(), texture, handle)?;
        }
        R::ColorTarget(slot) => {
            let target = &state.render_targets().color(slot as usize);
            match target.readiness(true) {
                MaxwellThreeDAttachmentReadiness::Unprogrammed
                | MaxwellThreeDAttachmentReadiness::Disabled => (),
                MaxwellThreeDAttachmentReadiness::Ready => builder.color_target(slot, target)?,
                _ => return Err(MaxwellThreeDResourceError::IncompleteState { role }),
            }
        }
        R::DepthStencilTarget => builder.depth_target(&state.render_targets().depth_stencil())?,
        R::TextureHeaders | R::Samplers | R::BlitSource | R::BlitDestination => {
            return Err(MaxwellThreeDResourceError::IncompleteState { role });
        }
    }
    let resource = builder
        .resources
        .into_iter()
        .find(|resolved| resolved.role() == role)
        .map(Arc::new);
    Ok(RetainedBinding {
        address_space: address_space.id(),
        state: state.resource_state_identity(role),
        size,
        sampled,
        resource,
        sampler: builder
            .samplers
            .into_iter()
            .find(|sampler| sampler.role() == role),
        descriptor_reads: builder.descriptor_reads.into_boxed_slice(),
        mapping_generation: address_space.mapping_generation().get(),
    })
}

struct ResourceBuilder<'a, 'm> {
    address_space: &'a MaxwellGpuMemory,
    access: &'a mut nixe_gpu::MemoryAccess<'m>,
    retained_backings: &'a mut MaxwellThreeDRetainedBackingCache,
    sample_mode: Option<MaxwellThreeDSampleMode>,
    resources: Vec<MaxwellThreeDResolvedResource>,
    samplers: Vec<MaxwellThreeDResolvedSampler>,
    descriptor_reads: Vec<MaxwellThreeDDescriptorRead>,
    compute: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DescriptorTableKind {
    Image,
    Sampler,
}

#[derive(Debug)]
struct DescriptorTableEntry {
    bytes: [u8; 32],
    version: u64,
    read_epoch: u64,
}

#[derive(Debug, Default)]
struct DescriptorTable {
    base: Option<GpuVirtualAddress>,
    limit: u32,
    entries: Vec<Option<DescriptorTableEntry>>,
    epoch: u64,
    next_version: u64,
}

impl DescriptorTable {
    fn synchronize(&mut self, base: GpuVirtualAddress, limit: u32) {
        if self.base != Some(base) || self.limit != limit {
            self.base = Some(base);
            self.limit = limit;
            self.entries.clear();
        }
    }

    fn next_epoch(&mut self) {
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("descriptor epoch exhausted");
    }

    fn read(
        &mut self,
        role: MaxwellThreeDResourceRole,
        index: u32,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        batch: &mut MaxwellGpuReadBatch<'_>,
    ) -> Result<([u8; 32], u64), MaxwellThreeDResourceError> {
        if index > self.limit {
            return Err(MaxwellThreeDResourceError::DescriptorIndexOutOfRange { role, index });
        }
        let base = self
            .base
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let address = space
            .checked_add(base, u64::from(index) * MAXWELL_DESCRIPTOR_SIZE)
            .map_err(|_| MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let index = index as usize;
        if self.entries.len() <= index {
            self.entries
                .try_reserve(index + 1 - self.entries.len())
                .map_err(|_| MaxwellThreeDResourceError::ResourceExhausted)?;
            self.entries.resize_with(index + 1, || None);
        }
        let entry = &mut self.entries[index];
        if entry
            .as_ref()
            .is_none_or(|entry| entry.read_epoch != self.epoch)
        {
            let mut bytes = [0; 32];
            batch
                .read_into(address, &mut bytes, access)
                .map_err(|error| MaxwellThreeDResourceError::Resolution { role, error })?;
            if entry.as_ref().is_none_or(|entry| entry.bytes != bytes) {
                self.next_version = self
                    .next_version
                    .checked_add(1)
                    .expect("descriptor version exhausted");
                *entry = Some(DescriptorTableEntry {
                    bytes,
                    version: self.next_version,
                    read_epoch: self.epoch,
                });
            } else {
                entry.as_mut().unwrap().read_epoch = self.epoch;
            }
        }
        let entry = entry.as_ref().unwrap();
        Ok((entry.bytes, entry.version))
    }
}

#[derive(Debug)]
enum MaxwellThreeDDescriptorRead {
    Direct {
        role: MaxwellThreeDResourceRole,
        address: GpuVirtualAddress,
        size: u8,
        bytes: [u8; 32],
        pointer_source: Option<MaxwellResolvedRange>,
    },
    Table {
        role: MaxwellThreeDResourceRole,
        kind: DescriptorTableKind,
        index: u32,
        version: u64,
    },
}
impl MaxwellThreeDDescriptorRead {
    fn refresh(
        &self,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        tables: &mut [DescriptorTable; 2],
        batch: &mut MaxwellGpuReadBatch<'_>,
    ) -> Result<bool, MaxwellThreeDResourceError> {
        match self {
            Self::Direct {
                role,
                address,
                size,
                bytes,
                ..
            } => {
                let mut current = [0; 32];
                batch
                    .read_into(*address, &mut current[..usize::from(*size)], access)
                    .map_err(|error| MaxwellThreeDResourceError::Resolution {
                        role: *role,
                        error,
                    })?;
                Ok(current == *bytes)
            }
            Self::Table {
                role,
                kind,
                index,
                version,
            } => {
                let (_, current) =
                    tables[*kind as usize].read(*role, *index, space, access, batch)?;
                Ok(current == *version)
            }
        }
    }
}

#[derive(Debug)]
struct RetainedBinding {
    address_space: crate::MaxwellAddressSpaceId,
    state: MaxwellThreeDResourceStateIdentity,
    size: Option<MaxwellShaderBufferSize>,
    sampled: bool,
    resource: Option<Arc<MaxwellThreeDResolvedResource>>,
    sampler: Option<MaxwellThreeDResolvedSampler>,
    descriptor_reads: Box<[MaxwellThreeDDescriptorRead]>,
    mapping_generation: u64,
}

#[derive(Debug)]
struct RetainedBindingGroup {
    roles: Box<[MaxwellThreeDResourceRole]>,
    sizes: Box<[(MaxwellThreeDResourceRole, MaxwellShaderBufferSize)]>,
    bindings: Vec<RetainedBinding>,
    state: MaxwellThreeDResourceGroupIdentity,
    address_space: crate::MaxwellAddressSpaceId,
    mapping_generation: u64,
    has_descriptor_reads: bool,
    resources: Option<Arc<[Arc<MaxwellThreeDResolvedResource>]>>,
    samplers: Option<Arc<[MaxwellThreeDResolvedSampler]>>,
}

impl MaxwellThreeDResourceRole {
    const fn binding_group(self) -> usize {
        match self {
            Self::VertexStream(_) | Self::IndexBuffer => 0,
            Self::ConstantBuffer { .. } | Self::StorageBuffer { .. } => 1,
            Self::TextureHeaders
            | Self::Samplers
            | Self::SampledImage { .. }
            | Self::Sampler(_) => 2,
            Self::ColorTarget(_)
            | Self::DepthStencilTarget
            | Self::BlitSource
            | Self::BlitDestination => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ColorBlitKey {
    address_space: crate::MaxwellAddressSpaceId,
    surfaces: [crate::engines::twod::blit::BlitSurface; 2],
    destination_compression: bool,
    resolve: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SolidImageKey {
    address_space: crate::MaxwellAddressSpaceId,
    surface: crate::engines::twod::blit::BlitSurface,
    zeta: bool,
    compression: bool,
}

#[derive(Debug)]
struct ColorBlitCacheEntry {
    resources: Arc<MaxwellThreeDResolvedResources>,
    mapping_generation: Cell<u64>,
    last_used: Cell<u64>,
}

pub(crate) struct MaxwellThreeDResolvedResourceCache {
    storage_alignment: u64,
    blits: HashMap<ColorBlitKey, ColorBlitCacheEntry>,
    solid_images: HashMap<SolidImageKey, ColorBlitCacheEntry>,
    groups: [Option<RetainedBindingGroup>; 4],
    selected: MaxwellThreeDResolvedResources,
    retained_backings: MaxwellThreeDRetainedBackingCache,
    next_use: Cell<u64>,
}

impl Default for MaxwellThreeDResolvedResourceCache {
    fn default() -> Self {
        Self {
            storage_alignment: 1,
            blits: HashMap::default(),
            solid_images: HashMap::default(),
            groups: Default::default(),
            selected: MaxwellThreeDResolvedResources::default(),
            retained_backings: Default::default(),
            next_use: Cell::new(0),
        }
    }
}
impl MaxwellThreeDResolvedResourceCache {
    pub(crate) fn new(storage_alignment: u64) -> Self {
        Self {
            storage_alignment: storage_alignment.max(1),
            ..Self::default()
        }
    }

    pub(crate) fn resolve_solid_image(
        &mut self,
        request: &crate::engines::twod::MaxwellTwoDSolidOperation,
        address_space: &MaxwellGpuMemory,
        limit: usize,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Arc<MaxwellThreeDResolvedResources>, MaxwellThreeDResourceError> {
        self.retained_backings.set_view_limit(limit);
        let key = SolidImageKey {
            address_space: address_space.id(),
            surface: request.destination,
            zeta: request.zeta,
            compression: request.compression,
        };
        if let Some(entry) = self.solid_images.get(&key) {
            let generation = address_space.mapping_generation().get();
            if entry.mapping_generation.get() == generation
                || entry.resources.validate_mappings(address_space).is_ok()
            {
                entry.mapping_generation.set(generation);
                entry.last_used.set(self.take_use());
                return Ok(Arc::clone(&entry.resources));
            }
        }
        let surface = request.destination;
        let role = if request.zeta {
            MaxwellThreeDResourceRole::DepthStencilTarget
        } else {
            MaxwellThreeDResourceRole::BlitDestination
        };
        let (format, kind, guest_format) = if request.zeta {
            // COLOR_RENDER_TO_ZETA preserves the packed pixel value, while the
            // PTE kind identifies depth/stencil packing independently of 2D color format.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L578-L581
            let address = address_space.address(surface.address).map_err(|error| {
                MaxwellThreeDResourceError::Resolution {
                    role,
                    error: MaxwellGpuAccessError::Address(error),
                }
            })?;
            let range = address_space
                .resolve_range(address, 1, MemoryPermissions::WRITE)
                .map_err(|error| MaxwellThreeDResourceError::Resolution { role, error })?;
            let actual = range.segments()[0].mapping().kind();
            let depth_format = match actual {
                MAXWELL_S8Z24_2CZ_KIND => MaxwellThreeDDepthStencilFormat::Stencil8Z24,
                MAXWELL_Z24S8_2CZ_KIND => MaxwellThreeDDepthStencilFormat::Z24Stencil8,
                _ => {
                    return Err(MaxwellThreeDResourceError::UnsupportedKind {
                        role,
                        expected: MAXWELL_Z24S8_2CZ_KIND,
                        actual,
                    });
                }
            };
            (
                ImageFormat::Depth24UnormStencil8Uint,
                ImageKind::DepthStencil,
                MaxwellThreeDGuestImageFormat::DepthStencil(depth_format),
            )
        } else {
            (
                color_image_format(surface.format, role)?,
                ImageKind::Color,
                MaxwellThreeDGuestImageFormat::Color(MaxwellThreeDColorTargetFormat::Color(
                    surface.format,
                )),
            )
        };
        let mut builder =
            ResourceBuilder::new(address_space, None, &mut self.retained_backings, access);
        builder.sample_mode = Some(MaxwellThreeDSampleMode::Samples1x1);
        let description = image_description(MaxwellImageDescriptionRequest {
            dimension: ImageDimension::Two,
            extent: ImageExtent {
                width: surface.width,
                height: surface.height,
                depth: 1,
            },
            format,
            kind,
            layers: 1,
            role,
        })?;
        builder.image(
            role,
            MaxwellThreeDUnresolvedAddress::new(
                (surface.address >> 32) as u8,
                surface.address as u32,
            ),
            description,
            match surface.layout {
                crate::engines::twod::blit::BlitSurfaceLayout::Pitch(_) => {
                    MaxwellThreeDImageLayout::PitchLinear
                }
                crate::engines::twod::blit::BlitSurfaceLayout::BlockLinear(block_height_log2) => {
                    MaxwellThreeDImageLayout::BlockLinear {
                        block_height_log2,
                        block_depth_log2: 0,
                    }
                }
            },
            None,
            0,
            guest_format,
            request.compression,
            match surface.layout {
                crate::engines::twod::blit::BlitSurfaceLayout::Pitch(pitch) => Some(pitch),
                _ => None,
            },
        )?;
        let resources = Arc::new(builder.finish()?.0);
        if self.solid_images.len() >= limit && !self.solid_images.contains_key(&key) {
            let oldest = *self
                .solid_images
                .iter()
                .min_by_key(|(_, v)| v.last_used.get())
                .expect("nonzero cache limit")
                .0;
            self.solid_images.remove(&oldest);
        }
        self.solid_images.insert(
            key,
            ColorBlitCacheEntry {
                resources: Arc::clone(&resources),
                mapping_generation: Cell::new(address_space.mapping_generation().get()),
                last_used: Cell::new(self.take_use()),
            },
        );
        Ok(resources)
    }
    pub(crate) fn resolve_color_images(
        &mut self,
        request: &crate::engines::twod::MaxwellTwoDBlitOperation,
        address_space: &MaxwellGpuMemory,
        limit: usize,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Arc<MaxwellThreeDResolvedResources>, MaxwellThreeDResourceError> {
        self.retained_backings.set_view_limit(limit);
        let key = ColorBlitKey {
            address_space: address_space.id(),
            surfaces: request.images,
            destination_compression: request.destination_compression,
            resolve: matches!(
                request.kind,
                crate::engines::twod::blit::BlitKind::Downsample2x { .. }
            ),
        };
        if let Some(entry) = self.blits.get(&key) {
            let generation = address_space.mapping_generation().get();
            if entry.mapping_generation.get() == generation
                || entry.resources.validate_mappings(address_space).is_ok()
            {
                entry.mapping_generation.set(generation);
                entry.last_used.set(self.take_use());
                return Ok(Arc::clone(&entry.resources));
            }
        }
        let mut builder =
            ResourceBuilder::new(address_space, None, &mut self.retained_backings, access);
        for (index, surface) in request.images.iter().enumerate() {
            let role = if index == 0 {
                MaxwellThreeDResourceRole::BlitSource
            } else {
                MaxwellThreeDResourceRole::BlitDestination
            };
            // Point copies preserve the packed texels, including depth/stencil.
            // PTE kinds identify the resident representation even when 2D uses
            // a four-byte color format for the raw copy.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/twod/cl902d.h
            let address = address_space.address(surface.address).map_err(|error| {
                MaxwellThreeDResourceError::Resolution {
                    role,
                    error: MaxwellGpuAccessError::Address(error),
                }
            })?;
            let range = address_space
                .resolve_range(
                    address,
                    1,
                    if index == 0 {
                        MemoryPermissions::READ
                    } else {
                        MemoryPermissions::WRITE
                    },
                )
                .map_err(|error| MaxwellThreeDResourceError::Resolution { role, error })?;
            let pte_kind = range.segments()[0].mapping().kind();
            let (format, image_kind, guest_format) = match pte_kind {
                MAXWELL_S8Z24_2CZ_KIND | MAXWELL_Z24S8_2CZ_KIND => (
                    ImageFormat::Depth24UnormStencil8Uint,
                    ImageKind::DepthStencil,
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        if pte_kind == MAXWELL_S8Z24_2CZ_KIND {
                            MaxwellThreeDDepthStencilFormat::Stencil8Z24
                        } else {
                            MaxwellThreeDDepthStencilFormat::Z24Stencil8
                        },
                    ),
                ),
                _ => (
                    color_image_format(surface.format, role)?,
                    ImageKind::Color,
                    MaxwellThreeDGuestImageFormat::Color(MaxwellThreeDColorTargetFormat::Color(
                        surface.format,
                    )),
                ),
            };
            builder.sample_mode = Some(
                if index == 0 && key.resolve && pte_kind == MAXWELL_C32_MS4_2CBR_KIND {
                    MaxwellThreeDSampleMode::Samples2x2
                } else {
                    MaxwellThreeDSampleMode::Samples1x1
                },
            );
            let description = image_description(MaxwellImageDescriptionRequest {
                dimension: ImageDimension::Two,
                extent: ImageExtent {
                    width: surface.width,
                    height: surface.height,
                    depth: 1,
                },
                format,
                kind: image_kind,
                layers: 1,
                role,
            })?;
            builder.image(
                role,
                MaxwellThreeDUnresolvedAddress::new(
                    (surface.address >> 32) as u8,
                    surface.address as u32,
                ),
                description,
                match surface.layout {
                    crate::engines::twod::blit::BlitSurfaceLayout::Pitch(_) => {
                        MaxwellThreeDImageLayout::PitchLinear
                    }
                    crate::engines::twod::blit::BlitSurfaceLayout::BlockLinear(
                        block_height_log2,
                    ) => MaxwellThreeDImageLayout::BlockLinear {
                        block_height_log2,
                        block_depth_log2: 0,
                    },
                },
                None,
                0,
                guest_format,
                if index == 0 {
                    matches!(
                        surface.layout,
                        crate::engines::twod::blit::BlitSurfaceLayout::BlockLinear(_)
                    )
                } else {
                    request.destination_compression
                },
                match surface.layout {
                    crate::engines::twod::blit::BlitSurfaceLayout::Pitch(pitch) => Some(pitch),
                    _ => None,
                },
            )?;
        }
        let resources = Arc::new(builder.finish()?.0);
        if self.blits.len() >= limit && !self.blits.contains_key(&key) {
            let oldest = *self
                .blits
                .iter()
                .min_by_key(|(_, value)| value.last_used.get())
                .expect("nonzero resolved-resource cache limit")
                .0;
            self.blits.remove(&oldest);
        }
        self.blits.insert(
            key,
            ColorBlitCacheEntry {
                resources: Arc::clone(&resources),
                mapping_generation: Cell::new(address_space.mapping_generation().get()),
                last_used: Cell::new(self.take_use()),
            },
        );
        Ok(resources)
    }
    pub(crate) fn resolve(
        &mut self,
        state: &MaxwellThreeDState,
        address_space: &MaxwellGpuMemory,
        required_roles: &[MaxwellThreeDResourceRole],
        limit: usize,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<&MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
        self.resolve_with_buffer_sizes(state, address_space, required_roles, &[], limit, access)
    }
    pub(crate) fn resolve_with_buffer_sizes(
        &mut self,
        state: &MaxwellThreeDState,
        address_space: &MaxwellGpuMemory,
        required_roles: &[MaxwellThreeDResourceRole],
        buffer_sizes: &[(MaxwellThreeDResourceRole, MaxwellShaderBufferSize)],
        limit: usize,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<&MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
        debug_assert!(required_roles.is_sorted());
        debug_assert!(buffer_sizes.is_sorted_by_key(|(role, _)| *role));
        let mut batch = address_space.read_batch();
        self.retained_backings.set_view_limit(limit);
        for table in &mut self.retained_backings.descriptor_tables {
            table.next_epoch();
        }
        let mut remaining = required_roles;
        for group in 0..4 {
            let end = remaining.partition_point(|role| role.binding_group() == group);
            let (selected_roles, rest) = remaining.split_at(end);
            remaining = rest;
            if selected_roles.is_empty() {
                self.selected.resources[group] = None;
                if group == 2 {
                    self.selected.samplers = None;
                }
                continue;
            }
            let size_start = buffer_sizes.partition_point(|(role, _)| role.binding_group() < group);
            let size_end = buffer_sizes.partition_point(|(role, _)| role.binding_group() <= group);
            let selected_sizes = &buffer_sizes[size_start..size_end];
            let generation = address_space.mapping_generation().get();
            if let Some(retained) = &mut self.groups[group]
                && retained.address_space == address_space.id()
                && retained.roles.as_ref() == selected_roles
                && retained.sizes.as_ref() == selected_sizes
            {
                let mut current = true;
                if retained.mapping_generation != generation
                    || !retained.state.matches(state, group)
                {
                    for entry in &mut retained.bindings {
                        if !entry.state.matches(state)
                            || !Self::binding_memory_is_current(
                                entry,
                                address_space,
                                access,
                                &mut self.retained_backings.descriptor_tables,
                                &mut batch,
                            )?
                        {
                            current = false;
                            break;
                        }
                    }
                } else if retained.has_descriptor_reads {
                    for entry in &retained.bindings {
                        for read in &entry.descriptor_reads {
                            if !read.refresh(
                                address_space,
                                access,
                                &mut self.retained_backings.descriptor_tables,
                                &mut batch,
                            )? {
                                current = false;
                                break;
                            }
                        }
                        if !current {
                            break;
                        }
                    }
                }
                if current {
                    retained.state = state.resource_group_identity(group);
                    retained.mapping_generation = generation;
                    if self.selected.resources[group]
                        .as_ref()
                        .zip(retained.resources.as_ref())
                        .is_none_or(|(a, b)| !Arc::ptr_eq(a, b))
                    {
                        self.selected.resources[group] = retained.resources.clone();
                    }
                    if group == 2
                        && self
                            .selected
                            .samplers
                            .as_ref()
                            .zip(retained.samplers.as_ref())
                            .is_none_or(|(a, b)| !Arc::ptr_eq(a, b))
                    {
                        self.selected.samplers = retained.samplers.clone();
                    }
                    continue;
                }
            }
            // Current binding slots are owned by this channel and indexed in
            // selected-role order. A changed group reuses only matching slots;
            // native backing views have their own bounded cache below.
            let mut prior = self.groups[group].take();
            let mut changed = prior.as_ref().is_none_or(|old| {
                old.roles.as_ref() != selected_roles || old.sizes.as_ref() != selected_sizes
            });
            let mut previous = prior.as_mut().map(|old| {
                let roles = std::mem::take(&mut old.roles);
                let bindings = std::mem::take(&mut old.bindings);
                roles.into_vec().into_iter().zip(bindings).peekable()
            });
            let mut bindings = Vec::with_capacity(selected_roles.len());
            let mut has_descriptor_reads = false;
            for &role in selected_roles {
                let size = buffer_sizes
                    .binary_search_by_key(&role, |(r, _)| *r)
                    .ok()
                    .map(|index| buffer_sizes[index].1);
                let sampled = matches!(role, MaxwellThreeDResourceRole::SampledImage { texture, .. } if required_roles.binary_search(&MaxwellThreeDResourceRole::Sampler(texture)).is_ok());
                let old = if let Some(previous) = &mut previous {
                    while previous
                        .peek()
                        .is_some_and(|(old_role, _)| *old_role < role)
                    {
                        previous.next();
                    }
                    if previous
                        .peek()
                        .is_some_and(|(old_role, _)| *old_role == role)
                    {
                        Some(previous.next().unwrap().1)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let entry = if let Some(mut entry) = old
                    && entry.address_space == address_space.id()
                    && entry.size == size
                    && entry.sampled == sampled
                    && entry.state.matches(state)
                    && Self::binding_memory_is_current(
                        &mut entry,
                        address_space,
                        access,
                        &mut self.retained_backings.descriptor_tables,
                        &mut batch,
                    )? {
                    entry
                } else {
                    changed = true;
                    batch.release();
                    resolve_binding(
                        state,
                        address_space,
                        role,
                        size,
                        self.storage_alignment,
                        &mut self.retained_backings,
                        access,
                        sampled,
                    )?
                };
                has_descriptor_reads |= !entry.descriptor_reads.is_empty();
                bindings.push(entry);
            }
            let (resources, samplers) = if changed {
                let resources: Vec<_> = bindings
                    .iter()
                    .filter_map(|entry| entry.resource.as_ref().map(Arc::clone))
                    .collect();
                let samplers: Vec<_> = bindings.iter().filter_map(|entry| entry.sampler).collect();
                (
                    (!resources.is_empty()).then(|| resources.into()),
                    (!samplers.is_empty()).then(|| samplers.into()),
                )
            } else {
                let old = prior.as_mut().unwrap();
                (old.resources.take(), old.samplers.take())
            };
            self.groups[group] = Some(RetainedBindingGroup {
                roles: selected_roles.into(),
                sizes: selected_sizes.into(),
                bindings,
                state: state.resource_group_identity(group),
                address_space: address_space.id(),
                mapping_generation: generation,
                has_descriptor_reads,
                resources,
                samplers,
            });
            let retained = self.groups[group].as_ref().unwrap();
            if self.selected.resources[group]
                .as_ref()
                .zip(retained.resources.as_ref())
                .is_none_or(|(a, b)| !Arc::ptr_eq(a, b))
            {
                self.selected.resources[group] = retained.resources.clone();
            }
            if group == 2
                && self
                    .selected
                    .samplers
                    .as_ref()
                    .zip(retained.samplers.as_ref())
                    .is_none_or(|(a, b)| !Arc::ptr_eq(a, b))
            {
                self.selected.samplers = retained.samplers.clone();
            }
        }
        Ok(&self.selected)
    }
    fn binding_memory_is_current(
        entry: &mut RetainedBinding,
        address_space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
        tables: &mut [DescriptorTable; 2],
        batch: &mut MaxwellGpuReadBatch<'_>,
    ) -> Result<bool, MaxwellThreeDResourceError> {
        let generation = address_space.mapping_generation().get();
        if entry.mapping_generation != generation {
            if entry
                .resource
                .iter()
                .map(|resource| resource.source())
                .any(|source| {
                    source.segments().iter().any(|segment| {
                        !address_space.retained_mapping_is_current(segment.mapping())
                    })
                })
                || entry.descriptor_reads.iter().any(|read| {
                    let MaxwellThreeDDescriptorRead::Direct { pointer_source, .. } = read else {
                        return false;
                    };
                    pointer_source.as_ref().is_some_and(|source| {
                        source
                            .segments()
                            .iter()
                            .any(|s| !address_space.retained_mapping_is_current(s.mapping()))
                    })
                })
            {
                return Ok(false);
            }
            entry.mapping_generation = generation;
        }
        for read in &entry.descriptor_reads {
            if !read.refresh(address_space, access, tables, batch)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn take_use(&self) -> u64 {
        let next = self.next_use.get();
        self.next_use.set(
            next.checked_add(1)
                .expect("resolved-resource cache LRU sequence exhausted"),
        );
        next
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RetainedBackingKey {
    address_space: crate::MaxwellAddressSpaceId,
    offset: GpuVirtualAddress,
    size: u64,
    permissions: u8,
}

impl From<&MaxwellResolvedRange> for RetainedBackingKey {
    fn from(source: &MaxwellResolvedRange) -> Self {
        Self {
            address_space: source.address_space(),
            offset: source.offset(),
            size: source.size(),
            permissions: source.permissions().bits(),
        }
    }
}

#[derive(Debug)]
struct RetainedBackingCacheEntry {
    mapping_generation: Cell<Option<nixe_gpu::MappingGeneration>>,
    source: MaxwellResolvedRange,
    retained: RetainedResourceBacking,
}

// Cache consumed view descriptions independently of unrelated draw-state domains.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ResolvedViewDefinition {
    Buffer {
        address: MaxwellThreeDUnresolvedAddress,
        size: u64,
    },
    Sampled([u8; 32]),
    Attachment {
        address: MaxwellThreeDUnresolvedAddress,
        description: ImageDescription,
        layout: MaxwellThreeDImageLayout,
        array_pitch_dwords: Option<u32>,
        selected_layer: u16,
        guest_format: MaxwellThreeDGuestImageFormat,
        compression_enabled: bool,
        row_pitch_bytes: Option<u32>,
        sample_mode: Option<MaxwellThreeDSampleMode>,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ResolvedViewKey {
    address_space: crate::MaxwellAddressSpaceId,
    role: MaxwellThreeDResourceRole,
    definition: ResolvedViewDefinition,
}

#[derive(Debug)]
struct CachedResolvedView {
    key: ResolvedViewKey,
    resource: MaxwellThreeDResolvedResource,
    mapping_generation: u64,
    referenced: bool,
}

/// Owner-local mapping topology and resource-view cache. Descriptor bytes are
/// still read in submission order; views are keyed by their actual descriptions.
pub(crate) struct MaxwellThreeDRetainedBackingCache {
    entries: nixe_gpu::WorkingSet<RetainedBackingKey, RetainedBackingCacheEntry>,
    views: HashMap<ResolvedViewKey, usize>,
    view_entries: Vec<CachedResolvedView>,
    view_limit: usize,
    view_clock: usize,
    samplers: nixe_gpu::WorkingSet<[u8; 32], MaxwellThreeDResolvedSampler>,
    descriptor_tables: [DescriptorTable; 2],
    compute_descriptor_tables: [DescriptorTable; 2],
}

impl Default for MaxwellThreeDRetainedBackingCache {
    fn default() -> Self {
        Self {
            entries: nixe_gpu::WorkingSet::new(4096, 64 * 1024 * 1024),
            views: HashMap::default(),
            view_entries: Vec::new(),
            view_limit: 0,
            view_clock: 0,
            samplers: nixe_gpu::WorkingSet::new(4096, 1024 * 1024),
            descriptor_tables: Default::default(),
            compute_descriptor_tables: Default::default(),
        }
    }
}
impl std::fmt::Debug for MaxwellThreeDRetainedBackingCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedBackingCache")
            .field("mappings", &self.entries.usage())
            .field("views", &self.view_entries.len())
            .finish()
    }
}

impl MaxwellThreeDRetainedBackingCache {
    fn resolve_cached(
        &mut self,
        space: &MaxwellGpuMemory,
        address: GpuVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Option<MaxwellResolvedRange> {
        // Physical content changes do not invalidate translation. A changed VM
        // generation checks only the retained mapping leases; ordinary hits
        // need neither a VM traversal nor a page/alias scan.
        // https://github.com/yangjieshao/Ryujinx/blob/2505a1abcd5f31e4364bfb7f9313546f1840e5e1/src/Ryujinx.Graphics.Gpu/Memory/BufferCache.cs
        let key = RetainedBackingKey {
            address_space: space.id(),
            offset: address,
            size,
            permissions: permissions.bits(),
        };
        let entry = self.entries.get(&key)?;
        let generation = space.mapping_generation();
        if entry.mapping_generation.get() != Some(generation) {
            if entry
                .source
                .segments()
                .iter()
                .any(|s| !space.retained_mapping_is_current(s.mapping()))
            {
                return None;
            }
            entry.mapping_generation.set(Some(generation));
        }
        Some(entry.source.clone())
    }

    fn view(
        &mut self,
        key: &ResolvedViewKey,
        address_space: &MaxwellGpuMemory,
    ) -> Option<MaxwellThreeDResolvedResource> {
        let entry = &mut self.view_entries[*self.views.get(key)?];
        // Buffer interpretation is independent of its contents. Descriptor
        // readers and backend uploads observe the live canonical bytes; do not
        // arm another CPU-write dependency merely to recreate buffer metadata.
        let generation = address_space.mapping_generation().get();
        if entry.mapping_generation != generation {
            if entry
                .resource
                .source()
                .segments()
                .iter()
                .any(|segment| !address_space.retained_mapping_is_current(segment.mapping()))
            {
                return None;
            }
            entry.mapping_generation = generation;
        }
        entry.referenced = true;
        Some(entry.resource.clone())
    }

    fn set_view_limit(&mut self, limit: usize) {
        if self.view_limit == limit {
            return;
        }
        self.view_limit = limit;
        if self.view_entries.len() > limit {
            self.view_entries.truncate(limit);
            self.views.clear();
            self.views.extend(
                self.view_entries
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| (entry.key, index)),
            );
            self.view_clock = 0;
        }
    }

    fn remember_view(
        &mut self,
        key: ResolvedViewKey,
        resource: MaxwellThreeDResolvedResource,
        address_space: &MaxwellGpuMemory,
    ) {
        if self.view_limit == 0 {
            return;
        }
        let entry = CachedResolvedView {
            key,
            resource,
            mapping_generation: address_space.mapping_generation().get(),
            referenced: true,
        };
        if let Some(&index) = self.views.get(&key) {
            self.view_entries[index] = entry;
            return;
        }
        let index = if self.view_entries.len() < self.view_limit {
            let index = self.view_entries.len();
            self.view_entries.push(entry);
            index
        } else {
            // Clock replacement avoids a full HashMap scan for every transient
            // constant-buffer view. Hits only set a bit; misses amortize scans.
            while self.view_entries[self.view_clock].referenced {
                self.view_entries[self.view_clock].referenced = false;
                self.view_clock = (self.view_clock + 1) % self.view_entries.len();
            }
            let index = self.view_clock;
            self.views.remove(&self.view_entries[index].key);
            self.view_entries[index] = entry;
            self.view_clock = (index + 1) % self.view_entries.len();
            index
        };
        self.views.insert(key, index);
    }

    fn retain(
        &mut self,
        source: &MaxwellResolvedRange,
        role: MaxwellThreeDResourceRole,
    ) -> Result<RetainedResourceBacking, MaxwellThreeDResourceError> {
        let key = RetainedBackingKey::from(source);
        if let Some(entry) = self.entries.get(&key)
            && entry.source == *source
        {
            return Ok(entry.retained.clone());
        }
        let retained = retained_backing(source, role)?;
        let bytes = std::mem::size_of::<RetainedBackingCacheEntry>()
            + std::mem::size_of_val(source.segments())
            + retained.backing.mapping().metadata_bytes()
            + std::mem::size_of_val(retained.mappings.as_ref());
        self.entries.insert(
            key,
            RetainedBackingCacheEntry {
                mapping_generation: Cell::new(None),
                source: source.clone(),
                retained: retained.clone(),
            },
            bytes,
        );
        Ok(retained)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RetainedResourceBacking {
    backing: BackingView,
    allocation_description: GpuAllocationDescription,
    mappings: Arc<[MaxwellThreeDMappingReference]>,
}

struct SampledMipChain {
    layouts: Vec<(u64, ImageMemoryLayout)>,
    layer_stride: u64,
    layer_size: u64,
}

struct MaxwellImageDescriptionRequest {
    dimension: ImageDimension,
    extent: ImageExtent,
    format: ImageFormat,
    kind: ImageKind,
    layers: u16,
    role: MaxwellThreeDResourceRole,
}

impl<'a, 'm> ResourceBuilder<'a, 'm> {
    fn descriptor_read(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: GpuVirtualAddress,
        size: u8,
        pointer_source: Option<MaxwellResolvedRange>,
    ) -> Result<[u8; 32], MaxwellThreeDResourceError> {
        let mut bytes = [0; 32];
        self.address_space
            .read_into(address, &mut bytes[..usize::from(size)], self.access)
            .map_err(|error| MaxwellThreeDResourceError::Resolution { role, error })?;
        self.descriptor_reads
            .push(MaxwellThreeDDescriptorRead::Direct {
                role,
                address,
                size,
                bytes,
                pointer_source,
            });
        Ok(bytes)
    }

    fn reuse_view(&mut self, key: &ResolvedViewKey) -> Result<bool, MaxwellThreeDResourceError> {
        let Some(resource) = self.retained_backings.view(key, self.address_space) else {
            return Ok(false);
        };
        self.resources.push(resource);
        Ok(true)
    }

    fn remember_last_view(&mut self, key: ResolvedViewKey) {
        self.retained_backings.remember_view(
            key,
            self.resources.last().unwrap().clone(),
            self.address_space,
        );
    }

    fn new(
        address_space: &'a MaxwellGpuMemory,
        sample_mode: Option<MaxwellThreeDSampleMode>,
        retained_backings: &'a mut MaxwellThreeDRetainedBackingCache,
        access: &'a mut nixe_gpu::MemoryAccess<'m>,
    ) -> Self {
        Self {
            address_space,
            access,
            retained_backings,
            sample_mode,
            resources: Vec::new(),
            samplers: Vec::new(),
            descriptor_reads: Vec::new(),
            compute: false,
        }
    }

    /// Tegra 2D mip chains: each level has a reduced GOB height, and array
    /// layers align the complete chain rather than every level to the base block.
    /// https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/util.cpp
    fn sampled_mip_layouts(
        extent: ImageExtent,
        format: ImageFormat,
        levels: u8,
        block: [u8; 2],
        role: MaxwellThreeDResourceRole,
    ) -> Result<SampledMipChain, MaxwellThreeDResourceError> {
        let ImageExtent {
            width,
            height,
            depth,
        } = extent;
        let [block_height_log2, block_depth_log2] = block;
        let [bw, bh] = format.block_extent();
        let bytes = u64::from(format.plane_bytes_per_block(0).unwrap());
        let mut layouts = Vec::with_capacity(usize::from(levels));
        let mut offset = 0_u64;
        for level in 0..levels {
            let columns = (width >> level).max(1).div_ceil(bw);
            let rows = (height >> level).max(1).div_ceil(bh);
            let slices = (depth >> level).max(1);
            let mut gob_depth = block_depth_log2;
            let mut gob_height = block_height_log2;
            if levels > 1 {
                while gob_height > 0 && u64::from(rows) <= (8_u64 << (gob_height - 1)) {
                    gob_height -= 1;
                }
            }
            if levels > 1 {
                while gob_depth > 0 && u64::from(slices) <= (1u64 << (gob_depth - 1)) {
                    gob_depth -= 1;
                }
            }
            let padded_slices = align_up(u64::from(slices), 1u64 << gob_depth, role)?;
            let size = align_up(u64::from(columns) * bytes, 64, role)?
                .checked_mul(align_up(u64::from(rows), 8_u64 << gob_height, role)?)
                .and_then(|size| size.checked_mul(padded_slices))
                .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
            layouts.push((
                offset,
                ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                    block_width_log2: 0,
                    block_height_log2: gob_height,
                    block_depth_log2: gob_depth,
                    layer_stride: size,
                }),
            ));
            offset = offset
                .checked_add(size)
                .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        }
        // Match Tegra's layer alignment, including compressed block-height rounding.
        let mut gob_height = block_height_log2;
        let aligned_height = u64::from(height.div_ceil(bh)) * u64::from(bh);
        while gob_height > 0 && aligned_height <= (8_u64 << (gob_height - 1)) {
            gob_height -= 1;
        }
        let mut gob_depth = block_depth_log2;
        while gob_depth > 0 && u64::from(depth) <= (1u64 << (gob_depth - 1)) {
            gob_depth -= 1;
        }
        let stride = align_up(offset, 512_u64 << (gob_height + gob_depth), role)?;
        Ok(SampledMipChain {
            layouts,
            layer_stride: stride,
            layer_size: offset,
        })
    }

    fn retained_backing(
        &mut self,
        source: &MaxwellResolvedRange,
        role: MaxwellThreeDResourceRole,
    ) -> Result<RetainedResourceBacking, MaxwellThreeDResourceError> {
        self.retained_backings.retain(source, role)
    }

    fn buffer(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: MaxwellThreeDUnresolvedAddress,
        size: u64,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let key = ResolvedViewKey {
            address_space: self.address_space.id(),
            role,
            definition: ResolvedViewDefinition::Buffer { address, size },
        };
        if self.reuse_view(&key)? {
            return Ok(());
        }
        let source = self.resolve(address, size, MemoryPermissions::READ, role)?;
        let retained = self.retained_backing(&source, role)?;
        let allocation_description = retained.allocation_description;
        let description = BufferDescription::new(size)
            .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        let id = BufferId::new(resource_id(self.resources.len())?);
        let view = BufferView::new(id, description, 0, retained.backing)
            .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        self.resources.push(MaxwellThreeDResolvedResource::Buffer(
            MaxwellThreeDResolvedBuffer {
                role,
                access: MaxwellThreeDResourceAccess::Read,
                description,
                allocation_description,
                view,
                source,
                mappings: retained.mappings,
            },
        ));
        self.remember_last_view(key);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn storage_buffer(
        &mut self,
        state: &MaxwellThreeDState,
        role: MaxwellThreeDResourceRole,
        group: u8,
        slot: u8,
        byte_offset: u32,
        size: MaxwellShaderBufferSize,
        alignment: u64,
    ) -> Result<(), MaxwellThreeDResourceError> {
        // NVN storage descriptors contain a u64 GPU VA and a u32 byte size.
        // Descriptor contents are draw state, not shader translation identity.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/buffer_cache/buffer_cache.h (StorageBufferBinding)
        let binding = state
            .shader_bindings()
            .groups()
            .get(usize::from(group))
            .and_then(|group| group.constant_buffers().get(usize::from(slot)))
            .copied()
            .flatten()
            .filter(|binding| binding.enabled())
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let address = binding
            .address()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?
            .get();
        let cb_size = binding
            .size()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let descriptor_size = if size == MaxwellShaderBufferSize::Declared {
            16
        } else {
            8
        };
        if u64::from(byte_offset) + descriptor_size > u64::from(cb_size) {
            return Err(MaxwellThreeDResourceError::InvalidStorageBuffer {
                role,
                field: "global pointer exceeds constant buffer",
                value: u64::from(byte_offset),
            });
        }
        let pointer_address = address
            .checked_add(u64::from(byte_offset))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        if pointer_address >= 1 << 40 {
            return Err(MaxwellThreeDResourceError::AddressOutOfRange {
                role,
                address: pointer_address,
            });
        }
        let source = self.resolve(
            MaxwellThreeDUnresolvedAddress::new(
                (pointer_address >> 32) as u8,
                pointer_address as u32,
            ),
            descriptor_size,
            MemoryPermissions::READ,
            role,
        )?;
        let pointer_address = source.offset();
        let bytes =
            self.descriptor_read(role, pointer_address, descriptor_size as u8, Some(source))?;
        let target = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let target_size = match size {
            MaxwellShaderBufferSize::Proven(size) => size,
            MaxwellShaderBufferSize::Declared => {
                u64::from(u32::from_le_bytes(bytes[8..12].try_into().unwrap()))
            }
        };
        if target & 3 != 0 || target_size == 0 {
            return Err(MaxwellThreeDResourceError::InvalidStorageBuffer {
                role,
                field: "global buffer alignment/size",
                value: target,
            });
        }
        if target >= 1 << 40 {
            return Err(MaxwellThreeDResourceError::AddressOutOfRange {
                role,
                address: target,
            });
        }
        let aligned = target / alignment * alignment;
        self.buffer(
            role,
            MaxwellThreeDUnresolvedAddress::new((aligned >> 32) as u8, aligned as u32),
            target_size + target - aligned,
        )
    }

    fn constant_buffer(
        &mut self,
        bindings: &super::MaxwellThreeDShaderBindingState,
        group: u8,
        slot: u8,
        end: Option<u64>,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let role = MaxwellThreeDResourceRole::ConstantBuffer { group, slot };
        let Some(binding) = bindings.groups()[group as usize].constant_buffers()[slot as usize]
            .filter(|b| b.enabled())
        else {
            return Ok(());
        };
        let address = binding
            .address()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let size = u64::from(
            binding
                .size()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
        );
        self.buffer(role, address, end.map_or(size, |end| size.min(end)))
    }
    fn sampler_pool(
        &mut self,
        bindings: &super::MaxwellThreeDShaderBindingState,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let maximum = if bindings.sampler_binding().value()
            == Some(&MaxwellThreeDSamplerBindingMode::ViaTextureHeader)
        {
            bindings.texture_headers().maximum_index().value().copied()
        } else {
            bindings.samplers().maximum_index().value().copied()
        };
        self.descriptor_pool(
            MaxwellThreeDResourceRole::Samplers,
            bindings.samplers().address().map(|a| a.get()),
            maximum,
        )
    }

    fn descriptor_pool(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: Option<u64>,
        maximum_index: Option<u32>,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let programmed = address.is_some() || maximum_index.is_some();
        if !programmed {
            return Ok(());
        }
        let address = address.ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let maximum = maximum_index.ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let base = self
            .address_space
            .address(address)
            .map_err(|_| MaxwellThreeDResourceError::AddressOutOfRange { role, address })?;
        let kind = match role {
            MaxwellThreeDResourceRole::TextureHeaders => DescriptorTableKind::Image,
            MaxwellThreeDResourceRole::Samplers => DescriptorTableKind::Sampler,
            _ => unreachable!("only TIC and TSC pools contain table descriptors"),
        };
        let tables = if self.compute {
            &mut self.retained_backings.compute_descriptor_tables
        } else {
            &mut self.retained_backings.descriptor_tables
        };
        // Synchronize table state; only consumed descriptors require mapped RAM.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/descriptor_table.h
        tables[kind as usize].synchronize(base, maximum);
        Ok(())
    }

    fn descriptor_bytes(
        &mut self,
        role: MaxwellThreeDResourceRole,
        index: u32,
    ) -> Result<[u8; 32], MaxwellThreeDResourceError> {
        let kind = match role {
            MaxwellThreeDResourceRole::TextureHeaders => DescriptorTableKind::Image,
            MaxwellThreeDResourceRole::Samplers => DescriptorTableKind::Sampler,
            _ => unreachable!("only TIC and TSC pools contain table descriptors"),
        };
        let tables = if self.compute {
            &mut self.retained_backings.compute_descriptor_tables
        } else {
            &mut self.retained_backings.descriptor_tables
        };
        let (bytes, version) = tables[kind as usize].read(
            role,
            index,
            self.address_space,
            self.access,
            &mut self.address_space.read_batch(),
        )?;
        self.descriptor_reads
            .push(MaxwellThreeDDescriptorRead::Table {
                role,
                kind,
                index,
                version,
            });
        Ok(bytes)
    }

    fn sampled_texture(
        &mut self,
        maxwell_headers: Option<bool>,
        sampler_mode: Option<MaxwellThreeDSamplerBindingMode>,
        texture_reference: MaxwellThreeDTextureReference,
        raw_handle: u32,
        dimension: MaxwellThreeDTextureDimension,
        needs_sampler: bool,
    ) -> Result<(), MaxwellThreeDResourceError> {
        // An unprogrammed selector retains the Maxwell class reset mode. Only
        // an explicit false selects the legacy texture-header interpretation;
        // the TIC decoder below independently requires a Maxwell v3 header.
        // The public class definition confirms this is a one-bit selector:
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1210-L1213
        if !uses_maxwell_texture_headers(maxwell_headers.as_ref()) {
            return Err(MaxwellThreeDResourceError::UnsupportedTextureBindingMode {
                descriptor: raw_handle,
            });
        }
        // Maxwell resource handles use TIC in bits 0..20 and TSC in bits
        // 20..32 when the tables are independent. Via-header mode uses the
        // same raw index for both tables. Public corroborating definitions:
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/include/deko3d.h#L711-L724
        // https://source.hodakov.me/hdkv/yuzu/src/commit/55bf3dbf5ddaa3f7c1c3efade5553b07499fe289/src/video_core/textures/texture.h#L147-L165
        let image_index = if !needs_sampler {
            // Texel fetch consumes only the TIC portion, never TSC selection/state.
            raw_handle & 0x000f_ffff
        } else {
            match sampler_mode {
                Some(mode) => texture_descriptor_pair(mode, raw_handle).0,
                None => {
                    return Err(MaxwellThreeDResourceError::IncompleteState {
                        role: MaxwellThreeDResourceRole::Sampler(texture_reference),
                    });
                }
            }
        };
        let tic = self.descriptor_bytes(MaxwellThreeDResourceRole::TextureHeaders, image_index)?;
        self.sampled_image(texture_reference, dimension, image_index, tic)?;
        Ok(())
    }
    fn sampler(
        &mut self,
        sampler_mode: Option<MaxwellThreeDSamplerBindingMode>,
        texture_reference: MaxwellThreeDTextureReference,
        raw: u32,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let mode = sampler_mode.ok_or(MaxwellThreeDResourceError::IncompleteState {
            role: MaxwellThreeDResourceRole::Sampler(texture_reference),
        })?;
        let (_, sampler_index) = texture_descriptor_pair(mode, raw);
        let tsc = self.descriptor_bytes(MaxwellThreeDResourceRole::Samplers, sampler_index)?;
        let mut sampler = if let Some(sampler) = self.retained_backings.samplers.get(&tsc) {
            *sampler
        } else {
            let sampler = decode_sampler(texture_reference, sampler_index, tsc)?;
            self.retained_backings.samplers.insert(
                tsc,
                sampler,
                std::mem::size_of_val(&sampler) + 32,
            );
            sampler
        };
        sampler.role = MaxwellThreeDResourceRole::Sampler(texture_reference);
        self.samplers.push(sampler);
        Ok(())
    }

    fn texture_handle(
        &mut self,
        texture_reference: MaxwellThreeDTextureReference,
        primary: Option<(u64, u64)>,
        secondary: Option<(u64, u64)>,
    ) -> Result<u32, MaxwellThreeDResourceError> {
        let primary = self.texture_handle_word(
            texture_reference,
            texture_reference.constant_buffer_slot,
            texture_reference.byte_offset,
            primary,
        )? << texture_reference.shift_left;
        let secondary = if let Some((slot, offset, shift)) = texture_reference.secondary {
            self.texture_handle_word(texture_reference, slot, offset, secondary)? << shift
        } else {
            0
        };
        Ok(primary | secondary)
    }

    fn texture_handle_word(
        &mut self,
        texture_reference: MaxwellThreeDTextureReference,
        slot: u8,
        byte_offset: u32,
        binding: Option<(u64, u64)>,
    ) -> Result<u32, MaxwellThreeDResourceError> {
        let role = MaxwellThreeDResourceRole::ConstantBuffer {
            group: texture_reference.group,
            slot,
        };
        let (base, size) = binding.ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        if u64::from(byte_offset) + 4 > size {
            return Err(MaxwellThreeDResourceError::TextureHandleOutOfRange {
                texture_reference,
                constant_buffer_size: size,
            });
        }
        let address = base
            .checked_add(u64::from(byte_offset))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let source = self.resolve(
            MaxwellThreeDUnresolvedAddress::new((address >> 32) as u8, address as u32),
            4,
            MemoryPermissions::READ,
            role,
        )?;
        let bytes = self.descriptor_read(role, source.offset(), 4, Some(source))?;
        Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
    }

    fn sampled_image(
        &mut self,
        texture_reference: MaxwellThreeDTextureReference,
        required_dimension: MaxwellThreeDTextureDimension,
        descriptor_index: u32,
        bytes: [u8; 32],
    ) -> Result<(), MaxwellThreeDResourceError> {
        // Maxwell 2 TIC field placement follows deko3d's pinned public
        // descriptor definition, itself cross-referenced to envytools:
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_image_control_block.h
        let words = descriptor_words(bytes);
        let role = MaxwellThreeDResourceRole::SampledImage {
            texture: texture_reference,
            dimension: required_dimension,
        };
        let key = ResolvedViewKey {
            address_space: self.address_space.id(),
            role,
            definition: ResolvedViewDefinition::Sampled(bytes),
        };
        if self.reuse_view(&key)? {
            return Ok(());
        }
        let format_word = words[0];
        let image_format = (format_word & 0x7f) as u8;
        let components = [7, 10, 13, 16].map(|shift| ((format_word >> shift) & 7) as u8);
        let swizzle = [19, 22, 25, 28].map(|shift| ((format_word >> shift) & 7) as u8);
        let srgb = words[4] & (1 << 22) != 0;
        let format = decode_sampled_texture_format(
            image_format,
            components,
            swizzle,
            srgb,
            format_word >> 31,
        )
        .ok_or(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
            descriptor: descriptor_index,
            field: "format/component/swizzle",
            value: format_word,
        })?;
        let sample_swizzle =
            if !matches!(
                format,
                ImageFormat::Bgra8Unorm
                    | ImageFormat::Bgra8Srgb
                    | ImageFormat::Rgb565Unorm
                    | ImageFormat::Depth24UnormStencil8Uint
                    | ImageFormat::Depth32Float
            ) && !(matches!(format, ImageFormat::R8Unorm | ImageFormat::Depth16Unorm)
                && swizzle == [2, 0, 0, 7])
            {
                Swizzle(swizzle.map(|component| match component {
                    0 => ComponentSwizzle::Zero,
                    2 => ComponentSwizzle::Red,
                    3 => ComponentSwizzle::Green,
                    4 => ComponentSwizzle::Blue,
                    5 => ComponentSwizzle::Alpha,
                    7 => ComponentSwizzle::One,
                    _ => unreachable!("validated TIC component mapping"),
                }))
            } else {
                // Existing format tuples include their native component mapping.
                Swizzle::IDENTITY
            };
        let header_version = ((words[2] >> 21) & 7) as u8;
        let texture_type = ((words[4] >> 23) & 0xf) as u8;
        let width = (words[4] & 0xffff) + 1;
        let height = (words[5] & 0xffff) + 1;
        let depth = ((words[5] >> 16) & 0x3fff) + 1;
        let block_width_log2 = (words[3] & 7) as u8;
        let block_height_log2 = ((words[3] >> 3) & 7) as u8;
        let block_depth_log2 = ((words[3] >> 6) & 7) as u8;
        let mip_max = ((words[3] >> 28) & 0xf) as u8;
        let view_mip_min = (words[7] & 0xf) as u8;
        let view_mip_max = ((words[7] >> 4) & 0xf) as u8;
        let msaa = ((words[7] >> 8) & 0xf) as u8;
        let sparse = words[5] & (1 << 30) != 0;
        let normalized = words[5] & (1 << 31) != 0;
        let layer_base = ((words[4] >> 16) & 7)
            | (((words[2] >> 16) & 0x1f) << 3)
            | (((words[2] >> 29) & 7) << 8);
        let (required_texture_type, layers) = match required_dimension {
            MaxwellThreeDTextureDimension::Two => (1, 1_u16),
            MaxwellThreeDTextureDimension::Three => (2, 1),
            // A single TIC cubemap has depth one and six independent faces.
            // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/texture_cache/image_info.cpp
            MaxwellThreeDTextureDimension::Cube => (3, 6),
            MaxwellThreeDTextureDimension::CubeArray => (
                8,
                u16::try_from(depth * 6).map_err(|_| {
                    MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                        descriptor: descriptor_index,
                        field: "cube-array face count",
                        value: depth,
                    }
                })?,
            ),
            MaxwellThreeDTextureDimension::TwoArray => (
                5,
                u16::try_from(depth).map_err(|_| {
                    MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                        descriptor: descriptor_index,
                        field: "2D-array layer count",
                        value: depth,
                    }
                })?,
            ),
        };
        let pitch_linear = header_version == 2;
        let cube = matches!(
            required_dimension,
            MaxwellThreeDTextureDimension::Cube | MaxwellThreeDTextureDimension::CubeArray
        );
        let texture_type_matches = texture_type == required_texture_type
            || (texture_type == 7
                && matches!(
                    required_dimension,
                    MaxwellThreeDTextureDimension::Two | MaxwellThreeDTextureDimension::TwoArray
                ))
            || (required_dimension == MaxwellThreeDTextureDimension::CubeArray
                && texture_type == 3)
            || (required_dimension == MaxwellThreeDTextureDimension::TwoArray && texture_type == 1);
        // TEX's array coordinate does not require multiple storage layers. A
        // non-array TIC can be sampled through a one-layer array view, retaining
        // the instruction's coordinate layout and host layer clamping.
        // Eden likewise preserves TEX dimensionality when it differs from TIC:
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/ir_opt/texture_pass.cpp
        if !matches!(header_version, 2 | 3)
            || !texture_type_matches
            || (cube && (pitch_linear || width != height))
            || (pitch_linear && depth != 1)
            || (!matches!(texture_type, 2 | 5 | 8) && depth != 1)
            || (required_dimension == MaxwellThreeDTextureDimension::Two && depth != 1)
            || (!pitch_linear
                && (block_width_log2 != 0 || block_depth_log2 > 5 || block_height_log2 > 5))
            || (pitch_linear && mip_max != 0)
            || view_mip_min != 0
            || view_mip_max != mip_max
            || msaa != 0
            || sparse
            || !normalized
            || layer_base != 0
        {
            log::debug!(
                "unsupported Maxwell TIC shape: descriptor={descriptor_index} dimension={required_dimension:?} words={words:08x?}"
            );
            return Err(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                descriptor: descriptor_index,
                field: "sampled texture shape",
                value: words[2] ^ words[3] ^ words[4] ^ words[5] ^ words[7],
            });
        }
        // Width spacing changes Tegra GOB row alignment; the packed mip
        // layout below currently represents spacing zero only.
        if !pitch_linear && (words[3] >> 10) & 7 != 0 {
            return Err(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                descriptor: descriptor_index,
                field: "tile-width spacing",
                value: words[3],
            });
        }
        let description = ImageDescription::new(
            if cube {
                ImageDimension::Cube
            } else if required_dimension == MaxwellThreeDTextureDimension::Three {
                ImageDimension::Three
            } else {
                ImageDimension::Two
            },
            ImageExtent {
                width,
                height,
                depth: if required_dimension == MaxwellThreeDTextureDimension::Three {
                    depth
                } else {
                    1
                },
            },
            format,
            if format.is_depth_stencil() {
                ImageKind::DepthStencil
            } else {
                ImageKind::Color
            },
            mip_max + 1,
            layers,
            SampleCount::One,
        )
        .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role })?;
        let address = u64::from(words[1]) | (u64::from(words[2] & 0xffff) << 32);
        let [block_width, block_height] = format.block_extent();
        let bytes_per_block = format
            .plane_bytes_per_block(0)
            .expect("decoded sampled image has one plane");
        let row_bytes = u64::from(width.div_ceil(block_width)) * u64::from(bytes_per_block);
        let SampledMipChain {
            layouts: mip_layouts,
            layer_stride,
            layer_size,
        } = if pitch_linear {
            // Pitch occupies bits 5:20 of the byte stride; the low five bits
            // are implicitly zero. Pitched storage has a single 2D layer.
            let row_pitch = u64::from(words[3] & 0xffff) << 5;
            if row_pitch < row_bytes {
                return Err(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                    descriptor: descriptor_index,
                    field: "pitch smaller than texel row",
                    value: row_pitch as u32,
                });
            }
            let layer_stride = row_pitch
                .checked_mul(u64::from(height.div_ceil(block_height)))
                .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
            SampledMipChain {
                layouts: vec![(
                    0,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch,
                        layer_stride,
                    },
                )],
                layer_stride,
                layer_size: layer_stride,
            }
        } else {
            Self::sampled_mip_layouts(
                description.extent(),
                format,
                mip_max + 1,
                [block_height_log2, block_depth_log2],
                role,
            )?
        };
        let layout = match mip_layouts[0].1 {
            ImageMemoryLayout::BlockLinear(mut blocks) => {
                blocks.layer_stride = layer_stride;
                ImageMemoryLayout::BlockLinear(blocks)
            }
            layout => layout,
        };
        let unresolved = MaxwellThreeDUnresolvedAddress::new((address >> 32) as u8, address as u32);
        let size = if layers == 1 {
            layer_size
        } else {
            layer_stride
                .checked_mul(u64::from(layers))
                .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?
        };
        let source = self.resolve(unresolved, size, MemoryPermissions::READ, role)?;
        let actual_kind = source
            .segments()
            .first()
            .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?
            .mapping()
            .kind();
        // TIC and render-target descriptors use different format encodings,
        // but refer to the same compressed allocation. Only texel formats
        // (not BC blocks) may use these color-compression families. Lowering
        // must supply an existing resident image; these bytes cannot be uploaded.
        let compressed_kind = match format {
            ImageFormat::Depth16Unorm => Some(MAXWELL_Z16_2Z_KIND),
            ImageFormat::Depth24UnormStencil8Uint => Some(MAXWELL_Z24S8_2CZ_KIND),
            _ => (!format.is_depth_stencil() && block_width == 1 && block_height == 1)
                .then(|| single_sample_compressed_color_kind(u64::from(bytes_per_block)))
                .flatten(),
        };
        let expected_kind = if pitch_linear {
            MAXWELL_PITCH_KIND
        } else {
            MAXWELL_GENERIC_BLOCK_LINEAR_KIND
        };
        if source.segments().iter().any(|segment| {
            segment.mapping().kind() != actual_kind
                || if pitch_linear {
                    !matches!(
                        actual_kind,
                        MAXWELL_PITCH_KIND | MAXWELL_PITCH_NO_SWIZZLE_KIND
                    )
                } else {
                    actual_kind != MAXWELL_GENERIC_BLOCK_LINEAR_KIND
                        && Some(actual_kind) != compressed_kind
                }
        }) {
            return Err(MaxwellThreeDResourceError::UnsupportedKind {
                role,
                expected: expected_kind,
                actual: actual_kind,
            });
        }
        let retained = self.retained_backing(&source, role)?;
        let allocation_description = retained.allocation_description;
        let id = ImageId::new(resource_id(self.resources.len())?);
        let mut image_bindings = Vec::with_capacity(mip_layouts.len() * usize::from(layers));
        if mip_layouts.len() == 1 {
            image_bindings.push((
                ImageSubresourceRange {
                    plane: 0,
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: layers,
                },
                layout,
                retained.backing.clone(),
            ));
        } else {
            // Array layers contain whole mip chains. Bind each level/layer's
            // actual bytes separately so interleaved levels never overlap.
            for layer in 0..layers {
                for (level, &(mip_offset, mip_layout)) in mip_layouts.iter().enumerate() {
                    let offset = u64::from(layer) * layer_stride + mip_offset;
                    let ImageMemoryLayout::BlockLinear(blocks) = mip_layout else {
                        unreachable!()
                    };
                    let range = retained
                        .backing
                        .mapping()
                        .slice(offset, blocks.layer_stride)
                        .map_err(|error| MaxwellThreeDResourceError::Memory(Arc::new(error)))?;
                    let backing = BackingView::new(
                        retained.backing.allocation(),
                        allocation_description,
                        retained
                            .backing
                            .allocation_offset()
                            .checked_add(offset)
                            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?,
                        range,
                    )
                    .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
                    image_bindings.push((
                        ImageSubresourceRange {
                            plane: 0,
                            mip_level: level as u8,
                            base_layer: layer,
                            layer_count: 1,
                        },
                        mip_layout,
                        backing,
                    ));
                }
            }
        }
        let view = ImageView::new(id, description, sample_swizzle, image_bindings)
            .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        self.resources.push(MaxwellThreeDResolvedResource::Image(
            MaxwellThreeDResolvedImage {
                role,
                access: MaxwellThreeDResourceAccess::Read,
                description,
                allocation_description,
                view,
                source,
                mappings: retained.mappings,
                guest_layout: MaxwellThreeDPreservedImageLayout {
                    layout,
                    pte_kind: actual_kind,
                    // A TIC cannot establish whether previous render-target
                    // writes used compression. Require the resident representation
                    // for compressed color kinds instead of importing opaque bytes.
                    compression_enabled: !pitch_linear
                        && actual_kind != MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                },
                guest_format: if format == ImageFormat::Depth24UnormStencil8Uint {
                    // TIC Z24S8 and SET_ZT_FORMAT describe the same packed texels.
                    // Keep the storage encoding independent of texture selectors.
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        MaxwellThreeDDepthStencilFormat::Z24Stencil8,
                    )
                } else if format == ImageFormat::Depth16Unorm {
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        MaxwellThreeDDepthStencilFormat::Z16,
                    )
                } else if format == ImageFormat::Depth32Float {
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        MaxwellThreeDDepthStencilFormat::ZFloat32,
                    )
                } else {
                    MaxwellThreeDGuestImageFormat::Texture(format_word)
                },
            },
        ));
        self.remember_last_view(key);
        Ok(())
    }

    fn color_target(
        &mut self,
        index: u8,
        target: &MaxwellThreeDColorTargetState,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let role = MaxwellThreeDResourceRole::ColorTarget(index);
        let address = unresolved(
            target.address_upper().value(),
            target.address_lower().value(),
        )
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let width = *target
            .width()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let height = *target
            .height()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let third = *target
            .third_dimension()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let kind = *target
            .kind()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let guest_format = target
            .format()
            .value()
            .copied()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let format = color_image_format(guest_format.raw(), role)?;
        // GM20B color surface formats explicitly distinguish UNORM and sRGB.
        // deko3d programs RGBA8Srgb/BGRA8Srgb directly without SET_SRGB_WRITE;
        // the latter's zero reset must not strip the attachment's transfer function.
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/format_traits.inc#L46-L47
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_3d_base.cpp#L50-L55
        let (dimension, depth, layers, selected_layer) = match kind {
            MaxwellThreeDImageKind::Array => {
                let layers = u16::try_from(third)
                    .map_err(|_| MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
                let layer = *target
                    .layer()
                    .value()
                    .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
                if layer >= layers {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                (ImageDimension::Two, 1, layers, layer)
            }
            MaxwellThreeDImageKind::ThreeDimensional => (ImageDimension::Three, third, 1, 0),
        };
        let description = image_description(MaxwellImageDescriptionRequest {
            dimension,
            extent: ImageExtent {
                width,
                height,
                depth,
            },
            format,
            kind: ImageKind::Color,
            layers,
            role,
        })?;
        self.image(
            role,
            address,
            description,
            *target
                .layout()
                .value()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
            target.array_pitch().value().copied(),
            selected_layer,
            MaxwellThreeDGuestImageFormat::Color(guest_format),
            target.compression().value()
                == Some(&super::MaxwellThreeDColorCompressionMode::Enabled),
            None,
        )
    }

    fn depth_target(
        &mut self,
        target: &MaxwellThreeDDepthStencilTargetState,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let role = MaxwellThreeDResourceRole::DepthStencilTarget;
        if target.sparse().value().is_some_and(|value| value & 1 != 0) {
            return Err(MaxwellThreeDResourceError::UnsupportedSparseDepth {
                value: *target.sparse().value().unwrap(),
            });
        }
        let address = unresolved(
            target.address_upper().value(),
            target.address_lower().value(),
        )
        .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let guest_format = *target
            .format()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let format = match guest_format {
            MaxwellThreeDDepthStencilFormat::Z16 => ImageFormat::Depth16Unorm,
            MaxwellThreeDDepthStencilFormat::Z24Stencil8
            | MaxwellThreeDDepthStencilFormat::Stencil8Z24 => ImageFormat::Depth24UnormStencil8Uint,
            MaxwellThreeDDepthStencilFormat::ZFloat32 => ImageFormat::Depth32Float,
            MaxwellThreeDDepthStencilFormat::ZFloat32X24Stencil8 => {
                ImageFormat::Depth32FloatStencil8Uint
            }
            format => {
                return Err(MaxwellThreeDResourceError::UnsupportedDepthFormat {
                    role,
                    format: format as u32,
                });
            }
        };
        let width = *target
            .width()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let height = *target
            .height()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let control = *target
            .array_control()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let layers = match control {
            MaxwellThreeDDepthArrayControl::ThirdDimensionDefinesArraySize => *target
                .third_dimension()
                .value()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
            MaxwellThreeDDepthArrayControl::ArraySizeIsOne => 1,
        };
        let selected_layer = *target
            .layer()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        if selected_layer >= layers {
            return Err(MaxwellThreeDResourceError::ContradictoryState { role });
        }
        let description = image_description(MaxwellImageDescriptionRequest {
            dimension: ImageDimension::Two,
            extent: ImageExtent {
                width,
                height,
                depth: 1,
            },
            format,
            kind: ImageKind::DepthStencil,
            layers,
            role,
        })?;
        self.image(
            role,
            address,
            description,
            *target
                .layout()
                .value()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
            target.array_pitch().value().copied(),
            selected_layer,
            MaxwellThreeDGuestImageFormat::DepthStencil(guest_format),
            target.compression().value() == Some(&super::MaxwellThreeDZCompressionMode::Enabled),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn image(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: MaxwellThreeDUnresolvedAddress,
        description: ImageDescription,
        layout: MaxwellThreeDImageLayout,
        array_pitch_dwords: Option<u32>,
        selected_layer: u16,
        guest_format: MaxwellThreeDGuestImageFormat,
        compression_enabled: bool,
        row_pitch_bytes: Option<u32>,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let key = ResolvedViewKey {
            address_space: self.address_space.id(),
            role,
            definition: ResolvedViewDefinition::Attachment {
                address,
                description,
                layout,
                array_pitch_dwords,
                selected_layer,
                guest_format,
                compression_enabled,
                row_pitch_bytes,
                sample_mode: self.sample_mode,
            },
        };
        if self.reuse_view(&key)? {
            return Ok(());
        }
        let samples = match self
            .sample_mode
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?
        {
            MaxwellThreeDSampleMode::Samples1x1 => SampleCount::One,
            MaxwellThreeDSampleMode::Samples2x2 => SampleCount::Four,
            mode => return Err(MaxwellThreeDResourceError::UnsupportedSampleMode { role, mode }),
        };
        // Target WIDTH/HEIGHT count stored samples, not raster pixels. Keep
        // their full footprint for mappings/pitches while the neutral image
        // carries pixel dimensions and a separate sample count.
        // https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp
        let extent = description.extent();
        let description = if samples == SampleCount::Four {
            if !extent.width.is_multiple_of(2)
                || !extent.height.is_multiple_of(2)
                // Only the 32-bit color MS4 storage family is decoded below.
                // Wider single-sample formats must not inherit its PTE kind.
                || (matches!(guest_format, MaxwellThreeDGuestImageFormat::Color(_))
                    && description.format().plane_bytes_per_texel(0) != Some(4))
                || description.dimension() != ImageDimension::Two
                || !matches!(
                    layout,
                    MaxwellThreeDImageLayout::BlockLinear {
                        block_depth_log2: 0,
                        ..
                    }
                )
                || !matches!(
                    guest_format,
                    MaxwellThreeDGuestImageFormat::Color(_)
                        | MaxwellThreeDGuestImageFormat::DepthStencil(
                            MaxwellThreeDDepthStencilFormat::Stencil8Z24
                                | MaxwellThreeDDepthStencilFormat::Z24Stencil8
                        )
                )
            {
                return Err(MaxwellThreeDResourceError::UnsupportedImageLayout { role });
            }
            ImageDescription::new(
                description.dimension(),
                ImageExtent {
                    width: extent.width / 2,
                    height: extent.height / 2,
                    depth: 1,
                },
                description.format(),
                description.kind(),
                1,
                description.array_layers(),
                samples,
            )
            .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role })?
        } else {
            description
        };
        let bpp = u64::from(
            description
                .format()
                .plane_bytes_per_texel(0)
                .ok_or(MaxwellThreeDResourceError::UnsupportedImageLayout { role })?,
        );
        let compact_row = u64::from(extent.width)
            .checked_mul(bpp)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let compact_layer = compact_row
            .checked_mul(u64::from(extent.height))
            .and_then(|size| size.checked_mul(u64::from(extent.depth)))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        // Maxwell render-target and zeta ARRAY_PITCH registers count dwords,
        // not bytes. This is visible in deko3d's 1280x720 target, whose
        // captured 0x000f0000 value denotes a 0x003c0000-byte layer stride.
        // Keep the conversion at the guest/neutral boundary so the neutral
        // image layout remains byte-addressed.
        // https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/310c1f50beb77fc5c6f9075029973161d4e51a4a/src/video_core/texture_cache/image_info.cpp#L126-L177
        let array_pitch = array_pitch_dwords.map(|pitch| u64::from(pitch) * 4);
        let (neutral_layout, layer_stride, expected_kind, generic_kind_allowed) = match layout {
            MaxwellThreeDImageLayout::PitchLinear => {
                let row_pitch = row_pitch_bytes.map(u64::from).unwrap_or(compact_row);
                if row_pitch < compact_row || compression_enabled {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                let stride = array_pitch
                    .filter(|value| *value != 0)
                    .unwrap_or(row_pitch * u64::from(extent.height));
                (
                    ImageMemoryLayout::PitchLinear {
                        row_pitch,
                        layer_stride: stride,
                    },
                    stride,
                    MAXWELL_PITCH_KIND,
                    false,
                )
            }
            MaxwellThreeDImageLayout::BlockLinear {
                block_height_log2,
                block_depth_log2,
            } => {
                let row = align_up(compact_row, 64, role)?;
                let rows = align_up(u64::from(extent.height), 8_u64 << block_height_log2, role)?;
                let depths = align_up(u64::from(extent.depth), 1_u64 << block_depth_log2, role)?;
                let minimum = row
                    .checked_mul(rows)
                    .and_then(|size| size.checked_mul(depths))
                    .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
                let stride = array_pitch.filter(|value| *value != 0).unwrap_or(minimum);
                if samples == SampleCount::Four && stride < minimum {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                let compressed_color_kind = match guest_format {
                    MaxwellThreeDGuestImageFormat::Color(_) if samples == SampleCount::Four => {
                        Some(MAXWELL_C32_MS4_2CBR_KIND)
                    }
                    MaxwellThreeDGuestImageFormat::Color(_) => {
                        single_sample_compressed_color_kind(bpp)
                    }
                    _ => None,
                };
                let compressed_depth_kind = match guest_format {
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        MaxwellThreeDDepthStencilFormat::Z16,
                    ) if samples == SampleCount::One => Some(MAXWELL_Z16_2Z_KIND),
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        MaxwellThreeDDepthStencilFormat::Stencil8Z24,
                    ) => Some(if samples == SampleCount::Four {
                        MAXWELL_S8Z24_MS4_2CZ_KIND
                    } else {
                        MAXWELL_S8Z24_2CZ_KIND
                    }),
                    MaxwellThreeDGuestImageFormat::DepthStencil(
                        MaxwellThreeDDepthStencilFormat::Z24Stencil8,
                    ) => Some(if samples == SampleCount::Four {
                        MAXWELL_Z24S8_MS4_2CZ_KIND
                    } else {
                        MAXWELL_Z24S8_2CZ_KIND
                    }),
                    _ => None,
                };
                let expected_kind = compressed_color_kind
                    .or(compressed_depth_kind)
                    .unwrap_or(MAXWELL_GENERIC_BLOCK_LINEAR_KIND);
                (
                    ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                        block_width_log2: 0,
                        block_height_log2,
                        block_depth_log2,
                        layer_stride: stride,
                    }),
                    stride,
                    expected_kind,
                    samples == SampleCount::One
                        && (compressed_color_kind.is_some()
                            || (compressed_depth_kind.is_some() && !compression_enabled)),
                )
            }
        };
        if layer_stride < compact_layer {
            return Err(MaxwellThreeDResourceError::ContradictoryState { role });
        }
        let layer_offset = u64::from(selected_layer)
            .checked_mul(layer_stride)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let address_value = address
            .get()
            .checked_add(layer_offset)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let resolved_address =
            MaxwellThreeDUnresolvedAddress::new((address_value >> 32) as u8, address_value as u32);
        let layer_count = 1;
        let size = layer_stride
            .checked_mul(u64::from(layer_count))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let source = self.resolve(
            resolved_address,
            size,
            if role == MaxwellThreeDResourceRole::BlitSource {
                MemoryPermissions::READ
            } else {
                MemoryPermissions::WRITE
            },
            role,
        )?;
        let actual_kind = source
            .segments()
            .first()
            .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?
            .mapping()
            .kind();
        if source.segments().iter().any(|segment| {
            segment.mapping().kind() != actual_kind
                || !image_kind_matches(
                    layout,
                    expected_kind,
                    generic_kind_allowed,
                    segment.mapping().kind(),
                )
        }) {
            return Err(MaxwellThreeDResourceError::UnsupportedKind {
                role,
                expected: expected_kind,
                actual: source
                    .segments()
                    .iter()
                    .find(|segment| segment.mapping().kind() != expected_kind)
                    .unwrap()
                    .mapping()
                    .kind(),
            });
        }
        let retained = self.retained_backing(&source, role)?;
        let allocation_description = retained.allocation_description;
        let image_id = ImageId::new(resource_id(self.resources.len())?);
        let view = ImageView::new(
            image_id,
            description,
            Swizzle::IDENTITY,
            vec![(
                ImageSubresourceRange {
                    plane: 0,
                    mip_level: 0,
                    base_layer: selected_layer,
                    layer_count,
                },
                neutral_layout,
                retained.backing,
            )],
        )
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        self.resources.push(MaxwellThreeDResolvedResource::Image(
            MaxwellThreeDResolvedImage {
                role,
                access: if role == MaxwellThreeDResourceRole::BlitSource {
                    MaxwellThreeDResourceAccess::Read
                } else {
                    MaxwellThreeDResourceAccess::Write
                },
                description,
                allocation_description,
                view,
                source,
                mappings: retained.mappings,
                guest_layout: MaxwellThreeDPreservedImageLayout {
                    layout: neutral_layout,
                    pte_kind: actual_kind,
                    compression_enabled,
                },
                guest_format,
            },
        ));
        self.remember_last_view(key);
        Ok(())
    }

    fn resolve(
        &mut self,
        address: MaxwellThreeDUnresolvedAddress,
        size: u64,
        permissions: MemoryPermissions,
        role: MaxwellThreeDResourceRole,
    ) -> Result<MaxwellResolvedRange, MaxwellThreeDResourceError> {
        let bits = self
            .address_space
            .profile()
            .virtual_address()
            .address_bits()
            .bits();
        let address = GpuVirtualAddress::try_new(address.get(), bits).map_err(|_| {
            MaxwellThreeDResourceError::AddressOutOfRange {
                role,
                address: address.get(),
            }
        })?;
        if let Some(source) =
            self.retained_backings
                .resolve_cached(self.address_space, address, size, permissions)
        {
            return Ok(source);
        }
        self.address_space
            .resolve_range(address, size, permissions)
            .map_err(|error| {
                log::debug!(
                    "Maxwell resource resolution failed: role={role:?} address={:#x} size={size:#x}: {error}",
                    address.get(),
                );
                MaxwellThreeDResourceError::Resolution { role, error }
            })
    }

    fn finish(
        self,
    ) -> Result<
        (
            MaxwellThreeDResolvedResources,
            Box<[MaxwellThreeDDescriptorRead]>,
        ),
        MaxwellThreeDResourceError,
    > {
        let mut groups: [Vec<Arc<MaxwellThreeDResolvedResource>>; 4] = Default::default();
        for resource in self.resources {
            groups[resource.role().binding_group()].push(Arc::new(resource));
        }
        let result = MaxwellThreeDResolvedResources {
            resources: groups.map(|mut group| {
                group.sort_unstable_by_key(|r| r.role());
                (!group.is_empty()).then(|| group.into())
            }),
            samplers: (!self.samplers.is_empty()).then(|| self.samplers.into()),
        };
        result.validate_mappings(self.address_space)?;
        Ok((result, self.descriptor_reads.into_boxed_slice()))
    }
}

// Preserve precision, numeric representation, transfer function and channel
// order for render targets and 2D resolves. The L components denote sRGB;
// alpha remains linear UNORM in those formats.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
fn color_image_format(
    format: u8,
    role: MaxwellThreeDResourceRole,
) -> Result<ImageFormat, MaxwellThreeDResourceError> {
    match format {
        0xca => Ok(ImageFormat::Rgba16Float),
        0xcf => Ok(ImageFormat::Bgra8Unorm),
        0xd0 => Ok(ImageFormat::Bgra8Srgb),
        0xd1 => Ok(ImageFormat::Rgb10a2Unorm),
        0xd5 => Ok(ImageFormat::Rgba8Unorm),
        0xd6 => Ok(ImageFormat::Rgba8Srgb),
        0xe0 => Ok(ImageFormat::Rg11b10Ufloat),
        0xf2 => Ok(ImageFormat::R16Float),
        0xf3 => Ok(ImageFormat::R8Unorm),
        _ => Err(MaxwellThreeDResourceError::UnsupportedColorFormat { role, format }),
    }
}

/// Decodes verified Maxwell TIC format/component/swizzle tuples into their
/// neutral storage and sampling semantics. Numeric component and swizzle values are
/// defined by deko3d's pinned public Maxwell tables:
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
fn decode_sampled_texture_format(
    image_format: u8,
    components: [u8; 4],
    swizzle: [u8; 4],
    srgb: bool,
    reserved: u32,
) -> Option<ImageFormat> {
    if reserved != 0 || !swizzle.iter().all(|c| matches!(c, 0 | 2..=5 | 7)) {
        return None;
    }
    if components == [2; 4] {
        let astc = match (image_format, srgb) {
            (0x40, false) => Some(ImageFormat::Astc4x4Unorm),
            (0x40, true) => Some(ImageFormat::Astc4x4Srgb),
            (0x50, false) => Some(ImageFormat::Astc5x4Unorm),
            (0x50, true) => Some(ImageFormat::Astc5x4Srgb),
            (0x41, false) => Some(ImageFormat::Astc5x5Unorm),
            (0x41, true) => Some(ImageFormat::Astc5x5Srgb),
            (0x51, false) => Some(ImageFormat::Astc6x5Unorm),
            (0x51, true) => Some(ImageFormat::Astc6x5Srgb),
            (0x42, false) => Some(ImageFormat::Astc6x6Unorm),
            (0x42, true) => Some(ImageFormat::Astc6x6Srgb),
            (0x55, false) => Some(ImageFormat::Astc8x5Unorm),
            (0x55, true) => Some(ImageFormat::Astc8x5Srgb),
            (0x52, false) => Some(ImageFormat::Astc8x6Unorm),
            (0x52, true) => Some(ImageFormat::Astc8x6Srgb),
            (0x44, false) => Some(ImageFormat::Astc8x8Unorm),
            (0x44, true) => Some(ImageFormat::Astc8x8Srgb),
            (0x56, false) => Some(ImageFormat::Astc10x5Unorm),
            (0x56, true) => Some(ImageFormat::Astc10x5Srgb),
            (0x57, false) => Some(ImageFormat::Astc10x6Unorm),
            (0x57, true) => Some(ImageFormat::Astc10x6Srgb),
            (0x53, false) => Some(ImageFormat::Astc10x8Unorm),
            (0x53, true) => Some(ImageFormat::Astc10x8Srgb),
            (0x45, false) => Some(ImageFormat::Astc10x10Unorm),
            (0x45, true) => Some(ImageFormat::Astc10x10Srgb),
            (0x54, false) => Some(ImageFormat::Astc12x10Unorm),
            (0x54, true) => Some(ImageFormat::Astc12x10Srgb),
            (0x46, false) => Some(ImageFormat::Astc12x12Unorm),
            (0x46, true) => Some(ImageFormat::Astc12x12Srgb),
            _ => None,
        };
        if astc.is_some() {
            return astc;
        }
    }
    match (image_format, components, swizzle, srgb) {
        // B5G6R5 stores R in the low bits. The B,G,R swizzle makes the high
        // five bits the observable red channel (packed RGB565).
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/format_traits.inc
        (0x15, [2, 2, 2, 2], [4, 3, 2, 7], false) => Some(ImageFormat::Rgb565Unorm),
        // Z24S8 uses unsigned stencil R and normalized depth G. Selecting G
        // for RGB samples the depth aspect, with alpha forced to one.
        (0x29, [4, 2, 2, 2], [3, 3, 3, 7], false) => Some(ImageFormat::Depth24UnormStencil8Uint),
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/format_lookup_table.cpp
        (0x3a, [2, 2, 2, 2] | [2, 4, 4, 4], _, false) => Some(ImageFormat::Depth16Unorm),
        // ZF32 uses a float depth channel, replicated to RGB by its TIC mapping.
        // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/image_formats.h
        (0x2f, [7, 2, 2, 2], [2, 2, 2, 7], false) => Some(ImageFormat::Depth32Float),
        (0x27, [2, 2, 2, 2], _, false) => Some(ImageFormat::Bc4RUnorm),
        (0x28, [2, 2, 2, 2], _, false) => Some(ImageFormat::Bc5RgUnorm),
        // Component type 1 is SNORM; signed RGTC endpoints are not UNORM.
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
        (0x27, [1, 1, 1, 1], _, false) => Some(ImageFormat::Bc4RSnorm),
        (0x28, [1, 1, 1, 1], _, false) => Some(ImageFormat::Bc5RgSnorm),
        (0x17, [2, 2, 2, 2], _, srgb) => Some(if srgb {
            ImageFormat::Bc7RgbaSrgb
        } else {
            ImageFormat::Bc7RgbaUnorm
        }),
        (0x26, [2, 2, 2, 2], _, srgb) => Some(if srgb {
            ImageFormat::Bc3RgbaSrgb
        } else {
            ImageFormat::Bc3RgbaUnorm
        }),
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_compression_s3tc.txt
        // DXT1 storage and TIC selectors are independent. The opaque RGB
        // variant is equivalent only when no output observes the stored alpha.
        (0x24, [2, 2, 2, 2], mapping, srgb) => Some(match (mapping.contains(&5), srgb) {
            (false, false) => ImageFormat::Bc1RgbUnorm,
            (false, true) => ImageFormat::Bc1RgbSrgb,
            (true, false) => ImageFormat::Bc1RgbaUnorm,
            (true, true) => ImageFormat::Bc1RgbaSrgb,
        }),
        // Single-channel masks can independently route R or constants to
        // every sampled component, including white RGB with red as alpha.
        (0x1d, [2, 2, 2, 2], _, false) => Some(ImageFormat::R8Unorm),
        // Storage and view selectors are independent; luminance-alpha uses
        // R,R,R,G without changing the two-byte RG8 storage format.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/format_lookup_table.cpp
        (0x18, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rg8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], false) => Some(ImageFormat::Bgra8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], true) => Some(ImageFormat::Bgra8Srgb),
        (0x08, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgba8Unorm),
        (0x08, [2, 2, 2, 2], _, true) => Some(ImageFormat::Rgba8Srgb),
        (0x1b, [7, 7, 7, 7], _, false) => Some(ImageFormat::R16Float),
        (0x0c, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg16Float),
        (0x03, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rgba16Float),
        (0x21, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg11b10Ufloat),
        (0x09, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgb10a2Unorm),
        (0x0f, [7, 7, 7, 7], _, false) => Some(ImageFormat::R32Float),
        (0x04, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg32Float),
        (0x01, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rgba32Float),
        _ => None,
    }
}

fn retained_backing(
    source: &MaxwellResolvedRange,
    role: MaxwellThreeDResourceRole,
) -> Result<RetainedResourceBacking, MaxwellThreeDResourceError> {
    let first = source
        .segments()
        .first()
        .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?;
    let allocation = first.mapping().allocation();
    let allocation_offset = first.backing_offset();
    let mut expected_offset = allocation_offset;
    let mut contiguous = true;
    let mut physical = Vec::new();
    let mut mappings = Vec::new();
    for segment in source.segments() {
        let mapping = segment.mapping();
        contiguous &= mapping.allocation() == allocation
            && segment.backing_offset() == expected_offset
            && mapping.physical().bytes() == first.mapping().physical().bytes();
        physical.push(
            mapping
                .physical()
                .slice(segment.backing_offset(), segment.size())
                .map_err(|error| MaxwellThreeDResourceError::Resolution {
                    role,
                    error: MaxwellGpuAccessError::Backing(Arc::new(error)),
                })?,
        );
        mappings.push(MaxwellThreeDMappingReference {
            mapping: mapping.id(),
            generation: mapping.generation(),
            allocation: mapping.allocation(),
            backing_offset: segment.backing_offset(),
            size: segment.size(),
            kind: mapping.kind(),
        });
        expected_offset = expected_offset
            .checked_add(segment.size())
            .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?;
    }
    let physical = nixe_gpu::physical::PhysicalMapping::join(&physical).map_err(|error| {
        MaxwellThreeDResourceError::Resolution {
            role,
            error: MaxwellGpuAccessError::Backing(Arc::new(error)),
        }
    })?;
    // A GPU virtual resource may span independent nvmap objects. Its canonical
    // scatter/gather range remains the authority for content and aliases; a
    // separate logical allocation owns the assembled host view, without copying
    // guest memory or pretending the constituent objects have one identity.
    // Reserve the upper half for these frontend-owned logical allocations.
    const LOGICAL_ALLOCATION_BASE: u64 = 1 << 63;
    static NEXT_LOGICAL_ALLOCATION: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(LOGICAL_ALLOCATION_BASE);
    if source
        .segments()
        .iter()
        .any(|segment| segment.mapping().allocation().get() >= LOGICAL_ALLOCATION_BASE)
    {
        return Err(MaxwellThreeDResourceError::ResourceExhausted);
    }
    let (allocation, allocation_size, allocation_offset) = if contiguous {
        (
            GpuAllocationId::new(allocation.get()),
            first.mapping().physical().bytes(),
            allocation_offset,
        )
    } else {
        let id = NEXT_LOGICAL_ALLOCATION
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |value| value.checked_add(1),
            )
            .map_err(|_| MaxwellThreeDResourceError::ResourceExhausted)?;
        (GpuAllocationId::new(id), source.size(), 0)
    };
    let allocation_description = GpuAllocationDescription::new(allocation_size, 1)
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
    let backing = BackingView::new(
        allocation,
        allocation_description,
        allocation_offset,
        physical,
    )
    .map_err(|error| {
        log::debug!("invalid retained backing: role={role:?} error={error}");
        MaxwellThreeDResourceError::InvalidNeutralView { role }
    })?;
    Ok(RetainedResourceBacking {
        backing,
        allocation_description,
        mappings: mappings.into(),
    })
}

fn unresolved(upper: Option<&u8>, lower: Option<&u32>) -> Option<MaxwellThreeDUnresolvedAddress> {
    Some(MaxwellThreeDUnresolvedAddress::new(*upper?, *lower?))
}
fn inclusive_size(
    address: MaxwellThreeDUnresolvedAddress,
    limit: MaxwellThreeDUnresolvedAddress,
    role: MaxwellThreeDResourceRole,
) -> Result<u64, MaxwellThreeDResourceError> {
    limit
        .get()
        .checked_sub(address.get())
        .and_then(|size| size.checked_add(1))
        .ok_or(MaxwellThreeDResourceError::ContradictoryState { role })
}
fn resource_id(index: usize) -> Result<u64, MaxwellThreeDResourceError> {
    u64::try_from(index)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or(MaxwellThreeDResourceError::ResourceExhausted)
}
fn align_up(
    value: u64,
    alignment: u64,
    role: MaxwellThreeDResourceRole,
) -> Result<u64, MaxwellThreeDResourceError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })
}

const fn single_sample_compressed_color_kind(bytes_per_texel: u64) -> Option<u8> {
    match bytes_per_texel {
        4 => Some(MAXWELL_C32_2CRA_KIND),
        8 => Some(MAXWELL_C64_2CRA_KIND),
        16 => Some(MAXWELL_C128_2CR_KIND),
        _ => None,
    }
}

const fn image_kind_matches(
    layout: MaxwellThreeDImageLayout,
    expected: u8,
    generic_kind_allowed: bool,
    actual: u8,
) -> bool {
    match layout {
        MaxwellThreeDImageLayout::PitchLinear => {
            matches!(actual, MAXWELL_PITCH_KIND | MAXWELL_PITCH_NO_SWIZZLE_KIND)
        }
        MaxwellThreeDImageLayout::BlockLinear { .. } => {
            actual == expected
                || (generic_kind_allowed && actual == MAXWELL_GENERIC_BLOCK_LINEAR_KIND)
        }
    }
}

fn image_description(
    request: MaxwellImageDescriptionRequest,
) -> Result<ImageDescription, MaxwellThreeDResourceError> {
    let extent = ImageExtent::new(
        request.extent.width,
        request.extent.height,
        request.extent.depth,
    )
    .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role: request.role })?;
    ImageDescription::new(
        request.dimension,
        extent,
        request.format,
        request.kind,
        1,
        request.layers,
        SampleCount::One,
    )
    .map_err(|_| MaxwellThreeDResourceError::ContradictoryState { role: request.role })
}

#[cfg(test)]
fn read_descriptor_bytes(
    range: &nixe_gpu::physical::PhysicalMapping,
    offset: u64,
) -> Result<[u8; 32], MaxwellThreeDResourceError> {
    let mut bytes = [0; 32];
    read_backing_bytes(range, offset, &mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
fn read_backing_bytes(
    range: &nixe_gpu::physical::PhysicalMapping,
    offset: u64,
    bytes: &mut [u8],
) -> Result<(), MaxwellThreeDResourceError> {
    let inputs = range
        .pin(offset, bytes.len() as u64)
        .map_err(|e| MaxwellThreeDResourceError::Memory(Arc::new(e)))?;
    let inputs = nixe_memory::CapturedMemory::new(inputs)
        .map_err(|e| MaxwellThreeDResourceError::Memory(Arc::new(e)))?;
    inputs[0]
        .allocation()
        .machine()
        .read_captured_into(&inputs, 0, bytes)
        .map_err(|e| MaxwellThreeDResourceError::Memory(Arc::new(e)))?
        .into_ready()
        .expect("RAM fixture read cannot park");
    Ok(())
}

fn descriptor_words(bytes: [u8; 32]) -> [u32; 8] {
    std::array::from_fn(|index| {
        u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
    })
}

const fn texture_descriptor_pair(
    mode: MaxwellThreeDSamplerBindingMode,
    raw_handle: u32,
) -> (u32, u32) {
    match mode {
        MaxwellThreeDSamplerBindingMode::Independent => {
            (raw_handle & 0x000f_ffff, raw_handle >> 20)
        }
        MaxwellThreeDSamplerBindingMode::ViaTextureHeader => (raw_handle, raw_handle),
    }
}

fn uses_maxwell_texture_headers(selector: Option<&bool>) -> bool {
    selector.copied().unwrap_or(true)
}

fn decode_sampler(
    texture_reference: MaxwellThreeDTextureReference,
    descriptor: u32,
    bytes: [u8; 32],
) -> Result<MaxwellThreeDResolvedSampler, MaxwellThreeDResourceError> {
    // Maxwell TSC field placement and public enums follow deko3d's pinned
    // descriptor definition and generator:
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_sampler_control_block.h
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/tsc_generate.cpp
    let words = descriptor_words(bytes);
    let role = MaxwellThreeDResourceRole::Sampler(texture_reference);
    if ((words[1] >> 10) & 3) != 0 {
        return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "reduction filter",
            value: words[0] ^ words[1],
        });
    }
    let compare = (words[0] & (1 << 9) != 0).then(|| {
        use nixe_gpu::DepthCompareOperation as C;
        [
            C::Never,
            C::Less,
            C::Equal,
            C::LessEqual,
            C::Greater,
            C::NotEqual,
            C::GreaterEqual,
            C::Always,
        ][((words[0] >> 10) & 7) as usize]
    });
    let address_mode = |raw: u32| match raw {
        0 => Ok(AddressMode::Repeat),
        1 => Ok(AddressMode::MirroredRepeat),
        2 => Ok(AddressMode::ClampToEdge),
        3 => Ok(AddressMode::ClampToBorder),
        _ => Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "address mode",
            value: raw,
        }),
    };
    let filter = |raw: u32, field| match raw {
        1 => Ok(FilterMode::Nearest),
        2 => Ok(FilterMode::Linear),
        _ => Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field,
            value: raw,
        }),
    };
    let mip_filter = match (words[1] >> 6) & 3 {
        1 | 2 => FilterMode::Nearest,
        3 => FilterMode::Linear,
        value => {
            return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                descriptor,
                field: "mip filter",
                value,
            });
        }
    };
    let bias = ((words[1] >> 12) & 0x1fff) as i32;
    let signed_bias = (bias << 19) >> 19;
    let lod_min_fixed = (words[2] & 0xfff) as u16;
    let lod_max_fixed = ((words[2] >> 12) & 0xfff) as u16;
    if lod_min_fixed > lod_max_fixed {
        return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "LOD clamp",
            value: words[2] & 0x00ff_ffff,
        });
    }
    let max_anisotropy = [1, 2, 4, 6, 8, 10, 12, 16][((words[0] >> 20) & 7) as usize];
    let address_modes = [
        address_mode(words[0] & 7)?,
        address_mode((words[0] >> 3) & 7)?,
        address_mode((words[0] >> 6) & 7)?,
    ];
    let border_color = address_modes
        .contains(&AddressMode::ClampToBorder)
        .then_some([words[4], words[5], words[6], words[7]]);
    if let Some(color) = border_color
        && words[0] & (1 << 13) != 0
    {
        let srgb = [
            words[2] >> 24,
            (words[3] >> 12) & 255,
            (words[3] >> 20) & 255,
        ];
        for (bits, encoded) in color[..3].iter().copied().zip(srgb) {
            if (bits == 0 && encoded != 0) || (bits == 1.0_f32.to_bits() && encoded != 255) {
                return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                    descriptor,
                    field: "distinct sRGB border color",
                    value: words[2] ^ words[3],
                });
            }
        }
    }
    Ok(MaxwellThreeDResolvedSampler {
        role,
        compare,
        min_filter: filter((words[1] >> 4) & 3, "minification filter")?,
        mag_filter: filter(words[1] & 3, "magnification filter")?,
        mip_filter,
        lod_bias_fixed: signed_bias as i16,
        address_modes,
        border_color,
        lod_min_fixed,
        lod_max_fixed,
        max_anisotropy,
    })
}

#[derive(Clone, Debug)]
pub enum MaxwellThreeDResourceError {
    InvalidStorageBuffer {
        role: MaxwellThreeDResourceRole,
        field: &'static str,
        value: u64,
    },
    IncompleteState {
        role: MaxwellThreeDResourceRole,
    },
    ContradictoryState {
        role: MaxwellThreeDResourceRole,
    },
    AddressOutOfRange {
        role: MaxwellThreeDResourceRole,
        address: u64,
    },
    ArithmeticOverflow {
        role: MaxwellThreeDResourceRole,
    },
    Resolution {
        role: MaxwellThreeDResourceRole,
        error: MaxwellGpuAccessError,
    },
    StaleMapping {
        mapping: MaxwellMappingId,
        generation: nixe_gpu::MappingGeneration,
    },
    UnsupportedColorFormat {
        role: MaxwellThreeDResourceRole,
        format: u8,
    },
    UnsupportedSparseDepth {
        value: u8,
    },
    UnsupportedDepthFormat {
        role: MaxwellThreeDResourceRole,
        format: u32,
    },
    UnsupportedKind {
        role: MaxwellThreeDResourceRole,
        expected: u8,
        actual: u8,
    },
    UnsupportedImageLayout {
        role: MaxwellThreeDResourceRole,
    },
    UnsupportedSampleMode {
        role: MaxwellThreeDResourceRole,
        mode: MaxwellThreeDSampleMode,
    },
    DescriptorIndexOutOfRange {
        role: MaxwellThreeDResourceRole,
        index: u32,
    },
    TextureHandleOutOfRange {
        texture_reference: MaxwellThreeDTextureReference,
        constant_buffer_size: u64,
    },
    UnsupportedTextureBindingMode {
        descriptor: u32,
    },
    UnsupportedTextureDescriptor {
        descriptor: u32,
        field: &'static str,
        value: u32,
    },
    UnsupportedSamplerDescriptor {
        descriptor: u32,
        field: &'static str,
        value: u32,
    },
    InvalidNeutralView {
        role: MaxwellThreeDResourceRole,
    },
    Memory(Arc<nixe_memory::MemoryError>),
    ResourceExhausted,
}

impl Display for MaxwellThreeDResourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStorageBuffer { role, field, value } => write!(
                formatter,
                "invalid Maxwell storage buffer: role={role:?} {field}={value:#x}"
            ),
            Self::IncompleteState { role } => {
                write!(formatter, "incomplete Maxwell resource state: {role:?}")
            }
            Self::ContradictoryState { role } => {
                write!(formatter, "contradictory Maxwell resource state: {role:?}")
            }
            Self::AddressOutOfRange { role, address } => write!(
                formatter,
                "Maxwell resource address is outside the profile: role={role:?} address=0x{address:016x}"
            ),
            Self::ArithmeticOverflow { role } => {
                write!(formatter, "Maxwell resource range overflows: {role:?}")
            }
            Self::Resolution { role, error } => write!(
                formatter,
                "Maxwell resource resolution failed: role={role:?} error=[{error}]"
            ),
            Self::StaleMapping {
                mapping,
                generation,
            } => write!(
                formatter,
                "Maxwell resource retained a stale mapping: {mapping} {generation}"
            ),
            Self::UnsupportedColorFormat { role, format } => write!(
                formatter,
                "Maxwell color format has no neutral interpretation: role={role:?} format=0x{format:02x}"
            ),
            Self::UnsupportedSparseDepth { value } => write!(
                formatter,
                "sparse depth attachment semantics are not implemented: SET_ZT_SPARSE={value:#x}"
            ),
            Self::UnsupportedDepthFormat { role, format } => write!(
                formatter,
                "Maxwell depth format has no neutral interpretation: role={role:?} format=0x{format:02x}"
            ),
            Self::UnsupportedKind {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "Maxwell PTE kind contradicts image layout: role={role:?} expected=0x{expected:02x} actual=0x{actual:02x}"
            ),
            Self::UnsupportedImageLayout { role } => write!(
                formatter,
                "Maxwell image layout has no neutral interpretation: {role:?}"
            ),
            Self::UnsupportedSampleMode { role, mode } => write!(
                formatter,
                "Maxwell sample mode has no neutral image interpretation: role={role:?} mode={mode:?}"
            ),
            Self::DescriptorIndexOutOfRange { role, index } => write!(
                formatter,
                "Maxwell descriptor index exceeds its programmed pool: role={role:?} index={index}"
            ),
            Self::TextureHandleOutOfRange {
                texture_reference,
                constant_buffer_size,
            } => write!(
                formatter,
                "Maxwell texture handle is outside its constant buffer: reference={texture_reference:?} constant-buffer-size={constant_buffer_size}"
            ),
            Self::UnsupportedTextureBindingMode { descriptor } => write!(
                formatter,
                "Maxwell texture descriptor {descriptor} uses an unsupported texture/sampler binding mode"
            ),
            Self::UnsupportedTextureDescriptor {
                descriptor,
                field,
                value,
            } => write!(
                formatter,
                "Maxwell texture descriptor {descriptor} has unsupported {field}: value=0x{value:08x}"
            ),
            Self::UnsupportedSamplerDescriptor {
                descriptor,
                field,
                value,
            } => write!(
                formatter,
                "Maxwell sampler descriptor {descriptor} has unsupported {field}: value=0x{value:08x}"
            ),
            Self::InvalidNeutralView { role } => write!(
                formatter,
                "resolved Maxwell resource cannot form a neutral view: {role:?}"
            ),
            Self::Memory(error) => write!(formatter, "resource memory access failed: {error}"),
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell resource resolution exhausted host bookkeeping")
            }
        }
    }
}

impl std::error::Error for MaxwellThreeDResourceError {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nixe_gpu::{AddressMode, BlockLinearLayout, FilterMode, ImageMemoryLayout};
    use nixe_memory::MemoryPermissions;

    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };

    use super::{
        MAXWELL_C32_2CRA_KIND, MAXWELL_C64_2CRA_KIND, MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
        MaxwellThreeDPreservedImageLayout, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
        MaxwellThreeDRetainedBackingCache, MaxwellThreeDTextureDimension,
        decode_sampled_texture_format, decode_sampler, image_kind_matches, read_backing_bytes,
        read_descriptor_bytes, texture_descriptor_pair, uses_maxwell_texture_headers,
    };

    fn descriptor_bytes(words: [u32; 8]) -> [u8; 32] {
        let mut bytes = [0; 32];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn volume_tic_preserves_depth_mips_and_slice_geometry() {
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let mut space = MaxwellGpuAddressSpace::new(
            machine,
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let address = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: crate::test_memory::ranges(&allocation),
                backing_offset: 0,
                size: 4096,
                allocation_alignment: 4096,
                page_size: 0,
                kind: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap()
            .offset()
            .get();
        let format = 0x08
            | (2 << 7)
            | (2 << 10)
            | (2 << 13)
            | (2 << 16)
            | (2 << 19)
            | (3 << 22)
            | (4 << 25)
            | (5 << 28);
        let words = [
            format,
            address as u32,
            (address >> 32) as u32 | (3 << 21),
            (2 << 6) | (2 << 28),
            7 | (2 << 23),
            7 | (3 << 16) | (1 << 31),
            0,
            2 << 4,
        ];
        let mut retained = MaxwellThreeDRetainedBackingCache::default();
        let mut access = crate::test_memory::access(space.memory().machine());
        let mut builder =
            super::ResourceBuilder::new(space.memory(), None, &mut retained, &mut access);
        builder
            .sampled_image(
                super::MaxwellThreeDTextureReference::new(4, 0, 0),
                MaxwellThreeDTextureDimension::Three,
                0,
                descriptor_bytes(words),
            )
            .unwrap();
        let super::MaxwellThreeDResolvedResource::Image(image) = &builder.resources[0] else {
            panic!("expected volume")
        };
        assert_eq!(
            image.description().dimension(),
            nixe_gpu::ImageDimension::Three
        );
        assert_eq!(
            image.description().extent(),
            nixe_gpu::ImageExtent {
                width: 8,
                height: 8,
                depth: 4
            }
        );
        assert_eq!(image.description().array_layers(), 1);
        assert_eq!(image.description().mip_levels(), 3);
        assert_eq!(image.source().size(), 3584);
        assert_eq!(image.view().bindings().len(), 3);
        for (level, binding) in image.view().bindings().iter().enumerate() {
            assert_eq!(binding.subresources().mip_level, level as u8);
            assert_eq!(binding.subresources().layer_count, 1);
            let nixe_gpu::ImageMemoryLayout::BlockLinear(block) = binding.layout() else {
                panic!("expected GOB layout")
            };
            assert_eq!(block.block_depth_log2, 2u8.saturating_sub(level as u8));
        }
    }

    #[test]
    fn linked_sampler_table_uses_tic_limit_without_resolving_a_buffer() {
        let machine = crate::test_memory::machine();
        let space = MaxwellGpuAddressSpace::new(
            machine,
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let mut channel = crate::engines::tests::three_d_channel();
        let mut retained = MaxwellThreeDRetainedBackingCache::default();
        let mut access = crate::test_memory::access(space.memory().machine());
        for (mode, maximum, expected_limit) in [(1, 63, 63), (0, 63, 0), (1, 31, 31)] {
            for (method, argument) in [
                (0x155c, 0),
                (0x1560, 0x4000),
                (0x1564, 0),
                (0x157c, maximum),
                (0x1234, mode),
            ] {
                crate::engines::tests::program_three_d(&mut channel, method, argument);
            }
            let mut builder =
                super::ResourceBuilder::new(space.memory(), None, &mut retained, &mut access);
            builder
                .sampler_pool(&channel.three_d().shader_bindings())
                .unwrap();
            assert!(builder.resources.is_empty());
            let table = &builder.retained_backings.descriptor_tables
                [super::DescriptorTableKind::Sampler as usize];
            assert_eq!(table.limit, expected_limit);
            assert_eq!(table.base.unwrap().get(), 0x4000);
        }
    }

    #[test]
    fn descriptor_table_reuses_one_read_per_draw_and_invalidates_changed_bytes() {
        let machine = crate::test_memory::machine();
        let mut space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let mapping = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: crate::test_memory::ranges(&allocation),
                backing_offset: 0,
                size: 4096,
                allocation_alignment: 4096,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: nixe_memory::MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let base = mapping.offset();
        let role = MaxwellThreeDResourceRole::TextureHeaders;
        let mut table = super::DescriptorTable::default();
        table.synchronize(base, 1);
        table.next_epoch();
        let mut access = crate::test_memory::access(space.memory().machine());
        let mut batch = space.memory().read_batch();
        let (initial, first_version) = table
            .read(role, 0, space.memory(), &mut access, &mut batch)
            .unwrap();
        assert_eq!(initial, [0; 32]);
        batch.release();
        crate::test_memory::write(&allocation, 0, &[7; 32]).unwrap();
        assert_eq!(
            table
                .read(role, 0, space.memory(), &mut access, &mut batch)
                .unwrap(),
            (initial, first_version)
        );
        table.next_epoch();
        let (changed, next_version) = table
            .read(role, 0, space.memory(), &mut access, &mut batch)
            .unwrap();
        assert_eq!(changed, [7; 32]);
        assert_ne!(next_version, first_version);
        table.synchronize(space.memory().checked_add(base, 32).unwrap(), 0);
        table.next_epoch();
        let (_, moved_version) = table
            .read(role, 0, space.memory(), &mut access, &mut batch)
            .unwrap();
        assert_ne!(moved_version, next_version);
    }

    #[test]
    fn virtual_resource_spans_independent_allocations_with_real_aliases() {
        let machine = crate::test_memory::machine();
        let mut space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocations = [
            crate::test_memory::allocate(&machine, 0x1000),
            crate::test_memory::allocate(&machine, 0x1000),
        ];
        let mappings: Vec<_> = allocations
            .iter()
            .enumerate()
            .map(|(index, allocation)| {
                space
                    .map(MaxwellMapRequest {
                        allocation: MaxwellAllocationId::new(index as u64 + 1),
                        backing: crate::test_memory::ranges(allocation),
                        backing_offset: 0,
                        size: 0x1000,
                        allocation_alignment: 0x1000,
                        page_size: 0,
                        kind: 0,
                        cacheable: true,
                        permissions: MemoryPermissions::READ_WRITE,
                        fixed_offset: None,
                    })
                    .unwrap()
            })
            .collect();
        assert_eq!(
            mappings[1].offset().get(),
            mappings[0].offset().get() + 0x1000
        );
        let role = MaxwellThreeDResourceRole::ConstantBuffer { group: 4, slot: 1 };
        let source = space
            .memory()
            .resolve_range(
                space.address(mappings[0].offset().get() + 0xff0).unwrap(),
                32,
                MemoryPermissions::READ,
            )
            .unwrap();
        let mut cache = MaxwellThreeDRetainedBackingCache::default();
        let retained = cache.retain(&source, role).unwrap();
        assert_eq!(retained.allocation_description.size(), 32);
        assert_eq!(retained.backing.allocation_offset(), 0);
        assert_eq!(retained.mappings[0].allocation, mappings[0].allocation());
        assert_eq!(retained.mappings[1].allocation, mappings[1].allocation());
        let repeated = cache.retain(&source, role).unwrap();
        assert_eq!(retained.backing.allocation(), repeated.backing.allocation());
        for (index, mapping) in mappings.iter().enumerate() {
            let half = space
                .memory()
                .resolve_range(
                    space
                        .address(mapping.offset().get() + if index == 0 { 0xff0 } else { 0 })
                        .unwrap(),
                    16,
                    MemoryPermissions::READ,
                )
                .unwrap();
            let alias = cache.retain(&half, role).unwrap();
            assert!(retained.backing.overlaps(&alias.backing));
            crate::test_memory::write(
                &allocations[index],
                if index == 0 { 0xff0 } else { 0 },
                &[index as u8 + 1; 16],
            )
            .unwrap();
        }
        let mut actual = [0; 32];
        super::read_backing_bytes(retained.backing.mapping(), 0, &mut actual).unwrap();
        assert_eq!(actual[..16], [1; 16]);
        assert_eq!(actual[16..], [2; 16]);
        let fresh = cache.retain(&source, role).unwrap();
        assert_eq!(fresh.backing.allocation(), retained.backing.allocation());
    }

    #[test]
    fn generic_block_linear_color_bytes_remain_direct_when_compression_state_is_enabled() {
        let layout = MaxwellThreeDPreservedImageLayout {
            layout: ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: 4,
                block_depth_log2: 0,
                layer_stride: 0x20_0000,
            }),
            pte_kind: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
            compression_enabled: true,
        };

        assert!(layout.requires_materialization());
        assert!(layout.has_direct_canonical_representation());
        assert!(layout.supports_canonical_texel_storage());
    }

    #[test]
    fn compressed_depth_kind_still_requires_a_materialization_boundary() {
        let layout = MaxwellThreeDPreservedImageLayout {
            layout: ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: 4,
                block_depth_log2: 0,
                layer_stride: 0x20_0000,
            }),
            pte_kind: 0x17,
            compression_enabled: true,
        };

        assert!(!layout.has_direct_canonical_representation());
        assert!(!layout.supports_canonical_texel_storage());
    }

    #[test]
    fn compressed_color_kinds_require_resident_content_with_either_write_selector() {
        for pte_kind in [
            super::MAXWELL_C32_2CRA_KIND,
            super::MAXWELL_C64_2CRA_KIND,
            super::MAXWELL_C128_2CR_KIND,
        ] {
            for compression_enabled in [false, true] {
                let layout = MaxwellThreeDPreservedImageLayout {
                    layout: ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                        block_width_log2: 0,
                        block_height_log2: 4,
                        block_depth_log2: 0,
                        layer_stride: 0x20_0000,
                    }),
                    pte_kind,
                    compression_enabled,
                };
                assert!(layout.requires_materialization());
                assert!(!layout.has_direct_canonical_representation());
                assert!(layout.supports_canonical_texel_storage());
            }
        }
    }

    #[test]
    fn c32_block_linear_kind_accepts_generic_storage_but_rejects_c64_storage() {
        let maxwell_layout = super::MaxwellThreeDImageLayout::BlockLinear {
            block_height_log2: 4,
            block_depth_log2: 0,
        };

        assert!(image_kind_matches(
            maxwell_layout,
            MAXWELL_C32_2CRA_KIND,
            true,
            MAXWELL_C32_2CRA_KIND,
        ));
        assert!(image_kind_matches(
            maxwell_layout,
            MAXWELL_C32_2CRA_KIND,
            true,
            MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
        ));
        assert!(!image_kind_matches(
            maxwell_layout,
            MAXWELL_C32_2CRA_KIND,
            true,
            MAXWELL_C64_2CRA_KIND,
        ));
    }

    #[test]
    fn tsc_repeat_linear_sampler_becomes_an_exact_neutral_description() {
        let words = [0, 2 | (2 << 4) | (3 << 6), (15 * 256) << 12, 0, 0, 0, 0, 0];
        let texture_reference = super::MaxwellThreeDTextureReference::new(4, 3, 0x20);

        let sampler = decode_sampler(texture_reference, 8, descriptor_bytes(words)).unwrap();
        let description = sampler.description().unwrap();

        assert_eq!(
            sampler.role(),
            super::MaxwellThreeDResourceRole::Sampler(texture_reference)
        );
        assert_eq!(description.min_filter, FilterMode::Linear);
        assert_eq!(description.mag_filter, FilterMode::Linear);
        assert_eq!(description.mip_filter, FilterMode::Linear);
        assert_eq!(description.address_modes, [AddressMode::Repeat; 3]);
        assert_eq!(description.lod_min, 0.0);
        assert_eq!(description.lod_max, 15.0);
        assert_eq!(description.max_anisotropy, 1.0);
    }

    #[test]
    fn tsc_preserves_explicit_border_colors_and_ignores_unused_border_words() {
        let texture_reference = super::MaxwellThreeDTextureReference::new(4, 3, 0x20);
        for color in [[0; 4], [0, 0, 0, 1.0_f32.to_bits()], [1.0_f32.to_bits(); 4]] {
            let mut words = [
                3 | (3 << 3) | (3 << 6),
                2 | (2 << 4) | (3 << 6),
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            words[4..].copy_from_slice(&color);
            let description = decode_sampler(texture_reference, 8, descriptor_bytes(words))
                .unwrap()
                .description()
                .unwrap();
            assert_eq!(description.address_modes, [AddressMode::ClampToBorder; 3]);
            assert_eq!(description.border_color, Some(color));
            words[0] = 0;
            assert_eq!(
                decode_sampler(texture_reference, 8, descriptor_bytes(words))
                    .unwrap()
                    .description()
                    .unwrap()
                    .border_color,
                None
            );
        }
    }

    #[test]
    fn tsc_depth_comparisons_preserve_every_function_and_reduction_stays_explicit() {
        use nixe_gpu::DepthCompareOperation as C;
        let texture = super::MaxwellThreeDTextureReference::new(4, 3, 0x20);
        let mut words = [0; 8];
        words[1] = 1 | (1 << 4) | (1 << 6);
        for (raw, compare) in [
            C::Never,
            C::Less,
            C::Equal,
            C::LessEqual,
            C::Greater,
            C::NotEqual,
            C::GreaterEqual,
            C::Always,
        ]
        .into_iter()
        .enumerate()
        {
            words[0] = (1 << 9) | ((raw as u32) << 10);
            assert_eq!(
                decode_sampler(texture, 8, descriptor_bytes(words))
                    .unwrap()
                    .description()
                    .unwrap()
                    .compare,
                Some(compare)
            );
        }
        words[0] &= !(1 << 9);
        assert_eq!(
            decode_sampler(texture, 8, descriptor_bytes(words))
                .unwrap()
                .description()
                .unwrap()
                .compare,
            None
        );
        words[1] |= 1 << 10;
        assert!(matches!(
            decode_sampler(texture, 8, descriptor_bytes(words)),
            Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                field: "reduction filter",
                ..
            })
        ));
    }

    #[test]
    fn raw_texture_handle_selects_tic_and_tsc_for_each_binding_mode() {
        assert_eq!(
            texture_descriptor_pair(
                super::MaxwellThreeDSamplerBindingMode::Independent,
                0xabc0_0008,
            ),
            (8, 0xabc)
        );
        assert_eq!(
            texture_descriptor_pair(super::MaxwellThreeDSamplerBindingMode::ViaTextureHeader, 8),
            (8, 8)
        );
    }

    #[test]
    fn texture_header_selector_uses_the_maxwell_reset_mode_until_programmed() {
        assert!(uses_maxwell_texture_headers(None));
        assert!(uses_maxwell_texture_headers(Some(&true)));
        assert!(!uses_maxwell_texture_headers(Some(&false)));
    }

    #[test]
    fn paired_texture_handles_reject_invalid_slots_and_shifts() {
        use super::MaxwellThreeDTextureReference as T;
        assert!(T::expression(4, (1, 0, 32), None).is_none());
        assert!(T::expression(4, (1, 0, 0), Some((32, 0, 0))).is_none());
        assert!(T::expression(super::MAXWELL_BIND_GROUP_COUNT as u8, (1, 0, 0), None).is_none());
    }

    #[test]
    fn bc5_luminance_alpha_descriptor_keeps_both_compressed_channels() {
        let word = 0x3492_4928_u32;
        let components = [7, 10, 13, 16].map(|shift| ((word >> shift) & 7) as u8);
        let swizzle = [19, 22, 25, 28].map(|shift| ((word >> shift) & 7) as u8);
        let format =
            decode_sampled_texture_format((word & 0x7f) as u8, components, swizzle, false, 0)
                .unwrap();
        assert_eq!(format, nixe_gpu::ImageFormat::Bc5RgUnorm);
        assert_eq!(swizzle, [2, 2, 2, 3]);
        assert_eq!(format.block_extent(), [4, 4]);
        assert_eq!(format.plane_bytes_per_block(0), Some(16));
        assert_eq!(
            decode_sampled_texture_format(0x28, components, swizzle, true, 0),
            None
        );
    }

    #[test]
    fn signed_rgtc_tic_selects_snorm_storage_without_losing_view_channels() {
        for (word, expected) in [
            (0x70d124a8, nixe_gpu::ImageFormat::Bc5RgSnorm),
            (0x70d124a7, nixe_gpu::ImageFormat::Bc4RSnorm),
        ] {
            let components = [7, 10, 13, 16].map(|shift| ((word >> shift) & 7) as u8);
            let swizzle = [19, 22, 25, 28].map(|shift| ((word >> shift) & 7) as u8);
            assert_eq!(
                decode_sampled_texture_format((word & 127) as u8, components, swizzle, false, 0),
                Some(expected)
            );
            assert_eq!(swizzle, [2, 3, 0, 7]);
            assert_eq!(
                decode_sampled_texture_format((word & 127) as u8, components, swizzle, true, 0),
                None
            );
        }
    }

    #[test]
    fn bc7_tic_preserves_linear_srgb_storage_and_rgba_mapping() {
        let word = 0x58d24917;
        let components = [7, 10, 13, 16].map(|shift| ((word >> shift) & 7) as u8);
        let swizzle = [19, 22, 25, 28].map(|shift| ((word >> shift) & 7) as u8);
        for (srgb, format) in [
            (false, nixe_gpu::ImageFormat::Bc7RgbaUnorm),
            (true, nixe_gpu::ImageFormat::Bc7RgbaSrgb),
        ] {
            assert_eq!(
                decode_sampled_texture_format(0x17, components, swizzle, srgb, 0),
                Some(format)
            );
            assert_eq!(format.block_extent(), [4, 4]);
            assert_eq!(format.plane_bytes_per_block(0), Some(16));
        }
    }

    #[test]
    fn sampler_lod_bias_sign_extends_all_thirteen_fixed_point_bits() {
        let texture = super::MaxwellThreeDTextureReference::new(4, 0, 0);
        for (raw, expected) in [
            (0, 0.0),
            (0x1e00, -2.0),
            (0x1000, -16.0),
            (0x0fff, 4095.0 / 256.0),
            (1, 1.0 / 256.0),
        ] {
            let mut words = [0; 8];
            words[1] = 2 | (2 << 4) | (3 << 6) | (raw << 12);
            words[2] = (8 * 256) << 12;
            assert_eq!(
                decode_sampler(texture, 338, descriptor_bytes(words))
                    .unwrap()
                    .description()
                    .unwrap()
                    .lod_bias,
                expected
            );
        }
    }

    #[test]
    fn astc_descriptor_uses_encoded_block_footprint_and_preserves_swizzle() {
        let word = 0x58d2_4944_u32;
        let components = [7, 10, 13, 16].map(|shift| ((word >> shift) & 7) as u8);
        let swizzle = [19, 22, 25, 28].map(|shift| ((word >> shift) & 7) as u8);
        let format =
            decode_sampled_texture_format((word & 0x7f) as u8, components, swizzle, false, 0)
                .unwrap();
        assert_eq!(format, nixe_gpu::ImageFormat::Astc8x8Unorm);
        assert_eq!(format.block_extent(), [8, 8]);
        assert_eq!(format.plane_bytes_per_block(0), Some(16));
        assert_eq!(
            decode_sampled_texture_format(0x44, components, swizzle, true, 0),
            Some(nixe_gpu::ImageFormat::Astc8x8Srgb)
        );
        assert_eq!(
            decode_sampled_texture_format(0x44, [7; 4], swizzle, false, 0),
            None
        );
        assert_eq!(
            decode_sampled_texture_format(0x44, components, [1; 4], false, 0),
            None
        );
        assert_eq!(
            decode_sampled_texture_format(0x44, components, swizzle, false, 1),
            None
        );
    }

    #[test]
    fn rgb565_descriptor_swizzle_selects_high_red_low_blue_and_opaque_alpha() {
        let word = 0x74e2_4915_u32;
        let components = [7, 10, 13, 16].map(|shift| ((word >> shift) & 7) as u8);
        let swizzle = [19, 22, 25, 28].map(|shift| ((word >> shift) & 7) as u8);
        assert_eq!(
            decode_sampled_texture_format(
                (word & 0x7f) as u8,
                components,
                swizzle,
                false,
                word >> 31
            ),
            Some(nixe_gpu::ImageFormat::Rgb565Unorm)
        );
        assert_eq!(
            decode_sampled_texture_format(0x15, components, swizzle, true, 0),
            None
        );
        assert_eq!(
            decode_sampled_texture_format(0x15, components, [2, 3, 4, 7], false, 0),
            None
        );
    }

    #[test]
    fn r8_target_and_texture_use_the_same_single_channel_storage() {
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
        let format =
            super::color_image_format(0xf3, MaxwellThreeDResourceRole::ColorTarget(0)).unwrap();
        assert_eq!(format, nixe_gpu::ImageFormat::R8Unorm);
        assert_eq!(format.plane_bytes_per_block(0), Some(1));
        assert_eq!(
            decode_sampled_texture_format(0x1d, [2; 4], [2, 0, 0, 7], false, 0),
            Some(format)
        );
    }

    #[test]
    fn r16_float_target_and_texture_use_the_same_half_float_storage() {
        let format =
            super::color_image_format(0xf2, MaxwellThreeDResourceRole::ColorTarget(0)).unwrap();
        assert_eq!(format, nixe_gpu::ImageFormat::R16Float);
        assert_eq!(format.plane_bytes_per_block(0), Some(2));
        assert_eq!(
            decode_sampled_texture_format(0x1b, [7; 4], [2, 0, 0, 7], false, 0),
            Some(format)
        );
        for raw in [0xee, 0xef, 0xf0, 0xf1] {
            assert!(
                super::color_image_format(raw, MaxwellThreeDResourceRole::ColorTarget(0)).is_err()
            );
        }
    }

    #[test]
    fn rgba16_float_target_and_texture_encodings_have_the_same_interpretation() {
        let target =
            super::color_image_format(0xca, MaxwellThreeDResourceRole::ColorTarget(0)).unwrap();
        assert_eq!(target, nixe_gpu::ImageFormat::Rgba16Float);
        assert_eq!(
            decode_sampled_texture_format(0x03, [7; 4], [2, 3, 4, 5], false, 0),
            Some(target)
        );
        // Equal texel width does not make the signed/unsigned integer formats
        // or the RGBX variant interchangeable with four floating components.
        for raw in [0xc8, 0xc9, 0xce] {
            assert!(
                super::color_image_format(raw, MaxwellThreeDResourceRole::ColorTarget(0)).is_err()
            );
        }
    }

    #[test]
    fn packed_rgb10a2_target_and_texture_preserve_unorm_channels_and_storage() {
        let target =
            super::color_image_format(0xd1, MaxwellThreeDResourceRole::ColorTarget(0)).unwrap();
        assert_eq!(target, nixe_gpu::ImageFormat::Rgb10a2Unorm);
        assert_eq!(target.plane_bytes_per_texel(0), Some(4));
        for swizzle in [[2, 3, 4, 7], [4, 3, 2, 5], [2, 2, 2, 5], [7, 7, 7, 2]] {
            assert_eq!(
                decode_sampled_texture_format(0x09, [2; 4], swizzle, false, 0),
                Some(target)
            );
        }
        assert_eq!(
            decode_sampled_texture_format(0x09, [2; 4], [2, 3, 4, 5], false, 0),
            Some(target)
        );
        for (components, srgb, reserved) in
            [([4; 4], false, 0), ([2; 4], true, 0), ([2; 4], false, 1)]
        {
            assert_eq!(
                decode_sampled_texture_format(0x09, components, [2, 3, 4, 5], srgb, reserved),
                None
            );
        }
    }

    #[test]
    fn sampled_storage_preserves_independent_alpha_and_constant_component_views() {
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 0x1000);
        let mut space = MaxwellGpuAddressSpace::new(
            machine,
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let address = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: crate::test_memory::ranges(&allocation),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                cacheable: true,
                fixed_offset: None,
                permissions: MemoryPermissions::READ_WRITE,
            })
            .unwrap()
            .offset()
            .get();
        for (word, format, swizzle) in [
            (
                0x5b6a4908,
                nixe_gpu::ImageFormat::Rgba8Unorm,
                nixe_gpu::Swizzle([nixe_gpu::ComponentSwizzle::Alpha; 4]),
            ),
            (
                0x78d24909,
                nixe_gpu::ImageFormat::Rgb10a2Unorm,
                nixe_gpu::Swizzle([
                    nixe_gpu::ComponentSwizzle::Red,
                    nixe_gpu::ComponentSwizzle::Green,
                    nixe_gpu::ComponentSwizzle::Blue,
                    nixe_gpu::ComponentSwizzle::One,
                ]),
            ),
        ] {
            let words = [
                word,
                address as u32,
                (address >> 32) as u32 | (3 << 21),
                0,
                7 | (1 << 23),
                7 | (1 << 31),
                0,
                0,
            ];
            let mut access = crate::test_memory::access(space.memory().machine());
            let mut retained = MaxwellThreeDRetainedBackingCache::default();
            let mut builder =
                super::ResourceBuilder::new(space.memory(), None, &mut retained, &mut access);
            builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    MaxwellThreeDTextureDimension::Two,
                    0,
                    descriptor_bytes(words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(image) = &builder.resources[0] else {
                panic!("expected sampled image")
            };
            assert_eq!(image.description.format(), format);
            assert_eq!(image.view.swizzle(), swizzle);
        }
    }

    #[test]
    fn packed_unsigned_float_target_and_texture_preserve_storage_and_opaque_alpha() {
        // deko3d format_traits.inc: BF10GF11RF11, Float components, R,G,B,OneFloat.
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/format_traits.inc
        let target =
            super::color_image_format(0xe0, MaxwellThreeDResourceRole::ColorTarget(0)).unwrap();
        assert_eq!(target, nixe_gpu::ImageFormat::Rg11b10Ufloat);
        assert_eq!(target.plane_bytes_per_texel(0), Some(4));
        assert_eq!(
            decode_sampled_texture_format(0x21, [7; 4], [2, 3, 4, 7], false, 0),
            Some(target)
        );
        for (components, swizzle, srgb, reserved) in [
            ([2; 4], [2, 3, 4, 7], false, 0),
            ([7; 4], [2, 3, 4, 7], true, 0),
            ([7; 4], [2, 3, 4, 7], false, 1),
        ] {
            assert_eq!(
                decode_sampled_texture_format(0x21, components, swizzle, srgb, reserved),
                None
            );
        }
    }

    #[test]
    fn sampled_texture_formats_cover_captured_r32_float_and_direct_host_family() {
        assert_eq!(
            decode_sampled_texture_format(0x0f, [7; 4], [2, 0, 0, 7], false, 0),
            Some(nixe_gpu::ImageFormat::R32Float)
        );
        for (format, components, swizzle, expected) in [
            (0x1d, [2; 4], [2, 0, 0, 7], nixe_gpu::ImageFormat::R8Unorm),
            (0x18, [2; 4], [2, 3, 0, 7], nixe_gpu::ImageFormat::Rg8Unorm),
            (0x1b, [7; 4], [2, 0, 0, 7], nixe_gpu::ImageFormat::R16Float),
            (0x0c, [7; 4], [2, 3, 0, 7], nixe_gpu::ImageFormat::Rg16Float),
            (
                0x03,
                [7; 4],
                [2, 3, 4, 5],
                nixe_gpu::ImageFormat::Rgba16Float,
            ),
            (0x04, [7; 4], [2, 3, 0, 7], nixe_gpu::ImageFormat::Rg32Float),
            (
                0x01,
                [7; 4],
                [2, 3, 4, 5],
                nixe_gpu::ImageFormat::Rgba32Float,
            ),
        ] {
            assert_eq!(
                decode_sampled_texture_format(format, components, swizzle, false, 0),
                Some(expected)
            );
        }
        assert_eq!(
            decode_sampled_texture_format(0x0f, [7; 4], [2, 0, 0, 7], true, 0),
            None
        );
        assert_eq!(
            decode_sampled_texture_format(0x0f, [7; 4], [2, 0, 0, 7], false, 1),
            None
        );
    }

    #[test]
    fn bc1_texture_descriptor_preserves_rgb_alpha_and_srgb_semantics() {
        let word = 0x78d2_4924_u32;
        let components = [7, 10, 13, 16].map(|shift| ((word >> shift) & 7) as u8);
        let swizzle = [19, 22, 25, 28].map(|shift| ((word >> shift) & 7) as u8);
        assert_eq!(components, [2; 4]);
        assert_eq!(swizzle, [2, 3, 4, 7]);
        for (alpha, srgb, expected) in [
            (7, false, nixe_gpu::ImageFormat::Bc1RgbUnorm),
            (7, true, nixe_gpu::ImageFormat::Bc1RgbSrgb),
            (5, false, nixe_gpu::ImageFormat::Bc1RgbaUnorm),
            (5, true, nixe_gpu::ImageFormat::Bc1RgbaSrgb),
        ] {
            assert_eq!(
                decode_sampled_texture_format(
                    (word & 0x7f) as u8,
                    components,
                    [2, 3, 4, alpha],
                    srgb,
                    word >> 31
                ),
                Some(expected)
            );
        }
        let captured = 0x08d2_4924_u32;
        let captured_swizzle = [19, 22, 25, 28].map(|shift| ((captured >> shift) & 7) as u8);
        assert_eq!(captured_swizzle, [2, 3, 4, 0]);
        assert_eq!(
            decode_sampled_texture_format(0x24, components, captured_swizzle, false, 0),
            Some(nixe_gpu::ImageFormat::Bc1RgbUnorm)
        );
        for mapping in [[5, 3, 4, 7], [2, 5, 4, 0], [7, 7, 7, 5]] {
            assert_eq!(
                decode_sampled_texture_format(0x24, components, mapping, false, 0),
                Some(nixe_gpu::ImageFormat::Bc1RgbaUnorm)
            );
            assert_eq!(
                decode_sampled_texture_format(0x24, components, mapping, true, 0),
                Some(nixe_gpu::ImageFormat::Bc1RgbaSrgb)
            );
        }
        for mapping in [[2, 3, 4, 1], [6, 3, 4, 0]] {
            assert_eq!(
                decode_sampled_texture_format(0x24, components, mapping, false, 0),
                None
            );
        }
        assert_eq!(
            decode_sampled_texture_format(0x24, [7; 4], swizzle, false, 0),
            None
        );
        assert_eq!(
            decode_sampled_texture_format(0x24, components, [4, 3, 2, 7], false, 0),
            Some(nixe_gpu::ImageFormat::Bc1RgbUnorm)
        );
        assert_eq!(
            decode_sampled_texture_format(0x24, components, swizzle, false, 1),
            None
        );
    }

    #[test]
    fn bc3_descriptor_preserves_interpolated_alpha_srgb_and_view_swizzles() {
        for (srgb, expected) in [
            (false, nixe_gpu::ImageFormat::Bc3RgbaUnorm),
            (true, nixe_gpu::ImageFormat::Bc3RgbaSrgb),
        ] {
            for swizzle in [[2, 3, 4, 5], [4, 3, 2, 7], [5; 4]] {
                assert_eq!(
                    decode_sampled_texture_format(0x26, [2; 4], swizzle, srgb, 0),
                    Some(expected)
                );
            }
            assert_eq!(expected.block_extent(), [4, 4]);
            assert_eq!(expected.plane_bytes_per_block(0), Some(16));
            assert_eq!(expected.plane_bytes_per_texel(0), None);
        }
        assert_eq!(
            decode_sampled_texture_format(0x26, [7; 4], [2, 3, 4, 5], false, 0),
            None
        );
    }

    #[test]
    fn pitch_texture_resolves_padded_rows_and_nomipmap_2d_type() {
        let machine = crate::test_memory::machine();
        let mut address_space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocation = crate::test_memory::allocate(&machine, 0x1000);
        let backing = crate::test_memory::ranges(&allocation);
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                size: backing.iter().map(|range| range.bytes()).sum::<u64>(),
                backing,
                backing_offset: 0,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: super::MAXWELL_PITCH_KIND,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        for texture_type in [1, 7] {
            let mut words = [
                0x74e2_4915,
                address as u32,
                (address >> 32) as u32 | (2 << 21),
                2,
                18 | (texture_type << 23),
                16 | (1 << 31),
                0,
                0,
            ];
            let mut memory_access = crate::test_memory::access(address_space.memory().machine());
            let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
            let mut builder = super::ResourceBuilder::new(
                address_space.memory(),
                None,
                &mut retained_cache,
                &mut memory_access,
            );
            builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    MaxwellThreeDTextureDimension::Two,
                    256,
                    descriptor_bytes(words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(image) = &builder.resources[0] else {
                panic!("expected sampled image");
            };
            assert_eq!(
                image.description.format(),
                nixe_gpu::ImageFormat::Rgb565Unorm
            );
            assert_eq!(
                image.view.bindings()[0].backing().mapping().bytes(),
                64 * 17
            );
            assert_eq!(
                image.view.bindings()[0].layout(),
                ImageMemoryLayout::PitchLinear {
                    row_pitch: 64,
                    layer_stride: 64 * 17,
                }
            );
            assert!(!image.guest_layout.requires_materialization());
            let mut rg_words = words;
            rg_words[0] = 0x3492_4918; // R,R,R,G from RG8 storage.
            let mut rg_builder_access =
                crate::test_memory::access(address_space.memory().machine());
            let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
            let mut rg_builder = super::ResourceBuilder::new(
                address_space.memory(),
                None,
                &mut retained_cache,
                &mut rg_builder_access,
            );
            rg_builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    MaxwellThreeDTextureDimension::Two,
                    374,
                    descriptor_bytes(rg_words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(image) = &rg_builder.resources[0]
            else {
                panic!("expected sampled RG8 image");
            };
            assert_eq!(image.description.format(), nixe_gpu::ImageFormat::Rg8Unorm);
            assert_eq!(
                image.view.swizzle(),
                nixe_gpu::Swizzle([
                    nixe_gpu::ComponentSwizzle::Red,
                    nixe_gpu::ComponentSwizzle::Red,
                    nixe_gpu::ComponentSwizzle::Red,
                    nixe_gpu::ComponentSwizzle::Green,
                ])
            );
            assert_eq!(
                image.view.bindings()[0].backing().mapping().bytes(),
                64 * 17
            );
            let mut mask_words = words;
            mask_words[0] = 0x2ffa_491d; // 1,1,1,R from single-channel storage.
            let mut mask_builder_access =
                crate::test_memory::access(address_space.memory().machine());
            let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
            let mut mask_builder = super::ResourceBuilder::new(
                address_space.memory(),
                None,
                &mut retained_cache,
                &mut mask_builder_access,
            );
            mask_builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    MaxwellThreeDTextureDimension::Two,
                    361,
                    descriptor_bytes(mask_words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(mask) = &mask_builder.resources[0]
            else {
                panic!("expected R8 mask");
            };
            assert_eq!(mask.description.format(), nixe_gpu::ImageFormat::R8Unorm);
            assert_eq!(
                mask.view.swizzle(),
                nixe_gpu::Swizzle([
                    nixe_gpu::ComponentSwizzle::One,
                    nixe_gpu::ComponentSwizzle::One,
                    nixe_gpu::ComponentSwizzle::One,
                    nixe_gpu::ComponentSwizzle::Red
                ])
            );
            assert_eq!(mask.view.bindings()[0].backing().mapping().bytes(), 64 * 17);
            let mut array_builder_access =
                crate::test_memory::access(address_space.memory().machine());
            let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
            let mut array_builder = super::ResourceBuilder::new(
                address_space.memory(),
                None,
                &mut retained_cache,
                &mut array_builder_access,
            );
            array_builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    MaxwellThreeDTextureDimension::TwoArray,
                    257,
                    descriptor_bytes(mask_words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(array_mask) =
                &array_builder.resources[0]
            else {
                panic!("expected single-layer array view");
            };
            assert_eq!(array_mask.description.array_layers(), 1);
            assert_eq!(
                array_mask.view.bindings()[0].backing().mapping().bytes(),
                64 * 17
            );
            words[3] = 1; // Thirty-two bytes cannot hold nineteen RGB565 texels.
            assert!(
                builder
                    .sampled_image(
                        super::MaxwellThreeDTextureReference::new(4, 0, 0),
                        MaxwellThreeDTextureDimension::Two,
                        256,
                        descriptor_bytes(words)
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn depth_array_tic_preserves_native_storage_kind_and_layers() {
        let machine = crate::test_memory::machine();
        use nixe_gpu::ImageFormat;
        let mut space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocation = crate::test_memory::allocate(&machine, 0x1000);
        let backing = crate::test_memory::ranges(&allocation);
        let map = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                size: backing.iter().map(|range| range.bytes()).sum::<u64>(),
                backing,
                backing_offset: 0,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = map.offset().get();
        for (code, r_type, swizzle, format, guest) in [
            (
                0x3a,
                2,
                [2, 0, 0, 7],
                ImageFormat::Depth16Unorm,
                super::MaxwellThreeDDepthStencilFormat::Z16,
            ),
            (
                0x29,
                4,
                [3, 3, 3, 7],
                ImageFormat::Depth24UnormStencil8Uint,
                super::MaxwellThreeDDepthStencilFormat::Z24Stencil8,
            ),
            (
                0x2f,
                7,
                [2, 2, 2, 7],
                ImageFormat::Depth32Float,
                super::MaxwellThreeDDepthStencilFormat::ZFloat32,
            ),
        ] {
            let format_word = code
                | (r_type << 7)
                | (2 << 10)
                | (2 << 13)
                | (2 << 16)
                | (swizzle[0] << 19)
                | (swizzle[1] << 22)
                | (swizzle[2] << 25)
                | (swizzle[3] << 28);
            let words = [
                format_word,
                address as u32,
                (address >> 32) as u32 | (3 << 21),
                0,
                7 | (5 << 23),
                7 | (1 << 16) | (1 << 31),
                0,
                0,
            ];
            let mut memory_access = crate::test_memory::access(space.memory().machine());
            let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
            let mut builder = super::ResourceBuilder::new(
                space.memory(),
                None,
                &mut retained_cache,
                &mut memory_access,
            );
            builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    MaxwellThreeDTextureDimension::TwoArray,
                    0,
                    descriptor_bytes(words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(image) = &builder.resources[0] else {
                panic!("expected depth image")
            };
            assert_eq!(image.description().format(), format);
            assert_eq!(image.view().swizzle(), nixe_gpu::Swizzle::IDENTITY);
            assert_eq!(
                image.description().kind(),
                nixe_gpu::ImageKind::DepthStencil
            );
            assert_eq!(image.description().array_layers(), 2);
            assert_eq!(image.source().size(), 1024);
            assert_eq!(
                image.guest_format(),
                super::MaxwellThreeDGuestImageFormat::DepthStencil(guest)
            );
        }
    }

    #[test]
    fn bc3_mip_chain_resolves_all_levels_and_separates_array_layers() {
        let machine = crate::test_memory::machine();
        let mut address_space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocation = crate::test_memory::allocate(&machine, 0x48000);
        let backing = crate::test_memory::ranges(&allocation);
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                size: backing.iter().map(|range| range.bytes()).sum::<u64>(),
                backing,
                backing_offset: 0,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        let offsets = [0, 16384, 20480, 21504, 22016, 22528, 23040, 23552];
        let sizes = [16384, 4096, 1024, 512, 512, 512, 512, 512];
        let heights = [2, 1, 0, 0, 0, 0, 0, 0];
        for layers in [1, 2, 6, 12] {
            let words = [
                0x58d24926,
                address as u32,
                (address >> 32) as u32 | (3 << 21),
                (7 << 28) | (2 << 3),
                127 | (1 << 22)
                    | ((if layers == 1 {
                        1
                    } else if layers == 6 {
                        3
                    } else if layers == 12 {
                        8
                    } else {
                        5
                    }) << 23),
                127 | ((if layers >= 6 {
                    layers / 6 - 1
                } else {
                    layers - 1
                }) << 16)
                    | (1 << 31),
                0x03000000,
                7 << 4,
            ];
            let mut memory_access = crate::test_memory::access(address_space.memory().machine());
            let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
            let mut builder = super::ResourceBuilder::new(
                address_space.memory(),
                None,
                &mut retained_cache,
                &mut memory_access,
            );
            builder
                .sampled_image(
                    super::MaxwellThreeDTextureReference::new(4, 0, 0),
                    if layers == 1 {
                        MaxwellThreeDTextureDimension::Two
                    } else if layers == 6 {
                        MaxwellThreeDTextureDimension::Cube
                    } else if layers == 12 {
                        MaxwellThreeDTextureDimension::CubeArray
                    } else {
                        MaxwellThreeDTextureDimension::TwoArray
                    },
                    0,
                    descriptor_bytes(words),
                )
                .unwrap();
            let super::MaxwellThreeDResolvedResource::Image(image) = &builder.resources[0] else {
                panic!("expected sampled image")
            };
            assert_eq!(image.description().mip_levels(), 8);
            assert_eq!(image.description().array_layers(), layers as u16);
            assert_eq!(
                image.description().dimension(),
                if layers >= 6 {
                    nixe_gpu::ImageDimension::Cube
                } else {
                    nixe_gpu::ImageDimension::Two
                }
            );
            assert_eq!(
                image.source().size(),
                if layers == 1 {
                    0x5e00
                } else {
                    0x6000 * u64::from(layers)
                }
            );
            assert_eq!(image.view().bindings().len(), 8 * layers as usize);
            for layer in 0..layers {
                for mip in 0..8 {
                    let binding = &image.view().bindings()[layer as usize * 8 + mip];
                    assert_eq!(binding.subresources().mip_level, mip as u8);
                    assert_eq!(binding.subresources().base_layer, layer as u16);
                    assert_eq!(
                        binding.backing().allocation_offset(),
                        layer as u64 * 0x6000 + offsets[mip]
                    );
                    assert_eq!(binding.backing().mapping().bytes(), sizes[mip]);
                    assert_eq!(
                        binding.layout(),
                        ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                            block_width_log2: 0,
                            block_height_log2: heights[mip],
                            block_depth_log2: 0,
                            layer_stride: sizes[mip],
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn bc1_sampled_image_resolves_only_its_compressed_block_storage() {
        let machine = crate::test_memory::machine();
        let mut address_space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocation = crate::test_memory::allocate(&machine, 0x8000);
        let backing = crate::test_memory::ranges(&allocation);
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                size: backing.iter().map(|range| range.bytes()).sum::<u64>(),
                backing,
                backing_offset: 0,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        let words = [
            0x78d2_4924,
            address as u32,
            ((address >> 32) as u32) | (3 << 21),
            3 << 3,
            255 | (1 << 23),
            255 | (1 << 31),
            0,
            0,
        ];
        let mut memory_access = crate::test_memory::access(address_space.memory().machine());
        let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
        let mut builder = super::ResourceBuilder::new(
            address_space.memory(),
            None,
            &mut retained_cache,
            &mut memory_access,
        );
        builder
            .sampled_image(
                super::MaxwellThreeDTextureReference::new(4, 0, 0),
                MaxwellThreeDTextureDimension::Two,
                0,
                descriptor_bytes(words),
            )
            .unwrap();
        let super::MaxwellThreeDResolvedResource::Image(image) = &builder.resources[0] else {
            panic!("expected sampled image");
        };
        assert_eq!(
            image.description.format(),
            nixe_gpu::ImageFormat::Bc1RgbUnorm
        );
        assert_eq!(image.view.bindings()[0].backing().mapping().bytes(), 0x8000);
        assert!(matches!(
            image.view.bindings()[0].layout(),
            ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                layer_stride: 0x8000,
                ..
            })
        ));
        let mut bc4_words = words;
        bc4_words[0] = 0x7492_4927;
        let mut bc4_builder_access = crate::test_memory::access(address_space.memory().machine());
        let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
        let mut bc4_builder = super::ResourceBuilder::new(
            address_space.memory(),
            None,
            &mut retained_cache,
            &mut bc4_builder_access,
        );
        bc4_builder
            .sampled_image(
                super::MaxwellThreeDTextureReference::new(4, 0, 0),
                MaxwellThreeDTextureDimension::Two,
                0,
                descriptor_bytes(bc4_words),
            )
            .unwrap();
        let super::MaxwellThreeDResolvedResource::Image(bc4) = &bc4_builder.resources[0] else {
            panic!("expected BC4 sampled image");
        };
        assert_eq!(bc4.description().format(), nixe_gpu::ImageFormat::Bc4RUnorm);
        assert_eq!(
            bc4.view().swizzle(),
            nixe_gpu::Swizzle([
                nixe_gpu::ComponentSwizzle::Red,
                nixe_gpu::ComponentSwizzle::Red,
                nixe_gpu::ComponentSwizzle::Red,
                nixe_gpu::ComponentSwizzle::One,
            ])
        );
        assert_eq!(bc4.view().bindings()[0].backing().mapping().bytes(), 0x8000);
        // Fetch-only access must not read a sampler descriptor or require its
        // selector. Even garbage TSC bits in the handle are irrelevant.
        crate::test_memory::write(&allocation, 0, &descriptor_bytes(words)).unwrap();
        crate::test_memory::write(&allocation, 0x100, &0xfff0_0000_u32.to_le_bytes()).unwrap();
        let mut memory_access = crate::test_memory::access(address_space.memory().machine());
        let mut retained_cache = MaxwellThreeDRetainedBackingCache::default();
        let mut builder = super::ResourceBuilder::new(
            address_space.memory(),
            None,
            &mut retained_cache,
            &mut memory_access,
        );
        builder
            .descriptor_pool(
                MaxwellThreeDResourceRole::TextureHeaders,
                Some(address),
                Some(0xfffff),
            )
            .unwrap();
        let reference = super::MaxwellThreeDTextureReference::new(4, 0, 0);
        let handle = builder
            .texture_handle(reference, Some((address + 0x100, 4)), None)
            .unwrap();
        builder
            .sampled_texture(
                None,
                None,
                reference,
                handle,
                MaxwellThreeDTextureDimension::Two,
                false,
            )
            .unwrap();
        assert_eq!(builder.resources.len(), 1); // only the actual sampled image
        assert!(builder.samplers.is_empty());
        assert_eq!(builder.descriptor_reads.len(), 2); // handle and TIC, no TSC
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let unchanged = builder.descriptor_reads[0]
            .refresh(
                builder.address_space,
                builder.access,
                &mut builder.retained_backings.descriptor_tables,
                &mut builder.address_space.read_batch(),
            )
            .unwrap();
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert!(unchanged);
        assert_eq!(allocations, 0);
        crate::test_memory::write(&allocation, 0x100, &0_u32.to_le_bytes()).unwrap();
        assert!(
            !builder.descriptor_reads[0]
                .refresh(
                    builder.address_space,
                    builder.access,
                    &mut builder.retained_backings.descriptor_tables,
                    &mut builder.address_space.read_batch(),
                )
                .unwrap()
        );
    }

    #[test]
    fn retained_translation_cache_is_bounded_and_rejects_unmapped_generations() {
        let machine = crate::test_memory::machine();
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let mut space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: crate::test_memory::ranges(&allocation),
                backing_offset: 0,
                size: 4096,
                allocation_alignment: 4096,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset();
        let permissions = MemoryPermissions::READ;
        let source = space
            .memory()
            .resolve_range(address, 32, permissions)
            .unwrap();
        let mut cache = MaxwellThreeDRetainedBackingCache {
            entries: nixe_gpu::WorkingSet::new(2, 64 * 1024),
            ..Default::default()
        };
        let retained = cache
            .retain(&source, MaxwellThreeDResourceRole::TextureHeaders)
            .unwrap();
        assert!(
            cache
                .resolve_cached(space.memory(), address, 32, permissions)
                .is_some()
        );
        ALLOCATIONS.with(|n| n.set(Some(0)));
        for _ in 0..1000 {
            std::hint::black_box(
                cache
                    .resolve_cached(space.memory(), address, 32, permissions)
                    .unwrap(),
            );
        }
        assert_eq!(ALLOCATIONS.with(|n| n.replace(None).unwrap()), 0);
        crate::test_memory::write(&allocation, 0, &[7]).unwrap();
        assert!(
            cache
                .resolve_cached(space.memory(), address, 32, permissions)
                .is_some()
        );
        space.unmap(address).unwrap();
        assert!(
            cache
                .resolve_cached(space.memory(), address, 32, permissions)
                .is_none()
        );
        assert!(
            space
                .memory()
                .resolve_range(address, 32, permissions)
                .is_err()
        );
        let replacement = crate::test_memory::allocate(&machine, 4096);
        space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(2),
                backing: crate::test_memory::ranges(&replacement),
                backing_offset: 0,
                size: 4096,
                allocation_alignment: 4096,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        assert!(
            cache
                .resolve_cached(space.memory(), address, 32, permissions)
                .is_none()
        );
        for offset in [0, 64, 128] {
            let source = space
                .memory()
                .resolve_range(
                    space.address(address.get() + offset).unwrap(),
                    32,
                    permissions,
                )
                .unwrap();
            cache
                .retain(&source, MaxwellThreeDResourceRole::TextureHeaders)
                .unwrap();
        }
        assert_eq!(cache.entries.usage().0, 2);
        assert!(
            cache
                .resolve_cached(space.memory(), address, 32, permissions)
                .is_none()
        );
        let mut old = [0; 1];
        read_backing_bytes(retained.backing.mapping(), 0, &mut old).unwrap();
        assert_eq!(old, [7]);
    }

    #[test]
    fn retained_backing_cache_reuses_topology_across_writes_and_remaps() {
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 0x2000);
        let mut address_space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let backing = crate::test_memory::ranges(&allocation);
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                size: backing.iter().map(|range| range.bytes()).sum::<u64>(),
                backing,
                backing_offset: 0,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let source = address_space
            .memory()
            .resolve_range(mapping.offset(), 0x1000, MemoryPermissions::READ)
            .unwrap();
        let role = MaxwellThreeDResourceRole::VertexStream(0);
        let mut cache = MaxwellThreeDRetainedBackingCache::default();

        let first = cache.retain(&source, role).unwrap();
        let descriptor = [0x5a; 32];
        crate::test_memory::write(&allocation, 0, &descriptor).unwrap();
        assert_eq!(
            read_descriptor_bytes(first.backing.mapping(), 0).unwrap(),
            descriptor
        );

        crate::test_memory::write(&allocation, 0x1000, &[1]).unwrap();
        let after_disjoint_write = cache.retain(&source, role).unwrap();
        assert!(Arc::ptr_eq(&first.mappings, &after_disjoint_write.mappings));

        crate::test_memory::write(&allocation, 0, &[2]).unwrap();
        let after_overlapping_write = cache.retain(&source, role).unwrap();
        assert!(Arc::ptr_eq(
            &after_disjoint_write.mappings,
            &after_overlapping_write.mappings
        ));
        assert!(
            first
                .backing
                .mapping()
                .eq(after_overlapping_write.backing.mapping())
        );
        assert_eq!(
            read_descriptor_bytes(after_overlapping_write.backing.mapping(), 0).unwrap()[0],
            2
        );

        address_space.unmap(mapping.offset()).unwrap();
        let replacement = crate::test_memory::allocate(&machine, 0x2000);
        let replacement_backing = crate::test_memory::ranges(&replacement);
        let remapped = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(2),
                size: replacement_backing
                    .iter()
                    .map(|range| range.bytes())
                    .sum::<u64>(),
                backing: replacement_backing,
                backing_offset: 0,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        assert_eq!(remapped.offset(), mapping.offset());
        let remapped_source = address_space
            .memory()
            .resolve_range(remapped.offset(), 0x1000, MemoryPermissions::READ)
            .unwrap();
        let after_remap = cache.retain(&remapped_source, role).unwrap();
        assert!(!Arc::ptr_eq(
            &after_overlapping_write.mappings,
            &after_remap.mappings
        ));
    }
}
