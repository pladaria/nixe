//! Atomic resource resolution for immutable `MAXWELL_B` state snapshots.
//!
//! Maxwell GPU virtual addresses and PTE kinds are interpreted here, while
//! canonical storage and neutral resource views retain their own identities.
//! No host layout, swizzle operation, or backend object crosses this boundary.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{
    AddressMode, BackingView, BlockLinearLayout, BufferDescription, BufferId, BufferView,
    FilterMode, GpuAllocationDescription, GpuAllocationId, GpuVirtualAddress, ImageDescription,
    ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind, ImageMemoryLayout,
    ImageSubresourceRange, ImageView, SampleCount, SamplerDescription, Swizzle,
};
use nixe_memory::{
    CanonicalBackingRange, CanonicalCpuWriteDependency, CanonicalRangeAccessError,
    CanonicalRangeError, CanonicalWriteBatch, CanonicalWriteBatchError, MemoryPermissions,
};

use crate::{
    MaxwellAllocationId, MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellMappingId,
    MaxwellResolvedRange,
};

use super::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MaxwellThreeDAttachmentReadiness,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetState, MaxwellThreeDDepthStencilFormat,
    MaxwellThreeDDepthStencilTargetState, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDImageKind, MaxwellThreeDImageLayout,
    MaxwellThreeDSampleMode, MaxwellThreeDSamplerBindingMode, MaxwellThreeDState,
    MaxwellThreeDUnresolvedAddress,
};

// Public Switch NVIDIA memory kinds. Compressed color/depth kinds remain
// unsupported until their layout semantics are modeled rather than treated as
// generic block-linear storage.
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
// Public Switch kind table names 0x17 as S8Z24_2CZ; deko3d selects it for
// compressed, single-sample S8Z24 depth images.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h#L35-L45
// https://github.com/devkitPro/deko3d/blob/6ee80db52aac0168303fc2f6417232997e464999/source/maxwell/image_formats.cpp#L35-L47
const MAXWELL_S8Z24_2CZ_KIND: u8 = 0x17;
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
    TextureHeaders,
    Samplers,
    SampledImage {
        texture: MaxwellThreeDTextureReference,
        dimension: MaxwellThreeDTextureDimension,
    },
    Sampler(MaxwellThreeDTextureReference),
    ColorTarget(u8),
    DepthStencilTarget,
}

/// Shader-visible dimensionality required from a Maxwell sampled-image TIC.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellThreeDTextureDimension {
    Two,
    TwoArray,
}

/// Draw-local location of a raw Maxwell TIC/TSC handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaxwellThreeDTextureReference {
    group: u8,
    constant_buffer_slot: u8,
    byte_offset: u32,
}

impl MaxwellThreeDTextureReference {
    #[must_use]
    pub const fn new(group: u8, constant_buffer_slot: u8, byte_offset: u32) -> Self {
        Self {
            group,
            constant_buffer_slot,
            byte_offset,
        }
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
    generation: nixe_memory::MappingGeneration,
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
    pub const fn generation(self) -> nixe_memory::MappingGeneration {
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

    pub(super) fn shared_mappings(&self) -> Arc<[MaxwellThreeDMappingReference]> {
        Arc::clone(&self.mappings)
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
    Color(MaxwellThreeDColorTargetFormat),
    DepthStencil(MaxwellThreeDDepthStencilFormat),
    Texture(u32),
}

/// Source-preserving neutral interpretation of one Maxwell TSC entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedSampler {
    role: MaxwellThreeDResourceRole,
    min_filter: FilterMode,
    mag_filter: FilterMode,
    mip_filter: FilterMode,
    address_modes: [AddressMode; 3],
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
        self.compression_enabled
    }

    /// Returns whether the guest bytes already have a direct canonical
    /// representation. Disabled compression is direct regardless of the
    /// compressible PTE family; with compression enabled, only generic 16Bx2
    /// mappings are already canonical.
    #[must_use]
    pub const fn has_direct_canonical_representation(self) -> bool {
        !self.compression_enabled || self.pte_kind == MAXWELL_GENERIC_BLOCK_LINEAR_KIND
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
    cpu_writes: Option<CanonicalCpuWriteDependency>,
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
    pub(super) fn shared_mappings(&self) -> Arc<[MaxwellThreeDMappingReference]> {
        Arc::clone(&self.mappings)
    }
    pub(super) fn cpu_write_dependency(&self) -> Option<&CanonicalCpuWriteDependency> {
        self.cpu_writes.as_ref()
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

    fn backing_view(&self) -> &BackingView {
        match self {
            Self::Buffer(value) => value.view.backing(),
            Self::Image(value) => value.view.bindings()[0].backing(),
        }
    }

    fn source(&self) -> &MaxwellResolvedRange {
        match self {
            Self::Buffer(value) => &value.source,
            Self::Image(value) => &value.source,
        }
    }
}

/// Canonical alias relationship retained independently of either GPU VA.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDResourceAlias {
    first: u16,
    second: u16,
}

impl MaxwellThreeDResourceAlias {
    #[must_use]
    pub const fn first(self) -> usize {
        self.first as usize
    }
    #[must_use]
    pub const fn second(self) -> usize {
        self.second as usize
    }
}

/// Dirty subresources remain guest-layout resources until a later boundary
/// explicitly converts, uploads, downloads, or clears them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDDirtySubresources {
    entries: BTreeSet<MaxwellThreeDDirtySubresource>,
}

/// One dirty plane, mip level, and layer range of a resolved image.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaxwellThreeDDirtySubresource {
    resource: u16,
    subresources: ImageSubresourceRange,
}

impl MaxwellThreeDDirtySubresource {
    #[must_use]
    pub const fn resource(self) -> usize {
        self.resource as usize
    }
    #[must_use]
    pub const fn subresources(self) -> ImageSubresourceRange {
        self.subresources
    }
}

impl MaxwellThreeDDirtySubresources {
    fn mark(&mut self, resource: u16, subresources: ImageSubresourceRange) {
        self.entries.insert(MaxwellThreeDDirtySubresource {
            resource,
            subresources,
        });
    }
    pub fn clear(&mut self, resource: usize) {
        if let Ok(resource) = u16::try_from(resource) {
            self.entries.retain(|entry| entry.resource != resource);
        }
    }
    #[must_use]
    pub fn contains(&self, resource: usize) -> bool {
        u16::try_from(resource)
            .is_ok_and(|resource| self.entries.iter().any(|entry| entry.resource == resource))
    }
    #[must_use]
    pub fn entries(&self) -> impl ExactSizeIterator<Item = MaxwellThreeDDirtySubresource> + '_ {
        self.entries.iter().copied()
    }
}

/// Immutable all-or-nothing interpretation of one 3D state snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDResolvedResources {
    address_space_generation: nixe_memory::MappingGeneration,
    resources: Box<[MaxwellThreeDResolvedResource]>,
    samplers: Box<[MaxwellThreeDResolvedSampler]>,
    aliases: Box<[MaxwellThreeDResourceAlias]>,
    dirty: MaxwellThreeDDirtySubresources,
}

impl MaxwellThreeDResolvedResources {
    #[must_use]
    pub const fn address_space_generation(&self) -> nixe_memory::MappingGeneration {
        self.address_space_generation
    }
    #[must_use]
    pub fn resources(&self) -> &[MaxwellThreeDResolvedResource] {
        &self.resources
    }
    #[must_use]
    pub fn samplers(&self) -> &[MaxwellThreeDResolvedSampler] {
        &self.samplers
    }
    #[must_use]
    pub fn aliases(&self) -> &[MaxwellThreeDResourceAlias] {
        &self.aliases
    }
    #[must_use]
    pub const fn dirty_subresources(&self) -> &MaxwellThreeDDirtySubresources {
        &self.dirty
    }
    /// Marks one resolved image dirty while keeping its preserved guest
    /// layout. Buffer indices and unknown indices are rejected.
    pub fn mark_image_dirty(&mut self, resource: usize) -> Result<(), MaxwellThreeDResourceError> {
        let dirty_resource =
            u16::try_from(resource).map_err(|_| MaxwellThreeDResourceError::ResourceExhausted)?;
        match self.resources.get(resource) {
            Some(MaxwellThreeDResolvedResource::Image(image)) => {
                for binding in image.view.bindings() {
                    self.dirty.mark(dirty_resource, binding.subresources());
                }
                Ok(())
            }
            Some(MaxwellThreeDResolvedResource::Buffer(_)) => {
                Err(MaxwellThreeDResourceError::NotAnImage { resource })
            }
            None => Err(MaxwellThreeDResourceError::UnknownResource { resource }),
        }
    }

    pub fn clear_image_dirty(&mut self, resource: usize) {
        self.dirty.clear(resource);
    }

    /// Revalidates the complete retained mapping set before any consumer uses
    /// a resource prefix.
    pub fn validate_mappings(
        &self,
        address_space: &MaxwellGpuAddressSpace,
    ) -> Result<(), MaxwellThreeDResourceError> {
        for resource in &self.resources {
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

/// Resolves every referenced range and constructs no visible prefix on error.
pub fn resolve_maxwell_three_d_resources(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
    resolve_maxwell_three_d_resources_for_roles_with_staged_writes(
        state,
        address_space,
        &[],
        None,
        true,
    )
}

/// Resolves only the explicitly consumed resource roles.
///
/// Unlike [`resolve_maxwell_three_d_resources`], unrelated partially
/// programmed state is outside this operation-scoped boundary. Every named
/// role remains strict when its state is consumed.
pub fn resolve_maxwell_three_d_resources_for_roles(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    required_roles: &[MaxwellThreeDResourceRole],
) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
    resolve_maxwell_three_d_resources_for_roles_with_staged_writes(
        state,
        address_space,
        required_roles,
        None,
        false,
    )
}

pub(crate) fn resolve_maxwell_three_d_resources_for_roles_with_staged_writes(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    required_roles: &[MaxwellThreeDResourceRole],
    staged_writes: Option<&CanonicalWriteBatch>,
    inspect_complete_state: bool,
) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
    resolve_maxwell_three_d_resources_for_roles_with_staged_writes_and_cache(
        state,
        address_space,
        required_roles,
        staged_writes,
        inspect_complete_state,
        None,
    )
}

pub(crate) fn resolve_maxwell_three_d_resources_for_roles_with_staged_writes_and_cache(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    required_roles: &[MaxwellThreeDResourceRole],
    staged_writes: Option<&CanonicalWriteBatch>,
    inspect_complete_state: bool,
    retained_backings: Option<&mut MaxwellThreeDRetainedBackingCache>,
) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
    let sample_mode = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::SampleMode)
        .value()
        .and_then(|value| match value {
            MaxwellThreeDFixedFunctionValue::SampleMode(value) => Some(*value),
            _ => None,
        });
    let mut builder =
        ResourceBuilder::new(address_space, sample_mode, staged_writes, retained_backings);

    for (index, stream) in state.vertex_input().streams().iter().enumerate() {
        let role = MaxwellThreeDResourceRole::VertexStream(index as u8);
        if (inspect_complete_state || required_roles.contains(&role))
            && stream
                .format()
                .value()
                .is_some_and(|format| format.enabled())
        {
            let address = stream
                .address()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
            let limit = stream
                .limit()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
            builder.buffer(role, address, inclusive_size(address, limit, role)?)?;
        }
    }

    let index = state.vertex_input().index();
    let index_programmed = [
        index.address_upper().raw(),
        index.address_lower().raw(),
        index.limit_upper().raw(),
        index.limit_lower().raw(),
    ]
    .iter()
    .any(Option::is_some);
    let index_required = required_roles.contains(&MaxwellThreeDResourceRole::IndexBuffer);
    // MME shadow state may contain an unrelated partial binding. Complete-state
    // inspection validates any programmed prefix; operation-scoped resolution
    // requires the binding only for a trigger that explicitly names the role.
    if (inspect_complete_state && index_programmed) || index_required {
        let address = unresolved(index.address_upper().value(), index.address_lower().value())
            .ok_or(MaxwellThreeDResourceError::IncompleteState {
                role: MaxwellThreeDResourceRole::IndexBuffer,
            })?;
        let limit = unresolved(index.limit_upper().value(), index.limit_lower().value()).ok_or(
            MaxwellThreeDResourceError::IncompleteState {
                role: MaxwellThreeDResourceRole::IndexBuffer,
            },
        )?;
        builder.buffer(
            MaxwellThreeDResourceRole::IndexBuffer,
            address,
            inclusive_size(address, limit, MaxwellThreeDResourceRole::IndexBuffer)?,
        )?;
    }

    let bindings = state.shader_bindings();
    for group in 0..MAXWELL_BIND_GROUP_COUNT {
        for slot in 0..MAXWELL_CONSTANT_BUFFER_SLOT_COUNT {
            let role = MaxwellThreeDResourceRole::ConstantBuffer {
                group: group as u8,
                slot: slot as u8,
            };
            if !inspect_complete_state
                && !constant_buffer_is_required(required_roles, group as u8, slot as u8)
            {
                continue;
            }
            let Some(binding) = bindings.groups()[group].constant_buffers()[slot] else {
                continue;
            };
            if !binding.enabled() {
                continue;
            }
            let address = binding
                .address()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
            let size = u64::from(
                binding
                    .size()
                    .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
            );
            builder.buffer(role, address, size)?;
        }
    }
    let texture_headers_required = required_roles.iter().any(|role| {
        matches!(
            role,
            MaxwellThreeDResourceRole::TextureHeaders
                | MaxwellThreeDResourceRole::SampledImage { .. }
        )
    });
    if inspect_complete_state || texture_headers_required {
        builder.descriptor_pool(
            MaxwellThreeDResourceRole::TextureHeaders,
            bindings.texture_headers(),
        )?;
    }
    let samplers_required = required_roles.iter().any(|role| {
        matches!(
            role,
            MaxwellThreeDResourceRole::Samplers | MaxwellThreeDResourceRole::Sampler(_)
        )
    });
    if inspect_complete_state || samplers_required {
        builder.descriptor_pool(MaxwellThreeDResourceRole::Samplers, bindings.samplers())?;
    }
    let mut descriptors = BTreeMap::new();
    for role in required_roles {
        if let MaxwellThreeDResourceRole::SampledImage { texture, dimension } = role {
            if let Some(previous) = descriptors.insert(*texture, *dimension)
                && previous != *dimension
            {
                return Err(MaxwellThreeDResourceError::ContradictoryState { role: *role });
            }
        }
    }
    for (texture_reference, dimension) in descriptors {
        if !required_roles.contains(&MaxwellThreeDResourceRole::Sampler(texture_reference)) {
            return Err(MaxwellThreeDResourceError::IncompleteState {
                role: MaxwellThreeDResourceRole::Sampler(texture_reference),
            });
        }
        builder.sampled_texture(bindings, texture_reference, dimension)?;
    }

    for (index, target) in state.render_targets().color().iter().enumerate() {
        let role = MaxwellThreeDResourceRole::ColorTarget(index as u8);
        if !inspect_complete_state && !required_roles.contains(&role) {
            continue;
        }
        match target.readiness(true) {
            MaxwellThreeDAttachmentReadiness::Unprogrammed
            | MaxwellThreeDAttachmentReadiness::Disabled => {}
            MaxwellThreeDAttachmentReadiness::Ready => builder.color_target(index as u8, target)?,
            _ => {
                return Err(MaxwellThreeDResourceError::IncompleteState { role });
            }
        }
    }
    let depth_required = required_roles.contains(&MaxwellThreeDResourceRole::DepthStencilTarget);
    let depth_explicitly_unselected = state.render_targets().depth_target_count().value()
        == Some(&super::MaxwellThreeDDepthTargetCount::None);
    if depth_required
        || (inspect_complete_state
            && !depth_explicitly_unselected
            && depth_is_programmed(state.render_targets().depth_stencil()))
    {
        builder.depth_target(state.render_targets().depth_stencil())?;
    }

    builder.finish()
}

struct ResourceBuilder<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    staged_writes: Option<&'a CanonicalWriteBatch>,
    retained_backings: Option<&'a mut MaxwellThreeDRetainedBackingCache>,
    sample_mode: Option<MaxwellThreeDSampleMode>,
    resources: Vec<MaxwellThreeDResolvedResource>,
    samplers: Vec<MaxwellThreeDResolvedSampler>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct RetainedBackingCacheEntry {
    source: MaxwellResolvedRange,
    retained: RetainedResourceBacking,
}

/// Transactional cache for page-versioned backing views derived from stable
/// Maxwell mappings. CPU-write epochs make the common validation path O(1)
/// per backing store. Lost journal history conservatively rebuilds the entry
/// from authoritative page generations.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct MaxwellThreeDRetainedBackingCache {
    entries: Arc<BTreeMap<RetainedBackingKey, Arc<RetainedBackingCacheEntry>>>,
}

impl MaxwellThreeDRetainedBackingCache {
    fn retain(
        &mut self,
        source: &MaxwellResolvedRange,
        role: MaxwellThreeDResourceRole,
    ) -> Result<RetainedResourceBacking, MaxwellThreeDResourceError> {
        let key = RetainedBackingKey::from(source);
        if let Some(entry) = self.entries.get(&key)
            && entry.source == *source
            && entry
                .retained
                .cpu_writes
                .as_ref()
                .is_some_and(CanonicalCpuWriteDependency::remains_current)
        {
            return Ok(entry.retained.clone());
        }

        let retained = retained_backing(source, role)?;
        if retained.cpu_writes.is_none() {
            Arc::make_mut(&mut self.entries).remove(&key);
            return Ok(retained);
        }
        Arc::make_mut(&mut self.entries).insert(
            key,
            Arc::new(RetainedBackingCacheEntry {
                source: source.clone(),
                retained: retained.clone(),
            }),
        );
        Ok(retained)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RetainedResourceBacking {
    backing: BackingView,
    allocation_description: GpuAllocationDescription,
    mappings: Arc<[MaxwellThreeDMappingReference]>,
    cpu_writes: Option<CanonicalCpuWriteDependency>,
}

struct MaxwellImageDescriptionRequest {
    dimension: ImageDimension,
    extent: ImageExtent,
    format: ImageFormat,
    kind: ImageKind,
    layers: u16,
    role: MaxwellThreeDResourceRole,
}

impl<'a> ResourceBuilder<'a> {
    fn new(
        address_space: &'a MaxwellGpuAddressSpace,
        sample_mode: Option<MaxwellThreeDSampleMode>,
        staged_writes: Option<&'a CanonicalWriteBatch>,
        retained_backings: Option<&'a mut MaxwellThreeDRetainedBackingCache>,
    ) -> Self {
        Self {
            address_space,
            staged_writes,
            retained_backings,
            sample_mode,
            resources: Vec::new(),
            samplers: Vec::new(),
        }
    }

    fn retained_backing(
        &mut self,
        source: &MaxwellResolvedRange,
        role: MaxwellThreeDResourceRole,
    ) -> Result<RetainedResourceBacking, MaxwellThreeDResourceError> {
        match self.retained_backings.as_deref_mut() {
            Some(cache) => cache.retain(source, role),
            None => retained_backing(source, role),
        }
    }

    fn buffer(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: MaxwellThreeDUnresolvedAddress,
        size: u64,
    ) -> Result<(), MaxwellThreeDResourceError> {
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
        Ok(())
    }

    fn descriptor_pool(
        &mut self,
        role: MaxwellThreeDResourceRole,
        pool: &super::MaxwellThreeDDescriptorPoolState,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let programmed = pool.address_upper().raw().is_some()
            || pool.address_lower().raw().is_some()
            || pool.maximum_index().raw().is_some();
        if !programmed {
            return Ok(());
        }
        let address = pool
            .address()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let maximum = *pool
            .maximum_index()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let size = u64::from(maximum)
            .checked_add(1)
            .and_then(|count| count.checked_mul(MAXWELL_DESCRIPTOR_SIZE))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        self.buffer(role, address, size)
    }

    fn descriptor_bytes(
        &self,
        role: MaxwellThreeDResourceRole,
        index: u32,
    ) -> Result<[u8; 32], MaxwellThreeDResourceError> {
        let buffer = self
            .resources
            .iter()
            .find_map(|resource| match resource {
                MaxwellThreeDResolvedResource::Buffer(buffer) if buffer.role() == role => {
                    Some(buffer)
                }
                _ => None,
            })
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let offset = u64::from(index)
            .checked_mul(MAXWELL_DESCRIPTOR_SIZE)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        if offset
            .checked_add(MAXWELL_DESCRIPTOR_SIZE)
            .is_none_or(|end| end > buffer.description().size())
        {
            return Err(MaxwellThreeDResourceError::DescriptorIndexOutOfRange { role, index });
        }
        read_descriptor_bytes(buffer.view().backing().range(), offset, self.staged_writes)
    }

    fn sampled_texture(
        &mut self,
        bindings: &super::MaxwellThreeDShaderBindingState,
        texture_reference: MaxwellThreeDTextureReference,
        dimension: MaxwellThreeDTextureDimension,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let raw_handle = self.texture_handle(texture_reference)?;
        // An unprogrammed selector retains the Maxwell class reset mode. Only
        // an explicit false selects the legacy texture-header interpretation;
        // the TIC decoder below independently requires a Maxwell v3 header.
        // The public class definition confirms this is a one-bit selector:
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1210-L1213
        if !uses_maxwell_texture_headers(bindings.maxwell_texture_headers().value()) {
            return Err(MaxwellThreeDResourceError::UnsupportedTextureBindingMode {
                descriptor: raw_handle,
            });
        }
        // Maxwell resource handles use TIC in bits 0..20 and TSC in bits
        // 20..32 when the tables are independent. Via-header mode uses the
        // same raw index for both tables. Public corroborating definitions:
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/include/deko3d.h#L711-L724
        // https://source.hodakov.me/hdkv/yuzu/src/commit/55bf3dbf5ddaa3f7c1c3efade5553b07499fe289/src/video_core/textures/texture.h#L147-L165
        let (image_index, sampler_index) = match bindings.sampler_binding().value() {
            Some(mode) => texture_descriptor_pair(*mode, raw_handle),
            None => {
                return Err(MaxwellThreeDResourceError::IncompleteState {
                    role: MaxwellThreeDResourceRole::Sampler(texture_reference),
                });
            }
        };
        let tic = self.descriptor_bytes(MaxwellThreeDResourceRole::TextureHeaders, image_index)?;
        let tsc = self.descriptor_bytes(MaxwellThreeDResourceRole::Samplers, sampler_index)?;
        self.sampled_image(texture_reference, dimension, image_index, tic)?;
        self.samplers
            .push(decode_sampler(texture_reference, sampler_index, tsc)?);
        Ok(())
    }

    fn texture_handle(
        &self,
        texture_reference: MaxwellThreeDTextureReference,
    ) -> Result<u32, MaxwellThreeDResourceError> {
        let role = MaxwellThreeDResourceRole::ConstantBuffer {
            group: texture_reference.group,
            slot: texture_reference.constant_buffer_slot,
        };
        let buffer = self
            .resources
            .iter()
            .find_map(|resource| match resource {
                MaxwellThreeDResolvedResource::Buffer(buffer) if buffer.role() == role => {
                    Some(buffer)
                }
                _ => None,
            })
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
        let end = u64::from(texture_reference.byte_offset)
            .checked_add(4)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        if end > buffer.description().size() {
            return Err(MaxwellThreeDResourceError::TextureHandleOutOfRange {
                texture_reference,
                constant_buffer_size: buffer.description().size(),
            });
        }
        let mut bytes = [0_u8; 4];
        read_backing_bytes(
            buffer.view().backing().range(),
            u64::from(texture_reference.byte_offset),
            &mut bytes,
            self.staged_writes,
        )?;
        Ok(u32::from_le_bytes(bytes))
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
        if header_version != 3
            || texture_type != required_texture_type
            || (required_dimension == MaxwellThreeDTextureDimension::Two && depth != 1)
            || block_width_log2 != 0
            || block_depth_log2 != 0
            || mip_max != 0
            || view_mip_min != 0
            || view_mip_max != 0
            || msaa != 0
            || sparse
            || !normalized
            || layer_base != 0
        {
            return Err(MaxwellThreeDResourceError::UnsupportedTextureDescriptor {
                descriptor: descriptor_index,
                field: "2D/2D-array block-linear shape",
                value: words[2] ^ words[3] ^ words[4] ^ words[5] ^ words[7],
            });
        }
        let address = u64::from(words[1]) | (u64::from(words[2] & 0xffff) << 32);
        let row = align_up(u64::from(width) * 4, 64, role)?;
        let rows = align_up(u64::from(height), 8_u64 << block_height_log2, role)?;
        let layer_stride = row
            .checked_mul(rows)
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let unresolved = MaxwellThreeDUnresolvedAddress::new((address >> 32) as u8, address as u32);
        let size = layer_stride
            .checked_mul(u64::from(layers))
            .ok_or(MaxwellThreeDResourceError::ArithmeticOverflow { role })?;
        let source = self.resolve(unresolved, size, MemoryPermissions::READ, role)?;
        let actual_kind = source
            .segments()
            .first()
            .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?
            .mapping()
            .kind();
        if source
            .segments()
            .iter()
            .any(|segment| segment.mapping().kind() != MAXWELL_GENERIC_BLOCK_LINEAR_KIND)
        {
            return Err(MaxwellThreeDResourceError::UnsupportedKind {
                role,
                expected: MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
                actual: actual_kind,
            });
        }
        let description = image_description(MaxwellImageDescriptionRequest {
            dimension: ImageDimension::Two,
            extent: ImageExtent {
                width,
                height,
                depth: 1,
            },
            format,
            kind: ImageKind::Color,
            layers,
            role,
        })?;
        let retained = self.retained_backing(&source, role)?;
        let allocation_description = retained.allocation_description;
        let cpu_writes = retained.cpu_writes.clone();
        let layout = ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2,
            block_height_log2,
            block_depth_log2,
            layer_stride,
        });
        let id = ImageId::new(resource_id(self.resources.len())?);
        let view = ImageView::new(
            id,
            description,
            Swizzle::IDENTITY,
            vec![(
                ImageSubresourceRange {
                    plane: 0,
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: layers,
                },
                layout,
                retained.backing,
            )],
        )
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
                cpu_writes,
                guest_layout: MaxwellThreeDPreservedImageLayout {
                    layout,
                    pte_kind: actual_kind,
                    compression_enabled: false,
                },
                guest_format: MaxwellThreeDGuestImageFormat::Texture(format_word),
            },
        ));
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
        // NVIDIA names both packed encodings separately, while their logical
        // components are the same 24-bit depth plus 8-bit stencil pair. Keep
        // `guest_format` so later layout conversion can distinguish packing.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L947-L974
        let format = match guest_format {
            MaxwellThreeDColorTargetFormat::Color(0xcf) => ImageFormat::Bgra8Unorm,
            MaxwellThreeDColorTargetFormat::Color(0xd5) => ImageFormat::Rgba8Unorm,
            format => {
                return Err(MaxwellThreeDResourceError::UnsupportedColorFormat {
                    role,
                    format: format.raw(),
                });
            }
        };
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
        )
    }

    fn depth_target(
        &mut self,
        target: &MaxwellThreeDDepthStencilTargetState,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let role = MaxwellThreeDResourceRole::DepthStencilTarget;
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
        let third = u32::from(
            *target
                .third_dimension()
                .value()
                .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?,
        );
        let kind = *target
            .kind()
            .value()
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?;
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
            MaxwellThreeDImageKind::ThreeDimensional => {
                if target.layer().value().is_some_and(|layer| *layer != 0) {
                    return Err(MaxwellThreeDResourceError::ContradictoryState { role });
                }
                (ImageDimension::Three, third, 1, 0)
            }
        };
        let description = image_description(MaxwellImageDescriptionRequest {
            dimension,
            extent: ImageExtent {
                width,
                height,
                depth,
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
    ) -> Result<(), MaxwellThreeDResourceError> {
        match self
            .sample_mode
            .ok_or(MaxwellThreeDResourceError::IncompleteState { role })?
        {
            MaxwellThreeDSampleMode::Samples1x1 => {}
            mode => return Err(MaxwellThreeDResourceError::UnsupportedSampleMode { role, mode }),
        }
        let bpp = u64::from(
            description
                .format()
                .plane_bytes_per_texel(0)
                .ok_or(MaxwellThreeDResourceError::UnsupportedImageLayout { role })?,
        );
        let extent = description.extent();
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
                let stride = array_pitch
                    .filter(|value| *value != 0)
                    .unwrap_or(compact_layer);
                (
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: compact_row,
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
                let compressed_color_kind = match guest_format {
                    MaxwellThreeDGuestImageFormat::Color(_) => {
                        single_sample_compressed_color_kind(bpp)
                    }
                    _ => None,
                };
                let expected_kind = compressed_color_kind.unwrap_or_else(|| {
                    if compression_enabled
                        && guest_format
                            == MaxwellThreeDGuestImageFormat::DepthStencil(
                                MaxwellThreeDDepthStencilFormat::Stencil8Z24,
                            )
                    {
                        MAXWELL_S8Z24_2CZ_KIND
                    } else {
                        MAXWELL_GENERIC_BLOCK_LINEAR_KIND
                    }
                });
                (
                    ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                        block_width_log2: 0,
                        block_height_log2,
                        block_depth_log2,
                        layer_stride: stride,
                    }),
                    stride,
                    expected_kind,
                    compressed_color_kind.is_some(),
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
        let source = self.resolve(resolved_address, size, MemoryPermissions::WRITE, role)?;
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
        let cpu_writes = retained.cpu_writes.clone();
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
                access: MaxwellThreeDResourceAccess::Write,
                description,
                allocation_description,
                view,
                source,
                mappings: retained.mappings,
                cpu_writes,
                guest_layout: MaxwellThreeDPreservedImageLayout {
                    layout: neutral_layout,
                    pte_kind: actual_kind,
                    compression_enabled,
                },
                guest_format,
            },
        ));
        Ok(())
    }

    fn resolve(
        &self,
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
        self.address_space
            .resolve_range(address, size, permissions)
            .map_err(|error| MaxwellThreeDResourceError::Resolution { role, error })
    }

    fn finish(self) -> Result<MaxwellThreeDResolvedResources, MaxwellThreeDResourceError> {
        let mut aliases = Vec::new();
        for right in 0..self.resources.len() {
            for left in 0..right {
                if self.resources[left]
                    .backing_view()
                    .overlaps(self.resources[right].backing_view())
                {
                    if contradictory_image_alias(&self.resources[left], &self.resources[right]) {
                        return Err(MaxwellThreeDResourceError::ContradictoryAlias {
                            first: self.resources[left].role(),
                            second: self.resources[right].role(),
                        });
                    }
                    aliases.push(MaxwellThreeDResourceAlias {
                        first: u16::try_from(left)
                            .map_err(|_| MaxwellThreeDResourceError::ResourceExhausted)?,
                        second: u16::try_from(right)
                            .map_err(|_| MaxwellThreeDResourceError::ResourceExhausted)?,
                    });
                }
            }
        }
        let result = MaxwellThreeDResolvedResources {
            address_space_generation: self.address_space.mapping_generation(),
            resources: self.resources.into_boxed_slice(),
            samplers: self.samplers.into_boxed_slice(),
            aliases: aliases.into_boxed_slice(),
            dirty: MaxwellThreeDDirtySubresources::default(),
        };
        result.validate_mappings(self.address_space)?;
        Ok(result)
    }
}

fn constant_buffer_is_required(
    required_roles: &[MaxwellThreeDResourceRole],
    group: u8,
    slot: u8,
) -> bool {
    required_roles.iter().any(|role| match role {
        MaxwellThreeDResourceRole::ConstantBuffer {
            group: required_group,
            slot: required_slot,
        } => *required_group == group && *required_slot == slot,
        MaxwellThreeDResourceRole::SampledImage { texture, .. }
        | MaxwellThreeDResourceRole::Sampler(texture) => {
            texture.group == group && texture.constant_buffer_slot == slot
        }
        _ => false,
    })
}

/// Decodes only Maxwell TIC format/component/swizzle tuples with a direct
/// neutral and wgpu representation. Numeric component and swizzle values are
/// defined by deko3d's pinned public Maxwell tables:
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
fn decode_sampled_texture_format(
    image_format: u8,
    components: [u8; 4],
    swizzle: [u8; 4],
    srgb: bool,
    reserved: u32,
) -> Option<ImageFormat> {
    if reserved != 0 {
        return None;
    }
    match (image_format, components, swizzle, srgb) {
        (0x1d, [2, 2, 2, 2], [2, 0, 0, 7], false) => Some(ImageFormat::R8Unorm),
        (0x18, [2, 2, 2, 2], [2, 3, 0, 7], false) => Some(ImageFormat::Rg8Unorm),
        (0x08, [2, 2, 2, 2], [2, 3, 4, 5], false) => Some(ImageFormat::Rgba8Unorm),
        (0x08, [2, 2, 2, 2], [2, 3, 4, 5], true) => Some(ImageFormat::Rgba8Srgb),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], false) => Some(ImageFormat::Bgra8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], true) => Some(ImageFormat::Bgra8Srgb),
        (0x1b, [7, 7, 7, 7], [2, 0, 0, 7], false) => Some(ImageFormat::R16Float),
        (0x0c, [7, 7, 7, 7], [2, 3, 0, 7], false) => Some(ImageFormat::Rg16Float),
        (0x03, [7, 7, 7, 7], [2, 3, 4, 5], false) => Some(ImageFormat::Rgba16Float),
        (0x0f, [7, 7, 7, 7], [2, 0, 0, 7], false) => Some(ImageFormat::R32Float),
        (0x04, [7, 7, 7, 7], [2, 3, 0, 7], false) => Some(ImageFormat::Rg32Float),
        (0x01, [7, 7, 7, 7], [2, 3, 4, 5], false) => Some(ImageFormat::Rgba32Float),
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
    let mut canonical = Vec::new();
    let mut mappings = Vec::new();
    for segment in source.segments() {
        let mapping = segment.mapping();
        if mapping.allocation() != allocation || segment.backing_offset() != expected_offset {
            return Err(MaxwellThreeDResourceError::DiscontiguousAllocation);
        }
        mapping
            .backing()
            .snapshot_subrange_into(segment.backing_offset(), segment.size(), &mut canonical)
            .map_err(MaxwellThreeDResourceError::Canonical)?;
        mappings.push(MaxwellThreeDMappingReference {
            mapping: mapping.id(),
            generation: mapping.generation(),
            allocation,
            backing_offset: segment.backing_offset(),
            size: segment.size(),
            kind: mapping.kind(),
        });
        expected_offset = expected_offset
            .checked_add(segment.size())
            .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?;
    }
    let range =
        CanonicalBackingRange::new(canonical).map_err(MaxwellThreeDResourceError::Canonical)?;
    let cpu_writes = CanonicalCpuWriteDependency::capture(&range);
    let allocation_description = GpuAllocationDescription::new(allocation_size(source)?, 1)
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
    let backing = BackingView::new(
        GpuAllocationId::new(allocation.get()),
        allocation_description,
        allocation_offset,
        range,
    )
    .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
    Ok(RetainedResourceBacking {
        backing,
        allocation_description,
        mappings: mappings.into(),
        cpu_writes,
    })
}

fn allocation_size(source: &MaxwellResolvedRange) -> Result<u64, MaxwellThreeDResourceError> {
    let first = source
        .segments()
        .first()
        .ok_or(MaxwellThreeDResourceError::ResourceExhausted)?;
    let size = first.mapping().backing().size();
    if source
        .segments()
        .iter()
        .any(|segment| segment.mapping().backing().size() != size)
    {
        return Err(MaxwellThreeDResourceError::DiscontiguousAllocation);
    }
    Ok(size)
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

fn read_descriptor_bytes(
    range: &CanonicalBackingRange,
    offset: u64,
    staged_writes: Option<&CanonicalWriteBatch>,
) -> Result<[u8; 32], MaxwellThreeDResourceError> {
    let mut bytes = [0; 32];
    read_backing_bytes(range, offset, &mut bytes, staged_writes)?;
    Ok(bytes)
}

fn read_backing_bytes(
    range: &CanonicalBackingRange,
    offset: u64,
    bytes: &mut [u8],
    staged_writes: Option<&CanonicalWriteBatch>,
) -> Result<(), MaxwellThreeDResourceError> {
    if let Some(staged_writes) = staged_writes {
        staged_writes
            .read_staged(range, offset, bytes)
            .map_err(MaxwellThreeDResourceError::StagedCanonicalAccess)?;
    } else {
        range
            .read(offset, bytes)
            .map_err(MaxwellThreeDResourceError::CanonicalAccess)?;
    }
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
    if words[0] & (1 << 9) != 0 || ((words[1] >> 10) & 3) != 0 {
        return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "depth comparison or reduction filter",
            value: words[0] ^ words[1],
        });
    }
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
    if signed_bias != 0 {
        return Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
            descriptor,
            field: "LOD bias",
            value: bias as u32,
        });
    }
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
    Ok(MaxwellThreeDResolvedSampler {
        role,
        min_filter: filter((words[1] >> 4) & 3, "minification filter")?,
        mag_filter: filter(words[1] & 3, "magnification filter")?,
        mip_filter,
        address_modes: [
            address_mode(words[0] & 7)?,
            address_mode((words[0] >> 3) & 7)?,
            address_mode((words[0] >> 6) & 7)?,
        ],
        lod_min_fixed,
        lod_max_fixed,
        max_anisotropy,
    })
}

fn depth_is_programmed(target: &MaxwellThreeDDepthStencilTargetState) -> bool {
    // SET_Z_COMPRESSION configures an operation performed on an attachment; it
    // does not describe or bind one. Keep it out of this presence test so a
    // selector write cannot fabricate an incomplete depth resource.
    [
        target.address_upper().raw(),
        target.address_lower().raw(),
        target.format().raw(),
        target.layout().raw(),
        target.width().raw(),
        target.height().raw(),
        target.third_dimension().raw(),
        target.kind().raw(),
    ]
    .iter()
    .any(Option::is_some)
}

fn contradictory_image_alias(
    first: &MaxwellThreeDResolvedResource,
    second: &MaxwellThreeDResolvedResource,
) -> bool {
    match (first, second) {
        (
            MaxwellThreeDResolvedResource::Image(first),
            MaxwellThreeDResolvedResource::Image(second),
        ) => {
            first.description != second.description
                || first.guest_layout != second.guest_layout
                || first.guest_format != second.guest_format
        }
        _ => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDResourceError {
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
        generation: nixe_memory::MappingGeneration,
    },
    UnsupportedColorFormat {
        role: MaxwellThreeDResourceRole,
        format: u8,
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
    DiscontiguousAllocation,
    ContradictoryAlias {
        first: MaxwellThreeDResourceRole,
        second: MaxwellThreeDResourceRole,
    },
    InvalidNeutralView {
        role: MaxwellThreeDResourceRole,
    },
    UnknownResource {
        resource: usize,
    },
    NotAnImage {
        resource: usize,
    },
    Canonical(CanonicalRangeError),
    CanonicalAccess(CanonicalRangeAccessError),
    StagedCanonicalAccess(CanonicalWriteBatchError),
    ResourceExhausted,
}

impl Display for MaxwellThreeDResourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
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
            Self::DiscontiguousAllocation => formatter
                .write_str("one resource range crosses non-contiguous allocation identities"),
            Self::ContradictoryAlias { first, second } => write!(
                formatter,
                "aliased image resources have contradictory interpretations: first={first:?} second={second:?}"
            ),
            Self::InvalidNeutralView { role } => write!(
                formatter,
                "resolved Maxwell resource cannot form a neutral view: {role:?}"
            ),
            Self::UnknownResource { resource } => {
                write!(
                    formatter,
                    "unknown resolved Maxwell resource: index={resource}"
                )
            }
            Self::NotAnImage { resource } => {
                write!(
                    formatter,
                    "resolved Maxwell resource is not an image: index={resource}"
                )
            }
            Self::Canonical(error) => {
                write!(formatter, "canonical resource snapshot failed: {error}")
            }
            Self::CanonicalAccess(error) => {
                write!(formatter, "canonical descriptor read failed: {error}")
            }
            Self::StagedCanonicalAccess(error) => {
                write!(formatter, "staged descriptor read failed: {error}")
            }
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
    use nixe_memory::{CanonicalAllocation, CanonicalWriteBatch, MemoryPermissions};

    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };

    use super::{
        MAXWELL_C32_2CRA_KIND, MAXWELL_C64_2CRA_KIND, MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
        MaxwellThreeDPreservedImageLayout, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
        MaxwellThreeDRetainedBackingCache, MaxwellThreeDTextureDimension,
        constant_buffer_is_required, decode_sampled_texture_format, decode_sampler,
        image_kind_matches, read_backing_bytes, read_descriptor_bytes, texture_descriptor_pair,
        uses_maxwell_texture_headers,
    };

    fn descriptor_bytes(words: [u32; 8]) -> [u8; 32] {
        let mut bytes = [0; 32];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
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
    fn tsc_depth_comparison_is_rejected_instead_of_losing_semantics() {
        let mut words = [0; 8];
        words[0] = 1 << 9;
        let texture_reference = super::MaxwellThreeDTextureReference::new(4, 3, 0x20);

        assert!(matches!(
            decode_sampler(texture_reference, 8, descriptor_bytes(words)),
            Err(MaxwellThreeDResourceError::UnsupportedSamplerDescriptor {
                descriptor: 8,
                field: "depth comparison or reduction filter",
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
    fn descriptor_reads_see_ordered_staged_uploads_without_publishing_them() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut writes = CanonicalWriteBatch::new();
        let expected = descriptor_bytes([
            0x58d2_4908,
            0x093d_7000,
            0x0060_0000,
            0x0007_0020,
            0xe880_00ff,
            0x8000_00ff,
            0x0300_0000,
            0,
        ]);
        writes.stage(&range, 8 * 32, &expected).unwrap();

        assert_eq!(
            read_descriptor_bytes(&range, 8 * 32, Some(&writes)).unwrap(),
            expected
        );
        assert_eq!(
            read_descriptor_bytes(&range, 8 * 32, None).unwrap(),
            [0; 32]
        );
    }

    #[test]
    fn bindless_texture_handle_reads_the_staged_constant_buffer_value() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut writes = CanonicalWriteBatch::new();
        writes
            .stage(&range, 0x20, &0xabc0_0008_u32.to_le_bytes())
            .unwrap();
        let mut staged = [0; 4];
        let mut published = [0; 4];

        read_backing_bytes(&range, 0x20, &mut staged, Some(&writes)).unwrap();
        read_backing_bytes(&range, 0x20, &mut published, None).unwrap();

        assert_eq!(u32::from_le_bytes(staged), 0xabc0_0008);
        assert_eq!(published, [0; 4]);
    }

    #[test]
    fn sampled_resources_require_the_constant_buffer_that_stores_their_handle() {
        let texture = super::MaxwellThreeDTextureReference::new(4, 0, 0x690);
        let roles = [
            MaxwellThreeDResourceRole::SampledImage {
                texture,
                dimension: MaxwellThreeDTextureDimension::TwoArray,
            },
            MaxwellThreeDResourceRole::Sampler(texture),
            MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 2 },
        ];

        assert!(constant_buffer_is_required(&roles, 4, 0));
        assert!(constant_buffer_is_required(&roles, 0, 2));
        assert!(!constant_buffer_is_required(&roles, 4, 1));
        assert!(!constant_buffer_is_required(&roles, 3, 0));
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
    fn retained_backing_cache_uses_store_epoch_then_exact_page_overlap() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                size: backing.size(),
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
            .resolve_range(mapping.offset(), 0x1000, MemoryPermissions::READ)
            .unwrap();
        let role = MaxwellThreeDResourceRole::VertexStream(0);
        let mut cache = MaxwellThreeDRetainedBackingCache::default();

        let first = cache.retain(&source, role).unwrap();
        let staged_descriptor = [0x5a; 32];
        let mut staged_writes = CanonicalWriteBatch::new();
        staged_writes
            .stage(
                source.segments()[0].mapping().backing(),
                0,
                &staged_descriptor,
            )
            .unwrap();
        assert_eq!(
            read_descriptor_bytes(first.backing.range(), 0, Some(&staged_writes)).unwrap(),
            staged_descriptor
        );
        assert_eq!(
            read_descriptor_bytes(first.backing.range(), 0, None).unwrap(),
            [0; 32]
        );

        allocation.write(0x1000, &[1]).unwrap();
        let after_disjoint_write = cache.retain(&source, role).unwrap();
        assert!(Arc::ptr_eq(&first.mappings, &after_disjoint_write.mappings));

        allocation.write(0, &[2]).unwrap();
        let after_overlapping_write = cache.retain(&source, role).unwrap();
        assert!(!Arc::ptr_eq(
            &after_disjoint_write.mappings,
            &after_overlapping_write.mappings
        ));
        assert_eq!(
            after_overlapping_write.backing.range().segments()[0].content_generation(),
            source.segments()[0].mapping().backing().segments()[0]
                .content_generation()
                .next()
                .unwrap()
        );

        address_space.unmap(mapping.offset()).unwrap();
        let replacement = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let replacement_backing = replacement
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let remapped = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(2),
                size: replacement_backing.size(),
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
            .resolve_range(remapped.offset(), 0x1000, MemoryPermissions::READ)
            .unwrap();
        let after_remap = cache.retain(&remapped_source, role).unwrap();
        assert!(!Arc::ptr_eq(
            &after_overlapping_write.mappings,
            &after_remap.mappings
        ));
    }
}
