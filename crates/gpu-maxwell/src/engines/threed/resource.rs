//! Atomic resource resolution for immutable `MAXWELL_B` state snapshots.
//!
//! Maxwell GPU virtual addresses and PTE kinds are interpreted here, while
//! canonical storage and neutral resource views retain their own identities.
//! No host layout, swizzle operation, or backend object crosses this boundary.

use std::{
    collections::BTreeSet,
    fmt::{Display, Formatter},
};

use nixe_gpu::{
    BackingView, BlockLinearLayout, BufferDescription, BufferId, BufferView,
    GpuAllocationDescription, GpuAllocationId, GpuVirtualAddress, ImageDescription, ImageDimension,
    ImageExtent, ImageFormat, ImageId, ImageKind, ImageMemoryLayout, ImageSubresourceRange,
    ImageView, SampleCount, Swizzle,
};
use nixe_memory::{CanonicalBackingRange, CanonicalRangeError, MemoryPermissions};

use crate::{
    MaxwellAllocationId, MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellMappingId,
    MaxwellResolvedRange,
};

use super::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MaxwellThreeDAttachmentReadiness,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetState, MaxwellThreeDDepthStencilFormat,
    MaxwellThreeDDepthStencilTargetState, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDImageKind, MaxwellThreeDImageLayout,
    MaxwellThreeDSampleMode, MaxwellThreeDState, MaxwellThreeDUnresolvedAddress,
};

// Public Switch NVIDIA memory kinds. Compressed color/depth kinds remain
// unsupported until their layout semantics are modeled rather than treated as
// generic block-linear storage.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h#L18-L250
const MAXWELL_PITCH_KIND: u8 = 0x00;
const MAXWELL_PITCH_NO_SWIZZLE_KIND: u8 = 0xfd;
const MAXWELL_GENERIC_BLOCK_LINEAR_KIND: u8 = 0xfe;
const MAXWELL_DESCRIPTOR_SIZE: u64 = 32;

/// Frontend role of one completely resolved resource.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellThreeDResourceRole {
    VertexStream(u8),
    IndexBuffer,
    ConstantBuffer { group: u8, slot: u8 },
    TextureHeaders,
    Samplers,
    ColorTarget(u8),
    DepthStencilTarget,
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
    mappings: Box<[MaxwellThreeDMappingReference]>,
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
    mappings: Box<[MaxwellThreeDMappingReference]>,
    guest_layout: MaxwellThreeDPreservedImageLayout,
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

    fn backing(&self) -> &CanonicalBackingRange {
        match self {
            Self::Buffer(value) => value.view.backing().range(),
            Self::Image(value) => value.view.bindings()[0].backing().range(),
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
    let sample_mode = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::SampleMode)
        .value()
        .and_then(|value| match value {
            MaxwellThreeDFixedFunctionValue::SampleMode(value) => Some(*value),
            _ => None,
        });
    let mut builder = ResourceBuilder::new(address_space, sample_mode);

    for (index, stream) in state.vertex_input().streams().iter().enumerate() {
        if stream
            .format()
            .value()
            .is_some_and(|format| format.enabled())
        {
            let address = stream
                .address()
                .ok_or(MaxwellThreeDResourceError::IncompleteState {
                    role: MaxwellThreeDResourceRole::VertexStream(index as u8),
                })?;
            let limit = stream
                .limit()
                .ok_or(MaxwellThreeDResourceError::IncompleteState {
                    role: MaxwellThreeDResourceRole::VertexStream(index as u8),
                })?;
            builder.buffer(
                MaxwellThreeDResourceRole::VertexStream(index as u8),
                address,
                inclusive_size(address, limit)?,
            )?;
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
    if index_programmed {
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
            inclusive_size(address, limit)?,
        )?;
    }

    let bindings = state.shader_bindings();
    for group in 0..MAXWELL_BIND_GROUP_COUNT {
        for slot in 0..MAXWELL_CONSTANT_BUFFER_SLOT_COUNT {
            let Some(binding) = bindings.groups()[group].constant_buffers()[slot] else {
                continue;
            };
            if !binding.enabled() {
                continue;
            }
            let role = MaxwellThreeDResourceRole::ConstantBuffer {
                group: group as u8,
                slot: slot as u8,
            };
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
    builder.descriptor_pool(
        MaxwellThreeDResourceRole::TextureHeaders,
        bindings.texture_headers(),
    )?;
    builder.descriptor_pool(MaxwellThreeDResourceRole::Samplers, bindings.samplers())?;

    for (index, target) in state.render_targets().color().iter().enumerate() {
        match target.readiness(true) {
            MaxwellThreeDAttachmentReadiness::Unprogrammed
            | MaxwellThreeDAttachmentReadiness::Disabled => {}
            MaxwellThreeDAttachmentReadiness::Ready => builder.color_target(index as u8, target)?,
            _ => {
                return Err(MaxwellThreeDResourceError::IncompleteState {
                    role: MaxwellThreeDResourceRole::ColorTarget(index as u8),
                });
            }
        }
    }
    if depth_is_programmed(state.render_targets().depth_stencil()) {
        builder.depth_target(state.render_targets().depth_stencil())?;
    }

    builder.finish()
}

struct ResourceBuilder<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    sample_mode: Option<MaxwellThreeDSampleMode>,
    resources: Vec<MaxwellThreeDResolvedResource>,
}

struct RetainedResourceBacking {
    range: CanonicalBackingRange,
    allocation: MaxwellAllocationId,
    allocation_offset: u64,
    mappings: Box<[MaxwellThreeDMappingReference]>,
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
    ) -> Self {
        Self {
            address_space,
            sample_mode,
            resources: Vec::new(),
        }
    }

    fn buffer(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: MaxwellThreeDUnresolvedAddress,
        size: u64,
    ) -> Result<(), MaxwellThreeDResourceError> {
        let source = self.resolve(address, size, MemoryPermissions::READ, role)?;
        let retained = retained_backing(&source)?;
        let allocation_description = GpuAllocationDescription::new(allocation_size(&source)?, 1)
            .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        let backing = BackingView::new(
            GpuAllocationId::new(retained.allocation.get()),
            allocation_description,
            retained.allocation_offset,
            retained.range,
        )
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        let description = BufferDescription::new(size)
            .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        let id = BufferId::new(resource_id(self.resources.len())?);
        let view = BufferView::new(id, description, 0, backing)
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
        let format = match target.format().value().copied() {
            Some(MaxwellThreeDColorTargetFormat::Color(0xcf)) => ImageFormat::Bgra8Unorm,
            Some(MaxwellThreeDColorTargetFormat::Color(0xd5)) => ImageFormat::Rgba8Unorm,
            Some(format) => {
                return Err(MaxwellThreeDResourceError::UnsupportedColorFormat {
                    role,
                    format: format.raw(),
                });
            }
            None => return Err(MaxwellThreeDResourceError::IncompleteState { role }),
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
        let format = match target.format().value() {
            Some(MaxwellThreeDDepthStencilFormat::Z16) => ImageFormat::Depth16Unorm,
            Some(MaxwellThreeDDepthStencilFormat::Z24Stencil8) => {
                ImageFormat::Depth24UnormStencil8Uint
            }
            Some(MaxwellThreeDDepthStencilFormat::ZFloat32) => ImageFormat::Depth32Float,
            Some(MaxwellThreeDDepthStencilFormat::ZFloat32X24Stencil8) => {
                ImageFormat::Depth32FloatStencil8Uint
            }
            Some(format) => {
                return Err(MaxwellThreeDResourceError::UnsupportedDepthFormat {
                    role,
                    format: *format as u32,
                });
            }
            None => return Err(MaxwellThreeDResourceError::IncompleteState { role }),
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
        let (dimension, depth, layers) = match kind {
            MaxwellThreeDImageKind::Array => (
                ImageDimension::Two,
                1,
                u16::try_from(third)
                    .map_err(|_| MaxwellThreeDResourceError::ArithmeticOverflow { role })?,
            ),
            MaxwellThreeDImageKind::ThreeDimensional => (ImageDimension::Three, third, 1),
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
            0,
        )
    }

    fn image(
        &mut self,
        role: MaxwellThreeDResourceRole,
        address: MaxwellThreeDUnresolvedAddress,
        description: ImageDescription,
        layout: MaxwellThreeDImageLayout,
        array_pitch: Option<u32>,
        selected_layer: u16,
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
        let (neutral_layout, layer_stride, expected_kind) = match layout {
            MaxwellThreeDImageLayout::PitchLinear => {
                let stride = array_pitch
                    .filter(|value| *value != 0)
                    .map(u64::from)
                    .unwrap_or(compact_layer);
                (
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: compact_row,
                        layer_stride: stride,
                    },
                    stride,
                    MAXWELL_PITCH_KIND,
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
                let stride = array_pitch
                    .filter(|value| *value != 0)
                    .map(u64::from)
                    .unwrap_or(minimum);
                (
                    ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                        block_width_log2: 0,
                        block_height_log2,
                        block_depth_log2,
                        layer_stride: stride,
                    }),
                    stride,
                    MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
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
        let layer_count = if description.dimension() == ImageDimension::Three {
            1
        } else if selected_layer == 0 && description.kind() == ImageKind::DepthStencil {
            description.array_layers()
        } else {
            1
        };
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
                || !layout_accepts_kind(layout, segment.mapping().kind())
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
        let retained = retained_backing(&source)?;
        let allocation_description = GpuAllocationDescription::new(allocation_size(&source)?, 1)
            .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
        let backing = BackingView::new(
            GpuAllocationId::new(retained.allocation.get()),
            allocation_description,
            retained.allocation_offset,
            retained.range,
        )
        .map_err(|_| MaxwellThreeDResourceError::InvalidNeutralView { role })?;
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
                backing,
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
                guest_layout: MaxwellThreeDPreservedImageLayout {
                    layout: neutral_layout,
                    pte_kind: actual_kind,
                },
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
                if ranges_overlap(
                    self.resources[left].backing(),
                    self.resources[right].backing(),
                ) {
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
            aliases: aliases.into_boxed_slice(),
            dirty: MaxwellThreeDDirtySubresources::default(),
        };
        result.validate_mappings(self.address_space)?;
        Ok(result)
    }
}

fn retained_backing(
    source: &MaxwellResolvedRange,
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
        let range = mapping
            .backing()
            .snapshot_subrange(segment.backing_offset(), segment.size())
            .map_err(MaxwellThreeDResourceError::Canonical)?;
        canonical.extend_from_slice(range.segments());
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
    Ok(RetainedResourceBacking {
        range: CanonicalBackingRange::new(canonical)
            .map_err(MaxwellThreeDResourceError::Canonical)?,
        allocation,
        allocation_offset,
        mappings: mappings.into_boxed_slice(),
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
) -> Result<u64, MaxwellThreeDResourceError> {
    limit
        .get()
        .checked_sub(address.get())
        .and_then(|size| size.checked_add(1))
        .ok_or(MaxwellThreeDResourceError::ResourceExhausted)
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

fn layout_accepts_kind(layout: MaxwellThreeDImageLayout, kind: u8) -> bool {
    match layout {
        MaxwellThreeDImageLayout::PitchLinear => {
            matches!(kind, MAXWELL_PITCH_KIND | MAXWELL_PITCH_NO_SWIZZLE_KIND)
        }
        MaxwellThreeDImageLayout::BlockLinear { .. } => kind == MAXWELL_GENERIC_BLOCK_LINEAR_KIND,
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

fn ranges_overlap(first: &CanonicalBackingRange, second: &CanonicalBackingRange) -> bool {
    first.segments().iter().any(|left| {
        second.segments().iter().any(|right| {
            left.page() == right.page()
                && left.offset() < right.offset() + right.size()
                && right.offset() < left.offset() + left.size()
        })
    })
}

fn contradictory_image_alias(
    first: &MaxwellThreeDResolvedResource,
    second: &MaxwellThreeDResolvedResource,
) -> bool {
    match (first, second) {
        (
            MaxwellThreeDResolvedResource::Image(first),
            MaxwellThreeDResolvedResource::Image(second),
        ) => first.description != second.description || first.guest_layout != second.guest_layout,
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
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell resource resolution exhausted host bookkeeping")
            }
        }
    }
}

impl std::error::Error for MaxwellThreeDResourceError {}
