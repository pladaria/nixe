//! Atomic lowering of validated `MAXWELL_B` clear and draw snapshots.
//!
//! This boundary produces only backend-independent `nixe-gpu` resources and
//! operations. Shader translation is supplied as typed T10 evidence; this
//! module never treats Maxwell code as a neutral or host shader.

use std::{
    cell::Cell,
    collections::HashMap,
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, AlphaCompareOperation, AlphaTest, AttachmentLoad,
    AttachmentStore, BackendCapabilities, BackendCapabilityError, BackendResourceCreateInfo,
    BarrierOperation, BufferId, BufferRange, BufferRegion, BufferView, CapabilityRequirements,
    ClearOperation, ClearValue, CommandDescriptionError, DepthCompareOperation, DepthState,
    DescriptorKind, DescriptorTableBinding, DescriptorTableDescription, DescriptorTableId,
    DrawArguments, DrawOperation, FrontendSubmissionId, GpuCacheConfiguration, GpuCommand,
    GpuOperation, ImageId, ImageOrigin, ImageRegion, ImageSubresourceRange, ImageView,
    OperationSubmission, PipelineDescription, PipelineId, PipelineKind, PipelineStages,
    PreparedDraw, PrimitiveTopology, RenderAttachment, RenderPassDescription, RenderPassId,
    RenderPassOperation, ResourceAccess, ResourceDependency, ResourceTransition, ResourceUsage,
    SamplerId, ShaderDescription, ShaderId, ShaderResourceKind, ShaderStage, TriangleRasterization,
    VertexAttribute, VertexBufferLayout, VertexComponentCount, VertexComponentWidth, VertexFormat,
    VertexStepMode, ViewportTransform,
};
use nixe_memory::CanonicalCpuWriteDependency;

use crate::MaxwellMethodSource;
use crate::shader::{
    MaxwellShaderTranslationError, MaxwellShaderTranslationInputs,
    MaxwellShaderTranslationSourceKey, MaxwellStagedShaderWrite, MaxwellTranslatedShaderProgram,
    prepare_maxwell_shader_translation_inputs_from_source,
    prepare_maxwell_shader_translation_source, translate_prepared_maxwell_shader_programs,
};
#[cfg(debug_assertions)]
use crate::shader::{MaxwellShaderTranslationKey, MaxwellShaderTranslationSource};

use super::{
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAlphaToCoverageOverride,
    MaxwellThreeDAntiAliasedLineEnable, MaxwellThreeDApiMandatedEarlyZ, MaxwellThreeDBegin,
    MaxwellThreeDBlendEnableCommon, MaxwellThreeDClipIdTestEnable,
    MaxwellThreeDColorCompressionMode, MaxwellThreeDColorReductionThresholdsEnable,
    MaxwellThreeDCompareOp, MaxwellThreeDConditionalLoadConstantBuffer,
    MaxwellThreeDConservativeRasterEnable, MaxwellThreeDCoverageToColor, MaxwellThreeDCsaaEnable,
    MaxwellThreeDDirectlyAddressableMemory, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFillViaTriangleMode, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDHybridAntiAliasControl,
    MaxwellThreeDIteratedBlend, MaxwellThreeDLogicOp, MaxwellThreeDPatchSize,
    MaxwellThreeDPixelShaderClampRange, MaxwellThreeDPixelShaderInterlockControl,
    MaxwellThreeDPointCenterMode, MaxwellThreeDPointSpriteSelect,
    MaxwellThreeDPolygonClipGeneratedEdge, MaxwellThreeDPolygonMode,
    MaxwellThreeDPostZPixelShaderImask, MaxwellThreeDProvokingVertex,
    MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderTargetIndexOffset,
    MaxwellThreeDRenderTargetLayer, MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources,
    MaxwellThreeDResourceRole, MaxwellThreeDSampleLocationGroup, MaxwellThreeDSeparateFragmentData,
    MaxwellThreeDShadeMode, MaxwellThreeDShaderLocalMemoryPerWarpSize, MaxwellThreeDShaderStage,
    MaxwellThreeDState, MaxwellThreeDTextureDimension, MaxwellThreeDTirControl,
    MaxwellThreeDTirMode, MaxwellThreeDVertexNumericalType, MaxwellThreeDViewportCoordinateSwizzle,
    MaxwellThreeDViewportPixelCenter, MaxwellThreeDViewportScaleOffsetEnable,
};

#[derive(Clone, Debug)]
struct DrawAttachmentSelection {
    colors: Vec<(u8, usize)>,
    depth_stencil: Option<usize>,
}

impl DrawAttachmentSelection {
    fn attachment_indices(&self) -> Vec<usize> {
        self.attachment_indices_iter().collect()
    }

    fn attachment_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.colors
            .iter()
            .map(|(_, index)| *index)
            .chain(self.depth_stencil)
    }

    fn color_targets(&self) -> impl Iterator<Item = u8> + '_ {
        self.colors.iter().map(|(target, _)| *target)
    }
}

/// One execution trigger retained at its exact method location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDOperationTrigger {
    ClearSurface {
        source: MaxwellMethodSource,
    },
    DrawVertexArray {
        source: MaxwellMethodSource,
        vertex_count: u32,
    },
}

impl MaxwellThreeDOperationTrigger {
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::ClearSurface { source } | Self::DrawVertexArray { source, .. } => source,
        }
    }

    /// Appends the resources consumed directly by this trigger.
    ///
    /// Shader translation contributes constant-buffer, texture, and sampler
    /// roles separately. Keeping the trigger-specific selection exhaustive
    /// ensures that future indexed draws cannot silently inherit non-indexed
    /// resolution.
    pub(crate) fn append_resource_roles(
        self,
        state: &MaxwellThreeDState,
        roles: &mut Vec<MaxwellThreeDResourceRole>,
    ) {
        match self {
            Self::ClearSurface { .. } => {
                if let Some(surface) = state.render_targets().clear().last_surface().value() {
                    if surface.color_mask() != 0 {
                        roles.push(MaxwellThreeDResourceRole::ColorTarget(
                            surface.color_target(),
                        ));
                    }
                    if surface.depth() || surface.stencil() {
                        roles.push(MaxwellThreeDResourceRole::DepthStencilTarget);
                    }
                }
            }
            Self::DrawVertexArray { .. } => {
                for attribute in state.vertex_input().attributes() {
                    if let Some(attribute) = attribute.value().filter(|value| value.enabled()) {
                        let role = MaxwellThreeDResourceRole::VertexStream(attribute.stream());
                        if !roles.contains(&role) {
                            roles.push(role);
                        }
                    }
                }
                if let Some(selection) = state.render_targets().color_target_selection().value() {
                    roles.extend(
                        selection
                            .active_targets()
                            .iter()
                            .copied()
                            .map(MaxwellThreeDResourceRole::ColorTarget),
                    );
                }
                if draw_depth_stencil_resource_required(state) {
                    roles.push(MaxwellThreeDResourceRole::DepthStencilTarget);
                }
            }
        }
    }
}

/// Stable evidence that T10 translated one enabled Maxwell shader stage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDTranslatedShader {
    stage: ShaderStage,
    shader: ShaderId,
    cache_fingerprint: u128,
    directly_addressable_memory: Option<MaxwellThreeDDirectlyAddressableMemory>,
    maximum_api_visible_calls: u16,
}

impl MaxwellThreeDTranslatedShader {
    #[must_use]
    pub(crate) const fn new(
        stage: ShaderStage,
        shader: ShaderId,
        cache_fingerprint: u128,
        directly_addressable_memory: Option<MaxwellThreeDDirectlyAddressableMemory>,
        maximum_api_visible_calls: u16,
    ) -> Self {
        Self {
            stage,
            shader,
            cache_fingerprint,
            directly_addressable_memory,
            maximum_api_visible_calls,
        }
    }
    #[must_use]
    pub const fn stage(self) -> ShaderStage {
        self.stage
    }
    #[must_use]
    pub const fn shader(self) -> ShaderId {
        self.shader
    }
    /// Guest shader-memory configuration consumed by this shader, if any.
    /// This is never inferred from host cache topology or unrelated state.
    #[must_use]
    pub const fn directly_addressable_memory(
        self,
    ) -> Option<MaxwellThreeDDirectlyAddressableMemory> {
        self.directly_addressable_memory
    }

    /// Conservative maximum established by T10 for the Maxwell calls whose
    /// execution is governed by `SET_API_VISIBLE_CALL_LIMIT`.
    #[must_use]
    pub const fn maximum_api_visible_calls(self) -> u16 {
        self.maximum_api_visible_calls
    }
}

/// Shader-declared use of one already resolved frontend resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDShaderResourceUse {
    role: MaxwellThreeDResourceRole,
    binding: u8,
    kind: DescriptorKind,
    stages: PipelineStages,
    usage: Option<ResourceUsage>,
}

impl MaxwellThreeDShaderResourceUse {
    pub fn new(
        role: MaxwellThreeDResourceRole,
        binding: u8,
        kind: DescriptorKind,
        stages: PipelineStages,
        usage: Option<ResourceUsage>,
    ) -> Result<Self, MaxwellThreeDLoweringError> {
        if let Some(usage) = usage {
            let _ = AccessScope::new(stages, AccessMode::Read, usage)
                .map_err(|_| MaxwellThreeDLoweringError::InvalidShaderResourceUse { role })?;
        } else if kind != DescriptorKind::Sampler {
            return Err(MaxwellThreeDLoweringError::InvalidShaderResourceUse { role });
        }
        Ok(Self {
            role,
            binding,
            kind,
            stages,
            usage,
        })
    }
    #[must_use]
    pub const fn role(self) -> MaxwellThreeDResourceRole {
        self.role
    }
}

/// Immutable T10 input to draw lowering. Absence is a typed boundary, not a
/// fabricated shader or an empty pipeline.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDTranslatedShaders {
    identity: Arc<()>,
    shaders: Box<[MaxwellThreeDTranslatedShader]>,
    resources: Box<[MaxwellThreeDShaderResourceUse]>,
}

impl MaxwellThreeDTranslatedShaders {
    pub(crate) fn new(
        shaders: Vec<MaxwellThreeDTranslatedShader>,
        resources: Vec<MaxwellThreeDShaderResourceUse>,
    ) -> Result<Self, MaxwellThreeDLoweringError> {
        if shaders.is_empty() {
            return Err(MaxwellThreeDLoweringError::ShaderTranslationRequired);
        }
        for (index, shader) in shaders.iter().enumerate() {
            if shader.stage == ShaderStage::Compute
                || shaders[index + 1..]
                    .iter()
                    .any(|other| other.stage == shader.stage)
            {
                return Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders);
            }
        }
        for (index, resource) in resources.iter().enumerate() {
            if resources[index + 1..].contains(resource) {
                return Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders);
            }
        }
        Ok(Self {
            identity: Arc::new(()),
            shaders: shaders.into_boxed_slice(),
            resources: resources.into_boxed_slice(),
        })
    }
    #[must_use]
    pub fn shaders(&self) -> &[MaxwellThreeDTranslatedShader] {
        &self.shaders
    }
    #[must_use]
    pub fn resources(&self) -> &[MaxwellThreeDShaderResourceUse] {
        &self.resources
    }

    fn identity(&self) -> Arc<()> {
        Arc::clone(&self.identity)
    }

    fn has_identity(&self, identity: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.identity, identity)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ViewKey {
    Buffer {
        description: nixe_gpu::BufferDescription,
        buffer_offset: u64,
        backing: nixe_gpu::BackingView,
        mappings: Arc<[super::MaxwellThreeDMappingReference]>,
    },
    Image {
        description: nixe_gpu::ImageDescription,
        swizzle: nixe_gpu::Swizzle,
        guest_format: super::MaxwellThreeDGuestImageFormat,
        guest_pte_kind: u8,
        guest_compression_enabled: bool,
        bindings: Box<
            [(
                ImageSubresourceRange,
                nixe_gpu::ImageMemoryLayout,
                nixe_gpu::BackingView,
            )],
        >,
        mappings: Arc<[super::MaxwellThreeDMappingReference]>,
    },
}

impl ViewKey {
    fn matches_resource(&self, resource: &MaxwellThreeDResolvedResource) -> bool {
        match (self, resource) {
            (
                Self::Buffer {
                    description,
                    buffer_offset,
                    backing,
                    mappings,
                },
                MaxwellThreeDResolvedResource::Buffer(current),
            ) => {
                *description == current.description()
                    && *buffer_offset == current.view().buffer_offset()
                    && same_canonical_backing(backing, current.view().backing())
                    && mappings.as_ref() == current.mappings()
            }
            (
                Self::Image {
                    description,
                    swizzle,
                    guest_format,
                    guest_pte_kind,
                    guest_compression_enabled,
                    bindings,
                    mappings,
                },
                MaxwellThreeDResolvedResource::Image(current),
            ) => {
                *description == current.description()
                    && *swizzle == current.view().swizzle()
                    && *guest_format == current.guest_format()
                    && *guest_pte_kind == current.guest_layout().pte_kind()
                    && *guest_compression_enabled
                        == current.guest_layout().requires_materialization()
                    && mappings.as_ref() == current.mappings()
                    && bindings.len() == current.view().bindings().len()
                    && bindings.iter().zip(current.view().bindings()).all(
                        |((subresources, layout, backing), current)| {
                            *subresources == current.subresources()
                                && *layout == current.layout()
                                && same_canonical_backing(backing, current.backing())
                        },
                    )
            }
            _ => false,
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        (0..self.backing_count()).any(|left| {
            (0..other.backing_count()).any(|right| {
                self.backing(left)
                    .expect("backing index is bounded by backing_count")
                    .overlaps(
                        other
                            .backing(right)
                            .expect("backing index is bounded by backing_count"),
                    )
            })
        })
    }

    fn backing_count(&self) -> usize {
        match self {
            Self::Buffer { .. } => 1,
            Self::Image { bindings, .. } => bindings.len(),
        }
    }

    fn backing(&self, index: usize) -> Option<&nixe_gpu::BackingView> {
        match self {
            Self::Buffer { backing, .. } => (index == 0).then_some(backing),
            Self::Image { bindings, .. } => bindings.get(index).map(|(_, _, backing)| backing),
        }
    }

    /// Returns whether an already-created backend image still represents the
    /// same guest image bytes after a mapping-only identity change.
    ///
    /// Mapping identifiers are deliberately excluded: Maxwell may bind the
    /// same canonical pages through another GPU virtual mapping without
    /// changing their contents. Any overlapping CPU write, layout change, or
    /// physical backing change makes the representation non-reusable.
    fn same_domain_as_image(&self, image: &super::MaxwellThreeDResolvedImage) -> bool {
        let Self::Image {
            description,
            swizzle,
            guest_format,
            guest_pte_kind,
            guest_compression_enabled,
            bindings,
            ..
        } = self
        else {
            return false;
        };
        *description == image.description()
            && *swizzle == image.view().swizzle()
            && *guest_format == image.guest_format()
            && *guest_pte_kind == image.guest_layout().pte_kind()
            && *guest_compression_enabled == image.guest_layout().requires_materialization()
            && bindings.len() == image.view().bindings().len()
            && bindings.iter().zip(image.view().bindings()).all(
                |((recorded_subresources, recorded_layout, recorded_backing), current)| {
                    *recorded_subresources == current.subresources()
                        && *recorded_layout == current.layout()
                        && same_canonical_backing(recorded_backing, current.backing())
                },
            )
    }
}

fn same_canonical_backing(left: &nixe_gpu::BackingView, right: &nixe_gpu::BackingView) -> bool {
    left.range() == right.range()
        || (left.range().segments().len() == right.range().segments().len()
            && left
                .range()
                .segments()
                .iter()
                .zip(right.range().segments())
                .all(|(left, right)| {
                    left.page() == right.page()
                        && left.offset() == right.offset()
                        && left.size() == right.size()
                }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ColorRepresentationBinding {
    subresources: ImageSubresourceRange,
    layout: nixe_gpu::ImageMemoryLayout,
    backing: nixe_gpu::BackingView,
}

/// Stable neutral representation state. GPU virtual mappings and backend view
/// identities are deliberately excluded: neither changes the represented
/// bytes. Canonical pages, byte ranges and image layout define the domain.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ColorRepresentationRecord {
    description: nixe_gpu::ImageDescription,
    swizzle: nixe_gpu::Swizzle,
    guest_format: super::MaxwellThreeDGuestImageFormat,
    guest_pte_kind: u8,
    guest_compression_enabled: bool,
    bindings: Box<[ColorRepresentationBinding]>,
    cpu_writes: Option<CanonicalCpuWriteDependency>,
}

impl ColorRepresentationRecord {
    fn same_domain_as_image(&self, image: &super::MaxwellThreeDResolvedImage) -> bool {
        self.description == image.description()
            && self.swizzle == image.view().swizzle()
            && self.guest_format == image.guest_format()
            && self.guest_pte_kind == image.guest_layout().pte_kind()
            && self.guest_compression_enabled == image.guest_layout().requires_materialization()
            && self.bindings.len() == image.view().bindings().len()
            && self.bindings.iter().zip(image.view().bindings()).all(
                |(recorded_binding, current_binding)| {
                    recorded_binding.subresources == current_binding.subresources()
                        && recorded_binding.layout == current_binding.layout()
                        && same_canonical_backing(
                            &recorded_binding.backing,
                            current_binding.backing(),
                        )
                },
            )
    }

    fn remains_materialized_for(&self, image: &super::MaxwellThreeDResolvedImage) -> bool {
        if !self.same_domain_as_image(image) {
            return false;
        }
        self.cpu_writes
            .as_ref()
            .is_some_and(CanonicalCpuWriteDependency::remains_current)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ViewRecord {
    key: ViewKey,
    dependency: ResourceDependency,
    materialization: ViewMaterialization,
    cpu_writes: Option<CanonicalCpuWriteDependency>,
}

impl ViewRecord {
    fn remains_current_for_image(&self, image: &super::MaxwellThreeDResolvedImage) -> bool {
        self.key.same_domain_as_image(image)
            && self
                .cpu_writes
                .as_ref()
                .is_some_and(CanonicalCpuWriteDependency::remains_current)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ViewMaterialization {
    Direct,
    CompressedColor,
    CompressedDepthStencil { depth: bool, stencil: bool },
}

impl ViewMaterialization {
    const fn supports_depth_stencil(self, depth: bool, stencil: bool) -> bool {
        match self {
            Self::Direct => true,
            Self::CompressedDepthStencil {
                depth: materialized_depth,
                stencil: materialized_stencil,
            } => (!depth || materialized_depth) && (!stencil || materialized_stencil),
            Self::CompressedColor => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RenderPassRecord {
    description: RenderPassDescription,
    id: RenderPassId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DescriptorRecord {
    roles: Box<[MaxwellThreeDResourceRole]>,
    bindings: Box<[u8]>,
    dependencies: Box<[ResourceDependency]>,
    id: DescriptorTableId,
}

#[derive(Debug)]
struct PreparedDrawRecord {
    state: super::state::MaxwellThreeDDrawStateIdentity,
    resources: Arc<()>,
    shaders: Arc<()>,
    operations: [GpuOperation; 3],
    dirty_images: Arc<[usize]>,
}

impl PreparedDrawRecord {
    fn matches(
        &self,
        state: &MaxwellThreeDState,
        resources: &MaxwellThreeDResolvedResources,
        shaders: &MaxwellThreeDTranslatedShaders,
    ) -> bool {
        self.state.matches(state)
            && resources.has_identity(&self.resources)
            && shaders.has_identity(&self.shaders)
    }

    fn operations(
        &self,
        arguments: DrawArguments,
    ) -> Result<[GpuOperation; 3], MaxwellThreeDLoweringError> {
        Ok([
            self.operations[0].clone(),
            self.operations[1]
                .with_draw_arguments(arguments)
                .map_err(MaxwellThreeDLoweringError::Command)?,
            self.operations[2].clone(),
        ])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SamplerRecord {
    sampler: super::MaxwellThreeDResolvedSampler,
    id: SamplerId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShaderTranslationRecord {
    #[cfg(debug_assertions)]
    key: Option<MaxwellShaderTranslationKey>,
    id: ShaderId,
    module: nixe_gpu::ShaderBackendModule,
    published: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShaderTranslationSetRecord {
    #[cfg(debug_assertions)]
    inputs: MaxwellShaderTranslationInputs,
    programs: Arc<[MaxwellTranslatedShaderProgram]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShaderTranslationSourceRecord {
    #[cfg(debug_assertions)]
    source: MaxwellShaderTranslationSource,
    inputs: MaxwellShaderTranslationInputs,
    programs: Arc<[MaxwellTranslatedShaderProgram]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShaderStateRecord {
    state: super::MaxwellThreeDShaderStateIdentity,
    inputs: MaxwellShaderTranslationInputs,
    programs: Arc<[MaxwellTranslatedShaderProgram]>,
    translated: Option<Arc<MaxwellThreeDTranslatedShaders>>,
}

#[derive(Debug)]
struct FingerprintedRecord<T> {
    value: T,
    last_used: Cell<u64>,
}

/// Owner-local fingerprint cache with O(1), allocation-free hits and exact LRU
/// stamps. It is mutated only by ordered Maxwell lowering.
#[derive(Debug)]
struct FingerprintCache<T> {
    records: HashMap<u128, FingerprintedRecord<T>>,
    next_use: Cell<u64>,
}

impl<T> Default for FingerprintCache<T> {
    fn default() -> Self {
        Self {
            records: HashMap::new(),
            next_use: Cell::new(1),
        }
    }
}

impl<T> FingerprintCache<T> {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn get(&self, fingerprint: u128) -> Option<&T> {
        let record = self.records.get(&fingerprint)?;
        record.last_used.set(self.take_use());
        Some(&record.value)
    }

    fn take_use(&self) -> u64 {
        let next = self.next_use.get();
        self.next_use.set(
            next.checked_add(1)
                .expect("GPU cache LRU sequence exhausted"),
        );
        next
    }

    fn push(&mut self, fingerprint: u128, value: T) {
        let last_used = self.take_use();
        assert!(
            self.records
                .insert(
                    fingerprint,
                    FingerprintedRecord {
                        value,
                        last_used: Cell::new(last_used),
                    }
                )
                .is_none(),
            "duplicate GPU cache fingerprint insertion"
        );
    }

    fn get_mut(&mut self, fingerprint: u128) -> Option<&mut T> {
        let last_used = self.take_use();
        let record = self.records.get_mut(&fingerprint)?;
        record.last_used.set(last_used);
        Some(&mut record.value)
    }

    fn replace(&mut self, fingerprint: u128, value: T) {
        let last_used = self.take_use();
        if let Some(record) = self.records.get_mut(&fingerprint) {
            record.value = value;
            record.last_used.set(last_used);
        } else {
            self.records.insert(
                fingerprint,
                FingerprintedRecord {
                    value,
                    last_used: Cell::new(last_used),
                },
            );
        }
    }

    fn remove_lru(&mut self) -> (u128, T) {
        let fingerprint = self
            .records
            .iter()
            .min_by_key(|(_, record)| record.last_used.get())
            .map(|(fingerprint, _)| *fingerprint)
            .expect("LRU eviction requires a non-empty cache");
        let removed = self
            .records
            .remove(&fingerprint)
            .expect("selected LRU fingerprint remains present");
        (fingerprint, removed.value)
    }
}

/// Frontend-owned derived identity cache. It contains no backend handles and
/// changes only while lowering ordered frontend work.
#[derive(Debug)]
pub struct MaxwellThreeDLoweringCache {
    configuration: GpuCacheConfiguration,
    revision: u64,
    next_identity: u64,
    allocations: Vec<(
        nixe_gpu::GpuAllocationId,
        nixe_gpu::GpuAllocationDescription,
    )>,
    views: Vec<ViewRecord>,
    color_materializations: Vec<ColorRepresentationRecord>,
    graphics_pipeline: Option<PipelineId>,
    render_passes: Vec<RenderPassRecord>,
    descriptors: Vec<DescriptorRecord>,
    prepared_draw: Option<PreparedDrawRecord>,
    samplers: Vec<SamplerRecord>,
    shader_translation_sets: FingerprintCache<ShaderTranslationSetRecord>,
    shader_translation_sources: FingerprintCache<ShaderTranslationSourceRecord>,
    shader_state: Option<ShaderStateRecord>,
    shader_translations: FingerprintCache<ShaderTranslationRecord>,
    retired_resources: Vec<ResourceDependency>,
    accesses: Vec<(AccessTarget, AccessScope)>,
    resolved_resources: super::MaxwellThreeDResolvedResourceCache,
    resource_roles: Vec<MaxwellThreeDResourceRole>,
    mme_methods: Vec<crate::MaxwellMethodDispatch>,
    mme_parameters: Vec<u32>,
}

impl Default for MaxwellThreeDLoweringCache {
    fn default() -> Self {
        Self::new(GpuCacheConfiguration::default())
    }
}

impl MaxwellThreeDLoweringCache {
    #[must_use]
    pub fn new(configuration: GpuCacheConfiguration) -> Self {
        Self {
            configuration,
            revision: 0,
            next_identity: 1,
            allocations: Vec::new(),
            views: Vec::new(),
            color_materializations: Vec::new(),
            graphics_pipeline: None,
            render_passes: Vec::new(),
            descriptors: Vec::new(),
            prepared_draw: None,
            samplers: Vec::new(),
            shader_translation_sets: FingerprintCache::default(),
            shader_translation_sources: FingerprintCache::default(),
            shader_state: None,
            shader_translations: FingerprintCache::default(),
            retired_resources: Vec::new(),
            accesses: Vec::new(),
            resolved_resources: super::MaxwellThreeDResolvedResourceCache::default(),
            resource_roles: Vec::new(),
            mme_methods: Vec::new(),
            mme_parameters: Vec::new(),
        }
    }
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.views.len()
    }
    pub(crate) fn resolved_resources_mut(
        &mut self,
    ) -> &mut super::MaxwellThreeDResolvedResourceCache {
        &mut self.resolved_resources
    }

    pub(crate) const fn resource_cache_limit(&self) -> usize {
        self.configuration.pipeline_entries()
    }

    pub(crate) fn take_resource_roles(&mut self) -> Vec<MaxwellThreeDResourceRole> {
        let mut roles = std::mem::take(&mut self.resource_roles);
        roles.clear();
        roles
    }

    pub(crate) fn recycle_resource_roles(&mut self, roles: Vec<MaxwellThreeDResourceRole>) {
        self.resource_roles = roles;
    }

    pub(crate) fn take_mme_scratch(&mut self) -> (Vec<crate::MaxwellMethodDispatch>, Vec<u32>) {
        let mut methods = std::mem::take(&mut self.mme_methods);
        let mut parameters = std::mem::take(&mut self.mme_parameters);
        methods.clear();
        parameters.clear();
        (methods, parameters)
    }

    pub(crate) fn recycle_mme_scratch(
        &mut self,
        methods: Vec<crate::MaxwellMethodDispatch>,
        parameters: Vec<u32>,
    ) {
        self.mme_methods = methods;
        self.mme_parameters = parameters;
    }

    #[cfg(test)]
    pub(crate) fn shader_translation_count(&self) -> usize {
        self.shader_translations.len()
    }

    #[cfg(test)]
    pub(crate) fn shader_translation_set_count(&self) -> usize {
        self.shader_translation_sets.len()
    }

    /// Reuses a complete translation before rebuilding verified IR or WGSL.
    /// The frontend owns and updates the versioned input snapshot directly.
    pub(crate) fn resolve_shader_translation_inputs(
        &mut self,
        inputs: MaxwellShaderTranslationInputs,
    ) -> Result<Arc<[MaxwellTranslatedShaderProgram]>, MaxwellShaderTranslationError> {
        let fingerprint = inputs.fingerprint();
        if let Some(record) = self.shader_translation_sets.get(fingerprint) {
            #[cfg(debug_assertions)]
            assert_eq!(
                record.inputs, inputs,
                "XXH3-128 collision or incomplete shader-set cache key"
            );
            return Ok(Arc::clone(&record.programs));
        }

        log::debug!("Maxwell shader translation cache miss: fingerprint={fingerprint:032x}");

        let programs: Arc<[MaxwellTranslatedShaderProgram]> =
            translate_prepared_maxwell_shader_programs(&inputs)?.into();
        self.shader_translation_sets.push(
            fingerprint,
            ShaderTranslationSetRecord {
                #[cfg(debug_assertions)]
                inputs,
                programs: Arc::clone(&programs),
            },
        );
        while self.shader_translation_sets.len() > self.configuration.shader_entries() {
            let (evicted, _) = self.shader_translation_sets.remove_lru();
            log::debug!(
                "Maxwell shader translation cache evicted LRU set: fingerprint={evicted:032x}"
            );
        }
        Ok(programs)
    }

    pub(crate) fn resolve_shader_translation_source(
        &mut self,
        source: MaxwellShaderTranslationSourceKey<'_>,
        address_space: &crate::MaxwellGpuAddressSpace,
    ) -> Result<Arc<[MaxwellTranslatedShaderProgram]>, MaxwellShaderTranslationError> {
        let source_fingerprint = source.fingerprint();
        if let Some(record) = self.shader_translation_sources.get(source_fingerprint) {
            #[cfg(debug_assertions)]
            assert!(
                source.matches(&record.source),
                "XXH3-128 collision or incomplete shader-source cache key"
            );
            if record.inputs.source_is_current(address_space) {
                return Ok(Arc::clone(&record.programs));
            }
        }

        log::debug!(
            "Maxwell shader source cache miss or stale entry: fingerprint={source_fingerprint:032x}"
        );

        let source = source.materialize();
        let inputs = prepare_maxwell_shader_translation_inputs_from_source(&source, address_space)?;
        let programs = self.resolve_shader_translation_inputs(inputs.clone())?;
        self.shader_translation_sources.replace(
            source_fingerprint,
            ShaderTranslationSourceRecord {
                #[cfg(debug_assertions)]
                source,
                inputs,
                programs: Arc::clone(&programs),
            },
        );
        while self.shader_translation_sources.len() > self.configuration.shader_entries() {
            let (evicted, _) = self.shader_translation_sources.remove_lru();
            log::debug!("Maxwell shader source cache evicted LRU set: fingerprint={evicted:032x}");
        }
        Ok(programs)
    }

    /// Reuses the shader set directly from the retained semantic state before
    /// constructing or hashing a source key. Ordered writes only invalidate
    /// this path when they overlap bytes which were actually decoded.
    pub(crate) fn resolve_shader_translation_for_state(
        &mut self,
        state: &MaxwellThreeDState,
        staged_writes: &[MaxwellStagedShaderWrite],
        address_space: &crate::MaxwellGpuAddressSpace,
    ) -> Result<Arc<[MaxwellTranslatedShaderProgram]>, MaxwellShaderTranslationError> {
        if let Some(record) = &self.shader_state
            && record.state.matches(state)
            && record.inputs.source_is_current(address_space)
            && record.inputs.staged_writes_are_irrelevant(staged_writes)
        {
            return Ok(Arc::clone(&record.programs));
        }

        let source = prepare_maxwell_shader_translation_source(state, staged_writes)?;
        let fingerprint = source.fingerprint();
        let programs = self.resolve_shader_translation_source(source, address_space)?;
        let inputs = self
            .shader_translation_sources
            .get(fingerprint)
            .expect("resolved shader source was retained")
            .inputs
            .clone();
        self.shader_state = Some(ShaderStateRecord {
            state: state.shader_state_identity(),
            inputs,
            programs: Arc::clone(&programs),
            translated: None,
        });
        Ok(programs)
    }

    pub(crate) fn reuse_translated_shaders_for_state(
        &self,
        state: &MaxwellThreeDState,
        staged_writes: &[MaxwellStagedShaderWrite],
        address_space: &crate::MaxwellGpuAddressSpace,
    ) -> Option<Arc<MaxwellThreeDTranslatedShaders>> {
        let record = self.shader_state.as_ref()?;
        if !record.state.matches(state)
            || !record.inputs.source_is_current(address_space)
            || !record.inputs.staged_writes_are_irrelevant(staged_writes)
        {
            return None;
        }
        record.translated.as_ref().map(Arc::clone)
    }

    pub(crate) fn retain_translated_shader_state(
        &mut self,
        programs: &Arc<[MaxwellTranslatedShaderProgram]>,
        translated: Arc<MaxwellThreeDTranslatedShaders>,
    ) {
        let record = self
            .shader_state
            .as_mut()
            .expect("translated shaders follow a resolved shader state");
        assert!(
            Arc::ptr_eq(&record.programs, programs),
            "translated shaders must describe the current shader state"
        );
        record.translated = Some(translated);
    }

    /// Resolves immutable T10 products to stable logical shader identities.
    pub(crate) fn stage_shader_translations(
        &mut self,
        programs: &[MaxwellTranslatedShaderProgram],
    ) -> Result<MaxwellThreeDTranslatedShaders, MaxwellThreeDLoweringError> {
        let mut shaders = Vec::with_capacity(programs.len());
        let mut resources: Vec<MaxwellThreeDShaderResourceUse> = Vec::new();
        for program in programs {
            let fingerprint = program.fingerprint();
            let id = if let Some(record) = self.shader_translations.get(fingerprint) {
                #[cfg(debug_assertions)]
                assert_eq!(
                    record.key.as_ref(),
                    Some(program.key()),
                    "XXH3-128 collision or incomplete shader cache key"
                );
                record.id
            } else {
                log::debug!(
                    "Maxwell translated shader cache miss: stage={:?} fingerprint={fingerprint:032x}",
                    program.stage()
                );
                let id = ShaderId::new(take_identity(self)?);
                self.shader_translations.push(
                    fingerprint,
                    ShaderTranslationRecord {
                        #[cfg(debug_assertions)]
                        key: Some(program.key().clone()),
                        id,
                        module: program.module().clone(),
                        published: false,
                    },
                );
                self.enforce_shader_translation_limit();
                id
            };
            shaders.push(MaxwellThreeDTranslatedShader::new(
                program.stage(),
                id,
                fingerprint,
                program.directly_addressable_memory(),
                program.maximum_api_visible_calls(),
            ));
            let stages = shader_pipeline_stages(program.stage())?;
            for resource in program.resources() {
                let (role, kind, usage) = match resource.kind() {
                    ShaderResourceKind::ConstantBuffer
                        if resource.readable() && !resource.writable() =>
                    {
                        (
                            MaxwellThreeDResourceRole::ConstantBuffer {
                                group: program
                                    .bind_group()
                                    .ok_or(MaxwellThreeDLoweringError::InvalidTranslatedShaders)?,
                                slot: program
                                    .local_resource_binding(resource.binding())
                                    .ok_or(MaxwellThreeDLoweringError::InvalidTranslatedShaders)?,
                            },
                            DescriptorKind::Buffer,
                            Some(ResourceUsage::StorageBuffer),
                        )
                    }
                    ShaderResourceKind::SampledImage | ShaderResourceKind::SampledImage2DArray
                        if resource.readable() && !resource.writable() =>
                    {
                        let texture = program
                            .texture_bindings()
                            .iter()
                            .copied()
                            .find(|binding| binding.image_binding() == resource.binding())
                            .ok_or(MaxwellThreeDLoweringError::InvalidTranslatedShaders)?;
                        (
                            MaxwellThreeDResourceRole::SampledImage {
                                texture: super::MaxwellThreeDTextureReference::new(
                                    program.bind_group().ok_or(
                                        MaxwellThreeDLoweringError::InvalidTranslatedShaders,
                                    )?,
                                    program.texture_constant_buffer_slot().ok_or(
                                        MaxwellThreeDLoweringError::InvalidTranslatedShaders,
                                    )?,
                                    texture.constant_buffer_byte_offset(),
                                ),
                                dimension: match texture.image_kind() {
                                    ShaderResourceKind::SampledImage => {
                                        MaxwellThreeDTextureDimension::Two
                                    }
                                    ShaderResourceKind::SampledImage2DArray => {
                                        MaxwellThreeDTextureDimension::TwoArray
                                    }
                                    _ => {
                                        return Err(
                                            MaxwellThreeDLoweringError::InvalidTranslatedShaders,
                                        );
                                    }
                                },
                            },
                            DescriptorKind::SampledImage,
                            Some(ResourceUsage::SampledImage),
                        )
                    }
                    ShaderResourceKind::Sampler if resource.readable() && !resource.writable() => {
                        let texture = program
                            .texture_bindings()
                            .iter()
                            .copied()
                            .find(|binding| binding.sampler_binding() == resource.binding())
                            .ok_or(MaxwellThreeDLoweringError::InvalidTranslatedShaders)?;
                        (
                            MaxwellThreeDResourceRole::Sampler(
                                super::MaxwellThreeDTextureReference::new(
                                    program.bind_group().ok_or(
                                        MaxwellThreeDLoweringError::InvalidTranslatedShaders,
                                    )?,
                                    program.texture_constant_buffer_slot().ok_or(
                                        MaxwellThreeDLoweringError::InvalidTranslatedShaders,
                                    )?,
                                    texture.constant_buffer_byte_offset(),
                                ),
                            ),
                            DescriptorKind::Sampler,
                            None,
                        )
                    }
                    _ => return Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders),
                };
                if let Some(existing) = resources.iter_mut().find(|existing| existing.role == role)
                {
                    if existing.binding != resource.binding() || existing.kind != kind {
                        return Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders);
                    }
                    existing.stages = existing.stages.union(stages);
                } else {
                    resources.push(MaxwellThreeDShaderResourceUse::new(
                        role,
                        resource.binding(),
                        kind,
                        stages,
                        usage,
                    )?);
                }
            }
        }
        MaxwellThreeDTranslatedShaders::new(shaders, resources)
    }

    fn enforce_shader_translation_limit(&mut self) {
        while self.shader_translations.len() > self.configuration.shader_entries() {
            let (fingerprint, retired) = self.shader_translations.remove_lru();
            log::debug!(
                "Maxwell translated shader cache evicted LRU shader: id={} fingerprint={fingerprint:032x}",
                retired.id
            );
            if !retired.published {
                continue;
            }
            if self.prepared_draw.as_ref().is_some_and(|prepared| {
                prepared.operations[1]
                    .dependencies()
                    .contains(&ResourceDependency::Shader(retired.id))
            }) {
                self.prepared_draw = None;
            }
            self.retired_resources
                .push(ResourceDependency::Shader(retired.id));
        }
    }

    #[cfg(test)]
    pub(crate) fn seed_test_shader_translations(
        &mut self,
        shaders: &MaxwellThreeDTranslatedShaders,
    ) {
        for shader in shaders.shaders() {
            if let Some(record) = self.shader_translations.get(shader.cache_fingerprint) {
                assert_eq!(record.id, shader.shader());
                continue;
            }
            let ir = nixe_gpu::VerifiedShaderIr::verify(nixe_gpu::ShaderIr::new(
                shader.stage(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![nixe_gpu::ShaderInstruction::new(
                    nixe_gpu::ShaderSourceLocation::new(0),
                    nixe_gpu::ShaderPredicate::Always,
                    nixe_gpu::ShaderOperation::Exit,
                )],
            ))
            .expect("synthetic unit-test shader is valid");
            let module =
                nixe_gpu::lower_shader_ir_to_wgsl(&ir).expect("synthetic unit-test shader lowers");
            self.shader_translations.push(
                shader.cache_fingerprint,
                ShaderTranslationRecord {
                    #[cfg(debug_assertions)]
                    key: None,
                    id: shader.shader(),
                    module,
                    published: false,
                },
            );
        }
    }
}

/// Committed frontend record retained independently from backend handles.
pub struct MaxwellThreeDLoweredWork {
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
    dirty_images: Arc<[usize]>,
}

impl MaxwellThreeDLoweredWork {
    #[must_use]
    pub fn resource_creations(&self) -> &[BackendResourceCreateInfo] {
        &self.creations
    }
    #[must_use]
    pub fn resource_invalidations(&self) -> &[ResourceDependency] {
        &self.invalidations
    }
    #[must_use]
    pub const fn submission(&self) -> &OperationSubmission {
        &self.submission
    }
    #[must_use]
    pub fn dirty_images(&self) -> &[usize] {
        &self.dirty_images
    }
}

/// Lowers one exact trigger directly into frontend-owned derived caches.
///
/// A lowering failure is terminal for the guest submission. Derived caches are
/// not guest-visible state, so cloning them for rollback would only preserve a
/// path which cannot resume.
#[allow(clippy::too_many_arguments)]
pub fn lower_maxwell_three_d_operation(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    translated_shaders: Option<&MaxwellThreeDTranslatedShaders>,
    submission: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    capabilities: &BackendCapabilities,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
    let work = lower_maxwell_three_d_operation_into_cache(
        state,
        resources,
        trigger,
        translated_shaders,
        submission,
        predecessors,
        cache,
    )?;
    for creation in work.resource_creations() {
        let requirements = creation
            .capability_requirements()
            .map_err(|_| MaxwellThreeDLoweringError::InvalidResourceCreation)?;
        capabilities
            .negotiate(&requirements)
            .map_err(MaxwellThreeDLoweringError::Capability)?;
    }
    capabilities
        .negotiate_all(&work.submission().capability_requirements())
        .map_err(MaxwellThreeDLoweringError::Capability)?;
    Ok(work)
}

/// Lowers into the frontend-owned cache used by ordered execution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_maxwell_three_d_operation_into_cache(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    translated_shaders: Option<&MaxwellThreeDTranslatedShaders>,
    submission: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
    if let (MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. }, Some(shaders)) =
        (trigger, translated_shaders)
    {
        let arguments = draw_arguments(state, vertex_count)?;
        let prepared = cache
            .prepared_draw
            .as_ref()
            .filter(|prepared| prepared.matches(state, resources, shaders))
            .map(|prepared| {
                Ok::<_, MaxwellThreeDLoweringError>((
                    prepared.operations(arguments)?,
                    Arc::clone(&prepared.dirty_images),
                ))
            })
            .transpose()?;
        if let Some((commands, dirty_images)) = prepared {
            let invalidations = std::mem::take(&mut cache.retired_resources);
            return finish_lowered_work(
                cache,
                submission,
                predecessors,
                Vec::new(),
                invalidations,
                commands,
                dirty_images,
            );
        }
    }
    if let Some(mode) = state.render_enable().execution_mode()
        && mode != MaxwellThreeDRenderEnableMode::Enabled
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedRenderEnableMode(
            mode,
        ));
    }
    if state
        .render_enable()
        .conditional_load_constant_buffer()
        .value()
        == Some(&MaxwellThreeDConditionalLoadConstantBuffer::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedConditionalLoadConstantBufferSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(MaxwellThreeDFixedFunctionValue::ShadeMode(MaxwellThreeDShadeMode::Flat)) =
        state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
            .value()
    {
        // Smooth preserves the interpolation selected by translated shader
        // inputs and therefore needs no fixed-function override. Flat shading
        // changes the primitive-wide source value and remains a typed boundary
        // until T10 represents that override explicitly.
        return Err(MaxwellThreeDLoweringError::UnsupportedShadeModeSemantics(
            MaxwellThreeDShadeMode::Flat,
        ));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ProvokingVertex)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::ProvokingVertex(
            MaxwellThreeDProvokingVertex::First,
        ))
    {
        return Err(
            MaxwellThreeDLoweringError::UnsupportedProvokingVertexSemantics(
                MaxwellThreeDProvokingVertex::First,
            ),
        );
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::TwoSidedLightEnable)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedTwoSidedLightSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ColorClampEnable)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedColorClampSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(MaxwellThreeDFixedFunctionValue::PixelShaderSaturate(value)) = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::PixelShaderSaturate)
        .value()
        && let Some(output) = value.first_enabled_output()
    {
        return Err(
            MaxwellThreeDLoweringError::UnsupportedPixelShaderSaturateSemantics {
                output,
                range: value
                    .clamp_range(output)
                    .expect("enabled output is within the eight-output register"),
            },
        );
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.shader_bindings().has_enabled_pipeline()
    {
        let local_memory = state.shader_execution().shader_local_memory();
        if local_memory.region_is_partially_programmed() {
            return Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_SHADER_LOCAL_MEMORY_A-D",
            ));
        }
        if let Some(default_size_per_warp) = local_memory
            .default_size_per_warp()
            .value()
            .copied()
            .filter(|size| size.bytes() != 0)
        {
            if local_memory.address().is_none() || local_memory.size().is_none() {
                return Err(MaxwellThreeDLoweringError::IncompleteDraw(
                    "SET_SHADER_LOCAL_MEMORY_A-D",
                ));
            }
            return Err(
                MaxwellThreeDLoweringError::UnsupportedShaderLocalMemorySemantics {
                    default_size_per_warp,
                },
            );
        }
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.color_reduction().enable().value()
        == Some(&MaxwellThreeDColorReductionThresholdsEnable::Enabled)
    {
        // NVIDIA exposes a dedicated activation method, so merely programming
        // a threshold is not enough to make it effective. Once explicitly
        // enabled, however, the current neutral pipeline cannot represent the
        // reduction decision and must stop before cache/backend effects.
        return Err(MaxwellThreeDLoweringError::UnsupportedColorReductionSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.constant_color_rendering().enabled().value() == Some(&true)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedConstantColorRenderingSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.shader_execution().api_mandated_early_z().value()
        == Some(&MaxwellThreeDApiMandatedEarlyZ::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedApiMandatedEarlyZSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.coverage().post_z_pixel_shader_imask().value()
        == Some(&MaxwellThreeDPostZPixelShaderImask::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedPostZPixelShaderImaskSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(value) = state
        .shader_execution()
        .pixel_shader_interlock_control()
        .value()
        .copied()
        .filter(|value| value.conflict_detection_enabled())
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedPixelShaderInterlockSemantics(value));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(base_vertex) = state
        .vertex_input()
        .assembly()
        .global_base_vertex_index()
        .value()
        .copied()
        .filter(|value| *value != 0)
    {
        // The neutral non-indexed draw currently has one first-vertex value,
        // which controls both vertex-buffer addressing and the shader-visible
        // vertex index. Maxwell's global base changes only the latter; mapping
        // it to first_vertex would therefore silently fetch different data.
        return Err(MaxwellThreeDLoweringError::UnsupportedGlobalBaseVertexIndex(base_vertex));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.coverage().csaa_enable().value() == Some(&MaxwellThreeDCsaaEnable::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedCsaaSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(value) = state
        .coverage()
        .coverage_to_color()
        .value()
        .copied()
        .filter(|value| value.enabled())
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedCoverageToColorSemantics(value));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(value) = state
        .coverage()
        .alpha_to_coverage_override()
        .value()
        .copied()
        .filter(|value| value.raw() != 0)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedAlphaToCoverageOverrideSemantics(value));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.coverage().tir_mode().value() == Some(&MaxwellThreeDTirMode::RasterNTargetM)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedTirSemantics {
            control: state.coverage().tir_control().value().copied(),
        });
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(value) = state
        .coverage()
        .hybrid_anti_alias_control()
        .value()
        .copied()
        .filter(|value| !value.is_single_pass_per_fragment())
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedHybridAntiAliasSemantics(value));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some((group, value)) = state
        .coverage()
        .sample_locations()
        .iter()
        .enumerate()
        .find_map(|(group, register)| {
            register
                .value()
                .copied()
                .filter(|value| !value.is_centered())
                .map(|value| (group as u8, value))
        })
    {
        return Err(
            MaxwellThreeDLoweringError::UnsupportedSampleLocationsSemantics { group, value },
        );
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state.ps_output_sample_mask_effective() == Some(true)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedPsOutputSampleMaskSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        draw_viewport_transform(state)?;
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some((viewport, swizzle)) = state
        .fixed_function()
        .viewport()
        .iter()
        .enumerate()
        .find_map(|(viewport, state)| {
            state
                .coordinate_swizzle()
                .value()
                .copied()
                .filter(|swizzle| !swizzle.is_identity())
                .map(|swizzle| (viewport as u8, swizzle))
        })
    {
        return Err(
            MaxwellThreeDLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                viewport,
                swizzle,
            },
        );
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedWindowClipSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(
            MaxwellThreeDClipIdTestEnable::Enabled,
        ))
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedClipIdTestSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        if state.viewport().pixel_center().value()
            == Some(&MaxwellThreeDViewportPixelCenter::Integers)
        {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedViewportPixelCenterSemantics(
                    MaxwellThreeDViewportPixelCenter::Integers,
                ),
            );
        }
        if let Some(mode) = state
            .raster()
            .fill_via_triangle()
            .value()
            .copied()
            .filter(|mode| *mode == MaxwellThreeDFillViaTriangleMode::FillAll)
        {
            return Err(MaxwellThreeDLoweringError::UnsupportedFillViaTriangleSemantics(mode));
        }
        if state.raster().conservative_raster().value()
            == Some(&MaxwellThreeDConservativeRasterEnable::Enabled)
        {
            return Err(MaxwellThreeDLoweringError::UnsupportedConservativeRasterSemantics);
        }
        if state
            .vertex_input()
            .primitive()
            .active_begin()
            .is_some_and(|begin| matches!(begin.topology(), 4..=6))
        {
            if state.raster().polygon_smooth_enable().value() == Some(&true) {
                return Err(MaxwellThreeDLoweringError::UnsupportedPolygonSmoothSemantics);
            }
            if state.raster().polygon_stipple_enable().value() == Some(&true) {
                return Err(MaxwellThreeDLoweringError::UnsupportedPolygonStippleSemantics);
            }
        }
        if state.shader_bindings().has_enabled_pipeline()
            && state
                .shader_bindings()
                .program_region()
                .is_partially_programmed()
        {
            return Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_PROGRAM_REGION_A/B",
            ));
        }
        if state
            .vertex_input()
            .primitive()
            .active_begin()
            .is_some_and(|begin| begin.topology() == 14)
        {
            let patch_size = state
                .vertex_input()
                .primitive()
                .patch_size()
                .value()
                .copied()
                .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_PATCH"))?;
            if patch_size.control_points() == 0 {
                return Err(MaxwellThreeDLoweringError::InvalidPatchSize(patch_size));
            }
            return Err(MaxwellThreeDLoweringError::UnsupportedPatchSemantics(
                patch_size,
            ));
        }
        if state
            .vertex_input()
            .primitive()
            .active_begin()
            .is_some_and(|begin| begin.topology() == 0)
        {
            if let Some(value) = state
                .raster()
                .attribute_point_size()
                .value()
                .copied()
                .filter(|value| value.enabled())
            {
                return Err(
                    MaxwellThreeDLoweringError::UnsupportedAttributePointSizeSemantics {
                        slot: value.slot(),
                    },
                );
            }
            if state.raster().point_sprite_enable().value() == Some(&true) {
                return Err(MaxwellThreeDLoweringError::UnsupportedPointSpriteSemantics);
            }
            if state.raster().anti_aliased_point_enable().value() == Some(&true) {
                return Err(MaxwellThreeDLoweringError::UnsupportedAntiAliasedPointSemantics);
            }
            if let Some(select) = state
                .raster()
                .point_sprite_select()
                .value()
                .copied()
                .filter(|select| select.affects_point_coordinates())
            {
                return Err(
                    MaxwellThreeDLoweringError::UnsupportedPointSpriteCoordinatesSemantics(select),
                );
            }
            if let Some(mode) = state.raster().point_center_mode().value().copied() {
                return Err(MaxwellThreeDLoweringError::UnsupportedPointCenterSemantics(
                    mode,
                ));
            }
        }
        if state.edge_flag_affects_draw() {
            return Err(MaxwellThreeDLoweringError::UnsupportedEdgeFlagSemantics(
                MaxwellThreeDEdgeFlag::Disabled,
            ));
        }
        validate_line_rasterization_state(state)?;
    }
    if let MaxwellThreeDOperationTrigger::ClearSurface { source } = trigger
        && state.render_targets().clear().last_surface().source() != Some(source)
    {
        return Err(MaxwellThreeDLoweringError::TriggerStateMismatch);
    }
    let draw_attachments = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => None,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => {
            Some(select_draw_attachments(state, resources)?)
        }
    };
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        let attachments = draw_attachments
            .as_ref()
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
        validate_draw_surface_clip(state, resources, attachments)?;
        if attachments.colors.len() > 1
            && state.render_targets().separate_fragment_data().value()
                == Some(&MaxwellThreeDSeparateFragmentData::Disabled)
        {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedReplicatedColorTargetOutputSemantics,
            );
        }
        if (!attachments.colors.is_empty() || attachments.depth_stencil.is_some())
            && let Some(value) = state
                .render_targets()
                .render_target_index_offset()
                .value()
                .copied()
                .filter(|value| value.enabled())
        {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedRenderTargetIndexOffsetSemantics(value),
            );
        }
        if (!attachments.colors.is_empty() || attachments.depth_stencil.is_some())
            && let Some(value) = state
                .render_targets()
                .render_target_layer()
                .value()
                .copied()
                .filter(|value| {
                    value.affects_draw_layering(
                        state
                            .shader_bindings()
                            .has_enabled_stage(MaxwellThreeDShaderStage::Geometry),
                    )
                })
        {
            return Err(MaxwellThreeDLoweringError::UnsupportedRenderTargetLayerSemantics(value));
        }
        validate_draw_iterated_blend_state(state, attachments)?;
        validate_draw_blending_state(state, attachments)?;
        validate_draw_logic_op_state(state, attachments)?;
        validate_draw_color_write_state(state, attachments)?;
        draw_alpha_test_state(state)?;
    }
    validate_compressed_depth_materialization(
        state,
        resources,
        trigger,
        draw_attachments.as_ref(),
        cache,
    )?;
    validate_compressed_color_materialization(
        state,
        resources,
        trigger,
        draw_attachments.as_ref(),
        cache,
    )?;
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        validate_draw_stencil_state(state)?;
    }
    let shaders = if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        Some(translated_shaders.ok_or(MaxwellThreeDLoweringError::ShaderTranslationRequired)?)
    } else {
        None
    };
    if let Some(shaders) = shaders {
        validate_visible_call_limit(state, shaders)?;
        validate_shader_memory_configuration(state, shaders)?;
    }
    let resource_indices = operation_resource_indices(
        state,
        resources,
        trigger,
        draw_attachments.as_ref(),
        shaders,
    )?;
    let mut creations = Vec::new();
    let mut invalidations = std::mem::take(&mut cache.retired_resources);
    let resource_bindings = prepare_resources(
        resources,
        &resource_indices,
        cache,
        &mut creations,
        &mut invalidations,
    )?;
    let sampler_bindings = prepare_samplers(resources, cache, &mut creations, &mut invalidations)?;
    let (commands, dirty_images) = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { source: _ } => {
            let lowered = lower_clear(state, resources, &resource_bindings)?;
            record_clear_materialization(state, resources, cache)?;
            lowered
        }
        MaxwellThreeDOperationTrigger::DrawVertexArray {
            source: _,
            vertex_count,
        } => {
            let attachments = draw_attachments
                .as_ref()
                .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
            let lowered = lower_draw(
                state,
                resources,
                &resource_bindings,
                &sampler_bindings,
                shaders.ok_or(MaxwellThreeDLoweringError::ShaderTranslationRequired)?,
                attachments,
                vertex_count,
                cache,
                &mut creations,
            )?;
            record_draw_color_materializations(state, resources, attachments, cache)?;
            lowered
        }
    };
    finish_lowered_work(
        cache,
        submission,
        predecessors,
        creations,
        invalidations,
        commands,
        dirty_images,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_lowered_work(
    cache: &mut MaxwellThreeDLoweringCache,
    submission: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    creations: Vec<BackendResourceCreateInfo>,
    invalidations: Vec<ResourceDependency>,
    commands: impl IntoIterator<Item = GpuOperation>,
    dirty_images: Arc<[usize]>,
) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
    let operations = sequence_with_transitions(commands, cache)?;
    let submission = OperationSubmission::new(submission, predecessors, operations)
        .map_err(MaxwellThreeDLoweringError::Command)?;
    cache.revision = cache
        .revision
        .checked_add(1)
        .ok_or(MaxwellThreeDLoweringError::ResourceExhausted)?;
    Ok(MaxwellThreeDLoweredWork {
        creations: creations.into_boxed_slice(),
        invalidations: invalidations.into_boxed_slice(),
        submission,
        dirty_images,
    })
}

fn validate_shader_memory_configuration(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<(), MaxwellThreeDLoweringError> {
    for shader in shaders.shaders() {
        let Some(required) = shader.directly_addressable_memory() else {
            continue;
        };
        let configured = state
            .shader_execution()
            .l1_configuration()
            .value()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_L1_CONFIGURATION",
            ))?;
        if required != configured {
            return Err(
                MaxwellThreeDLoweringError::TranslatedShaderMemoryConfigurationMismatch {
                    stage: shader.stage(),
                    configured,
                    required,
                },
            );
        }
    }
    Ok(())
}

fn validate_visible_call_limit(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<(), MaxwellThreeDLoweringError> {
    let Some(limit) = state
        .shader_execution()
        .visible_call_limit()
        .value()
        .and_then(|value| value.limit())
    else {
        return Ok(());
    };
    if let Some(shader) = shaders
        .shaders()
        .iter()
        .find(|shader| shader.maximum_api_visible_calls() > limit)
    {
        return Err(MaxwellThreeDLoweringError::VisibleCallLimitExceeded {
            stage: shader.stage(),
            required: shader.maximum_api_visible_calls(),
            limit,
        });
    }
    Ok(())
}

fn validate_draw_blending_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellThreeDLoweringError> {
    if attachments.colors.is_empty() {
        return Ok(());
    }

    let per_target = match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::Boolean(value)) => *value,
        None => {
            return Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: None,
                field: "SET_BLEND_STATE_PER_TARGET",
            });
        }
        Some(_) => {
            return Err(MaxwellThreeDLoweringError::ContradictoryState {
                reason: "blend-state selection register has the wrong typed value",
            });
        }
    };

    if !per_target {
        return match state.fixed_function().blend_enable_common().value() {
            None => Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: None,
                field: "SET_BLEND_ENABLE_COMMON",
            }),
            Some(MaxwellThreeDBlendEnableCommon::Disabled) => Ok(()),
            Some(MaxwellThreeDBlendEnableCommon::Enabled) => {
                validate_common_blend_equation_state(state)?;
                Err(MaxwellThreeDLoweringError::UnsupportedBlendSemantics { target: None })
            }
        };
    }

    let mut enabled_target = None;
    for target in attachments.color_targets() {
        match state.fixed_function().blend_enable()[target as usize].value() {
            None => {
                return Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                    target: Some(target),
                    field: "SET_BLEND(i)",
                });
            }
            Some(false) => {}
            Some(true) => {
                validate_per_target_blend_equation_state(state, target)?;
                enabled_target.get_or_insert(target);
            }
        }
    }
    if let Some(target) = enabled_target {
        return Err(MaxwellThreeDLoweringError::UnsupportedBlendSemantics {
            target: Some(target),
        });
    }
    Ok(())
}

fn validate_draw_iterated_blend_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellThreeDLoweringError> {
    if attachments.colors.is_empty() {
        return Ok(());
    }
    let controls = state.fixed_function().blend_controls();
    let Some(value) = controls
        .iterated_blend()
        .value()
        .copied()
        .filter(|value| value.enabled())
    else {
        return Ok(());
    };
    Err(
        MaxwellThreeDLoweringError::UnsupportedIteratedBlendSemantics {
            value,
            pass_count: controls
                .iterated_blend_pass_count()
                .value()
                .map(|value| value.pass_count()),
        },
    )
}

fn validate_draw_logic_op_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellThreeDLoweringError> {
    if attachments.colors.is_empty()
        || state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LogicOpEnable)
            .value()
            != Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Ok(());
    }
    let function = match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::LogicOpFunction)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::LogicOp(value)) => *value,
        None => return Err(MaxwellThreeDLoweringError::IncompleteLogicOpState),
        Some(_) => {
            return Err(MaxwellThreeDLoweringError::ContradictoryState {
                reason: "logic-operation function register has the wrong typed value",
            });
        }
    };
    Err(MaxwellThreeDLoweringError::UnsupportedLogicOpSemantics(
        function,
    ))
}

fn validate_draw_color_write_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellThreeDLoweringError> {
    let fixed = state.fixed_function();
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(single)) = fixed
        .register(MaxwellThreeDFixedFunctionRegister::SingleColorTargetWriteControl)
        .value()
    else {
        // Preserve compatibility with snapshots predating this modeled
        // register. Once explicitly programmed, its selected masks are fully
        // validated below rather than guessed.
        return Ok(());
    };
    for target in attachments.color_targets() {
        let mask_register = if *single { 0 } else { target };
        let mask = fixed.color_mask()[mask_register as usize]
            .value()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteColorWriteState {
                target,
                mask_register,
            })?;
        if !mask.all_enabled() {
            return Err(MaxwellThreeDLoweringError::UnsupportedColorWriteMask {
                target,
                mask_register,
                mask,
            });
        }
    }
    Ok(())
}

fn draw_alpha_test_state(
    state: &MaxwellThreeDState,
) -> Result<Option<AlphaTest>, MaxwellThreeDLoweringError> {
    let fixed = state.fixed_function();
    if fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable)
        .value()
        != Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Ok(None);
    }
    let reference = match fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestReference)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::FloatBits(value)) => *value,
        None => {
            return Err(MaxwellThreeDLoweringError::IncompleteAlphaTestState(
                "reference",
            ));
        }
        Some(_) => {
            return Err(MaxwellThreeDLoweringError::ContradictoryState {
                reason: "alpha-test reference register has the wrong typed value",
            });
        }
    };
    let function = match fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestFunction)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::Compare(value)) => *value,
        None => {
            return Err(MaxwellThreeDLoweringError::IncompleteAlphaTestState(
                "function",
            ));
        }
        Some(_) => {
            return Err(MaxwellThreeDLoweringError::ContradictoryState {
                reason: "alpha-test function register has the wrong typed value",
            });
        }
    };
    let comparison = match function {
        MaxwellThreeDCompareOp::Never => AlphaCompareOperation::Never,
        MaxwellThreeDCompareOp::Less => AlphaCompareOperation::Less,
        MaxwellThreeDCompareOp::Equal => AlphaCompareOperation::Equal,
        MaxwellThreeDCompareOp::LessEqual => AlphaCompareOperation::LessEqual,
        MaxwellThreeDCompareOp::Greater => AlphaCompareOperation::Greater,
        MaxwellThreeDCompareOp::NotEqual => AlphaCompareOperation::NotEqual,
        MaxwellThreeDCompareOp::GreaterEqual => AlphaCompareOperation::GreaterEqual,
        MaxwellThreeDCompareOp::Always => AlphaCompareOperation::Always,
    };
    Ok(Some(AlphaTest {
        comparison,
        reference_bits: reference.get(),
    }))
}

fn validate_common_blend_equation_state(
    state: &MaxwellThreeDState,
) -> Result<(), MaxwellThreeDLoweringError> {
    let fixed = state.fixed_function();
    let separate_alpha = require_common_blend_value(
        fixed,
        MaxwellThreeDFixedFunctionRegister::BlendSeparateAlpha,
        "SET_BLEND_SEPARATE_FOR_ALPHA",
    )?;
    for (register, field) in [
        (
            MaxwellThreeDFixedFunctionRegister::BlendColorOp,
            "SET_BLEND_OP_COLOR",
        ),
        (
            MaxwellThreeDFixedFunctionRegister::BlendColorSource,
            "SET_BLEND_COEFF_SOURCE_COLOR",
        ),
        (
            MaxwellThreeDFixedFunctionRegister::BlendColorDestination,
            "SET_BLEND_COEFF_DESTINATION_COLOR",
        ),
    ] {
        require_common_blend_value(fixed, register, field)?;
    }
    if matches!(
        separate_alpha,
        MaxwellThreeDFixedFunctionValue::Boolean(true)
    ) {
        for (register, field) in [
            (
                MaxwellThreeDFixedFunctionRegister::BlendAlphaOp,
                "SET_BLEND_OP_ALPHA",
            ),
            (
                MaxwellThreeDFixedFunctionRegister::BlendAlphaSource,
                "SET_BLEND_COEFF_SOURCE_ALPHA",
            ),
            (
                MaxwellThreeDFixedFunctionRegister::BlendAlphaDestination,
                "SET_BLEND_COEFF_DESTINATION_ALPHA",
            ),
        ] {
            require_common_blend_value(fixed, register, field)?;
        }
    }
    Ok(())
}

fn require_common_blend_value(
    fixed: &super::MaxwellThreeDFixedFunctionState,
    register: MaxwellThreeDFixedFunctionRegister,
    field: &'static str,
) -> Result<MaxwellThreeDFixedFunctionValue, MaxwellThreeDLoweringError> {
    fixed.register(register).value().copied().ok_or(
        MaxwellThreeDLoweringError::IncompleteBlendState {
            target: None,
            field,
        },
    )
}

fn validate_per_target_blend_equation_state(
    state: &MaxwellThreeDState,
    target: u8,
) -> Result<(), MaxwellThreeDLoweringError> {
    let values = &state.fixed_function().per_target_blend()[target as usize];
    let separate_alpha =
        values[0]
            .value()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: Some(target),
                field: "SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA",
            })?;
    for (index, field) in [
        (1, "SET_BLEND_PER_TARGET_OP_COLOR"),
        (2, "SET_BLEND_PER_TARGET_COEFF_SOURCE_COLOR"),
        (3, "SET_BLEND_PER_TARGET_COEFF_DESTINATION_COLOR"),
    ] {
        values[index]
            .value()
            .ok_or(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: Some(target),
                field,
            })?;
    }
    if matches!(
        separate_alpha,
        MaxwellThreeDFixedFunctionValue::Boolean(true)
    ) {
        for (index, field) in [
            (4, "SET_BLEND_PER_TARGET_OP_ALPHA"),
            (5, "SET_BLEND_PER_TARGET_COEFF_SOURCE_ALPHA"),
            (6, "SET_BLEND_PER_TARGET_COEFF_DESTINATION_ALPHA"),
        ] {
            values[index]
                .value()
                .ok_or(MaxwellThreeDLoweringError::IncompleteBlendState {
                    target: Some(target),
                    field,
                })?;
        }
    }
    Ok(())
}

fn validate_line_rasterization_state(
    state: &MaxwellThreeDState,
) -> Result<(), MaxwellThreeDLoweringError> {
    let topology = state
        .vertex_input()
        .primitive()
        .active_begin()
        .map(|begin| begin.topology());
    let direct_line_primitive = topology.is_some_and(|topology| matches!(topology, 1 | 3));
    let polygon_primitive = topology.is_some_and(|topology| matches!(topology, 4..=6));
    let polygon_line_mode = [
        MaxwellThreeDFixedFunctionRegister::FrontPolygonMode,
        MaxwellThreeDFixedFunctionRegister::BackPolygonMode,
    ]
    .into_iter()
    .any(|register| {
        matches!(
            state.fixed_function().register(register).value(),
            Some(MaxwellThreeDFixedFunctionValue::PolygonMode(
                MaxwellThreeDPolygonMode::Line
            ))
        )
    });
    if !direct_line_primitive && !(polygon_primitive && polygon_line_mode) {
        return Ok(());
    }

    if state.line().anti_aliased_line_enable().value()
        == Some(&MaxwellThreeDAntiAliasedLineEnable::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedAntiAliasedLineSemantics);
    }
    if state.line().stipple_enable().value() == Some(&true) {
        let parameters = state.line().stipple_parameters().value().copied().ok_or(
            MaxwellThreeDLoweringError::IncompleteDraw("SET_LINE_STIPPLE_PARAMETERS"),
        )?;
        return Err(
            MaxwellThreeDLoweringError::UnsupportedLineStippleSemantics {
                factor: parameters.factor(),
                pattern: parameters.pattern(),
            },
        );
    }
    if polygon_primitive
        && polygon_line_mode
        && state.line().polygon_clip_generated_edge().value()
            == Some(&MaxwellThreeDPolygonClipGeneratedEdge::DoNotDrawLine)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedPolygonClipGeneratedEdgeSemantics);
    }

    match state.line().aliased_line_width_enable().value() {
        None => Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_ALIASED_LINE_WIDTH_ENABLE",
        )),
        Some(MaxwellThreeDAliasedLineWidthEnable::Disabled) => state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LineWidth)
            .value()
            .is_some()
            .then_some(())
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_LINE_WIDTH_FLOAT",
            )),
        Some(MaxwellThreeDAliasedLineWidthEnable::Enabled) => {
            Err(MaxwellThreeDLoweringError::UnsupportedAliasedLineWidthSemantics)
        }
    }
}

fn validate_compressed_depth_materialization(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<(), MaxwellThreeDLoweringError> {
    let Some((index, image)) =
        resources
            .resources()
            .iter()
            .enumerate()
            .find_map(|(index, resource)| match resource {
                MaxwellThreeDResolvedResource::Image(image)
                    if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget
                        && image.guest_layout().requires_materialization() =>
                {
                    Some((index, image))
                }
                _ => None,
            })
    else {
        return Ok(());
    };
    let (consumes_depth, consumes_stencil) = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => state
            .render_targets()
            .clear()
            .last_surface()
            .value()
            .map_or((false, false), |surface| {
                (surface.depth(), surface.stencil())
            }),
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => {
            if draw_attachments.is_some_and(|attachments| attachments.depth_stencil == Some(index))
            {
                draw_depth_stencil_aspects(state)
            } else {
                (false, false)
            }
        }
    };
    if !consumes_depth && !consumes_stencil {
        return Ok(());
    }
    if cache.views.iter().any(|record| {
        record.remains_current_for_image(image)
            && record
                .materialization
                .supports_depth_stencil(consumes_depth, consumes_stencil)
    }) {
        return Ok(());
    }
    if matches!(trigger, MaxwellThreeDOperationTrigger::ClearSurface { .. })
        && depth_clear_fully_initializes(state, image, consumes_depth, consumes_stencil)?
    {
        return Ok(());
    }
    Err(MaxwellThreeDLoweringError::CompressedDepthImportRequired {
        kind: image.guest_layout().pte_kind(),
    })
}

fn depth_clear_fully_initializes(
    state: &MaxwellThreeDState,
    image: &super::MaxwellThreeDResolvedImage,
    depth: bool,
    stencil: bool,
) -> Result<bool, MaxwellThreeDLoweringError> {
    if !depth && !stencil {
        return Ok(false);
    }
    let clear = state.render_targets().clear();
    if clear
        .surface_control()
        .value()
        .is_some_and(|control| stencil && control.respect_stencil_mask())
    {
        return Ok(false);
    }
    clear_fully_covers_image(state, image)
}

fn clear_fully_covers_image(
    state: &MaxwellThreeDState,
    image: &super::MaxwellThreeDResolvedImage,
) -> Result<bool, MaxwellThreeDLoweringError> {
    let extent = image.description().extent();
    Ok(MaxwellThreeDClearRegions::from_state(state)?
        .for_attachment(extent.width, extent.height)
        .fully_covers(extent.width, extent.height))
}

fn record_clear_materialization(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<(), MaxwellThreeDLoweringError> {
    let surface = state
        .render_targets()
        .clear()
        .last_surface()
        .value()
        .copied()
        .ok_or(MaxwellThreeDLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
    if surface.color_mask() != 0
        && state.render_targets().color()[surface.color_target() as usize]
            .compression()
            .value()
            == Some(&MaxwellThreeDColorCompressionMode::Enabled)
    {
        let index = resource_index(
            resources,
            MaxwellThreeDResourceRole::ColorTarget(surface.color_target()),
        )?;
        record_color_materialization(resolved_image(resources, index)?, cache);
    }
    if surface.depth() || surface.stencil() {
        let index = resource_index(resources, MaxwellThreeDResourceRole::DepthStencilTarget)?;
        let position = cache
            .views
            .iter()
            .position(|record| record.key.matches_resource(&resources.resources()[index]))
            .ok_or(MaxwellThreeDLoweringError::InvalidResolvedView {
                role: MaxwellThreeDResourceRole::DepthStencilTarget,
            })?;
        let materialization = cache.views[position].materialization;
        if let ViewMaterialization::CompressedDepthStencil { depth, stencil } = materialization {
            let depth = depth || surface.depth();
            let stencil = stencil || surface.stencil();
            if materialization != (ViewMaterialization::CompressedDepthStencil { depth, stencil }) {
                cache
                    .views
                    .get_mut(position)
                    .expect("depth view position came from the same cache")
                    .materialization =
                    ViewMaterialization::CompressedDepthStencil { depth, stencil };
            }
        }
    }
    Ok(())
}

fn record_draw_color_materializations(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<(), MaxwellThreeDLoweringError> {
    for target in attachments.color_targets().filter(|target| {
        state.render_targets().color()[*target as usize]
            .compression()
            .value()
            == Some(&MaxwellThreeDColorCompressionMode::Enabled)
    }) {
        let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
        record_color_materialization(resolved_image(resources, index)?, cache);
    }
    Ok(())
}

fn record_color_materialization(
    image: &super::MaxwellThreeDResolvedImage,
    cache: &mut MaxwellThreeDLoweringCache,
) {
    if let Some(position) = cache
        .color_materializations
        .iter()
        .position(|previous| previous.same_domain_as_image(image))
    {
        if cache.color_materializations[position].remains_materialized_for(image) {
            return;
        }
        *cache
            .color_materializations
            .get_mut(position)
            .expect("materialization position came from the same cache") =
            color_representation_record(image);
        return;
    }
    cache
        .color_materializations
        .push(color_representation_record(image));
}

fn validate_compressed_color_materialization(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<(), MaxwellThreeDLoweringError> {
    match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => {
            if let Some(surface) = state
                .render_targets()
                .clear()
                .last_surface()
                .value()
                .filter(|surface| surface.color_mask() != 0)
            {
                validate_compressed_color_target(
                    state,
                    resources,
                    cache,
                    surface.color_target(),
                    surface.color_mask() == 0xf,
                )?;
            }
        }
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => {
            for target in draw_attachments
                .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?
                .color_targets()
            {
                validate_compressed_color_target(state, resources, cache, target, false)?;
            }
        }
    }
    Ok(())
}

fn validate_compressed_color_target(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    cache: &MaxwellThreeDLoweringCache,
    target: u8,
    complete_clear: bool,
) -> Result<(), MaxwellThreeDLoweringError> {
    if state.render_targets().color()[target as usize]
        .compression()
        .value()
        != Some(&MaxwellThreeDColorCompressionMode::Enabled)
    {
        return Ok(());
    }
    let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
    let image = resolved_image(resources, index)?;
    if cache
        .color_materializations
        .iter()
        .any(|previous| previous.remains_materialized_for(image))
        || (complete_clear && clear_fully_covers_image(state, image)?)
    {
        return Ok(());
    }
    Err(MaxwellThreeDLoweringError::CompressedColorImportRequired { target })
}

fn operation_resource_indices(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
    shaders: Option<&MaxwellThreeDTranslatedShaders>,
) -> Result<Vec<usize>, MaxwellThreeDLoweringError> {
    let mut indices = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => {
            let surface = state
                .render_targets()
                .clear()
                .last_surface()
                .value()
                .copied()
                .ok_or(MaxwellThreeDLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
            let mut indices = Vec::new();
            if surface.color_mask() != 0
                && let Ok(index) = resource_index(
                    resources,
                    MaxwellThreeDResourceRole::ColorTarget(surface.color_target()),
                )
            {
                indices.push(index);
            }
            if (surface.depth() || surface.stencil())
                && let Ok(index) =
                    resource_index(resources, MaxwellThreeDResourceRole::DepthStencilTarget)
            {
                indices.push(index);
            }
            indices
        }
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => draw_resource_indices(
            state,
            resources,
            draw_attachments.ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?,
            shaders.ok_or(MaxwellThreeDLoweringError::ShaderTranslationRequired)?,
        )?,
    };
    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn draw_resource_indices(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<Vec<usize>, MaxwellThreeDLoweringError> {
    let mut indices = attachments.attachment_indices();
    for (stream, state) in state.vertex_input().streams().iter().enumerate() {
        if state
            .format()
            .value()
            .is_some_and(|format| format.enabled())
        {
            indices.push(resource_index(
                resources,
                MaxwellThreeDResourceRole::VertexStream(stream as u8),
            )?);
        }
    }
    for resource in shaders.resources() {
        if !matches!(resource.role(), MaxwellThreeDResourceRole::Sampler(_)) {
            indices.push(resource_index(resources, resource.role())?);
        }
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn select_draw_attachments(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
) -> Result<DrawAttachmentSelection, MaxwellThreeDLoweringError> {
    let selection = state
        .render_targets()
        .color_target_selection()
        .value()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
    for (slot, target) in selection.active_targets().iter().copied().enumerate() {
        if selection.active_targets()[..slot].contains(&target) {
            return Err(MaxwellThreeDLoweringError::DuplicateColorTargetRoute { target });
        }
    }
    let mut colors = Vec::with_capacity(selection.target_count() as usize);
    for (slot, target) in selection.active_targets().iter().copied().enumerate() {
        let configured = &state.render_targets().color()[target as usize];
        match configured.readiness(true) {
            super::MaxwellThreeDAttachmentReadiness::Unprogrammed => {
                return Err(MaxwellThreeDLoweringError::ColorTargetRouteUnprogrammed {
                    slot: slot as u8,
                    target,
                });
            }
            super::MaxwellThreeDAttachmentReadiness::Disabled => {
                return Err(MaxwellThreeDLoweringError::ColorTargetRouteDisabled {
                    slot: slot as u8,
                    target,
                });
            }
            super::MaxwellThreeDAttachmentReadiness::Ready => {}
            _ => {
                return Err(MaxwellThreeDLoweringError::ColorTargetRouteIncomplete {
                    slot: slot as u8,
                    target,
                });
            }
        }
        let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
        let image = resolved_image(resources, index)?;
        if image.description().kind() != nixe_gpu::ImageKind::Color {
            return Err(MaxwellThreeDLoweringError::ResolvedResourceKindMismatch);
        }
        colors.push((target, index));
    }
    let depth_stencil = draw_depth_stencil_attachment_required(state)
        .then(|| {
            resources.resources().iter().position(|resource| {
                resource.role() == MaxwellThreeDResourceRole::DepthStencilTarget
            })
        })
        .flatten();
    if let Some(index) = depth_stencil {
        let image = resolved_image(resources, index)?;
        if image.description().kind() != nixe_gpu::ImageKind::DepthStencil {
            return Err(MaxwellThreeDLoweringError::ResolvedResourceKindMismatch);
        }
    }
    Ok(DrawAttachmentSelection {
        colors,
        depth_stencil,
    })
}

/// A configured depth/stencil target is not an attachment dependency when both
/// fragment tests are explicitly disabled. Unknown state stays conservative:
/// it must not silently discard a guest depth/stencil dependency.
fn draw_depth_stencil_attachment_required(state: &MaxwellThreeDState) -> bool {
    let (depth, stencil) = draw_depth_stencil_aspects(state);
    depth || stencil
}

/// Unknown test-enable state must still be validated by pipeline lowering, but
/// it does not prove that this operation references depth/stencil memory.
fn draw_depth_stencil_resource_required(state: &MaxwellThreeDState) -> bool {
    let (depth, stencil) = draw_depth_stencil_enable_state(state);
    depth == Some(true) || stencil == Some(true)
}

/// Returns the aspects that a draw may observe. Missing enable state remains
/// conservative and therefore requires the corresponding guest contents.
fn draw_depth_stencil_aspects(state: &MaxwellThreeDState) -> (bool, bool) {
    let (depth, stencil) = draw_depth_stencil_enable_state(state);
    (depth.unwrap_or(true), stencil.unwrap_or(true))
}

fn draw_depth_stencil_enable_state(state: &MaxwellThreeDState) -> (Option<bool>, Option<bool>) {
    let boolean = |register| {
        state
            .fixed_function()
            .register(register)
            .value()
            .and_then(|value| match value {
                MaxwellThreeDFixedFunctionValue::Boolean(value) => Some(*value),
                _ => None,
            })
    };
    (
        boolean(MaxwellThreeDFixedFunctionRegister::DepthTestEnable),
        boolean(MaxwellThreeDFixedFunctionRegister::StencilTestEnable),
    )
}

fn draw_depth_state(state: &MaxwellThreeDState) -> Result<DepthState, MaxwellThreeDLoweringError> {
    let register = |register| state.fixed_function().register(register).value();
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(test_enabled)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable)
    else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw("SET_DEPTH_TEST"));
    };
    if !test_enabled {
        return Ok(DepthState::DISABLED);
    }
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(write_enabled)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthWriteEnable)
    else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_DEPTH_WRITE",
        ));
    };
    let Some(MaxwellThreeDFixedFunctionValue::Compare(compare)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthCompare)
    else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw("SET_DEPTH_FUNC"));
    };
    Ok(DepthState::new(
        true,
        *write_enabled,
        neutral_depth_compare(*compare),
    ))
}

fn validate_draw_stencil_state(
    state: &MaxwellThreeDState,
) -> Result<(), MaxwellThreeDLoweringError> {
    match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::StencilTestEnable)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::Boolean(true)) => {
            let two_sided = state
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable)
                .value()
                == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true));
            Err(MaxwellThreeDLoweringError::UnsupportedStencilTestSemantics { two_sided })
        }
        _ => Ok(()),
    }
}

const fn neutral_depth_compare(compare: MaxwellThreeDCompareOp) -> DepthCompareOperation {
    match compare {
        MaxwellThreeDCompareOp::Never => DepthCompareOperation::Never,
        MaxwellThreeDCompareOp::Less => DepthCompareOperation::Less,
        MaxwellThreeDCompareOp::Equal => DepthCompareOperation::Equal,
        MaxwellThreeDCompareOp::LessEqual => DepthCompareOperation::LessEqual,
        MaxwellThreeDCompareOp::Greater => DepthCompareOperation::Greater,
        MaxwellThreeDCompareOp::NotEqual => DepthCompareOperation::NotEqual,
        MaxwellThreeDCompareOp::GreaterEqual => DepthCompareOperation::GreaterEqual,
        MaxwellThreeDCompareOp::Always => DepthCompareOperation::Always,
    }
}

#[cfg(test)]
const fn depth_stencil_attachment_required(
    depth_test_enabled: Option<bool>,
    stencil_test_enabled: Option<bool>,
) -> bool {
    !matches!(
        (depth_test_enabled, stencil_test_enabled),
        (Some(false), Some(false))
    )
}

fn validate_draw_surface_clip(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellThreeDLoweringError> {
    let horizontal = state.fixed_function().surface_clip_horizontal().value();
    let vertical = state.fixed_function().surface_clip_vertical().value();
    let (Some(horizontal), Some(vertical)) = (horizontal, vertical) else {
        return if horizontal.is_none() && vertical.is_none() {
            Ok(())
        } else {
            Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_SURFACE_CLIP_HORIZONTAL/VERTICAL",
            ))
        };
    };

    for index in attachments.attachment_indices() {
        let image = resolved_image(resources, index)?;
        let extent = image.description().extent();
        if horizontal.origin() != 0
            || vertical.origin() != 0
            || u32::from(horizontal.extent()) < extent.width
            || u32::from(vertical.extent()) < extent.height
        {
            return Err(MaxwellThreeDLoweringError::UnsupportedSurfaceClipSemantics);
        }
    }
    Ok(())
}

fn draw_viewport_transform(
    state: &MaxwellThreeDState,
) -> Result<Option<ViewportTransform>, MaxwellThreeDLoweringError> {
    let enabled = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(
            MaxwellThreeDViewportScaleOffsetEnable::Enabled,
        ));
    if !enabled {
        return Ok(None);
    }

    // The current draw contract selects viewport zero. T10 must provide
    // explicit viewport-index output evidence before another slot can be used.
    let viewport = &state.fixed_function().viewport()[0];
    let scale = viewport
        .scale()
        .each_ref()
        .map(|register| register.value().copied())
        .map(|value| value.map(|value| f32::from_bits(value.get())));
    let offset = viewport
        .offset()
        .each_ref()
        .map(|register| register.value().copied())
        .map(|value| value.map(|value| f32::from_bits(value.get())));
    let [Some(scale_x), Some(scale_y), Some(scale_z)] = scale else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_VIEWPORT_SCALE_X/Y/Z(0)",
        ));
    };
    let [Some(offset_x), Some(offset_y), Some(offset_z)] = offset else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_VIEWPORT_OFFSET_X/Y/Z(0)",
        ));
    };
    let Some(clip_min_z) = viewport
        .clip_min_z()
        .value()
        .copied()
        .map(|value| f32::from_bits(value.get()))
    else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_VIEWPORT_CLIP_MIN_Z(0)",
        ));
    };
    let Some(clip_max_z) = viewport
        .clip_max_z()
        .value()
        .copied()
        .map(|value| f32::from_bits(value.get()))
    else {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_VIEWPORT_CLIP_MAX_Z(0)",
        ));
    };
    ViewportTransform::new(
        [scale_x, scale_y, scale_z],
        [offset_x, offset_y, offset_z],
        [clip_min_z, clip_max_z],
    )
    .map(Some)
    .map_err(MaxwellThreeDLoweringError::Command)
}

fn prepare_resources(
    resources: &MaxwellThreeDResolvedResources,
    indices: &[usize],
    cache: &mut MaxwellThreeDLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
    invalidations: &mut Vec<ResourceDependency>,
) -> Result<Vec<Option<ResourceDependency>>, MaxwellThreeDLoweringError> {
    let mut result = vec![None; resources.resources().len()];
    for index in indices {
        let resource = resources
            .resources()
            .get(*index)
            .ok_or(MaxwellThreeDLoweringError::ResourceExhausted)?;
        let (allocation, allocation_description) = match resource {
            MaxwellThreeDResolvedResource::Buffer(value) => (
                value.view().backing().allocation(),
                value.allocation_description(),
            ),
            MaxwellThreeDResolvedResource::Image(value) => (
                value.view().bindings()[0].backing().allocation(),
                value.allocation_description(),
            ),
        };
        match cache.allocations.iter().find(|(id, _)| *id == allocation) {
            Some((_, current)) if *current != allocation_description => {
                return Err(MaxwellThreeDLoweringError::AllocationDescriptionChanged {
                    allocation,
                });
            }
            Some(_) => {}
            None => {
                cache.allocations.push((allocation, allocation_description));
                creations.push(BackendResourceCreateInfo::Allocation {
                    id: allocation,
                    description: allocation_description,
                });
            }
        }

        if let Some(record) = cache
            .views
            .iter()
            .find(|record| record.key.matches_resource(resource))
        {
            result[*index] = Some(record.dependency);
            continue;
        }
        // A compressed attachment initialized by a previous complete clear is
        // represented by its retained backend texture, not by importable guest
        // bytes. Preserve that texture across mapping-only identity changes;
        // validation above has already rejected any operation requiring an
        // unmaterialized aspect, and remains_current_for_image rejects CPU
        // writes or a changed image domain.
        if let MaxwellThreeDResolvedResource::Image(image) = resource
            && image.guest_layout().requires_materialization()
            && let Some(position) = cache
                .views
                .iter()
                .position(|record| record.remains_current_for_image(image))
        {
            let record = cache
                .views
                .get_mut(position)
                .expect("materialized view position came from the same cache");
            record.key = view_key(resource);
            result[*index] = Some(record.dependency);
            continue;
        }
        let key = view_key(resource);
        let invalidated = cache
            .views
            .iter()
            .filter(|record| {
                record.key.overlaps(&key) && !result.contains(&Some(record.dependency))
            })
            .map(|record| record.dependency)
            .collect::<Vec<_>>();
        cache
            .views
            .retain(|record| !invalidated.contains(&record.dependency));
        cache.accesses.retain(|(target, _)| {
            !invalidated
                .iter()
                .any(|dependency| dependency_matches_target(*dependency, *target))
        });
        let invalidated_descriptors = cache
            .descriptors
            .iter()
            .filter(|record| {
                record
                    .dependencies
                    .iter()
                    .any(|dependency| invalidated.contains(dependency))
            })
            .map(|record| ResourceDependency::DescriptorTable(record.id))
            .collect::<Vec<_>>();
        cache.descriptors.retain(|record| {
            !invalidated_descriptors.contains(&ResourceDependency::DescriptorTable(record.id))
        });
        for dependency in invalidated_descriptors {
            if !invalidations.contains(&dependency) {
                invalidations.push(dependency);
            }
        }
        for dependency in invalidated {
            if !invalidations.contains(&dependency) {
                invalidations.push(dependency);
            }
        }

        let materialization = match resource {
            MaxwellThreeDResolvedResource::Buffer(_) => ViewMaterialization::Direct,
            MaxwellThreeDResolvedResource::Image(value)
                if !value.guest_layout().requires_materialization() =>
            {
                ViewMaterialization::Direct
            }
            MaxwellThreeDResolvedResource::Image(value)
                if value.role() == MaxwellThreeDResourceRole::DepthStencilTarget =>
            {
                ViewMaterialization::CompressedDepthStencil {
                    depth: false,
                    stencil: false,
                }
            }
            MaxwellThreeDResolvedResource::Image(_) => ViewMaterialization::CompressedColor,
        };
        let cpu_writes = match resource {
            MaxwellThreeDResolvedResource::Buffer(_) => None,
            MaxwellThreeDResolvedResource::Image(value) => value.cpu_write_dependency().cloned(),
        };
        let dependency = match resource {
            MaxwellThreeDResolvedResource::Buffer(value) => {
                let id = BufferId::new(take_identity(cache)?);
                let view = BufferView::new(
                    id,
                    value.description(),
                    value.view().buffer_offset(),
                    value.view().backing().clone(),
                )
                .map_err(|_| MaxwellThreeDLoweringError::InvalidResolvedView {
                    role: value.role(),
                })?;
                creations.push(BackendResourceCreateInfo::Buffer {
                    id,
                    description: value.description(),
                    view: Some(view),
                });
                ResourceDependency::Buffer(id)
            }
            MaxwellThreeDResolvedResource::Image(value) => {
                let id = ImageId::new(take_identity(cache)?);
                let bindings = value
                    .view()
                    .bindings()
                    .iter()
                    .map(|binding| {
                        (
                            binding.subresources(),
                            binding.layout(),
                            binding.backing().clone(),
                        )
                    })
                    .collect();
                let view =
                    ImageView::new(id, value.description(), value.view().swizzle(), bindings)
                        .map_err(|_| MaxwellThreeDLoweringError::InvalidResolvedView {
                            role: value.role(),
                        })?;
                creations.push(BackendResourceCreateInfo::Image {
                    id,
                    description: value.description(),
                    view: value
                        .guest_layout()
                        .has_direct_canonical_representation()
                        .then_some(view),
                });
                ResourceDependency::Image(id)
            }
        };
        cache.views.push(ViewRecord {
            key,
            dependency,
            materialization,
            cpu_writes,
        });
        result[*index] = Some(dependency);
    }
    Ok(result)
}

fn binding_at(
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    index: usize,
) -> Result<ResourceDependency, MaxwellThreeDLoweringError> {
    bindings.get(index).and_then(|binding| *binding).ok_or(
        MaxwellThreeDLoweringError::MissingResolvedResource {
            role: resources
                .resources()
                .get(index)
                .ok_or(MaxwellThreeDLoweringError::ResourceExhausted)?
                .role(),
        },
    )
}

fn prepare_samplers(
    resources: &MaxwellThreeDResolvedResources,
    cache: &mut MaxwellThreeDLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
    invalidations: &mut Vec<ResourceDependency>,
) -> Result<Vec<(MaxwellThreeDResourceRole, ResourceDependency)>, MaxwellThreeDLoweringError> {
    let mut result = Vec::with_capacity(resources.samplers().len());
    for sampler in resources.samplers().iter().copied() {
        if let Some(record) = cache
            .samplers
            .iter()
            .find(|record| record.sampler == sampler)
            .copied()
        {
            result.push((sampler.role(), ResourceDependency::Sampler(record.id)));
            continue;
        }
        let retired = cache
            .samplers
            .iter()
            .filter(|record| record.sampler.role() == sampler.role())
            .map(|record| ResourceDependency::Sampler(record.id))
            .collect::<Vec<_>>();
        cache
            .samplers
            .retain(|record| record.sampler.role() != sampler.role());
        let retired_descriptors = cache
            .descriptors
            .iter()
            .filter(|record| {
                record
                    .dependencies
                    .iter()
                    .any(|dependency| retired.contains(dependency))
            })
            .map(|record| ResourceDependency::DescriptorTable(record.id))
            .collect::<Vec<_>>();
        cache.descriptors.retain(|record| {
            !retired_descriptors.contains(&ResourceDependency::DescriptorTable(record.id))
        });
        for dependency in retired.into_iter().chain(retired_descriptors) {
            if !invalidations.contains(&dependency) {
                invalidations.push(dependency);
            }
        }
        let id = SamplerId::new(take_identity(cache)?);
        creations.push(BackendResourceCreateInfo::Sampler {
            id,
            description: sampler.description().map_err(|_| {
                MaxwellThreeDLoweringError::InvalidResolvedView {
                    role: sampler.role(),
                }
            })?,
        });
        cache.samplers.push(SamplerRecord { sampler, id });
        result.push((sampler.role(), ResourceDependency::Sampler(id)));
    }
    Ok(result)
}

fn shader_resource_dependency(
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    samplers: &[(MaxwellThreeDResourceRole, ResourceDependency)],
    role: MaxwellThreeDResourceRole,
) -> Result<ResourceDependency, MaxwellThreeDLoweringError> {
    if let MaxwellThreeDResourceRole::Sampler(_) = role {
        return samplers
            .iter()
            .find_map(|(candidate, dependency)| (*candidate == role).then_some(*dependency))
            .ok_or(MaxwellThreeDLoweringError::MissingResolvedResource { role });
    }
    let index = resource_index(resources, role)?;
    binding_at(resources, bindings, index)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaxwellThreeDClearRegion {
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

impl MaxwellThreeDClearRegion {
    const fn attachment(width: u32, height: u32) -> Self {
        Self {
            min_x: 0,
            max_x: width,
            min_y: 0,
            max_y: height,
        }
    }

    fn intersect(
        &mut self,
        horizontal: super::MaxwellThreeDRectangle,
        vertical: super::MaxwellThreeDRectangle,
    ) {
        self.min_x = self.min_x.max(u32::from(horizontal.min));
        self.max_x = self.max_x.min(u32::from(horizontal.max));
        self.min_y = self.min_y.max(u32::from(vertical.min));
        self.max_y = self.max_y.min(u32::from(vertical.max));
    }

    const fn is_empty(self) -> bool {
        self.min_x >= self.max_x || self.min_y >= self.max_y
    }

    const fn fully_covers(self, width: u32, height: u32) -> bool {
        self.min_x == 0 && self.max_x == width && self.min_y == 0 && self.max_y == height
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MaxwellThreeDClearRegions {
    clear: Option<(super::MaxwellThreeDRectangle, super::MaxwellThreeDRectangle)>,
    scissor: Option<(super::MaxwellThreeDRectangle, super::MaxwellThreeDRectangle)>,
    viewport_clip: Option<(super::MaxwellThreeDRectangle, super::MaxwellThreeDRectangle)>,
}

impl MaxwellThreeDClearRegions {
    fn from_state(state: &MaxwellThreeDState) -> Result<Self, MaxwellThreeDLoweringError> {
        let clear = state.render_targets().clear();
        let control = clear.surface_control().value().copied();
        let clear = control
            .is_none_or(|control| control.use_clear_rect())
            .then(|| {
                Ok((
                    clear.horizontal().value().copied().ok_or(
                        MaxwellThreeDLoweringError::IncompleteClear("horizontal rectangle"),
                    )?,
                    clear.vertical().value().copied().ok_or(
                        MaxwellThreeDLoweringError::IncompleteClear("vertical rectangle"),
                    )?,
                ))
            })
            .transpose()?;
        let scissor = control
            .is_some_and(|control| control.use_scissor_zero())
            .then(|| {
                let scissor = &state.fixed_function().scissor()[0];
                Ok((
                    scissor.horizontal().value().copied().ok_or(
                        MaxwellThreeDLoweringError::IncompleteClear("SET_SCISSOR_HORIZONTAL(0)"),
                    )?,
                    scissor.vertical().value().copied().ok_or(
                        MaxwellThreeDLoweringError::IncompleteClear("SET_SCISSOR_VERTICAL(0)"),
                    )?,
                ))
            })
            .transpose()?;
        let viewport_clip = control
            .is_some_and(|control| control.use_viewport_clip_zero())
            .then(|| {
                let viewport = &state.fixed_function().viewport()[0];
                Ok((
                    viewport.clip_horizontal().value().copied().ok_or(
                        MaxwellThreeDLoweringError::IncompleteClear(
                            "SET_VIEWPORT_CLIP_HORIZONTAL(0)",
                        ),
                    )?,
                    viewport.clip_vertical().value().copied().ok_or(
                        MaxwellThreeDLoweringError::IncompleteClear(
                            "SET_VIEWPORT_CLIP_VERTICAL(0)",
                        ),
                    )?,
                ))
            })
            .transpose()?;
        Ok(Self {
            clear,
            scissor,
            viewport_clip,
        })
    }

    fn for_attachment(self, width: u32, height: u32) -> MaxwellThreeDClearRegion {
        let mut region = MaxwellThreeDClearRegion::attachment(width, height);
        for (horizontal, vertical) in [self.clear, self.scissor, self.viewport_clip]
            .into_iter()
            .flatten()
        {
            region.intersect(horizontal, vertical);
        }
        region
    }
}

fn clear_image_region(
    image: ImageId,
    subresources: ImageSubresourceRange,
    attachment_width: u32,
    attachment_height: u32,
    array_layer: u16,
    regions: MaxwellThreeDClearRegions,
) -> Result<ImageRegion, MaxwellThreeDLoweringError> {
    if array_layer != subresources.base_layer {
        return Err(MaxwellThreeDLoweringError::ClearOutsideAttachment);
    }
    let region = regions.for_attachment(attachment_width, attachment_height);
    if region.is_empty() {
        return Err(MaxwellThreeDLoweringError::EmptyClearRectangle);
    }
    Ok(ImageRegion {
        image,
        subresources,
        origin: ImageOrigin {
            x: region.min_x,
            y: region.min_y,
            z: 0,
        },
        extent: nixe_gpu::ImageExtent {
            width: region.max_x - region.min_x,
            height: region.max_y - region.min_y,
            depth: 1,
        },
    })
}

fn lower_clear(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
) -> Result<(Vec<GpuOperation>, Arc<[usize]>), MaxwellThreeDLoweringError> {
    let clear = state.render_targets().clear();
    let surface = clear
        .last_surface()
        .value()
        .copied()
        .ok_or(MaxwellThreeDLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
    if surface.color_mask() == 0 && !surface.depth() && !surface.stencil() {
        return Err(MaxwellThreeDLoweringError::EmptyClearMask);
    }
    let control = clear.surface_control().value().copied();
    if surface.stencil() && control.is_some_and(|control| control.respect_stencil_mask()) {
        return Err(MaxwellThreeDLoweringError::UnsupportedClearStencilMaskSemantics);
    }
    let regions = MaxwellThreeDClearRegions::from_state(state)?;
    let mut operations = Vec::new();
    let mut dirty = Vec::new();
    if surface.color_mask() != 0 {
        if surface.color_mask() != 0xf {
            return Err(MaxwellThreeDLoweringError::PartialColorClearUnsupported {
                mask: surface.color_mask(),
            });
        }
        let index = resource_index(
            resources,
            MaxwellThreeDResourceRole::ColorTarget(surface.color_target()),
        )?;
        let image = resolved_image(resources, index)?;
        let image_id = image_dependency(binding_at(resources, bindings, index)?)?;
        let subresources = image.view().bindings()[0].subresources();
        let region = clear_image_region(
            image_id,
            subresources,
            image.description().extent().width,
            image.description().extent().height,
            surface.array_layer(),
            regions,
        )?;
        let mut color = [0.0; 4];
        for (component, output) in clear.color().iter().zip(&mut color) {
            *output = f32::from_bits(
                component
                    .value()
                    .ok_or(MaxwellThreeDLoweringError::IncompleteClear("color value"))?
                    .get(),
            );
        }
        let operation = ClearOperation::image(
            region,
            image.description().kind(),
            image.description().format(),
            image.description().samples(),
            ClearValue::Color(color),
        )
        .map_err(MaxwellThreeDLoweringError::Command)?;
        operations.push(GpuOperation::new(
            GpuCommand::Clear(operation),
            [],
            [],
            CapabilityRequirements::none(),
        ));
        dirty.push(index);
    }
    if surface.depth() || surface.stencil() {
        let index = resource_index(resources, MaxwellThreeDResourceRole::DepthStencilTarget)?;
        let image = resolved_image(resources, index)?;
        let image_id = image_dependency(binding_at(resources, bindings, index)?)?;
        let subresources = image.view().bindings()[0].subresources();
        let region = clear_image_region(
            image_id,
            subresources,
            image.description().extent().width,
            image.description().extent().height,
            surface.array_layer(),
            regions,
        )?;
        let depth = surface
            .depth()
            .then(|| {
                clear
                    .depth()
                    .value()
                    .map(|value| f32::from_bits(value.get()))
                    .ok_or(MaxwellThreeDLoweringError::IncompleteClear("depth value"))
            })
            .transpose()?;
        let stencil = surface
            .stencil()
            .then(|| {
                clear
                    .stencil()
                    .value()
                    .copied()
                    .ok_or(MaxwellThreeDLoweringError::IncompleteClear("stencil value"))
            })
            .transpose()?;
        let value = match (depth, stencil) {
            (Some(depth), None) => ClearValue::Depth(depth),
            (None, Some(stencil)) => ClearValue::Stencil(stencil),
            (Some(depth), Some(stencil)) => ClearValue::DepthStencil { depth, stencil },
            (None, None) => unreachable!("depth/stencil clear branch requires one aspect"),
        };
        let operation = ClearOperation::image(
            region,
            image.description().kind(),
            image.description().format(),
            image.description().samples(),
            value,
        )
        .map_err(MaxwellThreeDLoweringError::Command)?;
        operations.push(GpuOperation::new(
            GpuCommand::Clear(operation),
            [],
            [],
            CapabilityRequirements::none(),
        ));
        dirty.push(index);
    }
    Ok((operations, dirty.into()))
}

#[allow(clippy::too_many_arguments)]
fn lower_draw(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    sampler_bindings: &[(MaxwellThreeDResourceRole, ResourceDependency)],
    shaders: &MaxwellThreeDTranslatedShaders,
    attachment_selection: &DrawAttachmentSelection,
    vertex_count: u32,
    cache: &mut MaxwellThreeDLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
) -> Result<(Vec<GpuOperation>, Arc<[usize]>), MaxwellThreeDLoweringError> {
    let arguments = draw_arguments(state, vertex_count)?;
    let DrawArguments::NonIndexed { first_vertex, .. } = arguments else {
        unreachable!("Maxwell vertex-array draw arguments are non-indexed")
    };
    validate_shader_stages(state, shaders)?;
    for translated in &shaders.shaders {
        let record = cache
            .shader_translations
            .get(translated.cache_fingerprint)
            .ok_or(MaxwellThreeDLoweringError::InvalidTranslatedShaders)?;
        #[cfg(debug_assertions)]
        assert_eq!(
            record.id, translated.shader,
            "XXH3-128 collision or inconsistent translated shader identity"
        );
        if record.module.stage() != translated.stage {
            return Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders);
        }
        if !record.published {
            creations.push(BackendResourceCreateInfo::Shader {
                id: record.id,
                description: ShaderDescription {
                    stage: translated.stage,
                },
                module: record.module.clone(),
            });
            cache
                .shader_translations
                .get_mut(translated.cache_fingerprint)
                .expect("validated shader translation fingerprint exists")
                .published = true;
        }
    }
    let topology = primitive_topology(
        state
            .vertex_input()
            .primitive()
            .active_begin()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("BEGIN"))?,
    )?;
    let mut vertex_buffers = Vec::new();
    for (index, stream) in state.vertex_input().streams().iter().enumerate() {
        let Some(stream_format) = stream.format().value().filter(|value| value.enabled()) else {
            continue;
        };
        let attributes = state
            .vertex_input()
            .attributes()
            .iter()
            .enumerate()
            .filter_map(|(location, attribute)| {
                attribute
                    .value()
                    .filter(|attribute| {
                        attribute.enabled() && usize::from(attribute.stream()) == index
                    })
                    .map(|attribute| (location, *attribute))
            })
            .map(|(location, attribute)| {
                Ok(VertexAttribute {
                    format: neutral_vertex_format(location as u8, attribute)?,
                    offset: u64::from(attribute.offset()),
                    shader_location: location as u32,
                })
            })
            .collect::<Result<Vec<_>, MaxwellThreeDLoweringError>>()?;
        if attributes.is_empty() {
            continue;
        }
        let resource = resource_index(
            resources,
            MaxwellThreeDResourceRole::VertexStream(index as u8),
        )?;
        let buffer = resolved_buffer(resources, resource)?;
        let region = BufferRegion {
            buffer: buffer_dependency(binding_at(resources, bindings, resource)?)?,
            range: BufferRange::new(0, buffer.description().size()).map_err(|_| {
                MaxwellThreeDLoweringError::InvalidResolvedView {
                    role: buffer.role(),
                }
            })?,
        };
        let instanced = stream.instanced().value().copied().unwrap_or(false);
        let frequency = stream.frequency().value().copied().unwrap_or(1);
        if instanced && frequency != 1 {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedVertexInstanceDivisor {
                    stream: index as u8,
                    divisor: frequency,
                },
            );
        }
        vertex_buffers.push(
            VertexBufferLayout::new(
                region,
                u64::from(stream_format.stride()),
                if instanced {
                    VertexStepMode::Instance
                } else {
                    VertexStepMode::Vertex
                },
                attributes,
            )
            .map_err(MaxwellThreeDLoweringError::Command)?,
        );
    }
    let triangle_rasterization = match state
        .raster()
        .fill_via_triangle()
        .value()
        .copied()
        .unwrap_or(MaxwellThreeDFillViaTriangleMode::Disabled)
    {
        MaxwellThreeDFillViaTriangleMode::Disabled => TriangleRasterization::Fill,
        MaxwellThreeDFillViaTriangleMode::FillBoundingBox => {
            if topology != PrimitiveTopology::Triangles {
                return Err(MaxwellThreeDLoweringError::UnsupportedFillRectangleDraw(
                    "primitive topology is not a triangle list",
                ));
            }
            if !first_vertex.is_multiple_of(3) || !vertex_count.is_multiple_of(3) {
                return Err(MaxwellThreeDLoweringError::UnsupportedFillRectangleDraw(
                    "vertex range is not aligned to complete triangles",
                ));
            }
            if vertex_buffers
                .iter()
                .any(|layout| layout.step_mode == VertexStepMode::Vertex)
            {
                return Err(MaxwellThreeDLoweringError::UnsupportedFillRectangleDraw(
                    "per-vertex attributes require vertex-pulling expansion",
                ));
            }
            TriangleRasterization::FillRectangle
        }
        MaxwellThreeDFillViaTriangleMode::FillAll => {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedFillViaTriangleSemantics(
                    MaxwellThreeDFillViaTriangleMode::FillAll,
                ),
            );
        }
    };
    let attachments = attachment_records(resources, bindings, attachment_selection)?;
    if attachments.is_empty() {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw("render target"));
    }
    let required_indices = draw_resource_indices(state, resources, attachment_selection, shaders)?;
    reject_draw_aliases(
        resources,
        &required_indices,
        &attachment_selection.attachment_indices(),
    )?;
    let render_pass_description = RenderPassDescription::new(
        attachments
            .iter()
            .map(|attachment| nixe_gpu::RenderPassAttachmentDescription {
                kind: attachment.kind,
                format: attachment.format,
                samples: attachment.samples,
            })
            .collect(),
    )
    .map_err(|_| MaxwellThreeDLoweringError::InvalidResourceCreation)?;
    let render_pass = if let Some(record) = cache
        .render_passes
        .iter()
        .find(|record| record.description == render_pass_description)
    {
        record.id
    } else {
        let id = RenderPassId::new(take_identity(cache)?);
        cache.render_passes.push(RenderPassRecord {
            description: render_pass_description.clone(),
            id,
        });
        creations.push(BackendResourceCreateInfo::RenderPass {
            id,
            description: render_pass_description.clone(),
        });
        id
    };

    let descriptor_roles = shaders
        .resources
        .iter()
        .map(|resource| resource.role)
        .collect::<Vec<_>>();
    let descriptor_binding_numbers = shaders
        .resources
        .iter()
        .map(|resource| resource.binding)
        .collect::<Vec<_>>();
    let descriptor_dependencies = shaders
        .resources
        .iter()
        .map(|resource| {
            shader_resource_dependency(resources, bindings, sampler_bindings, resource.role)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor_tables = if descriptor_roles.is_empty() {
        Vec::new()
    } else if let Some(record) = cache.descriptors.iter().find(|record| {
        record.roles.as_ref() == descriptor_roles.as_slice()
            && record.bindings.as_ref() == descriptor_binding_numbers.as_slice()
            && record.dependencies.as_ref() == descriptor_dependencies.as_slice()
    }) {
        vec![record.id]
    } else {
        let id = DescriptorTableId::new(take_identity(cache)?);
        let description = DescriptorTableDescription::new(
            shaders
                .resources
                .iter()
                .map(|resource| resource.kind)
                .collect(),
        )
        .map_err(|_| MaxwellThreeDLoweringError::InvalidTranslatedShaders)?;
        cache.descriptors.push(DescriptorRecord {
            roles: descriptor_roles.into_boxed_slice(),
            bindings: descriptor_binding_numbers.clone().into_boxed_slice(),
            dependencies: descriptor_dependencies.clone().into_boxed_slice(),
            id,
        });
        creations.push(BackendResourceCreateInfo::DescriptorTable {
            id,
            description,
            bindings: descriptor_binding_numbers
                .iter()
                .copied()
                .zip(descriptor_dependencies.iter().copied())
                .map(|(binding, resource)| DescriptorTableBinding { binding, resource })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        });
        vec![id]
    };

    for index in attachment_selection.attachment_indices_iter() {
        resolved_image(resources, index)?;
    }
    let pipeline = if let Some(pipeline) = cache.graphics_pipeline {
        pipeline
    } else {
        let id = PipelineId::new(take_identity(cache)?);
        cache.graphics_pipeline = Some(id);
        creations.push(BackendResourceCreateInfo::Pipeline {
            id,
            description: PipelineDescription {
                kind: PipelineKind::Graphics,
            },
        });
        id
    };

    let mut shader_accesses = Vec::new();
    let mut shader_dependencies = shaders
        .shaders
        .iter()
        .map(|shader| ResourceDependency::Shader(shader.shader))
        .collect::<Vec<_>>();
    for resource_use in &shaders.resources {
        let dependency =
            shader_resource_dependency(resources, bindings, sampler_bindings, resource_use.role)?;
        if !shader_dependencies.contains(&dependency) {
            shader_dependencies.push(dependency);
        }
        let Some(usage) = resource_use.usage else {
            continue;
        };
        let index = resource_index(resources, resource_use.role)?;
        let target = match &resources.resources()[index] {
            MaxwellThreeDResolvedResource::Buffer(buffer) => AccessTarget::Buffer {
                buffer: buffer_dependency(binding_at(resources, bindings, index)?)?,
                range: BufferRange::new(0, buffer.description().size()).map_err(|_| {
                    MaxwellThreeDLoweringError::InvalidResolvedView {
                        role: buffer.role(),
                    }
                })?,
            },
            MaxwellThreeDResolvedResource::Image(image) => AccessTarget::Image {
                image: image_dependency(binding_at(resources, bindings, index)?)?,
                subresources: image.view().bindings()[0].subresources(),
            },
        };
        shader_accesses.push(ResourceAccess::new(
            target,
            AccessScope::new(resource_use.stages, AccessMode::Read, usage).map_err(|_| {
                MaxwellThreeDLoweringError::InvalidShaderResourceUse {
                    role: resource_use.role,
                }
            })?,
        ));
    }
    let mut draw = PreparedDraw::new(
        pipeline,
        render_pass,
        topology,
        descriptor_tables,
        vertex_buffers,
        None,
    )
    .map_err(MaxwellThreeDLoweringError::Command)?;
    draw = draw.with_triangle_rasterization(triangle_rasterization);
    if let Some(alpha_test) = draw_alpha_test_state(state)? {
        draw = draw.with_alpha_test(alpha_test);
    }
    if let Some(viewport_transform) = draw_viewport_transform(state)? {
        draw = draw.with_viewport_transform(viewport_transform);
    }
    if attachment_selection.depth_stencil.is_some() {
        draw = draw.with_depth_state(draw_depth_state(state)?);
    }
    let draw = Arc::new(draw);
    let operations = [
        GpuOperation::new(
            GpuCommand::RenderPass(
                RenderPassOperation::begin(draw.render_pass, render_pass_description, attachments)
                    .map_err(MaxwellThreeDLoweringError::Command)?,
            ),
            [],
            [],
            CapabilityRequirements::none(),
        ),
        GpuOperation::new(
            GpuCommand::Draw(
                DrawOperation::new(Arc::clone(&draw), arguments)
                    .map_err(MaxwellThreeDLoweringError::Command)?,
            ),
            shader_accesses,
            shader_dependencies,
            CapabilityRequirements::new(
                shaders
                    .shaders
                    .iter()
                    .map(|shader| nixe_gpu::CapabilityRequirement::ShaderStage(shader.stage)),
            ),
        ),
        GpuOperation::new(
            GpuCommand::RenderPass(RenderPassOperation::end(draw.render_pass)),
            [],
            [],
            CapabilityRequirements::none(),
        ),
    ];
    let record = PreparedDrawRecord {
        state: state.draw_state_identity(),
        resources: resources.identity(),
        shaders: shaders.identity(),
        operations,
        dirty_images: attachment_selection.attachment_indices().into(),
    };
    let operations = record.operations(arguments)?;
    let dirty = Arc::clone(&record.dirty_images);
    cache.prepared_draw = Some(record);
    Ok((operations.into(), dirty))
}

fn sequence_with_transitions(
    commands: impl IntoIterator<Item = GpuOperation>,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<Vec<GpuOperation>, MaxwellThreeDLoweringError> {
    let commands = commands.into_iter();
    let mut result = Vec::with_capacity(commands.size_hint().0);
    for command in commands {
        let mut transitions = Vec::new();
        for access in command.accesses() {
            if let Some((_, before)) = cache
                .accesses
                .iter()
                .find(|(target, _)| *target == access.target())
                && *before != access.scope()
            {
                transitions.push(
                    ResourceTransition::new(access.target(), *before, access.scope())
                        .map_err(|_| MaxwellThreeDLoweringError::InvalidTransition)?,
                );
            }
        }
        if !transitions.is_empty() {
            result.push(GpuOperation::new(
                GpuCommand::Barrier(
                    BarrierOperation::new(transitions)
                        .map_err(MaxwellThreeDLoweringError::Command)?,
                ),
                [],
                [],
                CapabilityRequirements::none(),
            ));
        }
        for access in command.accesses() {
            let previous = cache
                .accesses
                .iter()
                .find(|(target, _)| *target == access.target())
                .map(|(_, scope)| *scope);
            if previous.is_some_and(|scope| scope != access.scope()) {
                let (_, scope) = cache
                    .accesses
                    .iter_mut()
                    .find(|(target, _)| *target == access.target())
                    .expect("access found immediately before mutation");
                *scope = access.scope();
            } else if previous.is_none() {
                cache.accesses.push((access.target(), access.scope()));
            }
        }
        result.push(command);
    }
    Ok(result)
}

fn validate_shader_stages(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<(), MaxwellThreeDLoweringError> {
    let mut expected_count = 0;
    for pipeline in state.shader_bindings().pipeline() {
        if pipeline.enabled().value() != Some(&true) {
            continue;
        }
        let stage = match pipeline
            .stage()
            .value()
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("shader stage"))?
        {
            MaxwellThreeDShaderStage::Vertex => ShaderStage::Vertex,
            MaxwellThreeDShaderStage::TessellationInit => ShaderStage::TessellationControl,
            MaxwellThreeDShaderStage::Tessellation => ShaderStage::TessellationEvaluation,
            MaxwellThreeDShaderStage::Geometry => ShaderStage::Geometry,
            MaxwellThreeDShaderStage::Pixel => ShaderStage::Fragment,
            MaxwellThreeDShaderStage::VertexCullBeforeFetch => {
                return Err(MaxwellThreeDLoweringError::UnsupportedShaderStage(
                    MaxwellThreeDShaderStage::VertexCullBeforeFetch,
                ));
            }
        };
        expected_count += 1;
        if !shaders.shaders.iter().any(|shader| shader.stage == stage) {
            return Err(MaxwellThreeDLoweringError::TranslatedShaderStageMismatch);
        }
    }
    if expected_count == 0 || expected_count != shaders.shaders.len() {
        return Err(MaxwellThreeDLoweringError::TranslatedShaderStageMismatch);
    }
    Ok(())
}

fn shader_pipeline_stages(
    stage: ShaderStage,
) -> Result<PipelineStages, MaxwellThreeDLoweringError> {
    match stage {
        ShaderStage::Vertex => Ok(PipelineStages::VERTEX_SHADER),
        ShaderStage::TessellationControl => Ok(PipelineStages::TESSELLATION_CONTROL_SHADER),
        ShaderStage::TessellationEvaluation => Ok(PipelineStages::TESSELLATION_EVALUATION_SHADER),
        ShaderStage::Geometry => Ok(PipelineStages::GEOMETRY_SHADER),
        ShaderStage::Fragment => Ok(PipelineStages::FRAGMENT_SHADER),
        ShaderStage::Compute => Err(MaxwellThreeDLoweringError::InvalidTranslatedShaders),
    }
}

fn primitive_topology(
    begin: MaxwellThreeDBegin,
) -> Result<PrimitiveTopology, MaxwellThreeDLoweringError> {
    if begin.preserve_primitive_id() {
        return Err(MaxwellThreeDLoweringError::UnsupportedPrimitiveIdContinuation);
    }
    if begin.split_mode() != 0 {
        return Err(MaxwellThreeDLoweringError::UnsupportedPrimitiveSplitMode(
            begin.split_mode(),
        ));
    }
    match begin.topology() {
        0 => Ok(PrimitiveTopology::Points),
        1 => Ok(PrimitiveTopology::Lines),
        3 => Ok(PrimitiveTopology::LineStrip),
        4 => Ok(PrimitiveTopology::Triangles),
        5 => Ok(PrimitiveTopology::TriangleStrip),
        6 => Ok(PrimitiveTopology::TriangleFan),
        14 => Ok(PrimitiveTopology::Patches),
        topology => Err(MaxwellThreeDLoweringError::UnsupportedTopology(topology)),
    }
}

fn neutral_first_instance(base: u32, relative: u32) -> Result<u32, MaxwellThreeDLoweringError> {
    base.checked_add(relative)
        .ok_or(MaxwellThreeDLoweringError::InstanceIndexOverflow { base, relative })
}

fn draw_arguments(
    state: &MaxwellThreeDState,
    vertex_count: u32,
) -> Result<DrawArguments, MaxwellThreeDLoweringError> {
    if vertex_count == 0 {
        return Err(MaxwellThreeDLoweringError::EmptyDraw);
    }
    let first_vertex = *state
        .vertex_input()
        .primitive()
        .vertex_array_start()
        .value()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
            "VERTEX_ARRAY_START",
        ))?;
    let base_instance = state
        .vertex_input()
        .assembly()
        .global_base_instance_index()
        .value()
        .copied()
        .unwrap_or(0);
    let relative_instance = state.vertex_input().primitive().instance_index();
    Ok(DrawArguments::NonIndexed {
        first_vertex,
        vertex_count,
        first_instance: neutral_first_instance(base_instance, relative_instance)?,
        instance_count: 1,
    })
}

fn neutral_vertex_format(
    attribute: u8,
    format: super::MaxwellThreeDVertexAttributeFormat,
) -> Result<VertexFormat, MaxwellThreeDLoweringError> {
    let widths = format
        .component_widths()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_VERTEX_ATTRIBUTE_A",
        ))?;
    let numerical = format
        .numerical_type()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_VERTEX_ATTRIBUTE_A",
        ))?;
    if format.swap_red_blue() {
        return Err(
            MaxwellThreeDLoweringError::UnsupportedVertexAttributeFormat {
                attribute,
                component_widths: widths,
                numerical_type: numerical,
                swap_red_blue: true,
            },
        );
    }

    // Maxwell field values are pinned to NVIDIA's public class header:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cl9097.h#L1021-L1055
    let scaled_layout = || {
        let (width, components) = match widths.raw() {
            0x1d => (VertexComponentWidth::Bits8, VertexComponentCount::One),
            0x18 => (VertexComponentWidth::Bits8, VertexComponentCount::Two),
            0x13 => (VertexComponentWidth::Bits8, VertexComponentCount::Three),
            0x0a => (VertexComponentWidth::Bits8, VertexComponentCount::Four),
            0x1b => (VertexComponentWidth::Bits16, VertexComponentCount::One),
            0x0f => (VertexComponentWidth::Bits16, VertexComponentCount::Two),
            0x05 => (VertexComponentWidth::Bits16, VertexComponentCount::Three),
            0x03 => (VertexComponentWidth::Bits16, VertexComponentCount::Four),
            0x12 => (VertexComponentWidth::Bits32, VertexComponentCount::One),
            0x04 => (VertexComponentWidth::Bits32, VertexComponentCount::Two),
            0x02 => (VertexComponentWidth::Bits32, VertexComponentCount::Three),
            0x01 => (VertexComponentWidth::Bits32, VertexComponentCount::Four),
            _ => return None,
        };
        Some((width, components))
    };
    let vertex = match (widths.raw(), numerical) {
        (0x01, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32x4,
        (0x02, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32x3,
        (0x04, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32x2,
        (0x12, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float32,
        (0x03, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::Float) => VertexFormat::Float16x2,
        (0x01, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32x4,
        (0x02, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32x3,
        (0x04, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32x2,
        (0x12, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint32,
        (0x01, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32x4,
        (0x02, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32x3,
        (0x04, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32x2,
        (0x12, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint32,
        (0x03, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint16x2,
        (0x03, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint16x2,
        (0x03, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm16x2,
        (0x03, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm16x4,
        (0x0f, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm16x2,
        (0x0a, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint8x4,
        (0x18, MaxwellThreeDVertexNumericalType::SignedInteger) => VertexFormat::Sint8x2,
        (0x0a, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint8x4,
        (0x18, MaxwellThreeDVertexNumericalType::UnsignedInteger) => VertexFormat::Uint8x2,
        (0x0a, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm8x4,
        (0x18, MaxwellThreeDVertexNumericalType::SignedNormalized) => VertexFormat::Snorm8x2,
        (0x0a, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm8x4,
        (0x18, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => VertexFormat::Unorm8x2,
        (0x30, MaxwellThreeDVertexNumericalType::UnsignedNormalized) => {
            VertexFormat::Unorm10_10_10_2
        }
        (_, MaxwellThreeDVertexNumericalType::UnsignedScaled) => {
            let (width, components) = scaled_layout().ok_or(
                MaxwellThreeDLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                },
            )?;
            VertexFormat::Uscaled { width, components }
        }
        (_, MaxwellThreeDVertexNumericalType::SignedScaled) => {
            let (width, components) = scaled_layout().ok_or(
                MaxwellThreeDLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                },
            )?;
            VertexFormat::Sscaled { width, components }
        }
        _ => {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                },
            );
        }
    };
    Ok(vertex)
}

fn attachment_records(
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    selection: &DrawAttachmentSelection,
) -> Result<Vec<RenderAttachment>, MaxwellThreeDLoweringError> {
    selection
        .attachment_indices()
        .into_iter()
        .map(|index| {
            let image = resolved_image(resources, index)?;
            Ok(RenderAttachment {
                image: image_dependency(binding_at(resources, bindings, index)?)?,
                subresources: image.view().bindings()[0].subresources(),
                kind: image.description().kind(),
                format: image.description().format(),
                samples: image.description().samples(),
                load: AttachmentLoad::Load,
                store: AttachmentStore::Store,
            })
        })
        .collect()
}

fn reject_draw_aliases(
    resources: &MaxwellThreeDResolvedResources,
    required_indices: &[usize],
    attachment_indices: &[usize],
) -> Result<(), MaxwellThreeDLoweringError> {
    for alias in resources.aliases() {
        if !required_indices.contains(&alias.first()) || !required_indices.contains(&alias.second())
        {
            continue;
        }
        let first = resources.resources()[alias.first()].role();
        let second = resources.resources()[alias.second()].role();
        let first_writes = attachment_indices.contains(&alias.first());
        let second_writes = attachment_indices.contains(&alias.second());
        if first_writes || second_writes {
            return Err(MaxwellThreeDLoweringError::AliasedDrawResources { first, second });
        }
    }
    Ok(())
}

fn view_key(resource: &MaxwellThreeDResolvedResource) -> ViewKey {
    match resource {
        MaxwellThreeDResolvedResource::Buffer(value) => ViewKey::Buffer {
            description: value.description(),
            buffer_offset: value.view().buffer_offset(),
            backing: value.view().backing().clone(),
            mappings: value.shared_mappings(),
        },
        MaxwellThreeDResolvedResource::Image(value) => ViewKey::Image {
            description: value.description(),
            swizzle: value.view().swizzle(),
            guest_format: value.guest_format(),
            guest_pte_kind: value.guest_layout().pte_kind(),
            guest_compression_enabled: value.guest_layout().requires_materialization(),
            bindings: value
                .view()
                .bindings()
                .iter()
                .map(|binding| {
                    (
                        binding.subresources(),
                        binding.layout(),
                        binding.backing().clone(),
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            mappings: value.shared_mappings(),
        },
    }
}

fn color_representation_record(
    image: &super::MaxwellThreeDResolvedImage,
) -> ColorRepresentationRecord {
    ColorRepresentationRecord {
        description: image.description(),
        swizzle: image.view().swizzle(),
        guest_format: image.guest_format(),
        guest_pte_kind: image.guest_layout().pte_kind(),
        guest_compression_enabled: image.guest_layout().requires_materialization(),
        bindings: image
            .view()
            .bindings()
            .iter()
            .map(|binding| ColorRepresentationBinding {
                subresources: binding.subresources(),
                layout: binding.layout(),
                backing: binding.backing().clone(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        cpu_writes: image.cpu_write_dependency().cloned(),
    }
}

fn resource_index(
    resources: &MaxwellThreeDResolvedResources,
    role: MaxwellThreeDResourceRole,
) -> Result<usize, MaxwellThreeDLoweringError> {
    resources
        .resources()
        .iter()
        .position(|resource| resource.role() == role)
        .ok_or(MaxwellThreeDLoweringError::MissingResolvedResource { role })
}

fn resolved_buffer(
    resources: &MaxwellThreeDResolvedResources,
    index: usize,
) -> Result<&super::MaxwellThreeDResolvedBuffer, MaxwellThreeDLoweringError> {
    match &resources.resources()[index] {
        MaxwellThreeDResolvedResource::Buffer(value) => Ok(value),
        _ => Err(MaxwellThreeDLoweringError::ResolvedResourceKindMismatch),
    }
}

fn resolved_image(
    resources: &MaxwellThreeDResolvedResources,
    index: usize,
) -> Result<&super::MaxwellThreeDResolvedImage, MaxwellThreeDLoweringError> {
    match &resources.resources()[index] {
        MaxwellThreeDResolvedResource::Image(value) => Ok(value),
        _ => Err(MaxwellThreeDLoweringError::ResolvedResourceKindMismatch),
    }
}

fn buffer_dependency(
    dependency: ResourceDependency,
) -> Result<BufferId, MaxwellThreeDLoweringError> {
    match dependency {
        ResourceDependency::Buffer(id) => Ok(id),
        _ => Err(MaxwellThreeDLoweringError::ResolvedResourceKindMismatch),
    }
}

fn image_dependency(dependency: ResourceDependency) -> Result<ImageId, MaxwellThreeDLoweringError> {
    match dependency {
        ResourceDependency::Image(id) => Ok(id),
        _ => Err(MaxwellThreeDLoweringError::ResolvedResourceKindMismatch),
    }
}

fn dependency_matches_target(dependency: ResourceDependency, target: AccessTarget) -> bool {
    matches!(
        (dependency, target),
        (ResourceDependency::Buffer(left), AccessTarget::Buffer { buffer: right, .. }) if left == right
    ) || matches!(
        (dependency, target),
        (ResourceDependency::Image(left), AccessTarget::Image { image: right, .. }) if left == right
    )
}

fn take_identity(
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<u64, MaxwellThreeDLoweringError> {
    let value = cache.next_identity;
    cache.next_identity = value
        .checked_add(1)
        .ok_or(MaxwellThreeDLoweringError::ResourceExhausted)?;
    Ok(value)
}

/// Typed failure before any cache or backend effect is published.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDLoweringError {
    ContradictoryState {
        reason: &'static str,
    },
    TriggerStateMismatch,
    UnsupportedRenderEnableMode(MaxwellThreeDRenderEnableMode),
    UnsupportedConditionalLoadConstantBufferSemantics,
    VisibleCallLimitExceeded {
        stage: ShaderStage,
        required: u16,
        limit: u16,
    },
    UnsupportedColorReductionSemantics,
    UnsupportedConstantColorRenderingSemantics,
    UnsupportedApiMandatedEarlyZSemantics,
    UnsupportedPostZPixelShaderImaskSemantics,
    UnsupportedPixelShaderInterlockSemantics(MaxwellThreeDPixelShaderInterlockControl),
    UnsupportedGlobalBaseVertexIndex(u32),
    UnsupportedCsaaSemantics,
    UnsupportedCoverageToColorSemantics(MaxwellThreeDCoverageToColor),
    UnsupportedAlphaToCoverageOverrideSemantics(MaxwellThreeDAlphaToCoverageOverride),
    UnsupportedTirSemantics {
        control: Option<MaxwellThreeDTirControl>,
    },
    UnsupportedHybridAntiAliasSemantics(MaxwellThreeDHybridAntiAliasControl),
    UnsupportedSampleLocationsSemantics {
        group: u8,
        value: MaxwellThreeDSampleLocationGroup,
    },
    UnsupportedPsOutputSampleMaskSemantics,
    UnsupportedReplicatedColorTargetOutputSemantics,
    UnsupportedRenderTargetIndexOffsetSemantics(MaxwellThreeDRenderTargetIndexOffset),
    UnsupportedRenderTargetLayerSemantics(MaxwellThreeDRenderTargetLayer),
    UnsupportedShaderLocalMemorySemantics {
        default_size_per_warp: MaxwellThreeDShaderLocalMemoryPerWarpSize,
    },
    UnsupportedViewportPixelCenterSemantics(MaxwellThreeDViewportPixelCenter),
    UnsupportedViewportCoordinateSwizzleSemantics {
        viewport: u8,
        swizzle: MaxwellThreeDViewportCoordinateSwizzle,
    },
    UnsupportedSurfaceClipSemantics,
    UnsupportedWindowClipSemantics,
    UnsupportedClipIdTestSemantics,
    UnsupportedStencilTestSemantics {
        two_sided: bool,
    },
    UnsupportedClearStencilMaskSemantics,
    UnsupportedAliasedLineWidthSemantics,
    UnsupportedAntiAliasedLineSemantics,
    UnsupportedLineStippleSemantics {
        factor: u8,
        pattern: u16,
    },
    UnsupportedPolygonClipGeneratedEdgeSemantics,
    UnsupportedVertexAttributeFormat {
        attribute: u8,
        component_widths: super::MaxwellThreeDVertexComponentWidths,
        numerical_type: super::MaxwellThreeDVertexNumericalType,
        swap_red_blue: bool,
    },
    UnsupportedVertexInstanceDivisor {
        stream: u8,
        divisor: u32,
    },
    InvalidPatchSize(MaxwellThreeDPatchSize),
    UnsupportedPatchSemantics(MaxwellThreeDPatchSize),
    UnsupportedPointSpriteCoordinatesSemantics(MaxwellThreeDPointSpriteSelect),
    UnsupportedAttributePointSizeSemantics {
        slot: u8,
    },
    UnsupportedPointSpriteSemantics,
    UnsupportedAntiAliasedPointSemantics,
    UnsupportedPointCenterSemantics(MaxwellThreeDPointCenterMode),
    UnsupportedFillViaTriangleSemantics(MaxwellThreeDFillViaTriangleMode),
    UnsupportedFillRectangleDraw(&'static str),
    UnsupportedConservativeRasterSemantics,
    UnsupportedPolygonSmoothSemantics,
    UnsupportedPolygonStippleSemantics,
    UnsupportedEdgeFlagSemantics(MaxwellThreeDEdgeFlag),
    UnsupportedShadeModeSemantics(MaxwellThreeDShadeMode),
    UnsupportedProvokingVertexSemantics(MaxwellThreeDProvokingVertex),
    UnsupportedTwoSidedLightSemantics,
    UnsupportedColorClampSemantics,
    UnsupportedPixelShaderSaturateSemantics {
        output: u8,
        range: MaxwellThreeDPixelShaderClampRange,
    },
    UnsupportedBlendSemantics {
        target: Option<u8>,
    },
    UnsupportedIteratedBlendSemantics {
        value: MaxwellThreeDIteratedBlend,
        pass_count: Option<u8>,
    },
    IncompleteLogicOpState,
    UnsupportedLogicOpSemantics(MaxwellThreeDLogicOp),
    IncompleteColorWriteState {
        target: u8,
        mask_register: u8,
    },
    UnsupportedColorWriteMask {
        target: u8,
        mask_register: u8,
        mask: super::MaxwellThreeDColorMask,
    },
    IncompleteAlphaTestState(&'static str),
    CompressedDepthImportRequired {
        kind: u8,
    },
    CompressedColorImportRequired {
        target: u8,
    },
    ShaderTranslationRequired,
    InvalidTranslatedShaders,
    TranslatedShaderStageMismatch,
    TranslatedShaderMemoryConfigurationMismatch {
        stage: ShaderStage,
        configured: MaxwellThreeDDirectlyAddressableMemory,
        required: MaxwellThreeDDirectlyAddressableMemory,
    },
    UnsupportedShaderStage(MaxwellThreeDShaderStage),
    InvalidShaderResourceUse {
        role: MaxwellThreeDResourceRole,
    },
    MissingResolvedResource {
        role: MaxwellThreeDResourceRole,
    },
    ResolvedResourceKindMismatch,
    InvalidResolvedView {
        role: MaxwellThreeDResourceRole,
    },
    AllocationDescriptionChanged {
        allocation: nixe_gpu::GpuAllocationId,
    },
    IncompleteClear(&'static str),
    EmptyClearMask,
    EmptyClearRectangle,
    PartialColorClearUnsupported {
        mask: u8,
    },
    ClearOutsideAttachment,
    IncompleteDraw(&'static str),
    IncompleteBlendState {
        target: Option<u8>,
        field: &'static str,
    },
    ColorTargetRouteUnprogrammed {
        slot: u8,
        target: u8,
    },
    ColorTargetRouteDisabled {
        slot: u8,
        target: u8,
    },
    ColorTargetRouteIncomplete {
        slot: u8,
        target: u8,
    },
    DuplicateColorTargetRoute {
        target: u8,
    },
    EmptyDraw,
    UnsupportedPrimitiveIdContinuation,
    UnsupportedPrimitiveSplitMode(u8),
    InstanceIndexOverflow {
        base: u32,
        relative: u32,
    },
    UnsupportedTopology(u8),
    AliasedDrawResources {
        first: MaxwellThreeDResourceRole,
        second: MaxwellThreeDResourceRole,
    },
    InvalidTransition,
    InvalidResourceCreation,
    Command(CommandDescriptionError),
    Capability(BackendCapabilityError),
    ResourceExhausted,
}

impl Display for MaxwellThreeDLoweringError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContradictoryState { reason } => {
                write!(formatter, "contradictory Maxwell 3D state: {reason}")
            }
            Self::TriggerStateMismatch => {
                formatter.write_str("3D trigger does not match its immutable state snapshot")
            }
            Self::UnsupportedRenderEnableMode(mode) => write!(
                formatter,
                "MAXWELL_B render-enable mode has no verified neutral execution: mode={mode:?}"
            ),
            Self::UnsupportedConditionalLoadConstantBufferSemantics => formatter.write_str(
                "MAXWELL_B conditional constant-buffer load has no verified execution semantics",
            ),
            Self::VisibleCallLimitExceeded {
                stage,
                required,
                limit,
            } => write!(
                formatter,
                "translated Maxwell shader exceeds SET_API_VISIBLE_CALL_LIMIT: stage={stage:?} required={required} limit={limit}"
            ),
            Self::UnsupportedColorReductionSemantics => formatter.write_str(
                "MAXWELL_B enabled color reduction has no verified neutral threshold evaluation or color-output semantics",
            ),
            Self::UnsupportedConstantColorRenderingSemantics => formatter.write_str(
                "MAXWELL_B enabled constant-color rendering is not represented by the neutral pipeline",
            ),
            Self::UnsupportedApiMandatedEarlyZSemantics => formatter.write_str(
                "MAXWELL_B API-mandated early depth/stencil ordering is not represented by the neutral pipeline",
            ),
            Self::UnsupportedPostZPixelShaderImaskSemantics => formatter.write_str(
                "MAXWELL_B post-Z pixel-shader invocation mask is not represented by the neutral pipeline",
            ),
            Self::UnsupportedPixelShaderInterlockSemantics(value) => write!(
                formatter,
                "MAXWELL_B pixel-shader interlock is not represented by the neutral pipeline: control={value:?}"
            ),
            Self::UnsupportedGlobalBaseVertexIndex(value) => write!(
                formatter,
                "MAXWELL_B global base vertex index cannot be represented independently from vertex-buffer addressing: value={value}"
            ),
            Self::UnsupportedCsaaSemantics => formatter.write_str(
                "MAXWELL_B enabled CSAA has no verified coverage sampling, resolve, capability, or coherency semantics",
            ),
            Self::UnsupportedCoverageToColorSemantics(value) => write!(
                formatter,
                "MAXWELL_B coverage-to-color output is not represented by the neutral pipeline: color-target={}",
                value.color_target()
            ),
            Self::UnsupportedAlphaToCoverageOverrideSemantics(value) => write!(
                formatter,
                "MAXWELL_B alpha-to-coverage override qualification is not represented by the neutral pipeline: qualify-by-aa={} qualify-by-ps-sample-mask={}",
                value.qualify_by_anti_alias_enable(),
                value.qualify_by_pixel_shader_sample_mask()
            ),
            Self::UnsupportedTirSemantics { control } => write!(
                formatter,
                "MAXWELL_B enabled target-independent rasterization has no neutral raster, coverage, alpha-to-coverage, or query representation: control={control:?}"
            ),
            Self::UnsupportedHybridAntiAliasSemantics(value) => write!(
                formatter,
                "MAXWELL_B hybrid antialiasing is not represented by the neutral raster pipeline: passes={} centroid={:?} passes-extended={}",
                value.passes(),
                value.centroid(),
                value.passes_extended()
            ),
            Self::UnsupportedSampleLocationsSemantics { group, value } => write!(
                formatter,
                "MAXWELL_B custom sample locations are not represented by the neutral raster pipeline: group={group} raw=0x{:08x}",
                value.raw()
            ),
            Self::UnsupportedPsOutputSampleMaskSemantics => formatter.write_str(
                "MAXWELL_B effective pixel-shader sample-mask output has no shader translation or neutral backend representation",
            ),
            Self::UnsupportedReplicatedColorTargetOutputSemantics => formatter.write_str(
                "MAXWELL_B disabled separate MRT fragment data requires replicating fragment color output zero to every active color target",
            ),
            Self::UnsupportedRenderTargetIndexOffsetSemantics(value) => write!(
                formatter,
                "MAXWELL_B viewport-index render-target routing is not represented by the neutral attachment contract: mode={value:?}"
            ),
            Self::UnsupportedRenderTargetLayerSemantics(value) => write!(
                formatter,
                "MAXWELL_B render-target layer routing is not represented by shader translation or the neutral attachment contract: layer={} control={:?}",
                value.layer(),
                value.control()
            ),
            Self::UnsupportedShaderLocalMemorySemantics {
                default_size_per_warp,
            } => write!(
                formatter,
                "MAXWELL_B active shader-local-memory allocation has no translated-shader or neutral backend representation: default-size-per-warp={}",
                default_size_per_warp.bytes()
            ),
            Self::UnsupportedViewportPixelCenterSemantics(center) => write!(
                formatter,
                "MAXWELL_B viewport pixel-center convention is not represented by the neutral pipeline contract: center={center:?}"
            ),
            Self::UnsupportedViewportCoordinateSwizzleSemantics { viewport, swizzle } => write!(
                formatter,
                "MAXWELL_B viewport coordinate swizzle is not represented by the neutral pipeline contract: viewport={viewport} components={:?}",
                swizzle.components()
            ),
            Self::UnsupportedSurfaceClipSemantics => formatter.write_str(
                "MAXWELL_B programmed surface clip has no verified neutral draw-time region-composition semantics",
            ),
            Self::UnsupportedWindowClipSemantics => formatter.write_str(
                "MAXWELL_B enabled window clipping has no neutral pipeline or backend rasterization semantics",
            ),
            Self::UnsupportedClipIdTestSemantics => formatter.write_str(
                "MAXWELL_B enabled clip-ID testing has no implemented extent, surface-ID, comparison, or backend rasterization semantics",
            ),
            Self::UnsupportedStencilTestSemantics { two_sided } => write!(
                formatter,
                "MAXWELL_B enabled stencil testing has no neutral pipeline representation: two-sided={two_sided}"
            ),
            Self::UnsupportedClearStencilMaskSemantics => formatter.write_str(
                "MAXWELL_B stencil-masked clear has no neutral backend representation",
            ),
            Self::UnsupportedAliasedLineWidthSemantics => formatter.write_str(
                "MAXWELL_B aliased line-width selection has no represented width register or host rasterization semantics",
            ),
            Self::UnsupportedAntiAliasedLineSemantics => formatter.write_str(
                "MAXWELL_B anti-aliased line rasterization has no neutral backend representation",
            ),
            Self::UnsupportedLineStippleSemantics { factor, pattern } => write!(
                formatter,
                "MAXWELL_B line stippling has no neutral backend representation: factor={factor} pattern=0x{pattern:04x}"
            ),
            Self::UnsupportedPolygonClipGeneratedEdgeSemantics => formatter.write_str(
                "MAXWELL_B suppression of polygon-clip-generated edges has no neutral backend representation",
            ),
            Self::UnsupportedVertexAttributeFormat {
                attribute,
                component_widths,
                numerical_type,
                swap_red_blue,
            } => write!(
                formatter,
                "MAXWELL_B vertex attribute has no exact neutral format: attribute={attribute} component-widths=0x{:02x} numerical-type={numerical_type:?} swap-red-blue={swap_red_blue}",
                component_widths.raw()
            ),
            Self::UnsupportedVertexInstanceDivisor { stream, divisor } => write!(
                formatter,
                "MAXWELL_B vertex stream instance divisor is not representable: stream={stream} divisor={divisor}"
            ),
            Self::InvalidPatchSize(size) => write!(
                formatter,
                "MAXWELL_B patch draw has an invalid control-point count: {}",
                size.control_points()
            ),
            Self::UnsupportedPatchSemantics(size) => write!(
                formatter,
                "MAXWELL_B patch control-point count is not represented by the neutral pipeline contract: {}",
                size.control_points()
            ),
            Self::UnsupportedPointSpriteCoordinatesSemantics(select) => write!(
                formatter,
                "MAXWELL_B generated point-sprite coordinates are not represented by shader translation or the neutral pipeline contract: texture-mask=0x{:03x} r-mode={:?} origin={:?}",
                select.generated_texture_mask(),
                select.r_mode(),
                select.origin()
            ),
            Self::UnsupportedAttributePointSizeSemantics { slot } => write!(
                formatter,
                "MAXWELL_B shader-provided point size is not represented by shader or neutral backend lowering: slot={slot}"
            ),
            Self::UnsupportedPointSpriteSemantics => formatter.write_str(
                "MAXWELL_B enabled point-sprite rasterization is not represented by the neutral backend",
            ),
            Self::UnsupportedAntiAliasedPointSemantics => formatter.write_str(
                "MAXWELL_B anti-aliased point rasterization is not represented by the neutral backend",
            ),
            Self::UnsupportedPointCenterSemantics(mode) => write!(
                formatter,
                "MAXWELL_B point-center convention is not represented by the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedFillViaTriangleSemantics(mode) => write!(
                formatter,
                "MAXWELL_B fill-via-triangle mode is not represented by the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedFillRectangleDraw(reason) => write!(
                formatter,
                "MAXWELL_B fill-rectangle draw is not representable: {reason}"
            ),
            Self::UnsupportedConservativeRasterSemantics => formatter.write_str(
                "MAXWELL_B conservative rasterization is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedPolygonSmoothSemantics => formatter.write_str(
                "MAXWELL_B polygon smoothing is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedPolygonStippleSemantics => formatter.write_str(
                "MAXWELL_B polygon stippling is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedEdgeFlagSemantics(flag) => write!(
                formatter,
                "MAXWELL_B disabled polygon edge flag is not represented by the neutral pipeline contract: flag={flag:?}"
            ),
            Self::UnsupportedShadeModeSemantics(mode) => write!(
                formatter,
                "MAXWELL_B shade mode is not representable in the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedProvokingVertexSemantics(vertex) => write!(
                formatter,
                "MAXWELL_B provoking vertex is not representable in the neutral pipeline or shader interpolation contract: vertex={vertex:?}"
            ),
            Self::UnsupportedTwoSidedLightSemantics => formatter.write_str(
                "MAXWELL_B enabled two-sided fixed-function lighting is not represented by shader or neutral backend lowering",
            ),
            Self::UnsupportedColorClampSemantics => formatter.write_str(
                "MAXWELL_B enabled color clamping is not represented by shader or neutral backend lowering",
            ),
            Self::UnsupportedPixelShaderSaturateSemantics { output, range } => write!(
                formatter,
                "MAXWELL_B pixel-shader output saturation is not represented by shader or neutral backend lowering: output={output} range={range:?}"
            ),
            Self::UnsupportedBlendSemantics { target } => match target {
                Some(target) => write!(
                    formatter,
                    "MAXWELL_B enabled blend state is not representable in the neutral pipeline contract: target={target}"
                ),
                None => formatter.write_str(
                    "MAXWELL_B enabled common blend state is not representable in the neutral pipeline contract",
                ),
            },
            Self::UnsupportedIteratedBlendSemantics { value, pass_count } => write!(
                formatter,
                "MAXWELL_B iterated blending has no neutral backend representation: color={} alpha={} pass-count={pass_count:?}",
                value.color_enabled(),
                value.alpha_enabled()
            ),
            Self::IncompleteLogicOpState => formatter.write_str(
                "MAXWELL_B enabled logic operations require SET_LOGIC_OP_FUNC",
            ),
            Self::UnsupportedLogicOpSemantics(function) => write!(
                formatter,
                "MAXWELL_B color logic operation {:?} is not represented by the neutral render pipeline",
                function
            ),
            Self::IncompleteColorWriteState {
                target,
                mask_register,
            } => write!(
                formatter,
                "MAXWELL_B color target {target} selects unprogrammed SET_CT_WRITE({mask_register})",
            ),
            Self::UnsupportedColorWriteMask {
                target,
                mask_register,
                mask,
            } => write!(
                formatter,
                "MAXWELL_B partial color writes are not represented by the neutral render pipeline: target={target} mask-register={mask_register} mask=0x{:04x}",
                mask.raw()
            ),
            Self::IncompleteAlphaTestState(field) => write!(
                formatter,
                "MAXWELL_B enabled alpha testing requires SET_ALPHA_{field}"
            ),
            Self::CompressedDepthImportRequired { kind } => write!(
                formatter,
                "Maxwell compressed depth contents require materialization before use: kind=0x{kind:02x}"
            ),
            Self::CompressedColorImportRequired { target } => write!(
                formatter,
                "Maxwell compressed color contents require materialization before use: target={target}"
            ),
            Self::ShaderTranslationRequired => {
                formatter.write_str("Maxwell shader translation is required before draw lowering")
            }
            Self::InvalidTranslatedShaders => {
                formatter.write_str("translated shader evidence is empty, duplicated, or invalid")
            }
            Self::TranslatedShaderStageMismatch => formatter
                .write_str("translated shader stages do not match enabled Maxwell pipeline stages"),
            Self::TranslatedShaderMemoryConfigurationMismatch {
                stage,
                configured,
                required,
            } => write!(
                formatter,
                "translated shader requires a different Maxwell directly addressable memory configuration: stage={stage:?} configured-bytes={} translated-for-bytes={}",
                configured.bytes(),
                required.bytes()
            ),
            Self::UnsupportedShaderStage(stage) => write!(
                formatter,
                "Maxwell shader stage has no neutral lowering: {stage:?}"
            ),
            Self::InvalidShaderResourceUse { role } => write!(
                formatter,
                "translated shader declares an invalid resource use: role={role:?}"
            ),
            Self::MissingResolvedResource { role } => write!(
                formatter,
                "complete 3D snapshot lacks resolved resource: role={role:?}"
            ),
            Self::ResolvedResourceKindMismatch => {
                formatter.write_str("resolved 3D resource kind contradicts its role")
            }
            Self::InvalidResolvedView { role } => write!(
                formatter,
                "resolved 3D view cannot be re-identified neutrally: role={role:?}"
            ),
            Self::AllocationDescriptionChanged { allocation } => write!(
                formatter,
                "cached GPU allocation changed immutable description: {allocation}"
            ),
            Self::IncompleteClear(field) => {
                write!(formatter, "clear state is incomplete: missing={field}")
            }
            Self::EmptyClearMask => {
                formatter.write_str("CLEAR_SURFACE selects no color, depth, or stencil component")
            }
            Self::EmptyClearRectangle => formatter.write_str(
                "clear rectangle, scissor, and viewport-clip intersection is empty",
            ),
            Self::PartialColorClearUnsupported { mask } => write!(
                formatter,
                "partial color-channel clear is not represented yet: mask={mask:#x}"
            ),
            Self::ClearOutsideAttachment => {
                formatter.write_str("clear rectangle or layer lies outside the resolved attachment")
            }
            Self::IncompleteDraw(field) => {
                write!(formatter, "draw state is incomplete: missing={field}")
            }
            Self::IncompleteBlendState { target, field } => match target {
                Some(target) => write!(
                    formatter,
                    "blend state is incomplete: target={target} missing={field}"
                ),
                None => write!(formatter, "common blend state is incomplete: missing={field}"),
            },
            Self::ColorTargetRouteUnprogrammed { slot, target } => write!(
                formatter,
                "SET_CT_SELECT routes output slot {slot} to unprogrammed color target {target}"
            ),
            Self::ColorTargetRouteDisabled { slot, target } => write!(
                formatter,
                "SET_CT_SELECT routes output slot {slot} to disabled color target {target}"
            ),
            Self::ColorTargetRouteIncomplete { slot, target } => write!(
                formatter,
                "SET_CT_SELECT routes output slot {slot} to incomplete color target {target}"
            ),
            Self::DuplicateColorTargetRoute { target } => write!(
                formatter,
                "SET_CT_SELECT routes one color target more than once: target={target}"
            ),
            Self::EmptyDraw => formatter.write_str("draw vertex count is zero"),
            Self::UnsupportedPrimitiveIdContinuation => formatter
                .write_str("BEGIN requests unsupported primitive-ID continuation semantics"),
            Self::UnsupportedPrimitiveSplitMode(mode) => write!(
                formatter,
                "BEGIN requests unsupported split-primitive semantics: mode={mode}"
            ),
            Self::InstanceIndexOverflow { base, relative } => write!(
                formatter,
                "Maxwell instance index exceeds the neutral u32 domain: base={base} relative={relative}"
            ),
            Self::UnsupportedTopology(topology) => write!(
                formatter,
                "primitive topology has no neutral lowering: topology={topology:#x}"
            ),
            Self::AliasedDrawResources { first, second } => write!(
                formatter,
                "draw has a read/write or attachment alias without modeled feedback semantics: first={first:?} second={second:?}"
            ),
            Self::InvalidTransition => {
                formatter.write_str("derived neutral resource transition is invalid")
            }
            Self::InvalidResourceCreation => {
                formatter.write_str("derived neutral resource creation is invalid")
            }
            Self::Command(error) => {
                write!(formatter, "neutral command construction failed: {error}")
            }
            Self::Capability(error) => write!(
                formatter,
                "backend capabilities cannot represent complete 3D operation: {error}"
            ),
            Self::ResourceExhausted => {
                formatter.write_str("3D lowering exhausted host resources or identities")
            }
        }
    }
}

impl std::error::Error for MaxwellThreeDLoweringError {}

#[cfg(test)]
mod tests {
    use nixe_gpu::{
        DepthCompareOperation, GpuCacheConfiguration, PrimitiveTopology, ResourceDependency,
        ShaderId, ShaderInstruction, ShaderIr, ShaderOperation, ShaderPredicate,
        ShaderSourceLocation, ShaderStage, VerifiedShaderIr, VertexComponentCount,
        VertexComponentWidth, VertexFormat,
    };

    use crate::{MaxwellThreeDBegin, MaxwellThreeDCompareOp, MaxwellThreeDVertexAttributeFormat};

    use super::{
        FingerprintCache, MaxwellThreeDLoweringCache, MaxwellThreeDLoweringError,
        ShaderTranslationRecord, depth_stencil_attachment_required, neutral_depth_compare,
        neutral_first_instance, neutral_vertex_format, primitive_topology,
    };

    #[test]
    fn fingerprint_index_tracks_hits_for_lru_eviction() {
        let mut cache = FingerprintCache::default();
        cache.push(11, "first");
        cache.push(22, "second");

        assert_eq!(cache.get(11), Some(&"first"));
        assert_eq!(cache.remove_lru(), (22, "second"));
        assert_eq!(cache.get(22), None);
        assert_eq!(cache.get(11), Some(&"first"));
    }

    #[test]
    fn published_shader_translation_storage_is_bounded_and_retires_evictions() {
        let verified = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![ShaderInstruction::new(
                ShaderSourceLocation::new(0),
                ShaderPredicate::Always,
                ShaderOperation::Exit,
            )],
        ))
        .unwrap();
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&verified).unwrap();
        let configuration = GpuCacheConfiguration::new(6, 1, 1, 1, 1).unwrap();
        let mut cache = MaxwellThreeDLoweringCache::new(configuration);
        for raw in 1..=7 {
            cache.shader_translations.push(
                raw as u128,
                ShaderTranslationRecord {
                    #[cfg(debug_assertions)]
                    key: None,
                    id: ShaderId::new(raw as u64),
                    module: module.clone(),
                    published: true,
                },
            );
            cache.enforce_shader_translation_limit();
        }

        assert_eq!(cache.shader_translations.len(), 6);
        assert_eq!(
            cache.retired_resources.as_slice(),
            [ResourceDependency::Shader(ShaderId::new(1))]
        );
    }

    #[test]
    fn every_maxwell_depth_comparison_has_an_exact_neutral_mapping() {
        let cases = [
            (MaxwellThreeDCompareOp::Never, DepthCompareOperation::Never),
            (MaxwellThreeDCompareOp::Less, DepthCompareOperation::Less),
            (MaxwellThreeDCompareOp::Equal, DepthCompareOperation::Equal),
            (
                MaxwellThreeDCompareOp::LessEqual,
                DepthCompareOperation::LessEqual,
            ),
            (
                MaxwellThreeDCompareOp::Greater,
                DepthCompareOperation::Greater,
            ),
            (
                MaxwellThreeDCompareOp::NotEqual,
                DepthCompareOperation::NotEqual,
            ),
            (
                MaxwellThreeDCompareOp::GreaterEqual,
                DepthCompareOperation::GreaterEqual,
            ),
            (
                MaxwellThreeDCompareOp::Always,
                DepthCompareOperation::Always,
            ),
        ];
        for (maxwell, neutral) in cases {
            assert_eq!(neutral_depth_compare(maxwell), neutral);
        }
    }

    #[test]
    fn depth_stencil_attachment_is_omitted_only_when_both_tests_are_explicitly_disabled() {
        assert!(!depth_stencil_attachment_required(Some(false), Some(false)));

        for state in [
            (Some(true), Some(false)),
            (Some(false), Some(true)),
            (Some(true), Some(true)),
            (None, Some(false)),
            (Some(false), None),
            (None, None),
        ] {
            assert!(depth_stencil_attachment_required(state.0, state.1));
        }
    }

    #[test]
    fn simple_triangle_float3_attributes_lower_to_exact_neutral_formats() {
        let position = MaxwellThreeDVertexAttributeFormat::parse(0x3840_0000).unwrap();
        let color = MaxwellThreeDVertexAttributeFormat::parse(0x3840_0600).unwrap();

        assert_eq!(
            neutral_vertex_format(0, position),
            Ok(VertexFormat::Float32x3)
        );
        assert_eq!(neutral_vertex_format(1, color), Ok(VertexFormat::Float32x3));
        assert_eq!(position.offset(), 0);
        assert_eq!(color.offset(), 12);
    }

    #[test]
    fn scaled_vertex_family_preserves_width_count_and_signedness() {
        for (width_raw, width, components) in [
            (0x1d, VertexComponentWidth::Bits8, VertexComponentCount::One),
            (0x18, VertexComponentWidth::Bits8, VertexComponentCount::Two),
            (
                0x13,
                VertexComponentWidth::Bits8,
                VertexComponentCount::Three,
            ),
            (
                0x0a,
                VertexComponentWidth::Bits8,
                VertexComponentCount::Four,
            ),
            (
                0x1b,
                VertexComponentWidth::Bits16,
                VertexComponentCount::One,
            ),
            (
                0x0f,
                VertexComponentWidth::Bits16,
                VertexComponentCount::Two,
            ),
            (
                0x05,
                VertexComponentWidth::Bits16,
                VertexComponentCount::Three,
            ),
            (
                0x03,
                VertexComponentWidth::Bits16,
                VertexComponentCount::Four,
            ),
            (
                0x12,
                VertexComponentWidth::Bits32,
                VertexComponentCount::One,
            ),
            (
                0x04,
                VertexComponentWidth::Bits32,
                VertexComponentCount::Two,
            ),
            (
                0x02,
                VertexComponentWidth::Bits32,
                VertexComponentCount::Three,
            ),
            (
                0x01,
                VertexComponentWidth::Bits32,
                VertexComponentCount::Four,
            ),
        ] {
            for (type_raw, expected) in [
                (5, VertexFormat::Uscaled { width, components }),
                (6, VertexFormat::Sscaled { width, components }),
            ] {
                let format =
                    MaxwellThreeDVertexAttributeFormat::parse((width_raw << 21) | (type_raw << 27))
                        .unwrap();
                assert_eq!(neutral_vertex_format(0, format), Ok(expected));
            }
        }
    }

    #[test]
    fn instance_begin_modes_do_not_change_primitive_topology() {
        for instance in [0, 1, 2] {
            let begin = MaxwellThreeDBegin::parse(4 | (instance << 26)).unwrap();
            assert_eq!(primitive_topology(begin), Ok(PrimitiveTopology::Triangles));
        }
    }

    #[test]
    fn primitive_continuation_modes_remain_precise_fatal_boundaries() {
        let primitive_id = MaxwellThreeDBegin::parse(4 | (1 << 24)).unwrap();
        assert_eq!(
            primitive_topology(primitive_id),
            Err(MaxwellThreeDLoweringError::UnsupportedPrimitiveIdContinuation)
        );

        for split_mode in 1..=3 {
            let split = MaxwellThreeDBegin::parse(4 | (split_mode << 29)).unwrap();
            assert_eq!(
                primitive_topology(split),
                Err(MaxwellThreeDLoweringError::UnsupportedPrimitiveSplitMode(
                    split_mode as u8
                ))
            );
        }
    }

    #[test]
    fn relative_instance_is_added_to_base_instance_without_loss() {
        assert_eq!(neutral_first_instance(7, 2), Ok(9));
        assert_eq!(
            neutral_first_instance(u32::MAX, 1),
            Err(MaxwellThreeDLoweringError::InstanceIndexOverflow {
                base: u32::MAX,
                relative: 1,
            })
        );
    }
}
