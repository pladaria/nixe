//! Atomic lowering of validated `MAXWELL_B` clear and draw snapshots.
//!
//! This boundary produces only backend-independent `nixe-gpu` resources and
//! operations. Shader translation is supplied as typed T10 evidence; this
//! module never treats Maxwell code as a neutral or host shader.

use std::fmt::{Display, Formatter};

use nixe_gpu::{
    AccessMode, AccessScope, AccessTarget, AttachmentLoad, AttachmentStore, BackendCapabilities,
    BackendCapabilityError, BackendResourceCreateInfo, BarrierOperation, BufferId, BufferRange,
    BufferRegion, BufferView, CapabilityAgreement, CapabilityRequirements, ClearOperation,
    ClearValue, CommandDescriptionError, DescriptorKind, DescriptorTableDescription,
    DescriptorTableId, DrawArguments, DrawOperation, FrontendSubmissionId, GpuCommand,
    GpuOperation, ImageId, ImageOrigin, ImageRegion, ImageSubresourceRange, ImageView,
    OperationSubmission, PipelineDescription, PipelineId, PipelineKind, PipelineStages,
    PrimitiveTopology, RenderAttachment, RenderPassDescription, RenderPassId, RenderPassOperation,
    ResourceAccess, ResourceDependency, ResourceTransition, ResourceUsage, ShaderDescription,
    ShaderId, ShaderStage, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
    ViewportTransform,
};

use crate::MaxwellMethodSource;
use crate::shader::{MaxwellShaderTranslationKey, MaxwellTranslatedShaderProgram};

use super::{
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAntiAliasedLineEnable, MaxwellThreeDBegin,
    MaxwellThreeDBlendEnableCommon, MaxwellThreeDClipIdTestEnable,
    MaxwellThreeDColorCompressionMode, MaxwellThreeDColorReductionThresholdsEnable,
    MaxwellThreeDConditionalLoadConstantBuffer, MaxwellThreeDConservativeRasterEnable,
    MaxwellThreeDCsaaEnable, MaxwellThreeDDirectlyAddressableMemory, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFillViaTriangleMode, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDHybridAntiAliasControl, MaxwellThreeDLogicOp,
    MaxwellThreeDPatchSize, MaxwellThreeDPixelShaderClampRange, MaxwellThreeDPointCenterMode,
    MaxwellThreeDPointSpriteSelect, MaxwellThreeDPolygonMode, MaxwellThreeDProvokingVertex,
    MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderTargetIndexOffset,
    MaxwellThreeDRenderTargetLayer, MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources,
    MaxwellThreeDResourceRole, MaxwellThreeDSampleLocationGroup, MaxwellThreeDSeparateFragmentData,
    MaxwellThreeDShadeMode, MaxwellThreeDShaderLocalMemoryPerWarpSize, MaxwellThreeDShaderStage,
    MaxwellThreeDState, MaxwellThreeDVertexNumericalType, MaxwellThreeDViewportCoordinateSwizzle,
    MaxwellThreeDViewportPixelCenter, MaxwellThreeDViewportScaleOffsetEnable,
};

#[derive(Clone, Debug)]
struct DrawAttachmentSelection {
    colors: Vec<(u8, usize)>,
    depth_stencil: Option<usize>,
}

impl DrawAttachmentSelection {
    fn attachment_indices(&self) -> Vec<usize> {
        self.colors
            .iter()
            .map(|(_, index)| *index)
            .chain(self.depth_stencil)
            .collect()
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
}

/// Stable evidence that T10 translated one enabled Maxwell shader stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTranslatedShader {
    stage: ShaderStage,
    shader: ShaderId,
    directly_addressable_memory: MaxwellThreeDDirectlyAddressableMemory,
    maximum_api_visible_calls: u16,
}

impl MaxwellThreeDTranslatedShader {
    #[must_use]
    pub(crate) const fn new(
        stage: ShaderStage,
        shader: ShaderId,
        directly_addressable_memory: MaxwellThreeDDirectlyAddressableMemory,
        maximum_api_visible_calls: u16,
    ) -> Self {
        Self {
            stage,
            shader,
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
    /// Guest shader-memory configuration for which T10 produced this shader.
    /// This is never inferred from host cache topology.
    #[must_use]
    pub const fn directly_addressable_memory(self) -> MaxwellThreeDDirectlyAddressableMemory {
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDShaderResourceUse {
    role: MaxwellThreeDResourceRole,
    stages: PipelineStages,
    usage: ResourceUsage,
}

impl MaxwellThreeDShaderResourceUse {
    pub fn new(
        role: MaxwellThreeDResourceRole,
        stages: PipelineStages,
        usage: ResourceUsage,
    ) -> Result<Self, MaxwellThreeDLoweringError> {
        let _ = AccessScope::new(stages, AccessMode::Read, usage)
            .map_err(|_| MaxwellThreeDLoweringError::InvalidShaderResourceUse { role })?;
        Ok(Self {
            role,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTranslatedShaders {
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ViewKey {
    Buffer {
        description: nixe_gpu::BufferDescription,
        buffer_offset: u64,
        backing: nixe_gpu::BackingView,
        mappings: Box<[super::MaxwellThreeDMappingReference]>,
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
        mappings: Box<[super::MaxwellThreeDMappingReference]>,
    },
}

impl ViewKey {
    fn overlaps(&self, other: &Self) -> bool {
        self.backings().iter().any(|left| {
            other
                .backings()
                .iter()
                .any(|right| canonical_backings_overlap(left, right))
        })
    }

    fn backings(&self) -> Vec<&nixe_gpu::BackingView> {
        match self {
            Self::Buffer { backing, .. } => vec![backing],
            Self::Image { bindings, .. } => {
                bindings.iter().map(|(_, _, backing)| backing).collect()
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ViewRecord {
    key: ViewKey,
    dependency: ResourceDependency,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PipelineAttachmentKey(
    MaxwellThreeDResourceRole,
    nixe_gpu::ImageDescription,
    ImageId,
);

#[derive(Clone, Debug, Eq, PartialEq)]
struct PipelineKey {
    method_dependencies: Box<[Option<u32>]>,
    attachments: Box<[PipelineAttachmentKey]>,
    resource_dependencies: Box<[(MaxwellThreeDResourceRole, ResourceDependency)]>,
    shaders: MaxwellThreeDTranslatedShaders,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PipelineRecord {
    key: PipelineKey,
    id: PipelineId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RenderPassRecord {
    description: RenderPassDescription,
    id: RenderPassId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DescriptorRecord {
    roles: Box<[MaxwellThreeDResourceRole]>,
    dependencies: Box<[ResourceDependency]>,
    id: DescriptorTableId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShaderTranslationRecord {
    key: Option<MaxwellShaderTranslationKey>,
    id: ShaderId,
    module: nixe_gpu::ShaderBackendModule,
    published: bool,
}

/// Frontend-owned derived identity cache. It contains no backend handles and
/// changes only through [`MaxwellThreeDLoweringPlan::commit_cache`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDLoweringCache {
    revision: u64,
    next_identity: u64,
    allocations: Vec<(
        nixe_gpu::GpuAllocationId,
        nixe_gpu::GpuAllocationDescription,
    )>,
    views: Vec<ViewRecord>,
    pipelines: Vec<PipelineRecord>,
    render_passes: Vec<RenderPassRecord>,
    descriptors: Vec<DescriptorRecord>,
    shader_translations: Vec<ShaderTranslationRecord>,
    retired_shader_resources: Vec<ResourceDependency>,
    accesses: Vec<(AccessTarget, AccessScope)>,
}

impl Default for MaxwellThreeDLoweringCache {
    fn default() -> Self {
        Self {
            revision: 0,
            next_identity: 1,
            allocations: Vec::new(),
            views: Vec::new(),
            pipelines: Vec::new(),
            render_passes: Vec::new(),
            descriptors: Vec::new(),
            shader_translations: Vec::new(),
            retired_shader_resources: Vec::new(),
            accesses: Vec::new(),
        }
    }
}

impl MaxwellThreeDLoweringCache {
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    #[must_use]
    pub fn view_count(&self) -> usize {
        self.views.len()
    }
    #[must_use]
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    #[cfg(test)]
    pub(crate) fn shader_translation_count(&self) -> usize {
        self.shader_translations.len()
    }

    /// Resolves immutable T10 products to stable logical shader identities.
    pub(crate) fn stage_shader_translations(
        &mut self,
        programs: &[MaxwellTranslatedShaderProgram],
        directly_addressable_memory: MaxwellThreeDDirectlyAddressableMemory,
    ) -> Result<MaxwellThreeDTranslatedShaders, MaxwellThreeDLoweringError> {
        let mut shaders = Vec::with_capacity(programs.len());
        for program in programs {
            let id = if let Some(record) = self
                .shader_translations
                .iter()
                .find(|record| record.key.as_ref() == Some(program.key()))
            {
                record.id
            } else {
                let mut retired_ids = Vec::new();
                self.shader_translations.retain(|record| {
                    let replaced = record
                        .key
                        .as_ref()
                        .is_some_and(|key| key.same_program_binding(program.key()));
                    if replaced && record.published {
                        retired_ids.push(record.id);
                    }
                    !replaced
                });
                if !retired_ids.is_empty() {
                    let mut retired_pipelines = Vec::new();
                    self.pipelines.retain(|record| {
                        let replaced = record
                            .key
                            .shaders
                            .shaders()
                            .iter()
                            .any(|shader| retired_ids.contains(&shader.shader()));
                        if replaced {
                            retired_pipelines.push(record.id);
                        }
                        !replaced
                    });
                    self.retired_shader_resources.extend(
                        retired_pipelines
                            .into_iter()
                            .map(ResourceDependency::Pipeline),
                    );
                    self.retired_shader_resources
                        .extend(retired_ids.into_iter().map(ResourceDependency::Shader));
                }
                let id = ShaderId::new(take_identity(self)?);
                self.shader_translations.push(ShaderTranslationRecord {
                    key: Some(program.key().clone()),
                    id,
                    module: program.module().clone(),
                    published: false,
                });
                id
            };
            shaders.push(MaxwellThreeDTranslatedShader::new(
                program.stage(),
                id,
                directly_addressable_memory,
                program.maximum_api_visible_calls(),
            ));
        }
        MaxwellThreeDTranslatedShaders::new(shaders, Vec::new())
    }

    #[cfg(test)]
    pub(crate) fn seed_test_shader_translations(
        &mut self,
        shaders: &MaxwellThreeDTranslatedShaders,
    ) {
        for shader in shaders.shaders() {
            if self
                .shader_translations
                .iter()
                .any(|record| record.id == shader.shader())
            {
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
            self.shader_translations.push(ShaderTranslationRecord {
                key: None,
                id: shader.shader(),
                module,
                published: false,
            });
        }
    }
}

/// Immutable preflight result. Resource creation, backend submission, and
/// cache publication remain separate caller-controlled phases.
pub struct MaxwellThreeDUnnegotiatedLoweringPlan {
    expected_revision: u64,
    candidate: MaxwellThreeDLoweringCache,
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
    dirty_images: Box<[usize]>,
}

impl MaxwellThreeDUnnegotiatedLoweringPlan {
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

    /// Negotiates the already complete neutral description with one real
    /// backend capability set, without changing frontend cache state.
    pub fn negotiate(
        self,
        capabilities: &BackendCapabilities,
    ) -> Result<MaxwellThreeDLoweringPlan, MaxwellThreeDLoweringError> {
        for creation in &self.creations {
            let requirements = creation
                .capability_requirements()
                .map_err(|_| MaxwellThreeDLoweringError::InvalidResourceCreation)?;
            capabilities
                .negotiate(&requirements)
                .map_err(MaxwellThreeDLoweringError::Capability)?;
        }
        let agreement = capabilities
            .negotiate_all(&self.submission.capability_requirements())
            .map_err(MaxwellThreeDLoweringError::Capability)?;
        Ok(MaxwellThreeDLoweringPlan {
            expected_revision: self.expected_revision,
            candidate: self.candidate,
            creations: self.creations,
            invalidations: self.invalidations,
            submission: self.submission,
            agreement,
            dirty_images: self.dirty_images,
        })
    }

    pub(crate) fn stage_cache(
        self,
        cache: &mut MaxwellThreeDLoweringCache,
    ) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
        commit_lowering_cache(
            self.expected_revision,
            self.candidate,
            self.creations,
            self.invalidations,
            self.submission,
            self.dirty_images,
            cache,
        )
    }
}

/// Immutable preflight result negotiated with one backend capability set.
pub struct MaxwellThreeDLoweringPlan {
    expected_revision: u64,
    candidate: MaxwellThreeDLoweringCache,
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
    agreement: CapabilityAgreement,
    dirty_images: Box<[usize]>,
}

impl MaxwellThreeDLoweringPlan {
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
    pub const fn capability_agreement(&self) -> CapabilityAgreement {
        self.agreement
    }
    #[must_use]
    pub fn dirty_images(&self) -> &[usize] {
        &self.dirty_images
    }

    /// Publishes derived identities only after the caller has successfully
    /// completed its resource-ownership and backend-submission phases.
    pub fn commit_cache(
        self,
        cache: &mut MaxwellThreeDLoweringCache,
    ) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
        commit_lowering_cache(
            self.expected_revision,
            self.candidate,
            self.creations,
            self.invalidations,
            self.submission,
            self.dirty_images,
            cache,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn commit_lowering_cache(
    expected_revision: u64,
    mut candidate: MaxwellThreeDLoweringCache,
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
    dirty_images: Box<[usize]>,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
    if cache.revision != expected_revision {
        return Err(MaxwellThreeDLoweringError::CacheChanged {
            expected: expected_revision,
            actual: cache.revision,
        });
    }
    candidate.revision = candidate
        .revision
        .checked_add(1)
        .ok_or(MaxwellThreeDLoweringError::ResourceExhausted)?;
    *cache = candidate;
    Ok(MaxwellThreeDLoweredWork {
        creations,
        invalidations,
        submission,
        dirty_images,
    })
}

/// Committed frontend record retained independently from backend handles.
pub struct MaxwellThreeDLoweredWork {
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
    dirty_images: Box<[usize]>,
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

/// Preflights one exact trigger without changing resolved resources, cache, or
/// any concrete backend.
#[allow(clippy::too_many_arguments)]
pub fn preflight_maxwell_three_d_operation(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    translated_shaders: Option<&MaxwellThreeDTranslatedShaders>,
    submission: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    capabilities: &BackendCapabilities,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<MaxwellThreeDLoweringPlan, MaxwellThreeDLoweringError> {
    preflight_maxwell_three_d_operation_unnegotiated(
        state,
        resources,
        trigger,
        translated_shaders,
        submission,
        predecessors,
        cache,
    )?
    .negotiate(capabilities)
}

/// Produces a complete neutral operation description before selecting or
/// claiming support from a concrete backend.
#[allow(clippy::too_many_arguments)]
pub fn preflight_maxwell_three_d_operation_unnegotiated(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    translated_shaders: Option<&MaxwellThreeDTranslatedShaders>,
    submission: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<MaxwellThreeDUnnegotiatedLoweringPlan, MaxwellThreeDLoweringError> {
    state.validate_cross_registers().map_err(|error| {
        MaxwellThreeDLoweringError::ContradictoryState {
            reason: error.reason,
        }
    })?;
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
    ) && state.coverage().csaa_enable().value() == Some(&MaxwellThreeDCsaaEnable::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedCsaaSemantics);
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
            .filter(|mode| *mode != MaxwellThreeDFillViaTriangleMode::Disabled)
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
            .vertex_array_restart_enabled()
            .value()
            == Some(&super::MaxwellThreeDVertexArrayPrimitiveRestartEnable::Enabled)
        {
            let vertex_count = match trigger {
                MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. } => vertex_count,
                MaxwellThreeDOperationTrigger::ClearSurface { .. } => unreachable!(),
            };
            let topology = state
                .vertex_input()
                .primitive()
                .active_begin()
                .copied()
                .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("BEGIN"))?
                .topology();
            if !vertex_array_restart_is_neutral(topology, vertex_count) {
                return Err(
                    MaxwellThreeDLoweringError::UnsupportedVertexArrayPrimitiveRestartSemantics {
                        topology,
                        vertex_count,
                    },
                );
            }
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
                .filter(|value| value.affects_draw_layering())
        {
            return Err(MaxwellThreeDLoweringError::UnsupportedRenderTargetLayerSemantics(value));
        }
        validate_draw_blending_state(state, attachments)?;
        validate_draw_logic_op_state(state, attachments)?;
        validate_draw_color_write_state(state, attachments)?;
        validate_draw_alpha_test_state(state)?;
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
    let mut candidate = cache.clone();
    let mut creations = Vec::new();
    let mut invalidations = std::mem::take(&mut candidate.retired_shader_resources);
    let resource_bindings = prepare_resources(
        resources,
        &resource_indices,
        &mut candidate,
        &mut creations,
        &mut invalidations,
    )?;
    let (commands, dirty_images) = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { source: _ } => {
            lower_clear(state, resources, &resource_bindings)?
        }
        MaxwellThreeDOperationTrigger::DrawVertexArray {
            source: _,
            vertex_count,
        } => {
            let attachments = draw_attachments
                .as_ref()
                .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
            lower_draw(
                state,
                resources,
                &resource_bindings,
                shaders.ok_or(MaxwellThreeDLoweringError::ShaderTranslationRequired)?,
                attachments,
                vertex_count,
                &mut candidate,
                &mut creations,
            )?
        }
    };
    let operations = sequence_with_transitions(commands, &mut candidate)?;
    let submission = OperationSubmission::new(submission, predecessors, operations)
        .map_err(MaxwellThreeDLoweringError::Command)?;
    Ok(MaxwellThreeDUnnegotiatedLoweringPlan {
        expected_revision: cache.revision,
        candidate,
        creations: creations.into_boxed_slice(),
        invalidations: invalidations.into_boxed_slice(),
        submission,
        dirty_images: dirty_images.into_boxed_slice(),
    })
}

fn validate_shader_memory_configuration(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<(), MaxwellThreeDLoweringError> {
    let configured = state
        .shader_execution()
        .l1_configuration()
        .value()
        .copied()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_L1_CONFIGURATION",
        ))?;
    for shader in shaders.shaders() {
        let required = shader.directly_addressable_memory();
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
    let active_targets = attachments.color_targets().collect::<Vec<_>>();
    if active_targets.is_empty() {
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
    for target in active_targets {
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

fn validate_draw_alpha_test_state(
    state: &MaxwellThreeDState,
) -> Result<(), MaxwellThreeDLoweringError> {
    let fixed = state.fixed_function();
    if fixed
        .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable)
        .value()
        != Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Ok(());
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
    Err(MaxwellThreeDLoweringError::UnsupportedAlphaTestSemantics {
        function,
        reference,
    })
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
    let consumes_depth = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => state
            .render_targets()
            .clear()
            .last_surface()
            .value()
            .is_some_and(|surface| surface.depth() || surface.stencil()),
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => {
            draw_attachments.is_some_and(|attachments| attachments.depth_stencil == Some(index))
        }
    };
    if !consumes_depth {
        return Ok(());
    }
    let key = view_key(&resources.resources()[index]);
    if cache.views.iter().any(|record| record.key == key) {
        return Ok(());
    }
    if matches!(trigger, MaxwellThreeDOperationTrigger::ClearSurface { .. })
        && depth_clear_fully_initializes(state, image)?
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
) -> Result<bool, MaxwellThreeDLoweringError> {
    let clear = state.render_targets().clear();
    let surface = clear
        .last_surface()
        .value()
        .ok_or(MaxwellThreeDLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
    if !(surface.depth() && surface.stencil()) {
        return Ok(false);
    }
    if clear
        .surface_control()
        .value()
        .is_some_and(|control| control.respect_stencil_mask())
    {
        return Ok(false);
    }
    clear_fully_covers_image(state, image)
}

fn clear_fully_covers_image(
    state: &MaxwellThreeDState,
    image: &super::MaxwellThreeDResolvedImage,
) -> Result<bool, MaxwellThreeDLoweringError> {
    let clear = state.render_targets().clear();
    let control = clear.surface_control().value().copied();
    if control.is_some_and(|control| control.use_scissor_zero() || control.use_viewport_clip_zero())
    {
        return Ok(false);
    }
    if control.is_some_and(|control| !control.use_clear_rect()) {
        return Ok(true);
    }
    let Some(horizontal) = clear.horizontal().value() else {
        return Ok(false);
    };
    let Some(vertical) = clear.vertical().value() else {
        return Ok(false);
    };
    Ok(horizontal.min == 0
        && vertical.min == 0
        && u32::from(horizontal.max) == image.description().extent().width
        && u32::from(vertical.max) == image.description().extent().height)
}

fn validate_compressed_color_materialization(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<(), MaxwellThreeDLoweringError> {
    let consumed_target = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => state
            .render_targets()
            .clear()
            .last_surface()
            .value()
            .filter(|surface| surface.color_mask() != 0)
            .map(|surface| surface.color_target()),
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => draw_attachments
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("SET_CT_SELECT"))?
            .color_targets()
            .find(|target| {
                state.render_targets().color()[*target as usize]
                    .compression()
                    .value()
                    == Some(&MaxwellThreeDColorCompressionMode::Enabled)
            }),
    };
    let Some(target) = consumed_target.filter(|target| {
        state.render_targets().color()[*target as usize]
            .compression()
            .value()
            == Some(&MaxwellThreeDColorCompressionMode::Enabled)
    }) else {
        return Ok(());
    };
    let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
    let image = resolved_image(resources, index)?;
    let key = view_key(&resources.resources()[index]);
    if cache.views.iter().any(|record| record.key == key) {
        return Ok(());
    }
    let full_clear = matches!(trigger, MaxwellThreeDOperationTrigger::ClearSurface { .. })
        && state
            .render_targets()
            .clear()
            .last_surface()
            .value()
            .is_some_and(|surface| surface.color_mask() == 0xf)
        && clear_fully_covers_image(state, image)?;
    if !full_clear {
        return Err(MaxwellThreeDLoweringError::CompressedColorImportRequired { target });
    }
    Ok(())
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
        indices.push(resource_index(resources, resource.role())?);
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
    depth_stencil_attachment_required(
        boolean(MaxwellThreeDFixedFunctionRegister::DepthTestEnable),
        boolean(MaxwellThreeDFixedFunctionRegister::StencilTestEnable),
    )
}

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
    ViewportTransform::new([scale_x, scale_y, scale_z], [offset_x, offset_y, offset_z])
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

        let key = view_key(resource);
        if let Some(record) = cache.views.iter().find(|record| record.key == key) {
            result[*index] = Some(record.dependency);
            continue;
        }
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
        let invalidated_pipelines = cache
            .pipelines
            .iter()
            .filter(|record| {
                record.key.attachments.iter().any(|attachment| {
                    invalidated.contains(&ResourceDependency::Image(attachment.2))
                }) || record
                    .key
                    .resource_dependencies
                    .iter()
                    .any(|(_, dependency)| invalidated.contains(dependency))
            })
            .map(|record| ResourceDependency::Pipeline(record.id))
            .collect::<Vec<_>>();
        cache.pipelines.retain(|record| {
            !invalidated_pipelines.contains(&ResourceDependency::Pipeline(record.id))
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
        for dependency in invalidated_pipelines {
            if !invalidations.contains(&dependency) {
                invalidations.push(dependency);
            }
        }
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
        cache.views.push(ViewRecord { key, dependency });
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

fn clear_image_region(
    image: ImageId,
    subresources: ImageSubresourceRange,
    attachment_width: u32,
    attachment_height: u32,
    array_layer: u16,
    rectangle: Option<(super::MaxwellThreeDRectangle, super::MaxwellThreeDRectangle)>,
) -> Result<ImageRegion, MaxwellThreeDLoweringError> {
    if array_layer != subresources.base_layer {
        return Err(MaxwellThreeDLoweringError::ClearOutsideAttachment);
    }
    let (origin, width, height) = match rectangle {
        Some((horizontal, vertical)) => {
            let width = u32::from(horizontal.max.saturating_sub(horizontal.min));
            let height = u32::from(vertical.max.saturating_sub(vertical.min));
            if width == 0 || height == 0 {
                return Err(MaxwellThreeDLoweringError::EmptyClearRectangle);
            }
            if u32::from(horizontal.max) > attachment_width
                || u32::from(vertical.max) > attachment_height
            {
                return Err(MaxwellThreeDLoweringError::ClearOutsideAttachment);
            }
            (
                ImageOrigin {
                    x: u32::from(horizontal.min),
                    y: u32::from(vertical.min),
                    z: 0,
                },
                width,
                height,
            )
        }
        None => (
            ImageOrigin { x: 0, y: 0, z: 0 },
            attachment_width,
            attachment_height,
        ),
    };
    Ok(ImageRegion {
        image,
        subresources,
        origin,
        extent: nixe_gpu::ImageExtent {
            width,
            height,
            depth: 1,
        },
    })
}

fn lower_clear(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
) -> Result<(Vec<GpuOperation>, Vec<usize>), MaxwellThreeDLoweringError> {
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
    if control.is_some_and(|control| control.use_scissor_zero()) {
        return Err(MaxwellThreeDLoweringError::UnsupportedClearScissorSemantics);
    }
    if control.is_some_and(|control| control.use_viewport_clip_zero()) {
        return Err(MaxwellThreeDLoweringError::UnsupportedClearViewportClipSemantics);
    }
    if surface.stencil() && control.is_some_and(|control| control.respect_stencil_mask()) {
        return Err(MaxwellThreeDLoweringError::UnsupportedClearStencilMaskSemantics);
    }
    // An unprogrammed control retains the pre-existing explicit-rectangle
    // contract without claiming an undocumented hardware reset value.
    let rectangle = if control.is_none_or(|control| control.use_clear_rect()) {
        Some((
            clear.horizontal().value().copied().ok_or(
                MaxwellThreeDLoweringError::IncompleteClear("horizontal rectangle"),
            )?,
            clear.vertical().value().copied().ok_or(
                MaxwellThreeDLoweringError::IncompleteClear("vertical rectangle"),
            )?,
        ))
    } else {
        None
    };
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
            rectangle,
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
        if !(surface.depth() && surface.stencil()) {
            return Err(MaxwellThreeDLoweringError::PartialDepthStencilClearUnsupported);
        }
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
            rectangle,
        )?;
        let depth = f32::from_bits(
            clear
                .depth()
                .value()
                .ok_or(MaxwellThreeDLoweringError::IncompleteClear("depth value"))?
                .get(),
        );
        let stencil = *clear
            .stencil()
            .value()
            .ok_or(MaxwellThreeDLoweringError::IncompleteClear("stencil value"))?;
        let operation = ClearOperation::image(
            region,
            image.description().kind(),
            image.description().format(),
            image.description().samples(),
            ClearValue::DepthStencil { depth, stencil },
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
    Ok((operations, dirty))
}

#[allow(clippy::too_many_arguments)]
fn lower_draw(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    shaders: &MaxwellThreeDTranslatedShaders,
    attachment_selection: &DrawAttachmentSelection,
    vertex_count: u32,
    cache: &mut MaxwellThreeDLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
) -> Result<(Vec<GpuOperation>, Vec<usize>), MaxwellThreeDLoweringError> {
    if vertex_count == 0 {
        return Err(MaxwellThreeDLoweringError::EmptyDraw);
    }
    validate_shader_stages(state, shaders)?;
    for translated in &shaders.shaders {
        let record = cache
            .shader_translations
            .iter_mut()
            .find(|record| record.id == translated.shader)
            .ok_or(MaxwellThreeDLoweringError::InvalidTranslatedShaders)?;
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
            record.published = true;
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
    let first_vertex = *state
        .vertex_input()
        .primitive()
        .vertex_array_start()
        .value()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
            "VERTEX_ARRAY_START",
        ))?;

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
    if vertex_buffers.is_empty() {
        return Err(MaxwellThreeDLoweringError::IncompleteDraw("vertex stream"));
    }

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
    let descriptor_dependencies = shaders
        .resources
        .iter()
        .map(|resource| {
            resource_index(resources, resource.role)
                .and_then(|index| binding_at(resources, bindings, index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor_tables = if descriptor_roles.is_empty() {
        Vec::new()
    } else if let Some(record) = cache.descriptors.iter().find(|record| {
        record.roles.as_ref() == descriptor_roles.as_slice()
            && record.dependencies.as_ref() == descriptor_dependencies.as_slice()
    }) {
        vec![record.id]
    } else {
        let id = DescriptorTableId::new(take_identity(cache)?);
        let description = DescriptorTableDescription::new(
            descriptor_roles
                .iter()
                .map(|role| descriptor_kind(*role))
                .collect(),
        )
        .map_err(|_| MaxwellThreeDLoweringError::InvalidTranslatedShaders)?;
        cache.descriptors.push(DescriptorRecord {
            roles: descriptor_roles.into_boxed_slice(),
            dependencies: descriptor_dependencies.clone().into_boxed_slice(),
            id,
        });
        creations.push(BackendResourceCreateInfo::DescriptorTable { id, description });
        vec![id]
    };

    let pipeline_key = PipelineKey {
        method_dependencies: state
            .pipeline_dependencies(&attachment_selection.color_targets().collect::<Vec<_>>()),
        attachments: attachment_pipeline_key(resources, bindings, attachment_selection)?,
        resource_dependencies: shaders
            .resources
            .iter()
            .zip(descriptor_dependencies.iter().copied())
            .map(|(resource, dependency)| (resource.role, dependency))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        shaders: shaders.clone(),
    };
    let pipeline = if let Some(record) = cache
        .pipelines
        .iter()
        .find(|record| record.key == pipeline_key)
    {
        record.id
    } else {
        let id = PipelineId::new(take_identity(cache)?);
        cache.pipelines.push(PipelineRecord {
            key: pipeline_key,
            id,
        });
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
            AccessScope::new(resource_use.stages, AccessMode::Read, resource_use.usage).map_err(
                |_| MaxwellThreeDLoweringError::InvalidShaderResourceUse {
                    role: resource_use.role,
                },
            )?,
        ));
        let dependency = binding_at(resources, bindings, index)?;
        if !shader_dependencies.contains(&dependency) {
            shader_dependencies.push(dependency);
        }
    }
    let mut draw = DrawOperation::new(
        pipeline,
        render_pass,
        topology,
        descriptor_tables,
        vertex_buffers,
        None,
        DrawArguments::NonIndexed {
            first_vertex,
            vertex_count,
            first_instance: 0,
            instance_count: 1,
        },
    )
    .map_err(MaxwellThreeDLoweringError::Command)?;
    if let Some(viewport_transform) = draw_viewport_transform(state)? {
        draw = draw.with_viewport_transform(viewport_transform);
    }
    let begin = RenderPassOperation::begin(render_pass, render_pass_description, attachments)
        .map_err(MaxwellThreeDLoweringError::Command)?;
    let operations = vec![
        GpuOperation::new(
            GpuCommand::RenderPass(begin),
            [],
            [],
            CapabilityRequirements::none(),
        ),
        GpuOperation::new(
            GpuCommand::Draw(draw),
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
            GpuCommand::RenderPass(RenderPassOperation::end(render_pass)),
            [],
            [],
            CapabilityRequirements::none(),
        ),
    ];
    let dirty = attachment_selection.attachment_indices();
    Ok((operations, dirty))
}

fn sequence_with_transitions(
    commands: Vec<GpuOperation>,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<Vec<GpuOperation>, MaxwellThreeDLoweringError> {
    let mut result = Vec::new();
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
            if let Some((_, scope)) = cache
                .accesses
                .iter_mut()
                .find(|(target, _)| *target == access.target())
            {
                *scope = access.scope();
            } else {
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
    let mut expected = Vec::new();
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
        expected.push(stage);
    }
    if expected.is_empty()
        || expected.len() != shaders.shaders.len()
        || expected
            .iter()
            .any(|stage| !shaders.shaders.iter().any(|shader| shader.stage == *stage))
    {
        return Err(MaxwellThreeDLoweringError::TranslatedShaderStageMismatch);
    }
    Ok(())
}

fn primitive_topology(
    begin: MaxwellThreeDBegin,
) -> Result<PrimitiveTopology, MaxwellThreeDLoweringError> {
    if begin.preserve_primitive_id() || begin.instance_id() != 0 || begin.split_mode() != 0 {
        return Err(MaxwellThreeDLoweringError::UnsupportedBeginMode);
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

fn attachment_pipeline_key(
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    selection: &DrawAttachmentSelection,
) -> Result<Box<[PipelineAttachmentKey]>, MaxwellThreeDLoweringError> {
    selection
        .attachment_indices()
        .into_iter()
        .map(|index| {
            let image = resolved_image(resources, index)?;
            Ok(PipelineAttachmentKey(
                image.role(),
                image.description(),
                image_dependency(binding_at(resources, bindings, index)?)?,
            ))
        })
        .collect::<Result<Vec<_>, MaxwellThreeDLoweringError>>()
        .map(Vec::into_boxed_slice)
}

fn view_key(resource: &MaxwellThreeDResolvedResource) -> ViewKey {
    match resource {
        MaxwellThreeDResolvedResource::Buffer(value) => ViewKey::Buffer {
            description: value.description(),
            buffer_offset: value.view().buffer_offset(),
            backing: value.view().backing().clone(),
            mappings: value.mappings().to_vec().into_boxed_slice(),
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
            mappings: value.mappings().to_vec().into_boxed_slice(),
        },
    }
}

fn canonical_backings_overlap(left: &nixe_gpu::BackingView, right: &nixe_gpu::BackingView) -> bool {
    left.range().segments().iter().any(|left| {
        right.range().segments().iter().any(|right| {
            if left.page() != right.page() {
                return false;
            }
            let left_end = left.offset() + left.size();
            let right_end = right.offset() + right.size();
            left.offset() < right_end && right.offset() < left_end
        })
    })
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

fn descriptor_kind(role: MaxwellThreeDResourceRole) -> DescriptorKind {
    match role {
        MaxwellThreeDResourceRole::Samplers => DescriptorKind::Sampler,
        MaxwellThreeDResourceRole::TextureHeaders => DescriptorKind::SampledImage,
        MaxwellThreeDResourceRole::ColorTarget(_)
        | MaxwellThreeDResourceRole::DepthStencilTarget => DescriptorKind::StorageImage,
        _ => DescriptorKind::Buffer,
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
    UnsupportedCsaaSemantics,
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
    UnsupportedClearStencilMaskSemantics,
    UnsupportedClearScissorSemantics,
    UnsupportedClearViewportClipSemantics,
    UnsupportedAliasedLineWidthSemantics,
    UnsupportedAntiAliasedLineSemantics,
    UnsupportedLineStippleSemantics {
        factor: u8,
        pattern: u16,
    },
    UnsupportedVertexArrayPrimitiveRestartSemantics {
        topology: u8,
        vertex_count: u32,
    },
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
    UnsupportedAlphaTestSemantics {
        function: super::MaxwellThreeDCompareOp,
        reference: super::MaxwellThreeDRawValue,
    },
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
    PartialDepthStencilClearUnsupported,
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
    UnsupportedBeginMode,
    UnsupportedTopology(u8),
    AliasedDrawResources {
        first: MaxwellThreeDResourceRole,
        second: MaxwellThreeDResourceRole,
    },
    InvalidTransition,
    InvalidResourceCreation,
    Command(CommandDescriptionError),
    Capability(BackendCapabilityError),
    CacheChanged {
        expected: u64,
        actual: u64,
    },
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
            Self::UnsupportedCsaaSemantics => formatter.write_str(
                "MAXWELL_B enabled CSAA has no verified coverage sampling, resolve, capability, or coherency semantics",
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
            Self::UnsupportedClearStencilMaskSemantics => formatter.write_str(
                "MAXWELL_B stencil-masked clear has no neutral backend representation",
            ),
            Self::UnsupportedClearScissorSemantics => formatter.write_str(
                "MAXWELL_B clear constrained by scissor 0 has no verified region-composition semantics",
            ),
            Self::UnsupportedClearViewportClipSemantics => formatter.write_str(
                "MAXWELL_B clear constrained by viewport clip 0 has no verified region-composition semantics",
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
            Self::UnsupportedVertexArrayPrimitiveRestartSemantics {
                topology,
                vertex_count,
            } => write!(
                formatter,
                "MAXWELL_B enabled vertex-array primitive restart may change primitive segmentation: topology={topology:#x} vertex-count={vertex_count}"
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
            Self::UnsupportedAlphaTestSemantics {
                function,
                reference,
            } => write!(
                formatter,
                "MAXWELL_B alpha testing is not represented by shader or neutral backend lowering: function={function:?} reference-bits=0x{:08x}",
                reference.get()
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
            Self::EmptyClearRectangle => formatter.write_str("clear rectangle is empty"),
            Self::PartialColorClearUnsupported { mask } => write!(
                formatter,
                "partial color-channel clear is not represented yet: mask={mask:#x}"
            ),
            Self::PartialDepthStencilClearUnsupported => {
                formatter.write_str("a partial depth/stencil aspect clear is not represented yet")
            }
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
            Self::UnsupportedBeginMode => formatter
                .write_str("BEGIN selects unsupported primitive-ID, instance, or split semantics"),
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
            Self::CacheChanged { expected, actual } => write!(
                formatter,
                "3D lowering cache changed after preflight: expected-revision={expected} actual-revision={actual}"
            ),
            Self::ResourceExhausted => {
                formatter.write_str("3D lowering exhausted host resources or identities")
            }
        }
    }
}

impl std::error::Error for MaxwellThreeDLoweringError {}

/// Returns whether restarting at this complete non-indexed draw boundary cannot
/// change primitive assembly. The published Maxwell ABI names this control but
/// does not define its segmentation algorithm, so only complete point, line,
/// and triangle lists are accepted; connected and incomplete topologies remain
/// typed failures.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1084-L1090>
const fn vertex_array_restart_is_neutral(topology: u8, vertex_count: u32) -> bool {
    match topology {
        0 => true,
        1 => vertex_count.is_multiple_of(2),
        4 => vertex_count.is_multiple_of(3),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use nixe_gpu::VertexFormat;

    use crate::MaxwellThreeDVertexAttributeFormat;

    use super::{
        depth_stencil_attachment_required, neutral_vertex_format, vertex_array_restart_is_neutral,
    };

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
    fn vertex_array_restart_is_neutral_only_at_complete_list_boundaries() {
        assert!(vertex_array_restart_is_neutral(0, 1));
        assert!(vertex_array_restart_is_neutral(0, 7));
        assert!(vertex_array_restart_is_neutral(1, 2));
        assert!(vertex_array_restart_is_neutral(1, 8));
        assert!(vertex_array_restart_is_neutral(4, 3));
        assert!(vertex_array_restart_is_neutral(4, 12));

        assert!(!vertex_array_restart_is_neutral(1, 3));
        assert!(!vertex_array_restart_is_neutral(4, 2));
        assert!(!vertex_array_restart_is_neutral(3, 4));
        assert!(!vertex_array_restart_is_neutral(5, 6));
        assert!(!vertex_array_restart_is_neutral(6, 6));
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
}
