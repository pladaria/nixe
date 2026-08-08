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
    ResourceAccess, ResourceDependency, ResourceTransition, ResourceUsage, ShaderId, ShaderStage,
};

use crate::MaxwellMethodSource;

use super::{
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDBegin, MaxwellThreeDBlendEnableCommon,
    MaxwellThreeDColorCompressionMode, MaxwellThreeDCsaaEnable, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDPolygonMode, MaxwellThreeDRenderEnableMode,
    MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources, MaxwellThreeDResourceRole,
    MaxwellThreeDShadeMode, MaxwellThreeDShaderStage, MaxwellThreeDSmTimeoutCounterBit,
    MaxwellThreeDState, MaxwellThreeDVisibleCallLimit, MaxwellThreeDZCompressionMode,
    MaxwellThreeDZCullStatsEnable,
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
    translation_generation: u64,
}

impl MaxwellThreeDTranslatedShader {
    #[must_use]
    pub const fn new(stage: ShaderStage, shader: ShaderId, translation_generation: u64) -> Self {
        Self {
            stage,
            shader,
            translation_generation,
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
    #[must_use]
    pub const fn translation_generation(self) -> u64 {
        self.translation_generation
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
    pub fn new(
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
}

/// Immutable preflight result. Resource creation, backend submission, and
/// cache publication remain separate caller-controlled phases.
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
        mut self,
        cache: &mut MaxwellThreeDLoweringCache,
    ) -> Result<MaxwellThreeDLoweredWork, MaxwellThreeDLoweringError> {
        if cache.revision != self.expected_revision {
            return Err(MaxwellThreeDLoweringError::CacheChanged {
                expected: self.expected_revision,
                actual: cache.revision,
            });
        }
        self.candidate.revision = self
            .candidate
            .revision
            .checked_add(1)
            .ok_or(MaxwellThreeDLoweringError::ResourceExhausted)?;
        *cache = self.candidate;
        Ok(MaxwellThreeDLoweredWork {
            creations: self.creations,
            invalidations: self.invalidations,
            submission: self.submission,
            dirty_images: self.dirty_images,
        })
    }
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
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(MaxwellThreeDFixedFunctionValue::ShadeMode(mode)) = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
        .value()
    {
        // The current neutral pipeline contract cannot attest whether shader
        // interpolation implements Maxwell flat or smooth shading. Reject the
        // draw before cache/backend effects instead of assuming a host default.
        return Err(MaxwellThreeDLoweringError::UnsupportedShadeModeSemantics(
            *mode,
        ));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(limit) = state
        .shader_execution()
        .visible_call_limit()
        .value()
        .copied()
        .filter(|limit| limit.limit().is_some())
    {
        // The public ABI verifies the selector values but not what hardware
        // counts as an API-visible call or where the limit is enforced. Keep
        // active limiting ahead of cache/backend effects. `NoCheck` is the
        // only encoding whose absence of a limiting effect is explicit.
        return Err(MaxwellThreeDLoweringError::UnsupportedVisibleCallLimitSemantics(limit));
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) && let Some(value) = state
        .shader_execution()
        .sm_timeout_counter_bit()
        .value()
        .copied()
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedSmTimeoutIntervalSemantics(value));
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
    ) && state.zcull().stats_enable().value() == Some(&MaxwellThreeDZCullStatsEnable::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedZCullStatsSemantics);
    }
    if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        if state
            .vertex_input()
            .primitive()
            .vertex_array_restart_enabled()
            .value()
            == Some(&super::MaxwellThreeDVertexArrayPrimitiveRestartEnable::Enabled)
        {
            return Err(
                MaxwellThreeDLoweringError::UnsupportedVertexArrayPrimitiveRestartSemantics,
            );
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
        validate_draw_blending_state(state, attachments)?;
        reject_unsupported_z_compression(state, resources, trigger)?;
        reject_unsupported_color_compression(state, resources, trigger, Some(attachments))?;
    }
    let shaders = if matches!(
        trigger,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
    ) {
        Some(translated_shaders.ok_or(MaxwellThreeDLoweringError::ShaderTranslationRequired)?)
    } else {
        None
    };
    let resource_indices = operation_resource_indices(
        state,
        resources,
        trigger,
        draw_attachments.as_ref(),
        shaders,
    )?;
    let mut candidate = cache.clone();
    let mut creations = Vec::new();
    let mut invalidations = Vec::new();
    let resource_bindings = prepare_resources(
        resources,
        &resource_indices,
        &mut candidate,
        &mut creations,
        &mut invalidations,
    )?;
    let (commands, dirty_images) = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { source: _ } => {
            reject_unsupported_z_compression(state, resources, trigger)?;
            reject_unsupported_color_compression(state, resources, trigger, None)?;
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
    for creation in &creations {
        let requirements = creation
            .capability_requirements()
            .map_err(|_| MaxwellThreeDLoweringError::InvalidResourceCreation)?;
        capabilities
            .negotiate(&requirements)
            .map_err(MaxwellThreeDLoweringError::Capability)?;
    }
    let agreement = capabilities
        .negotiate_all(&submission.capability_requirements())
        .map_err(MaxwellThreeDLoweringError::Capability)?;
    Ok(MaxwellThreeDLoweringPlan {
        expected_revision: cache.revision,
        candidate,
        creations: creations.into_boxed_slice(),
        invalidations: invalidations.into_boxed_slice(),
        submission,
        agreement,
        dirty_images: dirty_images.into_boxed_slice(),
    })
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
        .begin()
        .value()
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

fn reject_unsupported_z_compression(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
) -> Result<(), MaxwellThreeDLoweringError> {
    if state.render_targets().depth_stencil().compression().value()
        != Some(&MaxwellThreeDZCompressionMode::Enabled)
    {
        return Ok(());
    }
    let consumes_depth = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => state
            .render_targets()
            .clear()
            .last_surface()
            .value()
            .is_some_and(|surface| surface.depth() || surface.stencil()),
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. } => resources
            .resources()
            .iter()
            .any(|resource| resource.role() == MaxwellThreeDResourceRole::DepthStencilTarget),
    };
    if consumes_depth {
        return Err(MaxwellThreeDLoweringError::UnsupportedZCompressionSemantics);
    }
    Ok(())
}

fn reject_unsupported_color_compression(
    state: &MaxwellThreeDState,
    _resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
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
    if let Some(target) = consumed_target
        && state.render_targets().color()[target as usize]
            .compression()
            .value()
            == Some(&MaxwellThreeDColorCompressionMode::Enabled)
    {
        return Err(MaxwellThreeDLoweringError::UnsupportedColorCompressionSemantics { target });
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
    let depth_stencil = resources
        .resources()
        .iter()
        .position(|resource| resource.role() == MaxwellThreeDResourceRole::DepthStencilTarget);
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
                    view: Some(view),
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
    let horizontal =
        clear
            .horizontal()
            .value()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle",
            ))?;
    let vertical =
        clear
            .vertical()
            .value()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteClear(
                "vertical rectangle",
            ))?;
    let width = u32::from(horizontal.max.saturating_sub(horizontal.min));
    let height = u32::from(vertical.max.saturating_sub(vertical.min));
    if width == 0 || height == 0 {
        return Err(MaxwellThreeDLoweringError::EmptyClearRectangle);
    }
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
        if surface.array_layer() != subresources.base_layer
            || u32::from(horizontal.max) > image.description().extent().width
            || u32::from(vertical.max) > image.description().extent().height
        {
            return Err(MaxwellThreeDLoweringError::ClearOutsideAttachment);
        }
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
            ImageRegion {
                image: image_id,
                subresources,
                origin: ImageOrigin {
                    x: u32::from(horizontal.min),
                    y: u32::from(vertical.min),
                    z: 0,
                },
                extent: nixe_gpu::ImageExtent {
                    width,
                    height,
                    depth: 1,
                },
            },
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
            ImageRegion {
                image: image_id,
                subresources,
                origin: ImageOrigin {
                    x: u32::from(horizontal.min),
                    y: u32::from(vertical.min),
                    z: 0,
                },
                extent: nixe_gpu::ImageExtent {
                    width,
                    height,
                    depth: 1,
                },
            },
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
    if operations.is_empty() {
        return Err(MaxwellThreeDLoweringError::EmptyClearMask);
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
    let topology = primitive_topology(
        state
            .vertex_input()
            .primitive()
            .begin()
            .value()
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
        if !stream.format().value().is_some_and(|value| value.enabled()) {
            continue;
        }
        let resource = resource_index(
            resources,
            MaxwellThreeDResourceRole::VertexStream(index as u8),
        )?;
        let buffer = resolved_buffer(resources, resource)?;
        vertex_buffers.push(BufferRegion {
            buffer: buffer_dependency(binding_at(resources, bindings, resource)?)?,
            range: BufferRange::new(0, buffer.description().size()).map_err(|_| {
                MaxwellThreeDLoweringError::InvalidResolvedView {
                    role: buffer.role(),
                }
            })?,
        });
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
    let draw = DrawOperation::new(
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
#[derive(Debug)]
pub enum MaxwellThreeDLoweringError {
    ContradictoryState {
        reason: &'static str,
    },
    TriggerStateMismatch,
    UnsupportedRenderEnableMode(MaxwellThreeDRenderEnableMode),
    UnsupportedSmTimeoutIntervalSemantics(MaxwellThreeDSmTimeoutCounterBit),
    UnsupportedVisibleCallLimitSemantics(MaxwellThreeDVisibleCallLimit),
    UnsupportedCsaaSemantics,
    UnsupportedZCullStatsSemantics,
    UnsupportedAliasedLineWidthSemantics,
    UnsupportedVertexArrayPrimitiveRestartSemantics,
    UnsupportedShadeModeSemantics(MaxwellThreeDShadeMode),
    UnsupportedBlendSemantics {
        target: Option<u8>,
    },
    UnsupportedZCompressionSemantics,
    UnsupportedColorCompressionSemantics {
        target: u8,
    },
    ShaderTranslationRequired,
    InvalidTranslatedShaders,
    TranslatedShaderStageMismatch,
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
            Self::UnsupportedSmTimeoutIntervalSemantics(value) => write!(
                formatter,
                "MAXWELL_B SM timeout interval has no verified temporal semantics: counter-bit={}",
                value.get()
            ),
            Self::UnsupportedVisibleCallLimitSemantics(limit) => match limit.limit() {
                Some(call_limit) => write!(
                    formatter,
                    "MAXWELL_B API-visible call limiting is not implemented: selector=0x{:x} limit={call_limit}",
                    limit.raw()
                ),
                None => formatter.write_str(
                    "MAXWELL_B API-visible call limiting reported an unsupported boundary for NO_CHECK",
                ),
            },
            Self::UnsupportedCsaaSemantics => formatter.write_str(
                "MAXWELL_B enabled CSAA has no verified coverage sampling, resolve, capability, or coherency semantics",
            ),
            Self::UnsupportedZCullStatsSemantics => formatter.write_str(
                "MAXWELL_B enabled Z-cull statistics have no implemented counter accumulation, visibility, or reporting semantics",
            ),
            Self::UnsupportedAliasedLineWidthSemantics => formatter.write_str(
                "MAXWELL_B aliased line-width selection has no represented width register or host rasterization semantics",
            ),
            Self::UnsupportedVertexArrayPrimitiveRestartSemantics => formatter.write_str(
                "MAXWELL_B enabled vertex-array primitive restart has no verified marker, segmentation, draw-accounting, or neutral backend semantics",
            ),
            Self::UnsupportedShadeModeSemantics(mode) => write!(
                formatter,
                "MAXWELL_B shade mode is not representable in the neutral pipeline contract: mode={mode:?}"
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
            Self::UnsupportedZCompressionSemantics => formatter.write_str(
                "MAXWELL_B enabled Z compression has no verified representation or coherency semantics",
            ),
            Self::UnsupportedColorCompressionSemantics { target } => write!(
                formatter,
                "MAXWELL_B enabled color compression has no verified representation or coherency semantics: target={target}"
            ),
            Self::ShaderTranslationRequired => {
                formatter.write_str("Maxwell shader translation is required before draw lowering")
            }
            Self::InvalidTranslatedShaders => {
                formatter.write_str("translated shader evidence is empty, duplicated, or invalid")
            }
            Self::TranslatedShaderStageMismatch => formatter
                .write_str("translated shader stages do not match enabled Maxwell pipeline stages"),
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
