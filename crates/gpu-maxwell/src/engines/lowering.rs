//! Shared frontend resource ownership and ordered graphics/compute lowering.
//!
//! This boundary produces only backend-independent `nixe-gpu` resources and
//! operations. Maxwell shader translation supplies verified neutral IR; raw
//! guest code never reaches the host backend as a shader module.

mod buffer;
#[cfg(test)]
mod buffer_cache_tests;
mod color;
#[cfg(test)]
mod compressed_sampling_tests;
mod compute;
mod indexed;
mod multisample;
mod raster;
mod resolve;
mod sampled_alias;

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

use super::threed::{
    MaxwellShaderStage, MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAlphaToCoverageOverride,
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
    MaxwellThreeDShadeMode, MaxwellThreeDShaderLocalMemoryPerWarpSize, MaxwellThreeDState,
    MaxwellThreeDTextureDimension, MaxwellThreeDTirControl, MaxwellThreeDTirMode,
    MaxwellThreeDVertexNumericalType, MaxwellThreeDViewportCoordinateSwizzle,
    MaxwellThreeDViewportPixelCenter, MaxwellThreeDViewportScaleOffsetEnable,
    MaxwellThreeDViewportSwizzleComponent,
};

#[derive(Clone, Debug)]
struct DrawAttachmentSelection {
    colors: Vec<(u8, usize)>,
    color_outputs: [nixe_gpu::ColorOutputState; 8],
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
    DrawIndexBuffer {
        source: MaxwellMethodSource,
        index_count: u32,
    },
}

impl MaxwellThreeDOperationTrigger {
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::ClearSurface { source }
            | Self::DrawVertexArray { source, .. }
            | Self::DrawIndexBuffer { source, .. } => source,
        }
    }

    pub(crate) const fn is_draw(self) -> bool {
        matches!(
            self,
            Self::DrawVertexArray { .. } | Self::DrawIndexBuffer { .. }
        )
    }

    const fn is_indexed(self) -> bool {
        matches!(self, Self::DrawIndexBuffer { .. })
    }

    /// Appends the resources consumed directly by this trigger.
    ///
    /// Shader translation contributes constant-buffer, texture, and sampler
    /// roles separately. Keeping the trigger-specific selection exhaustive
    /// ensures that indexed draws also resolve the index storage.
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
            Self::DrawVertexArray { .. } | Self::DrawIndexBuffer { .. } => {
                if self.is_indexed() {
                    roles.push(MaxwellThreeDResourceRole::IndexBuffer);
                }
                roles.extend(
                    consumed_vertex_streams(state).map(MaxwellThreeDResourceRole::VertexStream),
                );
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

/// A stream's enable bit alone does not cause vertex fetches. Draws can leave
/// old streams configured while disabling all their attributes (for example,
/// when a subsequent shader generates geometry from the vertex ID).
fn consumed_vertex_streams(state: &MaxwellThreeDState) -> impl Iterator<Item = u8> {
    let mut mask = state.vertex_input().attributes().iter().enumerate().fold(
        0_u32,
        |mask, (index, attribute)| match attribute.value().filter(|attribute| {
            attribute.enabled() && state.vertex_input().attribute_skip_mask(index as u8) != 15
        }) {
            Some(attribute) => mask | (1 << attribute.stream()),
            None => mask,
        },
    );
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let stream = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        Some(stream)
    })
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
    ) -> Result<Self, MaxwellLoweringError> {
        if let Some(usage) = usage {
            let _ = AccessScope::new(stages, AccessMode::Read, usage)
                .map_err(|_| MaxwellLoweringError::InvalidShaderResourceUse { role })?;
        } else if kind != DescriptorKind::Sampler {
            return Err(MaxwellLoweringError::InvalidShaderResourceUse { role });
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
    ) -> Result<Self, MaxwellLoweringError> {
        if shaders.is_empty() {
            return Err(MaxwellLoweringError::ShaderTranslationRequired);
        }
        for (index, shader) in shaders.iter().enumerate() {
            if shader.stage == ShaderStage::Compute
                || shaders[index + 1..]
                    .iter()
                    .any(|other| other.stage == shader.stage)
            {
                return Err(MaxwellLoweringError::InvalidTranslatedShaders);
            }
        }
        for (index, resource) in resources.iter().enumerate() {
            if resources[index + 1..].contains(resource) {
                return Err(MaxwellLoweringError::InvalidTranslatedShaders);
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
        mappings: Arc<[super::threed::MaxwellThreeDMappingReference]>,
    },
    Image {
        description: nixe_gpu::ImageDescription,
        swizzle: nixe_gpu::Swizzle,
        guest_format: super::threed::MaxwellThreeDGuestImageFormat,
        guest_pte_kind: u8,
        guest_compression_enabled: bool,
        bindings: Box<
            [(
                ImageSubresourceRange,
                nixe_gpu::ImageMemoryLayout,
                nixe_gpu::BackingView,
            )],
        >,
        mappings: Arc<[super::threed::MaxwellThreeDMappingReference]>,
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
    fn same_domain_as_image(&self, image: &super::threed::MaxwellThreeDResolvedImage) -> bool {
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
            && (same_guest_image_interpretation(
                *guest_format,
                *guest_compression_enabled,
                image.guest_format(),
                image.guest_layout().requires_materialization(),
            ) || (image.role() == MaxwellThreeDResourceRole::BlitSource
                && *guest_format == image.guest_format()))
            && *guest_pte_kind == image.guest_layout().pte_kind()
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

impl MaxwellLoweringCache {
    pub(crate) fn inline_image_word(
        &self,
        target: &crate::MaxwellResolvedRange,
        hint: Option<usize>,
    ) -> Result<Option<(usize, ImageRegion)>, MaxwellLoweringError> {
        let [segment] = target.segments() else {
            return Ok(None);
        };
        let matches = |record: &ViewRecord| {
            let ViewKey::Image { bindings, .. } = &record.key else {
                return false;
            };
            record.materialization == ViewMaterialization::CompressedColor
                && bindings.len() == 1
                && bindings[0].2.allocation().get() == segment.mapping().allocation().get()
                && segment.backing_offset() >= bindings[0].2.allocation_offset()
                && segment.backing_offset() + 4
                    <= bindings[0].2.allocation_offset() + bindings[0].2.range().size()
        };
        let index = hint
            .filter(|index| self.views.get(*index).is_some_and(&matches))
            .or_else(|| self.views.iter().position(matches));
        let Some(index) = index else {
            return Ok(None);
        };
        let record = &self.views[index];
        let ViewKey::Image {
            description,
            bindings,
            ..
        } = &record.key
        else {
            unreachable!();
        };
        if hint != Some(index)
            && (!record
                .cpu_writes
                .as_ref()
                .is_some_and(CanonicalCpuWriteDependency::remains_current))
        {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "inline image upload requires a current producer",
            });
        }
        if description.format().plane_bytes_per_texel(0) != Some(4)
            || description.samples() != nixe_gpu::SampleCount::One
            || bindings[0].0.layer_count != 1
        {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "inline image upload requires single-layer C32 storage",
            });
        }
        let nixe_gpu::ImageMemoryLayout::BlockLinear(layout) = bindings[0].1 else {
            return Ok(None);
        };
        let offset = segment.backing_offset() - bindings[0].2.allocation_offset();
        let (x, y) = inline_block_linear_position(
            offset,
            description.extent().width,
            layout.block_height_log2,
        );
        if x % 4 != 0
            || x / 4 >= u64::from(description.extent().width)
            || y >= u64::from(description.extent().height)
        {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "inline image upload targets padding or an unaligned texel",
            });
        }
        Ok(Some((
            index,
            ImageRegion {
                image: image_dependency(record.dependency)?,
                subresources: bindings[0].0,
                origin: ImageOrigin {
                    x: (x / 4) as u32,
                    y: y as u32,
                    z: 0,
                },
                extent: nixe_gpu::ImageExtent {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            },
        )))
    }

    pub(crate) fn lower_inline_images(
        &mut self,
        uploads: Vec<(ImageRegion, Vec<u8>)>,
        submission: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        let revision = self.revision.saturating_add(1);
        let mut commands = Vec::with_capacity(uploads.len());
        for (destination, bytes) in uploads {
            if let Some(record) = self
                .views
                .iter_mut()
                .find(|record| record.dependency == ResourceDependency::Image(destination.image))
            {
                record.write_revision = revision;
                let incomplete = !record.uninitialized_color_regions.is_empty();
                resolve::subtract_initialized_rect(
                    &mut record.uninitialized_color_regions,
                    [
                        destination.origin.x,
                        destination.origin.y,
                        destination.origin.x + destination.extent.width,
                        destination.origin.y + destination.extent.height,
                    ],
                );
                if incomplete && record.uninitialized_color_regions.is_empty() {
                    let ViewKey::Image {
                        description,
                        swizzle,
                        guest_format,
                        guest_pte_kind,
                        guest_compression_enabled,
                        bindings,
                        ..
                    } = &record.key
                    else {
                        unreachable!();
                    };
                    self.color_materializations.push(ColorRepresentationRecord {
                        description: *description,
                        swizzle: *swizzle,
                        guest_format: *guest_format,
                        guest_pte_kind: *guest_pte_kind,
                        guest_compression_enabled: *guest_compression_enabled,
                        bindings: bindings
                            .iter()
                            .map(
                                |(subresources, layout, backing)| ColorRepresentationBinding {
                                    subresources: *subresources,
                                    layout: *layout,
                                    backing: backing.clone(),
                                },
                            )
                            .collect(),
                        cpu_writes: record.cpu_writes.clone(),
                    });
                }
            }
            commands.push(GpuOperation::new(
                GpuCommand::UploadImage {
                    destination,
                    bytes: bytes.into(),
                },
                [],
                [],
                CapabilityRequirements::none(),
            ));
        }
        finish_lowered_work(
            self,
            submission,
            predecessors,
            Vec::new(),
            Vec::new(),
            commands,
            Arc::from([]),
        )
    }
}

fn inline_block_linear_position(offset: u64, width: u32, block_height_log2: u8) -> (u64, u64) {
    // Inverse of Tegra's documented 16Bx2 GOB address mapping.
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c
    let height = 1_u64 << block_height_log2;
    let row = (u64::from(width) * 4).div_ceil(64) * 512 * height;
    let gob = offset % 512;
    let x = (offset % row) / (512 * height) * 64 + gob / 256 * 32 + gob % 64 / 32 * 16 + gob % 16;
    let y = offset / row * 8 * height
        + offset % (512 * height) / 512 * 8
        + gob % 256 / 64 * 2
        + gob % 32 / 16;
    (x, y)
}

fn same_canonical_backing(left: &nixe_gpu::BackingView, right: &nixe_gpu::BackingView) -> bool {
    // Reject distinct byte coverage using the compressed span index before
    // comparing potentially thousands of retained page segments. The ordered
    // segment comparison below still distinguishes differently ordered aliases.
    if left.canonical_spans() != right.canonical_spans() {
        return false;
    }
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

/// Exact neutral descriptions and layouts are checked by the caller. Color
/// targets and TICs encode the same texel format in different register domains;
/// their raw encodings must not prevent reuse of the rendered image.
/// A TIC has no write-compression selector: reading an existing representation
/// does not depend on whether the producer enabled compression of its writes.
fn same_guest_image_interpretation(
    left: super::threed::MaxwellThreeDGuestImageFormat,
    left_compression: bool,
    right: super::threed::MaxwellThreeDGuestImageFormat,
    right_compression: bool,
) -> bool {
    use super::threed::MaxwellThreeDGuestImageFormat::{Color, Texture};
    (left == right && left_compression == right_compression)
        || matches!(
            (left, right),
            (Color(_), Texture(_)) | (Texture(_), Color(_))
        )
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
    guest_format: super::threed::MaxwellThreeDGuestImageFormat,
    guest_pte_kind: u8,
    guest_compression_enabled: bool,
    bindings: Box<[ColorRepresentationBinding]>,
    cpu_writes: Option<CanonicalCpuWriteDependency>,
}

impl ColorRepresentationRecord {
    fn same_domain_as_image(&self, image: &super::threed::MaxwellThreeDResolvedImage) -> bool {
        self.description == image.description()
            && self.swizzle == image.view().swizzle()
            && (same_guest_image_interpretation(
                self.guest_format,
                self.guest_compression_enabled,
                image.guest_format(),
                image.guest_layout().requires_materialization(),
            ) || (image.role() == MaxwellThreeDResourceRole::BlitSource
                && self.guest_format == image.guest_format()))
            && self.guest_pte_kind == image.guest_layout().pte_kind()
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

    fn remains_materialized_for(&self, image: &super::threed::MaxwellThreeDResolvedImage) -> bool {
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
    write_revision: u64,
    last_used: u64,
    uninitialized_color_regions: Vec<[u32; 4]>,
    uninitialized_depth_stencil_regions: [Vec<[u32; 4]>; 2],
}

impl ViewRecord {
    fn remains_current_for_image(&self, image: &super::threed::MaxwellThreeDResolvedImage) -> bool {
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
    CopiedColor { source: ImageId, revision: u64 },
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
            Self::CompressedColor | Self::CopiedColor { .. } => false,
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
    kinds: Box<[DescriptorKind]>,
    bindings: Box<[u8]>,
    dependencies: Box<[ResourceDependency]>,
    id: DescriptorTableId,
}

#[derive(Debug)]
struct PreparedDrawRecord {
    indexed: bool,
    state: super::threed::state::MaxwellThreeDDrawStateIdentity,
    resources: Arc<()>,
    shaders: Arc<()>,
    operations: [GpuOperation; 3],
    dirty_images: Arc<[usize]>,
    sampled_aliases: Box<[ResourceDependency]>,
}

impl PreparedDrawRecord {
    fn matches(
        &self,
        state: &MaxwellThreeDState,
        resources: &MaxwellThreeDResolvedResources,
        shaders: &MaxwellThreeDTranslatedShaders,
        indexed: bool,
    ) -> bool {
        self.indexed == indexed
            && self.state.matches(state)
            && resources.has_identity(&self.resources)
            && shaders.has_identity(&self.shaders)
    }

    fn operations(
        &self,
        arguments: DrawArguments,
    ) -> Result<[GpuOperation; 3], MaxwellLoweringError> {
        Ok([
            self.operations[0].clone(),
            self.operations[1]
                .with_draw_arguments(arguments)
                .map_err(MaxwellLoweringError::Command)?,
            self.operations[2].clone(),
        ])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SamplerRecord {
    sampler: super::threed::MaxwellThreeDResolvedSampler,
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

#[derive(Clone, Debug)]
struct ShaderStateRecord {
    state: super::threed::MaxwellThreeDShaderStateIdentity,
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
pub struct MaxwellLoweringCache {
    configuration: GpuCacheConfiguration,
    revision: u64,
    next_identity: u64,
    allocations: Vec<(
        nixe_gpu::GpuAllocationId,
        nixe_gpu::GpuAllocationDescription,
    )>,
    views: Vec<ViewRecord>,
    color_materializations: Vec<ColorRepresentationRecord>,
    image_alias_copies: Vec<GpuOperation>,
    graphics_pipeline: Option<PipelineId>,
    compute_pipeline: Option<PipelineId>,
    compute_shaders: FingerprintCache<compute::ComputeShaderRecord>,
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
    resolved_resources: super::threed::MaxwellThreeDResolvedResourceCache,
    resource_roles: Vec<MaxwellThreeDResourceRole>,
    mme_methods: Vec<crate::MaxwellMethodDispatch>,
    mme_parameters: Vec<u32>,
}

impl Default for MaxwellLoweringCache {
    fn default() -> Self {
        Self::new(GpuCacheConfiguration::default())
    }
}

impl MaxwellLoweringCache {
    #[must_use]
    pub fn new(configuration: GpuCacheConfiguration) -> Self {
        Self {
            configuration,
            revision: 0,
            next_identity: 1,
            allocations: Vec::new(),
            views: Vec::new(),
            color_materializations: Vec::new(),
            image_alias_copies: Vec::new(),
            graphics_pipeline: None,
            compute_pipeline: None,
            compute_shaders: FingerprintCache::default(),
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
            resolved_resources: super::threed::MaxwellThreeDResolvedResourceCache::default(),
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
    ) -> &mut super::threed::MaxwellThreeDResolvedResourceCache {
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
    ) -> Result<MaxwellThreeDTranslatedShaders, MaxwellLoweringError> {
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
                                    .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?,
                                slot: program
                                    .local_resource_binding(resource.binding())
                                    .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?,
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
                            .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?;
                        (
                            MaxwellThreeDResourceRole::SampledImage {
                                texture: super::threed::MaxwellThreeDTextureReference::new(
                                    program
                                        .bind_group()
                                        .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?,
                                    program
                                        .texture_constant_buffer_slot()
                                        .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?,
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
                                        return Err(MaxwellLoweringError::InvalidTranslatedShaders);
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
                            .find(|binding| binding.sampler_binding() == Some(resource.binding()))
                            .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?;
                        (
                            MaxwellThreeDResourceRole::Sampler(
                                super::threed::MaxwellThreeDTextureReference::new(
                                    program
                                        .bind_group()
                                        .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?,
                                    program
                                        .texture_constant_buffer_slot()
                                        .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?,
                                    texture.constant_buffer_byte_offset(),
                                ),
                            ),
                            DescriptorKind::Sampler,
                            None,
                        )
                    }
                    _ => return Err(MaxwellLoweringError::InvalidTranslatedShaders),
                };
                if let Some(existing) = resources.iter_mut().find(|existing| existing.role == role)
                {
                    if existing.binding != resource.binding() || existing.kind != kind {
                        return Err(MaxwellLoweringError::InvalidTranslatedShaders);
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
            let module = nixe_gpu::ShaderBackendModule::new(ir);
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
pub struct MaxwellLoweredWork {
    creations: Box<[BackendResourceCreateInfo]>,
    invalidations: Box<[ResourceDependency]>,
    submission: OperationSubmission,
    dirty_images: Arc<[usize]>,
}

impl MaxwellLoweredWork {
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
    cache: &mut MaxwellLoweringCache,
) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
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
            .map_err(|_| MaxwellLoweringError::InvalidResourceCreation)?;
        capabilities
            .negotiate(&requirements)
            .map_err(MaxwellLoweringError::Capability)?;
    }
    capabilities
        .negotiate_all(&work.submission().capability_requirements())
        .map_err(MaxwellLoweringError::Capability)?;
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
    cache: &mut MaxwellLoweringCache,
) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
    if trigger.is_draw()
        && let Some(shaders) = translated_shaders
    {
        let arguments = draw_arguments(state, trigger)?;
        let prepared = cache
            .prepared_draw
            .as_ref()
            .filter(|prepared| {
                prepared.matches(state, resources, shaders, trigger.is_indexed())
                    && prepared.sampled_aliases.iter().all(|dependency| {
                        cache
                            .views
                            .iter()
                            .find(|record| record.dependency == *dependency)
                            .is_some_and(|record| sampled_alias::copy_is_current(record, cache))
                    })
            })
            .map(|prepared| {
                Ok::<_, MaxwellLoweringError>((
                    prepared.operations(arguments)?,
                    Arc::clone(&prepared.dirty_images),
                ))
            })
            .transpose()?;
        if let Some((commands, dirty_images)) = prepared {
            for index in dirty_images.iter() {
                let image = resolved_image(resources, *index)?;
                record_image_write(image, cache);
            }
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
    let mut raster_state = None;
    let tessellation = if trigger.is_draw() {
        super::threed::tessellation::draw_state(state)?
    } else {
        None
    };
    if let Some(mode) = state.render_enable().execution_mode()
        && mode != MaxwellThreeDRenderEnableMode::Enabled
    {
        return Err(MaxwellLoweringError::UnsupportedRenderEnableMode(mode));
    }
    if state
        .render_enable()
        .conditional_load_constant_buffer()
        .value()
        == Some(&MaxwellThreeDConditionalLoadConstantBuffer::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedConditionalLoadConstantBufferSemantics);
    }
    if trigger.is_draw()
        && let Some(MaxwellThreeDFixedFunctionValue::ShadeMode(MaxwellThreeDShadeMode::Flat)) =
            state
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
                .value()
    {
        // Smooth preserves the interpolation selected by translated shader
        // inputs and therefore needs no fixed-function override. Flat shading
        // changes the primitive-wide source value and remains a typed boundary
        // until T10 represents that override explicitly.
        return Err(MaxwellLoweringError::UnsupportedShadeModeSemantics(
            MaxwellThreeDShadeMode::Flat,
        ));
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ProvokingVertex)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::ProvokingVertex(
                MaxwellThreeDProvokingVertex::First,
            ))
    {
        return Err(MaxwellLoweringError::UnsupportedProvokingVertexSemantics(
            MaxwellThreeDProvokingVertex::First,
        ));
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::TwoSidedLightEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellLoweringError::UnsupportedTwoSidedLightSemantics);
    }
    // SET_COLOR_CLAMP applies to legacy vertex COLOR/BCOLOR attributes,
    // not generic varyings or fragment outputs. Their SPH maps/attribute
    // transfers are rejected explicitly by the shader translator until that
    // interface is implemented, so no supported draw consumes this clamp.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/gallium/drivers/nouveau/nvc0/nvc0_state.c#L231-L233
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/gallium/drivers/nouveau/nvc0/nvc0_program.c#L53-L54
    if trigger.is_draw()
        && let Some(MaxwellThreeDFixedFunctionValue::PixelShaderSaturate(value)) = state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::PixelShaderSaturate)
            .value()
        && let Some(output) = value.first_enabled_output()
    {
        return Err(
            MaxwellLoweringError::UnsupportedPixelShaderSaturateSemantics {
                output,
                range: value
                    .clamp_range(output)
                    .expect("enabled output is within the eight-output register"),
            },
        );
    }
    if trigger.is_draw() && state.shader_bindings().has_enabled_pipeline() {
        let local_memory = state.shader_execution().shader_local_memory();
        if local_memory.region_is_partially_programmed() {
            return Err(MaxwellLoweringError::IncompleteDraw(
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
                return Err(MaxwellLoweringError::IncompleteDraw(
                    "SET_SHADER_LOCAL_MEMORY_A-D",
                ));
            }
            return Err(
                MaxwellLoweringError::UnsupportedShaderLocalMemorySemantics {
                    default_size_per_warp,
                },
            );
        }
    }
    if trigger.is_draw()
        && state.color_reduction().enable().value()
            == Some(&MaxwellThreeDColorReductionThresholdsEnable::Enabled)
    {
        // NVIDIA exposes a dedicated activation method, so merely programming
        // a threshold is not enough to make it effective. Once explicitly
        // enabled, however, the current neutral pipeline cannot represent the
        // reduction decision and must stop before cache/backend effects.
        return Err(MaxwellLoweringError::UnsupportedColorReductionSemantics);
    }
    if trigger.is_draw() && state.constant_color_rendering().enabled().value() == Some(&true) {
        return Err(MaxwellLoweringError::UnsupportedConstantColorRenderingSemantics);
    }
    if trigger.is_draw()
        && state.shader_execution().api_mandated_early_z().value()
            == Some(&MaxwellThreeDApiMandatedEarlyZ::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedApiMandatedEarlyZSemantics);
    }
    if trigger.is_draw()
        && state.coverage().post_ps_initial_coverage().value() == Some(&true)
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::AlphaTestEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        // The host implements alpha test through shader discard. Its
        // interaction with a pre-PS initial mask needs explicit lowering.
        return Err(MaxwellLoweringError::UnsupportedPostPsInitialCoverageSemantics);
    }
    if trigger.is_draw()
        && state.coverage().post_z_pixel_shader_imask().value()
            == Some(&MaxwellThreeDPostZPixelShaderImask::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedPostZPixelShaderImaskSemantics);
    }
    if trigger.is_draw()
        && let Some(value) = state
            .shader_execution()
            .pixel_shader_interlock_control()
            .value()
            .copied()
            .filter(|value| value.conflict_detection_enabled())
    {
        return Err(MaxwellLoweringError::UnsupportedPixelShaderInterlockSemantics(value));
    }
    if !trigger.is_indexed()
        && trigger.is_draw()
        && let Some(base_vertex) = state
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
        return Err(MaxwellLoweringError::UnsupportedGlobalBaseVertexIndex(
            base_vertex,
        ));
    }
    if !trigger.is_indexed()
        && trigger.is_draw()
        && let Some(base) = state
            .vertex_input()
            .assembly()
            .vertex_id_base()
            .value()
            .copied()
            .filter(|value| *value != 0)
    {
        // A zero base preserves the existing shader-visible index. Nonzero
        // bases need a separate shader-ID adjustment, not first_vertex (which
        // would also change vertex-buffer fetches). Keep this register in the
        // prepared-draw semantic revision so a cached draw cannot bypass it.
        return Err(MaxwellLoweringError::UnsupportedVertexIdBase(base));
    }
    if trigger.is_draw()
        && state.coverage().csaa_enable().value() == Some(&MaxwellThreeDCsaaEnable::Enabled)
    {
        return Err(MaxwellLoweringError::UnsupportedCsaaSemantics);
    }
    // Dither footprint is only configuration while alpha-to-coverage is off.
    // Neither coverage generation (including dithering) nor alpha-to-one is
    // represented by the current neutral pipeline. Reject their activation,
    // including after a cached draw, rather than ignoring a stored selector.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
    if trigger.is_draw()
        && let Some(MaxwellThreeDFixedFunctionValue::AlphaControl {
            alpha_to_coverage,
            alpha_to_one,
        }) = state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::AlphaToCoverageEnable)
            .value()
        && (*alpha_to_coverage || *alpha_to_one)
    {
        return Err(MaxwellLoweringError::UnsupportedAntiAliasAlphaControl {
            alpha_to_coverage: *alpha_to_coverage,
            alpha_to_one: *alpha_to_one,
        });
    }
    if trigger.is_draw()
        && let Some(value) = state
            .coverage()
            .coverage_to_color()
            .value()
            .copied()
            .filter(|value| value.enabled())
    {
        return Err(MaxwellLoweringError::UnsupportedCoverageToColorSemantics(
            value,
        ));
    }
    if trigger.is_draw()
        && let Some(value) = state
            .coverage()
            .alpha_to_coverage_override()
            .value()
            .copied()
            .filter(|value| value.raw() != 0)
    {
        return Err(MaxwellLoweringError::UnsupportedAlphaToCoverageOverrideSemantics(value));
    }
    if trigger.is_draw()
        && state.coverage().tir_mode().value() == Some(&MaxwellThreeDTirMode::RasterNTargetM)
    {
        return Err(MaxwellLoweringError::UnsupportedTirSemantics {
            control: state.coverage().tir_control().value().copied(),
        });
    }
    if trigger.is_draw()
        && let Some(value) = state
            .coverage()
            .hybrid_anti_alias_control()
            .value()
            .copied()
            .filter(|value| !value.is_single_pass_per_fragment())
    {
        return Err(MaxwellLoweringError::UnsupportedHybridAntiAliasSemantics(
            value,
        ));
    }
    if trigger.is_draw() {
        multisample::validate(state)?;
    }
    if trigger.is_draw() {
        draw_viewport_transform(state)?;
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true))
    {
        return Err(MaxwellLoweringError::UnsupportedWindowClipSemantics);
    }
    if trigger.is_draw()
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
            .value()
            == Some(&MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(
                MaxwellThreeDClipIdTestEnable::Enabled,
            ))
    {
        return Err(MaxwellLoweringError::UnsupportedClipIdTestSemantics);
    }
    if trigger.is_draw() {
        if state.viewport().pixel_center().value()
            == Some(&MaxwellThreeDViewportPixelCenter::Integers)
        {
            return Err(
                MaxwellLoweringError::UnsupportedViewportPixelCenterSemantics(
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
            return Err(MaxwellLoweringError::UnsupportedFillViaTriangleSemantics(
                mode,
            ));
        }
        if state.raster().conservative_raster().value()
            == Some(&MaxwellThreeDConservativeRasterEnable::Enabled)
        {
            return Err(MaxwellLoweringError::UnsupportedConservativeRasterSemantics);
        }
        if state.shader_bindings().has_enabled_pipeline()
            && state
                .shader_bindings()
                .program_region()
                .is_partially_programmed()
        {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_PROGRAM_REGION_A/B",
            ));
        }
        if state.generated_primitive() == Some(super::threed::state::GeneratedPrimitive::Points) {
            if let Some(value) = state
                .raster()
                .attribute_point_size()
                .value()
                .copied()
                .filter(|value| value.enabled())
            {
                return Err(
                    MaxwellLoweringError::UnsupportedAttributePointSizeSemantics {
                        slot: value.slot(),
                    },
                );
            }
            if state.raster().point_sprite_enable().value() == Some(&true) {
                return Err(MaxwellLoweringError::UnsupportedPointSpriteSemantics);
            }
            if state.raster().anti_aliased_point_enable().value() == Some(&true) {
                return Err(MaxwellLoweringError::UnsupportedAntiAliasedPointSemantics);
            }
            if let Some(select) = state
                .raster()
                .point_sprite_select()
                .value()
                .copied()
                .filter(|select| select.affects_point_coordinates())
            {
                return Err(
                    MaxwellLoweringError::UnsupportedPointSpriteCoordinatesSemantics(select),
                );
            }
            if let Some(mode) = state.raster().point_center_mode().value().copied() {
                return Err(MaxwellLoweringError::UnsupportedPointCenterSemantics(mode));
            }
        }
        validate_direct_line_rasterization_state(state)?;
        raster_state = Some(raster::draw_state(state)?);
    }
    if let MaxwellThreeDOperationTrigger::ClearSurface { source } = trigger
        && state.render_targets().clear().last_surface().source() != Some(source)
    {
        return Err(MaxwellLoweringError::TriggerStateMismatch);
    }
    let mut draw_attachments = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => None,
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
        | MaxwellThreeDOperationTrigger::DrawIndexBuffer { .. } => {
            Some(select_draw_attachments(state, resources)?)
        }
    };
    if trigger.is_draw() {
        let attachments = draw_attachments
            .as_mut()
            .ok_or(MaxwellLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
        if attachments.colors.len() > 1
            && state.render_targets().separate_fragment_data().value()
                == Some(&MaxwellThreeDSeparateFragmentData::Disabled)
        {
            return Err(MaxwellLoweringError::UnsupportedReplicatedColorTargetOutputSemantics);
        }
        if (!attachments.colors.is_empty() || attachments.depth_stencil.is_some())
            && let Some(value) = state
                .render_targets()
                .render_target_index_offset()
                .value()
                .copied()
                .filter(|value| value.enabled())
        {
            return Err(MaxwellLoweringError::UnsupportedRenderTargetIndexOffsetSemantics(value));
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
                            .has_enabled_stage(MaxwellShaderStage::Geometry),
                    )
                })
        {
            return Err(MaxwellLoweringError::UnsupportedRenderTargetLayerSemantics(
                value,
            ));
        }
        validate_draw_iterated_blend_state(state, attachments)?;
        attachments.color_outputs = color::draw_color_outputs(state, resources, attachments)?;
        validate_draw_logic_op_state(state, attachments)?;
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
    if trigger.is_draw() {
        validate_draw_stencil_state(state)?;
    }
    let shaders = if trigger.is_draw() {
        Some(translated_shaders.ok_or(MaxwellLoweringError::ShaderTranslationRequired)?)
    } else {
        None
    };
    if let Some(shaders) = shaders {
        multisample::validate_shader_mask(state, shaders, cache)?;
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
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
        | MaxwellThreeDOperationTrigger::DrawIndexBuffer { .. } => {
            let attachments = draw_attachments
                .as_ref()
                .ok_or(MaxwellLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
            let lowered = lower_draw(
                state,
                resources,
                &resource_bindings,
                &sampler_bindings,
                shaders.ok_or(MaxwellLoweringError::ShaderTranslationRequired)?,
                attachments,
                draw_arguments(state, trigger)?,
                tessellation,
                raster_state.expect("draw consumes raster state"),
                cache,
                &mut creations,
            )?;
            record_draw_color_materializations(resources, attachments, cache)?;
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
    cache: &mut MaxwellLoweringCache,
    submission: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    creations: Vec<BackendResourceCreateInfo>,
    mut invalidations: Vec<ResourceDependency>,
    commands: impl IntoIterator<Item = GpuOperation>,
    dirty_images: Arc<[usize]>,
) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
    let copies = std::mem::take(&mut cache.image_alias_copies);
    let operations = sequence_with_transitions(copies.into_iter().chain(commands), cache)?;
    trim_read_only_buffer_views(cache, &operations, &mut invalidations);
    let submission = OperationSubmission::new(submission, predecessors, operations)
        .map_err(MaxwellLoweringError::Command)?;
    cache.revision = cache
        .revision
        .checked_add(1)
        .ok_or(MaxwellLoweringError::ResourceExhausted)?;
    Ok(MaxwellLoweredWork {
        creations: creations.into_boxed_slice(),
        invalidations: invalidations.into_boxed_slice(),
        submission,
        dirty_images,
    })
}

fn validate_shader_memory_configuration(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<(), MaxwellLoweringError> {
    for shader in shaders.shaders() {
        let Some(required) = shader.directly_addressable_memory() else {
            continue;
        };
        let configured = state
            .shader_execution()
            .l1_configuration()
            .value()
            .copied()
            .ok_or(MaxwellLoweringError::IncompleteDraw("SET_L1_CONFIGURATION"))?;
        if required != configured {
            return Err(
                MaxwellLoweringError::TranslatedShaderMemoryConfigurationMismatch {
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
) -> Result<(), MaxwellLoweringError> {
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
        return Err(MaxwellLoweringError::VisibleCallLimitExceeded {
            stage: shader.stage(),
            required: shader.maximum_api_visible_calls(),
            limit,
        });
    }
    Ok(())
}

fn validate_draw_iterated_blend_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellLoweringError> {
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
    Err(MaxwellLoweringError::UnsupportedIteratedBlendSemantics {
        value,
        pass_count: controls
            .iterated_blend_pass_count()
            .value()
            .map(|value| value.pass_count()),
    })
}

fn validate_draw_logic_op_state(
    state: &MaxwellThreeDState,
    attachments: &DrawAttachmentSelection,
) -> Result<(), MaxwellLoweringError> {
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
        None => return Err(MaxwellLoweringError::IncompleteLogicOpState),
        Some(_) => {
            return Err(MaxwellLoweringError::ContradictoryState {
                reason: "logic-operation function register has the wrong typed value",
            });
        }
    };
    Err(MaxwellLoweringError::UnsupportedLogicOpSemantics(function))
}

fn draw_alpha_test_state(
    state: &MaxwellThreeDState,
) -> Result<Option<AlphaTest>, MaxwellLoweringError> {
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
            return Err(MaxwellLoweringError::IncompleteAlphaTestState("reference"));
        }
        Some(_) => {
            return Err(MaxwellLoweringError::ContradictoryState {
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
            return Err(MaxwellLoweringError::IncompleteAlphaTestState("function"));
        }
        Some(_) => {
            return Err(MaxwellLoweringError::ContradictoryState {
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

fn validate_direct_line_rasterization_state(
    state: &MaxwellThreeDState,
) -> Result<(), MaxwellLoweringError> {
    let direct_line_primitive =
        state.generated_primitive() == Some(super::threed::state::GeneratedPrimitive::Lines);
    if !direct_line_primitive {
        return Ok(());
    }

    if state.line().anti_aliased_line_enable().value()
        == Some(&MaxwellThreeDAntiAliasedLineEnable::Enabled)
    {
        return raster::smooth_line(state).map(|_| ());
    }
    if state.line().stipple_enable().value() == Some(&true) {
        let parameters = state.line().stipple_parameters().value().copied().ok_or(
            MaxwellLoweringError::IncompleteDraw("SET_LINE_STIPPLE_PARAMETERS"),
        )?;
        return Err(MaxwellLoweringError::UnsupportedLineStippleSemantics {
            factor: parameters.factor(),
            pattern: parameters.pattern(),
        });
    }
    match state.line().aliased_line_width_enable().value() {
        None => Err(MaxwellLoweringError::IncompleteDraw(
            "SET_ALIASED_LINE_WIDTH_ENABLE",
        )),
        Some(MaxwellThreeDAliasedLineWidthEnable::Disabled) => state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::LineWidth)
            .value()
            .is_some()
            .then_some(())
            .ok_or(MaxwellLoweringError::IncompleteDraw("SET_LINE_WIDTH_FLOAT")),
        Some(MaxwellThreeDAliasedLineWidthEnable::Enabled) => {
            Err(MaxwellLoweringError::UnsupportedAliasedLineWidthSemantics)
        }
    }
}

fn validate_compressed_depth_materialization(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
    cache: &MaxwellLoweringCache,
) -> Result<(), MaxwellLoweringError> {
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
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
        | MaxwellThreeDOperationTrigger::DrawIndexBuffer { .. } => {
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
    Err(MaxwellLoweringError::CompressedDepthImportRequired {
        kind: image.guest_layout().pte_kind(),
    })
}

fn depth_clear_fully_initializes(
    state: &MaxwellThreeDState,
    image: &super::threed::MaxwellThreeDResolvedImage,
    depth: bool,
    stencil: bool,
) -> Result<bool, MaxwellLoweringError> {
    if !depth && !stencil {
        return Ok(false);
    }
    if stencil && !clear_stencil_mask_is_full(state)? {
        return Ok(false);
    }
    clear_fully_covers_image(state, image)
}

/// An enabled mask of 0xff still overwrites the entire eight-bit stencil
/// aspect. Use the operation snapshot: deko3d restores SET_STENCIL_MASK via
/// MME shadow replay immediately after issuing the clear.
/// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_3d_base.cpp#L470-L511
fn clear_stencil_mask_is_full(state: &MaxwellThreeDState) -> Result<bool, MaxwellLoweringError> {
    if !state
        .render_targets()
        .clear()
        .surface_control()
        .value()
        .is_some_and(|control| control.respect_stencil_mask())
    {
        return Ok(true);
    }
    match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::FrontStencilWriteMask)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::Mask(mask)) => Ok(*mask == 0xff),
        _ => Err(MaxwellLoweringError::IncompleteClear("SET_STENCIL_MASK")),
    }
}

fn clear_fully_covers_image(
    state: &MaxwellThreeDState,
    image: &super::threed::MaxwellThreeDResolvedImage,
) -> Result<bool, MaxwellLoweringError> {
    let extent = image.description().extent();
    Ok(MaxwellThreeDClearRegions::from_state(state)?
        .for_attachment(extent.width, extent.height)
        .fully_covers(extent.width, extent.height))
}

fn record_clear_materialization(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    cache: &mut MaxwellLoweringCache,
) -> Result<(), MaxwellLoweringError> {
    let surface = state
        .render_targets()
        .clear()
        .last_surface()
        .value()
        .copied()
        .ok_or(MaxwellLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
    if surface.color_mask() != 0 {
        let index = resource_index(
            resources,
            MaxwellThreeDResourceRole::ColorTarget(surface.color_target()),
        )?;
        let image = resolved_image(resources, index)?;
        record_image_write(image, cache);
        // Track the same storage requirement consumed by validation and
        // resource creation. MSAA sample storage needs a resident image even
        // when compression of subsequent writes is disabled or unspecified.
        if image.guest_layout().requires_materialization() {
            record_color_materialization(image, cache);
        }
    }
    if surface.depth() || surface.stencil() {
        let index = resource_index(resources, MaxwellThreeDResourceRole::DepthStencilTarget)?;
        let position = cache
            .views
            .iter()
            .position(|record| record.key.matches_resource(&resources.resources()[index]))
            .ok_or(MaxwellLoweringError::InvalidResolvedView {
                role: MaxwellThreeDResourceRole::DepthStencilTarget,
            })?;
        let materialization = cache.views[position].materialization;
        if let ViewMaterialization::CompressedDepthStencil { depth, stencil } = materialization {
            if surface.depth() {
                cache.views[position].uninitialized_depth_stencil_regions[0].clear();
            }
            if surface.stencil() {
                cache.views[position].uninitialized_depth_stencil_regions[1].clear();
            }
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
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
    cache: &mut MaxwellLoweringCache,
) -> Result<(), MaxwellLoweringError> {
    for target in attachments.color_targets() {
        let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
        let image = resolved_image(resources, index)?;
        record_image_write(image, cache);
        if image.guest_layout().requires_materialization() {
            record_color_materialization(image, cache);
        }
    }
    Ok(())
}

fn record_image_write(
    image: &super::threed::MaxwellThreeDResolvedImage,
    cache: &mut MaxwellLoweringCache,
) {
    let revision = cache.revision.saturating_add(1);
    if let Some(record) = cache
        .views
        .iter_mut()
        .find(|record| record.key.same_domain_as_image(image))
    {
        record.write_revision = revision;
    }
}

fn record_color_materialization(
    image: &super::threed::MaxwellThreeDResolvedImage,
    cache: &mut MaxwellLoweringCache,
) {
    if let Some(record) = cache
        .views
        .iter_mut()
        .find(|record| record.key.same_domain_as_image(image))
    {
        record.uninitialized_color_regions.clear();
    }
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
    cache: &MaxwellLoweringCache,
) -> Result<(), MaxwellLoweringError> {
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
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
        | MaxwellThreeDOperationTrigger::DrawIndexBuffer { .. } => {
            for target in draw_attachments
                .ok_or(MaxwellLoweringError::IncompleteDraw("SET_CT_SELECT"))?
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
    cache: &MaxwellLoweringCache,
    target: u8,
    complete_clear: bool,
) -> Result<(), MaxwellLoweringError> {
    if state.render_targets().color()[target as usize]
        .compression()
        .value()
        != Some(&MaxwellThreeDColorCompressionMode::Enabled)
        && state
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::SampleMode)
            .value()
            != Some(&MaxwellThreeDFixedFunctionValue::SampleMode(
                super::threed::MaxwellThreeDSampleMode::Samples2x2,
            ))
    {
        return Ok(());
    }
    let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
    let image = resolved_image(resources, index)?;
    if !image.guest_layout().requires_materialization() {
        return Ok(());
    }
    if cache
        .color_materializations
        .iter()
        .any(|previous| previous.remains_materialized_for(image))
        || (complete_clear && clear_fully_covers_image(state, image)?)
    {
        return Ok(());
    }
    Err(MaxwellLoweringError::CompressedColorImportRequired { target })
}

fn operation_resource_indices(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    trigger: MaxwellThreeDOperationTrigger,
    draw_attachments: Option<&DrawAttachmentSelection>,
    shaders: Option<&MaxwellThreeDTranslatedShaders>,
) -> Result<Vec<usize>, MaxwellLoweringError> {
    let mut indices = match trigger {
        MaxwellThreeDOperationTrigger::ClearSurface { .. } => {
            let surface = state
                .render_targets()
                .clear()
                .last_surface()
                .value()
                .copied()
                .ok_or(MaxwellLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
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
        MaxwellThreeDOperationTrigger::DrawVertexArray { .. }
        | MaxwellThreeDOperationTrigger::DrawIndexBuffer { .. } => draw_resource_indices(
            state,
            resources,
            draw_attachments.ok_or(MaxwellLoweringError::IncompleteDraw("SET_CT_SELECT"))?,
            shaders.ok_or(MaxwellLoweringError::ShaderTranslationRequired)?,
        )?,
    };
    if trigger.is_indexed() {
        indices.push(resource_index(
            resources,
            MaxwellThreeDResourceRole::IndexBuffer,
        )?);
    }
    indices.sort_unstable();
    indices.dedup();
    Ok(indices)
}

fn draw_resource_indices(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<Vec<usize>, MaxwellLoweringError> {
    let mut indices = attachments.attachment_indices();
    for stream in consumed_vertex_streams(state) {
        indices.push(resource_index(
            resources,
            MaxwellThreeDResourceRole::VertexStream(stream),
        )?);
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
) -> Result<DrawAttachmentSelection, MaxwellLoweringError> {
    let selection = state
        .render_targets()
        .color_target_selection()
        .value()
        .ok_or(MaxwellLoweringError::IncompleteDraw("SET_CT_SELECT"))?;
    for (slot, target) in selection.active_targets().iter().copied().enumerate() {
        if selection.active_targets()[..slot].contains(&target) {
            return Err(MaxwellLoweringError::DuplicateColorTargetRoute { target });
        }
    }
    let mut colors = Vec::with_capacity(selection.target_count() as usize);
    for (slot, target) in selection.active_targets().iter().copied().enumerate() {
        let configured = &state.render_targets().color()[target as usize];
        match configured.readiness(true) {
            super::threed::MaxwellThreeDAttachmentReadiness::Unprogrammed => {
                return Err(MaxwellLoweringError::ColorTargetRouteUnprogrammed {
                    slot: slot as u8,
                    target,
                });
            }
            super::threed::MaxwellThreeDAttachmentReadiness::Disabled => {
                return Err(MaxwellLoweringError::ColorTargetRouteDisabled {
                    slot: slot as u8,
                    target,
                });
            }
            super::threed::MaxwellThreeDAttachmentReadiness::Ready => {}
            _ => {
                return Err(MaxwellLoweringError::ColorTargetRouteIncomplete {
                    slot: slot as u8,
                    target,
                });
            }
        }
        let index = resource_index(resources, MaxwellThreeDResourceRole::ColorTarget(target))?;
        let image = resolved_image(resources, index)?;
        if image.description().kind() != nixe_gpu::ImageKind::Color {
            return Err(MaxwellLoweringError::ResolvedResourceKindMismatch);
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
            return Err(MaxwellLoweringError::ResolvedResourceKindMismatch);
        }
    }
    Ok(DrawAttachmentSelection {
        colors,
        color_outputs: [nixe_gpu::ColorOutputState::REPLACE; 8],
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
    // SET_ZT_SELECT.TARGET_COUNT=0 unbinds Z independently of the test enables.
    // Only explicit absence suppresses consumption; an unknown selector must
    // not hide a missing descriptor. ClearSurface has its own resource roles.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2748-L2749
    if state.render_targets().depth_target_count().value()
        == Some(&super::threed::MaxwellThreeDDepthTargetCount::None)
    {
        return (Some(false), Some(false));
    }
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

fn draw_depth_state(state: &MaxwellThreeDState) -> Result<DepthState, MaxwellLoweringError> {
    let register = |register| state.fixed_function().register(register).value();
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(test_enabled)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable)
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_DEPTH_TEST"));
    };
    if !test_enabled {
        return Ok(DepthState::DISABLED);
    }
    let Some(MaxwellThreeDFixedFunctionValue::Boolean(write_enabled)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthWriteEnable)
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_DEPTH_WRITE"));
    };
    let Some(MaxwellThreeDFixedFunctionValue::Compare(compare)) =
        register(MaxwellThreeDFixedFunctionRegister::DepthCompare)
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_DEPTH_FUNC"));
    };
    Ok(DepthState::new(
        true,
        *write_enabled,
        neutral_depth_compare(*compare),
    ))
}

fn validate_draw_stencil_state(state: &MaxwellThreeDState) -> Result<(), MaxwellLoweringError> {
    match draw_depth_stencil_enable_state(state).1 {
        Some(true) => {
            let two_sided = state
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::TwoSidedStencilTestEnable)
                .value()
                == Some(&MaxwellThreeDFixedFunctionValue::Boolean(true));
            Err(MaxwellLoweringError::UnsupportedStencilTestSemantics { two_sided })
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

fn draw_scissor(
    state: &MaxwellThreeDState,
    resources: &MaxwellThreeDResolvedResources,
    attachments: &DrawAttachmentSelection,
) -> Result<nixe_gpu::ScissorRect, MaxwellLoweringError> {
    let (mut width, mut height) = (u32::MAX, u32::MAX);
    for index in attachments.attachment_indices() {
        let extent = resolved_image(resources, index)?.description().extent();
        width = width.min(extent.width);
        height = height.min(extent.height);
    }
    draw_scissor_region(state, width, height)
}

fn draw_scissor_region(
    state: &MaxwellThreeDState,
    width: u32,
    height: u32,
) -> Result<nixe_gpu::ScissorRect, MaxwellLoweringError> {
    // Surface clip uses origin + extent; scissor uses half-open min/max.
    // Both restrict window-space fragments, preserving vertex execution and
    // shader-visible coordinates even when their intersection is empty.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1386-L1392
    // https://docs.rs/wgpu/30.0.1/wgpu/struct.RenderPass.html#method.set_scissor_rect
    let mut region = MaxwellThreeDRasterRegion::attachment(width, height);
    let horizontal = state.fixed_function().surface_clip_horizontal().value();
    let vertical = state.fixed_function().surface_clip_vertical().value();
    match (horizontal, vertical) {
        (Some(horizontal), Some(vertical)) => {
            region.min_x = u32::from(horizontal.origin()).min(width);
            region.max_x =
                (u32::from(horizontal.origin()) + u32::from(horizontal.extent())).min(width);
            region.min_y = u32::from(vertical.origin()).min(height);
            region.max_y =
                (u32::from(vertical.origin()) + u32::from(vertical.extent())).min(height);
        }
        (None, None) => {}
        _ => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_SURFACE_CLIP_HORIZONTAL/VERTICAL",
            ));
        }
    }
    let scissor = &state.fixed_function().scissor()[0];
    if scissor.enable().value() == Some(&true) {
        let mut scissor_y =
            scissor
                .vertical()
                .value()
                .copied()
                .ok_or(MaxwellLoweringError::IncompleteDraw(
                    "SET_SCISSOR_VERTICAL(0)",
                ))?;
        if lower_left_window_origin(state) {
            let height = window_origin_height(state)?;
            scissor_y = super::threed::MaxwellThreeDRectangle {
                min: height.saturating_sub(scissor_y.max),
                max: height.saturating_sub(scissor_y.min),
            };
        }
        region.intersect(
            scissor
                .horizontal()
                .value()
                .copied()
                .ok_or(MaxwellLoweringError::IncompleteDraw(
                    "SET_SCISSOR_HORIZONTAL(0)",
                ))?,
            scissor_y,
        );
    }
    Ok(nixe_gpu::ScissorRect {
        x: region.min_x.min(width),
        y: region.min_y.min(height),
        width: region.max_x.saturating_sub(region.min_x),
        height: region.max_y.saturating_sub(region.min_y),
    })
}

fn lower_left_window_origin(state: &MaxwellThreeDState) -> bool {
    matches!(state.fixed_function().register(MaxwellThreeDFixedFunctionRegister::WindowOrigin).value(),
        Some(MaxwellThreeDFixedFunctionValue::Mask(value)) if value & 1 != 0)
}

fn window_origin_height(state: &MaxwellThreeDState) -> Result<u16, MaxwellLoweringError> {
    state
        .fixed_function()
        .surface_clip_vertical()
        .value()
        .map(|axis| axis.extent())
        .ok_or(MaxwellLoweringError::IncompleteDraw(
            "SET_SURFACE_CLIP_VERTICAL",
        ))
}

fn draw_viewport_transform(
    state: &MaxwellThreeDState,
) -> Result<Option<ViewportTransform>, MaxwellLoweringError> {
    let enabled = state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
        .value()
        == Some(&MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(
            MaxwellThreeDViewportScaleOffsetEnable::Enabled,
        ));
    // The current draw contract selects viewport zero. Shader-selected or
    // replicated viewports require a distinct draw contract.
    let viewport = &state.fixed_function().viewport()[0];
    if let Some(&precision) = viewport.snap_grid_precision().value()
        && precision != [0, 0]
    {
        return Err(MaxwellLoweringError::UnsupportedViewportSnapGridPrecision {
            viewport: 0,
            precision,
        });
    }
    let swizzle = viewport.coordinate_swizzle().value().copied();
    let reflect_y = if let Some(swizzle) = swizzle {
        use MaxwellThreeDViewportSwizzleComponent::{NegativeY, PositiveW, PositiveX, PositiveZ};
        if swizzle.is_identity() {
            false
        } else if enabled && swizzle.components() == [PositiveX, NegativeY, PositiveZ, PositiveW] {
            true
        } else {
            return Err(
                MaxwellLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                    viewport: 0,
                    swizzle,
                },
            );
        }
    } else {
        false
    };
    if !enabled {
        if lower_left_window_origin(state) {
            return Err(MaxwellLoweringError::UnsupportedWindowOrigin(1));
        }
        return Ok(None);
    }
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
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_SCALE_X/Y/Z(0)",
        ));
    };
    let [Some(offset_x), Some(offset_y), Some(offset_z)] = offset else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_OFFSET_X/Y/Z(0)",
        ));
    };
    let Some(clip_min_z) = viewport
        .clip_min_z()
        .value()
        .copied()
        .map(|value| f32::from_bits(value.get()))
    else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_CLIP_MIN_Z(0)",
        ));
    };
    let Some(clip_max_z) = viewport
        .clip_max_z()
        .value()
        .copied()
        .map(|value| f32::from_bits(value.get()))
    else {
        return Err(MaxwellLoweringError::IncompleteDraw(
            "SET_VIEWPORT_CLIP_MAX_Z(0)",
        ));
    };
    // NV_viewport_swizzle acts before clipping and perspective division.
    // A Y-only reflection preserves the symmetric -w..w clip volume and w;
    // its exact window transform is therefore (y/w)*(-scale_y)+offset_y.
    // Fold the sign into the existing neutral transform; the backend applies
    // it at vertex output, preserving varyings, depth, and polygon facing.
    // https://registry.khronos.org/OpenGL/extensions/NV/NV_viewport_swizzle.txt
    let negative_one_to_one = match state
        .fixed_function()
        .register(MaxwellThreeDFixedFunctionRegister::ViewportClipControl)
        .value()
    {
        Some(MaxwellThreeDFixedFunctionValue::ClipControl(control)) => control.raw() & 1 == 0,
        _ => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_VIEWPORT_CLIP_CONTROL",
            ));
        }
    };
    // ClipMin/MaxZ are post-transform pixel bounds, not viewport endpoints.
    // Unbounded values are legal. The affine range comes from the configured
    // clip volume and ScaleZ/OffsetZ, without guessing the guest API's mode.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3372-L3399
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L98-L110
    let depth_range = [
        if negative_one_to_one {
            offset_z - scale_z
        } else {
            offset_z
        },
        offset_z + scale_z,
    ];
    if clip_min_z.is_nan()
        || clip_max_z.is_nan()
        || clip_min_z > depth_range[0].min(depth_range[1])
        || clip_max_z < depth_range[0].max(depth_range[1])
    {
        return Err(MaxwellLoweringError::UnsupportedViewportPixelDepthBounds);
    }
    let mut effective_scale_y = if reflect_y { -scale_y } else { scale_y };
    let mut effective_offset_y = offset_y;
    if lower_left_window_origin(state) {
        // Convert bottom-left window coordinates to the neutral top-left
        // framebuffer convention: y_host = surface_height - y_guest.
        // Scissor bounds use the same origin, independently of FLIP_Y facing.
        // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L128-L134
        effective_scale_y = -effective_scale_y;
        effective_offset_y = f32::from(window_origin_height(state)?) - offset_y;
    }
    ViewportTransform::new(
        [scale_x, effective_scale_y, scale_z],
        [offset_x, effective_offset_y, offset_z],
        depth_range,
    )
    .map(|transform| transform.with_negative_one_to_one_depth(negative_one_to_one))
    .map(Some)
    .map_err(MaxwellLoweringError::Command)
}

fn prepare_resources(
    resources: &MaxwellThreeDResolvedResources,
    indices: &[usize],
    cache: &mut MaxwellLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
    invalidations: &mut Vec<ResourceDependency>,
) -> Result<Vec<Option<ResourceDependency>>, MaxwellLoweringError> {
    let mut result = vec![None; resources.resources().len()];
    for index in indices {
        let resource = resources
            .resources()
            .get(*index)
            .ok_or(MaxwellLoweringError::ResourceExhausted)?;
        if let MaxwellThreeDResolvedResource::Buffer(value) = resource {
            result[*index] = Some(buffer::prepare_buffer(
                value.description(),
                value.allocation_description(),
                value.view().backing().clone(),
                value.shared_mappings(),
                cache,
                creations,
                invalidations,
                result.iter().flatten().copied(),
            )?);
            continue;
        }
        if let MaxwellThreeDResolvedResource::Image(image) = resource
            && matches!(image.role(), MaxwellThreeDResourceRole::SampledImage { .. })
            && image.guest_layout().requires_materialization()
        {
            // Compressed guest bytes are opaque. Sampling is legal only while
            // the initialized representation is still resident and no CPU write
            // has invalidated it. Never create a blank texture or import bytes.
            let resident = cache
                .views
                .iter()
                .find(|record| record.remains_current_for_image(image));
            if let Some(record) = resident
                && (sampled_alias::copy_is_current(record, cache)
                    || record.materialization == ViewMaterialization::Direct
                    || (record.materialization == ViewMaterialization::CompressedColor
                        && cache
                            .color_materializations
                            .iter()
                            .any(|previous| previous.remains_materialized_for(image))))
            {
                // Direct residents have canonical initialization from their
                // producer; opaque residents require the recorded clear.
                result[*index] = Some(record.dependency);
                continue;
            }
            if let Some(dependency) = sampled_alias::prepare(image, cache, creations)? {
                result[*index] = Some(dependency);
                continue;
            }
            return Err(MaxwellLoweringError::CompressedSampledImageImportRequired {
                role: image.role(),
                kind: image.guest_layout().pte_kind(),
            });
        }
        let image = resolved_image(resources, *index)?;
        let allocation = image.view().bindings()[0].backing().allocation();
        let allocation_description = image.allocation_description();
        match cache.allocations.iter().find(|(id, _)| *id == allocation) {
            Some((_, current)) if *current != allocation_description => {
                return Err(MaxwellLoweringError::AllocationDescriptionChanged { allocation });
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

        if let Some(record) = cache.views.iter().find(|record| {
            record.key.matches_resource(resource)
                && (!image.guest_layout().requires_materialization()
                    || record.remains_current_for_image(image))
        }) {
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
        retire_overlapping_views(&key, result.iter().flatten().copied(), cache, invalidations);
        let materialization = if !image.guest_layout().requires_materialization() {
            ViewMaterialization::Direct
        } else if image.role() == MaxwellThreeDResourceRole::DepthStencilTarget {
            ViewMaterialization::CompressedDepthStencil {
                depth: false,
                stencil: false,
            }
        } else {
            ViewMaterialization::CompressedColor
        };
        let cpu_writes = Some(image.cpu_write_dependency().clone());
        let id = ImageId::new(take_identity(cache)?);
        let bindings = image
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
        let view = ImageView::new(id, image.description(), image.view().swizzle(), bindings)
            .map_err(|_| MaxwellLoweringError::InvalidResolvedView { role: image.role() })?;
        creations.push(BackendResourceCreateInfo::Image {
            id,
            description: image.description(),
            view: image
                .guest_layout()
                .has_direct_canonical_representation()
                .then_some(view),
        });
        let dependency = ResourceDependency::Image(id);
        cache.views.push(ViewRecord {
            key,
            dependency,
            materialization,
            cpu_writes,
            write_revision: 0,
            last_used: 0,
            uninitialized_depth_stencil_regions: if matches!(
                materialization,
                ViewMaterialization::CompressedDepthStencil { .. }
            ) {
                let extent = image.description().extent();
                std::array::from_fn(|_| vec![[0, 0, extent.width, extent.height]])
            } else {
                Default::default()
            },
            uninitialized_color_regions: if materialization == ViewMaterialization::CompressedColor
                && !image.guest_layout().has_direct_canonical_representation()
            {
                let extent = image.description().extent();
                vec![[0, 0, extent.width, extent.height]]
            } else {
                Vec::new()
            },
        });
        result[*index] = Some(dependency);
    }
    Ok(result)
}

fn retire_overlapping_views(
    key: &ViewKey,
    retained: impl Iterator<Item = ResourceDependency> + Clone,
    cache: &mut MaxwellLoweringCache,
    invalidations: &mut Vec<ResourceDependency>,
) {
    let invalidated = cache
        .views
        .iter()
        .filter(|record| {
            record.key.overlaps(key)
                && !retained
                    .clone()
                    .any(|dependency| dependency == record.dependency)
        })
        .map(|record| record.dependency)
        .collect::<Vec<_>>();
    retire_view_dependencies(&invalidated, cache, invalidations);
}

fn retire_view_dependencies(
    invalidated: &[ResourceDependency],
    cache: &mut MaxwellLoweringCache,
    invalidations: &mut Vec<ResourceDependency>,
) {
    if !invalidated.is_empty() {
        cache.prepared_draw = None;
    }
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
    for dependency in invalidated.iter().copied() {
        if !invalidations.contains(&dependency) {
            invalidations.push(dependency);
        }
    }
}

// Dynamic vertex/index/uniform slices are derived read-only host views. Keeping
// every slice of a guest ring buffer forever makes alias checks, transitions,
// and backend residency selection grow with the number of rendered frames.
// This budget bounds that metadata without evicting canonical guest bytes.
const MAX_CACHED_READ_ONLY_BUFFER_VIEWS: usize = 256;

fn trim_read_only_buffer_views(
    cache: &mut MaxwellLoweringCache,
    operations: &[GpuOperation],
    invalidations: &mut Vec<ResourceDependency>,
) {
    let limit = cache
        .resource_cache_limit()
        .min(MAX_CACHED_READ_ONLY_BUFFER_VIEWS);
    if cache.views.len() <= limit {
        return;
    }
    let eligible = |record: &&ViewRecord| {
        matches!(record.key, ViewKey::Buffer { .. }) && record.write_revision == 0
    };
    let count = cache.views.iter().filter(eligible).count();
    if count <= limit {
        return;
    }
    // The ordered backend submission retires these resources only after its
    // completion. Also protect all views used by this delivery, including
    // indirect references in descriptor tables. GPU-written views remain
    // pinned: discarding them would require canonical writeback first.
    let mut protected = operations
        .iter()
        .flat_map(GpuOperation::dependencies)
        .copied()
        .collect::<Vec<_>>();
    for descriptor in &cache.descriptors {
        if protected.contains(&ResourceDependency::DescriptorTable(descriptor.id)) {
            protected.extend_from_slice(&descriptor.dependencies);
        }
    }
    let mut candidates = cache
        .views
        .iter()
        .filter(eligible)
        .filter(|record| !protected.contains(&record.dependency))
        .map(|record| (record.last_used, record.dependency))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(last_used, _)| *last_used);
    let retired = candidates
        .into_iter()
        .take(count - limit)
        .map(|(_, dependency)| dependency)
        .collect::<Vec<_>>();
    retire_view_dependencies(&retired, cache, invalidations);
}

fn binding_at(
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    index: usize,
) -> Result<ResourceDependency, MaxwellLoweringError> {
    bindings.get(index).and_then(|binding| *binding).ok_or(
        MaxwellLoweringError::MissingResolvedResource {
            role: resources
                .resources()
                .get(index)
                .ok_or(MaxwellLoweringError::ResourceExhausted)?
                .role(),
        },
    )
}

fn prepare_samplers(
    resources: &MaxwellThreeDResolvedResources,
    cache: &mut MaxwellLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
    invalidations: &mut Vec<ResourceDependency>,
) -> Result<Vec<(MaxwellThreeDResourceRole, ResourceDependency)>, MaxwellLoweringError> {
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
                MaxwellLoweringError::InvalidResolvedView {
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
) -> Result<ResourceDependency, MaxwellLoweringError> {
    if let MaxwellThreeDResourceRole::Sampler(_) = role {
        return samplers
            .iter()
            .find_map(|(candidate, dependency)| (*candidate == role).then_some(*dependency))
            .ok_or(MaxwellLoweringError::MissingResolvedResource { role });
    }
    let index = resource_index(resources, role)?;
    binding_at(resources, bindings, index)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MaxwellThreeDRasterRegion {
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

impl MaxwellThreeDRasterRegion {
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
        horizontal: super::threed::MaxwellThreeDRectangle,
        vertical: super::threed::MaxwellThreeDRectangle,
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
    clear: Option<(
        super::threed::MaxwellThreeDRectangle,
        super::threed::MaxwellThreeDRectangle,
    )>,
    scissor: Option<(
        super::threed::MaxwellThreeDRectangle,
        super::threed::MaxwellThreeDRectangle,
    )>,
    viewport_clip: Option<(
        super::threed::MaxwellThreeDClipAxis,
        super::threed::MaxwellThreeDClipAxis,
    )>,
}

impl MaxwellThreeDClearRegions {
    fn from_state(state: &MaxwellThreeDState) -> Result<Self, MaxwellLoweringError> {
        let clear = state.render_targets().clear();
        let control = clear.surface_control().value().copied();
        let clear = control
            .is_none_or(|control| control.use_clear_rect())
            .then(|| {
                Ok((
                    clear.horizontal().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("horizontal rectangle"),
                    )?,
                    clear
                        .vertical()
                        .value()
                        .copied()
                        .ok_or(MaxwellLoweringError::IncompleteClear("vertical rectangle"))?,
                ))
            })
            .transpose()?;
        let scissor = control
            .is_some_and(|control| control.use_scissor_zero())
            .then(|| {
                let scissor = &state.fixed_function().scissor()[0];
                Ok((
                    scissor.horizontal().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_SCISSOR_HORIZONTAL(0)"),
                    )?,
                    scissor.vertical().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_SCISSOR_VERTICAL(0)"),
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
                        MaxwellLoweringError::IncompleteClear("SET_VIEWPORT_CLIP_HORIZONTAL(0)"),
                    )?,
                    viewport.clip_vertical().value().copied().ok_or(
                        MaxwellLoweringError::IncompleteClear("SET_VIEWPORT_CLIP_VERTICAL(0)"),
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

    fn for_attachment(self, width: u32, height: u32) -> MaxwellThreeDRasterRegion {
        let mut region = MaxwellThreeDRasterRegion::attachment(width, height);
        for (horizontal, vertical) in [self.clear, self.scissor].into_iter().flatten() {
            region.intersect(horizontal, vertical);
        }
        if let Some((horizontal, vertical)) = self.viewport_clip {
            region.min_x = region.min_x.max(u32::from(horizontal.origin()));
            region.max_x = region
                .max_x
                .min(u32::from(horizontal.origin()) + u32::from(horizontal.extent()));
            region.min_y = region.min_y.max(u32::from(vertical.origin()));
            region.max_y = region
                .max_y
                .min(u32::from(vertical.origin()) + u32::from(vertical.extent()));
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
) -> Result<ImageRegion, MaxwellLoweringError> {
    if array_layer != subresources.base_layer {
        return Err(MaxwellLoweringError::ClearOutsideAttachment);
    }
    let region = regions.for_attachment(attachment_width, attachment_height);
    if region.is_empty() {
        return Err(MaxwellLoweringError::EmptyClearRectangle);
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
) -> Result<(Vec<GpuOperation>, Arc<[usize]>), MaxwellLoweringError> {
    let clear = state.render_targets().clear();
    let surface = clear
        .last_surface()
        .value()
        .copied()
        .ok_or(MaxwellLoweringError::IncompleteClear("CLEAR_SURFACE"))?;
    if surface.color_mask() == 0 && !surface.depth() && !surface.stencil() {
        return Err(MaxwellLoweringError::EmptyClearMask);
    }
    if surface.stencil() && !clear_stencil_mask_is_full(state)? {
        return Err(MaxwellLoweringError::UnsupportedClearStencilMaskSemantics);
    }
    let regions = MaxwellThreeDClearRegions::from_state(state)?;
    let mut operations = Vec::new();
    let mut dirty = Vec::new();
    if surface.color_mask() != 0 {
        if surface.color_mask() != 0xf {
            return Err(MaxwellLoweringError::PartialColorClearUnsupported {
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
                    .ok_or(MaxwellLoweringError::IncompleteClear("color value"))?
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
        .map_err(MaxwellLoweringError::Command)?;
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
                    .ok_or(MaxwellLoweringError::IncompleteClear("depth value"))
            })
            .transpose()?;
        let stencil = surface
            .stencil()
            .then(|| {
                clear
                    .stencil()
                    .value()
                    .copied()
                    .ok_or(MaxwellLoweringError::IncompleteClear("stencil value"))
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
        .map_err(MaxwellLoweringError::Command)?;
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
    arguments: DrawArguments,
    tessellation: Option<nixe_gpu::TessellationState>,
    raster: raster::DrawRasterState,
    cache: &mut MaxwellLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
) -> Result<(Vec<GpuOperation>, Arc<[usize]>), MaxwellLoweringError> {
    validate_shader_stages(state, shaders)?;
    for translated in &shaders.shaders {
        let record = cache
            .shader_translations
            .get(translated.cache_fingerprint)
            .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?;
        #[cfg(debug_assertions)]
        assert_eq!(
            record.id, translated.shader,
            "XXH3-128 collision or inconsistent translated shader identity"
        );
        if record.module.stage() != translated.stage {
            return Err(MaxwellLoweringError::InvalidTranslatedShaders);
        }
        if translated.stage == ShaderStage::Vertex {
            validate_vertex_attribute_skip_masks(state, record.module.ir().ir())?;
        }
        if translated.stage == ShaderStage::Fragment
            && lower_left_window_origin(state)
            && record.module.ir().ir().inputs().iter().any(|input| {
                input.location() == nixe_gpu::ShaderIoLocation::Position && input.component() == 1
            })
        {
            return Err(MaxwellLoweringError::UnsupportedWindowOriginFragmentPosition);
        }
        if let Some(tessellation) = tessellation {
            super::threed::tessellation::validate_default_level_inputs(
                tessellation.control,
                record.module.ir().ir(),
            )?;
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
            .ok_or(MaxwellLoweringError::IncompleteDraw("BEGIN"))?,
    )?;
    let mut vertex_buffers = Vec::new();
    for index in consumed_vertex_streams(state).map(usize::from) {
        let stream = &state.vertex_input().streams()[index];
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
                        attribute.enabled()
                            && usize::from(attribute.stream()) == index
                            && state.vertex_input().attribute_skip_mask(location as u8) != 15
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
            .collect::<Result<Vec<_>, MaxwellLoweringError>>()?;
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
                MaxwellLoweringError::InvalidResolvedView {
                    role: buffer.role(),
                }
            })?,
        };
        let instanced = stream.instanced().value().copied().unwrap_or(false);
        let frequency = stream.frequency().value().copied().unwrap_or(1);
        if instanced && frequency != 1 {
            return Err(MaxwellLoweringError::UnsupportedVertexInstanceDivisor {
                stream: index as u8,
                divisor: frequency,
            });
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
            .map_err(MaxwellLoweringError::Command)?,
        );
    }
    let triangle_rasterization = match state
        .raster()
        .fill_via_triangle()
        .value()
        .copied()
        .unwrap_or(MaxwellThreeDFillViaTriangleMode::Disabled)
    {
        MaxwellThreeDFillViaTriangleMode::Disabled => raster.triangles,
        MaxwellThreeDFillViaTriangleMode::FillBoundingBox => {
            let DrawArguments::NonIndexed {
                first_vertex,
                vertex_count,
                ..
            } = arguments
            else {
                return Err(MaxwellLoweringError::UnsupportedFillRectangleDraw(
                    "indexed rectangle expansion",
                ));
            };
            if topology != PrimitiveTopology::Triangles {
                return Err(MaxwellLoweringError::UnsupportedFillRectangleDraw(
                    "primitive topology is not a triangle list",
                ));
            }
            if !first_vertex.is_multiple_of(3) || !vertex_count.is_multiple_of(3) {
                return Err(MaxwellLoweringError::UnsupportedFillRectangleDraw(
                    "vertex range is not aligned to complete triangles",
                ));
            }
            if vertex_buffers
                .iter()
                .any(|layout| layout.step_mode == VertexStepMode::Vertex)
            {
                return Err(MaxwellLoweringError::UnsupportedFillRectangleDraw(
                    "per-vertex attributes require vertex-pulling expansion",
                ));
            }
            TriangleRasterization::FillRectangle
        }
        MaxwellThreeDFillViaTriangleMode::FillAll => {
            return Err(MaxwellLoweringError::UnsupportedFillViaTriangleSemantics(
                MaxwellThreeDFillViaTriangleMode::FillAll,
            ));
        }
    };
    let attachments = attachment_records(resources, bindings, attachment_selection)?;
    if attachments.is_empty() {
        return Err(MaxwellLoweringError::IncompleteDraw("render target"));
    }
    let mut required_indices =
        draw_resource_indices(state, resources, attachment_selection, shaders)?;
    let index_buffer = if matches!(arguments, DrawArguments::Indexed { .. }) {
        let index = resource_index(resources, MaxwellThreeDResourceRole::IndexBuffer)?;
        required_indices.push(index);
        let buffer = resolved_buffer(resources, index)?;
        Some((
            BufferRegion {
                buffer: buffer_dependency(binding_at(resources, bindings, index)?)?,
                range: BufferRange::new(0, buffer.description().size()).map_err(|_| {
                    MaxwellLoweringError::InvalidResolvedView {
                        role: buffer.role(),
                    }
                })?,
            },
            indexed::index_type(state)?,
        ))
    } else {
        None
    };
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
    .map_err(|_| MaxwellLoweringError::InvalidResourceCreation)?;
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

    let descriptor_bindings = shaders
        .resources
        .iter()
        .map(|resource| {
            Ok(DescriptorTableBinding {
                binding: resource.binding,
                resource: shader_resource_dependency(
                    resources,
                    bindings,
                    sampler_bindings,
                    resource.role,
                )?,
            })
        })
        .collect::<Result<Vec<_>, MaxwellLoweringError>>()?;
    let descriptor_tables = prepare_descriptors(
        descriptor_bindings,
        shaders.resources.iter().map(|r| r.kind).collect(),
        cache,
        creations,
    )?;
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
                    MaxwellLoweringError::InvalidResolvedView {
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
                MaxwellLoweringError::InvalidShaderResourceUse {
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
        index_buffer,
    )
    .map_err(MaxwellLoweringError::Command)?;
    draw = draw.with_triangle_rasterization(triangle_rasterization);
    if state.generated_primitive() == Some(super::threed::state::GeneratedPrimitive::Lines)
        && state.line().anti_aliased_line_enable().value()
            == Some(&MaxwellThreeDAntiAliasedLineEnable::Enabled)
    {
        draw.line_rasterization = Some(raster::smooth_line(state)?);
    }
    draw.front_face = raster.front_face;
    draw.cull_mode = raster.cull_mode;
    draw.scissor = Some(draw_scissor(state, resources, attachment_selection)?);
    draw.tessellation = tessellation;
    draw.color_outputs = attachment_selection.color_outputs;
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
                    .map_err(MaxwellLoweringError::Command)?,
            ),
            [],
            [],
            CapabilityRequirements::none(),
        ),
        GpuOperation::new(
            GpuCommand::Draw(
                DrawOperation::new(Arc::clone(&draw), arguments)
                    .map_err(MaxwellLoweringError::Command)?,
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
        indexed: matches!(arguments, DrawArguments::Indexed { .. }),
        state: state.draw_state_identity(),
        resources: resources.identity(),
        shaders: shaders.identity(),
        operations,
        dirty_images: attachment_selection.attachment_indices().into(),
        sampled_aliases: bindings
            .iter()
            .flatten()
            .copied()
            .filter(|dependency| {
                cache.views.iter().any(|record| {
                    record.dependency == *dependency
                        && matches!(
                            record.materialization,
                            ViewMaterialization::CopiedColor { .. }
                        )
                })
            })
            .collect(),
    };
    let operations = record.operations(arguments)?;
    let dirty = Arc::clone(&record.dirty_images);
    cache.prepared_draw = Some(record);
    Ok((operations.into(), dirty))
}

fn prepare_descriptors(
    bindings: Vec<DescriptorTableBinding>,
    kinds: Vec<DescriptorKind>,
    cache: &mut MaxwellLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
) -> Result<Vec<DescriptorTableId>, MaxwellLoweringError> {
    if bindings.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(record) = cache.descriptors.iter().find(|record| {
        record.kinds.as_ref() == kinds.as_slice()
            && record.bindings.len() == bindings.len()
            && record
                .bindings
                .iter()
                .zip(&record.dependencies)
                .zip(&bindings)
                .all(|((&binding, &resource), actual)| {
                    binding == actual.binding && resource == actual.resource
                })
    }) {
        return Ok(vec![record.id]);
    }
    let id = DescriptorTableId::new(take_identity(cache)?);
    let description = DescriptorTableDescription::new(kinds.clone())
        .map_err(|_| MaxwellLoweringError::InvalidTranslatedShaders)?;
    cache.descriptors.push(DescriptorRecord {
        id,
        kinds: kinds.into(),
        bindings: bindings.iter().map(|b| b.binding).collect(),
        dependencies: bindings.iter().map(|b| b.resource).collect(),
    });
    creations.push(BackendResourceCreateInfo::DescriptorTable {
        id,
        description,
        bindings: bindings.into(),
    });
    Ok(vec![id])
}

fn sequence_with_transitions(
    commands: impl IntoIterator<Item = GpuOperation>,
    cache: &mut MaxwellLoweringCache,
) -> Result<Vec<GpuOperation>, MaxwellLoweringError> {
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
                        .map_err(|_| MaxwellLoweringError::InvalidTransition)?,
                );
            }
        }
        if !transitions.is_empty() {
            result.push(GpuOperation::new(
                GpuCommand::Barrier(
                    BarrierOperation::new(transitions).map_err(MaxwellLoweringError::Command)?,
                ),
                [],
                [],
                CapabilityRequirements::none(),
            ));
        }
        for access in command.accesses() {
            let dependency = access.target().dependency();
            if matches!(dependency, ResourceDependency::Buffer(_))
                && let Some(record) = cache
                    .views
                    .iter_mut()
                    .find(|record| record.dependency == dependency)
            {
                record.last_used = cache.revision.saturating_add(1);
                if access.scope().mode().writes() {
                    record.write_revision = cache.revision.saturating_add(1);
                }
            }
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

fn validate_vertex_attribute_skip_masks(
    state: &MaxwellThreeDState,
    ir: &nixe_gpu::ShaderIr,
) -> Result<(), MaxwellLoweringError> {
    for input in ir.inputs() {
        if let nixe_gpu::ShaderIoLocation::Generic(attribute) = input.location()
            && state.vertex_input().attribute_skip_mask(attribute) & (1 << input.component()) != 0
        {
            // Skipped components receive the DA default, not the fetched vertex
            // value. Reject consumption until that constant-input path is lowered.
            return Err(MaxwellLoweringError::UnsupportedSkippedVertexComponent {
                attribute,
                component: input.component(),
            });
        }
    }
    Ok(())
}

fn validate_shader_stages(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
) -> Result<(), MaxwellLoweringError> {
    let mut expected_count = 0;
    for pipeline in state.shader_bindings().pipeline() {
        if pipeline.enabled().value() != Some(&true) {
            continue;
        }
        let stage = match pipeline
            .stage()
            .value()
            .ok_or(MaxwellLoweringError::IncompleteDraw("shader stage"))?
        {
            MaxwellShaderStage::Vertex => ShaderStage::Vertex,
            MaxwellShaderStage::TessellationInit => ShaderStage::TessellationControl,
            MaxwellShaderStage::Tessellation => ShaderStage::TessellationEvaluation,
            MaxwellShaderStage::Geometry => ShaderStage::Geometry,
            MaxwellShaderStage::Pixel => ShaderStage::Fragment,
            stage @ (MaxwellShaderStage::VertexCullBeforeFetch | MaxwellShaderStage::Compute) => {
                return Err(MaxwellLoweringError::UnsupportedShaderStage(*stage));
            }
        };
        expected_count += 1;
        if !shaders.shaders.iter().any(|shader| shader.stage == stage) {
            return Err(MaxwellLoweringError::TranslatedShaderStageMismatch);
        }
    }
    if expected_count == 0 || expected_count != shaders.shaders.len() {
        return Err(MaxwellLoweringError::TranslatedShaderStageMismatch);
    }
    Ok(())
}

fn shader_pipeline_stages(stage: ShaderStage) -> Result<PipelineStages, MaxwellLoweringError> {
    match stage {
        ShaderStage::Vertex => Ok(PipelineStages::VERTEX_SHADER),
        ShaderStage::TessellationControl => Ok(PipelineStages::TESSELLATION_CONTROL_SHADER),
        ShaderStage::TessellationEvaluation => Ok(PipelineStages::TESSELLATION_EVALUATION_SHADER),
        ShaderStage::Geometry => Ok(PipelineStages::GEOMETRY_SHADER),
        ShaderStage::Fragment => Ok(PipelineStages::FRAGMENT_SHADER),
        ShaderStage::Compute => Err(MaxwellLoweringError::InvalidTranslatedShaders),
    }
}

fn primitive_topology(
    begin: MaxwellThreeDBegin,
) -> Result<PrimitiveTopology, MaxwellLoweringError> {
    if begin.preserve_primitive_id() {
        return Err(MaxwellLoweringError::UnsupportedPrimitiveIdContinuation);
    }
    if begin.split_mode() != 0 {
        return Err(MaxwellLoweringError::UnsupportedPrimitiveSplitMode(
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
        // NVB197_BEGIN_OP_QUADS; keep primitive assembly backend-independent.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
        7 => Ok(PrimitiveTopology::Quads),
        14 => Ok(PrimitiveTopology::Patches),
        topology => Err(MaxwellLoweringError::UnsupportedTopology(topology)),
    }
}

fn neutral_first_instance(base: u32, relative: u32) -> Result<u32, MaxwellLoweringError> {
    base.checked_add(relative)
        .ok_or(MaxwellLoweringError::InstanceIndexOverflow { base, relative })
}

fn draw_arguments(
    state: &MaxwellThreeDState,
    trigger: MaxwellThreeDOperationTrigger,
) -> Result<DrawArguments, MaxwellLoweringError> {
    if let MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count, .. } = trigger {
        return indexed::draw_arguments(state, index_count);
    }
    let MaxwellThreeDOperationTrigger::DrawVertexArray { vertex_count, .. } = trigger else {
        unreachable!("only drawing triggers have draw arguments")
    };
    if vertex_count == 0 {
        return Err(MaxwellLoweringError::EmptyDraw);
    }
    let first_vertex = *state
        .vertex_input()
        .primitive()
        .vertex_array_start()
        .value()
        .ok_or(MaxwellLoweringError::IncompleteDraw("VERTEX_ARRAY_START"))?;
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
    format: super::threed::MaxwellThreeDVertexAttributeFormat,
) -> Result<VertexFormat, MaxwellLoweringError> {
    let widths = format
        .component_widths()
        .ok_or(MaxwellLoweringError::IncompleteDraw(
            "SET_VERTEX_ATTRIBUTE_A",
        ))?;
    let numerical = format
        .numerical_type()
        .ok_or(MaxwellLoweringError::IncompleteDraw(
            "SET_VERTEX_ATTRIBUTE_A",
        ))?;
    if format.swap_red_blue() {
        return Err(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
            attribute,
            component_widths: widths,
            numerical_type: numerical,
            swap_red_blue: true,
        });
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
            let (width, components) =
                scaled_layout().ok_or(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                })?;
            VertexFormat::Uscaled { width, components }
        }
        (_, MaxwellThreeDVertexNumericalType::SignedScaled) => {
            let (width, components) =
                scaled_layout().ok_or(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
                    attribute,
                    component_widths: widths,
                    numerical_type: numerical,
                    swap_red_blue: false,
                })?;
            VertexFormat::Sscaled { width, components }
        }
        _ => {
            return Err(MaxwellLoweringError::UnsupportedVertexAttributeFormat {
                attribute,
                component_widths: widths,
                numerical_type: numerical,
                swap_red_blue: false,
            });
        }
    };
    Ok(vertex)
}

fn attachment_records(
    resources: &MaxwellThreeDResolvedResources,
    bindings: &[Option<ResourceDependency>],
    selection: &DrawAttachmentSelection,
) -> Result<Vec<RenderAttachment>, MaxwellLoweringError> {
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
) -> Result<(), MaxwellLoweringError> {
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
            return Err(MaxwellLoweringError::AliasedDrawResources { first, second });
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
    image: &super::threed::MaxwellThreeDResolvedImage,
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
        cpu_writes: Some(image.cpu_write_dependency().clone()),
    }
}

fn resource_index(
    resources: &MaxwellThreeDResolvedResources,
    role: MaxwellThreeDResourceRole,
) -> Result<usize, MaxwellLoweringError> {
    resources
        .resources()
        .iter()
        .position(|resource| resource.role() == role)
        .ok_or(MaxwellLoweringError::MissingResolvedResource { role })
}

fn resolved_buffer(
    resources: &MaxwellThreeDResolvedResources,
    index: usize,
) -> Result<&super::threed::MaxwellThreeDResolvedBuffer, MaxwellLoweringError> {
    match &resources.resources()[index] {
        MaxwellThreeDResolvedResource::Buffer(value) => Ok(value),
        _ => Err(MaxwellLoweringError::ResolvedResourceKindMismatch),
    }
}

fn resolved_image(
    resources: &MaxwellThreeDResolvedResources,
    index: usize,
) -> Result<&super::threed::MaxwellThreeDResolvedImage, MaxwellLoweringError> {
    match &resources.resources()[index] {
        MaxwellThreeDResolvedResource::Image(value) => Ok(value),
        _ => Err(MaxwellLoweringError::ResolvedResourceKindMismatch),
    }
}

fn buffer_dependency(dependency: ResourceDependency) -> Result<BufferId, MaxwellLoweringError> {
    match dependency {
        ResourceDependency::Buffer(id) => Ok(id),
        _ => Err(MaxwellLoweringError::ResolvedResourceKindMismatch),
    }
}

fn image_dependency(dependency: ResourceDependency) -> Result<ImageId, MaxwellLoweringError> {
    match dependency {
        ResourceDependency::Image(id) => Ok(id),
        _ => Err(MaxwellLoweringError::ResolvedResourceKindMismatch),
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

fn take_identity(cache: &mut MaxwellLoweringCache) -> Result<u64, MaxwellLoweringError> {
    let value = cache.next_identity;
    cache.next_identity = value
        .checked_add(1)
        .ok_or(MaxwellLoweringError::ResourceExhausted)?;
    Ok(value)
}

/// Typed failure before any cache or backend effect is published.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellLoweringError {
    ComputeLaunch(crate::MaxwellComputeLaunchError),
    ComputeShader(MaxwellShaderTranslationError),
    BufferBacking(String),
    UnsupportedMultisampleState(&'static str),
    BlitSourceNotResident,
    UnsupportedPolygonRasterization(&'static str),
    UnsupportedWindowOrigin(u32),
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
    UnsupportedPostPsInitialCoverageSemantics,
    UnsupportedViewportPixelDepthBounds,
    UnsupportedWindowOriginFragmentPosition,
    UnsupportedPostZPixelShaderImaskSemantics,
    UnsupportedPixelShaderInterlockSemantics(MaxwellThreeDPixelShaderInterlockControl),
    UnsupportedGlobalBaseVertexIndex(u32),
    UnsupportedVertexIdBase(u32),
    UnsupportedIndexFormat(super::threed::MaxwellThreeDIndexElementSize),
    UnsupportedIndexedDraw(&'static str),
    UnsupportedCsaaSemantics,
    UnsupportedAntiAliasAlphaControl {
        alpha_to_coverage: bool,
        alpha_to_one: bool,
    },
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
    UnsupportedViewportSnapGridPrecision {
        viewport: u8,
        precision: [u8; 2],
    },
    UnsupportedViewportCoordinateSwizzleSemantics {
        viewport: u8,
        swizzle: MaxwellThreeDViewportCoordinateSwizzle,
    },
    UnsupportedWindowClipSemantics,
    UnsupportedClipIdTestSemantics,
    UnsupportedStencilTestSemantics {
        two_sided: bool,
    },
    UnsupportedClearStencilMaskSemantics,
    UnsupportedAliasedLineWidthSemantics,
    UnsupportedLineStippleSemantics {
        factor: u8,
        pattern: u16,
    },
    UnsupportedPolygonClipGeneratedEdgeSemantics,
    UnsupportedSkippedVertexComponent {
        attribute: u8,
        component: u8,
    },
    UnsupportedVertexAttributeFormat {
        attribute: u8,
        component_widths: super::threed::MaxwellThreeDVertexComponentWidths,
        numerical_type: super::threed::MaxwellThreeDVertexNumericalType,
        swap_red_blue: bool,
    },
    UnsupportedVertexInstanceDivisor {
        stream: u8,
        divisor: u32,
    },
    InvalidPatchSize(MaxwellThreeDPatchSize),
    TessellationStageTopology,
    TessellationMode {
        value: super::threed::MaxwellThreeDTessellationMode,
        source: Option<MaxwellMethodSource>,
        reason: super::threed::MaxwellTessellationModeError,
    },
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
    UnsupportedPixelShaderSaturateSemantics {
        output: u8,
        range: MaxwellThreeDPixelShaderClampRange,
    },
    UnsupportedBlendFactor {
        target: Option<u8>,
        value: u32,
    },
    UnsupportedBlendFormat {
        target: u8,
        format: nixe_gpu::ImageFormat,
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
    IncompleteAlphaTestState(&'static str),
    CompressedDepthImportRequired {
        kind: u8,
    },
    CompressedColorImportRequired {
        target: u8,
    },
    CompressedSampledImageImportRequired {
        role: MaxwellThreeDResourceRole,
        kind: u8,
    },
    ShaderTranslationRequired,
    InvalidTranslatedShaders,
    TranslatedShaderStageMismatch,
    TranslatedShaderMemoryConfigurationMismatch {
        stage: ShaderStage,
        configured: MaxwellThreeDDirectlyAddressableMemory,
        required: MaxwellThreeDDirectlyAddressableMemory,
    },
    UnsupportedShaderStage(MaxwellShaderStage),
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

impl Display for MaxwellLoweringError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComputeLaunch(error) => Display::fmt(error, formatter),
            Self::ComputeShader(error) => Display::fmt(error, formatter),
            Self::BufferBacking(reason) => write!(formatter, "Maxwell buffer backing cannot be lowered: {reason}"),
            Self::BlitSourceNotResident => formatter.write_str("Maxwell color blit requires a current, fully initialized resident source"),
            Self::UnsupportedMultisampleState(reason) => write!(formatter, "Maxwell multisample state has no neutral lowering: {reason}"),
            Self::UnsupportedPolygonRasterization(reason) => write!(formatter, "MAXWELL_B polygon rasterization is unsupported: {reason}"),
            Self::UnsupportedWindowOrigin(value) => write!(formatter,
                "MAXWELL_B window origin/flip has no neutral coordinate lowering: value=0x{value:x}"),
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
            Self::UnsupportedPostPsInitialCoverageSemantics => formatter.write_str("MAXWELL_B pre-shader initial coverage for post-shader tests is not represented by the neutral pipeline"),
            Self::UnsupportedViewportPixelDepthBounds => formatter.write_str("MAXWELL_B pixel depth bounds narrow the affine viewport range and require explicit clip/clamp lowering"),
            Self::UnsupportedWindowOriginFragmentPosition => formatter.write_str("MAXWELL_B lower-left fragment position Y requires shader coordinate lowering"),
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
            Self::UnsupportedVertexIdBase(value) => write!(
                formatter,
                "MAXWELL_B SET_VERTEX_ID_BASE requires an independent shader vertex-ID adjustment: value=0x{value:08x}"
            ),
            Self::UnsupportedIndexFormat(format) => write!(formatter, "Maxwell index format has no neutral representation: {format:?}"),
            Self::UnsupportedIndexedDraw(reason) => write!(formatter, "Maxwell indexed draw is not represented: {reason}"),
            Self::UnsupportedCsaaSemantics => formatter.write_str(
                "MAXWELL_B enabled CSAA has no verified coverage sampling, resolve, capability, or coherency semantics",
            ),
            Self::UnsupportedAntiAliasAlphaControl { alpha_to_coverage, alpha_to_one } => write!(
                formatter,
                "MAXWELL_B alpha-to-coverage/dithering or alpha-to-one is not represented by the neutral pipeline: alpha-to-coverage={alpha_to_coverage} alpha-to-one={alpha_to_one}"
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
            Self::UnsupportedViewportSnapGridPrecision { viewport, precision } => write!(formatter, "MAXWELL_B viewport {viewport} increased snap-grid precision is unsupported: X={} Y={}", precision[0], precision[1]),
            Self::UnsupportedViewportCoordinateSwizzleSemantics { viewport, swizzle } => write!(
                formatter,
                "MAXWELL_B viewport coordinate swizzle is not represented by the neutral pipeline contract: viewport={viewport} components={:?}",
                swizzle.components()
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
                "MAXWELL_B partial stencil write mask on clear has no neutral backend representation",
            ),
            Self::UnsupportedAliasedLineWidthSemantics => formatter.write_str(
                "MAXWELL_B aliased line-width selection has no represented width register or host rasterization semantics",
            ),
            Self::UnsupportedLineStippleSemantics { factor, pattern } => write!(
                formatter,
                "MAXWELL_B line stippling has no neutral backend representation: factor={factor} pattern=0x{pattern:04x}"
            ),
            Self::UnsupportedPolygonClipGeneratedEdgeSemantics => formatter.write_str(
                "MAXWELL_B suppression of polygon-clip-generated edges has no neutral backend representation",
            ),
            Self::UnsupportedSkippedVertexComponent { attribute, component } => write!(formatter,
                "vertex shader consumes a skipped DA attribute component: attribute={attribute} component={component}; DA default input lowering is not implemented"),
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
            Self::TessellationStageTopology => formatter.write_str("MAXWELL_B tessellation shaders require patch topology"),
            Self::TessellationMode { value, source, reason } => write!(formatter, "MAXWELL_B tessellation mode cannot be consumed: value={:#x} reason={reason:?} source={source:?}", value.raw()),
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
            Self::UnsupportedPixelShaderSaturateSemantics { output, range } => write!(
                formatter,
                "MAXWELL_B pixel-shader output saturation is not represented by shader or neutral backend lowering: output={output} range={range:?}"
            ),
            Self::UnsupportedBlendFactor { target, value } => write!(formatter,
                "MAXWELL_B blend factor requires unsupported coupled/constant/dual-source semantics: target={target:?} value=0x{value:04x}"),
            Self::UnsupportedBlendFormat { target, format } => write!(formatter,
                "MAXWELL_B blending currently requires RGBA8/BGRA8 UNORM or sRGB: target={target} format={format:?}"),
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
            Self::IncompleteAlphaTestState(field) => write!(
                formatter,
                "MAXWELL_B enabled alpha testing requires SET_ALPHA_{field}"
            ),
            Self::CompressedDepthImportRequired { kind } => write!(
                formatter,
                "Maxwell compressed depth contents require materialization before use: kind=0x{kind:02x}"
            ),
            Self::CompressedSampledImageImportRequired { role, kind } => write!(
                formatter,
                "Maxwell compressed sampled image has no current materialized resident image: role={role:?} kind={kind:#04x}"
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
                formatter.write_str("GPU lowering exhausted host resources or identities")
            }
        }
    }
}

impl std::error::Error for MaxwellLoweringError {}

#[cfg(test)]
mod tests {
    #[test]
    fn draw_fragment_bounds_compose_surface_origin_extent_and_scissor() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        use nixe_gpu::ScissorRect;
        let mut channel = three_d_channel();
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 0,
                y: 0,
                width: 64,
                height: 32
            }
        );
        program_three_d(&mut channel, 0x0ff4, (40 << 16) | 5);
        assert!(matches!(
            super::draw_scissor_region(channel.three_d(), 64, 32),
            Err(MaxwellLoweringError::IncompleteDraw(_))
        ));
        program_three_d(&mut channel, 0x0ff8, (20 << 16) | 7);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 5,
                y: 7,
                width: 40,
                height: 20
            }
        );
        program_three_d(&mut channel, 0x0e00, 1);
        assert!(matches!(
            super::draw_scissor_region(channel.three_d(), 64, 32),
            Err(MaxwellLoweringError::IncompleteDraw(_))
        ));
        program_three_d(&mut channel, 0x0e04, (50 << 16) | 12);
        program_three_d(&mut channel, 0x0e08, (23 << 16) | 2);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 12,
                y: 7,
                width: 33,
                height: 16
            }
        );
        program_three_d(&mut channel, 0x0e04, (80 << 16) | 70);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 64,
                y: 7,
                width: 0,
                height: 16
            }
        );
        program_three_d(&mut channel, 0x0e00, 0);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 16, 12).unwrap(),
            ScissorRect {
                x: 5,
                y: 7,
                width: 11,
                height: 5
            }
        );
        program_three_d(&mut channel, 0x0ff4, 100);
        assert_eq!(
            super::draw_scissor_region(channel.three_d(), 64, 32).unwrap(),
            ScissorRect {
                x: 64,
                y: 7,
                width: 0,
                height: 20
            }
        );
    }

    #[test]
    fn snap_grid_precision_only_rejects_the_consumed_viewport() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x0a1c, 0);
        program_three_d(&mut channel, 0x0bfc, 0x1f1f);
        assert!(super::draw_viewport_transform(channel.three_d()).is_ok());
        let register = channel.three_d().fixed_function().viewport()[15].snap_grid_precision();
        assert_eq!(register.value(), Some(&[31, 31]));
        assert_eq!(
            register.source().unwrap().method(),
            nixe_gpu::GpuMethodId(0x0bfc)
        );
        program_three_d(&mut channel, 0x0a1c, 0x0302);
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedViewportSnapGridPrecision {
                viewport: 0,
                precision: [2, 3]
            })
        ));
        program_three_d(&mut channel, 0x0a1c, 0);
        assert!(super::draw_viewport_transform(channel.three_d()).is_ok());
    }

    #[test]
    fn lower_left_origin_reflects_viewport_and_scissor_independently_of_facing() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        for (method, value) in [
            (0x0a00, 32.0_f32.to_bits()),
            (0x0a04, 16.0_f32.to_bits()),
            (0x0a08, 1.0_f32.to_bits()),
            (0x0a0c, 37.0_f32.to_bits()),
            (0x0a10, 19.0_f32.to_bits()),
            (0x0a14, 0.0_f32.to_bits()),
            (0x193c, 1),
            (0x192c, 1),
            (0x13ac, 1),
            (0x0ff4, 128 << 16),
            (0x0ff8, 64 << 16),
            (0x0e00, 1),
            (0x0e04, (100 << 16) | 8),
            (0x0e08, (20 << 16) | 4),
        ] {
            program_three_d(&mut channel, method, value);
        }
        for origin in [1, 0x11] {
            program_three_d(&mut channel, 0x13ac, origin);
            let viewport = super::draw_viewport_transform(channel.three_d())
                .unwrap()
                .unwrap();
            assert_eq!(viewport.scale(), [32.0, -16.0, 1.0]);
            assert_eq!(viewport.offset(), [37.0, 45.0, 0.0]);
            assert_eq!(
                super::draw_scissor_region(channel.three_d(), 128, 64).unwrap(),
                nixe_gpu::ScissorRect {
                    x: 8,
                    y: 44,
                    width: 92,
                    height: 16
                }
            );
        }
        program_three_d(&mut channel, 0x0a18, 0x6430);
        assert_eq!(
            super::draw_viewport_transform(channel.three_d())
                .unwrap()
                .unwrap()
                .scale()[1],
            16.0
        );
    }

    #[test]
    fn viewport_depth_mode_and_unbounded_pixel_limits_are_independent() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        for (method, argument) in [
            (0x0a00, 640.0_f32.to_bits()),
            (0x0a04, 360.0_f32.to_bits()),
            (0x0a08, 0.5_f32.to_bits()),
            (0x0a0c, 640.0_f32.to_bits()),
            (0x0a10, 360.0_f32.to_bits()),
            (0x0a14, 0.5_f32.to_bits()),
            (0x0c08, f32::NEG_INFINITY.to_bits()),
            (0x0c0c, f32::INFINITY.to_bits()),
            (0x192c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(MaxwellLoweringError::IncompleteDraw(
                "SET_VIEWPORT_CLIP_CONTROL"
            ))
        ));
        program_three_d(&mut channel, 0x193c, 0x281c);
        let gl = super::draw_viewport_transform(channel.three_d())
            .unwrap()
            .unwrap();
        assert_eq!(gl.depth_range(), [0.0, 1.0]);
        assert!(gl.depth_clip_negative_one_to_one());
        program_three_d(&mut channel, 0x193c, 0x281d);
        let zero = super::draw_viewport_transform(channel.three_d())
            .unwrap()
            .unwrap();
        assert_eq!(zero.depth_range(), [0.5, 1.0]);
        assert!(!zero.depth_clip_negative_one_to_one());
        program_three_d(&mut channel, 0x0c08, 0.75_f32.to_bits());
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedViewportPixelDepthBounds)
        ));
    }

    #[test]
    fn viewport_y_swizzle_composes_with_both_scale_signs_and_preserves_offsets() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x193c, 0);
        for (method, argument) in [
            (0x0a00, 32.0_f32.to_bits()),
            (0x0a08, 0.5_f32.to_bits()),
            (0x0a0c, 37.0_f32.to_bits()),
            (0x0a10, 19.0_f32.to_bits()),
            (0x0a14, 0.5_f32.to_bits()),
            (0x0c08, 0.0_f32.to_bits()),
            (0x0c0c, 1.0_f32.to_bits()),
            (0x192c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        for scale in [-16.0_f32, 16.0] {
            program_three_d(&mut channel, 0x0a04, scale.to_bits());
            for (swizzle, expected_y) in [(0x6420, scale), (0x6430, -scale)] {
                program_three_d(&mut channel, 0x0a18, swizzle);
                // Unconsumed viewport state must not reject viewport zero.
                program_three_d(&mut channel, 0x0a38, 0x7654);
                let transform = super::draw_viewport_transform(channel.three_d())
                    .unwrap()
                    .unwrap();
                assert_eq!(transform.scale(), [32.0, expected_y, 0.5]);
                assert_eq!(transform.offset(), [37.0, 19.0, 0.5]);
                assert_eq!(transform.depth_range(), [0.0, 1.0]);
            }
        }
        program_three_d(&mut channel, 0x0a18, 0x6421);
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(
                MaxwellLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                    viewport: 0,
                    ..
                }
            )
        ));
        program_three_d(&mut channel, 0x0a18, 0x6430);
        program_three_d(&mut channel, 0x192c, 0);
        assert!(matches!(
            super::draw_viewport_transform(channel.three_d()),
            Err(
                MaxwellLoweringError::UnsupportedViewportCoordinateSwizzleSemantics {
                    viewport: 0,
                    ..
                }
            )
        ));
    }

    use nixe_gpu::{
        DepthCompareOperation, GpuCacheConfiguration, PrimitiveTopology, ResourceDependency,
        ShaderId, ShaderInstruction, ShaderIr, ShaderOperation, ShaderPredicate,
        ShaderSourceLocation, ShaderStage, VerifiedShaderIr, VertexComponentCount,
        VertexComponentWidth, VertexFormat,
    };

    use crate::{MaxwellThreeDBegin, MaxwellThreeDCompareOp, MaxwellThreeDVertexAttributeFormat};

    use super::{
        FingerprintCache, MaxwellLoweringCache, MaxwellLoweringError, ShaderTranslationRecord,
        depth_stencil_attachment_required, neutral_depth_compare, neutral_first_instance,
        neutral_vertex_format, primitive_topology,
    };

    #[test]
    fn consumed_vertex_streams_deduplicate_attributes_and_cover_stream_31() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let mut channel = three_d_channel();
        for (method, argument) in [
            (0x1160, 0x3820_001f),
            (0x1164, 0x3820_001f),
            (0x1168, 0x3820_0003),
            (0x116c, 0x3820_0040),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        assert_eq!(
            super::consumed_vertex_streams(channel.three_d()).collect::<Vec<_>>(),
            [3, 31]
        );
    }

    #[test]
    fn da_attribute_skip_masks_gate_fetch_without_hiding_consumed_components() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        use nixe_gpu::{ShaderInterfaceElement, ShaderIoLocation, ShaderScalarType};
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x1160, 0x3820_001f);
        program_three_d(&mut channel, 0x1180, 0x3820_0003);
        program_three_d(&mut channel, 0x11a0, 0x3820_0004);
        program_three_d(&mut channel, 0x11c0, 0x3820_0005);
        for (method, value) in [(0x1120, 0xf), (0x1124, 0xf), (0x1128, 0xf), (0x112c, 0xf)] {
            program_three_d(&mut channel, method, value);
            let index = (method - 0x1120) as usize / 4;
            let register = &channel.three_d().vertex_input().attribute_skip_masks()[index];
            assert_eq!(register.raw(), Some(value));
            assert_eq!(register.source().unwrap().method().0, method);
        }
        assert!(
            super::consumed_vertex_streams(channel.three_d())
                .next()
                .is_none()
        );
        let ir = ShaderIr::new(
            ShaderStage::Vertex,
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Generic(0),
                    1,
                    ShaderScalarType::Float32,
                    None,
                )
                .unwrap(),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        assert!(matches!(
            super::validate_vertex_attribute_skip_masks(channel.three_d(), &ir),
            Err(MaxwellLoweringError::UnsupportedSkippedVertexComponent {
                attribute: 0,
                component: 1
            })
        ));
        // Component zero remains skipped; component one is fetched.
        program_three_d(&mut channel, 0x1120, 0xd);
        assert!(super::validate_vertex_attribute_skip_masks(channel.three_d(), &ir).is_ok());
        assert_eq!(
            super::consumed_vertex_streams(channel.three_d()).collect::<Vec<_>>(),
            [31]
        );
    }

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
        let module = nixe_gpu::ShaderBackendModule::new(verified);
        let configuration = GpuCacheConfiguration::new(6, 1, 1, 1, 1).unwrap();
        let mut cache = MaxwellLoweringCache::new(configuration);
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
            let begin = MaxwellThreeDBegin::parse(7 | (instance << 26)).unwrap();
            assert_eq!(primitive_topology(begin), Ok(PrimitiveTopology::Quads));
        }
    }

    #[test]
    fn primitive_continuation_modes_remain_precise_fatal_boundaries() {
        let primitive_id = MaxwellThreeDBegin::parse(4 | (1 << 24)).unwrap();
        assert_eq!(
            primitive_topology(primitive_id),
            Err(MaxwellLoweringError::UnsupportedPrimitiveIdContinuation)
        );

        for split_mode in 1..=3 {
            let split = MaxwellThreeDBegin::parse(4 | (split_mode << 29)).unwrap();
            assert_eq!(
                primitive_topology(split),
                Err(MaxwellLoweringError::UnsupportedPrimitiveSplitMode(
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
            Err(MaxwellLoweringError::InstanceIndexOverflow {
                base: u32::MAX,
                relative: 1,
            })
        );
    }
}
