//! Native frontend preparation. Maxwell dirty revisions and mapping/CPU proofs
//! select immutable semantic state; no renderer IDs or old register IR are built.
use super::*;
mod compute;
mod identity;
mod image;
mod interface;
use crate::MaxwellGpuMemory;
use crate::engines::threed::{MaxwellThreeDResolvedBinding, MaxwellThreeDResolvedImage};
use crate::execution::{MaxwellFrontend, MaxwellSubmissionExecutionError as Error};
use identity::{ImageKey, ParameterKey, ParameterSelection, ProgramKey, ProgramLookup};
use interface::ShaderInterface;
use nixe_gpu::native::*;
use nixe_gpu::shader_module::ResourceId;

type GraphicsSourceKey =
    [Option<crate::recompiler::GraphicsSource>; crate::MAXWELL_PIPELINE_SHADER_COUNT];

type ComputeSourceKey = (crate::MaxwellAddressSpaceId, u64, [u32; 3], u32, u32, u32);

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeFrontendStatistics {
    pub draws: u64,
    pub prepared_draws: u64,
    pub prepared_parameters: u64,
    pub reused_parameters: u64,
    pub reused_programs: u64,
    pub shader_captures: u64,
    pub revalidated_captures: u64,
    pub restored_captures: u64,
    pub warm_draws: u64,
}
#[derive(Clone)]
struct Code {
    state: super::super::threed::state::MaxwellThreeDShaderStateIdentity,
    address_space: crate::MaxwellAddressSpaceId,
    source: GraphicsSourceKey,
    snapshot: Arc<crate::recompiler::CapturedProgram>,
    code: Arc<GuestShaderCode>,
    interface: Arc<ShaderInterface>,
}
struct Prepared {
    state: super::super::threed::state::MaxwellThreeDDrawStateIdentity,
    resource_groups: [u64; 4],
    attachments: [u32; 4],
    code: Arc<GuestShaderCode>,
    indexed: bool,
    draw: Arc<GuestDrawState>,
}
// Keep native selections in their channel slots, as Eden keeps current buffer
// and texture bindings. A changed descriptor does not make unrelated native
// views depend on another lookup in the bounded general caches.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
#[derive(Default)]
struct ProgramPreparation {
    program: Option<Arc<GuestProgram>>,
    bindings: Vec<(ResourceId, GuestBinding)>,
    binding_revisions: Vec<u64>,
    resource_groups: [u64; 4],
}
#[derive(Clone, Eq, PartialEq)]
struct PresentationKey {
    backing: nixe_gpu::BackingView,
    extent: [u32; 2],
    format: nixe_gpu::PresentationImageFormat,
    layout: nixe_gpu::ImageMemoryLayout,
    row_pitch: u32,
}
// Dirty revisions and current bindings belong to the channel. Immutable content
// caches and the compiler below belong to the command owner and survive switches.
struct ChannelPreparation {
    render_enabled: bool,
    resources: super::super::threed::MaxwellThreeDResolvedResourceCache,
    selection: Option<(u64, Arc<GuestShaderCode>, bool, bool)>,
    code: Option<Code>,
    prepared: Option<Prepared>,
    program: ProgramPreparation,
    prepared_compute: Option<Arc<GuestProgram>>,
    compute_resources: Option<super::super::compute::ComputeResources>,
    compute: Option<Arc<crate::recompiler::CapturedProgram>>,
}
pub struct MaxwellNativeFrontend {
    presentation: Vec<(PresentationKey, Arc<[GuestImagePlane]>)>,
    source_codes: nixe_gpu::WorkingSet<(crate::MaxwellAddressSpaceId, GraphicsSourceKey), Code>,
    interfaces: nixe_gpu::WorkingSet<nixe_gpu::shader_identity::ContentId, Arc<ShaderInterface>>,
    parameter_states: nixe_gpu::WorkingSet<ParameterKey, ParameterSelection>,
    programs: nixe_gpu::WorkingSet<ProgramKey, Arc<GuestProgram>>,
    draw_vertices: Vec<GuestVertexBuffer>,
    draws: nixe_gpu::WorkingSet<Arc<GuestDrawState>, Arc<GuestDrawState>>,
    next: u64,
    parameters: Vec<u32>,
    pub statistics: NativeFrontendStatistics,
    image_views: nixe_gpu::WorkingSet<ImageKey, GuestImage>,
    capture_compiler: Option<nixe_shader_recompiler::Compiler>,
    capture_memory: crate::recompiler::MemoryCache,
    capture_archive: Option<crate::recompiler::archive::CaptureArchive>,
    configuration: nixe_gpu::shader_compilation::Configuration,
    compute_sources:
        nixe_gpu::WorkingSet<ComputeSourceKey, Arc<crate::recompiler::CapturedProgram>>,
    captured_codes:
        nixe_gpu::WorkingSet<nixe_gpu::shader_identity::ContentId, Arc<GuestShaderCode>>,

    channel: ChannelPreparation,
    current_channel: Option<crate::MaxwellChannelId>,
    channels: std::collections::BTreeMap<crate::MaxwellChannelId, Option<ChannelPreparation>>,
}
impl ChannelPreparation {
    fn new(alignment: u64) -> Self {
        Self {
            render_enabled: true,
            resources: MaxwellThreeDResolvedResourceCache::new(alignment),
            selection: None,
            code: None,
            prepared: None,
            program: ProgramPreparation::default(),
            prepared_compute: None,
            compute_resources: None,
            compute: None,
        }
    }
}
fn compilation(error: impl Display) -> Error {
    Error::Native(error.to_string().into())
}
fn lowering<T>(result: Result<T, MaxwellLoweringError>) -> Result<T, Error> {
    result.map_err(Error::Lowering)
}
impl MaxwellNativeFrontend {
    pub fn new(configuration: nixe_gpu::shader_compilation::Configuration) -> Self {
        let channel = ChannelPreparation::new(configuration.host.min_ssbo_alignment);
        Self {
            presentation: Vec::new(),
            source_codes: nixe_gpu::WorkingSet::new(2048, 32 * 1024 * 1024),
            interfaces: nixe_gpu::WorkingSet::new(2048, 8 * 1024 * 1024),
            parameter_states: nixe_gpu::WorkingSet::new(8192, 16 * 1024 * 1024),
            programs: nixe_gpu::WorkingSet::new(8192, 32 * 1024 * 1024),
            draw_vertices: Vec::new(),
            draws: nixe_gpu::WorkingSet::new(8192, 64 * 1024 * 1024),
            next: 1,
            parameters: Vec::new(),
            statistics: Default::default(),
            image_views: nixe_gpu::WorkingSet::new(4096, IMAGE_VIEW_BYTES),
            capture_compiler: None,
            capture_memory: Default::default(),
            capture_archive: None,
            configuration,
            compute_sources: nixe_gpu::WorkingSet::new(256, 8 * 1024 * 1024),
            captured_codes: nixe_gpu::WorkingSet::new(2048, 64 * 1024 * 1024),

            channel,
            current_channel: None,
            channels: Default::default(),
        }
    }
    pub fn set_capture_cache(
        &mut self,
        directory: &std::path::Path,
        namespace: nixe_gpu::shader_identity::ContentId,
        budget: u64,
    ) {
        let mut archive = crate::recompiler::archive::CaptureArchive::new(
            directory,
            namespace,
            self.configuration.clone(),
            budget,
        );
        archive.load(&mut self.captured_codes);
        self.capture_archive = Some(archive);
    }
    fn conversion_image(
        &mut self,
        source: nixe_gpu::ConversionImage,
    ) -> Result<GuestImage, MaxwellLoweringError> {
        let key = ImageKey {
            encoding: super::super::threed::MaxwellThreeDGuestImageFormat::Canonical(
                source.description.format(),
            ),
            description: source.description,
            canonical_texels: source.view.has_canonical_texels(),
            view: source.view.clone(),
        };
        if let Some(image) = self.image_views.get(&key) {
            return Ok(image.clone());
        }
        let image = image::conversion(&source)
            .map_err(|error| MaxwellLoweringError::ImageBacking(error.to_string()))?;
        Self::remember_image(&mut self.image_views, key, &image)
            .map_err(|error| MaxwellLoweringError::ImageBacking(error.to_string()))?;
        Ok(image)
    }
    fn identity(next: &mut u64) -> nixe_gpu::PreparedStateId {
        let id = *next;
        *next = id
            .checked_add(1)
            .expect("native prepared identity exhausted");
        nixe_gpu::PreparedStateId::new(id, 1)
    }
    fn ensure_code(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), Error> {
        if let Some(code) = &mut self.channel.code
            && code.state.matches_code(state)
            && code.address_space == space.id()
            && code
                .snapshot
                .stages
                .iter()
                .all(|stage| stage.memory_is_current(space))
        {
            return Ok(());
        }
        let mut source: GraphicsSourceKey = [None; crate::MAXWELL_PIPELINE_SHADER_COUNT];
        for (index, source_slot) in source.iter_mut().enumerate() {
            let pipeline = state.shader_bindings().pipeline(index);
            if pipeline.enabled().value() != Some(&true) {
                continue;
            }
            *source_slot =
                Some(crate::recompiler::graphics_source(state, index).map_err(compilation)?);
        }
        // A shader switch selects its captured code and interface directly.
        // Construct compilation bindings only for a missing/stale capture,
        // like Eden's address-indexed shader cache and invalidation proofs.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/shader_cache.h
        if let Some(code) = self.source_codes.get_mut(&(space.id(), source))
            && code
                .snapshot
                .stages
                .iter()
                .all(|stage| stage.memory_is_current(space))
        {
            code.state = state.shader_state_identity();
            self.channel.code = Some(code.clone());
            return Ok(());
        }
        let requests: Vec<_> = source
            .iter()
            .enumerate()
            .filter_map(|(index, source)| {
                source.as_ref().map(|source| {
                    crate::recompiler::graphics_request(state, index)
                        .map(|request| (source.group, request))
                        .map_err(compilation)
                })
            })
            .collect::<Result<_, _>>()?;
        let valid = |snapshot: &crate::recompiler::CapturedProgram| {
            snapshot.stages.len() == requests.len()
                && snapshot
                    .stages
                    .iter()
                    .zip(&requests)
                    .all(|(stage, (_, request))| stage.request_is_current(request, space))
        };
        let cached = self
            .channel
            .code
            .as_ref()
            .filter(|code| code.source == source)
            .map(|code| Arc::clone(&code.snapshot))
            .or_else(|| {
                self.source_codes
                    .get(&(space.id(), source))
                    .map(|code| Arc::clone(&code.snapshot))
            });
        let (cached, refreshed) = match cached {
            Some(snapshot) if valid(&snapshot) => (Some(snapshot), None),
            Some(snapshot) => (
                None,
                snapshot
                    .refresh(&requests, space, &mut self.capture_memory, access)
                    .map_err(compilation)?,
            ),
            None => (None, None),
        };
        let snapshot =
            if let Some(snapshot) = cached {
                snapshot
            } else if let Some(snapshot) = refreshed {
                self.statistics.revalidated_captures += 1;
                Arc::new(snapshot)
            } else if let Some(snapshot) = self.capture_archive.as_ref().and_then(|archive| {
                archive.restore(&requests, space, &mut self.capture_memory, access)
            }) {
                self.statistics.restored_captures += 1;
                Arc::new(snapshot)
            } else {
                if self.capture_compiler.is_none() {
                    self.capture_compiler =
                        Some(nixe_shader_recompiler::Compiler::new().map_err(compilation)?);
                }
                let snapshot = Arc::new(
                    crate::recompiler::capture_program(
                        self.capture_compiler.as_mut().unwrap(),
                        space,
                        requests,
                        &self.configuration,
                        &mut self.capture_memory,
                        access,
                    )
                    .map_err(compilation)?,
                );
                self.statistics.shader_captures += 1;
                snapshot
            };
        let inputs = &snapshot.inputs;
        let code = if let Some(code) = self.captured_codes.get(&inputs.identity) {
            Arc::clone(code)
        } else {
            let code = Arc::clone(inputs);
            let weight = inputs.retained_bytes();
            self.captured_codes
                .insert(inputs.identity, Arc::clone(&code), weight);
            code
        };
        if let Some(archive) = &mut self.capture_archive {
            archive.remember(&code);
        }
        let interface = match self.interfaces.get(&code.identity) {
            Some(interface) => Arc::clone(interface),
            None => {
                let interface = Arc::new(ShaderInterface::new(inputs)?);
                self.interfaces.insert(
                    code.identity,
                    Arc::clone(&interface),
                    interface.retained_bytes(),
                );
                interface
            }
        };
        let code = Code {
            state: state.shader_state_identity(),
            address_space: space.id(),
            source,
            snapshot,
            code,
            interface,
        };
        self.source_codes.insert(
            (space.id(), source),
            code.clone(),
            code.snapshot.retained_bytes() + code.interface.retained_bytes(),
        );
        self.channel.code = Some(code);
        Ok(())
    }
    fn image(
        image_views: &mut nixe_gpu::WorkingSet<ImageKey, GuestImage>,
        next: &mut u64,
        resolved: &MaxwellThreeDResolvedImage,
    ) -> Result<GuestImage, Error> {
        let canonical_texels = resolved
            .guest_layout()
            .has_direct_canonical_representation();
        let key = ImageKey {
            encoding: resolved.guest_format(),
            description: resolved.description(),
            canonical_texels,
            view: resolved.view().clone(),
        };
        if let Some(image) = image_views.get(&key) {
            return Ok(image.clone());
        }
        let image = GuestImage {
            view_identity: Some(Self::identity(next)),
            sample_grid: image::sample_grid(resolved.description().samples())?,
            description: resolved.description(),
            swizzle: resolved.view().swizzle(),
            canonical_texels,
            planes: image::planes(resolved)?,
        };
        Self::remember_image(image_views, key, &image)?;
        Ok(image)
    }
    fn remember_image(
        image_views: &mut nixe_gpu::WorkingSet<ImageKey, GuestImage>,
        key: ImageKey,
        image: &GuestImage,
    ) -> Result<(), Error> {
        let size = plane_metadata_bytes(image)
            + std::mem::size_of::<ImageKey>()
            + std::mem::size_of_val(key.view.bindings());
        if size > IMAGE_VIEW_BYTES {
            return Err(compilation(
                "native image address metadata exceeds the 512 MiB frontend cache budget",
            ));
        }
        image_views.insert(key, image.clone(), size);
        Ok(())
    }
    fn draw(
        &mut self,
        state: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        instances: Option<(u32, u32)>,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, Error> {
        let begin = *state
            .vertex_input()
            .primitive()
            .active_begin()
            .ok_or_else(|| compilation("draw consumes an unavailable BEGIN sequence"))?;
        if state.shader_execution().stream_output_enable().value() == Some(&true) {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedStreamOutputSemantics,
            ));
        }
        self.statistics.draws += 1;
        self.ensure_code(state, space, access)?;
        let code = self.channel.code.as_ref().unwrap();
        let inline = state.inline_index().is_some();
        let same_selection = self.channel.selection.as_ref().is_some_and(
            |(revision, shader, indexed, old_inline)| {
                *revision == state.resource_selection_revision()
                    && Arc::ptr_eq(shader, &code.code)
                    && *indexed == trigger.is_indexed()
                    && *old_inline == inline
            },
        );
        if !same_selection {
            self.channel.resources.select(
                &code.interface.resources,
                trigger.resource_slots(state, code.interface.vertex_attributes),
            );
            self.channel.selection = Some((
                state.resource_selection_revision(),
                Arc::clone(&code.code),
                trigger.is_indexed(),
                inline,
            ));
        }
        let resources = self
            .channel
            .resources
            .resolve(state, space, 4096, access)
            .map_err(Error::ThreeDResource)?;
        let mut arguments = lowering(native_arguments(state, trigger))?;
        if let Some((first, count)) = instances {
            let base = state
                .vertex_input()
                .assembly()
                .global_base_instance_index()
                .value()
                .copied()
                .unwrap_or(0);
            let first = lowering(neutral_first_instance(base, first))?;
            match &mut arguments {
                DrawArguments::NonIndexed {
                    first_instance,
                    instance_count,
                    ..
                }
                | DrawArguments::Indexed {
                    first_instance,
                    instance_count,
                    ..
                } => {
                    *first_instance = first;
                    *instance_count = count;
                }
            }
        }
        if let Some(p) = &self.channel.prepared
            && p.state.matches(state)
            && p.resource_groups == resources.revisions()
            && Arc::ptr_eq(&p.code, &code.code)
            && p.indexed == trigger.is_indexed()
            && p.draw
                .index
                .as_ref()
                .is_some_and(|i| matches!(i, GuestIndexBuffer::Inline(_)))
                == state.inline_index().is_some()
            && match (&p.draw.index, state.inline_index()) {
                (Some(GuestIndexBuffer::Inline(a)), Some(b)) => Arc::ptr_eq(a, b),
                (_, None) => true,
                _ => false,
            }
        {
            self.statistics.warm_draws += 1;
            return Ok(NativeWork {
                frontend,
                command: NativeCommand::Draw {
                    state: Arc::clone(&p.draw),
                    arguments,
                },
            });
        }
        let resource_groups = resources.revisions();
        let attachments = state.attachment_selection();
        let shader = Arc::clone(&code.code);
        let shader_pixel_depth = code.interface.pixel_depth;
        let vertex_attributes = code.interface.vertex_attributes;
        let binding_roles = Arc::clone(&code.interface.bindings);
        let mut selection = None;
        let (colors, depth_stencil) = if let Some(previous) = &self.channel.prepared
            && previous.resource_groups[3] == resource_groups[3]
            && previous.attachments == attachments
        {
            (
                Arc::clone(&previous.draw.colors),
                previous.draw.depth_stencil.clone(),
            )
        } else {
            let selected = selection.insert(lowering(select_draw_attachments(state, resources))?);
            let colors: Arc<[_]> = selected
                .colors
                .iter()
                .map(|(_, role)| {
                    Self::image(
                        &mut self.image_views,
                        &mut self.next,
                        lowering(resolved_image(resources, *role))?,
                    )
                })
                .collect::<Result<_, _>>()?;
            let depth = selected
                .depth_stencil
                .map(|role| {
                    Self::image(
                        &mut self.image_views,
                        &mut self.next,
                        lowering(resolved_image(resources, role))?,
                    )
                })
                .transpose()?;
            (colors, depth)
        };
        let mut extent = [u32::MAX; 2];
        for image in colors.iter().chain(depth_stencil.iter()) {
            extent[0] = extent[0].min(image.description.extent().width);
            extent[1] = extent[1].min(image.description.extent().height);
        }
        if extent.contains(&u32::MAX) {
            return Err(compilation("draw has no attachment extent"));
        }
        let parameters = if let Some(prepared) = &self.channel.prepared
            && prepared.state.matches_parameters(state)
            && Arc::ptr_eq(&prepared.code, &shader)
            && prepared.indexed == trigger.is_indexed()
        {
            self.statistics.reused_parameters += 1;
            Arc::clone(&prepared.draw.parameters)
        } else {
            let mut formats = [None; 8];
            for (slot, image) in formats.iter_mut().zip(colors.iter()) {
                *slot = Some(image.description.format());
            }
            let key = ParameterKey {
                shader: shader.identity,
                registers: state.parameter_fingerprint(),
                indexed: trigger.is_indexed(),
                extent,
                colors: formats,
                depth_stencil: depth_stencil
                    .as_ref()
                    .map(|image| image.description.format()),
                samples: colors
                    .first()
                    .or(depth_stencil.as_ref())
                    .unwrap()
                    .description
                    .samples(),
            };
            // Select from canonical register inputs before interpreting or
            // allocating derived state. A hash collision is only a cache miss;
            // every relevant register must match before validation is reused.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.cpp
            if let Some(selected) = self.parameter_states.get(&key)
                && state.matches_parameter_words(&selected.words)
            {
                self.statistics.reused_parameters += 1;
                Arc::clone(&selected.parameters)
            } else {
                if selection.is_none() {
                    selection = Some(lowering(select_draw_attachments(state, resources))?);
                }
                let selection = selection.as_ref().unwrap();
                let (tessellation, raster) = lowering(validate_consumed_state(state, trigger))?;
                if state.shader_execution().stream_output_enable().value() == Some(&true) {
                    return Err(Error::Lowering(
                        MaxwellLoweringError::UnsupportedStreamOutputSemantics,
                    ));
                }
                let raster = raster.unwrap();
                let color_outputs =
                    lowering(color::draw_color_outputs(state, resources, selection))?;
                lowering(validate_draw_iterated_blend_state(state, selection))?;
                lowering(validate_draw_logic_op_state(state, selection))?;
                let mut vertices = Vec::new();
                for stream in consumed_vertex_streams(state, vertex_attributes) {
                    let s = &state.vertex_input().streams(stream as usize);
                    let format = s
                        .format()
                        .value()
                        .filter(|f| f.enabled())
                        .copied()
                        .ok_or_else(|| compilation("consumed vertex stream is disabled"))?;
                    let instanced = s.instanced().value().copied().unwrap_or(false);
                    if instanced && s.frequency().value().copied().unwrap_or(1) != 1 {
                        return Err(Error::Lowering(
                            MaxwellLoweringError::UnsupportedVertexInstanceDivisor {
                                stream,
                                divisor: *s.frequency().value().unwrap(),
                            },
                        ));
                    }
                    let attributes = vertex_attribute_locations(vertex_attributes)
                        .map(|location| (location, state.vertex_input().attributes(location)))
                        .filter_map(|(location, attr)| {
                            attr.value()
                                .filter(|a| {
                                    a.enabled()
                                        && a.stream() == stream
                                        && state.vertex_input().attribute_skip_mask(location as u8)
                                            != 15
                                })
                                .map(|a| {
                                    neutral_vertex_format(location as u8, *a).map(|format| {
                                        VertexAttribute {
                                            format,
                                            offset: u64::from(a.offset()),
                                            shader_location: location as u32,
                                        }
                                    })
                                })
                        })
                        .collect::<Result<Box<[_]>, _>>()
                        .map_err(Error::Lowering)?;
                    vertices.push(GuestVertexLayout {
                        binding: u32::from(stream),
                        stride: u32::from(format.stride()),
                        step: if instanced {
                            VertexStepMode::Instance
                        } else {
                            VertexStepMode::Vertex
                        },
                        attributes,
                    });
                }
                let depth = if depth_stencil.is_some() {
                    lowering(draw_depth_state(state))?
                } else {
                    DepthState::DISABLED
                };
                let pixel_depth = depth.test_enabled || shader_pixel_depth;
                let viewport =
                    lowering(draw_viewport_transform_with_depth_usage(state, pixel_depth))?;
                let position_transform = if viewport.is_none() {
                    Some(
                        nixe_gpu::VertexPositionTransform::framebuffer(extent[0], extent[1])
                            .map_err(compilation)?,
                    )
                } else {
                    None
                };
                let mut blend_constants = [0.; 4];
                for (out, r) in blend_constants.iter_mut().zip([
                    MaxwellThreeDFixedFunctionRegister::BlendConstantRed,
                    MaxwellThreeDFixedFunctionRegister::BlendConstantGreen,
                    MaxwellThreeDFixedFunctionRegister::BlendConstantBlue,
                    MaxwellThreeDFixedFunctionRegister::BlendConstantAlpha,
                ]) {
                    if let Some(MaxwellThreeDFixedFunctionValue::FloatBits(v)) =
                        state.fixed_function().register(r).value()
                    {
                        *out = f32::from_bits(v.get());
                    }
                }
                let parameters = GuestDrawParameters {
                    shader_state: GuestShaderState {
                        point_size_bits: state
                            .raster()
                            .point_size()
                            .value()
                            .map(|value| value.bits()),
                        force_early_z: state.shader_execution().api_mandated_early_z().value()
                            == Some(&crate::MaxwellThreeDApiMandatedEarlyZ::Enabled),
                        y_negate: matches!(state.fixed_function().register(MaxwellThreeDFixedFunctionRegister::WindowOrigin).value(), Some(MaxwellThreeDFixedFunctionValue::Mask(origin)) if origin & 1 != 0),
                    },
                    identity: Self::identity(&mut self.next),
                    vertices: vertices.into_boxed_slice(),
                    color_formats: colors.iter().map(|i| i.description.format()).collect(),
                    depth_stencil_format: depth_stencil.as_ref().map(|i| i.description.format()),
                    samples: colors
                        .first()
                        .or(depth_stencil.as_ref())
                        .unwrap()
                        .description
                        .samples(),
                    extent,
                    render_area: lowering(draw_surface_region(state, extent[0], extent[1]))?
                        .scissor(extent[0], extent[1]),
                    topology: lowering(primitive_topology(begin))?,
                    restart: state.vertex_input().primitive().restart_enabled().value()
                        == Some(&true),
                    provoking_first: state
                        .fixed_function()
                        .register(MaxwellThreeDFixedFunctionRegister::ProvokingVertex)
                        .value()
                        == Some(&MaxwellThreeDFixedFunctionValue::ProvokingVertex(
                            MaxwellThreeDProvokingVertex::First,
                        )),
                    raster: if state.raster().fill_via_triangle().value()
                        == Some(&MaxwellThreeDFillViaTriangleMode::FillBoundingBox)
                    {
                        TriangleRasterization::FillRectangle
                    } else {
                        raster.triangles
                    },
                    line: if state.generated_primitive()
                        == Some(super::super::threed::state::GeneratedPrimitive::Lines)
                        && state.line().anti_aliased_line_enable().value()
                            == Some(&MaxwellThreeDAntiAliasedLineEnable::Enabled)
                    {
                        Some(lowering(raster::smooth_line(state))?)
                    } else {
                        None
                    },
                    front_face: raster.front_face,
                    cull: raster.cull_mode,
                    depth_bias: raster.depth_bias,
                    viewport,
                    position_transform,
                    scissor: lowering(draw_scissor(state, resources, selection))?,
                    depth,
                    stencil: lowering(draw_stencil_test_state(state))?,
                    color_outputs,
                    blend_constants,
                    alpha_test: lowering(draw_alpha_test_state(state))?,
                    tessellation,
                    sample_mask: u64::MAX,
                    alpha_to_coverage: false,
                };
                let parameters = Arc::new(parameters);
                let words = state.parameter_words();
                let bytes =
                    identity::parameter_bytes(&parameters) + std::mem::size_of_val(words.as_ref());
                self.statistics.prepared_parameters += 1;
                self.parameter_states.insert(
                    key,
                    ParameterSelection {
                        words,
                        parameters: Arc::clone(&parameters),
                    },
                    bytes,
                );
                parameters
            }
        };
        let program = Self::native_program(
            &mut self.programs,
            &mut self.channel.program,
            &mut self.statistics,
            &mut self.image_views,
            &mut self.next,
            &shader,
            &binding_roles,
            resources,
        )?;
        // Vertex formats/strides belong to pipeline state. Retain physical
        // selections by their actual slots, independently of layout changes.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        let retained_vertices = self
            .channel
            .prepared
            .as_ref()
            .filter(|previous| {
                previous.draw.vertices.len() == parameters.vertices.len()
                    && parameters
                        .vertices
                        .iter()
                        .zip(previous.draw.vertices.iter())
                        .all(|(vertex, old)| vertex.binding == old.binding)
                    && (previous.resource_groups[0] == resource_groups[0]
                        || parameters
                            .vertices
                            .iter()
                            .zip(previous.draw.vertices.iter())
                            .all(|(vertex, old)| {
                                resolved_buffer(
                                    resources,
                                    MaxwellThreeDResourceRole::VertexStream(vertex.binding as u8),
                                )
                                .is_ok_and(|buffer| {
                                    &old.backing == buffer.view().backing().mapping()
                                })
                            }))
            })
            .map(|previous| Arc::clone(&previous.draw.vertices));
        if retained_vertices.is_none() {
            self.draw_vertices.clear();
            for layout in parameters.vertices.iter() {
                let role = MaxwellThreeDResourceRole::VertexStream(layout.binding as u8);
                self.draw_vertices.push(GuestVertexBuffer {
                    binding: layout.binding,
                    backing: lowering(resolved_buffer(resources, role))?
                        .view()
                        .backing()
                        .mapping()
                        .clone(),
                });
            }
        }
        let vertices = retained_vertices.as_deref().unwrap_or(&self.draw_vertices);
        let index = if let Some(bytes) = state.inline_index() {
            Some(GuestIndexBuffer::Inline(Arc::clone(bytes)))
        } else if trigger.is_indexed() {
            let i = MaxwellThreeDResourceRole::IndexBuffer;
            Some(GuestIndexBuffer::Canonical(
                lowering(resolved_buffer(resources, i))?
                    .view()
                    .backing()
                    .mapping()
                    .clone(),
                match state
                    .vertex_input()
                    .index()
                    .element_size()
                    .value()
                    .copied()
                    .ok_or_else(|| compilation("missing SET_INDEX_BUFFER_E"))?
                {
                    super::super::threed::MaxwellThreeDIndexElementSize::OneByte => {
                        nixe_gpu::IndexType::Uint8
                    }
                    super::super::threed::MaxwellThreeDIndexElementSize::TwoBytes => {
                        nixe_gpu::IndexType::Uint16
                    }
                    super::super::threed::MaxwellThreeDIndexElementSize::FourBytes => {
                        nixe_gpu::IndexType::Uint32
                    }
                },
            ))
        } else {
            None
        };
        let key = GuestDrawStateKey {
            program: program.identity,
            parameters: parameters.identity,
            vertices,
            index: index.as_ref(),
            colors: &colors,
            depth_stencil: depth_stencil.as_ref(),
        };
        let draw = if let Some(cached) = self.draws.get(&key) {
            self.statistics.warm_draws += 1;
            Arc::clone(cached)
        } else {
            self.statistics.prepared_draws += 1;
            let draw = Arc::new(GuestDrawState {
                identity: Self::identity(&mut self.next),
                program,
                vertices: retained_vertices
                    .unwrap_or_else(|| Arc::from(self.draw_vertices.as_slice())),
                index,
                colors,
                depth_stencil,
                parameters,
            });
            self.draws.insert(
                Arc::clone(&draw),
                Arc::clone(&draw),
                identity::draw_bytes(&draw),
            );
            draw
        };
        // Canonical proofs come from this resolution, independently of the
        // immutable physical bindings retained by the draw template.
        self.channel.prepared = Some(Prepared {
            state: state.draw_state_identity(),
            resource_groups,
            attachments,
            code: shader,
            indexed: trigger.is_indexed(),
            draw: Arc::clone(&draw),
        });
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Draw {
                state: draw,
                arguments,
            },
        })
    }
}

const IMAGE_VIEW_BYTES: usize = 512 * 1024 * 1024;

fn plane_metadata_bytes(image: &GuestImage) -> usize {
    image
        .planes
        .iter()
        .map(|p| {
            std::mem::size_of::<GuestImagePlane>()
                + std::mem::size_of::<nixe_gpu::physical::Footprint>()
                + p.footprint.mapping().metadata_bytes()
        })
        .sum()
}

impl MaxwellNativeFrontend {
    fn binding(
        image_views: &mut nixe_gpu::WorkingSet<ImageKey, GuestImage>,
        next: &mut u64,
        binding: MaxwellThreeDResolvedBinding<'_>,
    ) -> Result<GuestBinding, Error> {
        Ok(match binding {
            MaxwellThreeDResolvedBinding::Sampler(sampler) => {
                GuestBinding::Sampler(sampler.description().map_err(Error::ThreeDResource)?)
            }
            MaxwellThreeDResolvedBinding::Resource(resource) => match resource {
                MaxwellThreeDResolvedResource::Buffer { view: b, .. } => {
                    GuestBinding::Buffer(b.view().backing().mapping().clone())
                }
                MaxwellThreeDResolvedResource::Image { view, role } => GuestBinding::Image {
                    image: Self::image(image_views, next, view)?,
                    arrayed: matches!(
                        *role,
                        MaxwellThreeDResourceRole::SampledImage {
                            dimension: super::super::threed::MaxwellThreeDTextureDimension::TwoArray
                                | super::super::threed::MaxwellThreeDTextureDimension::CubeArray,
                            ..
                        }
                    ),
                },
            },
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn native_program(
        programs: &mut nixe_gpu::WorkingSet<ProgramKey, Arc<GuestProgram>>,
        selected: &mut ProgramPreparation,
        statistics: &mut NativeFrontendStatistics,
        image_views: &mut nixe_gpu::WorkingSet<ImageKey, GuestImage>,
        next: &mut u64,
        shader: &Arc<GuestShaderCode>,
        roles: &[interface::ShaderBinding],
        resources: &MaxwellThreeDResolvedResources,
    ) -> Result<Arc<GuestProgram>, Error> {
        let same_shader = selected.program.as_ref().is_some_and(|program| {
            Arc::ptr_eq(&program.shader, shader) && program.bindings.len() == roles.len()
        });
        let resource_groups = resources.revisions();
        if same_shader
            && selected.resource_groups[1] == resource_groups[1]
            && selected.resource_groups[2] == resource_groups[2]
        {
            statistics.reused_programs += 1;
            return Ok(Arc::clone(selected.program.as_ref().unwrap()));
        }
        if !same_shader {
            selected.bindings.clear();
            selected.binding_revisions.clear();
        }
        let mut changed = !same_shader;
        for (index, binding_slot) in roles.iter().enumerate() {
            let (id, role) = (binding_slot.id, binding_slot.role);
            let revision = resources
                .binding_revision(binding_slot.location)
                .ok_or_else(|| {
                    compilation(format!("shader consumes unavailable binding slot {role:?}"))
                })?;
            if same_shader && selected.binding_revisions[index] == revision {
                continue;
            }
            let binding = Self::binding(
                image_views,
                next,
                resources.binding_at(binding_slot.location).ok_or_else(|| {
                    compilation(format!("shader consumes unavailable binding {role:?}"))
                })?,
            )?;
            if same_shader {
                selected.binding_revisions[index] = revision;
                let old = &mut selected.bindings[index];
                if old.0 != id || !same_binding(&old.1, &binding) {
                    *old = (id, binding);
                    changed = true;
                }
            } else {
                selected.bindings.push((id, binding));
                selected.binding_revisions.push(revision);
            }
        }
        let program = if changed {
            // General caches resolve only a changed selection. The worker
            // receives its immutable payload with the existing physical leases.
            let key = ProgramLookup {
                shader: shader.identity,
                shared_memory_bytes: 0,
                bindings: &selected.bindings,
            };
            if let Some(program) = programs.get(&key) {
                statistics.reused_programs += 1;
                Arc::clone(program)
            } else {
                let candidate = Arc::new(GuestProgram {
                    identity: Self::identity(next),
                    shader: Arc::clone(shader),
                    bindings: selected.bindings.to_vec().into_boxed_slice(),
                    shared_memory_bytes: 0,
                });
                let key = ProgramKey(Arc::clone(&candidate));
                programs.insert(
                    key,
                    Arc::clone(&candidate),
                    identity::program_bytes(&candidate),
                );
                candidate
            }
        } else {
            statistics.reused_programs += 1;
            Arc::clone(selected.program.as_ref().unwrap())
        };
        selected.resource_groups = resource_groups;
        selected.program = Some(Arc::clone(&program));
        Ok(program)
    }
}

fn same_binding(a: &GuestBinding, b: &GuestBinding) -> bool {
    match (a, b) {
        (GuestBinding::NullBuffer, GuestBinding::NullBuffer) => true,
        // Program preparation depends on physical bindings, not CPU content
        // epochs. Native consumption still checks visibility on every draw.
        (GuestBinding::Buffer(a), GuestBinding::Buffer(b)) => a == b,
        (
            GuestBinding::Image {
                image: a,
                arrayed: aa,
            },
            GuestBinding::Image {
                image: b,
                arrayed: ab,
            },
        ) => aa == ab && a.same_prepared_view(b),
        (GuestBinding::Sampler(a), GuestBinding::Sampler(b)) => a == b,
        _ => false,
    }
}
fn native_arguments(
    state: &MaxwellThreeDState,
    trigger: MaxwellThreeDOperationTrigger,
) -> Result<DrawArguments, MaxwellLoweringError> {
    if let MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count, .. } = trigger {
        let input = state.vertex_input();
        let first_index = if state.inline_index().is_some() {
            0
        } else {
            *input
                .index()
                .first()
                .value()
                .ok_or(MaxwellLoweringError::IncompleteDraw("SET_INDEX_BUFFER_F"))?
        };
        let assembly = input.assembly();
        let base = assembly
            .global_base_vertex_index()
            .value()
            .copied()
            .unwrap_or(0);
        if base != assembly.vertex_id_base().value().copied().unwrap_or(0) {
            return Err(MaxwellLoweringError::UnsupportedIndexedDraw(
                "distinct vertex-fetch and shader-ID bases",
            ));
        }
        return Ok(DrawArguments::Indexed {
            first_index,
            index_count,
            vertex_offset: base as i32,
            first_instance: neutral_first_instance(
                assembly
                    .global_base_instance_index()
                    .value()
                    .copied()
                    .unwrap_or(0),
                input.primitive().instance_index(),
            )?,
            instance_count: 1,
        });
    }
    draw_arguments(state, trigger)
}
impl MaxwellFrontend for MaxwellNativeFrontend {
    fn bind_channel(&mut self, channel: crate::MaxwellChannelId) {
        if self.current_channel == Some(channel) {
            return;
        }
        let next = self
            .channels
            .entry(channel)
            .or_default()
            .take()
            .unwrap_or_else(|| ChannelPreparation::new(self.configuration.host.min_ssbo_alignment));
        let previous = std::mem::replace(&mut self.channel, next);
        if let Some(id) = self.current_channel.replace(channel) {
            *self.channels.get_mut(&id).unwrap() = Some(previous);
        }
    }
    fn close_channel(&mut self, channel: crate::MaxwellChannelId) {
        self.channels.remove(&channel);
        if self.current_channel == Some(channel) {
            self.current_channel = None;
            self.channel = ChannelPreparation::new(self.configuration.host.min_ssbo_alignment);
        }
    }

    fn lower_presentation(
        &mut self,
        request: nixe_gpu::PresentationImageRequest,
    ) -> Result<NativePresentationImage, Error> {
        let key = PresentationKey {
            backing: request.backing.clone(),
            extent: [request.width, request.height],
            format: request.format,
            layout: request.layout,
            row_pitch: request.row_pitch,
        };
        let planes = if let Some((_, planes)) =
            self.presentation.iter().find(|(cached, _)| *cached == key)
        {
            Arc::clone(planes)
        } else {
            let planes = image::presentation_planes(&request)?;
            if self.presentation.len() == 16 {
                self.presentation.remove(0);
            }
            self.presentation.push((key, Arc::clone(&planes)));
            planes
        };
        Ok(NativePresentationImage {
            inputs: request.inputs,
            planes,
            extent: [request.width, request.height],
            format: request.format,
            allow_canonical_import: request.allow_canonical_import,
        })
    }

    fn lower_three_d(
        &mut self,
        state: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        instances: Option<(u32, u32)>,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, Error> {
        if trigger.is_draw() {
            return self.draw(state, trigger, space, frontend, instances, access);
        }
        lowering(validate_consumed_state(state, trigger))?;
        self.channel.selection = None;
        let resources = self
            .channel
            .resources
            .select(
                &MaxwellThreeDResourceSelection::default(),
                trigger.resource_slots(state, 0),
            )
            .resolve(state, space, 4096, access)
            .map_err(Error::ThreeDResource)?;
        let clear = state.render_targets().clear();
        let surface = *clear
            .last_surface()
            .value()
            .ok_or_else(|| compilation("CLEAR_SURFACE state is incomplete"))?;
        if surface.array_layer() != 0 {
            return Err(compilation(
                "native clear array-layer selection requires a selected attachment layer",
            ));
        }
        if surface.color_mask() == 0 && !surface.depth() && !surface.stencil() {
            return Err(Error::Lowering(MaxwellLoweringError::EmptyClearMask));
        }
        if surface.stencil() && !lowering(clear_stencil_mask_is_full(state))? {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedClearStencilMaskSemantics,
            ));
        }
        let regions = lowering(MaxwellThreeDClearRegions::from_state(state))?;
        let mut commands = smallvec::SmallVec::<[_; 2]>::new();
        for resource in resources.resources() {
            let index = resource.role();
            let MaxwellThreeDResolvedResource::Image { view: image, .. } = resource else {
                continue;
            };
            let extent = image.description().extent();
            let r = regions.for_attachment(extent.width, extent.height);
            if r.is_empty() {
                continue;
            }
            let value = if matches!(index, MaxwellThreeDResourceRole::ColorTarget(_)) {
                if surface.color_mask() != 15 {
                    return Err(Error::Lowering(
                        MaxwellLoweringError::PartialColorClearUnsupported {
                            mask: surface.color_mask(),
                        },
                    ));
                }
                let mut color = [0.; 4];
                for (index, out) in color.iter_mut().enumerate() {
                    let c = clear.color(index);
                    *out = f32::from_bits(
                        c.value()
                            .ok_or_else(|| compilation("missing clear color"))?
                            .get(),
                    );
                }
                ClearValue::Color(color)
            } else {
                let depth = surface
                    .depth()
                    .then(|| {
                        clear
                            .depth()
                            .value()
                            .map(|v| f32::from_bits(v.get()))
                            .ok_or_else(|| compilation("missing clear depth"))
                    })
                    .transpose()?;
                let stencil = surface
                    .stencil()
                    .then(|| {
                        clear
                            .stencil()
                            .value()
                            .copied()
                            .ok_or_else(|| compilation("missing clear stencil"))
                    })
                    .transpose()?;
                match (depth, stencil) {
                    (Some(depth), Some(stencil)) => ClearValue::DepthStencil { depth, stencil },
                    (Some(d), None) => ClearValue::Depth(d),
                    (None, Some(s)) => ClearValue::Stencil(s),
                    _ => continue,
                }
            };
            commands.push(NativeCommand::Clear {
                image: Self::image(
                    &mut self.image_views,
                    &mut self.next,
                    lowering(resolved_image(resources, index))?,
                )?,
                rectangle: nixe_gpu::ScissorRect {
                    x: r.min_x,
                    y: r.min_y,
                    width: r.max_x - r.min_x,
                    height: r.max_y - r.min_y,
                },
                value,
            });
        }
        Ok(NativeWork {
            frontend,
            command: if commands.len() == 1 {
                commands.pop().unwrap()
            } else {
                NativeCommand::Sequence(commands.into_vec().into_boxed_slice())
            },
        })
    }
    fn lower_launch(
        &mut self,
        launch: &super::super::compute::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, Error> {
        launch.validate_execution().map_err(Error::ComputeLaunch)?;
        let capture = self.capture_compute(launch, space, access)?;
        let inputs = &capture.inputs;
        let code = if let Some(code) = self.captured_codes.get(&inputs.identity) {
            Arc::clone(code)
        } else {
            let code = Arc::clone(inputs);
            let weight = inputs.retained_bytes();
            self.captured_codes
                .insert(inputs.identity, Arc::clone(&code), weight);
            code
        };
        if let Some(program) = &self.channel.prepared_compute
            && Arc::ptr_eq(&program.shader, &code)
            && let Some(resources) = &mut self.channel.compute_resources
            && resources
                .refresh(
                    launch,
                    space,
                    &mut self.capture_memory,
                    access,
                    &mut self.channel.resources,
                )
                .map_err(Error::ComputeLaunch)?
        {
            return Ok(NativeWork {
                frontend,
                command: NativeCommand::Dispatch {
                    program: Arc::clone(program),
                    groups: launch.workgroups(),
                },
            });
        }
        let mut resources = launch
            .native_resources(
                &inputs.stages[0].info,
                self.configuration.host.min_ssbo_alignment.max(1),
                space,
                &mut self.capture_memory,
                access,
            )
            .map_err(Error::ComputeLaunch)?;
        resources.textures = self
            .channel
            .resources
            .resolve_compute_textures(launch, &inputs.stages[0].info, space, access)
            .map_err(Error::ThreeDResource)?;
        let mut bindings = Vec::new();
        for (resource, range) in &resources.bindings {
            let Some(range) = range else {
                bindings.push((*resource, GuestBinding::NullBuffer));
                continue;
            };
            let retained = lowering(buffer::retain(range))?;
            bindings.push((
                *resource,
                GuestBinding::Buffer(retained.backing.mapping().clone()),
            ));
        }
        if let Some(textures) = &resources.textures {
            for &(id, location) in &textures.bindings {
                bindings.push((
                    id,
                    Self::binding(
                        &mut self.image_views,
                        &mut self.next,
                        textures.resources.binding_at(location).ok_or_else(|| {
                            compilation(format!("compute consumes unavailable resource {id:?}"))
                        })?,
                    )?,
                ));
            }
        }
        bindings.sort_unstable_by_key(|(id, _)| *id);
        self.channel.compute_resources = Some(resources);
        if let Some(old) = &self.channel.prepared_compute
            && Arc::ptr_eq(&old.shader, &code)
            && old.bindings.len() == bindings.len()
            && old
                .bindings
                .iter()
                .zip(&bindings)
                .all(|((id, a), (other, b))| id == other && same_binding(a, b))
        {
            return Ok(NativeWork {
                frontend,
                command: NativeCommand::Dispatch {
                    program: Arc::clone(old),
                    groups: launch.workgroups(),
                },
            });
        }
        let identity = Self::identity(&mut self.next);
        let program = Arc::new(GuestProgram {
            identity,
            shader: code,
            bindings: bindings.into_boxed_slice(),
            shared_memory_bytes: 0,
        });
        self.channel.prepared_compute = Some(Arc::clone(&program));
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Dispatch {
                program,
                groups: launch.workgroups(),
            },
        })
    }
    fn lower_solid(
        &mut self,
        request: &crate::engines::MaxwellTwoDSolidOperation,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, Error> {
        let resources = self
            .channel
            .resources
            .resolve_solid_image(request, space, 4096, access)
            .map_err(Error::ThreeDResource)?;
        let role = if request.zeta {
            MaxwellThreeDResourceRole::DepthStencilTarget
        } else {
            MaxwellThreeDResourceRole::BlitDestination
        };
        let image = lowering(resolved_image(&resources, role))?;
        let value = lowering(resolve::solid_clear_value(
            request.color,
            request.destination.format,
            image.guest_format(),
        ))?;
        let [x, y, x1, y1] = request.rectangle;
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Clear {
                image: Self::image(&mut self.image_views, &mut self.next, image)?,
                rectangle: nixe_gpu::ScissorRect {
                    x,
                    y,
                    width: x1 - x,
                    height: y1 - y,
                },
                value,
            },
        })
    }
    fn lower_blit(
        &mut self,
        request: &crate::MaxwellTwoDBlitOperation,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<NativeWork, Error> {
        use super::super::twod::blit::BlitKind;
        let resources = self
            .channel
            .resources
            .resolve_color_images(request, space, 4096, access)
            .map_err(Error::ThreeDResource)?;
        let source = Self::image(
            &mut self.image_views,
            &mut self.next,
            lowering(resolved_image(
                &resources,
                MaxwellThreeDResourceRole::BlitSource,
            ))?,
        )?;
        let destination = Self::image(
            &mut self.image_views,
            &mut self.next,
            lowering(resolved_image(
                &resources,
                MaxwellThreeDResourceRole::BlitDestination,
            ))?,
        )?;
        let rect = |x, y, width, height| nixe_gpu::ScissorRect {
            x,
            y,
            width,
            height,
        };
        let (source_rectangle, destination_rectangle, linear, resolve) = match request.kind {
            BlitKind::Copy { origins, extent } | BlitKind::Convert { origins, extent } => (
                rect(origins[0].x, origins[0].y, extent.width, extent.height),
                rect(origins[1].x, origins[1].y, extent.width, extent.height),
                false,
                source.description.samples() != nixe_gpu::SampleCount::One,
            ),
            BlitKind::Downsample2x { filter } => {
                let destination_extent = destination.description.extent();
                let resolve = source.description.samples() != nixe_gpu::SampleCount::One;
                let source_extent = if resolve {
                    destination_extent
                } else {
                    source.description.extent()
                };
                (
                    rect(0, 0, source_extent.width, source_extent.height),
                    rect(0, 0, destination_extent.width, destination_extent.height),
                    filter == nixe_gpu::BlitFilter::Linear,
                    resolve,
                )
            }
            BlitKind::Scale {
                filter,
                first,
                step,
                destination_origin,
                extent,
            } => {
                for axis in 0..2 {
                    let count = if axis == 0 {
                        extent.width
                    } else {
                        extent.height
                    };
                    let end = i128::from(first[axis])
                        + i128::from(step[axis]) * i128::from(count.saturating_sub(1));
                    if i64::try_from(end).is_err()
                        || first[axis].checked_sub(1 << 31).is_none()
                        || i64::try_from(end - (1_i128 << 31)).is_err()
                    {
                        return Err(Error::Lowering(MaxwellLoweringError::UnsupportedBlit(
                            "fixed-point sampling exceeds its signed 64-bit accumulator",
                        )));
                    }
                }
                let extent_source = source.description.extent();
                (
                    rect(0, 0, extent_source.width, extent_source.height),
                    rect(
                        destination_origin.x,
                        destination_origin.y,
                        extent.width,
                        extent.height,
                    ),
                    filter == nixe_gpu::BlitFilter::Linear,
                    false,
                )
            }
        };
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Blit {
                source,
                destination,
                source_rectangle,
                destination_rectangle,
                linear,
                resolve,
                sampling: match request.kind {
                    BlitKind::Scale { first, step, .. } => Some((first, step)),
                    _ => None,
                },
            },
        })
    }
    fn render_enabled(&self) -> bool {
        self.channel.render_enabled
    }
    fn update_render_condition(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), Error> {
        use MaxwellThreeDRenderEnableMode as Mode;
        let condition = state.render_enable();
        let mode = condition
            .execution_mode()
            .ok_or_else(|| compilation("invalid render condition mode or override"))?;
        // Evaluate in command order when SET_RENDER_ENABLE_C executes,
        // and retain the result until the next condition command. An unchanged
        // mode is still an evaluation; override writes and ordinary draws do
        // not reevaluate it or reread query RAM.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
        self.channel.render_enabled = match mode {
            Mode::Enabled => true,
            Mode::Disabled => false,
            _ => {
                let address = condition
                    .address()
                    .ok_or_else(|| compilation("render condition address is incomplete"))?;
                let mut bytes = [0; 24];
                space
                    .read_into(
                        space.address(address.get()).map_err(compilation)?,
                        &mut bytes,
                        access,
                    )
                    .map_err(compilation)?;
                let words = std::array::from_fn::<_, 6, _>(|i| {
                    u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
                });
                match mode {
                    Mode::Conditional => words[0] != 0 && words[1] != 0,
                    Mode::RenderIfEqual => words[0] == words[4] && words[1] == words[5],
                    Mode::RenderIfNotEqual => words[0] != words[4] || words[1] != words[5],
                    _ => unreachable!(),
                }
            }
        };
        Ok(())
    }
    fn take_mme_scratch(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.parameters)
    }
    fn recycle_mme_scratch(&mut self, parameters: Vec<u32>) {
        self.parameters = parameters;
    }
    fn lower_nv12_conversion(
        &mut self,
        destination: nixe_gpu::ConversionImage,
        source: nixe_gpu::Nv12Planes,
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<NativeWork, MaxwellLoweringError> {
        Ok(NativeWork {
            frontend: FrontendSubmissionId::new(u64::MAX - Self::identity(&mut self.next).slot()),
            command: NativeCommand::ConvertNv12 {
                destination: Box::new(self.conversion_image(destination)?),
                source,
                transform,
            },
        })
    }
}

impl Drop for MaxwellNativeFrontend {
    fn drop(&mut self) {
        if let Some(archive) = &self.capture_archive {
            // Address-indexed proofs retain only the current code version.
            // Persist every surviving canonical variant, including code that
            // was replaced at the same GPU VA before owner shutdown.
            for (_, inputs) in self.captured_codes.iter() {
                archive.persist(inputs);
            }
        }
        log::info!("native Maxwell preparation: {:?}", self.statistics);
    }
}

#[cfg(test)]
mod producer_tests {
    use super::*;
    use crate::MaxwellGpuAddressSpace;
    use crate::engines::twod::blit::{BlitKind, BlitSurface, BlitSurfaceLayout};
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };
    use nixe_memory::MemoryPermissions;

    #[test]
    fn channel_preparation_retains_distinct_bindings_with_equal_local_revisions() {
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        use crate::engines::tests::{program_three_d, three_d_channel};
        let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
        let mut channels = [three_d_channel(), three_d_channel()];
        for (index, channel) in channels.iter_mut().enumerate() {
            for (method, value) in [
                (0x2380, 64),
                (0x2384, (address >> 32) as u32),
                (0x2388, address as u32 + index as u32 * 64),
                (0x2410, 1),
            ] {
                program_three_d(channel, method, value);
            }
        }
        let role = MaxwellThreeDResourceRole::ConstantBuffer { group: 0, slot: 0 };
        let mut frontend = MaxwellNativeFrontend::new(Default::default());
        let shader = Arc::new(GuestShaderCode {
            identity: nixe_gpu::shader_identity::ContentId::of(b"binding-selection", &[]),
            stages: Box::new([]),
        });
        let mut selections = Vec::new();
        let mut programs = Vec::new();
        for index in [0, 1, 0, 1] {
            frontend.bind_channel(crate::MaxwellChannelId::new(index as u64 + 1));
            let selected = frontend
                .channel
                .resources
                .select(
                    &MaxwellThreeDResourceSelection::new(&[role], &[]),
                    Default::default(),
                )
                .resolve(
                    channels[index].three_d(),
                    space.memory(),
                    4096,
                    &mut crate::test_memory::access(space.memory().machine()),
                )
                .unwrap();
            let MaxwellThreeDResolvedResource::Buffer { view, .. } =
                selected.resource(role).unwrap()
            else {
                panic!("constant buffer")
            };
            assert_eq!(view.source().offset().get(), address + index as u64 * 64);
            if selections.len() < 2 {
                selections.push(selected.clone());
                programs.push(
                    MaxwellNativeFrontend::native_program(
                        &mut frontend.programs,
                        &mut frontend.channel.program,
                        &mut frontend.statistics,
                        &mut frontend.image_views,
                        &mut frontend.next,
                        &shader,
                        &[interface::ShaderBinding {
                            id: ResourceId::new(0),
                            role,
                            location: crate::engines::threed::MaxwellThreeDBindingLocation {
                                group: 1,
                                index: 0,
                                sampler: false,
                            },
                        }],
                        &selections[index],
                    )
                    .unwrap(),
                );
            }
        }
        assert!(!Arc::ptr_eq(&programs[0], &programs[1]));
        let entries = frontend.programs.usage();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        for index in (0..100).map(|i| i % 2) {
            frontend.bind_channel(crate::MaxwellChannelId::new(index as u64 + 1));
            let selected = MaxwellNativeFrontend::native_program(
                &mut frontend.programs,
                &mut frontend.channel.program,
                &mut frontend.statistics,
                &mut frontend.image_views,
                &mut frontend.next,
                &shader,
                &[interface::ShaderBinding {
                    id: ResourceId::new(0),
                    role,
                    location: crate::engines::threed::MaxwellThreeDBindingLocation {
                        group: 1,
                        index: 0,
                        sampler: false,
                    },
                }],
                &selections[index],
            )
            .unwrap();
            assert!(Arc::ptr_eq(&selected, &programs[index]));
        }
        let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
        assert_eq!(allocations, 0);
        assert_eq!(entries, frontend.programs.usage());
    }

    #[test]
    fn sampler_changes_preserve_current_native_images_without_general_cache_entries() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        use crate::engines::threed::{
            MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
        };
        let (allocation, space, address) = crate::shader::test_support::mapped_memory();
        let mut channel = three_d_channel();
        for (method, value) in [
            (0x2380, 64),
            (0x2384, (address >> 32) as u32),
            (0x2388, address as u32),
            (0x2410, 1),
            (0x1574, (address >> 32) as u32),
            (0x1578, address as u32 + 64),
            (0x157c, 0),
            (0x155c, (address >> 32) as u32),
            (0x1560, address as u32 + 128),
            (0x1564, 0),
        ] {
            program_three_d(&mut channel, method, value);
        }
        let descriptor = |words: [u32; 8]| {
            words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let format = 0x08
            | (2 << 7)
            | (2 << 10)
            | (2 << 13)
            | (2 << 16)
            | (2 << 19)
            | (3 << 22)
            | (4 << 25)
            | (5 << 28);
        crate::test_memory::write(
            &allocation,
            64,
            &descriptor([
                format,
                address as u32 + 256,
                (address >> 32) as u32 | (2 << 21),
                1,
                1 << 23,
                1 << 31,
                0,
                0,
            ]),
        )
        .unwrap();
        crate::test_memory::write(
            &allocation,
            128,
            &descriptor([0, 2 | (2 << 4) | (3 << 6), (15 * 256) << 12, 0, 0, 0, 0, 0]),
        )
        .unwrap();
        let texture = MaxwellThreeDTextureReference::new(0, 0, 0);
        let roles = [
            MaxwellThreeDResourceRole::SampledImage {
                texture,
                dimension: MaxwellThreeDTextureDimension::Two,
                with_sampler: true,
            },
            MaxwellThreeDResourceRole::Sampler(texture),
        ];
        let shader = Arc::new(GuestShaderCode {
            identity: nixe_gpu::shader_identity::ContentId::of(b"native-slot-retention", &[]),
            stages: Box::new([]),
        });
        let mut frontend = MaxwellNativeFrontend::new(Default::default());
        // The bounded cache cannot retain this view. Current channel bindings
        // must still retain their native interpretation and physical lifetime.
        frontend.image_views = nixe_gpu::WorkingSet::new(1, 1);
        let prepare = |frontend: &mut MaxwellNativeFrontend| {
            let resources = frontend
                .channel
                .resources
                .select(
                    &MaxwellThreeDResourceSelection::new(&roles, &[]),
                    Default::default(),
                )
                .resolve(
                    channel.three_d(),
                    space.memory(),
                    4096,
                    &mut crate::test_memory::access(space.memory().machine()),
                )
                .unwrap();
            let bindings = std::array::from_fn::<_, 2, _>(|index| interface::ShaderBinding {
                id: ResourceId::new(index as u32),
                role: roles[index],
                location: resources.binding_location(roles[index]).unwrap(),
            });
            MaxwellNativeFrontend::native_program(
                &mut frontend.programs,
                &mut frontend.channel.program,
                &mut frontend.statistics,
                &mut frontend.image_views,
                &mut frontend.next,
                &shader,
                &bindings,
                resources,
            )
            .unwrap()
        };
        let first = prepare(&mut frontend);
        assert_eq!(frontend.image_views.usage(), (0, 0));
        let GuestBinding::Image { image: initial, .. } = &first.bindings[0].1 else {
            panic!("sampled image")
        };
        let GuestBinding::Sampler(initial_sampler) = &first.bindings[1].1 else {
            panic!("sampler")
        };
        assert_eq!(initial_sampler.mag_filter, nixe_gpu::FilterMode::Linear);
        // Mutable TSC bytes change without a register write. Only the sampler
        // binding changes; the old immutable worker payload remains valid.
        crate::test_memory::write(
            &allocation,
            132,
            &(1_u32 | (2 << 4) | (3 << 6)).to_le_bytes(),
        )
        .unwrap();
        let second = prepare(&mut frontend);
        assert!(!Arc::ptr_eq(&first, &second));
        let GuestBinding::Image {
            image: retained, ..
        } = &second.bindings[0].1
        else {
            panic!("sampled image")
        };
        assert_eq!(initial.view_identity, retained.view_identity);
        let GuestBinding::Sampler(changed_sampler) = &second.bindings[1].1 else {
            panic!("sampler")
        };
        assert_eq!(changed_sampler.mag_filter, nixe_gpu::FilterMode::Nearest);
        assert_eq!(initial_sampler.mag_filter, nixe_gpu::FilterMode::Linear);
        // Texel content belongs to visibility, independently of native bindings.
        crate::test_memory::write(&allocation, 256, &[1, 2, 3, 4]).unwrap();
        assert!(Arc::ptr_eq(&second, &prepare(&mut frontend)));
    }

    #[test]
    fn render_condition_is_evaluated_at_commands_and_retained_per_channel() {
        use crate::engines::tests::{program_three_d, three_d_channel};
        let (allocation, space, address) = crate::shader::test_support::mapped_memory();
        let mut channel = three_d_channel();
        let mut frontend = MaxwellNativeFrontend::new(Default::default());
        frontend.bind_channel(crate::MaxwellChannelId::new(1));
        program_three_d(&mut channel, 0x1550, (address >> 32) as u32);
        program_three_d(&mut channel, 0x1554, address as u32);
        for (mode, words, expected) in [
            (0, [1, 1, 0, 0, 2, 2], false),
            (1, [0; 6], true),
            (2, [1, 1, 0, 0, 0, 0], true),
            (2, [1, 0, 0, 0, 0, 0], false),
            (3, [3, 7, 99, 88, 3, 7], true),
            (3, [3, 7, 99, 88, 3, 8], false),
            (4, [3, 7, 99, 88, 3, 7], false),
            (4, [3, 7, 99, 88, 4, 7], true),
        ] {
            let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            crate::test_memory::write(&allocation, 0, &bytes).unwrap();
            program_three_d(&mut channel, 0x1558, mode);
            frontend
                .update_render_condition(
                    channel.three_d(),
                    space.memory(),
                    &mut crate::test_memory::access(space.memory().machine()),
                )
                .unwrap();
            assert_eq!(frontend.channel.render_enabled, expected, "mode={mode}");
        }
        program_three_d(&mut channel, 0x1558, 4);
        crate::test_memory::write(&allocation, 0, &[0; 24]).unwrap();
        frontend
            .update_render_condition(
                channel.three_d(),
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert!(!frontend.channel.render_enabled);
        // Later memory changes do not change the decision before another
        // condition command.
        crate::test_memory::write(&allocation, 0, &1_u32.to_le_bytes()).unwrap();
        assert!(!frontend.render_enabled());
        frontend
            .update_render_condition(
                channel.three_d(),
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert!(frontend.channel.render_enabled);
        // An override write leaves the retained decision alone until the next
        // mode command. Its forced result does not read query bytes.
        program_three_d(&mut channel, 0x1944, 2);
        assert!(frontend.render_enabled());
        program_three_d(&mut channel, 0x1558, 4);
        frontend
            .update_render_condition(channel.three_d(), space.memory(), &mut |_, _| {
                panic!("override must not read memory")
            })
            .unwrap();
        assert!(!frontend.channel.render_enabled);
        frontend.bind_channel(crate::MaxwellChannelId::new(2));
        assert!(frontend.channel.render_enabled);
        frontend.bind_channel(crate::MaxwellChannelId::new(1));
        assert!(!frontend.channel.render_enabled);
    }

    #[test]
    fn alternating_native_shader_sources_share_code_and_observe_committed_changes() {
        use crate::shader::test_support::{mapped_memory, write_shader_words};
        let (allocation, space, address) = mapped_memory();
        let mut vertex = [0u32; 20];
        vertex[0] = 0x0002_0461;
        let mut pixel = [0u32; 20];
        pixel[0] = 0x0002_5462;
        pixel[18] = 1;
        let encode = |header: [u32; 20], instruction: u64| {
            header
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .chain(
                    [instruction, 0xe300_0000_0007_000f, 0, 0]
                        .into_iter()
                        .flat_map(u64::to_le_bytes),
                )
                .collect::<Vec<_>>()
        };
        let mov = |bits: u32| 0x0100_0000_0007_f000 | u64::from(bits) << 20;
        crate::test_memory::write(&allocation, 0, &encode(vertex, 0xe300_0000_0007_000f)).unwrap();
        crate::test_memory::write(&allocation, 0x100, &encode(pixel, mov(1f32.to_bits()))).unwrap();
        crate::test_memory::write(&allocation, 0x200, &encode(pixel, mov(2f32.to_bits()))).unwrap();
        let mut channel = crate::engines::tests::three_d_channel();
        for (method, value) in [
            (0x2140, 0x50),
            (0x192c, 1),
            (0x1608, (address >> 32) as u32),
            (0x160c, address as u32),
            (0x2000, 0x11),
            (0x2004, 0),
            (0x200c, 4),
            (0x2040, 0x51),
            (0x2044, 0x100),
            (0x204c, 4),
        ] {
            crate::engines::tests::program_three_d(&mut channel, method, value);
        }
        let mut frontend = MaxwellNativeFrontend::new(Default::default());
        frontend.bind_channel(crate::MaxwellChannelId::new(1));
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(
                channel.three_d(),
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        let code = Arc::clone(&frontend.channel.code.as_ref().unwrap().code);
        frontend.bind_channel(crate::MaxwellChannelId::new(2));
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(channel.three_d(), space.memory(), &mut |_, _| {
                panic!("shared capture should be current")
            })
            .unwrap();
        assert!(Arc::ptr_eq(
            &code,
            &frontend.channel.code.as_ref().unwrap().code
        ));
        frontend.close_channel(crate::MaxwellChannelId::new(2));
        frontend.bind_channel(crate::MaxwellChannelId::new(2));
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(channel.three_d(), space.memory(), &mut |_, _| {
                panic!("closing a channel must not evict shared code")
            })
            .unwrap();
        assert!(Arc::ptr_eq(
            &code,
            &frontend.channel.code.as_ref().unwrap().code
        ));
        assert_eq!(frontend.statistics.shader_captures, 1);
        frontend.bind_channel(crate::MaxwellChannelId::new(1));
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(channel.three_d(), space.memory(), &mut |_, _| {
                panic!("switching back must preserve the channel's current code")
            })
            .unwrap();
        assert!(Arc::ptr_eq(
            &code,
            &frontend.channel.code.as_ref().unwrap().code
        ));
        // Rebinding an unused uniform slot does not reconstruct capture requests
        // or touch shader memory, even though the CB register group is dirty.
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        let mut no_read = |_, _| panic!("unchanged code must not read guest memory");
        for (method, value) in [
            (0x2380, 64),
            (0x2384, (address >> 32) as u32),
            (0x2388, address as u32 + 0x800),
        ] {
            crate::engines::tests::program_three_d(&mut channel, method, value);
        }
        let mut allocations = 0;
        for i in 0..1000 {
            crate::engines::tests::program_three_d(&mut channel, 0x2410, (i % 2) << 4 | 1);
            ALLOCATIONS.with(|n| n.set(Some(0)));
            frontend
                .ensure_code(channel.three_d(), space.memory(), &mut no_read)
                .unwrap();
            allocations += ALLOCATIONS.with(|n| n.replace(None).unwrap());
        }
        assert_eq!(allocations, 0);
        assert_eq!(frontend.statistics.shader_captures, 1);
        assert_eq!(frontend.statistics.revalidated_captures, 0);
        let first = Arc::clone(&frontend.channel.code.as_ref().unwrap().code);
        let interface = Arc::clone(&frontend.channel.code.as_ref().unwrap().interface);
        assert_eq!(first.stages.len(), 2);
        crate::engines::tests::program_three_d(&mut channel, 0x2044, 0x200);
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(
                channel.three_d(),
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert!(!Arc::ptr_eq(
            &first,
            &frontend.channel.code.as_ref().unwrap().code
        ));
        crate::engines::tests::program_three_d(&mut channel, 0x2044, 0x100);
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(
                channel.three_d(),
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert!(Arc::ptr_eq(
            &first,
            &frontend.channel.code.as_ref().unwrap().code
        ));
        assert!(Arc::ptr_eq(
            &interface,
            &frontend.channel.code.as_ref().unwrap().interface
        ));
        assert_eq!(frontend.statistics.shader_captures, 2);
        let word = mov(3f32.to_bits());
        write_shader_words(
            space.memory(),
            &[
                (address + 0x150, word as u32),
                (address + 0x154, (word >> 32) as u32),
            ],
        );
        crate::engines::tests::program_three_d(&mut channel, 0x0d78, 3);
        frontend
            .ensure_code(
                channel.three_d(),
                space.memory(),
                &mut crate::test_memory::access(space.memory().machine()),
            )
            .unwrap();
        assert_ne!(
            first.identity,
            frontend.channel.code.as_ref().unwrap().code.identity
        );
        assert_eq!(frontend.statistics.shader_captures, 3);
    }

    #[test]
    fn single_sample_downsample_is_a_filtered_blit_with_distinct_rectangles() {
        let machine = crate::test_memory::machine();
        let mut space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mut addresses = Vec::new();
        for i in 1..=2 {
            let storage = crate::test_memory::allocate(&machine, 0x10000);
            let address = space
                .map(MaxwellMapRequest {
                    allocation: MaxwellAllocationId::new(i),
                    backing: crate::test_memory::ranges(&storage),
                    backing_offset: 0,
                    size: 0x10000,
                    allocation_alignment: 4096,
                    page_size: 0,
                    kind: 0xfe,
                    cacheable: true,
                    fixed_offset: None,
                    permissions: MemoryPermissions::READ_WRITE,
                })
                .unwrap()
                .offset()
                .get();
            addresses.push(address);
        }
        let mut frontend = MaxwellNativeFrontend::new(Default::default());
        let mut channel = crate::engines::tests::three_d_channel();
        let source = crate::engines::tests::program_three_d(&mut channel, 0x19d0, 0x3c);
        let request = crate::MaxwellTwoDBlitOperation {
            source,
            images: [
                BlitSurface {
                    address: addresses[0],
                    width: 32,
                    height: 16,
                    format: 0xcf,
                    layout: BlitSurfaceLayout::BlockLinear(0),
                },
                BlitSurface {
                    address: addresses[1],
                    width: 16,
                    height: 8,
                    format: 0xcf,
                    layout: BlitSurfaceLayout::BlockLinear(0),
                },
            ],
            destination_compression: false,
            kind: BlitKind::Downsample2x {
                filter: nixe_gpu::BlitFilter::Linear,
            },
        };
        let work = frontend
            .lower_blit(
                &request,
                space.memory(),
                FrontendSubmissionId::new(1),
                &mut crate::test_memory::apply,
            )
            .unwrap();
        let NativeCommand::Blit {
            source_rectangle,
            destination_rectangle,
            linear,
            resolve,
            ..
        } = work.command
        else {
            panic!("expected native blit")
        };
        assert_eq!([source_rectangle.width, source_rectangle.height], [32, 16]);
        assert_eq!(
            [destination_rectangle.width, destination_rectangle.height],
            [16, 8]
        );
        assert!(linear);
        assert!(!resolve);
    }
}

#[cfg(test)]
mod clear_tests {
    use super::*;
    #[test]
    fn warm_single_clear_uses_no_resource_or_command_snapshots() {
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        use crate::engines::tests::{program_three_d, three_d_channel};
        let (_allocation, space, address) = crate::shader::test_support::mapped_memory();
        let mut channel = three_d_channel();
        for (method, value) in [
            (0x0800, (address >> 32) as u32),
            (0x0804, address as u32),
            (0x0808, 16),
            (0x080c, 4),
            (0x0810, 0xd5),
            (0x0814, 0x1000),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x0d6c, 4 << 16),
            (0x0d70, 4 << 16),
            (0x0d80, 0),
            (0x0d84, 0),
            (0x0d88, 0),
            (0x0d8c, 0),
            (0x1534, 0),
        ] {
            program_three_d(&mut channel, method, value);
        }
        let source = program_three_d(&mut channel, 0x19d0, 0x3c);
        let mut frontend = MaxwellNativeFrontend::new(Default::default());
        let mut access = crate::test_memory::access(space.memory().machine());
        let trigger = MaxwellThreeDOperationTrigger::ClearSurface { source };
        frontend
            .lower_three_d(
                channel.three_d(),
                trigger,
                space.memory(),
                FrontendSubmissionId::new(1),
                None,
                &mut access,
            )
            .unwrap();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        let work = frontend
            .lower_three_d(
                channel.three_d(),
                trigger,
                space.memory(),
                FrontendSubmissionId::new(2),
                None,
                &mut access,
            )
            .unwrap();
        let count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
        println!("AUDIT FOLLOWUP warmed_color_clear allocations={count}");
        assert!(matches!(work.command, NativeCommand::Clear { .. }));
        assert_eq!(count, 0);
    }
}
