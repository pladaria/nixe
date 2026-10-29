//! Native frontend preparation. Maxwell dirty revisions and mapping/CPU proofs
//! select immutable semantic state; no renderer IDs or old register IR are built.
use super::*;
mod compute;
mod identity;
mod image;
mod interface;
use crate::MaxwellGpuMemory;
use crate::execution::{MaxwellFrontend, MaxwellSubmissionExecutionError as Error};
use identity::{ImageKey, ParameterKey, ProgramKey};
use interface::ShaderInterface;
use nixe_gpu::native::*;
use nixe_gpu::shader_module::ResourceId;

type GraphicsSourceKey = [Option<(nixe_shader_recompiler::Stage, u64, Option<u8>, u32)>;
    crate::MAXWELL_PIPELINE_SHADER_COUNT];

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
    pub warm_draws: u64,
}
struct Code {
    state: super::super::threed::state::MaxwellThreeDShaderStateIdentity,
    address_space: crate::MaxwellAddressSpaceId,
    mapping_generation: u64,
    code_generation: u64,
    source: GraphicsSourceKey,
    snapshot: Arc<crate::recompiler::CapturedProgram>,
    code: Arc<GuestShaderCode>,
    interface: Arc<ShaderInterface>,
}
struct Prepared {
    state: super::super::threed::state::MaxwellThreeDDrawStateIdentity,
    resources: MaxwellThreeDResolvedResources,
    code: Arc<GuestShaderCode>,
    indexed: bool,
    draw: Arc<GuestDrawState>,
    roles: Arc<[MaxwellThreeDResourceRole]>,
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
    draw_roles: Vec<MaxwellThreeDResourceRole>,
    code: Option<Code>,
    prepared: Option<Prepared>,
    prepared_compute: Option<Arc<GuestProgram>>,
    compute_resources: Option<super::super::compute::ComputeResources>,
    compute: Option<Arc<crate::recompiler::CapturedProgram>>,
}
pub struct MaxwellNativeFrontend {
    presentation: Vec<(PresentationKey, Arc<[GuestImagePlane]>)>,
    source_codes: nixe_gpu::WorkingSet<
        (crate::MaxwellAddressSpaceId, GraphicsSourceKey),
        Arc<crate::recompiler::CapturedProgram>,
    >,
    interfaces: nixe_gpu::WorkingSet<nixe_gpu::shader_identity::ContentId, Arc<ShaderInterface>>,
    parameter_states: nixe_gpu::WorkingSet<ParameterKey, Arc<GuestDrawParameters>>,
    programs: nixe_gpu::WorkingSet<ProgramKey, Arc<GuestProgram>>,
    draws: nixe_gpu::WorkingSet<Arc<GuestDrawState>, Arc<GuestDrawState>>,
    next: u64,
    parameters: Vec<u32>,
    pub statistics: NativeFrontendStatistics,
    image_views: nixe_gpu::WorkingSet<ImageKey, GuestImage>,
    capture_compiler: Option<nixe_shader_recompiler::Compiler>,
    capture_memory: crate::recompiler::MemoryCache,
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
            draw_roles: Vec::new(),
            code: None,
            prepared: None,
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
            draws: nixe_gpu::WorkingSet::new(8192, 64 * 1024 * 1024),
            next: 1,
            parameters: Vec::new(),
            statistics: Default::default(),
            image_views: nixe_gpu::WorkingSet::new(4096, IMAGE_VIEW_BYTES),
            capture_compiler: None,
            capture_memory: Default::default(),
            configuration,
            compute_sources: nixe_gpu::WorkingSet::new(256, 8 * 1024 * 1024),
            captured_codes: nixe_gpu::WorkingSet::new(2048, 64 * 1024 * 1024),

            channel,
            current_channel: None,
            channels: Default::default(),
        }
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
        self.remember_image(key, &image)
            .map_err(|error| MaxwellLoweringError::ImageBacking(error.to_string()))?;
        Ok(image)
    }
    fn identity(&mut self) -> nixe_gpu::PreparedStateId {
        let id = self.next;
        self.next = id
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
        let mapping_generation = space.mapping_generation().get();
        let code_generation = space.machine().code_generation();
        if let Some(code) = &mut self.channel.code
            && code.state.matches_code(state)
            && code.address_space == space.id()
        {
            if code.mapping_generation == mapping_generation
                && code.code_generation == code_generation
            {
                return Ok(());
            }
            if code
                .snapshot
                .stages
                .iter()
                .all(|stage| stage.memory_is_current(space))
            {
                code.mapping_generation = mapping_generation;
                code.code_generation = code_generation;
                return Ok(());
            }
        }
        let mut source: GraphicsSourceKey = [None; crate::MAXWELL_PIPELINE_SHADER_COUNT];
        let mut requests = Vec::new();
        for (index, source_slot) in source.iter_mut().enumerate() {
            let pipeline = state.shader_bindings().pipeline(index);
            if pipeline.enabled().value() != Some(&true) {
                continue;
            }
            let request = crate::recompiler::graphics_request(state, index).map_err(compilation)?;
            *source_slot = Some((
                request.metadata.stage,
                request.address,
                pipeline.effective_group(),
                request.metadata.texture_bound_buffer.0,
            ));
            requests.push((pipeline.effective_group(), request));
        }
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
            .or_else(|| self.source_codes.get(&(space.id(), source)).cloned());
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
        let snapshot = if let Some(snapshot) = cached {
            snapshot
        } else if let Some(snapshot) = refreshed {
            self.statistics.revalidated_captures += 1;
            let snapshot = Arc::new(snapshot);
            self.source_codes.insert(
                (space.id(), source),
                Arc::clone(&snapshot),
                snapshot.retained_bytes(),
            );
            snapshot
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
            let weight = snapshot.retained_bytes();
            self.source_codes
                .insert((space.id(), source), Arc::clone(&snapshot), weight);
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
        self.channel.code = Some(Code {
            state: state.shader_state_identity(),
            address_space: space.id(),
            mapping_generation,
            code_generation,
            source,
            snapshot,
            code,
            interface,
        });
        Ok(())
    }
    fn image(
        &mut self,
        resources: &MaxwellThreeDResolvedResources,
        index: MaxwellThreeDResourceRole,
    ) -> Result<GuestImage, Error> {
        let resolved = lowering(resolved_image(resources, index))?;
        let canonical_texels = resolved
            .guest_layout()
            .has_direct_canonical_representation();
        let key = ImageKey {
            encoding: resolved.guest_format(),
            description: resolved.description(),
            canonical_texels,
            view: resolved.view().clone(),
        };
        if let Some(image) = self.image_views.get(&key) {
            return Ok(image.clone());
        }
        let image = GuestImage {
            view_identity: Some(self.identity()),
            sample_grid: image::sample_grid(resolved.description().samples())?,
            description: resolved.description(),
            swizzle: resolved.view().swizzle(),
            canonical_texels,
            planes: image::planes(resolved)?,
        };
        self.remember_image(key, &image)?;
        Ok(image)
    }
    fn remember_image(&mut self, key: ImageKey, image: &GuestImage) -> Result<(), Error> {
        let size = plane_metadata_bytes(image)
            + std::mem::size_of::<ImageKey>()
            + std::mem::size_of_val(key.view.bindings());
        if size > IMAGE_VIEW_BYTES {
            return Err(compilation(
                "native image address metadata exceeds the 512 MiB frontend cache budget",
            ));
        }
        self.image_views.insert(key, image.clone(), size);
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
        let reuse_roles = self.channel.prepared.as_ref().is_some_and(|p| {
            p.state.matches_parameters(state)
                && Arc::ptr_eq(&p.code, &code.code)
                && p.indexed == trigger.is_indexed()
                && p.draw
                    .index
                    .as_ref()
                    .is_some_and(|i| matches!(i, GuestIndexBuffer::Inline(_)))
                    == state.inline_index().is_some()
        });
        let roles = if reuse_roles {
            self.channel.prepared.as_ref().unwrap().roles.as_ref()
        } else {
            self.channel.draw_roles.clear();
            self.channel
                .draw_roles
                .extend_from_slice(&code.interface.roles);
            trigger.append_resource_roles(state, &mut self.channel.draw_roles);
            self.channel.draw_roles.sort_unstable();
            self.channel.draw_roles.dedup();
            self.channel.draw_roles.as_slice()
        };
        let resources = self
            .channel
            .resources
            .resolve_with_buffer_sizes(state, space, roles, &code.interface.sizes, 4096, access)
            .map_err(Error::ThreeDResource)?;
        let prepared_roles = roles;
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
            && p.resources.same_bindings(resources)
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
        let resources = resources.clone();
        let prepared_roles = if reuse_roles {
            Arc::clone(&self.channel.prepared.as_ref().unwrap().roles)
        } else {
            Arc::from(prepared_roles)
        };
        let shader = Arc::clone(&code.code);
        let shader_pixel_depth = code.interface.pixel_depth;
        let binding_roles = Arc::clone(&code.interface.bindings);
        let mut selection = None;
        let (colors, depth_stencil) = if let Some(previous) = &self.channel.prepared
            && previous.resources.same_group(&resources, 3)
            && previous.state.matches_parameters(state)
        {
            (
                Arc::clone(&previous.draw.colors),
                previous.draw.depth_stencil.clone(),
            )
        } else {
            let selected = selection.insert(lowering(select_draw_attachments(state, &resources))?);
            let colors: Arc<[_]> = selected
                .colors
                .iter()
                .map(|(_, role)| self.image(&resources, *role))
                .collect::<Result<_, _>>()?;
            let depth = selected
                .depth_stencil
                .map(|role| self.image(&resources, role))
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
            if selection.is_none() {
                selection = Some(lowering(select_draw_attachments(state, &resources))?);
            }
            let selection = selection.as_ref().unwrap();
            let (tessellation, raster) = lowering(validate_consumed_state(state, trigger))?;
            if state.shader_execution().stream_output_enable().value() == Some(&true) {
                return Err(Error::Lowering(
                    MaxwellLoweringError::UnsupportedStreamOutputSemantics,
                ));
            }
            let raster = raster.unwrap();
            let color_outputs = lowering(color::draw_color_outputs(state, &resources, selection))?;
            lowering(validate_draw_iterated_blend_state(state, selection))?;
            lowering(validate_draw_logic_op_state(state, selection))?;
            let mut vertices = Vec::new();
            for stream in consumed_vertex_streams(state) {
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
                let attributes = (0..32)
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
            let viewport = lowering(draw_viewport_transform_with_depth_usage(state, pixel_depth))?;
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
                    base_instance: state
                        .vertex_input()
                        .assembly()
                        .global_base_instance_index()
                        .value()
                        .copied()
                        .unwrap_or(0),
                    point_size_bits: state
                        .raster()
                        .point_size()
                        .value()
                        .map(|value| value.bits()),
                    force_early_z: state.shader_execution().api_mandated_early_z().value()
                        == Some(&crate::MaxwellThreeDApiMandatedEarlyZ::Enabled),
                    y_negate: matches!(state.fixed_function().register(MaxwellThreeDFixedFunctionRegister::WindowOrigin).value(), Some(MaxwellThreeDFixedFunctionValue::Mask(origin)) if origin & 1 != 0),
                },
                identity: self.identity(),
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
                restart: state.vertex_input().primitive().restart_enabled().value() == Some(&true),
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
                scissor: lowering(draw_scissor(state, &resources, selection))?,
                depth,
                stencil: lowering(draw_stencil_test_state(state))?,
                color_outputs,
                blend_constants,
                alpha_test: lowering(draw_alpha_test_state(state))?,
                tessellation,
                sample_mask: u64::MAX,
                alpha_to_coverage: false,
            };
            let candidate = Arc::new(parameters);
            let key = ParameterKey(Arc::clone(&candidate));
            if let Some(parameters) = self.parameter_states.get(&key) {
                self.statistics.reused_parameters += 1;
                Arc::clone(parameters)
            } else {
                self.statistics.prepared_parameters += 1;
                self.parameter_states.insert(
                    key,
                    Arc::clone(&candidate),
                    identity::parameter_bytes(&candidate),
                );
                candidate
            }
        };
        let program = self.native_program(&shader, &binding_roles, &resources)?;
        let vertices = if let Some(previous) = &self.channel.prepared
            && previous.draw.parameters.vertices == parameters.vertices
            && parameters.vertices.iter().all(|vertex| {
                previous.resources.same_resource(
                    &resources,
                    MaxwellThreeDResourceRole::VertexStream(vertex.binding as u8),
                )
            }) {
            Arc::clone(&previous.draw.vertices)
        } else {
            parameters
                .vertices
                .iter()
                .map(|layout| {
                    let index = MaxwellThreeDResourceRole::VertexStream(layout.binding as u8);
                    Ok(GuestVertexBuffer {
                        binding: layout.binding,
                        backing: lowering(resolved_buffer(&resources, index))?
                            .view()
                            .backing()
                            .mapping()
                            .clone(),
                    })
                })
                .collect::<Result<Arc<[_]>, Error>>()?
        };
        let index = if let Some(bytes) = state.inline_index() {
            Some(GuestIndexBuffer::Inline(Arc::clone(bytes)))
        } else if trigger.is_indexed() {
            let i = MaxwellThreeDResourceRole::IndexBuffer;
            Some(GuestIndexBuffer::Canonical(
                lowering(resolved_buffer(&resources, i))?
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
        let draw = GuestDrawState {
            identity: self.identity(),
            program,
            vertices,
            index,
            colors,
            depth_stencil,
            parameters,
        };
        let draw = if let Some(cached) = self.draws.get(&draw) {
            self.statistics.warm_draws += 1;
            Arc::clone(cached)
        } else {
            self.statistics.prepared_draws += 1;
            let draw = Arc::new(draw);
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
            resources,
            code: shader,
            indexed: trigger.is_indexed(),
            draw: Arc::clone(&draw),
            roles: prepared_roles,
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
        &mut self,
        resources: &MaxwellThreeDResolvedResources,
        role: MaxwellThreeDResourceRole,
    ) -> Result<GuestBinding, Error> {
        Ok(if matches!(role, MaxwellThreeDResourceRole::Sampler(_)) {
            let sampler = resources
                .sampler(role)
                .ok_or_else(|| compilation("absent consumed sampler"))?;
            GuestBinding::Sampler(sampler.description().map_err(Error::ThreeDResource)?)
        } else {
            let index = role;
            match resources
                .resource(role)
                .ok_or_else(|| compilation("absent consumed resource"))?
            {
                MaxwellThreeDResolvedResource::Buffer(b) => {
                    GuestBinding::Buffer(b.view().backing().mapping().clone())
                }
                MaxwellThreeDResolvedResource::Image(_) => GuestBinding::Image {
                    image: self.image(resources, index)?,
                    arrayed: matches!(
                        role,
                        MaxwellThreeDResourceRole::SampledImage {
                            dimension: super::super::threed::MaxwellThreeDTextureDimension::TwoArray
                                | super::super::threed::MaxwellThreeDTextureDimension::CubeArray,
                            ..
                        }
                    ),
                },
            }
        })
    }
    fn native_program(
        &mut self,
        shader: &Arc<GuestShaderCode>,
        roles: &[(ResourceId, MaxwellThreeDResourceRole)],
        resources: &MaxwellThreeDResolvedResources,
    ) -> Result<Arc<GuestProgram>, Error> {
        if let Some(previous) = &self.channel.prepared
            && Arc::ptr_eq(&previous.code, shader)
            && previous.resources.same_group(resources, 1)
            && previous.resources.same_group(resources, 2)
        {
            self.statistics.reused_programs += 1;
            return Ok(Arc::clone(&previous.draw.program));
        }
        let previous = self
            .channel
            .prepared
            .as_ref()
            .filter(|p| {
                Arc::ptr_eq(&p.code, shader) && p.draw.program.bindings.len() == roles.len()
            })
            .map(|p| Arc::clone(&p.draw.program));
        let mut changed = previous.is_none().then(|| Vec::with_capacity(roles.len()));
        for (index, &(id, role)) in roles.iter().enumerate() {
            let binding = self.binding(resources, role)?;
            if changed.is_none() {
                let old = &previous.as_ref().unwrap().bindings;
                if old[index].0 == id && same_binding(&old[index].1, &binding) {
                    continue;
                }
                // Allocate only at the first changed binding. Its prefix
                // was proved equal and can retain the previous objects.
                let mut bindings = Vec::with_capacity(roles.len());
                bindings.extend_from_slice(&old[..index]);
                changed = Some(bindings);
            }
            changed.as_mut().unwrap().push((id, binding));
        }
        if let Some(bindings) = changed {
            let candidate = Arc::new(GuestProgram {
                identity: self.identity(),
                shader: Arc::clone(shader),
                bindings: bindings.into_boxed_slice(),
                shared_memory_bytes: 0,
            });
            let key = ProgramKey(Arc::clone(&candidate));
            if let Some(program) = self.programs.get(&key) {
                self.statistics.reused_programs += 1;
                Ok(Arc::clone(program))
            } else {
                self.programs.insert(
                    key,
                    Arc::clone(&candidate),
                    identity::program_bytes(&candidate),
                );
                Ok(candidate)
            }
        } else {
            self.statistics.reused_programs += 1;
            Ok(previous.unwrap())
        }
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
        let mut roles = Vec::new();
        trigger.append_resource_roles(state, &mut roles);
        let resources = self
            .channel
            .resources
            .resolve(state, space, &roles, 4096, access)
            .map_err(Error::ThreeDResource)?
            .clone();
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
        let mut commands = Vec::new();
        for resource in resources.resources() {
            let index = resource.role();
            let MaxwellThreeDResolvedResource::Image(image) = resource else {
                continue;
            };
            let extent = image.description().extent();
            let r = regions.for_attachment(extent.width, extent.height);
            if r.is_empty() {
                continue;
            }
            let value = if matches!(image.role(), MaxwellThreeDResourceRole::ColorTarget(_)) {
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
                image: self.image(&resources, index)?,
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
            command: NativeCommand::Sequence(commands.into_boxed_slice()),
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
            for &(id, role) in &textures.bindings {
                bindings.push((id, self.binding(&textures.resources, role)?));
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
        let identity = self.identity();
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
                image: self.image(&resources, role)?,
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
        let source = self.image(&resources, MaxwellThreeDResourceRole::BlitSource)?;
        let destination = self.image(&resources, MaxwellThreeDResourceRole::BlitDestination)?;
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
        let mode = condition.execution_mode().unwrap_or(Mode::Enabled);
        // Evaluate in command order when SET_RENDER_ENABLE_C/OVERRIDE executes,
        // and retain the result until the next condition command. An unchanged
        // mode is still an evaluation; ordinary draws do not reread query RAM.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/engines/maxwell_3d.cpp
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
            frontend: FrontendSubmissionId::new(u64::MAX - self.identity().slot()),
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
        // Override does not consume an address or read query bytes.
        program_three_d(&mut channel, 0x1944, 2);
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
