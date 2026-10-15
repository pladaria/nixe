//! Native frontend preparation. Maxwell dirty revisions and mapping/CPU proofs
//! select immutable semantic state; no renderer IDs or old register IR are built.
use super::*;
mod image;
use crate::execution::{MaxwellFrontend, MaxwellSubmissionExecutionError as Error};
use crate::{
    MaxwellGpuAddressSpace, MaxwellShaderCompiler, MaxwellShaderSnapshot, MaxwellSsaBindingSource,
};
use nixe_gpu::native::*;
use nixe_gpu::ssa::{ResourceId, compile::CompilePool};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeFrontendStatistics {
    pub draws: u64,
    pub prepared_draws: u64,
    pub shader_captures: u64,
    pub warm_draws: u64,
}
struct Code {
    state: super::super::threed::state::MaxwellThreeDShaderStateIdentity,
    source: crate::shader::MaxwellShaderTranslationSourceKey,
    snapshot: MaxwellShaderSnapshot,
    translated: Arc<crate::MaxwellShaderCode>,
    code: Arc<GuestShaderCode>,
    bindings: Box<[(ResourceId, MaxwellThreeDResourceRole)]>,
    sizes: Box<[(MaxwellThreeDResourceRole, MaxwellShaderBufferSize)]>,
    roles: Vec<MaxwellThreeDResourceRole>,
}
struct Prepared {
    state: super::super::threed::state::MaxwellThreeDDrawStateIdentity,
    resources: Arc<MaxwellThreeDResolvedResources>,
    code: Arc<GuestShaderCode>,
    indexed: bool,
    draw: Arc<GuestDrawState>,
    accesses: Arc<[nixe_gpu::BackingView]>,
    roles: Box<[MaxwellThreeDResourceRole]>,
}
pub struct MaxwellNativeFrontend {
    presentation: Vec<(nixe_gpu::PresentationImageRequest, NativePresentationImage)>,
    compiler: MaxwellShaderCompiler,
    resources: super::super::threed::MaxwellThreeDResolvedResourceCache,
    draw_roles: Vec<MaxwellThreeDResourceRole>,
    code: Option<Code>,
    prepared: Option<Prepared>,
    next: u64,
    methods: Vec<crate::MaxwellMethodDispatch>,
    parameters: Vec<u32>,
    pub statistics: NativeFrontendStatistics,
    image_view_bytes: usize,
    image_views: Vec<(
        super::super::threed::MaxwellThreeDGuestImageFormat,
        nixe_gpu::ImageView,
        GuestImage,
    )>,
    compute_request: Option<(u64, u8, Option<[u32; 3]>)>,
    prepared_compute: Option<(Arc<GuestProgram>, Arc<[nixe_gpu::BackingView]>)>,
    compute: Option<(
        MaxwellShaderSnapshot,
        Arc<crate::MaxwellShaderCode>,
        Arc<GuestShaderCode>,
    )>,
}
fn compilation(error: impl Display) -> Error {
    Error::Native(error.to_string().into())
}
fn lowering<T>(result: Result<T, MaxwellLoweringError>) -> Result<T, Error> {
    result.map_err(Error::Lowering)
}
fn ready<T>(pending: nixe_gpu::ssa::compile::Pending<T>) -> Result<Arc<T>, Error> {
    loop {
        if let Some(value) = pending.wait_timeout(Duration::from_millis(20)) {
            return value.map_err(|e| compilation(&*e));
        }
    }
}
impl MaxwellNativeFrontend {
    pub fn new(pool: Arc<CompilePool>) -> Self {
        Self {
            presentation: Vec::new(),
            compiler: MaxwellShaderCompiler::new(pool, 2048, 64 * 1024 * 1024),
            resources: Default::default(),
            draw_roles: Vec::new(),
            code: None,
            prepared: None,
            next: 1,
            methods: Vec::new(),
            parameters: Vec::new(),
            statistics: Default::default(),
            image_views: Vec::new(),
            image_view_bytes: 0,
            compute: None,
            compute_request: None,
            prepared_compute: None,
        }
    }
    fn compile(
        &self,
        snapshot: MaxwellShaderSnapshot,
    ) -> Result<Arc<crate::MaxwellShaderCode>, Error> {
        loop {
            match self.compiler.request(snapshot.clone()) {
                Ok(p) => return ready(p),
                Err(
                    nixe_gpu::ssa::compile::CompileError::Backpressure
                    | nixe_gpu::ssa::compile::CompileError::Budget,
                ) => std::thread::sleep(Duration::from_millis(1)),
                Err(e) => return Err(compilation(e)),
            }
        }
    }
    fn identity(&mut self) -> nixe_gpu::PreparedStateId {
        let id = self.next;
        self.next = self
            .next
            .checked_add(1)
            .expect("native prepared identity exhausted");
        nixe_gpu::PreparedStateId::new(id, 1)
    }
    fn ensure_code(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
    ) -> Result<(), Error> {
        let current = self.code.as_ref().is_some_and(|c| {
            c.snapshot.source_is_current(space)
                && c.snapshot.staged_writes_are_irrelevant(writes)
                && c.snapshot
                    .indirect_bindings_current(state.shader_bindings())
        });
        if current && self.code.as_ref().unwrap().state.matches(state) {
            return Ok(());
        }
        let source = crate::shader::prepare_maxwell_shader_translation_source(state)
            .map_err(Error::ShaderTranslation)?;
        if current
            && let Some(code) = &mut self.code
            && code.source == source
        {
            // Attribute offsets/stream layouts can change while the consumed
            // shader input types stay identical. Retain code and its resource
            // reflection; code/table bytes still need their freshness proof.
            code.state = state.shader_state_identity();
            return Ok(());
        }
        let snapshot = MaxwellShaderSnapshot::capture_source(&source, space, writes)
            .map_err(Error::ShaderTranslation)?;
        self.statistics.shader_captures += 1;
        let translated = self.compile(snapshot.clone())?;
        let code = if let Some(old) = &self.code
            && old.code.identity == snapshot.identity()
        {
            Arc::clone(&old.code)
        } else {
            Arc::new(GuestShaderCode {
                identity: snapshot.identity(),
                functions: translated
                    .programs
                    .iter()
                    .map(|p| p.function.clone())
                    .collect(),
            })
        };
        let mut bindings = BTreeMap::new();
        let mut sizes = BTreeMap::new();
        for program in &translated.programs {
            let extents: BTreeMap<_, _> = program
                .global_buffer_extents([1; 3], [1; 3])
                .into_vec()
                .into_iter()
                .collect();
            for binding in &program.bindings {
                let group = binding
                    .binding_group
                    .ok_or_else(|| compilation("graphics binding has no guest group"))?;
                let texture = |offset| -> Result<_, Error> {
                    Ok(super::super::threed::MaxwellThreeDTextureReference::new(
                        group,
                        binding
                            .texture_constant_buffer_slot
                            .ok_or_else(|| compilation("texture has no handle CB slot"))?,
                        offset,
                    ))
                };
                let role = match binding.source {
                    MaxwellSsaBindingSource::ConstantBuffer { slot } => {
                        MaxwellThreeDResourceRole::ConstantBuffer { group, slot }
                    }
                    MaxwellSsaBindingSource::GlobalBuffer {
                        constant_buffer: slot,
                        byte_offset,
                        runtime_extent,
                    } => {
                        let role = MaxwellThreeDResourceRole::StorageBuffer {
                            group,
                            slot,
                            byte_offset,
                        };
                        let size = if runtime_extent {
                            MaxwellShaderBufferSize::Declared
                        } else {
                            MaxwellShaderBufferSize::Proven(
                                *extents.get(&binding.resource).ok_or_else(|| {
                                    compilation("storage access has no proved extent")
                                })?,
                            )
                        };
                        sizes
                            .entry(role)
                            .and_modify(|old: &mut MaxwellShaderBufferSize| *old = old.union(size))
                            .or_insert(size);
                        role
                    }
                    MaxwellSsaBindingSource::Image {
                        handle_byte_offset,
                        kind,
                    } => {
                        let dimension = match kind {
                            ShaderResourceKind::SampledImage
                            | ShaderResourceKind::SampledDepthImage => {
                                MaxwellThreeDTextureDimension::Two
                            }
                            ShaderResourceKind::SampledImage2DArray
                            | ShaderResourceKind::SampledDepthImage2DArray => {
                                MaxwellThreeDTextureDimension::TwoArray
                            }
                            ShaderResourceKind::SampledImageCube => {
                                MaxwellThreeDTextureDimension::Cube
                            }
                            _ => {
                                return Err(compilation(format!(
                                    "unsupported native Maxwell TIC class {kind:?}"
                                )));
                            }
                        };
                        MaxwellThreeDResourceRole::SampledImage {
                            texture: texture(handle_byte_offset)?,
                            dimension,
                        }
                    }
                    MaxwellSsaBindingSource::Sampler {
                        handle_byte_offset, ..
                    } => MaxwellThreeDResourceRole::Sampler(texture(handle_byte_offset)?),
                };
                bindings.insert(binding.resource, role);
            }
        }
        let mut roles: Vec<_> = bindings.values().copied().collect();
        roles.sort_unstable();
        roles.dedup();
        self.code = Some(Code {
            state: state.shader_state_identity(),
            source,
            snapshot,
            translated,
            code,
            bindings: bindings.into_iter().collect(),
            sizes: sizes.into_iter().collect(),
            roles,
        });
        Ok(())
    }
    fn image(
        &mut self,
        resources: &Arc<MaxwellThreeDResolvedResources>,
        index: usize,
    ) -> Result<GuestImage, Error> {
        let resolved = lowering(resolved_image(resources, index))?;
        let canonical_texels = resolved
            .guest_layout()
            .has_direct_canonical_representation();
        if let Some((_, _, image)) =
            self.image_views
                .iter()
                .rev()
                .find(|(encoding, view, image)| {
                    *encoding == resolved.guest_format()
                        && image.description == resolved.description()
                        && image.canonical_texels == canonical_texels
                        && view.swizzle() == resolved.view().swizzle()
                        && view.bindings() == resolved.view().bindings()
                })
        {
            return Ok(image.clone());
        }
        let image = GuestImage {
            sample_grid: image::sample_grid(resolved.description().samples())?,
            description: resolved.description(),
            swizzle: resolved.view().swizzle(),
            canonical_texels,
            planes: image::planes(resolved)?,
        };
        let size = plane_metadata_bytes(&image);
        const IMAGE_VIEW_BYTES: usize = 512 * 1024 * 1024;
        while !self.image_views.is_empty()
            && (self.image_views.len() >= 4096
                || self.image_view_bytes.saturating_add(size) > IMAGE_VIEW_BYTES)
        {
            let (_, _, old) = self.image_views.remove(0);
            self.image_view_bytes -= plane_metadata_bytes(&old);
        }
        if size > IMAGE_VIEW_BYTES {
            return Err(compilation(
                "native image address metadata exceeds the 512 MiB frontend cache budget",
            ));
        }
        self.image_view_bytes += size;
        self.image_views.push((
            resolved.guest_format(),
            resolved.view().clone(),
            image.clone(),
        ));
        Ok(image)
    }
    fn draw(
        &mut self,
        state: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
    ) -> Result<NativeWork, Error> {
        if state.shader_execution().stream_output_enable().value() == Some(&true) {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedStreamOutputSemantics,
            ));
        }
        self.statistics.draws += 1;
        self.ensure_code(state, space, writes)?;
        let code = self.code.as_ref().unwrap();
        let roles = if let Some(p) = &self.prepared
            && p.state.matches(state)
            && Arc::ptr_eq(&p.code, &code.code)
            && p.indexed == trigger.is_indexed()
        {
            p.roles.as_ref()
        } else {
            self.draw_roles.clear();
            self.draw_roles.extend_from_slice(&code.roles);
            trigger.append_resource_roles(state, &mut self.draw_roles);
            self.draw_roles.sort_unstable();
            self.draw_roles.dedup();
            self.draw_roles.as_slice()
        };
        let resources = self
            .resources
            .resolve_with_buffer_sizes(state, space, roles, &code.sizes, Some(writes), false, 4096)
            .map_err(Error::ThreeDResource)?;
        let prepared_roles = roles;
        let arguments = lowering(native_arguments(state, trigger))?;
        if let Some(p) = &self.prepared
            && p.state.matches(state)
            && Arc::ptr_eq(&p.resources, &resources)
            && Arc::ptr_eq(&p.code, &code.code)
            && p.indexed == trigger.is_indexed()
        {
            self.statistics.warm_draws += 1;
            return Ok(NativeWork {
                frontend,
                command: NativeCommand::Draw {
                    state: Arc::clone(&p.draw),
                    arguments,
                },
                canonical_accesses: Arc::clone(&p.accesses),
            });
        }
        let prepared_roles: Box<[_]> = prepared_roles.into();
        let shader = Arc::clone(&code.code);
        let translated = Arc::clone(&code.translated);
        let binding_roles = code.bindings.clone();
        let (tessellation, raster) = lowering(validate_consumed_state(state, trigger))?;
        if state
            .render_enable()
            .execution_mode()
            .is_some_and(|m| m != MaxwellThreeDRenderEnableMode::Enabled)
        {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedRenderEnableMode(
                    state.render_enable().execution_mode().unwrap(),
                ),
            ));
        }
        if state.shader_execution().stream_output_enable().value() == Some(&true) {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedStreamOutputSemantics,
            ));
        }
        let raster = raster.unwrap();
        let mut selection = lowering(select_draw_attachments(state, &resources))?;
        selection.color_outputs =
            lowering(color::draw_color_outputs(state, &resources, &selection))?;
        lowering(validate_draw_iterated_blend_state(state, &selection))?;
        lowering(validate_draw_logic_op_state(state, &selection))?;
        let bindings = self.native_bindings(&binding_roles, &resources)?;
        let colors: Box<_> = selection
            .colors
            .iter()
            .map(|(_, i)| self.image(&resources, *i))
            .collect::<Result<_, _>>()?;
        let depth_stencil = selection
            .depth_stencil
            .map(|i| self.image(&resources, i))
            .transpose()?;
        let mut extent = [u32::MAX; 2];
        for image in colors.iter().chain(depth_stencil.iter()) {
            extent[0] = extent[0].min(image.description.extent().width);
            extent[1] = extent[1].min(image.description.extent().height);
        }
        if extent.contains(&u32::MAX) {
            return Err(compilation("draw has no attachment extent"));
        }
        let mut vertices = Vec::new();
        for stream in consumed_vertex_streams(state) {
            let s = &state.vertex_input().streams()[stream as usize];
            let format = s
                .format()
                .value()
                .filter(|f| f.enabled())
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
            let attributes = state
                .vertex_input()
                .attributes()
                .iter()
                .enumerate()
                .filter_map(|(location, attr)| {
                    attr.value()
                        .filter(|a| {
                            a.enabled()
                                && a.stream() == stream
                                && state.vertex_input().attribute_skip_mask(location as u8) != 15
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
            let index = lowering(resource_index(
                &resources,
                MaxwellThreeDResourceRole::VertexStream(stream),
            ))?;
            let buffer = lowering(resolved_buffer(&resources, index))?;
            vertices.push(GuestVertexBuffer {
                binding: u32::from(stream),
                backing: buffer.view().backing().range().clone(),
                stride: u32::from(format.stride()),
                step: if instanced {
                    VertexStepMode::Instance
                } else {
                    VertexStepMode::Vertex
                },
                attributes,
            });
        }
        let index = if trigger.is_indexed() {
            let i = lowering(resource_index(
                &resources,
                MaxwellThreeDResourceRole::IndexBuffer,
            ))?;
            Some((
                lowering(resolved_buffer(&resources, i))?
                    .view()
                    .backing()
                    .range()
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
        let depth = if depth_stencil.is_some() {
            lowering(draw_depth_state(state))?
        } else {
            DepthState::DISABLED
        };
        let pixel_depth = depth.test_enabled
            || translated
                .programs
                .iter()
                .filter(|p| p.function.stage == ShaderStage::Fragment)
                .any(|p| {
                    p.function.inputs.iter().any(|i| {
                        i.location() == nixe_gpu::ShaderIoLocation::Position && i.component() == 2
                    }) || p
                        .function
                        .outputs
                        .iter()
                        .any(|i| i.location() == nixe_gpu::ShaderIoLocation::FragmentDepth)
                });
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
        let program = GuestProgram {
            identity: self.identity(),
            shader: Arc::clone(&shader),
            bindings,
            shared_memory_bytes: 0,
            local_memory_bytes: 0,
        };
        let draw = Arc::new(GuestDrawState {
            program,
            vertices: vertices.into_boxed_slice(),
            index,
            colors,
            depth_stencil,
            extent,
            topology: lowering(primitive_topology(
                *state.vertex_input().primitive().active_begin().unwrap(),
            ))?,
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
            scissor: lowering(draw_scissor(state, &resources, &selection))?,
            depth,
            stencil: lowering(draw_stencil_test_state(state))?,
            color_outputs: selection
                .colors
                .iter()
                .map(|(target, _)| selection.color_outputs[*target as usize])
                .collect(),
            blend_constants,
            alpha_test: lowering(draw_alpha_test_state(state))?,
            tessellation,
            sample_mask: u64::MAX,
            alpha_to_coverage: false,
        });
        let accesses = accesses(&resources);
        self.statistics.prepared_draws += 1;
        self.prepared = Some(Prepared {
            state: state.draw_state_identity(),
            resources,
            code: shader,
            indexed: trigger.is_indexed(),
            draw: Arc::clone(&draw),
            accesses: Arc::clone(&accesses),
            roles: prepared_roles,
        });
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Draw {
                state: draw,
                arguments,
            },
            canonical_accesses: accesses,
        })
    }
}

fn plane_metadata_bytes(image: &GuestImage) -> usize {
    image
        .planes
        .iter()
        .map(|p| std::mem::size_of::<GuestImagePlane>() + p.backing.metadata_bytes())
        .sum()
}

fn accesses(resources: &MaxwellThreeDResolvedResources) -> Arc<[nixe_gpu::BackingView]> {
    resources
        .resources()
        .iter()
        .flat_map(|r| match r {
            MaxwellThreeDResolvedResource::Buffer(b) => vec![b.view().backing().clone()],
            MaxwellThreeDResolvedResource::Image(i) => i
                .view()
                .bindings()
                .iter()
                .map(|b| b.backing().clone())
                .collect(),
        })
        .collect()
}
impl MaxwellNativeFrontend {
    fn native_bindings(
        &mut self,
        roles: &[(ResourceId, MaxwellThreeDResourceRole)],
        resources: &Arc<MaxwellThreeDResolvedResources>,
    ) -> Result<Box<[(ResourceId, GuestBinding)]>, Error> {
        roles
            .iter()
            .map(|&(id, role)| {
                let binding = if matches!(role, MaxwellThreeDResourceRole::Sampler(_)) {
                    let sampler = resources
                        .samplers()
                        .iter()
                        .find(|s| s.role() == role)
                        .ok_or_else(|| compilation("absent consumed sampler"))?;
                    GuestBinding::Sampler(sampler.description().map_err(Error::ThreeDResource)?)
                } else {
                    let index = lowering(resource_index(resources, role))?;
                    match &resources.resources()[index] {
                        MaxwellThreeDResolvedResource::Buffer(b) => {
                            GuestBinding::Buffer(b.view().backing().range().clone())
                        }
                        MaxwellThreeDResolvedResource::Image(_) => {
                            GuestBinding::Image(self.image(resources, index)?)
                        }
                    }
                };
                Ok((id, binding))
            })
            .collect()
    }
}

fn native_arguments(
    state: &MaxwellThreeDState,
    trigger: MaxwellThreeDOperationTrigger,
) -> Result<DrawArguments, MaxwellLoweringError> {
    if let MaxwellThreeDOperationTrigger::DrawIndexBuffer { index_count, .. } = trigger {
        let input = state.vertex_input();
        let first_index = *input
            .index()
            .first()
            .value()
            .ok_or(MaxwellLoweringError::IncompleteDraw("SET_INDEX_BUFFER_F"))?;
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
    type Work = NativeWork;
    fn lower_presentation(
        &mut self,
        request: nixe_gpu::PresentationImageRequest,
    ) -> Result<NativePresentationImage, Error> {
        if let Some((_, image)) = self.presentation.iter().find(|(key, _)| {
            key.backing == request.backing
                && key.width == request.width
                && key.height == request.height
                && key.format == request.format
                && key.layout == request.layout
                && key.row_pitch == request.row_pitch
        }) {
            let mut image = image.clone();
            image.allow_canonical_import = request.allow_canonical_import;
            return Ok(image);
        }
        let planes = image::presentation_planes(&request)?;
        let image = NativePresentationImage {
            planes,
            extent: [request.width, request.height],
            format: request.format,
            allow_canonical_import: request.allow_canonical_import,
        };
        if self.presentation.len() == 16 {
            self.presentation.remove(0);
        }
        self.presentation.push((request, image.clone()));
        Ok(image)
    }
    fn lower_three_d(
        &mut self,
        state: &MaxwellThreeDState,
        trigger: MaxwellThreeDOperationTrigger,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        _predecessors: &[FrontendSubmissionId],
    ) -> Result<NativeWork, Error> {
        if let Some(mode) = state.render_enable().execution_mode()
            && mode != MaxwellThreeDRenderEnableMode::Enabled
        {
            return Err(Error::Lowering(
                MaxwellLoweringError::UnsupportedRenderEnableMode(mode),
            ));
        }
        if trigger.is_draw() {
            return self.draw(state, trigger, space, writes, frontend);
        }
        lowering(validate_consumed_state(state, trigger))?;
        let mut roles = Vec::new();
        trigger.append_resource_roles(state, &mut roles);
        let resources = self
            .resources
            .resolve(state, space, &roles, Some(writes), false, 4096)
            .map_err(Error::ThreeDResource)?;
        let clear = state.render_targets().clear();
        let surface = *clear
            .last_surface()
            .value()
            .ok_or_else(|| compilation("CLEAR_SURFACE state is incomplete"))?;
        if clear.last_surface().source() != Some(trigger.source()) {
            return Err(Error::Lowering(MaxwellLoweringError::TriggerStateMismatch));
        }
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
        for (index, resource) in resources.resources().iter().enumerate() {
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
                for (out, c) in color.iter_mut().zip(clear.color()) {
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
            canonical_accesses: accesses(&resources),
        })
    }
    fn lower_launch(
        &mut self,
        launch: &super::super::compute::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        _predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<NativeWork, Error> {
        launch.validate_execution().map_err(Error::ComputeLaunch)?;
        let request = launch.native_request();
        // Source relocation never retains the old mapping proof.
        let request_key = (
            request.address,
            request.register_count,
            request.workgroup_size,
        );
        let reuse = self.compute_request == Some(request_key)
            && self.compute.as_ref().is_some_and(|(s, _, _)| {
                s.source_is_current(space) && s.staged_writes_are_irrelevant(writes)
            });
        if !reuse {
            let snapshot = MaxwellShaderSnapshot::capture_compute(space, writes, request)
                .map_err(Error::ShaderTranslation)?;
            let translated = self.compile(snapshot.clone())?;
            let code = Arc::new(GuestShaderCode {
                identity: snapshot.identity(),
                functions: translated
                    .programs
                    .iter()
                    .map(|p| p.function.clone())
                    .collect(),
            });
            self.compute = Some((snapshot, translated, code));
            self.compute_request = Some(request_key);
        }
        let (_, translated, code) = self.compute.as_ref().unwrap();
        let resources = launch
            .native_resources(&translated.programs[0], space, writes)
            .map_err(Error::ComputeLaunch)?;
        let mut bindings = Vec::new();
        let mut canonical = Vec::new();
        for (resource, range) in resources {
            let retained = lowering(buffer::retain(&range))?;
            bindings.push((
                resource,
                GuestBinding::Buffer(retained.backing.range().clone()),
            ));
            canonical.push(retained.backing);
        }
        if let Some((old, accesses)) = &self.prepared_compute
            && Arc::ptr_eq(&old.shader, code)
            && old.bindings.len() == bindings.len()
            && old
                .bindings
                .iter()
                .zip(&bindings)
                .all(|((id, a), (other, b))| {
                    id == other
                        && matches!((a,b),(GuestBinding::Buffer(a),GuestBinding::Buffer(b)) if a==b)
                })
        {
            return Ok(NativeWork {
                frontend,
                command: NativeCommand::Dispatch {
                    program: Arc::clone(old),
                    groups: launch.workgroups(),
                },
                canonical_accesses: Arc::clone(accesses),
            });
        }
        let code = Arc::clone(code);
        let identity = self.identity();
        let program = Arc::new(GuestProgram {
            identity,
            shader: code,
            bindings: bindings.into_boxed_slice(),
            shared_memory_bytes: 0,
            local_memory_bytes: 0,
        });
        let canonical_accesses: Arc<[_]> = canonical.into();
        self.prepared_compute = Some((Arc::clone(&program), Arc::clone(&canonical_accesses)));
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Dispatch {
                program,
                groups: launch.workgroups(),
            },
            canonical_accesses,
        })
    }
    fn lower_solid(
        &mut self,
        request: &crate::engines::MaxwellTwoDSolidOperation,
        space: &MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        _predecessors: &[FrontendSubmissionId],
    ) -> Result<NativeWork, Error> {
        let resources = self
            .resources
            .resolve_solid_image(request, space, 4096)
            .map_err(Error::ThreeDResource)?;
        let image = lowering(resolved_image(&resources, 0))?;
        let value = lowering(resolve::solid_clear_value(
            request.color,
            request.destination.format,
            image.guest_format(),
        ))?;
        let [x, y, x1, y1] = request.rectangle;
        Ok(NativeWork {
            frontend,
            command: NativeCommand::Clear {
                image: self.image(&resources, 0)?,
                rectangle: nixe_gpu::ScissorRect {
                    x,
                    y,
                    width: x1 - x,
                    height: y1 - y,
                },
                value,
            },
            canonical_accesses: accesses(&resources),
        })
    }
    fn lower_blit(
        &mut self,
        request: &crate::MaxwellTwoDBlitOperation,
        space: &MaxwellGpuAddressSpace,
        frontend: FrontendSubmissionId,
        _predecessors: &[FrontendSubmissionId],
    ) -> Result<NativeWork, Error> {
        use super::super::twod::blit::BlitKind;
        let resources = self
            .resources
            .resolve_color_images(request, space, 4096)
            .map_err(Error::ThreeDResource)?;
        let source = self.image(&resources, 0)?;
        let destination = self.image(&resources, 1)?;
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
                let extent = destination.description.extent();
                (
                    rect(0, 0, extent.width, extent.height),
                    rect(0, 0, extent.width, extent.height),
                    filter == nixe_gpu::BlitFilter::Linear,
                    true,
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
            canonical_accesses: accesses(&resources),
        })
    }
    fn take_mme_scratch(&mut self) -> (Vec<crate::MaxwellMethodDispatch>, Vec<u32>) {
        (
            std::mem::take(&mut self.methods),
            std::mem::take(&mut self.parameters),
        )
    }
    fn recycle_mme_scratch(
        &mut self,
        methods: Vec<crate::MaxwellMethodDispatch>,
        parameters: Vec<u32>,
    ) {
        self.methods = methods;
        self.parameters = parameters;
    }
    fn lower_nv12_conversion(
        &mut self,
        images: [nixe_gpu::ConversionImage; 3],
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<NativeWork, MaxwellLoweringError> {
        let canonical_accesses = images
            .iter()
            .flat_map(|i| i.view.bindings().iter().map(|b| b.backing().clone()))
            .collect();
        Ok(NativeWork {
            frontend: FrontendSubmissionId::new(u64::MAX - self.next),
            command: NativeCommand::ConvertNv12 {
                images: Box::new(images.map(|i| GuestImage {
                    sample_grid: [1, 1],
                    planes: Arc::from([]),
                    description: i.description,
                    canonical_texels: i.view.has_canonical_texels(),
                    swizzle: i.view.swizzle(),
                })),
                transform,
            },
            canonical_accesses,
        })
    }
    fn lower_decoded_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        bytes: [Arc<[u8]>; 2],
        widths: [u32; 2],
    ) -> Result<NativeWork, MaxwellLoweringError> {
        let canonical_accesses = images
            .iter()
            .flat_map(|i| i.view.bindings().iter().map(|b| b.backing().clone()))
            .collect();
        Ok(NativeWork {
            frontend: FrontendSubmissionId::new(u64::MAX - self.next),
            command: NativeCommand::UploadPlanes {
                images: Box::new(images.map(|i| GuestImage {
                    sample_grid: [1, 1],
                    planes: Arc::from([]),
                    description: i.description,
                    canonical_texels: i.view.has_canonical_texels(),
                    swizzle: i.view.swizzle(),
                })),
                bytes,
                widths,
            },
            canonical_accesses,
        })
    }
    fn lower_device_plane_uploads(
        &mut self,
        _images: [nixe_gpu::ConversionImage; 2],
        _frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<NativeWork, MaxwellLoweringError> {
        Err(MaxwellLoweringError::UnsupportedBlit(
            "external video producer integration is task 13",
        ))
    }
}

impl Drop for MaxwellNativeFrontend {
    fn drop(&mut self) {
        log::info!("native Maxwell preparation: {:?}", self.statistics);
    }
}
