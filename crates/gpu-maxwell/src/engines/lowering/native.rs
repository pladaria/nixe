//! Native frontend preparation. Maxwell dirty revisions and mapping/CPU proofs
//! select immutable semantic state; no renderer IDs or old register IR are built.
use super::*;
mod compute;
mod identity;
mod image;
mod interface;
use crate::execution::{MaxwellFrontend, MaxwellSubmissionExecutionError as Error};
use crate::{
    MaxwellGpuAddressSpace, MaxwellShaderCompiler, MaxwellShaderSnapshot, MaxwellSsaBindingSource,
};
use identity::{DrawKey, ImageKey, ParameterKey, ProgramKey};
use interface::ShaderInterface;
use nixe_gpu::native::*;
use nixe_gpu::ssa::{ResourceId, compile::CompilePool};
use std::time::Duration;

type ComputeSourceKey = (u64, u8, Option<[u32; 3]>);

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeFrontendStatistics {
    pub draws: u64,
    pub prepared_draws: u64,
    pub prepared_parameters: u64,
    pub reused_parameters: u64,
    pub reused_programs: u64,
    pub shader_captures: u64,
    pub warm_draws: u64,
}
struct Code {
    state: super::super::threed::state::MaxwellThreeDShaderStateIdentity,
    source: crate::shader::MaxwellShaderTranslationSourceKey,
    snapshot: Arc<MaxwellShaderSnapshot>,
    translated: Arc<crate::MaxwellShaderCode>,
    code: Arc<GuestShaderCode>,
    interface: Arc<ShaderInterface>,
}
struct Prepared {
    state: super::super::threed::state::MaxwellThreeDDrawStateIdentity,
    resources: Arc<MaxwellThreeDResolvedResources>,
    code: Arc<GuestShaderCode>,
    indexed: bool,
    draw: Arc<GuestDrawState>,
    accesses: Arc<[nixe_gpu::BackingView]>,
    roles: Arc<[MaxwellThreeDResourceRole]>,
}
pub struct MaxwellNativeFrontend {
    presentation: Vec<(nixe_gpu::PresentationImageRequest, NativePresentationImage)>,
    compiler: MaxwellShaderCompiler,
    resources: super::super::threed::MaxwellThreeDResolvedResourceCache,
    draw_roles: Vec<MaxwellThreeDResourceRole>,
    code: Option<Code>,
    source_codes: nixe_gpu::WorkingSet<
        crate::shader::MaxwellShaderTranslationSourceKey,
        (
            Arc<MaxwellShaderSnapshot>,
            std::sync::Weak<crate::MaxwellShaderCode>,
        ),
    >,
    prepared: Option<Prepared>,
    interfaces: nixe_gpu::WorkingSet<nixe_gpu::ssa::artifact::ContentId, Arc<ShaderInterface>>,
    parameter_states: nixe_gpu::WorkingSet<ParameterKey, Arc<GuestDrawParameters>>,
    programs: nixe_gpu::WorkingSet<ProgramKey, Arc<GuestProgram>>,
    draws: nixe_gpu::WorkingSet<DrawKey, Arc<GuestDrawState>>,
    next: u64,
    methods: Vec<crate::MaxwellMethodDispatch>,
    parameters: Vec<u32>,
    pub statistics: NativeFrontendStatistics,
    image_views: nixe_gpu::WorkingSet<ImageKey, GuestImage>,
    compute_request: Option<ComputeSourceKey>,
    compute_sources: nixe_gpu::WorkingSet<ComputeSourceKey, Arc<MaxwellShaderSnapshot>>,
    compute_variants: nixe_gpu::WorkingSet<
        (
            nixe_gpu::ssa::artifact::ContentId,
            nixe_gpu::ssa::UniformSelection,
        ),
        Arc<crate::MaxwellShaderCode>,
    >,
    prepared_compute: Option<(Arc<GuestProgram>, Arc<[nixe_gpu::BackingView]>)>,
    compute: Option<(Arc<MaxwellShaderSnapshot>, Arc<crate::MaxwellShaderCode>)>,
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
            source_codes: nixe_gpu::WorkingSet::new(2048, 32 * 1024 * 1024),
            prepared: None,
            interfaces: nixe_gpu::WorkingSet::new(2048, 8 * 1024 * 1024),
            parameter_states: nixe_gpu::WorkingSet::new(8192, 16 * 1024 * 1024),
            programs: nixe_gpu::WorkingSet::new(8192, 32 * 1024 * 1024),
            draws: nixe_gpu::WorkingSet::new(8192, 64 * 1024 * 1024),
            next: 1,
            methods: Vec::new(),
            parameters: Vec::new(),
            statistics: Default::default(),
            image_views: nixe_gpu::WorkingSet::new(4096, IMAGE_VIEW_BYTES),
            compute: None,
            compute_request: None,
            compute_sources: nixe_gpu::WorkingSet::new(256, 8 * 1024 * 1024),
            compute_variants: nixe_gpu::WorkingSet::new(256, 32 * 1024 * 1024),
            prepared_compute: None,
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
        let cached = self
            .source_codes
            .get(&source)
            .filter(|(snapshot, _)| {
                snapshot.source_is_current(space)
                    && snapshot.staged_writes_are_irrelevant(writes)
                    && snapshot.indirect_bindings_current(state.shader_bindings())
            })
            .cloned();
        let (snapshot, translated) = if let Some((snapshot, code)) = cached {
            let code = match code.upgrade() {
                Some(code) => code,
                None => {
                    let code = self.compile(snapshot.as_ref().clone())?;
                    self.source_codes.insert(
                        source.clone(),
                        (Arc::clone(&snapshot), Arc::downgrade(&code)),
                        snapshot.retained_bytes() + std::mem::size_of_val(&source),
                    );
                    code
                }
            };
            (snapshot, code)
        } else {
            let snapshot = Arc::new(
                MaxwellShaderSnapshot::capture_source(
                    &source,
                    state.shader_bindings(),
                    space,
                    writes,
                )
                .map_err(Error::ShaderTranslation)?,
            );
            self.statistics.shader_captures += 1;
            let translated = self.compile(snapshot.as_ref().clone())?;
            let weight = snapshot.retained_bytes() + std::mem::size_of_val(&source);
            self.source_codes.insert(
                source.clone(),
                (Arc::clone(&snapshot), Arc::downgrade(&translated)),
                weight,
            );
            (snapshot, translated)
        };
        let code = Arc::clone(&translated.native);
        let interface = match self.interfaces.get(&code.identity) {
            Some(interface) => Arc::clone(interface),
            None => {
                let interface = Arc::new(ShaderInterface::new(&translated)?);
                self.interfaces.insert(
                    code.identity,
                    Arc::clone(&interface),
                    interface.retained_bytes(),
                );
                interface
            }
        };
        self.code = Some(Code {
            state: state.shader_state_identity(),
            source,
            snapshot,
            translated,
            code,
            interface,
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
        let reuse_roles = self.prepared.as_ref().is_some_and(|p| {
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
            self.prepared.as_ref().unwrap().roles.as_ref()
        } else {
            self.draw_roles.clear();
            self.draw_roles.extend_from_slice(&code.interface.roles);
            trigger.append_resource_roles(state, &mut self.draw_roles);
            self.draw_roles.sort_unstable();
            self.draw_roles.dedup();
            self.draw_roles.as_slice()
        };
        let resources = self
            .resources
            .resolve_with_buffer_sizes(
                state,
                space,
                roles,
                &code.interface.sizes,
                Some(writes),
                false,
                4096,
            )
            .map_err(Error::ThreeDResource)?;
        let prepared_roles = roles;
        let arguments = lowering(native_arguments(state, trigger))?;
        if let Some(p) = &self.prepared
            && p.state.matches(state)
            && Arc::ptr_eq(&p.resources, &resources)
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
                canonical_accesses: Arc::clone(&p.accesses),
            });
        }
        let prepared_roles = if reuse_roles {
            Arc::clone(&self.prepared.as_ref().unwrap().roles)
        } else {
            Arc::from(prepared_roles)
        };
        let shader = Arc::clone(&code.code);
        let translated = Arc::clone(&code.translated);
        let binding_roles = Arc::clone(&code.interface.bindings);
        let mut selection = lowering(select_draw_attachments(state, &resources))?;
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
        let parameters = if let Some(prepared) = &self.prepared
            && prepared.state.matches_parameters(state)
            && Arc::ptr_eq(&prepared.code, &shader)
            && prepared.indexed == trigger.is_indexed()
        {
            self.statistics.reused_parameters += 1;
            Arc::clone(&prepared.draw.parameters)
        } else {
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
            selection.color_outputs =
                lowering(color::draw_color_outputs(state, &resources, &selection))?;
            lowering(validate_draw_iterated_blend_state(state, &selection))?;
            lowering(validate_draw_logic_op_state(state, &selection))?;
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
            let pixel_depth = depth.test_enabled
                || translated
                    .programs
                    .iter()
                    .filter(|p| p.function.stage == ShaderStage::Fragment)
                    .any(|p| {
                        p.function.inputs.iter().any(|i| {
                            i.location() == nixe_gpu::ShaderIoLocation::Position
                                && i.component() == 2
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
            let parameters = GuestDrawParameters {
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
        let vertices = parameters
            .vertices
            .iter()
            .map(|layout| {
                let index = lowering(resource_index(
                    &resources,
                    MaxwellThreeDResourceRole::VertexStream(layout.binding as u8),
                ))?;
                Ok(GuestVertexBuffer {
                    binding: layout.binding,
                    backing: lowering(resolved_buffer(&resources, index))?
                        .view()
                        .backing()
                        .range()
                        .clone(),
                })
            })
            .collect::<Result<_, Error>>()?;
        let index = if let Some(bytes) = state.inline_index() {
            Some(GuestIndexBuffer::Inline(Arc::clone(bytes)))
        } else if trigger.is_indexed() {
            let i = lowering(resource_index(
                &resources,
                MaxwellThreeDResourceRole::IndexBuffer,
            ))?;
            Some(GuestIndexBuffer::Canonical(
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
        let draw = Arc::new(GuestDrawState {
            identity: self.identity(),
            program,
            vertices,
            index,
            colors,
            depth_stencil,
            parameters,
        });
        let key = DrawKey(Arc::clone(&draw));
        let draw = if let Some(cached) = self.draws.get(&key) {
            self.statistics.warm_draws += 1;
            Arc::clone(cached)
        } else {
            self.statistics.prepared_draws += 1;
            self.draws
                .insert(key, Arc::clone(&draw), identity::draw_bytes(&draw));
            draw
        };
        // Canonical proofs come from this resolution, independently of the
        // immutable physical bindings retained by the draw template.
        let accesses = accesses(&resources);
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

const IMAGE_VIEW_BYTES: usize = 512 * 1024 * 1024;

fn plane_metadata_bytes(image: &GuestImage) -> usize {
    image
        .planes
        .iter()
        .map(|p| std::mem::size_of::<GuestImagePlane>() + p.backing.metadata_bytes())
        .sum()
}

fn accesses(resources: &MaxwellThreeDResolvedResources) -> Arc<[nixe_gpu::BackingView]> {
    let mut accesses = Vec::with_capacity(resources.resources().len());
    for resource in resources.resources() {
        match resource {
            MaxwellThreeDResolvedResource::Buffer(b) => accesses.push(b.view().backing().clone()),
            MaxwellThreeDResolvedResource::Image(i) => {
                accesses.extend(i.view().bindings().iter().map(|b| b.backing().clone()));
            }
        }
    }
    accesses.into()
}
impl MaxwellNativeFrontend {
    fn native_program(
        &mut self,
        shader: &Arc<GuestShaderCode>,
        roles: &[(ResourceId, MaxwellThreeDResourceRole)],
        resources: &Arc<MaxwellThreeDResolvedResources>,
    ) -> Result<Arc<GuestProgram>, Error> {
        let previous = self
            .prepared
            .as_ref()
            .filter(|p| {
                Arc::ptr_eq(&p.code, shader) && p.draw.program.bindings.len() == roles.len()
            })
            .map(|p| Arc::clone(&p.draw.program));
        let mut changed = previous.is_none().then(|| Vec::with_capacity(roles.len()));
        for (index, &(id, role)) in roles.iter().enumerate() {
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
                    MaxwellThreeDResolvedResource::Image(_) => GuestBinding::Image {
                        image: self.image(resources, index)?,
                        arrayed: matches!(
                            role,
                            MaxwellThreeDResourceRole::SampledImage {
                                dimension:
                                    super::super::threed::MaxwellThreeDTextureDimension::TwoArray,
                                ..
                            }
                        ),
                    },
                }
            };
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
        // Program preparation depends on physical bindings, not CPU content
        // epochs. Native consumption still checks visibility on every draw.
        (GuestBinding::Buffer(a), GuestBinding::Buffer(b)) => {
            a.byte_mapping_key() == b.byte_mapping_key()
        }
        (
            GuestBinding::Image {
                image: a,
                arrayed: aa,
            },
            GuestBinding::Image {
                image: b,
                arrayed: ab,
            },
        ) => aa == ab && a.same_view(b),
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
            && self.compute.as_ref().is_some_and(|(s, _)| {
                s.source_is_current(space)
                    && s.staged_writes_are_irrelevant(writes)
                    && s.compute_indirect_bindings_current(|slot| launch.constant_buffer(slot))
            });
        if !reuse {
            let cached = self
                .compute_sources
                .get(&request_key)
                .filter(|snapshot| {
                    snapshot.source_is_current(space)
                        && snapshot.staged_writes_are_irrelevant(writes)
                        && snapshot
                            .compute_indirect_bindings_current(|slot| launch.constant_buffer(slot))
                })
                .cloned();
            let snapshot = if let Some(snapshot) = cached {
                snapshot
            } else {
                let snapshot = Arc::new(
                    MaxwellShaderSnapshot::capture_compute(space, writes, request, |slot| {
                        launch.constant_buffer(slot)
                    })
                    .map_err(Error::ShaderTranslation)?,
                );
                self.compute_sources.insert(
                    request_key,
                    Arc::clone(&snapshot),
                    snapshot.retained_bytes(),
                );
                self.statistics.shader_captures += 1;
                snapshot
            };
            let translated = self.compile(snapshot.as_ref().clone())?;
            self.compute = Some((snapshot, translated));
            self.compute_request = Some(request_key);
        }
        let (_, translated) = self.compute.as_ref().unwrap();
        let translated = self.specialize_compute(Arc::clone(translated), launch, space, writes)?;
        let code = &translated.native;
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
        let mut native = Vec::with_capacity(3);
        for image in images {
            native.push(self.conversion_image(image)?);
        }
        Ok(NativeWork {
            frontend: FrontendSubmissionId::new(u64::MAX - self.identity().slot()),
            command: NativeCommand::ConvertNv12 {
                images: native.into_boxed_slice().try_into().unwrap(),
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
        let mut native = Vec::with_capacity(2);
        for image in images {
            native.push(self.conversion_image(image)?);
        }
        Ok(NativeWork {
            frontend: FrontendSubmissionId::new(u64::MAX - self.identity().slot()),
            command: NativeCommand::UploadPlanes {
                images: native.into_boxed_slice().try_into().unwrap(),
                bytes,
                widths,
            },
            canonical_accesses,
        })
    }
    fn lower_device_plane_uploads(
        &mut self,
        images: [nixe_gpu::ConversionImage; 2],
        frame: nixe_gpu::DeviceImageFrame,
    ) -> Result<NativeWork, MaxwellLoweringError> {
        let canonical_accesses = images
            .iter()
            .flat_map(|i| i.view.bindings().iter().map(|b| b.backing().clone()))
            .collect();
        let mut native = Vec::with_capacity(2);
        for image in images {
            native.push(self.conversion_image(image)?);
        }
        Ok(NativeWork {
            frontend: FrontendSubmissionId::new(u64::MAX - self.identity().slot()),
            command: NativeCommand::ImportDecodedPlanes {
                images: native.into_boxed_slice().try_into().unwrap(),
                frame,
            },
            canonical_accesses,
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
    use crate::engines::twod::blit::{BlitKind, BlitSurface, BlitSurfaceLayout};
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    #[test]
    fn alternating_native_shader_sources_share_code_and_observe_staged_changes() {
        use crate::shader::test_support::{canonical_shader_writes, mapped_memory};
        let (allocation, space, address) = mapped_memory();
        let mut vertex = [0u32; 20];
        vertex[0] = 0x0002_0461;
        let mut pixel = [0u32; 20];
        pixel[0] = 0x0002_5462;
        pixel[4] = 0x000f_f000;
        pixel[6] = 0x77;
        pixel[13] = 0x0007_f000;
        pixel[18] = 1;
        let encode = |header: [u32; 20], instruction: u64| {
            header
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .chain(
                    [0, instruction, 0xe300_0000_0007_000f, 0]
                        .into_iter()
                        .flat_map(u64::to_le_bytes),
                )
                .collect::<Vec<_>>()
        };
        let mov = |bits: u32| 0x0100_0000_0007_f000 | u64::from(bits) << 20;
        allocation
            .write(0, &encode(vertex, 0xe300_0000_0007_000f))
            .unwrap();
        allocation
            .write(0x100, &encode(pixel, mov(1f32.to_bits())))
            .unwrap();
        allocation
            .write(0x200, &encode(pixel, mov(2f32.to_bits())))
            .unwrap();
        let mut channel = crate::engines::tests::three_d_channel();
        for (method, value) in [
            (0x2140, 0x50),
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
        let mut frontend = MaxwellNativeFrontend::new(Arc::new(CompilePool::new(1, 8).unwrap()));
        let writes = CanonicalWriteBatch::new();
        frontend
            .ensure_code(channel.three_d(), &space, &writes)
            .unwrap();
        let first = Arc::clone(&frontend.code.as_ref().unwrap().code);
        let interface = Arc::clone(&frontend.code.as_ref().unwrap().interface);
        assert!(Arc::ptr_eq(
            &first.functions[0],
            &frontend.code.as_ref().unwrap().translated.programs[0].function
        ));
        crate::engines::tests::program_three_d(&mut channel, 0x2044, 0x200);
        frontend
            .ensure_code(channel.three_d(), &space, &writes)
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &frontend.code.as_ref().unwrap().code));
        crate::engines::tests::program_three_d(&mut channel, 0x2044, 0x100);
        frontend
            .ensure_code(channel.three_d(), &space, &writes)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &frontend.code.as_ref().unwrap().code));
        assert!(Arc::ptr_eq(
            &interface,
            &frontend.code.as_ref().unwrap().interface
        ));
        assert_eq!(frontend.statistics.shader_captures, 2);
        let word = mov(3f32.to_bits());
        let staged = canonical_shader_writes(
            &space,
            &[
                (address + 0x158, word as u32),
                (address + 0x15c, (word >> 32) as u32),
            ],
        );
        frontend
            .ensure_code(channel.three_d(), &space, &staged)
            .unwrap();
        assert_ne!(
            first.identity,
            frontend.code.as_ref().unwrap().code.identity
        );
        assert_eq!(frontend.statistics.shader_captures, 3);
    }

    #[test]
    fn single_sample_downsample_is_a_filtered_blit_with_distinct_rectangles() {
        let mut space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mut addresses = Vec::new();
        for i in 1..=2 {
            let storage = CanonicalAllocation::zeroed(0x10000, 4096).unwrap();
            let address = space
                .map(MaxwellMapRequest {
                    allocation: MaxwellAllocationId::new(i),
                    backing: storage
                        .backing_range(MemoryPermissions::READ_WRITE)
                        .unwrap(),
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
        let mut frontend = MaxwellNativeFrontend::new(Arc::new(CompilePool::new(1, 8).unwrap()));
        let mut channel = crate::engines::tests::three_d_channel();
        crate::engines::tests::program_three_d(&mut channel, 0x19d0, 0x3c);
        let source = channel
            .three_d()
            .render_targets()
            .clear()
            .last_surface()
            .source()
            .unwrap();
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
            .lower_blit(&request, &space, FrontendSubmissionId::new(1), &[])
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
