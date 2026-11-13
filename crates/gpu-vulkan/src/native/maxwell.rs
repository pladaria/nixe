//! Console-specific Vulkan rendering from the selected Maxwell channel.
//! Channel slots persist independently of shared compiled code and native work.
use model::engines::lowering::*;
use model::engines::threed::*;
use nixe_gpu::*;
use nixe_gpu_maxwell as model;
use std::{cell::RefCell, fmt::Display, rc::Rc, sync::Arc};
pub(in crate::native) mod blending;
mod compute;
pub(crate) mod dynamic_state;
pub(in crate::native) mod fixed_state;
mod geometry;
mod graphics;
mod handles;
pub(in crate::native) mod identity;
pub(crate) mod image;
pub(in crate::native) mod registers;
pub(super) mod reuse;
mod targets;
pub(in crate::native) mod textures;
pub(in crate::native) mod vertex_input;
use super::pipeline_cache::PipelineCache;
use identity::ImageKey;
use model::MaxwellGpuMemory;
use model::{MaxwellFrontend, MaxwellSubmissionExecutionError as Error};
use nixe_gpu::native::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeFrontendStatistics {
    pub draws: u64,
    pub shader_captures: u64,
    pub prepared_graphics: u64,
    pub fixed_state_refreshes: u64,
}
/// RefreshStages owns the renderer's current per-stage shader facts. Pipelines
/// are keyed by raw bytecode hashes, not by captured binding combinations.
#[derive(Default)]
pub(in crate::native) struct GraphicsShaders {
    pub(in crate::native) hashes: [u64; model::MAXWELL_PIPELINE_SHADER_COUNT],
    pub(in crate::native) infos:
        [Option<Arc<nixe_gpu::shader_memory::ShaderInfo>>; model::MAXWELL_PIPELINE_SHADER_COUNT],
    pub(in crate::native) groups: [Option<u8>; 5],
    pub(in crate::native) tessellation_enabled: bool,
}
pub struct MaxwellVulkanRasterizer {
    renderer: Rc<RefCell<Option<super::NativeGpuExecution>>>,
    shader_cache: nixe_gpu::shader_memory::ShaderCache,
    shaders: Option<GraphicsShaders>,
    pipelines: PipelineCache,
    profile: Arc<crate::DeviceProfile>,
    next: u64,
    pub statistics: NativeFrontendStatistics,
    image_views: nixe_gpu::WorkingSet<ImageKey, GuestImage>,

    render_enabled: bool,
    current_channel: Option<model::MaxwellChannelId>,
    channels: std::collections::BTreeMap<model::MaxwellChannelId, Option<bool>>,
}
fn rendered_description(
    info: AttachmentInfo,
    role: MaxwellThreeDResourceRole,
) -> Result<ImageDescription, Error> {
    model::engines::threed::attachment_description(info, role).map_err(Error::ThreeDResource)
}
fn compilation(error: impl Display) -> Error {
    Error::Native(error.to_string().into())
}
fn lowering<T>(result: Result<T, MaxwellLoweringError>) -> Result<T, Error> {
    result.map_err(Error::Lowering)
}
impl MaxwellVulkanRasterizer {
    pub fn new(renderer: Rc<RefCell<Option<super::NativeGpuExecution>>>) -> Self {
        let (shader_cache, pipelines, profile) = {
            let backend = renderer.borrow();
            let backend = backend.as_ref().expect("installed renderer");
            (
                backend.shader_cache(),
                PipelineCache::new(
                    Arc::clone(&backend.compiler),
                    &backend.residency.device().profile().shader_compiler,
                ),
                backend.residency.device().profile_handle(),
            )
        };
        Self {
            renderer,
            shader_cache,
            shaders: None,
            pipelines,
            profile,
            next: 1,
            statistics: Default::default(),
            image_views: nixe_gpu::WorkingSet::new(4096, IMAGE_VIEW_BYTES),

            render_enabled: true,
            current_channel: None,
            channels: Default::default(),
        }
    }
    pub fn load_disk_resources(
        &mut self,
        directory: &std::path::Path,
        stop: &std::sync::atomic::AtomicBool,
    ) {
        self.pipelines
            .load_disk_resources(directory, &self.profile, stop);
    }
    fn conversion_image(
        &mut self,
        source: nixe_gpu::ConversionImage,
    ) -> Result<GuestImage, MaxwellLoweringError> {
        let key = ImageKey {
            encoding: model::engines::threed::MaxwellThreeDGuestImageFormat::Canonical(
                source.description.format(),
            ),
            description: source.description,

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
    fn refresh_shaders(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), Error> {
        let dirty = state.graphics_dirty().take(GraphicsDirty::SHADERS) != 0;
        if !dirty && self.shaders.is_some() {
            return Ok(());
        }
        let mut shaders = GraphicsShaders::default();
        for index in 0..model::MAXWELL_PIPELINE_SHADER_COUNT {
            let binding = state.shader_bindings();
            let pipeline = binding.pipeline(index);
            if pipeline.enabled().value() != Some(&true) {
                continue;
            }
            if index == 5 && state.register_words()[0x037c / 4] == 0 {
                continue;
            }
            let source = model::recompiler::graphics_source(state, index).map_err(compilation)?;
            let gpu = space.address(source.address).map_err(compilation)?;
            let device = space.device_address(gpu).map_err(compilation)?;
            let info = if let Some(info) = self.shader_cache.try_get(
                device,
                nixe_shader_recompiler::environment::ShaderIsa::Maxwell,
            ) {
                info
            } else {
                let request =
                    model::recompiler::graphics_request(state, index).map_err(compilation)?;
                self.shader_cache
                    .register(
                        device,
                        nixe_shader_recompiler::environment::ShaderIsa::Maxwell,
                        model::recompiler::analyze_code(space, &request, access)
                            .map_err(compilation)?,
                    )
                    .map_err(compilation)?
            };
            let stage = nixe_gpu::shader_compilation::neutral_stage(source.stage) as usize;
            shaders.tessellation_enabled |= matches!(
                source.stage,
                nixe_shader_recompiler::Stage::TessellationControl
                    | nixe_shader_recompiler::Stage::TessellationEval
            );
            if shaders.groups[stage].is_some() && shaders.groups[stage] != source.group {
                return Err(compilation(
                    "dual vertex stages require the same constant-buffer group",
                ));
            }
            shaders.groups[stage] = source.group;
            shaders.hashes[index] = info.unique_hash;
            shaders.infos[index] = Some(info);
        }
        self.shaders = Some(shaders);
        Ok(())
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

impl MaxwellFrontend for MaxwellVulkanRasterizer {
    fn sync_guest_memory(&mut self) {
        self.renderer
            .borrow()
            .as_ref()
            .expect("installed renderer")
            .residency
            .caches
            .sync_guest_memory();
    }
    fn service_host_requests(&mut self) -> Result<(), Error> {
        let mut renderer = self.renderer.borrow_mut();
        let renderer = renderer.as_mut().expect("installed renderer");
        let (steps, anti_aliasing_steps) = renderer
            .state
            .host
            .as_ref()
            .map_or((0, 0), |host| host.client.take_setting_steps());
        if anti_aliasing_steps != 0 {
            renderer
                .state
                .frames
                .cycle_anti_aliasing(anti_aliasing_steps);
        }
        if steps == 0 {
            return Ok(());
        }
        let current = renderer.residency.device().resolution_scale;
        let scales = crate::RESOLUTION_SCALES;
        let index = scales
            .iter()
            .position(|scale| {
                scale.up_scale == current.up_scale && scale.down_shift == current.down_shift
            })
            .ok_or_else(|| {
                compilation("render resolution is outside the supported hotkey cycle")
            })?;
        let resolution = scales[(index + usize::from(steps)) % scales.len()];
        renderer
            .residency
            .access()
            .change_resolution(resolution)
            .map_err(compilation)?;
        self.pipelines.change_resolution(resolution);
        log::info!(
            "render resolution scale changed to {}x",
            resolution.up_factor
        );
        Ok(())
    }
    fn report_samples_passed(
        &mut self,
        target: Vec<nixe_memory::AllocationRange>,
        short: bool,
        timestamp: u64,
        frontend: FrontendSubmissionId,
    ) -> Result<(), Error> {
        let mut renderer = self.renderer.borrow_mut();
        renderer
            .as_mut()
            .expect("installed renderer")
            .execution()
            .record(frontend, |native| {
                native
                    .residency
                    .report_samples_passed(&target, short, timestamp)
            })
            .map_err(compilation)
    }
    fn reset_samples_passed(&mut self) -> Result<(), Error> {
        let mut renderer = self.renderer.borrow_mut();
        renderer
            .as_mut()
            .expect("installed renderer")
            .residency
            .access()
            .device_mut()
            .reset_samples_passed()
            .map_err(compilation)
    }
    fn bind_gpu_memory(&mut self, memory: &model::MaxwellGpuMemoryHandle) {
        let renderer = self.renderer.borrow();
        let caches = &renderer
            .as_ref()
            .expect("installed renderer")
            .residency
            .caches;
        let pending_reads = caches.pending_read_count();
        let memory_sync = caches.guest_memory_sync();
        let interface: Arc<dyn model::MaxwellGpuMemoryInterface> = caches.clone();
        memory.bind_rasterizer(Arc::downgrade(&interface), pending_reads, memory_sync);
    }
    fn create_channel(&mut self, channel: model::MaxwellChannelId) {
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .create_channel(
                channel.get(),
                model::MAXWELL_VERTEX_STREAM_COUNT,
                model::MAXWELL_BIND_GROUP_COUNT,
                model::MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
            );
        assert!(
            self.channels.insert(channel, Some(true)).is_none(),
            "frontend channel already exists"
        );
    }
    fn bind_channel(&mut self, channel: model::MaxwellChannelId, state: &MaxwellThreeDState) {
        if self.current_channel == Some(channel) {
            return;
        }
        // Bind the canonical dirty bank without discarding global shader/key
        // selections, as RasterizerVulkan::BindChannel/StateTracker::InvalidateState.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L2057-L2070
        state.graphics_dirty().mark(GraphicsDirty::ALL);
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .bind_channel(channel.get());
        let next = self
            .channels
            .get_mut(&channel)
            .expect("registered frontend channel")
            .take()
            .expect("inactive frontend channel has storage");
        let previous = std::mem::replace(&mut self.render_enabled, next);
        if let Some(id) = self.current_channel.replace(channel) {
            *self.channels.get_mut(&id).unwrap() = Some(previous);
        }
    }
    fn bind_graphics_uniform(
        &mut self,
        group: u8,
        slot: u8,
        binding: Option<model::MaxwellThreeDConstantBufferBinding>,
        space: &MaxwellGpuMemory,
    ) -> Result<(), Error> {
        let source = binding
            .map(|binding| {
                let address = space
                    .address(binding.address().get())
                    .map_err(compilation)?;
                let address = space.device_address(address).map_err(compilation)?;
                Ok::<_, Error>(GuestBufferRange {
                    address,
                    size: u64::from(binding.size()),
                })
            })
            .transpose()?;
        let mut native = self.renderer.borrow_mut();
        let native = native.as_mut().expect("installed renderer");
        native
            .residency
            .runtime
            .buffer_bindings
            .bind_graphics_uniform(group, slot, source);
        Ok(())
    }
    fn close_channel(&mut self, channel: model::MaxwellChannelId) {
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .close_channel(channel.get());
        self.channels.remove(&channel);
        if self.current_channel == Some(channel) {
            self.current_channel = None;
            self.render_enabled = true;
        }
    }

    fn draw(
        &mut self,
        state: &MaxwellThreeDState,
        draw: &MaxwellThreeDDrawState,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), Error> {
        self.refresh_shaders(state, space, access)?;
        let shaders = self.shaders.as_ref().unwrap();
        let pipeline_reused = self
            .pipelines
            .current_graphics(
                &self.profile,
                &mut self.statistics,
                shaders,
                state,
                draw,
                space,
                access,
            )
            .map_err(compilation)?;
        let slot = self
            .pipelines
            .current_graphics
            .expect("selected graphics pipeline");
        let pipeline = &self.pipelines.graphics_pipelines[slot];
        let groups = shaders.groups;
        let mut renderer = self.renderer.borrow_mut();
        let renderer = renderer.as_mut().expect("installed renderer");
        self.statistics.draws += 1;
        renderer.state.statistics.reuse.pipeline_hits += u64::from(pipeline_reused);
        let result = renderer
            .execution()
            .record(frontend, |native| {
                pipeline.draw(native, state, draw, space, groups)
            })
            .map_err(compilation);
        if result.is_ok() {
            state.graphics_dirty().take(GraphicsDirty::PARAMETERS);
        }
        result
    }
    fn clear(
        &mut self,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<(), Error> {
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .execution()
            .record(frontend, |native| {
                native
                    .clear_current(state, space)
                    .map(|()| true)
                    .map_err(|e| crate::VulkanError::Compilation(e.to_string().into()))
            })
            .map(|_| ())
            .map_err(compilation)
    }

    fn launch(
        &mut self,
        launch: &model::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<(), Error> {
        launch.validate_execution().map_err(Error::ComputeLaunch)?;
        let address = space
            .address(launch.shader_address())
            .map_err(compilation)?;
        let address = space.device_address(address).map_err(compilation)?;
        let shader = if let Some(shader) = self.shader_cache.try_get(
            address,
            nixe_shader_recompiler::environment::ShaderIsa::Maxwell,
        ) {
            shader
        } else {
            self.shader_cache
                .register(
                    address,
                    nixe_shader_recompiler::environment::ShaderIsa::Maxwell,
                    model::recompiler::analyze_code(space, &launch.capture_request(), access)
                        .map_err(compilation)?,
                )
                .map_err(compilation)?
        };
        let slot = self
            .pipelines
            .current_compute(launch, &shader, space, access, &mut self.statistics)
            .map_err(compilation)?;
        let pipeline = &self.pipelines.compute_pipelines[slot];
        let mut renderer = self.renderer.borrow_mut();
        let renderer = renderer.as_mut().expect("installed renderer");
        renderer
            .execution()
            .record(frontend, |native| {
                pipeline.dispatch(native, launch, space, launch.workgroups())
            })
            .map_err(compilation)
    }
    fn solid(
        &mut self,
        request: &model::engines::MaxwellTwoDSolidOperation,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<(), Error> {
        let info = solid_image_info(request, space).map_err(Error::ThreeDResource)?;
        let role = if request.zeta {
            MaxwellThreeDResourceRole::DepthStencilTarget
        } else {
            MaxwellThreeDResourceRole::BlitDestination
        };
        let value = lowering(resolve::solid_clear_value(
            request.color,
            request.destination.format,
            info.guest_format,
        ))?;
        let [x, y, x1, y1] = request.rectangle;
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .execution()
            .record(frontend, |native| {
                native.flush_work()?;
                let image = native.render_target_view(info, role, space)?;
                native.clear_native(
                    image,
                    nixe_gpu::ScissorRect {
                        x,
                        y,
                        width: x1 - x,
                        height: y1 - y,
                    },
                    value,
                )?;
                Ok(())
            })
            .map_err(compilation)
    }
    fn accelerate_dma(
        &mut self,
        operation: &model::MaxwellMemoryCopyOperation,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<bool, Error> {
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .execution()
            .record(frontend, |native| {
                native.residency.runtime.device.realize_deferred_clear()?;
                native.residency.accelerate_dma(operation, space)
            })
            .map_err(compilation)
    }
    fn blit(
        &mut self,
        request: &model::MaxwellTwoDBlitOperation,
        space: &MaxwellGpuMemory,
        frontend: FrontendSubmissionId,
    ) -> Result<(), Error> {
        use model::engines::twod::blit::BlitKind;
        let infos = blit_image_infos(request, space).map_err(Error::ThreeDResource)?;
        let source_description =
            rendered_description(infos[0], MaxwellThreeDResourceRole::BlitSource)?;
        let destination_description =
            rendered_description(infos[1], MaxwellThreeDResourceRole::BlitDestination)?;
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
                source_description.samples() != nixe_gpu::SampleCount::One,
            ),
            BlitKind::Downsample2x { filter } => {
                let destination_extent = destination_description.extent();
                let resolve = source_description.samples() != nixe_gpu::SampleCount::One;
                let source_extent = if resolve {
                    destination_extent
                } else {
                    source_description.extent()
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
                destination_origin,
                extent,
                ..
            } => {
                let extent_source = source_description.extent();
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
        self.renderer
            .borrow_mut()
            .as_mut()
            .expect("installed renderer")
            .execution()
            .record(frontend, |native| {
                // Eden TextureCacheRuntime::BlitImage uses BlitColor for all
                // single-sample color surfaces. Consume their selected views,
                // including format conversions, mip levels and array layers.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L1264-L1282
                let color = source_description.kind() == ImageKind::Color
                    && destination_description.kind() == ImageKind::Color
                    && source_description.samples() == SampleCount::One
                    && destination_description.samples() == SampleCount::One;
                let sampling = if color {
                    Some(match request.kind {
                        BlitKind::Scale { first, step, .. } => (first, step),
                        _ => {
                            let step = [
                                (i64::from(source_rectangle.width) << 32)
                                    .checked_div(i64::from(destination_rectangle.width))
                                    .ok_or(crate::VulkanError::InvalidResource(
                                        "color blit destination width is zero",
                                    ))?,
                                (i64::from(source_rectangle.height) << 32)
                                    .checked_div(i64::from(destination_rectangle.height))
                                    .ok_or(crate::VulkanError::InvalidResource(
                                        "color blit destination height is zero",
                                    ))?,
                            ];
                            let first = [
                                (i64::from(source_rectangle.x) << 32) + step[0] / 2,
                                (i64::from(source_rectangle.y) << 32) + step[1] / 2,
                            ];
                            (first, step)
                        }
                    })
                } else {
                    None
                };
                if sampling.is_some() {
                    native.ensure_blit_pipeline(super::images::format(
                        destination_description.format(),
                    ))?;
                }
                native.residency.runtime.device.realize_deferred_clear()?;
                let retained = native.residency.runtime.constructing.len();
                let result =
                    (|| {
                        let source = native.render_target_view(
                            infos[0],
                            MaxwellThreeDResourceRole::BlitSource,
                            space,
                        )?;
                        native.residency.runtime.constructing.push(
                            crate::residency::Resource::Image(
                                native.residency.texture_cache.views[source].image,
                            ),
                        );
                        let destination = native.render_target_view(
                            infos[1],
                            MaxwellThreeDResourceRole::BlitDestination,
                            space,
                        )?;
                        native.residency.runtime.constructing.push(
                            crate::residency::Resource::Image(
                                native.residency.texture_cache.views[destination].image,
                            ),
                        );
                        native.blit_native(
                            source,
                            destination,
                            source_rectangle,
                            destination_rectangle,
                            linear,
                            resolve,
                            sampling,
                        )?;
                        Ok(())
                    })();
                native.residency.runtime.constructing.truncate(retained);
                result
            })
            .map_err(compilation)
    }
    fn render_enabled(&self) -> bool {
        self.render_enabled
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
        self.render_enabled = match mode {
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
    fn lower_nv12_conversion(
        &mut self,
        destination: nixe_gpu::ConversionImage,
        source: nixe_gpu::Nv12Planes,
        transform: nixe_gpu::Nv12ColorTransform,
    ) -> Result<NativeNv12Conversion, MaxwellLoweringError> {
        let submission = self.next;
        self.next = submission
            .checked_add(1)
            .expect("native submission counter exhausted");
        Ok(NativeNv12Conversion {
            frontend: FrontendSubmissionId::new(u64::MAX - submission),
            destination: Box::new(self.conversion_image(destination)?),
            source,
            transform,
        })
    }
}

impl Drop for MaxwellVulkanRasterizer {
    fn drop(&mut self) {
        log::info!("native Maxwell preparation: {:?}", self.statistics);
    }
}

#[cfg(test)]
mod allocations;
