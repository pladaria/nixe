//! Native pipeline construction. Hash/normalization and feature decisions are
//! cache-miss/changed-state work; prepared references bypass this path entirely.
use super::*;
use super::{
    driver::DriverCaches,
    layout::{LayoutCache, stage_flags},
};
use crate::{VulkanDevice, VulkanError, device::DeviceOwner};
use ash::vk;
use nixe_gpu::{ShaderStage, shader_identity::ContentId};
use rustc_hash::FxHashMap as HashMap;
use std::{
    sync::{Arc, Condvar, Mutex},
    time::Instant,
};

mod state;
pub use state::*;
mod create;
use create::GraphicsCompilation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineStrategy {
    Monolithic,
    Libraries,
}
#[derive(Clone, Debug)]
pub enum PipelineRecipe {
    Compute(Arc<ShaderArtifact>),
    Graphics {
        shader: Arc<ShaderArtifact>,
        state: GraphicsState,
    },
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineKey {
    shader: ContentId,
    state: Option<GraphicsState>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineMetrics {
    pub wall_ns: u64,
    pub driver_ns: Option<u64>,
    pub application_cache_hit: bool,
    pub link_ns: u64,
    pub libraries_created: u32,
    pub library_wall_ns: u64,
    pub library_driver_ns: Option<u64>,
}

pub struct NativePipeline {
    pub(crate) owner: Arc<DeviceOwner>,
    pub(crate) handle: vk::Pipeline,
    pub layout: Arc<PipelineLayout>,
    pub bind_point: vk::PipelineBindPoint,
    pub dynamic_states: Box<[vk::DynamicState]>,
    pub metrics: PipelineMetrics,
    pub(crate) graphics: Option<GraphicsState>,
    pub(crate) resources: Box<[crate::residency::execution::ShaderAccess]>,
    pub(crate) host_uniforms: Box<[nixe_shader_recompiler::artifact::HostUniformLayout]>,
    pub(crate) binding_origins: Box<[super::layout::BindingOrigin]>,
    _libraries: Box<[Arc<Library>]>,
}
impl Drop for NativePipeline {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_pipeline(self.handle, None);
        }
    }
}
struct Library {
    owner: Arc<DeviceOwner>,
    handle: vk::Pipeline,
    _layout: Arc<PipelineLayout>,
    metrics: PipelineMetrics,
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_pipeline(self.handle, None);
        }
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LibraryKey {
    part: u32,
    shaders: Box<[ContentId]>,
    layout: LayoutKey,
    state: GraphicsState,
}
struct LibraryCell {
    result: Mutex<Option<Result<Arc<Library>, Arc<CompileError>>>>,
    ready: Condvar,
}
struct LibraryState {
    entries: HashMap<LibraryKey, (Arc<LibraryCell>, u64)>,
    clock: u64,
}
struct Libraries {
    map: Mutex<LibraryState>,
    limit: usize,
}
impl Libraries {
    fn get(
        &self,
        key: LibraryKey,
        create: impl FnOnce() -> Result<Library, CompileError>,
    ) -> Result<(Arc<Library>, bool), CompileError> {
        #[cfg(feature = "performance-counters")]
        let capture_key = nixe_gpu::compilation_capture::Key::new(&key);
        let (cell, create_here) = {
            let mut map = self.map.lock().unwrap();
            map.clock = map.clock.wrapping_add(1);
            let clock = map.clock;
            if let Some((cell, touched)) = map.entries.get_mut(&key) {
                *touched = clock;
                (Arc::clone(cell), false)
            } else {
                if map.entries.len() >= self.limit {
                    let victim = map
                        .entries
                        .iter()
                        .filter(|(_, (cell, _))| cell.result.lock().unwrap().is_some())
                        .min_by_key(|(_, (_, tick))| *tick)
                        .map(|(key, _)| key.clone());
                    let Some(victim) = victim else {
                        return Err(CompileError::Budget);
                    };
                    #[cfg(feature = "performance-counters")]
                    nixe_gpu::compilation_capture::Key::new(&victim).event("evict", usize::MAX, 0);
                    map.entries.remove(&victim);
                }
                let cell = Arc::new(LibraryCell {
                    result: Mutex::new(None),
                    ready: Condvar::new(),
                });
                #[cfg(feature = "performance-counters")]
                if nixe_gpu::compilation_capture::enabled() {
                    capture_key.detail("identity", usize::MAX, 0, &format!("{key:?}"));
                }
                map.entries.insert(key, (Arc::clone(&cell), clock));
                (cell, true)
            }
        };
        #[cfg(feature = "performance-counters")]
        capture_key.event(if create_here { "begin" } else { "hit" }, usize::MAX, 0);
        #[cfg(feature = "performance-counters")]
        let started = Instant::now();
        if create_here {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(create))
                .unwrap_or_else(|_| {
                    Err(CompileError::Failed(
                        "pipeline library compiler panicked".into(),
                    ))
                })
                .map(Arc::new)
                .map_err(Arc::new);
            #[cfg(feature = "performance-counters")]
            capture_key.event(
                if result.is_ok() { "end" } else { "failed" },
                usize::MAX,
                started.elapsed().as_nanos() as u64,
            );
            *cell.result.lock().unwrap() = Some(result);
            cell.ready.notify_all();
        }
        let result = cell.result.lock().unwrap();
        let result = cell.ready.wait_while(result, |r| r.is_none()).unwrap();
        #[cfg(feature = "performance-counters")]
        if !create_here {
            capture_key.event("wait_ns", usize::MAX, started.elapsed().as_nanos() as u64);
        }
        result
            .as_ref()
            .unwrap()
            .as_ref()
            .map(|lib| (Arc::clone(lib), create_here))
            .map_err(|e| (**e).clone())
    }
}
struct Compiler {
    owner: Arc<DeviceOwner>,
    driver: DriverCaches,
    layouts: LayoutCache,
    libraries: Libraries,
    strategy: PipelineStrategy,
    inline_modules: bool,
    feedback: bool,
    cache_control: bool,
    features: vk::PhysicalDeviceFeatures,
    limits: vk::PhysicalDeviceLimits,
    dynamic: Box<[vk::DynamicState]>,
    provoking: bool,
    lines: [bool; 3],
    triangle_fans: bool,
    fill_rectangle: bool,
}
pub struct PipelineCompiler {
    pool: Arc<CompilePool>,
    cache: CompilationCache<PipelineKey, NativePipeline>,
    compiler: Arc<Compiler>,
}
impl PipelineCompiler {
    pub fn new(
        device: &VulkanDevice,
        pool: Arc<CompilePool>,
        config: &CacheConfiguration,
    ) -> Result<Self, VulkanError> {
        let strategy = if device.profile().graphics_pipeline_library
            && device.profile().graphics_pipeline_library_fast_linking
        {
            PipelineStrategy::Libraries
        } else {
            PipelineStrategy::Monolithic
        };
        Self::with_strategy(device, pool, config, strategy)
    }
    pub(crate) fn with_strategy(
        device: &VulkanDevice,
        pool: Arc<CompilePool>,
        config: &CacheConfiguration,
        strategy: PipelineStrategy,
    ) -> Result<Self, VulkanError> {
        if pool.workers() != config.workers
            || config.pipeline_entries == 0
            || config.layout_entries == 0
        {
            return Err(VulkanError::InvalidResource(
                "pipeline worker/cache configuration mismatch",
            ));
        }
        if strategy == PipelineStrategy::Libraries && !device.profile().graphics_pipeline_library {
            return Err(VulkanError::InvalidResource(
                "graphics pipeline libraries are unavailable",
            ));
        }
        let base = device.profile().dynamic_state.base;
        let mut dynamic = vec![
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        if base {
            dynamic.extend([
                vk::DynamicState::CULL_MODE,
                vk::DynamicState::FRONT_FACE,
                vk::DynamicState::DEPTH_TEST_ENABLE,
                vk::DynamicState::DEPTH_WRITE_ENABLE,
                vk::DynamicState::DEPTH_COMPARE_OP,
                vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
                vk::DynamicState::STENCIL_TEST_ENABLE,
                vk::DynamicState::STENCIL_OP,
            ]);
        }
        if device.profile().core.depth_bounds != 0 {
            dynamic.push(vk::DynamicState::DEPTH_BOUNDS);
        }
        Ok(Self {
            pool,
            cache: CompilationCache::new(
                config.pipeline_entries,
                config.pipeline_entries * std::mem::size_of::<NativePipeline>(),
            ),
            compiler: Arc::new(Compiler {
                owner: Arc::clone(&device.owner),
                driver: DriverCaches::new(device, config)?,
                layouts: LayoutCache::new(device, config.layout_entries),
                libraries: Libraries {
                    map: Mutex::new(LibraryState {
                        entries: HashMap::default(),
                        clock: 0,
                    }),
                    limit: config.pipeline_entries,
                },
                strategy,
                inline_modules: device.profile().graphics_pipeline_library,
                feedback: device.profile().pipeline_creation_feedback,
                cache_control: device.profile().pipeline_creation_cache_control,
                features: device.profile().core,
                limits: device.profile().limits,
                dynamic: dynamic.into_boxed_slice(),
                fill_rectangle: device.profile().fill_rectangle,
                provoking: device.profile().provoking_vertex_last,
                lines: [
                    device.profile().rectangular_lines,
                    device.profile().smooth_lines,
                    device.profile().bresenham_lines,
                ],
                triangle_fans: device
                    .profile()
                    .portability
                    .as_ref()
                    .is_none_or(|p| p.triangle_fans),
            }),
        })
    }
    pub fn strategy(&self) -> PipelineStrategy {
        self.compiler.strategy
    }
    pub fn request(&self, recipe: PipelineRecipe) -> Result<Pending<NativePipeline>, CompileError> {
        let (shader, state) = match recipe {
            PipelineRecipe::Compute(shader) => (shader, None),
            PipelineRecipe::Graphics { shader, state } => (shader, Some(state)),
        };
        if let Some(state) = &state {
            self.compiler.check_graphics(state)?;
        }
        let state = state.map(|s| self.compiler.normalize(s));
        let key = PipelineKey {
            shader: shader.identity,
            state: state.clone(),
        };
        let compiler = Arc::clone(&self.compiler);
        #[cfg(feature = "performance-counters")]
        let captured = key.clone();
        self.cache.request(
            &self.pool,
            key,
            std::mem::size_of::<NativePipeline>(),
            move |worker| {
                #[cfg(feature = "performance-counters")]
                if nixe_gpu::compilation_capture::enabled() {
                    nixe_gpu::compilation_capture::Key::new(&captured).detail(
                        "identity",
                        worker,
                        0,
                        &format!(
                            "{captured:?} stages={:?} strategy={:?}",
                            shader
                                .stages
                                .iter()
                                .map(|s| (s.stage, s.identity.hex(), s.module.words.len()))
                                .collect::<Vec<_>>(),
                            compiler.strategy
                        ),
                    );
                }
                let result = compiler.create(worker, shader, state);
                #[cfg(feature = "performance-counters")]
                if let Ok(pipeline) = &result {
                    nixe_gpu::compilation_capture::Key::new(&captured).detail(
                        "metrics",
                        worker,
                        pipeline.metrics.wall_ns,
                        &format!("{:?}", pipeline.metrics),
                    );
                }
                result
            },
        )
    }
    pub fn checkpoint(&self) -> Result<(), CompileError> {
        self.compiler.driver.checkpoint()
    }
    pub fn usage(&self) -> (usize, usize) {
        self.cache.usage()
    }
    pub fn library_entries(&self) -> usize {
        self.compiler.libraries.map.lock().unwrap().entries.len()
    }
}
impl Compiler {
    fn normalize(&self, s: GraphicsState) -> GraphicsState {
        s.normalized(&self.dynamic)
    }
    fn create(
        &self,
        worker: usize,
        shader: Arc<ShaderArtifact>,
        state: Option<GraphicsState>,
    ) -> Result<NativePipeline, CompileError> {
        self.check_stages(&shader, state.as_ref())?;
        let layout = self.layouts.get(LayoutKey::reflect(&shader)?)?;
        let binding_origins = super::layout::binding_origins(&shader, &layout.key)?;
        let start = Instant::now();
        let mut metrics = PipelineMetrics::default();
        let mut linked_libraries = Vec::new();
        let (handle, bind_point, dynamic) = if let Some(state) = &state {
            self.check_graphics(state)?;
            let handle = if self.strategy == PipelineStrategy::Libraries {
                let mut libraries = Vec::new();
                for part in [
                    vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE,
                    vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
                    vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER,
                    vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE,
                ] {
                    if state.raster_discard
                        && matches!(
                            part,
                            vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER
                                | vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE
                        )
                    {
                        continue;
                    }
                    let key = self.library_key(&shader, &layout, state, part);
                    let (library, created) = self.libraries.get(key, || {
                        let mut library_metrics = PipelineMetrics::default();
                        let library_start = Instant::now();
                        let handle = self.graphics(
                            worker,
                            &shader,
                            &layout,
                            state,
                            GraphicsCompilation::Library(part),
                            &mut library_metrics,
                        )?;
                        Ok(Library {
                            owner: Arc::clone(&self.owner),
                            handle,
                            _layout: Arc::clone(&layout),
                            metrics: {
                                library_metrics.wall_ns = library_start.elapsed().as_nanos() as u64;
                                library_metrics
                            },
                        })
                    })?;
                    metrics.libraries_created += u32::from(created);
                    if created {
                        metrics.library_wall_ns += library.metrics.wall_ns;
                        if let Some(duration) = library.metrics.driver_ns {
                            metrics.library_driver_ns =
                                Some(metrics.library_driver_ns.unwrap_or(0) + duration);
                        }
                    }
                    libraries.push(library);
                }
                let handles: Vec<_> = libraries.iter().map(|l| l.handle).collect();
                let link_start = Instant::now();
                let handle = self.graphics(
                    worker,
                    &shader,
                    &layout,
                    state,
                    GraphicsCompilation::Link(&handles),
                    &mut metrics,
                )?;
                metrics.link_ns = link_start.elapsed().as_nanos() as u64;
                // Cache eviction releases index ownership, not the executable's
                // linked components. The batch pipeline lease keeps the complete
                // compiled dependency graph alive through completion.
                linked_libraries = libraries;
                handle
            } else {
                self.graphics(
                    worker,
                    &shader,
                    &layout,
                    state,
                    GraphicsCompilation::Monolithic,
                    &mut metrics,
                )?
            };
            (
                handle,
                vk::PipelineBindPoint::GRAPHICS,
                self.dynamic.clone(),
            )
        } else {
            (
                self.compute(worker, &shader, &layout, &mut metrics)?,
                vk::PipelineBindPoint::COMPUTE,
                Box::new([]) as Box<[vk::DynamicState]>,
            )
        };
        metrics.wall_ns = start.elapsed().as_nanos() as u64;
        nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::HostPipelineCompilations, 1);
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::HostPipelineCompilationNanoseconds,
            metrics.wall_ns,
        );
        let resources = crate::residency::execution::shader_accesses(&shader, &layout.key);
        Ok(NativePipeline {
            owner: Arc::clone(&self.owner),
            handle,
            layout,
            bind_point,
            dynamic_states: dynamic,
            metrics,
            resources,
            binding_origins,
            host_uniforms: shader
                .stages
                .iter()
                .filter_map(|stage| stage.guest.as_ref())
                .flat_map(|stage| stage.host_uniforms.iter().cloned())
                .collect(),
            graphics: state,
            _libraries: linked_libraries.into_boxed_slice(),
        })
    }
    fn check_stages(
        &self,
        shader: &ShaderArtifact,
        state: Option<&GraphicsState>,
    ) -> Result<(), CompileError> {
        let compute = state.is_none();
        if compute && (shader.stages.len() != 1 || shader.stages[0].stage != ShaderStage::Compute)
            || !compute && !shader.stages.iter().any(|s| s.stage == ShaderStage::Vertex)
        {
            return Err(CompileError::Failed(
                "native pipeline has invalid shader stage set".into(),
            ));
        }
        if compute {
            let stage = &shader.stages[0];
            let size = stage.workgroup_size.ok_or_else(|| {
                CompileError::Failed("compute stage is missing workgroup metadata".into())
            })?;
            if size.contains(&0)
                || size
                    .iter()
                    .zip(self.limits.max_compute_work_group_size)
                    .any(|(&n, max)| n > max)
                || size
                    .iter()
                    .try_fold(1u32, |a, &b| a.checked_mul(b))
                    .is_none_or(|n| n > self.limits.max_compute_work_group_invocations)
                || stage.shared_memory_bytes > self.limits.max_compute_shared_memory_size
            {
                return Err(CompileError::Failed(
                    "compute workgroup/shared memory exceeds enabled host limits".into(),
                ));
            }
        }
        let mut stages = vk::ShaderStageFlags::empty();
        for s in &shader.stages {
            if s.output_control_points
                .is_some_and(|n| n > self.limits.max_tessellation_patch_size)
            {
                return Err(CompileError::Failed(
                    "control shader output patch exceeds enabled host limit".into(),
                ));
            }
            let flag = stage_flags(s.stage);
            if stages.contains(flag) {
                return Err(CompileError::Failed("duplicate shader stage".into()));
            }
            stages |= flag;
            if !compute && s.stage == ShaderStage::Compute {
                return Err(CompileError::Failed(
                    "compute shader in graphics pipeline".into(),
                ));
            }
            if matches!(
                s.stage,
                ShaderStage::TessellationControl | ShaderStage::TessellationEvaluation
            ) && self.features.tessellation_shader == 0
                || s.stage == ShaderStage::Geometry && self.features.geometry_shader == 0
            {
                return Err(CompileError::Failed(
                    "shader stage requires an unavailable host feature".into(),
                ));
            }
            if s.module.bindings.iter().any(|b| b.writable)
                && match s.stage {
                    ShaderStage::Fragment => self.features.fragment_stores_and_atomics == 0,
                    ShaderStage::Compute => false,
                    _ => self.features.vertex_pipeline_stores_and_atomics == 0,
                }
            {
                return Err(CompileError::Failed(
                    "shader storage writes require the enabled stage stores/atomics feature".into(),
                ));
            }
        }
        if stages.contains(vk::ShaderStageFlags::TESSELLATION_CONTROL)
            != stages.contains(vk::ShaderStageFlags::TESSELLATION_EVALUATION)
        {
            return Err(CompileError::Failed(
                "tessellation requires both control and evaluation stages".into(),
            ));
        }
        if let Some(s) = state {
            if stages.contains(vk::ShaderStageFlags::TESSELLATION_CONTROL)
                != (s.topology == vk::PrimitiveTopology::PATCH_LIST)
                || s.topology == vk::PrimitiveTopology::PATCH_LIST && s.patch_points == 0
            {
                return Err(CompileError::Failed(
                    "patch topology/control stages/cardinality disagree".into(),
                ));
            }
            if !s.raster_discard && !stages.contains(vk::ShaderStageFlags::FRAGMENT) {
                return Err(CompileError::Failed(
                    "rasterization requires a fragment shader".into(),
                ));
            }
        }
        Ok(())
    }
    fn check_graphics(&self, s: &GraphicsState) -> Result<(), CompileError> {
        if s.colors.len() > self.limits.max_color_attachments as usize
            || s.viewport_count == 0
            || s.viewport_count > self.limits.max_viewports
            || s.patch_points > self.limits.max_tessellation_patch_size
        {
            return Err(CompileError::Failed(
                "graphics state exceeds enabled host limits".into(),
            ));
        }
        for (required, enabled, name) in [
            (s.depth_clamp, self.features.depth_clamp, "depthClamp"),
            (
                matches!(s.polygon, vk::PolygonMode::LINE | vk::PolygonMode::POINT),
                self.features.fill_mode_non_solid,
                "fillModeNonSolid",
            ),
            (s.logic_enabled, self.features.logic_op, "logicOp"),
            (s.depth_bounds, self.features.depth_bounds, "depthBounds"),
            (
                s.viewport_count > 1,
                self.features.multi_viewport,
                "multiViewport",
            ),
        ] {
            if required && enabled == 0 {
                return Err(CompileError::Failed(
                    format!("graphics state requires enabled {name}").into(),
                ));
            }
        }
        if s.vertex_bindings.len() > self.limits.max_vertex_input_bindings as usize
            || s.vertex_attributes.len() > self.limits.max_vertex_input_attributes as usize
            || s.vertex_bindings.iter().any(|b| {
                b.binding >= self.limits.max_vertex_input_bindings
                    || b.stride > self.limits.max_vertex_input_binding_stride
            })
            || s.vertex_attributes.iter().any(|a| {
                a.location >= self.limits.max_vertex_input_attributes
                    || a.offset > self.limits.max_vertex_input_attribute_offset
                    || !s.vertex_bindings.iter().any(|b| b.binding == a.binding)
            })
        {
            return Err(CompileError::Failed(
                "vertex input exceeds host bounds or references an absent binding".into(),
            ));
        }
        if s.polygon == vk::PolygonMode::FILL_RECTANGLE_NV && !self.fill_rectangle {
            return Err(CompileError::Failed(
                "fill rectangle requires VK_NV_fill_rectangle or an exact primitive lowering"
                    .into(),
            ));
        }
        if s.provoking == vk::ProvokingVertexModeEXT::LAST_VERTEX && !self.provoking {
            return Err(CompileError::Failed(
                "last provoking vertex requires enabled provokingVertexLast".into(),
            ));
        }
        let line = match s.line_mode {
            vk::LineRasterizationModeEXT::DEFAULT => true,
            vk::LineRasterizationModeEXT::RECTANGULAR => self.lines[0],
            vk::LineRasterizationModeEXT::RECTANGULAR_SMOOTH => self.lines[1],
            vk::LineRasterizationModeEXT::BRESENHAM => self.lines[2],
            _ => false,
        };
        if !line {
            return Err(CompileError::Failed(
                "requested line rasterization mode is unavailable".into(),
            ));
        }
        if s.topology == vk::PrimitiveTopology::TRIANGLE_FAN && !self.triangle_fans {
            return Err(CompileError::Failed(
                "triangle fan requires an exact frontend index lowering on this portability device"
                    .into(),
            ));
        }
        if s.primitive_restart
            && !matches!(
                s.topology,
                vk::PrimitiveTopology::LINE_STRIP
                    | vk::PrimitiveTopology::TRIANGLE_STRIP
                    | vk::PrimitiveTopology::TRIANGLE_FAN
            )
        {
            return Err(CompileError::Failed("primitive restart outside native strip/fan topology requires frontend index lowering".into()));
        }
        if s.vertex_bindings.iter().enumerate().any(|(i, b)| {
            s.vertex_bindings[..i]
                .iter()
                .any(|p| p.binding == b.binding)
        }) || s.vertex_attributes.iter().enumerate().any(|(i, a)| {
            s.vertex_attributes[..i]
                .iter()
                .any(|p| p.location == a.location)
        }) {
            return Err(CompileError::Failed(
                "duplicate vertex binding or attribute location".into(),
            ));
        }
        for a in &s.vertex_attributes {
            let properties = unsafe {
                self.owner
                    .instance
                    .instance
                    .get_physical_device_format_properties(self.owner.physical, a.format)
            };
            if !properties
                .buffer_features
                .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
            {
                return Err(CompileError::Failed(format!("native vertex format {:?} lacks VERTEX_BUFFER support; use exact SSA pulling",a.format).into()));
            }
        }
        for &(format, blend) in &s.colors {
            if format == vk::Format::UNDEFINED {
                continue;
            }
            let p = unsafe {
                self.owner
                    .instance
                    .instance
                    .get_physical_device_format_properties(self.owner.physical, format)
            };
            let required = vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | if blend.enabled {
                    vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
                } else {
                    vk::FormatFeatureFlags::empty()
                };
            if !p.optimal_tiling_features.contains(required) {
                return Err(CompileError::Failed(
                    "color format lacks native attachment/blend support".into(),
                ));
            }
        }
        if self.features.independent_blend == 0 && s.colors.windows(2).any(|p| p[0].1 != p[1].1) {
            return Err(CompileError::Failed(
                "independent per-target blend/masks require independentBlend".into(),
            ));
        }
        if !s.samples.as_raw().is_power_of_two() {
            return Err(CompileError::Failed(
                "raster sample count must be one supported power-of-two flag".into(),
            ));
        }
        Ok(())
    }
    fn library_key(
        &self,
        shader: &ShaderArtifact,
        layout: &PipelineLayout,
        s: &GraphicsState,
        part: vk::GraphicsPipelineLibraryFlagsEXT,
    ) -> LibraryKey {
        let mut state = s.clone();
        let vi = part == vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE;
        let pre = part == vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS;
        let fs = part == vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER;
        let out = part == vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE;
        if !vi {
            state.vertex_bindings = Box::new([]);
            state.vertex_attributes = Box::new([]);
            state.topology = vk::PrimitiveTopology::TRIANGLE_LIST;
            state.primitive_restart = false;
        }
        if !pre {
            state.patch_points = 0;
            state.polygon = vk::PolygonMode::FILL;
            state.cull = vk::CullModeFlags::NONE;
            state.front_face = vk::FrontFace::COUNTER_CLOCKWISE;
            state.depth_clamp = false;
            state.raster_discard = false;
            state.depth_bias = false;
            state.viewport_count = 1;
            state.provoking = vk::ProvokingVertexModeEXT::FIRST_VERTEX;
            state.line_mode = vk::LineRasterizationModeEXT::DEFAULT;
        }
        if !fs {
            state.depth_test = false;
            state.depth_write = false;
            state.depth_compare = vk::CompareOp::ALWAYS;
            state.depth_bounds = false;
            state.stencil_test = false;
            state.front = Default::default();
            state.back = Default::default();
        }
        if !fs && !out {
            state.samples = vk::SampleCountFlags::TYPE_1;
            state.sample_mask = u64::MAX;
            state.alpha_to_coverage = false;
            state.depth_format = vk::Format::UNDEFINED;
            state.stencil_format = vk::Format::UNDEFINED;
        }
        if !out {
            state.colors = Box::new([]);
            state.logic_enabled = false;
            state.logic_op = vk::LogicOp::COPY;
        }
        let shaders = shader
            .stages
            .iter()
            .filter(|s| {
                if pre {
                    s.stage != ShaderStage::Fragment
                } else {
                    fs && s.stage == ShaderStage::Fragment
                }
            })
            .map(|s| s.identity)
            .collect();
        LibraryKey {
            part: part.as_raw(),
            shaders,
            layout: if vi || out {
                LayoutKey {
                    bindings: Box::new([]),
                    push_bytes: 0,
                    push_stages: vk::ShaderStageFlags::empty(),
                }
            } else {
                layout.key.clone()
            },
            state,
        }
    }
}
