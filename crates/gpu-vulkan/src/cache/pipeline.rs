//! A guest cache miss constructs one complete pipeline, as GraphicsPipeline.
//! Keys and hit decisions belong to the graphics/compute owner, not this builder.
use super::*;
use super::{
    driver::DriverCache,
    layout::{LayoutBuilder, stage_flags},
};
use crate::{VulkanDevice, VulkanError, device::DeviceOwner};
use ash::vk;
use nixe_gpu::ShaderStage;
use std::{
    sync::{Arc, Condvar, Mutex, OnceLock},
    time::Instant,
};
mod state;
pub use state::*;
mod create;
#[derive(Clone, Debug)]
pub enum PipelineRecipe {
    Compute(Arc<ShaderArtifact>),
    Graphics {
        shader: Arc<ShaderArtifact>,
        state: GraphicsState,
    },
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineMetrics {
    pub wall_ns: u64,
    pub driver_ns: Option<u64>,
    pub application_cache_hit: bool,
}
pub type NativePipeline = super::NativeObject<NativePipelineData>;
pub struct NativePipelineData {
    pub(crate) owner: Arc<DeviceOwner>,
    modules: create::ShaderModules,
    counters: nixe_gpu::cache_counters::CacheCounters,
    build: OnceLock<Result<Executable, Arc<CompileError>>>,
    build_lock: Mutex<()>,
    build_ready: Condvar,
    pub layout: Arc<PipelineLayout>,
    pub bind_point: vk::PipelineBindPoint,
    pub dynamic_states: Box<[vk::DynamicState]>,
    pub(crate) graphics: Option<GraphicsState>,
    pub(crate) resources: Box<[crate::residency::AccessScope]>,
    pub(crate) image_bindings: Box<[usize]>,
    pub(crate) shader_writes: bool,
}
impl NativePipelineData {
    fn executable(&self) -> Result<&Executable, Arc<CompileError>> {
        let build = if let Some(build) = self.build.get() {
            build
        } else {
            let lock = self.build_lock.lock().unwrap();
            drop(
                self.build_ready
                    .wait_while(lock, |_| self.build.get().is_none())
                    .unwrap(),
            );
            self.build.get().unwrap()
        };
        build.as_ref().map_err(Arc::clone)
    }
    pub(crate) fn handle(&self) -> Result<vk::Pipeline, Arc<CompileError>> {
        self.executable().map(|build| build.handle)
    }
    pub fn metrics(&self) -> Result<PipelineMetrics, Arc<CompileError>> {
        self.executable().map(|build| build.metrics)
    }
}
struct Executable {
    handle: vk::Pipeline,
    metrics: PipelineMetrics,
}
impl Drop for NativePipelineData {
    fn drop(&mut self) {
        self.counters.remove_entries(1);
        if let Some(Ok(build)) = self.build.get() {
            unsafe {
                self.owner.device.destroy_pipeline(build.handle, None);
            }
        }
    }
}
struct Compiler {
    owner: Arc<DeviceOwner>,
    driver: Arc<DriverCache>,
    layouts: LayoutBuilder,
    feedback: bool,
    features: vk::PhysicalDeviceFeatures,
    limits: vk::PhysicalDeviceLimits,
    formats: [vk::FormatProperties; 185],
    dynamic: Box<[vk::DynamicState]>,
    provoking: bool,
    lines: [bool; 3],
    triangle_fans: bool,
    primitive_restart: [bool; 2],
    fill_rectangle: bool,
    vertex_divisor: bool,
    vertex_zero_divisor: bool,
    max_vertex_divisor: u32,
}
pub struct PipelineCompiler {
    pool: Arc<CompilePool>,
    serialization: Arc<CompilePool>,
    compiler: Arc<Compiler>,
    objects: Arc<super::NativeObjects>,
    graphics: nixe_gpu::cache_counters::CacheCounters,
    compute: nixe_gpu::cache_counters::CacheCounters,
}
impl Drop for PipelineCompiler {
    fn drop(&mut self) {
        self.pool.wait_for_requests();
        self.serialization.wait_for_requests();
        if let Err(error) = self.compiler.driver.checkpoint() {
            log::warn!("Vulkan driver cache write: {error}");
        }
    }
}
impl PipelineCompiler {
    pub fn new(
        device: &VulkanDevice,
        pool: Arc<CompilePool>,
        config: &CacheConfiguration,
    ) -> Result<Self, VulkanError> {
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
                vk::DynamicState::PRIMITIVE_TOPOLOGY,
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
        if device.profile().dynamic_state.base2 {
            dynamic.extend([
                vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
                vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
                vk::DynamicState::DEPTH_BIAS_ENABLE,
            ]);
        }
        if base {
            // BindVertexBuffers2 supplies strides even with SetVertexInputEXT.
            // Declare the state emitted by that command for VUID-08608.
            // https://docs.vulkan.org/spec/latest/chapters/drawing.html#VUID-vkCmdDrawIndexed-None-08608
            dynamic.push(vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
        }
        if device.profile().dynamic_state.vertex_input {
            dynamic.push(vk::DynamicState::VERTEX_INPUT_EXT);
        }
        if device.profile().dynamic_state.blending() {
            dynamic.extend([
                vk::DynamicState::COLOR_BLEND_ENABLE_EXT,
                vk::DynamicState::COLOR_BLEND_EQUATION_EXT,
                vk::DynamicState::COLOR_WRITE_MASK_EXT,
            ]);
        }
        if device.profile().core.depth_bounds != 0 {
            dynamic.push(vk::DynamicState::DEPTH_BOUNDS);
        }
        if device.profile().attachment_feedback_loop {
            dynamic.push(vk::DynamicState::ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT);
        }
        Ok(Self {
            objects: Arc::clone(&device.objects),
            pool,
            graphics: nixe_gpu::cache_counters::CacheCounters::new(0, 0),
            compute: nixe_gpu::cache_counters::CacheCounters::new(0, 0),
            serialization: Arc::new(
                CompilePool::with_name(1, "gpu-cache-store")
                    .map_err(|e| VulkanError::Loader(e.to_string().into()))?,
            ),
            compiler: Arc::new(Compiler {
                owner: Arc::clone(&device.owner),
                driver: Arc::new(DriverCache::new(device, config.directory.as_deref())?),
                layouts: LayoutBuilder::new(device),
                feedback: device.profile().pipeline_creation_feedback,
                features: device.profile().core,
                limits: device.profile().limits,
                formats: device.profile().formats,
                dynamic: dynamic.into_boxed_slice(),
                fill_rectangle: device.profile().fill_rectangle,
                vertex_divisor: device.profile().dynamic_state.vertex_divisor,
                vertex_zero_divisor: device.profile().dynamic_state.vertex_zero_divisor,
                max_vertex_divisor: device.profile().dynamic_state.max_vertex_divisor,
                provoking: device.profile().provoking_vertex_last,
                primitive_restart: device.profile().primitive_restart,
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
    pub fn request(&self, recipe: PipelineRecipe) -> Result<NativePipeline, CompileError> {
        let pipeline = self.prepare(recipe)?;
        let compiler = Arc::clone(&self.compiler);
        let building = pipeline.clone();
        self.pool
            .submit(move |worker| Self::finish(&compiler, &building, worker))?;
        Ok(pipeline)
    }
    pub(crate) fn build_immediate(
        &self,
        recipe: PipelineRecipe,
        worker: usize,
    ) -> Result<NativePipeline, CompileError> {
        let pipeline = self.prepare(recipe)?;
        Self::finish(&self.compiler, &pipeline, worker);
        pipeline.handle().map_err(|e| (*e).clone())?;
        Ok(pipeline)
    }
    fn prepare(&self, recipe: PipelineRecipe) -> Result<NativePipeline, CompileError> {
        let (shader, state) = match recipe {
            PipelineRecipe::Compute(shader) => (shader, None),
            PipelineRecipe::Graphics { shader, state } => (shader, Some(state)),
        };
        self.compiler.check_stages(&shader, state.as_ref())?;
        if let Some(state) = &state {
            self.compiler.check_graphics(state)?;
        }
        let state = state.map(|s| self.compiler.normalize(s));
        let bind_point = if state.is_some() {
            vk::PipelineBindPoint::GRAPHICS
        } else {
            vk::PipelineBindPoint::COMPUTE
        };
        let key = LayoutKey::reflect(&shader)?;
        let layout = if shader.stages.iter().all(|stage| stage.guest.is_some()) {
            self.compiler.layouts.create_guest(key, bind_point)?
        } else {
            self.compiler.layouts.create(key)?
        };
        let resources = crate::residency::execution::shader_accesses(&shader, &layout.key);
        let dynamic_states = state.as_ref().map_or_else(
            || Box::new([]) as Box<[vk::DynamicState]>,
            |s| self.compiler.dynamic_states(s),
        );
        let counters = if bind_point == vk::PipelineBindPoint::GRAPHICS {
            &self.graphics
        } else {
            &self.compute
        };
        let pipeline = self.objects.pipeline(NativePipelineData {
            owner: Arc::clone(&self.compiler.owner),
            modules: self.compiler.modules(&shader)?,
            counters: counters.clone(),
            build: OnceLock::new(),
            build_lock: Mutex::new(()),
            build_ready: Condvar::new(),
            layout: Arc::clone(&layout),
            bind_point,
            dynamic_states,
            shader_writes: resources.iter().any(|scope| scope.writes()),
            image_bindings: layout
                .key
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    matches!(
                        b.kind,
                        vk::DescriptorType::SAMPLED_IMAGE
                            | vk::DescriptorType::STORAGE_IMAGE
                            | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    )
                    .then_some(i)
                })
                .collect(),
            resources,
            graphics: state,
        });
        counters.miss();
        counters.add_entries(1);
        Ok(pipeline)
    }
    fn finish(compiler: &Compiler, pipeline: &NativePipeline, worker: usize) {
        nixe_gpu::trace::event("pipeline_compile", worker as u64, 0, 0);
        let _trace = nixe_gpu::trace::span("pipeline_build", worker as u64, 0, 0);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compiler.build(pipeline)))
                .unwrap_or_else(|_| {
                    Err(CompileError::Failed(
                        "pipeline construction panicked".into(),
                    ))
                })
                .map_err(Arc::new);
        let _lock = pipeline.build_lock.lock().unwrap();
        assert!(
            pipeline.build.set(result).is_ok(),
            "pipeline completed twice"
        );
        pipeline.build_ready.notify_all();
    }
    pub(crate) fn queue(
        &self,
        job: impl FnOnce(usize) + Send + 'static,
    ) -> Result<(), CompileError> {
        self.pool.submit(job)
    }
    pub(crate) fn serialization(&self) -> Arc<CompilePool> {
        Arc::clone(&self.serialization)
    }
    pub(crate) fn wait_for_requests(&self) {
        self.pool.wait_for_requests();
    }
    pub(crate) fn graphics_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.graphics.clone()
    }
    pub(crate) fn compute_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.compute.clone()
    }
    pub fn checkpoint(&self) -> Result<usize, CompileError> {
        self.pool.wait_for_requests();
        self.serialization.wait_for_requests();
        Ok(self.compiler.driver.checkpoint()?)
    }
    pub(crate) fn queue_checkpoint(
        &self,
        complete: impl FnOnce(usize) + Send + 'static,
    ) -> Result<(), CompileError> {
        let driver = Arc::clone(&self.compiler.driver);
        self.serialization
            .submit(move |_| match driver.checkpoint() {
                Ok(size) => complete(size),
                Err(error) => {
                    log::warn!("Vulkan driver cache write: {error}");
                    complete(0);
                }
            })
    }
}
impl Compiler {
    fn dynamic_states(&self, state: &GraphicsState) -> Box<[vk::DynamicState]> {
        self.dynamic
            .iter()
            .copied()
            .filter(|&value| {
                value != vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE
                    || !state.vertex_bindings.is_empty()
                    || self.dynamic.contains(&vk::DynamicState::VERTEX_INPUT_EXT)
            })
            .collect()
    }
    fn normalize(&self, s: GraphicsState) -> GraphicsState {
        s.normalized(&self.dynamic)
    }
    fn build(&self, pipeline: &NativePipeline) -> Result<Executable, CompileError> {
        let start = Instant::now();
        let mut metrics = PipelineMetrics::default();
        let handle = if let Some(state) = &pipeline.graphics {
            self.graphics(&pipeline.modules, &pipeline.layout, state, &mut metrics)?
        } else {
            self.compute(&pipeline.modules, &pipeline.layout, &mut metrics)?
        };
        metrics.wall_ns = start.elapsed().as_nanos() as u64;
        nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::HostPipelineCompilations, 1);
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::HostPipelineCompilationNanoseconds,
            metrics.wall_ns,
        );
        Ok(Executable { handle, metrics })
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
        for (index, &(binding, divisor)) in s.vertex_divisors.iter().enumerate() {
            if !self.vertex_divisor
                || (divisor == 0 && !self.vertex_zero_divisor)
                || divisor > self.max_vertex_divisor
                || !s
                    .vertex_bindings
                    .iter()
                    .any(|b| b.binding == binding && b.rate == vk::VertexInputRate::INSTANCE)
                || s.vertex_divisors[..index]
                    .iter()
                    .any(|&(b, _)| b == binding)
            {
                return Err(CompileError::Failed(
                    format!(
                        "vertex binding {binding}: unsupported native instance divisor {divisor}"
                    )
                    .into(),
                ));
            }
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
            && !crate::cache::restart_supported(s.topology, self.primitive_restart)
        {
            return Err(CompileError::Failed(
                "primitive restart requires enabled topology-list/patch-list restart support"
                    .into(),
            ));
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
        for a in s.vertex_attributes.iter() {
            if !usize::try_from(a.format.as_raw())
                .ok()
                .and_then(|index| self.formats.get(index))
                .map(|p| {
                    p.buffer_features
                        .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
                })
                .unwrap_or(false)
            {
                return Err(CompileError::Failed(format!("native vertex format {:?} lacks VERTEX_BUFFER support; use exact SSA pulling",a.format).into()));
            }
        }
        for &(format, blend) in s.colors.iter() {
            if format == vk::Format::UNDEFINED {
                continue;
            }
            let p = self
                .formats
                .get(format.as_raw() as usize)
                .ok_or_else(|| CompileError::Failed("unsupported native color format".into()))?;
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
}
