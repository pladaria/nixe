//! Concrete native execution selected at startup. Cold compilation is polled by
//! the owner; prepared references bypass content hashes and descriptor updates.
use crate::residency::Resource;
use crate::{cache::*, *};
use ash::vk;
use nixe_gpu::{BackendRuntimeError, FrontendSubmissionId, GpuExecution, ResidentImage, native::*};
use rustc_hash::FxHashMap as HashMap;
use std::{collections::VecDeque, sync::Arc, time::Duration};
mod completion;
mod pipeline_cache;
pub(crate) mod pipeline_resources;

pub(crate) mod maxwell;
pub(crate) use maxwell::blending::Blending;
pub(crate) use maxwell::vertex_input::VertexInput;
pub use maxwell::{MaxwellVulkanRasterizer, NativeFrontendStatistics};
pub(crate) mod images;
mod state;
use crate::residency::color_blit_pipeline;

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeExecutionStatistics {
    pub draws: u64,
    pub batches: u64,
    pub submits: u64,
    pub dispatches: u64,
    pub dynamic_state_refreshes: u64,
    pub reuse: maxwell::reuse::ReuseStatistics,
}
pub struct NativeGpuExecution {
    // The fencing thread joins before the native timeline is destroyed.
    state: NativeState,
    compiler: Arc<PipelineCompiler>,
    residency: VulkanResidency,
}

// One operation borrows the canonical channel/pipeline state and resource
// owners together, through preparation and command recording.
struct NativeExecution<'a> {
    state: &'a mut NativeState,
    compiler: &'a Arc<PipelineCompiler>,
    residency: crate::residency::Residency<'a>,
}
struct NativeState {
    // Drop/join before residency destroys the timeline semaphore, including
    // failure paths that do not reach explicit teardown.
    fences: completion::FenceManager,
    frames: crate::presentation::Frames,
    host: Option<crate::presentation::Host>,
    descriptors: DescriptorCache,
    dynamic: maxwell::dynamic_state::DynamicState,
    current_context: Option<u64>,
    blit_pipelines: HashMap<vk::Format, NativePipeline>,
    blit_samplers: [Sampler; 2],
    video_pipelines: HashMap<vk::Format, NativePipeline>,
    video_inputs: Option<crate::residency::Nv12Inputs>,
    draw_counter: u32,
    last_draw: Option<maxwell::reuse::ResolvedDraw>,
    pub statistics: NativeExecutionStatistics,
}
fn runtime(error: VulkanError) -> BackendRuntimeError {
    match error {
        VulkanError::Memory(cause) => BackendRuntimeError::Memory(cause),
        error => BackendRuntimeError::Backend(error.to_string().into()),
    }
}
impl NativeGpuExecution {
    fn execution(&mut self) -> NativeExecution<'_> {
        NativeExecution {
            state: &mut self.state,
            compiler: &self.compiler,
            residency: self.residency.access(),
        }
    }
    pub fn attach_presentation(
        &mut self,
        client: crate::presentation::PresentationClient,
        vsync: bool,
    ) -> Result<(), VulkanError> {
        self.execution().attach_presentation(client, vsync)
    }
    pub fn new(
        memory: nixe_memory::device_memory::DeviceMemory,
        device: VulkanDevice,
        configuration: CacheConfiguration,
        resident_bytes: u64,
    ) -> Result<Self, VulkanError> {
        let driver_pool = Arc::new(
            CompilePool::with_name(configuration.workers, "gpu-pipeline")
                .map_err(|e| VulkanError::Loader(e.to_string().into()))?,
        );
        let pipelines = Arc::new(PipelineCompiler::new(&device, driver_pool, &configuration)?);
        let descriptors = DescriptorCache::new(&device);
        let make_sampler = |filter| {
            vk::SamplerCreateInfo::default()
                .min_filter(filter)
                .mag_filter(filter)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .unnormalized_coordinates(true)
        };
        let mut residency = VulkanResidency::new(memory, device, resident_bytes)?;
        let blit_samplers = {
            let mut cache = residency.access();
            let samplers = &mut cache.texture_cache.samplers;
            [
                (samplers.get(&make_sampler(vk::Filter::NEAREST))?).clone(),
                (samplers.get(&make_sampler(vk::Filter::LINEAR))?).clone(),
            ]
        };
        residency
            .access()
            .enable_sample_conversion((pipelines).clone());
        residency.access().enable_quad_indices((pipelines).clone());
        residency.access().enable_scaling(Arc::clone(&pipelines));
        residency
            .access()
            .device_mut()
            .enable_sample_queries(Arc::clone(&pipelines))?;
        residency
            .memory()
            .enable_cpu_write_collection()
            .map_err(VulkanError::from)?;
        let fences =
            completion::FenceManager::new(residency.device(), Arc::clone(&residency.caches))?;
        Ok(Self {
            state: NativeState {
                fences,
                frames: crate::presentation::Frames::new(residency.device())?,
                host: None,
                descriptors,
                dynamic: Default::default(),
                current_context: None,
                blit_pipelines: HashMap::default(),
                blit_samplers,
                video_pipelines: HashMap::default(),
                video_inputs: None,
                draw_counter: 0,
                last_draw: None,
                statistics: Default::default(),
            },
            compiler: pipelines,
            residency,
        })
    }
    pub(crate) fn shader_cache(&self) -> nixe_gpu_maxwell::recompiler::ShaderCache {
        self.residency.shader_cache()
    }
    pub fn device_profile(&self) -> &DeviceProfile {
        self.residency.device().profile()
    }
    /// Retain this read-only history handle before moving execution to its owner.
    pub fn frame_metrics(&self) -> nixe_gpu::frame_metrics::FrameMetrics {
        self.residency
            .device()
            .submission
            .frame_metrics
            .history
            .clone()
    }
    pub fn cache_monitor(&self) -> CacheMonitor {
        CacheMonitor::new([
            ("shader cache", self.shader_cache().cache_counters()),
            ("graphics pipelines", self.compiler.graphics_counters()),
            ("compute pipelines", self.compiler.compute_counters()),
            ("descriptor sets", self.residency.descriptor_counters()),
        ])
    }
    pub fn recording_statistics(&self) -> RecordingStatistics {
        self.residency.device().recording_statistics()
    }
    pub fn residency_statistics(&self) -> ResidencyStatistics {
        self.residency.statistics()
    }
}

impl NativeExecution<'_> {
    fn service_host(&mut self) -> Result<(), BackendRuntimeError> {
        if self.state.host.is_none() {
            self.residency.maintain_budget(false).map_err(runtime)?;
        }
        let Some(mut host) = self.state.host.take() else {
            return Ok(());
        };
        let result = host.service(
            &mut self.residency,
            &mut self.state.frames,
            self.compiler,
            &mut self.state.descriptors,
        );
        if let Err(error) = &result {
            host.client.fail(error.to_string());
        }
        self.state.host = Some(host);
        result.map_err(runtime)
    }
    pub fn attach_presentation(
        &mut self,
        client: crate::presentation::PresentationClient,
        vsync: bool,
    ) -> Result<(), VulkanError> {
        self.state.host = Some(crate::presentation::Host::new(
            self.residency.device(),
            client,
            vsync,
        )?);
        Ok(())
    }
    fn prepare_nv12_pipeline(&mut self, format: vk::Format) -> Result<(), VulkanError> {
        if !self.state.video_pipelines.contains_key(&format) {
            let pending = match crate::residency::nv12_pipeline(self.compiler, format) {
                Ok(pending) => pending,
                Err(error) => {
                    return Err(VulkanError::Compilation(error.to_string().into()));
                }
            };
            self.state.video_pipelines.insert(format, pending);
        }
        Ok(())
    }
    fn ensure_blit_pipeline(&mut self, format: vk::Format) -> Result<(), VulkanError> {
        if !self.state.blit_pipelines.contains_key(&format) {
            let pending = match color_blit_pipeline(self.compiler, format) {
                Ok(p) => p,
                Err(e) => return Err(VulkanError::Compilation(e.to_string().into())),
            };
            self.state.blit_pipelines.insert(format, pending);
        }
        Ok(())
    }
    fn record_conversion(&mut self, conversion: &NativeNv12Conversion) -> Result<(), VulkanError> {
        self.residency.runtime.device.realize_deferred_clear()?;
        let NativeNv12Conversion {
            destination,
            source,
            transform,
            ..
        } = conversion;
        self.prepare_nv12_pipeline(images::format(destination.description.format()))?;
        let extent = crate::residency::Nv12Inputs::dimensions(source)?;
        if self
            .state
            .video_inputs
            .as_ref()
            .is_none_or(|inputs| inputs.extent[0] < extent[0] || inputs.extent[1] < extent[1])
        {
            let mut capacity = extent;
            if let Some(old) = self.state.video_inputs.take() {
                capacity = std::array::from_fn(|axis| capacity[axis].max(old.extent[axis]));
            }
            self.state.video_inputs = Some(crate::residency::Nv12Inputs::new(
                &mut self.residency,
                capacity,
            )?);
        }
        self.state
            .video_inputs
            .as_mut()
            .unwrap()
            .upload(&mut self.residency, source)?;
        let output = self.allocate_guest_image(destination)?;
        let pipeline =
            (self.state.video_pipelines[&images::format(destination.description.format())]).clone();
        let conversion = self.residency.prepare_nv12(
            pipeline,
            &mut self.state.descriptors,
            self.state.video_inputs.as_ref().unwrap(),
            output,
            0,
            *transform,
        )?;
        self.residency.convert_nv12(&conversion)?;
        Ok(())
    }
    fn flush_work(&mut self) -> Result<(), VulkanError> {
        // Desktop Eden records eight-operation chunks and submits a bounded
        // command-list prefix at 4096 operations. GPU idleness is not a submit
        // condition; presentation, coherence and guest fences flush explicitly.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L983-L1001
        self.state.draw_counter += 1;
        if self.state.draw_counter & 7 != 7 {
            return Ok(());
        }
        if self.state.draw_counter < 4096 {
            self.residency.device_mut().dispatch.dispatch_work()?;
        } else {
            self.residency.flush()?;
            self.state.draw_counter = 0;
        }
        Ok(())
    }

    fn record<T>(
        &mut self,
        frontend: FrontendSubmissionId,
        operation: impl FnOnce(&mut Self) -> Result<T, VulkanError>,
    ) -> Result<T, BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        let _trace = nixe_gpu::trace::span("backend_record", frontend.get(), 0, 1);
        #[cfg(feature = "performance-counters")]
        let started = std::time::Instant::now();
        let result = operation(self).map_err(runtime)?;
        let work = self.residency.device().current_work();
        nixe_gpu::trace::detail_event("backend_accept", frontend.get(), 0, work.tick());
        #[cfg(feature = "performance-counters")]
        {
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::BackendSubmissionNanoseconds,
                started.elapsed().as_nanos() as u64,
            );
            nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::BackendSubmissions, 1);
        }
        self.state.statistics.batches += 1;
        self.state.statistics.submits = self.residency.device().submitted_tick();
        Ok(result)
    }
}
impl NativeGpuExecution {
    fn fence_caches(&mut self, action: Option<nixe_gpu::FenceAction>) -> Result<(), VulkanError> {
        let mut residency = self.residency.access();
        residency.commit_texture_downloads()?;
        residency.commit_buffer_downloads()?;
        let work = residency.visibility_work();
        if work.is_some_and(|work| work.tick() > residency.device().scheduled_tick())
            && !residency.device_mut().submission.defer_flush()
        {
            residency.flush()?;
        }
        self.state.fences.queue(work, true, action)?;
        drop(residency);
        // SignalFence queues its committed prefix before InvalidateGPUCache.
        nixe_gpu::OwnerMemory::invalidate_gpu_cache(self).map_err(VulkanError::Memory)?;
        Ok(())
    }
}
// ChannelSetupCaches selects the actual cache-owned texture bank under its
// mutex. Current selections remain visible to CPU-side image deletion.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/control/channel_state_cache.inc
impl NativeGpuExecution {
    fn create_channel(&mut self, channel: u64) {
        self.residency
            .caches
            .texture_cache
            .lock()
            .unwrap()
            .bindings
            .create_channel(channel);
        self.residency.runtime.buffer_bindings.create_channel(
            channel,
            nixe_gpu_maxwell::MAXWELL_VERTEX_STREAM_COUNT,
            nixe_gpu_maxwell::MAXWELL_BIND_GROUP_COUNT,
            nixe_gpu_maxwell::MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
        );
    }
    fn bind_channel(&mut self, context: u64) {
        self.residency.runtime.buffer_bindings.bind_channel(context);
        self.residency
            .caches
            .texture_cache
            .lock()
            .unwrap()
            .bindings
            .bind_channel(context);
        if self.state.current_context == Some(context) {
            return;
        }
        if let Some(index) = self.residency.runtime.device.submission.recording {
            self.residency.runtime.device.submission.slots[index]
                .graphics_state
                .invalidate();
        }
        self.state.current_context = Some(context);
    }
    fn close_channel(&mut self, context: u64) {
        self.residency
            .runtime
            .buffer_bindings
            .close_channel(context);
        self.residency
            .caches
            .texture_cache
            .lock()
            .unwrap()
            .bindings
            .close_channel(context);
        if self.state.current_context == Some(context) {
            self.state.current_context = None;
        }
    }
}
impl Drop for NativeGpuExecution {
    fn drop(&mut self) {
        self.residency.memory().disable_cpu_write_collection();
    }
}
impl NativeExecution<'_> {
    fn bind_host_wakeup(
        &mut self,
        wake: Arc<dyn Fn() + Send + Sync>,
        submit: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), BackendRuntimeError> {
        self.state.fences.bind_wakeup(Arc::clone(&wake), submit);
        if let Some(host) = &self.state.host {
            host.client.bind_wakeup(wake);
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<WorkTicket, BackendRuntimeError> {
        self.state.draw_counter = 0;
        self.state.last_draw = None;
        self.residency.flush().map_err(runtime)
    }

    fn end_command_list(&mut self) -> Result<(), BackendRuntimeError> {
        self.state.draw_counter = 0;
        let device = self.residency.device_mut();
        if device.submission.finish_command_list(&device.dispatch) {
            self.flush()?;
        }
        Ok(())
    }
    fn flush_deferred(&mut self) -> Result<(), BackendRuntimeError> {
        if self.residency.device().submission.flush_pending {
            self.flush()?;
        }
        Ok(())
    }
    fn submit(&mut self, action: Option<nixe_gpu::FenceAction>) -> Result<(), BackendRuntimeError> {
        let work = self.flush()?;
        self.state
            .fences
            .queue(Some(work), false, action)
            .map_err(runtime)
    }
    fn flush_region(&mut self, address: u64, bytes: u64) -> Result<(), BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.residency.flush_region(address, bytes).map_err(runtime)
    }
    fn access_memory(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        mut operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.residency
            .access_memory(&ranges, &mut operation)
            .map_err(runtime)
    }

    fn queue_device_access(
        &mut self,
        request: nixe_memory::DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.residency
            .queue_device_access(request)
            .map_err(runtime)?;
        Ok(())
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
        self.state.fences.check().map_err(runtime)?;
        self.residency.progress_native_work().map_err(runtime)?;
        self.service_host()
    }
    fn retire_physical_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
        extents: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.residency
            .retire_physical_memory(machine, allocation, extents)
            .map_err(runtime)
    }
    fn release_guest_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError> {
        self.residency.check_machine(machine).map_err(runtime)?;
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.progress()
    }
    fn allocation_pressure(
        &mut self,
        machine: nixe_memory::MachineId,
        _: u64,
    ) -> Result<(), BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.residency.check_machine(machine).map_err(runtime)?;
        self.residency.flush().map_err(runtime)?;
        Ok(())
    }
    fn compose_frame(
        &mut self,
        request: &nixe_gpu_maxwell::PresentationImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        self.state.draw_counter = 0;
        self.state.last_draw = None;
        #[cfg(feature = "performance-counters")]
        {
            let queue = &self.residency.device().submission;
            nixe_gpu::trace::event(
                "frame_dependency_wait",
                queue.dependency_wait_ns,
                queue.recording_wait_ns,
                0,
            );
        }
        let reuse = self.state.statistics.reuse;
        nixe_gpu::trace::event(
            "frame_reuse",
            reuse.eligible_draws,
            reuse.draw_hits,
            reuse.pipeline_hits,
        );
        nixe_gpu::trace::event(
            "frame_texture_reuse",
            reuse.texture_attempts,
            reuse.texture_hits,
            reuse.memo_builds,
        );
        for (index, reasons) in reuse.draw_misses.chunks_exact(3).enumerate() {
            nixe_gpu::trace::event(
                ["frame_draw_miss0", "frame_draw_miss1", "frame_draw_miss2"][index],
                reasons[0],
                reasons[1],
                reasons[2],
            );
        }
        for (index, reasons) in reuse.texture_misses.chunks_exact(3).enumerate() {
            nixe_gpu::trace::event(
                ["frame_texture_miss0", "frame_texture_miss1"][index],
                reasons[0],
                reasons[1],
                reasons[2],
            );
        }
        nixe_gpu::trace::event(
            "frame_work",
            self.state.statistics.draws,
            self.state.statistics.dispatches,
            self.state.statistics.batches,
        );
        nixe_gpu::trace::event(
            "frame_queue",
            self.residency.device().submitted_tick(),
            self.residency.device().recording_statistics().draw_calls,
            self.residency.statistics().native_readback_bytes,
        );
        nixe_gpu::trace::event(
            "frame_memory",
            self.residency.statistics().cpu_uploaded_bytes,
            self.residency.statistics().cpu_streamed_bytes,
            self.residency.statistics().gpu_copied_bytes,
        );
        nixe_gpu::trace::event(
            "frame_buffers",
            self.residency.statistics().buffer_allocations,
            self.residency.statistics().buffer_merges,
            0,
        );
        let render_extent = if request.transform.rotate_90_clockwise {
            [request.crop.height, request.crop.width]
        } else {
            [request.crop.width, request.crop.height]
        };
        // Eden's display layer composites at the window size. Keep the actual
        // render dimensions independent of this presentation canvas.
        let extent = self
            .state
            .host
            .as_ref()
            .map(|h| h.client.output_size())
            .filter(|s| !s.contains(&0))
            .unwrap_or_else(|| render_extent.map(|v| self.residency.resolution().scale(v)));
        let format = self
            .state
            .host
            .as_ref()
            .map_or(vk::Format::R8G8B8A8_UNORM, |h| h.output_format());
        let request = NativePresentationImage {
            crop: request.crop,
            transform: request.transform,
            address: request.mapping.address() + request.offset,
            size: request.size,
            layout: request.layout,
            row_pitch: request.row_pitch,
            extent: [request.width, request.height],
            format: request.format,
            allow_canonical_import: request.allow_canonical_import,
        };
        let image = self
            .state
            .frames
            .compose(
                &mut self.residency,
                self.compiler,
                &mut self.state.descriptors,
                self.state.host.as_mut(),
                request,
                extent,
                format,
            )
            .map_err(runtime)?;
        Ok(image)
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        self.residency
            .invalidate_device_mappings()
            .map_err(runtime)?;
        if let Some(host) = &mut self.state.host {
            host.shutdown(&mut self.residency).map_err(runtime)?;
        }
        #[cfg(target_os = "linux")]
        log::info!("native Vulkan residency: {:?}", self.residency.statistics());
        log::info!(
            "native Vulkan recording: {:?}",
            self.residency.device().recording_statistics()
        );
        log::info!("native Vulkan preparation: {:?}", self.state.statistics);
        log::info!(
            "native Vulkan execution: draws={} dispatches={} batches={} submits={} descriptors={:?}",
            self.state.statistics.draws,
            self.state.statistics.dispatches,
            self.state.statistics.batches,
            self.state.statistics.submits,
            self.state.descriptors.statistics()
        );
        self.residency.flush().map_err(runtime)?;
        self.residency
            .device()
            .validation_monitor()
            .check()
            .map_err(runtime)?;
        Ok(())
    }
}

impl GpuExecution for NativeGpuExecution {
    type Conversion = NativeNv12Conversion;
    type ImageRequest = nixe_gpu_maxwell::PresentationImageRequest;
    fn begin_frame_work(&mut self) {
        // Start before taking resource-cache locks so contention is measured.
        let device = &mut self.residency.runtime.device;
        device.submission.begin_frame_work(&mut device.dispatch);
    }
    fn begin_frame_work_at(&mut self, start: std::time::Instant) {
        let device = &mut self.residency.runtime.device;
        device.submission.frame_metrics.begin_cpu_at(start);
        device.submission.begin_frame_work(&mut device.dispatch);
    }
    fn frame_idle(&mut self, start: std::time::Instant, end: std::time::Instant) {
        let device = &mut self.residency.runtime.device;
        device
            .submission
            .frame_idle(start, end, device.dispatch.completed_recording());
    }
    fn device_uuid(&self) -> Option<[u8; 16]> {
        Some(self.residency.device().profile().device_uuid)
    }
    fn bind_host_wakeup(
        &mut self,
        wake: Arc<dyn Fn() + Send + Sync>,
        submit: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), BackendRuntimeError> {
        self.execution().bind_host_wakeup(wake, submit)
    }
    fn flush(&mut self) -> Result<(), BackendRuntimeError> {
        self.execution().flush().map(|_| ())
    }

    fn end_command_list(&mut self) -> Result<(), BackendRuntimeError> {
        self.execution().end_command_list()
    }
    fn flush_deferred(&mut self) -> Result<(), BackendRuntimeError> {
        self.execution().flush_deferred()
    }
    fn convert_nv12(&mut self, conversion: &Self::Conversion) -> Result<(), BackendRuntimeError> {
        self.execution().record(conversion.frontend, |native| {
            native.record_conversion(conversion)
        })
    }

    fn report_memory(
        &mut self,
        frontend: FrontendSubmissionId,
        target: Vec<nixe_memory::AllocationRange>,
        bytes: Arc<[u8]>,
        fence: bool,
    ) -> Result<(), BackendRuntimeError> {
        let action = self.execution().record(frontend, |native| {
            native.residency.runtime.device.realize_deferred_clear()?;
            native.residency.report_query(target, bytes)
        })?;
        if fence {
            self.fence_caches(Some(action)).map_err(runtime)?;
        } else {
            self.state.fences.sync_operation(action);
        }
        Ok(())
    }
    fn order_device(&mut self, frontend: FrontendSubmissionId) -> Result<(), BackendRuntimeError> {
        self.execution()
            .record(frontend, |native| {
                native.residency.runtime.device.realize_deferred_clear()?;
                native.residency.order_device()?;
                Ok(())
            })
            .map(|_| ())
    }
    fn cache_maintenance(
        &mut self,
        frontend: FrontendSubmissionId,
        _operation: nixe_gpu::CacheMaintenanceOperation,
    ) -> Result<(), BackendRuntimeError> {
        // Actual cache invalidation is owned by OwnerMemory and resource scopes;
        // this ordered boundary also materializes any preceding attachment clears.
        self.execution()
            .record(frontend, |native| {
                native.residency.runtime.device.realize_deferred_clear()?;
                Ok(())
            })
            .map(|_| ())
    }
    fn submit(&mut self, action: Option<nixe_gpu::FenceAction>) -> Result<(), BackendRuntimeError> {
        self.execution().submit(action)
    }
    fn signal_fence(
        &mut self,
        action: Option<nixe_gpu::FenceAction>,
    ) -> Result<(), BackendRuntimeError> {
        self.fence_caches(action).map_err(runtime)
    }

    fn queue_device_access(
        &mut self,
        request: nixe_memory::DeviceAccessRequest,
    ) -> Result<(), BackendRuntimeError> {
        nixe_gpu::OwnerMemory::invalidate_gpu_cache(self).map_err(BackendRuntimeError::Memory)?;
        self.execution().queue_device_access(request)
    }
    fn progress(&mut self) -> Result<(), BackendRuntimeError> {
        self.execution().progress()
    }
    fn retire_physical_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
        extents: &[nixe_memory::Extent],
    ) -> Result<(), BackendRuntimeError> {
        self.execution()
            .retire_physical_memory(machine, allocation, extents)
    }
    fn release_guest_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
    ) -> Result<(), BackendRuntimeError> {
        self.execution().release_guest_memory(machine, allocation)
    }
    fn allocation_pressure(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: u64,
    ) -> Result<(), BackendRuntimeError> {
        self.execution().allocation_pressure(machine, allocation)
    }
    fn composite(
        &mut self,
        request: nixe_gpu_maxwell::PresentationImageRequest,
        swap_interval: u32,
        pacing: nixe_gpu::frame_metrics::FramePacing,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        let device = &mut self.residency.runtime.device;
        device
            .submission
            .begin_composite(&mut device.dispatch, pacing);
        let image = self.execution().compose_frame(&request)?;
        // Composite/Present own this boundary outside resource-cache locks.
        // Composition retains its actual native inputs and output.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/renderer_vulkan.cpp#L204-L248
        self.residency
            .runtime
            .device
            .dispatch
            .wait_worker()
            .map_err(runtime)?;
        let mut residency = self.residency.access();
        residency.tick_frame();
        residency.maintain_budget(true).map_err(runtime)?;
        // Recording-worker and cache-maintenance waits are production costs.
        residency
            .device_mut()
            .submission
            .end_frame(image.lease().producer().slot(), swap_interval);
        Ok(image)
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        // WaitPendingFences(true) commits a final cache prefix before waiting.
        self.fence_caches(None).map_err(runtime)?;
        self.execution().flush_deferred()?;
        self.state.fences.wait_pending().map_err(runtime)?;
        self.execution().teardown()
    }
}

impl nixe_gpu::OwnerMemory for NativeGpuExecution {
    fn invalidate_gpu_cache(&mut self) -> Result<(), Arc<nixe_memory::MemoryError>> {
        let memory = self.residency.memory().clone();
        memory
            .gather_gpu_dirty_memory(|ranges| {
                self.residency
                    .access()
                    .write_cpu_ranges(ranges.iter().copied())
            })
            .map_err(|error| owner_memory_error(runtime(error)))
    }

    fn flush_region(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), Arc<nixe_memory::MemoryError>> {
        self.invalidate_gpu_cache()?;
        self.execution()
            .flush_region(address, bytes)
            .map_err(owner_memory_error)
    }
    fn invalidate_regions(
        &mut self,
        regions: &[GuestBufferRange],
    ) -> Result<(), Arc<nixe_memory::MemoryError>> {
        self.residency
            .access()
            .write_regions(regions)
            .map_err(|error| owner_memory_error(runtime(error)))
    }
    fn inline_memory(
        &mut self,
        regions: &[GuestBufferRange],
    ) -> Result<(), Arc<nixe_memory::MemoryError>> {
        self.residency
            .access()
            .write_regions(regions)
            .map_err(|error| owner_memory_error(runtime(error)))
    }
    fn access_memory(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, Arc<nixe_memory::MemoryError>> {
        self.invalidate_gpu_cache()?;
        self.execution()
            .access_memory(ranges, operation)
            .map_err(owner_memory_error)
    }
}
fn owner_memory_error(error: BackendRuntimeError) -> Arc<nixe_memory::MemoryError> {
    match error {
        BackendRuntimeError::Memory(cause) => cause,
        other => Arc::new(nixe_memory::MemoryError::Execution(
            other.to_string().into(),
        )),
    }
}
