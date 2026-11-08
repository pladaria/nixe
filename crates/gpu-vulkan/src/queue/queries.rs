//! SamplesStreamer query banks and GPU prefix accumulation, as in Eden.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_query_cache.cpp
use super::*;
use crate::cache::{
    BindingRecord, DescriptorCache, DescriptorResource, NativePipeline, PipelineCompiler,
    PipelineRecipe, ShaderArtifact,
};
use nixe_gpu::{
    ShaderStage,
    shader_module::{Binding, DescriptorKind, ResourceId},
};
use std::collections::VecDeque;
const CAPACITY: u32 = 256;
const SCAN_BYTES: u64 = 2048 * 8;
struct Pool {
    owner: Arc<DeviceOwner>,
    handle: vk::QueryPool,
    input: Arc<AllocatedBuffer>,
    _output: Arc<AllocatedBuffer>,
    binding: Arc<BindingRecord>,
}
impl Drop for Pool {
    fn drop(&mut self) {
        // QueueState drains native recording and GPU use before dropping banks.
        unsafe { self.owner.device.destroy_query_pool(self.handle, None) };
    }
}
struct Bank {
    pool: Arc<Pool>,
    first: u32,
    used: u32,
    work: WorkTicket,
}
struct Pending {
    pool: Arc<Pool>,
    work: WorkTicket,
}
struct Scan {
    accumulation: Arc<AllocatedBuffer>,
    pipeline: NativePipeline,
    descriptors: DescriptorCache,
}
#[derive(Default)]
pub(crate) struct Samples {
    pub(crate) enabled: bool,
    current: Option<Bank>,
    active: Option<(vk::QueryPool, u32)>,
    pending: VecDeque<Pending>,
    free: Vec<Arc<Pool>>,
    scan: Option<Scan>,
}
impl Samples {
    pub fn pause(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        if let Some((pool, index)) = self.active.take() {
            dispatch.end_occlusion(pool, index);
        }
    }
    fn collect(&mut self, completed: u64) {
        while self
            .pending
            .front()
            .is_some_and(|p| p.work.tick <= completed)
        {
            self.free.push(self.pending.pop_front().unwrap().pool);
        }
    }
}
impl VulkanDevice {
    pub(crate) fn enable_sample_queries(
        &mut self,
        compiler: Arc<PipelineCompiler>,
    ) -> Result<(), VulkanError> {
        let subgroup = &self.profile.subgroup;
        let extended = subgroup.stages.contains(vk::ShaderStageFlags::COMPUTE)
            && subgroup.operations.contains(
                vk::SubgroupFeatureFlags::BASIC
                    | vk::SubgroupFeatureFlags::ARITHMETIC
                    | vk::SubgroupFeatureFlags::SHUFFLE
                    | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE,
            );
        let bytes: &[u8] = if extended {
            include_bytes!("../../shaders/spv/query-prefix.comp.spv")
        } else {
            include_bytes!("../../shaders/spv/query-prefix-nosubgroups.comp.spv")
        };
        let bindings = (0..3)
            .map(|i| Binding {
                resource: ResourceId::new(i),
                binding: i,
                kind: DescriptorKind::StorageBuffer,
                count: 1,
                sampler: None,
                readable: true,
                writable: i != 0,
            })
            .collect();
        let mut stage = crate::residency::helper_stage(ShaderStage::Compute, bytes, bindings, 16);
        stage.workgroup_size = Some([if extended { 256 } else { 512 }, 1, 1]);
        stage.shared_memory_bytes = if extended { 1024 } else { 16384 };
        let pipeline = compiler
            .request(PipelineRecipe::Compute(Arc::new(ShaderArtifact::new(
                Box::new([stage]),
            ))))
            .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
        let accumulation = self.query_buffer(8, "sample accumulation")?;
        unsafe { self.dispatch.cmd_fill_buffer(accumulation.handle, 0, 8, 0) };
        self.submission.samples.scan = Some(Scan {
            accumulation,
            pipeline,
            descriptors: DescriptorCache::new(self),
        });
        Ok(())
    }
    fn query_buffer(&self, bytes: u64, name: &str) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        self.allocate_buffer(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            AllocationMode::Managed,
            name,
        )
        .map(Arc::new)
    }
    pub(crate) fn set_samples_enabled(&mut self, enabled: bool) -> Result<(), VulkanError> {
        self.submission.samples.enabled = enabled;
        if enabled
            && self.submission.samples.active.is_none()
            && self
                .submission
                .samples
                .current
                .as_ref()
                .is_some_and(|b| b.used == CAPACITY)
        {
            self.end_rendering()?;
        }
        self.prepare_samples(enabled)
    }
    pub(crate) fn reset_samples_passed(&mut self) -> Result<(), VulkanError> {
        self.end_rendering()?;
        let samples = &mut self.submission.samples;
        // Abandon queries preceding this reset; their pools stay alive until
        // native completion. Previously exported reports own separate buffers.
        if let Some(bank) = &mut samples.current {
            bank.first = bank.used;
        }
        let accumulation = samples.scan.as_ref().ok_or(VulkanError::InvalidResource(
            "sample queries require the shared pipeline compiler",
        ))?;
        unsafe {
            self.dispatch
                .cmd_fill_buffer(accumulation.accumulation.handle, 0, 8, 0)
        };
        Ok(())
    }
    fn scan_samples(&mut self) -> Result<(), VulkanError> {
        let samples = &self.submission.samples;
        let Some(bank) = &samples.current else {
            return Ok(());
        };
        let count = bank.used - bank.first;
        if count == 0 {
            return Ok(());
        }
        let scan = samples.scan.as_ref().expect("initialized query scanner");
        self.dispatch.prepare_occlusion_results(
            bank.pool.handle,
            bank.first,
            count,
            bank.pool.input.handle,
            SCAN_BYTES,
        );
        let mut push = [0_u8; 16];
        for (i, value) in [count, count, count - 1, 0].into_iter().enumerate() {
            push[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let offsets = vec![0; scan.pipeline.layout.dynamic_bindings.len()];
        self.configure_pipeline(
            &scan.pipeline.clone(),
            &push,
            Some(&bank.pool.binding.clone()),
            &offsets,
            None,
        )?;
        unsafe { self.dispatch.cmd_dispatch(1, 1, 1) };
        self.dispatch.complete_query_scan();
        let work = self.current_work();
        let bank = self.submission.samples.current.as_mut().unwrap();
        bank.first = bank.used;
        bank.work = work;
        Ok(())
    }
    pub(crate) fn sample_report_buffer(&mut self) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        self.end_rendering()?;
        self.scan_samples()?;
        self.submission
            .samples
            .scan
            .as_ref()
            .map(|s| Arc::clone(&s.accumulation))
            .ok_or(VulkanError::InvalidResource(
                "sample queries require the shared pipeline compiler",
            ))
    }
    pub(crate) fn prepare_samples(&mut self, enabled: bool) -> Result<(), VulkanError> {
        if !enabled {
            self.submission.samples.pause(&mut self.dispatch);
            return Ok(());
        }
        let samples = &mut self.submission.samples;
        if samples.active.is_some() {
            return Ok(());
        }
        samples.collect(self.submission.progress.completed());
        if samples
            .current
            .as_ref()
            .is_some_and(|bank| bank.used == CAPACITY)
        {
            self.scan_samples()?;
            let bank = self.submission.samples.current.take().unwrap();
            self.submission.samples.pending.push_back(Pending {
                pool: bank.pool,
                work: bank.work,
            });
        }
        if self.submission.samples.current.is_none() {
            let pool =
                if let Some(pool) = self.submission.samples.free.pop() {
                    pool
                } else {
                    let input = self.query_buffer(SCAN_BYTES, "query input")?;
                    let output = self.query_buffer(SCAN_BYTES, "query prefix results")?;
                    let scan = self.submission.samples.scan.as_mut().ok_or(
                        VulkanError::InvalidResource(
                            "sample queries require the shared pipeline compiler",
                        ),
                    )?;
                    let binding = scan.descriptors.get(
                        &scan.pipeline.layout,
                        &[
                            DescriptorResource::Buffer {
                                buffer: Arc::clone(&input),
                                offset: 0,
                                range: SCAN_BYTES,
                            },
                            DescriptorResource::Buffer {
                                buffer: Arc::clone(&output),
                                offset: 0,
                                range: SCAN_BYTES,
                            },
                            DescriptorResource::Buffer {
                                buffer: Arc::clone(&scan.accumulation),
                                offset: 0,
                                range: 8,
                            },
                        ],
                    )?;
                    let handle = unsafe {
                        self.owner.device.create_query_pool(
                            &vk::QueryPoolCreateInfo::default()
                                .query_type(vk::QueryType::OCCLUSION)
                                .query_count(CAPACITY),
                            None,
                        )
                    }
                    .map_err(|e| VulkanError::vk("vkCreateQueryPool(occlusion)", e))?;
                    Arc::new(Pool {
                        owner: Arc::clone(&self.owner),
                        handle,
                        input,
                        _output: output,
                        binding,
                    })
                };
            self.dispatch.reset_occlusion_pool(pool.handle, CAPACITY);
            self.submission.samples.current = Some(Bank {
                pool,
                first: 0,
                used: 0,
                work: self.current_work(),
            });
        }
        Ok(())
    }
    pub(crate) fn begin_samples(&mut self, enabled: bool) {
        if !enabled || self.submission.samples.active.is_some() {
            return;
        }
        let work = self.current_work();
        let bank = self
            .submission
            .samples
            .current
            .as_mut()
            .expect("prepared sample query bank");
        let index = bank.used;
        bank.used += 1;
        bank.work = work;
        let pool = bank.pool.handle;
        self.dispatch
            .begin_occlusion(pool, index, self.profile.core.occlusion_query_precise != 0);
        self.submission.samples.active = Some((pool, index));
    }
}
