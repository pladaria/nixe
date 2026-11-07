//! Select core/KHR function pointers once; recording has no API-version branch.
use crate::{VulkanError, profile::ApiPath};
use ash::{Device, Instance, vk};

mod arena;
mod record;
mod slot;
mod state;
mod transfer;
pub(crate) use state::DynamicState;
mod worker;
pub(crate) use slot::CommandSlot;
use worker::Recorder;
pub(crate) use worker::{QueueSubmission, RecordingHandle};

#[derive(Clone, Copy)]
struct Functions {
    submit: vk::PFN_vkQueueSubmit2,
    barrier: vk::PFN_vkCmdPipelineBarrier2,
    begin: vk::PFN_vkCmdBeginRenderPass,
    end: vk::PFN_vkCmdEndRenderPass,
    #[cfg(feature = "binding-probe")]
    timestamp: vk::PFN_vkCmdWriteTimestamp2,
    state: Option<StateDispatch>,
    state2: Option<State2Dispatch>,
    feedback: Option<vk::PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT>,
    vertex_input: Option<vk::PFN_vkCmdSetVertexInputEXT>,
    vertex_buffers2: Option<vk::PFN_vkCmdBindVertexBuffers2>,
    blending: Option<BlendingDispatch>,
    push_descriptor: Option<vk::PFN_vkCmdPushDescriptorSetWithTemplateKHR>,
}

#[derive(Clone, Copy)]
struct BlendingDispatch {
    enable: vk::PFN_vkCmdSetColorBlendEnableEXT,
    equation: vk::PFN_vkCmdSetColorBlendEquationEXT,
    mask: vk::PFN_vkCmdSetColorWriteMaskEXT,
}

pub(crate) struct Dispatch {
    recorder: Recorder,
    pub dynamic_mask: u16,
    pub release_swapchain_images: Option<vk::PFN_vkReleaseSwapchainImagesEXT>,
}

impl Dispatch {
    pub(crate) fn begin_command_buffer(
        &mut self,
        slot: std::sync::Arc<CommandSlot>,
        tick: u64,
        progress: std::sync::Arc<crate::queue::progress::SubmissionProgress>,
        timeline: vk::Semaphore,
    ) -> Result<(), VulkanError> {
        self.recorder.begin(slot, tick, progress, timeline)
    }
    pub(crate) fn submit(&mut self, work: QueueSubmission) -> Result<(), VulkanError> {
        self.recorder.submit(work)
    }

    pub(crate) fn discard_pending(&mut self) {
        self.recorder.discard_pending();
    }

    pub fn new(
        instance: &Instance,
        device: &Device,
        profile: &crate::DeviceProfile,
        queue_mutex: std::sync::Arc<std::sync::Mutex<()>>,
    ) -> Result<Self, VulkanError> {
        let path = profile.api_path;
        let maintenance = profile.swapchain_maintenance;
        let dynamic = profile.dynamic_state.base;
        let feedback_loop = profile.attachment_feedback_loop;
        let push_descriptor = profile.max_push_descriptors != 0;
        let (submit, barrier) = match path {
            ApiPath::Core13 => (
                device.fp_v1_3().queue_submit2,
                device.fp_v1_3().cmd_pipeline_barrier2,
            ),
            ApiPath::Khr12 => {
                let sync = ash::khr::synchronization2::Device::new(instance, device);
                (
                    sync.fp().queue_submit2_khr,
                    sync.fp().cmd_pipeline_barrier2_khr,
                )
            }
        };
        let begin = device.fp_v1_0().cmd_begin_render_pass;
        let end = device.fp_v1_0().cmd_end_render_pass;
        #[cfg(feature = "binding-probe")]
        let timestamp = match path {
            ApiPath::Core13 => device.fp_v1_3().cmd_write_timestamp2,
            ApiPath::Khr12 => {
                ash::khr::synchronization2::Device::new(instance, device)
                    .fp()
                    .cmd_write_timestamp2_khr
            }
        };
        let release_swapchain_images = match maintenance {
            crate::SwapchainMaintenance::Disabled => None,
            crate::SwapchainMaintenance::Ext | crate::SwapchainMaintenance::Khr => {
                let name = if maintenance == crate::SwapchainMaintenance::Khr {
                    c"vkReleaseSwapchainImagesKHR"
                } else {
                    c"vkReleaseSwapchainImagesEXT"
                };
                // SAFETY: an enabled extension defines this ABI; KHR/EXT aliases
                // have identical parameters in the registry. Resolve once.
                let raw = unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }
                    .ok_or(VulkanError::InvalidResource(
                        "enabled swapchain maintenance command is missing",
                    ))?;
                Some(unsafe {
                    std::mem::transmute::<
                        unsafe extern "system" fn(),
                        vk::PFN_vkReleaseSwapchainImagesEXT,
                    >(raw)
                })
            }
        };
        let state = dynamic.then(|| match path {
            ApiPath::Core13 => {
                let f = device.fp_v1_3();
                StateDispatch {
                    topology: f.cmd_set_primitive_topology,
                    cull: f.cmd_set_cull_mode,
                    front: f.cmd_set_front_face,
                    depth_test: f.cmd_set_depth_test_enable,
                    depth_write: f.cmd_set_depth_write_enable,
                    depth_compare: f.cmd_set_depth_compare_op,
                    depth_bounds: f.cmd_set_depth_bounds_test_enable,
                    stencil_test: f.cmd_set_stencil_test_enable,
                    stencil: f.cmd_set_stencil_op,
                }
            }
            ApiPath::Khr12 => {
                let extension = ash::ext::extended_dynamic_state::Device::new(instance, device);
                let f = extension.fp();
                StateDispatch {
                    topology: f.cmd_set_primitive_topology_ext,
                    cull: f.cmd_set_cull_mode_ext,
                    front: f.cmd_set_front_face_ext,
                    depth_test: f.cmd_set_depth_test_enable_ext,
                    depth_write: f.cmd_set_depth_write_enable_ext,
                    depth_compare: f.cmd_set_depth_compare_op_ext,
                    depth_bounds: f.cmd_set_depth_bounds_test_enable_ext,
                    stencil_test: f.cmd_set_stencil_test_enable_ext,
                    stencil: f.cmd_set_stencil_op_ext,
                }
            }
        });
        let state2 = profile.dynamic_state.base2.then(|| match path {
            ApiPath::Core13 => {
                let f = device.fp_v1_3();
                State2Dispatch {
                    restart: f.cmd_set_primitive_restart_enable,
                    discard: f.cmd_set_rasterizer_discard_enable,
                    bias: f.cmd_set_depth_bias_enable,
                }
            }
            ApiPath::Khr12 => {
                let extension = ash::ext::extended_dynamic_state2::Device::new(instance, device);
                let f = extension.fp();
                State2Dispatch {
                    restart: f.cmd_set_primitive_restart_enable_ext,
                    discard: f.cmd_set_rasterizer_discard_enable_ext,
                    bias: f.cmd_set_depth_bias_enable_ext,
                }
            }
        });
        let dynamic_mask =
            if state.is_some() { 0x3ff } else { 0 } | if state2.is_some() { 0x1c00 } else { 0 };
        let feedback = feedback_loop.then(|| {
            ash::ext::attachment_feedback_loop_dynamic_state::Device::new(instance, device)
                .fp()
                .cmd_set_attachment_feedback_loop_enable_ext
        });
        let recorder = Recorder::new(
            device.clone(),
            Functions {
                submit,
                barrier,
                begin,
                end,
                #[cfg(feature = "binding-probe")]
                timestamp,
                state,
                state2,
                feedback,
                vertex_buffers2: dynamic.then(|| match path {
                    ApiPath::Core13 => device.fp_v1_3().cmd_bind_vertex_buffers2,
                    ApiPath::Khr12 => {
                        ash::ext::extended_dynamic_state::Device::new(instance, device)
                            .fp()
                            .cmd_bind_vertex_buffers2_ext
                    }
                }),
                vertex_input: profile.dynamic_state.vertex_input.then(|| {
                    ash::ext::vertex_input_dynamic_state::Device::new(instance, device)
                        .fp()
                        .cmd_set_vertex_input_ext
                }),
                blending: profile.dynamic_state.blending().then(|| {
                    let extension =
                        ash::ext::extended_dynamic_state3::Device::new(instance, device);
                    let f = extension.fp();
                    BlendingDispatch {
                        enable: f.cmd_set_color_blend_enable_ext,
                        equation: f.cmd_set_color_blend_equation_ext,
                        mask: f.cmd_set_color_write_mask_ext,
                    }
                }),
                push_descriptor: push_descriptor.then(|| {
                    ash::khr::push_descriptor::Device::new(instance, device)
                        .fp()
                        .cmd_push_descriptor_set_with_template_khr
                }),
            },
            queue_mutex,
        )?;
        Ok(Self {
            recorder,
            dynamic_mask,
            release_swapchain_images,
        })
    }
    #[cfg(feature = "binding-probe")]
    pub unsafe fn timestamp(
        &mut self,
        stage: vk::PipelineStageFlags2,
        pool: vk::QueryPool,
        index: u32,
    ) {
        self.recorder.record(move |ctx| {
            unsafe { (ctx.functions.timestamp)(ctx.command, stage, pool, index) };
            Ok(())
        });
    }
    #[cfg(feature = "binding-probe")]
    pub(crate) unsafe fn record<F>(&mut self, function: F)
    where
        F: Fn(&mut worker::Context) -> Result<(), crate::queue::progress::NativeFailure>
            + Send
            + 'static,
    {
        self.recorder.record(function);
    }
    pub fn dispatch_work(&mut self) -> Result<(), VulkanError> {
        self.recorder.flush_chunks()
    }
    /// Scheduler::WaitWorker dispatches the current chunk and waits only for
    /// native recording, independently of queue/GPU completion.
    pub(crate) fn wait_worker(&mut self) -> Result<(), VulkanError> {
        self.recorder.finish()
    }
    pub fn begin_submission_timing(&mut self, pool: vk::QueryPool) {
        self.recorder.record(move |ctx| {
            unsafe {
                ctx.device.cmd_reset_query_pool(ctx.upload, pool, 0, 2);
                ctx.device.cmd_write_timestamp(
                    ctx.upload,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    0,
                );
            }
            Ok(())
        });
    }
    pub fn recording_handle(&self) -> RecordingHandle {
        self.recorder.handle.clone()
    }
    pub unsafe fn cmd_set_feedback_loop(&mut self, aspects: vk::ImageAspectFlags) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .feedback
                    .expect("enabled attachment feedback state"))(
                    ctx.command, aspects
                )
            };
            Ok(())
        });
    }
    pub unsafe fn cmd_set_cull_mode(&mut self, value: vk::CullModeFlags) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .cull)(ctx.command, value);
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_front_face(&mut self, value: vk::FrontFace) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .front)(ctx.command, value);
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_test_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .depth_test)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_write_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .depth_write)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_compare_op(&mut self, value: vk::CompareOp) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .depth_compare)(ctx.command, value);
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_bounds_test_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .depth_bounds)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_test_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .stencil_test)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_primitive_topology(&mut self, value: vk::PrimitiveTopology) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .topology)(ctx.command, value);
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_primitive_restart_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state2
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .restart)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_rasterizer_discard_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state2
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .discard)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_depth_bias_enable(&mut self, value: bool) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state2
                    .as_ref()
                    .expect("enabled native dynamic state")
                    .bias)(ctx.command, u32::from(value));
            }
            Ok(())
        });
    }
    pub unsafe fn cmd_set_stencil_op(
        &mut self,
        face: vk::StencilFaceFlags,
        value: crate::cache::StencilState,
    ) {
        self.recorder.record(move |ctx| {
            unsafe {
                (ctx.functions
                    .state
                    .as_ref()
                    .expect("enabled native stencil state")
                    .stencil)(
                    ctx.command,
                    face,
                    value.fail,
                    value.pass,
                    value.depth_fail,
                    value.compare,
                );
            }
            Ok(())
        });
    }
    pub unsafe fn dynamic_state(&mut self, state: &crate::cache::GraphicsState) -> u16 {
        let s = DynamicState::from(state);
        let changed = self.dynamic_mask;
        if changed == 0 {
            return 0;
        }
        self.recorder.record(move |ctx| {
            let command = ctx.command;
            unsafe {
                if let Some(state) = ctx.functions.state.as_ref() {
                    if changed & (1 << 9) != 0 {
                        (state.topology)(command, s.topology);
                    }
                    if changed & (1 << 0) != 0 {
                        (state.cull)(command, s.cull);
                    }
                    if changed & (1 << 1) != 0 {
                        (state.front)(command, s.front_face);
                    }
                    if changed & (1 << 2) != 0 {
                        (state.depth_test)(command, u32::from(s.depth_test));
                    }
                    if changed & (1 << 3) != 0 {
                        (state.depth_write)(command, u32::from(s.depth_write));
                    }
                    if changed & (1 << 4) != 0 {
                        (state.depth_compare)(command, s.depth_compare);
                    }
                    if changed & (1 << 5) != 0 {
                        (state.depth_bounds)(command, u32::from(s.depth_bounds));
                    }
                    if changed & (1 << 6) != 0 {
                        (state.stencil_test)(command, u32::from(s.stencil_test));
                    }
                    for (index, (face, stencil)) in [
                        (vk::StencilFaceFlags::FRONT, s.front),
                        (vk::StencilFaceFlags::BACK, s.back),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if changed & (1 << (7 + index)) == 0 {
                            continue;
                        }
                        (state.stencil)(
                            command,
                            face,
                            stencil.fail,
                            stencil.pass,
                            stencil.depth_fail,
                            stencil.compare,
                        );
                    }
                }
                if let Some(state) = ctx.functions.state2.as_ref() {
                    if changed & (1 << 10) != 0 {
                        (state.restart)(command, u32::from(s.primitive_restart));
                    }
                    if changed & (1 << 11) != 0 {
                        (state.discard)(command, u32::from(s.raster_discard));
                    }
                    if changed & (1 << 12) != 0 {
                        (state.bias)(command, u32::from(s.depth_bias));
                    }
                }
            }
            Ok(())
        });
        changed
    }
    /// All chains must be null; array elements contain only native handles and scalars.
    pub unsafe fn barrier(&mut self, info: &vk::DependencyInfo<'_>) {
        debug_assert!(info.p_next.is_null());
        let memory = if info.memory_barrier_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    info.p_memory_barriers,
                    info.memory_barrier_count as usize,
                )
            }
        };
        let buffers = if info.buffer_memory_barrier_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    info.p_buffer_memory_barriers,
                    info.buffer_memory_barrier_count as usize,
                )
            }
        };
        let images = if info.image_memory_barrier_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    info.p_image_memory_barriers,
                    info.image_memory_barrier_count as usize,
                )
            }
        };
        debug_assert!(
            memory.iter().all(|v| v.p_next.is_null())
                && buffers.iter().all(|v| v.p_next.is_null())
                && images.iter().all(|v| v.p_next.is_null())
        );
        let flags = info.dependency_flags;
        // SAFETY: these structs have no borrowed fields beyond their null pNext.
        // Copy their owned values directly into the capture arena, without a temporary vector.
        unsafe {
            self.recorder.record_arrays(
                memory,
                buffers,
                images,
                move |ctx, memory, buffers, images| {
                    let info = vk::DependencyInfo::default()
                        .dependency_flags(flags)
                        .memory_barriers(memory)
                        .buffer_memory_barriers(buffers)
                        .image_memory_barriers(images);
                    (ctx.functions.barrier)(ctx.command, &info);
                    Ok(())
                },
            )
        };
    }
    pub unsafe fn begin_render_pass(&mut self, info: &vk::RenderPassBeginInfo<'_>) {
        debug_assert!(info.p_next.is_null());
        let clears = if info.clear_value_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(info.p_clear_values, info.clear_value_count as usize)
            }
        };
        let pass = info.render_pass;
        let framebuffer = info.framebuffer;
        let area = info.render_area;
        unsafe {
            self.recorder.record_array(clears, move |ctx, clear| {
                let info = vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(clear);
                (ctx.functions.begin)(ctx.command, &info, vk::SubpassContents::INLINE);
                Ok(())
            });
        }
    }
    pub unsafe fn end_rendering(&mut self) {
        self.recorder.record(|ctx| {
            unsafe { (ctx.functions.end)(ctx.command) };
            Ok(())
        });
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StateDispatch {
    pub topology: vk::PFN_vkCmdSetPrimitiveTopology,
    pub cull: vk::PFN_vkCmdSetCullMode,
    pub front: vk::PFN_vkCmdSetFrontFace,
    pub depth_test: vk::PFN_vkCmdSetDepthTestEnable,
    pub depth_write: vk::PFN_vkCmdSetDepthWriteEnable,
    pub depth_compare: vk::PFN_vkCmdSetDepthCompareOp,
    pub depth_bounds: vk::PFN_vkCmdSetDepthBoundsTestEnable,
    pub stencil_test: vk::PFN_vkCmdSetStencilTestEnable,
    pub stencil: vk::PFN_vkCmdSetStencilOp,
}

#[derive(Clone, Copy)]
struct State2Dispatch {
    restart: vk::PFN_vkCmdSetPrimitiveRestartEnable,
    discard: vk::PFN_vkCmdSetRasterizerDiscardEnable,
    bias: vk::PFN_vkCmdSetDepthBiasEnable,
}
