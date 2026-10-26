//! Select core/KHR function pointers once; recording has no API-version branch.
use crate::{VulkanError, profile::ApiPath};
use ash::{Device, Instance, vk};

mod arena;
mod record;
mod slot;
mod state;
mod worker;
pub(crate) use slot::CommandSlot;
use worker::Recorder;
pub(crate) use worker::{QueueSubmission, RecordingHandle};

#[derive(Clone, Copy)]
struct Functions {
    submit: vk::PFN_vkQueueSubmit2,
    barrier: vk::PFN_vkCmdPipelineBarrier2,
    begin: vk::PFN_vkCmdBeginRendering,
    end: vk::PFN_vkCmdEndRendering,
    #[cfg(feature = "binding-probe")]
    timestamp: vk::PFN_vkCmdWriteTimestamp2,
    state: Option<StateDispatch>,
    feedback: Option<vk::PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT>,
}

pub(crate) struct Dispatch {
    recorder: Recorder,
    pub state: Option<StateDispatch>,
    pub release_swapchain_images: Option<vk::PFN_vkReleaseSwapchainImagesEXT>,
}
impl Dispatch {
    pub(crate) fn begin_command_buffer(
        &mut self,
        slot: std::sync::Arc<CommandSlot>,
        tick: u64,
        progress: std::sync::Arc<crate::queue::progress::SubmissionProgress>,
    ) -> Result<(), VulkanError> {
        self.recorder.begin(slot, tick, progress)
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
        path: ApiPath,
        maintenance: crate::SwapchainMaintenance,
        dynamic: bool,
        feedback_loop: bool,
        queue_mutex: std::sync::Arc<std::sync::Mutex<()>>,
    ) -> Result<Self, VulkanError> {
        let (submit, barrier, begin, end) = match path {
            ApiPath::Core13 => {
                let f = device.fp_v1_3();
                (
                    f.queue_submit2,
                    f.cmd_pipeline_barrier2,
                    f.cmd_begin_rendering,
                    f.cmd_end_rendering,
                )
            }
            ApiPath::Khr12 => {
                let sync = ash::khr::synchronization2::Device::new(instance, device);
                let rendering = ash::khr::dynamic_rendering::Device::new(instance, device);
                (
                    sync.fp().queue_submit2_khr,
                    sync.fp().cmd_pipeline_barrier2_khr,
                    rendering.fp().cmd_begin_rendering_khr,
                    rendering.fp().cmd_end_rendering_khr,
                )
            }
        };
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
                feedback,
            },
            queue_mutex,
        )?;
        Ok(Self {
            recorder,
            state,
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
            ctx.flush_draw();
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
    pub fn recording_handle(&self) -> RecordingHandle {
        self.recorder.handle.clone()
    }
    pub unsafe fn cmd_set_feedback_loop(&mut self, aspects: vk::ImageAspectFlags) {
        self.recorder.record(move |ctx| {
            ctx.flush_draw();
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
    pub unsafe fn dynamic_state(&mut self, state: &crate::cache::GraphicsState) {
        let s = state::DynamicState::from(state);
        self.recorder.record(move |ctx| {
            ctx.flush_draw();
            let state = ctx
                .functions
                .state
                .as_ref()
                .expect("enabled native dynamic state");
            let command = ctx.command;
            unsafe {
                (state.cull)(command, s.cull);
                (state.front)(command, s.front_face);
                (state.depth_test)(command, u32::from(s.depth_test));
                (state.depth_write)(command, u32::from(s.depth_write));
                (state.depth_compare)(command, s.depth_compare);
                (state.depth_bounds)(command, u32::from(s.depth_bounds));
                (state.stencil_test)(command, u32::from(s.stencil_test));
                for (face, stencil) in [
                    (vk::StencilFaceFlags::FRONT, s.front),
                    (vk::StencilFaceFlags::BACK, s.back),
                ] {
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
            Ok(())
        });
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
                    ctx.flush_draw();
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
    pub unsafe fn begin_rendering(&mut self, info: &vk::RenderingInfo<'_>) {
        debug_assert!(info.p_next.is_null());
        let own = |v: &vk::RenderingAttachmentInfo<'_>| {
            debug_assert!(v.p_next.is_null());
            vk::RenderingAttachmentInfo::default()
                .image_view(v.image_view)
                .image_layout(v.image_layout)
                .resolve_mode(v.resolve_mode)
                .resolve_image_view(v.resolve_image_view)
                .resolve_image_layout(v.resolve_image_layout)
                .load_op(v.load_op)
                .store_op(v.store_op)
                .clear_value(v.clear_value)
        };
        let colors = if info.color_attachment_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    info.p_color_attachments,
                    info.color_attachment_count as usize,
                )
            }
        };
        debug_assert!(colors.iter().all(|v| v.p_next.is_null()));
        let depth = unsafe { info.p_depth_attachment.as_ref() }.map(own);
        let stencil = unsafe { info.p_stencil_attachment.as_ref() }.map(own);
        let flags = info.flags;
        let area = info.render_area;
        let layers = info.layer_count;
        let mask = info.view_mask;
        unsafe {
            self.recorder.record_array(colors, move |ctx, colors| {
                ctx.flush_draw();
                let mut info = vk::RenderingInfo::default()
                    .flags(flags)
                    .render_area(area)
                    .layer_count(layers)
                    .view_mask(mask)
                    .color_attachments(colors);
                if let Some(depth) = &depth {
                    info = info.depth_attachment(depth);
                }
                if let Some(stencil) = &stencil {
                    info = info.stencil_attachment(stencil);
                }
                (ctx.functions.begin)(ctx.command, &info);
                Ok(())
            })
        };
    }
    pub unsafe fn end_rendering(&mut self) {
        self.recorder.record(|ctx| {
            ctx.flush_draw();
            unsafe { (ctx.functions.end)(ctx.command) };
            Ok(())
        });
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StateDispatch {
    pub cull: vk::PFN_vkCmdSetCullMode,
    pub front: vk::PFN_vkCmdSetFrontFace,
    pub depth_test: vk::PFN_vkCmdSetDepthTestEnable,
    pub depth_write: vk::PFN_vkCmdSetDepthWriteEnable,
    pub depth_compare: vk::PFN_vkCmdSetDepthCompareOp,
    pub depth_bounds: vk::PFN_vkCmdSetDepthBoundsTestEnable,
    pub stencil_test: vk::PFN_vkCmdSetStencilTestEnable,
    pub stencil: vk::PFN_vkCmdSetStencilOp,
}
