//! Select core/KHR function pointers once; recording has no API-version branch.
use crate::{VulkanError, profile::ApiPath};
use ash::{Device, Instance, vk};

mod commands;
mod record;
mod worker;
pub(crate) use commands::Command;
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
}

pub(crate) struct Dispatch {
    recorder: Recorder,
    pub state: Option<StateDispatch>,
    pub release_swapchain_images: Option<vk::PFN_vkReleaseSwapchainImagesEXT>,
}
impl Dispatch {
    pub fn new(
        instance: &Instance,
        device: &Device,
        path: ApiPath,
        maintenance: crate::SwapchainMaintenance,
        dynamic: bool,
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
        &self,
        command: vk::CommandBuffer,
        stage: vk::PipelineStageFlags2,
        pool: vk::QueryPool,
        query: u32,
    ) {
        self.recorder.push(Command::Timestamp {
            command,
            stage,
            pool,
            index: query,
        });
    }
    pub fn recording_handle(&self) -> RecordingHandle {
        self.recorder.handle.clone()
    }
    pub unsafe fn dynamic_state(
        &self,
        command: vk::CommandBuffer,
        state: &crate::cache::GraphicsState,
    ) {
        self.recorder.push(Command::Dynamic {
            command,
            state: commands::DynamicState::from(state),
        });
    }
    pub unsafe fn barrier(&self, command: vk::CommandBuffer, info: &vk::DependencyInfo<'_>) {
        debug_assert!(info.p_next.is_null());
        // SAFETY: the caller supplies live arrays; all entries are copied now.
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
        self.recorder.push(Command::Barrier {
            command,
            flags: info.dependency_flags,
            memory: memory
                .iter()
                .map(|v| {
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(v.src_stage_mask)
                        .src_access_mask(v.src_access_mask)
                        .dst_stage_mask(v.dst_stage_mask)
                        .dst_access_mask(v.dst_access_mask)
                })
                .collect(),
            buffers: buffers
                .iter()
                .map(|v| {
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(v.src_stage_mask)
                        .src_access_mask(v.src_access_mask)
                        .dst_stage_mask(v.dst_stage_mask)
                        .dst_access_mask(v.dst_access_mask)
                        .src_queue_family_index(v.src_queue_family_index)
                        .dst_queue_family_index(v.dst_queue_family_index)
                        .buffer(v.buffer)
                        .offset(v.offset)
                        .size(v.size)
                })
                .collect(),
            images: images
                .iter()
                .map(|v| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(v.src_stage_mask)
                        .src_access_mask(v.src_access_mask)
                        .dst_stage_mask(v.dst_stage_mask)
                        .dst_access_mask(v.dst_access_mask)
                        .src_queue_family_index(v.src_queue_family_index)
                        .dst_queue_family_index(v.dst_queue_family_index)
                        .image(v.image)
                        .old_layout(v.old_layout)
                        .new_layout(v.new_layout)
                        .subresource_range(v.subresource_range)
                })
                .collect(),
        });
    }
    pub unsafe fn begin_rendering(&self, command: vk::CommandBuffer, info: &vk::RenderingInfo<'_>) {
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
        // SAFETY: all caller-owned attachments are copied into this command.
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
        let depth = unsafe { info.p_depth_attachment.as_ref() }.map(own);
        let stencil = unsafe { info.p_stencil_attachment.as_ref() }.map(own);
        self.recorder.push(Command::BeginRendering {
            command,
            flags: info.flags,
            area: info.render_area,
            layers: info.layer_count,
            mask: info.view_mask,
            colors: colors.iter().map(own).collect(),
            depth,
            stencil,
        });
    }
    pub unsafe fn end_rendering(&self, command: vk::CommandBuffer) {
        self.recorder.push(Command::EndRendering { command });
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
