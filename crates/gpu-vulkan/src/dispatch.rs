//! Select core/KHR function pointers once; recording has no API-version branch.
use crate::{VulkanError, profile::ApiPath};
use ash::{Device, Instance, vk};

pub(crate) struct Dispatch {
    pub state: Option<StateDispatch>,
    submit: vk::PFN_vkQueueSubmit2,
    barrier: vk::PFN_vkCmdPipelineBarrier2,
    begin: vk::PFN_vkCmdBeginRendering,
    end: vk::PFN_vkCmdEndRendering,
    #[cfg(feature = "binding-probe")]
    timestamp: vk::PFN_vkCmdWriteTimestamp2,
    pub release_swapchain_images: Option<vk::PFN_vkReleaseSwapchainImagesEXT>,
}
impl Dispatch {
    pub fn new(
        instance: &Instance,
        device: &Device,
        path: ApiPath,
        maintenance: crate::SwapchainMaintenance,
        dynamic: bool,
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
        Ok(Self {
            state,
            submit,
            barrier,
            begin,
            end,
            #[cfg(feature = "binding-probe")]
            timestamp,
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
        unsafe {
            (self.timestamp)(command, stage, pool, query);
        }
    }
    pub unsafe fn submit(
        &self,
        queue: vk::Queue,
        info: &vk::SubmitInfo2<'_>,
        fence: vk::Fence,
    ) -> Result<(), VulkanError> {
        // SAFETY: caller owns queue synchronization and retains submission data.
        let result = unsafe { (self.submit)(queue, 1, info, fence) };
        if result == vk::Result::SUCCESS {
            Ok(())
        } else {
            Err(VulkanError::vk("vkQueueSubmit2", result))
        }
    }
    pub unsafe fn barrier(&self, cmd: vk::CommandBuffer, info: &vk::DependencyInfo<'_>) {
        // SAFETY: caller provides a recording buffer and valid resource scopes.
        unsafe { (self.barrier)(cmd, info) }
    }
    pub unsafe fn begin_rendering(&self, cmd: vk::CommandBuffer, info: &vk::RenderingInfo<'_>) {
        // SAFETY: caller provides a recording buffer outside any rendering scope.
        unsafe { (self.begin)(cmd, info) }
    }
    pub unsafe fn end_rendering(&self, cmd: vk::CommandBuffer) {
        // SAFETY: caller pairs this with begin_rendering on this command buffer.
        unsafe { (self.end)(cmd) }
    }
}

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
