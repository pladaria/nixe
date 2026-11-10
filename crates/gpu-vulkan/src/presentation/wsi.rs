//! Copy completion protects frame storage; acquisition protects per-image semaphores.
//! https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
use super::*;
use crate::device::DeviceOwner;
use ash::vk;

pub(super) fn select_present_mode(
    vsync: bool,
    modes: &[vk::PresentModeKHR],
) -> Result<vk::PresentModeKHR, VulkanError> {
    // IMMEDIATE does not wait for vertical blank; MAILBOX still does.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPresentModeKHR.html
    let mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        vk::PresentModeKHR::IMMEDIATE
    };
    if modes.contains(&mode) {
        Ok(mode)
    } else {
        Err(VulkanError::InvalidResource(if vsync {
            "surface does not support FIFO presentation for VSync"
        } else {
            "VSync is disabled, but the surface does not support IMMEDIATE presentation"
        }))
    }
}

fn surface_format(owner: &DeviceOwner) -> Result<vk::SurfaceFormatKHR, VulkanError> {
    let surface = owner
        .instance
        .surface
        .as_ref()
        .ok_or(VulkanError::InvalidResource("presentation has no surface"))?;
    let formats = unsafe {
        surface
            .loader
            .get_physical_device_surface_formats(owner.physical, surface.handle)
    }
    .map_err(|e| VulkanError::vk("surface formats", e))?;
    let format = formats
        .iter()
        .find(|f| {
            f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                && matches!(
                    f.format,
                    vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
                )
        })
        .or_else(|| {
            formats.iter().find(|f| {
                f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    && matches!(
                        f.format,
                        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                    )
            })
        })
        .copied()
        .ok_or(VulkanError::InvalidResource(
            "surface requires nonlinear RGBA8/BGRA8 UNORM or sRGB",
        ))?;

    Ok(format)
}

/// Native scheduling synchronization crosses the owner boundary without guest reads.
struct Context {
    progress: Arc<crate::queue::progress::SubmissionProgress>,
    owner: Arc<DeviceOwner>,
    queues: crate::device::QueueSelection,
    timeline: vk::Semaphore,
    vsync: bool,
}
impl Context {
    fn drain(&self, work: WorkTicket) -> Result<(), VulkanError> {
        // The owner publishes complete work before handing out this ticket.
        // Wait for the native worker to accept it before querying the timeline.
        let _ = self.progress.wait_submitted(work.tick(), None);
        if self
            .progress
            .failure()
            .is_some_and(|failure| failure.cause.device_lost())
        {
            return Ok(());
        }
        let tick = work.tick().min(self.progress.submitted());
        if tick == 0 {
            return Ok(());
        }
        let semaphores = [self.timeline];
        let values = [tick];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        match unsafe { self.owner.device.wait_semaphores(&info, u64::MAX) } {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => Ok(()),
            Err(error) => Err(VulkanError::vk("WSI graphics drain", error)),
        }
    }
}
pub(crate) struct CopyCommands {
    owner: Arc<DeviceOwner>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}
impl CopyCommands {
    pub(super) fn new(device: &VulkanDevice) -> Result<Arc<Self>, VulkanError> {
        let pool = unsafe {
            device.owner.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.queues().graphics_compute_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("presentation command pool", e))?;
        let mut result = Self {
            owner: Arc::clone(&device.owner),
            pool,
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };
        result.command = unsafe {
            device.owner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| VulkanError::vk("presentation command buffer", e))?[0];
        result.fence = unsafe {
            device.owner.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("presentation copy fence", e))?;
        Ok(Arc::new(result))
    }
    fn ready(&self) -> Result<bool, VulkanError> {
        unsafe { self.owner.device.get_fence_status(self.fence) }
            .map_err(|e| VulkanError::vk("presentation copy completion", e))
    }
    fn wait(&self) -> Result<(), VulkanError> {
        unsafe {
            self.owner
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
        }
        .or_else(|e| {
            if e == vk::Result::ERROR_DEVICE_LOST {
                Ok(())
            } else {
                Err(e)
            }
        })
        .map_err(|e| VulkanError::vk("presentation copy retirement", e))
    }
    fn submit(
        &self,
        context: &Context,
        frame: &PresentFrame,
        target: &Acquired,
    ) -> Result<(), VulkanError> {
        // A frame's command buffer and source remain in use until its copy completes.
        self.wait()?;
        context.progress.wait_submitted(frame.work.tick(), None)?;
        let d = &self.owner.device;
        let source = frame.storage.view.image.handle;
        let range = super::render::color_range();
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let barrier = |image, old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(range)
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src)
                .dst_access_mask(dst)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        };
        let before = [
            barrier(
                source,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            ),
            barrier(
                target.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            ),
        ];
        let after = [
            barrier(
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            ),
            barrier(
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ,
            ),
        ];
        unsafe {
            d.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .map_err(|e| VulkanError::vk("reset presentation command buffer", e))?;
            d.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| VulkanError::vk("begin presentation copy", e))?;
            d.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &before,
            );
            if target.blit_supported {
                let region = vk::ImageBlit::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: frame.storage.extent[0] as i32,
                            y: frame.storage.extent[1] as i32,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: target.extent.width as i32,
                            y: target.extent.height as i32,
                            z: 1,
                        },
                    ]);
                d.cmd_blit_image(
                    self.command,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    target.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                    vk::Filter::LINEAR,
                );
            } else {
                if frame.storage.format != target.format {
                    return Err(VulkanError::InvalidResource(
                        "presentation format conversion requires BLIT_DST support",
                    ));
                }
                let region = vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width: frame.storage.extent[0].min(target.extent.width),
                        height: frame.storage.extent[1].min(target.extent.height),
                        depth: 1,
                    });
                d.cmd_copy_image(
                    self.command,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    target.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
            d.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &after,
            );
            d.end_command_buffer(self.command)
                .map_err(|e| VulkanError::vk("end presentation copy", e))?;
            d.reset_fences(&[self.fence])
                .map_err(|e| VulkanError::vk("reset presentation copy fence", e))?;
            let semaphores = [context.timeline, target.wait];
            let values = [frame.work.tick(), 0];
            let signals = [target.signal];
            let signal_values = [0];
            let stages = [vk::PipelineStageFlags::TRANSFER; 2];
            let commands = [self.command];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&values)
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .command_buffers(&commands)
                .wait_semaphores(&semaphores)
                .wait_dst_stage_mask(&stages)
                .signal_semaphores(&signals)
                .push_next(&mut timeline);
            let _queue = self
                .owner
                .queue_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let queue = d.get_device_queue(context.queues.graphics_compute_family, 0);
            d.queue_submit(queue, &[submit], self.fence)
                .map_err(|e| VulkanError::vk("submit presentation copy", e))?;
        }
        nixe_gpu::trace::event(
            "present_copy",
            frame.work.tick(),
            u64::from(target.extent.width),
            u64::from(target.extent.height),
        );
        Ok(())
    }
}
impl Drop for CopyCommands {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_fence(self.fence, None);
            self.owner.device.destroy_command_pool(self.pool, None);
        }
    }
}
struct Image {
    handle: vk::Image,
    copy: Option<Arc<CopyCommands>>,
    copy_submitted: bool,
    source: Option<Arc<FrameStorage>>,
    finished: vk::Semaphore,
}
struct Chain {
    owner: Arc<DeviceOwner>,
    loader: ash::khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<Image>,
    format: vk::Format,
    blit_supported: bool,
    extent: vk::Extent2D,
    last_work: Option<WorkTicket>,
}
impl Drop for Chain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.handle, None);
            for image in &mut self.images {
                image.source.take();
                image.copy.take();
                self.owner.device.destroy_semaphore(image.finished, None);
            }
        }
    }
}
struct Acquire {
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    copy: Option<Arc<CopyCommands>>,
    issued: bool,
}
#[derive(Clone)]
pub(crate) struct Acquired {
    pub format: vk::Format,
    blit_supported: bool,
    pub image: vk::Image,
    pub extent: vk::Extent2D,
    pub wait: vk::Semaphore,
    pub signal: vk::Semaphore,
    index: u32,
    acquire: usize,
}
struct Swapchains {
    owner: Arc<DeviceOwner>,
    current: Option<Chain>,
    acquires: Vec<Acquire>,
    held: Option<(u32, usize, bool)>,
    dirty: bool,
    requested: [u32; 2],
    present_queue: vk::Queue,
    safe_to_drop: bool,
    vsync: bool,
}
impl Swapchains {
    pub fn new(device: &Context) -> Result<Self, VulkanError> {
        let family = device.queues.present_family.unwrap();
        let present_queue = unsafe { device.owner.device.get_device_queue(family, 0) };
        let result = Self {
            owner: Arc::clone(&device.owner),
            current: None,
            acquires: Vec::new(),
            held: None,
            dirty: true,
            requested: [0, 0],
            present_queue,
            safe_to_drop: true,
            vsync: device.vsync,
        };
        Ok(result)
    }
    fn reserve_acquires(&mut self, count: usize) -> Result<(), VulkanError> {
        while self.acquires.len() < count {
            let semaphore = unsafe {
                self.owner
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|e| VulkanError::vk("acquire semaphore", e))?;
            self.acquires.push(Acquire {
                semaphore,
                fence: vk::Fence::null(),
                copy: None,
                issued: false,
            });
            let fence = unsafe {
                self.owner
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .map_err(|e| VulkanError::vk("acquire fence", e))?;
            self.acquires.last_mut().unwrap().fence = fence;
        }
        Ok(())
    }
    pub fn resize(&mut self, size: [u32; 2]) {
        if size != self.requested {
            self.requested = size;
            self.dirty = true;
        }
    }
    pub fn update(&mut self, device: &Context) -> Result<bool, VulkanError> {
        if let Some(chain) = &mut self.current {
            for image in &mut chain.images {
                if image.copy_submitted && image.copy.as_ref().unwrap().ready()? {
                    image.source = None;
                }
            }
        }
        if self.requested.contains(&0) {
            return Ok(false);
        }
        if self.held.is_some() {
            return Ok(true);
        }
        if !self.dirty {
            return Ok(true);
        }
        let surface = device.owner.instance.surface.as_ref().unwrap();
        let capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(device.owner.physical, surface.handle)
        }
        .map_err(|e| VulkanError::vk("surface capabilities", e))?;
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: self.requested[0].clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: self.requested[1].clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        let format = surface_format(&device.owner)?;
        if !capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            return Err(VulkanError::InvalidResource(
                "surface does not support presentation transfers",
            ));
        }
        let mut count = capabilities.min_image_count.saturating_add(1).max(3);
        if capabilities.max_image_count != 0 {
            count = count.min(capabilities.max_image_count);
        }
        if !capabilities
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            return Err(VulkanError::InvalidResource(
                "native presentation requires opaque window compositing",
            ));
        }
        if !capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            return Err(VulkanError::InvalidResource(
                "native presentation requires an identity surface transform",
            ));
        }
        let modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(device.owner.physical, surface.handle)
        }
        .map_err(|e| VulkanError::vk("surface present modes", e))?;
        let present_mode = select_present_mode(self.vsync, &modes)?;
        log::info!(
            "Vulkan presentation: vsync={} mode={present_mode:?}",
            self.vsync
        );
        let graphics = device.queues.graphics_compute_family;
        let present = device.queues.present_family.unwrap();
        let families = [graphics, present];
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(
                self.current
                    .as_ref()
                    .map_or(vk::SwapchainKHR::null(), |c| c.handle),
            );
        // Separate present families share this WSI image explicitly; ordinary
        // guest resources remain exclusive to the graphics/compute owner.
        if graphics != present {
            info = info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        let loader =
            ash::khr::swapchain::Device::new(&device.owner.instance.instance, &device.owner.device);
        // Resize retires the previous native work before replacing its swapchain.
        // Ordinary frames keep using the asynchronous copy/present worker.
        self.drain(device)?;
        let handle = unsafe { loader.create_swapchain(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateSwapchainKHR", e))?;
        let mut chain = Chain {
            owner: Arc::clone(&device.owner),
            loader,
            handle,
            images: Vec::new(),
            format: format.format,
            blit_supported: unsafe {
                device
                    .owner
                    .instance
                    .instance
                    .get_physical_device_format_properties(device.owner.physical, format.format)
            }
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::BLIT_DST),
            extent,
            last_work: None,
        };
        for image in unsafe { chain.loader.get_swapchain_images(handle) }
            .map_err(|e| VulkanError::vk("swapchain images", e))?
        {
            chain.images.push(Image {
                handle: image,
                copy: None,
                copy_submitted: false,
                source: None,
                finished: vk::Semaphore::null(),
            });
            let entry = chain.images.last_mut().unwrap();
            entry.finished = unsafe {
                device
                    .owner
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|e| VulkanError::vk("present semaphore", e))?;
        }
        self.reserve_acquires(chain.images.len())?;
        self.current = Some(chain);
        self.dirty = false;
        Ok(true)
    }
    pub fn acquire(&mut self, device: &Context) -> Result<Option<Acquired>, VulkanError> {
        let Some(chain) = &self.current else {
            return Ok(None);
        };
        let (index, acquire, suboptimal) = if let Some(held) = self.held {
            held
        } else {
            let mut available = None;
            for (index, acquire) in self.acquires.iter().enumerate() {
                if let Some(copy) = &acquire.copy
                    && !copy.ready()?
                {
                    continue;
                }
                if acquire.issued
                    && !unsafe { device.owner.device.get_fence_status(acquire.fence) }
                        .map_err(|e| VulkanError::vk("acquire completion status", e))?
                {
                    continue;
                }
                available = Some(index);
                break;
            }
            let Some(acquire) = available else {
                return Ok(None);
            };
            self.acquires[acquire].issued = false;
            let a = &self.acquires[acquire];
            unsafe { device.owner.device.reset_fences(&[a.fence]) }
                .map_err(|e| VulkanError::vk("reset acquire fence", e))?;
            // Smooth Motion can keep returning TIMEOUT when polled with zero.
            // Wait briefly on this WSI worker so interposed presentation can
            // advance, while keeping shutdown responsive if no image is ready.
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkAcquireNextImageKHR.html
            match unsafe {
                chain
                    .loader
                    .acquire_next_image(chain.handle, 1_000_000, a.semaphore, a.fence)
            } {
                Ok((index, suboptimal)) => {
                    self.acquires[acquire].issued = true;
                    self.held = Some((index, acquire, suboptimal));
                    self.safe_to_drop = false;
                    (index, acquire, suboptimal)
                }
                Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => return Ok(None),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.dirty = true;
                    return Ok(None);
                }
                Err(error) => return Err(VulkanError::vk("vkAcquireNextImageKHR", error)),
            }
        };
        // A ready acquire fence proves the WSI handoff has finished. No
        // unavailable image is allowed to gate ordinary guest queue progress.
        if !unsafe {
            device
                .owner
                .device
                .get_fence_status(self.acquires[acquire].fence)
        }
        .map_err(|e| VulkanError::vk("acquired image readiness", e))?
        {
            return Ok(None);
        }
        let image = &chain.images[index as usize];
        if image.copy_submitted && !image.copy.as_ref().unwrap().ready()? {
            return Ok(None);
        }
        // Acquiring this image and completing its acquire fence also completes
        // the previous presentation's use of its per-image wait semaphore.
        // https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
        self.dirty |= suboptimal;
        Ok(Some(Acquired {
            format: chain.format,
            blit_supported: chain.blit_supported,
            image: image.handle,
            extent: chain.extent,
            wait: self.acquires[acquire].semaphore,
            signal: image.finished,
            index,
            acquire,
        }))
    }
    pub fn present(
        &mut self,
        device: &Context,
        frame: Acquired,
        source: PresentFrame,
    ) -> Result<bool, VulkanError> {
        let work = source.work;
        let chain = self.current.as_mut().unwrap();
        let image = &mut chain.images[frame.index as usize];
        let copy = &source.storage.copy;
        copy.submit(device, &source, &frame)?;
        image.copy = Some(Arc::clone(copy));
        image.copy_submitted = true;
        self.acquires[frame.acquire].copy = Some(Arc::clone(copy));
        image.source = Some(source.storage);
        let chains = [chain.handle];
        let indices = [frame.index];
        let waits = [frame.signal];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&chains)
            .image_indices(&indices);
        chain.last_work = Some(work);
        let result = {
            let _queue = device
                .owner
                .queue_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { chain.loader.queue_present(self.present_queue, &info) }
        };
        // OOM does not enqueue or consume the wait semaphore. Keep acquisition
        // until shutdown drains its copy. Out-of-date/surface-lost enqueue waits.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkQueuePresentKHR.html
        if matches!(
            result,
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        ) {
            return Err(VulkanError::vk("vkQueuePresentKHR", result.unwrap_err()));
        }
        self.held = None;
        match result {
            Ok(suboptimal) => {
                self.dirty |= suboptimal;
                Ok(true)
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty = true;
                Ok(false)
            }
            Err(error) => Err(VulkanError::vk("vkQueuePresentKHR", error)),
        }
    }
    fn drain(&mut self, device: &Context) -> Result<(), VulkanError> {
        if let Some(work) = self.current.as_ref().and_then(|c| c.last_work) {
            device.drain(work)?;
        }
        if let Some((_, acquire, _)) = self.held {
            unsafe {
                device
                    .owner
                    .device
                    .wait_for_fences(&[self.acquires[acquire].fence], true, u64::MAX)
            }
            .or_else(|error| {
                if error == vk::Result::ERROR_DEVICE_LOST {
                    Ok(())
                } else {
                    Err(error)
                }
            })
            .map_err(|e| VulkanError::vk("final acquire completion", e))?;
            self.held = None;
        }
        // Drain native submissions on resize/shutdown before releasing their
        // swapchain, copy storage and semaphores. No drain is needed per frame.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkDeviceWaitIdle.html
        {
            let _queue = device
                .owner
                .queue_mutex
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { device.owner.device.device_wait_idle() }
                .or_else(|error| {
                    if error == vk::Result::ERROR_DEVICE_LOST {
                        Ok(())
                    } else {
                        Err(error)
                    }
                })
                .map_err(|e| VulkanError::vk("presentation device drain", e))?;
        }
        self.safe_to_drop = true;
        Ok(())
    }
    pub fn shutdown(&mut self, device: &Context) -> Result<(), VulkanError> {
        self.drain(device)?;
        self.current = None;
        Ok(())
    }
}
impl Drop for Swapchains {
    fn drop(&mut self) {
        if !self.safe_to_drop {
            log::error!("WSI resources destroyed without graphics completion");
            std::process::abort();
        }
        unsafe {
            for acquire in &self.acquires {
                self.owner.device.destroy_semaphore(acquire.semaphore, None);
                self.owner.device.destroy_fence(acquire.fence, None);
            }
        }
    }
}

/// All operations externally synchronized against a swapchain stay on this
/// worker, including acquire/recreate/release. Submit and present synchronize
/// their native queue calls with the scheduler, including a shared VkQueue.
pub(crate) struct Presenter {
    sender: mpsc::SyncSender<Command>,
    receiver: mpsc::Receiver<Event>,
    worker: Option<std::thread::JoinHandle<Result<(), VulkanError>>>,
    busy: bool,
    format: vk::Format,
}
enum Command {
    Present(PresentFrame),
    Shutdown,
}
enum Event {
    Idle,
    Failed(VulkanError),
}
impl Presenter {
    pub fn new(
        device: &VulkanDevice,
        client: PresentationClient,
        vsync: bool,
    ) -> Result<Self, VulkanError> {
        if device.owner.instance.surface.is_none() {
            return Err(VulkanError::InvalidResource(
                "native presentation requires the startup window surface",
            ));
        }
        let format = surface_format(&device.owner)?.format;
        let context = Context {
            progress: Arc::clone(&device.submission.progress),
            owner: Arc::clone(&device.owner),
            queues: crate::device::QueueSelection {
                graphics_compute_family: device.queues().graphics_compute_family,
                present_family: device.queues().present_family,
            },
            timeline: device.submission.timeline,
            vsync,
        };
        let mut chains = Swapchains::new(&context)?;
        let (sender, commands) = mpsc::sync_channel(2);
        let (events, receiver) = mpsc::sync_channel(2);
        let worker = std::thread::Builder::new()
            .name("nixe-vulkan-present".into())
            .spawn(move || {
                let mut pending: Option<PresentFrame> = None;
                let mut last_sequence = None;
                let result = (|| -> Result<(), VulkanError> {
                    loop {
                        let command = if pending.is_some() {
                            match commands.recv_timeout(std::time::Duration::from_millis(1)) {
                                Ok(c) => Some(c),
                                Err(mpsc::RecvTimeoutError::Timeout) => None,
                                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        } else {
                            match commands.recv() {
                                Ok(c) => Some(c),
                                Err(_) => break,
                            }
                        };
                        match command {
                            Some(Command::Shutdown) => break,
                            Some(Command::Present(frame)) => {
                                assert!(pending.is_none(), "one admitted composed frame");
                                chains.resize(frame.storage.extent);
                                pending = Some(frame);
                            }
                            None => {}
                        }
                        if pending.is_some()
                            && chains.update(&context)?
                            && let Some(target) = chains.acquire(&context)?
                        {
                            let frame = pending.take().unwrap();
                            let sequence = frame.sequence;
                            let work = frame.work;
                            let accepted = chains.present(&context, target, frame)?;
                            if accepted {
                                let new_frame = sequence.is_some() && sequence != last_sequence;
                                if new_frame {
                                    nixe_gpu::trace::event(
                                        "host_present",
                                        sequence.unwrap(),
                                        work.tick(),
                                        0,
                                    );
                                    last_sequence = sequence;
                                }
                                client.record_presented(new_frame);
                            }
                            events
                                .send(Event::Idle)
                                .map_err(|_| VulkanError::QueueStopped)?;
                            client.wake_owner();
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    client.fail(error.to_string());
                    let _ = events.send(Event::Failed(error));
                }
                let retired = chains.shutdown(&context);
                if let Err(error) = &retired {
                    log::error!("WSI retirement failed: {error}");
                }
                retired
            })
            .map_err(|e| VulkanError::WindowHandle(e.to_string().into_boxed_str()))?;
        Ok(Self {
            sender,
            receiver,
            worker: Some(worker),
            busy: false,
            format,
        })
    }
    pub fn format(&self) -> vk::Format {
        self.format
    }
    pub fn poll(&mut self) -> Result<(), VulkanError> {
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                Event::Idle => self.busy = false,
                Event::Failed(error) => return Err(error),
            }
        }
        Ok(())
    }
    pub fn ready(&mut self) -> Result<bool, VulkanError> {
        self.poll()?;
        Ok(!self.busy)
    }
    pub fn present(&mut self, frame: PresentFrame) -> Result<(), VulkanError> {
        self.sender
            .send(Command::Present(frame))
            .map_err(|_| VulkanError::QueueStopped)?;
        self.busy = true;
        Ok(())
    }
    pub fn shutdown(&mut self) -> Result<(), VulkanError> {
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        let _ = self.sender.send(Command::Shutdown);
        worker.join().map_err(|_| VulkanError::QueueStopped)?
    }
}
impl Drop for Presenter {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            log::error!("cannot retire Vulkan presentation: {error}");
            std::process::abort();
        }
    }
}
