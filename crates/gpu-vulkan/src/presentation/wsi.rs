//! Presentation release fences are distinct from ordinary GPU timeline uses.
//! https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
use super::*;
use crate::device::DeviceOwner;
use ash::vk;
/// Only read-only timeline queries cross the graphics-owner boundary.
struct Context {
    owner: Arc<DeviceOwner>,
    queues: crate::device::QueueSelection,
    timeline: vk::Semaphore,
    mutable_format: bool,
    acquire_count: usize,
}
impl Context {
    fn completed(&self) -> Result<u64, VulkanError> {
        unsafe { self.owner.device.get_semaphore_counter_value(self.timeline) }
            .map_err(|e| VulkanError::vk("WSI graphics completion", e))
    }
    fn wait(&self, work: WorkTicket) -> Result<(), VulkanError> {
        let semaphores = [self.timeline];
        let values = [work.tick()];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.owner.device.wait_semaphores(&info, u64::MAX) }
            .map_err(|e| VulkanError::vk("WSI graphics drain", e))
    }
}
struct Image {
    handle: vk::Image,
    view: vk::ImageView,
    finished: vk::Semaphore,
    present: vk::Fence,
    presented: bool,
}
struct Chain {
    owner: Arc<DeviceOwner>,
    loader: ash::khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    last_work: Option<WorkTicket>,
}
impl Chain {
    fn releasable(&self, completed: u64) -> Result<bool, VulkanError> {
        if self.last_work.is_some_and(|w| w.tick() > completed) {
            return Ok(false);
        }
        for image in &self.images {
            if image.presented
                && !unsafe { self.owner.device.get_fence_status(image.present) }
                    .map_err(|e| VulkanError::vk("presentation release status", e))?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
impl Drop for Chain {
    fn drop(&mut self) {
        unsafe {
            for image in &self.images {
                self.owner.device.destroy_image_view(image.view, None);
                self.owner.device.destroy_semaphore(image.finished, None);
                self.owner.device.destroy_fence(image.present, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}
struct Acquire {
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    work: Option<WorkTicket>,
    issued: bool,
}
#[derive(Clone, Copy)]
pub(crate) struct Acquired {
    pub format: vk::Format,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub wait: vk::Semaphore,
    pub signal: vk::Semaphore,
    index: u32,
    acquire: usize,
}
struct Swapchains {
    owner: Arc<DeviceOwner>,
    current: Option<Chain>,
    retired: Vec<Chain>,
    acquires: Vec<Acquire>,
    held: Option<(u32, usize, bool)>,
    dirty: bool,
    requested: [u32; 2],
    present_queue: vk::Queue,
    safe_to_drop: bool,
}
impl Swapchains {
    pub fn new(device: &Context) -> Result<Self, VulkanError> {
        let family = device.queues.present_family.unwrap();
        let present_queue = unsafe {
            device
                .owner
                .device
                .get_device_queue(family, device.queues.present_queue_index)
        };
        let mut result = Self {
            owner: Arc::clone(&device.owner),
            current: None,
            retired: Vec::new(),
            acquires: Vec::new(),
            held: None,
            dirty: true,
            requested: [0, 0],
            present_queue,
            safe_to_drop: true,
        };
        for _ in 0..device.acquire_count {
            let semaphore = unsafe {
                result
                    .owner
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|e| VulkanError::vk("acquire semaphore", e))?;
            result.acquires.push(Acquire {
                semaphore,
                fence: vk::Fence::null(),
                work: None,
                issued: false,
            });
            let fence = unsafe {
                result
                    .owner
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .map_err(|e| VulkanError::vk("acquire fence", e))?;
            result.acquires.last_mut().unwrap().fence = fence;
        }
        Ok(result)
    }
    pub fn resize(&mut self, size: [u32; 2]) {
        if size != self.requested {
            self.requested = size;
            self.dirty = true;
        }
    }
    pub fn update(&mut self, device: &Context) -> Result<bool, VulkanError> {
        let completed = device.completed()?;
        let mut at = 0;
        while at < self.retired.len() {
            if self.retired[at].releasable(completed)? {
                self.retired.remove(at);
            } else {
                at += 1;
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
        // Bounds rapid resize generations without blocking guest progress.
        if self.retired.len() == 8 {
            return Ok(false);
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
        let formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(device.owner.physical, surface.handle)
        }
        .map_err(|e| VulkanError::vk("surface formats", e))?;
        let format = formats
            .iter()
            .find(|f| {
                f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR && matches!(
                    f.format,
                    vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
                )
            })
            .or_else(|| {
                formats.iter().find(|f| {
                    device.mutable_format && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR && matches!(
                        f.format,
                        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                    )
                })
            })
            .copied()
            .ok_or(VulkanError::InvalidResource(
                "surface requires nonlinear RGBA8/BGRA8 UNORM, or an sRGB surface with VK_KHR_swapchain_mutable_format",
            ))?;
        if !capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        {
            return Err(VulkanError::InvalidResource(
                "surface cannot render color attachments",
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
        let view_format = match format.format {
            vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
            format => format,
        };
        let view_formats = [format.format, view_format];
        let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(
                self.current
                    .as_ref()
                    .map_or(vk::SwapchainKHR::null(), |c| c.handle),
            );
        // An UNORM attachment view preserves egui's nonlinear blending even
        // when WSI advertises only sRGB images; no extra pass or gamma shader.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_swapchain_mutable_format.html
        if view_format != format.format {
            info = info
                .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
                .push_next(&mut format_list);
        }
        // Separate present families share this WSI image explicitly; ordinary
        // guest resources remain exclusive to the graphics/compute owner.
        if graphics != present {
            info = info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        let loader =
            ash::khr::swapchain::Device::new(&device.owner.instance.instance, &device.owner.device);
        let handle = unsafe { loader.create_swapchain(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateSwapchainKHR", e))?;
        let mut chain = Chain {
            owner: Arc::clone(&device.owner),
            loader,
            handle,
            images: Vec::new(),
            format: view_format,
            extent,
            last_work: None,
        };
        for image in unsafe { chain.loader.get_swapchain_images(handle) }
            .map_err(|e| VulkanError::vk("swapchain images", e))?
        {
            chain.images.push(Image {
                handle: image,
                view: vk::ImageView::null(),
                finished: vk::Semaphore::null(),
                present: vk::Fence::null(),
                presented: false,
            });
            let entry = chain.images.last_mut().unwrap();
            entry.view = unsafe {
                device.owner.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(view_format)
                        .subresource_range(super::render::color_range()),
                    None,
                )
            }
            .map_err(|e| VulkanError::vk("swapchain view", e))?;
            entry.finished = unsafe {
                device
                    .owner
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(|e| VulkanError::vk("present semaphore", e))?;
            entry.present = unsafe {
                device.owner.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }
            .map_err(|e| VulkanError::vk("present release fence", e))?;
        }
        if let Some(old) = self.current.replace(chain) {
            self.retired.push(old);
        }
        self.dirty = false;
        Ok(true)
    }
    pub fn acquire(&mut self, device: &Context) -> Result<Option<Acquired>, VulkanError> {
        let completed = device.completed()?;
        let Some(chain) = &self.current else {
            return Ok(None);
        };
        let (index, acquire, suboptimal) = if let Some(held) = self.held {
            held
        } else {
            let mut available = None;
            for (index, acquire) in self.acquires.iter().enumerate() {
                if acquire.work.is_some_and(|w| w.tick() > completed) {
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
            match unsafe {
                chain
                    .loader
                    .acquire_next_image(chain.handle, 0, a.semaphore, a.fence)
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
        if image.presented
            && !unsafe { device.owner.device.get_fence_status(image.present) }
                .map_err(|e| VulkanError::vk("present release status", e))?
        {
            return Ok(None);
        }
        self.dirty |= suboptimal;
        Ok(Some(Acquired {
            format: chain.format,
            image: image.handle,
            view: image.view,
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
        work: WorkTicket,
    ) -> Result<bool, VulkanError> {
        let chain = self.current.as_mut().unwrap();
        let image = &mut chain.images[frame.index as usize];
        unsafe { device.owner.device.reset_fences(&[image.present]) }
            .map_err(|e| VulkanError::vk("reset present release fence", e))?;
        let fences = [image.present];
        let mut release = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
        let chains = [chain.handle];
        let indices = [frame.index];
        let waits = [frame.signal];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&chains)
            .image_indices(&indices)
            .push_next(&mut release);
        let result = unsafe { chain.loader.queue_present(self.present_queue, &info) };
        // OOM does not enqueue or consume the wait semaphore. Keep acquisition
        // for explicit release after graphics completion, not a never-signaled
        // presentation fence. Out-of-date/surface-lost do enqueue their waits.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkQueuePresentKHR.html
        if matches!(
            result,
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        ) {
            image.presented = false;
            return Err(VulkanError::vk("vkQueuePresentKHR", result.unwrap_err()));
        }
        image.presented = true;
        chain.last_work = Some(work);
        self.acquires[frame.acquire].work = Some(work);
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
    pub fn shutdown(&mut self, device: &Context) -> Result<(), VulkanError> {
        if let Some(work) = self.current.as_ref().and_then(|c| c.last_work) {
            device.wait(work)?;
        }
        if let Some((_, acquire, _)) = self.held.take() {
            unsafe {
                device
                    .owner
                    .device
                    .wait_for_fences(&[self.acquires[acquire].fence], true, u64::MAX)
            }
            .map_err(|e| VulkanError::vk("final acquire completion", e))?;
            // Destruction needs completed acquire/graphics/presentation uses,
            // not a separate release of an unused acquired image. No continuing
            // swapchain exists here. Avoid an unnecessary surface-dependent call.
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkDestroySwapchainKHR.html
        }
        for chain in self.current.iter().chain(&self.retired) {
            if let Some(work) = chain.last_work {
                device.wait(work)?;
            }
            let fences: Vec<_> = chain
                .images
                .iter()
                .filter(|i| i.presented)
                .map(|i| i.present)
                .collect();
            if !fences.is_empty() {
                unsafe { device.owner.device.wait_for_fences(&fences, true, u64::MAX) }
                    .map_err(|e| VulkanError::vk("final presentation release", e))?;
            }
        }
        self.safe_to_drop = true;
        self.current = None;
        self.retired.clear();
        Ok(())
    }
}
impl Drop for Swapchains {
    fn drop(&mut self) {
        if !self.safe_to_drop {
            log::error!("WSI resources destroyed without presentation release proof");
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
/// worker, including acquire/recreate/release. Its VkQueue is distinct from
/// the graphics queue; a blocked driver present cannot stop guest execution.
pub(crate) struct Presenter {
    sender: mpsc::SyncSender<Command>,
    receiver: mpsc::Receiver<Event>,
    worker: Option<std::thread::JoinHandle<Result<(), VulkanError>>>,
    busy: bool,
    acquired: Option<Acquired>,
}
enum Command {
    Acquire([u32; 2]),
    Submit(Acquired, WorkTicket, Option<u64>),
    Graphics(WorkTicket),
    Shutdown,
}
enum Event {
    Acquired(Acquired),
    Idle,
    Failed(VulkanError),
}
impl Presenter {
    pub fn new(device: &VulkanDevice, client: PresentationClient) -> Result<Self, VulkanError> {
        if device.owner.instance.surface.is_none() {
            return Err(VulkanError::InvalidResource(
                "native presentation requires the startup window surface",
            ));
        }
        if device.profile().swapchain_maintenance == SwapchainMaintenance::Disabled {
            let instance = &device.owner.instance;
            let error = if !instance.surface_maintenance_khr && !instance.surface_maintenance_ext {
                VulkanError::SurfaceMaintenanceUnavailable
            } else {
                VulkanError::SwapchainMaintenanceUnavailable
            };
            return Err(error);
        }
        let context = Context {
            owner: Arc::clone(&device.owner),
            queues: crate::device::QueueSelection {
                graphics_compute_family: device.queues().graphics_compute_family,
                present_family: device.queues().present_family,
                present_queue_index: device.queues().present_queue_index,
            },
            timeline: device.submission.timeline,
            mutable_format: device.profile().swapchain_mutable_format,
            // Pools are allocated lazily; use configured capacity at startup.
            acquire_count: device.submission.configuration.command_slots,
        };
        let mut chains = Swapchains::new(&context)?;
        let (sender, commands) = mpsc::sync_channel(2);
        let (events, receiver) = mpsc::sync_channel(2);
        let worker = std::thread::Builder::new()
            .name("nixe-vulkan-present".into())
            .spawn(move || {
                let mut requested = false;
                let mut handed_off = false;
                let mut last_sequence = None;
                let result = (|| -> Result<(), VulkanError> {
                    loop {
                        let command = if requested && !handed_off {
                            match commands.recv_timeout(std::time::Duration::from_millis(1)) {
                                Ok(command) => Some(command),
                                Err(mpsc::RecvTimeoutError::Timeout) => None,
                                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        } else {
                            match commands.recv() {
                                Ok(command) => Some(command),
                                Err(_) => break,
                            }
                        };
                        match command {
                            Some(Command::Shutdown) => break,
                            Some(Command::Acquire(size)) => {
                                chains.resize(size);
                                requested = true;
                            }
                            Some(Command::Graphics(work)) => {
                                chains.current.as_mut().unwrap().last_work = Some(work);
                            }
                            Some(Command::Submit(frame, work, sequence)) => {
                                let accepted = chains.present(&context, frame, work)?;
                                if accepted
                                    && sequence != last_sequence
                                    && let Some(fresh_sequence) = sequence
                                {
                                    nixe_gpu::trace::event(
                                        "host_present",
                                        fresh_sequence,
                                        work.tick(),
                                        0,
                                    );
                                    last_sequence = sequence;
                                    client.record_presented();
                                }
                                requested = false;
                                handed_off = false;
                                events
                                    .send(Event::Idle)
                                    .map_err(|_| VulkanError::QueueStopped)?;
                            }
                            None => {}
                        }
                        if requested
                            && !handed_off
                            && chains.update(&context)?
                            && let Some(frame) = chains.acquire(&context)?
                        {
                            events
                                .send(Event::Acquired(frame))
                                .map_err(|_| VulkanError::QueueStopped)?;
                            handed_off = true;
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
            acquired: None,
        })
    }
    pub fn poll(&mut self) -> Result<(), VulkanError> {
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                Event::Acquired(frame) => self.acquired = Some(frame),
                Event::Idle => self.busy = false,
                Event::Failed(error) => return Err(error),
            }
        }
        Ok(())
    }
    pub fn acquire(&mut self, size: [u32; 2]) -> Result<Option<Acquired>, VulkanError> {
        self.poll()?;
        if let Some(frame) = self.acquired {
            return Ok(Some(frame));
        }
        if !self.busy && !size.contains(&0) {
            self.sender
                .send(Command::Acquire(size))
                .map_err(|_| VulkanError::QueueStopped)?;
            self.busy = true;
        }
        Ok(None)
    }
    pub fn graphics(&self, work: WorkTicket) -> Result<(), VulkanError> {
        self.sender
            .send(Command::Graphics(work))
            .map_err(|_| VulkanError::QueueStopped)
    }
    pub fn present(
        &mut self,
        frame: Acquired,
        work: WorkTicket,
        sequence: Option<u64>,
    ) -> Result<(), VulkanError> {
        self.sender
            .send(Command::Submit(frame, work, sequence))
            .map_err(|_| VulkanError::QueueStopped)?;
        self.acquired = None;
        Ok(())
    }
    pub fn shutdown(&mut self) -> Result<(), VulkanError> {
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        let _ = self.sender.send(Command::Shutdown);
        worker
            .join()
            .map_err(|_| VulkanError::InvalidResource("presentation worker panicked"))?
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
