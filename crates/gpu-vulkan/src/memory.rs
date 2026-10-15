//! Owned allocations; host mapping obeys non-coherent atom isolation.
use crate::{VulkanDevice, VulkanError, device::DeviceOwner};
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AllocationMode {
    #[default]
    Managed,
    Dedicated,
}

pub struct AllocatedBuffer {
    pub(crate) owner: Arc<DeviceOwner>,
    pub(crate) handle: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
    pub(crate) usage: vk::BufferUsageFlags,
    pub(crate) charge: Option<AllocationCharge>,
}
pub struct AllocatedImage {
    pub(crate) owner: Arc<DeviceOwner>,
    pub(crate) handle: vk::Image,
    allocation: Option<Allocation>,
    pub(crate) charge: Option<AllocationCharge>,
}

pub(crate) fn aligned_host_requirements(
    mut requirements: vk::MemoryRequirements,
    atom: u64,
) -> Result<vk::MemoryRequirements, VulkanError> {
    // Adjacent mapped suballocations must not share a non-coherent atom. Pad
    // both alignment and size so flush/invalidate stays in this allocation.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkMappedMemoryRange.html
    requirements.alignment = requirements.alignment.max(atom);
    requirements.size =
        requirements
            .size
            .checked_add(atom - 1)
            .ok_or(VulkanError::InvalidResource(
                "host allocation size overflow",
            ))?
            & !(atom - 1);
    Ok(requirements)
}
// The final partial atom is legal only at the allocation boundary.
// https://docs.vulkan.org/refpages/latest/refpages/source/VkMappedMemoryRange.html
pub(crate) fn atom_range(
    offset: u64,
    length: u64,
    allocation_size: u64,
    atom: u64,
) -> Result<(u64, u64), VulkanError> {
    let end = offset
        .checked_add(length)
        .filter(|end| *end <= allocation_size)
        .ok_or(VulkanError::InvalidResource(
            "mapped atom range exceeds allocation",
        ))?;
    let start = offset & !(atom - 1);
    let end = end
        .checked_add(atom - 1)
        .ok_or(VulkanError::InvalidResource("mapped atom range overflow"))?
        & !(atom - 1);
    Ok((start, end.min(allocation_size) - start))
}
impl VulkanDevice {
    pub fn allocate_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        mode: AllocationMode,
        label: &str,
    ) -> Result<AllocatedBuffer, VulkanError> {
        if size == 0 || usage.is_empty() {
            return Err(VulkanError::InvalidResource(
                "buffer size and usage must be nonzero",
            ));
        }
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && !self.profile().buffer_device_address
        {
            return Err(VulkanError::InvalidResource(
                "buffer device address feature is not enabled",
            ));
        }
        allocate_buffer(&self.owner, size, usage, location, mode, label)
    }

    pub fn allocate_image_2d(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mode: AllocationMode,
        label: &str,
    ) -> Result<AllocatedImage, VulkanError> {
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage,
            flags: vk::ImageCreateFlags::empty(),
        };
        allocate_image_charged(&self.owner, description, mode, label, |_| Ok(None))
    }
}
pub(crate) fn allocate_buffer(
    owner: &Arc<DeviceOwner>,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    mode: AllocationMode,
    label: &str,
) -> Result<AllocatedBuffer, VulkanError> {
    allocate_buffer_charged(owner, size, usage, location, mode, label, |_| Ok(None))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn allocate_buffer_charged(
    owner: &Arc<DeviceOwner>,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    mode: AllocationMode,
    label: &str,
    charge: impl FnOnce(u64) -> Result<Option<AllocationCharge>, VulkanError>,
) -> Result<AllocatedBuffer, VulkanError> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: this device is retained; create info has no foreign pointers.
    let handle = unsafe { owner.device.create_buffer(&info, None) }
        .map_err(|e| VulkanError::vk("vkCreateBuffer", e))?;
    let mut buffer = AllocatedBuffer {
        owner: Arc::clone(owner),
        handle,
        allocation: None,
        size,
        usage,
        charge: None,
    };
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    let info = vk::BufferMemoryRequirementsInfo2::default().buffer(handle);
    unsafe {
        owner
            .device
            .get_buffer_memory_requirements2(&info, &mut requirements);
    }
    let requirements = requirements.memory_requirements;
    let dedicated =
        mode == AllocationMode::Dedicated || dedicated.requires_dedicated_allocation != 0;
    // A dedicated allocation must use the exact driver-reported size. Its
    // final partial atom is isolated by the allocation boundary itself.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryDedicatedAllocateInfo.html
    let requirements = if !dedicated
        && matches!(
            location,
            MemoryLocation::CpuToGpu | MemoryLocation::GpuToCpu
        ) {
        aligned_host_requirements(requirements, owner.non_coherent_atom_size)?
    } else {
        requirements
    };
    let scheme = if dedicated {
        AllocationScheme::DedicatedBuffer(handle)
    } else {
        AllocationScheme::GpuAllocatorManaged
    };
    buffer.charge = charge(requirements.size)?;
    let allocation = owner
        .allocator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
        .expect("device allocator initialized")
        .allocate(&AllocationCreateDesc {
            name: label,
            requirements,
            location,
            linear: true,
            allocation_scheme: scheme,
        })?;
    buffer.allocation = Some(allocation);
    let allocation = buffer.allocation.as_ref().expect("allocated buffer");
    // SAFETY: this allocation meets the buffer's original requirements, and
    // is retained until the buffer is destroyed. Account for suballocation offset.
    unsafe {
        owner
            .device
            .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
    }
    .map_err(|e| VulkanError::vk("vkBindBufferMemory", e))?;
    owner.name(handle, label)?;
    Ok(buffer)
}
impl AllocatedBuffer {
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn is_dedicated(&self) -> bool {
        self.allocation
            .as_ref()
            .is_some_and(Allocation::is_dedicated)
    }
    pub(crate) fn host_alignment(&self) -> u64 {
        let allocation = self.allocation.as_ref().expect("bound buffer");
        if allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            4
        } else {
            self.owner.non_coherent_atom_size.max(4)
        }
    }
    fn mapped_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<vk::MappedMemoryRange<'_>, VulkanError> {
        let allocation = self.allocation.as_ref().expect("bound buffer");
        let (start, size) = atom_range(
            offset,
            length,
            allocation.size(),
            self.owner.non_coherent_atom_size,
        )?;
        Ok(vk::MappedMemoryRange::default()
            .memory(unsafe { allocation.memory() })
            .offset(allocation.offset() + start)
            .size(size))
    }
    pub(crate) unsafe fn write_range(&self, offset: u64, bytes: &[u8]) -> Result<(), VulkanError> {
        self.check_range(offset, bytes.len() as u64)?;
        let allocation = self.allocation.as_ref().expect("bound buffer");
        let ptr = allocation.mapped_ptr().ok_or(VulkanError::InvalidResource(
            "upload memory is not host visible",
        ))?;
        let offset_host = usize::try_from(offset).map_err(|_| {
            VulkanError::InvalidResource("mapped offset exceeds host address space")
        })?;
        // SAFETY: caller excludes other host accesses and GPU use of these atoms; mapped_ptr includes
        // the suballocation offset and the requested range is in bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                ptr.as_ptr().cast::<u8>().add(offset_host),
                bytes.len(),
            );
        }
        if !allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            let range = self.mapped_range(offset, bytes.len() as u64)?;
            unsafe { self.owner.device.flush_mapped_memory_ranges(&[range]) }
                .map_err(|e| VulkanError::vk("vkFlushMappedMemoryRanges", e))?;
        }
        Ok(())
    }
    pub(crate) unsafe fn read_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, VulkanError> {
        self.check_range(offset, length)?;
        let length_host = usize::try_from(length)
            .map_err(|_| VulkanError::InvalidResource("readback cannot fit host address space"))?;
        let offset_host = usize::try_from(offset).map_err(|_| {
            VulkanError::InvalidResource("mapped offset exceeds host address space")
        })?;
        let allocation = self.allocation.as_ref().expect("bound buffer");
        let ptr = allocation.mapped_ptr().ok_or(VulkanError::InvalidResource(
            "readback memory is not host visible",
        ))?;
        // SAFETY: caller completed GPU writes and a HOST_READ dependency. Only
        // this atom-isolated slice is invalidated, never unrelated in-flight data.
        if !allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            let range = self.mapped_range(offset, length)?;
            unsafe { self.owner.device.invalidate_mapped_memory_ranges(&[range]) }
                .map_err(|e| VulkanError::vk("vkInvalidateMappedMemoryRanges", e))?;
        }
        Ok(unsafe {
            std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>().add(offset_host), length_host)
        }
        .to_vec())
    }
    fn check_range(&self, offset: u64, length: u64) -> Result<(), VulkanError> {
        if length == 0 || offset.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(VulkanError::InvalidResource(
                "mapped range is empty or exceeds buffer",
            ));
        }
        Ok(())
    }
}
impl AllocatedImage {
    pub fn is_dedicated(&self) -> bool {
        self.allocation
            .as_ref()
            .is_some_and(Allocation::is_dedicated)
    }
}
impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        // SAFETY: only internal recording sees the handle, retaining it until
        // completion. Destroy the bound object before returning its allocation.
        unsafe {
            self.owner.device.destroy_buffer(self.handle, None);
        }
        if let Some(allocation) = self.allocation.take()
            && let Err(e) = self
                .owner
                .allocator
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut()
                .expect("live allocator")
                .free(allocation)
        {
            log::error!("Vulkan buffer free failed: {e}");
        }
    }
}
impl Drop for AllocatedImage {
    fn drop(&mut self) {
        // SAFETY: the same retained-completion/ownership rule as AllocatedBuffer.
        unsafe {
            self.owner.device.destroy_image(self.handle, None);
        }
        if let Some(allocation) = self.allocation.take()
            && let Err(e) = self
                .owner
                .allocator
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut()
                .expect("live allocator")
                .free(allocation)
        {
            log::error!("Vulkan image free failed: {e}");
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flush_invalidate_ranges_cover_only_the_consumed_atoms() {
        assert_eq!(atom_range(256, 257, 1024, 256).unwrap(), (256, 512));
        assert_eq!(atom_range(256, 1, 257, 256).unwrap(), (256, 1));
        assert_eq!(atom_range(512, 256, 1024, 256).unwrap(), (512, 256));
        assert!(atom_range(1000, 25, 1024, 256).is_err());
        assert!(atom_range(u64::MAX, 1, u64::MAX, 256).is_err());
    }
    #[test]
    fn non_coherent_ranges_isolate_neighboring_suballocations_and_reject_overflow() {
        let requirements = vk::MemoryRequirements {
            size: 257,
            alignment: 16,
            memory_type_bits: 1,
        };
        let padded = aligned_host_requirements(requirements, 256).unwrap();
        assert_eq!(
            (padded.size, padded.alignment, padded.memory_type_bits),
            (512, 256, 1)
        );
        assert!(
            aligned_host_requirements(
                vk::MemoryRequirements {
                    size: u64::MAX,
                    ..requirements
                },
                256
            )
            .is_err()
        );
    }
}

/// The charge follows the native allocation through timeline retirement.
/// Metadata eviction alone cannot release this budget.
#[derive(Clone)]
pub(crate) struct AllocationBudget {
    used: Arc<std::sync::atomic::AtomicU64>,
    pub limit: u64,
}
pub(crate) struct AllocationCharge {
    budget: AllocationBudget,
    bytes: u64,
}
impl AllocationBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            limit,
        }
    }
    pub fn used(&self) -> u64 {
        self.used.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn reserve(&self, bytes: u64) -> Option<AllocationCharge> {
        self.used
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |used| used.checked_add(bytes).filter(|total| *total <= self.limit),
            )
            .ok()?;
        Some(AllocationCharge {
            budget: self.clone(),
            bytes,
        })
    }
}
impl Drop for AllocationCharge {
    fn drop(&mut self) {
        self.budget
            .used
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}
/// Native description: guest tiling and encoding belong to the selected frontend.
/// Extent/mips/layers/formats are validated against the selected host when consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeImageDescription {
    pub image_type: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}
pub(crate) fn allocate_image_charged(
    owner: &Arc<DeviceOwner>,
    description: NativeImageDescription,
    mode: AllocationMode,
    label: &str,
    charge: impl FnOnce(u64) -> Result<Option<AllocationCharge>, VulkanError>,
) -> Result<AllocatedImage, VulkanError> {
    let d = description;
    if d.extent.width == 0
        || d.extent.height == 0
        || d.extent.depth == 0
        || d.mip_levels == 0
        || d.array_layers == 0
        || d.usage.is_empty()
    {
        return Err(VulkanError::InvalidResource("empty native image"));
    }
    let max_dimension = d.extent.width.max(d.extent.height).max(d.extent.depth);
    if d.samples.as_raw().count_ones() != 1
        || d.mip_levels > u32::BITS - max_dimension.leading_zeros()
        || (d.image_type == vk::ImageType::TYPE_1D && (d.extent.height != 1 || d.extent.depth != 1))
        || (d.image_type == vk::ImageType::TYPE_2D && d.extent.depth != 1)
        || (d.image_type == vk::ImageType::TYPE_3D && d.array_layers != 1)
    {
        return Err(VulkanError::InvalidResource(
            "invalid native image dimensions, mips or samples",
        ));
    }
    let supported = unsafe {
        owner
            .instance
            .instance
            .get_physical_device_image_format_properties(
                owner.physical,
                d.format,
                d.image_type,
                vk::ImageTiling::OPTIMAL,
                d.usage,
                d.flags,
            )
    }
    .map_err(|e| VulkanError::vk("vkGetPhysicalDeviceImageFormatProperties", e))?;
    if d.extent.width > supported.max_extent.width
        || d.extent.height > supported.max_extent.height
        || d.extent.depth > supported.max_extent.depth
        || d.mip_levels > supported.max_mip_levels
        || d.array_layers > supported.max_array_layers
        || !supported.sample_counts.contains(d.samples)
    {
        return Err(VulkanError::InvalidResource(
            "native image exceeds host format limits",
        ));
    }
    let info = vk::ImageCreateInfo::default()
        .image_type(d.image_type)
        .format(d.format)
        .extent(d.extent)
        .mip_levels(d.mip_levels)
        .array_layers(d.array_layers)
        .samples(d.samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(d.usage)
        .flags(d.flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let handle = unsafe { owner.device.create_image(&info, None) }
        .map_err(|e| VulkanError::vk("vkCreateImage", e))?;
    let mut image = AllocatedImage {
        owner: Arc::clone(owner),
        handle,
        allocation: None,
        charge: None,
    };
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    unsafe {
        owner.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(handle),
            &mut requirements,
        );
    }
    image.charge = charge(requirements.memory_requirements.size)?;
    image.allocation = Some(
        owner
            .allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
            .expect("allocator")
            .allocate(&AllocationCreateDesc {
                name: label,
                requirements: requirements.memory_requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: if mode == AllocationMode::Dedicated
                    || dedicated.requires_dedicated_allocation != 0
                {
                    AllocationScheme::DedicatedImage(handle)
                } else {
                    AllocationScheme::GpuAllocatorManaged
                },
            })?,
    );
    let allocation = image.allocation.as_ref().expect("allocated image");
    unsafe {
        owner
            .device
            .bind_image_memory(handle, allocation.memory(), allocation.offset())
    }
    .map_err(|e| VulkanError::vk("vkBindImageMemory", e))?;
    owner.name(handle, label)?;
    Ok(image)
}

/// Independently leased view; its image remains alive until the last view/batch.
pub struct AllocatedImageView {
    pub(crate) image: Arc<AllocatedImage>,
    pub(crate) handle: vk::ImageView,
}
impl AllocatedImageView {
    pub(crate) fn from_handle(image: Arc<AllocatedImage>, handle: vk::ImageView) -> Self {
        Self { image, handle }
    }
}
impl Drop for AllocatedImageView {
    fn drop(&mut self) {
        unsafe {
            self.image
                .owner
                .device
                .destroy_image_view(self.handle, None);
        }
    }
}
