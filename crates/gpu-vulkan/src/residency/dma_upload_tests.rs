use super::*;
use nixe_gpu::native::{
    ImageAddressSpace, ImageInfo, ImagePacking, MappedGpuRange, MemoryCopy, MemoryCopySurface,
};
use nixe_gpu::physical::{ByteRect, ImageFootprint, ImageStorage};
use nixe_gpu::{
    BlockLinearLayout, ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    ImageMemoryLayout, SampleCount, Swizzle,
};
use nixe_memory::{
    AllocationRange, Machine, MemoryError, WriteSource, device_memory::DeviceMemory,
};

mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

struct Space(u64);
const SPACE: Space = Space(1);
impl ImageAddressSpace for Space {
    fn image_space_id(&self) -> u64 {
        self.0
    }
    fn image_device_address(&self, a: u64) -> Result<Option<u64>, MemoryError> {
        Ok(Some(a))
    }
    fn image_mapped_ranges(
        &self,
        a: u64,
        n: u64,
    ) -> Result<impl Iterator<Item = MappedGpuRange>, MemoryError> {
        Ok(std::iter::once(MappedGpuRange {
            gpu: a,
            device: a,
            size: n,
        }))
    }
}
fn info(address: u64, format: ImageFormat, w: u32, h: u32, layout: ImageMemoryLayout) -> ImageInfo {
    ImageInfo {
        address,
        description: ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(w, h, 1).unwrap(),
            format,
            if format.is_depth_stencil() {
                ImageKind::DepthStencil
            } else {
                ImageKind::Color
            },
            1,
            1,
            SampleCount::One,
        )
        .unwrap(),
        swizzle: Swizzle::IDENTITY,
        layout,
        mip_layouts: Arc::from([(0, layout)]),
        layer_stride: 4096,
        maybe_unaligned_layer_stride: 4096,
        size: 4096,
        packing: ImagePacking::Native,
        storage: false,
        sample_grid: [1, 1],
    }
}

#[test]
fn explicit_render_targets_publish_each_gpu_version_on_demand() -> Result<(), VulkanError> {
    let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(4096)?;
    let memory = DeviceMemory::new(machine, 32);
    let address = 0x10000;
    let _mapping = memory.map(
        address,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 4096)?]),
        true,
    )?;
    let mut cache = VulkanResidency::new(
        memory.clone(),
        device,
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let mut r = cache.access();
    let render_target = r.insert_guest_image(
        &info(
            address,
            ImageFormat::R32Uint,
            1,
            1,
            ImageMemoryLayout::PitchLinear {
                row_pitch: 128,
                layer_stride: 4096,
            },
        ),
        &Space(crate::native::NVN_IMAGE_SPACE_ID),
    )?;
    for value in [0x1020_3040u32, 0x5060_7080] {
        r.clear_color_image(
            render_target,
            0,
            vk::ClearColorValue {
                uint32: [value, 0, 0, 0],
            },
        )?;
        assert!(r.texture_cache.flush_area::<false>(address, 4).is_none());
        let area = r.texture_cache.flush_area::<true>(address, 4).unwrap();
        assert!(!area.preemptive);
        r.download_texture_memory(area.start, area.end - area.start)?;
        let mut word = [0; 4];
        memory.reader().read_block_unsafe(address, &mut word)?;
        assert_eq!(u32::from_le_bytes(word), value);
    }
    drop(r);
    validation.check()?;
    Ok(())
}

#[test]
fn dma_preserves_partial_color_alias_and_fence_order() -> Result<(), VulkanError> {
    check_alias(ImageFormat::R32Uint)
}
#[test]
fn dma_preserves_partial_depth_alias_and_fence_order() -> Result<(), VulkanError> {
    check_alias(ImageFormat::Depth32Float)
}
fn check_alias(format: ImageFormat) -> Result<(), VulkanError> {
    let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(8192)?;
    let memory = DeviceMemory::new(machine, 32);
    let (src, dst, alias) = (0x10000, 0x20000, 0x40000);
    let pixels = AllocationRange::new(allocation.pin(), 4096, 4096)?;
    let _src = memory.map(
        src,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 4096)?]),
        true,
    )?;
    let _dst = memory.map(dst, Arc::from([pixels.clone()]), true)?;
    let _alias = memory.map(alias, Arc::from([pixels]), true)?;
    let mut cache = VulkanResidency::new(
        memory.clone(),
        device,
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let mut r = cache.access();
    let tiled = ImageMemoryLayout::BlockLinearAmpere(BlockLinearLayout {
        block_width_log2: 0,
        block_height_log2: 0,
        block_depth_log2: 0,
        layer_stride: 4096,
    });
    let target = r.insert_guest_image(&info(dst, ImageFormat::R8Unorm, 64, 6, tiled), &SPACE)?;
    let other = r.insert_guest_image(
        &info(
            alias,
            format,
            64,
            8,
            ImageMemoryLayout::PitchLinear {
                row_pitch: 256,
                layer_stride: 4096,
            },
        ),
        &SPACE,
    )?;
    let clear = |r: &mut Residency<'_>, byte: u8| {
        let bits = u32::from_le_bytes([byte; 4]);
        if format.is_depth_stencil() {
            r.clear_depth_stencil(other, 0, f32::from_bits(bits), 0)
        } else {
            r.clear_color_image(
                other,
                0,
                vk::ClearColorValue {
                    uint32: [bits, 0, 0, 0],
                },
            )
        }
    };
    clear(&mut r, 0x3c)?;
    r.texture_cache
        .pending_downloads
        .push(texture_download::PendingDownload::Image(other));
    r.commit_texture_downloads()?;
    let old = r.texture_visibility_work().unwrap();
    let source: Vec<u8> = (0..384).map(|i| (i * 23 + 7) as u8).collect();
    memory.write_block_unsafe(src, WriteSource::Bytes(&source))?;
    let layout = ImageFootprint {
        depth_slice: 0,
        base: 0,
        columns: 64,
        rows: 6,
        packed_bytes: 1,
        aspect_offset: 0,
        aspect_bytes: 1,
        storage: ImageStorage::Pitch { row_pitch: 64 },
        native_base: 0,
        native_element_bytes: 1,
        native_row_pitch: 64,
        rect: ByteRect {
            x: 0,
            y: 0,
            width: 64,
            height: 6,
        },
    };
    let dst_layout = ImageFootprint {
        storage: ImageStorage::BlockLinearAmpere {
            block_height_log2: 0,
            block_depth_log2: 0,
        },
        ..layout
    };
    let copy = MemoryCopy {
        source: MemoryCopySurface {
            address: src,
            bytes: 384,
            layout,
        },
        destination: MemoryCopySurface {
            address: dst,
            bytes: 512,
            layout: dst_layout,
        },
    };
    let expected_dma = Footprint::new(
        memory.read_mapping(dst, 512)?,
        nixe_gpu::physical::WriteDomain::Image(dst_layout),
    )?
    .repack_bytes(
        Footprint::new(
            memory.read_mapping(src, 384)?,
            nixe_gpu::physical::WriteDomain::Image(layout),
        )?
        .as_ref(),
        0,
        &source,
    )?;
    let mut expected = [0x3c; 2048];
    let mut at = 0;
    for range in &expected_dma.ranges {
        let len = (range.end - range.start) as usize;
        expected[range.start as usize..range.end as usize]
            .copy_from_slice(&expected_dma.bytes[at..at + len]);
        at += len;
    }
    assert!(r.accelerate_dma(&copy, &SPACE)?);
    assert!(r.texture_cache.get(other).dma.pending());
    assert_eq!(r.statistics().native_readback_bytes, 0);
    let area = r
        .texture_cache
        .flush_area::<false>(alias, 4)
        .expect("DMA bytes are already CPU-visible");
    assert_eq!(
        (area.start, area.end, area.preemptive),
        (alias, alias + 4, true)
    );
    r.download_texture_memory(alias, 4)?;
    assert!(r.texture_cache.get(other).dma.pending());
    assert_eq!(r.statistics().native_readback_bytes, 0);
    r.flush()?;
    r.runtime
        .device
        .wait_for_work(old, std::time::Duration::MAX)?;
    r.texture_cache.pop_async_flush(&mut r.buffer_cache)?;
    let mut result = [0u8; 2048];
    memory.reader().read_block_unsafe(alias, &mut result)?;
    assert_eq!(result, expected, "earlier fence overwrote later DMA");
    // Read the primary native image independently of the write-through RAM copy.
    r.texture_cache
        .pending_downloads
        .push(texture_download::PendingDownload::Image(target));
    r.commit_texture_downloads()?;
    let target_work = r.texture_visibility_work().unwrap();
    r.flush()?;
    r.runtime
        .device
        .wait_for_work(target_work, std::time::Duration::MAX)?;
    memory.write_block_unsafe(dst, WriteSource::Bytes(&[0xee; 16]))?;
    r.texture_cache.pop_async_flush(&mut r.buffer_cache)?;
    memory.reader().read_block_unsafe(alias, &mut result)?;
    assert_eq!(result, expected, "native DMA target differs from source");
    // A read-only consumer must patch exactly the lanes shared with the DMA.
    if format.is_depth_stencil() {
        r.change_resolution(crate::ResolutionScaling::new(2, 0))?;
        r.scale_image(other, true, false)?;
    }
    r.prepare_image(other, false, false)?;
    assert!(!r.texture_cache.get(other).dma.pending());
    r.download_image_memory(other)?;
    memory.reader().read_block_unsafe(alias, &mut result)?;
    assert_eq!(
        result, expected,
        "native alias lost untouched texels or retained old DMA bytes"
    );
    // A clear wins over all older patches; a later fence must publish the clear.
    clear(&mut r, 0x2b)?;
    assert!(r.accelerate_dma(&copy, &SPACE)?);
    assert!(r.texture_cache.get(other).dma.pending());
    clear(&mut r, 0x39)?;
    assert!(!r.texture_cache.get(other).dma.pending());
    r.download_image_memory(other)?;
    memory.reader().read_block_unsafe(alias, &mut result)?;
    assert_eq!(result, [0x39; 2048]);
    // A partial CPU store to the mixed-authority image preserves other GPU texels.
    clear(&mut r, 0x30)?;
    r.texture_cache
        .pending_downloads
        .push(texture_download::PendingDownload::Image(other));
    r.commit_texture_downloads()?;
    let last_fence = r.texture_visibility_work().unwrap();
    assert!(r.accelerate_dma(&copy, &SPACE)?);
    if format.is_depth_stencil() {
        r.scale_image(other, true, false)?;
    }
    let mut reference = [0x30; 2048];
    let mut at = 0;
    for range in &expected_dma.ranges {
        let n = (range.end - range.start) as usize;
        reference[range.start as usize..range.end as usize]
            .copy_from_slice(&expected_dma.bytes[at..at + n]);
        at += n;
    }
    reference[800..804].copy_from_slice(&[1, 2, 3, 4]);
    let caches = Arc::clone(r.caches);
    drop(r);
    use nixe_memory::device_memory::DeviceMemoryInterface;
    let extent = nixe_memory::Extent {
        start: memory.physical_address(alias + 800).unwrap(),
        bytes: 4,
    };
    assert!(!caches.can_collect_cpu_write(extent));
    caches.on_cpu_write(alias + 800, 4);
    assert!(caches.must_flush_region(alias + 800, 4));
    let mut r = cache.access();
    let store = AllocationRange::new(allocation.pin(), 4096 + 800, 4)?;
    r.access_memory(
        &[store],
        &mut nixe_memory::AccessOperation::Write(vec![1, 2, 3, 4]),
    )?;
    memory.reader().read_block_unsafe(alias, &mut result)?;
    assert_eq!(result, reference);
    r.flush()?;
    r.runtime
        .device
        .wait_for_work(last_fence, std::time::Duration::MAX)?;
    r.texture_cache.pop_async_flush(&mut r.buffer_cache)?;
    memory.reader().read_block_unsafe(alias, &mut result)?;
    assert_eq!(
        result, reference,
        "older image fence overwrote a later CPU store"
    );
    assert!(r.texture_cache.get(target).initialized);
    drop(r);
    drop(cache);
    validation.check()?;
    Ok(())
}

#[test]
fn inline_upload_preserves_gpu_texels_and_supersedes_an_earlier_fence() -> Result<(), VulkanError> {
    let Some(device) = support::device_for_test(crate::ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4096)?.allocate(4096)?;
    let pixels = AllocationRange::new(allocation.pin(), 0, 4096)?;
    let memory = DeviceMemory::new(machine, 32);
    let (image_address, alias) = (0x10000, 0x20000);
    let _image_map = memory.map(image_address, Arc::from([pixels.clone()]), true)?;
    let _alias_map = memory.map(alias, Arc::from([pixels]), true)?;
    let mut cache = VulkanResidency::new(
        memory.clone(),
        device,
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let mut r = cache.access();
    let layout = ImageMemoryLayout::PitchLinear {
        row_pitch: 256,
        layer_stride: 4096,
    };
    let image = r.insert_guest_image(
        &info(image_address, ImageFormat::R32Uint, 64, 16, layout),
        &SPACE,
    )?;
    r.clear_color_image(
        image,
        0,
        vk::ClearColorValue {
            uint32: [0x41414141, 0, 0, 0],
        },
    )?;
    r.texture_cache
        .pending_downloads
        .push(texture_download::PendingDownload::Image(image));
    r.commit_texture_downloads()?;
    let fence = r.texture_visibility_work().unwrap();

    let tail = alias + 4092;
    r.prepare_inline_write(tail, 4)?;
    memory.write_block_unsafe(tail, WriteSource::Bytes(&[1, 2, 3, 4]))?;
    r.write_regions(&[nixe_gpu::native::GuestBufferRange {
        address: tail,
        size: 4,
    }])?;
    let mut prefix = [0u8; 4];
    memory
        .reader()
        .read_block_unsafe(image_address, &mut prefix)?;
    assert_eq!(
        prefix, [0x41; 4],
        "untouched GPU texels must be in RAM before the older fence publishes"
    );
    r.flush()?;
    r.runtime
        .device
        .wait_for_work(fence, std::time::Duration::MAX)?;
    r.texture_cache.pop_async_flush(&mut r.buffer_cache)?;

    let mut result = [0u8; 4096];
    memory
        .reader()
        .read_block_unsafe(image_address, &mut result)?;
    assert_eq!(result[..4092], [0x41; 4092]);
    assert_eq!(result[4092..], [1, 2, 3, 4]);
    drop(r);
    drop(cache);
    validation.check()?;
    Ok(())
}
