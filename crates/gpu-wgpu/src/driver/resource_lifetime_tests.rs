use super::*;
use nixe_gpu::{
    BackendInstanceId, BackendResourceKind, GpuAllocationDescription, GpuAllocationId, ImageExtent,
    ImageId, ImageKind, ImageView, Swizzle,
};
use nixe_memory::{CanonicalAllocation, MemoryPermissions, NonCpuDeviceId};

#[test]
fn full_clear_reclaims_superseded_generations_and_their_page_index_entries() {
    let instance = BackendInstanceId::new(810);
    let device_id = NonCpuDeviceId::new(810);
    let Ok(initialized) = crate::initialize_backend(instance, device_id, Default::default()) else {
        return;
    };
    let context = initialized.presentation_context();
    let mut driver = WgpuBackendDriver::new(
        instance,
        WgpuExecutionContext {
            device: context.device().clone(),
            queue: context.queue().clone(),
            queue_access: context.queue_access().clone(),
        },
        Arc::new(WgpuVisibilityCoordinator::new(device_id)),
        None,
        None,
        GpuCacheConfiguration::default(),
    );
    let allocation = CanonicalAllocation::zeroed(64, 4096).unwrap();
    let backing = BackingView::new(
        GpuAllocationId::new(1),
        GpuAllocationDescription::new(64, 4).unwrap(),
        0,
        allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
    )
    .unwrap();
    let page = backing.range().segments()[0].page();
    let description = ImageDescription::new(
        ImageDimension::Two,
        ImageExtent::new(4, 4, 1).unwrap(),
        ImageFormat::Rgba8Unorm,
        ImageKind::Color,
        1,
        1,
        SampleCount::One,
    )
    .unwrap();
    let subresources = ImageSubresourceRange {
        plane: 0,
        mip_level: 0,
        base_layer: 0,
        layer_count: 1,
    };
    let image = ImageId::new(1);
    let info = BackendResourceCreateInfo::Image {
        id: image,
        description,
        view: Some(
            ImageView::new(
                image,
                description,
                Swizzle::IDENTITY,
                vec![(
                    subresources,
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: 16,
                        layer_stride: 64,
                    },
                    backing,
                )],
            )
            .unwrap(),
        ),
    };
    let target = ImageRegion {
        image,
        subresources,
        origin: ImageOrigin { x: 0, y: 0, z: 0 },
        extent: description.extent(),
    };
    for generation in 1..8 {
        let handle =
            BackendResourceHandle::new(instance, 0, generation, BackendResourceKind::Image);
        driver.create_resource(handle, &info).unwrap();
        assert_eq!(
            driver.page_resources.get(page).len(),
            if generation == 1 { 1 } else { 2 }
        );
        driver
            .reclaim_cleared_retired_images(
                handle,
                ImageRegion {
                    extent: ImageExtent::new(2, 4, 1).unwrap(),
                    ..target
                },
            )
            .unwrap();
        assert_eq!(driver.retired_resources.len(), usize::from(generation != 1));
        driver
            .reclaim_cleared_retired_images(handle, target)
            .unwrap();
        assert!(driver.retired_resources.is_empty());
        assert_eq!(driver.resident_resources, 1);
        assert_eq!(driver.resident_resource_bytes, 64);
        assert_eq!(driver.page_resources.get(page).len(), 1);
        record_device_write(
            driver.resource_record_mut(handle).unwrap(),
            nixe_gpu::AccessTarget::Image {
                image,
                subresources,
            },
            u64::from(generation),
        )
        .unwrap();
        driver.destroy_resource(handle).unwrap();
        assert_eq!(driver.retired_resources.len(), 1);
        let mut writes = Vec::new();
        let entry = driver.page_resources.get(page)[0];
        collect_demanded_writebacks(
            entry.handle,
            driver.resource_record(entry.handle).unwrap(),
            entry.binding,
            &mut writes,
        );
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].handle(), handle);
    }
}
