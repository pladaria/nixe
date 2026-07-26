use std::sync::{Arc, Mutex};

use nixe_memory::{
    CanonicalAllocation, CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint,
    DeviceVisibilityRequest, MemoryPermissions, NonCpuDeviceId, VisibilityCoordinator,
    VisibilityCoordinatorError,
};

#[derive(Default)]
struct RecordingDevice {
    uploads: Mutex<Vec<(DeviceVisibilityRequest, Box<[u8]>)>>,
}

impl VisibilityCoordinator for RecordingDevice {
    fn make_device_visible(
        &self,
        request: DeviceVisibilityRequest,
        canonical_bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        self.uploads
            .lock()
            .unwrap()
            .push((request, canonical_bytes.into()));
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        _request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        Err(VisibilityCoordinatorError::new(
            "read-only acceptance device has no writeback",
        ))
    }
}

#[test]
fn gpu_contract_accesses_a_page_spanning_range_without_cpu_context() {
    let allocation = CanonicalAllocation::zeroed(0x1800, 0x1000).unwrap();
    allocation.write(0xfff, &[0x11, 0x22]).unwrap();
    let range = allocation.backing_range(MemoryPermissions::READ).unwrap();
    let device = Arc::new(RecordingDevice::default());
    let coordinator: Arc<dyn VisibilityCoordinator> = device.clone();
    let declaration =
        DeviceAccessDeclaration::read(NonCpuDeviceId::new(1), DeviceVisibilityPoint::new(7));

    range
        .prepare_device_access(declaration, coordinator)
        .unwrap();

    let uploads = device.uploads.lock().unwrap();
    assert_eq!(uploads.len(), 2);
    assert_eq!(uploads[0].0.page, range.segments()[0].page());
    assert_eq!(uploads[1].0.page, range.segments()[1].page());
    assert_ne!(uploads[0].0.page, uploads[1].0.page);
    assert_eq!(uploads[0].1[0xfff], 0x11);
    assert_eq!(uploads[1].1[0], 0x22);
    assert_eq!(uploads[0].0.device, declaration.device());
    assert_eq!(uploads[1].0.visible_at, declaration.device_visible_at());
}
