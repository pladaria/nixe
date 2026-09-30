//! Physical residency and logical generation are independent native binding keys.
use super::*;
use nixe_gpu::{
    Backend, BackendInstanceId, BufferDescription, BufferId, CacheMaintenanceOperation,
};
use std::collections::HashSet;

// The neutral backend owns resource resolution. This test-only driver observes
// that boundary, forces real budget eviction, then delegates normal submission.
struct ResidencyProbe {
    driver: WgpuBackendDriver,
    previous: Option<BufferKey>,
}

impl BackendDriver for ResidencyProbe {
    fn create_resource(
        &mut self,
        handle: BackendResourceHandle,
        info: &BackendResourceCreateInfo,
    ) -> Result<(), BackendDriverError> {
        self.driver.create_resource(handle, info)
    }

    fn destroy_resource(
        &mut self,
        handle: BackendResourceHandle,
    ) -> Result<(), BackendDriverError> {
        self.driver.destroy_resource(handle)
    }

    fn submit(
        &mut self,
        submission: &AcceptedBackendSubmission<'_>,
    ) -> Result<(), BackendDriverError> {
        let resources = submission.resources();
        self.driver.ensure_resident(resources)?;
        let handle = dependency_handle(resources, ResourceDependency::Buffer(BufferId::new(1)))?;
        let before = BufferKey {
            handle,
            buffer: self.driver.buffer(handle)?.clone(),
        };
        if let Some(previous) = &self.previous {
            assert_eq!(previous.handle.slot(), handle.slot());
            assert_ne!(previous.handle.generation(), handle.generation());
            // Generation participates even if a host allocation were reused.
            assert_ne!(
                previous,
                &BufferKey {
                    handle,
                    buffer: previous.buffer.clone()
                }
            );
        }

        // WGPU's Eq/Hash use immutable object identity, not the mutable device.
        #[expect(clippy::mutable_key_type)]
        let keys = HashSet::from([before.clone()]);
        assert!(keys.contains(&before));
        // Reserve the whole count budget without allocating thousands of objects.
        // This calls the production eviction path, not a replacement test policy.
        self.driver
            .ensure_residency_budget(MAX_RESIDENT_RESOURCE_COUNT, 0, None)?;
        assert!(self.driver.resource_record(handle)?.host.is_none());
        assert_eq!(self.driver.resident_resources, 0);
        assert_eq!(self.driver.resident_resource_bytes, 0);
        self.driver.ensure_resident(resources)?;
        let after = BufferKey {
            handle,
            buffer: self.driver.buffer(handle)?.clone(),
        };
        assert_eq!(before.handle, after.handle);
        assert_ne!(before.buffer, after.buffer);
        assert!(
            !keys.contains(&after),
            "recreated host buffer must miss the old binding key"
        );
        assert_eq!(self.driver.resident_resources, 1);
        assert_eq!(self.driver.resident_resource_bytes, 64);
        // Retain the old host identity across submission and logical destruction.
        self.previous = Some(before);
        self.driver.submit(submission)
    }

    fn has_completed(&mut self, token: BackendSubmissionToken) -> Result<bool, BackendDriverError> {
        self.driver.has_completed(token)
    }

    fn wait_for_completion(
        &mut self,
        token: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        self.driver.wait_for_completion(token)
    }

    fn release_submission(
        &mut self,
        token: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        self.driver.release_submission(token)
    }

    fn acquire_presentable_image(
        &mut self,
        request: nixe_gpu::PresentationImageRequest,
    ) -> Result<nixe_gpu::ResidentImage, BackendDriverError> {
        self.driver.acquire_presentable_image(request)
    }

    fn teardown(&mut self) -> Result<(), BackendDriverError> {
        self.driver.teardown()
    }
}

#[test]
#[ignore = "requires a Vulkan adapter; explicitly tests physical WGPU resource recreation"]
fn native_binding_keys_distinguish_residency_recreation_and_slot_reuse() {
    let instance = BackendInstanceId::new(811);
    let device_id = nixe_memory::NonCpuDeviceId::new(811);
    let Some(initialized) =
        crate::test_hardware::initialize_backend(instance, device_id, Default::default())
    else {
        return;
    };
    assert_eq!(initialized.adapter.backend, crate::HostBackend::Vulkan);
    let context = initialized.presentation_context();
    let driver = WgpuBackendDriver::new(
        instance,
        WgpuExecutionContext {
            native_vulkan: initialized.adapter.native_vulkan,
            device: context.device().clone(),
            queue: context.queue().clone(),
            queue_access: context.queue_access().clone(),
        },
        Arc::new(WgpuVisibilityCoordinator::new(device_id)),
        None,
        None,
        GpuCacheConfiguration::default(),
    );
    let mut backend = Backend::new(
        instance,
        initialized.runtime.capabilities().clone(),
        ResidencyProbe {
            driver,
            previous: None,
        },
    );
    for serial in 1..=4 {
        let handle = backend
            .create_resource(BackendResourceCreateInfo::Buffer {
                id: BufferId::new(1),
                description: BufferDescription::new(64).unwrap(),
                view: None,
            })
            .unwrap();
        // No simulated draw: this operation only resolves the live dependency.
        // Native rendered pixels and descriptor lifetimes have separate tests.
        let submission = nixe_gpu::OperationSubmission::new(
            nixe_gpu::FrontendSubmissionId::new(serial),
            vec![],
            vec![nixe_gpu::GpuOperation::new(
                GpuCommand::CacheMaintenance(CacheMaintenanceOperation::InvalidateSamplerCaches),
                [],
                [ResourceDependency::Buffer(BufferId::new(1))],
                nixe_gpu::CapabilityRequirements::none(),
            )],
        )
        .unwrap();
        let token = backend.submit(&submission).unwrap();
        backend.wait_for_completion(token).unwrap();
        backend.release_submission(token).unwrap();
        backend.destroy_resource(handle).unwrap();
    }
    backend.teardown().unwrap();
}
