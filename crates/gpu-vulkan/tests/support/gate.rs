use crate::{VulkanDevice, device::DeviceOwner};
use ash::vk;
use std::sync::Arc;
// Deterministically hold real GPU submissions in flight without timing guesses.
// This external test gate is not another renderer completion timeline.
pub(crate) struct Gate {
    owner: Arc<DeviceOwner>,
    pub(crate) handle: vk::Semaphore,
    completion: vk::Semaphore,
    pub(crate) last_use: u64,
    signalled: bool,
}
impl Gate {
    pub(crate) fn new(device: &VulkanDevice) -> Self {
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let handle = unsafe {
            device.owner.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .unwrap();
        Self {
            owner: Arc::clone(&device.owner),
            handle,
            completion: device.submission.timeline,
            last_use: 0,
            signalled: false,
        }
    }
    pub(crate) fn signal(&mut self) {
        if !self.signalled {
            unsafe {
                self.owner.device.signal_semaphore(
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(self.handle)
                        .value(1),
                )
            }
            .unwrap();
            self.signalled = true;
        }
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        self.signal();
        if self.last_use != 0 {
            let semaphores = [self.completion];
            let values = [self.last_use];
            unsafe {
                self.owner.device.wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values),
                    u64::MAX,
                )
            }
            .unwrap();
        }
        unsafe { self.owner.device.destroy_semaphore(self.handle, None) };
    }
}

