//! CPU callbacks copy local bytes only; GPU work stays on the queue owner.
use nixe_gpu::BackendVisibilityRequester;
use nixe_memory::{
    CanonicalPageId, CpuVisibilityRequest, DeviceVisibilityPoint, DeviceVisibilityRequest,
    VisibilityCoordinator, VisibilityCoordinatorError,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, mpsc},
};

pub(super) struct CpuPage {
    pub bytes: Box<[u8]>,
    pub completed: Option<DeviceVisibilityPoint>,
}
pub struct VulkanVisibility {
    pub(super) pages: Mutex<HashMap<CanonicalPageId, CpuPage>>,
    pub(super) acknowledgements: mpsc::Sender<CpuVisibilityRequest>,
    requester: OnceLock<Arc<dyn BackendVisibilityRequester>>,
}
impl VulkanVisibility {
    pub(super) fn new(acknowledgements: mpsc::Sender<CpuVisibilityRequest>) -> Self {
        Self {
            pages: Mutex::new(HashMap::new()),
            acknowledgements,
            requester: OnceLock::new(),
        }
    }
    pub fn bind_requester(
        &self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), VisibilityCoordinatorError> {
        self.requester.set(requester).map_err(|_| {
            VisibilityCoordinatorError::new("Vulkan visibility requester already bound")
        })
    }
}
impl VisibilityCoordinator for VulkanVisibility {
    fn cache_cpu_page(
        &self,
        request: DeviceVisibilityRequest,
        bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        if request.size != bytes.len() {
            return Err(VisibilityCoordinatorError::new(
                "canonical page cache size mismatch",
            ));
        }
        self.pages.lock().unwrap().insert(
            request.page,
            CpuPage {
                bytes: bytes.into(),
                completed: None,
            },
        );
        Ok(())
    }
    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        self.requester.get().ok_or_else(|| VisibilityCoordinatorError::new("Vulkan CPU visibility needs an owner requester; owner-local snapshots must use an inline resolver"))?.make_cpu_visible(request)
    }
    fn release_cpu_page(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<(), VisibilityCoordinatorError> {
        let mut pages = self.pages.lock().unwrap();
        if pages.get(&request.page).is_some_and(|page| {
            page.completed
                .is_some_and(|point| point <= request.visible_at)
        }) {
            pages.remove(&request.page);
            self.acknowledgements.send(request).map_err(|_| {
                VisibilityCoordinatorError::new("Vulkan residency owner has stopped")
            })?;
        }
        Ok(())
    }
}
