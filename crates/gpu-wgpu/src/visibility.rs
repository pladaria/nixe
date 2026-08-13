//! Canonical-page mirrors used by the conservative staging policy.

use std::collections::HashMap;
use std::sync::Mutex;

use nixe_gpu::BackingView;
use nixe_memory::{
    CanonicalPageId, CpuVisibilityRequest, DeviceVisibilityRequest, NonCpuDeviceId,
    VisibilityCoordinator, VisibilityCoordinatorError,
};

/// Thread-safe canonical-page mirror shared by submission and completion code.
pub struct WgpuVisibilityCoordinator {
    device: NonCpuDeviceId,
    pages: Mutex<HashMap<CanonicalPageId, Box<[u8]>>>,
}

impl WgpuVisibilityCoordinator {
    pub(crate) fn new(device: NonCpuDeviceId) -> Self {
        Self {
            device,
            pages: Mutex::new(HashMap::new()),
        }
    }

    #[must_use]
    pub const fn device(&self) -> NonCpuDeviceId {
        self.device
    }

    pub(crate) fn write_backing(
        &self,
        backing: &BackingView,
        bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        if bytes.len() != backing.size() as usize {
            return Err(VisibilityCoordinatorError::new(
                "backend writeback size does not match the canonical backing view",
            ));
        }
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let mut source = 0_usize;
        for segment in backing.range().segments() {
            let size = usize::try_from(segment.size())
                .map_err(|_| VisibilityCoordinatorError::new("segment size overflows usize"))?;
            let offset = usize::try_from(segment.offset())
                .map_err(|_| VisibilityCoordinatorError::new("segment offset overflows usize"))?;
            let page = pages.get_mut(&segment.page()).ok_or_else(|| {
                VisibilityCoordinatorError::new(
                    "GPU writeback reached a page which was not prepared for device access",
                )
            })?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| VisibilityCoordinatorError::new("page range overflows"))?;
            let source_end = source
                .checked_add(size)
                .ok_or_else(|| VisibilityCoordinatorError::new("source range overflows"))?;
            if end > page.len() || source_end > bytes.len() {
                return Err(VisibilityCoordinatorError::new(
                    "GPU writeback exceeds its prepared canonical page",
                ));
            }
            page[offset..end].copy_from_slice(&bytes[source..source_end]);
            source = source_end;
        }
        Ok(())
    }
}

impl VisibilityCoordinator for WgpuVisibilityCoordinator {
    fn make_device_visible(
        &self,
        request: DeviceVisibilityRequest,
        canonical_bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        if request.device != self.device {
            return Err(VisibilityCoordinatorError::new(
                "visibility request targets another device",
            ));
        }
        if canonical_bytes.len() != request.size {
            return Err(VisibilityCoordinatorError::new(
                "canonical upload does not contain one complete page",
            ));
        }
        self.pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?
            .insert(request.page, canonical_bytes.into());
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        if request.device != self.device {
            return Err(VisibilityCoordinatorError::new(
                "visibility request targets another device",
            ));
        }
        let pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let bytes = pages.get(&request.page).ok_or_else(|| {
            VisibilityCoordinatorError::new("wgpu page has no completed device mirror")
        })?;
        if bytes.len() != request.size {
            return Err(VisibilityCoordinatorError::new(
                "wgpu page mirror has an unexpected size",
            ));
        }
        Ok(bytes.clone())
    }
}
