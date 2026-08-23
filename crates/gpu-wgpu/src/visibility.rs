//! Canonical-page mirrors and demanded readback routing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use nixe_gpu::{BackendVisibilityRequester, BackingView};
use nixe_memory::{
    CanonicalPageId, CpuVisibilityRequest, DeviceVisibilityPoint, DeviceVisibilityRequest,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};

struct PageMirror {
    bytes: Box<[u8]>,
    completed: Option<DeviceVisibilityPoint>,
}

/// Thread-safe canonical-page mirrors shared by submission and completion code.
pub(crate) struct WgpuVisibilityCoordinator {
    device: NonCpuDeviceId,
    pages: Mutex<HashMap<CanonicalPageId, PageMirror>>,
    requester: OnceLock<Arc<dyn BackendVisibilityRequester>>,
}

impl WgpuVisibilityCoordinator {
    pub(crate) fn new(device: NonCpuDeviceId) -> Self {
        Self {
            device,
            pages: Mutex::new(HashMap::new()),
            requester: OnceLock::new(),
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
            if end > page.bytes.len() || source_end > bytes.len() {
                return Err(VisibilityCoordinatorError::new(
                    "GPU writeback exceeds its prepared canonical page",
                ));
            }
            page.bytes[offset..end].copy_from_slice(&bytes[source..source_end]);
            source = source_end;
        }
        Ok(())
    }

    pub(crate) fn bind_requester(
        &self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), VisibilityCoordinatorError> {
        self.requester.set(requester).map_err(|_| {
            VisibilityCoordinatorError::new("wgpu visibility requester is already bound")
        })
    }

    pub(crate) fn read_backing(
        &self,
        backing: &BackingView,
        output: &mut [u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        if output.len() != backing.size() as usize {
            return Err(VisibilityCoordinatorError::new(
                "backend mirror read size does not match the canonical backing view",
            ));
        }
        let pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let mut destination = 0_usize;
        for segment in backing.range().segments() {
            let size = usize::try_from(segment.size())
                .map_err(|_| VisibilityCoordinatorError::new("segment size overflows usize"))?;
            let offset = usize::try_from(segment.offset())
                .map_err(|_| VisibilityCoordinatorError::new("segment offset overflows usize"))?;
            let page = pages.get(&segment.page()).ok_or_else(|| {
                VisibilityCoordinatorError::new("canonical page has no device mirror")
            })?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| VisibilityCoordinatorError::new("page range overflows"))?;
            let destination_end = destination
                .checked_add(size)
                .ok_or_else(|| VisibilityCoordinatorError::new("destination range overflows"))?;
            output[destination..destination_end].copy_from_slice(&page.bytes[offset..end]);
            destination = destination_end;
        }
        Ok(())
    }

    pub(crate) fn take_completed_page(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let page = pages
            .get(&request.page)
            .ok_or_else(|| VisibilityCoordinatorError::new("wgpu page has no device mirror"))?;
        if !page
            .completed
            .is_some_and(|completed| completed >= request.visible_at)
        {
            return Err(VisibilityCoordinatorError::new(
                "wgpu page mirror has not reached the requested visibility point",
            ));
        }
        if page.bytes.len() != request.size {
            return Err(VisibilityCoordinatorError::new(
                "wgpu page mirror has an unexpected size",
            ));
        }
        Ok(pages
            .remove(&request.page)
            .expect("validated page remains present while locked")
            .bytes)
    }

    pub(crate) fn mark_backing_completed(
        &self,
        backing: &BackingView,
        point: DeviceVisibilityPoint,
    ) -> Result<(), VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        for segment in backing.range().segments() {
            let page = pages.get_mut(&segment.page()).ok_or_else(|| {
                VisibilityCoordinatorError::new(
                    "completed GPU write has no prepared canonical page mirror",
                )
            })?;
            page.completed = Some(page.completed.map_or(point, |current| current.max(point)));
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
            .insert(
                request.page,
                PageMirror {
                    bytes: canonical_bytes.into(),
                    completed: None,
                },
            );
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
        let completed = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?
            .get(&request.page)
            .and_then(|page| page.completed)
            .is_some_and(|completed| completed >= request.visible_at);
        if completed {
            return self.take_completed_page(request);
        }
        let requester = self.requester.get().ok_or_else(|| {
            VisibilityCoordinatorError::new("wgpu visibility requester is not bound")
        })?;
        requester.make_cpu_visible(request)
    }
}
