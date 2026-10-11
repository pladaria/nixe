//! Canonical-page mirrors and demanded readback routing.

use rustc_hash::FxHashMap as HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use nixe_gpu::BackendVisibilityRequester;
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
            pages: Mutex::new(HashMap::default()),
            requester: OnceLock::new(),
        }
    }

    #[must_use]
    pub const fn device(&self) -> NonCpuDeviceId {
        self.device
    }

    /// A texture readback covers an entire layer, but only still-cached pages
    /// belong to this device's visibility transaction. CPU-published pages have
    /// already released their mirrors and must never be written back again.
    /// Hold the mirror lock across padding preservation and format conversion:
    /// publication/release of another page cannot split those two operations.
    pub(crate) fn update_cached_backing(
        &self,
        backing: &nixe_memory::CanonicalBackingRange,
        required_page: CanonicalPageId,
        update: impl FnOnce(&mut [u8]) -> Result<(), VisibilityCoordinatorError>,
    ) -> Result<(), VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        if !pages.contains_key(&required_page) {
            return Err(VisibilityCoordinatorError::new(
                "demanded GPU page has no prepared mirror",
            ));
        }
        let size = usize::try_from(backing.size())
            .map_err(|_| VisibilityCoordinatorError::new("image backing size overflows usize"))?;
        let mut bytes = vec![0; size];
        let mut position = 0;
        for segment in backing.segments() {
            let size = segment.size() as usize;
            let offset = segment.offset() as usize;
            if let Some(page) = pages.get(&segment.page()) {
                let source = page.bytes.get(offset..offset + size).ok_or_else(|| {
                    VisibilityCoordinatorError::new("image exceeds its prepared canonical page")
                })?;
                bytes[position..position + size].copy_from_slice(source);
            }
            position += size;
        }
        update(&mut bytes)?;
        position = 0;
        for segment in backing.segments() {
            let size = segment.size() as usize;
            let offset = segment.offset() as usize;
            if let Some(page) = pages.get_mut(&segment.page()) {
                page.bytes[offset..offset + size]
                    .copy_from_slice(&bytes[position..position + size]);
            }
            position += size;
        }
        Ok(())
    }

    pub(crate) fn write_page_range(
        &self,
        page: CanonicalPageId,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let mirror = pages.get_mut(&page).ok_or_else(|| {
            VisibilityCoordinatorError::new(
                "GPU writeback reached a page which was not prepared for device access",
            )
        })?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| VisibilityCoordinatorError::new("page range overflows"))?;
        let destination = mirror.bytes.get_mut(offset..end).ok_or_else(|| {
            VisibilityCoordinatorError::new("GPU writeback exceeds its prepared canonical page")
        })?;
        destination.copy_from_slice(bytes);
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

    pub(crate) fn copy_completed_page(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        let pages = self
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
        // The caller may discard this result if a GPU owner advances while
        // the readback is in flight. Only canonical publication acknowledges
        // that the mirror can be released, never completion of the transfer.
        Ok(page.bytes.clone())
    }

    pub(crate) fn mark_page_completed(
        &self,
        page: CanonicalPageId,
        point: DeviceVisibilityPoint,
    ) -> Result<(), VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let mirror = pages.get_mut(&page).ok_or_else(|| {
            VisibilityCoordinatorError::new(
                "completed GPU write has no prepared canonical page mirror",
            )
        })?;
        mirror.completed = Some(mirror.completed.map_or(point, |current| current.max(point)));
        Ok(())
    }
}

impl VisibilityCoordinator for WgpuVisibilityCoordinator {
    fn release_cpu_page(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<(), VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        if pages.get(&request.page).is_some_and(|mirror| {
            mirror
                .completed
                .is_some_and(|point| point <= request.visible_at)
        }) {
            pages.remove(&request.page);
        }
        Ok(())
    }

    fn cache_cpu_page(
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
            return self.copy_completed_page(request);
        }
        let requester = self.requester.get().ok_or_else(|| {
            VisibilityCoordinatorError::new("wgpu visibility requester is not bound")
        })?;
        requester.make_cpu_visible(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    #[test]
    fn image_readback_updates_only_device_pages_and_preserves_their_padding() {
        let allocation = CanonicalAllocation::zeroed(8192, 4096).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let coordinator = WgpuVisibilityCoordinator::new(NonCpuDeviceId::new(1));
        let point = DeviceVisibilityPoint::new(2);
        for segment in range.segments() {
            coordinator
                .cache_cpu_page(
                    DeviceVisibilityRequest {
                        page: segment.page(),
                        size: 4096,
                        device: coordinator.device(),
                        visible_at: point,
                    },
                    &[0xaa; 4096],
                )
                .unwrap();
        }
        let first = range.segments()[0].page();
        let second = range.segments()[1].page();
        let request = |page| CpuVisibilityRequest {
            page,
            size: 4096,
            device: coordinator.device(),
            visible_at: point,
        };
        coordinator.mark_page_completed(first, point).unwrap();
        coordinator.release_cpu_page(request(first)).unwrap();
        // A whole-layer texture copy contains data for the CPU-published first
        // page too. Only the second page may consume it, retaining row padding.
        coordinator
            .update_cached_backing(&range, second, |bytes| {
                bytes[..16].fill(0x11);
                bytes[4096..4112].fill(0x22);
                Ok(())
            })
            .unwrap();
        coordinator.mark_page_completed(second, point).unwrap();
        let bytes = coordinator.copy_completed_page(request(second)).unwrap();
        assert_eq!(&bytes[..16], &[0x22; 16]);
        assert!(bytes[16..].iter().all(|b| *b == 0xaa));
        assert!(coordinator.copy_completed_page(request(first)).is_err());
        assert!(
            coordinator
                .update_cached_backing(&range, first, |_| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn readback_mirrors_survive_retries_until_canonical_publication() {
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = range.segments()[0].page();
        let coordinator = WgpuVisibilityCoordinator::new(NonCpuDeviceId::new(1));
        let request = CpuVisibilityRequest {
            page,
            size: 4096,
            device: coordinator.device(),
            visible_at: DeviceVisibilityPoint::new(2),
        };
        coordinator
            .cache_cpu_page(
                DeviceVisibilityRequest {
                    page,
                    size: request.size,
                    device: request.device,
                    visible_at: DeviceVisibilityPoint::new(1),
                },
                &[0; 4096],
            )
            .unwrap();
        coordinator.write_page_range(page, 0, &[0x11]).unwrap();
        coordinator
            .mark_page_completed(page, request.visible_at)
            .unwrap();
        assert_eq!(coordinator.make_cpu_visible(request).unwrap()[0], 0x11);
        // Another CPU request or a newer device owner may race publication.
        assert_eq!(coordinator.make_cpu_visible(request).unwrap()[0], 0x11);
        let next = CpuVisibilityRequest {
            visible_at: DeviceVisibilityPoint::new(4),
            ..request
        };
        coordinator.write_page_range(page, 0, &[0x44]).unwrap();
        coordinator
            .mark_page_completed(page, next.visible_at)
            .unwrap();
        coordinator.release_cpu_page(request).unwrap();
        assert_eq!(coordinator.make_cpu_visible(next).unwrap()[0], 0x44);
        coordinator.release_cpu_page(next).unwrap();
        assert!(coordinator.copy_completed_page(next).is_err());
    }
}
