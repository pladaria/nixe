//! Canonical-page mirrors and demanded readback routing.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use nixe_gpu::BackendVisibilityRequester;
use nixe_memory::{
    CanonicalPageId, CpuVisibilityRequest, DeviceVisibilityPoint, DeviceVisibilityRequest,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};

struct PageMirror {
    bytes: Box<[u8]>,
    completed: Option<DeviceVisibilityPoint>,
    // Exact physical bytes authored by ordered uploads since the last CPU epoch.
    known: Vec<std::ops::Range<usize>>,
}

// Interval metadata is bounded independently of command count. Exhaustion loses
// only an optimization: dirty resource ranges remain the readback authority.
const MAX_KNOWN_INTERVALS: usize = 64;
impl PageMirror {
    fn unknown(&self, range: std::ops::Range<usize>) -> Vec<std::ops::Range<usize>> {
        let mut output = Vec::new();
        let mut start = range.start;
        for known in &self.known {
            if known.end <= start {
                continue;
            }
            if known.start >= range.end {
                break;
            }
            if start < known.start {
                output.push(start..known.start.min(range.end));
            }
            start = start.max(known.end);
        }
        if start < range.end {
            output.push(start..range.end);
        }
        output
    }
    fn merge_readback(&mut self, offset: usize, bytes: &[u8]) {
        // An older aliased resource must not overwrite newer, known uploads.
        if self.known.is_empty() {
            self.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
            return;
        }
        for range in self.unknown(offset..offset + bytes.len()) {
            self.bytes[range.clone()]
                .copy_from_slice(&bytes[range.start - offset..range.end - offset]);
        }
    }
    fn invalidate(&mut self, range: std::ops::Range<usize>) {
        self.completed = None;
        if self.known.is_empty() {
            return;
        }
        let mut index = 0;
        while index < self.known.len() {
            let old = self.known[index].clone();
            if old.end <= range.start || old.start >= range.end {
                index += 1;
                continue;
            }
            if old.start < range.start && old.end > range.end {
                if self.known.len() == MAX_KNOWN_INTERVALS {
                    self.known.clear();
                    return;
                }
                self.known[index].end = range.start;
                self.known.insert(index + 1, range.end..old.end);
                break;
            } else if old.start < range.start {
                self.known[index].end = range.start;
                index += 1;
            } else if old.end > range.end {
                self.known[index].start = range.end;
                break;
            } else {
                self.known.remove(index);
            }
        }
    }

    fn remember(&mut self, range: std::ops::Range<usize>, bytes: &[u8]) {
        self.bytes[range.clone()].copy_from_slice(bytes);
        self.known.push(range);
        self.known.sort_unstable_by_key(|range| range.start);
        let mut count = 0;
        for index in 0..self.known.len() {
            let range = self.known[index].clone();
            if count > 0 && self.known[count - 1].end >= range.start {
                self.known[count - 1].end = self.known[count - 1].end.max(range.end);
            } else {
                self.known[count] = range;
                count += 1;
            }
        }
        self.known.truncate(if count > MAX_KNOWN_INTERVALS {
            0
        } else {
            count
        });
    }
}

#[derive(Default)]
struct PageMirrors {
    pages: HashMap<CanonicalPageId, PageMirror>,
    // Index only; bytes and exact coverage remain in each existing mirror.
    known_pages: BTreeSet<CanonicalPageId>,
}

/// Thread-safe canonical-page mirrors shared by submission and completion code.
pub(crate) struct WgpuVisibilityCoordinator {
    device: NonCpuDeviceId,
    pages: Mutex<PageMirrors>,
    requester: OnceLock<Arc<dyn BackendVisibilityRequester>>,
}

impl WgpuVisibilityCoordinator {
    pub(crate) fn new(device: NonCpuDeviceId) -> Self {
        Self {
            device,
            pages: Mutex::new(PageMirrors::default()),
            requester: OnceLock::new(),
        }
    }

    #[must_use]
    pub const fn device(&self) -> NonCpuDeviceId {
        self.device
    }

    pub(crate) fn write_backing(
        &self,
        backing: &nixe_memory::CanonicalBackingRange,
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
        for segment in backing.segments() {
            let size = usize::try_from(segment.size())
                .map_err(|_| VisibilityCoordinatorError::new("segment size overflows usize"))?;
            let offset = usize::try_from(segment.offset())
                .map_err(|_| VisibilityCoordinatorError::new("segment offset overflows usize"))?;
            let page = pages.pages.get_mut(&segment.page()).ok_or_else(|| {
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
            page.merge_readback(offset, &bytes[source..source_end]);
            source = source_end;
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
        let mirror = pages.pages.get_mut(&page).ok_or_else(|| {
            VisibilityCoordinatorError::new(
                "GPU writeback reached a page which was not prepared for device access",
            )
        })?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| VisibilityCoordinatorError::new("page range overflows"))?;
        mirror.bytes.get(offset..end).ok_or_else(|| {
            VisibilityCoordinatorError::new("GPU writeback exceeds its prepared canonical page")
        })?;
        mirror.merge_readback(offset, bytes);
        Ok(())
    }

    /// Invalidate physical aliases, or retain immutable upload bytes at their
    /// command position. This never completes a fence or changes page ownership.
    pub(crate) fn update_known(
        &self,
        backing: &nixe_memory::CanonicalBackingRange,
        offset: u64,
        size: u64,
        bytes: Option<&[u8]>,
    ) -> Result<(), VisibilityCoordinatorError> {
        offset
            .checked_add(size)
            .filter(|end| *end <= backing.size())
            .ok_or_else(|| {
                VisibilityCoordinatorError::new("known range exceeds canonical backing")
            })?;
        if bytes.is_some_and(|bytes| bytes.len() as u64 != size) {
            return Err(VisibilityCoordinatorError::new(
                "known upload size mismatch",
            ));
        }
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let (first, last) = backing.page_identity_bounds();
        if bytes.is_none() && pages.known_pages.range(first..=last).next().is_none() {
            return Ok(());
        }
        let mut source = 0;
        for (segment, page_offset, segment_size) in backing
            .subrange_segments(offset, size)
            .map_err(|error| VisibilityCoordinatorError::new(error.to_string()))?
        {
            let id = segment.page();
            if bytes.is_none() && !pages.known_pages.contains(&id) {
                continue;
            }
            let page = pages.pages.get_mut(&id).ok_or_else(|| {
                VisibilityCoordinatorError::new("known write has no prepared page mirror")
            })?;
            let start = page_offset as usize;
            let stop = start + segment_size as usize;
            if stop > page.bytes.len() {
                return Err(VisibilityCoordinatorError::new(
                    "known range exceeds page mirror",
                ));
            }
            if let Some(bytes) = bytes {
                page.remember(start..stop, &bytes[source..source + segment_size as usize]);
                source += segment_size as usize;
            } else {
                page.invalidate(start..stop);
            }
            let known = !page.known.is_empty();
            if known {
                pages.known_pages.insert(id);
            } else {
                pages.known_pages.remove(&id);
            }
        }
        Ok(())
    }

    pub(crate) fn unknown_ranges(
        &self,
        page: CanonicalPageId,
        offset: usize,
        size: usize,
    ) -> Result<Vec<std::ops::Range<usize>>, VisibilityCoordinatorError> {
        let pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let page = pages
            .pages
            .get(&page)
            .ok_or_else(|| VisibilityCoordinatorError::new("demand has no page mirror"))?;
        let end = offset
            .checked_add(size)
            .filter(|end| *end <= page.bytes.len())
            .ok_or_else(|| VisibilityCoordinatorError::new("demand exceeds page mirror"))?;
        Ok(page.unknown(offset..end))
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
        backing: &nixe_memory::CanonicalBackingRange,
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
        for segment in backing.segments() {
            let size = usize::try_from(segment.size())
                .map_err(|_| VisibilityCoordinatorError::new("segment size overflows usize"))?;
            let offset = usize::try_from(segment.offset())
                .map_err(|_| VisibilityCoordinatorError::new("segment offset overflows usize"))?;
            let page = pages.pages.get(&segment.page()).ok_or_else(|| {
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
        self.completed_page(request, false)
    }

    pub(crate) fn copy_completed_page(
        &self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        self.completed_page(request, true)
    }

    fn completed_page(
        &self,
        request: CpuVisibilityRequest,
        retain: bool,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        let page = pages
            .pages
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
        if retain {
            return Ok(page.bytes.clone());
        }
        pages.known_pages.remove(&request.page);
        Ok(pages
            .pages
            .remove(&request.page)
            .expect("validated page remains present while locked")
            .bytes)
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
        let mirror = pages.pages.get_mut(&page).ok_or_else(|| {
            VisibilityCoordinatorError::new(
                "completed GPU write has no prepared canonical page mirror",
            )
        })?;
        mirror.completed = Some(mirror.completed.map_or(point, |current| current.max(point)));
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
        let mut mirrors = self
            .pages
            .lock()
            .map_err(|_| VisibilityCoordinatorError::new("wgpu page mirror is poisoned"))?;
        mirrors.known_pages.remove(&request.page);
        mirrors.pages.insert(
            request.page,
            PageMirror {
                bytes: canonical_bytes.into(),
                completed: None,
                known: Vec::new(),
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
            .pages
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_knowledge_is_partial_and_unknown_alias_writes_invalidate_it() {
        let mut page = PageMirror {
            bytes: vec![0xaa; 64].into(),
            completed: Some(DeviceVisibilityPoint::new(1)),
            known: Vec::new(),
        };
        page.remember(4..12, &[1; 8]);
        assert_eq!(page.unknown(0..16), vec![0..4, 12..16]);
        page.merge_readback(0, &[2; 16]);
        assert_eq!(
            &page.bytes[0..16],
            &[2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        );
        page.invalidate(7..9);
        assert_eq!(page.completed, None);
        assert_eq!(page.known, vec![4..7, 9..12]);
        page.merge_readback(0, &[3; 16]);
        assert_eq!(&page.bytes[4..12], &[1, 1, 1, 3, 3, 1, 1, 1]);
        page.remember(7..9, &[4; 2]);
        assert_eq!(page.known, vec![4..12]);
    }
    #[test]
    fn knowledge_budget_falls_back_to_real_readback_without_completing_a_page() {
        let mut page = PageMirror {
            bytes: vec![0; 1024].into(),
            completed: None,
            known: Vec::new(),
        };
        for index in 0..=MAX_KNOWN_INTERVALS {
            page.remember(index * 4..index * 4 + 1, &[1]);
        }
        assert!(page.known.is_empty());
        assert_eq!(page.unknown(0..1024), vec![0..1024]);
        assert_eq!(page.completed, None);
    }
    #[test]
    fn a_new_cpu_epoch_discards_upload_knowledge_and_completion() {
        let allocation = nixe_memory::CanonicalAllocation::zeroed(64, 4096).unwrap();
        let backing = allocation
            .backing_range(nixe_memory::MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = backing.segments()[0].page();
        let coordinator = WgpuVisibilityCoordinator::new(NonCpuDeviceId::new(900));
        let request = DeviceVisibilityRequest {
            page,
            size: 64,
            device: coordinator.device(),
            visible_at: DeviceVisibilityPoint::new(1),
        };
        coordinator.make_device_visible(request, &[0; 64]).unwrap();
        coordinator
            .update_known(&backing, 4, 4, Some(&[1; 4]))
            .unwrap();
        coordinator
            .mark_page_completed(page, request.visible_at)
            .unwrap();
        coordinator
            .make_device_visible(
                DeviceVisibilityRequest {
                    visible_at: DeviceVisibilityPoint::new(2),
                    ..request
                },
                &[2; 64],
            )
            .unwrap();
        assert_eq!(
            coordinator.unknown_ranges(page, 0, 64).unwrap(),
            vec![0..64]
        );
        assert!(
            coordinator
                .copy_completed_page(CpuVisibilityRequest {
                    page,
                    size: 64,
                    device: coordinator.device(),
                    visible_at: request.visible_at
                })
                .is_err()
        );
    }
}
