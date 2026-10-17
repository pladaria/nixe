//! Demanded transfers are recorded by the queue owner and scattered only after
//! their real timeline point. CPU callers wait in the mailbox, not this recorder.
use super::*;
use crate::ReadbackTicket;
use std::collections::VecDeque;

pub(super) struct Scatter {
    pub source: usize,
    pub destination: usize,
    pub length: usize,
}
pub(super) struct Transfer {
    pub ticket: ReadbackTicket,
    pub bytes: u64,
    pub copies: Vec<Scatter>,
    pub windows: Vec<(readback::WindowKey, usize, usize)>,
}
pub(super) struct CpuReadback {
    baseline: Arc<[u8]>,
    pub bytes: Box<[u8]>,
    pub transfers: VecDeque<Transfer>,
    pub waiting_windows: usize,
    completed: DeviceVisibilityPoint,
    sealed: bool,
}
impl VulkanResidency {
    pub(crate) fn cpu_readback_submission_pending(&self) -> Result<bool, VulkanError> {
        let submitted = self.device.submitted_tick();
        for job in self.cpu_readbacks.values() {
            if self
                .device
                .visibility_work(job.completed)?
                .is_some_and(|work| work.tick() > submitted)
                || job
                    .transfers
                    .back()
                    .is_some_and(|transfer| transfer.ticket.work().tick() > submitted)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn cpu_readback_point(&self) -> Result<Option<u64>, VulkanError> {
        let mut first = None;
        for job in self.cpu_readbacks.values() {
            let point = if let Some(transfer) = job.transfers.front() {
                Some(transfer.ticket.work().tick())
            } else {
                self.device
                    .visibility_work(job.completed)?
                    .map(WorkTicket::tick)
            };
            if let Some(point) = point
                && point > self.device.completed_tick()
            {
                first = Some(first.map_or(point, |old: u64| old.min(point)));
            }
        }
        Ok(first)
    }

    pub(super) fn progress_cpu_readbacks(&mut self) -> Result<(), VulkanError> {
        let completed = self.device.poll_progress()?;
        let mut windows = Vec::new();
        for job in self.cpu_readbacks.values_mut() {
            while job
                .transfers
                .front()
                .is_some_and(|t| t.ticket.work().tick() <= completed)
            {
                let transfer = job.transfers.pop_front().unwrap();
                // readback observes the already cached completed point: no
                // Vulkan wait or speculative host read is needed here.
                let data = self.device.readback(transfer.ticket)?;
                self.statistics.native_readback_bytes += transfer.bytes;
                for copy in transfer.copies {
                    job.bytes[copy.destination..copy.destination + copy.length]
                        .copy_from_slice(&data[copy.source..copy.source + copy.length]);
                }
                for (key, offset, length) in transfer.windows {
                    windows.push((key, Box::<[u8]>::from(&data[offset..offset + length])));
                }
            }
        }
        for (key, bytes) in windows {
            if let Some(waiters) = self.image_readbacks.pending.remove(&key) {
                for (request, copies) in waiters {
                    if let Some(job) = self.cpu_readbacks.get_mut(&request) {
                        for copy in copies {
                            job.bytes[copy.destination..copy.destination + copy.length]
                                .copy_from_slice(&bytes[copy.source..copy.source + copy.length]);
                        }
                        job.waiting_windows -= 1;
                    }
                }
            }
            self.image_readbacks.install(
                key,
                bytes,
                self.device
                    .submission
                    .configuration
                    .arena_bytes
                    .saturating_mul(2) as usize,
            );
        }
        Ok(())
    }

    /// Full staging is actual backpressure. Retire completed demands first;
    /// wait only if a real submitted use is required to free the bounded arena.
    pub(super) fn reserve_cpu_readback(
        &mut self,
        bytes: u64,
        alignment: u64,
    ) -> Result<(usize, crate::queue::Slice, ReadbackTicket), VulkanError> {
        loop {
            self.progress_cpu_readbacks()?;
            match self
                .device
                .submission
                .reserve_readback_aligned(bytes, alignment)
            {
                Ok(reservation) => return Ok(reservation),
                Err(error @ VulkanError::QueuePressure { pending, .. }) => {
                    let submitted = self.flush()?;
                    let demand = self
                        .cpu_readback_point()?
                        .map(|t| self.device.submission.ticket(t));
                    let point = pending.or(submitted).or(demand);
                    let Some(work) = point else { return Err(error) };
                    let _trace = nixe_gpu::trace::span("resource_pressure_wait", work.tick(), 1, 0);
                    self.device.wait_for_work(work, std::time::Duration::MAX)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub(super) fn enqueue_cpu_transfer(
        &mut self,
        request: CpuVisibilityRequest,
        transfer: Transfer,
    ) {
        let job = self
            .cpu_readbacks
            .get_mut(&request)
            .expect("registered demand");
        // A download's host ticket is readiness, not a new guest content
        // version. CPU acknowledgement must still name the actual producer.
        job.transfers.push_back(transfer);
    }

    pub(crate) fn request_cpu_page(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Option<Box<[u8]>>, VulkanError> {
        if request.device != NonCpuDeviceId::new(self.instance.get()) {
            return Err(VulkanError::InvalidResource(
                "CPU visibility requested from foreign Vulkan owner",
            ));
        }
        self.progress_cpu_readbacks()?;
        if !self.cpu_readbacks.contains_key(&request) {
            let (baseline, bytes) = {
                let pages = self.visibility.pages.lock().unwrap();
                let page = pages
                    .get(&request.page)
                    .ok_or(VulkanError::InvalidResource(
                        "CPU visibility has no preserved page baseline",
                    ))?;
                if page.bytes.len() != request.size {
                    return Err(VulkanError::InvalidResource(
                        "CPU visibility page size mismatch",
                    ));
                }
                if page.completed.is_some_and(|p| p >= request.visible_at) {
                    return Ok(Some(page.bytes.as_ref().into()));
                }
                (Arc::clone(&page.baseline), page.bytes.as_ref().into())
            };
            let writes = self
                .index
                .pages
                .get(&request.page)
                .map(|p| p.resolved_writes())
                .unwrap_or_default();
            let point = writes
                .iter()
                .fold(request.visible_at, |p, w| p.max(w.point));
            self.cpu_readbacks.insert(
                request,
                CpuReadback {
                    bytes,
                    baseline,
                    transfers: VecDeque::new(),
                    waiting_windows: 0,
                    completed: point,
                    sealed: false,
                },
            );
            nixe_gpu::trace::detail_event(
                "readback_page",
                request.visible_at.get(),
                request.page.store().get(),
                request.page.page().get(),
            );
            if let Err(error) = self.record_cpu_authorities(request, &writes) {
                if let Some(job) = self.cpu_readbacks.remove(&request) {
                    for transfer in job.transfers {
                        self.device.discard_readback(transfer.ticket)?;
                    }
                }
                return Err(error);
            }
            self.statistics.canonical_readback_bytes +=
                writes.iter().map(|w| w.end - w.start).sum::<u64>();
            self.cpu_readbacks.get_mut(&request).unwrap().sealed = true;
            // A capture of this recording's producer freezes its byte version.
            // Reading an older prefix must not split unrelated current writes.
            if self.device.visibility_work(point)?.is_some_and(|work| {
                self.device
                    .submission
                    .recording
                    .is_some_and(|index| self.device.submission.slots[index].tick == work.tick())
            }) {
                self.device.end_visibility_scope();
            }
            self.progress_cpu_readbacks()?;
        }
        let ready = self.cpu_readbacks.get(&request).is_some_and(|job| {
            job.sealed
                && job.transfers.is_empty()
                && job.waiting_windows == 0
                && job.completed.get() <= self.device.completed_visibility()
        });
        if !ready {
            return Ok(None);
        }
        let job = self.cpu_readbacks.remove(&request).unwrap();
        let mut pages = self.visibility.pages.lock().unwrap();
        if let Some(page) = pages.get_mut(&request.page)
            && Arc::ptr_eq(&page.baseline, &job.baseline)
            && page.completed.is_none_or(|p| p <= job.completed)
        {
            page.bytes = Arc::from(job.bytes.as_ref());
            page.completed = Some(job.completed);
        }
        Ok(Some(job.bytes))
    }

    /// Owner-local memory transformations genuinely consume these bytes. They
    /// use the same transfer implementation with a targeted synchronous wait.
    pub fn materialize_cpu_page(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VulkanError> {
        let _trace = nixe_gpu::trace::span(
            "cpu_readback",
            request.visible_at.get(),
            request.size as u64,
            0,
        );
        loop {
            if let Some(bytes) = self.request_cpu_page(request)? {
                return Ok(bytes);
            }
            self.flush()?;
            let point = self
                .cpu_readback_point()?
                .ok_or(VulkanError::InvalidResource(
                    "pending CPU demand has no producing point",
                ))?;
            self.device.wait_for_work(
                self.device.submission.ticket(point),
                std::time::Duration::MAX,
            )?;
        }
    }
}
