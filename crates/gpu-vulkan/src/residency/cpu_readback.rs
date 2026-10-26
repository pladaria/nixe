//! Exact owned operations complete after their actual native transfers. Staging
//! contains only requested producer bytes, with no whole-page RAM baseline.
use super::*;
use crate::ReadbackTicket;
use nixe_gpu::authority::{AccessReadback, ReadbackPublication};
use nixe_memory::{AccessRequestId, DeviceAccessRequest};
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
    plans: Vec<AccessReadback<Resource>>,
    pub bytes: Box<[u8]>,
    pub transfers: VecDeque<Transfer>,
    pub waiting_windows: usize,
    required_tick: u64,
    sealed: bool,
}
impl VulkanResidency {
    pub(crate) fn device_access_submission_pending(&self) -> bool {
        let scheduled = self.device.scheduled_tick();
        self.cpu_readbacks.values().any(|job| {
            job.required_tick > scheduled
                || job
                    .transfers
                    .back()
                    .is_some_and(|t| t.ticket.work().tick() > scheduled)
        }) || self
            .buffer_downloads
            .back()
            .is_some_and(|job| job.work().tick() > scheduled)
            && self.authority.access_requests().any(|id| {
                let request = self.authority.access_request(id).expect("pending access");
                !matches!(request.operation(), nixe_memory::AccessOperation::Write(_))
                    && self
                        .buffer_downloads
                        .iter()
                        .any(|job| job.work().tick() > scheduled && job.overlaps(request.ranges()))
            })
    }

    pub(crate) fn readback_point(&self) -> Option<u64> {
        let completed = self.device.completed_tick();
        self.cpu_readbacks
            .values()
            .filter_map(|job| {
                let point = if let Some(transfer) = job.transfers.front() {
                    Some(transfer.ticket.work().tick())
                } else if job.waiting_windows == 0 {
                    Some(job.required_tick)
                } else {
                    // Another job owns the shared window's real transfer ticket.
                    None
                };
                point.filter(|&tick| tick > completed)
            })
            .chain(
                self.buffer_downloads
                    .front()
                    .map(|job| job.work().tick())
                    .filter(|&tick| tick > completed),
            )
            .min()
    }

    pub(super) fn progress_readbacks(&mut self) -> Result<(), VulkanError> {
        let completed = self.device.poll_progress()?;
        self.publish_buffer_downloads(completed)?;
        let mut windows = Vec::new();
        for job in self.cpu_readbacks.values_mut() {
            while job
                .transfers
                .front()
                .is_some_and(|t| t.ticket.work().tick() <= completed)
            {
                let transfer = job.transfers.pop_front().expect("completed transfer");
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
                (crate::queue::staging::READBACK_WINDOW_BYTES * 8) as usize,
            );
        }
        let ready: Vec<_> = self
            .cpu_readbacks
            .iter()
            .filter_map(|(&id, job)| {
                (job.sealed
                    && job.transfers.is_empty()
                    && job.waiting_windows == 0
                    && (job.required_tick <= completed
                        || self.authority.access_request(id).is_none()))
                .then_some(id)
            })
            .collect();
        for id in ready {
            let job = self.cpu_readbacks.remove(&id).expect("completed demand");
            if self
                .authority
                .access_request(id)
                .is_some_and(DeviceAccessRequest::is_cancelled)
            {
                self.cancel_device_access(id)?;
            }
            // A cancelled job can still own a native window used by another
            // request. Retire its native tickets normally, but never its RAM.
            if self.authority.access_request(id).is_none() {
                continue;
            }
            let mut dirty = Vec::new();
            for plan in job.plans {
                let start = plan.source.logical as usize;
                let end = start + plan.source.bytes as usize;
                match self
                    .authority
                    .publish_access_readback(&self.memory, plan, &job.bytes[start..end], &mut dirty)
                    .map_err(memory_error)?
                {
                    ReadbackPublication::Published { bytes } => {
                        self.statistics.canonical_readback_bytes += bytes;
                    }
                    ReadbackPublication::Obsolete | ReadbackPublication::Replan => break,
                }
            }
            self.invalidate_cpu_resources(&dirty)?;
            self.authority
                .retire_unused_device_pages(&self.memory, &dirty)?;
            if self.authority.access_can_complete(id) {
                self.authority.complete_access(&self.memory, id)?;
            }
        }
        Ok(())
    }

    pub(super) fn enqueue_cpu_transfer(&mut self, request: AccessRequestId, transfer: Transfer) {
        self.cpu_readbacks
            .get_mut(&request)
            .expect("registered demand")
            .transfers
            .push_back(transfer);
    }

    pub(crate) fn fail_device_accesses(&mut self, cause: Arc<nixe_memory::MemoryError>) {
        self.authority.fail_accesses(cause);
    }
    pub(crate) fn queue_device_access(
        &mut self,
        request: DeviceAccessRequest,
    ) -> Result<(), VulkanError> {
        self.authority.queue_access(request)?;
        self.progress_device_accesses()
    }

    pub(crate) fn cancel_device_access(
        &mut self,
        request: AccessRequestId,
    ) -> Result<(), VulkanError> {
        self.authority.cancel_access(&self.memory, request)?;
        Ok(())
    }

    /// The authority retains the operation and its physical dependency through
    /// application. No CPU receives permission to perform a later RAM write.
    pub(crate) fn progress_device_accesses(&mut self) -> Result<(), VulkanError> {
        loop {
            self.progress_readbacks()?;
            let pending: Vec<_> = self.authority.access_requests().collect();
            let mut progressed = false;
            for id in pending {
                if self
                    .authority
                    .access_request(id)
                    .is_some_and(DeviceAccessRequest::is_cancelled)
                {
                    self.cancel_device_access(id)?;
                    progressed = true;
                    continue;
                }
                if self.cpu_readbacks.contains_key(&id) || !self.authority.access_can_prepare(id) {
                    continue;
                }
                let request = self
                    .authority
                    .access_request(id)
                    .expect("pending exact access");
                if !matches!(request.operation(), nixe_memory::AccessOperation::Write(_))
                    && self
                        .buffer_downloads
                        .iter()
                        .any(|download| download.overlaps(request.ranges()))
                {
                    // A fence already owns this transfer. Let its exact publication
                    // satisfy the CPU demand instead of downloading the bytes twice.
                    continue;
                }
                let plans = self.authority.prepare_access(id).map_err(memory_error)?;
                progressed = true;
                if plans.is_empty() {
                    let memory = self.memory.clone();
                    self.consume_cpu_writes(memory.collect_cpu_writes()?)?;
                    self.authority.complete_access(&self.memory, id)?;
                    continue;
                }
                let bytes = self
                    .authority
                    .access_request(id)
                    .expect("pending request")
                    .bytes();
                let bytes = plans.iter().fold(bytes, |size, plan| {
                    size.max((plan.source.logical + plan.source.bytes) as usize)
                });
                let required_tick = plans
                    .iter()
                    .map(|p| p.source.producer.expect("device source").tick.0)
                    .max()
                    .expect("nonempty readback plan");
                let sources: Vec<_> = plans.iter().map(|p| p.source).collect();
                self.cpu_readbacks.insert(
                    id,
                    CpuReadback {
                        plans,
                        bytes: vec![0; bytes].into_boxed_slice(),
                        transfers: VecDeque::new(),
                        waiting_windows: 0,
                        required_tick,
                        sealed: false,
                    },
                );
                // Large demands can retire completed staging during recording;
                // sealing prevents those partial transfers completing a request.
                self.record_access_readbacks(id, &sources)?;
                self.cpu_readbacks
                    .get_mut(&id)
                    .expect("recording demand")
                    .sealed = true;
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Retain the demand without blocking the command owner on native completion.
    pub(crate) fn begin_access_memory(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_gpu::OwnerAccess, VulkanError> {
        if let Some(result) = self.memory.try_access_captured(&ranges, &operation)? {
            return Ok(nixe_gpu::OwnerAccess::Ready(result));
        }
        let (request, mut result) = self.memory.prepare_device_access(ranges, operation)?;
        self.queue_device_access(request)?;
        self.progress_device_accesses()?;
        if let Some(result) = result.poll() {
            return result
                .map(nixe_gpu::OwnerAccess::Ready)
                .map_err(VulkanError::from);
        }
        Ok(nixe_gpu::OwnerAccess::Pending(result))
    }

    /// A command-owner read uses the same authority in place. It never submits
    /// an access to its own endpoint or waits for its own inbox to service it.
    pub(crate) fn access_memory(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
        operation: nixe_memory::AccessOperation,
    ) -> Result<nixe_memory::AccessResult, VulkanError> {
        let mut result = match self.begin_access_memory(ranges, operation)? {
            nixe_gpu::OwnerAccess::Ready(result) => return Ok(result),
            nixe_gpu::OwnerAccess::Pending(result) => result,
        };
        loop {
            self.progress_device_accesses()?;
            if let Some(result) = result.poll() {
                return result.map_err(VulkanError::from);
            }
            self.flush()?;
            self.progress_device_accesses()?;
            if let Some(result) = result.poll() {
                return result.map_err(VulkanError::from);
            }
            let point = self.readback_point().ok_or(VulkanError::InvalidResource(
                "pending exact owner access has no native progress point",
            ))?;
            self.device.wait_for_work(
                self.device.submission.ticket(point),
                std::time::Duration::MAX,
            )?;
        }
    }
}
