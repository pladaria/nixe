//! Buffer writes become CPU-coherent in batches before guest fence completion.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
use super::*;
use nixe_gpu::authority::ResolvedBytes;

pub(super) struct Download {
    source: Source,
    snapshot: ResolvedBytes<Resource>,
}
enum Source {
    Native(crate::ReadbackTicket),
    Literal {
        work: WorkTicket,
        bytes: Arc<[u8]>,
        offset: usize,
    },
}
impl Download {
    pub(super) fn overlaps(&self, ranges: &[nixe_memory::AllocationRange]) -> bool {
        ranges.iter().any(|range| {
            range.allocation().identity() == self.snapshot.allocation
                && range.physical_extents().any(|extent| {
                    extent.start.get() < self.snapshot.start.get() + self.snapshot.bytes
                        && self.snapshot.start.get() < extent.start.get() + extent.bytes
                })
        })
    }
    pub(super) fn work(&self) -> WorkTicket {
        match &self.source {
            Source::Native(ticket) => ticket.work(),
            Source::Literal { work, .. } => *work,
        }
    }
}
impl VulkanResidency {
    pub(crate) fn commit_buffer_downloads(&mut self) -> Result<(), VulkanError> {
        let mut pending = std::mem::take(&mut self.buffer_writes);
        let limit = self.device.submission.configuration.arena_bytes;
        for (&id, writes) in &mut pending {
            let Ok(buffer) = self.buffer(id) else {
                continue;
            };
            let mapping = Arc::clone(buffer.contents.geometry[0].mapping());
            normalize(writes);
            for &(start, bytes) in writes.iter() {
                let footprint = Footprint::new(
                    Arc::clone(&mapping),
                    nixe_gpu::physical::WriteDomain::Buffer { start, bytes },
                )?;
                let snapshots = self.authority.resolve_footprint(&footprint)?;
                for snapshot in snapshots {
                    let Some(producer) = snapshot
                        .producer
                        .filter(|p| p.resource == Resource::Buffer(id) && !p.materialized)
                    else {
                        continue;
                    };
                    let mut offset = 0;
                    while offset < snapshot.bytes {
                        while self.buffer_downloads.len() >= 256 {
                            self.flush()?;
                            let Some(download) = self.buffer_downloads.front() else {
                                break;
                            };
                            let work = download.work();
                            self.device.wait_for_work(work, std::time::Duration::MAX)?;
                            self.progress_readbacks()?;
                        }
                        let bytes = (snapshot.bytes - offset).min(limit);
                        let mut part = snapshot;
                        part.start = nixe_memory::ArenaOffset::new(part.start.get() + offset);
                        part.logical += offset;
                        part.bytes = bytes;
                        part.producer.as_mut().unwrap().native += offset;
                        // Literal releases still publish RAM at their fence.
                        // Retain their existing payload instead of uploading
                        // and downloading bytes that the owner already knows.
                        let literal = self
                            .buffer(id)?
                            .known
                            .iter()
                            .find(|known| {
                                known.offset <= producer.native + offset
                                    && producer.native + offset + bytes
                                        <= known.offset + known.bytes.len() as u64
                            })
                            .map(|known| (known.offset, Arc::clone(&known.bytes)));
                        let source = if let Some((base, literal)) = literal {
                            Source::Literal {
                                work: self.device.work_ticket()?,
                                bytes: literal,
                                offset: (producer.native + offset - base) as usize,
                            }
                        } else {
                            Source::Native(self.download_buffer(
                                id,
                                producer.native + offset,
                                bytes,
                            )?)
                        };
                        self.buffer_downloads.push_back(Download {
                            source,
                            snapshot: part,
                        });
                        offset += bytes;
                    }
                }
            }
        }
        pending.clear();
        self.buffer_writes = pending;
        Ok(())
    }

    pub(crate) fn publish_buffer_downloads(&mut self, completed: u64) -> Result<(), VulkanError> {
        let mut ranges = Vec::new();
        while self
            .buffer_downloads
            .front()
            .is_some_and(|download| download.work().tick() <= completed)
        {
            let download = self.buffer_downloads.pop_front().unwrap();
            let mut dirty = Vec::new();
            let published = match download.source {
                Source::Native(ticket) => {
                    let bytes = self.device.readback(ticket)?;
                    self.statistics.native_readback_bytes += download.snapshot.bytes;
                    let memory = self.memory.clone();
                    self.authority.publish_readback(
                        &memory,
                        download.snapshot,
                        &bytes,
                        &mut dirty,
                    )?
                }
                Source::Literal { bytes, offset, .. } => {
                    let end = offset + download.snapshot.bytes as usize;
                    let bytes = bytes.get(offset..end).ok_or(VulkanError::InvalidResource(
                        "literal fence publication exceeds its producer",
                    ))?;
                    let memory = self.memory.clone();
                    self.authority.publish_readback(
                        &memory,
                        download.snapshot,
                        bytes,
                        &mut dirty,
                    )?
                }
            };
            self.invalidate_cpu_buffers(&dirty)?;
            ranges.extend(dirty);
            for span in published {
                self.statistics.canonical_readback_bytes += span.bytes;
                if let Some(mapping) = self.authority.mapping(span.producer.unwrap().resource, 0)
                    && let Some(pin) =
                        mapping.pin_physical(span.allocation, span.start, span.bytes)?
                {
                    ranges.push(pin);
                }
            }
        }
        self.authority
            .retire_unused_device_pages(&self.memory, &ranges)?;
        Ok(())
    }
    pub(crate) fn drain_buffer_downloads(&mut self) -> Result<(), VulkanError> {
        while let Some(download) = self.buffer_downloads.front() {
            self.device
                .wait_for_work(download.work(), std::time::Duration::MAX)?;
            self.publish_buffer_downloads(self.device.completed_tick())?;
        }
        Ok(())
    }
}
