//! Buffer writes become CPU-coherent in batches before guest fence completion.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
use super::*;
use crate::transfer::host_read_barrier;
use nixe_gpu::authority::ResolvedBytes;

pub(super) struct Download {
    work: Option<WorkTicket>,
    ticket: Option<crate::ReadbackTicket>,
    publications: Vec<Publication>,
}

struct Publication {
    source: Source,
    snapshot: ResolvedBytes<Resource>,
}
enum Source {
    Native { offset: usize },
    Literal { bytes: Arc<[u8]>, offset: usize },
}
impl Download {
    pub(super) fn overlaps(&self, ranges: &[nixe_memory::AllocationRange]) -> bool {
        self.publications.iter().any(|publication| {
            ranges.iter().any(|range| {
                range.allocation().identity() == publication.snapshot.allocation
                    && range.physical_extents().any(|extent| {
                        extent.start.get()
                            < publication.snapshot.start.get() + publication.snapshot.bytes
                            && publication.snapshot.start.get() < extent.start.get() + extent.bytes
                    })
            })
        })
    }
    pub(super) fn work(&self) -> Option<WorkTicket> {
        self.work
    }
}
impl VulkanResidency {
    pub(super) fn mark_buffer_write(&mut self, id: BufferId, mut start: u64, bytes: u64) {
        // Keep an interval union rather than a write log between guest fences.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        if bytes == 0 {
            return;
        }
        let ranges = self.buffer_writes.entry(id).or_default();
        let mut end = start + bytes;
        if let Some((&previous, &previous_end)) = ranges.range(..=start).next_back()
            && previous_end >= start
        {
            if previous_end >= end {
                return;
            }
            start = previous;
        }
        end = end.max(*ranges.entry(start).or_insert(end));
        while let Some((&next, &next_end)) = ranges
            .range((std::ops::Bound::Excluded(start), std::ops::Bound::Unbounded))
            .next()
        {
            if next > end {
                break;
            }
            ranges.remove(&next);
            end = end.max(next_end);
        }
        *ranges.get_mut(&start).unwrap() = end;
    }
    pub(crate) fn buffer_visibility_work(&self) -> Option<WorkTicket> {
        self.buffer_downloads.back().and_then(Download::work)
    }
    pub(crate) fn buffer_publications_pending(&self) -> bool {
        self.buffer_visibility_work()
            .is_some_and(|work| work.tick() > self.device.scheduled_tick())
    }
    pub(crate) fn commit_buffer_downloads(&mut self) -> Result<(), VulkanError> {
        if self.buffer_writes.is_empty() {
            return Ok(());
        }
        let mut pending = std::mem::take(&mut self.buffer_writes);
        let mut publications = Vec::new();
        let mut groups = Vec::new();
        let mut total = 0u64;
        for (&id, writes) in &pending {
            let Ok(buffer) = self.buffer(id) else {
                continue;
            };
            let mapping = Arc::clone(buffer.contents.geometry[0].mapping());
            let mut copies = Vec::new();
            for (&start, &end) in writes {
                let footprint = Footprint::new(
                    Arc::clone(&mapping),
                    nixe_gpu::physical::WriteDomain::Buffer {
                        start,
                        bytes: end - start,
                    },
                )?;
                for snapshot in self.authority.resolve_footprint(&footprint)? {
                    let Some(producer) = snapshot
                        .producer
                        .filter(|p| p.resource == Resource::Buffer(id) && !p.materialized)
                    else {
                        continue;
                    };
                    let literal = self
                        .buffer(id)?
                        .known
                        .iter()
                        .find(|known| {
                            known.offset <= producer.native
                                && producer.native + snapshot.bytes
                                    <= known.offset + known.bytes.len() as u64
                        })
                        .map(|known| (known.offset, Arc::clone(&known.bytes)));
                    let source = if let Some((base, bytes)) = literal {
                        Source::Literal {
                            bytes,
                            offset: (producer.native - base) as usize,
                        }
                    } else {
                        // One reservation for the fence prefix; avoid cache-line
                        // sharing between regions, like CommitAsyncFlushesHigh.
                        let offset = total;
                        total = crate::queue::align_up(
                            total.checked_add(snapshot.bytes).ok_or(
                                VulkanError::InvalidResource("fence download size overflow"),
                            )?,
                            64,
                        )
                        .ok_or(VulkanError::InvalidResource(
                            "fence download alignment overflow",
                        ))?;
                        copies.push(
                            vk::BufferCopy::default()
                                .src_offset(producer.native)
                                .dst_offset(offset)
                                .size(snapshot.bytes),
                        );
                        Source::Native {
                            offset: usize::try_from(offset).map_err(|_| {
                                VulkanError::InvalidResource(
                                    "fence download exceeds host address space",
                                )
                            })?,
                        }
                    };
                    publications.push(Publication { source, snapshot });
                }
            }
            if !copies.is_empty() {
                groups.push((id, copies));
            }
        }
        let mut work = self.buffer_visibility_work();
        let ticket = if total != 0 {
            let (slice, ticket) = self.device.submission.reserve_readback_aligned(
                &mut self.device.dispatch,
                total,
                64,
            )?;
            let target = self.device.submission.staging_buffer(&slice).handle;
            let mut sources = Vec::with_capacity(groups.len());
            for (id, copies) in &mut groups {
                let mut source = None;
                for copy in copies.iter_mut() {
                    source = Some(self.buffer_access_range(
                        *id,
                        copy.src_offset,
                        copy.size,
                        AccessScope::new(
                            vk::PipelineStageFlags2::COPY,
                            vk::AccessFlags2::TRANSFER_READ,
                        ),
                    )?);
                    copy.dst_offset += slice.offset;
                }
                sources.push(source.expect("nonempty copy group"));
            }
            self.barriers.emit(&mut self.device);
            for ((_, copies), source) in groups.iter().zip(sources) {
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(source, target, copies);
                }
            }
            host_read_barrier(&mut self.device, target, slice.offset, total);
            work = Some(ticket.work());
            Some(ticket)
        } else {
            None
        };
        if !publications.is_empty() {
            self.buffer_downloads.push_back(Download {
                work,
                ticket,
                publications,
            });
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
            .is_some_and(|download| download.work().is_none_or(|work| work.tick() <= completed))
        {
            let download = self.buffer_downloads.pop_front().unwrap();
            let staging = download
                .ticket
                .map(|ticket| self.device.completed_readback(ticket))
                .transpose()?;
            for publication in download.publications {
                let bytes = match &publication.source {
                    Source::Native { offset } => {
                        self.statistics.native_readback_bytes += publication.snapshot.bytes;
                        &staging.as_ref().expect("native download reservation")
                            [*offset..*offset + publication.snapshot.bytes as usize]
                    }
                    Source::Literal { bytes, offset } => bytes
                        .get(*offset..*offset + publication.snapshot.bytes as usize)
                        .ok_or(VulkanError::InvalidResource(
                            "literal fence publication exceeds its producer",
                        ))?,
                };
                let mut dirty = Vec::new();
                let published = self.authority.publish_readback(
                    &self.memory,
                    publication.snapshot,
                    bytes,
                    &mut dirty,
                )?;
                self.invalidate_cpu_resources(&dirty)?;
                ranges.extend(dirty);
                for span in published {
                    self.statistics.canonical_readback_bytes += span.bytes;
                    if let Some(mapping) =
                        self.authority.mapping(span.producer.unwrap().resource, 0)
                        && let Some(pin) =
                            mapping.pin_physical(span.allocation, span.start, span.bytes)?
                    {
                        ranges.push(pin);
                    }
                }
            }
        }
        self.authority
            .retire_unused_device_pages(&self.memory, &ranges)?;
        Ok(())
    }
    pub(crate) fn drain_buffer_downloads(&mut self) -> Result<(), VulkanError> {
        while let Some(download) = self.buffer_downloads.front() {
            if let Some(work) = download.work() {
                self.device.wait_for_work(work, std::time::Duration::MAX)?;
            }
            self.publish_buffer_downloads(self.device.completed_tick())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::tests::{backing_allocation, buffer, cache, cpu_read};

    #[test]
    fn fence_batch_groups_regions_and_retains_later_overlapping_producers() {
        let Some(mut cache) = cache(crate::ApiPreference::Auto, 16 * 1024 * 1024) else {
            return;
        };
        let validation = cache.device.validation_monitor();
        let first = backing_allocation(&cache, 4096, 4096, 0x19);
        let second = backing_allocation(&cache, 4096, 4096, 0x19);
        let ids = [
            buffer(&mut cache, Arc::clone(&first.1)),
            buffer(&mut cache, Arc::clone(&second.1)),
        ];
        for (index, &id) in ids.iter().enumerate() {
            cache
                .fill_buffer(id, 4, 8, 0x37373737 + index as u32)
                .unwrap();
            cache
                .fill_buffer(id, 80, 12, 0x42424242 + index as u32)
                .unwrap();
        }
        cache.commit_buffer_downloads().unwrap();
        assert_eq!(cache.buffer_downloads.len(), 1);
        let batch = &cache.buffer_downloads[0];
        assert_eq!(batch.ticket.as_ref().unwrap().slice.size, 256);
        assert_eq!(batch.publications.len(), 4);
        assert!(
            batch
                .publications
                .iter()
                .all(|p| matches!(p.source, Source::Native { offset } if offset % 64 == 0))
        );
        assert_eq!(cache.device.queue_usage().retained_readbacks, 1);
        let prefix = batch.work.unwrap();
        cache.flush().unwrap();
        cache.fill_buffer(ids[0], 4, 4, 0x71717171).unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        cache
            .device
            .wait_for_work(prefix, std::time::Duration::MAX)
            .unwrap();
        cache.publish_buffer_downloads(prefix.tick()).unwrap();
        assert_eq!(cache.buffer_downloads.len(), 1);
        assert_eq!(cache.device.queue_usage().retained_readbacks, 1);
        cache.drain_buffer_downloads().unwrap();
        assert_eq!(cache.device.queue_usage().retained_readbacks, 0);
        let a = cpu_read(&mut cache, &first.1);
        let b = cpu_read(&mut cache, &second.1);
        assert_eq!(&a[4..8], &[0x71; 4]);
        assert_eq!(&a[8..12], &[0x37; 4]);
        assert_eq!(&a[80..92], &[0x42; 12]);
        assert_eq!(&b[4..12], &0x37373738u32.to_le_bytes().repeat(2));
        assert_eq!(&b[80..92], &0x42424243u32.to_le_bytes().repeat(3));
        assert_eq!(&a[12..80], &[0x19; 68]);
        drop(cache);
        validation.check().unwrap();
    }
}
