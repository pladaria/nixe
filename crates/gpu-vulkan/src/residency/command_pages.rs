//! Retained producer copies of physical command pages. Readers hold only this
//! metadata lock, never the buffer/texture cache or the native recorder.
//! https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L3230-L3395
use super::*;
use nixe_gpu_maxwell::CommandRead;
use std::sync::Mutex;
const PAGE: u64 = 4096;
#[derive(Default)]
pub(super) struct CommandPages {
    pages: Mutex<HashMap<u64, Page>>,
}
#[derive(Default)]
struct Page {
    drawn: bool,
    copy: Option<Copy>,
}
struct Copy {
    version: u64,
    begin: usize,
    end: usize,
    ticket: crate::ReadbackTicket,
}
impl CommandPages {
    pub(super) fn draw_write(
        &self,
        device: &nixe_memory::device_memory::DeviceMemory,
        address: u64,
        bytes: u64,
    ) {
        let mut pages = None;
        device.visit_command_pages(address, bytes, |_, physical| {
            let pages = pages.get_or_insert_with(|| self.pages.lock().unwrap());
            let page = pages.entry(physical).or_default();
            page.drawn = true;
            page.copy = None;
        });
    }
    pub(super) fn read(
        &self,
        device: &nixe_memory::device_memory::DeviceMemory,
        address: u64,
        output: &mut [u8],
        past_sync: bool,
    ) -> Result<CommandRead, VulkanError> {
        let Some(physical) = device.physical_address(address) else {
            return Ok(CommandRead::Unknown);
        };
        let at = physical.get();
        let offset = (at & (PAGE - 1)) as usize;
        if offset + output.len() > PAGE as usize {
            return Ok(CommandRead::Unknown);
        }
        let pages = self.pages.lock().unwrap();
        let page = pages.get(&(at & !(PAGE - 1)));
        // Draws may still be queued beyond the last copy/compute producer.
        // Only these pages require the host-sync frontier before reading.
        if page.is_some_and(|page| page.drawn) && !past_sync {
            return Ok(CommandRead::NeedsSync);
        }
        let (version, synced) = device.command_page_state(at);
        if synced {
            return Ok(CommandRead::Synced);
        }
        let Some(page) = page else {
            return Ok(CommandRead::Unknown);
        };
        if page.drawn {
            return Ok(CommandRead::Unknown);
        }
        let Some(copy) = &page.copy else {
            return Ok(CommandRead::Unknown);
        };
        if copy.version != version || offset < copy.begin || offset + output.len() > copy.end {
            return Ok(CommandRead::Unknown);
        }
        let Some(bytes) = copy.ticket.try_completed()? else {
            return Ok(CommandRead::Pending(copy.ticket.work().tick()));
        };
        output.copy_from_slice(&bytes[offset..offset + output.len()]);
        Ok(CommandRead::Copied)
    }
}
impl Residency<'_> {
    pub(super) fn copy_command_pages(&mut self) -> Result<(), VulkanError> {
        let mut requests = std::mem::take(&mut self.buffer_cache.command_copy_requests);
        requests.sort_by_key(|&(_, page, _)| page);
        let mut copied = false;
        for index in 0..requests.len() {
            let (buffer, physical, address) = requests[index];
            if requests
                .get(index + 1)
                .is_some_and(|&(_, page, _)| page == physical)
            {
                continue;
            }
            let entry = &self.buffer_cache.buffers[buffer];
            let base = entry.device_address.expect("mapped command buffer");
            let begin = base.max(address);
            let end = (base + entry.native.size()).min(address + PAGE);
            if begin >= end {
                continue;
            }
            if self
                .caches
                .command_pages
                .pages
                .lock()
                .unwrap()
                .get(&physical)
                .is_some_and(|p| p.drawn)
            {
                continue;
            }
            let version = self.runtime.device_memory.command_version(physical);
            self.ensure_buffer_range(buffer, begin - base, end - begin)?;
            let source = self.use_buffer(buffer, begin - base, end - begin)?;
            let (slice, ticket) = self
                .runtime
                .device
                .submission
                .reserve_readback_aligned(PAGE, 64)?;
            let destination = self.runtime.device.submission.staging_buffer(&slice).handle;
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            unsafe {
                self.runtime.device.dispatch.cmd_copy_buffer(
                    source,
                    destination,
                    &[vk::BufferCopy::default()
                        .src_offset(begin - base)
                        .dst_offset(slice.offset + begin - address)
                        .size(end - begin)],
                    true,
                );
            }
            crate::transfer::host_read_barrier(
                &mut self.runtime.device,
                destination,
                slice.offset,
                PAGE,
            );
            self.caches
                .command_pages
                .pages
                .lock()
                .unwrap()
                .entry(physical)
                .or_default()
                .copy = Some(Copy {
                version,
                begin: (begin - address) as usize,
                end: (end - address) as usize,
                ticket,
            });
            self.buffer_cache.native_readback_bytes += end - begin;
            copied = true;
        }
        requests.clear();
        self.buffer_cache.command_copy_requests = requests;
        if copied {
            self.flush()?;
        }
        Ok(())
    }
}
impl BufferCache {
    pub(super) fn note_command_write(&mut self, buffer: BufferId, address: u64, bytes: u64) {
        if !self.collect_command_copies {
            self.command_pages
                .draw_write(&self.device_memory, address, bytes);
            return;
        }
        self.device_memory
            .visit_command_pages(address, bytes, |at, physical| {
                self.command_copy_requests.push((buffer, physical, at));
            });
    }
}
