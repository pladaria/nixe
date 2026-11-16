//! Command fetches follow host synchronization, independently of MME data reads.
//! Producer snapshots and asynchronous completion are shared with other engines.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/dma_pusher.cpp
use super::*;
use ampere::RenderMemory;
use nixe_gpu::command_memory::{CommandMemory, CommandRead};

struct Span {
    offset: usize,
    bytes: usize,
    device: u64,
    host: nixe_memory::HostAddress,
}
#[derive(Default)]
pub(super) struct Fetch {
    started: bool,
    sync: Option<u64>,
    tick: Option<u64>,
    pending: Option<PendingAccess>,
    completed: Option<Vec<u8>>,
    bytes: Vec<u8>,
    copied: usize,
    mapping: Option<[u64; 4]>,
    spans: smallvec::SmallVec<[Span; 4]>,
}
impl Fetch {
    fn resolve(&mut self, view: &mut GpuMemory<'_>, address: u64, size: usize) -> Result<bool> {
        let identity = [
            address,
            size as u64,
            view.mapping_generation(),
            view.memory.generation(),
        ];
        if self.mapping == Some(identity) {
            return Ok(true);
        }
        self.mapping = None;
        self.copied = 0;
        self.spans.clear();
        view.commit(address, size as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        if let Some((device, host)) = view
            .mapped_read_span(address, size as u64)?
            .and_then(|span| span.at(address, size as u64))
        {
            self.spans.push(Span {
                offset: 0,
                bytes: size,
                device,
                host,
            });
        } else {
            let mut offset = 0;
            while offset < size {
                let at = address + offset as u64;
                let bytes = (size - offset).min((4096 - (at & 4095)) as usize);
                let Some((device, host)) = view
                    .mapped_read_span(at, bytes as u64)?
                    .and_then(|span| span.at(at, bytes as u64))
                else {
                    return Ok(false);
                };
                self.spans.push(Span {
                    offset,
                    bytes,
                    device,
                    host,
                });
                offset += bytes;
            }
        }
        self.mapping = Some([
            address,
            size as u64,
            view.mapping_generation(),
            view.memory.generation(),
        ]);
        Ok(true)
    }
    fn finish(&mut self) -> Vec<u8> {
        self.started = false;
        self.sync = None;
        self.tick = None;
        self.copied = 0;
        self.mapping = None;
        self.spans.clear();
        std::mem::take(&mut self.bytes)
    }
    /// Recycle initialized storage after interpretation; command bytes are always
    /// refreshed, including the unfetched tail after a semaphore acquire.
    pub fn recycle(&mut self, bytes: Vec<u8>) {
        self.bytes = bytes;
    }
    pub fn started(&self) -> bool {
        self.started
    }
    pub fn ready(&mut self, commands: &dyn CommandMemory) -> Result<bool> {
        if let Some(tick) = self.tick {
            return commands
                .command_tick_done(tick)
                .map_err(|e| e.to_string().into());
        }
        if let Some(pending) = &mut self.pending {
            let Some(result) = pending.poll() else {
                return Ok(false);
            };
            let AccessResult::Read(bytes) = result.map_err(|e| e.to_string().into_boxed_str())?
            else {
                return Err("invalid command memory completion".into());
            };
            self.completed = Some(bytes);
            self.pending = None;
        }
        Ok(true)
    }
}
impl Front<'_> {
    fn commands_written(&self, fetch: &mut Fetch, address: u64, size: usize) -> Result<bool> {
        let mut view = GpuMemory::new(self.memory.space);
        if !fetch.resolve(&mut view, address, size)? {
            return Ok(true);
        }
        Ok(fetch.spans.iter().any(|span| {
            view.memory
                .may_be_gpu_written(span.device, span.bytes as u64)
        }))
    }

    pub fn read_commands(
        &mut self,
        fetch: &mut Fetch,
        address: u64,
        size: usize,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(bytes) = fetch.completed.take() {
            fetch.bytes = bytes;
            return Ok(Some(fetch.finish()));
        }
        if !fetch.started {
            self.finish_writes();
            let written = self.commands_written(fetch, address, size)?;
            let target = if written {
                self.dependencies.writer
            } else {
                self.dependencies.synced_writer
            };
            self.writer
                .wait_until(target)
                .map_err(|_| "GPU stopped before command producer")?;
            // Recheck after synchronized producers: this can be a page's first GPU write.
            if !written && target < self.dependencies.writer {
                let written = self.commands_written(fetch, address, size)?;
                if written {
                    self.writer
                        .wait_until(self.dependencies.writer)
                        .map_err(|_| "GPU stopped before command producer")?;
                }
            }
            fetch.started = true;
            fetch.sync = Some(self.dependencies.synced);
        }
        fetch.tick = None;
        fetch.bytes.resize(size, 0);
        loop {
            let result = {
                let mut view = GpuMemory::new(self.memory.space);
                let complete = fetch.resolve(&mut view, address, size)?;
                let mut result = if complete {
                    CommandRead::Copied
                } else {
                    CommandRead::Unknown
                };
                // The guard above protects all cached pointers. A generation
                // change resets both admission and copied progress before use.
                let mut offset = fetch.copied;
                let mut spans = fetch.spans.iter();
                let mut span = spans.find(|span| span.offset + span.bytes > offset);
                while offset < size {
                    let Some(current) = span else {
                        break;
                    };
                    let within = offset - current.offset;
                    let device = current.device + within as u64;
                    let count = (current.bytes - within).min((4096 - (device & 4095)) as usize);
                    let host = nixe_memory::HostAddress::from_pointer(unsafe {
                        current.host.as_ptr().add(within)
                    });
                    let part = &mut fetch.bytes[offset..offset + count];
                    let state = if view.memory.register_command_pages(device, count as u64) {
                        self.commands
                            .read_command(device, part, fetch.sync.is_none())
                            .map_err(|e| e.to_string().into_boxed_str())?
                    } else {
                        CommandRead::Synced
                    };
                    match state {
                        CommandRead::Synced => unsafe {
                            host.read_into(part);
                        },
                        CommandRead::Copied => (),
                        CommandRead::Unknown
                            if !self.commands.must_flush_region(device, count as u64) =>
                        unsafe {
                            host.read_into(part);
                        },
                        other => {
                            result = other;
                            break;
                        }
                    }
                    offset += count;
                    fetch.copied = offset;
                    if offset == current.offset + current.bytes {
                        span = spans.next();
                    }
                }
                result
            };
            match result {
                CommandRead::Copied => {
                    return Ok(Some(fetch.finish()));
                }
                CommandRead::Pending(tick) => {
                    fetch.tick = Some(tick);
                    return Ok(None);
                }
                CommandRead::NeedsSync => {
                    let target = fetch
                        .sync
                        .take()
                        .expect("command read requested host sync twice");
                    self.writer
                        .wait_until(target)
                        .map_err(|_| "GPU stopped before command host sync")?;
                }
                CommandRead::Unknown => {
                    // No retained producer exists. Materialize its ordered prefix once;
                    // the coherent access can then park this channel while other work runs.
                    self.finish_draws()?;
                    self.writer
                        .wait_until(self.writer.position())
                        .map_err(|_| "GPU stopped before command publication")?;
                    let capture = Arc::new(
                        GpuMemory::new(self.memory.space)
                            .capture(address, size as u64)
                            .map_err(|e| e.to_string().into_boxed_str())?,
                    );
                    match self
                        .memory
                        .machine
                        .read_captured_owned(&capture.ranges, 0, std::mem::take(&mut fetch.bytes))
                        .map_err(|e| e.to_string().into_boxed_str())?
                    {
                        Access::Ready(bytes) => {
                            fetch.bytes = bytes;
                            return Ok(Some(fetch.finish()));
                        }
                        Access::Pending(mut pending) => {
                            pending.retain_owner(capture);
                            let thread = thread::current();
                            pending.set_waker(Arc::new(move || thread.unpark()));
                            fetch.pending = Some(pending);
                            return Ok(None);
                        }
                    }
                }
                CommandRead::Synced => unreachable!("RAM words are copied by the command reader"),
            }
        }
    }
}
