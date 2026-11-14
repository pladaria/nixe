//! Command fetches follow host synchronization, independently of MME data reads.
//! Producer snapshots and asynchronous completion are shared with other engines.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/dma_pusher.cpp
use super::*;
use ampere::RenderMemory;
use nixe_gpu::command_memory::{CommandMemory, CommandRead};

#[derive(Default)]
pub(super) struct Fetch {
    started: bool,
    sync: Option<u64>,
    tick: Option<u64>,
    pending: Option<PendingAccess>,
    completed: Option<Vec<u8>>,
    bytes: Vec<u8>,
    copied: usize,
    mapping: Option<[u64; 2]>,
}
impl Fetch {
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
    fn commands_written(&self, address: u64, size: usize) -> Result<bool> {
        let view = GpuMemory::new(self.memory.space);
        for (at, bytes) in view
            .image_mapped_ranges(address, size as u64)
            .map_err(|e| e.to_string().into_boxed_str())?
        {
            if let Some(device) = view
                .image_device_address(at)
                .map_err(|e| e.to_string().into_boxed_str())?
                && view.memory.may_be_gpu_written(device, bytes)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn read_commands(
        &mut self,
        fetch: &mut Fetch,
        address: u64,
        size: usize,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(bytes) = fetch.completed.take() {
            *fetch = Fetch::default();
            return Ok(Some(bytes));
        }
        if !fetch.started {
            self.finish_writes();
            let written = self.commands_written(address, size)?;
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
                let written = self.commands_written(address, size)?;
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
                let identity = [view.mapping_generation(), view.memory.generation()];
                if fetch.mapping != Some(identity) {
                    fetch.copied = 0;
                    fetch.mapping = Some(identity);
                }
                let mut result = CommandRead::Copied;
                let mut offset = fetch.copied;
                while offset < size {
                    let at = address + offset as u64;
                    let count = (size - offset).min((4096 - (at & 4095)) as usize);
                    let Some((device, host)) = view.read_span(at, count as u64)? else {
                        result = CommandRead::Unknown;
                        break;
                    };
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
                }
                result
            };
            match result {
                CommandRead::Copied => {
                    let bytes = std::mem::take(&mut fetch.bytes);
                    *fetch = Fetch::default();
                    return Ok(Some(bytes));
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
                            *fetch = Fetch::default();
                            return Ok(Some(bytes));
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
