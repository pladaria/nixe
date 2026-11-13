//! Per-stage uniform uploads retained for the current command buffer.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/buffer_cache/uniform_stream_reuse.h
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/buffer_cache/buffer_cache.h#L1723
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

const STAGES: usize = 5;
const SLOTS: usize = nixe_gpu_maxwell::MAXWELL_CONSTANT_BUFFER_SLOT_COUNT;
const MAX_BYTES: usize = 4096;
const MAX_WRITES: usize = 16;

struct Slot {
    allocation: Option<crate::queue::Slice>,
    tick: u64,
    address: u64,
    size: usize,
    bytes: [u8; MAX_BYTES],
}
impl Default for Slot {
    fn default() -> Self {
        Self {
            allocation: None,
            tick: 0,
            address: 0,
            size: 0,
            bytes: [0; MAX_BYTES],
        }
    }
}

pub(super) struct UniformStreams {
    slots: Box<[Slot]>,
    scratch: [u8; MAX_BYTES],
    clean: [u32; STAGES],
    // Half-open ranges for inline data and CPU/GPU notifications. Never spills.
    writes: smallvec::SmallVec<[(u64, u64); MAX_WRITES]>,
    writes_need_aliases: bool,
    sync: Arc<AtomicU64>,
    seen_sync: u64,
    context: [u64; 4],
    groups: [Option<u8>; STAGES],
}
impl UniformStreams {
    pub(super) fn new(sync: Arc<AtomicU64>) -> Self {
        Self {
            slots: (0..STAGES * SLOTS).map(|_| Slot::default()).collect(),
            scratch: [0; MAX_BYTES],
            clean: [0; STAGES],
            writes: Default::default(),
            writes_need_aliases: false,
            sync,
            seen_sync: 0,
            context: [0; 4],
            groups: [None; STAGES],
        }
    }
    pub(crate) fn invalidate(&mut self) {
        self.clean = [0; STAGES];
        self.writes.clear();
        self.writes_need_aliases = false;
    }
    fn begin(
        &mut self,
        context: [u64; 4],
        groups: [Option<u8>; STAGES],
        binds: [u32; nixe_gpu_maxwell::MAXWELL_BIND_GROUP_COUNT],
    ) {
        if self.context != context {
            self.invalidate();
            self.context = context;
        }
        for (stage, &group) in groups.iter().enumerate() {
            if self.groups[stage] != group {
                self.clean[stage] = 0;
            } else if let Some(group) = group {
                self.clean[stage] &= !binds[usize::from(group)];
            }
        }
        self.groups = groups;
    }
    pub(crate) fn forget<const STAGE: usize>(&mut self, index: usize) {
        if STAGE < STAGES {
            self.clean[STAGE] &= !(1 << index);
        }
    }
    fn note_write(&mut self, address: u64, size: u64, aliases: bool) {
        if size == 0 || self.clean == [0; STAGES] {
            return;
        }
        let end = address + size;
        self.writes_need_aliases |= aliases;
        // AppendRange: contiguous CB uploads extend the last range directly;
        // overlapping writes merge before consuming another bounded entry.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/nx2_fast_draw.h#L249-L285
        if let Some(last) = self.writes.last_mut()
            && last.1 == address
        {
            last.1 = end;
            return;
        }
        if let Some(range) = self
            .writes
            .iter_mut()
            .find(|range| address <= range.1 && range.0 <= end)
        {
            range.0 = range.0.min(address);
            range.1 = range.1.max(end);
            return;
        }
        if self.writes.len() == MAX_WRITES {
            // A widened/lost range ends uniform keeps; no other consumer needs
            // an approximation of the writes. Byte comparison remains available.
            self.invalidate();
        } else {
            self.writes.push((address, end));
        }
    }
    fn apply_writes(&mut self, device: &nixe_memory::device_memory::DeviceMemory, tick: u64) {
        // Query publication can run on the fence worker while preparation holds
        // the buffer cache. Consume its serial before each attempted keep.
        let sync = self.sync.load(Ordering::Acquire);
        if self.seen_sync != sync {
            self.invalidate();
            self.seen_sync = sync;
        }
        if self.writes.is_empty() {
            return;
        }
        for stage in 0..STAGES {
            let mut mask = self.clean[stage];
            while mask != 0 {
                let index = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let slot = &self.slots[stage * SLOTS + index];
                if slot.tick != tick {
                    self.clean[stage] &= !(1 << index);
                    continue;
                }
                let overlaps = |begin: u64, size: u64| {
                    self.writes
                        .iter()
                        .any(|&(address, end)| address < begin + size && begin < end)
                };
                let mut written = overlaps(slot.address, slot.size as u64);
                if !written && self.writes_need_aliases {
                    // Resolve aliases of the small retained range, never every
                    // page of a potentially large GPU destination. CPU write
                    // notifications already contain all device aliases.
                    device.apply_device_aliases(slot.address, slot.size as u64, |at, bytes| {
                        written = written || overlaps(at, bytes);
                    });
                }
                if written {
                    self.clean[stage] &= !(1 << index);
                }
            }
        }
        self.writes.clear();
        self.writes_need_aliases = false;
    }
}

impl BufferCache {
    pub(super) fn note_uniform_write(&mut self, address: u64, size: u64) {
        self.uniform_streams.note_write(address, size, false);
    }
    pub(super) fn note_uniform_gpu_write(&mut self, address: u64, size: u64) {
        self.uniform_streams.note_write(address, size, true);
    }
}

impl Residency<'_> {
    pub(super) fn kept_uniform<const STAGE: usize>(
        &mut self,
        index: usize,
        address: u64,
        size: u64,
    ) -> Option<crate::queue::Slice> {
        if STAGE >= STAGES {
            return None;
        }
        let tick = self.runtime.device.current_work().tick();
        let streams = &mut self.buffer_cache.uniform_streams;
        streams.apply_writes(&self.runtime.device_memory, tick);
        let slot = &streams.slots[STAGE * SLOTS + index];
        if streams.clean[STAGE] & (1 << index) != 0
            && slot.address == address
            && slot.size as u64 == size
            && slot.tick == tick
        {
            self.runtime.statistics.uniform_keeps += 1;
            return slot.allocation;
        }
        streams.forget::<STAGE>(index);
        None
    }
    pub(crate) fn begin_graphics_uniforms(
        &mut self,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        groups: [Option<u8>; STAGES],
    ) {
        let binds = std::mem::take(&mut self.runtime.buffer_bindings.channel.uniform_bind_changes);
        self.buffer_cache.uniform_streams.begin(
            [
                self.runtime
                    .buffer_bindings
                    .current_channel()
                    .expect("bound graphics channel"),
                space.id().get(),
                space.mapping_generation().get(),
                self.runtime.device_memory.generation(),
            ],
            groups,
            binds,
        );
    }
    pub(super) fn stream_guest_uniform<const STAGE: usize>(
        &mut self,
        index: usize,
        address: u64,
        range: BufferRange,
        alignment: u64,
    ) -> Result<crate::queue::Slice, VulkanError> {
        let size = range.size as usize;
        // Compute uniforms and larger unaligned banks retain their direct upload path.
        if STAGE >= STAGES || size > MAX_BYTES {
            let slice = self.runtime.device.submission.reserve_slice(
                range.size,
                crate::queue::staging::Usage::Upload,
                alignment,
            )?;
            // SAFETY: this is a fresh upload reservation, not yet referenced by a command.
            unsafe {
                self.runtime
                    .device
                    .submission
                    .staging
                    .write_with(&slice, |bytes| {
                        self.runtime
                            .device_reader
                            .read_block_unsafe(address, bytes)
                            .map_err(memory_error)
                    })?;
            }
            self.runtime.statistics.cpu_streamed_bytes += range.size;
            return Ok(slice);
        }
        let streams = &mut self.buffer_cache.uniform_streams;
        let tick = self.runtime.device.current_work().tick();
        let slot = &mut streams.slots[STAGE * SLOTS + index];
        let same_size = slot.size == size;
        let current = slot.tick == tick && slot.allocation.is_some();
        let bytes = &mut streams.scratch[..size];
        self.runtime
            .device_reader
            .read_block_unsafe(address, bytes)
            .map_err(memory_error)?;
        self.runtime.statistics.uniform_read_bytes += range.size;
        let same_bytes = same_size && slot.bytes[..size] == *bytes;
        let slice = if same_bytes && current {
            self.runtime.statistics.uniform_reuses += 1;
            slot.allocation.unwrap()
        } else {
            let slice = self.runtime.device.submission.reserve_slice(
                range.size,
                crate::queue::staging::Usage::Upload,
                alignment,
            )?;
            // SAFETY: only the new allocation is written; retained uploads stay immutable.
            unsafe {
                self.runtime
                    .device
                    .submission
                    .staging
                    .write_with(&slice, |target| {
                        target.copy_from_slice(bytes);
                        Ok(())
                    })?;
            }
            self.runtime.statistics.cpu_streamed_bytes += range.size;
            self.runtime.statistics.uniform_uploads += 1;
            slot.allocation = Some(slice);
            // The pool retains this allocation through the command buffer that first uses it.
            slot.tick = self.runtime.device.current_work().tick();
            if !same_bytes {
                slot.bytes[..size].copy_from_slice(bytes);
                slot.size = size;
            }
            slice
        };
        slot.address = address;
        streams.clean[STAGE] |= 1 << index;
        Ok(slice)
    }
}
