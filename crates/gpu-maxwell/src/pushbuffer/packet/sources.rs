//! Capture command bytes before execution can release their guest storage.
use super::*;

const MAX_COMMAND_CAPTURE_BYTES: usize = 64 * 1024 * 1024;

/// Reusable storage owned by the GPU command processor, never a native worker.
#[derive(Default)]
pub struct MaxwellCommandSourceScratch {
    bytes: Vec<u8>,
    ends: Vec<usize>,
    packet: PacketScratch,
}

impl MaxwellCommandSourceScratch {
    pub(crate) fn take_packet_scratch(&mut self) -> PacketScratch {
        std::mem::take(&mut self.packet)
    }
    pub(crate) fn recycle_packet_scratch(&mut self, packet: PacketScratch) {
        self.packet = packet;
    }
}

pub(crate) struct SubmissionWords<'a> {
    submission: &'a MaxwellValidatedGpfifoSubmission,
    address_space: &'a MaxwellGpuAddressSpace,
    scratch: &'a mut MaxwellCommandSourceScratch,
    synchronize: &'a mut dyn FnMut() -> Result<(), Box<str>>,
    next_entry: usize,
    captured_start: usize,
    captured_end: usize,
    current_entry: u32,
    current_begin: usize,
    current_offset: usize,
    current_end: usize,
    current_segment: usize,
    failed: bool,
}
impl<'a> SubmissionWords<'a> {
    pub(crate) fn new(
        submission: &'a MaxwellValidatedGpfifoSubmission,
        address_space: &'a MaxwellGpuAddressSpace,
        scratch: &'a mut MaxwellCommandSourceScratch,
        synchronize: &'a mut dyn FnMut() -> Result<(), Box<str>>,
    ) -> Self {
        Self {
            submission,
            address_space,
            scratch,
            synchronize,
            next_entry: 0,
            captured_start: 0,
            captured_end: 0,
            current_entry: 0,
            current_begin: 0,
            current_offset: 0,
            current_end: 0,
            current_segment: 0,
            failed: false,
        }
    }
    fn capture_group(&mut self) -> Result<(), MaxwellGpfifoSourceError> {
        if self.next_entry == 0 {
            self.submission.validate_sources(self.address_space)?;
        }
        let entries = self.submission.pushbuffers();
        if entries[self.next_entry].entry().sync() == crate::MaxwellGpfifoSyncMode::Wait {
            (self.synchronize)().map_err(|detail| MaxwellGpfifoSourceError::Synchronization {
                entry_index: self.next_entry as u32,
                detail,
            })?;
        }
        let end = entries[self.next_entry + 1..]
            .iter()
            .position(|entry| entry.entry().sync() == crate::MaxwellGpfifoSyncMode::Wait)
            .map_or(entries.len(), |offset| self.next_entry + 1 + offset);
        let size = entries[self.next_entry..end]
            .iter()
            .try_fold(0usize, |size, entry| {
                size.checked_add(entry.entry().byte_count() as usize)
                    .ok_or(MaxwellGpfifoSourceError::ArithmeticOverflow)
            })?;
        if size > MAX_COMMAND_CAPTURE_BYTES {
            return Err(MaxwellGpfifoSourceError::CaptureTooLarge {
                bytes: size,
                limit: MAX_COMMAND_CAPTURE_BYTES,
            });
        }
        self.scratch.bytes.clear();
        self.scratch.ends.clear();
        self.scratch
            .bytes
            .try_reserve(size)
            .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
        self.scratch
            .ends
            .try_reserve(end - self.next_entry)
            .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
        self.scratch.bytes.resize(size, 0);
        let mut at = 0;
        for (offset, entry) in entries[self.next_entry..end].iter().enumerate() {
            let next = at + entry.entry().byte_count() as usize;
            self.submission.read_entry(
                self.address_space,
                (self.next_entry + offset) as u32,
                &mut self.scratch.bytes[at..next],
            )?;
            self.scratch.ends.push(next);
            at = next;
        }
        self.captured_start = self.next_entry;
        self.captured_end = end;
        Ok(())
    }
    fn load_next_entry(&mut self) -> Result<bool, MaxwellGpfifoSourceError> {
        if self.next_entry == self.submission.pushbuffers().len() {
            return Ok(false);
        }
        if self.next_entry == self.captured_end {
            self.capture_group()?;
        }
        let index = self.next_entry - self.captured_start;
        self.current_begin = if index == 0 {
            0
        } else {
            self.scratch.ends[index - 1]
        };
        self.current_offset = self.current_begin;
        self.current_end = self.scratch.ends[index];
        self.current_entry = self.next_entry as u32;
        self.current_segment = 0;
        self.next_entry += 1;
        Ok(true)
    }
}
impl Iterator for SubmissionWords<'_> {
    type Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        while self.current_offset == self.current_end {
            match self.load_next_entry() {
                Ok(true) => {}
                Ok(false) => return None,
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            }
        }
        let word_offset = (self.current_offset - self.current_begin) / 4;
        let value = u32::from_le_bytes(
            self.scratch.bytes[self.current_offset..self.current_offset + 4]
                .try_into()
                .unwrap(),
        );
        self.current_offset += 4;
        let pushbuffer = &self.submission.pushbuffers()[self.current_entry as usize];
        let address = pushbuffer.entry().address().get() + word_offset as u64 * 4;
        let segments = pushbuffer.source().segments();
        while address
            >= segments[self.current_segment].gpu_offset().get()
                + segments[self.current_segment].size()
        {
            self.current_segment += 1;
        }
        let mapping = segments[self.current_segment].mapping();
        Some(Ok(MaxwellPushbufferWord::new(
            value,
            MaxwellGpfifoSourceLocation {
                channel: self.submission.channel(),
                frontend: self.submission.frontend(),
                entry_index: self.current_entry,
                pushbuffer: pushbuffer.entry().address(),
                word_offset: word_offset as u64,
                mapping: mapping.id(),
                generation: mapping.generation(),
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use nixe_gpu::FrontendSubmissionId;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};
    use std::cell::Cell;

    fn exercise_capture(wait: bool) {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(0, &1u32.to_le_bytes()).unwrap();
        allocation.write(4, &2u32.to_le_bytes()).unwrap();
        let mut space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        channel.bind_address_space(space.id()).unwrap();
        let mut descriptors = Vec::new();
        for index in 0..2 {
            let address = mapping.offset().get() + index * 4;
            descriptors.extend_from_slice(&(address as u32).to_le_bytes());
            let flags = if index == 1 && wait { 1 << 31 } else { 0 };
            descriptors
                .extend_from_slice(&((address >> 32) as u32 | (1 << 10) | flags).to_le_bytes());
        }
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            2,
            MaxwellGpfifoSubmitRequest {
                entry_count: 2,
                flags: 4,
                fence_id: 0,
                fence_value: 0,
            },
            &descriptors,
        )
        .unwrap();
        let submission =
            resolve_gpfifo_submission(&channel, FrontendSubmissionId::new(1), decoded, &space)
                .unwrap();
        let mut scratch = MaxwellCommandSourceScratch::default();
        let waits = Cell::new(0);
        let mut synchronize = || {
            waits.set(waits.get() + 1);
            Ok(())
        };
        let mut words = SubmissionWords::new(&submission, &space, &mut scratch, &mut synchronize);
        assert_eq!(words.next().unwrap().unwrap().value(), 1);
        assert_eq!(waits.get(), 0);
        // A preceding effect releases storage that the guest then repurposes.
        allocation.write(4, &99u32.to_le_bytes()).unwrap();
        let second = words.next().unwrap().unwrap();
        assert_eq!(second.value(), if wait { 99 } else { 2 });
        assert_eq!(second.location().entry_index, 1);
        assert_eq!(second.location().word_offset, 0);
        assert_eq!(waits.get(), usize::from(wait));
        assert!(words.next().is_none());
    }
    #[test]
    fn pending_command_entries_keep_captured_bytes_after_a_guest_rewrite() {
        exercise_capture(false);
    }
    #[test]
    fn sync_wait_captures_following_sources_after_the_real_dependency() {
        exercise_capture(true);
    }
}
