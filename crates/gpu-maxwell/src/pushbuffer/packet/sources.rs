//! Consume live contiguous command RAM; gather scattered/device sources once.
use super::*;

/// Reusable storage owned by the GPU command processor, never a native worker.
#[derive(Default)]
pub struct MaxwellCommandSourceScratch {
    bytes: Vec<u8>,
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

pub(crate) struct SubmissionWords<'a, 'm> {
    submission: &'a MaxwellValidatedGpfifoSubmission,
    address_space: &'a MaxwellGpuMemory,
    access: &'a mut nixe_gpu::MemoryAccess<'m>,
    scratch: &'a mut MaxwellCommandSourceScratch,
    synchronize: &'a mut dyn FnMut() -> Result<(), Box<str>>,
    next_entry: usize,
    current_entry: u32,
    current_offset: usize,
    current_end: usize,
    current_segment: usize,
    direct: Option<(nixe_memory::HostAddress, nixe_memory::AllocationRange)>,
    failed: bool,
}
impl<'a, 'm> SubmissionWords<'a, 'm> {
    pub(crate) fn new(
        submission: &'a MaxwellValidatedGpfifoSubmission,
        address_space: &'a MaxwellGpuMemory,
        scratch: &'a mut MaxwellCommandSourceScratch,
        synchronize: &'a mut dyn FnMut() -> Result<(), Box<str>>,
        access: &'a mut nixe_gpu::MemoryAccess<'m>,
    ) -> Self {
        Self {
            submission,
            address_space,
            access,
            scratch,
            synchronize,
            next_entry: 0,
            current_entry: 0,
            current_offset: 0,
            current_end: 0,
            current_segment: 0,
            direct: None,
            failed: false,
        }
    }
    fn load_next_entry(&mut self) -> Result<bool, MaxwellGpfifoSourceError> {
        self.direct = None;
        let entries = self.submission.pushbuffers();
        let Some(entry) = entries.get(self.next_entry) else {
            return Ok(false);
        };
        if entry.entry().sync() == crate::MaxwellGpfifoSyncMode::Wait {
            (self.synchronize)().map_err(|detail| MaxwellGpfifoSourceError::Synchronization {
                entry_index: self.next_entry as u32,
                detail,
            })?;
        }
        // Earlier methods can produce the next entry's indirect parameters.
        // Resolve only when the owner consumes it, as in Eden DmaPusher::Step.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/dma_pusher.cpp
        let size = entry.entry().byte_count() as usize;
        self.scratch.bytes.clear();
        self.direct = self
            .address_space
            .command_ram(entry.source())
            .map_err(|error| MaxwellGpfifoSourceError::Access {
                location: self
                    .submission
                    .word_location(self.next_entry as u32, 0)
                    .expect("nonempty retained command entry"),
                error,
            })?;
        if self.direct.is_none() {
            self.scratch
                .bytes
                .try_reserve(size)
                .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
            self.scratch.bytes.resize(size, 0);
            self.submission.read_entry(
                self.address_space,
                self.next_entry as u32,
                &mut self.scratch.bytes,
                self.access,
            )?;
        }
        self.current_offset = 0;
        self.current_end = size;
        self.current_entry = self.next_entry as u32;
        self.current_segment = 0;
        self.next_entry += 1;
        Ok(true)
    }
}
impl Iterator for SubmissionWords<'_, '_> {
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
        let word_offset = self.current_offset / 4;
        let value = if let Some((pointer, _range)) = &self.direct {
            // SAFETY: the current entry retains its physical range. The cursor
            // is within that checked entry; no Rust RAM reference escapes and
            // only this command owner resolves device effects.
            unsafe { pointer.read_u32(self.current_offset) }
        } else {
            u32::from_le_bytes(
                self.scratch.bytes[self.current_offset..self.current_offset + 4]
                    .try_into()
                    .unwrap(),
            )
        };
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
