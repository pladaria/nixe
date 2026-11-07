//! Consume contiguous command RAM directly, gathering only split physical spans.
use super::MaxwellDmaArguments;
use crate::{
    MaxwellGpfifoSourceError, MaxwellGpfifoSourceLocation, MaxwellGpuMemory,
    MaxwellValidatedGpfifoSubmission,
};

#[derive(Default)]
pub struct MaxwellCommandSourceScratch {
    bytes: Vec<u8>,
}
impl MaxwellCommandSourceScratch {
    pub fn recycle(&mut self, completed: Self) {
        if completed.bytes.capacity() > self.bytes.capacity() {
            *self = completed;
        }
    }
}

pub(crate) struct CommandReader {
    // Raw RAM remains mapped for the machine's lifetime, including suspension
    // and guest remaps. Native resource inputs retain their own physical pins.
    _machine: nixe_memory::Machine,
    next_entry: usize,
    origin: Option<MaxwellGpfifoSourceLocation>,
    end: usize,
    gathered: Option<Vec<u8>>,
    address: Option<nixe_memory::HostAddress>,
}
impl CommandReader {
    pub(crate) fn new(machine: nixe_memory::Machine) -> Self {
        Self {
            _machine: machine,
            next_entry: 0,
            origin: None,
            end: 0,
            gathered: None,
            address: None,
        }
    }
    /// Eden's SafeRead GuestMemory uses GetSpan followed by FlushRegion, or
    /// ReadBlock into reusable scratch storage when the span is not contiguous.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/guest_memory.h#L123-L163
    pub(crate) fn load_entry<H: crate::MaxwellCommandHost>(
        &mut self,
        submission: &MaxwellValidatedGpfifoSubmission,
        memory: &MaxwellGpuMemory,
        scratch: &mut MaxwellCommandSourceScratch,
        host: &mut H,
    ) -> Result<bool, MaxwellGpfifoSourceError> {
        self.finish(scratch);
        while let Some(entry) = submission.decoded().entries().get(self.next_entry).copied() {
            let entry_index = self.next_entry;
            self.next_entry += 1;
            self.end = entry.byte_count() as usize;
            if self.end == 0 {
                continue;
            }
            let location = MaxwellGpfifoSourceLocation {
                channel: submission.channel(),
                frontend: submission.frontend(),
                entry_index: entry_index as u32,
                pushbuffer: entry.address(),
                word_offset: 0,
            };
            let span = memory
                .get_span(entry.address(), entry.byte_count())
                .map_err(|error| MaxwellGpfifoSourceError::Resolution {
                    channel: submission.channel(),
                    frontend: submission.frontend(),
                    entry_index: entry_index as u32,
                    pushbuffer: entry.address(),
                    address_space_generation: memory.mapping_generation(),
                    error,
                })?;
            self.address = Some(if let Some(span) = span {
                memory
                    .flush_region(entry.address(), entry.byte_count(), host)
                    .map_err(|error| MaxwellGpfifoSourceError::Access { location, error })?;
                span
            } else {
                scratch
                    .bytes
                    .try_reserve(self.end.saturating_sub(scratch.bytes.len()))
                    .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
                scratch.bytes.resize(self.end, 0);
                memory
                    .read_into(entry.address(), &mut scratch.bytes, host)
                    .map_err(|error| MaxwellGpfifoSourceError::Access { location, error })?;
                let bytes = std::mem::take(&mut scratch.bytes);
                let address = nixe_memory::HostAddress::from_pointer(bytes.as_ptr().cast_mut());
                self.gathered = Some(bytes);
                address
            });
            self.origin = Some(location);
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn location_at(&self, offset: usize) -> MaxwellGpfifoSourceLocation {
        let mut origin = self.origin.unwrap();
        origin.word_offset = offset as u64 / 4;
        origin
    }
    pub(crate) fn len(&self) -> usize {
        self.end / 4
    }
    pub(crate) fn word(&self, index: usize) -> u32 {
        debug_assert!(index < self.len());
        // SAFETY: GetSpan bounds contiguous RAM in the retained machine's arena;
        // otherwise the reader owns gathered storage. No RAM reference is formed.
        unsafe { self.address.unwrap().read_u32(index * 4) }
    }
    pub(crate) fn arguments(&self, first: usize, count: usize) -> MaxwellDmaArguments<'_> {
        assert!(first <= self.len() && count <= self.len() - first);
        MaxwellDmaArguments::new(self, self.address.unwrap(), first * 4, count)
    }
    pub(super) fn gathered_bytes(&self, begin: usize, end: usize) -> Option<&[u8]> {
        self.gathered.as_ref().map(|bytes| &bytes[begin..end])
    }
    pub(crate) fn finish(&mut self, scratch: &mut MaxwellCommandSourceScratch) {
        if let Some(bytes) = self.gathered.take() {
            scratch.bytes = bytes;
        }
        self.origin = None;
        self.end = 0;
        self.address = None;
    }
}
