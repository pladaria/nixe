//! Retained command bytes. Clean RAM is copied on the command owner; native
//! producers are published by the preparation owner before their bytes are read.
use super::MaxwellDmaArguments;
use crate::{
    MaxwellAddressSpaceId, MaxwellGpfifoEntry, MaxwellGpfifoSourceError,
    MaxwellGpfifoSourceLocation, MaxwellGpuMemory,
};

/// One fetch prefix, ending before the next SYNC_WAIT entry. The owner publishes
/// cache producers before copying RAM; parsing never borrows native resources.
/// https://github.com/NVIDIA/open-gpu-doc/blob/ab27fc22db5de0d02a4cabe08e555663b62db4d4/manuals/volta/gv100/dev_pbdma.ref.txt#L104-L138
pub struct MaxwellCommandFetch {
    pub space: MaxwellAddressSpaceId,
    pub origin: MaxwellGpfifoSourceLocation,
    pub entries: std::sync::Arc<[MaxwellGpfifoEntry]>,
    pub range: std::ops::Range<usize>,
    pub bytes: Vec<u8>,
    /// Validated prefix retained across the coherent owner handoff.
    pub(crate) copied: usize,
    pub(crate) generation: Option<(u64, u64)>,
}
impl MaxwellCommandFetch {
    pub(crate) fn invalidate_tail(&mut self, offset: usize) {
        self.copied = self.copied.min(offset);
    }
    pub fn take_storage(&mut self) -> Self {
        Self {
            space: self.space,
            origin: self.origin,
            entries: std::sync::Arc::clone(&self.entries),
            range: self.range.clone(),
            bytes: std::mem::take(&mut self.bytes),
            copied: self.copied,
            generation: self.generation,
        }
    }
    pub(crate) fn prepare_storage(&mut self) -> Result<(), MaxwellGpfifoSourceError> {
        let count = self.entries[self.range.clone()]
            .iter()
            .try_fold(0usize, |count, entry| {
                count.checked_add(entry.byte_count() as usize)
            })
            .ok_or(MaxwellGpfifoSourceError::ArithmeticOverflow)?;
        self.bytes
            .try_reserve(count.saturating_sub(self.bytes.len()))
            .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
        self.bytes.resize(count.max(self.bytes.len()), 0);
        Ok(())
    }
    pub fn read(
        mut self,
        memory: &mut MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Self, MaxwellGpfifoSourceError> {
        self.prepare_storage()?;
        if self.generation
            != Some((
                memory.mapping_generation().get(),
                memory.device_memory().device().generation(),
            ))
        {
            self.copied = 0;
        }
        memory
            .flush_caching(access)
            .map_err(|error| MaxwellGpfifoSourceError::Access {
                location: self.origin,
                error,
            })?;
        let mut offset = 0;
        for (index, entry) in self.entries[self.range.clone()].iter().enumerate() {
            let end = offset + entry.byte_count() as usize;
            if self.copied < end {
                let start = offset.max(self.copied);
                let address = memory
                    .address(entry.address().get() + (start - offset) as u64)
                    .map_err(|error| MaxwellGpfifoSourceError::Access {
                        location: MaxwellGpfifoSourceLocation {
                            entry_index: self.origin.entry_index + index as u32,
                            pushbuffer: entry.address(),
                            word_offset: (start - offset) as u64 / 4,
                            ..self.origin
                        },
                        error: crate::MaxwellGpuAccessError::Address(error),
                    })?;
                memory
                    .read_command_into(address, &mut self.bytes[start..end], access)
                    .map_err(|error| MaxwellGpfifoSourceError::Access {
                        location: MaxwellGpfifoSourceLocation {
                            entry_index: self.origin.entry_index + index as u32,
                            pushbuffer: entry.address(),
                            word_offset: (start - offset) as u64 / 4,
                            ..self.origin
                        },
                        error,
                    })?;
                self.copied = end;
            }
            offset = end;
        }
        Ok(self)
    }
}

pub(crate) struct CommandReader<'a> {
    origin: MaxwellGpfifoSourceLocation,
    bytes: &'a [u8],
}
impl<'a> CommandReader<'a> {
    pub(crate) fn new(origin: MaxwellGpfifoSourceLocation, bytes: &'a [u8]) -> Self {
        Self { origin, bytes }
    }
    pub(crate) fn location_at(&self, offset: usize) -> MaxwellGpfifoSourceLocation {
        MaxwellGpfifoSourceLocation {
            word_offset: offset as u64 / 4,
            ..self.origin
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.bytes.len() / 4
    }
    pub(crate) fn word(&self, index: usize) -> u32 {
        u32::from_le_bytes(self.bytes[index * 4..index * 4 + 4].try_into().unwrap())
    }
    pub(crate) fn arguments(&self, first: usize, count: usize) -> MaxwellDmaArguments<'_> {
        MaxwellDmaArguments::owned(
            self.location_at(first * 4),
            &self.bytes[first * 4..(first + count) * 4],
        )
    }
}
