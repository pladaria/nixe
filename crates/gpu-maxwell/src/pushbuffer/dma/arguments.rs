//! Borrowed literal words from a coherent command fetch or ordered method batch.
use super::MaxwellPushbufferSubchannel;
use crate::MaxwellMethodSource;
use nixe_gpu::GpuMethodId;
use std::ops::{Bound, RangeBounds};

#[derive(Clone, Copy)]
pub(crate) struct MaxwellDmaArguments<'a> {
    origin: crate::MaxwellGpfifoSourceLocation,
    bytes: &'a [u8],
}
impl<'a> MaxwellDmaArguments<'a> {
    pub(crate) fn owned(origin: crate::MaxwellGpfifoSourceLocation, bytes: &'a [u8]) -> Self {
        Self { origin, bytes }
    }
    pub(crate) fn bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub(crate) fn len(self) -> usize {
        self.bytes.len() / 4
    }
    #[inline]
    pub(crate) fn argument_at(self, index: usize) -> u32 {
        u32::from_le_bytes(self.bytes[index * 4..index * 4 + 4].try_into().unwrap())
    }
    pub(crate) fn source(
        self,
        index: usize,
        method: GpuMethodId,
        subchannel: MaxwellPushbufferSubchannel,
    ) -> MaxwellMethodSource {
        MaxwellMethodSource::dma(
            crate::MaxwellGpfifoSourceLocation {
                word_offset: self.origin.word_offset + index as u64,
                ..self.origin
            },
            subchannel,
            method,
            self.argument_at(index),
        )
    }
    pub(crate) fn write_source(self) -> nixe_memory::WriteSource<'a> {
        nixe_memory::WriteSource::Bytes(self.bytes)
    }
    pub(crate) fn arguments(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        (0..self.len()).map(move |index| self.argument_at(index))
    }
    pub(crate) fn append_arguments(
        self,
        output: &mut Vec<u32>,
    ) -> Result<(), std::collections::TryReserveError> {
        output.try_reserve(self.len())?;
        output.extend(self.arguments());
        Ok(())
    }
    pub(crate) fn slice(self, range: impl RangeBounds<usize>) -> Self {
        let begin = match range.start_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => i + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(i) => i + 1,
            Bound::Excluded(i) => *i,
            Bound::Unbounded => self.len(),
        };
        Self {
            origin: crate::MaxwellGpfifoSourceLocation {
                word_offset: self.origin.word_offset + begin as u64,
                ..self.origin
            },
            bytes: &self.bytes[begin * 4..end * 4],
        }
    }
}
