//! A borrowed DMA argument array; the method and count belong to DMA state.
use super::{CommandReader, MaxwellPushbufferSubchannel};
use crate::MaxwellMethodSource;
use nixe_gpu::GpuMethodId;
use nixe_memory::HostAddress;
use std::ops::{Bound, RangeBounds};

#[derive(Clone, Copy)]
pub(crate) struct MaxwellDmaArguments<'a> {
    reader: &'a CommandReader,
    address: HostAddress,
    offset: usize,
    count: usize,
}
impl<'a> MaxwellDmaArguments<'a> {
    pub(super) fn new(
        reader: &'a CommandReader,
        address: HostAddress,
        offset: usize,
        count: usize,
    ) -> Self {
        Self {
            reader,
            // SAFETY: CommandReader bounds the accepted entry before borrowing it.
            address: HostAddress::from_pointer(unsafe { address.as_ptr().add(offset) }),
            offset,
            count,
        }
    }
    pub(crate) fn len(self) -> usize {
        self.count
    }
    #[inline]
    pub(crate) fn argument_at(self, index: usize) -> u32 {
        debug_assert!(index < self.count);
        // SAFETY: this span borrows the accepted entry and is bounded at creation.
        unsafe { self.address.read_u32(index * 4) }
    }
    pub(crate) fn source(
        self,
        index: usize,
        method: GpuMethodId,
        subchannel: MaxwellPushbufferSubchannel,
    ) -> MaxwellMethodSource {
        debug_assert!(index < self.count);
        MaxwellMethodSource::dma(
            self.reader.location_at(self.offset + index * 4),
            subchannel,
            method,
            self.argument_at(index),
        )
    }
    pub(crate) fn ram_address(self) -> HostAddress {
        self.address
    }
    pub(crate) fn gathered_bytes(self) -> Option<&'a [u8]> {
        self.reader
            .gathered_bytes(self.offset, self.offset + self.count * 4)
    }
    pub(crate) fn write_source(self) -> nixe_memory::WriteSource<'a> {
        if let Some(bytes) = self.gathered_bytes() {
            nixe_memory::WriteSource::Bytes(bytes)
        } else {
            nixe_memory::WriteSource::Ram {
                address: self.ram_address(),
                offset: 0,
                bytes: self.count * 4,
            }
        }
    }
    pub(crate) fn arguments(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        (0..self.count).map(move |index| self.argument_at(index))
    }
    pub(crate) fn append_arguments(
        self,
        output: &mut Vec<u32>,
    ) -> Result<(), std::collections::TryReserveError> {
        output.try_reserve(self.count)?;
        let start = output.len();
        // SAFETY: entry admission bounds the source; reserve established disjoint
        // destination storage. The borrowed reader owns the RAM arena or gathered data.
        unsafe {
            self.address
                .copy_words_to(0, output.as_mut_ptr().add(start), self.count);
            output.set_len(start + self.count);
        }
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
            Bound::Unbounded => self.count,
        };
        assert!(
            begin <= end && end <= self.count,
            "DMA arguments outside accepted entry"
        );
        Self {
            // SAFETY: the subspan was bounded above, including the empty tail.
            address: HostAddress::from_pointer(unsafe { self.address.as_ptr().add(begin * 4) }),
            offset: self.offset + begin * 4,
            count: end - begin,
            ..self
        }
    }
}
