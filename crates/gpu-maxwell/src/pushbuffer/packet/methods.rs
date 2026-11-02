//! Bounded command words. The accepted entry owns storage and diagnostic context.
use super::*;
use std::ops::{Bound, RangeBounds};

#[derive(Clone, Copy)]
enum SpanData<'a> {
    Live {
        address: nixe_memory::HostAddress,
        stream: &'a sources::CommandStream,
        offset: u32,
        first: u32,
    },
    #[cfg(test)]
    Fixture {
        packet: &'a MaxwellDecodedMethodPacket,
        offset: usize,
    },
}
#[derive(Clone, Copy)]
pub struct MaxwellMethodSpan<'a> {
    data: SpanData<'a>,
    mode: MaxwellMethodPacketMode,
    count: u32,
}
impl<'a> MaxwellMethodSpan<'a> {
    #[cfg(test)]
    pub(super) fn new(packet: &'a MaxwellDecodedMethodPacket) -> Self {
        Self {
            data: SpanData::Fixture { packet, offset: 0 },
            mode: packet.mode,
            count: u32::try_from(packet.methods.len()).expect("hardware packet method bound"),
        }
    }
    pub fn len(self) -> usize {
        self.count as usize
    }
    pub fn is_empty(self) -> bool {
        self.count == 0
    }
    pub fn first(self) -> Option<MaxwellDecodedMethod> {
        self.get(0)
    }
    pub fn last(self) -> Option<MaxwellDecodedMethod> {
        self.len().checked_sub(1).and_then(|i| self.get(i))
    }
    #[inline]
    pub(crate) fn method_at(self, index: usize) -> GpuMethodId {
        debug_assert!(index < self.len());
        match self.data {
            SpanData::Live { first, .. } => method_id(method_dword(self.mode, first, index as u16)),
            #[cfg(test)]
            SpanData::Fixture { packet, offset } => packet.methods[offset + index].method,
        }
    }
    #[inline]
    pub(crate) fn argument_at(self, index: usize) -> u32 {
        debug_assert!(index < self.len());
        match self.data {
            // SAFETY: the span is bounded when the entry is parsed; its borrow
            // retains both the physical pin and any owned gathered storage.
            SpanData::Live { address, .. } => unsafe { address.read_u32(index * 4) },
            #[cfg(test)]
            SpanData::Fixture { packet, offset } => packet.methods[offset + index].argument,
        }
    }
    pub(crate) fn mode(self) -> MaxwellMethodPacketMode {
        self.mode
    }
    pub fn get(self, index: usize) -> Option<MaxwellDecodedMethod> {
        if index >= self.len() {
            return None;
        }
        let source = match self.data {
            SpanData::Live { stream, offset, .. } => {
                stream.location_at(offset as usize + index * 4)
            }
            #[cfg(test)]
            SpanData::Fixture { packet, offset } => packet.methods[offset + index].argument_source,
        };
        Some(MaxwellDecodedMethod {
            method: self.method_at(index),
            argument: self.argument_at(index),
            argument_source: source,
        })
    }
    pub(super) fn live(
        stream: &'a sources::CommandStream,
        address: nixe_memory::HostAddress,
        offset: usize,
        count: usize,
        first: u32,
        mode: MaxwellMethodPacketMode,
    ) -> Self {
        // Entry sizes and packet counts come from bounded hardware fields.
        debug_assert!(offset <= u32::MAX as usize && count <= u16::MAX as usize);
        Self {
            data: SpanData::Live {
                address,
                stream,
                offset: offset as u32,
                first,
            },
            mode,
            count: count as u32,
        }
    }
    pub(crate) fn ram_address(self) -> Option<nixe_memory::HostAddress> {
        match self.data {
            SpanData::Live { address, .. } => Some(address),
            #[cfg(test)]
            SpanData::Fixture { .. } => None,
        }
    }
    pub(crate) fn gathered_bytes(self) -> Option<&'a [u8]> {
        match self.data {
            SpanData::Live { stream, offset, .. } => {
                stream.gathered_bytes(offset as usize, offset as usize + self.len() * 4)
            }
            #[cfg(test)]
            SpanData::Fixture { .. } => None,
        }
    }
    pub(crate) fn arguments(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        (0..self.len()).map(move |index| self.argument_at(index))
    }
    pub(crate) fn append_arguments(
        self,
        output: &mut Vec<u32>,
    ) -> Result<(), std::collections::TryReserveError> {
        let count = self.len();
        output.try_reserve(count)?;
        match self.data {
            SpanData::Live { address, .. } => {
                let start = output.len();
                // SAFETY: entry admission bounds the complete span, its borrow
                // retains storage, and reserve established the disjoint output.
                unsafe {
                    address.copy_words_to(0, output.as_mut_ptr().add(start), count);
                    output.set_len(start + count);
                }
            }
            #[cfg(test)]
            SpanData::Fixture { .. } => output.extend(self.arguments()),
        }
        Ok(())
    }
    pub(crate) fn last_writes(mut self) -> impl Iterator<Item = MaxwellDecodedMethod> + 'a {
        std::iter::from_fn(move || {
            let method = self.last()?;
            self.count = match self.mode {
                MaxwellMethodPacketMode::Incrementing => self.count - 1,
                MaxwellMethodPacketMode::IncrementOnce if self.count > 1 => 1,
                _ => 0,
            };
            Some(method)
        })
    }
    pub fn slice(self, range: impl RangeBounds<usize>) -> Self {
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
        assert!(
            begin <= end && end <= self.len(),
            "method span outside packet"
        );
        // Normalize the cursor once when slicing, as DmaPusher advances its
        // method and data pointer. Argument access then needs no base-index
        // addition or repeated increment-once classification.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/dma_pusher.cpp
        let data = match self.data {
            SpanData::Live {
                address,
                stream,
                offset,
                first,
            } => SpanData::Live {
                // SAFETY: slicing was bounded above, including the empty tail.
                address: nixe_memory::HostAddress::from_pointer(unsafe {
                    address.as_ptr().add(begin * 4)
                }),
                stream,
                offset: offset + begin as u32 * 4,
                first: method_dword(self.mode, first, begin as u16),
            },
            #[cfg(test)]
            SpanData::Fixture { packet, offset } => SpanData::Fixture {
                packet,
                offset: offset + begin,
            },
        };
        Self {
            data,
            count: (end - begin) as u32,
            mode: if self.mode == MaxwellMethodPacketMode::IncrementOnce && begin != 0 {
                MaxwellMethodPacketMode::NonIncrementing
            } else {
                self.mode
            },
        }
    }
    pub fn iter(self) -> MaxwellMethodIterator<'a> {
        MaxwellMethodIterator {
            span: self,
            next: 0,
        }
    }
}
pub struct MaxwellMethodIterator<'a> {
    span: MaxwellMethodSpan<'a>,
    next: usize,
}
impl Iterator for MaxwellMethodIterator<'_> {
    type Item = MaxwellDecodedMethod;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let method = self.span.get(self.next)?;
        self.next += 1;
        Some(method)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.span.len() - self.next;
        (left, Some(left))
    }
}
impl DoubleEndedIterator for MaxwellMethodIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next == self.span.len() {
            return None;
        }
        let method = self.span.last();
        self.span.count -= 1;
        method
    }
}
impl ExactSizeIterator for MaxwellMethodIterator<'_> {}
impl<'a> IntoIterator for MaxwellMethodSpan<'a> {
    type Item = MaxwellDecodedMethod;
    type IntoIter = MaxwellMethodIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
