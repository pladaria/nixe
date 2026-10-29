//! Bounded command words. The accepted entry owns storage and diagnostic context.
use super::*;
use std::ops::{Bound, RangeBounds};

#[derive(Clone, Copy)]
enum SpanData<'a> {
    Live {
        address: nixe_memory::HostAddress,
        stream: &'a sources::CommandStream,
        offset: usize,
        first: u32,
    },
    #[cfg(test)]
    Fixture(&'a MaxwellDecodedMethodPacket),
}
#[derive(Clone, Copy)]
pub struct MaxwellMethodSpan<'a> {
    data: SpanData<'a>,
    mode: MaxwellMethodPacketMode,
    begin: usize,
    end: usize,
}
impl<'a> MaxwellMethodSpan<'a> {
    #[cfg(test)]
    pub(super) fn new(packet: &'a MaxwellDecodedMethodPacket) -> Self {
        let end = packet.methods.len();
        Self {
            data: SpanData::Fixture(packet),
            mode: packet.mode,
            begin: 0,
            end,
        }
    }
    pub fn len(self) -> usize {
        self.end - self.begin
    }
    pub fn is_empty(self) -> bool {
        self.begin == self.end
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
        let index = self.begin + index;
        match self.data {
            SpanData::Live { first, .. } => method_id(method_dword(self.mode, first, index as u16)),
            #[cfg(test)]
            SpanData::Fixture(packet) => packet.methods[index].method,
        }
    }
    #[inline]
    pub(crate) fn argument_at(self, index: usize) -> u32 {
        debug_assert!(index < self.len());
        let index = self.begin + index;
        match self.data {
            // SAFETY: the span is bounded when the entry is parsed; its borrow
            // retains both the physical pin and any owned gathered storage.
            SpanData::Live { address, .. } => unsafe { address.read_u32(index * 4) },
            #[cfg(test)]
            SpanData::Fixture(packet) => packet.methods[index].argument,
        }
    }
    #[cfg(test)]
    pub(crate) fn mode(self) -> MaxwellMethodPacketMode {
        if self.mode == MaxwellMethodPacketMode::IncrementOnce && self.begin > 0 {
            MaxwellMethodPacketMode::NonIncrementing
        } else {
            self.mode
        }
    }
    pub fn get(self, index: usize) -> Option<MaxwellDecodedMethod> {
        if index >= self.len() {
            return None;
        }
        let source = match self.data {
            SpanData::Live { stream, offset, .. } => {
                stream.location_at(offset + (self.begin + index) * 4)
            }
            #[cfg(test)]
            SpanData::Fixture(packet) => packet.methods[self.begin + index].argument_source,
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
        Self {
            data: SpanData::Live {
                address,
                stream,
                offset,
                first,
            },
            mode,
            begin: 0,
            end: count,
        }
    }
    pub(crate) fn ram_bytes(self) -> Option<(nixe_memory::HostAddress, usize)> {
        match self.data {
            SpanData::Live { address, .. } => Some((address, self.begin * 4)),
            #[cfg(test)]
            SpanData::Fixture(_) => None,
        }
    }
    pub(crate) fn gathered_bytes(self) -> Option<&'a [u8]> {
        match self.data {
            SpanData::Live { stream, offset, .. } => {
                stream.gathered_bytes(offset + self.begin * 4, offset + self.end * 4)
            }
            #[cfg(test)]
            SpanData::Fixture(_) => None,
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
                    address.copy_words_to(self.begin * 4, output.as_mut_ptr().add(start), count);
                    output.set_len(start + count);
                }
            }
            #[cfg(test)]
            SpanData::Fixture(_) => output.extend(self.arguments()),
        }
        Ok(())
    }
    pub(crate) fn last_writes(mut self) -> impl Iterator<Item = MaxwellDecodedMethod> + 'a {
        std::iter::from_fn(move || {
            let method = self.last()?;
            self.end = match self.mode {
                MaxwellMethodPacketMode::Incrementing => self.end - 1,
                MaxwellMethodPacketMode::IncrementOnce if self.begin == 0 && self.end > 1 => 1,
                _ => self.begin,
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
        Self {
            begin: self.begin + begin,
            end: self.begin + end,
            ..self
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
        self.span.end -= 1;
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
