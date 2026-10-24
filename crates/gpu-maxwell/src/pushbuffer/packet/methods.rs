//! Compact arguments with provenance recorded only when its source run changes.
use super::*;
use std::ops::{Bound, RangeBounds};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum PacketMethods {
    Empty,
    One(MaxwellDecodedMethod),
    #[cfg(test)]
    Fixture(Box<[MaxwellDecodedMethod]>),
}
#[derive(Clone, Copy)]
enum SpanData<'a> {
    Packet(&'a MaxwellDecodedMethodPacket),
    Live {
        words: &'a sources::EntryWords,
        first_method: u32,
        source: MaxwellGpfifoSourceLocation,
        offset: usize,
    },
}
#[derive(Clone, Copy)]
pub struct MaxwellMethodSpan<'a> {
    data: SpanData<'a>,
    mode: MaxwellMethodPacketMode,
    begin: usize,
    end: usize,
}
impl<'a> MaxwellMethodSpan<'a> {
    pub(super) fn new(packet: &'a MaxwellDecodedMethodPacket) -> Self {
        let end = match &packet.methods {
            PacketMethods::Empty => 0,
            PacketMethods::One(_) => 1,
            #[cfg(test)]
            PacketMethods::Fixture(methods) => methods.len(),
        };
        Self {
            data: SpanData::Packet(packet),
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
    #[inline(always)]
    pub fn get(self, index: usize) -> Option<MaxwellDecodedMethod> {
        if index >= self.len() {
            return None;
        }
        let index = self.begin + index;
        let SpanData::Packet(packet) = self.data else {
            let SpanData::Live {
                words,
                first_method,
                mut source,
                offset,
            } = self.data
            else {
                unreachable!()
            };
            source.word_offset += index as u64;
            return Some(MaxwellDecodedMethod {
                method: method_id(method_dword(self.mode, first_method, index as u16)),
                argument: words.read(offset + index * 4),
                argument_source: source,
            });
        };
        Some(match &packet.methods {
            PacketMethods::Empty => unreachable!("empty method span"),
            PacketMethods::One(method) => *method,
            #[cfg(test)]
            PacketMethods::Fixture(methods) => methods[index],
        })
    }
    pub(crate) fn live(
        words: &'a sources::EntryWords,
        offset: usize,
        count: usize,
        first_method: u32,
        mode: MaxwellMethodPacketMode,
        source: MaxwellGpfifoSourceLocation,
    ) -> Self {
        Self {
            data: SpanData::Live {
                words,
                first_method,
                source,
                offset,
            },
            mode,
            begin: 0,
            end: count,
        }
    }
    pub(crate) fn ram_bytes(self) -> Option<(nixe_memory::HostAddress, usize)> {
        match self.data {
            SpanData::Live {
                words: sources::EntryWords::Ram { address, .. },
                offset,
                ..
            } => Some((*address, offset + self.begin * 4)),
            _ => None,
        }
    }
    pub(crate) fn gathered_bytes(self) -> Option<&'a [u8]> {
        match self.data {
            SpanData::Live {
                words: sources::EntryWords::Gathered(bytes),
                offset,
                ..
            } => Some(&bytes[offset + self.begin * 4..offset + self.end * 4]),
            _ => None,
        }
    }
    pub(crate) fn arguments(self) -> impl ExactSizeIterator<Item = u32> + 'a {
        (self.begin..self.end).map(move |i| match self.data {
            SpanData::Live { words, offset, .. } => words.read(offset + i * 4),
            SpanData::Packet(packet) => match &packet.methods {
                PacketMethods::Empty => unreachable!(),
                PacketMethods::One(method) => method.argument,
                #[cfg(test)]
                PacketMethods::Fixture(methods) => methods[i].argument,
            },
        })
    }
    /// Last write of each distinct register, in reverse execution order.
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
            data: self.data,
            mode: self.mode,
            begin: self.begin + begin,
            end: self.begin + end,
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
