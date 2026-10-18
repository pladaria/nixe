//! Compact arguments with provenance recorded only when its source run changes.
use super::*;
use std::ops::{Bound, RangeBounds};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SourceRun {
    pub index: u16,
    pub location: MaxwellGpfifoSourceLocation,
}
#[derive(Default)]
pub(crate) struct PacketScratch {
    pub(super) arguments: Vec<u32>,
    pub(super) sources: Vec<SourceRun>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum PacketMethods {
    Empty,
    One(MaxwellDecodedMethod),
    Many {
        first_method: u32,
        arguments: Vec<u32>,
        first_source: MaxwellGpfifoSourceLocation,
        sources: Vec<SourceRun>,
    },
}
#[derive(Clone, Copy)]
pub struct MaxwellMethodSpan<'a> {
    packet: &'a MaxwellDecodedMethodPacket,
    begin: usize,
    end: usize,
}
impl<'a> MaxwellMethodSpan<'a> {
    pub(super) fn new(packet: &'a MaxwellDecodedMethodPacket) -> Self {
        let end = match &packet.methods {
            PacketMethods::Empty => 0,
            PacketMethods::One(_) => 1,
            PacketMethods::Many { arguments, .. } => arguments.len(),
        };
        Self {
            packet,
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
    pub fn get(self, index: usize) -> Option<MaxwellDecodedMethod> {
        if index >= self.len() {
            return None;
        }
        let index = self.begin + index;
        Some(match &self.packet.methods {
            PacketMethods::Empty => unreachable!("empty method span"),
            PacketMethods::One(method) => *method,
            PacketMethods::Many {
                first_method,
                arguments,
                first_source,
                sources,
            } => {
                let run = sources.partition_point(|run| usize::from(run.index) <= index);
                let (base, mut location) = if run == 0 {
                    (0, *first_source)
                } else {
                    (
                        usize::from(sources[run - 1].index),
                        sources[run - 1].location,
                    )
                };
                location.word_offset += (index - base) as u64;
                MaxwellDecodedMethod {
                    method: method_id(method_dword(self.packet.mode, *first_method, index as u16)),
                    argument: arguments[index],
                    argument_source: location,
                }
            }
        })
    }
    /// Raw packet words remain a block through upload/index consumers.
    pub(crate) fn arguments(self) -> &'a [u32] {
        match &self.packet.methods {
            PacketMethods::Empty => &[],
            PacketMethods::One(method) => {
                &std::slice::from_ref(&method.argument)[self.begin..self.end]
            }
            PacketMethods::Many { arguments, .. } => &arguments[self.begin..self.end],
        }
    }
    /// Last write of each distinct register, in reverse execution order.
    pub(crate) fn last_writes(mut self) -> impl Iterator<Item = MaxwellDecodedMethod> + 'a {
        std::iter::from_fn(move || {
            let method = self.last()?;
            self.end = match self.packet.mode {
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
            packet: self.packet,
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
