//! Retained input coordinates indexed at acceptance, never rebuilt per access.
use crate::{AllocationRange, Extent, MachineId, MemoryError};
use smallvec::SmallVec;

#[derive(Clone, Debug)]
struct Span {
    end: u64,
    range: AllocationRange,
}

/// An immutable ordered input. Each descriptor contains its logical end so
/// subrange operations can seek without traversing preceding mappings.
#[derive(Clone, Debug)]
pub struct CapturedMemory {
    spans: SmallVec<[Span; 1]>,
    pub(crate) machine: Option<MachineId>,
}
impl CapturedMemory {
    pub fn new(ranges: impl IntoIterator<Item = AllocationRange>) -> Result<Self, MemoryError> {
        let ranges = ranges.into_iter();
        let mut spans = SmallVec::with_capacity(ranges.size_hint().0);
        let mut machine = None;
        let mut end = 0u64;
        for range in ranges {
            let id = range.allocation().machine().identity();
            if machine.is_some_and(|owner| owner != id) {
                return Err(MemoryError::Invalid(
                    "captured input spans different machines",
                ));
            }
            machine = Some(id);
            end = end
                .checked_add(range.bytes())
                .ok_or(MemoryError::Invalid("captured input length overflows"))?;
            spans.push(Span { end, range });
        }
        Ok(Self { spans, machine })
    }
    pub fn bytes(&self) -> u64 {
        self.spans.last().map_or(0, |span| span.end)
    }
    pub fn len(&self) -> usize {
        self.spans.len()
    }
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
    #[inline]
    fn selected(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = (&AllocationRange, u64, u64)>, MemoryError> {
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= self.bytes())
            .ok_or(MemoryError::Invalid(
                "physical access exceeds accepted input",
            ))?;
        let first = self.spans.partition_point(|span| span.end <= offset);
        Ok(self.spans[first..]
            .iter()
            .take_while(move |span| span.end - span.range.bytes() < end && bytes != 0)
            .map(move |span| {
                let begin = span.end - span.range.bytes();
                let low = offset.max(begin);
                (&span.range, low - begin, span.end.min(end) - low)
            }))
    }
    #[inline]
    pub(crate) fn physical_extents(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<impl Iterator<Item = Extent>, MemoryError> {
        Ok(self
            .selected(offset, bytes)?
            .flat_map(|(range, offset, bytes)| {
                range
                    .allocation()
                    .physical_extents(range.offset() + offset, bytes)
            }))
    }
    /// Independent exact leases for an operation that escapes the input owner.
    pub fn capture(&self, offset: u64, bytes: u64) -> Result<Vec<AllocationRange>, MemoryError> {
        self.selected(offset, bytes)?
            .map(|(range, offset, bytes)| range.slice(offset, bytes))
            .collect()
    }
    pub(crate) fn views(
        &self,
        offset: u64,
        bytes: u64,
    ) -> Result<Vec<AllocationRange>, MemoryError> {
        self.selected(offset, bytes)?
            .map(|(range, offset, bytes)| {
                AllocationRange::new(range.allocation().clone(), range.offset() + offset, bytes)
            })
            .collect()
    }
}
impl std::ops::Index<usize> for CapturedMemory {
    type Output = AllocationRange;
    fn index(&self, index: usize) -> &Self::Output {
        &self.spans[index].range
    }
}
impl From<AllocationRange> for CapturedMemory {
    fn from(range: AllocationRange) -> Self {
        Self {
            machine: Some(range.allocation().machine().identity()),
            spans: smallvec::smallvec![Span {
                end: range.bytes(),
                range
            }],
        }
    }
}

#[cfg(test)]
mod tests;
