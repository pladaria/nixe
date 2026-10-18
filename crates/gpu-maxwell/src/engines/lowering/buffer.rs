//! Buffer views shared by graphics and compute; content ownership stays canonical.
use super::*;
use crate::MaxwellResolvedRange;
use nixe_gpu::{BackingView, GpuAllocationDescription, GpuAllocationId};
use nixe_memory::CanonicalBackingRange;

pub(super) struct RetainedBuffer {
    pub backing: BackingView,
}

pub(super) fn retain(range: &MaxwellResolvedRange) -> Result<RetainedBuffer, MaxwellLoweringError> {
    let fail = |reason: &str| MaxwellLoweringError::BufferBacking(reason.to_owned());
    let first = range
        .segments()
        .first()
        .ok_or_else(|| fail("empty range"))?;
    let allocation = first.mapping().allocation();
    let allocation_size = first.mapping().backing().size();
    let offset = first.backing_offset();
    let mut end = offset;
    let mut segments = Vec::new();
    for segment in range.segments() {
        if segment.mapping().allocation() != allocation
            || segment.backing_offset() != end
            || segment.mapping().backing().size() != allocation_size
        {
            return Err(fail("discontiguous allocation"));
        }
        segment
            .mapping()
            .backing()
            .snapshot_subrange_into(segment.backing_offset(), segment.size(), &mut segments)
            .map_err(|error| fail(&error.to_string()))?;
        end = end
            .checked_add(segment.size())
            .ok_or_else(|| fail("range overflow"))?;
    }
    let canonical = CanonicalBackingRange::new(segments).map_err(|e| fail(&e.to_string()))?;
    let allocation_description =
        GpuAllocationDescription::new(allocation_size, 1).map_err(|e| fail(&e.to_string()))?;
    let backing = BackingView::new(
        GpuAllocationId::new(allocation.get()),
        allocation_description,
        offset,
        canonical,
    )
    .map_err(|e| fail(&e.to_string()))?;
    Ok(RetainedBuffer { backing })
}
