//! Buffer views shared by graphics and compute; content ownership stays canonical.
use super::*;
use crate::MaxwellResolvedRange;
use nixe_gpu::physical::PhysicalMapping;
use nixe_gpu::{BackingView, GpuAllocationDescription, GpuAllocationId};

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
    let allocation_size = first.mapping().physical().bytes();
    let offset = first.backing_offset();
    let mut end = offset;
    let mut segments = Vec::new();
    for segment in range.segments() {
        if segment.mapping().allocation() != allocation
            || segment.backing_offset() != end
            || segment.mapping().physical().bytes() != allocation_size
        {
            return Err(fail("discontiguous allocation"));
        }
        segments.push(
            segment
                .mapping()
                .physical()
                .slice(segment.backing_offset(), segment.size())
                .map_err(|error| fail(&error.to_string()))?,
        );
        end = end
            .checked_add(segment.size())
            .ok_or_else(|| fail("range overflow"))?;
    }
    let physical = PhysicalMapping::join(&segments).map_err(|e| fail(&e.to_string()))?;
    let allocation_description =
        GpuAllocationDescription::new(allocation_size, 1).map_err(|e| fail(&e.to_string()))?;
    let backing = BackingView::new(
        GpuAllocationId::new(allocation.get()),
        allocation_description,
        offset,
        physical,
    )
    .map_err(|e| fail(&e.to_string()))?;
    Ok(RetainedBuffer { backing })
}
