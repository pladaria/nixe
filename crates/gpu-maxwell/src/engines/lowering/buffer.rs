//! Buffer views shared by graphics and compute; content ownership stays canonical.
use super::*;
use crate::{MaxwellResolvedRange, MaxwellThreeDMappingReference};
use nixe_gpu::{BackingView, BufferDescription, GpuAllocationDescription, GpuAllocationId};
use nixe_memory::CanonicalBackingRange;

pub(super) struct RetainedBuffer {
    pub description: BufferDescription,
    pub allocation_description: GpuAllocationDescription,
    pub backing: BackingView,
    pub mappings: Arc<[MaxwellThreeDMappingReference]>,
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
    let mut mappings = Vec::new();
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
        mappings.push(MaxwellThreeDMappingReference::from_segment(segment));
    }
    let canonical = CanonicalBackingRange::new(segments).map_err(|e| fail(&e.to_string()))?;
    let description = BufferDescription::new(end - offset).map_err(|e| fail(&e.to_string()))?;
    let allocation_description =
        GpuAllocationDescription::new(allocation_size, 1).map_err(|e| fail(&e.to_string()))?;
    let backing = BackingView::new(
        GpuAllocationId::new(allocation.get()),
        allocation_description,
        offset,
        canonical,
    )
    .map_err(|e| fail(&e.to_string()))?;
    Ok(RetainedBuffer {
        description,
        allocation_description,
        backing,
        mappings: mappings.into(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_buffer(
    description: BufferDescription,
    allocation_description: GpuAllocationDescription,
    backing: BackingView,
    mappings: Arc<[MaxwellThreeDMappingReference]>,
    cache: &mut MaxwellLoweringCache,
    creations: &mut Vec<BackendResourceCreateInfo>,
    invalidations: &mut Vec<ResourceDependency>,
    retained: impl Iterator<Item = ResourceDependency> + Clone,
) -> Result<ResourceDependency, MaxwellLoweringError> {
    let key = ViewKey::Buffer {
        description,
        buffer_offset: 0,
        backing: backing.clone(),
        mappings,
    };
    if let Some(record) = cache.views.iter().find(|record| match &record.key {
        ViewKey::Buffer {
            description: current,
            buffer_offset,
            backing: old,
            mappings: old_mappings,
        } => {
            *current == description
                && *buffer_offset == 0
                && same_canonical_backing(old, &backing)
                && matches!(&key, ViewKey::Buffer { mappings, .. } if mappings == old_mappings)
        }
        _ => false,
    }) {
        return Ok(record.dependency);
    }
    let allocation = backing.allocation();
    match cache.allocations.iter().find(|(id, _)| *id == allocation) {
        Some((_, current)) if *current != allocation_description => {
            return Err(MaxwellLoweringError::AllocationDescriptionChanged { allocation });
        }
        Some(_) => {}
        None => {
            cache.allocations.push((allocation, allocation_description));
            creations.push(BackendResourceCreateInfo::Allocation {
                id: allocation,
                description: allocation_description,
            });
        }
    }
    retire_overlapping_views(&key, retained, cache, invalidations);
    let id = BufferId::new(take_identity(cache)?);
    let view = BufferView::new(id, description, 0, backing)
        .map_err(|e| MaxwellLoweringError::BufferBacking(e.to_string()))?;
    creations.push(BackendResourceCreateInfo::Buffer {
        id,
        description,
        view: Some(view),
    });
    let dependency = ResourceDependency::Buffer(id);
    cache.views.push(ViewRecord {
        key,
        dependency,
        materialization: ViewMaterialization::Direct,
        cpu_writes: None,
        write_revision: 0,
        last_used: 0,
        uninitialized_color_regions: Vec::new(),
        uninitialized_depth_stencil_regions: Default::default(),
    });
    Ok(dependency)
}
