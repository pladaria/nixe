//! Retained canonical identity and command-order lifetime of resource versions.
use std::collections::BTreeMap;

use nixe_gpu::{AccessTarget, BackingView, BufferRange, BufferRegion, ResourceDependency};

use super::{MaxwellResolvedRange, MaxwellSubmissionExecutionStep};

#[derive(Clone)]
pub(crate) struct ResidentResource {
    pub dependency: ResourceDependency,
    pub backings: Box<[BackingView]>,
    pub buffer_offset: Option<u64>,
}

struct Version {
    resource: ResidentResource,
    created: usize,
    retired: usize,
    aliases: Vec<usize>,
}

pub(super) struct Residency {
    versions: Vec<Version>,
    identities: BTreeMap<ResourceDependency, usize>,
    buffers: BTreeMap<u64, Vec<usize>>,
}

impl Residency {
    pub fn new(
        initial: Vec<ResidentResource>,
        steps: &mut [MaxwellSubmissionExecutionStep],
    ) -> Self {
        let mut result = Self {
            versions: Vec::new(),
            identities: BTreeMap::new(),
            buffers: BTreeMap::new(),
        };
        for resource in initial {
            result.insert(resource, 0);
        }
        for (index, step) in steps.iter_mut().enumerate() {
            if let MaxwellSubmissionExecutionStep::Gpu(work) = step {
                for resource in std::mem::take(&mut work.resident_resources).into_vec() {
                    result.insert(resource, index);
                }
                for dependency in work.resource_invalidations() {
                    if let Some(&version) = result.identities.get(dependency) {
                        result.versions[version].retired = index;
                    }
                }
            }
        }
        // Sweep compressed physical intervals. Disjoint resource versions do
        // not require pairwise comparisons of their retained page topology.
        let mut spans = Vec::new();
        for (version, record) in result.versions.iter().enumerate() {
            for backing in &record.resource.backings {
                spans.extend(
                    backing
                        .canonical_spans()
                        .iter()
                        .copied()
                        .map(|span| (version, span)),
                );
            }
        }
        spans.sort_unstable_by_key(|(_, span)| (span.first_page(), span.first_offset()));
        let mut active: Vec<(usize, nixe_gpu::CanonicalBackingSpan)> = Vec::new();
        for (b, span) in spans {
            active.retain(|(_, previous)| {
                !previous.ends_before(span.first_page(), span.first_offset())
            });
            for &(a, previous) in &active {
                if a != b && previous.overlaps(span) && !result.versions[a].aliases.contains(&b) {
                    result.versions[a].aliases.push(b);
                    result.versions[b].aliases.push(a);
                }
            }
            active.push((b, span));
        }
        result
    }

    fn insert(&mut self, resource: ResidentResource, created: usize) {
        let index = self.versions.len();
        assert!(
            self.identities.insert(resource.dependency, index).is_none(),
            "resource identities are unique"
        );
        if resource.buffer_offset.is_some()
            && let [backing] = resource.backings.as_ref()
        {
            self.buffers
                .entry(backing.allocation().get())
                .or_default()
                .push(index);
        }
        self.versions.push(Version {
            resource,
            created,
            retired: usize::MAX,
            aliases: Vec::new(),
        });
    }

    pub fn is_live(&self, dependency: ResourceDependency, index: usize) -> bool {
        self.identities.get(&dependency).is_some_and(|&v| {
            let version = &self.versions[v];
            version.created <= index && index < version.retired
        })
    }

    pub fn buffer(
        &self,
        target: &MaxwellResolvedRange,
        index: usize,
        hint: &mut Option<usize>,
    ) -> Option<BufferRegion> {
        let [segment] = target.segments() else {
            return None;
        };
        let region = |version: &Version| {
            if version.created > index || index >= version.retired {
                return None;
            }
            let ResourceDependency::Buffer(buffer) = version.resource.dependency else {
                return None;
            };
            let [backing] = version.resource.backings.as_ref() else {
                return None;
            };
            let relative = segment
                .backing_offset()
                .checked_sub(backing.allocation_offset())?;
            if segment.mapping().allocation().get() != backing.allocation().get()
                || relative.checked_add(target.size())? > backing.size()
            {
                return None;
            }
            Some(BufferRegion {
                buffer,
                range: BufferRange::new(
                    version.resource.buffer_offset?.checked_add(relative)?,
                    target.size(),
                )
                .ok()?,
            })
        };
        if let Some(v) = *hint
            && let Some(region) = region(&self.versions[v])
        {
            return Some(region);
        }
        let (v, region) = self
            .buffers
            .get(&segment.mapping().allocation().get())?
            .iter()
            .copied()
            .find_map(|v| region(&self.versions[v]).map(|region| (v, region)))?;
        *hint = Some(v);
        Some(region)
    }

    pub fn inline_buffer(
        &self,
        target: &MaxwellResolvedRange,
        index: usize,
        hint: &mut Option<usize>,
    ) -> Option<BufferRegion> {
        // Host resource definitions can be installed before command execution.
        // A future read-only version can receive its preceding inline writes
        // provided no other version consumes those canonical bytes meanwhile.
        let [segment] = target.segments() else {
            return None;
        };
        let candidate = |v: usize| {
            let version = &self.versions[v];
            if version.created <= index || version.retired <= index {
                return None;
            }
            let ResourceDependency::Buffer(buffer) = version.resource.dependency else {
                return None;
            };
            let [backing] = version.resource.backings.as_ref() else {
                return None;
            };
            let relative = segment
                .backing_offset()
                .checked_sub(backing.allocation_offset())?;
            if segment.mapping().allocation().get() != backing.allocation().get()
                || relative.checked_add(target.size())? > backing.size()
            {
                return None;
            }
            Some(BufferRegion {
                buffer,
                range: BufferRange::new(
                    version.resource.buffer_offset?.checked_add(relative)?,
                    target.size(),
                )
                .ok()?,
            })
        };
        if let Some(v) = *hint
            && let Some(region) = candidate(v)
        {
            return Some(region);
        }
        if let Some(region) = self.buffer(target, index, hint) {
            return Some(region);
        }
        let (v, region) = self
            .buffers
            .get(&segment.mapping().allocation().get())?
            .iter()
            .copied()
            .find_map(|v| candidate(v).map(|r| (v, r)))?;
        *hint = Some(v);
        Some(region)
    }

    pub fn has_later_alias(
        &self,
        dependency: ResourceDependency,
        index: usize,
        target: &MaxwellResolvedRange,
    ) -> bool {
        let Some(&version) = self.identities.get(&dependency) else {
            return true;
        };
        self.versions[version].aliases.iter().any(|&other| {
            let other = &self.versions[other];
            other.retired > index
                && other
                    .resource
                    .backings
                    .iter()
                    .any(|backing| overlaps_target(backing, target, false))
        })
    }

    pub fn disjoint(&self, dependency: ResourceDependency, target: &MaxwellResolvedRange) -> bool {
        if !matches!(
            dependency,
            ResourceDependency::Buffer(_) | ResourceDependency::Image(_)
        ) {
            return true;
        }
        self.identities.get(&dependency).is_some_and(|&v| {
            self.versions[v]
                .resource
                .backings
                .iter()
                .all(|backing| !overlaps_target(backing, target, true))
        })
    }

    pub fn written_pages(
        &self,
        target: AccessTarget,
        pages: &mut std::collections::BTreeSet<nixe_memory::CanonicalPageId>,
    ) -> bool {
        let dependency = target.dependency();
        let Some(&v) = self.identities.get(&dependency) else {
            return !matches!(
                dependency,
                ResourceDependency::Buffer(_) | ResourceDependency::Image(_)
            );
        };
        let resource = &self.versions[v].resource;
        for backing in &resource.backings {
            let (start, end) = match target {
                AccessTarget::Buffer { range, .. } => {
                    let offset = resource.buffer_offset.unwrap_or(0);
                    (
                        range.offset().saturating_sub(offset),
                        (range.offset() + range.size())
                            .saturating_sub(offset)
                            .min(backing.size()),
                    )
                }
                _ => (0, backing.size()),
            };
            let mut base = 0;
            for segment in backing.range().segments() {
                if start < base + segment.size() && base < end {
                    pages.insert(segment.page());
                }
                base += segment.size();
                if base >= end {
                    break;
                }
            }
        }
        true
    }

    pub fn cpu_conflicts(
        target: &MaxwellResolvedRange,
        pages: &std::collections::BTreeSet<nixe_memory::CanonicalPageId>,
    ) -> bool {
        !pages.is_empty()
            && target.segments().iter().any(|mapping| {
                mapping
                    .mapping()
                    .backing()
                    .subrange_segments(mapping.backing_offset(), mapping.size())
                    .expect("resolved subrange")
                    .any(|(segment, _, _)| pages.contains(&segment.page()))
            })
    }
}

// CPU visibility is page-granular; transfer aliasing is byte-granular. Walk
// only the resolved subrange, never the surrounding allocation's other pages.
fn overlaps_target(backing: &BackingView, target: &MaxwellResolvedRange, pages: bool) -> bool {
    target.segments().iter().any(|mapping| {
        mapping
            .mapping()
            .backing()
            .subrange_segments(mapping.backing_offset(), mapping.size())
            .expect("resolved subrange")
            .any(|(segment, from, size)| {
                backing.canonical_spans().iter().any(|span| {
                    let page = segment.page();
                    let start = if page == span.first_page() {
                        span.first_offset()
                    } else {
                        0
                    };
                    let end = if page == span.last_page() {
                        span.last_end()
                    } else {
                        u64::MAX
                    };
                    span.first_page() <= page
                        && page <= span.last_page()
                        && (pages || (start < from + size && from < end))
                })
            })
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAllocationId, MaxwellGpuAddressSpace, MaxwellMapRequest,
        SWITCH_1_GM20B_PROFILE,
    };
    use nixe_gpu::{BufferId, GpuAllocationDescription, GpuAllocationId};
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    pub(in crate::execution) fn fixture() -> (MaxwellGpuAddressSpace, u64, ResidentResource) {
        let allocation = CanonicalAllocation::zeroed(0x3000, 0x1000).unwrap();
        let canonical = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        space
            .initialize(crate::MaxwellAddressSpaceInitialization {
                big_page_size: 0x20000,
                ..Default::default()
            })
            .unwrap();
        let mapping = space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: canonical.clone(),
                backing_offset: 0,
                size: 0x3000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let resource = ResidentResource {
            dependency: ResourceDependency::Buffer(BufferId::new(1)),
            buffer_offset: Some(0),
            backings: vec![
                BackingView::new(
                    GpuAllocationId::new(1),
                    GpuAllocationDescription::new(0x3000, 4).unwrap(),
                    0,
                    canonical,
                )
                .unwrap(),
            ]
            .into(),
        };
        (space, mapping.offset().get(), resource)
    }

    #[test]
    fn versions_exist_only_between_creation_and_retirement() {
        let (space, address, resource) = fixture();
        let dependency = resource.dependency;
        let target = space
            .resolve_range(space.address(address).unwrap(), 4, MemoryPermissions::WRITE)
            .unwrap();
        let mut residency = Residency::new(vec![], &mut []);
        residency.insert(resource, 5);
        residency.versions[0].retired = 10;
        let mut hint = None;
        assert!(residency.buffer(&target, 4, &mut hint).is_none());
        assert!(residency.inline_buffer(&target, 4, &mut hint).is_some());
        assert!(residency.buffer(&target, 5, &mut hint).is_some());
        assert!(residency.is_live(dependency, 9));
        assert!(!residency.is_live(dependency, 10));
        assert!(residency.buffer(&target, 10, &mut hint).is_none());
        assert!(residency.inline_buffer(&target, 10, &mut hint).is_none());
    }

    #[test]
    fn physical_aliases_block_only_their_covered_bytes_and_lifetime() {
        let (space, address, resource) = fixture();
        let backing = &resource.backings[0];
        let alias = ResidentResource {
            dependency: ResourceDependency::Buffer(BufferId::new(2)),
            buffer_offset: Some(0),
            backings: vec![
                BackingView::new(
                    GpuAllocationId::new(99),
                    GpuAllocationDescription::new(0x1000, 4).unwrap(),
                    0,
                    backing.range().snapshot_subrange(0x1000, 0x1000).unwrap(),
                )
                .unwrap(),
            ]
            .into(),
        };
        let mut residency = Residency::new(vec![resource, alias], &mut []);
        let resolve = |offset| {
            space
                .resolve_range(
                    space.address(address + offset).unwrap(),
                    4,
                    MemoryPermissions::WRITE,
                )
                .unwrap()
        };
        let first = resolve(0);
        let second = resolve(0x1000);
        let dependency = ResourceDependency::Buffer(BufferId::new(1));
        assert!(!residency.has_later_alias(dependency, 0, &first));
        assert!(residency.has_later_alias(dependency, 0, &second));
        assert!(residency.disjoint(ResourceDependency::Buffer(BufferId::new(2)), &first));
        residency.versions[1].retired = 4;
        assert!(!residency.has_later_alias(dependency, 4, &second));
    }

    #[test]
    fn partial_gpu_writes_conflict_at_page_granularity_and_include_aliases() {
        let (space, address, resource) = fixture();
        let residency = Residency::new(vec![resource], &mut []);
        let mut pages = std::collections::BTreeSet::new();
        assert!(residency.written_pages(
            AccessTarget::Buffer {
                buffer: BufferId::new(1),
                range: BufferRange::new(0x1004, 4).unwrap()
            },
            &mut pages
        ));
        assert_eq!(pages.len(), 1);
        let resolve = |offset, size| {
            space
                .resolve_range(
                    space.address(address + offset).unwrap(),
                    size,
                    MemoryPermissions::WRITE,
                )
                .unwrap()
        };
        assert!(!Residency::cpu_conflicts(&resolve(0, 4), &pages));
        assert!(Residency::cpu_conflicts(&resolve(0x1000, 4), &pages));
        assert!(Residency::cpu_conflicts(&resolve(0xffe, 4), &pages));
        assert!(!Residency::cpu_conflicts(&resolve(0x2000, 4), &pages));
    }
}
