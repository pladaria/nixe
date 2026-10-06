//! Immutable dependency indexing and exact accesses shared by submission consumers.
use crate::{AccessMode, AccessTarget, GpuOperation, PipelineStages, ResourceDependency};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedAccess {
    target: AccessTarget,
    mode: AccessMode,
    stages: PipelineStages,
    dependency_index: usize,
    first_operation: usize,
}

impl PlannedAccess {
    #[must_use]
    pub const fn first_operation(self) -> usize {
        self.first_operation
    }
    #[must_use]
    pub const fn target(self) -> AccessTarget {
        self.target
    }
    #[must_use]
    pub const fn mode(self) -> AccessMode {
        self.mode
    }
    #[must_use]
    pub const fn stages(self) -> PipelineStages {
        self.stages
    }
    #[must_use]
    pub const fn dependency_index(self) -> usize {
        self.dependency_index
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubmissionAccessPlan {
    dependencies: Box<[ResourceDependency]>,
    accesses: Box<[PlannedAccess]>,
    operation_dependencies: Box<[Box<[usize]>]>,
}

impl SubmissionAccessPlan {
    pub(crate) fn compile(operations: &[GpuOperation]) -> Self {
        let dependencies = operations
            .iter()
            .flat_map(|operation| operation.dependencies().iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Box<[_]>>();
        // Sweep endpoints within each resource/aspect. The work is O(A log A),
        // including partially overlapping intervals; it never widens a read
        // into a neighboring write-only interval or a different mip/plane.
        let mut groups: BTreeMap<(usize, u8, u8), Vec<AccessInterval>> = BTreeMap::new();
        let mut accesses = Vec::new();
        for (operation_index, operation) in operations.iter().enumerate() {
            for access in operation.accesses() {
                let dependency_index = dependencies
                    .binary_search(&access.target().dependency())
                    .expect("operation access retains its dependency");
                let (start, end, plane, mip) = match access.target() {
                    AccessTarget::Buffer { range, .. } => (range.offset(), range.end(), 0, 0),
                    AccessTarget::Queries { range, .. } => (
                        u64::from(range.first()),
                        u64::from(range.first()) + u64::from(range.count()),
                        0,
                        0,
                    ),
                    AccessTarget::Image { subresources, .. } => {
                        if subresources.layer_count == 0
                            || subresources
                                .base_layer
                                .checked_add(subresources.layer_count)
                                .is_none()
                        {
                            // Keep malformed declarations intact for the backend's
                            // precise validation error, rather than hiding them.
                            accesses.push(PlannedAccess {
                                target: access.target(),
                                mode: access.scope().mode(),
                                stages: access.scope().stages(),
                                dependency_index,
                                first_operation: operation_index,
                            });
                            continue;
                        }
                        (
                            u64::from(subresources.base_layer),
                            u64::from(subresources.base_layer)
                                + u64::from(subresources.layer_count),
                            subresources.plane,
                            subresources.mip_level,
                        )
                    }
                };
                groups
                    .entry((dependency_index, plane, mip))
                    .or_default()
                    .push(AccessInterval {
                        start,
                        end,
                        mode: access.scope().mode(),
                        stages: access.scope().stages(),
                        operation: operation_index,
                    });
            }
        }
        for ((dependency_index, plane, mip), intervals) in groups {
            let mut events = Vec::with_capacity(intervals.len() * 2);
            for (index, interval) in intervals.iter().enumerate() {
                events.push((interval.start, index, true));
                events.push((interval.end, index, false));
            }
            events.sort_unstable();
            let mut active = ActiveAccesses::default();
            let mut previous = events[0].0;
            let mut cursor = 0;
            while cursor < events.len() {
                let position = events[cursor].0;
                if previous < position
                    && let Some(first_operation) = active.first_operation()
                {
                    let target = match dependencies[dependency_index] {
                        ResourceDependency::Buffer(buffer) => AccessTarget::Buffer {
                            buffer,
                            range: crate::BufferRange::new(previous, position - previous)
                                .expect("checked access endpoints"),
                        },
                        ResourceDependency::Image(image) => AccessTarget::Image {
                            image,
                            subresources: crate::ImageSubresourceRange {
                                plane,
                                mip_level: mip,
                                base_layer: previous as u16,
                                layer_count: (position - previous) as u16,
                            },
                        },
                        ResourceDependency::QueryPool(pool) => AccessTarget::Queries {
                            pool,
                            range: crate::QueryRange::new(
                                previous as u32,
                                (position - previous) as u32,
                            )
                            .expect("checked query endpoints"),
                        },
                        _ => unreachable!("access target dependency"),
                    };
                    let planned = PlannedAccess {
                        target,
                        mode: active.mode(),
                        stages: active.stages(),
                        dependency_index,
                        first_operation,
                    };
                    if let Some(last) = accesses.last_mut()
                        && merge_adjacent(last, planned)
                    {
                    } else {
                        accesses.push(planned);
                    }
                }
                while cursor < events.len() && events[cursor].0 == position {
                    let (_, index, add) = events[cursor];
                    active.update(&intervals[index], add);
                    cursor += 1;
                }
                previous = position;
            }
        }
        let operation_dependencies = operations
            .iter()
            .map(|operation| {
                operation
                    .dependencies()
                    .iter()
                    .map(|dependency| {
                        dependencies
                            .binary_search(dependency)
                            .expect("compiled dependency")
                    })
                    .collect()
            })
            .collect();
        Self {
            dependencies,
            accesses: accesses.into_boxed_slice(),
            operation_dependencies,
        }
    }

    #[must_use]
    pub fn dependencies(&self) -> &[ResourceDependency] {
        &self.dependencies
    }
    #[must_use]
    pub fn accesses(&self) -> &[PlannedAccess] {
        &self.accesses
    }
    #[must_use]
    pub fn operation_dependencies(&self, operation: usize) -> &[usize] {
        &self.operation_dependencies[operation]
    }
}

struct AccessInterval {
    start: u64,
    end: u64,
    mode: AccessMode,
    stages: PipelineStages,
    operation: usize,
}

const STAGES: [PipelineStages; 14] = [
    PipelineStages::COPY,
    PipelineStages::VERTEX_INPUT,
    PipelineStages::VERTEX_SHADER,
    PipelineStages::TESSELLATION_CONTROL_SHADER,
    PipelineStages::TESSELLATION_EVALUATION_SHADER,
    PipelineStages::GEOMETRY_SHADER,
    PipelineStages::FRAGMENT_SHADER,
    PipelineStages::EARLY_DEPTH_STENCIL,
    PipelineStages::LATE_DEPTH_STENCIL,
    PipelineStages::COLOR_OUTPUT,
    PipelineStages::COMPUTE_SHADER,
    PipelineStages::QUERY,
    PipelineStages::INDIRECT,
    PipelineStages::PRESENT,
];

#[derive(Default)]
struct ActiveAccesses {
    reads: usize,
    writes: usize,
    stages: [usize; 14],
    operations: BTreeMap<usize, usize>,
}
impl ActiveAccesses {
    fn update(&mut self, interval: &AccessInterval, add: bool) {
        let count = |value: &mut usize| if add { *value += 1 } else { *value -= 1 };
        if interval.mode.reads() {
            count(&mut self.reads);
        }
        if interval.mode.writes() {
            count(&mut self.writes);
        }
        for (index, stage) in STAGES.iter().enumerate() {
            if interval.stages.contains(*stage) {
                count(&mut self.stages[index]);
            }
        }
        count(self.operations.entry(interval.operation).or_default());
        if self.operations[&interval.operation] == 0 {
            self.operations.remove(&interval.operation);
        }
    }
    fn first_operation(&self) -> Option<usize> {
        self.operations
            .first_key_value()
            .map(|(operation, _)| *operation)
    }
    fn mode(&self) -> AccessMode {
        match (self.reads != 0, self.writes != 0) {
            (true, true) => AccessMode::ReadWrite,
            (true, false) => AccessMode::Read,
            (false, true) => AccessMode::Write,
            _ => unreachable!("empty access set"),
        }
    }
    fn stages(&self) -> PipelineStages {
        let mut selected = STAGES
            .iter()
            .enumerate()
            .filter(|(index, _)| self.stages[*index] != 0)
            .map(|(_, stage)| *stage);
        let first = selected.next().expect("validated nonempty access stages");
        selected.fold(first, PipelineStages::union)
    }
}

fn merge_adjacent(left: &mut PlannedAccess, right: PlannedAccess) -> bool {
    if left.mode != right.mode
        || left.stages != right.stages
        || left.dependency_index != right.dependency_index
        || left.first_operation != right.first_operation
    {
        return false;
    }
    left.target = match (left.target, right.target) {
        (AccessTarget::Buffer { buffer, range: a }, AccessTarget::Buffer { range: b, .. })
            if a.end() == b.offset() =>
        {
            AccessTarget::Buffer {
                buffer,
                range: crate::BufferRange::new(a.offset(), a.size() + b.size())
                    .expect("adjacent checked intervals"),
            }
        }
        (
            AccessTarget::Image {
                image,
                subresources: a,
            },
            AccessTarget::Image {
                subresources: b, ..
            },
        ) if a.plane == b.plane
            && a.mip_level == b.mip_level
            && a.base_layer + a.layer_count == b.base_layer =>
        {
            AccessTarget::Image {
                image,
                subresources: crate::ImageSubresourceRange {
                    layer_count: a.layer_count + b.layer_count,
                    ..a
                },
            }
        }
        (AccessTarget::Queries { pool, range: a }, AccessTarget::Queries { range: b, .. })
            if a.first() + a.count() == b.first() =>
        {
            AccessTarget::Queries {
                pool,
                range: crate::QueryRange::new(a.first(), a.count() + b.count())
                    .expect("adjacent checked intervals"),
            }
        }
        _ => return false,
    };
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BufferId, BufferRange, BufferRegion, CapabilityRequirements, CopyOperation, GpuCommand,
    };
    fn copy(source: BufferRange, destination: BufferRange) -> GpuOperation {
        GpuOperation::new(
            GpuCommand::Copy(CopyOperation::BufferToBuffer {
                source: BufferRegion {
                    buffer: BufferId::new(1),
                    range: source,
                },
                destination: BufferRegion {
                    buffer: BufferId::new(1),
                    range: destination,
                },
            }),
            [],
            [],
            CapabilityRequirements::none(),
        )
    }
    #[test]
    fn overlapping_accesses_keep_exact_modes_and_dense_identity() {
        let operation = copy(
            BufferRange::new(0, 16).unwrap(),
            BufferRange::new(8, 16).unwrap(),
        );
        let plan = SubmissionAccessPlan::compile(&[operation]);
        assert_eq!(
            plan.dependencies(),
            &[ResourceDependency::Buffer(BufferId::new(1))]
        );
        let expected = [
            (0, 8, AccessMode::Read),
            (8, 8, AccessMode::ReadWrite),
            (16, 8, AccessMode::Write),
        ];
        for (access, (offset, size, mode)) in plan.accesses().iter().zip(expected) {
            assert_eq!(
                access.target(),
                AccessTarget::Buffer {
                    buffer: BufferId::new(1),
                    range: BufferRange::new(offset, size).unwrap()
                }
            );
            assert_eq!(access.mode(), mode);
            assert_eq!(access.dependency_index(), 0);
            assert_eq!(access.stages(), PipelineStages::COPY);
        }
        assert_eq!(plan.accesses().len(), 3);
    }
    #[test]
    fn adjacent_equal_scopes_merge_but_operation_order_is_retained() {
        let operation = copy(
            BufferRange::new(0, 8).unwrap(),
            BufferRange::new(32, 8).unwrap(),
        );
        let another = copy(
            BufferRange::new(4, 16).unwrap(),
            BufferRange::new(40, 8).unwrap(),
        );
        let plan = SubmissionAccessPlan::compile(&[operation, another]);
        assert_eq!(plan.operation_dependencies(0), &[0]);
        assert_eq!(plan.operation_dependencies(1), &[0]);
        assert_eq!(
            plan.accesses()[0].target(),
            AccessTarget::Buffer {
                buffer: BufferId::new(1),
                range: BufferRange::new(0, 8).unwrap()
            }
        );
        assert_eq!(plan.accesses()[1].first_operation(), 1);
        // The frontier changes exactly when the first consuming operation changes.
        assert_eq!(
            plan.accesses()[1].target(),
            AccessTarget::Buffer {
                buffer: BufferId::new(1),
                range: BufferRange::new(8, 12).unwrap()
            }
        );
    }
}
