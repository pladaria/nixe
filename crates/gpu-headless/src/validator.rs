//! Atomic validation of neutral resources and submissions.

use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

use nixe_gpu::{
    AcceptedBackendSubmission, AccessScope, AccessTarget, BackendDriver, BackendDriverError,
    BackendResourceCreateInfo, BackendResourceHandle, BackendSubmissionToken, BackingView,
    BufferId, BufferRange, FrontendSubmissionId, GpuCommand, ImageId, ImageSubresourceRange,
    QueryPoolId, QueryRange, RenderPassId, RenderPassOperation, ResourceAccess, ResourceDependency,
};

use crate::timeline::{HeadlessCompletionController, SharedTimeline, TimelineState};

#[derive(Clone)]
struct ResourceRecord {
    info: BackendResourceCreateInfo,
}

#[derive(Clone, Debug)]
enum Footprint {
    Buffer {
        handle: BackendResourceHandle,
        id: BufferId,
        range: BufferRange,
        canonical: Option<(BackingView, u64, u64)>,
    },
    Image {
        handle: BackendResourceHandle,
        id: ImageId,
        subresources: ImageSubresourceRange,
        canonical: Vec<BackingView>,
    },
    Queries {
        handle: BackendResourceHandle,
        pool: QueryPoolId,
        range: QueryRange,
    },
}

#[derive(Clone)]
struct AccessState {
    footprint: Footprint,
    scope: AccessScope,
    owner: FrontendSubmissionId,
    prepared_by_barrier: bool,
}

#[derive(Clone, Default)]
struct ValidationState {
    accesses: Vec<AccessState>,
    ancestors: HashMap<FrontendSubmissionId, HashSet<FrontendSubmissionId>>,
    active_render_pass: Option<RenderPassId>,
    active_queries: HashSet<(QueryPoolId, u32)>,
}

/// Deterministic concrete consumer of [`nixe_gpu`]'s backend contract.
pub struct HeadlessBackendDriver {
    resources: HashMap<BackendResourceHandle, ResourceRecord>,
    validation: ValidationState,
    submissions: HashSet<BackendSubmissionToken>,
    timeline: SharedTimeline,
    torn_down: bool,
}

impl HeadlessBackendDriver {
    /// Creates an empty validator and a cloneable manual completion control.
    #[must_use]
    pub fn new() -> (Self, HeadlessCompletionController) {
        let timeline = Arc::new(Mutex::new(TimelineState::default()));
        (
            Self {
                resources: HashMap::new(),
                validation: ValidationState::default(),
                submissions: HashSet::new(),
                timeline: Arc::clone(&timeline),
                torn_down: false,
            },
            HeadlessCompletionController { timeline },
        )
    }

    /// Number of resources currently owned by the concrete backend.
    #[must_use]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Number of accepted submissions not yet released.
    #[must_use]
    pub fn submission_count(&self) -> usize {
        self.submissions.len()
    }

    fn validate_submission(
        &self,
        accepted: &AcceptedBackendSubmission<'_>,
    ) -> Result<ValidationState, HeadlessValidationError> {
        let submission = accepted.submission();
        let resources = accepted
            .resources()
            .iter()
            .map(|resolved| (resolved.dependency(), resolved.handle()))
            .collect::<HashMap<_, _>>();
        let mut next = self.validation.clone();
        let mut ancestors = HashSet::new();
        for predecessor in submission.predecessors() {
            ancestors.insert(*predecessor);
            if let Some(transitive) = next.ancestors.get(predecessor) {
                ancestors.extend(transitive.iter().copied());
            }
        }

        for (operation_index, operation) in submission.operations().iter().enumerate() {
            validate_operation_order(&mut next, operation.command(), operation_index)?;
            match operation.command() {
                GpuCommand::Barrier(barrier) => {
                    for transition in barrier.transitions() {
                        let footprint = self.resolve_footprint(transition.target(), &resources)?;
                        apply_transition(
                            &mut next,
                            footprint,
                            transition.before(),
                            transition.after(),
                            submission.id(),
                            &ancestors,
                            operation_index,
                        )?;
                    }
                }
                _ => {
                    for access in operation.accesses() {
                        let footprint = self.resolve_footprint(access.target(), &resources)?;
                        apply_access(
                            &mut next,
                            footprint,
                            *access,
                            submission.id(),
                            &ancestors,
                            operation_index,
                        )?;
                    }
                }
            }
        }
        next.ancestors.insert(submission.id(), ancestors);
        Ok(next)
    }

    fn resolve_footprint(
        &self,
        target: AccessTarget,
        resources: &HashMap<ResourceDependency, BackendResourceHandle>,
    ) -> Result<Footprint, HeadlessValidationError> {
        match target {
            AccessTarget::Buffer { buffer, range } => {
                let (handle, record) =
                    self.resource(ResourceDependency::Buffer(buffer), resources)?;
                let BackendResourceCreateInfo::Buffer {
                    description, view, ..
                } = &record.info
                else {
                    return Err(HeadlessValidationError::ResourceKindMismatch);
                };
                if range.end() > description.size() {
                    return Err(HeadlessValidationError::AccessOutOfBounds { target });
                }
                let canonical = if let Some(view) = view {
                    if range.offset() < view.buffer_offset()
                        || range.end() > view.buffer_offset() + view.size()
                    {
                        return Err(HeadlessValidationError::AccessOutsideBacking { target });
                    }
                    Some((
                        view.backing().clone(),
                        range.offset() - view.buffer_offset(),
                        range.size(),
                    ))
                } else {
                    None
                };
                Ok(Footprint::Buffer {
                    handle,
                    id: buffer,
                    range,
                    canonical,
                })
            }
            AccessTarget::Image {
                image,
                subresources,
            } => {
                let (handle, record) =
                    self.resource(ResourceDependency::Image(image), resources)?;
                let BackendResourceCreateInfo::Image {
                    description, view, ..
                } = &record.info
                else {
                    return Err(HeadlessValidationError::ResourceKindMismatch);
                };
                if !valid_image_range(*description, subresources) {
                    return Err(HeadlessValidationError::AccessOutOfBounds { target });
                }
                let mut canonical = Vec::new();
                if let Some(view) = view {
                    for binding in view.bindings() {
                        if image_ranges_overlap(binding.subresources(), subresources) {
                            canonical.push(binding.backing().clone());
                        }
                    }
                    if !bindings_cover(view, subresources) {
                        return Err(HeadlessValidationError::AccessOutsideBacking { target });
                    }
                }
                Ok(Footprint::Image {
                    handle,
                    id: image,
                    subresources,
                    canonical,
                })
            }
            AccessTarget::Queries { pool, range } => {
                let (handle, record) =
                    self.resource(ResourceDependency::QueryPool(pool), resources)?;
                let BackendResourceCreateInfo::QueryPool { description, .. } = &record.info else {
                    return Err(HeadlessValidationError::ResourceKindMismatch);
                };
                if range.first().saturating_add(range.count()) > description.count() {
                    return Err(HeadlessValidationError::AccessOutOfBounds { target });
                }
                Ok(Footprint::Queries {
                    handle,
                    pool,
                    range,
                })
            }
        }
    }

    fn resource<'a>(
        &'a self,
        dependency: ResourceDependency,
        resources: &HashMap<ResourceDependency, BackendResourceHandle>,
    ) -> Result<(BackendResourceHandle, &'a ResourceRecord), HeadlessValidationError> {
        let handle = resources
            .get(&dependency)
            .ok_or(HeadlessValidationError::MissingResolvedResource(dependency))?;
        let record = self
            .resources
            .get(handle)
            .ok_or(HeadlessValidationError::MissingBackendResource(*handle))?;
        Ok((*handle, record))
    }

    fn timeline(&self) -> Result<std::sync::MutexGuard<'_, TimelineState>, BackendDriverError> {
        self.timeline
            .lock()
            .map_err(|_| BackendDriverError::failure("headless completion state is poisoned"))
    }

    fn check_device(&mut self) -> Result<(), BackendDriverError> {
        let timeline = self.timeline()?;
        let device_loss = timeline.device_loss.clone();
        let torn_down = timeline.torn_down;
        drop(timeline);
        if let Some(reason) = device_loss {
            self.resources.clear();
            self.submissions.clear();
            self.validation = ValidationState::default();
            let mut timeline = self.timeline()?;
            timeline.accepted.clear();
            timeline.completed.clear();
            return Err(BackendDriverError::device_lost(reason));
        }
        if self.torn_down || torn_down {
            return Err(BackendDriverError::failure("headless backend is torn down"));
        }
        Ok(())
    }
}

impl BackendDriver for HeadlessBackendDriver {
    fn create_resource(
        &mut self,
        handle: BackendResourceHandle,
        info: &BackendResourceCreateInfo,
    ) -> Result<(), BackendDriverError> {
        self.check_device()?;
        if self.resources.contains_key(&handle) {
            return Err(BackendDriverError::failure(
                "duplicate headless resource handle",
            ));
        }
        self.resources
            .insert(handle, ResourceRecord { info: info.clone() });
        Ok(())
    }

    fn destroy_resource(
        &mut self,
        handle: BackendResourceHandle,
    ) -> Result<(), BackendDriverError> {
        self.check_device()?;
        if self.resources.remove(&handle).is_none() {
            return Err(BackendDriverError::failure(
                "unknown headless resource handle",
            ));
        }
        self.validation
            .accesses
            .retain(|access| access.footprint.handle() != handle);
        Ok(())
    }

    fn submit(
        &mut self,
        submission: &AcceptedBackendSubmission<'_>,
    ) -> Result<(), BackendDriverError> {
        self.check_device()?;
        let next = self
            .validate_submission(submission)
            .map_err(|error| BackendDriverError::failure(error.to_string()))?;
        let mut timeline = self.timeline()?;
        if !timeline.accepted.insert(submission.token()) {
            return Err(BackendDriverError::failure(
                "duplicate headless submission token",
            ));
        }
        drop(timeline);
        self.submissions.insert(submission.token());
        self.validation = next;
        Ok(())
    }

    fn has_completed(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<bool, BackendDriverError> {
        self.check_device()?;
        let timeline = self.timeline()?;
        if !timeline.accepted.contains(&submission) {
            return Err(BackendDriverError::failure(
                "unknown headless submission token",
            ));
        }
        Ok(timeline.completed.contains(&submission))
    }

    fn release_submission(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        self.check_device()?;
        let mut timeline = self.timeline()?;
        if !timeline.completed.contains(&submission) || !timeline.accepted.contains(&submission) {
            return Err(BackendDriverError::failure(
                "headless submission is not complete",
            ));
        }
        timeline.completed.remove(&submission);
        timeline.accepted.remove(&submission);
        drop(timeline);
        self.submissions.remove(&submission);
        Ok(())
    }

    fn teardown(&mut self) -> Result<(), BackendDriverError> {
        if self.torn_down {
            return Ok(());
        }
        self.check_device()?;
        let mut timeline = self.timeline()?;
        timeline.accepted.clear();
        timeline.completed.clear();
        timeline.torn_down = true;
        drop(timeline);
        self.resources.clear();
        self.submissions.clear();
        self.validation = ValidationState::default();
        self.torn_down = true;
        Ok(())
    }
}

impl Footprint {
    const fn handle(&self) -> BackendResourceHandle {
        match self {
            Self::Buffer { handle, .. }
            | Self::Image { handle, .. }
            | Self::Queries { handle, .. } => *handle,
        }
    }
}

fn validate_operation_order(
    state: &mut ValidationState,
    command: &GpuCommand,
    operation_index: usize,
) -> Result<(), HeadlessValidationError> {
    match command {
        GpuCommand::RenderPass(RenderPassOperation::Begin { render_pass, .. }) => {
            if let Some(active) = state.active_render_pass {
                return Err(HeadlessValidationError::NestedRenderPass {
                    operation_index,
                    active,
                    requested: *render_pass,
                });
            }
            state.active_render_pass = Some(*render_pass);
        }
        GpuCommand::RenderPass(RenderPassOperation::End { render_pass }) => {
            if state.active_render_pass != Some(*render_pass) {
                return Err(HeadlessValidationError::RenderPassEndMismatch {
                    operation_index,
                    active: state.active_render_pass,
                    requested: *render_pass,
                });
            }
            state.active_render_pass = None;
        }
        GpuCommand::Draw(draw) if state.active_render_pass != Some(draw.render_pass) => {
            return Err(HeadlessValidationError::DrawOutsideRenderPass {
                operation_index,
                requested: draw.render_pass,
                active: state.active_render_pass,
            });
        }
        GpuCommand::Query(query) => validate_query_order(state, query, operation_index)?,
        _ => {}
    }
    Ok(())
}

fn validate_query_order(
    state: &mut ValidationState,
    query: &nixe_gpu::QueryOperation,
    operation_index: usize,
) -> Result<(), HeadlessValidationError> {
    use nixe_gpu::QueryOperation;
    match query {
        QueryOperation::Begin { pool, query, .. } => {
            if !state.active_queries.insert((*pool, *query)) {
                return Err(HeadlessValidationError::QueryAlreadyActive {
                    operation_index,
                    pool: *pool,
                    query: *query,
                });
            }
        }
        QueryOperation::End { pool, query, .. } => {
            if !state.active_queries.remove(&(*pool, *query)) {
                return Err(HeadlessValidationError::QueryNotActive {
                    operation_index,
                    pool: *pool,
                    query: *query,
                });
            }
        }
        QueryOperation::Reset { pool, range, .. } => {
            if state.active_queries.iter().any(|(active_pool, query)| {
                *active_pool == *pool
                    && *query >= range.first()
                    && *query < range.first() + range.count()
            }) {
                return Err(HeadlessValidationError::ActiveQueryReset { operation_index });
            }
        }
        QueryOperation::WriteTimestamp { .. } | QueryOperation::Resolve { .. } => {}
    }
    Ok(())
}

fn apply_access(
    state: &mut ValidationState,
    footprint: Footprint,
    access: ResourceAccess,
    owner: FrontendSubmissionId,
    ancestors: &HashSet<FrontendSubmissionId>,
    operation_index: usize,
) -> Result<(), HeadlessValidationError> {
    for previous in &state.accesses {
        if !footprints_overlap(&footprint, &previous.footprint) {
            continue;
        }
        if previous.owner != owner && !ancestors.contains(&previous.owner) {
            return Err(HeadlessValidationError::UnorderedAccess {
                operation_index,
                previous: previous.owner,
                current: owner,
            });
        }
        if previous.scope != access.scope()
            || (previous.scope.mode().writes() || access.scope().mode().writes())
                && !previous.prepared_by_barrier
        {
            return Err(HeadlessValidationError::MissingBarrier { operation_index });
        }
    }
    state
        .accesses
        .retain(|previous| !footprints_overlap(&footprint, &previous.footprint));
    state.accesses.push(AccessState {
        footprint,
        scope: access.scope(),
        owner,
        prepared_by_barrier: false,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_transition(
    state: &mut ValidationState,
    footprint: Footprint,
    before: AccessScope,
    after: AccessScope,
    owner: FrontendSubmissionId,
    ancestors: &HashSet<FrontendSubmissionId>,
    operation_index: usize,
) -> Result<(), HeadlessValidationError> {
    let overlapping = state
        .accesses
        .iter()
        .filter(|previous| footprints_overlap(&footprint, &previous.footprint))
        .collect::<Vec<_>>();
    if overlapping.is_empty() {
        return Err(HeadlessValidationError::UnknownTransitionState { operation_index });
    }
    for previous in overlapping {
        if previous.owner != owner && !ancestors.contains(&previous.owner) {
            return Err(HeadlessValidationError::UnorderedAccess {
                operation_index,
                previous: previous.owner,
                current: owner,
            });
        }
        if previous.scope != before {
            return Err(HeadlessValidationError::TransitionBeforeMismatch { operation_index });
        }
    }
    state
        .accesses
        .retain(|previous| !footprints_overlap(&footprint, &previous.footprint));
    state.accesses.push(AccessState {
        footprint,
        scope: after,
        owner,
        prepared_by_barrier: true,
    });
    Ok(())
}

fn footprints_overlap(left: &Footprint, right: &Footprint) -> bool {
    match (left, right) {
        (
            Footprint::Buffer {
                handle: left_handle,
                id: left_id,
                range: left_range,
                canonical: left_canonical,
            },
            Footprint::Buffer {
                handle: right_handle,
                id: right_id,
                range: right_range,
                canonical: right_canonical,
            },
        ) => {
            (left_handle == right_handle
                && left_id == right_id
                && ranges_overlap(
                    left_range.offset(),
                    left_range.size(),
                    right_range.offset(),
                    right_range.size(),
                ))
                || match (left_canonical, right_canonical) {
                    (
                        Some((left, left_offset, left_size)),
                        Some((right, right_offset, right_size)),
                    ) => backing_slices_overlap(
                        left,
                        *left_offset,
                        *left_size,
                        right,
                        *right_offset,
                        *right_size,
                    ),
                    _ => false,
                }
        }
        (
            Footprint::Image {
                handle: left_handle,
                id: left_id,
                subresources: left_range,
                canonical: left_canonical,
            },
            Footprint::Image {
                handle: right_handle,
                id: right_id,
                subresources: right_range,
                canonical: right_canonical,
            },
        ) => {
            (left_handle == right_handle
                && left_id == right_id
                && image_ranges_overlap(*left_range, *right_range))
                || left_canonical.iter().any(|left| {
                    right_canonical.iter().any(|right| {
                        backing_slices_overlap(left, 0, left.size(), right, 0, right.size())
                    })
                })
        }
        (
            Footprint::Buffer {
                canonical: Some((buffer, offset, size)),
                ..
            },
            Footprint::Image { canonical, .. },
        )
        | (
            Footprint::Image { canonical, .. },
            Footprint::Buffer {
                canonical: Some((buffer, offset, size)),
                ..
            },
        ) => canonical
            .iter()
            .any(|image| backing_slices_overlap(buffer, *offset, *size, image, 0, image.size())),
        (
            Footprint::Queries {
                handle: left_handle,
                pool: left_pool,
                range: left_range,
            },
            Footprint::Queries {
                handle: right_handle,
                pool: right_pool,
                range: right_range,
            },
        ) => {
            left_handle == right_handle
                && left_pool == right_pool
                && ranges_overlap(
                    u64::from(left_range.first()),
                    u64::from(left_range.count()),
                    u64::from(right_range.first()),
                    u64::from(right_range.count()),
                )
        }
        _ => false,
    }
}

fn backing_slices_overlap(
    left: &BackingView,
    left_offset: u64,
    left_size: u64,
    right: &BackingView,
    right_offset: u64,
    right_size: u64,
) -> bool {
    canonical_fragments(left, left_offset, left_size).any(|(left_segment, left_start, left_len)| {
        canonical_fragments(right, right_offset, right_size).any(
            |(right_segment, right_start, right_len)| {
                left_segment.page() == right_segment.page()
                    && ranges_overlap(left_start, left_len, right_start, right_len)
            },
        )
    })
}

fn canonical_fragments(
    backing: &BackingView,
    offset: u64,
    size: u64,
) -> impl Iterator<Item = (&nixe_gpu::CanonicalBackingSegment, u64, u64)> {
    let end = offset + size;
    let mut cursor = 0_u64;
    backing
        .range()
        .segments()
        .iter()
        .filter_map(move |segment| {
            let segment_start = cursor;
            let segment_end = cursor + segment.size();
            cursor = segment_end;
            let start = offset.max(segment_start);
            let finish = end.min(segment_end);
            (start < finish).then(|| {
                (
                    segment,
                    segment.offset() + start - segment_start,
                    finish - start,
                )
            })
        })
}

const fn ranges_overlap(
    left_offset: u64,
    left_size: u64,
    right_offset: u64,
    right_size: u64,
) -> bool {
    left_offset < right_offset + right_size && right_offset < left_offset + left_size
}

fn image_ranges_overlap(left: ImageSubresourceRange, right: ImageSubresourceRange) -> bool {
    left.plane == right.plane
        && left.mip_level == right.mip_level
        && ranges_overlap(
            u64::from(left.base_layer),
            u64::from(left.layer_count),
            u64::from(right.base_layer),
            u64::from(right.layer_count),
        )
}

fn bindings_cover(view: &nixe_gpu::ImageView, requested: ImageSubresourceRange) -> bool {
    let requested_end = u32::from(requested.base_layer) + u32::from(requested.layer_count);
    let mut intervals = view
        .bindings()
        .iter()
        .map(|binding| binding.subresources())
        .filter(|candidate| {
            candidate.plane == requested.plane && candidate.mip_level == requested.mip_level
        })
        .map(|candidate| {
            (
                u32::from(candidate.base_layer),
                u32::from(candidate.base_layer) + u32::from(candidate.layer_count),
            )
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable();
    let mut covered_until = u32::from(requested.base_layer);
    for (start, end) in intervals {
        if end <= covered_until || start >= requested_end {
            continue;
        }
        if start > covered_until {
            return false;
        }
        covered_until = end.min(requested_end);
        if covered_until == requested_end {
            return true;
        }
    }
    false
}

fn valid_image_range(
    description: nixe_gpu::ImageDescription,
    range: ImageSubresourceRange,
) -> bool {
    range.layer_count != 0
        && range.plane < description.format().plane_count()
        && range.mip_level < description.mip_levels()
        && range
            .base_layer
            .checked_add(range.layer_count)
            .is_some_and(|end| end <= description.array_layers())
}

/// Deterministic rejection produced before a headless submission mutates state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeadlessValidationError {
    MissingResolvedResource(ResourceDependency),
    MissingBackendResource(BackendResourceHandle),
    ResourceKindMismatch,
    AccessOutOfBounds {
        target: AccessTarget,
    },
    AccessOutsideBacking {
        target: AccessTarget,
    },
    UnorderedAccess {
        operation_index: usize,
        previous: FrontendSubmissionId,
        current: FrontendSubmissionId,
    },
    MissingBarrier {
        operation_index: usize,
    },
    UnknownTransitionState {
        operation_index: usize,
    },
    TransitionBeforeMismatch {
        operation_index: usize,
    },
    NestedRenderPass {
        operation_index: usize,
        active: RenderPassId,
        requested: RenderPassId,
    },
    RenderPassEndMismatch {
        operation_index: usize,
        active: Option<RenderPassId>,
        requested: RenderPassId,
    },
    DrawOutsideRenderPass {
        operation_index: usize,
        requested: RenderPassId,
        active: Option<RenderPassId>,
    },
    QueryAlreadyActive {
        operation_index: usize,
        pool: QueryPoolId,
        query: u32,
    },
    QueryNotActive {
        operation_index: usize,
        pool: QueryPoolId,
        query: u32,
    },
    ActiveQueryReset {
        operation_index: usize,
    },
}

impl Display for HeadlessValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "headless validation failed: {self:?}")
    }
}

impl std::error::Error for HeadlessValidationError {}
