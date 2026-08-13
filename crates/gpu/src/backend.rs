//! Validated lifetime boundary between neutral GPU work and a host backend.
//!
//! [`Backend`] owns all instance, generation, capability, dependency, and
//! lifecycle validation. A concrete [`BackendDriver`] observes a resource or
//! submission only after the complete request has passed those checks.

use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};

use crate::{
    BackendCapabilities, BackendCapabilityError, BackendCompletionError, BackendCompletionSource,
    BackendInstanceId, BackendSubmissionToken, BufferDescription, BufferId, BufferView,
    BufferViewError, CapabilityAgreement, CapabilityRequirement, CapabilityRequirements,
    DescriptorTableDescription, DescriptorTableId, FrontendSubmissionId, GpuAllocationDescription,
    GpuAllocationId, ImageDescription, ImageId, ImageView, ImageViewError, OperationSubmission,
    PipelineDescription, PipelineId, QueryPoolDescription, QueryPoolId, RenderPassDescription,
    RenderPassId, ResourceDependency, SamplerDescription, SamplerId, ShaderBackendModule,
    ShaderDescription, ShaderId, ShaderStage,
};

/// Semantic kind carried by every backend resource handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BackendResourceKind {
    Allocation,
    Buffer,
    Image,
    Sampler,
    Shader,
    Pipeline,
    DescriptorTable,
    RenderPass,
    QueryPool,
}

/// Pointer-free, instance-scoped and generation-safe backend resource handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BackendResourceHandle {
    instance: BackendInstanceId,
    slot: u64,
    generation: u32,
    kind: BackendResourceKind,
}

impl BackendResourceHandle {
    #[must_use]
    pub const fn new(
        instance: BackendInstanceId,
        slot: u64,
        generation: u32,
        kind: BackendResourceKind,
    ) -> Self {
        Self {
            instance,
            slot,
            generation,
            kind,
        }
    }

    #[must_use]
    pub const fn instance(self) -> BackendInstanceId {
        self.instance
    }

    #[must_use]
    pub const fn slot(self) -> u64 {
        self.slot
    }

    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    #[must_use]
    pub const fn kind(self) -> BackendResourceKind {
        self.kind
    }
}

impl Display for BackendResourceHandle {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "backend-resource[{} slot=0x{:016x} generation={} kind={:?}]",
            self.instance, self.slot, self.generation, self.kind
        )
    }
}

/// Complete immutable input used to create one backend resource.
///
/// Logical descriptions remain separate from optional backing views. Shader
/// modules are backend-language source produced from verified neutral IR, never
/// guest ISA bytes or host handles.
#[derive(Clone, Debug, PartialEq)]
pub enum BackendResourceCreateInfo {
    Allocation {
        id: GpuAllocationId,
        description: GpuAllocationDescription,
    },
    Buffer {
        id: BufferId,
        description: BufferDescription,
        view: Option<BufferView>,
    },
    Image {
        id: ImageId,
        description: ImageDescription,
        view: Option<ImageView>,
    },
    Sampler {
        id: SamplerId,
        description: SamplerDescription,
    },
    Shader {
        id: ShaderId,
        description: ShaderDescription,
        module: ShaderBackendModule,
    },
    Pipeline {
        id: PipelineId,
        description: PipelineDescription,
    },
    DescriptorTable {
        id: DescriptorTableId,
        description: DescriptorTableDescription,
    },
    RenderPass {
        id: RenderPassId,
        description: RenderPassDescription,
    },
    QueryPool {
        id: QueryPoolId,
        description: QueryPoolDescription,
    },
}

impl BackendResourceCreateInfo {
    #[must_use]
    pub const fn dependency(&self) -> ResourceDependency {
        match self {
            Self::Allocation { id, .. } => ResourceDependency::Allocation(*id),
            Self::Buffer { id, .. } => ResourceDependency::Buffer(*id),
            Self::Image { id, .. } => ResourceDependency::Image(*id),
            Self::Sampler { id, .. } => ResourceDependency::Sampler(*id),
            Self::Shader { id, .. } => ResourceDependency::Shader(*id),
            Self::Pipeline { id, .. } => ResourceDependency::Pipeline(*id),
            Self::DescriptorTable { id, .. } => ResourceDependency::DescriptorTable(*id),
            Self::RenderPass { id, .. } => ResourceDependency::RenderPass(*id),
            Self::QueryPool { id, .. } => ResourceDependency::QueryPool(*id),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> BackendResourceKind {
        dependency_kind(self.dependency())
    }

    fn backing_allocations(&self) -> Vec<GpuAllocationId> {
        match self {
            Self::Buffer {
                view: Some(view), ..
            } => vec![view.backing().allocation()],
            Self::Image {
                view: Some(view), ..
            } => {
                let mut allocations = Vec::new();
                for binding in view.bindings() {
                    let allocation = binding.backing().allocation();
                    if !allocations.contains(&allocation) {
                        allocations.push(allocation);
                    }
                }
                allocations
            }
            _ => Vec::new(),
        }
    }

    /// Capabilities required before this complete resource can be created.
    pub fn capability_requirements(
        &self,
    ) -> Result<CapabilityRequirements, BackendResourceValidationError> {
        Ok(match self {
            Self::Image { description, .. } => CapabilityRequirements::new([
                CapabilityRequirement::ImageFormat(description.format()),
                CapabilityRequirement::SampleCount(description.samples()),
            ]),
            Self::Shader { description, .. } => {
                CapabilityRequirements::new([CapabilityRequirement::ShaderStage(description.stage)])
            }
            Self::DescriptorTable { description, .. } => {
                CapabilityRequirements::new([CapabilityRequirement::DescriptorBindings(
                    u32::try_from(description.bindings().len()).map_err(|_| {
                        BackendResourceValidationError::DescriptorBindingCountOverflow
                    })?,
                )])
            }
            Self::QueryPool { description, .. } => {
                CapabilityRequirements::new([CapabilityRequirement::QueryKind(description.kind())])
            }
            _ => CapabilityRequirements::none(),
        })
    }
}

/// One fully resolved logical dependency supplied to a concrete backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedResourceDependency {
    dependency: ResourceDependency,
    handle: BackendResourceHandle,
}

impl ResolvedResourceDependency {
    #[must_use]
    pub const fn dependency(self) -> ResourceDependency {
        self.dependency
    }

    #[must_use]
    pub const fn handle(self) -> BackendResourceHandle {
        self.handle
    }
}

/// Completely validated submission view passed to a concrete backend.
pub struct AcceptedBackendSubmission<'a> {
    token: BackendSubmissionToken,
    submission: &'a OperationSubmission,
    agreement: CapabilityAgreement,
    resources: &'a [ResolvedResourceDependency],
}

impl AcceptedBackendSubmission<'_> {
    #[must_use]
    pub const fn token(&self) -> BackendSubmissionToken {
        self.token
    }

    #[must_use]
    pub const fn submission(&self) -> &OperationSubmission {
        self.submission
    }

    #[must_use]
    pub const fn capability_agreement(&self) -> CapabilityAgreement {
        self.agreement
    }

    #[must_use]
    pub const fn resources(&self) -> &[ResolvedResourceDependency] {
        self.resources
    }
}

/// Concrete backend operations behind the validated neutral boundary.
///
/// Implementations must treat each callback atomically. Device loss is
/// distinguished from an ordinary backend failure so [`Backend`] can enter a
/// deterministic terminal state and discard every neutral ownership record.
pub trait BackendDriver {
    fn create_resource(
        &mut self,
        handle: BackendResourceHandle,
        info: &BackendResourceCreateInfo,
    ) -> Result<(), BackendDriverError>;

    fn destroy_resource(&mut self, handle: BackendResourceHandle)
    -> Result<(), BackendDriverError>;

    fn submit(
        &mut self,
        submission: &AcceptedBackendSubmission<'_>,
    ) -> Result<(), BackendDriverError>;

    fn has_completed(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<bool, BackendDriverError>;

    fn release_submission(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError>;

    fn teardown(&mut self) -> Result<(), BackendDriverError>;
}

/// Failure reported by a concrete backend without leaking its object types.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendDriverError {
    Failure(Box<str>),
    DeviceLost(Box<str>),
}

impl BackendDriverError {
    #[must_use]
    pub fn failure(message: impl Into<Box<str>>) -> Self {
        Self::Failure(message.into())
    }

    #[must_use]
    pub fn device_lost(message: impl Into<Box<str>>) -> Self {
        Self::DeviceLost(message.into())
    }
}

impl Display for BackendDriverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failure(message) => write!(formatter, "backend operation failed: {message}"),
            Self::DeviceLost(message) => write!(formatter, "backend device lost: {message}"),
        }
    }
}

impl std::error::Error for BackendDriverError {}

/// Observable lifecycle state of one neutral backend instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendState {
    Active,
    DeviceLost,
    TornDown,
}

struct ResourceSlot {
    generation: u32,
    record: Option<ResourceRecord>,
}

struct ResourceRecord {
    dependency: ResourceDependency,
    backing_allocations: Box<[GpuAllocationId]>,
    allocation_description: Option<GpuAllocationDescription>,
}

struct SubmissionSlot {
    generation: u32,
    record: Option<SubmissionRecord>,
}

struct SubmissionRecord {
    resources: Box<[BackendResourceHandle]>,
}

/// Validated neutral backend instance consumed by the composition root.
pub struct Backend<D> {
    instance: BackendInstanceId,
    capabilities: BackendCapabilities,
    driver: D,
    state: BackendState,
    device_loss_reason: Option<Box<str>>,
    resources: Vec<ResourceSlot>,
    resources_by_dependency: HashMap<ResourceDependency, BackendResourceHandle>,
    submissions: Vec<SubmissionSlot>,
    accepted_frontends: HashSet<FrontendSubmissionId>,
}

impl<D: BackendDriver> Backend<D> {
    #[must_use]
    pub fn new(instance: BackendInstanceId, capabilities: BackendCapabilities, driver: D) -> Self {
        Self {
            instance,
            capabilities,
            driver,
            state: BackendState::Active,
            device_loss_reason: None,
            resources: Vec::new(),
            resources_by_dependency: HashMap::new(),
            submissions: Vec::new(),
            accepted_frontends: HashSet::new(),
        }
    }

    #[must_use]
    pub const fn instance(&self) -> BackendInstanceId {
        self.instance
    }

    #[must_use]
    pub const fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    #[must_use]
    pub const fn state(&self) -> BackendState {
        self.state
    }

    #[must_use]
    pub fn device_loss_reason(&self) -> Option<&str> {
        self.device_loss_reason.as_deref()
    }

    #[must_use]
    pub const fn driver(&self) -> &D {
        &self.driver
    }

    /// Creates one resource after validating the complete request.
    pub fn create_resource(
        &mut self,
        info: BackendResourceCreateInfo,
    ) -> Result<BackendResourceHandle, BackendError> {
        self.require_active()?;
        let dependency = info.dependency();
        if self.resources_by_dependency.contains_key(&dependency) {
            return Err(BackendError::DuplicateResource(dependency));
        }
        self.validate_resource_info(&info)?;
        self.capabilities
            .negotiate(
                &info
                    .capability_requirements()
                    .map_err(BackendError::InvalidResource)?,
            )
            .map_err(BackendError::Capability)?;
        let backing_allocations = info.backing_allocations();
        for allocation in &backing_allocations {
            self.resolve_dependency(ResourceDependency::Allocation(*allocation))?;
        }

        let (slot, generation) = next_resource_slot(&self.resources)?;
        let handle = BackendResourceHandle::new(self.instance, slot, generation, info.kind());
        self.resources
            .try_reserve(usize::from(slot as usize == self.resources.len()))
            .map_err(|_| BackendError::ResourceExhausted)?;
        self.resources_by_dependency
            .try_reserve(1)
            .map_err(|_| BackendError::ResourceExhausted)?;

        if let Err(error) = self.driver.create_resource(handle, &info) {
            return Err(self.handle_driver_error(error));
        }
        let record = ResourceRecord {
            dependency,
            backing_allocations: backing_allocations.into_boxed_slice(),
            allocation_description: match info {
                BackendResourceCreateInfo::Allocation { description, .. } => Some(description),
                _ => None,
            },
        };
        commit_resource_slot(&mut self.resources, slot, generation, record);
        self.resources_by_dependency.insert(dependency, handle);
        Ok(handle)
    }

    /// Destroys a live resource. Referenced resources remain live until every
    /// accepted submission which retained them has been released.
    pub fn destroy_resource(&mut self, handle: BackendResourceHandle) -> Result<(), BackendError> {
        self.require_active()?;
        let record = self.validate_resource_handle(handle)?;
        if self.submissions.iter().any(|slot| {
            slot.record
                .as_ref()
                .is_some_and(|submission| submission.resources.contains(&handle))
        }) {
            return Err(BackendError::ResourceInUse(handle));
        }
        if let ResourceDependency::Allocation(allocation) = record.dependency
            && self.resources.iter().any(|slot| {
                slot.record.as_ref().is_some_and(|candidate| {
                    candidate.dependency != record.dependency
                        && candidate.backing_allocations.contains(&allocation)
                })
            })
        {
            return Err(BackendError::ResourceInUse(handle));
        }
        let dependency = record.dependency;
        if let Err(error) = self.driver.destroy_resource(handle) {
            return Err(self.handle_driver_error(error));
        }
        self.resources[handle.slot as usize].record = None;
        self.resources_by_dependency.remove(&dependency);
        Ok(())
    }

    /// Validates and atomically submits a complete neutral operation sequence.
    pub fn submit(
        &mut self,
        submission: &OperationSubmission,
    ) -> Result<BackendSubmissionToken, BackendError> {
        self.require_active()?;
        if self.accepted_frontends.contains(&submission.id()) {
            return Err(BackendError::DuplicateSubmission(submission.id()));
        }
        for predecessor in submission.predecessors() {
            if !self.accepted_frontends.contains(predecessor) {
                return Err(BackendError::UnknownPredecessor(*predecessor));
            }
        }
        let requirements = submission.capability_requirements();
        let agreement = self
            .capabilities
            .negotiate_all(&requirements)
            .map_err(BackendError::Capability)?;

        let mut resolved = Vec::new();
        for operation in submission.operations() {
            for dependency in operation.dependencies() {
                if resolved
                    .iter()
                    .any(|entry: &ResolvedResourceDependency| entry.dependency == *dependency)
                {
                    continue;
                }
                resolved.push(ResolvedResourceDependency {
                    dependency: *dependency,
                    handle: self.resolve_dependency(*dependency)?,
                });
            }
        }
        let (slot, generation) = next_submission_slot(&self.submissions)?;
        let token = BackendSubmissionToken::new(self.instance, slot, generation);
        self.submissions
            .try_reserve(usize::from(slot as usize == self.submissions.len()))
            .map_err(|_| BackendError::ResourceExhausted)?;
        self.accepted_frontends
            .try_reserve(1)
            .map_err(|_| BackendError::ResourceExhausted)?;

        let accepted = AcceptedBackendSubmission {
            token,
            submission,
            agreement,
            resources: &resolved,
        };
        if let Err(error) = self.driver.submit(&accepted) {
            return Err(self.handle_driver_error(error));
        }
        let record = SubmissionRecord {
            resources: resolved
                .iter()
                .map(|dependency| dependency.handle)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        commit_submission_slot(&mut self.submissions, slot, generation, record);
        self.accepted_frontends.insert(submission.id());
        Ok(token)
    }

    /// Returns host completion only. No canonical-memory visibility or guest
    /// timeline transition occurs at this boundary.
    pub fn has_completed(&mut self, token: BackendSubmissionToken) -> Result<bool, BackendError> {
        self.require_active()?;
        self.validate_submission_token(token)?;
        self.driver
            .has_completed(token)
            .map_err(|error| self.handle_driver_error(error))
    }

    /// Releases one host-complete token, making its slot reusable with a new
    /// generation. Frontend ordering history remains retained for successors.
    pub fn release_submission(
        &mut self,
        token: BackendSubmissionToken,
    ) -> Result<(), BackendError> {
        self.require_active()?;
        self.validate_submission_token(token)?;
        let completed = self
            .driver
            .has_completed(token)
            .map_err(|error| self.handle_driver_error(error))?;
        if !completed {
            return Err(BackendError::SubmissionIncomplete(token));
        }
        if let Err(error) = self.driver.release_submission(token) {
            return Err(self.handle_driver_error(error));
        }
        self.submissions[token.slot() as usize].record = None;
        Ok(())
    }

    /// Explicitly tears down the instance without fabricating completion.
    pub fn teardown(&mut self) -> Result<(), BackendError> {
        match self.state {
            BackendState::TornDown => return Ok(()),
            BackendState::DeviceLost => {
                self.clear_ownership();
                return Err(BackendError::DeviceLost(
                    self.device_loss_reason
                        .clone()
                        .unwrap_or_else(|| "unspecified device loss".into()),
                ));
            }
            BackendState::Active => {}
        }
        let result = self.driver.teardown();
        match result {
            Ok(()) => {
                self.clear_ownership();
                self.state = BackendState::TornDown;
                Ok(())
            }
            Err(error) => Err(self.handle_driver_error(error)),
        }
    }

    #[must_use]
    pub fn into_driver(self) -> D {
        self.driver
    }

    fn require_active(&self) -> Result<(), BackendError> {
        match self.state {
            BackendState::Active => Ok(()),
            BackendState::DeviceLost => Err(BackendError::DeviceLost(
                self.device_loss_reason
                    .clone()
                    .unwrap_or_else(|| "unspecified device loss".into()),
            )),
            BackendState::TornDown => Err(BackendError::TornDown),
        }
    }

    fn resolve_dependency(
        &self,
        dependency: ResourceDependency,
    ) -> Result<BackendResourceHandle, BackendError> {
        self.resources_by_dependency
            .get(&dependency)
            .copied()
            .ok_or(BackendError::UnknownResource(dependency))
    }

    fn validate_resource_info(&self, info: &BackendResourceCreateInfo) -> Result<(), BackendError> {
        match info {
            BackendResourceCreateInfo::Buffer {
                id,
                description,
                view: Some(view),
            } => {
                if view.buffer() != *id {
                    return Err(BackendError::InvalidResource(
                        BackendResourceValidationError::LogicalIdentityMismatch,
                    ));
                }
                BufferView::new(
                    *id,
                    *description,
                    view.buffer_offset(),
                    view.backing().clone(),
                )
                .map_err(|error| {
                    BackendError::InvalidResource(BackendResourceValidationError::BufferView(error))
                })?;
                self.validate_backing_view(view.backing())?;
            }
            BackendResourceCreateInfo::Image {
                id,
                description,
                view: Some(view),
            } => {
                if view.image() != *id {
                    return Err(BackendError::InvalidResource(
                        BackendResourceValidationError::LogicalIdentityMismatch,
                    ));
                }
                let bindings = view
                    .bindings()
                    .iter()
                    .map(|binding| {
                        (
                            binding.subresources(),
                            binding.layout(),
                            binding.backing().clone(),
                        )
                    })
                    .collect();
                ImageView::new(*id, *description, view.swizzle(), bindings).map_err(|error| {
                    BackendError::InvalidResource(BackendResourceValidationError::ImageView(error))
                })?;
                for binding in view.bindings() {
                    self.validate_backing_view(binding.backing())?;
                }
            }
            BackendResourceCreateInfo::Shader {
                description,
                module,
                ..
            } if description.stage != module.stage() => {
                return Err(BackendError::InvalidResource(
                    BackendResourceValidationError::ShaderStageMismatch {
                        description: description.stage,
                        module: module.stage(),
                    },
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_backing_view(&self, view: &crate::BackingView) -> Result<(), BackendError> {
        let dependency = ResourceDependency::Allocation(view.allocation());
        let handle = self.resolve_dependency(dependency)?;
        let record = self.validate_resource_handle(handle)?;
        let description = record
            .allocation_description
            .expect("allocation dependency has an allocation description");
        let end = view.allocation_offset().checked_add(view.size()).ok_or(
            BackendError::InvalidResource(BackendResourceValidationError::BackingRangeOverflow),
        )?;
        if end > description.size() {
            return Err(BackendError::InvalidResource(
                BackendResourceValidationError::BackingOutOfBounds {
                    allocation: view.allocation(),
                    offset: view.allocation_offset(),
                    size: view.size(),
                    allocation_size: description.size(),
                },
            ));
        }
        Ok(())
    }

    fn validate_resource_handle(
        &self,
        handle: BackendResourceHandle,
    ) -> Result<&ResourceRecord, BackendError> {
        if handle.instance != self.instance {
            return Err(BackendError::WrongInstance {
                expected: self.instance,
                observed: handle.instance,
            });
        }
        let Some(slot) = usize::try_from(handle.slot)
            .ok()
            .and_then(|slot| self.resources.get(slot))
        else {
            return Err(BackendError::StaleResource(handle));
        };
        if slot.generation != handle.generation {
            return Err(BackendError::StaleResource(handle));
        }
        let record = slot
            .record
            .as_ref()
            .ok_or(BackendError::StaleResource(handle))?;
        if dependency_kind(record.dependency) != handle.kind {
            return Err(BackendError::ResourceKindMismatch(handle));
        }
        Ok(record)
    }

    fn validate_submission_token(
        &self,
        token: BackendSubmissionToken,
    ) -> Result<&SubmissionRecord, BackendError> {
        if token.instance() != self.instance {
            return Err(BackendError::WrongInstance {
                expected: self.instance,
                observed: token.instance(),
            });
        }
        let Some(slot) = usize::try_from(token.slot())
            .ok()
            .and_then(|slot| self.submissions.get(slot))
        else {
            return Err(BackendError::StaleSubmission(token));
        };
        if slot.generation != token.generation() {
            return Err(BackendError::StaleSubmission(token));
        }
        slot.record
            .as_ref()
            .ok_or(BackendError::StaleSubmission(token))
    }

    fn handle_driver_error(&mut self, error: BackendDriverError) -> BackendError {
        match error {
            BackendDriverError::Failure(message) => BackendError::Driver(message),
            BackendDriverError::DeviceLost(message) => {
                self.state = BackendState::DeviceLost;
                self.device_loss_reason = Some(message.clone());
                self.clear_ownership();
                BackendError::DeviceLost(message)
            }
        }
    }

    fn clear_ownership(&mut self) {
        self.resources.clear();
        self.resources_by_dependency.clear();
        self.submissions.clear();
        self.accepted_frontends.clear();
    }
}

impl<D: BackendDriver> BackendCompletionSource for Backend<D> {
    fn has_completed(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<bool, BackendCompletionError> {
        Backend::has_completed(self, submission)
            .map_err(|error| BackendCompletionError::new(error.to_string()))
    }
}

const fn dependency_kind(dependency: ResourceDependency) -> BackendResourceKind {
    match dependency {
        ResourceDependency::Allocation(_) => BackendResourceKind::Allocation,
        ResourceDependency::Buffer(_) => BackendResourceKind::Buffer,
        ResourceDependency::Image(_) => BackendResourceKind::Image,
        ResourceDependency::Sampler(_) => BackendResourceKind::Sampler,
        ResourceDependency::Shader(_) => BackendResourceKind::Shader,
        ResourceDependency::Pipeline(_) => BackendResourceKind::Pipeline,
        ResourceDependency::DescriptorTable(_) => BackendResourceKind::DescriptorTable,
        ResourceDependency::RenderPass(_) => BackendResourceKind::RenderPass,
        ResourceDependency::QueryPool(_) => BackendResourceKind::QueryPool,
    }
}

fn next_resource_slot(slots: &[ResourceSlot]) -> Result<(u64, u32), BackendError> {
    if let Some((slot, state)) = slots
        .iter()
        .enumerate()
        .find(|(_, slot)| slot.record.is_none())
    {
        let generation = state
            .generation
            .checked_add(1)
            .ok_or(BackendError::GenerationExhausted)?;
        return Ok((slot as u64, generation));
    }
    Ok((slots.len() as u64, 1))
}

fn commit_resource_slot(
    slots: &mut Vec<ResourceSlot>,
    slot: u64,
    generation: u32,
    record: ResourceRecord,
) {
    let slot = slot as usize;
    if slot == slots.len() {
        slots.push(ResourceSlot {
            generation,
            record: Some(record),
        });
    } else {
        slots[slot] = ResourceSlot {
            generation,
            record: Some(record),
        };
    }
}

fn next_submission_slot(slots: &[SubmissionSlot]) -> Result<(u64, u32), BackendError> {
    if let Some((slot, state)) = slots
        .iter()
        .enumerate()
        .find(|(_, slot)| slot.record.is_none())
    {
        let generation = state
            .generation
            .checked_add(1)
            .ok_or(BackendError::GenerationExhausted)?;
        return Ok((slot as u64, generation));
    }
    Ok((slots.len() as u64, 1))
}

fn commit_submission_slot(
    slots: &mut Vec<SubmissionSlot>,
    slot: u64,
    generation: u32,
    record: SubmissionRecord,
) {
    let slot = slot as usize;
    if slot == slots.len() {
        slots.push(SubmissionSlot {
            generation,
            record: Some(record),
        });
    } else {
        slots[slot] = SubmissionSlot {
            generation,
            record: Some(record),
        };
    }
}

/// Typed failure at the neutral backend lifetime boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendError {
    Capability(BackendCapabilityError),
    InvalidResource(BackendResourceValidationError),
    DuplicateResource(ResourceDependency),
    UnknownResource(ResourceDependency),
    ResourceInUse(BackendResourceHandle),
    ResourceKindMismatch(BackendResourceHandle),
    StaleResource(BackendResourceHandle),
    DuplicateSubmission(FrontendSubmissionId),
    UnknownPredecessor(FrontendSubmissionId),
    StaleSubmission(BackendSubmissionToken),
    SubmissionIncomplete(BackendSubmissionToken),
    WrongInstance {
        expected: BackendInstanceId,
        observed: BackendInstanceId,
    },
    GenerationExhausted,
    ResourceExhausted,
    Driver(Box<str>),
    DeviceLost(Box<str>),
    TornDown,
}

impl Display for BackendError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capability(error) => error.fmt(formatter),
            Self::InvalidResource(error) => error.fmt(formatter),
            Self::DuplicateResource(resource) => write!(formatter, "duplicate {resource:?}"),
            Self::UnknownResource(resource) => write!(formatter, "unknown {resource:?}"),
            Self::ResourceInUse(handle) => write!(formatter, "resource is still in use: {handle}"),
            Self::ResourceKindMismatch(handle) => {
                write!(
                    formatter,
                    "resource handle kind does not match live slot: {handle}"
                )
            }
            Self::StaleResource(handle) => write!(formatter, "stale or destroyed {handle}"),
            Self::DuplicateSubmission(submission) => write!(formatter, "duplicate {submission}"),
            Self::UnknownPredecessor(submission) => {
                write!(formatter, "unknown submission predecessor: {submission}")
            }
            Self::StaleSubmission(submission) => {
                write!(formatter, "stale or released {submission}")
            }
            Self::SubmissionIncomplete(submission) => {
                write!(formatter, "cannot release incomplete {submission}")
            }
            Self::WrongInstance { expected, observed } => write!(
                formatter,
                "backend instance mismatch: expected {expected} observed {observed}"
            ),
            Self::GenerationExhausted => formatter.write_str("backend handle generation exhausted"),
            Self::ResourceExhausted => formatter.write_str("neutral backend bookkeeping exhausted"),
            Self::Driver(message) => write!(formatter, "backend operation failed: {message}"),
            Self::DeviceLost(message) => write!(formatter, "backend device lost: {message}"),
            Self::TornDown => formatter.write_str("backend instance has been torn down"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Failure while combining a neutral resource description and backing view at
/// the backend ownership boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendResourceValidationError {
    LogicalIdentityMismatch,
    BufferView(BufferViewError),
    ImageView(ImageViewError),
    ShaderStageMismatch {
        description: ShaderStage,
        module: ShaderStage,
    },
    BackingRangeOverflow,
    BackingOutOfBounds {
        allocation: GpuAllocationId,
        offset: u64,
        size: u64,
        allocation_size: u64,
    },
    DescriptorBindingCountOverflow,
}

impl Display for BackendResourceValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LogicalIdentityMismatch => formatter.write_str(
                "backend resource description and backing view name different logical resources",
            ),
            Self::BufferView(error) => write!(formatter, "invalid backend buffer view: {error}"),
            Self::ImageView(error) => write!(formatter, "invalid backend image view: {error}"),
            Self::ShaderStageMismatch {
                description,
                module,
            } => write!(
                formatter,
                "backend shader description stage {description:?} contradicts module stage {module:?}"
            ),
            Self::BackingRangeOverflow => {
                formatter.write_str("backend resource backing range overflows")
            }
            Self::BackingOutOfBounds {
                allocation,
                offset,
                size,
                allocation_size,
            } => write!(
                formatter,
                "backend resource backing exceeds live {allocation}: offset={offset:#x} size={size:#x} allocation-size={allocation_size:#x}"
            ),
            Self::DescriptorBindingCountOverflow => {
                formatter.write_str("descriptor binding count is not representable")
            }
        }
    }
}

impl std::error::Error for BackendResourceValidationError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        BackendFeatures, BackendLimits, BufferRange, BufferRegion, CopyOperation, GpuCommand,
        GpuOperation, ShaderInstruction, ShaderIr, ShaderOperation, ShaderPredicate,
        ShaderSourceLocation, VerifiedShaderIr, lower_shader_ir_to_wgsl,
    };

    #[derive(Default)]
    struct RecordingDriver {
        creates: Vec<BackendResourceHandle>,
        destroys: Vec<BackendResourceHandle>,
        submissions: Vec<BackendSubmissionToken>,
        completed: BTreeSet<BackendSubmissionToken>,
        next_error: Option<BackendDriverError>,
        teardown_count: usize,
    }

    impl RecordingDriver {
        fn result(&mut self) -> Result<(), BackendDriverError> {
            match self.next_error.take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
    }

    impl BackendDriver for RecordingDriver {
        fn create_resource(
            &mut self,
            handle: BackendResourceHandle,
            _info: &BackendResourceCreateInfo,
        ) -> Result<(), BackendDriverError> {
            self.result()?;
            self.creates.push(handle);
            Ok(())
        }

        fn destroy_resource(
            &mut self,
            handle: BackendResourceHandle,
        ) -> Result<(), BackendDriverError> {
            self.result()?;
            self.destroys.push(handle);
            Ok(())
        }

        fn submit(
            &mut self,
            submission: &AcceptedBackendSubmission<'_>,
        ) -> Result<(), BackendDriverError> {
            self.result()?;
            self.submissions.push(submission.token());
            Ok(())
        }

        fn has_completed(
            &mut self,
            submission: BackendSubmissionToken,
        ) -> Result<bool, BackendDriverError> {
            self.result()?;
            Ok(self.completed.contains(&submission))
        }

        fn release_submission(
            &mut self,
            _submission: BackendSubmissionToken,
        ) -> Result<(), BackendDriverError> {
            self.result()
        }

        fn teardown(&mut self) -> Result<(), BackendDriverError> {
            self.result()?;
            self.teardown_count += 1;
            Ok(())
        }
    }

    fn capabilities() -> BackendCapabilities {
        BackendCapabilities::new(
            BackendFeatures::COPY,
            [],
            [],
            [],
            [],
            BackendLimits {
                max_color_attachments: 0,
                max_descriptor_bindings: 0,
                max_compute_workgroups: [0; 3],
            },
        )
    }

    fn backend(instance: u64) -> Backend<RecordingDriver> {
        Backend::new(
            BackendInstanceId::new(instance),
            capabilities(),
            RecordingDriver::default(),
        )
    }

    fn buffer_info(id: u64) -> BackendResourceCreateInfo {
        BackendResourceCreateInfo::Buffer {
            id: BufferId::new(id),
            description: BufferDescription::new(64).unwrap(),
            view: None,
        }
    }

    fn copy_submission(id: u64, source: u64, destination: u64) -> OperationSubmission {
        let range = BufferRange::new(0, 16).unwrap();
        let copy = CopyOperation::buffer_to_buffer(
            BufferRegion {
                buffer: BufferId::new(source),
                range,
            },
            BufferRegion {
                buffer: BufferId::new(destination),
                range,
            },
        )
        .unwrap();
        let operation = GpuOperation::new(
            GpuCommand::Copy(copy),
            [],
            [],
            CapabilityRequirements::none(),
        );
        OperationSubmission::new(FrontendSubmissionId::new(id), Vec::new(), vec![operation])
            .unwrap()
    }

    #[test]
    fn shader_description_cannot_contradict_verified_module_stage() {
        let ir = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![ShaderInstruction::new(
                ShaderSourceLocation::new(0),
                ShaderPredicate::Always,
                ShaderOperation::Exit,
            )],
        ))
        .unwrap();
        let module = lower_shader_ir_to_wgsl(&ir).unwrap();
        let mut backend = backend(1);
        assert_eq!(
            backend.create_resource(BackendResourceCreateInfo::Shader {
                id: ShaderId::new(1),
                description: ShaderDescription {
                    stage: ShaderStage::Fragment,
                },
                module,
            }),
            Err(BackendError::InvalidResource(
                BackendResourceValidationError::ShaderStageMismatch {
                    description: ShaderStage::Fragment,
                    module: ShaderStage::Vertex,
                }
            ))
        );
        assert!(backend.driver().creates.is_empty());
    }

    #[test]
    fn stale_cross_instance_and_use_after_destroy_fail_before_driver_observation() {
        let mut backend = backend(1);
        let first = backend.create_resource(buffer_info(1)).unwrap();
        backend.destroy_resource(first).unwrap();
        assert_eq!(
            backend.destroy_resource(first),
            Err(BackendError::StaleResource(first))
        );
        let second = backend.create_resource(buffer_info(2)).unwrap();
        assert_eq!(second.slot(), first.slot());
        assert_eq!(second.generation(), first.generation() + 1);

        let foreign = BackendResourceHandle::new(
            BackendInstanceId::new(2),
            second.slot(),
            second.generation(),
            second.kind(),
        );
        assert!(matches!(
            backend.destroy_resource(foreign),
            Err(BackendError::WrongInstance { .. })
        ));
        assert_eq!(backend.driver().destroys, vec![first]);
    }

    #[test]
    fn malformed_submission_is_rejected_atomically_before_driver_observation() {
        let mut backend = backend(1);
        backend.create_resource(buffer_info(1)).unwrap();
        let submission = copy_submission(1, 1, 2);
        assert_eq!(
            backend.submit(&submission),
            Err(BackendError::UnknownResource(ResourceDependency::Buffer(
                BufferId::new(2)
            )))
        );
        assert!(backend.driver().submissions.is_empty());
    }

    #[test]
    fn completion_does_not_release_resources_or_publish_guest_progress() {
        let mut backend = backend(1);
        let source = backend.create_resource(buffer_info(1)).unwrap();
        backend.create_resource(buffer_info(2)).unwrap();
        let token = backend.submit(&copy_submission(1, 1, 2)).unwrap();
        let foreign = BackendSubmissionToken::new(
            BackendInstanceId::new(2),
            token.slot(),
            token.generation(),
        );
        assert!(matches!(
            backend.has_completed(foreign),
            Err(BackendError::WrongInstance { .. })
        ));
        assert_eq!(backend.has_completed(token), Ok(false));
        assert_eq!(
            backend.release_submission(token),
            Err(BackendError::SubmissionIncomplete(token))
        );
        assert_eq!(
            backend.destroy_resource(source),
            Err(BackendError::ResourceInUse(source))
        );

        backend.driver.completed.insert(token);
        assert_eq!(backend.has_completed(token), Ok(true));
        backend.release_submission(token).unwrap();
        backend.destroy_resource(source).unwrap();
        let replacement = backend.submit(&copy_submission(2, 2, 2)).unwrap();
        assert_eq!(replacement.slot(), token.slot());
        assert_eq!(replacement.generation(), token.generation() + 1);
        assert_eq!(
            backend.has_completed(token),
            Err(BackendError::StaleSubmission(token))
        );
    }

    #[test]
    fn device_loss_is_terminal_and_discards_neutral_ownership() {
        let mut backend = backend(1);
        backend.create_resource(buffer_info(1)).unwrap();
        backend.driver.next_error = Some(BackendDriverError::device_lost("removed"));
        assert_eq!(
            backend.create_resource(buffer_info(2)),
            Err(BackendError::DeviceLost("removed".into()))
        );
        assert_eq!(backend.state(), BackendState::DeviceLost);
        assert_eq!(backend.device_loss_reason(), Some("removed"));
        assert_eq!(
            backend.create_resource(buffer_info(3)),
            Err(BackendError::DeviceLost("removed".into()))
        );
        assert_eq!(backend.driver().creates.len(), 1);
    }

    #[test]
    fn explicit_teardown_is_idempotent_and_terminal() {
        let mut backend = backend(1);
        backend.create_resource(buffer_info(1)).unwrap();
        backend.teardown().unwrap();
        backend.teardown().unwrap();
        assert_eq!(backend.state(), BackendState::TornDown);
        assert_eq!(backend.driver().teardown_count, 1);
        assert_eq!(
            backend.create_resource(buffer_info(2)),
            Err(BackendError::TornDown)
        );
    }
}
