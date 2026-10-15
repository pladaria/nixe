//! Direct execution contract for a concrete GPU owner.
//!
//! Select and move the renderer into its worker at startup. This trait is used
//! with static dispatch; it does not require a validated backend/driver stack.

use std::sync::Arc;
use std::time::Duration;

use nixe_memory::CpuVisibilityRequest;

use crate::{
    BackendExecutionCompletion, BackendRuntimeError, BackendSubmissionToken,
    BackendVisibilityRequester, NeutralBackendRuntime, OperationSubmission,
    PresentationImageRequest, ResidentImage, ResourceDefinition, ResourceDependency,
};

/// Frontend-owned identity of an immutable prepared state revision.
///
/// Slots may be reused only with a new revision. IDs are scoped to the execution
/// session, never guest register encodings or host object handles.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PreparedStateId {
    slot: u64,
    revision: u64,
}

impl PreparedStateId {
    #[must_use]
    pub const fn new(slot: u64, revision: u64) -> Self {
        Self { slot, revision }
    }
    #[must_use]
    pub const fn slot(self) -> u64 {
        self.slot
    }
    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }
}

/// One ordered batch borrowing its immutable frontend-owned data.
///
/// Acceptance must capture input snapshots and retain canonical ranges needed
/// by asynchronous execution. A successful return is not guest completion.
/// Resource retirement is ordered after this batch's uses; no transactional
/// resource-creation rollback is part of this contract.
#[derive(Clone, Copy)]
pub struct ExecutionBatch<'a, C = OperationSubmission> {
    pub definitions: &'a [ResourceDefinition],
    pub retirements: &'a [ResourceDependency],
    pub commands: &'a C,
}

impl<'a> ExecutionBatch<'a> {
    /// Shader/resource effects in command order; the canonical access summary
    /// is not a replacement for this stream when computing host hazards.
    pub fn ordered_accesses(self) -> impl Iterator<Item = (usize, &'a crate::ResourceAccess)> {
        self.commands
            .operations()
            .iter()
            .enumerate()
            .flat_map(|(index, operation)| {
                operation
                    .accesses()
                    .iter()
                    .map(move |access| (index, access))
            })
    }
}

/// Interface consumed by Horizon's statically instantiated owner loop.
///
/// Access descriptions express guest effects in operation order. Canonical CPU
/// visibility is coordinated by nixe-memory; host hazards, queue ticks and
/// Vulkan barriers belong to the renderer. Poll/wait report ordered completion
/// only after host work and required canonical ownership transitions finish.
pub trait GpuExecution: Send + 'static {
    type Commands: Send + 'static;
    type ImageRequest: Send + 'static;
    /// Host control is serviced by the same concrete owner, outside draw loops.
    fn bind_host_wakeup(&mut self, _wake: Arc<dyn Fn() + Send + Sync>) {}
    fn service_host(&mut self) -> Result<(), BackendRuntimeError> {
        Ok(())
    }
    fn host_work_pending(&self) -> bool {
        false
    }
    fn external_image_device(&self) -> Option<[u8; 16]> {
        None
    }
    /// Poll cold preparation without waiting on compiler workers. No input
    /// acceptance or guest completion occurs until this returns true.
    fn prepare(
        &mut self,
        _batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<bool, BackendRuntimeError> {
        Ok(true)
    }
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_, Self::Commands>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError>;
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    /// Progresses outstanding work even when no later batch arrives.
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    /// A bounded GPU wait lets the owner revisit its control inbox. Timeout
    /// returns None and must never publish completion or release admission.
    fn wait_for_completion_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError>;
    fn bind_visibility_requester(
        &mut self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError>;
    /// The owner resolves its own requests directly, never through its inbox.
    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError>;
    fn acquire_presentable_image(
        &mut self,
        request: Self::ImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError>;
    /// Drains accepted host uses; cancels unaccepted work without guest success.
    fn teardown(&mut self) -> Result<(), BackendRuntimeError>;
}

/// Sole temporary seam to the old object-safe runtime. Delete in task 15.
///
/// The Vulkan renderer implements GpuExecution directly. It must not reuse this
/// bridge, the generic Backend<D>, or the legacy creation/rollback machinery.
pub struct LegacyGpuExecution(Box<dyn NeutralBackendRuntime>);

impl LegacyGpuExecution {
    #[must_use]
    pub fn new(runtime: Box<dyn NeutralBackendRuntime>) -> Self {
        Self(runtime)
    }
}

impl GpuExecution for LegacyGpuExecution {
    type Commands = OperationSubmission;
    type ImageRequest = PresentationImageRequest;
    fn external_image_device(&self) -> Option<[u8; 16]> {
        self.0.external_image_device()
    }
    fn submit(
        &mut self,
        batch: ExecutionBatch<'_>,
    ) -> Result<BackendSubmissionToken, BackendRuntimeError> {
        self.0
            .submit(batch.definitions, batch.retirements, batch.commands)
    }
    fn poll_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.0.poll_completion()
    }
    fn wait_for_completion(
        &mut self,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        self.0.wait_for_completion()
    }
    fn wait_for_completion_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<BackendExecutionCompletion>, BackendRuntimeError> {
        // Only the retiring oracle lacks a timed host wait. Do not block its
        // owner indefinitely; this adapter disappears with the legacy runtime.
        if let Some(completion) = self.0.poll_completion()? {
            return Ok(Some(completion));
        }
        std::thread::sleep(timeout);
        self.0.poll_completion()
    }
    fn bind_visibility_requester(
        &mut self,
        requester: Arc<dyn BackendVisibilityRequester>,
    ) -> Result<(), BackendRuntimeError> {
        self.0.bind_visibility_requester(requester)
    }
    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendRuntimeError> {
        self.0.make_cpu_visible(request)
    }
    fn acquire_presentable_image(
        &mut self,
        request: PresentationImageRequest,
    ) -> Result<ResidentImage, BackendRuntimeError> {
        self.0.acquire_presentable_image(request)
    }
    fn teardown(&mut self) -> Result<(), BackendRuntimeError> {
        self.0.teardown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BufferId, BufferRange, BufferRegion, CapabilityRequirements, CopyOperation, DrawArguments,
        DrawOperation, FrontendSubmissionId, GpuCommand, GpuOperation, PipelineId, PreparedDraw,
        PrimitiveTopology, RenderPassId,
    };

    #[test]
    fn ordered_accesses_keep_write_then_read_even_when_canonical_summary_merges_them() {
        let region = |id| BufferRegion {
            buffer: BufferId::new(id),
            range: BufferRange::new(0, 16).unwrap(),
        };
        let copy = |source, destination| {
            GpuOperation::new(
                GpuCommand::Copy(CopyOperation::BufferToBuffer {
                    source: region(source),
                    destination: region(destination),
                }),
                [],
                [],
                CapabilityRequirements::none(),
            )
        };
        let commands = OperationSubmission::new(
            FrontendSubmissionId::new(1),
            vec![],
            vec![copy(1, 2), copy(2, 3)],
        )
        .unwrap();
        let batch = ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: &commands,
        };
        let effects: Vec<_> = batch
            .ordered_accesses()
            .filter(|(_, access)| {
                access.target().dependency() == ResourceDependency::Buffer(BufferId::new(2))
            })
            .map(|(index, access)| (index, access.scope().mode()))
            .collect();
        assert_eq!(
            effects,
            [(0, crate::AccessMode::Write), (1, crate::AccessMode::Read)]
        );
    }

    #[test]
    fn queued_prepared_revisions_stay_immutable_and_warm_draws_share_the_record() {
        let mut state = PreparedDraw::new(
            PreparedStateId::new(8, 1),
            PipelineId::new(1),
            RenderPassId::new(1),
            PrimitiveTopology::Triangles,
            vec![],
            vec![],
            None,
        )
        .unwrap();
        let first_id = state.identity();
        state.edit_state().cull_mode = crate::CullMode::Back;
        assert_ne!(
            state.identity(),
            first_id,
            "mutable preparation must invalidate the earlier revision"
        );
        let mut prepared = Arc::new(state);
        let arguments = |first_vertex| DrawArguments::NonIndexed {
            first_vertex,
            vertex_count: 3,
            first_instance: 0,
            instance_count: 1,
        };
        let original = GpuOperation::new(
            GpuCommand::Draw(DrawOperation::new(Arc::clone(&prepared), arguments(0)).unwrap()),
            [],
            [],
            CapabilityRequirements::none(),
        );
        let warm = original.with_draw_arguments(arguments(3)).unwrap();
        let GpuCommand::Draw(first) = original.command() else {
            panic!()
        };
        let GpuCommand::Draw(next) = warm.command() else {
            panic!()
        };
        assert!(Arc::ptr_eq(&first.prepared, &next.prepared));
        assert_eq!(original.retained_bytes(), warm.retained_bytes());
        assert!(Arc::get_mut(&mut prepared).is_none());
        let mut changed = PreparedDraw::from_state(
            PreparedStateId::new(
                prepared.identity().slot(),
                prepared.identity().revision() + 1,
            ),
            prepared.state().clone(),
        );
        changed.edit_state().pipeline = PipelineId::new(2);
        assert_ne!(changed.identity(), first.prepared.identity());
        assert_eq!(first.prepared.pipeline, PipelineId::new(1));
        assert_eq!(next.arguments, arguments(3));
    }
}
