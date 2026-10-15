//! The command processor preserves ordering; the concrete lowerer owns its work representation.
use super::*;
use nixe_gpu::ExecutionBatch;

pub trait MaxwellWork: Send + 'static {
    type Batch: MaxwellWorkBatch<Self>;
    type ImageRequest: Send + 'static;
    fn retained_bytes(&self) -> usize;
    fn canonical_accesses(&self) -> (&[nixe_gpu::BackingView], bool);
    fn frontend(&self) -> FrontendSubmissionId;
}
pub trait MaxwellWorkSegment: Send {
    type Commands;
    fn execution_batch(&self) -> ExecutionBatch<'_, Self::Commands>;
    fn is_final_segment(&self) -> bool;
}
pub trait MaxwellWorkBatch<W: MaxwellWork + ?Sized>: Default + Send {
    type Segment: MaxwellWorkSegment;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn break_rendering(&mut self);
    fn ordering(&mut self, operation: &GpuOperation);
    fn work(&mut self, work: &W);
    fn finish(
        &mut self,
        frontend: FrontendSubmissionId,
        segment: FrontendSubmissionSegment,
        final_segment: bool,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Segment, MaxwellBackendExecutionError>;
}

#[derive(Default)]
pub struct LegacyWorkBatch {
    creations: Vec<ResourceDefinition>,
    invalidations: Vec<ResourceDependency>,
    operations: Vec<GpuOperation>,
    render_begin: Option<usize>,
}
impl MaxwellWork for MaxwellLoweredWork {
    type Batch = LegacyWorkBatch;
    type ImageRequest = nixe_gpu::PresentationImageRequest;
    fn retained_bytes(&self) -> usize {
        self.retained_bytes()
    }
    fn canonical_accesses(&self) -> (&[nixe_gpu::BackingView], bool) {
        self.canonical_accesses()
    }
    fn frontend(&self) -> FrontendSubmissionId {
        self.submission().id()
    }
}
impl MaxwellWorkSegment for MaxwellBackendSegment {
    type Commands = OperationSubmission;
    fn execution_batch(&self) -> ExecutionBatch<'_, Self::Commands> {
        ExecutionBatch {
            definitions: self.creations(),
            retirements: self.invalidations(),
            commands: self.submission(),
        }
    }
    fn is_final_segment(&self) -> bool {
        self.submission.is_final_segment()
    }
}
impl MaxwellWorkBatch<MaxwellLoweredWork> for LegacyWorkBatch {
    type Segment = MaxwellBackendSegment;
    fn len(&self) -> usize {
        self.operations.len()
    }
    fn break_rendering(&mut self) {
        self.render_begin = None;
    }
    fn ordering(&mut self, operation: &GpuOperation) {
        self.operations.push(operation.clone());
    }
    fn work(&mut self, work: &MaxwellLoweredWork) {
        self.creations.extend_from_slice(work.resource_creations());
        self.invalidations
            .extend_from_slice(work.resource_invalidations());
        append_batchable_operations(
            &mut self.operations,
            &mut self.render_begin,
            work.submission().operations(),
        );
    }
    fn finish(
        &mut self,
        frontend: FrontendSubmissionId,
        segment: FrontendSubmissionSegment,
        final_segment: bool,
        predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Segment, MaxwellBackendExecutionError> {
        let submission = OperationSubmission::new_segment(
            frontend,
            segment,
            final_segment,
            predecessors.to_vec(),
            std::mem::take(&mut self.operations),
        )
        .map_err(MaxwellBackendExecutionError::InvalidSubmission)?;
        self.render_begin = None;
        Ok(MaxwellBackendSegment {
            creations: std::mem::take(&mut self.creations).into_boxed_slice(),
            invalidations: std::mem::take(&mut self.invalidations).into_boxed_slice(),
            submission,
        })
    }
}

#[derive(Default)]
pub struct NativeWorkBatch {
    commands: Vec<nixe_gpu::native::NativeCommand>,
}
impl MaxwellWork for nixe_gpu::native::NativeWork {
    type Batch = NativeWorkBatch;
    type ImageRequest = nixe_gpu::native::NativePresentationImage;
    fn retained_bytes(&self) -> usize {
        self.retained_bytes()
    }
    fn canonical_accesses(&self) -> (&[nixe_gpu::BackingView], bool) {
        (&self.canonical_accesses, false)
    }
    fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }
}
impl MaxwellWorkSegment for nixe_gpu::native::NativeSubmission {
    type Commands = Self;
    fn execution_batch(&self) -> nixe_gpu::ExecutionBatch<'_, Self> {
        nixe_gpu::ExecutionBatch {
            definitions: &[],
            retirements: &[],
            commands: self,
        }
    }
    fn is_final_segment(&self) -> bool {
        self.final_segment
    }
}
impl MaxwellWorkBatch<nixe_gpu::native::NativeWork> for NativeWorkBatch {
    type Segment = nixe_gpu::native::NativeSubmission;
    fn len(&self) -> usize {
        self.commands.len()
    }
    fn break_rendering(&mut self) {}
    fn ordering(&mut self, op: &GpuOperation) {
        self.commands.push(match op.command() {
            GpuCommand::CacheMaintenance(c) => nixe_gpu::native::NativeCommand::Ordering(*c),
            GpuCommand::Barrier(b) => nixe_gpu::native::NativeCommand::Barrier(b.clone()),
            _ => unreachable!("command processor emits only ordering commands"),
        });
    }
    fn work(&mut self, work: &nixe_gpu::native::NativeWork) {
        self.commands.push(work.command.clone());
    }
    fn finish(
        &mut self,
        frontend: FrontendSubmissionId,
        segment: FrontendSubmissionSegment,
        final_segment: bool,
        _predecessors: &[FrontendSubmissionId],
    ) -> Result<Self::Segment, MaxwellBackendExecutionError> {
        Ok(nixe_gpu::native::NativeSubmission {
            frontend,
            segment,
            final_segment,
            commands: std::mem::take(&mut self.commands),
        })
    }
}
