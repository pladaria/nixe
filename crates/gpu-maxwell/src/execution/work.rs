//! The command processor preserves ordering; the concrete lowerer owns its work representation.
use super::*;
use nixe_gpu::ExecutionBatch;

pub trait MaxwellWork: Send + 'static {
    type Batch: MaxwellWorkBatch<Self>;
    type ImageRequest: Send + 'static;
    fn frontend(&self) -> FrontendSubmissionId;
}
pub trait MaxwellWorkSegment: Send {
    type Commands;
    fn execution_batch(&self) -> ExecutionBatch<'_, Self::Commands>;
    fn is_final_segment(&self) -> bool;
}
pub trait MaxwellWorkBatch<W: MaxwellWork + ?Sized>: Default + Send {
    type Segment: MaxwellWorkSegment;
    fn wait_for_idle(&mut self);
    fn write_memory(
        &mut self,
        target: nixe_memory::CanonicalBackingRange,
        offset: u64,
        bytes: &[u8],
        after_prior_work: bool,
    );
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn break_rendering(&mut self);
    fn ordering(&mut self, operation: &CacheMaintenanceOperation);
    fn work(&mut self, work: &W);
    fn finish(
        &mut self,
        frontend: FrontendSubmissionId,
        segment: FrontendSubmissionSegment,
        final_segment: bool,
    ) -> Result<Self::Segment, MaxwellBackendExecutionError>;
    /// Called after the backend accepts the commands and retains their uses.
    /// The recording container can then be reused before GPU completion.
    fn recycle_segment(&mut self, segment: Self::Segment) {
        drop(segment);
    }
}

#[derive(Default)]
pub struct NativeWorkBatch {
    commands: Vec<nixe_gpu::native::NativeCommand>,
}
impl MaxwellWork for nixe_gpu::native::NativeWork {
    type Batch = NativeWorkBatch;
    type ImageRequest = nixe_gpu::native::NativePresentationImage;
    fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }
}
impl MaxwellWorkSegment for nixe_gpu::native::NativeSubmission {
    type Commands = Self;
    fn execution_batch(&self) -> nixe_gpu::ExecutionBatch<'_, Self> {
        nixe_gpu::ExecutionBatch { commands: self }
    }
    fn is_final_segment(&self) -> bool {
        self.final_segment
    }
}
impl MaxwellWorkBatch<nixe_gpu::native::NativeWork> for NativeWorkBatch {
    type Segment = nixe_gpu::native::NativeSubmission;
    fn wait_for_idle(&mut self) {
        self.commands
            .push(nixe_gpu::native::NativeCommand::WaitForIdle);
    }
    fn write_memory(
        &mut self,
        target: nixe_memory::CanonicalBackingRange,
        offset: u64,
        bytes: &[u8],
        after_prior_work: bool,
    ) {
        self.commands
            .push(nixe_gpu::native::NativeCommand::WriteMemory {
                target,
                offset,
                bytes: Arc::from(bytes),
                after_prior_work,
            });
    }
    fn len(&self) -> usize {
        self.commands.len()
    }
    fn break_rendering(&mut self) {}
    fn ordering(&mut self, op: &CacheMaintenanceOperation) {
        self.commands
            .push(nixe_gpu::native::NativeCommand::Ordering(*op));
    }
    fn work(&mut self, work: &nixe_gpu::native::NativeWork) {
        self.commands.push(work.command.clone());
    }
    fn finish(
        &mut self,
        frontend: FrontendSubmissionId,
        segment: FrontendSubmissionSegment,
        final_segment: bool,
    ) -> Result<Self::Segment, MaxwellBackendExecutionError> {
        Ok(nixe_gpu::native::NativeSubmission {
            frontend,
            segment,
            final_segment,
            commands: std::mem::take(&mut self.commands),
        })
    }

    fn recycle_segment(&mut self, mut segment: Self::Segment) {
        debug_assert!(self.commands.is_empty());
        // submit() has already captured the native uses. Drop command leases,
        // retain the vector capacity, and never replay the accepted commands.
        segment.commands.clear();
        self.commands = segment.commands;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::native::{GuestProgram, GuestShaderCode, NativeCommand, NativeWork};

    #[test]
    fn accepted_native_segments_release_command_leases_and_reuse_empty_storage() {
        let frontend = FrontendSubmissionId::new(7);
        let program = Arc::new(GuestProgram {
            identity: nixe_gpu::PreparedStateId::new(1, 1),
            shader: Arc::new(GuestShaderCode {
                identity: nixe_gpu::shader_identity::ContentId::of(b"batch-retirement-test", &[]),
                stages: Box::new([]),
            }),
            bindings: Box::new([]),
            shared_memory_bytes: 0,
        });
        let mut batch = NativeWorkBatch::default();
        {
            let work = NativeWork {
                frontend,
                command: NativeCommand::Dispatch {
                    program: Arc::clone(&program),
                    groups: [3, 2, 1],
                },
            };
            batch.work(&work);
        }
        let first = batch
            .finish(frontend, FrontendSubmissionSegment::FIRST, false)
            .unwrap();
        let storage = first.commands.as_ptr();
        assert_eq!(first.commands.len(), 1);
        assert_eq!(Arc::strong_count(&program), 2);
        batch.recycle_segment(first);
        assert_eq!(Arc::strong_count(&program), 1);
        assert!(batch.is_empty());

        let operation = nixe_gpu::CacheMaintenanceOperation::InvalidateDeviceReadCaches;
        batch.ordering(&operation);
        let second = batch
            .finish(
                frontend,
                FrontendSubmissionSegment::FIRST.checked_next().unwrap(),
                true,
            )
            .unwrap();
        assert_eq!(second.commands.as_ptr(), storage);
        assert!(second.final_segment);
        assert!(matches!(
            second.commands.as_slice(),
            [NativeCommand::Ordering(
                nixe_gpu::CacheMaintenanceOperation::InvalidateDeviceReadCaches
            )]
        ));
        assert_eq!(Arc::strong_count(&program), 1);
    }
}
