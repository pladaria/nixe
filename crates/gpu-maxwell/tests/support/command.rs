use nixe_gpu::native::NativeWork;
use nixe_gpu_maxwell::{MaxwellCommandHost, MaxwellSubmissionExecutionError};
use nixe_memory::{AccessOperation, AccessResult, AllocationRange, Machine, MemoryError};
use std::sync::Arc;

pub struct Observer<'a> {
    pub memory: &'a Machine,
    pub consume: &'a mut dyn FnMut(NativeWork),
    pub controls: usize,
}
impl MaxwellCommandHost for Observer<'_> {
    async fn emit(&mut self, work: NativeWork) -> Result<(), MaxwellSubmissionExecutionError> {
        (self.consume)(work);
        Ok(())
    }
    unsafe fn memory_written(
        &mut self,
        _: nixe_memory::MachineId,
        _: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
        unreachable!("fixture does not produce native memory")
    }
    fn access(
        &mut self,
        ranges: Vec<AllocationRange>,
        operation: AccessOperation,
    ) -> Result<AccessResult, Arc<MemoryError>> {
        self.memory
            .apply_resolved_access(&ranges, &operation)
            .map_err(Arc::new)
    }
    fn control(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        self.controls += 1;
        Ok(())
    }
    fn timestamp(&self) -> u64 {
        0
    }
}

pub fn ready<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => panic!("fixture unexpectedly suspended"),
    }
}
