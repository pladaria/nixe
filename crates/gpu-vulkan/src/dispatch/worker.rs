//! Recycled native jobs, containing no guest-memory references.
use super::*;
use super::{arena::Arena, slot::CommandSlot};
use crate::queue::progress::{NativeFailure, SubmissionProgress};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

pub(crate) struct QueueSubmission {
    pub queue: vk::Queue,
    pub waits: smallvec::SmallVec<[vk::SemaphoreSubmitInfo<'static>; 1]>,
    pub signals: smallvec::SmallVec<[vk::SemaphoreSubmitInfo<'static>; 2]>,
    pub tick: u64,
    pub progress: Arc<SubmissionProgress>,
}
// SAFETY: submit entries contain native handles and scalar parameters. The
// enqueue boundary rejects pNext; each parameter array is owned by this job.
unsafe impl Send for QueueSubmission {}
struct WorkerScope {
    recording: Arc<RecordingState>,
    failure: Arc<OnceLock<NativeFailure>>,
    active: Option<(Arc<SubmissionProgress>, u64)>,
}
impl WorkerScope {
    fn activate(&mut self, progress: &Arc<SubmissionProgress>, tick: u64) -> bool {
        self.active = Some((Arc::clone(progress), tick));
        if let Some(failure) = progress.failure().filter(|f| tick >= f.tick) {
            let _ = self.failure.set(failure.cause);
        }
        if progress.cancelled() && tick > progress.submitted() {
            self.fail(NativeFailure::Stopped);
        }
        if let Some(cause) = self.failure.get() {
            progress.fail(tick, cause.clone());
            return false;
        }
        true
    }
    fn fail(&self, cause: NativeFailure) {
        if let Some((progress, tick)) = &self.active {
            progress.fail(*tick, cause.clone());
        }
        let _ = self.failure.set(cause);
    }
}
impl Drop for WorkerScope {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Some((progress, tick)) = &self.active {
                let failed = (*tick).max(progress.submitted().saturating_add(1));
                progress.fail(failed, NativeFailure::Panicked);
                let _ = self.failure.set(progress.failure().unwrap().cause);
            } else {
                let _ = self.failure.set(NativeFailure::Panicked);
            }
        }
        // A stopped or unwinding worker must wake drains even when queued
        // chunks remain. Abandoned captures are destroyed without execution.
        let pending = {
            let mut queue = self
                .recording
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            queue.running = false;
            std::mem::take(&mut queue.pending)
        };
        self.recording.event.notify_all();
        drop(pending);
    }
}
struct Begin {
    slot: Arc<CommandSlot>,
    tick: u64,
    progress: Arc<SubmissionProgress>,
    timeline: vk::Semaphore,
}
#[derive(Default)]
struct Chunk {
    begin: Option<Begin>,
    commands: Arena,
    submit: Option<QueueSubmission>,
    #[cfg(feature = "performance-counters")]
    queued_at: Option<std::time::Instant>,
}
impl Chunk {
    fn len(&self) -> usize {
        self.commands.len() + usize::from(self.begin.is_some()) + usize::from(self.submit.is_some())
    }
    fn clear(&mut self) {
        self.begin = None;
        self.commands.clear();
        self.submit = None;
    }
}
#[derive(Clone)]
pub(crate) struct RecordingHandle {
    inner: Arc<RecordingState>,
}
struct WorkQueue {
    pending: VecDeque<Box<Chunk>>,
    stopped: bool,
    running: bool,
}
struct RecordingState {
    queue: Mutex<WorkQueue>,
    event: Condvar,
    execution: Mutex<()>,
    // Stable unique chunk ownership follows Scheduler::chunk_reserve; moving
    // its box between the queue and reserve never copies command storage.
    #[allow(clippy::vec_box)]
    reserve: Mutex<Vec<Box<Chunk>>>,
    failure: Arc<OnceLock<NativeFailure>>,
}
pub(super) struct Recorder {
    pub(super) handle: RecordingHandle,
    chunk: Option<Box<Chunk>>,
    worker: Option<JoinHandle<()>>,
}
impl Recorder {
    pub(super) fn new(
        device: Device,
        functions: Functions,
        queue_mutex: Arc<Mutex<()>>,
    ) -> Result<Self, VulkanError> {
        let state = Arc::new(RecordingState {
            queue: Mutex::new(WorkQueue {
                pending: VecDeque::new(),
                stopped: false,
                running: true,
            }),
            event: Condvar::new(),
            execution: Mutex::new(()),
            reserve: Mutex::new(Vec::new()),
            failure: Arc::new(OnceLock::new()),
        });
        let worker_state = Arc::clone(&state);
        let worker = thread::Builder::new()
            .name("nixe-vk-record".into())
            .spawn(move || {
                let mut scope = WorkerScope {
                    recording: Arc::clone(&worker_state),
                    failure: Arc::clone(&worker_state.failure),
                    active: None,
                };
                let mut context = Context {
                    device,
                    functions,
                    command: vk::CommandBuffer::null(),
                    upload: vk::CommandBuffer::null(),
                    slot: None,
                    descriptors: crate::cache::DescriptorAllocator::default(),
                    descriptor_pools: Default::default(),
                    progress: None,
                    timeline: vk::Semaphore::null(),
                };
                // Scheduler::WorkerThread takes execution ownership before
                // releasing the queue lock. A drained queue then also admits
                // a precise WaitWorker barrier through the execution mutex.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp#L270-L322
                loop {
                    let queue = worker_state.queue.lock().unwrap();
                    let mut queue = worker_state
                        .event
                        .wait_while(queue, |queue| queue.pending.is_empty() && !queue.stopped)
                        .unwrap();
                    if queue.stopped {
                        break;
                    }
                    let mut chunk = queue.pending.pop_front().expect("queued native chunk");
                    worker_state.event.notify_all();
                    let execution = worker_state.execution.lock().unwrap();
                    drop(queue);
                    #[cfg(feature = "performance-counters")]
                    nixe_gpu::metrics::record(
                        nixe_gpu::metrics::Counter::NativeChunkHandoffNanoseconds,
                        chunk
                            .queued_at
                            .take()
                            .expect("dispatched chunk time")
                            .elapsed()
                            .as_nanos() as u64,
                    );
                    record_chunk(&mut scope, &mut context, &queue_mutex, &mut chunk);
                    drop(execution);
                    worker_state.reserve.lock().unwrap().push(chunk);
                }
            })
            .map_err(|error| {
                VulkanError::Loader(format!("Vulkan recording worker: {error}").into())
            })?;
        Ok(Self {
            handle: RecordingHandle { inner: state },
            chunk: Some(Box::default()),
            worker: Some(worker),
        })
    }
    pub(crate) fn begin(
        &mut self,
        slot: Arc<CommandSlot>,
        tick: u64,
        progress: Arc<SubmissionProgress>,
        timeline: vk::Semaphore,
    ) -> Result<(), VulkanError> {
        self.handle.check_failure()?;
        let chunk = self
            .chunk
            .as_mut()
            .ok_or_else(RecordingHandle::disconnected)?;
        debug_assert_eq!(chunk.len(), 0);
        chunk.begin = Some(Begin {
            slot,
            tick,
            progress,
            timeline,
        });
        Ok(())
    }
    pub(crate) fn submit(&mut self, work: QueueSubmission) -> Result<(), VulkanError> {
        self.handle.check_failure()?;
        let chunk = self
            .chunk
            .as_mut()
            .ok_or_else(RecordingHandle::disconnected)?;
        debug_assert!(chunk.submit.is_none());
        chunk.submit = Some(work);
        self.flush_chunks()
    }
    pub(super) fn record<F>(&mut self, function: F)
    where
        F: Fn(&mut Context) -> Result<(), NativeFailure> + Send + 'static,
    {
        // SAFETY: all three arrays are empty and the owned closure is Send.
        unsafe {
            self.record_arrays(&[], &[], &[], move |ctx, _: &[()], _: &[()], _: &[()]| {
                function(ctx)
            })
        };
    }
    pub(super) unsafe fn record_array<A: Copy, F>(&mut self, first: &[A], function: F)
    where
        F: Fn(&mut Context, &[A]) -> Result<(), NativeFailure> + Send + 'static,
    {
        unsafe {
            self.record_arrays(first, &[], &[], move |ctx, a, _: &[()], _: &[()]| {
                function(ctx, a)
            })
        };
    }
    pub(super) unsafe fn record_pair<A: Copy, B: Copy, F>(
        &mut self,
        first: &[A],
        second: &[B],
        function: F,
    ) where
        F: Fn(&mut Context, &[A], &[B]) -> Result<(), NativeFailure> + Send + 'static,
    {
        unsafe {
            self.record_arrays(first, second, &[], move |ctx, a, b, _: &[()]| {
                function(ctx, a, b)
            })
        };
    }
    pub(super) unsafe fn record_arrays<A: Copy, B: Copy, C: Copy, F>(
        &mut self,
        first: &[A],
        second: &[B],
        third: &[C],
        function: F,
    ) where
        F: Fn(&mut Context, &[A], &[B], &[C]) -> Result<(), NativeFailure> + Send + 'static,
    {
        unsafe {
            self.record_arrays4(first, second, third, &[], move |ctx, a, b, c, _: &[()]| {
                function(ctx, a, b, c)
            });
        }
    }
    pub(super) unsafe fn record_arrays4<A: Copy, B: Copy, C: Copy, D: Copy, F>(
        &mut self,
        first: &[A],
        second: &[B],
        third: &[C],
        fourth: &[D],
        function: F,
    ) where
        F: Fn(&mut Context, &[A], &[B], &[C], &[D]) -> Result<(), NativeFailure> + Send + 'static,
    {
        let lengths = [first.len(), second.len(), third.len(), fourth.len()]
            .map(|n| u32::try_from(n).expect("native argument count exceeds u32"));
        let plan = super::arena::Plan::new::<F, A, B, C, D>(lengths);
        let Some(chunk) = &self.chunk else {
            return;
        };
        if chunk.commands.len() != 0 && !chunk.commands.fits(&plan) {
            self.dispatch();
        }
        let Some(chunk) = &mut self.chunk else {
            return;
        };
        unsafe {
            chunk
                .commands
                .push((first, second, third, fourth), function, plan)
        };
    }
    fn dispatch(&mut self) {
        if self.chunk.as_ref().is_none_or(|chunk| chunk.len() == 0) {
            return;
        }
        // Transfer and recycle stable chunk ownership, as CommandChunk's
        // unique_ptr in Eden; neither queue copies the boundary parameters.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp
        let chunk = self.chunk.take().unwrap();
        nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::NativeQueuedChunks, 1);
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::NativeChunkCommands,
            chunk.len() as u64,
        );
        let _trace = nixe_gpu::trace::span(
            "native_chunk_enqueue",
            chunk.commands.identity(),
            chunk.len() as u64,
            0,
        );
        #[cfg(feature = "performance-counters")]
        let queued_at = std::time::Instant::now();
        #[cfg(feature = "performance-counters")]
        let mut chunk = chunk;
        #[cfg(feature = "performance-counters")]
        {
            chunk.queued_at = Some(queued_at);
        }
        let accepted = {
            let mut queue = self.handle.inner.queue.lock().unwrap();
            if queue.running && !queue.stopped {
                queue.pending.push_back(chunk);
                true
            } else {
                false
            }
        };
        #[cfg(feature = "performance-counters")]
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::NativeChunkEnqueueNanoseconds,
            queued_at.elapsed().as_nanos() as u64,
        );
        if !accepted {
            return;
        }
        self.handle.inner.event.notify_all();
        // AcquireNewChunk consumes the worker's reserve without waiting for
        // recording; only an empty reserve allocates another native block.
        self.chunk = Some(
            self.handle
                .inner
                .reserve
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_default(),
        );
    }
    pub(crate) fn flush_chunks(&mut self) -> Result<(), VulkanError> {
        self.handle.check_failure()?;
        self.dispatch();
        if self.chunk.is_none() {
            return Err(RecordingHandle::disconnected());
        }
        Ok(())
    }
    pub(crate) fn discard_pending(&mut self) {
        if let Some(chunk) = &mut self.chunk {
            chunk.clear();
        }
    }
    pub(crate) fn finish(&mut self) -> Result<(), VulkanError> {
        self.flush_chunks()?;
        self.handle.finish()
    }
}
impl RecordingHandle {
    pub(crate) fn check_failure(&self) -> Result<(), VulkanError> {
        self.native_failure()
            .map_or(Ok(()), |failure| Err(failure.error()))
    }
    pub(crate) fn native_failure(&self) -> Option<NativeFailure> {
        self.inner.failure.get().cloned()
    }
    pub(crate) fn is_device_lost(&self) -> bool {
        self.inner.failure.get().is_some_and(|failure| {
            matches!(
                failure,
                NativeFailure::Vulkan {
                    result: vk::Result::ERROR_DEVICE_LOST,
                    ..
                }
            )
        })
    }
    fn disconnected() -> VulkanError {
        VulkanError::InvalidResource("Vulkan recording worker stopped before recording completed")
    }
    pub(crate) fn finish(&self) -> Result<(), VulkanError> {
        // Scheduler::WaitWorker waits for the queue to drain, then takes the
        // execution lock. No control command or per-wait reply queue exists.
        let queue = self
            .inner
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let queue = self
            .inner
            .event
            .wait_while(queue, |queue| !queue.pending.is_empty() && queue.running)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let running = queue.running;
        drop(queue);
        let _execution = self.inner.execution.lock().unwrap_or_else(|poisoned| {
            // A panic can release execution before WorkerScope has finished
            // unwinding. Surface it before admitting descriptor payload reuse.
            let _ = self.inner.failure.set(NativeFailure::Panicked);
            poisoned.into_inner()
        });
        self.check_failure()?;
        if !running {
            return Err(Self::disconnected());
        }
        Ok(())
    }
}
impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.finish();
        self.handle
            .inner
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stopped = true;
        self.handle.inner.event.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
pub(crate) struct Context {
    pub device: Device,
    pub(super) functions: Functions,
    pub command: vk::CommandBuffer,
    pub upload: vk::CommandBuffer,
    pub(super) slot: Option<Arc<CommandSlot>>,
    pub(super) descriptors: crate::cache::DescriptorAllocator,
    pub(super) descriptor_pools: rustc_hash::FxHashSet<usize>,
    pub(super) progress: Option<(Arc<SubmissionProgress>, u64)>,
    pub(super) timeline: vk::Semaphore,
}
fn record_chunk(
    scope: &mut WorkerScope,
    context: &mut Context,
    queue_mutex: &Mutex<()>,
    chunk: &mut Chunk,
) {
    let count = chunk.len();
    let begin = chunk.begin.take();
    if let Some(begin) = &begin {
        scope.activate(&begin.progress, begin.tick);
    } else if let Some((progress, tick)) = &scope.active
        && progress.cancelled()
        && *tick > progress.submitted()
    {
        scope.fail(NativeFailure::Stopped);
    }
    let _trace = nixe_gpu::trace::span(
        "native_chunk_record",
        chunk.commands.identity(),
        count as u64,
        0,
    );
    if scope.failure.get().is_none()
        && let Some(begin) = begin
    {
        context.progress = Some((begin.progress, begin.tick));
        context.timeline = begin.timeline;
        context.slot = Some(begin.slot);
        context.descriptor_pools.clear();
        match context.slot.as_ref().unwrap().begin(&context.device) {
            Ok([command, upload]) => {
                context.command = command;
                context.upload = upload;
            }
            Err(error) => scope.fail(error),
        }
    }
    if scope.failure.get().is_none()
        && let Err(error) = chunk.commands.execute(context)
    {
        scope.fail(error);
    }
    // Successful execution already destroyed each capture. Only a failed or
    // cancelled suffix needs destruction here; recycling is constant-time.
    chunk.commands.clear();
    if let Some(work) = chunk.submit.take()
        && scope.activate(&work.progress, work.tick)
        && let Err(error) = submit_native(work, context, queue_mutex)
    {
        scope.fail(error);
    }
}

fn submit_native(
    work: QueueSubmission,
    context: &mut Context,
    queue_mutex: &Mutex<()>,
) -> Result<(), NativeFailure> {
    let _trace = nixe_gpu::trace::span("native_finalize_submit", work.tick, 0, 0);
    let device = &context.device;
    let functions = &context.functions;
    let visibility = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    unsafe {
        (functions.barrier)(
            context.upload,
            &vk::DependencyInfo::default().memory_barriers(&visibility),
        )
    };
    for command in [context.upload, context.command] {
        unsafe { device.end_command_buffer(command) }.map_err(|result| NativeFailure::Vulkan {
            operation: "vkEndCommandBuffer",
            result,
        })?;
    }
    let commands = [context.upload, context.command]
        .map(|command| vk::CommandBufferSubmitInfo::default().command_buffer(command));
    let info = vk::SubmitInfo2::default()
        .command_buffer_infos(&commands)
        .wait_semaphore_infos(&work.waits)
        .signal_semaphore_infos(&work.signals);
    let lock_trace = nixe_gpu::trace::span("native_queue_lock_wait", work.tick, 0, 0);
    let _queue = queue_mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    drop(lock_trace);
    if let Some(failure) = work.progress.failure().filter(|f| work.tick >= f.tick) {
        return Err(failure.cause);
    }
    if work.progress.cancelled() {
        return Err(NativeFailure::Stopped);
    }
    let submit_trace = nixe_gpu::trace::span("native_queue_submit", work.tick, 0, 0);
    #[cfg(feature = "performance-counters")]
    let started = std::time::Instant::now();
    let result = unsafe { (functions.submit)(work.queue, 1, &info, vk::Fence::null()) };
    // Publish successful native use before any diagnostic work can unwind.
    if result == vk::Result::SUCCESS {
        work.progress.submitted_successfully(work.tick);
    }
    drop(submit_trace);
    #[cfg(feature = "performance-counters")]
    nixe_gpu::metrics::record(
        nixe_gpu::metrics::Counter::HostQueueSubmitNanoseconds,
        started.elapsed().as_nanos() as u64,
    );
    if result != vk::Result::SUCCESS {
        return Err(NativeFailure::Vulkan {
            operation: "vkQueueSubmit2",
            result,
        });
    }
    nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::HostQueueSubmissions, 1);
    nixe_gpu::trace::event("queue_submit", work.tick, 0, 0);
    Ok(())
}
