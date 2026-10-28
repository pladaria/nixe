//! Recycled native jobs, containing no guest-memory references.
use super::*;
use super::{arena::Arena, slot::CommandSlot};
use crate::queue::progress::{NativeFailure, SubmissionProgress};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
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
}
impl Chunk {
    fn new() -> Box<Self> {
        Box::default()
    }
    fn len(&self) -> usize {
        self.commands.len() + usize::from(self.begin.is_some()) + usize::from(self.submit.is_some())
    }
    fn clear(&mut self) {
        self.begin = None;
        self.commands.clear();
        self.submit = None;
    }
}
enum Work {
    Commands {
        chunk: Box<Chunk>,
        #[cfg(feature = "performance-counters")]
        queued_at: std::time::Instant,
    },
    Finish(mpsc::SyncSender<()>),
    Stop,
}
#[derive(Clone)]
pub(crate) struct RecordingHandle {
    inner: Arc<RecordingState>,
}
struct RecordingState {
    sender: mpsc::Sender<Work>,
    failure: Arc<OnceLock<NativeFailure>>,
}
pub(super) struct Recorder {
    pub(super) handle: RecordingHandle,
    recycled: mpsc::Receiver<Box<Chunk>>,
    chunk: Option<Box<Chunk>>,
    worker: Option<JoinHandle<()>>,
}
impl Recorder {
    pub(super) fn new(
        device: Device,
        functions: Functions,
        queue_mutex: Arc<Mutex<()>>,
    ) -> Result<Self, VulkanError> {
        let (sender, receiver) = mpsc::channel();
        let (recycle, recycled) = mpsc::channel();
        let failure = Arc::new(OnceLock::new());
        let worker_failure = Arc::clone(&failure);
        let worker = thread::Builder::new()
            .name("nixe-vk-record".into())
            .spawn(move || {
                let mut scope = WorkerScope {
                    failure: Arc::clone(&worker_failure),
                    active: None,
                };
                let mut context = Context {
                    device,
                    functions,
                    command: vk::CommandBuffer::null(),
                    upload: vk::CommandBuffer::null(),
                    slot: None,
                    draw: None,
                    descriptors: crate::cache::DescriptorAllocator::default(),
                    descriptor_pools: Default::default(),
                    progress: None,
                    timeline: vk::Semaphore::null(),
                };
                while let Ok(work) = receiver.recv() {
                    match work {
                        Work::Commands {
                            mut chunk,
                            #[cfg(feature = "performance-counters")]
                            queued_at,
                        } => {
                            #[cfg(feature = "performance-counters")]
                            nixe_gpu::metrics::record(
                                nixe_gpu::metrics::Counter::NativeChunkHandoffNanoseconds,
                                queued_at.elapsed().as_nanos() as u64,
                            );
                            record_chunk(&mut scope, &mut context, &queue_mutex, &mut chunk);
                            if recycle.send(chunk).is_err() {
                                break;
                            }
                        }
                        Work::Finish(reply) => {
                            if worker_failure.get().is_none() {
                                context.flush_draw();
                            } else {
                                context.draw = None;
                            }
                            let _ = reply.send(());
                        }
                        Work::Stop => break,
                    }
                }
                if worker_failure.get().is_none() {
                    context.flush_draw();
                }
            })
            .map_err(|error| {
                VulkanError::Loader(format!("Vulkan recording worker: {error}").into())
            })?;
        Ok(Self {
            handle: RecordingHandle {
                inner: Arc::new(RecordingState { sender, failure }),
            },
            recycled,
            chunk: Some(Chunk::new()),
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
        let lengths = [first.len(), second.len(), third.len()]
            .map(|n| u32::try_from(n).expect("native argument count exceeds u32"));
        let plan = super::arena::Plan::new::<F, A, B, C>(lengths);
        let Some(chunk) = &self.chunk else {
            return;
        };
        if chunk.commands.len() != 0 && !chunk.commands.fits(&plan) {
            self.dispatch();
        }
        let Some(chunk) = &mut self.chunk else {
            return;
        };
        unsafe { chunk.commands.push((first, second, third), function, plan) };
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
        let result = self.handle.inner.sender.send(Work::Commands {
            chunk,
            #[cfg(feature = "performance-counters")]
            queued_at,
        });
        #[cfg(feature = "performance-counters")]
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::NativeChunkEnqueueNanoseconds,
            queued_at.elapsed().as_nanos() as u64,
        );
        if result.is_err() {
            return;
        }
        // Reuse the worker's finished storage, or grow without waiting for it.
        self.chunk = Some(self.recycled.try_recv().unwrap_or_else(|_| Chunk::new()));
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
    fn enqueue(&self, work: Work) -> Result<(), VulkanError> {
        self.check_failure()?;
        if self.inner.sender.send(work).is_err() {
            return Err(Self::disconnected());
        }
        Ok(())
    }
    pub(crate) fn finish(&self) -> Result<(), VulkanError> {
        let (reply, completed) = mpsc::sync_channel(1);
        self.enqueue(Work::Finish(reply))?;
        completed.recv().map_err(|_| Self::disconnected())?;
        self.check_failure()?;
        Ok(())
    }
}
impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.finish();
        let _ = self.handle.inner.sender.send(Work::Stop);
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
    draw: Option<PendingDraw>,
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
        context.flush_draw();
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
    if scope.failure.get().is_some() {
        context.draw = None;
    }
}

fn submit_native(
    work: QueueSubmission,
    context: &mut Context,
    queue_mutex: &Mutex<()>,
) -> Result<(), NativeFailure> {
    let _trace = nixe_gpu::trace::span("native_finalize_submit", work.tick, 0, 0);
    context.flush_draw();
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

#[derive(Clone, Copy)]
pub(super) enum PendingDraw {
    Draw {
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
}
impl Context {
    pub fn flush_draw(&mut self) {
        let Some(draw) = self.draw.take() else {
            return;
        };
        // SAFETY: the active command slot retains all native inputs through GPU use.
        unsafe {
            match draw {
                PendingDraw::Draw {
                    vertices,
                    instances,
                    first_vertex,
                    first_instance,
                } => self.device.cmd_draw(
                    self.command,
                    vertices,
                    instances,
                    first_vertex,
                    first_instance,
                ),
                PendingDraw::DrawIndexed {
                    indices,
                    instances,
                    first_index,
                    vertex_offset,
                    first_instance,
                } => self.device.cmd_draw_indexed(
                    self.command,
                    indices,
                    instances,
                    first_index,
                    vertex_offset,
                    first_instance,
                ),
            }
        }
    }
    pub(super) fn draw(&mut self, next: PendingDraw) {
        match (&mut self.draw, next) {
            (
                Some(PendingDraw::Draw {
                    vertices: old_vertices,
                    instances: old_instances,
                    first_vertex: old_vertex,
                    first_instance: old_instance,
                }),
                PendingDraw::Draw {
                    vertices,
                    instances,
                    first_vertex,
                    first_instance,
                },
            ) if *old_vertices == vertices
                && *old_vertex == first_vertex
                && old_instance.checked_add(*old_instances) == Some(first_instance) =>
            {
                if let Some(total) = old_instances.checked_add(instances) {
                    *old_instances = total;
                    return;
                }
            }
            (
                Some(PendingDraw::DrawIndexed {
                    indices: old_indices,
                    instances: old_instances,
                    first_index: old_index,
                    vertex_offset: old_vertex,
                    first_instance: old_instance,
                }),
                PendingDraw::DrawIndexed {
                    indices,
                    instances,
                    first_index,
                    vertex_offset,
                    first_instance,
                },
            ) if *old_indices == indices
                && *old_index == first_index
                && *old_vertex == vertex_offset
                && old_instance.checked_add(*old_instances) == Some(first_instance) =>
            {
                if let Some(total) = old_instances.checked_add(instances) {
                    *old_instances = total;
                    return;
                }
            }
            _ => {}
        }
        self.flush_draw();
        self.draw = Some(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    unsafe extern "system" fn end(_: vk::CommandBuffer) -> vk::Result {
        vk::Result::SUCCESS
    }
    unsafe extern "system" fn submit(
        queue: vk::Queue,
        _: u32,
        _: *const vk::SubmitInfo2<'_>,
        _: vk::Fence,
    ) -> vk::Result {
        if queue.as_raw() == 1 {
            vk::Result::SUCCESS
        } else {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        }
    }
    static CALLS_AFTER_FAILED_BEGIN: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    unsafe extern "system" fn begin(
        _: vk::CommandBuffer,
        _: *const vk::CommandBufferBeginInfo<'_>,
    ) -> vk::Result {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY
    }
    unsafe extern "system" fn end_rendering(_: vk::CommandBuffer) {
        CALLS_AFTER_FAILED_BEGIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    unsafe extern "system" fn barrier(_: vk::CommandBuffer, _: *const vk::DependencyInfo<'_>) {}
    fn recorder(queue_mutex: Arc<Mutex<()>>) -> Recorder {
        // Inject failures at native call boundaries in the actual worker.
        let device = unsafe {
            Device::load_with(
                |name| match name.to_bytes() {
                    b"vkEndCommandBuffer" => end as *const () as *const std::ffi::c_void,
                    b"vkBeginCommandBuffer" => begin as *const () as *const std::ffi::c_void,
                    b"vkCmdPipelineBarrier2" => barrier as *const () as *const std::ffi::c_void,
                    b"vkCmdEndRendering" => end_rendering as *const () as *const std::ffi::c_void,
                    _ => std::ptr::null(),
                },
                vk::Device::null(),
            )
        };
        let functions = Functions {
            submit,
            barrier: device.fp_v1_3().cmd_pipeline_barrier2,
            begin: device.fp_v1_3().cmd_begin_rendering,
            end: device.fp_v1_3().cmd_end_rendering,
            #[cfg(feature = "binding-probe")]
            timestamp: device.fp_v1_3().cmd_write_timestamp2,
            state: None,
            feedback: None,
        };
        Recorder::new(device, functions, queue_mutex).unwrap()
    }
    fn submission(queue: u64, tick: u64, submitted: &Arc<SubmissionProgress>) -> QueueSubmission {
        QueueSubmission {
            queue: vk::Queue::from_raw(queue),
            waits: Default::default(),
            signals: Default::default(),
            tick,
            progress: Arc::clone(submitted),
        }
    }
    #[test]
    fn failed_begin_cancels_commands_and_submit_in_the_same_chunk() {
        CALLS_AFTER_FAILED_BEGIN.store(0, std::sync::atomic::Ordering::Relaxed);
        let device = crate::VulkanDevice::new(crate::DeviceOptions::default(), None).unwrap();
        let [slot, _] = CommandSlot::allocate(
            Arc::clone(&device.owner),
            device.queues().graphics_compute_family,
        );
        let mut recorder = recorder(Arc::new(Mutex::new(())));
        let progress = Arc::new(SubmissionProgress::default());
        recorder
            .begin(slot, 1, Arc::clone(&progress), device.submission.timeline)
            .unwrap();
        recorder.record(|ctx| {
            ctx.flush_draw();
            unsafe { (ctx.functions.end)(ctx.command) };
            Ok(())
        });
        recorder.submit(submission(1, 1, &progress)).unwrap();
        assert!(matches!(
            recorder.finish(),
            Err(VulkanError::Vulkan {
                operation: "vkBeginCommandBuffer",
                result: vk::Result::ERROR_OUT_OF_HOST_MEMORY
            })
        ));
        assert_eq!(progress.submitted(), 0);
        assert_eq!(
            CALLS_AFTER_FAILED_BEGIN.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }
    struct CaptureDrop(Arc<std::sync::atomic::AtomicUsize>);
    impl CaptureDrop {
        fn expect_prefix(&self, expected: usize) {
            assert_eq!(self.0.load(std::sync::atomic::Ordering::SeqCst), expected);
        }
    }
    impl Drop for CaptureDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    #[test]
    fn arena_invokes_and_destroys_in_order_and_owns_aligned_oversized_arrays() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let drops = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Mutex::new(()));
        let blocked = gate.lock().unwrap();
        let mut recorder = recorder(Arc::new(Mutex::new(())));
        let waiting = Arc::clone(&gate);
        recorder.record(move |_| {
            drop(waiting.lock().unwrap());
            Ok(())
        });
        for index in 0..100 {
            let capture = CaptureDrop(Arc::clone(&drops));
            recorder.record(move |_| {
                capture.expect_prefix(index);
                Ok(())
            });
        }
        let mut values = vec![0x1234_5678_9abc_def0u128; 4096];
        let mut offsets = [17u64, 23];
        unsafe {
            recorder.record_pair(&values, &offsets, |_, values, offsets| {
                assert_eq!(values.len(), 4096);
                assert_eq!(values.as_ptr() as usize % align_of::<u128>(), 0);
                assert!(values.iter().all(|v| *v == 0x1234_5678_9abc_def0));
                assert_eq!(offsets, &[17, 23]);
                Ok(())
            })
        };
        values.fill(0);
        offsets.fill(0);
        drop(blocked);
        recorder.finish().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 100);
        // Reuse the same transport after executing the oversized capture.
        recorder.record(|_| Ok(()));
        recorder.finish().unwrap();
    }
    #[test]
    fn arena_failure_or_panic_destroys_unexecuted_captures_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for panic in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let calls = Arc::new(AtomicUsize::new(0));
            let mut recorder = recorder(Arc::new(Mutex::new(())));
            let handle = recorder.handle.clone();
            for index in 0..12 {
                let capture = CaptureDrop(Arc::clone(&drops));
                let calls = Arc::clone(&calls);
                recorder.record(move |_| {
                    capture.expect_prefix(index);
                    calls.fetch_add(1, Ordering::SeqCst);
                    if index == 3 {
                        assert!(!panic, "injected capture panic");
                        return Err(NativeFailure::Vulkan {
                            operation: "injected capture failure",
                            result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                        });
                    }
                    Ok(())
                });
            }
            assert!(recorder.finish().is_err());
            drop(recorder);
            assert_eq!(calls.load(Ordering::SeqCst), 4);
            assert_eq!(drops.load(Ordering::SeqCst), 12);
            assert!(matches!(handle.native_failure(), Some(NativeFailure::Panicked)) == panic);
        }
    }
    #[test]
    fn worker_panic_wakes_waiters_and_preserves_the_last_native_submission() {
        let progress = Arc::new(SubmissionProgress::default());
        progress.submitted_successfully(3);
        let failure = Arc::new(OnceLock::new());
        let pending = Arc::clone(&progress);
        let waiting = thread::spawn(move || pending.wait_submitted(4, None));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = WorkerScope {
                failure: Arc::clone(&failure),
                active: Some((Arc::clone(&progress), 3)),
            };
            panic!("injected native worker panic");
        }));
        assert!(result.is_err());
        assert!(matches!(
            waiting.join().unwrap(),
            Err(VulkanError::SubmissionFailed { tick: 4, .. })
        ));
        assert_eq!(progress.submitted(), 3);
        assert!(matches!(failure.get(), Some(NativeFailure::Panicked)));
    }
    #[test]
    fn failed_submission_retains_the_last_real_tick_and_stops_following_work() {
        let mut recorder = recorder(Arc::new(Mutex::new(())));
        let submitted = Arc::new(SubmissionProgress::default());
        recorder.submit(submission(1, 3, &submitted)).unwrap();
        recorder.submit(submission(2, 4, &submitted)).unwrap();
        assert!(matches!(
            recorder.finish(),
            Err(VulkanError::Vulkan {
                operation: "vkQueueSubmit2",
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            })
        ));
        assert_eq!(submitted.submitted(), 3);
        assert!(matches!(
            recorder.submit(submission(1, 5, &submitted)),
            Err(VulkanError::Vulkan {
                operation: "vkQueueSubmit2",
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            })
        ));
        assert_eq!(submitted.submitted(), 3);
    }

    #[test]
    fn queued_chunks_grow_without_admission_waits_and_failure_cancels_the_suffix() {
        let mutex = Arc::new(Mutex::new(()));
        let mut recorder = recorder(Arc::clone(&mutex));
        let progress = Arc::new(SubmissionProgress::default());
        recorder.submit(submission(1, 1, &progress)).unwrap();
        recorder.finish().unwrap();
        let (held, ready) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        std::thread::scope(|scope| {
            let holder = scope.spawn(move || {
                let _guard = mutex.lock().unwrap();
                held.send(()).unwrap();
                wait.recv_timeout(std::time::Duration::from_secs(3)).is_ok()
            });
            ready.recv().unwrap();
            recorder.submit(submission(2, 2, &progress)).unwrap();
            for tick in 3..64 {
                recorder.submit(submission(1, tick, &progress)).unwrap();
            }
            release.send(()).unwrap();
            assert!(
                holder.join().unwrap(),
                "enqueue must not wait for the recording worker"
            );
        });
        assert!(recorder.finish().is_err());
        assert_eq!(progress.submitted(), 1);
        assert_eq!(progress.failure().unwrap().tick, 2);
    }

    #[test]
    fn recording_panic_keeps_descriptor_pools_alive_for_the_submitted_gpu_prefix() {
        use crate::cache::*;
        use crate::{AllocatedBuffer, AllocationMode, MemoryLocation};
        use std::time::Duration;
        mod gate {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/support/gate.rs"
            ));
        }
        let Some(mut device) = crate::test_device::device_for_test(crate::ApiPreference::Auto)
        else {
            return;
        };
        let validation = device.validation_monitor();
        let workers = Arc::new(CompilePool::new(1, 8).unwrap());
        let pipelines = PipelineCompiler::new(
            &device,
            Arc::clone(&workers),
            &CacheConfiguration {
                workers: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let shader = crate::test_shaders::compile(
            &device.profile().shader_compiler.profile,
            [crate::test_shaders::increment(7, 4)],
        );
        let pipeline = pipelines.request(PipelineRecipe::Compute(shader)).unwrap();
        let buffer: Arc<AllocatedBuffer> = Arc::new(
            device
                .allocate_buffer(
                    8,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::GpuToCpu,
                    AllocationMode::Managed,
                    "descriptor pool panic prefix",
                )
                .unwrap(),
        );
        unsafe {
            buffer.write_range(0, &[0; 8]).unwrap();
        }
        let mut descriptors = DescriptorCache::new(&device);
        let record = descriptors
            .get(
                &pipeline.layout,
                &[DescriptorResource::Buffer {
                    buffer: Arc::clone(&buffer),
                    offset: 0,
                    range: 8,
                }],
            )
            .unwrap();
        device.bind_native_pipeline(&pipeline).unwrap();
        device
            .bind_native_descriptors(
                &pipeline,
                &record,
                &vec![0; pipeline.layout.dynamic_bindings.len()],
            )
            .unwrap();
        unsafe {
            device.dispatch.cmd_dispatch(1, 1, 1);
            let barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buffer.handle)
                .size(8)];
            device
                .dispatch
                .barrier(&vk::DependencyInfo::default().buffer_memory_barriers(&barriers));
        }
        let mut gate = gate::Gate::new(&device);
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(gate.handle)
            .value(1)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let work = unsafe {
            device
                .submission
                .flush(device.queue, &mut device.dispatch, &waits, &[])
        }
        .unwrap()
        .unwrap();
        gate.last_use = work.tick();
        device.dispatch.recording_handle().finish().unwrap();
        drop(record);
        drop(descriptors);
        device.begin_recording().unwrap();
        device
            .dispatch
            .recorder
            .record(|_| panic!("injected recording panic after native descriptor use"));
        let _ = device.flush();
        assert!(device.dispatch.recording_handle().finish().is_err());
        assert!(matches!(
            device.dispatch.recording_handle().native_failure(),
            Some(NativeFailure::Panicked)
        ));
        // The worker's allocator has unwound, but actual GPU pool use is pending.
        validation.check().unwrap();
        gate.signal();
        assert!(device.wait_for_work(work, Duration::from_secs(10)).unwrap());
        assert_eq!(
            unsafe { buffer.read_range(4, 4) }.unwrap(),
            7u32.to_le_bytes()
        );
        drop(gate);
        drop(buffer);
        drop(pipeline);
        drop(pipelines);
        drop(workers);
        drop(device);
        validation.check().unwrap();
    }
}
