//! Recycled bounded native jobs, containing no guest-memory references.
use super::*;
use crate::queue::progress::{NativeFailure, SubmissionProgress};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

const CHUNK_COMMANDS: usize = 128;
const QUEUED_CHUNKS: usize = 8;
pub(crate) struct QueueSubmission {
    pub queue: vk::Queue,
    pub command: vk::CommandBuffer,
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
enum Work {
    Commands(Vec<Command>),
    Begin {
        pool: vk::CommandPool,
        command: vk::CommandBuffer,
        tick: u64,
        progress: Arc<SubmissionProgress>,
    },
    Submit(QueueSubmission),
    Finish(mpsc::SyncSender<()>),
    Stop,
}
struct Producer {
    commands: Vec<Command>,
    failed: bool,
}
#[derive(Clone)]
pub(crate) struct RecordingHandle {
    inner: Arc<RecordingState>,
}
struct RecordingState {
    sender: mpsc::SyncSender<Work>,
    recycled: Mutex<mpsc::Receiver<Vec<Command>>>,
    producer: Mutex<Producer>,
    failure: Arc<OnceLock<NativeFailure>>,
}
pub(super) struct Recorder {
    pub(super) handle: RecordingHandle,
    worker: Option<JoinHandle<()>>,
}
impl Recorder {
    pub(super) fn new(
        device: Device,
        functions: Functions,
        queue_mutex: Arc<Mutex<()>>,
    ) -> Result<Self, VulkanError> {
        let (sender, receiver) = mpsc::sync_channel(QUEUED_CHUNKS);
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
                let mut draw = None;
                while let Ok(work) = receiver.recv() {
                    match &work {
                        Work::Begin { tick, progress, .. }
                        | Work::Submit(QueueSubmission { tick, progress, .. }) => {
                            scope.active = Some((Arc::clone(progress), *tick));
                            if let Some(failure) = progress.failure().filter(|f| *tick >= f.tick) {
                                let _ = worker_failure.set(failure.cause);
                            }
                        }
                        _ => {}
                    }
                    if let Some((progress, tick)) = &scope.active
                        && progress.cancelled()
                        && *tick > progress.submitted()
                    {
                        progress.fail(*tick, NativeFailure::Stopped);
                        let _ = worker_failure.set(progress.failure().unwrap().cause);
                    }
                    match work {
                        Work::Commands(mut commands) => {
                            if worker_failure.get().is_none() {
                                for command in commands.drain(..) {
                                    if let Err(error) =
                                        append_draw(&mut draw, command, &device, &functions)
                                    {
                                        scope.fail(error);
                                        break;
                                    }
                                }
                            } else {
                                commands.clear();
                            }
                            if recycle.send(commands).is_err() {
                                break;
                            }
                        }
                        Work::Begin {
                            pool,
                            command,
                            tick,
                            progress,
                        } => {
                            if worker_failure.get().is_none() {
                                if let Err(error) = flush_draw(&mut draw, &device, &functions) {
                                    scope.fail(error);
                                }
                                let result = unsafe {
                                    device.reset_command_pool(
                                        pool,
                                        vk::CommandPoolResetFlags::empty(),
                                    )
                                }
                                .map_err(|result| NativeFailure::Vulkan {
                                    operation: "vkResetCommandPool",
                                    result,
                                })
                                .and_then(|()| {
                                    unsafe {
                                        device.begin_command_buffer(
                                            command,
                                            &vk::CommandBufferBeginInfo::default().flags(
                                                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                                            ),
                                        )
                                    }
                                    .map_err(|result| {
                                        NativeFailure::Vulkan {
                                            operation: "vkBeginCommandBuffer",
                                            result,
                                        }
                                    })
                                });
                                if let Err(error) = result {
                                    progress.fail(tick, error.clone());
                                    let _ = worker_failure.set(error);
                                }
                            } else {
                                draw = None;
                                progress.fail(tick, worker_failure.get().unwrap().clone());
                            }
                        }
                        Work::Submit(work) => {
                            let _trace =
                                nixe_gpu::trace::span("native_finalize_submit", work.tick, 0, 0);
                            if worker_failure.get().is_none() {
                                if let Err(error) = flush_draw(&mut draw, &device, &functions) {
                                    scope.fail(error);
                                }
                                let result = unsafe { device.end_command_buffer(work.command) }
                                    .map_err(|result| NativeFailure::Vulkan {
                                        operation: "vkEndCommandBuffer",
                                        result,
                                    })
                                    .and_then(|()| {
                                        let commands = [vk::CommandBufferSubmitInfo::default()
                                            .command_buffer(work.command)];
                                        let info = vk::SubmitInfo2::default()
                                            .command_buffer_infos(&commands)
                                            .wait_semaphore_infos(&work.waits)
                                            .signal_semaphore_infos(&work.signals);
                                        let lock_trace = nixe_gpu::trace::span(
                                            "native_queue_lock_wait",
                                            work.tick,
                                            0,
                                            0,
                                        );
                                        let _queue = queue_mutex
                                            .lock()
                                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                                        drop(lock_trace);
                                        if let Some(failure) =
                                            work.progress.failure().filter(|f| work.tick >= f.tick)
                                        {
                                            return Err(failure.cause);
                                        }
                                        if work.progress.cancelled() {
                                            return Err(NativeFailure::Stopped);
                                        }
                                        let submit_trace = nixe_gpu::trace::span(
                                            "native_queue_submit",
                                            work.tick,
                                            0,
                                            0,
                                        );
                                        #[cfg(feature = "performance-counters")]
                                        let started = std::time::Instant::now();
                                        let result = unsafe {
                                            (functions.submit)(
                                                work.queue,
                                                1,
                                                &info,
                                                vk::Fence::null(),
                                            )
                                        };
                                        #[cfg(feature = "performance-counters")]
                                        let submit_ns = started.elapsed().as_nanos() as u64;
                                        if result == vk::Result::SUCCESS {
                                            // Record the actual native use before diagnostics or
                                            // any other fallible work can unwind this worker.
                                            work.progress.submitted_successfully(work.tick);
                                        }
                                        drop(submit_trace);
                                        #[cfg(feature = "performance-counters")]
                                        nixe_gpu::metrics::record(
                                            nixe_gpu::metrics::Counter::HostQueueSubmitNanoseconds,
                                            submit_ns,
                                        );
                                        if result == vk::Result::SUCCESS {
                                            Ok(())
                                        } else {
                                            Err(NativeFailure::Vulkan {
                                                operation: "vkQueueSubmit2",
                                                result,
                                            })
                                        }
                                    });
                                match result {
                                    Ok(()) => {
                                        nixe_gpu::metrics::record(
                                            nixe_gpu::metrics::Counter::HostQueueSubmissions,
                                            1,
                                        );
                                        nixe_gpu::trace::detail_event(
                                            "queue_submit",
                                            work.tick,
                                            0,
                                            0,
                                        );
                                    }
                                    Err(error) => {
                                        work.progress.fail(work.tick, error.clone());
                                        let _ = worker_failure.set(error);
                                    }
                                }
                            } else {
                                draw = None;
                                work.progress
                                    .fail(work.tick, worker_failure.get().unwrap().clone());
                            }
                        }
                        Work::Finish(reply) => {
                            if worker_failure.get().is_none() {
                                if let Err(error) = flush_draw(&mut draw, &device, &functions) {
                                    scope.fail(error);
                                }
                            } else {
                                draw = None;
                            }
                            let _ = reply.send(());
                        }
                        Work::Stop => break,
                    }
                }
                if worker_failure.get().is_none()
                    && let Err(error) = flush_draw(&mut draw, &device, &functions)
                {
                    scope.fail(error);
                }
            })
            .map_err(|error| {
                VulkanError::Loader(format!("Vulkan recording worker: {error}").into())
            })?;
        Ok(Self {
            handle: RecordingHandle {
                inner: Arc::new(RecordingState {
                    sender,
                    recycled: Mutex::new(recycled),
                    failure,
                    producer: Mutex::new(Producer {
                        commands: Vec::with_capacity(CHUNK_COMMANDS),
                        failed: false,
                    }),
                }),
            },
            worker: Some(worker),
        })
    }
    pub(crate) fn push(&self, command: Command) {
        self.handle.push(command);
    }
    pub(crate) fn finish(&self) -> Result<(), VulkanError> {
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
    pub(crate) fn push(&self, command: Command) {
        let mut producer = self
            .inner
            .producer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if producer.failed {
            return;
        }
        producer.commands.push(command);
        if producer.commands.len() == CHUNK_COMMANDS {
            self.dispatch(&mut producer);
        }
    }
    fn dispatch(&self, producer: &mut Producer) {
        if producer.commands.is_empty() || producer.failed {
            return;
        }
        let replacement = self
            .inner
            .recycled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .try_recv()
            .unwrap_or_else(|_| Vec::with_capacity(CHUNK_COMMANDS));
        let commands = std::mem::replace(&mut producer.commands, replacement);
        if self.inner.sender.send(Work::Commands(commands)).is_err() {
            producer.failed = true;
        }
    }
    fn enqueue(&self, work: Work) -> Result<(), VulkanError> {
        self.check_failure()?;
        let mut producer = self
            .inner
            .producer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.dispatch(&mut producer);
        if producer.failed {
            return Err(Self::disconnected());
        }
        if self.inner.sender.send(work).is_err() {
            producer.failed = true;
            return Err(Self::disconnected());
        }
        Ok(())
    }
    pub(crate) fn begin(
        &self,
        pool: vk::CommandPool,
        command: vk::CommandBuffer,
        tick: u64,
        progress: Arc<SubmissionProgress>,
    ) -> Result<(), VulkanError> {
        self.enqueue(Work::Begin {
            pool,
            command,
            tick,
            progress,
        })
    }
    pub(crate) fn submit(&self, work: QueueSubmission) -> Result<(), VulkanError> {
        if work
            .waits
            .iter()
            .chain(&work.signals)
            .any(|info| !info.p_next.is_null())
        {
            return Err(VulkanError::InvalidResource(
                "asynchronous queue submission requires owned semaphore parameters without pNext",
            ));
        }
        self.enqueue(Work::Submit(work))
    }
    pub(crate) fn finish(&self) -> Result<(), VulkanError> {
        let mut producer = self
            .inner
            .producer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.dispatch(&mut producer);
        if !producer.failed {
            let (reply, completed) = mpsc::sync_channel(1);
            producer.failed =
                self.inner.sender.send(Work::Finish(reply)).is_err() || completed.recv().is_err();
        }
        self.check_failure()?;
        if producer.failed {
            Err(Self::disconnected())
        } else {
            Ok(())
        }
    }
    pub(crate) fn discard_pending(&self) {
        self.inner
            .producer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .commands
            .clear();
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
fn flush_draw(
    draw: &mut Option<Command>,
    device: &Device,
    functions: &Functions,
) -> Result<(), NativeFailure> {
    if let Some(draw) = draw.take() {
        // SAFETY: each parameter was captured while the owner retained the
        // resources in its command slot. The ordered submit job follows this recording.
        unsafe { draw.execute(device, functions) }?;
    }
    Ok(())
}
fn append_draw(
    draw: &mut Option<Command>,
    command: Command,
    device: &Device,
    functions: &Functions,
) -> Result<(), NativeFailure> {
    match (&mut *draw, &command) {
        (
            Some(Command::Draw {
                command: old_cmd,
                vertices: old_vertices,
                instances: old_instances,
                first_vertex: old_vertex,
                first_instance: old_instance,
            }),
            Command::Draw {
                command: cmd,
                vertices,
                instances,
                first_vertex,
                first_instance,
            },
        ) if old_cmd == cmd
            && old_vertices == vertices
            && old_vertex == first_vertex
            && old_instance.checked_add(*old_instances) == Some(*first_instance) =>
        {
            if let Some(total) = old_instances.checked_add(*instances) {
                *old_instances = total;
                return Ok(());
            }
        }
        (
            Some(Command::DrawIndexed {
                command: old_cmd,
                indices: old_indices,
                instances: old_instances,
                first_index: old_index,
                vertex_offset: old_vertex,
                first_instance: old_instance,
            }),
            Command::DrawIndexed {
                command: cmd,
                indices,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            },
        ) if old_cmd == cmd
            && old_indices == indices
            && old_index == first_index
            && old_vertex == vertex_offset
            && old_instance.checked_add(*old_instances) == Some(*first_instance) =>
        {
            if let Some(total) = old_instances.checked_add(*instances) {
                *old_instances = total;
                return Ok(());
            }
        }
        _ => {}
    }
    flush_draw(draw, device, functions)?;
    if matches!(command, Command::Draw { .. } | Command::DrawIndexed { .. }) {
        *draw = Some(command);
    } else {
        // SAFETY: same retained-parameter contract as flush_draw.
        unsafe { command.execute(device, functions) }?;
    }
    Ok(())
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
    fn recorder() -> Recorder {
        // Only vkEndCommandBuffer and vkQueueSubmit2 run in this failure test.
        // A fake function table exercises the actual worker without a driver.
        let device = unsafe {
            Device::load_with(
                |name| {
                    if name == c"vkEndCommandBuffer" {
                        end as *const () as *const std::ffi::c_void
                    } else {
                        std::ptr::null()
                    }
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
        };
        Recorder::new(device, functions, Arc::new(Mutex::new(()))).unwrap()
    }
    fn submission(queue: u64, tick: u64, submitted: &Arc<SubmissionProgress>) -> QueueSubmission {
        QueueSubmission {
            queue: vk::Queue::from_raw(queue),
            command: vk::CommandBuffer::null(),
            waits: Default::default(),
            signals: Default::default(),
            tick,
            progress: Arc::clone(submitted),
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
        let recorder = recorder();
        let submitted = Arc::new(SubmissionProgress::default());
        recorder
            .handle
            .submit(submission(1, 3, &submitted))
            .unwrap();
        recorder
            .handle
            .submit(submission(2, 4, &submitted))
            .unwrap();
        assert!(matches!(
            recorder.finish(),
            Err(VulkanError::Vulkan {
                operation: "vkQueueSubmit2",
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            })
        ));
        assert_eq!(submitted.submitted(), 3);
        assert!(matches!(
            recorder.handle.submit(submission(1, 5, &submitted)),
            Err(VulkanError::Vulkan {
                operation: "vkQueueSubmit2",
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            })
        ));
        assert_eq!(submitted.submitted(), 3);
    }
}
