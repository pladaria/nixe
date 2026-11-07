//! Ordered asynchronous Host1x work. Completion follows canonical surface writes.
use super::{
    NvDrvCallError, NvDrvDeviceDescriptor, NvDrvDeviceKind, NvHostControl,
    UnsupportedNvDrvOperation, execution_error, video,
};
use nixe_gpu::{GuestSyncpointId, GuestSyncpointValue, GuestTimelinePoint, ReservedTimelinePoint};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicU8, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub(super) struct Job {
    pub descriptor: NvDrvDeviceDescriptor,
    pub request: u32,
    pub commands: Vec<Vec<u8>>,
    pub completion: Option<ReservedTimelinePoint>,
    pub mappings: video::Mappings,
    pub timeout: u32,
}
const RUNNING: u8 = 0;
const CLOSING: u8 = 1;
const STOPPING: u8 = 2;
pub(super) type Failure = Arc<Mutex<Option<(NvDrvDeviceDescriptor, u32, String)>>>;

/// Each device owns its command FIFO and engine state, as CDmaPusher does.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/cdma_pusher.cpp#L22-L95
#[derive(Debug)]
pub(in crate::nvdrv) struct Worker {
    sender: mpsc::Sender<Option<Job>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    stopped: Arc<AtomicU8>,
    progress: Arc<(Mutex<u64>, Condvar)>,
    failure: Failure,
    done: nixe_runtime::ReadableEventObject,
    descriptor: NvDrvDeviceDescriptor,
}
pub(super) fn require_healthy(failure: &Failure) -> Result<(), UnsupportedNvDrvOperation> {
    if let Some((descriptor, request, error)) = &*failure
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        let NvDrvCallError::Unsupported(operation) =
            execution_error(*descriptor, *request, error.clone())
        else {
            unreachable!()
        };
        return Err(operation);
    }
    Ok(())
}
impl Worker {
    pub(super) fn new(
        control: Arc<Mutex<NvHostControl>>,
        gpu: Option<Arc<super::super::gpu_executor::NvDrvGpuExecutor>>,
        descriptor: NvDrvDeviceDescriptor,
        frames: Arc<video::FrameQueue>,
        failure: Failure,
    ) -> Result<Self, NvDrvCallError> {
        let (sender, receiver) = mpsc::channel::<Option<Job>>();
        let stopped = Arc::new(AtomicU8::new(RUNNING));
        let thread_stopped = stopped.clone();
        let thread_failure = failure.clone();
        let (progress, (finished, done)) = {
            let mut control = control
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                control.host_progress(),
                nixe_runtime::EventObject::create_pair(control.kernel()),
            )
        };
        let thread_progress = progress.clone();
        let thread = thread::Builder::new()
            .name(format!("nixe-host1x-{}", descriptor.fd().raw()))
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let run = || -> Result<(), (u32, String)> {
                        let mut channel = Channel::default();
                        while let Ok(Some(job)) = receiver.recv() {
                            if thread_stopped.load(Ordering::Acquire) != RUNNING {
                                break;
                            }
                            require_healthy(&thread_failure)
                                .map_err(|e| (job.request, e.to_string()))?;
                            if !channel
                                .execute(
                                    &job,
                                    &control,
                                    gpu.as_deref(),
                                    &frames,
                                    &thread_progress,
                                    &thread_stopped,
                                    &thread_failure,
                                )
                                .map_err(|error| (job.request, error))?
                            {
                                break;
                            }
                        }
                        Ok(())
                    };
                    let result = run();
                    if descriptor.kind() == NvDrvDeviceKind::HostNvDec {
                        frames.close(descriptor.fd());
                    }
                    result
                }))
                .unwrap_or_else(|_| Err((0, "Host1x worker execution or cleanup panicked".into())));
                if let Err((request, error)) = result {
                    log::error!("Host1x execution failed: {error}");
                    let mut failed = thread_failure
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if failed.is_none() {
                        *failed = Some((descriptor, request, error));
                    }
                    drop(failed);
                    let mut generation = thread_progress
                        .0
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *generation = generation.wrapping_add(1);
                    thread_progress.1.notify_all();
                    drop(generation);
                    control
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .wake_on_execution_failure();
                }
                finished.signal();
            })
            .map_err(|_| NvDrvCallError::GuestResult(super::super::NV_INSUFFICIENT_MEMORY))?;
        Ok(Self {
            sender,
            thread: Mutex::new(Some(thread)),
            stopped,
            progress,
            failure,
            done,
            descriptor,
        })
    }
    pub(super) fn require_healthy(&self) -> Result<(), UnsupportedNvDrvOperation> {
        require_healthy(&self.failure)
    }
    pub(super) fn submit(&self, job: Job) -> Result<(), NvDrvCallError> {
        if self.stopped.load(Ordering::Acquire) != RUNNING {
            return Err(NvDrvCallError::GuestResult(super::super::NV_INVALID_STATE));
        }
        self.require_healthy()
            .map_err(NvDrvCallError::Unsupported)?;
        self.sender
            .send(Some(job))
            .map_err(|_| NvDrvCallError::GuestResult(super::super::NV_INVALID_STATE))
    }
    pub(super) fn stop(&self) {
        if self.stopped.swap(STOPPING, Ordering::AcqRel) == STOPPING {
            return;
        }
        let _guard = self
            .progress
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.progress.1.notify_all();
        let _ = self.sender.send(None);
    }
    pub(super) fn begin_close(&self) -> nixe_runtime::EventFuture {
        // A normal device close stops admission and drains the current command
        // list, as CDmaPusher's stop token does. Machine stop/cancel can abort
        // a passive syncpoint wait without inventing its completion.
        if self.stopped.fetch_max(CLOSING, Ordering::AcqRel) == RUNNING {
            let _ = self.sender.send(None);
        }
        self.done.wait_async()
    }
    pub(super) fn join(&self) -> Result<(), UnsupportedNvDrvOperation> {
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(thread) = thread
            && thread.join().is_err()
        {
            let NvDrvCallError::Unsupported(error) =
                execution_error(self.descriptor, 0, "Host1x worker cleanup panicked")
            else {
                unreachable!()
            };
            return Err(error);
        }
        Ok(())
    }
    pub(super) fn shutdown(&self) {
        self.stop();
        if let Err(error) = self.join() {
            log::error!("{error}");
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn wait(
    control: &Arc<Mutex<NvHostControl>>,
    progress: &Arc<(Mutex<u64>, Condvar)>,
    stopped: &AtomicU8,
    failure: &Failure,
    point: GuestTimelinePoint,
    timeout: u32,
) -> Result<bool, String> {
    let deadline = if timeout == 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_millis(u64::from(timeout)))
    };
    loop {
        require_healthy(failure).map_err(|error| error.to_string())?;
        if stopped.load(Ordering::Acquire) == STOPPING {
            return Ok(false);
        }
        let generation = *progress
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .host_point_reached(point)
        {
            Some(true) => return Ok(true),
            Some(false) => {}
            None => return Err(format!("unknown Host1x dependency {point:?}")),
        }
        let guard = progress
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *guard != generation
            || stopped.load(Ordering::Acquire) == STOPPING
            || failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
        {
            continue;
        }
        if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!("Host1x dependency timed out: {point:?}"));
            }
            drop(
                progress
                    .1
                    .wait_timeout(guard, remaining)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        } else {
            drop(
                progress
                    .1
                    .wait(guard)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        }
    }
}

/// The device thread consumes methods directly; waits and increments remain in
/// stream order and each execute borrows the device's current registers.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/cdma_pusher.cpp#L43-L119
struct Channel {
    packet: super::PacketState,
    registers: [u32; 0x446],
    decoder: Option<video::NvDec>,
    vic: super::vic::Vic,
}
impl Default for Channel {
    fn default() -> Self {
        Self {
            packet: Default::default(),
            registers: [0; 0x446],
            decoder: None,
            vic: Default::default(),
        }
    }
}
impl Channel {
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        job: &Job,
        control: &Arc<Mutex<NvHostControl>>,
        gpu: Option<&super::super::gpu_executor::NvDrvGpuExecutor>,
        frames: &Arc<video::FrameQueue>,
        progress: &Arc<(Mutex<u64>, Condvar)>,
        stopped: &AtomicU8,
        failure: &Failure,
    ) -> Result<bool, String> {
        let descriptor = job.descriptor;
        let mut remaining = job
            .completion
            .as_ref()
            .map_or(0, ReservedTimelinePoint::increments);
        for bytes in &job.commands {
            for word in bytes.chunks_exact(4) {
                let Some((class, method, argument)) = self
                    .packet
                    .word(u32::from_le_bytes(word.try_into().unwrap()))?
                else {
                    continue;
                };
                log::trace!(
                    "Host1x method: fd={} class={class:#x} method={method:#x} argument={argument:#010x}",
                    descriptor.fd()
                );
                if class == 1 {
                    match method {
                        0x4e => self.packet.wait_value = argument,
                        0x8 | 0x50 => {
                            let point = GuestTimelinePoint::new(
                                GuestSyncpointId::new(argument),
                                GuestSyncpointValue::new(self.packet.wait_value),
                            );
                            if !wait(control, progress, stopped, failure, point, job.timeout)? {
                                return Ok(false);
                            }
                        }
                        _ => return Err(format!("unsupported Host1x control method {method:#x}")),
                    }
                    continue;
                }
                let expected = if descriptor.kind() == NvDrvDeviceKind::HostNvDec {
                    0xf0
                } else {
                    0x5d
                };
                if class != expected {
                    return Err(format!("unsupported channel class {class:#x}"));
                }
                match method {
                    0 => {
                        if argument & !0xffff != 0 || argument >> 8 > 2 {
                            return Err("unsupported syncpoint increment condition".into());
                        }
                        let completion = job
                            .completion
                            .as_ref()
                            .filter(|point| {
                                point.point().syncpoint().get() == argument & 0xff && remaining != 0
                            })
                            .ok_or("submission contains an unreserved syncpoint increment")?;
                        let mut control = control
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        let actions = control
                            .consume_channel_increment(descriptor, job.request, completion)
                            .map_err(|error| format!("{error:?}"))?;
                        for action in actions {
                            action.signal_from_cpu();
                        }
                        control
                            .publish_channel_increment(descriptor, job.request, completion)
                            .map_err(|error| format!("{error:?}"))?;
                        remaining -= 1;
                    }
                    0x10 => self.packet.engine_method = argument,
                    0x11 => {
                        let method = self.packet.engine_method;
                        *self.registers.get_mut(method as usize).ok_or_else(|| {
                            format!("engine register out of range: {method:#x}")
                        })? = argument;
                        log::trace!(
                            "Host1x engine: fd={} method={method:#x} argument={argument:#010x}",
                            descriptor.fd()
                        );
                        if method == 0xc0 {
                            match descriptor.kind() {
                                NvDrvDeviceKind::HostNvDec => self
                                    .decoder
                                    .get_or_insert_with(|| {
                                        video::NvDec::new(
                                            gpu.and_then(|gpu| gpu.device_uuid()),
                                            frames.clone(),
                                            descriptor.fd(),
                                        )
                                    })
                                    .execute(&self.registers, &job.mappings)?,
                                NvDrvDeviceKind::HostVic => {
                                    self.vic
                                        .execute(&self.registers, &job.mappings, gpu, frames)?
                                }
                                _ => return Err("invalid Host1x engine".into()),
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "unsupported THI method {method:#x} argument={argument:#x}"
                        ));
                    }
                }
            }
        }
        if remaining != 0 {
            return Err("submission syncpoint increments do not match executed commands".into());
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdrv::{
        NvDrvDescriptorOwner, NvDrvFileDescriptor, NvDrvPermissionProfile, NvDrvSessionId,
    };
    use std::collections::BTreeMap;
    fn descriptor(fd: u32) -> NvDrvDeviceDescriptor {
        NvDrvDeviceDescriptor::open(
            NvDrvFileDescriptor::new(fd),
            NvDrvDeviceKind::HostVic,
            NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
            NvDrvPermissionProfile::Application,
        )
    }
    #[test]
    fn pending_dependencies_do_not_publish_completion_and_failures_are_retained() {
        let mut control = NvHostControl::new(&crate::test_memory::kernel());
        let producer = descriptor(1);
        let consumer = descriptor(2);
        let a = control.allocate_channel_syncpoint(producer, 0).unwrap();
        let b = control.allocate_channel_syncpoint(consumer, 0).unwrap();
        let dependency = control
            .reserve_channel_submission(producer, 0, a.syncpoint(), 1)
            .unwrap();
        let completion = control
            .reserve_channel_submission(consumer, 0, b.syncpoint(), 1)
            .unwrap();
        let point = completion.point();
        let control = Arc::new(Mutex::new(control));
        let worker = Worker::new(
            Arc::clone(&control),
            None,
            consumer,
            Arc::default(),
            Arc::default(),
        )
        .unwrap();
        worker
            .submit(Job {
                descriptor: consumer,
                request: 0,
                commands: vec![
                    [
                        1u32 << 6, // SETCLASS Host1x control.
                        (4 << 28) | (0x4e << 16) | dependency.point().value().get(),
                        (1 << 28) | (0x50 << 16) | 1,
                        dependency.point().syncpoint().get(),
                        0x5d << 6, // SETCLASS VIC THI.
                        (4 << 28) | b.syncpoint().get(),
                    ]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                ],
                completion: Some(completion),
                mappings: Arc::new(BTreeMap::new()),
                timeout: 0,
            })
            .unwrap();
        assert_eq!(
            control.lock().unwrap().host_point_reached(point),
            Some(false)
        );
        // Closing another device must not join this channel's unresolved wait.
        let other = Worker::new(
            control.clone(),
            None,
            descriptor(3),
            Arc::default(),
            Arc::default(),
        )
        .unwrap();
        crate::test_memory::run(other.begin_close());
        other.join().unwrap();
        assert_eq!(
            control.lock().unwrap().host_point_reached(point),
            Some(false)
        );
        for _ in 0..dependency.increments() {
            let mut control = control.lock().unwrap();
            control
                .consume_channel_increment(producer, 0, &dependency)
                .unwrap();
            control
                .publish_channel_increment(producer, 0, &dependency)
                .unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while control.lock().unwrap().host_point_reached(point) == Some(false) {
            worker.require_healthy().unwrap();
            assert!(
                Instant::now() < deadline,
                "Host1x dependency did not complete"
            );
            std::thread::yield_now();
        }
        worker.require_healthy().unwrap();
        assert_eq!(
            control.lock().unwrap().host_point_reached(point),
            Some(true)
        );
        let reservation = control
            .lock()
            .unwrap()
            .reserve_channel_submission(consumer, 0, b.syncpoint(), 1)
            .unwrap();
        let failed = reservation.point();
        worker
            .submit(Job {
                descriptor: consumer,
                request: 0,
                commands: vec![
                    [
                        0x5du32 << 6,
                        (4 << 28) | (0x10 << 16) | 0xc0,
                        (4 << 28) | (0x11 << 16),
                        (4 << 28) | b.syncpoint().get(),
                    ]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                ],
                completion: Some(reservation),
                mappings: Arc::new(BTreeMap::new()),
                timeout: 0,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.require_healthy().is_ok() {
            assert!(Instant::now() < deadline, "Host1x failure was not reported");
            std::thread::yield_now();
        }
        assert!(
            worker
                .require_healthy()
                .unwrap_err()
                .to_string()
                .contains("unmapped Host1x")
        );
        assert_eq!(
            control.lock().unwrap().host_point_reached(failed),
            Some(false)
        );
    }
    use crate::nvdrv::nvmap::{NvMapAllocationMetadata, NvMapCpuMapping, NvMapObjects, NvMapOwner};
    use nixe_memory::{
        AddressSpaceId, AllocationRange, DeviceAccessEndpoint, DeviceAccessRequest,
        GuestVirtualAddress,
    };
    struct Endpoint(mpsc::Sender<DeviceAccessRequest>);
    impl DeviceAccessEndpoint for Endpoint {
        fn submit(&self, r: DeviceAccessRequest) {
            self.0.send(r).unwrap();
        }
        fn cancel(&self, _: nixe_memory::AccessRequestId) {}
        fn allocation_released(&self, _: nixe_memory::MachineId, _: nixe_memory::AllocationId) {}
        fn physical_retired(
            &self,
            _: nixe_memory::MachineId,
            _: nixe_memory::AllocationId,
            _: Box<[nixe_memory::Extent]>,
        ) {
        }
        fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
    }
    fn job(mappings: video::Mappings) -> Job {
        let words = [
            0x5du32 << 6,
            (4 << 28) | (0x10 << 16) | 0x1c2,
            (4 << 28) | (0x11 << 16) | 0x10,
            (4 << 28) | (0x10 << 16) | 0xc0,
            (4 << 28) | (0x11 << 16) | 1,
        ];
        Job {
            descriptor: descriptor(2),
            request: 0x123,
            commands: vec![words.into_iter().flat_map(u32::to_le_bytes).collect()],
            completion: None,
            mappings,
            timeout: 0,
        }
    }

    #[test]
    fn in_stream_increments_satisfy_intermediate_waits_before_the_job_finishes() {
        let d = descriptor(2);
        let mut control = NvHostControl::new(&crate::test_memory::kernel());
        let initial = control.allocate_channel_syncpoint(d, 0).unwrap();
        let completion = control
            .reserve_channel_submission(d, 0, initial.syncpoint(), 2)
            .unwrap();
        let point = completion.point();
        let control = Arc::new(Mutex::new(control));
        let worker = Worker::new(control.clone(), None, d, Arc::default(), Arc::default()).unwrap();
        let words = [
            0x5du32 << 6,
            (4 << 28) | point.syncpoint().get(),
            1 << 6,
            (4 << 28) | (0x4e << 16) | point.value().get().wrapping_sub(1),
            (1 << 28) | (0x50 << 16) | 1,
            point.syncpoint().get(),
            0x5d << 6,
            (4 << 28) | point.syncpoint().get(),
        ];
        worker
            .submit(Job {
                descriptor: d,
                request: 0x456,
                commands: vec![words.into_iter().flat_map(u32::to_le_bytes).collect()],
                completion: Some(completion),
                mappings: Arc::default(),
                timeout: 10,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while control.lock().unwrap().host_point_reached(point) == Some(false) {
            worker.require_healthy().unwrap();
            assert!(
                Instant::now() < deadline,
                "in-stream dependency did not complete"
            );
            std::thread::yield_now();
        }
        worker.require_healthy().unwrap();
        worker.stop();
        worker.join().unwrap();
        assert_eq!(
            control.lock().unwrap().host_point_reached(point),
            Some(true)
        );
    }
    #[test]
    fn stopping_a_passive_wait_preserves_prior_increments_without_a_failure() {
        let producer = descriptor(1);
        let consumer = descriptor(2);
        let mut control = NvHostControl::new(&crate::test_memory::kernel());
        let a = control.allocate_channel_syncpoint(producer, 0).unwrap();
        let b = control.allocate_channel_syncpoint(consumer, 0).unwrap();
        let dependency = control
            .reserve_channel_submission(producer, 0, a.syncpoint(), 1)
            .unwrap();
        let completion = control
            .reserve_channel_submission(consumer, 0, b.syncpoint(), 1)
            .unwrap();
        let point = completion.point();
        let control = Arc::new(Mutex::new(control));
        let worker = Worker::new(
            control.clone(),
            None,
            consumer,
            Arc::default(),
            Arc::default(),
        )
        .unwrap();
        let words = [
            0x5du32 << 6,
            (4 << 28) | point.syncpoint().get(),
            1 << 6,
            (4 << 28) | (0x4e << 16) | dependency.point().value().get(),
            (1 << 28) | (0x50 << 16) | 1,
            dependency.point().syncpoint().get(),
        ];
        worker
            .submit(Job {
                descriptor: consumer,
                request: 0,
                commands: vec![words.into_iter().flat_map(u32::to_le_bytes).collect()],
                completion: Some(completion),
                mappings: Arc::default(),
                timeout: 0,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while control.lock().unwrap().host_point_reached(point) == Some(false) {
            worker.require_healthy().unwrap();
            assert!(
                Instant::now() < deadline,
                "worker did not execute its first increment"
            );
            std::thread::yield_now();
        }
        worker.stop();
        worker.join().unwrap();
        worker.require_healthy().unwrap();
        assert_eq!(
            control
                .lock()
                .unwrap()
                .host_point_reached(dependency.point()),
            Some(false)
        );
    }
}
