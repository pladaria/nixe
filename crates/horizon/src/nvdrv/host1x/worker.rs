//! Ordered asynchronous Host1x work. Completion follows canonical surface writes.
use super::{
    NvDrvCallError, NvDrvDeviceDescriptor, NvDrvDeviceKind, NvDrvFileDescriptor, NvHostControl,
    UnsupportedNvDrvOperation, execution_error, video,
};
use nixe_gpu::{GuestSyncpointId, GuestSyncpointValue, GuestTimelinePoint, ReservedTimelinePoint};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub(super) enum Operation {
    Wait(GuestTimelinePoint),
    Execute(Vec<u32>),
}
#[derive(Debug)]
pub(super) struct Job {
    pub descriptor: NvDrvDeviceDescriptor,
    pub request: u32,
    pub commands: Vec<Box<[u8]>>,
    pub advertised: BTreeMap<u32, u32>,
    pub completions: Vec<ReservedTimelinePoint>,
    pub mappings: video::Mappings,
    pub timeout: u32,
}
#[derive(Debug)]
enum Message {
    Job(Job),
    Close(NvDrvFileDescriptor, mpsc::Sender<()>),
    Stop,
}
type Failure = Arc<Mutex<Option<(NvDrvDeviceDescriptor, u32, String)>>>;

#[derive(Debug)]
pub(super) struct Worker {
    sender: mpsc::Sender<Message>,
    thread: Mutex<Option<JoinHandle<()>>>,
    stopped: Arc<AtomicBool>,
    progress: Arc<(Mutex<u64>, Condvar)>,
    failure: Failure,
}
impl Worker {
    pub(super) fn new(
        control: Arc<Mutex<NvHostControl>>,
        gpu: Option<Arc<super::super::gpu_executor::NvDrvGpuExecutor>>,
    ) -> Result<Self, NvDrvCallError> {
        let (sender, receiver) = mpsc::channel();
        let stopped = Arc::new(AtomicBool::new(false));
        let thread_stopped = Arc::clone(&stopped);
        let failure: Failure = Arc::default();
        let thread_failure = Arc::clone(&failure);
        let progress = control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .host_progress();
        let thread_progress = Arc::clone(&progress);
        let thread = thread::Builder::new()
            .name("nixe-host1x".into())
            .spawn(move || {
                let mut decoders = BTreeMap::new();
                let mut channels = BTreeMap::<NvDrvFileDescriptor, DecodeState>::new();
                while let Ok(message) = receiver.recv() {
                    match message {
                        Message::Stop => break,
                        Message::Close(fd, ack) => {
                            if decoders.remove(&fd).is_some() {
                                for state in channels.values_mut() {
                                    if state.vic.decoder == Some(fd) { state.vic = Default::default(); }
                                }
                            }
                            channels.remove(&fd);
                            let _ = ack.send(());
                        }
                        Message::Job(job) => {
                            if thread_stopped.load(Ordering::Acquire) {
                                break;
                            }
                            if thread_failure
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .is_some()
                            {
                                continue;
                            }
                            let execute = || -> Result<(), String> {
                                let state = channels.entry(job.descriptor.fd()).or_default();
                                let operations = decode(state, &job)?;
                                for operation in &operations {
                                    match operation {
                                        Operation::Wait(point) => wait(
                                            &control,
                                            &thread_progress,
                                            &thread_stopped,
                                            &thread_failure,
                                            *point,
                                            job.timeout,
                                        )?,
                                        Operation::Execute(registers) => {
                                            match job.descriptor.kind() {
                                                NvDrvDeviceKind::HostNvDec => {
                                                    if let std::collections::btree_map::Entry::Vacant(entry) = decoders.entry(job.descriptor.fd()) {
                                                        entry.insert(video::NvDec::new(gpu.as_deref().and_then(|gpu| gpu.device_uuid())));
                                                    }
                                                    decoders.get_mut(&job.descriptor.fd()).unwrap().execute(registers, &job.mappings)?;
                                                }
                                                NvDrvDeviceKind::HostVic => {
                                                    state.vic.execute(registers, &job.mappings, gpu.as_deref(), &mut decoders)?
                                                }
                                                _ => return Err("invalid Host1x engine".into()),
                                            }
                                        }
                                    }
                                }
                                let mut control = control
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                for completion in &job.completions {
                                    control
                                        .complete_channel_submission(
                                            job.descriptor,
                                            job.request,
                                            completion,
                                        )
                                        .map_err(|e| format!("{e:?}"))?;
                                }
                                Ok(())
                            };
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(execute))
                                    .unwrap_or_else(|_| Err("Host1x worker panicked".into()));
                            if let Err(error) = result
                                && !thread_stopped.load(Ordering::Acquire)
                            {
                                log::error!("Host1x execution failed: {error}");
                                let mut failure = thread_failure
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                if failure.is_none() {
                                    *failure = Some((job.descriptor, job.request, error));
                                }
                                drop(failure);
                                // Release channels waiting on the failed producer. They
                                // retain the original error and never publish completion.
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
                        }
                    }
                }
            })
            .map_err(|_| NvDrvCallError::GuestResult(super::super::NV_INSUFFICIENT_MEMORY))?;
        Ok(Self {
            sender,
            thread: Mutex::new(Some(thread)),
            stopped,
            progress,
            failure,
        })
    }
    pub(super) fn require_healthy(&self) -> Result<(), UnsupportedNvDrvOperation> {
        if let Some((descriptor, request, error)) = &*self
            .failure
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
    pub(super) fn submit(&self, job: Job) -> Result<(), NvDrvCallError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(NvDrvCallError::GuestResult(super::super::NV_INVALID_STATE));
        }
        self.require_healthy()
            .map_err(NvDrvCallError::Unsupported)?;
        self.sender
            .send(Message::Job(job))
            .map_err(|_| NvDrvCallError::GuestResult(super::super::NV_INVALID_STATE))
    }
    pub(super) fn close(&self, fd: NvDrvFileDescriptor) {
        let (sender, receiver) = mpsc::channel();
        if self.sender.send(Message::Close(fd, sender)).is_ok() {
            let _ = receiver.recv();
        }
    }
    pub(super) fn shutdown(&self) {
        self.stopped.store(true, Ordering::Release);
        {
            let _guard = self
                .progress
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.progress.1.notify_all();
        }
        let _ = self.sender.send(Message::Stop);
        if let Some(thread) = self
            .thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = thread.join();
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
    stopped: &AtomicBool,
    failure: &Failure,
    point: GuestTimelinePoint,
    timeout: u32,
) -> Result<(), String> {
    let deadline = if timeout == 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_millis(u64::from(timeout)))
    };
    loop {
        if stopped.load(Ordering::Acquire)
            || failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
        {
            return Err("Host1x stopped/failed while waiting for dependency".into());
        }
        let generation = *progress
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .point_reached(point)
        {
            Some(true) => return Ok(()),
            Some(false) => {}
            None => return Err(format!("unknown Host1x dependency {point:?}")),
        }
        let guard = progress
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *guard != generation
            || stopped.load(Ordering::Acquire)
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

struct DecodeState {
    packet: super::PacketState,
    registers: Vec<u32>,
    vic: super::vic::Vic,
}
impl Default for DecodeState {
    fn default() -> Self {
        Self {
            packet: Default::default(),
            registers: vec![0; 0x446],
            vic: Default::default(),
        }
    }
}
fn decode(channel: &mut DecodeState, job: &Job) -> Result<Vec<Operation>, String> {
    let decode = || -> Result<Vec<Operation>, NvDrvCallError> {
        let descriptor = job.descriptor;
        let request = job.request;
        let mut operations = Vec::new();
        let mut increments = BTreeMap::<u32, u32>::new();
        for bytes in &job.commands {
            for word in bytes.chunks_exact(4) {
                let word = u32::from_le_bytes(word.try_into().unwrap());
                let Some((class, method, argument)) = channel
                    .packet
                    .word(word)
                    .map_err(|error| super::execution_error(descriptor, request, error))?
                else {
                    continue;
                };
                log::trace!(
                    "Host1x method: fd={} class={class:#x} method={method:#x} argument={argument:#010x}",
                    descriptor.fd()
                );
                if class == 1 {
                    match method {
                        0x4e => channel.packet.wait_value = argument,
                        0x8 | 0x50 => {
                            let point = GuestTimelinePoint::new(
                                GuestSyncpointId::new(argument),
                                GuestSyncpointValue::new(channel.packet.wait_value),
                            );
                            operations.push(Operation::Wait(point));
                        }
                        _ => {
                            return Err(super::execution_error(
                                descriptor,
                                request,
                                format!("unsupported Host1x control method {method:#x}"),
                            ));
                        }
                    }
                    continue;
                }
                let expected = if descriptor.kind() == NvDrvDeviceKind::HostNvDec {
                    0xf0
                } else {
                    0x5d
                };
                if class != expected {
                    return Err(super::execution_error(
                        descriptor,
                        request,
                        format!("unsupported channel class {class:#x}"),
                    ));
                }
                match method {
                    0 => {
                        if argument & !0xffff != 0 || argument >> 8 > 2 {
                            return Err(super::execution_error(
                                descriptor,
                                request,
                                "unsupported syncpoint increment condition",
                            ));
                        }
                        let id = argument & 0xff;
                        let count = increments.entry(id).or_default();
                        *count = count.checked_add(1).ok_or(super::super::NV_OVERFLOW)?;
                    }
                    0x10 => channel.packet.engine_method = argument,
                    0x11 => {
                        let method = channel.packet.engine_method;
                        let register =
                            channel.registers.get_mut(method as usize).ok_or_else(|| {
                                super::execution_error(
                                    descriptor,
                                    request,
                                    format!("engine register out of range: {method:#x}"),
                                )
                            })?;
                        *register = argument;
                        log::trace!(
                            "Host1x engine: fd={} method={method:#x} argument={argument:#010x}",
                            descriptor.fd()
                        );
                        if method == 0xc0 {
                            operations.push(Operation::Execute(channel.registers.clone()));
                        }
                    }
                    _ => {
                        return Err(super::execution_error(
                            descriptor,
                            request,
                            format!("unsupported THI method {method:#x} argument={argument:#x}"),
                        ));
                    }
                }
            }
        }
        if job
            .advertised
            .iter()
            .any(|(id, count)| increments.remove(id).unwrap_or(0) != *count)
            || !increments.is_empty()
        {
            return Err(super::execution_error(
                descriptor,
                request,
                "submission syncpoint increments do not match executed commands",
            ));
        }
        Ok(operations)
    };
    let mut decode = decode;
    decode().map_err(|error| error.to_string())
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
        let worker = Worker::new(Arc::clone(&control), None).unwrap();
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
                advertised: BTreeMap::from([(b.syncpoint().get(), 1)]),
                completions: vec![completion],
                mappings: Arc::new(BTreeMap::new()),
                timeout: 0,
            })
            .unwrap();
        assert_eq!(control.lock().unwrap().point_reached(point), Some(false));
        control
            .lock()
            .unwrap()
            .complete_channel_submission(producer, 0, &dependency)
            .unwrap();
        worker.close(consumer.fd());
        worker.require_healthy().unwrap();
        assert_eq!(control.lock().unwrap().point_reached(point), Some(true));
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
                advertised: BTreeMap::from([(b.syncpoint().get(), 1)]),
                completions: vec![reservation],
                mappings: Arc::new(BTreeMap::new()),
                timeout: 0,
            })
            .unwrap();
        worker.close(consumer.fd());
        assert!(
            worker
                .require_healthy()
                .unwrap_err()
                .to_string()
                .contains("unmapped Host1x")
        );
        assert_eq!(control.lock().unwrap().point_reached(failed), Some(false));
    }
}
