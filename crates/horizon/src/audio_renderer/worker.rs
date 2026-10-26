//! Audio command execution on a dedicated worker. SDL only consumes prepared PCM.
//!
//! Automatic renderer scheduling and frame accounting:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/system.cpp

use super::{AudioRendererState, render::RenderBuffers};
use nixe_audio::{AudioBackend, AudioOutput, PcmFormat};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(crate) enum RenderError {
    Stopped,
    Memory(Arc<nixe_memory::MemoryError>),
    Diagnostic(Box<str>),
}
impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => f.write_str("DSP stopped"),
            Self::Memory(error) => write!(f, "DSP memory access: {error}"),
            Self::Diagnostic(detail) => f.write_str(detail),
        }
    }
}
impl From<Arc<nixe_memory::MemoryError>> for RenderError {
    fn from(error: Arc<nixe_memory::MemoryError>) -> Self {
        Self::Memory(error)
    }
}
impl From<Box<str>> for RenderError {
    fn from(detail: Box<str>) -> Self {
        Self::Diagnostic(detail)
    }
}
impl From<String> for RenderError {
    fn from(detail: String) -> Self {
        Self::Diagnostic(detail.into_boxed_str())
    }
}
impl From<&str> for RenderError {
    fn from(detail: &str) -> Self {
        Self::Diagnostic(detail.into())
    }
}

/// Device completion and shutdown share one wake predicate. The DSP owns its
/// state while waiting; this mutex protects only its host scheduling flags.
#[derive(Clone)]
pub(super) struct RenderMemory(Arc<(Mutex<WorkerControl>, Condvar)>);
impl RenderMemory {
    #[cfg(test)]
    pub(super) fn testing() -> Self {
        Self(Arc::new((
            Mutex::new(WorkerControl::default()),
            Condvar::new(),
        )))
    }
    pub(super) fn wait(
        &self,
        mut pending: nixe_memory::PendingAccess,
    ) -> Result<nixe_memory::AccessResult, RenderError> {
        let wake = self.0.clone();
        pending.set_waker(Arc::new(move || {
            // Synchronize with the check-to-sleep interval even when the GPU
            // completes before this worker has entered the condition wait.
            let _control = wake.0.lock().unwrap();
            wake.1.notify_all();
        }));
        let mut control = self.0.0.lock().unwrap();
        loop {
            if let Some(result) = pending.poll() {
                return result.map_err(RenderError::Memory);
            }
            if control.stop {
                return Err(RenderError::Stopped);
            }
            control = self.0.1.wait(control).unwrap();
        }
    }
}

const BLOCK: Duration = Duration::from_millis(5);
const BUFFER_COUNT: usize = 3;

#[derive(Debug)]
pub(super) struct RendererWorker {
    stop: Arc<(Mutex<WorkerControl>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

#[derive(Debug, Default)]
struct WorkerControl {
    stop: bool,
    paused: bool,
    parked: bool,
    finished: bool,
}

struct WorkerFinished(Arc<(Mutex<WorkerControl>, Condvar)>);
impl Drop for WorkerFinished {
    fn drop(&mut self) {
        self.0
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .finished = true;
        self.0.1.notify_all();
    }
}

impl RendererWorker {
    pub(super) fn start(
        state: &AudioRendererState,
        backend: &dyn AudioBackend,
        initial: &mut super::update::RendererUpdateState,
    ) -> Result<Self, Box<str>> {
        let stop = Arc::new((Mutex::new(WorkerControl::default()), Condvar::new()));
        let wake = stop.clone();
        let output = AudioOutput::open(
            backend,
            PcmFormat::STEREO_48KHZ,
            Arc::new(move |ready| {
                // Never take the renderer lock from SDL's callback. This notification
                // only releases producer backpressure; it is not a guest DSP event.
                if ready {
                    wake.1.notify_one();
                }
            }),
        )
        .map_err(|error| error.0.into_boxed_str())?;
        let params = state.parameters.clone();
        let devices = state.devices.clone();
        let mut buffers = RenderBuffers::new(&params);
        let mut free: Vec<_> = (0..BUFFER_COUNT)
            .map(|_| vec![0_i16; params.samples as usize * 2])
            .collect();
        let update = state.update.clone();
        let signal = state.signal.clone();
        let failure = state.failure.clone();
        let thread_stop = stop.clone();
        output.start().map_err(|error| error.0.into_boxed_str())?;
        initial.frames = 0;
        initial.started = true;
        let thread = match thread::Builder::new()
            .name("nixe-audren".into())
            .spawn(move || {
                // Release a pause waiter on normal completion and unwinding.
                let _finished = WorkerFinished(thread_stop.clone());
                let memory = RenderMemory(thread_stop.clone());
                let result = (|| -> Result<(), RenderError> {
                    let mut next = Instant::now();
                    loop {
                        let now = Instant::now();
                        let mut stopped = thread_stop.0.lock().unwrap();
                        if stopped.paused && !stopped.stop {
                            stopped.parked = true;
                            thread_stop.1.notify_all();
                            while stopped.paused && !stopped.stop {
                                stopped = thread_stop.1.wait(stopped).unwrap();
                            }
                            stopped.parked = false;
                            next = Instant::now();
                        }
                        if stopped.stop {
                            break;
                        }
                        if now < next {
                            drop(thread_stop.1.wait_timeout(stopped, next - now).unwrap());
                            continue;
                        }
                        drop(stopped);
                        while let Some(pcm) = output
                            .recycle_buffer()
                            .map_err(|error| error.0.into_boxed_str())?
                        {
                            free.push(pcm);
                        }
                        let Some(mut pcm) = free.pop() else {
                            // Full bounded queue: don't run ahead of host playback.
                            let stopped = thread_stop.0.lock().unwrap();
                            if stopped.stop {
                                break;
                            }
                            drop(thread_stop.1.wait_timeout(stopped, BLOCK).unwrap());
                            continue;
                        };
                        {
                            let mut state = update.take_blocking();
                            let has_sink = match buffers.render(
                                &memory,
                                &mut state,
                                &params,
                                &mut pcm,
                                devices.volume(crate::audio_device::ACTIVE_DEVICE),
                            ) {
                                Ok(has_sink) => has_sink,
                                Err(RenderError::Stopped) => break,
                                Err(error) => return Err(error),
                            };
                            if has_sink {
                                output
                                    .append(state.frames, pcm)
                                    .map_err(|error| error.0.into_boxed_str())?;
                            } else {
                                free.push(pcm);
                            }
                            state.frames += 1;
                            // Publish actual command execution while serialized with
                            // RequestUpdate's status snapshot and event reset.
                            signal.signal();
                        }
                        next += BLOCK;
                        if next < Instant::now() {
                            next = Instant::now();
                        }
                    }
                    output
                        .stop()
                        .map_err(|error| RenderError::Diagnostic(error.0.into_boxed_str()))
                })();
                if let Err(error) = result {
                    log::error!("audio renderer worker failed: {error}");
                    let mut state = update.take_blocking();
                    let _first_failure = failure.set(error);
                    state.started = false;
                    signal.signal();
                }
            }) {
            Ok(thread) => thread,
            Err(error) => {
                initial.started = false;
                return Err(format!("creating audio renderer worker: {error}").into());
            }
        };
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    fn set_paused(&self, paused: bool) {
        let mut control = self.stop.0.lock().unwrap();
        control.paused = paused;
        self.stop.1.notify_all();
        while paused && !control.parked && !control.finished {
            control = self.stop.1.wait(control).unwrap();
        }
    }
}

impl Drop for RendererWorker {
    fn drop(&mut self) {
        self.stop.0.lock().unwrap().stop = true;
        self.stop.1.notify_all();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("audio renderer worker panicked");
        }
    }
}

impl AudioRendererState {
    pub(super) fn set_host_paused(&self, paused: bool) {
        if let Some(worker) = self.worker.lock().unwrap().as_ref() {
            worker.set_paused(paused);
        }
    }
    pub(crate) async fn start(&self, backend: &dyn AudioBackend) -> Result<(), Box<str>> {
        let mut state = self.update.take().await;
        let mut worker = self.worker.lock().unwrap();
        if worker.is_none() {
            if self.parameters.performance_frames != 0 {
                return Err("Start: renderer performance histories are not implemented".into());
            }
            *worker = Some(RendererWorker::start(self, backend, &mut state)?);
            log::debug!(
                "audio renderer started: sample-rate={} samples={} pcm-buffers={BUFFER_COUNT}",
                self.parameters.sample_rate,
                self.parameters.samples
            );
        }
        state.frames = 0;
        Ok(())
    }

    pub(crate) fn stop(&self) {
        // The worker must finish before stop returns. It never owns the session,
        // so closing the last guest handle also stops and joins it without a cycle.
        let mut worker = self.worker.lock().unwrap();
        drop(worker.take());
        let mut state = self.update.take_blocking();
        state.started = false;
        log::debug!("audio renderer stopped: rendered-blocks={}", state.frames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_renderer::{AudioRendererManagerSession, PARAMETER_SIZE, RendererParameters};
    use crate::test_memory::{run, transfer};
    use nixe_audio::{AudioDevice, AudioError, AudioFeed};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Endpoint {
        requests: std::sync::mpsc::Sender<nixe_memory::DeviceAccessRequest>,
        cancelled: std::sync::mpsc::Sender<nixe_memory::AccessRequestId>,
    }
    impl nixe_memory::DeviceAccessEndpoint for Endpoint {
        fn submit(&self, request: nixe_memory::DeviceAccessRequest) {
            self.requests.send(request).unwrap();
        }
        fn cancel(&self, id: nixe_memory::AccessRequestId) {
            let _ = self.cancelled.send(id);
        }
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
    #[test]
    fn pending_dsp_reads_retain_exact_bytes_and_preserve_device_failures() {
        use crate::{audio_renderer::buffer::Attachment, test_memory::physical};
        use nixe_memory::{AccessResult, DeviceAccessEndpoint, MemoryError};
        let ranges = physical(&[0; 8]);
        let machine = ranges[0].allocation().machine().clone();
        let (send, receive) = std::sync::mpsc::channel();
        let (cancelled, _) = std::sync::mpsc::channel();
        let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint {
            requests: send,
            cancelled,
        });
        machine.set_device_endpoint(&endpoint);
        machine.publish_device_write(&ranges, || Ok(())).unwrap();
        for fail in [false, true] {
            let attachment = Attachment {
                address: 0x1000,
                ranges: nixe_memory::CapturedMemory::new(ranges.clone()).unwrap(),
                bytes: 8,
                pool: Some(0),
            };
            let worker = thread::spawn(move || {
                let mut bytes = [0; 3];
                attachment
                    .read(&RenderMemory::testing(), 2, &mut bytes)
                    .map(|()| bytes)
            });
            let request = receive.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(request.bytes(), 3);
            assert_eq!(request.ranges()[0].offset(), 2);
            if fail {
                let failure =
                    Arc::new(MemoryError::Execution("device lost during DSP read".into()));
                request.finish(Err(failure.clone()));
                let Err(RenderError::Memory(actual)) = worker.join().unwrap() else {
                    panic!("device failure was lost");
                };
                assert!(Arc::ptr_eq(&actual, &failure));
            } else {
                request.finish(Ok(AccessResult::Read(Box::new([4, 5, 6]))));
                assert_eq!(worker.join().unwrap().unwrap(), [4, 5, 6]);
            }
        }
    }
    #[test]
    fn shutdown_cancels_a_pending_device_read_without_waiting_for_gpu_completion() {
        use crate::test_memory::physical;
        use nixe_memory::{AccessOperation, DeviceAccessEndpoint};
        let ranges = physical(&[0; 8]);
        let machine = ranges[0].allocation().machine().clone();
        let (send, receive) = std::sync::mpsc::channel();
        let (cancelled, cancellations) = std::sync::mpsc::channel();
        let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint {
            requests: send,
            cancelled,
        });
        machine.set_device_endpoint(&endpoint);
        let pending = machine
            .request_device_access(ranges, AccessOperation::Read)
            .unwrap();
        let request = receive.recv().unwrap();
        let identity = request.identity();
        let memory = RenderMemory::testing();
        let stop = memory.0.clone();
        let worker = thread::spawn(move || memory.wait(pending));
        stop.0.lock().unwrap().stop = true;
        stop.1.notify_all();
        assert!(matches!(worker.join().unwrap(), Err(RenderError::Stopped)));
        assert_eq!(
            cancellations.recv_timeout(Duration::from_secs(2)).unwrap(),
            identity
        );
        assert!(request.is_cancelled());
    }

    #[derive(Default)]
    struct Backend {
        feed: Mutex<Option<AudioFeed>>,
        live: Arc<AtomicUsize>,
    }
    impl std::fmt::Debug for Backend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Backend")
        }
    }
    struct Device(Arc<AtomicUsize>);
    impl AudioDevice for Device {}
    impl Drop for Device {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    impl AudioBackend for Backend {
        fn output_layout(&self) -> Result<nixe_audio::ChannelLayout, nixe_audio::AudioError> {
            Ok(nixe_audio::ChannelLayout::Stereo)
        }

        fn open(&self, feed: AudioFeed) -> Result<Box<dyn AudioDevice>, AudioError> {
            *self.feed.lock().unwrap() = Some(feed);
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Device(self.live.clone())))
        }
    }
    fn wait_until(test: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !test() {
            assert!(
                Instant::now() < deadline,
                "worker failed to publish expected progress"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn worker_executes_bounded_blocks_recycles_pcm_and_joins_on_final_close() {
        let mut data = [0; PARAMETER_SIZE];
        for (offset, value) in [
            (0, 48000_u32),
            (4, 240),
            (8, 2),
            (20, 1),
            (48, u32::from_le_bytes(*b"REV7")),
        ] {
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let transfer = transfer(&vec![0; 4096]);
        let session = run(AudioRendererManagerSession::default().open(
            RendererParameters::decode(&data),
            transfer,
            4096,
            1,
            1,
        ))
        .unwrap()
        .unwrap();
        {
            let mut state = session.0.update.take_blocking();
            let mut mix = Box::new([0; 0x930]);
            mix[..4].copy_from_slice(&1_f32.to_le_bytes());
            mix[4..8].copy_from_slice(&48000_u32.to_le_bytes());
            mix[8..12].copy_from_slice(&2_u32.to_le_bytes());
            state.mixes = vec![Some(mix)];
            state.rebuild_mix_order().unwrap();
            state.effect_order = vec![Vec::new()];
            let mut sink = [0; 0x140];
            sink[0] = 1;
            sink[1] = 1;
            sink[0x120..0x124].copy_from_slice(&2_u32.to_le_bytes());
            sink[0x125] = 1;
            state.sinks = vec![sink];
        }
        let backend = Backend::default();
        run(session.0.start(&backend)).unwrap();
        let feed = backend.feed.lock().unwrap().take().unwrap();
        wait_until(|| session.0.update.take_blocking().frames == BUFFER_COUNT as u64);
        assert!(session.0.event.is_signalled());
        // No host consumption: execution must stay bounded, not fabricate a
        // renderer frame counter from elapsed wall time.
        thread::sleep(BLOCK * 4);
        assert_eq!(session.0.update.take_blocking().frames, BUFFER_COUNT as u64);
        feed.render(&mut [1; 480]);
        wait_until(|| session.0.update.take_blocking().frames == BUFFER_COUNT as u64 + 1);
        session.0.set_host_paused(true);
        let paused_frames = session.0.update.take_blocking().frames;
        feed.render(&mut [1; 480]);
        thread::sleep(BLOCK * 3);
        assert_eq!(session.0.update.take_blocking().frames, paused_frames);
        assert!(session.0.update.take_blocking().started);
        session.0.set_host_paused(false);
        wait_until(|| session.0.update.take_blocking().frames > paused_frames);
        session.0.set_host_paused(true);
        session.0.stop();
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
        assert!(!session.0.update.take_blocking().started);
        let stopped_frames = session.0.update.take_blocking().frames;
        thread::sleep(BLOCK * 2);
        assert_eq!(session.0.update.take_blocking().frames, stopped_frames);
        run(session.0.start(&backend)).unwrap();
        wait_until(|| session.0.update.take_blocking().frames > 0);
        let clone = session.clone();
        drop(session);
        assert_eq!(backend.live.load(Ordering::SeqCst), 1);
        drop(clone);
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn worker_failure_is_retained_and_signals_waiters_without_counting_a_block() {
        // Host errors are also tested through AudioFeed; use an unsupported
        // active sink here to exercise a DSP error after thread creation.
        let parameters = RendererParameters {
            sample_rate: 48000,
            samples: 240,
            mixes: 2,
            submixes: 0,
            voices: 0,
            sinks: 1,
            effects: 0,
            performance_frames: 0,
            splitters: 0,
            splitter_destinations: 0,
            revision: u32::from_le_bytes(*b"REV7"),
            rendering_device: 0,
            execution_mode: 0,
        };
        let transfer = transfer(&vec![0; 4096]);
        let session =
            run(AudioRendererManagerSession::default().open(parameters, transfer, 4096, 1, 1))
                .unwrap()
                .unwrap();
        let mut sink = [0; 0x140];
        sink[0] = 1;
        sink[1] = 1;
        sink[0x120..0x124].copy_from_slice(&4_u32.to_le_bytes());
        session.0.update.take_blocking().sinks = vec![sink];
        run(session.0.start(&Backend::default())).unwrap();
        wait_until(|| session.0.event.is_signalled());
        let state = session.0.update.take_blocking();
        assert_eq!(state.frames, 0);
        assert!(!state.started);
        assert!(
            session
                .0
                .failure
                .get()
                .unwrap()
                .to_string()
                .contains("4 channels")
        );
    }
}
