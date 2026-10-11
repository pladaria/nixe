//! Audio command execution on a dedicated worker. SDL only consumes prepared PCM.
//!
//! Automatic renderer scheduling and frame accounting:
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/system.cpp

use super::{AudioRendererState, render::RenderBuffers};
use nixe_audio::{AudioBackend, AudioOutput, PcmFormat};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const BLOCK: Duration = Duration::from_millis(5);
const BUFFER_COUNT: usize = 3;

#[derive(Debug)]
pub(super) struct RendererWorker {
    stop: Arc<(Mutex<bool>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl RendererWorker {
    pub(super) fn start(
        state: &AudioRendererState,
        backend: &dyn AudioBackend,
    ) -> Result<Self, Box<str>> {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
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
        let thread_stop = stop.clone();
        output.start().map_err(|error| error.0.into_boxed_str())?;
        {
            let mut state = update.lock().unwrap();
            state.frames = 0;
            state.started = true;
        }
        let thread = match thread::Builder::new()
            .name("nixe-audren".into())
            .spawn(move || {
                let result = (|| -> Result<(), Box<str>> {
                    let mut next = Instant::now();
                    loop {
                        let now = Instant::now();
                        let stopped = thread_stop.0.lock().unwrap();
                        if *stopped {
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
                            if *stopped {
                                break;
                            }
                            drop(thread_stop.1.wait_timeout(stopped, BLOCK).unwrap());
                            continue;
                        };
                        {
                            let mut state = update.lock().unwrap();
                            let has_sink = buffers.render(
                                &state,
                                &params,
                                &mut pcm,
                                devices.volume(crate::audio_device::ACTIVE_DEVICE),
                            )?;
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
                    output.stop().map_err(|error| error.0.into_boxed_str())
                })();
                if let Err(error) = result {
                    log::error!("audio renderer worker failed: {error}");
                    let mut state = update.lock().unwrap();
                    state.failure = Some(error);
                    state.started = false;
                    signal.signal();
                }
            }) {
            Ok(thread) => thread,
            Err(error) => {
                state.update.lock().unwrap().started = false;
                return Err(format!("creating audio renderer worker: {error}").into());
            }
        };
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for RendererWorker {
    fn drop(&mut self) {
        *self.stop.0.lock().unwrap() = true;
        self.stop.1.notify_one();
        if let Some(thread) = self.thread.take() {
            thread.join().expect("audio renderer worker panicked");
        }
    }
}

impl AudioRendererState {
    pub(crate) fn start(&self, backend: &dyn AudioBackend) -> Result<(), Box<str>> {
        let mut worker = self.worker.lock().unwrap();
        if worker.is_none() {
            if self.parameters.performance_frames != 0 {
                return Err("Start: renderer performance histories are not implemented".into());
            }
            *worker = Some(RendererWorker::start(self, backend)?);
            log::debug!(
                "audio renderer started: sample-rate={} samples={} pcm-buffers={BUFFER_COUNT}",
                self.parameters.sample_rate,
                self.parameters.samples
            );
        } else {
            // Start resets the elapsed-frame count even for an active renderer.
            self.update.lock().unwrap().frames = 0;
        }
        Ok(())
    }

    pub(crate) fn stop(&self) {
        // The worker must finish before stop returns. It never owns the session,
        // so closing the last guest handle also stops and joins it without a cycle.
        let mut worker = self.worker.lock().unwrap();
        drop(worker.take());
        let mut state = self.update.lock().unwrap();
        state.started = false;
        log::debug!("audio renderer stopped: rendered-blocks={}", state.frames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_renderer::{AudioRendererManagerSession, PARAMETER_SIZE, RendererParameters};
    use nixe_audio::{AudioDevice, AudioError, AudioFeed};
    use nixe_memory::{CanonicalAllocation, GuestVirtualAddress, MemoryPermissions};
    use nixe_runtime::TransferMemoryObject;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let transfer = TransferMemoryObject::new(
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::NONE,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
        );
        let session = AudioRendererManagerSession::default()
            .open(RendererParameters::decode(&data), transfer, 4096, 1, 1)
            .unwrap()
            .unwrap();
        {
            let mut state = session.0.update.lock().unwrap();
            let mut mix = Box::new([0; 0x930]);
            mix[..4].copy_from_slice(&1_f32.to_le_bytes());
            mix[4..8].copy_from_slice(&48000_u32.to_le_bytes());
            mix[8..12].copy_from_slice(&2_u32.to_le_bytes());
            state.mixes = vec![Some(mix)];
            let mut sink = [0; 0x140];
            sink[0] = 1;
            sink[1] = 1;
            sink[0x120..0x124].copy_from_slice(&2_u32.to_le_bytes());
            sink[0x125] = 1;
            state.sinks = vec![sink];
        }
        let backend = Backend::default();
        session.0.start(&backend).unwrap();
        let feed = backend.feed.lock().unwrap().take().unwrap();
        wait_until(|| session.0.update.lock().unwrap().frames == BUFFER_COUNT as u64);
        assert!(session.0.event.is_signalled());
        // No host consumption: execution must stay bounded, not fabricate a
        // renderer frame counter from elapsed wall time.
        thread::sleep(BLOCK * 4);
        assert_eq!(session.0.update.lock().unwrap().frames, BUFFER_COUNT as u64);
        feed.render(&mut [1; 480]);
        wait_until(|| session.0.update.lock().unwrap().frames == BUFFER_COUNT as u64 + 1);
        session.0.stop();
        assert_eq!(backend.live.load(Ordering::SeqCst), 0);
        assert!(!session.0.update.lock().unwrap().started);
        let stopped_frames = session.0.update.lock().unwrap().frames;
        thread::sleep(BLOCK * 2);
        assert_eq!(session.0.update.lock().unwrap().frames, stopped_frames);
        session.0.start(&backend).unwrap();
        wait_until(|| session.0.update.lock().unwrap().frames > 0);
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
            external_context_size: 0,
            revision: u32::from_le_bytes(*b"REV7"),
            rendering_device: 0,
            execution_mode: 0,
        };
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let transfer = TransferMemoryObject::new(
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::NONE,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
        );
        let session = AudioRendererManagerSession::default()
            .open(parameters, transfer, 4096, 1, 1)
            .unwrap()
            .unwrap();
        let mut sink = [0; 0x140];
        sink[0] = 1;
        sink[1] = 1;
        sink[0x120..0x124].copy_from_slice(&4_u32.to_le_bytes());
        session.0.update.lock().unwrap().sinks = vec![sink];
        session.0.start(&Backend::default()).unwrap();
        wait_until(|| session.0.event.is_signalled());
        let state = session.0.update.lock().unwrap();
        assert_eq!(state.frames, 0);
        assert!(!state.started);
        assert!(state.failure.as_ref().unwrap().contains("4 channels"));
    }
}
