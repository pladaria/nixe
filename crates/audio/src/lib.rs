//! Platform-neutral PCM ownership and consumption, with a host playback backend.

#[cfg(feature = "performance-counters")]
pub mod metrics;

use std::collections::VecDeque;
use std::fmt::{self, Debug};
use std::sync::{Arc, Mutex, MutexGuard};

mod host;
pub use host::{HostAudioBackend, HostAudioRuntime};

/// Speaker layout is explicit, rather than inferred from a sample count.
/// New layouts must be wired through Horizon negotiation and host channel maps.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChannelLayout {
    #[default]
    Stereo,
}

impl ChannelLayout {
    pub const fn channels(self) -> usize {
        match self {
            Self::Stereo => 2,
        }
    }
}

/// Interleaved signed-16-bit PCM. Sample format is fixed; rate and layout travel
/// with the feed so the host adapter does not impose console-specific settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcmFormat {
    pub sample_rate: std::num::NonZeroU32,
    pub layout: ChannelLayout,
}

impl PcmFormat {
    pub const STEREO_48KHZ: Self = Self {
        sample_rate: std::num::NonZeroU32::new(48_000).unwrap(),
        layout: ChannelLayout::Stereo,
    };

    pub const fn frame_bytes(self) -> usize {
        self.layout.channels() * size_of::<i16>()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioError(pub String);
impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for AudioError {}

/// A real device owns its stream until drop. Its callback consumes PCM through
/// AudioFeed; neither service calls nor elapsed time complete submitted buffers.
pub trait AudioDevice: Send {}

pub trait AudioBackend: Debug + Send + Sync {
    fn open(&self, feed: AudioFeed) -> Result<Box<dyn AudioDevice>, AudioError>;
}

#[derive(Debug)]
struct Buffer {
    tag: u64,
    samples: Vec<i16>,
    cursor: usize,
}

#[derive(Default)]
struct State {
    started: bool,
    queued: VecDeque<Buffer>,
    released: VecDeque<Buffer>,
    played_frames: u64,
    failure: Option<AudioError>,
}

/// Callback-side view of an interleaved signed-16-bit PCM playback queue.
#[derive(Clone)]
pub struct AudioFeed {
    format: PcmFormat,
    state: Arc<Mutex<State>>,
    event: Arc<dyn Fn(bool) + Send + Sync>,
}

impl AudioFeed {
    pub fn format(&self) -> PcmFormat {
        self.format
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Fill one playback quantum with interleaved signed-16-bit PCM.
    /// Empty queues produce silence, without advancing guest buffer completion.
    pub fn render(&self, output: &mut [i16]) {
        output.fill(0);
        let mut state = self.lock();
        if !state.started || state.failure.is_some() {
            return;
        }
        let mut written = 0;
        let mut released = false;
        let capacity = output.len() / self.format.layout.channels() * self.format.layout.channels();
        while written < capacity {
            let Some(buffer) = state.queued.front_mut() else {
                break;
            };
            let count = (buffer.samples.len() - buffer.cursor).min(capacity - written);
            output[written..written + count]
                .copy_from_slice(&buffer.samples[buffer.cursor..buffer.cursor + count]);
            buffer.cursor += count;
            written += count;
            if buffer.cursor == buffer.samples.len() {
                let buffer = state.queued.pop_front().unwrap();
                // Capacity was reserved by append. Keep sample storage until
                // the caller drains releases, avoiding allocator work here.
                state.released.push_back(buffer);
                released = true;
            }
        }
        #[cfg(feature = "performance-counters")]
        metrics::rendered(capacity, written);
        state.played_frames = state
            .played_frames
            .saturating_add((written / self.format.layout.channels()) as u64);
        if released {
            (self.event)(true);
        }
    }

    /// Device failures wake a waiting consumer, which receives the retained error.
    pub fn fail(&self, error: AudioError) {
        self.lock().failure = Some(error);
        (self.event)(true);
    }
}

pub struct AudioOutput {
    // Declared first so destruction stops callbacks before dropping our feed.
    _device: Box<dyn AudioDevice>,
    feed: AudioFeed,
}

impl Debug for AudioOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioOutput").finish_non_exhaustive()
    }
}

impl AudioOutput {
    pub fn open(
        backend: &dyn AudioBackend,
        format: PcmFormat,
        event: Arc<dyn Fn(bool) + Send + Sync>,
    ) -> Result<Self, AudioError> {
        let feed = AudioFeed {
            format,
            state: Arc::new(Mutex::new(State::default())),
            event,
        };
        let device = backend.open(feed.clone())?;
        Ok(Self {
            _device: device,
            feed,
        })
    }

    fn state(&self) -> Result<MutexGuard<'_, State>, AudioError> {
        let state = self.feed.lock();
        if let Some(error) = &state.failure {
            return Err(error.clone());
        }
        Ok(state)
    }

    pub fn started(&self) -> Result<bool, AudioError> {
        Ok(self.state()?.started)
    }

    pub fn start(&self) -> Result<(), AudioError> {
        if self.state()?.started {
            return Ok(());
        }
        self.state()?.started = true;
        Ok(())
    }

    pub fn stop(&self) -> Result<(), AudioError> {
        if !self.state()?.started {
            return Ok(());
        }
        let mut state = self.state()?;
        state.started = false;
        while let Some(buffer) = state.queued.pop_front() {
            state.released.push_back(buffer);
        }
        (self.feed.event)(!state.released.is_empty());
        Ok(())
    }

    pub fn append(&self, tag: u64, samples: Vec<i16>) -> Result<(), AudioError> {
        if !samples
            .len()
            .is_multiple_of(self.feed.format.layout.channels())
        {
            return Err(AudioError(
                "PCM buffer must contain complete stereo frames".into(),
            ));
        }
        let mut state = self.state()?;
        state
            .queued
            .try_reserve(1)
            .map_err(|e| AudioError(e.to_string()))?;
        let required = state.queued.len() + 1;
        state
            .released
            .try_reserve(required)
            .map_err(|e| AudioError(e.to_string()))?;
        state.queued.push_back(Buffer {
            tag,
            samples,
            cursor: 0,
        });
        Ok(())
    }

    pub fn contains(&self, tag: u64) -> Result<bool, AudioError> {
        let state = self.state()?;
        Ok(state
            .queued
            .iter()
            .chain(&state.released)
            .any(|buffer| buffer.tag == tag))
    }

    pub fn released(&self, max: usize) -> Result<Vec<u64>, AudioError> {
        let mut state = self.state()?;
        let count = max.min(state.released.len());
        let tags = state
            .released
            .drain(..count)
            .map(|buffer| buffer.tag)
            .collect();
        // Serialize reset with callback publication to avoid a lost wakeup.
        (self.feed.event)(!state.released.is_empty());
        Ok(tags)
    }

    pub fn owned_buffer_count(&self) -> Result<usize, AudioError> {
        let state = self.state()?;
        Ok(state.queued.len() + state.released.len())
    }

    pub fn buffer_count(&self) -> Result<usize, AudioError> {
        Ok(self.state()?.queued.len())
    }

    pub fn played_frames(&self) -> Result<u64, AudioError> {
        Ok(self.state()?.played_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Default)]
    struct Backend(Mutex<Option<AudioFeed>>);
    impl Debug for Backend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("Backend")
        }
    }
    struct Device;
    impl AudioDevice for Device {}
    impl AudioBackend for Backend {
        fn open(&self, feed: AudioFeed) -> Result<Box<dyn AudioDevice>, AudioError> {
            *self.0.lock().unwrap() = Some(feed);
            Ok(Box::new(Device))
        }
    }
    fn output() -> (AudioOutput, AudioFeed, Arc<AtomicBool>) {
        let backend = Backend::default();
        let ready = Arc::new(AtomicBool::new(false));
        let signal = ready.clone();
        let output = AudioOutput::open(
            &backend,
            PcmFormat::STEREO_48KHZ,
            Arc::new(move |r| signal.store(r, Ordering::SeqCst)),
        )
        .unwrap();
        let feed = backend.0.lock().unwrap().take().unwrap();
        (output, feed, ready)
    }

    #[test]
    fn consumption_releases_complete_buffers_and_retains_uncollected_ownership() {
        let (output, feed, ready) = output();
        output.append(11, vec![1, 2, 3, 4]).unwrap();
        output.append(22, vec![5, 6]).unwrap();
        let mut frame = [99; 2];
        feed.render(&mut frame);
        assert_eq!(frame, [0, 0]);
        assert_eq!(output.played_frames().unwrap(), 0);
        assert!(output.released(1).unwrap().is_empty());
        output.start().unwrap();
        feed.render(&mut frame);
        assert_eq!(frame, [1, 2]);
        assert!(!ready.load(Ordering::SeqCst));
        let mut rest = [99; 6];
        feed.render(&mut rest);
        assert_eq!(rest, [3, 4, 5, 6, 0, 0]);
        assert_eq!(output.played_frames().unwrap(), 3);
        assert_eq!(output.buffer_count().unwrap(), 0);
        assert_eq!(output.owned_buffer_count().unwrap(), 2);
        assert!(output.contains(11).unwrap());
        assert!(output.released(0).unwrap().is_empty());
        assert!(ready.load(Ordering::SeqCst));
        assert_eq!(output.released(1).unwrap(), [11]);
        assert!(ready.load(Ordering::SeqCst));
        assert!(!output.contains(11).unwrap());
        assert_eq!(output.released(1).unwrap(), [22]);
        assert!(!ready.load(Ordering::SeqCst));
        output.append(11, vec![7, 8]).unwrap();
        feed.render(&mut frame);
        assert_eq!(frame, [7, 8]);
        assert_eq!(output.released(1).unwrap(), [11]);
    }

    #[test]
    fn stop_releases_pending_buffers_without_counting_unplayed_samples_and_restart_works() {
        let (output, feed, ready) = output();
        output.append(1, vec![1, 2, 3, 4]).unwrap();
        output.stop().unwrap(); // Already stopped: preserves prequeued buffers.
        assert_eq!(output.buffer_count().unwrap(), 1);
        output.start().unwrap();
        feed.render(&mut [0; 2]);
        output.stop().unwrap();
        assert!(ready.load(Ordering::SeqCst));
        assert_eq!(output.released(1).unwrap(), [1]);
        assert_eq!(output.played_frames().unwrap(), 1);
        output.append(2, vec![9, 10]).unwrap();
        let mut frame = [0; 2];
        feed.render(&mut frame);
        assert_eq!(frame, [0, 0]);
        output.start().unwrap();
        feed.render(&mut frame);
        assert_eq!(frame, [9, 10]);
        assert_eq!(output.released(1).unwrap(), [2]);
    }

    #[test]
    fn device_failure_wakes_waiters_and_is_not_reported_as_successful_playback() {
        let (output, feed, ready) = output();
        output.append(1, vec![1, 2]).unwrap();
        output.start().unwrap();
        let failure = AudioError("device disconnected".into());
        feed.fail(failure.clone());
        assert!(ready.load(Ordering::SeqCst));
        assert_eq!(output.released(1), Err(failure.clone()));
        assert_eq!(output.started(), Err(failure));
        let mut frame = [1; 2];
        feed.render(&mut frame);
        assert_eq!(frame, [0, 0]);
    }
}
