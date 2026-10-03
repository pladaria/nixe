//! SDL's subsystem owner stays on the main thread. Stream creation and
//! destruction may run on the emulation thread; PCM is pulled by SDL itself.

use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use sdl3_sys::audio::*;

use crate::{AudioBackend, AudioDevice, AudioError, AudioFeed, ChannelLayout};

struct HostState {
    active: bool,
    layout: ChannelLayout,
    next_id: u64,
    streams: HashMap<u64, NativeStream>,
}

/// Main-thread owner. Close all streams before releasing SDL's audio subsystem,
/// including streams whose guest handles outlive the application's owner.
pub struct HostAudioRuntime {
    state: Arc<Mutex<HostState>>,
    _audio: sdl3::AudioSubsystem,
}

impl HostAudioRuntime {
    pub fn new(sdl: &sdl3::Sdl, layout: ChannelLayout) -> Result<Self, AudioError> {
        let audio = sdl.audio().map_err(error)?;
        Ok(Self {
            state: Arc::new(Mutex::new(HostState {
                active: true,
                layout,
                next_id: 0,
                streams: HashMap::new(),
            })),
            _audio: audio,
        })
    }

    pub fn backend(&self) -> Arc<HostAudioBackend> {
        Arc::new(HostAudioBackend {
            state: self.state.clone(),
        })
    }
}

impl Drop for HostAudioRuntime {
    fn drop(&mut self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active = false;
        for (_, stream) in state.streams.drain() {
            stream
                .feed
                .fail(AudioError("SDL audio output was shut down".into()));
            drop(stream);
        }
    }
}

pub struct HostAudioBackend {
    state: Arc<Mutex<HostState>>,
}

impl std::fmt::Debug for HostAudioBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SdlAudioBackend").finish_non_exhaustive()
    }
}

struct HostDevice {
    id: u64,
    state: Arc<Mutex<HostState>>,
}
impl AudioDevice for HostDevice {}
impl Drop for HostDevice {
    fn drop(&mut self) {
        // Serialize destruction with subsystem shutdown on the main thread.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .streams
            .remove(&self.id);
    }
}

struct Callback {
    feed: AudioFeed,
    scratch: Box<[i16]>,
}

struct NativeStream {
    stream: NonNull<SDL_AudioStream>,
    _callback: Box<Callback>,
    feed: AudioFeed,
}

// All operations used on a live stream are thread-safe. The callback has its
// own userdata and SDL serializes it with the stream lock. Stream destruction
// unbinds the device and waits out callbacks before we free userdata.
// https://wiki.libsdl.org/SDL3/SDL_OpenAudioDeviceStream
// https://wiki.libsdl.org/SDL3/SDL_DestroyAudioStream
unsafe impl Send for NativeStream {}

impl Drop for NativeStream {
    fn drop(&mut self) {
        // SAFETY: Unique owner; callback storage remains alive until this returns.
        unsafe {
            SDL_DestroyAudioStream(self.stream.as_ptr());
        }
    }
}

fn error(error: impl std::fmt::Display) -> AudioError {
    AudioError(error.to_string())
}

impl AudioBackend for HostAudioBackend {
    fn open(&self, feed: AudioFeed) -> Result<Box<dyn AudioDevice>, AudioError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.active {
            return Err(AudioError("SDL audio subsystem has been shut down".into()));
        }
        let format = feed.format();
        if format.layout != state.layout {
            return Err(AudioError(
                "PCM channel layout does not match the configured output".into(),
            ));
        }
        let id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| AudioError("audio stream identifiers exhausted".into()))?;
        let spec = SDL_AudioSpec {
            format: SDL_AUDIO_S16,
            channels: format.layout.channels() as c_int,
            freq: c_int::try_from(format.sample_rate.get()).map_err(error)?,
        };
        // Ten milliseconds of scratch storage, reused for arbitrary SDL demand.
        // SDL adapts this application's PCM to the actual device format/rate.
        // https://wiki.libsdl.org/SDL3/SDL_AudioStream
        let scratch = vec![
            0_i16;
            (format.sample_rate.get() as usize / 100).max(1)
                * format.layout.channels()
        ]
        .into_boxed_slice();
        let mut callback = Box::new(Callback { feed, scratch });
        // SAFETY: SDL starts this stream paused; userdata remains at its boxed
        // address through destruction. The audio subsystem is held by the owner.
        let pointer = unsafe {
            SDL_OpenAudioDeviceStream(
                SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
                &spec,
                Some(render),
                (&mut *callback as *mut Callback).cast(),
            )
        };
        let Some(stream) = NonNull::new(pointer) else {
            return Err(error(sdl3::get_error()));
        };
        let native = NativeStream {
            stream,
            feed: callback.feed.clone(),
            _callback: callback,
        };
        // Keep the host clock running; stopped guest sessions supply silence.
        // Pausing SDL would retain stale queued PCM across guest Stop/Start.
        if !unsafe { SDL_ResumeAudioStreamDevice(native.stream.as_ptr()) } {
            return Err(error(sdl3::get_error()));
        }
        state.next_id = id;
        state.streams.insert(id, native);
        Ok(Box::new(HostDevice {
            id,
            state: self.state.clone(),
        }))
    }
}

unsafe extern "C" fn render(
    userdata: *mut c_void,
    stream: *mut SDL_AudioStream,
    additional: c_int,
    _total: c_int,
) {
    // SAFETY: The stream lock serializes callbacks; NativeStream keeps this box
    // alive until SDL_DestroyAudioStream has detached and stopped the callback.
    let callback = unsafe { &mut *userdata.cast::<Callback>() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let frame_bytes = callback.feed.format().frame_bytes();
        let mut samples = (additional.max(0) as usize).div_ceil(frame_bytes)
            * callback.feed.format().layout.channels();
        while samples != 0 {
            let count = samples.min(callback.scratch.len());
            let pcm = &mut callback.scratch[..count];
            callback.feed.render(pcm);
            // SAFETY: SDL copies this fully initialized S16 PCM before returning.
            // https://wiki.libsdl.org/SDL3/SDL_PutAudioStreamData
            if !unsafe {
                SDL_PutAudioStreamData(
                    stream,
                    pcm.as_ptr().cast(),
                    std::mem::size_of_val(pcm) as c_int,
                )
            } {
                callback.feed.fail(error(sdl3::get_error()));
                break;
            }
            samples -= count;
        }
    }));
    if result.is_err() {
        callback
            .feed
            .fail(AudioError("SDL audio callback panicked".into()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioOutput, PcmFormat};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn sdl_backend_preserves_pcm_pitch_channels_and_stream_lifetimes() {
        // Native backend test only: SDL's dummy device, never a guest executable
        // or a real speaker. This is the only test initializing SDL in this crate.
        assert!(sdl3::hint::set_with_priority(
            "SDL_AUDIO_DRIVER",
            "dummy",
            &sdl3::hint::Hint::Override
        ));
        let sdl = sdl3::init().unwrap();
        let gamepads = sdl.gamepad().unwrap();
        let runtime = HostAudioRuntime::new(&sdl, ChannelLayout::Stereo).unwrap();
        verify_callback_pitch_and_channels();
        let backend = runtime.backend();
        let (send, receive) = mpsc::channel();
        let output = AudioOutput::open(
            backend.as_ref(),
            PcmFormat::STEREO_48KHZ,
            Arc::new(move |ready| {
                if ready {
                    let _ = send.send(());
                }
            }),
        )
        .unwrap();
        assert_eq!(backend.state.lock().unwrap().streams.len(), 1);
        output.append(0x1234, [123, -123].repeat(480)).unwrap();
        assert!(output.released(1).unwrap().is_empty());
        output.start().unwrap();
        receive.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(output.released(1).unwrap(), [0x1234]);
        assert_eq!(output.played_frames().unwrap(), 480);
        output.stop().unwrap();
        output.append(0x5678, [10, -10].repeat(480)).unwrap();
        output.start().unwrap();
        receive.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(output.released(1).unwrap(), [0x5678]);

        // Ordinary guest Close destroys its SDL stream on the guest thread.
        std::thread::spawn(move || drop(output)).join().unwrap();
        assert!(backend.state.lock().unwrap().streams.is_empty());
        let surviving =
            AudioOutput::open(backend.as_ref(), PcmFormat::STEREO_48KHZ, Arc::new(|_| {})).unwrap();
        drop(runtime);
        assert!(backend.state.lock().unwrap().streams.is_empty());
        assert!(
            surviving
                .started()
                .unwrap_err()
                .to_string()
                .contains("shut down")
        );
        assert!(
            AudioOutput::open(backend.as_ref(), PcmFormat::STEREO_48KHZ, Arc::new(|_| {}))
                .unwrap_err()
                .to_string()
                .contains("shut down")
        );
        drop(surviving);
        // Closing audio did not shut down SDL's independently held input subsystem.
        gamepads.gamepads().unwrap();
    }

    // Pull through the production callback and SDL's actual converter without
    // binding a speaker. Both rounds exchange 440/660 Hz between L/R, like the
    // audout demo; this tests PCM transport, not guest instruction execution.
    fn verify_callback_pitch_and_channels() {
        struct Device;
        impl AudioDevice for Device {}

        for output_rate in [48_000, 44_100, 96_000] {
            for frequencies in [[440, 660], [660, 440]] {
                let feed = AudioFeed {
                    format: PcmFormat::STEREO_48KHZ,
                    state: Arc::new(Mutex::new(crate::State::default())),
                    event: Arc::new(|_| {}),
                };
                let output = AudioOutput {
                    _device: Box::new(Device),
                    feed: feed.clone(),
                };
                let expected: Vec<i16> = (0..115_200)
                    .flat_map(|frame| {
                        frequencies.map(|frequency| {
                            let phase = frame * frequency % 48_000;
                            let slope = 8_000 * phase / 48_000;
                            (if phase < 24_000 {
                                slope - 2_000
                            } else {
                                6_000 - slope
                            }) as i16
                        })
                    })
                    .collect();
                for (index, pcm) in expected.chunks(9_600).enumerate() {
                    output.append(index as u64, pcm.to_vec()).unwrap();
                }
                output.start().unwrap();
                let source = SDL_AudioSpec {
                    format: SDL_AUDIO_S16,
                    channels: 2,
                    freq: 48_000,
                };
                let destination = SDL_AudioSpec {
                    freq: output_rate,
                    ..source
                };
                let mut callback = Box::new(Callback {
                    feed: feed.clone(),
                    scratch: vec![0; 960].into_boxed_slice(),
                });
                // SAFETY: Specs are valid; the unbound stream and its userdata
                // live until NativeStream drops, exactly as in the host backend.
                let stream = NonNull::new(unsafe { SDL_CreateAudioStream(&source, &destination) })
                    .expect("SDL creates an offline PCM stream");
                assert!(unsafe {
                    SDL_SetAudioStreamGetCallback(
                        stream.as_ptr(),
                        Some(render),
                        (&mut *callback as *mut Callback).cast(),
                    )
                });
                let native = NativeStream {
                    stream,
                    _callback: callback,
                    feed,
                };
                let mut captured = vec![0_i16; output_rate as usize * 12 / 5 * 2];
                let mut cursor = 0;
                let mut missing_frames = 0;
                // Uneven pull sizes cross scratch and guest buffer boundaries.
                for frames in [127, 512, 997, 2048].into_iter().cycle() {
                    if cursor == captured.len() {
                        break;
                    }
                    let count = (frames * 2).min(captured.len() - cursor);
                    let pcm = &mut captured[cursor..cursor + count];
                    // SAFETY: SDL writes at most the supplied slice's byte size.
                    let read = unsafe {
                        SDL_GetAudioStreamData(
                            native.stream.as_ptr(),
                            pcm.as_mut_ptr().cast(),
                            std::mem::size_of_val(pcm) as c_int,
                        )
                    };
                    assert!(read >= 0 && read as usize <= count * 2);
                    if read as usize != count * 2 {
                        assert_eq!(cursor, 0, "only initial resampler lookahead may be missing");
                    }
                    // Like a playback device, retain silence for any initial
                    // resampler lookahead that SDL cannot return yet.
                    missing_frames += (count * 2 - read as usize) / 4;
                    cursor += count;
                }
                eprintln!("SDL output={output_rate}Hz missing-frames={missing_frames}");
                if output_rate == 48_000 {
                    assert_eq!(captured, expected, "48 kHz transport must be bit-exact");
                }
                for (channel, expected_hz) in frequencies.into_iter().enumerate() {
                    // Interpolate positive-going zero crossings, excluding the
                    // resampler's boundary transient at either end of the stream.
                    let samples: Vec<_> = captured
                        .chunks_exact(2)
                        .map(|frame| f64::from(frame[channel]))
                        .collect();
                    let margin = output_rate as usize / 10;
                    let crossings: Vec<_> = (margin..samples.len() - margin - 1)
                        .filter(|&i| samples[i] < 0.0 && samples[i + 1] >= 0.0)
                        .map(|i| i as f64 - samples[i] / (samples[i + 1] - samples[i]))
                        .collect();
                    assert!(crossings.len() > 400);
                    let measured = (crossings.len() - 1) as f64 * f64::from(output_rate)
                        / (crossings.last().unwrap() - crossings[0]);
                    eprintln!(
                        "SDL output={output_rate}Hz channel={channel} expected={expected_hz}Hz measured={measured:.3}Hz"
                    );
                    assert!((measured - f64::from(expected_hz)).abs() < 0.1);
                }
                assert_eq!(output.played_frames().unwrap(), 115_200);
                assert_eq!(output.released(32).unwrap(), (0..24).collect::<Vec<_>>());
            }
        }
    }
}
