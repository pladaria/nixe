//! Console output routes and shared device-volume state. The virtual TV route
//! feeds the real host audio backend; logical port names are not host devices.
//!
//! https://switchbrew.org/wiki/Audio_services#IAudioDevice
//! https://github.com/eden-emulator/mirror/blob/master/src/audio_core/renderer/audio_device.cpp

use nixe_audio::{AudioBackend, AudioError};
use nixe_runtime::{EventObject, ExternalEventSource, ReadableEventObject};
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

pub(crate) const DEVICE_NAMES: [&str; 5] = [
    "AudioStereoJackOutput",
    "AudioBuiltInSpeakerOutput",
    "AudioTvOutput",
    "AudioUsbDeviceOutput",
    "AudioExternalOutput",
];
pub(crate) const OUTPUT_NAMES: [&str; 3] = [DEVICE_NAMES[1], DEVICE_NAMES[2], DEVICE_NAMES[4]];
pub(crate) const ACTIVE_DEVICE: usize = 2;

#[derive(Debug)]
pub(crate) struct AudioDeviceState {
    // Device gains are shared across manager/device connections, renderers and
    // AudioOut, including streams created after the volume was set.
    volumes: [AtomicU32; DEVICE_NAMES.len()],
}

impl Default for AudioDeviceState {
    fn default() -> Self {
        Self {
            volumes: std::array::from_fn(|_| AtomicU32::new(1_f32.to_bits())),
        }
    }
}

impl AudioDeviceState {
    pub(crate) fn volume(&self, device: usize) -> f32 {
        f32::from_bits(self.volumes[device].load(Ordering::Relaxed))
    }
    pub(crate) fn set_volume(&self, device: usize, volume: f32) {
        self.volumes[device].store(volume.to_bits(), Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
pub struct AudioDeviceSession {
    pub(crate) state: Arc<AudioDeviceState>,
    revision: u32,
    backend: Arc<dyn AudioBackend>,
    pub(crate) output_event: ReadableEventObject,
    pub(crate) input_event: ReadableEventObject,
    _applet_resource_user_id: u64,
}

impl AudioDeviceSession {
    pub(crate) fn new(
        state: Arc<AudioDeviceState>,
        revision: u32,
        applet_resource_user_id: u64,
        backend: Arc<dyn AudioBackend>,
    ) -> Result<Self, AudioError> {
        // Do not advertise an initialized output when its backend is unavailable.
        backend.output_layout()?;
        let (signal, output_event) =
            EventObject::create_pair_with_source(ExternalEventSource::Device);
        // Initial output-state publication. Queries return the same notification
        // and do not re-signal it or conflate it with DSP block completion.
        signal.signal();
        // The virtual console has no connected input route. Its notification is
        // separate and unsignalled; this does not advertise microphone support.
        let (_, input_event) = EventObject::create_pair_with_source(ExternalEventSource::Device);
        Ok(Self {
            state,
            revision,
            backend,
            output_event,
            input_event,
            _applet_resource_user_id: applet_resource_user_id,
        })
    }
    pub(crate) fn names(&self) -> &'static [&'static str] {
        // AudioUsbDeviceOutput is introduced by renderer REV4.
        // https://github.com/eden-emulator/mirror/blob/master/src/audio_core/common/feature_support.h
        &DEVICE_NAMES[..if self.revision >= 4 { 4 } else { 3 }]
    }
    pub(crate) fn volume_device(&self, name: &[u8]) -> Option<usize> {
        // Device API v2 adds logical output names. External output is a
        // volume route, while the legacy port list still has jack and USB.
        // https://switchbrew.org/wiki/Audio_services#IAudioDevice
        let count = if self.revision >= 10 {
            DEVICE_NAMES.len()
        } else {
            self.names().len()
        };
        DEVICE_NAMES[..count]
            .iter()
            .position(|device| device.as_bytes() == name)
    }
    pub(crate) fn channels(&self) -> Result<u32, AudioError> {
        Ok(self.backend.output_layout()?.channels() as u32)
    }
}
