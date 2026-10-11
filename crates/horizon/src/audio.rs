//! Horizon output-session lifetime and kernel event ownership.

use nixe_audio::{AudioBackend, AudioError, AudioOutput, PcmFormat};
use nixe_runtime::{EventObject, ExternalEventSource, ReadableEventObject};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct AudioOutManagerSession {
    pub(crate) backend: Option<Arc<dyn AudioBackend>>,
    pub(crate) devices: Arc<crate::audio_device::AudioDeviceState>,
}

#[derive(Clone, Debug)]
pub struct AudioOutSession {
    pub(crate) output: Arc<Mutex<AudioOutput>>,
    pub(crate) event: ReadableEventObject,
    pub(crate) devices: Arc<crate::audio_device::AudioDeviceState>,
}

impl AudioOutSession {
    pub(crate) fn open(
        backend: &dyn AudioBackend,
        format: PcmFormat,
        devices: Arc<crate::audio_device::AudioDeviceState>,
    ) -> Result<Self, AudioError> {
        let (signal, event) = EventObject::create_pair_with_source(ExternalEventSource::Device);
        let output = AudioOutput::open(
            backend,
            format,
            Arc::new(move |ready| {
                if ready {
                    signal.signal();
                } else {
                    signal.clear();
                }
            }),
        )?;
        Ok(Self {
            output: Arc::new(Mutex::new(output)),
            event,
            devices,
        })
    }
}
