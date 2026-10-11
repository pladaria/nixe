//! Horizon output-session lifetime and kernel event ownership.

use nixe_audio::{AudioBackend, AudioError, AudioOutput, PcmFormat};
use nixe_runtime::{EventObject, ExternalEventSource, ReadableEventObject};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct AudioOutManagerSession(pub(crate) Option<Arc<dyn AudioBackend>>);

#[derive(Clone, Debug)]
pub struct AudioOutSession {
    pub(crate) output: Arc<Mutex<AudioOutput>>,
    pub(crate) event: ReadableEventObject,
}

impl AudioOutSession {
    pub(crate) fn open(backend: &dyn AudioBackend, format: PcmFormat) -> Result<Self, AudioError> {
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
        })
    }
}
