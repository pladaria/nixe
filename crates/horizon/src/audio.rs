//! Horizon output-session lifetime and kernel event ownership.

use nixe_audio::{AudioBackend, AudioError, AudioOutput, PcmFormat};
use nixe_runtime::{EventObject, ExternalEventSource, ReadableEventObject};
use std::sync::{Arc, Mutex};

/// audout's currently implemented guest PCM format. Host adaptation is separate.
pub(crate) const OUTPUT_FORMAT: PcmFormat = PcmFormat::STEREO_48KHZ;

#[derive(Clone, Debug)]
pub struct AudioOutManagerSession(pub(crate) Option<Arc<dyn AudioBackend>>);

#[derive(Clone, Debug)]
pub struct AudioOutSession {
    pub(crate) output: Arc<Mutex<AudioOutput>>,
    pub(crate) event: ReadableEventObject,
}

impl AudioOutSession {
    pub(crate) fn open(backend: &dyn AudioBackend) -> Result<Self, AudioError> {
        let (signal, event) = EventObject::create_pair_with_source(ExternalEventSource::Device);
        let output = AudioOutput::open(
            backend,
            OUTPUT_FORMAT,
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
