//! Host packets; graphics recording stays on the GPU owner, WSI on its own queue.
use crate::*;
use std::sync::{Arc, Mutex, mpsc};
mod frames;
mod render;
mod wsi;
pub(crate) use frames::Frames;
pub(crate) use render::ScreenRenderer;
pub(crate) use wsi::Presenter;
mod host;
pub(crate) use host::Host;
#[cfg(test)]
mod tests;

pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures: egui::TexturesDelta,
    pub pixels_per_point: f32,
}
pub struct Screenshot {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}
pub struct HostFrame {
    pub frame: Option<Arc<nixe_video::PresentationFrame>>,
    pub size: [u32; 2],
    pub ui: Option<UiFrame>,
    pub screenshot: Option<mpsc::Sender<Result<Screenshot, String>>>,
}
impl Drop for HostFrame {
    fn drop(&mut self) {
        // An unconsumed packet is canceled (shutdown, failure or rejected
        // admission). Deltas already accepted by the owner move into Host::ui.
        if let Some(ui) = &mut self.ui {
            ui.textures.clear();
        }
    }
}
#[derive(Default)]
struct Control {
    pending: Option<HostFrame>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
    stopped: bool,
    failure: Option<String>,
    presented: std::collections::VecDeque<std::time::Instant>,
    notify: Option<Arc<dyn Fn() + Send + Sync>>,
}
/// One pending host packet. Font deltas are never overwritten or lost: the
/// main thread waits for admission before advancing egui's texture stream.
#[derive(Clone, Default)]
pub struct PresentationClient(Arc<Mutex<Control>>);
impl PresentationClient {
    pub fn set_notifier(&self, notify: Arc<dyn Fn() + Send + Sync>) {
        self.0.lock().unwrap().notify = Some(notify);
    }
    pub fn take_presented(&self) -> Vec<std::time::Instant> {
        self.0.lock().unwrap().presented.drain(..).collect()
    }
    pub(crate) fn record_presented(&self) {
        let notify = {
            let mut state = self.0.lock().unwrap();
            if state.presented.len() == 64 {
                state.presented.pop_front();
            }
            state.presented.push_back(std::time::Instant::now());
            state.notify.clone()
        };
        if let Some(notify) = notify {
            notify();
        }
    }
    pub fn ready(&self) -> Result<bool, String> {
        let state = self.0.lock().unwrap();
        if let Some(failure) = &state.failure {
            return Err(failure.clone());
        }
        Ok(!state.stopped && state.pending.is_none())
    }
    pub fn submit(&self, packet: HostFrame) -> Result<(), String> {
        let wake = {
            let mut state = self.0.lock().unwrap();
            if let Some(failure) = &state.failure {
                return Err(failure.clone());
            }
            if state.stopped {
                return Err("Vulkan presentation owner stopped".into());
            }
            if state.pending.is_some() {
                return Err("host packet submitted without admission".into());
            }
            state.pending = Some(packet);
            state.wake.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
        Ok(())
    }
    pub fn close(&self) {
        let wake = {
            let mut state = self.0.lock().unwrap();
            state.stopped = true;
            state.pending = None;
            state.wake.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
    }
    pub(crate) fn bind_wakeup(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.0.lock().unwrap().wake = Some(wake);
    }
    pub(crate) fn take(&self) -> Option<HostFrame> {
        self.0.lock().unwrap().pending.take()
    }
    pub(crate) fn fail(&self, error: String) {
        let (notify, wake) = {
            let mut state = self.0.lock().unwrap();
            state.failure = Some(error);
            (state.notify.clone(), state.wake.clone())
        };
        if let Some(notify) = notify {
            notify();
        }
        if let Some(wake) = wake {
            wake();
        }
    }
    pub fn stopped(&self) -> bool {
        self.0.lock().unwrap().stopped
    }
}

#[derive(Clone)]
pub(crate) struct Source {
    pub image: nixe_gpu::ResidentImage,
    pub crop: nixe_video::FrameCrop,
    pub transform: nixe_video::FrameTransform,
    pub sequence: u64,
}
impl Source {
    pub(crate) fn from_frame(frame: &nixe_video::PresentationFrame) -> Self {
        Self {
            image: frame.image().clone(),
            crop: frame.crop(),
            transform: frame.transform(),
            sequence: frame.sequence(),
        }
    }
    pub(crate) fn dimensions(&self) -> [u32; 2] {
        if self.transform.rotate_90_clockwise {
            [self.crop.height, self.crop.width]
        } else {
            [self.crop.width, self.crop.height]
        }
    }
}
