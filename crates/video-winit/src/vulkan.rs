//! Window/surface bootstrap for the read-only Vulkan device diagnostic.
//! Production presentation is initialized by WindowFrontend::create_vulkan_device.
use nixe_gpu_vulkan::{DeviceOptions, SurfaceTarget, VulkanDevice};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

/// Create the window/surface first so queue selection uses real present support.
/// The callback receives the sole concrete device owner and may move it to a
/// worker. The native window remains alive until its surface has been destroyed.
pub fn with_vulkan_device<F>(options: DeviceOptions, callback: F) -> Result<(), String>
where
    F: FnOnce(VulkanDevice) -> Result<(), String>,
{
    let event_loop = EventLoop::new().map_err(|e| format!("cannot create event loop: {e}"))?;
    let mut app = Bootstrap {
        options: Some(options),
        callback: Some(callback),
        result: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("Vulkan window event loop failed: {e}"))?;
    app.result
        .unwrap_or_else(|| Err("Vulkan window bootstrap exited before initialization".to_owned()))
}

struct Bootstrap<F> {
    options: Option<DeviceOptions>,
    callback: Option<F>,
    result: Option<Result<(), String>>,
}
impl<F: FnOnce(VulkanDevice) -> Result<(), String>> ApplicationHandler for Bootstrap<F> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(options) = self.options.take() else {
            return;
        };
        self.result = Some((|| {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes().with_title("nixe — Vulkan device check"),
                    )
                    .map_err(|e| format!("cannot create Vulkan window: {e}"))?,
            );
            let device = VulkanDevice::new(options, Some(SurfaceTarget::new(window)))
                .map_err(|e| e.to_string())?;
            self.callback
                .take()
                .expect("bootstrap callback consumed once")(device)
        })());
        event_loop.exit();
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if event == WindowEvent::CloseRequested {
            event_loop.exit();
        }
    }
}
