use super::*;
use crate::cache::{
    CacheConfiguration, CompilePool, DescriptorCache, PipelineCompiler, SamplerCache,
};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WindowHandle, XlibDisplayHandle, XlibWindowHandle,
};
use std::{ffi::c_void, time::Duration};
use x11_dl::xlib;

struct Window {
    xlib: xlib::Xlib,
    display: usize,
    screen: i32,
    window: std::ffi::c_ulong,
}
impl Window {
    fn new() -> Option<Arc<Self>> {
        if std::env::var_os("DISPLAY").is_none() {
            eprintln!("SKIP Vulkan WSI test: no X11 display");
            return None;
        }
        let xlib = xlib::Xlib::open().expect("load X11 for the available display");
        // Xlib may also be used by the Vulkan driver on its worker thread.
        unsafe {
            assert_ne!((xlib.XInitThreads)(), 0);
            let display = (xlib.XOpenDisplay)(std::ptr::null());
            assert!(!display.is_null(), "open the available X11 display");
            let screen = (xlib.XDefaultScreen)(display);
            let window = (xlib.XCreateSimpleWindow)(
                display,
                (xlib.XRootWindow)(display, screen),
                0,
                0,
                32,
                32,
                0,
                0,
                0,
            );
            assert_ne!(window, 0);
            (xlib.XMapWindow)(display, window);
            (xlib.XSync)(display, 0);
            Some(Arc::new(Self {
                xlib,
                display: display as usize,
                screen,
                window,
            }))
        }
    }
}
impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let raw = XlibDisplayHandle::new(
            std::ptr::NonNull::new(self.display as *mut c_void),
            self.screen,
        );
        // The surface retains this window and its display.
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Xlib(raw)) })
    }
}
impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        Ok(unsafe {
            WindowHandle::borrow_raw(RawWindowHandle::Xlib(XlibWindowHandle::new(self.window)))
        })
    }
}
impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            (self.xlib.XDestroyWindow)(self.display as *mut xlib::Display, self.window);
            (self.xlib.XCloseDisplay)(self.display as *mut xlib::Display);
        }
    }
}

#[test]
fn surface_availability_and_idle_wake_the_owner_without_another_host_packet() {
    let Some(window) = Window::new() else {
        return;
    };
    let Some(probe) = crate::test_device::device_for_test(crate::ApiPreference::Auto) else {
        return;
    };
    let adapter = probe.profile().name.clone();
    drop(probe);
    let device = VulkanDevice::new(
        crate::DeviceOptions {
            validation: crate::ValidationMode::Enabled,
            adapter: Some(adapter),
            ..Default::default()
        },
        Some(crate::SurfaceTarget::new(window)),
    )
    .unwrap();
    let validation = device.validation_monitor();
    let client = PresentationClient::default();
    let (wake, wakes) = mpsc::channel();
    client.bind_wakeup(Arc::new(move || {
        let _ = wake.send(());
    }));
    let mut host = Host::new(&device, client.clone(), false).unwrap();
    let config = CacheConfiguration {
        workers: 1,
        ..Default::default()
    };
    let workers = Arc::new(CompilePool::new(1, 8).unwrap());
    let pipelines = PipelineCompiler::new(&device, workers, &config).unwrap();
    let mut descriptors = DescriptorCache::new(&device);
    let mut samplers = SamplerCache::new(&device, 8);
    let mut cache = VulkanResidency::new(
        nixe_memory::Machine::new(64 * 1024 * 1024).unwrap(),
        device,
        16 * 1024 * 1024,
    )
    .unwrap();
    let mut frames = Frames::new();
    for _ in 0..2 {
        client
            .submit(HostFrame {
                frame: None,
                size: [32, 32],
                ui: None,
                screenshot: None,
            })
            .unwrap();
        wakes.recv_timeout(Duration::from_secs(5)).unwrap();
        let before = cache.device().scheduled_tick();
        host.service(
            &mut cache,
            &mut frames,
            &pipelines,
            &mut descriptors,
            &mut samplers,
        )
        .unwrap();
        assert_eq!(cache.device().scheduled_tick(), before);
        // The acquire result must be published before this notification.
        wakes
            .recv_timeout(Duration::from_secs(5))
            .expect("WSI acquisition wakes the parked owner");
        host.service(
            &mut cache,
            &mut frames,
            &pipelines,
            &mut descriptors,
            &mut samplers,
        )
        .unwrap();
        assert!(cache.device().scheduled_tick() > before);
        wakes
            .recv_timeout(Duration::from_secs(5))
            .expect("WSI idle wakes the owner");
        host.service(
            &mut cache,
            &mut frames,
            &pipelines,
            &mut descriptors,
            &mut samplers,
        )
        .unwrap();
    }
    host.shutdown(&mut cache).unwrap();
    drop(host);
    drop(cache);
    drop(descriptors);
    drop(samplers);
    drop(pipelines);
    validation.check().unwrap();
}
