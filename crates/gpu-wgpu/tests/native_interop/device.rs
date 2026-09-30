use std::ffi::CStr;
use std::sync::Mutex;

struct ValidationLog(Mutex<Vec<String>>);
static VALIDATION_LOG: ValidationLog = ValidationLog(Mutex::new(Vec::new()));

impl log::Log for ValidationLog {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
            if record.level() == log::Level::Error {
                self.0.lock().unwrap().push(record.args().to_string());
            }
        }
    }
    fn flush(&self) {}
}

pub fn enable_validation() {
    assert!(
        wgpu::InstanceFlags::default()
            .contains(wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION),
        "run this test in a debug build so production initialization enables validation"
    );
    enable_logging();
    // wgpu-hal 30 enables synchronization validation when this extension exists.
    // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/instance.rs
    unsafe {
        let entry = ash::Entry::load().expect("Vulkan loader required");
        let name = c"VK_LAYER_KHRONOS_validation";
        assert!(
            entry
                .enumerate_instance_layer_properties()
                .unwrap()
                .iter()
                .any(|p| CStr::from_ptr(p.layer_name.as_ptr()) == name),
            "Khronos validation layer required; this is not a skipped GPU test"
        );
        assert!(
            entry
                .enumerate_instance_extension_properties(Some(name))
                .unwrap()
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr())
                    == ash::ext::validation_features::NAME),
            "VK_EXT_validation_features required for synchronization validation"
        );
    }
}

pub fn enable_logging() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        log::set_logger(&VALIDATION_LOG).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
}

pub fn assert_validation_clean() {
    let errors = VALIDATION_LOG.0.lock().unwrap();
    assert!(
        errors.is_empty(),
        "Vulkan/wgpu errors:\n{}",
        errors.join("\n")
    );
}

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub capabilities: Option<nixe_gpu_wgpu::VulkanNativeCapabilities>,
    // Keep the actual backend's callbacks and teardown ownership through the test.
    _backend: Option<nixe_gpu_wgpu::InitializedWgpuBackend>,
}

impl Context {
    pub fn into_runtime(mut self) -> (Self, Box<dyn nixe_gpu::NeutralBackendRuntime>) {
        let runtime = self._backend.take().unwrap().into_runtime();
        (self, runtime)
    }

    pub fn new(tessellation: bool) -> Self {
        Self::with_cache(
            tessellation,
            nixe_gpu::GpuCacheConfiguration::new(6, 1, 1, 1, 4096).unwrap(),
        )
    }

    pub fn with_cache(tessellation: bool, cache: nixe_gpu::GpuCacheConfiguration) -> Self {
        Self::with_persistence(tessellation, cache, None)
    }

    pub fn with_persistence(
        tessellation: bool,
        cache: nixe_gpu::GpuCacheConfiguration,
        directory: Option<&std::path::Path>,
    ) -> Self {
        if tessellation {
            // Exercise production device creation, not a duplicate prototype.
            let backend = super::hardware::initialize_backend(
                nixe_gpu::BackendInstanceId::new(1),
                nixe_memory::NonCpuDeviceId::new(1),
                nixe_gpu_wgpu::WgpuBackendConfiguration {
                    host_backend: nixe_gpu_wgpu::HostBackend::Vulkan,
                    pipeline_cache_directory: directory.map(std::path::Path::to_path_buf),
                    // Production fixture deliberately evicts cached raw objects
                    // while earlier submissions still retain them.
                    cache,
                    ..Default::default()
                },
            )
            .unwrap();
            let caps = backend
                .adapter
                .native_vulkan
                .expect("native Vulkan device required");
            assert!(caps.tessellation_shader, "native tessellation unavailable");
            eprintln!(
                "production adapter: {:?}; enabled: {caps:?}",
                backend.adapter
            );
            let shared = backend.presentation_context();
            // The backend normally retains asynchronous errors. This opt-in test
            // must fail immediately instead of letting a recorded error go unread.
            shared
                .device()
                .on_uncaptured_error(std::sync::Arc::new(|e| panic!("{e}")));
            return Self {
                device: shared.device().clone(),
                queue: shared.queue().clone(),
                capabilities: Some(caps),
                _backend: Some(backend),
            };
        }
        // An ordinary device deliberately leaves the optional native feature off.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = super::hardware::adapter(&instance, wgpu::Backends::VULKAN)
            .expect("physical GPU disappeared after test preflight");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        Self {
            device,
            queue,
            capabilities: None,
            _backend: None,
        }
    }

    pub fn raw(&self) -> ash::Device {
        // The Context and each native owner retain a wgpu Device clone.
        unsafe {
            self.device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .unwrap()
                .raw_device()
                .clone()
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // Test cleanup/oracle only, never a handoff between normal/native work.
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
}
