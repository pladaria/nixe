//! NVIDIA's Linux implicit layer is enabled before Vulkan and host threads start.
//! An explicit layer counts successful downstream presents, including generated
//! frames. These are presentation requests, not confirmed display scanouts.
//! https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/nvpresent.html
use std::sync::Arc;

type ReadCount = unsafe extern "C" fn() -> u64;
type Attached = unsafe extern "C" fn() -> u32;

struct Counter {
    read: ReadCount,
    attached: Attached,
    // Keep the code mapped, and its manifest available, through device teardown.
    _library: libloading::Library,
    _files: tempfile::TempDir,
}

#[derive(Clone)]
pub struct SmoothMotion(Arc<Counter>);

impl SmoothMotion {
    /// Apply the startup preference and install the bundled presentation counter.
    ///
    /// # Safety
    /// Call only during single-threaded process startup, before any Vulkan,
    /// window, tracing, signal, CPU or GPU worker initialization. The Vulkan
    /// loader and NVIDIA read their environment when they are first loaded.
    pub unsafe fn configure(enabled: bool) -> Result<Option<Self>, String> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: the caller guarantees no concurrent environment readers.
            unsafe {
                std::env::set_var(
                    "NVPRESENT_ENABLE_SMOOTH_MOTION",
                    if enabled { "1" } else { "0" },
                );
            }
            if !enabled {
                return Ok(None);
            }
            // SAFETY: startup environment exclusivity is passed to initialize.
            unsafe { Self::initialize().map(Some) }
        }
        #[cfg(not(target_os = "linux"))]
        {
            if enabled {
                Err("NVIDIA Smooth Motion activation is currently supported only on Linux".into())
            } else {
                Ok(None)
            }
        }
    }

    pub fn presentation_count(&self) -> u64 {
        // SAFETY: Counter retains the library; this function reads an atomic.
        unsafe { (self.0.read)() }
    }

    pub fn attached(&self) -> bool {
        // SAFETY: same lifetime; the native layer publishes attached devices.
        unsafe { (self.0.attached)() != 0 }
    }

    #[cfg(target_os = "linux")]
    unsafe fn initialize() -> Result<Self, String> {
        use std::ffi::{CStr, OsString};
        let files = tempfile::Builder::new()
            .prefix("nixe-present-")
            .tempdir()
            .map_err(|e| format!("cannot create presentation counter directory: {e}"))?;
        let library = files.path().join("libnixe_present_counter.so");
        std::fs::write(
            &library,
            include_bytes!(concat!(env!("OUT_DIR"), "/libnixe_present_counter.so")),
        )
        .map_err(|e| format!("cannot extract presentation counter: {e}"))?;
        std::fs::write(
            files.path().join("VkLayer_nixe_present_counter.json"),
            r#"{
  "file_format_version": "1.2.0",
  "layer": {
    "name": "VK_LAYER_NIXE_present_counter",
    "type": "GLOBAL",
    "library_path": "./libnixe_present_counter.so",
    "api_version": "1.3.0",
    "implementation_version": "1",
    "description": "Nixe downstream presentation counter"
  }
}"#,
        )
        .map_err(|e| format!("cannot write presentation counter manifest: {e}"))?;
        // VK_LAYER_PATH overrides VK_ADD_LAYER_PATH, so extend whichever search
        // list the user selected. Explicit layers run below implicit layers.
        // https://vulkan.lunarg.com/doc/view/latest/linux/LoaderLayerInterface.html
        let path_key = if std::env::var_os("VK_LAYER_PATH").is_some() {
            "VK_LAYER_PATH"
        } else {
            "VK_ADD_LAYER_PATH"
        };
        let mut paths = vec![files.path().to_owned()];
        if let Some(existing) = std::env::var_os(path_key) {
            paths.extend(std::env::split_paths(&existing));
        }
        let paths = std::env::join_paths(paths).map_err(|e| e.to_string())?;
        let mut layers = std::env::var_os("VK_INSTANCE_LAYERS").unwrap_or_default();
        if !layers.is_empty() {
            layers.push(":");
        }
        layers.push(OsString::from("VK_LAYER_NIXE_present_counter"));
        // SAFETY: all environment changes finish before loading native libraries;
        // the caller guarantees single-threaded startup.
        unsafe {
            std::env::set_var(path_key, paths);
            std::env::set_var("VK_INSTANCE_LAYERS", layers);
        }
        // SAFETY: this bundled library's exports match the two C declarations.
        let (library, read, attached) = unsafe {
            let library = libloading::Library::new(&library).map_err(|e| e.to_string())?;
            let read = *library
                .get::<ReadCount>(b"nixePresentationCount\0")
                .map_err(|e| e.to_string())?;
            let attached = *library
                .get::<Attached>(b"nixePresentationCounterAttached\0")
                .map_err(|e| e.to_string())?;
            (library, read, attached)
        };
        // SAFETY: loader entry queries are synchronous; environment is final.
        let available = unsafe {
            let entry = ash::Entry::load().map_err(|e| e.to_string())?;
            entry
                .enumerate_instance_layer_properties()
                .map_err(|e| e.to_string())?
        };
        for required in [c"VK_LAYER_NV_present", c"VK_LAYER_NIXE_present_counter"] {
            if !available.iter().any(|layer| {
                // SAFETY: Vulkan layer names are NUL-terminated fixed arrays.
                unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) == required }
            }) {
                return Err(format!(
                    "NVIDIA Smooth Motion requires {}, but the Vulkan loader cannot find it",
                    required.to_string_lossy()
                ));
            }
        }
        log::info!("NVIDIA Smooth Motion requested; downstream presentation measurement enabled");
        Ok(Self(Arc::new(Counter {
            read,
            attached,
            _library: library,
            _files: files,
        })))
    }
}
