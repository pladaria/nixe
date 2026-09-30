//! Opt-in headless captures, linked only into the Linux integration test.
use std::ffi::{CString, c_char, c_void};
use std::path::PathBuf;

// ABI 1.0.0 prefix; later RenderDoc APIs append fields without changing it.
// Never invoke the unused entries. Function pointers retain their C ABI layout.
// https://github.com/baldurk/renderdoc/blob/v1.18/renderdoc/api/app/renderdoc_app.h
#[repr(C)]
struct Api {
    unused_before_path: [unsafe extern "C" fn(); 11],
    set_path: unsafe extern "C" fn(*const c_char),
    unused_before_start: [unsafe extern "C" fn(); 7],
    start: unsafe extern "C" fn(*mut c_void, *mut c_void),
    is_capturing: unsafe extern "C" fn() -> u32,
    end: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32,
}

pub struct Capture {
    _library: libloading::os::unix::Library,
    api: *const Api,
    directory: PathBuf,
}

impl Capture {
    pub fn from_environment() -> Option<Self> {
        let directory = PathBuf::from(std::env::var_os("NIXE_TEST_CAPTURE_DIR")?);
        assert!(
            directory.is_absolute() && directory.is_dir(),
            "capture directory must exist and be absolute"
        );
        // Loading RenderDoc after Vulkan device creation would be too late.
        // Require injection (LD_PRELOAD) before process startup instead.
        // https://github.com/baldurk/renderdoc/blob/v1.18/docs/in_application_api.rst
        unsafe {
            let library = libloading::os::unix::Library::this();
            let get_api = library
                .get::<unsafe extern "C" fn(u32, *mut *const Api) -> i32>(b"RENDERDOC_GetAPI\0")
                .expect("preload librenderdoc.so before starting this test process");
            let mut api = std::ptr::null();
            assert_eq!(
                get_api(10000, &mut api),
                1,
                "RenderDoc API 1.0.0 unavailable"
            );
            assert!(!api.is_null());
            Some(Self {
                _library: library,
                api,
                directory,
            })
        }
    }

    pub fn start(&self, name: &str) {
        use std::os::unix::ffi::OsStrExt;
        let path = CString::new(self.directory.join(name).as_os_str().as_bytes()).unwrap();
        // The fixture owns exactly one live device and no window.
        unsafe {
            ((*self.api).set_path)(path.as_ptr());
            assert_eq!(((*self.api).is_capturing)(), 0);
            ((*self.api).start)(std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!(
                ((*self.api).is_capturing)(),
                1,
                "RenderDoc did not start; check its Vulkan layer configuration"
            );
        }
    }

    pub fn end(&self) {
        unsafe {
            assert_eq!(
                ((*self.api).end)(std::ptr::null_mut(), std::ptr::null_mut()),
                1,
                "RenderDoc capture failed"
            );
        }
    }
}
