use nixe_cpu::state::a64::A64State;
use std::ffi::{CStr, c_char, c_void};

#[repr(C)]
pub struct Callbacks {
    pub code: unsafe extern "C" fn(*mut c_void, u64, *mut u32) -> bool,
    pub memory: unsafe extern "C" fn(*mut c_void, u64, u32, u32, *mut u64, *const u64) -> bool,
    pub counter: unsafe extern "C" fn(*mut c_void) -> u64,
    pub data_cache: unsafe extern "C" fn(*mut c_void, u64) -> bool,
}
#[repr(C)]
#[derive(Default)]
pub struct Exit {
    pub kind: u32,
    pub detail: u32,
    pub pc: u64,
    pub value: u64,
    pub ticks: u64,
}
unsafe extern "C" {
    pub fn nixe_dynarmic_resolve_fault(pc: u64, call: *mut u64, ret: *mut u64) -> bool;
    fn nixe_dynarmic_error() -> *const c_char;
    pub fn nixe_dynarmic_monitor_create(count: usize) -> *mut c_void;
    pub fn nixe_dynarmic_monitor_destroy(monitor: *mut c_void);
    pub fn nixe_dynarmic_create(
        cb: Callbacks,
        monitor: *mut c_void,
        id: usize,
        arena: usize,
        bits: usize,
        frequency: u32,
        ctr: u32,
        dczid: u32,
    ) -> *mut c_void;
    pub fn nixe_dynarmic_destroy(core: *mut c_void);
    pub fn nixe_dynarmic_halt(core: *mut c_void);
    pub fn nixe_dynarmic_clear_halt(core: *mut c_void);
    pub fn nixe_dynarmic_clear_exclusive(core: *mut c_void);
    pub fn nixe_dynarmic_invalidate(core: *mut c_void, addr: u64, size: usize) -> bool;
    pub fn nixe_dynarmic_load(core: *mut c_void, state: *const A64State);
    pub fn nixe_dynarmic_save(core: *mut c_void, state: *mut A64State);
    pub fn nixe_dynarmic_read(core: *mut c_void, reg: u32) -> u64;
    pub fn nixe_dynarmic_write(core: *mut c_void, reg: u32, value: u64);
    pub fn nixe_dynarmic_read_vector(core: *mut c_void, reg: u32, value: *mut u64);
    pub fn nixe_dynarmic_write_vector(core: *mut c_void, reg: u32, value: *const u64);
    pub fn nixe_dynarmic_run(
        core: *mut c_void,
        context: *mut c_void,
        budget: u64,
        result: *mut Exit,
    ) -> bool;
}
pub fn error() -> Box<str> {
    // The bridge owns this thread-local string until the next native failure.
    unsafe {
        CStr::from_ptr(nixe_dynarmic_error())
            .to_string_lossy()
            .into_owned()
            .into_boxed_str()
    }
}
