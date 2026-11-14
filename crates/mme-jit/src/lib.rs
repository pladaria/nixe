//! Shared AsmJit executable storage for generation-specific MME compilers.
#![cfg(target_arch = "x86_64")]
use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

unsafe extern "C" {
    fn nixe_mme_runtime_new() -> *mut c_void;
    fn nixe_mme_runtime_delete(runtime: *mut c_void);
    fn nixe_mme_release(runtime: *mut c_void, program: *const c_void);
}

/// Thread-confined allocator. Programs must retain their allocator until release.
#[derive(Debug)]
pub struct Runtime {
    pointer: NonNull<c_void>,
    owner: PhantomData<Rc<()>>,
}
impl Runtime {
    pub fn new() -> Option<Self> {
        // SAFETY: allocates an independent native runtime, returning null on failure.
        NonNull::new(unsafe { nixe_mme_runtime_new() }).map(|pointer| Self {
            pointer,
            owner: PhantomData,
        })
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.pointer.as_ptr()
    }
    /// # Safety
    /// `program` must be a live allocation from this runtime, no longer executing.
    pub unsafe fn release(&self, program: *const c_void) {
        // SAFETY: caller guarantees ownership and absence of executing users.
        unsafe { nixe_mme_release(self.as_ptr(), program) };
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: the runtime has unique ownership and its programs retain it.
        unsafe { nixe_mme_runtime_delete(self.as_ptr()) };
    }
}
