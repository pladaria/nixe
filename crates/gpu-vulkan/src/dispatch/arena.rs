//! Recycled typed captures, invoked and destroyed in place.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.h
use super::worker::Context;
use crate::queue::progress::NativeFailure;
use std::{alloc::Layout, mem::MaybeUninit};

const BYTES: usize = 32 * 1024;
type Invoke = unsafe fn(*mut Header, &mut Context) -> Result<(), NativeFailure>;
struct Vtable {
    invoke: Invoke,
    destroy: unsafe fn(*mut Header),
}
struct Header {
    table: &'static Vtable,
    next: u32,
    lengths: [u32; 3],
}
pub(super) struct Plan {
    layout: Layout,
    capture: usize,
    arrays: [usize; 3],
    lengths: [u32; 3],
}
impl Plan {
    pub fn new<F, A, B, C>(lengths: [u32; 3]) -> Self {
        const {
            assert!(align_of::<F>() <= 16);
            assert!(align_of::<A>() <= 16);
            assert!(align_of::<B>() <= 16);
            assert!(align_of::<C>() <= 16);
        }
        let (layout, capture) = Layout::new::<Header>().extend(Layout::new::<F>()).unwrap();
        let (layout, a) = layout
            .extend(Layout::array::<A>(lengths[0] as usize).unwrap())
            .unwrap();
        let (layout, b) = layout
            .extend(Layout::array::<B>(lengths[1] as usize).unwrap())
            .unwrap();
        let (layout, c) = layout
            .extend(Layout::array::<C>(lengths[2] as usize).unwrap())
            .unwrap();
        Self {
            layout: layout.align_to(16).unwrap().pad_to_align(),
            capture,
            arrays: [a, b, c],
            lengths,
        }
    }
}

pub(super) struct Arena {
    storage: Vec<MaybeUninit<u128>>,
    used: usize,
    cursor: usize,
    count: usize,
}
// SAFETY: captures are Send + 'static. Array admission is unsafe and requires
// owned, pointer-free native values; no borrowed Vulkan chains cross the queue.
unsafe impl Send for Arena {}
impl Arena {
    pub fn new() -> Self {
        Self {
            storage: Vec::with_capacity(BYTES / 16),
            used: 0,
            cursor: 0,
            count: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn identity(&self) -> u64 {
        self.storage.as_ptr() as u64
    }
    pub fn fits(&self, plan: &Plan) -> bool {
        self.used
            .checked_add(plan.layout.size())
            .is_some_and(|end| end <= self.storage.capacity() * 16)
    }
    /// `arrays` contain only owned Copy values safe to transfer to the worker.
    /// `plan` describes these capture/array types and lengths; it fits the arena
    /// or the arena is empty and may grow before placing its first capture.
    pub unsafe fn push<F, A: Copy, B: Copy, C: Copy>(
        &mut self,
        arrays: (&[A], &[B], &[C]),
        function: F,
        plan: Plan,
    ) where
        F: Fn(&mut Context, &[A], &[B], &[C]) -> Result<(), NativeFailure> + Send + 'static,
    {
        if self.count == 0 && plan.layout.size() > self.storage.capacity() * 16 {
            // One oversized operation is kept whole; storage grows only while
            // empty, so queued captures are never relocated.
            self.storage.reserve(plan.layout.size().div_ceil(16));
        }
        let next = self.used.checked_add(plan.layout.size()).unwrap();
        let next_u32 = u32::try_from(next).expect("native capture exceeds 4 GiB");
        debug_assert!(next <= self.storage.capacity() * 16);
        let base = unsafe { self.storage.as_mut_ptr().cast::<u8>().add(self.used) };
        unsafe {
            base.cast::<Header>().write(Header {
                table: &Vtable {
                    invoke: invoke::<F, A, B, C>,
                    destroy: destroy::<F>,
                },
                next: next_u32,
                lengths: plan.lengths,
            });
            base.add(plan.capture).cast::<F>().write(function);
            std::ptr::copy_nonoverlapping(
                arrays.0.as_ptr(),
                base.add(plan.arrays[0]).cast::<A>(),
                arrays.0.len(),
            );
            std::ptr::copy_nonoverlapping(
                arrays.1.as_ptr(),
                base.add(plan.arrays[1]).cast::<B>(),
                arrays.1.len(),
            );
            std::ptr::copy_nonoverlapping(
                arrays.2.as_ptr(),
                base.add(plan.arrays[2]).cast::<C>(),
                arrays.2.len(),
            );
        }
        self.used = next;
        self.count += 1;
    }
    pub fn execute(&mut self, context: &mut Context) -> Result<(), NativeFailure> {
        while self.count != 0 {
            let header = unsafe {
                self.storage
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(self.cursor)
                    .cast::<Header>()
            };
            let table = unsafe { (*header).table };
            self.cursor = unsafe { (*header).next as usize };
            self.count -= 1;
            // Advance before invoking: the guard owns this capture even if its
            // callback unwinds. Arena::drop owns only the unexecuted suffix.
            let _capture = Capture { header, table };
            unsafe { (table.invoke)(header, context) }?;
        }
        self.used = 0;
        self.cursor = 0;
        Ok(())
    }
    pub fn clear(&mut self) {
        while self.count != 0 {
            let header = unsafe {
                self.storage
                    .as_mut_ptr()
                    .cast::<u8>()
                    .add(self.cursor)
                    .cast::<Header>()
            };
            self.cursor = unsafe { (*header).next as usize };
            self.count -= 1;
            unsafe { ((*header).table.destroy)(header) };
        }
        self.used = 0;
        self.cursor = 0;
    }
}
impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for Arena {
    fn drop(&mut self) {
        self.clear();
    }
}
struct Capture {
    header: *mut Header,
    table: &'static Vtable,
}
impl Drop for Capture {
    fn drop(&mut self) {
        unsafe { (self.table.destroy)(self.header) };
    }
}
unsafe fn destroy<F>(header: *mut Header) {
    let (_, offset) = Layout::new::<Header>().extend(Layout::new::<F>()).unwrap();
    unsafe { std::ptr::drop_in_place(header.cast::<u8>().add(offset).cast::<F>()) };
}
unsafe fn invoke<F, A, B, C>(
    header: *mut Header,
    context: &mut Context,
) -> Result<(), NativeFailure>
where
    F: Fn(&mut Context, &[A], &[B], &[C]) -> Result<(), NativeFailure>,
{
    let lengths = unsafe { (*header).lengths };
    let plan = Plan::new::<F, A, B, C>(lengths);
    let base = header.cast::<u8>();
    unsafe {
        let function = &*base.add(plan.capture).cast::<F>();
        function(
            context,
            std::slice::from_raw_parts(base.add(plan.arrays[0]).cast::<A>(), lengths[0] as usize),
            std::slice::from_raw_parts(base.add(plan.arrays[1]).cast::<B>(), lengths[1] as usize),
            std::slice::from_raw_parts(base.add(plan.arrays[2]).cast::<C>(), lengths[2] as usize),
        )
    }
}
