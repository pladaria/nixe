use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! { pub(crate) static ALLOCATIONS: Cell<Option<u64>> = const {Cell::new(None)}; }
struct CountAllocations;
#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| {
            if let Some(value) = n.get() {
                n.set(Some(value + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| {
            if let Some(value) = n.get() {
                n.set(Some(value + 1));
            }
        });
        unsafe { System.realloc(p, layout, size) }
    }
}
