//! Ordinary command consumption must not allocate after the entry is retained.
use super::*;
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
fn run(
    channel: &mut MaxwellGpuChannel,
    stream: &mut crate::pushbuffer::packet::CommandStream,
    mme: &mut Option<MaxwellMethodDispatch>,
    parameters: &mut Vec<u32>,
) {
    stream.rewind();
    crate::test_memory::ready(stream_maxwell_engine_entry(
        channel,
        stream,
        None,
        mme,
        parameters,
        &mut async |_| Ok::<_, std::convert::Infallible>(()),
    ))
    .unwrap_or_else(|_| panic!("benchmark dispatch failed"));
}
#[test]
fn ordinary_register_admission_allocates_nothing_with_a_retained_snapshot() {
    let mut channel = three_d_channel();
    let snapshot = channel.three_d().clone();
    let mut streams = [packet(0x2380 / 4, 32), packet(0x2380 / 4, 64)]
        .map(|p| crate::pushbuffer::packet::CommandStream::fixture(p.packets()[0].view()));
    let mut mme = None;
    let mut parameters = Vec::new();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    for i in 0..1000 {
        run(&mut channel, &mut streams[i % 2], &mut mme, &mut parameters);
    }
    let count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert_eq!(count, 0);
    assert!(
        snapshot
            .shader_bindings()
            .selector()
            .size()
            .value()
            .is_none()
    );
    let (_, state) = channel.three_d_with_profile_mut();
    assert_eq!(
        state
            .operation_state()
            .shader_bindings()
            .selector()
            .size()
            .value(),
        Some(&64)
    );
}
