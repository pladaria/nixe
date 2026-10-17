//! Focused allocation contract and an opt-in dispatch benchmark. The latter
//! uses only the stable streaming API so the same harness can measure a frozen
//! prior source tree without changes to its production implementation.
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
    packet: &MaxwellDecodedPushbuffer,
    mme: &mut Vec<MaxwellMethodDispatch>,
    parameters: &mut Vec<u32>,
) {
    stream_maxwell_engine_packet(
        channel,
        FrontendSubmissionId::new(3),
        &packet.packets()[0],
        None,
        mme,
        parameters,
        &mut |_| Ok::<_, std::convert::Infallible>(()),
    )
    .unwrap_or_else(|_| panic!("benchmark dispatch failed"));
}
#[test]
fn ordinary_register_admission_allocates_nothing_with_a_retained_snapshot() {
    let mut channel = three_d_channel();
    let snapshot = channel.three_d().clone();
    let packets = [packet(0x2380 / 4, 32), packet(0x2380 / 4, 64)];
    let mut mme = Vec::new();
    let mut parameters = Vec::new();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    for i in 0..1000 {
        run(&mut channel, &packets[i % 2], &mut mme, &mut parameters);
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
    let (profile, state) = channel.three_d_with_profile_mut();
    threed::materialize_state(profile, state).unwrap();
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
#[test]
#[cfg(target_os = "linux")]
#[ignore = "opt-in matched dispatch throughput, not a correctness test"]
fn register_dispatch_throughput() {
    let iterations = std::env::var("NIXE_REGISTER_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100000);
    let mut channel = three_d_channel();
    let mut mme = Vec::new();
    let mut parameters = Vec::new();
    let values: Vec<_> = (0..64)
        .map(|i| {
            if i % 2 == 0 {
                1_f32.to_bits()
            } else {
                2_f32.to_bits()
            }
        })
        .collect();
    let data = non_incrementing_packet_on_subchannel(0, 0x0a00 / 4, &values);
    let consumer = packet(0x100 / 4, 0);
    for _ in 0..100 {
        run(&mut channel, &data, &mut mme, &mut parameters);
        run(&mut channel, &consumer, &mut mme, &mut parameters);
    }
    // The kernel counter includes user and system execution of this test thread.
    let cpu_ns = || -> u64 {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let cpu_before = cpu_ns();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        run(&mut channel, &data, &mut mme, &mut parameters);
        run(&mut channel, &consumer, &mut mme, &mut parameters);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let allocations = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    let cpu_nanoseconds = cpu_ns() - cpu_before;
    println!(
        "{{\"workload\":\"coalesced_viewport_registers\",\"iterations\":{iterations},\"methods\":{},\"seconds\":{elapsed},\"allocations\":{allocations},\"cpu_nanoseconds\":{cpu_nanoseconds}}}",
        iterations * 65
    );
}
