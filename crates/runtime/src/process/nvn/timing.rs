//! Optional AArch64 wrappers for timing resolved HLE calls and their host bridges.
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

pub(super) const ENTRY: usize = 0x180;
pub(super) const CALIBRATE: usize = 0x280;
pub(super) const CALIBRATED: usize = CALIBRATE + 23 * 4;
const SAMPLES: u128 = 4096;

// HLE entry points consume register arguments only. The wrapper keeps their
// original return address and the start tick on the calling guest thread's stack,
// so a wait or migration during the host call does not change the measurement.
pub(super) const CALL: &[u32] = &[
    0xf9400610, // ldr x16, [x16, #8]      per-name count and accumulated ticks
    0xa9be7bf0, // timed_call: stp x16, x30, [sp, #-32]!
    0xd5033fdf, // isb
    0xd53be02b, // mrs x11, cntpct_el0
    0xf9000beb, // str x11, [sp, #16]
    0xd63f0120, // blr x9
    0xd5033fdf, // isb
    0xd53be02b, // mrs x11, cntpct_el0
    0xf9400bec, // ldr x12, [sp, #16]
    0xa8c27bf0, // ldp x16, x30, [sp], #32
    0xcb0c016b, // sub x11, x11, x12
    0x91002210, // add x16, x16, #8
    0xc85f7e0c, // add_ticks: ldxr x12, [x16]
    0x8b0b018c, // add x12, x12, x11
    0xc80d7e0c, // stxr w13, x12, [x16]
    0x35ffffad, // cbnz w13, add_ticks
    0xd65f03c0, // ret
];

// Run the same timed call on an empty target. The first 64 iterations warm the
// native code. The resolver's two arguments and its return address survive.
pub(super) const BOOTSTRAP: &[u32] = &[
    0xa9bb53f3,                         // stp x19, x20, [sp, #-80]!
    0xa9017bf5,                         // stp x21, x30, [sp, #16]
    0xa90207e0,                         // stp x0, x1, [sp, #32]
    0xa9037fff,                         // stp xzr, xzr, [sp, #48]
    0x100002b3,                         // adr x19, empty
    0x9100c3f4,                         // add x20, sp, #48
    0x52800815,                         // mov w21, #64
    0xaa1303e9,                         // warm: mov x9, x19
    0xaa1403f0,                         // mov x16, x20
    0x97ffffb8,                         // bl timed_call
    0x710006b5,                         // subs w21, w21, #1
    0x54ffff81,                         // b.ne warm
    0xf9001fff,                         // str xzr, [sp, #56]
    0x52800015 | (SAMPLES as u32) << 5, // mov w21, #4096
    0xaa1303e9,                         // sample: mov x9, x19
    0xaa1403f0,                         // mov x16, x20
    0x97ffffb1,                         // bl timed_call
    0x710006b5,                         // subs w21, w21, #1
    0x54ffff81,                         // b.ne sample
    0xf9401fe2,                         // ldr x2, [sp, #56]       calibration ticks to report
    0xa94207e0,                         // ldp x0, x1, [sp, #32]
    0xa9417bf5,                         // ldp x21, x30, [sp, #16]
    0xa8c553f3,                         // ldp x19, x20, [sp], #80
    0xd4000001 | super::NVN_SVC << 5,
    0xd65f03c0, // ret
    0xd65f03c0, // empty: ret
];

#[derive(Default)]
pub(super) struct Calibration {
    started: AtomicBool,
    overhead_ns: OnceLock<u128>,
}

impl Calibration {
    pub(super) fn begin(&self) -> bool {
        !self.started.swap(true, Ordering::Relaxed)
    }

    pub(super) fn finish(&self, ticks: u64, frequency: u32) {
        let ns = (nanoseconds(ticks, frequency) + SAMPLES / 2) / SAMPLES;
        if self.overhead_ns.set(ns).is_ok() {
            log::info!("NVN HLE timing calibration: {ns} ns per call ({SAMPLES} samples)");
        }
    }

    pub(super) fn totals(&self, ticks: u64, invocations: u64, frequency: u32) -> (u128, u128) {
        let raw = nanoseconds(ticks, frequency);
        let overhead = self.overhead_ns.get().copied().unwrap_or(0);
        (raw, raw.saturating_sub(overhead * u128::from(invocations)))
    }
}

fn nanoseconds(ticks: u64, frequency: u32) -> u128 {
    u128::from(ticks) * 1_000_000_000 / u128::from(frequency)
}
