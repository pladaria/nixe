//! Bounded opt-in host timelines. Default builds erase all recording work.
#[cfg(feature = "capture")]
mod capture;
#[cfg(feature = "capture")]
pub use capture::{Capture, CpuSpan, Span, clock_ns, device_interval, enabled, epoch, event};
#[must_use]
pub const fn capture_compiled() -> bool {
    cfg!(feature = "capture")
}

#[cfg(not(feature = "capture"))]
#[inline(always)]
pub const fn enabled() -> bool {
    false
}
#[cfg(not(feature = "capture"))]
#[inline(always)]
pub fn event(_name: &'static str, _id: u64, _value: u64) {}
#[cfg(not(feature = "capture"))]
pub struct Span;
#[cfg(not(feature = "capture"))]
impl Span {
    #[inline(always)]
    pub fn new(_name: &'static str, _id: u64, _value: u64) -> Self {
        Self
    }
}

#[cfg(not(feature = "capture"))]
pub const fn epoch() -> u64 {
    0
}
#[cfg(not(feature = "capture"))]
pub const fn clock_ns() -> u64 {
    0
}
#[cfg(not(feature = "capture"))]
pub fn device_interval(_epoch: u64, _id: u64, _segment: u64, _start: Option<u64>, _duration: u64) {}

// Preserve the RAII scope API in erased builds too. This empty destructor is
// optimized away, including when a caller explicitly ends a timing interval.
#[cfg(not(feature = "capture"))]
impl Drop for Span {
    #[inline(always)]
    fn drop(&mut self) {}
}

/// Thread CPU timing is opt-in too; no clocks or branches survive default builds.
#[cfg(not(feature = "capture"))]
pub struct CpuSpan;
#[cfg(not(feature = "capture"))]
impl CpuSpan {
    #[inline(always)]
    pub fn new(_name: &'static str, _id: u64) -> Self {
        Self
    }
}
#[cfg(not(feature = "capture"))]
impl Drop for CpuSpan {
    #[inline(always)]
    fn drop(&mut self) {}
}
