//! Opt-in bounded execution events. Compiled away in ordinary builds.
//! Detailed spans and frame events are separate from aggregate counters so
//! production pacing can be measured without instrumenting every memory access.

#[cfg(feature = "execution-trace")]
mod capture;
#[cfg(feature = "execution-trace")]
pub use capture::{Span, detail_event, event, finish, span, start};

#[cfg(not(feature = "execution-trace"))]
pub struct Span;
#[cfg(not(feature = "execution-trace"))]
impl Drop for Span {
    #[inline(always)]
    fn drop(&mut self) {}
}
#[cfg(not(feature = "execution-trace"))]
#[inline(always)]
pub fn event(_: &'static str, _: u64, _: u64, _: u64) {}
#[cfg(not(feature = "execution-trace"))]
#[inline(always)]
pub fn detail_event(_: &'static str, _: u64, _: u64, _: u64) {}
#[cfg(not(feature = "execution-trace"))]
#[inline(always)]
pub fn span(_: &'static str, _: u64, _: u64, _: u64) -> Span {
    Span
}
