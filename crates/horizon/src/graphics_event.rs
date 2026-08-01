//! Horizon-owned graphics event-source vocabulary.

/// Guest graphics source associated with one runtime kernel event.
///
/// This identity remains in the Horizon ownership layer; `nixe-runtime` only
/// supplies the generic signaling primitive. Acquire and release fence sources
/// are named now so later fence integration cannot silently reuse a VSync,
/// BufferQueue, or GPU syncpoint event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphicsEventSource {
    ViVsync { display_id: u64 },
    BufferQueueAvailability { binder_id: i32 },
    GpuSyncpoint { event_slot: u8 },
    AcquireFence,
    ReleaseFence,
}
