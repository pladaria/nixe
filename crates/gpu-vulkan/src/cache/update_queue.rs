//! GPU-owner descriptor payload, consumed directly by native recording.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_update_descriptor.h
use super::DescriptorEntry;
use crate::{VulkanError, dispatch::Dispatch};
use std::{cell::UnsafeCell, ptr::NonNull};

const FRAMES_IN_FLIGHT: usize = 8;

pub(crate) struct UpdateDescriptorQueue {
    // Only the GPU owner writes. Native recording reads earlier reservations;
    // UnsafeCell permits those disjoint accesses without a mutable slice alias.
    payload: Box<[UnsafeCell<DescriptorEntry>]>,
    frame_payload_size: usize,
    frame: usize,
    cursor: usize,
    upload_start: usize,
    entries: usize,
}

/// A native recording parameter, not an owner of guest memory or a copied list.
/// Its queue outlives recording; frame rotation and overflow wait for the worker.
#[derive(Clone, Copy)]
pub(crate) struct DescriptorUpdate {
    data: NonNull<DescriptorEntry>,
    entries: usize,
}
// SAFETY: native jobs only read initialized reservations. The GPU owner does
// not overwrite them until WaitWorker, and residency drains recording on drop.
unsafe impl Send for DescriptorUpdate {}

impl DescriptorUpdate {
    pub(crate) fn is_empty(self) -> bool {
        self.entries == 0
    }
    /// The payload owner must remain live and this reservation must stay
    /// immutable until the native update command has consumed the slice.
    pub(crate) unsafe fn as_slice(&self) -> &[DescriptorEntry] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.entries) }
    }
}

impl UpdateDescriptorQueue {
    pub(crate) const GUEST_FRAME_PAYLOAD_SIZE: usize = 0x80000;
    pub(crate) const COMPUTE_FRAME_PAYLOAD_SIZE: usize = 0x20000;

    pub(crate) fn new(frame_payload_size: usize) -> Self {
        Self {
            payload: std::iter::repeat_with(|| {
                UnsafeCell::new(DescriptorEntry {
                    buffer: ash::vk::DescriptorBufferInfo::default(),
                })
            })
            .take(frame_payload_size * FRAMES_IN_FLIGHT)
            .collect(),
            frame_payload_size,
            frame: 0,
            cursor: 0,
            upload_start: 0,
            entries: 0,
        }
    }
}
impl UpdateDescriptorQueue {
    pub(crate) fn acquire(
        &mut self,
        dispatch: &mut Dispatch,
        entries: usize,
    ) -> Result<(), VulkanError> {
        // Acquire uses the reference's default reservation for empty layouts;
        // only actual entries advance the payload cursor.
        let reserve = if entries == 0 { 0x400 } else { entries };
        if reserve >= self.frame_payload_size {
            return Err(VulkanError::InvalidResource(
                "descriptor reservation exceeds the frame payload capacity",
            ));
        }
        let start = self.frame * self.frame_payload_size;
        if self.cursor - start + reserve >= self.frame_payload_size {
            log::warn!(
                "Descriptor payload overflow (used={}, reserve={}, capacity={})",
                self.cursor - start,
                reserve,
                self.frame_payload_size,
            );
            // Acquire reuses the frame reservation only after accepted native
            // updates have read it. This waits for recording, not GPU execution.
            dispatch.wait_worker()?;
            self.cursor = start;
        }
        self.upload_start = self.cursor;
        self.entries = entries;
        self.cursor += entries;
        Ok(())
    }
    pub(crate) fn write(&mut self, index: usize, entry: DescriptorEntry) {
        assert!(index < self.entries, "descriptor index exceeds reservation");
        // SAFETY: the reservation is in bounds and disjoint from all earlier
        // native captures; its complete union entry is initialized at once.
        unsafe { self.payload[self.upload_start + index].get().write(entry) };
    }
    pub(crate) fn update_data(&self) -> DescriptorUpdate {
        DescriptorUpdate {
            data: NonNull::new(self.payload[self.upload_start].get()).unwrap(),
            entries: self.entries,
        }
    }
    /// Called after the frame producer's WaitWorker, as synchronous
    /// PresentManager::Present followed by RasterizerVulkan::TickFrame.
    pub(crate) fn tick_frame(&mut self) {
        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        self.cursor = self.frame * self.frame_payload_size;
        self.upload_start = self.cursor;
        self.entries = 0;
    }
}
