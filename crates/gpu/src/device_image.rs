//! Device-resident decoded pixels crossing a host backend boundary.
use std::any::Any;
use std::sync::Arc;

/// Exported device-local buffer. Import duplicates these borrowed FDs; the
/// producer retains its imports until the consumer has completed. Enqueued GPU
/// writes signal the binary semaphore before Vulkan copies the visible planes.
#[derive(Clone, Copy, Debug)]
pub struct ExternalImageBuffer {
    pub fd: i32,
    pub semaphore_fd: i32,
    pub allocation_size: u64,
    pub device_uuid: [u8; 16],
    pub row_pitch: u64,
    pub chroma_offset: u64,
    /// Initialize all allocation bytes before the backend imports the buffer.
    pub initialize: bool,
}

/// Producer-owned import retained alongside the renderer's exported allocation.
/// Enqueued writes are ordered by the exported semaphore. Wait is only needed
/// when abandoning work.
pub trait DeviceImageTransfer: std::fmt::Debug + Send + Any {
    fn wait(&mut self) -> Result<(), Box<str>>;
}

pub trait DeviceImageSource: std::fmt::Debug + Send + Sync {
    /// Stable while a frame or its imported transfer retains the producer device.
    fn transfer_context(&self) -> usize {
        0
    }
    fn import_nv12(
        &self,
        _destination: ExternalImageBuffer,
    ) -> Result<Box<dyn DeviceImageTransfer>, Box<str>> {
        Err("this decoded image source cannot retain external imports".into())
    }
    fn enqueue_nv12(&self, _transfer: &mut dyn DeviceImageTransfer) -> Result<(), Box<str>> {
        Err("this decoded image source cannot enqueue external transfers".into())
    }
}

#[derive(Clone, Debug)]
pub struct DeviceImageFrame {
    pub width: u32,
    pub height: u32,
    pub source: Arc<dyn DeviceImageSource>,
}
impl PartialEq for DeviceImageFrame {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && Arc::ptr_eq(&self.source, &other.source)
    }
}
