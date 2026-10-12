//! Device-resident decoded pixels crossing a host backend boundary.
use std::sync::Arc;

/// Exported device-local buffer, borrowed only for the synchronous transfer.
/// The producer must finish its writes and signal the exported binary semaphore
/// before returning. The backend consumes that signal before copying the planes.
/// No CPU pixel mapping.
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

pub trait DeviceImageSource: std::fmt::Debug + Send + Sync {
    fn copy_nv12(&self, destination: ExternalImageBuffer) -> Result<(), Box<str>>;
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
