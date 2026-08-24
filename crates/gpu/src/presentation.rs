//! Backend-neutral transfer of resident images to host presentation.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use nixe_memory::CanonicalCpuWriteDependency;

use crate::{BackendInstanceId, BackingView, ImageDescription, ImageMemoryLayout};

/// Channel order declared by a host presentation buffer.
///
/// Transfer-function semantics remain owned by the producer image description;
/// Android's framebuffer descriptor does not distinguish UNORM from sRGB.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PresentationImageFormat {
    Rgba8,
    Rgbx8,
    Bgra8,
    Rgb565,
    Rgba4444,
}

/// Complete retained source for one GPU presentation operation.
///
/// The backing preserves canonical byte identity after the guest mapping or
/// nvmap handle disappears. Backends may export a compatible device-authored
/// image directly or import this source into a reusable resident image; they
/// must not materialize a software-rendered host frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresentationImageRequest {
    pub backing: BackingView,
    pub width: u32,
    pub height: u32,
    pub format: PresentationImageFormat,
    pub layout: ImageMemoryLayout,
    pub row_pitch: u32,
    pub cpu_writes: CanonicalCpuWriteDependency,
}

/// An immutable presentation lease over a backend-resident image.
///
/// Concrete payloads retain any host completion primitive required by their
/// presentation API. For WGPU, export and presentation use the same device and
/// queue, so queue ordering is the completion dependency and no neutral token
/// needs to duplicate backend synchronization state.
#[derive(Clone)]
pub struct ResidentImage {
    backend: BackendInstanceId,
    description: ImageDescription,
    payload: Arc<dyn Any + Send + Sync>,
}

impl ResidentImage {
    #[must_use]
    pub fn new(
        backend: BackendInstanceId,
        description: ImageDescription,
        payload: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            backend,
            description,
            payload,
        }
    }

    #[must_use]
    pub const fn backend(&self) -> BackendInstanceId {
        self.backend
    }

    #[must_use]
    pub const fn description(&self) -> ImageDescription {
        self.description
    }

    #[must_use]
    pub fn payload<T: Any>(&self) -> Option<&T> {
        self.payload.downcast_ref()
    }
}

impl Debug for ResidentImage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentImage")
            .field("backend", &self.backend)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}
