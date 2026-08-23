//! Backend-neutral transfer of resident images to host presentation.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use crate::{
    BackendInstanceId, BackendResourceHandle, BackendSubmissionToken, GpuAllocationId,
    ImageDescription,
};

/// Channel order declared by a host presentation buffer.
///
/// Transfer-function semantics remain owned by the producer image description;
/// Android's framebuffer descriptor does not distinguish UNORM from sRGB.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PresentationImageFormat {
    Rgba8,
    Bgra8,
}

/// Stable lookup key for the base level of a canonically backed image.
///
/// The allocation owner assigns the identity independently from guest handles
/// and mappings. A backend can therefore index presentation candidates without
/// walking canonical pages or host resources for every frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PresentationImageRequest {
    pub allocation: GpuAllocationId,
    pub allocation_offset: u64,
    pub width: u32,
    pub height: u32,
    pub format: PresentationImageFormat,
}

/// An immutable presentation lease over a backend-resident image.
///
/// `completion` names the sole backend timeline dependency which produced the
/// retained image. Concrete presenters may encode that dependency directly;
/// they must not infer readiness from guest fences or force CPU visibility.
#[derive(Clone)]
pub struct ResidentImage {
    backend: BackendInstanceId,
    resource: BackendResourceHandle,
    description: ImageDescription,
    completion: BackendSubmissionToken,
    payload: Arc<dyn Any + Send + Sync>,
}

impl ResidentImage {
    #[must_use]
    pub fn new(
        backend: BackendInstanceId,
        resource: BackendResourceHandle,
        description: ImageDescription,
        completion: BackendSubmissionToken,
        payload: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            backend,
            resource,
            description,
            completion,
            payload,
        }
    }

    #[must_use]
    pub const fn backend(&self) -> BackendInstanceId {
        self.backend
    }

    #[must_use]
    pub const fn resource(&self) -> BackendResourceHandle {
        self.resource
    }

    #[must_use]
    pub const fn description(&self) -> ImageDescription {
        self.description
    }

    #[must_use]
    pub const fn completion(&self) -> BackendSubmissionToken {
        self.completion
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
            .field("resource", &self.resource)
            .field("description", &self.description)
            .field("completion", &self.completion)
            .finish_non_exhaustive()
    }
}
