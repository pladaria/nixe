//! Backend-neutral transfer of resident images to host presentation.

use std::fmt::{Debug, Formatter};
use std::sync::{Arc, mpsc};

use nixe_memory::AllocationRange;

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
/// The accepted physical inputs survive CPU remap or nvmap-handle retirement.
/// Cached interpretation metadata holds weak identities; the graphics owner
/// consumes these inputs before delegating native recording.
#[derive(Clone, Debug)]
pub struct PresentationImageRequest {
    /// Whether canonical bytes can be decoded as this image. Opaque storage
    /// requires a current resident producer; importing it would corrupt pixels.
    pub allow_canonical_import: bool,
    pub backing: BackingView,
    pub width: u32,
    pub height: u32,
    pub format: PresentationImageFormat,
    pub layout: ImageMemoryLayout,
    pub row_pitch: u32,
    /// Exact accepted source pins, released after the graphics owner consumes it.
    pub inputs: Vec<AllocationRange>,
}

/// An immutable presentation lease over a backend-resident image.
///
/// Host objects remain owned by the renderer; frames carry typed release leases.
#[derive(Clone)]
pub struct ResidentImage {
    backend: BackendInstanceId,
    description: ImageDescription,
    lease: PresentationLease,
}
impl ResidentImage {
    #[must_use]
    pub fn from_lease(description: ImageDescription, lease: PresentationLease) -> Self {
        Self {
            backend: lease.producer().instance(),
            description,
            lease,
        }
    }
    #[must_use]
    pub fn lease(&self) -> &PresentationLease {
        &self.lease
    }

    #[must_use]
    pub const fn backend(&self) -> BackendInstanceId {
        self.backend
    }

    #[must_use]
    pub const fn description(&self) -> ImageDescription {
        self.description
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

/// Generation-safe identity in the GPU owner's presentation image cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PresentationImageId {
    pub slot: u64,
    pub generation: u32,
}

/// Exact frame version whose last consumer has released its lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentationRelease {
    pub image: PresentationImageId,
    pub content_version: u64,
    pub producer: crate::BackendSubmissionToken,
}

/// Immutable frame identity, producer ordering and release contract.
///
/// The owner must order the presentation read before any overwrite, or retain
/// a separate GPU snapshot of this version. A cache ID alone is insufficient.
/// Last drop returns the lease to the owner; it does not destroy a Vulkan
/// object or prove GPU/WSI completion. The owner retires after actual last use.
#[derive(Clone, Debug)]
pub struct PresentationLease(Arc<PresentationLeaseInner>);

#[derive(Debug)]
struct PresentationLeaseInner {
    release: PresentationRelease,
    owner: mpsc::Sender<PresentationRelease>,
}

impl PresentationLease {
    #[must_use]
    pub fn new(
        image: PresentationImageId,
        content_version: u64,
        producer: crate::BackendSubmissionToken,
        owner: mpsc::Sender<PresentationRelease>,
    ) -> Self {
        Self(Arc::new(PresentationLeaseInner {
            release: PresentationRelease {
                image,
                content_version,
                producer,
            },
            owner,
        }))
    }
    #[must_use]
    pub fn image(&self) -> PresentationImageId {
        self.0.release.image
    }
    #[must_use]
    pub fn content_version(&self) -> u64 {
        self.0.release.content_version
    }
    #[must_use]
    pub fn producer(&self) -> crate::BackendSubmissionToken {
        self.0.release.producer
    }
}

impl Drop for PresentationLeaseInner {
    fn drop(&mut self) {
        // An owner that has shut down already drained and destroyed its objects.
        let _ = self.owner.send(self.release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BackendSubmissionToken, ImageDimension, ImageExtent, ImageFormat, ImageKind, SampleCount,
    };

    #[test]
    fn lease_release_retains_exact_generation_version_and_order_until_last_drop() {
        let (owner, releases) = mpsc::channel();
        let producer = BackendSubmissionToken::new(BackendInstanceId::new(7), 20, 3);
        let id = PresentationImageId {
            slot: 4,
            generation: 2,
        };
        let lease = PresentationLease::new(id, 15, producer, owner);
        let retained = lease.clone();
        drop(lease);
        assert!(matches!(
            releases.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        assert_ne!(
            retained.image(),
            PresentationImageId {
                slot: 4,
                generation: 3
            }
        );
        drop(retained);
        assert_eq!(
            releases.recv().unwrap(),
            PresentationRelease {
                image: id,
                content_version: 15,
                producer
            }
        );
        assert!(matches!(
            releases.try_recv(),
            Err(mpsc::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn typed_resident_frame_never_exposes_an_opaque_host_payload() {
        let (owner, _) = mpsc::channel();
        let producer = BackendSubmissionToken::new(BackendInstanceId::new(7), 20, 3);
        let lease = PresentationLease::new(
            PresentationImageId {
                slot: 4,
                generation: 2,
            },
            15,
            producer,
            owner,
        );
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(1, 1, 1).unwrap(),
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .unwrap();
        let frame = ResidentImage::from_lease(description, lease);
        assert_eq!(frame.backend(), producer.instance());
        assert_eq!(frame.lease().content_version(), 15);
    }
}
