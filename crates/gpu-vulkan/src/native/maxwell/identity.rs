//! Image interpretation identities exclude mutable texel contents.
use super::model;
use std::hash::{Hash, Hasher};

/// Plane metadata depends on interpretation and physical bindings, never on a
/// resource-list ID or the current bytes. Hashing visits subresources, not the
/// image cache or the pages retained by each canonical mapping key.
#[derive(Clone)]
pub(in crate::native) struct ImageKey {
    pub encoding: model::engines::threed::MaxwellThreeDGuestImageFormat,
    pub description: nixe_gpu::ImageDescription,

    pub view: nixe_gpu::ImageView,
}
impl PartialEq for ImageKey {
    fn eq(&self, other: &Self) -> bool {
        self.encoding == other.encoding
            && self.description == other.description
            && self.view.swizzle() == other.view.swizzle()
            && (std::ptr::eq(self.view.bindings(), other.view.bindings())
                || (self.view.bindings().len() == other.view.bindings().len()
                    && self
                        .view
                        .bindings()
                        .iter()
                        .zip(other.view.bindings())
                        .all(|(a, b)| {
                            a.subresources() == b.subresources()
                                && a.layout() == b.layout()
                                && a.backing().mapping() == b.backing().mapping()
                        })))
    }
}
impl Eq for ImageKey {}
impl Hash for ImageKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.encoding.hash(state);
        self.description.hash(state);
        self.view.binding_fingerprint().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{
        BackingView, GpuAllocationDescription, GpuAllocationId, ImageDescription, ImageDimension,
        ImageExtent, ImageFormat, ImageId, ImageKind, ImageMemoryLayout, ImageSubresourceRange,
        ImageView, SampleCount, Swizzle,
    };
    use nixe_memory::Allocation;

    fn image_key(allocation: &Allocation, width: u32) -> ImageKey {
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(width, 8, 1).unwrap(),
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .unwrap();
        let backing = BackingView::new(
            GpuAllocationId::new(1),
            GpuAllocationDescription::new(4096, 1).unwrap(),
            0,
            model::test_memory::mapping(allocation),
        )
        .unwrap();
        ImageKey {
            encoding: model::engines::threed::MaxwellThreeDGuestImageFormat::Canonical(
                description.format(),
            ),
            description,

            view: ImageView::new(
                ImageId::new(1),
                description,
                Swizzle::IDENTITY,
                vec![(
                    ImageSubresourceRange {
                        plane: 0,
                        mip_level: 0,
                        base_layer: 0,
                        layer_count: 1,
                    },
                    ImageMemoryLayout::PitchLinear {
                        row_pitch: 64,
                        layer_stride: 512,
                    },
                    backing,
                )],
            )
            .unwrap(),
        }
    }

    #[test]
    fn image_metadata_survives_content_updates_and_resource_reidentification() {
        let machine = model::test_memory::machine();
        let allocation = model::test_memory::allocate(&machine, 4096);
        let first = image_key(&allocation, 16);
        let mut states = nixe_gpu::WorkingSet::new(4, 4096);
        states.insert(first.clone(), 7, 512);
        model::test_memory::write(&allocation, 0, &[1, 2, 3, 4]).unwrap();
        let mut updated = image_key(&allocation, 16);
        updated.view = updated.view.with_image(ImageId::new(19));
        assert_eq!(states.get(&updated), Some(&7));
        let unchanged = updated.clone();
        super::super::allocations::ALLOCATIONS.with(|n| n.set(Some(0)));
        for _ in 0..1000 {
            assert_eq!(states.get(&unchanged), Some(&7));
        }
        let allocations = super::super::allocations::ALLOCATIONS.with(|n| n.replace(None).unwrap());
        assert_eq!(allocations, 0);
        drop(first);
        assert_eq!(states.get(&updated), Some(&7));
        // Identical storage/pitch with a different extent must not reuse planes.
        assert_eq!(states.get(&image_key(&allocation, 8)), None);
        let replacement = model::test_memory::allocate(&machine, 4096);
        assert_eq!(states.get(&image_key(&replacement, 16)), None);
    }

    #[test]
    fn image_metadata_eviction_keeps_the_view_without_retaining_guest_ram() {
        let machine = model::test_memory::machine();
        let allocation = model::test_memory::allocate(&machine, 4096);
        let retained = image_key(&allocation, 16);
        let mut states = nixe_gpu::WorkingSet::new(1, 512);
        states.insert(retained.clone(), 7, 512);
        states.insert(image_key(&allocation, 8), 8, 512);
        assert_eq!(states.usage(), (1, 512));
        assert_eq!(states.get(&retained), None);
        model::test_memory::write(&allocation, 0, &[5]).unwrap();
        drop(allocation);
        assert!(
            retained.view.bindings()[0]
                .backing()
                .mapping()
                .pin(0, 1)
                .is_err()
        );
    }
}
