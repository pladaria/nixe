//! Semantic interning survives state switches without capturing mutable contents.
use super::*;
use std::hash::{Hash, Hasher};

/// Plane metadata depends on interpretation and physical bindings, never on a
/// resource-list ID or the current bytes. Hashing visits subresources, not the
/// image cache or the pages retained by each canonical mapping key.
#[derive(Clone)]
pub(super) struct ImageKey {
    pub encoding: super::super::super::threed::MaxwellThreeDGuestImageFormat,
    pub description: nixe_gpu::ImageDescription,
    pub canonical_texels: bool,
    pub view: nixe_gpu::ImageView,
}
impl PartialEq for ImageKey {
    fn eq(&self, other: &Self) -> bool {
        self.encoding == other.encoding
            && self.description == other.description
            && self.canonical_texels == other.canonical_texels
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
        self.canonical_texels.hash(state);
        self.view.binding_fingerprint().hash(state);
    }
}

#[derive(Clone)]
pub(super) struct ParameterKey(pub Arc<GuestDrawParameters>);
impl PartialEq for ParameterKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_state(&other.0)
    }
}
impl Eq for ParameterKey {}
impl Hash for ParameterKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash_state(state);
    }
}
#[derive(Clone)]
pub(super) struct ProgramKey(pub Arc<GuestProgram>);
impl PartialEq for ProgramKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.shader.identity == other.0.shader.identity
            && self.0.shared_memory_bytes == other.0.shared_memory_bytes
            && self.0.bindings.len() == other.0.bindings.len()
            && self
                .0
                .bindings
                .iter()
                .zip(&other.0.bindings)
                .all(|((a, av), (b, bv))| a == b && same_binding(av, bv))
    }
}
impl Eq for ProgramKey {}
impl Hash for ProgramKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.shader.identity.hash(state);
        self.0.shared_memory_bytes.hash(state);
        self.0.bindings.len().hash(state);
        for (id, binding) in &self.0.bindings {
            id.hash(state);
            std::mem::discriminant(binding).hash(state);
            match binding {
                GuestBinding::NullBuffer => (),
                GuestBinding::Buffer(b) => b.hash(state),
                GuestBinding::Image { image, arrayed } => {
                    image.view_identity.is_some().hash(state);
                    if let Some(identity) = image.view_identity {
                        identity.hash(state);
                    } else {
                        image.planes.hash(state);
                        image.description.hash(state);
                        image.swizzle.hash(state);
                        image.sample_grid.hash(state);
                        image.canonical_texels.hash(state);
                    }
                    arrayed.hash(state);
                }
                GuestBinding::Sampler(s) => {
                    s.compare.hash(state);
                    s.min_filter.hash(state);
                    s.mag_filter.hash(state);
                    s.mip_filter.hash(state);
                    s.address_modes.hash(state);
                    s.border_color.hash(state);
                    // Sampler descriptions compare finite floats numerically.
                    for f in [s.lod_bias, s.lod_min, s.lod_max, s.max_anisotropy] {
                        (if f == 0. { 0 } else { f.to_bits() }).hash(state);
                    }
                }
            }
        }
    }
}

pub(super) fn draw_bytes(draw: &GuestDrawState) -> usize {
    std::mem::size_of_val(draw)
        + std::mem::size_of_val(draw.vertices.as_ref())
        + draw
            .vertices
            .iter()
            .map(|v| v.backing.metadata_bytes())
            .sum::<usize>()
        + draw.index.as_ref().map_or(0, |b| match b {
            GuestIndexBuffer::Canonical(b, _) => b.metadata_bytes(),
            GuestIndexBuffer::Inline(bytes) => bytes.len(),
        })
        + draw
            .colors
            .iter()
            .chain(&draw.depth_stencil)
            .map(plane_metadata_bytes)
            .sum::<usize>()
}
pub(super) fn parameter_bytes(p: &GuestDrawParameters) -> usize {
    std::mem::size_of_val(p)
        + std::mem::size_of_val(p.vertices.as_ref())
        + p.vertices
            .iter()
            .map(|v| std::mem::size_of_val(v.attributes.as_ref()))
            .sum::<usize>()
        + std::mem::size_of_val(p.color_formats.as_ref())
        + std::mem::size_of_val(p.color_outputs.as_ref())
}
pub(super) fn program_bytes(p: &GuestProgram) -> usize {
    std::mem::size_of_val(p)
        + std::mem::size_of_val(p.bindings.as_ref())
        + p.bindings
            .iter()
            .map(|(_, b)| match b {
                GuestBinding::NullBuffer => 0,
                GuestBinding::Buffer(b) => b.metadata_bytes(),
                GuestBinding::Image { image, .. } => plane_metadata_bytes(image),
                GuestBinding::Sampler(_) => 0,
            })
            .sum::<usize>()
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
            crate::test_memory::mapping(allocation),
        )
        .unwrap();
        ImageKey {
            encoding: super::super::super::super::threed::MaxwellThreeDGuestImageFormat::Canonical(
                description.format(),
            ),
            description,
            canonical_texels: true,
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
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let first = image_key(&allocation, 16);
        let mut states = nixe_gpu::WorkingSet::new(4, 4096);
        states.insert(first.clone(), 7, 512);
        crate::test_memory::write(&allocation, 0, &[1, 2, 3, 4]).unwrap();
        let mut updated = image_key(&allocation, 16);
        updated.view = updated.view.with_image(ImageId::new(19));
        assert_eq!(states.get(&updated), Some(&7));
        let unchanged = updated.clone();
        crate::engines::tests::register_benchmark::ALLOCATIONS.with(|n| n.set(Some(0)));
        for _ in 0..1000 {
            assert_eq!(states.get(&unchanged), Some(&7));
        }
        let allocations = crate::engines::tests::register_benchmark::ALLOCATIONS
            .with(|n| n.replace(None).unwrap());
        assert_eq!(allocations, 0);
        drop(first);
        assert_eq!(states.get(&updated), Some(&7));
        // Identical storage/pitch with a different extent must not reuse planes.
        assert_eq!(states.get(&image_key(&allocation, 8)), None);
        let replacement = crate::test_memory::allocate(&machine, 4096);
        assert_eq!(states.get(&image_key(&replacement, 16)), None);
        updated.canonical_texels = false;
        assert_eq!(states.get(&updated), None);
    }

    #[test]
    fn image_metadata_eviction_keeps_the_view_without_retaining_guest_ram() {
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let retained = image_key(&allocation, 16);
        let mut states = nixe_gpu::WorkingSet::new(1, 512);
        states.insert(retained.clone(), 7, 512);
        states.insert(image_key(&allocation, 8), 8, 512);
        assert_eq!(states.usage(), (1, 512));
        assert_eq!(states.get(&retained), None);
        crate::test_memory::write(&allocation, 0, &[5]).unwrap();
        drop(allocation);
        assert!(
            retained.view.bindings()[0]
                .backing()
                .mapping()
                .pin(0, 1)
                .is_err()
        );
    }

    #[test]
    fn program_identity_ignores_cpu_versions_but_distinguishes_physical_bindings() {
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let backing = crate::test_memory::mapping(&allocation);
        let code = Arc::new(GuestShaderCode {
            identity: nixe_gpu::shader_identity::ContentId::of(b"program-intern-test", &[]),
            stages: Box::new([]),
        });
        let key = |backing| {
            ProgramKey(Arc::new(GuestProgram {
                identity: nixe_gpu::PreparedStateId::new(1, 1),
                shader: Arc::clone(&code),
                bindings: Box::new([(ResourceId::new(1), GuestBinding::Buffer(backing))]),
                shared_memory_bytes: 0,
            }))
        };
        let first = key(backing.clone());
        let mut states = nixe_gpu::WorkingSet::new(4, 1 << 20);
        states.insert(first.clone(), Arc::clone(&first.0), program_bytes(&first.0));
        crate::test_memory::write(&allocation, 0, &[1, 2, 3, 4]).unwrap();
        let updated = key(backing.clone());
        assert!(Arc::ptr_eq(states.get(&updated).unwrap(), &first.0));
        let replacement = crate::test_memory::allocate(&machine, 4096);
        let remapped = key(crate::test_memory::mapping(&replacement));
        assert!(states.get(&remapped).is_none());
    }
    #[test]
    fn reconstructed_image_planes_preserve_program_cache_identity() {
        let machine = crate::test_memory::machine();
        let allocation = crate::test_memory::allocate(&machine, 4096);
        let key = image_key(&allocation, 16);
        let a = super::super::image::conversion(&nixe_gpu::ConversionImage {
            description: key.description,
            view: key.view,
        })
        .unwrap();
        let mut b = a.clone();
        b.planes = Arc::from(a.planes.to_vec());
        assert!(a.same_view(&b));
        assert!(!Arc::ptr_eq(&a.planes, &b.planes));
        let code = Arc::new(GuestShaderCode {
            identity: nixe_gpu::shader_identity::ContentId::of(b"audit", &[]),
            stages: Box::new([]),
        });
        let key = |image| {
            ProgramKey(Arc::new(GuestProgram {
                identity: nixe_gpu::PreparedStateId::new(1, 1),
                shader: Arc::clone(&code),
                shared_memory_bytes: 0,
                bindings: Box::new([(
                    ResourceId::new(1),
                    GuestBinding::Image {
                        image,
                        arrayed: false,
                    },
                )]),
            }))
        };
        let a = key(a);
        let b = key(b);
        assert!(a == b);
        let hash = |key: &ProgramKey| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        let mut cache = nixe_gpu::WorkingSet::new(4, 4096);
        cache.insert(a.clone(), 7, 512);
        assert_eq!(hash(&a), hash(&b));
        assert_eq!(cache.get(&b), Some(&7));
    }
}
