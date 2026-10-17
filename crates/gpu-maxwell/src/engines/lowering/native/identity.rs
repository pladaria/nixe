//! Semantic interning survives state switches without capturing mutable contents.
use super::*;
use std::hash::{Hash, Hasher};

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
                GuestBinding::Buffer(b) => b.byte_mapping_key().hash(state),
                GuestBinding::Image { image, arrayed } => {
                    (Arc::as_ptr(&image.planes) as *const GuestImagePlane).hash(state);
                    image.description.hash(state);
                    image.swizzle.hash(state);
                    image.sample_grid.hash(state);
                    image.canonical_texels.hash(state);
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

#[derive(Clone)]
pub(super) struct DrawKey(pub Arc<GuestDrawState>);
impl PartialEq for DrawKey {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&self.0, &other.0);
        a.parameters.identity == b.parameters.identity
            && a.program.identity == b.program.identity
            && a.vertices.len() == b.vertices.len()
            && a.vertices.iter().zip(&b.vertices).all(|(a, b)| {
                a.binding == b.binding
                    && a.backing.byte_mapping_key() == b.backing.byte_mapping_key()
            })
            && match (&a.index, &b.index) {
                (None, None) => true,
                (Some((a, at)), Some((b, bt))) => {
                    at == bt && a.byte_mapping_key() == b.byte_mapping_key()
                }
                _ => false,
            }
            && a.colors.len() == b.colors.len()
            && a.colors.iter().zip(&b.colors).all(|(a, b)| a.same_view(b))
            && match (&a.depth_stencil, &b.depth_stencil) {
                (None, None) => true,
                (Some(a), Some(b)) => a.same_view(b),
                _ => false,
            }
    }
}
impl Eq for DrawKey {}
impl Hash for DrawKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a = &self.0;
        a.parameters.identity.hash(state);
        a.program.identity.hash(state);
        a.vertices.len().hash(state);
        for vertex in &a.vertices {
            vertex.binding.hash(state);
            vertex.backing.byte_mapping_key().hash(state);
        }
        a.index
            .as_ref()
            .map(|(backing, kind)| (backing.byte_mapping_key(), kind))
            .hash(state);
        a.colors.len().hash(state);
        for image in a.colors.iter().chain(&a.depth_stencil) {
            (Arc::as_ptr(&image.planes) as *const GuestImagePlane).hash(state);
            image.description.hash(state);
            image.swizzle.hash(state);
            image.sample_grid.hash(state);
            image.canonical_texels.hash(state);
        }
        a.depth_stencil.is_some().hash(state);
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
        + draw.index.as_ref().map_or(0, |(b, _)| b.metadata_bytes())
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
                GuestBinding::Buffer(b) => b.metadata_bytes(),
                GuestBinding::Image { image, .. } => plane_metadata_bytes(image),
                GuestBinding::Sampler(_) => 0,
            })
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    #[test]
    fn program_identity_ignores_cpu_versions_but_distinguishes_physical_bindings() {
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let code = Arc::new(GuestShaderCode {
            identity: nixe_gpu::ssa::artifact::ContentId::of(b"program-intern-test", &[]),
            functions: Box::new([]),
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
        allocation.write(0, &[1, 2, 3, 4]).unwrap();
        let updated = key(backing.snapshot_subrange(0, 4096).unwrap());
        assert!(Arc::ptr_eq(states.get(&updated).unwrap(), &first.0));
        let replacement = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        let remapped = key(replacement
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap());
        assert!(states.get(&remapped).is_none());
    }
}
