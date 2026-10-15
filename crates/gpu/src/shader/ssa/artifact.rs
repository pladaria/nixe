//! Stable native compiler identities. Hashing is cold work, never a draw key.
use super::spirv::Options;
use crate::{TessellationDomain, TessellationOutput, TessellationSpacing, TessellationWinding};
use sha2::{Digest, Sha256};

/// Bump whenever semantic lowering, the binding ABI or emitted code changes.
pub const COMPILER_SCHEMA: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContentId(pub [u8; 32]);
impl ContentId {
    /// Length framing makes distinct tuples unambiguous; all callers supply
    /// explicit little-endian semantic fields, never Rust struct memory/Hash.
    pub fn of(domain: &[u8], parts: &[&[u8]]) -> Self {
        let mut h = Sha256::new();
        for part in std::iter::once(domain).chain(parts.iter().copied()) {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part);
        }
        Self(h.finalize().into())
    }
    pub fn hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Selected once by the concrete guest frontend, independently of host features.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderNamespace {
    pub guest: [u8; 16],
    pub isa: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ArtifactKey(pub ContentId);
impl ArtifactKey {
    pub fn new(namespace: ShaderNamespace, source: ContentId, specialization: ContentId) -> Self {
        Self(ContentId::of(
            b"nixe-ssa-artifact",
            &[
                &COMPILER_SCHEMA.to_le_bytes(),
                &namespace.guest,
                &namespace.isa.to_le_bytes(),
                &source.0,
                &specialization.0,
            ],
        ))
    }
}

/// Exact emission inputs, including stage-dependent values, with a stable wire
/// representation. No device UUID/driver enters shader identity unless it changes
/// code; driver pipeline blobs have their own stricter compatibility namespace.
pub fn specialization(options: &[Options]) -> ContentId {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(options.len() as u32).to_le_bytes());
    for option in options {
        let s = option.shader;
        bytes.push(s.depth_clip_negative_one_to_one as u8);
        bytes.push(s.vertex_position_transform.is_some() as u8);
        if let Some(t) = s.vertex_position_transform {
            for v in t.scale().into_iter().chain(t.offset()) {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        bytes.extend_from_slice(&s.input_control_points.to_le_bytes());
        bytes.push(s.tessellation_mode.is_some() as u8);
        if let Some(t) = s.tessellation_mode {
            bytes.push(match t.domain {
                TessellationDomain::Isolines => 0,
                TessellationDomain::Triangles => 1,
                TessellationDomain::Quads => 2,
            });
            bytes.push(match t.spacing {
                TessellationSpacing::Equal => 0,
                TessellationSpacing::FractionalOdd => 1,
                TessellationSpacing::FractionalEven => 2,
            });
            bytes.push(match t.output {
                TessellationOutput::Points => 0,
                TessellationOutput::Lines => 1,
                TessellationOutput::Triangles(TessellationWinding::CounterClockwise) => 2,
                TessellationOutput::Triangles(TessellationWinding::Clockwise) => 3,
            });
        }
        bytes.extend([
            s.float32.denorm_preserve as u8,
            s.float32.rounding_mode_rte as u8,
            s.float32.signed_zero_inf_nan_preserve as u8,
            s.float32.fused_multiply_add as u8,
            s.float64.enabled as u8,
            s.float64.rounding_mode_rte as u8,
            s.float64.signed_zero_inf_nan_preserve as u8,
            option.integer64 as u8,
            option.subgroup_quad as u8,
        ]);
        bytes.extend_from_slice(&option.shared_memory_bytes.to_le_bytes());
        bytes.extend_from_slice(&option.local_memory_bytes.to_le_bytes());
    }
    ContentId::of(b"nixe-spirv-specialization", &[&bytes])
}
