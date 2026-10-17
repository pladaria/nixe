//! Stable native compiler identities. Hashing is cold work, never a draw key.
use super::spirv::Options;
use crate::{TessellationDomain, TessellationOutput, TessellationSpacing, TessellationWinding};
use sha2::{Digest, Sha256};

/// Bump whenever semantic lowering, the binding ABI or emitted code changes.
pub const COMPILER_SCHEMA: u32 = 17;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContentId(pub [u8; 32]);
impl ContentId {
    /// Length framing makes distinct tuples unambiguous; all callers supply
    /// explicit encodings, never raw Rust struct memory or pointer addresses.
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
            s.native_half_conversions as u8,
            s.native_half_fma as u8,
            s.float32.denorm_preserve as u8,
            s.float32.denorm_flush_to_zero as u8,
            s.float32.rounding_mode_rtz as u8,
            s.float32.independent_denormals as u8,
            s.float32.independent_rounding as u8,
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
        bytes.extend_from_slice(&option.uniform_buffer_bytes.to_le_bytes());
    }
    ContentId::of(b"nixe-spirv-specialization", &[&bytes])
}

/// Identity before SPIR-V emission, including linked IO in the semantic function
/// and only this stage's effective descriptor locations (not unused bindings).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StageKey(pub ContentId);
impl StageKey {
    pub fn new(
        namespace: ShaderNamespace,
        function: &super::Function,
        options: Options,
        bindings: &std::collections::BTreeMap<super::ResourceId, u32>,
    ) -> Self {
        use std::hash::{Hash, Hasher};
        // Structural encoding is compiler-schema scoped. Fixed-width numeric
        // writes avoid architecture-sized lengths or native-endian identities.
        struct Writer(Sha256);
        impl Hasher for Writer {
            fn finish(&self) -> u64 {
                u64::from_le_bytes(self.0.clone().finalize()[..8].try_into().unwrap())
            }
            fn write(&mut self, bytes: &[u8]) {
                self.0.update((bytes.len() as u64).to_le_bytes());
                self.0.update(bytes);
            }
            fn write_u8(&mut self, v: u8) {
                self.write(&v.to_le_bytes());
            }
            fn write_u16(&mut self, v: u16) {
                self.write(&v.to_le_bytes());
            }
            fn write_u32(&mut self, v: u32) {
                self.write(&v.to_le_bytes());
            }
            fn write_u64(&mut self, v: u64) {
                self.write(&v.to_le_bytes());
            }
            fn write_u128(&mut self, v: u128) {
                self.write(&v.to_le_bytes());
            }
            fn write_usize(&mut self, v: usize) {
                self.write_u64(v as u64);
            }
            fn write_i8(&mut self, v: i8) {
                self.write(&v.to_le_bytes());
            }
            fn write_i16(&mut self, v: i16) {
                self.write(&v.to_le_bytes());
            }
            fn write_i32(&mut self, v: i32) {
                self.write(&v.to_le_bytes());
            }
            fn write_i64(&mut self, v: i64) {
                self.write(&v.to_le_bytes());
            }
            fn write_i128(&mut self, v: i128) {
                self.write(&v.to_le_bytes());
            }
            fn write_isize(&mut self, v: isize) {
                self.write_i64(v as i64);
            }
        }
        let mut writer = Writer(Sha256::new());
        COMPILER_SCHEMA.hash(&mut writer);
        namespace.hash(&mut writer);
        function.hash(&mut writer);
        specialization(&[options]).hash(&mut writer);
        for resource in super::spirv::pipeline_bindings(&[function]).keys() {
            resource.hash(&mut writer);
            bindings[resource].hash(&mut writer);
        }
        Self(ContentId::of(
            b"nixe-semantic-stage-v1",
            &[&writer.0.finalize()],
        ))
    }
}
