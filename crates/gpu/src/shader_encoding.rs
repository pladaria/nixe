//! Versioned guest environment records. No compiled artifacts or cache identities.
use nixe_shader_recompiler::{
    artifact::ARTIFACT_ABI_VERSION, environment::*, maxwell::header::ProgramHeader, types::*,
};
use std::collections::BTreeMap;
const SCHEMA: u32 = 4;
trait Encode {
    fn encode(&self, out: &mut Vec<u8>);
}
trait Decode: Sized {
    fn decode(input: &mut Reader<'_>) -> Option<Self>;
}
struct Reader<'a> {
    remaining: &'a [u8],
}
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, tail) = self.remaining.split_at_checked(N)?;
        self.remaining = tail;
        head.try_into().ok()
    }
    fn count(&mut self) -> Option<usize> {
        let count = u32::decode(self)? as usize;
        (count <= self.remaining.len()).then_some(count)
    }
}
macro_rules! integer {
    ($($ty:ty),*) => {$(
        impl Encode for $ty { fn encode(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); } }
        impl Decode for $ty { fn decode(input: &mut Reader<'_>) -> Option<Self> { Some(Self::from_le_bytes(input.take()?)) } }
    )*};
}
integer!(u8, u16, u32, u64);
impl Encode for bool {
    fn encode(&self, out: &mut Vec<u8>) {
        (*self as u8).encode(out);
    }
}
impl Decode for bool {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        match u8::decode(input)? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}
impl Encode for f32 {
    fn encode(&self, out: &mut Vec<u8>) {
        self.to_bits().encode(out);
    }
}
impl Decode for f32 {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Self::from_bits(u32::decode(input)?))
    }
}
impl<T: Encode> Encode for Option<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        self.is_some().encode(out);
        if let Some(value) = self {
            value.encode(out);
        }
    }
}
impl<T: Decode> Decode for Option<T> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(if bool::decode(input)? {
            Some(T::decode(input)?)
        } else {
            None
        })
    }
}
impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode(&self, out: &mut Vec<u8>) {
        for value in self {
            value.encode(out);
        }
    }
}
impl<T: Decode, const N: usize> Decode for [T; N] {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        (0..N)
            .map(|_| T::decode(input))
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()
    }
}
impl<T: Encode> Encode for [T] {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.len() as u32).encode(out);
        for value in self {
            value.encode(out);
        }
    }
}
impl<T: Encode> Encode for Vec<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        self.as_slice().encode(out);
    }
}
impl<T: Decode> Decode for Vec<T> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        (0..input.count()?).map(|_| T::decode(input)).collect()
    }
}
impl<T: Encode> Encode for Box<[T]> {
    fn encode(&self, out: &mut Vec<u8>) {
        self.as_ref().encode(out);
    }
}
impl<T: Decode> Decode for Box<[T]> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Vec::decode(input)?.into_boxed_slice())
    }
}
impl<K: Encode, V: Encode> Encode for BTreeMap<K, V> {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.len() as u32).encode(out);
        for (key, value) in self {
            key.encode(out);
            value.encode(out);
        }
    }
}
impl<K: Decode + Ord, V: Decode> Decode for BTreeMap<K, V> {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        let mut map = Self::new();
        for _ in 0..input.count()? {
            if map.insert(K::decode(input)?, V::decode(input)?).is_some() {
                return None;
            }
        }
        Some(map)
    }
}
macro_rules! record {
    ($ty:ident { $($field:ident),* $(,)? }) => {
        impl Encode for $ty {
            fn encode(&self, out: &mut Vec<u8>) {
                let Self { $($field),* } = self;
                $($field.encode(out);)*
            }
        }
        impl Decode for $ty {
            fn decode(input: &mut Reader<'_>) -> Option<Self> { Some(Self { $($field: Decode::decode(input)?),* }) }
        }
    };
}
macro_rules! wrapper {
    ($($ty:ident),*) => {$(
        impl Encode for $ty { fn encode(&self, out: &mut Vec<u8>) { self.0.encode(out); } }
        impl Decode for $ty { fn decode(input: &mut Reader<'_>) -> Option<Self> { Some(Self(Decode::decode(input)?)) } }
    )*};
}
wrapper!(ProgramOffset, ByteOffset, CbufIndex, TextureHandle);
macro_rules! enumeration {
    ($($ty:ident),*) => {$(
        impl Encode for $ty { fn encode(&self, out: &mut Vec<u8>) { (*self as u32).encode(out); } }
        impl Decode for $ty { fn decode(input: &mut Reader<'_>) -> Option<Self> { Self::try_from(u32::decode(input)?).ok() } }
    )*};
}
enumeration!(Stage, ReplaceConstant, TextureType, TexturePixelFormat);
impl Encode for ProgramHeader {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bytes());
    }
}
impl Decode for ProgramHeader {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Self::from_bytes(&input.take()?))
    }
}
impl Encode for SnapshotEnvironment {
    fn encode(&self, out: &mut Vec<u8>) {
        self.metadata().encode(out);
        self.observations().encode(out);
    }
}
impl Decode for SnapshotEnvironment {
    fn decode(input: &mut Reader<'_>) -> Option<Self> {
        Some(Self::new(Decode::decode(input)?, Decode::decode(input)?))
    }
}

record!(CbufWordKey { index, offset });
record!(HandleKey {
    index,
    offset,
    shift_left,
    sec_index,
    sec_offset,
    sec_shift_left,
    count,
    has_secondary
});

record!(EnvironmentMetadata {
    sph,
    gp_passthrough_mask,
    stage,
    start_address,
    is_proprietary_driver,
    texture_bound_buffer,
    local_memory_size,
    shared_memory_size,
    workgroup_size,
    has_hle_macro_state
});

record!(Observations {
    instructions,
    cbuf_words,
    texture_handles,
    texture_types,
    texture_pixel_formats,
    texture_integer_formats,
    viewport_transform_state,
    replacements
});

pub fn encode_inputs(inputs: &crate::shader_compilation::ShaderInputs) -> Vec<u8> {
    let mut bytes = b"NXCAPT01".to_vec();
    SCHEMA.encode(&mut bytes);
    ARTIFACT_ABI_VERSION.encode(&mut bytes);
    (inputs.stages.len() as u32).encode(&mut bytes);
    for environment in &inputs.stages {
        environment.encode(&mut bytes);
    }
    bytes
}
pub fn decode_inputs(bytes: &[u8]) -> Option<crate::shader_compilation::ShaderInputs> {
    let mut input = Reader { remaining: bytes };
    if &input.take::<8>()? != b"NXCAPT01"
        || u32::decode(&mut input)? != SCHEMA
        || u32::decode(&mut input)? != ARTIFACT_ABI_VERSION
    {
        return None;
    }
    let count = input.count()?;
    if !(1..=6).contains(&count) {
        return None;
    }
    let stages = (0..count)
        .map(|_| Decode::decode(&mut input))
        .collect::<Option<Box<[_]>>>()?;
    input
        .remaining
        .is_empty()
        .then_some(crate::shader_compilation::ShaderInputs {
            stages,
            serializable: true,
        })
}
