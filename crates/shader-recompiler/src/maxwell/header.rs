use crate::{
    CompileError, Result,
    diagnostics::FailureCategory,
    types::{OutputTopology, PixelImap},
};

pub const PROGRAM_HEADER_SIZE: usize = 0x50;

/// The 80-byte graphics header, decoded in little-endian word order. Raw storage
/// preserves reserved fields and the overlapping VTG and fragment interpretations.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProgramHeader {
    words: [u32; 20],
}

macro_rules! fields {
    ($($name:ident: $word:literal, $start:literal, $width:literal;)*) => {
        $(pub const fn $name(&self) -> u32 {
            (self.words[$word] >> $start) & ((1u32 << $width) - 1)
        })*
    };
}

impl ProgramHeader {
    pub fn from_bytes(bytes: &[u8; PROGRAM_HEADER_SIZE]) -> Self {
        Self {
            words: std::array::from_fn(|i| {
                u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
            }),
        }
    }

    pub fn to_bytes(&self) -> [u8; PROGRAM_HEADER_SIZE] {
        let mut bytes = [0; PROGRAM_HEADER_SIZE];
        for (word, chunk) in self.words.iter().zip(bytes.chunks_exact_mut(4)) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    pub fn words(&self) -> &[u32; 20] {
        &self.words
    }

    fn byte(&self, offset: usize) -> u8 {
        (self.words[offset / 4] >> ((offset % 4) * 8)) as u8
    }

    fields! {
        sph_type: 0, 0, 5;
        version: 0, 5, 5;
        shader_type: 0, 10, 4;
        mrt_enable: 0, 14, 1;
        kills_pixels: 0, 15, 1;
        does_global_store: 0, 16, 1;
        sass_version: 0, 17, 4;
        common0_reserved1: 0, 21, 2;
        geometry_passthrough: 0, 24, 1;
        common0_reserved2: 0, 25, 1;
        does_load_or_store: 0, 26, 1;
        does_fp64: 0, 27, 1;
        stream_out_mask: 0, 28, 4;
        shader_local_memory_low_size: 1, 0, 24;
        per_patch_attribute_count: 1, 24, 8;
        shader_local_memory_high_size: 2, 0, 24;
        threads_per_input_primitive: 2, 24, 8;
        shader_local_memory_crs_size: 3, 0, 24;
        output_topology_raw: 3, 24, 4;
        common3_reserved: 3, 28, 4;
        max_output_vertices: 4, 0, 12;
        store_req_start: 4, 12, 8;
        common4_reserved: 4, 20, 4;
        store_req_end: 4, 24, 8;
    }

    pub fn output_topology(&self) -> Result<OutputTopology> {
        OutputTopology::try_from(self.output_topology_raw())
    }

    pub const fn local_memory_size(&self) -> u64 {
        self.shader_local_memory_low_size() as u64
            | ((self.shader_local_memory_high_size() as u64) << 24)
    }

    pub fn vtg_input_system_b(&self) -> SystemValuesB {
        SystemValuesB(self.byte(23))
    }
    pub fn vtg_output_system_b(&self) -> SystemValuesB {
        SystemValuesB(self.byte(53))
    }
    pub fn vtg_input_system_c(&self) -> SystemValuesC {
        SystemValuesC(u16::from_le_bytes([self.byte(42), self.byte(43)]))
    }
    pub fn vtg_output_system_c(&self) -> SystemValuesC {
        SystemValuesC(u16::from_le_bytes([self.byte(72), self.byte(73)]))
    }

    pub fn input_generic(&self, index: usize) -> [bool; 4] {
        assert!(index < 32);
        components(self.byte(24 + (index >> 1)) >> ((index % 2) * 4))
    }

    pub fn output_generic(&self, index: usize) -> [bool; 4] {
        assert!(index < 32);
        components(self.byte(54 + (index >> 1)) >> ((index % 2) * 4))
    }

    pub fn fragment_input_system_b(&self) -> SystemValuesB {
        SystemValuesB(self.byte(23))
    }

    pub fn generic_input_map(&self, attribute: usize) -> [PixelImap; 4] {
        assert!(attribute < 32);
        let vector = self.byte(24 + attribute);
        std::array::from_fn(|i| match (vector >> (i * 2)) & 3 {
            0 => PixelImap::Unused,
            1 => PixelImap::Constant,
            2 => PixelImap::Perspective,
            _ => PixelImap::ScreenLinear,
        })
    }

    pub fn is_generic_vector_active(&self, index: usize) -> bool {
        assert!(index < 32);
        self.byte(24 + index) != 0
    }

    pub const fn fragment_output_mask(&self) -> u32 {
        self.words[18]
    }
    pub const fn fragment_sample_mask(&self) -> bool {
        self.words[19] & 1 != 0
    }
    pub const fn fragment_depth(&self) -> bool {
        self.words[19] & 2 != 0
    }
    pub const fn fragment_output_reserved(&self) -> u32 {
        self.words[19] >> 2
    }

    pub fn enabled_output_components(&self, rt: usize) -> [bool; 4] {
        assert!(rt < 8);
        components((self.fragment_output_mask() >> (rt * 4)) as u8)
    }

    pub fn has_output_components(&self, rt: usize) -> bool {
        self.enabled_output_components(rt)
            .into_iter()
            .any(|value| value)
    }
}

impl TryFrom<&[u8]> for ProgramHeader {
    type Error = CompileError;
    fn try_from(bytes: &[u8]) -> Result<Self> {
        let bytes = bytes.try_into().map_err(|_| {
            CompileError::new(
                FailureCategory::InvalidArgument,
                format!("Graphics header requires 80 bytes, got {}", bytes.len()),
            )
        })?;
        Ok(Self::from_bytes(bytes))
    }
}

fn components(bits: u8) -> [bool; 4] {
    std::array::from_fn(|i| bits & (1 << i) != 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemValuesB(pub u8);

impl SystemValuesB {
    pub const fn primitive_array_id(self) -> bool {
        self.0 & 1 != 0
    }
    pub const fn rt_array_index(self) -> bool {
        self.0 & 2 != 0
    }
    pub const fn viewport_index(self) -> bool {
        self.0 & 4 != 0
    }
    pub const fn point_size(self) -> bool {
        self.0 & 8 != 0
    }
    pub const fn position_x(self) -> bool {
        self.0 & 16 != 0
    }
    pub const fn position_y(self) -> bool {
        self.0 & 32 != 0
    }
    pub const fn position_z(self) -> bool {
        self.0 & 64 != 0
    }
    pub const fn position_w(self) -> bool {
        self.0 & 128 != 0
    }
    pub const fn first(self) -> u8 {
        self.0 & 15
    }
    pub const fn position(self) -> u8 {
        self.0 >> 4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemValuesC(pub u16);

impl SystemValuesC {
    pub const fn clip_distances(self) -> u8 {
        self.0 as u8
    }
    pub const fn point_sprite_s(self) -> bool {
        self.0 & (1 << 8) != 0
    }
    pub const fn point_sprite_t(self) -> bool {
        self.0 & (1 << 9) != 0
    }
    pub const fn fog_coordinate(self) -> bool {
        self.0 & (1 << 10) != 0
    }
    pub const fn tessellation_eval_point_u(self) -> bool {
        self.0 & (1 << 12) != 0
    }
    pub const fn tessellation_eval_point_v(self) -> bool {
        self.0 & (1 << 13) != 0
    }
    pub const fn instance_id(self) -> bool {
        self.0 & (1 << 14) != 0
    }
    pub const fn vertex_id(self) -> bool {
        self.0 & (1 << 15) != 0
    }
}
