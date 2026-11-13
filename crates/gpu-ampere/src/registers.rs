//! Native register banks. Ordinary writes are interpreted when consumed by an
//! effect; method validity does not depend on a Maxwell structure's layout.
mod methods;
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Class {
    ThreeD = 0xc797,
    Compute = 0xc7c0,
    Inline = 0xa140,
    TwoD = 0x902d,
    Copy = 0xc7b5,
}
impl Class {
    pub fn from_id(id: u32) -> Result<Self> {
        match id {
            0xc797 => Ok(Self::ThreeD),
            0xc7c0 => Ok(Self::Compute),
            0xa140 => Ok(Self::Inline),
            0x902d => Ok(Self::TwoD),
            0xc7b5 => Ok(Self::Copy),
            _ => Err(format!("unsupported object class {id:#x}").into()),
        }
    }
    pub(crate) fn index(self) -> usize {
        match self {
            Self::ThreeD => 0,
            Self::Compute => 1,
            Self::Inline => 2,
            Self::TwoD => 3,
            Self::Copy => 4,
        }
    }
    pub(crate) fn mme(self) -> bool {
        matches!(self, Self::ThreeD | Self::Compute)
    }
}

pub struct Registers {
    pub class: Class,
    words: Box<[u32; 0xe00]>,
}
impl Registers {
    pub fn new(class: Class) -> Self {
        let mut registers = Self {
            class,
            words: Box::new([0; 0xe00]),
        };
        if class == Class::TwoD {
            // The 2D context starts with one slice, as the driver's transfer
            // initialization assumes. Subsequent guest writes remain canonical.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/engines/fermi_2d.cpp
            registers.set(0x20c, 1);
            registers.set(0x23c, 1);
            // @SPECULATIVE: native context reset values are not documented.
            // Use the enabled 2D render condition required by existing driver
            // initialization; the reference executes these blits unconditionally.
            registers.set(0x26c, 1);
        }
        registers
    }
    pub fn get(&self, method: u32) -> u32 {
        self.words[(method / 4) as usize]
    }
    pub(crate) fn set(&mut self, method: u32, value: u32) -> bool {
        let index = (method / 4) as usize;
        if self.words[index] == value {
            return false;
        }
        self.words[index] = value;
        true
    }
    pub fn address(&self, high: u32, low: u32) -> u64 {
        (u64::from(self.get(high)) << 32) | u64::from(self.get(low))
    }
}

pub(crate) fn method(class: Class, m: u32) -> Option<(&'static str, bool)> {
    if m & 3 != 0 || m >= 0x3800 {
        return None;
    }
    if let Some(scalar) = methods::scalar(class, m) {
        return Some(scalar);
    }
    // Indexed registers, with bounds from the public class layout. Gaps are
    // deliberately not admitted as state just because the bank has storage.
    let name = match class {
        Class::ThreeD => match m {
            0x240..=0x250 => "SET_ROOT_TABLE_VISIBILITY",
            0x380..=0x3f0 if (m - 0x380) % 32 <= 16 => "SET_STREAM_OUT_BUFFER",
            0x400..=0x4fc => "SET_POSITION_W_SCALED_OFFSET",
            0x600..=0x6fc => "SET_VERTEX_STREAM_SIZE",
            0x700..=0x738 if (m - 0x700) % 16 <= 8 => "SET_STREAM_OUT_CONTROL",
            0x800..=0x9e4 if (m - 0x800) % 64 <= 36 => "SET_COLOR_TARGET",
            0xa00..=0xbfc => "SET_VIEWPORT",
            0xc00..=0xcfc => "SET_VIEWPORT_CLIP",
            0xd00..=0xd3c => "SET_WINDOW_CLIP",
            0xd40..=0xd5c => "SET_CLIP_ID_EXTENT",
            0xd80..=0xd8c => "SET_COLOR_CLEAR_VALUE",
            0xe00..=0xef8 if (m - 0xe00) % 16 <= 8 => "SET_SCISSOR",
            0x1030..=0x106c => "SET_TIR_MODULATION_COEFFICIENT_TABLE",
            0x1120..=0x112c => "SET_DA_OUTPUT_ATTRIBUTE_SKIP_MASK",
            0x1160..=0x11dc => "SET_VERTEX_ATTRIBUTE",
            0x11e0..=0x11ec => "SET_ANTI_ALIAS_SAMPLE_POSITIONS",
            0x1240..=0x125c => "SET_POST_VTG_SHADER_ATTRIBUTE_SKIP_MASK",
            0x1360..=0x137c => "SET_BLEND",
            0x1480..=0x14fc => "SET_ITERATED_BLEND_CONSTANT",
            0x1670..=0x168c => "SET_LINE_SMOOTH_EDGE_TABLE",
            0x16c0..=0x16fc => "SET_SCG_GRAPHICS_SCHEDULING_PARAMETERS",
            0x1700..=0x177c => "SET_POLYGON_STIPPLE_PATTERN",
            0x1880..=0x18fc => "SET_VERTEX_STREAM_INSTANCE",
            0x19e0..=0x19fc => "SET_COLOR_COMPRESSION",
            0x1a00..=0x1a1c => "SET_CT_WRITE",
            0x1c00..=0x1dfc => "SET_VERTEX_STREAM",
            0x1e00..=0x1ef8 if (m - 0x1e00) % 32 <= 24 => "SET_BLEND_PER_TARGET",
            0x2000..=0x2160 if (m - 0x2000) % 64 <= 32 => "SET_PIPELINE_SHADER",
            0x2390..=0x23cc => return Some(("LOAD_CONSTANT_BUFFER", false)),
            0x2800..=0x2bfc => "SET_STREAM_OUT_LAYOUT_SELECT",
            0x2410..=0x2490 if (m - 0x2410).is_multiple_of(32) => {
                return Some(("BIND_GROUP_CONSTANT_BUFFER", false));
            }
            // @SPECULATIVE: these GA10F SDK initializer registers are ordinary
            // state in the reference, but absent from public AMPERE_B headers.
            // Preserve their values; graphics consumption must interpret them
            // before native drawing can be supported (not silently discard).
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/engines/maxwell_3d.h
            0x230
            | 0x2cc
            | 0xf80
            | 0xf60..=0xf70
            | 0x1108
            | 0x1148
            | 0x114c
            | 0x12a8..=0x12bc
            | 0x15b4 => "GA10F_INITIAL_STATE",
            0x2610..=0x2700 if m.is_multiple_of(16) => "GA10F_INITIAL_STATE",
            0x3400..=0x37fc => "SET_MME_SHADOW_SCRATCH",
            _ => return None,
        },
        Class::Compute => match m {
            // @SPECULATIVE: native compute scheduling state stored by Lichii;
            // public desktop C7C0 headers do not name this GA10F range.
            0xda0..=0xdb0 => "GA10F_COMPUTE_SCHEDULING_STATE",
            0x320..=0x41c => return Some(("LOAD_INLINE_QMD_DATA", m != 0x41c)),
            0x3400..=0x37fc => "SET_MME_SHADOW_SCRATCH",
            _ => return None,
        },
        _ => return None,
    };
    Some((name, true))
}
