//! Typed vertex-input and primitive topology state for `MAXWELL_B`.
//!
//! Method layouts and enum values come from NVIDIA's public `clb197.h` at
//! commit `9fdf5c4062007929d9f4e6cbad9c9771fe61b880`:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h

use crate::MaxwellMethodSource;

use super::state::MaxwellThreeDRegister;

pub const MAXWELL_VERTEX_STREAM_COUNT: usize = 32;
pub const MAXWELL_VERTEX_ATTRIBUTE_COUNT: usize = 32;
pub const MAXWELL_THREE_D_PRIMITIVE_AREA_MAX: u32 = 0x003f_ffff;

/// One constant vector selected for an absent vertex attribute.
///
/// The individual `SET_ATTRIBUTE_DEFAULT` fields expose different subsets of
/// these three vectors. Keeping the decoded vector rather than a bare bit
/// preserves that field-specific meaning.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3002-L3020>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDAttributeDefaultVector {
    Vector0000,
    Vector0001,
    Vector1111,
}

/// Complete typed value written by `SET_ATTRIBUTE_DEFAULT`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDAttributeDefaults {
    color_front_diffuse: MaxwellThreeDAttributeDefaultVector,
    color_front_specular: MaxwellThreeDAttributeDefaultVector,
    generic_vector: MaxwellThreeDAttributeDefaultVector,
    fixed_function_texture: MaxwellThreeDAttributeDefaultVector,
    dx9_color0: MaxwellThreeDAttributeDefaultVector,
    dx9_color1_to_color15: MaxwellThreeDAttributeDefaultVector,
}

impl MaxwellThreeDAttributeDefaults {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x3f != 0 {
            return None;
        }
        Some(Self {
            color_front_diffuse: if raw & 1 == 0 {
                MaxwellThreeDAttributeDefaultVector::Vector0001
            } else {
                MaxwellThreeDAttributeDefaultVector::Vector1111
            },
            color_front_specular: if raw & 2 == 0 {
                MaxwellThreeDAttributeDefaultVector::Vector0000
            } else {
                MaxwellThreeDAttributeDefaultVector::Vector0001
            },
            generic_vector: if raw & 4 == 0 {
                MaxwellThreeDAttributeDefaultVector::Vector0000
            } else {
                MaxwellThreeDAttributeDefaultVector::Vector0001
            },
            fixed_function_texture: if raw & 8 == 0 {
                MaxwellThreeDAttributeDefaultVector::Vector0000
            } else {
                MaxwellThreeDAttributeDefaultVector::Vector0001
            },
            dx9_color0: if raw & 0x10 == 0 {
                MaxwellThreeDAttributeDefaultVector::Vector0001
            } else {
                MaxwellThreeDAttributeDefaultVector::Vector1111
            },
            dx9_color1_to_color15: if raw & 0x20 == 0 {
                MaxwellThreeDAttributeDefaultVector::Vector0000
            } else {
                MaxwellThreeDAttributeDefaultVector::Vector0001
            },
        })
    }

    #[must_use]
    pub const fn color_front_diffuse(self) -> MaxwellThreeDAttributeDefaultVector {
        self.color_front_diffuse
    }

    #[must_use]
    pub const fn color_front_specular(self) -> MaxwellThreeDAttributeDefaultVector {
        self.color_front_specular
    }

    #[must_use]
    pub const fn generic_vector(self) -> MaxwellThreeDAttributeDefaultVector {
        self.generic_vector
    }

    #[must_use]
    pub const fn fixed_function_texture(self) -> MaxwellThreeDAttributeDefaultVector {
        self.fixed_function_texture
    }

    #[must_use]
    pub const fn dx9_color0(self) -> MaxwellThreeDAttributeDefaultVector {
        self.dx9_color0
    }

    #[must_use]
    pub const fn dx9_color1_to_color15(self) -> MaxwellThreeDAttributeDefaultVector {
        self.dx9_color1_to_color15
    }
}

/// Whether the vertex shader's vertex ID includes `SET_VERTEX_ARRAY_START`.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3087-L3090>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDVertexIdUsesArrayStart {
    Disabled = 0,
    Enabled = 0x1000,
}

impl MaxwellThreeDVertexIdUsesArrayStart {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            0x1000 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Data-assembly controls that affect values presented to the vertex shader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDVertexAssemblyState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDVertexAssemblyState<'_> {
    #[must_use]
    pub fn attribute_defaults(&self) -> MaxwellThreeDRegister<MaxwellThreeDAttributeDefaults> {
        self.registers
            .read(0x1610, MaxwellThreeDAttributeDefaults::parse)
    }
    #[must_use]
    pub fn vertex_id_uses_array_start(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDVertexIdUsesArrayStart> {
        self.registers.read(0x164c, |raw| {
            MaxwellThreeDVertexIdUsesArrayStart::parse(raw)
        })
    }
    #[must_use]
    pub fn vertex_id_base(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1118, Some)
    }
    #[must_use]
    pub fn global_base_vertex_index(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1434, Some)
    }
    #[must_use]
    pub fn global_base_instance_index(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1438, Some)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDUnresolvedAddress {
    upper: u8,
    lower: u32,
}

/// GPU address programmed by `SET_VERTEX_STREAM_SUBSTITUTE_A/B`.
///
/// NVIDIA's public class header defines this as one 40-bit address split into
/// an eight-bit upper field and a 32-bit lower field, but does not document
/// the condition that consumes it.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1255-L1259>
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDVertexStreamSubstituteState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDVertexStreamSubstituteState<'_> {
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(0x0f84, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x0f88, Some)
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper().value()?,
            *self.address_lower().value()?,
        ))
    }
}

impl MaxwellThreeDUnresolvedAddress {
    #[must_use]
    pub const fn new(upper: u8, lower: u32) -> Self {
        Self { upper, lower }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        (self.upper as u64) << 32 | self.lower as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDVertexStreamFormat {
    stride: u16,
    enabled: bool,
}

impl MaxwellThreeDVertexStreamFormat {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x1fff != 0 {
            return None;
        }
        Some(Self {
            stride: (raw & 0x0fff) as u16,
            enabled: raw & 0x1000 != 0,
        })
    }

    #[must_use]
    pub const fn stride(self) -> u16 {
        self.stride
    }

    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDVertexComponentWidths(u8);

impl MaxwellThreeDVertexComponentWidths {
    pub(super) const fn parse(raw: u8) -> Option<Self> {
        match raw {
            0x01 | 0x02 | 0x03 | 0x04 | 0x05 | 0x0a | 0x0f | 0x12 | 0x13 | 0x18 | 0x1b | 0x1d
            | 0x2f | 0x30 | 0x31 | 0x32 | 0x33 | 0x34 => Some(Self(raw)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn byte_size(self) -> u16 {
        match self.0 {
            0x01 => 16,
            0x02 => 12,
            0x03 => 8,
            0x04 => 8,
            0x05 => 6,
            0x13 => 3,
            0x0f | 0x12 => 4,
            0x18 | 0x1b | 0x32 => 2,
            0x1d | 0x34 => 1,
            _ => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDVertexNumericalType {
    SignedNormalized,
    UnsignedNormalized,
    SignedInteger,
    UnsignedInteger,
    UnsignedScaled,
    SignedScaled,
    Float,
}

impl MaxwellThreeDVertexNumericalType {
    pub(super) const fn parse(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(Self::SignedNormalized),
            2 => Some(Self::UnsignedNormalized),
            3 => Some(Self::SignedInteger),
            4 => Some(Self::UnsignedInteger),
            5 => Some(Self::UnsignedScaled),
            6 => Some(Self::SignedScaled),
            7 => Some(Self::Float),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDVertexAttributeFormat {
    stream: u8,
    enabled: bool,
    offset: u16,
    component_widths: Option<MaxwellThreeDVertexComponentWidths>,
    numerical_type: Option<MaxwellThreeDVertexNumericalType>,
    swap_red_blue: bool,
}

impl MaxwellThreeDVertexAttributeFormat {
    pub(in crate::engines) const fn parse(raw: u32) -> Option<Self> {
        let stream = (raw & 0x1f) as u8;
        let enabled = raw & (1 << 6) == 0;
        let offset = ((raw >> 7) & 0x3fff) as u16;
        let component_widths =
            MaxwellThreeDVertexComponentWidths::parse(((raw >> 21) & 0x3f) as u8);
        let numerical_type = MaxwellThreeDVertexNumericalType::parse(((raw >> 27) & 7) as u8);
        if enabled && (component_widths.is_none() || numerical_type.is_none()) {
            return None;
        }
        Some(Self {
            stream,
            enabled,
            offset,
            component_widths,
            numerical_type,
            swap_red_blue: raw >> 31 != 0,
        })
    }

    #[must_use]
    pub const fn stream(self) -> u8 {
        self.stream
    }
    #[must_use]
    pub const fn enabled(self) -> bool {
        self.enabled
    }
    #[must_use]
    pub const fn offset(self) -> u16 {
        self.offset
    }
    #[must_use]
    pub const fn component_widths(self) -> Option<MaxwellThreeDVertexComponentWidths> {
        self.component_widths
    }
    #[must_use]
    pub const fn numerical_type(self) -> Option<MaxwellThreeDVertexNumericalType> {
        self.numerical_type
    }
    #[must_use]
    pub const fn swap_red_blue(self) -> bool {
        self.swap_red_blue
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDVertexStreamState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    index: usize,
}
impl MaxwellThreeDVertexStreamState<'_> {
    #[must_use]
    pub fn format(&self) -> MaxwellThreeDRegister<MaxwellThreeDVertexStreamFormat> {
        self.registers.read(0x1c00 + self.index as u32 * 16, |raw| {
            MaxwellThreeDVertexStreamFormat::parse(raw)
        })
    }
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(0x1c04 + self.index as u32 * 16, |raw| {
            u8::try_from(raw).ok()
        })
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1c08 + self.index as u32 * 16, Some)
    }
    #[must_use]
    pub fn frequency(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1c0c + self.index as u32 * 16, Some)
    }
    #[must_use]
    pub fn limit_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers
            .read(0x1f00 + self.index as u32 * 8, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn limit_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1f04 + self.index as u32 * 8, Some)
    }
    #[must_use]
    pub fn instanced(&self) -> MaxwellThreeDRegister<bool> {
        self.registers.read(0x1880 + self.index as u32 * 4, |raw| {
            (raw <= 1).then_some(raw != 0)
        })
    }
    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper().value()?,
            *self.address_lower().value()?,
        ))
    }
    #[must_use]
    pub fn limit(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.limit_upper().value()?,
            *self.limit_lower().value()?,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDIndexElementSize {
    OneByte,
    TwoBytes,
    FourBytes,
}

impl MaxwellThreeDIndexElementSize {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::OneByte),
            1 => Some(Self::TwoBytes),
            2 => Some(Self::FourBytes),
            _ => None,
        }
    }
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::OneByte => 1,
            Self::TwoBytes => 2,
            Self::FourBytes => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDIndexBufferState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
}
impl MaxwellThreeDIndexBufferState<'_> {
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(0x17c8, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x17cc, Some)
    }
    #[must_use]
    pub fn limit_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers.read(0x17d0, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn limit_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x17d4, Some)
    }
    #[must_use]
    pub fn element_size(&self) -> MaxwellThreeDRegister<MaxwellThreeDIndexElementSize> {
        self.registers
            .read(0x17d8, MaxwellThreeDIndexElementSize::parse)
    }
    #[must_use]
    pub fn first(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x17dc, Some)
    }
    #[must_use]
    pub fn count(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x17e0, Some)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDPrimitiveTopology(u16);

impl MaxwellThreeDPrimitiveTopology {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x0001..=0x0005
            | 0x000a..=0x000e
            | 0x1001..=0x1003
            | 0x100f..=0x1018
            | 0x101a
            | 0x101b
                if raw != 0x1019 =>
            {
                Some(Self(raw as u16))
            }
            _ => None,
        }
    }
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Whether each non-indexed vertex-array draw starts a new primitive segment.
///
/// This selector is deliberately distinct from the indexed primitive-restart
/// enable and index registers. Each neutral draw already has an explicit
/// command boundary, so enabled restart is represented without a host pipeline
/// selector, including for connected topologies such as triangle strips.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1084-L1090>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDVertexArrayPrimitiveRestartEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDVertexArrayPrimitiveRestartEnable {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Primitive-area threshold for internal circular-buffer throttling.
///
/// NVIDIA's public ABI defines `PRIM_AREA` as bits 21:0, but exposes no
/// output-visible rendering selection through this method. The frontend keeps
/// the value and its source for diagnostics and future scheduling models while
/// excluding this internal throttle from logical pipeline identity.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L293-L294>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDPrimitiveCircularBufferThrottle(u32);

impl MaxwellThreeDPrimitiveCircularBufferThrottle {
    pub(super) const fn new(primitive_area: u32) -> Option<Self> {
        if primitive_area <= MAXWELL_THREE_D_PRIMITIVE_AREA_MAX {
            Some(Self(primitive_area))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn primitive_area(self) -> u32 {
        self.0
    }
}

/// Number of control points in one patch primitive.
///
/// NVIDIA publishes an eight-bit field but no narrower validity range in the
/// public class header. All encodings are therefore retained; execution-time
/// constraints are checked only when a patch topology consumes the value.
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1046-L1047>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDPatchSize(u8);

impl MaxwellThreeDPatchSize {
    #[must_use]
    pub const fn new(control_points: u8) -> Self {
        Self(control_points)
    }

    #[must_use]
    pub const fn control_points(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// Internal primitive-work scheduling policy selected by
/// `SET_BALANCED_PRIMITIVE_WORKLOAD`.
///
/// Both boolean fields are published in NVIDIA's public Kepler 3D class and
/// inherited by later Maxwell classes. They tune hardware scheduling rather
/// than the observable primitive or raster result:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cl9297.h#L476-L482>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDBalancedPrimitiveWorkload {
    in_unpartitioned_mode: bool,
    in_timesliced_mode: bool,
}

impl MaxwellThreeDBalancedPrimitiveWorkload {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x11 != 0 {
            return None;
        }
        Some(Self {
            in_unpartitioned_mode: raw & 1 != 0,
            in_timesliced_mode: raw & 0x10 != 0,
        })
    }

    #[must_use]
    pub const fn in_unpartitioned_mode(self) -> bool {
        self.in_unpartitioned_mode
    }

    #[must_use]
    pub const fn in_timesliced_mode(self) -> bool {
        self.in_timesliced_mode
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.in_unpartitioned_mode as u32 | ((self.in_timesliced_mode as u32) << 4)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct MaxwellThreeDPrimitiveSequence {
    begin: MaxwellThreeDRegister<MaxwellThreeDBegin>,
    end: MaxwellThreeDRegister<bool>,
    instance_index: u32,
    open: bool,
}
impl MaxwellThreeDPrimitiveSequence {
    pub(super) fn begin(&mut self, raw: u32, value: MaxwellThreeDBegin) {
        self.instance_index = match value.instance() {
            MaxwellThreeDBeginInstance::First => 0,
            MaxwellThreeDBeginInstance::Subsequent => self.instance_index.wrapping_add(1),
            MaxwellThreeDBeginInstance::Unchanged => self.instance_index,
        };
        self.begin = MaxwellThreeDRegister::decoded(raw, value);
        self.open = true;
    }
    pub(super) fn end(&mut self, raw: u32, value: bool) {
        self.end = MaxwellThreeDRegister::decoded(raw, value);
        self.open = false;
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDPrimitiveState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    pub(super) sequence: &'a MaxwellThreeDPrimitiveSequence,
}
impl<'a> MaxwellThreeDPrimitiveState<'a> {
    #[must_use]
    pub fn balanced_workload(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDBalancedPrimitiveWorkload> {
        self.registers.read(0x0374, |raw| {
            MaxwellThreeDBalancedPrimitiveWorkload::parse(raw)
        })
    }
    #[must_use]
    pub fn circular_buffer_throttle(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDPrimitiveCircularBufferThrottle> {
        self.registers.read(0x02d0, |raw| {
            MaxwellThreeDPrimitiveCircularBufferThrottle::new(raw)
        })
    }
    #[must_use]
    pub fn patch_size(&self) -> MaxwellThreeDRegister<MaxwellThreeDPatchSize> {
        self.registers.read(0x0dcc, |raw| {
            (raw <= 0xff).then_some(MaxwellThreeDPatchSize::new(raw as u8))
        })
    }
    #[must_use]
    pub fn topology_override(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x1948, |raw| (raw <= 1).then_some(raw != 0))
    }
    #[must_use]
    pub fn topology(&self) -> MaxwellThreeDRegister<MaxwellThreeDPrimitiveTopology> {
        self.registers
            .read(0x1970, MaxwellThreeDPrimitiveTopology::parse)
    }
    #[must_use]
    pub fn vertex_array_restart_enabled(
        &self,
    ) -> MaxwellThreeDRegister<MaxwellThreeDVertexArrayPrimitiveRestartEnable> {
        self.registers.read(0x0de8, |raw| {
            MaxwellThreeDVertexArrayPrimitiveRestartEnable::parse(raw)
        })
    }
    #[must_use]
    pub fn restart_enabled(&self) -> MaxwellThreeDRegister<bool> {
        self.registers
            .read(0x1644, |raw| (raw <= 1).then_some(raw != 0))
    }
    #[must_use]
    pub fn restart_index(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1648, Some)
    }
    #[must_use]
    pub fn vertex_array_start(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x0d74, Some)
    }
    #[must_use]
    pub const fn begin(&self) -> &'a MaxwellThreeDRegister<MaxwellThreeDBegin> {
        &self.sequence.begin
    }
    #[must_use]
    pub const fn end(&self) -> &'a MaxwellThreeDRegister<bool> {
        &self.sequence.end
    }
    #[must_use]
    pub const fn active_begin(&self) -> Option<&'a MaxwellThreeDBegin> {
        if self.sequence.open {
            self.sequence.begin.value()
        } else {
            None
        }
    }
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.sequence.open
    }
    #[must_use]
    pub const fn instance_index(&self) -> u32 {
        self.sequence.instance_index
    }
}

/// Instance-counter transition selected by `BEGIN.INSTANCE_ID`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDBeginInstance {
    First,
    Subsequent,
    Unchanged,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDBegin {
    topology: u8,
    preserve_primitive_id: bool,
    instance: MaxwellThreeDBeginInstance,
    split_mode: u8,
}

impl MaxwellThreeDBegin {
    pub(in crate::engines) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x6d00_ffff != 0 || raw & 0xffff > 0x0e || ((raw >> 26) & 3) == 3 {
            return None;
        }
        Some(Self {
            topology: (raw & 0xff) as u8,
            preserve_primitive_id: raw & (1 << 24) != 0,
            instance: match (raw >> 26) & 3 {
                0 => MaxwellThreeDBeginInstance::First,
                1 => MaxwellThreeDBeginInstance::Subsequent,
                2 => MaxwellThreeDBeginInstance::Unchanged,
                _ => return None,
            },
            split_mode: ((raw >> 29) & 3) as u8,
        })
    }
    #[must_use]
    pub const fn topology(self) -> u8 {
        self.topology
    }
    #[must_use]
    pub const fn preserve_primitive_id(self) -> bool {
        self.preserve_primitive_id
    }
    #[must_use]
    pub const fn instance_id(self) -> u8 {
        match self.instance {
            MaxwellThreeDBeginInstance::First => 0,
            MaxwellThreeDBeginInstance::Subsequent => 1,
            MaxwellThreeDBeginInstance::Unchanged => 2,
        }
    }
    #[must_use]
    pub const fn instance(self) -> MaxwellThreeDBeginInstance {
        self.instance
    }
    #[must_use]
    pub const fn split_mode(self) -> u8 {
        self.split_mode
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDVertexInputState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    pub(super) sequence: &'a MaxwellThreeDPrimitiveSequence,
}
impl<'a> MaxwellThreeDVertexInputState<'a> {
    #[must_use]
    pub fn attribute_skip_masks(&self, index: usize) -> MaxwellThreeDRegister<u32> {
        self.registers.read(0x1120 + index as u32 * 4, Some)
    }
    #[must_use]
    pub fn attribute_skip_mask(&self, attribute: u8) -> u8 {
        self.attribute_skip_masks(attribute as usize / 8)
            .value()
            .map_or(0, |value| ((value >> ((attribute % 8) * 4)) & 15) as u8)
    }
    #[must_use]
    pub fn attributes(
        &self,
        index: usize,
    ) -> MaxwellThreeDRegister<MaxwellThreeDVertexAttributeFormat> {
        self.registers.read(0x1160 + index as u32 * 4, |raw| {
            MaxwellThreeDVertexAttributeFormat::parse(raw)
        })
    }
    #[must_use]
    pub fn streams(&self, index: usize) -> MaxwellThreeDVertexStreamState<'a> {
        MaxwellThreeDVertexStreamState {
            registers: self.registers,
            index,
        }
    }
    #[must_use]
    pub fn index(&self) -> MaxwellThreeDIndexBufferState<'a> {
        MaxwellThreeDIndexBufferState {
            registers: self.registers,
        }
    }
    #[must_use]
    pub fn assembly(&self) -> MaxwellThreeDVertexAssemblyState<'a> {
        MaxwellThreeDVertexAssemblyState {
            registers: self.registers,
        }
    }
    #[must_use]
    pub fn stream_substitute(&self) -> MaxwellThreeDVertexStreamSubstituteState<'a> {
        MaxwellThreeDVertexStreamSubstituteState {
            registers: self.registers,
        }
    }
    #[must_use]
    pub fn primitive(&self) -> MaxwellThreeDPrimitiveState<'a> {
        MaxwellThreeDPrimitiveState {
            registers: self.registers,
            sequence: self.sequence,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDVertexInputWrite {
    Begin {
        value: MaxwellThreeDBegin,
        source: MaxwellMethodSource,
    },
    End {
        value: bool,
        source: MaxwellMethodSource,
    },
}
