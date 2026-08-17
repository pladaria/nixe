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
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDVertexAssemblyState {
    attribute_defaults: MaxwellThreeDRegister<MaxwellThreeDAttributeDefaults>,
    vertex_id_uses_array_start: MaxwellThreeDRegister<MaxwellThreeDVertexIdUsesArrayStart>,
}

impl MaxwellThreeDVertexAssemblyState {
    #[must_use]
    pub const fn attribute_defaults(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDAttributeDefaults> {
        &self.attribute_defaults
    }

    #[must_use]
    pub const fn vertex_id_uses_array_start(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDVertexIdUsesArrayStart> {
        &self.vertex_id_uses_array_start
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
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDVertexStreamSubstituteState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDVertexStreamSubstituteState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
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
    pub(super) const fn parse(raw: u32) -> Option<Self> {
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDVertexStreamState {
    format: MaxwellThreeDRegister<MaxwellThreeDVertexStreamFormat>,
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    limit_upper: MaxwellThreeDRegister<u8>,
    limit_lower: MaxwellThreeDRegister<u32>,
    frequency: MaxwellThreeDRegister<u32>,
    instanced: MaxwellThreeDRegister<bool>,
}

impl MaxwellThreeDVertexStreamState {
    #[must_use]
    pub const fn format(&self) -> &MaxwellThreeDRegister<MaxwellThreeDVertexStreamFormat> {
        &self.format
    }
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }
    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }
    #[must_use]
    pub const fn limit_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.limit_upper
    }
    #[must_use]
    pub const fn limit_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.limit_lower
    }
    #[must_use]
    pub const fn frequency(&self) -> &MaxwellThreeDRegister<u32> {
        &self.frequency
    }
    #[must_use]
    pub const fn instanced(&self) -> &MaxwellThreeDRegister<bool> {
        &self.instanced
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }

    #[must_use]
    pub fn limit(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.limit_upper.value()?,
            *self.limit_lower.value()?,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDIndexBufferState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    limit_upper: MaxwellThreeDRegister<u8>,
    limit_lower: MaxwellThreeDRegister<u32>,
    element_size: MaxwellThreeDRegister<MaxwellThreeDIndexElementSize>,
    first: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDIndexBufferState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }
    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }
    #[must_use]
    pub const fn limit_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.limit_upper
    }
    #[must_use]
    pub const fn limit_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.limit_lower
    }
    #[must_use]
    pub const fn element_size(&self) -> &MaxwellThreeDRegister<MaxwellThreeDIndexElementSize> {
        &self.element_size
    }
    #[must_use]
    pub const fn first(&self) -> &MaxwellThreeDRegister<u32> {
        &self.first
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDPrimitiveState {
    circular_buffer_throttle: MaxwellThreeDRegister<MaxwellThreeDPrimitiveCircularBufferThrottle>,
    patch_size: MaxwellThreeDRegister<MaxwellThreeDPatchSize>,
    topology_override: MaxwellThreeDRegister<bool>,
    topology: MaxwellThreeDRegister<MaxwellThreeDPrimitiveTopology>,
    vertex_array_restart_enabled:
        MaxwellThreeDRegister<MaxwellThreeDVertexArrayPrimitiveRestartEnable>,
    restart_enabled: MaxwellThreeDRegister<bool>,
    restart_index: MaxwellThreeDRegister<u32>,
    vertex_array_start: MaxwellThreeDRegister<u32>,
    begin: MaxwellThreeDRegister<MaxwellThreeDBegin>,
    end: MaxwellThreeDRegister<bool>,
    open: bool,
}

impl MaxwellThreeDPrimitiveState {
    #[must_use]
    pub const fn circular_buffer_throttle(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDPrimitiveCircularBufferThrottle> {
        &self.circular_buffer_throttle
    }

    #[must_use]
    pub const fn patch_size(&self) -> &MaxwellThreeDRegister<MaxwellThreeDPatchSize> {
        &self.patch_size
    }

    #[must_use]
    pub const fn topology_override(&self) -> &MaxwellThreeDRegister<bool> {
        &self.topology_override
    }
    #[must_use]
    pub const fn topology(&self) -> &MaxwellThreeDRegister<MaxwellThreeDPrimitiveTopology> {
        &self.topology
    }
    #[must_use]
    pub const fn vertex_array_restart_enabled(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDVertexArrayPrimitiveRestartEnable> {
        &self.vertex_array_restart_enabled
    }
    #[must_use]
    pub const fn restart_enabled(&self) -> &MaxwellThreeDRegister<bool> {
        &self.restart_enabled
    }
    #[must_use]
    pub const fn restart_index(&self) -> &MaxwellThreeDRegister<u32> {
        &self.restart_index
    }
    #[must_use]
    pub const fn vertex_array_start(&self) -> &MaxwellThreeDRegister<u32> {
        &self.vertex_array_start
    }
    #[must_use]
    pub const fn begin(&self) -> &MaxwellThreeDRegister<MaxwellThreeDBegin> {
        &self.begin
    }

    /// Last validated `END` write, retained independently from the preceding
    /// `BEGIN` so diagnostics can reconstruct the primitive sequence.
    #[must_use]
    pub const fn end(&self) -> &MaxwellThreeDRegister<bool> {
        &self.end
    }

    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.open
    }

    #[must_use]
    pub const fn active_begin(&self) -> Option<&MaxwellThreeDBegin> {
        if self.open { self.begin.value() } else { None }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDBegin {
    topology: u8,
    preserve_primitive_id: bool,
    instance_id: u8,
    split_mode: u8,
}

impl MaxwellThreeDBegin {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x6d00_ffff != 0 || raw & 0xffff > 0x0e || ((raw >> 26) & 3) == 3 {
            return None;
        }
        Some(Self {
            topology: (raw & 0xff) as u8,
            preserve_primitive_id: raw & (1 << 24) != 0,
            instance_id: ((raw >> 26) & 3) as u8,
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
        self.instance_id
    }
    #[must_use]
    pub const fn split_mode(self) -> u8 {
        self.split_mode
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDVertexInputState {
    streams: Box<[MaxwellThreeDVertexStreamState; MAXWELL_VERTEX_STREAM_COUNT]>,
    attributes: Box<
        [MaxwellThreeDRegister<MaxwellThreeDVertexAttributeFormat>; MAXWELL_VERTEX_ATTRIBUTE_COUNT],
    >,
    index: MaxwellThreeDIndexBufferState,
    primitive: MaxwellThreeDPrimitiveState,
    assembly: MaxwellThreeDVertexAssemblyState,
    stream_substitute: MaxwellThreeDVertexStreamSubstituteState,
}

impl Default for MaxwellThreeDVertexInputState {
    fn default() -> Self {
        Self {
            streams: Box::new(std::array::from_fn(|_| {
                MaxwellThreeDVertexStreamState::default()
            })),
            attributes: Box::new(std::array::from_fn(|_| MaxwellThreeDRegister::default())),
            index: MaxwellThreeDIndexBufferState::default(),
            primitive: MaxwellThreeDPrimitiveState::default(),
            assembly: MaxwellThreeDVertexAssemblyState::default(),
            stream_substitute: MaxwellThreeDVertexStreamSubstituteState::default(),
        }
    }
}

impl MaxwellThreeDVertexInputState {
    #[must_use]
    pub fn streams(&self) -> &[MaxwellThreeDVertexStreamState; MAXWELL_VERTEX_STREAM_COUNT] {
        &self.streams
    }
    #[must_use]
    pub fn attributes(
        &self,
    ) -> &[MaxwellThreeDRegister<MaxwellThreeDVertexAttributeFormat>; MAXWELL_VERTEX_ATTRIBUTE_COUNT]
    {
        &self.attributes
    }
    #[must_use]
    pub const fn index(&self) -> &MaxwellThreeDIndexBufferState {
        &self.index
    }
    #[must_use]
    pub const fn primitive(&self) -> &MaxwellThreeDPrimitiveState {
        &self.primitive
    }
    #[must_use]
    pub const fn assembly(&self) -> &MaxwellThreeDVertexAssemblyState {
        &self.assembly
    }

    #[must_use]
    pub const fn stream_substitute(&self) -> &MaxwellThreeDVertexStreamSubstituteState {
        &self.stream_substitute
    }

    pub(super) fn append_pipeline_dependencies(&self, dependencies: &mut Vec<Option<u32>>) {
        for stream in self.streams.iter() {
            dependencies.push(stream.format.raw());
            dependencies.push(stream.frequency.raw());
            dependencies.push(stream.instanced.raw());
        }
        dependencies.extend(self.attributes.iter().map(MaxwellThreeDRegister::raw));
        dependencies.push(self.index.element_size.raw());
        // Circular-buffer throttling changes internal scheduling, not the
        // logical vertex-input or pipeline configuration.
        dependencies.push(self.primitive.topology_override.raw());
        dependencies.push(self.primitive.topology.raw());
        dependencies.push(self.primitive.restart_enabled.raw());
        dependencies.push(self.primitive.restart_index.raw());
        if self.primitive.is_open() {
            dependencies.push(self.primitive.begin.raw());
        }
        if self
            .primitive
            .active_begin()
            .is_some_and(|begin| begin.topology() == 14)
        {
            dependencies.push(self.primitive.patch_size.raw());
        }
        dependencies.push(self.assembly.attribute_defaults.raw());
        dependencies.push(self.assembly.vertex_id_uses_array_start.raw());
        // The public ABI identifies the substitute address but not the state
        // that makes it active. An unconsumed address does not alter a host
        // pipeline key.
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDVertexInputWrite) {
        let raw = write.raw();
        let source = write.source();
        match write {
            MaxwellThreeDVertexInputWrite::PrimitiveCircularBufferThrottle { value, .. } => {
                self.primitive.circular_buffer_throttle =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::PatchSize { value, .. } => {
                self.primitive.patch_size = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamFormat { stream, value, .. } => {
                self.streams[stream as usize].format =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamAddressUpper { stream, value, .. } => {
                self.streams[stream as usize].address_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamAddressLower { stream, value, .. } => {
                self.streams[stream as usize].address_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamLimitUpper { stream, value, .. } => {
                self.streams[stream as usize].limit_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamLimitLower { stream, value, .. } => {
                self.streams[stream as usize].limit_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamFrequency { stream, value, .. } => {
                self.streams[stream as usize].frequency =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamInstanced { stream, value, .. } => {
                self.streams[stream as usize].instanced =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::Attribute {
                attribute, value, ..
            } => {
                self.attributes[attribute as usize] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::IndexAddressUpper { value, .. } => {
                self.index.address_upper = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::IndexAddressLower { value, .. } => {
                self.index.address_lower = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::IndexLimitUpper { value, .. } => {
                self.index.limit_upper = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::IndexLimitLower { value, .. } => {
                self.index.limit_lower = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::IndexElementSize { value, .. } => {
                self.index.element_size = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::IndexFirst { value, .. } => {
                self.index.first = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::TopologyOverride { value, .. } => {
                self.primitive.topology_override =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::Topology { value, .. } => {
                self.primitive.topology = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::VertexArrayPrimitiveRestartEnable { value, .. } => {
                self.primitive.vertex_array_restart_enabled =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::PrimitiveRestartEnable { value, .. } => {
                self.primitive.restart_enabled =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::PrimitiveRestartIndex { value, .. } => {
                self.primitive.restart_index = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::VertexArrayStart { value, .. } => {
                self.primitive.vertex_array_start =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::Begin { value, .. } => {
                self.primitive.begin = MaxwellThreeDRegister::programmed(raw, value, source);
                self.primitive.open = true;
            }
            MaxwellThreeDVertexInputWrite::End { value, .. } => {
                self.primitive.end = MaxwellThreeDRegister::programmed(raw, value, source);
                self.primitive.open = false;
            }
            MaxwellThreeDVertexInputWrite::AttributeDefaults { value, .. } => {
                self.assembly.attribute_defaults =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::VertexIdUsesArrayStart { value, .. } => {
                self.assembly.vertex_id_uses_array_start =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamSubstituteAddressUpper { value, .. } => {
                self.stream_substitute.address_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDVertexInputWrite::StreamSubstituteAddressLower { value, .. } => {
                self.stream_substitute.address_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDVertexInputWrite {
    PrimitiveCircularBufferThrottle {
        value: MaxwellThreeDPrimitiveCircularBufferThrottle,
        source: MaxwellMethodSource,
    },
    PatchSize {
        value: MaxwellThreeDPatchSize,
        source: MaxwellMethodSource,
    },
    StreamFormat {
        stream: u8,
        value: MaxwellThreeDVertexStreamFormat,
        source: MaxwellMethodSource,
    },
    StreamAddressUpper {
        stream: u8,
        value: u8,
        source: MaxwellMethodSource,
    },
    StreamAddressLower {
        stream: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    StreamLimitUpper {
        stream: u8,
        value: u8,
        source: MaxwellMethodSource,
    },
    StreamLimitLower {
        stream: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    StreamFrequency {
        stream: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    StreamInstanced {
        stream: u8,
        value: bool,
        source: MaxwellMethodSource,
    },
    Attribute {
        attribute: u8,
        value: MaxwellThreeDVertexAttributeFormat,
        source: MaxwellMethodSource,
    },
    IndexAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    IndexAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    IndexLimitUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    IndexLimitLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    IndexElementSize {
        value: MaxwellThreeDIndexElementSize,
        source: MaxwellMethodSource,
    },
    IndexFirst {
        value: u32,
        source: MaxwellMethodSource,
    },
    TopologyOverride {
        value: bool,
        source: MaxwellMethodSource,
    },
    Topology {
        value: MaxwellThreeDPrimitiveTopology,
        source: MaxwellMethodSource,
    },
    VertexArrayPrimitiveRestartEnable {
        value: MaxwellThreeDVertexArrayPrimitiveRestartEnable,
        source: MaxwellMethodSource,
    },
    PrimitiveRestartEnable {
        value: bool,
        source: MaxwellMethodSource,
    },
    PrimitiveRestartIndex {
        value: u32,
        source: MaxwellMethodSource,
    },
    VertexArrayStart {
        value: u32,
        source: MaxwellMethodSource,
    },
    Begin {
        value: MaxwellThreeDBegin,
        source: MaxwellMethodSource,
    },
    End {
        value: bool,
        source: MaxwellMethodSource,
    },
    AttributeDefaults {
        value: MaxwellThreeDAttributeDefaults,
        source: MaxwellMethodSource,
    },
    VertexIdUsesArrayStart {
        value: MaxwellThreeDVertexIdUsesArrayStart,
        source: MaxwellMethodSource,
    },
    StreamSubstituteAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    StreamSubstituteAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
}

impl MaxwellThreeDVertexInputWrite {
    pub(super) const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::PrimitiveCircularBufferThrottle { source, .. }
            | Self::PatchSize { source, .. }
            | Self::StreamFormat { source, .. }
            | Self::StreamAddressUpper { source, .. }
            | Self::StreamAddressLower { source, .. }
            | Self::StreamLimitUpper { source, .. }
            | Self::StreamLimitLower { source, .. }
            | Self::StreamFrequency { source, .. }
            | Self::StreamInstanced { source, .. }
            | Self::Attribute { source, .. }
            | Self::IndexAddressUpper { source, .. }
            | Self::IndexAddressLower { source, .. }
            | Self::IndexLimitUpper { source, .. }
            | Self::IndexLimitLower { source, .. }
            | Self::IndexElementSize { source, .. }
            | Self::IndexFirst { source, .. }
            | Self::TopologyOverride { source, .. }
            | Self::Topology { source, .. }
            | Self::VertexArrayPrimitiveRestartEnable { source, .. }
            | Self::PrimitiveRestartEnable { source, .. }
            | Self::PrimitiveRestartIndex { source, .. }
            | Self::VertexArrayStart { source, .. }
            | Self::Begin { source, .. }
            | Self::End { source, .. }
            | Self::AttributeDefaults { source, .. }
            | Self::VertexIdUsesArrayStart { source, .. }
            | Self::StreamSubstituteAddressUpper { source, .. }
            | Self::StreamSubstituteAddressLower { source, .. } => source,
        }
    }
    pub(super) const fn raw(self) -> u32 {
        self.source().argument()
    }
}
