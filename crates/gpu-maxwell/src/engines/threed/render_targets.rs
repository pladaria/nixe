//! Typed render-target, depth/stencil-target, and clear frontend state.

use crate::MaxwellMethodSource;

use super::state::MaxwellThreeDRegister;

pub const MAXWELL_COLOR_TARGET_COUNT: usize = 8;

/// Readiness of a frontend attachment description. A disabled attachment is
/// intentionally different from one never programmed or unsupported by the
/// selected immutable GPU profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDAttachmentReadiness {
    Unprogrammed,
    Disabled,
    Incomplete,
    Ready,
    Contradictory,
    ProfileUnavailable,
}

/// Guest image memory organization. This is not a host texture layout.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDImageLayout {
    BlockLinear {
        block_height_log2: u8,
        block_depth_log2: u8,
    },
    PitchLinear,
}

/// Meaning of the third color-target dimension.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDImageKind {
    Array,
    ThreeDimensional,
}

/// A verified Maxwell color-target format encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDColorTargetFormat {
    Disabled,
    Color(u8),
}

impl MaxwellThreeDColorTargetFormat {
    pub(super) fn parse(raw: u32) -> Option<Self> {
        let value = u8::try_from(raw).ok()?;
        if value == 0 {
            return Some(Self::Disabled);
        }
        // Exact values published by the pinned NVIDIA clb197 header. Holes
        // are not accepted as undocumented formats.
        const FORMATS: &[u8] = &[
            0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5,
            0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd5,
            0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0, 0xe3, 0xe4, 0xe5,
            0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf2, 0xf3,
            0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
        ];
        FORMATS.contains(&value).then_some(Self::Color(value))
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Disabled => 0,
            Self::Color(value) => value,
        }
    }
}

/// A verified Maxwell depth/stencil-target format encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDDepthStencilFormat {
    ZFloat32 = 0x0a,
    Z16 = 0x13,
    Z24Stencil8 = 0x14,
    X8Z24 = 0x15,
    Stencil8Z24 = 0x16,
    Stencil8 = 0x17,
    V8Z24 = 0x18,
    ZFloat32X24Stencil8 = 0x19,
    X8Z24X16V8Stencil8 = 0x1d,
    ZFloat32X16V8X8 = 0x1e,
    ZFloat32X16V8Stencil8 = 0x1f,
}

/// Depth-compression selector programmed by `SET_Z_COMPRESSION`.
///
/// This selector is not a memory kind or image layout. The public class
/// header establishes only these two values; representation and coherency
/// semantics remain an execution concern.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDZCompressionMode {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDZCompressionMode {
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

impl MaxwellThreeDDepthStencilFormat {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0x0a => Some(Self::ZFloat32),
            0x13 => Some(Self::Z16),
            0x14 => Some(Self::Z24Stencil8),
            0x15 => Some(Self::X8Z24),
            0x16 => Some(Self::Stencil8Z24),
            0x17 => Some(Self::Stencil8),
            0x18 => Some(Self::V8Z24),
            0x19 => Some(Self::ZFloat32X24Stencil8),
            0x1d => Some(Self::X8Z24X16V8Stencil8),
            0x1e => Some(Self::ZFloat32X16V8X8),
            0x1f => Some(Self::ZFloat32X16V8Stencil8),
            _ => None,
        }
    }
}

/// Source-preserving scalar whose interpretation remains frontend-only.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDRawValue(u32);

impl MaxwellThreeDRawValue {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Color-compression selector for one color target.
///
/// This type remains distinct from Z compression even though the currently
/// verified values share an encoding. It identifies neither a memory kind nor
/// an image layout and cannot create an attachment by itself.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDColorCompressionMode {
    Disabled = 0,
    Enabled = 1,
}

/// Ordered color-target routing selected by `SET_CT_SELECT`.
///
/// The complete register payload is retained, including selectors beyond
/// `target_count`; they are inactive for the current draw but have no
/// documented reset semantics and must not be discarded on write.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorTargetSelection {
    target_count: u8,
    targets: [u8; MAXWELL_COLOR_TARGET_COUNT],
}

impl MaxwellThreeDColorTargetSelection {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x0fff_ffff != 0 {
            return None;
        }
        let target_count = (raw & 0xf) as u8;
        if target_count as usize > MAXWELL_COLOR_TARGET_COUNT {
            return None;
        }
        let mut targets = [0; MAXWELL_COLOR_TARGET_COUNT];
        let mut index = 0;
        while index < MAXWELL_COLOR_TARGET_COUNT {
            targets[index] = ((raw >> (4 + index * 3)) & 7) as u8;
            index += 1;
        }
        Some(Self {
            target_count,
            targets,
        })
    }

    #[must_use]
    pub const fn target_count(self) -> u8 {
        self.target_count
    }

    #[must_use]
    pub const fn targets(self) -> [u8; MAXWELL_COLOR_TARGET_COUNT] {
        self.targets
    }

    #[must_use]
    pub fn active_targets(&self) -> &[u8] {
        &self.targets[..self.target_count as usize]
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        let mut raw = self.target_count as u32;
        let mut index = 0;
        while index < MAXWELL_COLOR_TARGET_COUNT {
            raw |= (self.targets[index] as u32) << (4 + index * 3);
            index += 1;
        }
        raw
    }
}

impl MaxwellThreeDColorCompressionMode {
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

/// One unresolved color attachment. Every member remains explicitly unset
/// until the guest programs it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDColorTargetState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    width: MaxwellThreeDRegister<u32>,
    height: MaxwellThreeDRegister<u32>,
    format: MaxwellThreeDRegister<MaxwellThreeDColorTargetFormat>,
    layout: MaxwellThreeDRegister<MaxwellThreeDImageLayout>,
    kind: MaxwellThreeDRegister<MaxwellThreeDImageKind>,
    third_dimension: MaxwellThreeDRegister<u32>,
    array_pitch: MaxwellThreeDRegister<u32>,
    layer: MaxwellThreeDRegister<u16>,
    compression: MaxwellThreeDRegister<MaxwellThreeDColorCompressionMode>,
}

impl MaxwellThreeDColorTargetState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }
    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }
    #[must_use]
    pub const fn width(&self) -> &MaxwellThreeDRegister<u32> {
        &self.width
    }
    #[must_use]
    pub const fn height(&self) -> &MaxwellThreeDRegister<u32> {
        &self.height
    }
    #[must_use]
    pub const fn format(&self) -> &MaxwellThreeDRegister<MaxwellThreeDColorTargetFormat> {
        &self.format
    }
    #[must_use]
    pub const fn layout(&self) -> &MaxwellThreeDRegister<MaxwellThreeDImageLayout> {
        &self.layout
    }
    #[must_use]
    pub const fn kind(&self) -> &MaxwellThreeDRegister<MaxwellThreeDImageKind> {
        &self.kind
    }
    #[must_use]
    pub const fn third_dimension(&self) -> &MaxwellThreeDRegister<u32> {
        &self.third_dimension
    }
    #[must_use]
    pub const fn array_pitch(&self) -> &MaxwellThreeDRegister<u32> {
        &self.array_pitch
    }
    #[must_use]
    pub const fn layer(&self) -> &MaxwellThreeDRegister<u16> {
        &self.layer
    }
    #[must_use]
    pub const fn compression(&self) -> &MaxwellThreeDRegister<MaxwellThreeDColorCompressionMode> {
        &self.compression
    }

    /// Classifies only attachment-description completeness and relationships.
    /// The compression selector is intentionally excluded: it configures work
    /// on an attachment but neither describes nor binds one. This function
    /// resolves no address and creates no image view.
    #[must_use]
    pub fn readiness(&self, profile_supports_format: bool) -> MaxwellThreeDAttachmentReadiness {
        let Some(format) = self.format.value().copied() else {
            return MaxwellThreeDAttachmentReadiness::Unprogrammed;
        };
        if format == MaxwellThreeDColorTargetFormat::Disabled {
            return MaxwellThreeDAttachmentReadiness::Disabled;
        }
        if !profile_supports_format {
            return MaxwellThreeDAttachmentReadiness::ProfileUnavailable;
        }
        let required = [
            self.address_upper.raw(),
            self.address_lower.raw(),
            self.width.raw(),
            self.height.raw(),
            self.layout.raw(),
            self.kind.raw(),
            self.third_dimension.raw(),
        ];
        if required.iter().any(Option::is_none) {
            return MaxwellThreeDAttachmentReadiness::Incomplete;
        }
        if self.width.value() == Some(&0)
            || self.height.value() == Some(&0)
            || self.third_dimension.value() == Some(&0)
        {
            return MaxwellThreeDAttachmentReadiness::Contradictory;
        }
        if self.kind.value() == Some(&MaxwellThreeDImageKind::ThreeDimensional)
            && self.layer.value().is_some_and(|layer| *layer != 0)
        {
            return MaxwellThreeDAttachmentReadiness::Contradictory;
        }
        MaxwellThreeDAttachmentReadiness::Ready
    }
}

/// One unresolved depth/stencil attachment.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDDepthStencilTargetState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    format: MaxwellThreeDRegister<MaxwellThreeDDepthStencilFormat>,
    layout: MaxwellThreeDRegister<MaxwellThreeDImageLayout>,
    width: MaxwellThreeDRegister<u32>,
    height: MaxwellThreeDRegister<u32>,
    third_dimension: MaxwellThreeDRegister<u16>,
    kind: MaxwellThreeDRegister<MaxwellThreeDImageKind>,
    array_pitch: MaxwellThreeDRegister<u32>,
    compression: MaxwellThreeDRegister<MaxwellThreeDZCompressionMode>,
}

impl MaxwellThreeDDepthStencilTargetState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }
    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }
    #[must_use]
    pub const fn format(&self) -> &MaxwellThreeDRegister<MaxwellThreeDDepthStencilFormat> {
        &self.format
    }
    #[must_use]
    pub const fn layout(&self) -> &MaxwellThreeDRegister<MaxwellThreeDImageLayout> {
        &self.layout
    }
    #[must_use]
    pub const fn width(&self) -> &MaxwellThreeDRegister<u32> {
        &self.width
    }
    #[must_use]
    pub const fn height(&self) -> &MaxwellThreeDRegister<u32> {
        &self.height
    }
    #[must_use]
    pub const fn third_dimension(&self) -> &MaxwellThreeDRegister<u16> {
        &self.third_dimension
    }
    #[must_use]
    pub const fn kind(&self) -> &MaxwellThreeDRegister<MaxwellThreeDImageKind> {
        &self.kind
    }
    #[must_use]
    pub const fn array_pitch(&self) -> &MaxwellThreeDRegister<u32> {
        &self.array_pitch
    }
    #[must_use]
    pub const fn compression(&self) -> &MaxwellThreeDRegister<MaxwellThreeDZCompressionMode> {
        &self.compression
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDRectangle {
    pub min: u16,
    pub max: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDClearSurface {
    depth: bool,
    stencil: bool,
    color_mask: u8,
    color_target: u8,
    array_layer: u16,
}

impl MaxwellThreeDClearSurface {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        let color_target = ((raw >> 6) & 0xf) as u8;
        if raw & !0x03ff_ffff != 0 || color_target as usize >= MAXWELL_COLOR_TARGET_COUNT {
            return None;
        }
        Some(Self {
            depth: raw & 1 != 0,
            stencil: raw & 2 != 0,
            color_mask: ((raw >> 2) & 0xf) as u8,
            color_target,
            array_layer: ((raw >> 10) & 0xffff) as u16,
        })
    }

    #[must_use]
    pub const fn depth(self) -> bool {
        self.depth
    }
    #[must_use]
    pub const fn stencil(self) -> bool {
        self.stencil
    }
    #[must_use]
    pub const fn color_mask(self) -> u8 {
        self.color_mask
    }
    #[must_use]
    pub const fn color_target(self) -> u8 {
        self.color_target
    }
    #[must_use]
    pub const fn array_layer(self) -> u16 {
        self.array_layer
    }
}

/// Values consumed by a later clear operation. `last_surface` records the
/// trigger but does not execute or lower it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDClearState {
    color: [MaxwellThreeDRegister<MaxwellThreeDRawValue>; 4],
    depth: MaxwellThreeDRegister<MaxwellThreeDRawValue>,
    stencil: MaxwellThreeDRegister<u8>,
    horizontal: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    vertical: MaxwellThreeDRegister<MaxwellThreeDRectangle>,
    last_surface: MaxwellThreeDRegister<MaxwellThreeDClearSurface>,
}

impl MaxwellThreeDClearState {
    #[must_use]
    pub const fn color(&self) -> &[MaxwellThreeDRegister<MaxwellThreeDRawValue>; 4] {
        &self.color
    }
    #[must_use]
    pub const fn depth(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRawValue> {
        &self.depth
    }
    #[must_use]
    pub const fn stencil(&self) -> &MaxwellThreeDRegister<u8> {
        &self.stencil
    }
    #[must_use]
    pub const fn horizontal(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.horizontal
    }
    #[must_use]
    pub const fn vertical(&self) -> &MaxwellThreeDRegister<MaxwellThreeDRectangle> {
        &self.vertical
    }
    #[must_use]
    pub const fn last_surface(&self) -> &MaxwellThreeDRegister<MaxwellThreeDClearSurface> {
        &self.last_surface
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDRenderTargetState {
    color: [MaxwellThreeDColorTargetState; MAXWELL_COLOR_TARGET_COUNT],
    color_target_selection: MaxwellThreeDRegister<MaxwellThreeDColorTargetSelection>,
    depth_stencil: MaxwellThreeDDepthStencilTargetState,
    clear: MaxwellThreeDClearState,
}

impl MaxwellThreeDRenderTargetState {
    #[must_use]
    pub const fn color(&self) -> &[MaxwellThreeDColorTargetState; MAXWELL_COLOR_TARGET_COUNT] {
        &self.color
    }
    #[must_use]
    pub const fn color_target_selection(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorTargetSelection> {
        &self.color_target_selection
    }
    #[must_use]
    pub const fn depth_stencil(&self) -> &MaxwellThreeDDepthStencilTargetState {
        &self.depth_stencil
    }
    #[must_use]
    pub const fn clear(&self) -> &MaxwellThreeDClearState {
        &self.clear
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDRenderTargetWrite) {
        let raw = write.raw();
        let source = write.source();
        match write {
            MaxwellThreeDRenderTargetWrite::ColorAddressUpper { target, value, .. } => {
                self.color[target as usize].address_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorAddressLower { target, value, .. } => {
                self.color[target as usize].address_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorWidth { target, value, .. } => {
                self.color[target as usize].width =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorHeight { target, value, .. } => {
                self.color[target as usize].height =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorFormat { target, value, .. } => {
                self.color[target as usize].format =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorLayout {
                target,
                layout,
                kind,
                ..
            } => {
                self.color[target as usize].layout =
                    MaxwellThreeDRegister::programmed(raw, layout, source);
                self.color[target as usize].kind =
                    MaxwellThreeDRegister::programmed(raw, kind, source);
            }
            MaxwellThreeDRenderTargetWrite::ColorThirdDimension { target, value, .. } => {
                self.color[target as usize].third_dimension =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorArrayPitch { target, value, .. } => {
                self.color[target as usize].array_pitch =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorLayer { target, value, .. } => {
                self.color[target as usize].layer =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorCompression { target, value, .. } => {
                self.color[target as usize].compression =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ColorTargetSelection { value, .. } => {
                self.color_target_selection = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthAddressUpper { value, .. } => {
                self.depth_stencil.address_upper =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthAddressLower { value, .. } => {
                self.depth_stencil.address_lower =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthFormat { value, .. } => {
                self.depth_stencil.format = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthLayout { value, .. } => {
                self.depth_stencil.layout = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthWidth { value, .. } => {
                self.depth_stencil.width = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthHeight { value, .. } => {
                self.depth_stencil.height = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthThirdDimension { value, kind, .. } => {
                self.depth_stencil.third_dimension =
                    MaxwellThreeDRegister::programmed(raw, value, source);
                self.depth_stencil.kind = MaxwellThreeDRegister::programmed(raw, kind, source);
            }
            MaxwellThreeDRenderTargetWrite::DepthArrayPitch { value, .. } => {
                self.depth_stencil.array_pitch =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::DepthCompression { value, .. } => {
                self.depth_stencil.compression =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ClearColor {
                component, value, ..
            } => {
                self.clear.color[component as usize] =
                    MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ClearDepth { value, .. } => {
                self.clear.depth = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ClearStencil { value, .. } => {
                self.clear.stencil = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ClearHorizontal { value, .. } => {
                self.clear.horizontal = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ClearVertical { value, .. } => {
                self.clear.vertical = MaxwellThreeDRegister::programmed(raw, value, source)
            }
            MaxwellThreeDRenderTargetWrite::ClearSurface { value, .. } => {
                self.clear.last_surface = MaxwellThreeDRegister::programmed(raw, value, source)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDRenderTargetWrite {
    ColorAddressUpper {
        target: u8,
        value: u8,
        source: MaxwellMethodSource,
    },
    ColorAddressLower {
        target: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    ColorWidth {
        target: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    ColorHeight {
        target: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    ColorFormat {
        target: u8,
        value: MaxwellThreeDColorTargetFormat,
        source: MaxwellMethodSource,
    },
    ColorLayout {
        target: u8,
        layout: MaxwellThreeDImageLayout,
        kind: MaxwellThreeDImageKind,
        source: MaxwellMethodSource,
    },
    ColorThirdDimension {
        target: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    ColorArrayPitch {
        target: u8,
        value: u32,
        source: MaxwellMethodSource,
    },
    ColorLayer {
        target: u8,
        value: u16,
        source: MaxwellMethodSource,
    },
    ColorCompression {
        target: u8,
        value: MaxwellThreeDColorCompressionMode,
        source: MaxwellMethodSource,
    },
    ColorTargetSelection {
        value: MaxwellThreeDColorTargetSelection,
        source: MaxwellMethodSource,
    },
    DepthAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    DepthAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    DepthFormat {
        value: MaxwellThreeDDepthStencilFormat,
        source: MaxwellMethodSource,
    },
    DepthLayout {
        value: MaxwellThreeDImageLayout,
        source: MaxwellMethodSource,
    },
    DepthWidth {
        value: u32,
        source: MaxwellMethodSource,
    },
    DepthHeight {
        value: u32,
        source: MaxwellMethodSource,
    },
    DepthThirdDimension {
        value: u16,
        kind: MaxwellThreeDImageKind,
        source: MaxwellMethodSource,
    },
    DepthArrayPitch {
        value: u32,
        source: MaxwellMethodSource,
    },
    DepthCompression {
        value: MaxwellThreeDZCompressionMode,
        source: MaxwellMethodSource,
    },
    ClearColor {
        component: u8,
        value: MaxwellThreeDRawValue,
        source: MaxwellMethodSource,
    },
    ClearDepth {
        value: MaxwellThreeDRawValue,
        source: MaxwellMethodSource,
    },
    ClearStencil {
        value: u8,
        source: MaxwellMethodSource,
    },
    ClearHorizontal {
        value: MaxwellThreeDRectangle,
        source: MaxwellMethodSource,
    },
    ClearVertical {
        value: MaxwellThreeDRectangle,
        source: MaxwellMethodSource,
    },
    ClearSurface {
        value: MaxwellThreeDClearSurface,
        source: MaxwellMethodSource,
    },
}

impl MaxwellThreeDRenderTargetWrite {
    pub(super) const fn source(self) -> MaxwellMethodSource {
        match self {
            Self::ColorAddressUpper { source, .. }
            | Self::ColorAddressLower { source, .. }
            | Self::ColorWidth { source, .. }
            | Self::ColorHeight { source, .. }
            | Self::ColorFormat { source, .. }
            | Self::ColorLayout { source, .. }
            | Self::ColorThirdDimension { source, .. }
            | Self::ColorArrayPitch { source, .. }
            | Self::ColorLayer { source, .. }
            | Self::ColorCompression { source, .. }
            | Self::ColorTargetSelection { source, .. }
            | Self::DepthAddressUpper { source, .. }
            | Self::DepthAddressLower { source, .. }
            | Self::DepthFormat { source, .. }
            | Self::DepthLayout { source, .. }
            | Self::DepthWidth { source, .. }
            | Self::DepthHeight { source, .. }
            | Self::DepthThirdDimension { source, .. }
            | Self::DepthArrayPitch { source, .. }
            | Self::DepthCompression { source, .. }
            | Self::ClearColor { source, .. }
            | Self::ClearDepth { source, .. }
            | Self::ClearStencil { source, .. }
            | Self::ClearHorizontal { source, .. }
            | Self::ClearVertical { source, .. }
            | Self::ClearSurface { source, .. } => source,
        }
    }
    pub(super) const fn raw(self) -> u32 {
        self.source().argument()
    }
}
