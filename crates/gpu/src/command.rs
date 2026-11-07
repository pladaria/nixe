//! Immutable backend-independent GPU operations and submission ordering.

use std::fmt::{Display, Formatter};

/// Origin of an image operation in texels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageOrigin {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Color conversion/scaling between texel rectangles. Coordinates map the
/// destination pixel centers to the source box; clamping occurs at its edges.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlitFilter {
    Point,
    Linear,
}

/// Neutral clear payload retaining the exact requested value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
    Buffer(u32),
    Color([f32; 4]),
    Depth(f32),
    Stencil(u8),
    DepthStencil { depth: f32, stencil: u8 },
}

impl ClearValue {}

/// Primitive assembly mode expected by a graphics draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PrimitiveTopology {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    /// Independent groups of four perimeter-ordered vertices.
    Quads,
    QuadStrip,
    Patches,
}

/// Comparison applied by the legacy fixed-function alpha test.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AlphaCompareOperation {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Optional fixed-function alpha test performed after fragment evaluation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AlphaTest {
    pub comparison: AlphaCompareOperation,
    pub reference_bits: u32,
}

/// Window-space fragment bounds. Empty rectangles still execute vertex shaders.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ScissorRect {
    /// Intersects with framebuffer bounds without changing viewport coordinates.
    #[must_use]
    pub fn clipped(self, width: u32, height: u32) -> Self {
        let x = self.x.min(width);
        let y = self.y.min(height);
        let end_x = self.x.saturating_add(self.width).min(width);
        let end_y = self.y.saturating_add(self.height).min(height);
        Self {
            x,
            y,
            width: end_x.saturating_sub(x),
            height: end_y.saturating_sub(y),
        }
    }
}

/// Affine conversion of the final shader X/Y output into clip coordinates.
/// It does not change shader-visible output reads or the Z/W components.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VertexPositionTransform {
    scale: [u32; 2],
    offset: [u32; 2],
}

impl VertexPositionTransform {
    /// Converts framebuffer pixel coordinates to clip X/Y for a positive-axis viewport.
    pub fn framebuffer(width: u32, height: u32) -> Result<Self, CommandDescriptionError> {
        if width == 0 || height == 0 {
            return Err(CommandDescriptionError::EmptyViewport);
        }
        Ok(Self {
            scale: [2.0 / width as f32, 2.0 / height as f32].map(f32::to_bits),
            offset: [-1.0_f32; 2].map(f32::to_bits),
        })
    }

    #[must_use]
    pub fn scale(self) -> [f32; 2] {
        self.scale.map(f32::from_bits)
    }

    #[must_use]
    pub fn offset(self) -> [f32; 2] {
        self.offset.map(f32::from_bits)
    }
}

/// Window-depth reconstruction for hosts whose viewport depth is limited to [0,1].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FragmentDepthTransform([u32; 4]);

impl FragmentDepthTransform {
    pub fn new(
        scale: f32,
        offset: f32,
        minimum: f32,
        maximum: f32,
    ) -> Result<Self, CommandDescriptionError> {
        if ![scale, offset, minimum, maximum]
            .into_iter()
            .all(f32::is_finite)
            || minimum < 0.0
            || maximum > 1.0
            || minimum > maximum
        {
            return Err(CommandDescriptionError::InvalidFragmentDepthTransform);
        }
        Ok(Self([scale, offset, minimum, maximum].map(f32::to_bits)))
    }

    #[must_use]
    pub fn coefficients(self) -> [f32; 4] {
        self.0.map(f32::from_bits)
    }
}

/// One fully specified direct draw dependency set.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

/// Physical width of one component in a scaled vertex attribute.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexComponentWidth {
    Bits8,
    Bits16,
    Bits32,
}

impl VertexComponentWidth {
    #[must_use]
    pub const fn bytes(self) -> u64 {
        match self {
            Self::Bits8 => 1,
            Self::Bits16 => 2,
            Self::Bits32 => 4,
        }
    }
}

/// Number of consecutive components in one scaled vertex attribute.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexComponentCount {
    One,
    Two,
    Three,
    Four,
}

impl VertexComponentCount {
    #[must_use]
    pub const fn get(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
        }
    }
}

/// Exact host-independent storage format of one vertex attribute.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexFormat {
    Uint8,
    Uint8x2,
    Uint8x4,
    Sint8x2,
    Sint8x4,
    Unorm8x2,
    Unorm8x4,
    Snorm8x2,
    Snorm8x4,
    Uint16x2,
    Uint16x4,
    Sint16x2,
    Sint16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Unorm10_10_10_2,
    Snorm10_10_10_2,
    /// Unsigned integer storage converted to equal-magnitude floating point.
    Uscaled {
        width: VertexComponentWidth,
        components: VertexComponentCount,
    },
    /// Signed integer storage converted to equal-magnitude floating point.
    Sscaled {
        width: VertexComponentWidth,
        components: VertexComponentCount,
    },
}

impl VertexFormat {
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::Uint8 => 1,
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2
            | Self::Float32
            | Self::Uint32
            | Self::Sint32
            | Self::Unorm10_10_10_2
            | Self::Snorm10_10_10_2 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4
            | Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
            Self::Uscaled { width, components } | Self::Sscaled { width, components } => {
                width.bytes() * components.get() as u64
            }
        }
    }

    #[must_use]
    pub const fn scaled_layout(self) -> Option<(bool, VertexComponentWidth, VertexComponentCount)> {
        match self {
            Self::Uscaled { width, components } => Some((false, width, components)),
            Self::Sscaled { width, components } => Some((true, width, components)),
            _ => None,
        }
    }

    /// Returns the signedness, storage width, and component count for integer
    /// formats whose shader-visible vector must be reconstructed explicitly.
    #[must_use]
    pub const fn integer_layout(
        self,
    ) -> Option<(bool, VertexComponentWidth, VertexComponentCount)> {
        use VertexComponentCount::{Four, One, Three, Two};
        use VertexComponentWidth::{Bits8, Bits16, Bits32};
        match self {
            Self::Uint8 => Some((false, Bits8, One)),
            Self::Uint8x2 => Some((false, Bits8, Two)),
            Self::Uint8x4 => Some((false, Bits8, Four)),
            Self::Sint8x2 => Some((true, Bits8, Two)),
            Self::Sint8x4 => Some((true, Bits8, Four)),
            Self::Uint16x2 => Some((false, Bits16, Two)),
            Self::Uint16x4 => Some((false, Bits16, Four)),
            Self::Sint16x2 => Some((true, Bits16, Two)),
            Self::Sint16x4 => Some((true, Bits16, Four)),
            Self::Uint32 => Some((false, Bits32, One)),
            Self::Uint32x2 => Some((false, Bits32, Two)),
            Self::Uint32x3 => Some((false, Bits32, Three)),
            Self::Uint32x4 => Some((false, Bits32, Four)),
            Self::Sint32 => Some((true, Bits32, One)),
            Self::Sint32x2 => Some((true, Bits32, Two)),
            Self::Sint32x3 => Some((true, Bits32, Three)),
            Self::Sint32x4 => Some((true, Bits32, Four)),
            _ => None,
        }
    }

    #[must_use]
    pub const fn requires_vertex_pulling(self) -> bool {
        matches!(self, Self::Snorm10_10_10_2)
            || self.scaled_layout().is_some()
            || self.integer_layout().is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

/// Host-independent depth comparison selected for a graphics draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DepthCompareOperation {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Backend-independent device cache-maintenance operation.
///
/// This is an ordering command, not a model of any host GPU's physical cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheMaintenanceOperation {
    /// Discard device-side cached reads so later reads observe canonical memory.
    InvalidateDeviceReadCaches,
    /// Discard cached texture fetches so later sampling observes current data.
    InvalidateTextureReadCaches,
    /// Discard cached texture descriptors so later fetches observe current headers.
    InvalidateTextureHeaderCaches,
    /// Discard cached sampler descriptors so later sampling observes current state.
    InvalidateSamplerCaches,
    /// Discard selected shader-side caches without implying an idle wait.
    InvalidateShaderCaches {
        instruction: bool,
        global_data: bool,
        constant: bool,
    },
    /// Make dirty device writes visible before later work on the submission.
    FlushDirtyDeviceWrites,
}

/// Whether existing attachment contents participate in a render pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttachmentLoad {
    Load,
    Clear(ClearValue),
    Discard,
}

/// Whether attachment contents remain defined after a render pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttachmentStore {
    Store,
    Discard,
}

/// Failure to construct an immutable neutral operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandDescriptionError {
    InvalidImageResolve,
    InvalidImageBlit,
    TessellationTopologyMismatch,
    CopySizeMismatch,
    ClearValueMismatch,
    NonFiniteClearValue,
    InvalidFragmentDepthTransform,
    EmptyViewport,
    EmptyDraw,
    IndexBufferMismatch,
    EmptyVertexStride,
    EmptyVertexAttributes,
    VertexAttributeOverflow,
    VertexAttributeExceedsStride,
    DuplicateVertexShaderLocation,
    EmptyDispatch,
    AttachmentKindMismatch,
    AttachmentCountMismatch,
    AttachmentDescriptionMismatch,
    DuplicateAttachment,
    EmptySubmission,
    TooManySubmissionSegments,
    SelfDependency,
    DuplicateSubmissionDependency,
}

impl Display for CommandDescriptionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidImageBlit => formatter.write_str("color blit requires distinct single-layer 2D texel rectangles"),
            Self::InvalidImageResolve => formatter.write_str("color resolve requires distinct images, matching regions and a multisampled source"),
            Self::TessellationTopologyMismatch => formatter.write_str("patch draw requires tessellation state and nonzero input control points; other topologies forbid it"),
            Self::CopySizeMismatch => {
                formatter.write_str("buffer copy source and destination sizes differ")
            }
            Self::ClearValueMismatch => {
                formatter.write_str("clear value does not match its resource kind")
            }
            Self::NonFiniteClearValue => formatter.write_str("clear value is not finite"),
            Self::InvalidFragmentDepthTransform => formatter.write_str("fragment depth transform requires finite coefficients and ordered bounds inside [0,1]"),

            Self::EmptyViewport => formatter.write_str("viewport transform has an empty axis"),
            Self::EmptyDraw => formatter.write_str("draw has no vertices, indices, or instances"),
            Self::IndexBufferMismatch => {
                formatter.write_str("draw arguments and index-buffer presence disagree")
            }
            Self::EmptyVertexStride => formatter.write_str("vertex buffer stride is zero"),
            Self::EmptyVertexAttributes => {
                formatter.write_str("vertex buffer layout has no attributes")
            }
            Self::VertexAttributeOverflow => {
                formatter.write_str("vertex attribute byte range overflows")
            }
            Self::VertexAttributeExceedsStride => {
                formatter.write_str("vertex attribute exceeds its buffer stride")
            }
            Self::DuplicateVertexShaderLocation => {
                formatter.write_str("draw has duplicate vertex shader locations")
            }
            Self::EmptyDispatch => formatter.write_str("dispatch has a zero workgroup dimension"),
            Self::AttachmentKindMismatch => {
                formatter.write_str("render attachment kind and format disagree")
            }
            Self::AttachmentCountMismatch => formatter
                .write_str("render attachment count does not match the render-pass description"),
            Self::AttachmentDescriptionMismatch => formatter
                .write_str("render attachment shape does not match the render-pass description"),
            Self::DuplicateAttachment => {
                formatter.write_str("render pass names one image subresource more than once")
            }
            Self::EmptySubmission => formatter.write_str("operation submission is empty"),
            Self::TooManySubmissionSegments => {
                formatter.write_str("submission contains too many ordered backend segments")
            }
            Self::SelfDependency => formatter.write_str("submission depends on itself"),
            Self::DuplicateSubmissionDependency => {
                formatter.write_str("submission contains a duplicate ordering dependency")
            }
        }
    }
}

impl std::error::Error for CommandDescriptionError {}
