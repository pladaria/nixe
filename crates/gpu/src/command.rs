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
    /// Independent groups of four perimeter-ordered vertices, with the fourth
    /// vertex supplying constant interpolants. Incomplete groups are discarded.
    Quads,
    Patches,
}

/// Rasterization rule applied after triangle assembly.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum TriangleRasterization {
    /// Cover samples inside the triangle edges.
    #[default]
    Fill,
    /// Cover the triangle's axis-aligned post-projection bounding box while
    /// retaining the original triangle's interpolation planes.
    FillRectangle,
    /// Rasterize triangle boundaries as rectangular lines. When `smooth` is
    /// set, coverage modulates fragment output zero's alpha after shading.
    /// Width is a positive finite float in framebuffer pixels, stored as bits
    /// for exact state identity. Backends validate host limits when consumed.
    Wireframe { width_bits: u32, smooth: bool },
}

/// Explicit rectangular coverage for line primitives, independently of polygon mode.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LineRasterization {
    pub width_bits: u32,
    pub smooth: bool,
}

/// Polygon depth offset: constant factor in host depth-format units, slope
/// factor and signed clamp. Float bits preserve exact pipeline identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DepthBias {
    pub constant_bits: u32,
    pub slope_scale_bits: u32,
    pub clamp_bits: u32,
}

/// Front-facing winding in framebuffer coordinates (X right, Y down), after
/// viewport transformation. Counterclockwise means negative shoelace area.
/// Changing the viewport determinant changes facing; it does not change this
/// state. Both WebGPU and Vulkan use this convention:
/// https://gpuweb.github.io/gpuweb/#polygon-rasterization
/// https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-polygons-basic
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

/// Faces discarded after polygon assembly, without suppressing shader execution
/// before rasterization. Points and direct lines are not affected.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
    FrontAndBack,
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

/// Width and interpretation of indices in an index buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndexType {
    Uint8,
    Uint16,
    Uint32,
}

/// Direct draw arguments. Indirect argument buffers are explicit accesses and
/// can be added as a later verified operation form without changing these.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DrawArguments {
    NonIndexed {
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    },
    Indexed {
        first_index: u32,
        index_count: u32,
        vertex_offset: i32,
        first_instance: u32,
        instance_count: u32,
    },
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

/// Backend-independent affine transform from normalized device coordinates to
/// framebuffer coordinates: `window = ndc * scale + offset`.
///
/// The depth interval is retained independently because guest APIs may program
/// clip-depth bounds that cannot be reconstructed from the Z coefficients.
///
/// IEEE-754 bit patterns are retained so negative viewport axes and exact guest
/// state survive frontend lowering without relying on one host API convention.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ViewportTransform {
    scale: [u32; 3],
    offset: [u32; 3],
    depth_range: [u32; 2],
    depth_clip_negative_one_to_one: bool,
    fragment_depth_transform: Option<FragmentDepthTransform>,
}

impl ViewportTransform {
    pub fn new(
        scale: [f32; 3],
        offset: [f32; 3],
        depth_range: [f32; 2],
    ) -> Result<Self, CommandDescriptionError> {
        if scale
            .into_iter()
            .chain(offset)
            .chain(depth_range)
            .any(|value| !value.is_finite())
        {
            return Err(CommandDescriptionError::NonFiniteViewportTransform);
        }
        if scale[0] == 0.0 || scale[1] == 0.0 {
            return Err(CommandDescriptionError::EmptyViewport);
        }
        Ok(Self {
            scale: scale.map(f32::to_bits),
            offset: offset.map(f32::to_bits),
            depth_range: depth_range.map(f32::to_bits),
            depth_clip_negative_one_to_one: false,
            fragment_depth_transform: None,
        })
    }

    #[must_use]
    pub fn scale(self) -> [f32; 3] {
        self.scale.map(f32::from_bits)
    }

    #[must_use]
    pub fn offset(self) -> [f32; 3] {
        self.offset.map(f32::from_bits)
    }

    /// Selects the homogeneous clip-space Z interval before the affine viewport.
    /// The default is zero-to-one; negative-one-to-one requires a final-stage
    /// position conversion on host APIs with a zero-to-one clip volume.
    pub fn with_negative_one_to_one_depth(mut self, enabled: bool) -> Self {
        self.depth_clip_negative_one_to_one = enabled;
        self
    }

    #[must_use]
    pub const fn depth_clip_negative_one_to_one(self) -> bool {
        self.depth_clip_negative_one_to_one
    }

    pub fn with_fragment_depth_transform(mut self, transform: FragmentDepthTransform) -> Self {
        self.fragment_depth_transform = Some(transform);
        self
    }

    #[must_use]
    pub const fn fragment_depth_transform(self) -> Option<FragmentDepthTransform> {
        self.fragment_depth_transform
    }

    /// Returns the affine viewport endpoints in window-space depth.
    #[must_use]
    pub fn depth_range(self) -> [f32; 2] {
        self.depth_range.map(f32::from_bits)
    }
}

impl DrawArguments {}

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

/// Effective depth-test state carried by one neutral draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DepthState {
    pub test_enabled: bool,
    pub write_enabled: bool,
    pub compare: DepthCompareOperation,
}

impl DepthState {
    pub const DISABLED: Self = Self {
        test_enabled: false,
        write_enabled: false,
        compare: DepthCompareOperation::Always,
    };

    #[must_use]
    pub const fn new(
        test_enabled: bool,
        write_enabled: bool,
        compare: DepthCompareOperation,
    ) -> Self {
        Self {
            test_enabled,
            write_enabled,
            compare,
        }
    }
}

/// Fixed-function stencil operations, applied before the eight-bit write mask.
/// https://docs.vulkan.org/spec/latest/chapters/fragops.html#fragops-stencil
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StencilFaceState {
    pub compare: DepthCompareOperation,
    pub fail: StencilOperation,
    pub depth_fail: StencilOperation,
    pub pass: StencilOperation,
    pub read_mask: u8,
    pub write_mask: u8,
    pub reference: u8,
}

impl StencilFaceState {
    #[must_use]
    pub const fn read_only(compare: DepthCompareOperation, read_mask: u8, reference: u8) -> Self {
        Self {
            compare,
            read_mask,
            reference,
            write_mask: 0,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        }
    }

    #[must_use]
    pub fn writes(self) -> bool {
        self.write_mask != 0
            && self
                .reachable_operations()
                .any(|op| op != StencilOperation::Keep)
    }

    #[must_use]
    pub fn uses_reference(self) -> bool {
        !matches!(
            self.compare,
            DepthCompareOperation::Always | DepthCompareOperation::Never
        ) || (self.write_mask != 0
            && self
                .reachable_operations()
                .any(|op| op == StencilOperation::Replace))
    }

    fn reachable_operations(self) -> impl Iterator<Item = StencilOperation> {
        [
            (self.fail, self.compare != DepthCompareOperation::Always),
            (
                self.depth_fail,
                self.compare != DepthCompareOperation::Never,
            ),
            (self.pass, self.compare != DepthCompareOperation::Never),
        ]
        .into_iter()
        .filter_map(|(op, reachable)| reachable.then_some(op))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StencilTestState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
}

impl StencilTestState {
    #[must_use]
    pub fn writes(self) -> bool {
        self.front.writes() || self.back.writes()
    }
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
    NonFiniteViewportTransform,
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
            Self::NonFiniteViewportTransform => {
                formatter.write_str("viewport transform contains a non-finite component")
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_transform_preserves_negative_axes_and_rejects_invalid_values() {
        let viewport =
            ViewportTransform::new([640.0, -360.0, 0.5], [640.0, 360.0, 0.5], [0.0, 1.0]).unwrap();
        assert_eq!(viewport.scale(), [640.0, -360.0, 0.5]);
        assert_eq!(viewport.offset(), [640.0, 360.0, 0.5]);
        assert_eq!(viewport.depth_range(), [0.0, 1.0]);
        assert_eq!(
            ViewportTransform::new([0.0, 1.0, 1.0], [0.0; 3], [0.0, 1.0]),
            Err(CommandDescriptionError::EmptyViewport)
        );
        assert_eq!(
            ViewportTransform::new([1.0, f32::NAN, 1.0], [0.0; 3], [0.0, 1.0]),
            Err(CommandDescriptionError::NonFiniteViewportTransform)
        );
        assert_eq!(
            ViewportTransform::new([1.0; 3], [0.0; 3], [0.0, f32::INFINITY]),
            Err(CommandDescriptionError::NonFiniteViewportTransform)
        );
    }
}
