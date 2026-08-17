//! Backend-independent GPU resource identities and immutable descriptions.

use std::fmt::{Display, Formatter};

macro_rules! resource_id {
    ($(#[$meta:meta])* $name:ident, $label:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Creates an identity from a value assigned by the resource owner.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Returns the owner-assigned numeric representation.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, concat!($label, "=0x{:016x}"), self.0)
            }
        }
    };
}

resource_id!(
    /// Logical buffer identity.
    BufferId,
    "buffer"
);
resource_id!(
    /// Logical image identity.
    ImageId,
    "image"
);
resource_id!(
    /// Logical sampler identity.
    SamplerId,
    "sampler"
);
resource_id!(
    /// Logical shader identity.
    ShaderId,
    "shader"
);
resource_id!(
    /// Logical pipeline identity.
    PipelineId,
    "pipeline"
);
resource_id!(
    /// Logical descriptor table identity.
    DescriptorTableId,
    "descriptor-table"
);
resource_id!(
    /// Logical render-pass identity.
    RenderPassId,
    "render-pass"
);
resource_id!(
    /// Logical query-pool identity.
    QueryPoolId,
    "query-pool"
);

/// Immutable logical buffer description, without backing storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferDescription {
    size: u64,
}

impl BufferDescription {
    /// Describes a non-empty byte-addressable buffer.
    pub const fn new(size: u64) -> Result<Self, ResourceDescriptionError> {
        if size == 0 {
            return Err(ResourceDescriptionError::EmptyBuffer);
        }
        Ok(Self { size })
    }

    /// Returns the logical byte size.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }
}

/// Dimensional interpretation of an image.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageDimension {
    One,
    Two,
    Three,
    Cube,
}

/// Non-zero base-level dimensions in texels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageExtent {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl ImageExtent {
    /// Creates an extent with no zero dimension.
    pub const fn new(
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<Self, ResourceDescriptionError> {
        if width == 0 || height == 0 || depth == 0 {
            return Err(ResourceDescriptionError::EmptyImageExtent);
        }
        Ok(Self {
            width,
            height,
            depth,
        })
    }
}

/// Backend-independent texel format.
///
/// Variants describe observable component semantics, not Maxwell or host API
/// numeric encodings. The set grows only when a verified frontend operation
/// requires another format.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageFormat {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Unorm,
    Bgra8Srgb,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,
    Depth16Unorm,
    Depth24UnormStencil8Uint,
    Depth32Float,
    Depth32FloatStencil8Uint,
}

impl ImageFormat {
    /// Returns the number of separately addressable planes.
    #[must_use]
    pub const fn plane_count(self) -> u8 {
        1
    }

    /// Returns whether this is a depth or depth/stencil format.
    #[must_use]
    pub const fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth24UnormStencil8Uint
                | Self::Depth32Float
                | Self::Depth32FloatStencil8Uint
        )
    }

    /// Returns the packed bytes per texel for the selected plane.
    #[must_use]
    pub const fn plane_bytes_per_texel(self, plane: u8) -> Option<u8> {
        if plane != 0 {
            return None;
        }
        Some(match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm | Self::R16Float | Self::Depth16Unorm => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rg16Float
            | Self::R32Float
            | Self::Depth24UnormStencil8Uint
            | Self::Depth32Float => 4,
            Self::Rgba16Float | Self::Rg32Float | Self::Depth32FloatStencil8Uint => 8,
            Self::Rgba32Float => 16,
        })
    }
}

/// Semantic use of the image's texels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageKind {
    Color,
    DepthStencil,
}

/// Fixed image sample count.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum SampleCount {
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
    Sixteen = 16,
}

/// Immutable logical image description, without layout or backing storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageDescription {
    dimension: ImageDimension,
    extent: ImageExtent,
    format: ImageFormat,
    kind: ImageKind,
    mip_levels: u8,
    array_layers: u16,
    samples: SampleCount,
}

impl ImageDescription {
    /// Creates a completely validated image description.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dimension: ImageDimension,
        extent: ImageExtent,
        format: ImageFormat,
        kind: ImageKind,
        mip_levels: u8,
        array_layers: u16,
        samples: SampleCount,
    ) -> Result<Self, ResourceDescriptionError> {
        if mip_levels == 0 || array_layers == 0 {
            return Err(ResourceDescriptionError::EmptyImageSubresources);
        }
        match dimension {
            ImageDimension::One if extent.height != 1 || extent.depth != 1 => {
                return Err(ResourceDescriptionError::InvalidImageDimension);
            }
            ImageDimension::Two | ImageDimension::Cube if extent.depth != 1 => {
                return Err(ResourceDescriptionError::InvalidImageDimension);
            }
            ImageDimension::Three if array_layers != 1 => {
                return Err(ResourceDescriptionError::InvalidImageDimension);
            }
            ImageDimension::Cube
                if extent.width != extent.height || !array_layers.is_multiple_of(6) =>
            {
                return Err(ResourceDescriptionError::InvalidCubeImage);
            }
            _ => {}
        }
        let maximum_mips = maximum_mip_levels(extent);
        if u32::from(mip_levels) > maximum_mips {
            return Err(ResourceDescriptionError::TooManyMipLevels {
                requested: mip_levels,
                maximum: maximum_mips as u8,
            });
        }
        if samples != SampleCount::One && (dimension != ImageDimension::Two || mip_levels != 1) {
            return Err(ResourceDescriptionError::InvalidMultisampleImage);
        }
        if format.is_depth_stencil() != (kind == ImageKind::DepthStencil) {
            return Err(ResourceDescriptionError::ImageKindFormatMismatch);
        }
        Ok(Self {
            dimension,
            extent,
            format,
            kind,
            mip_levels,
            array_layers,
            samples,
        })
    }

    #[must_use]
    pub const fn dimension(self) -> ImageDimension {
        self.dimension
    }
    #[must_use]
    pub const fn extent(self) -> ImageExtent {
        self.extent
    }
    #[must_use]
    pub const fn format(self) -> ImageFormat {
        self.format
    }
    #[must_use]
    pub const fn kind(self) -> ImageKind {
        self.kind
    }
    #[must_use]
    pub const fn mip_levels(self) -> u8 {
        self.mip_levels
    }
    #[must_use]
    pub const fn array_layers(self) -> u16 {
        self.array_layers
    }
    #[must_use]
    pub const fn samples(self) -> SampleCount {
        self.samples
    }

    /// Returns the dimensions of one validated mip level.
    #[must_use]
    pub fn mip_extent(self, mip_level: u8) -> Option<ImageExtent> {
        if mip_level >= self.mip_levels {
            return None;
        }
        Some(ImageExtent {
            width: mip_dimension(self.extent.width, mip_level),
            height: mip_dimension(self.extent.height, mip_level),
            depth: mip_dimension(self.extent.depth, mip_level),
        })
    }
}

fn maximum_mip_levels(extent: ImageExtent) -> u32 {
    u32::BITS
        - extent
            .width
            .max(extent.height)
            .max(extent.depth)
            .leading_zeros()
}

fn mip_dimension(base: u32, level: u8) -> u32 {
    let shifted = base.checked_shr(level as u32).unwrap_or(0);
    if shifted == 0 { 1 } else { shifted }
}

/// Texture filtering mode.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Addressing behavior outside normalized texture coordinates.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

/// Immutable logical sampler description.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDescription {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub address_modes: [AddressMode; 3],
    pub lod_min: f32,
    pub lod_max: f32,
    pub max_anisotropy: f32,
}

impl SamplerDescription {
    /// Validates finite LOD and anisotropy ranges without applying host limits.
    pub fn new(
        min_filter: FilterMode,
        mag_filter: FilterMode,
        mip_filter: FilterMode,
        address_modes: [AddressMode; 3],
        lod_min: f32,
        lod_max: f32,
        max_anisotropy: f32,
    ) -> Result<Self, ResourceDescriptionError> {
        if !lod_min.is_finite() || !lod_max.is_finite() || lod_min > lod_max {
            return Err(ResourceDescriptionError::InvalidLodRange);
        }
        if !max_anisotropy.is_finite() || max_anisotropy < 1.0 {
            return Err(ResourceDescriptionError::InvalidAnisotropy);
        }
        Ok(Self {
            min_filter,
            mag_filter,
            mip_filter,
            address_modes,
            lod_min,
            lod_max,
            max_anisotropy,
        })
    }
}

/// Stage consumed by one neutral shader resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderStage {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    Compute,
}

/// Immutable shader description. Shader IR is added at its dedicated boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderDescription {
    pub stage: ShaderStage,
}

/// Kind of neutral pipeline resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineKind {
    Graphics,
    Compute,
}

/// Immutable pipeline description. Detailed state is supplied by later operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineDescription {
    pub kind: PipelineKind,
}

/// Type of resource named by a descriptor slot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DescriptorKind {
    Buffer,
    SampledImage,
    StorageImage,
    Sampler,
}

/// One explicitly numbered neutral descriptor-table binding.
///
/// The binding names a logical resource rather than a backend handle so the
/// validated backend boundary remains instance- and API-independent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DescriptorTableBinding {
    pub binding: u8,
    pub resource: crate::ResourceDependency,
}

/// Immutable descriptor-table shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DescriptorTableDescription {
    bindings: Box<[DescriptorKind]>,
}

impl DescriptorTableDescription {
    pub fn new(bindings: Vec<DescriptorKind>) -> Result<Self, ResourceDescriptionError> {
        if bindings.is_empty() {
            return Err(ResourceDescriptionError::EmptyDescriptorTable);
        }
        Ok(Self {
            bindings: bindings.into_boxed_slice(),
        })
    }
    #[must_use]
    pub fn bindings(&self) -> &[DescriptorKind] {
        &self.bindings
    }
}

/// Immutable compatibility shape of one ordered render-pass attachment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RenderPassAttachmentDescription {
    pub kind: ImageKind,
    pub format: ImageFormat,
    pub samples: SampleCount,
}

/// Immutable ordered render-pass compatibility description.
///
/// Color attachment position is semantically relevant to fragment output
/// routing, so the description retains every attachment rather than merely
/// a count. Resource identities remain dynamic begin-operation data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderPassDescription {
    attachments: Box<[RenderPassAttachmentDescription]>,
}

impl RenderPassDescription {
    pub fn new(
        attachments: Vec<RenderPassAttachmentDescription>,
    ) -> Result<Self, ResourceDescriptionError> {
        let mut depth_stencil = false;
        let mut seen_depth_stencil = false;
        for attachment in &attachments {
            match attachment.kind {
                ImageKind::Color if !seen_depth_stencil => {}
                ImageKind::Color => {
                    return Err(ResourceDescriptionError::InvalidRenderPassAttachments);
                }
                ImageKind::DepthStencil if !depth_stencil => {
                    depth_stencil = true;
                    seen_depth_stencil = true;
                }
                ImageKind::DepthStencil => {
                    return Err(ResourceDescriptionError::InvalidRenderPassAttachments);
                }
            }
            if attachment.format.is_depth_stencil() != (attachment.kind == ImageKind::DepthStencil)
            {
                return Err(ResourceDescriptionError::InvalidRenderPassAttachments);
            }
        }
        Ok(Self {
            attachments: attachments.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn attachments(&self) -> &[RenderPassAttachmentDescription] {
        &self.attachments
    }

    #[must_use]
    pub fn color_attachment_count(&self) -> usize {
        self.attachments
            .iter()
            .filter(|attachment| attachment.kind == ImageKind::Color)
            .count()
    }

    #[must_use]
    pub fn has_depth_stencil(&self) -> bool {
        self.attachments
            .last()
            .is_some_and(|attachment| attachment.kind == ImageKind::DepthStencil)
    }
}

/// Type of values stored by a query pool.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QueryKind {
    Occlusion,
    Timestamp,
    PipelineStatistics,
}

/// Immutable query-pool shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryPoolDescription {
    kind: QueryKind,
    count: u32,
}

impl QueryPoolDescription {
    pub const fn new(kind: QueryKind, count: u32) -> Result<Self, ResourceDescriptionError> {
        if count == 0 {
            return Err(ResourceDescriptionError::EmptyQueryPool);
        }
        Ok(Self { kind, count })
    }
    #[must_use]
    pub const fn kind(self) -> QueryKind {
        self.kind
    }
    #[must_use]
    pub const fn count(self) -> u32 {
        self.count
    }
}

/// Failure to construct an immutable neutral resource description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceDescriptionError {
    EmptyBuffer,
    EmptyImageExtent,
    EmptyImageSubresources,
    InvalidImageDimension,
    InvalidCubeImage,
    TooManyMipLevels { requested: u8, maximum: u8 },
    InvalidMultisampleImage,
    ImageKindFormatMismatch,
    InvalidLodRange,
    InvalidAnisotropy,
    EmptyDescriptorTable,
    InvalidRenderPassAttachments,
    EmptyQueryPool,
}

impl Display for ResourceDescriptionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBuffer => formatter.write_str("buffer size is zero"),
            Self::EmptyImageExtent => formatter.write_str("image extent contains a zero dimension"),
            Self::EmptyImageSubresources => {
                formatter.write_str("image has no mip levels or array layers")
            }
            Self::InvalidImageDimension => {
                formatter.write_str("image extent or layer count is invalid for its dimension")
            }
            Self::InvalidCubeImage => formatter
                .write_str("cube image is not square or has an incomplete six-face layer set"),
            Self::TooManyMipLevels { requested, maximum } => write!(
                formatter,
                "image mip count exceeds its extent: requested={requested} maximum={maximum}"
            ),
            Self::InvalidMultisampleImage => {
                formatter.write_str("multisampled image must be two-dimensional with one mip level")
            }
            Self::ImageKindFormatMismatch => {
                formatter.write_str("image kind and texel format disagree")
            }
            Self::InvalidLodRange => {
                formatter.write_str("sampler LOD range is non-finite or reversed")
            }
            Self::InvalidAnisotropy => {
                formatter.write_str("sampler anisotropy is non-finite or less than one")
            }
            Self::EmptyDescriptorTable => formatter.write_str("descriptor table has no bindings"),
            Self::InvalidRenderPassAttachments => {
                formatter.write_str("render pass attachment sequence is invalid")
            }
            Self::EmptyQueryPool => formatter.write_str("query pool has no queries"),
        }
    }
}

impl std::error::Error for ResourceDescriptionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_description_checks_dimension_mips_samples_and_kind() {
        let extent = ImageExtent::new(64, 32, 1).unwrap();
        let image = ImageDescription::new(
            ImageDimension::Two,
            extent,
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            7,
            1,
            SampleCount::One,
        )
        .unwrap();
        assert_eq!(
            image.mip_extent(6),
            Some(ImageExtent {
                width: 1,
                height: 1,
                depth: 1
            })
        );
        assert_eq!(image.mip_extent(7), None);
        assert!(matches!(
            ImageDescription::new(
                ImageDimension::Two,
                extent,
                ImageFormat::Rgba8Unorm,
                ImageKind::Color,
                8,
                1,
                SampleCount::One
            ),
            Err(ResourceDescriptionError::TooManyMipLevels { .. })
        ));
        assert_eq!(
            ImageDescription::new(
                ImageDimension::Two,
                extent,
                ImageFormat::Depth32Float,
                ImageKind::Color,
                1,
                1,
                SampleCount::One
            ),
            Err(ResourceDescriptionError::ImageKindFormatMismatch)
        );
        assert_eq!(
            ImageDescription::new(
                ImageDimension::Two,
                extent,
                ImageFormat::Rgba8Unorm,
                ImageKind::Color,
                2,
                1,
                SampleCount::Four
            ),
            Err(ResourceDescriptionError::InvalidMultisampleImage)
        );
    }

    #[test]
    fn resource_id_domains_are_distinct_and_stably_formatted() {
        assert_eq!(BufferId::new(3).to_string(), "buffer=0x0000000000000003");
        assert_eq!(ImageId::new(3).to_string(), "image=0x0000000000000003");
        assert_eq!(
            DescriptorTableId::new(3).to_string(),
            "descriptor-table=0x0000000000000003"
        );
    }

    #[test]
    fn sampler_descriptor_and_query_validation_is_explicit() {
        assert_eq!(
            SamplerDescription::new(
                FilterMode::Nearest,
                FilterMode::Linear,
                FilterMode::Nearest,
                [AddressMode::Repeat; 3],
                2.0,
                1.0,
                1.0
            ),
            Err(ResourceDescriptionError::InvalidLodRange)
        );
        assert_eq!(
            QueryPoolDescription::new(QueryKind::Timestamp, 0),
            Err(ResourceDescriptionError::EmptyQueryPool)
        );
    }
}
