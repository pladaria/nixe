//! Explicit backend-independent resource access and usage transitions.

use std::fmt::{Display, Formatter};

use crate::{
    BufferId, DescriptorTableId, GpuAllocationId, ImageId, ImageSubresourceRange, PipelineId,
    QueryPoolId, RenderPassId, SamplerId, ShaderId,
};

/// A non-empty byte range within a logical buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferRange {
    offset: u64,
    size: u64,
}

impl BufferRange {
    /// Creates a range after checking that it is non-empty and cannot overflow.
    pub const fn new(offset: u64, size: u64) -> Result<Self, AccessDescriptionError> {
        if size == 0 {
            return Err(AccessDescriptionError::EmptyBufferRange);
        }
        if offset.checked_add(size).is_none() {
            return Err(AccessDescriptionError::BufferRangeOverflow);
        }
        Ok(Self { offset, size })
    }

    #[must_use]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }

    #[must_use]
    pub const fn end(self) -> u64 {
        self.offset + self.size
    }
}

/// A non-empty contiguous range in a query pool.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct QueryRange {
    first: u32,
    count: u32,
}

impl QueryRange {
    pub const fn new(first: u32, count: u32) -> Result<Self, AccessDescriptionError> {
        if count == 0 {
            return Err(AccessDescriptionError::EmptyQueryRange);
        }
        if first.checked_add(count).is_none() {
            return Err(AccessDescriptionError::QueryRangeOverflow);
        }
        Ok(Self { first, count })
    }

    #[must_use]
    pub const fn first(self) -> u32 {
        self.first
    }

    #[must_use]
    pub const fn count(self) -> u32 {
        self.count
    }
}

/// Pipeline execution stages participating in an access or ordering edge.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct PipelineStages(u16);

impl PipelineStages {
    pub const COPY: Self = Self(1 << 0);
    pub const VERTEX_INPUT: Self = Self(1 << 1);
    pub const VERTEX_SHADER: Self = Self(1 << 2);
    pub const TESSELLATION_CONTROL_SHADER: Self = Self(1 << 3);
    pub const TESSELLATION_EVALUATION_SHADER: Self = Self(1 << 4);
    pub const GEOMETRY_SHADER: Self = Self(1 << 5);
    pub const FRAGMENT_SHADER: Self = Self(1 << 6);
    pub const EARLY_DEPTH_STENCIL: Self = Self(1 << 7);
    pub const LATE_DEPTH_STENCIL: Self = Self(1 << 8);
    pub const COLOR_OUTPUT: Self = Self(1 << 9);
    pub const COMPUTE_SHADER: Self = Self(1 << 10);
    pub const QUERY: Self = Self(1 << 11);
    pub const INDIRECT: Self = Self(1 << 12);
    pub const PRESENT: Self = Self(1 << 13);

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Direction in which an operation observes canonical resource contents.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

impl AccessMode {
    #[must_use]
    pub const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    #[must_use]
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// Semantic use which a backend must preserve for one resource access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceUsage {
    TransferSource,
    TransferDestination,
    VertexBuffer,
    IndexBuffer,
    IndirectArguments,
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    ColorAttachment,
    DepthStencilAttachment,
    Query,
    QueryResolveDestination,
    Present,
}

impl ResourceUsage {
    #[must_use]
    pub const fn permits(self, mode: AccessMode) -> bool {
        match self {
            Self::TransferSource
            | Self::VertexBuffer
            | Self::IndexBuffer
            | Self::IndirectArguments
            | Self::UniformBuffer
            | Self::SampledImage
            | Self::Present => !mode.writes(),
            Self::TransferDestination | Self::QueryResolveDestination => !mode.reads(),
            Self::StorageBuffer
            | Self::StorageImage
            | Self::ColorAttachment
            | Self::DepthStencilAttachment
            | Self::Query => true,
        }
    }
}

/// Complete execution and visibility scope for one access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AccessScope {
    stages: PipelineStages,
    mode: AccessMode,
    usage: ResourceUsage,
}

impl AccessScope {
    pub const fn new(
        stages: PipelineStages,
        mode: AccessMode,
        usage: ResourceUsage,
    ) -> Result<Self, AccessDescriptionError> {
        if stages.is_empty() {
            return Err(AccessDescriptionError::EmptyPipelineStages);
        }
        if !usage.permits(mode) {
            return Err(AccessDescriptionError::UsageModeMismatch { usage, mode });
        }
        Ok(Self {
            stages,
            mode,
            usage,
        })
    }

    #[must_use]
    pub const fn stages(self) -> PipelineStages {
        self.stages
    }

    #[must_use]
    pub const fn mode(self) -> AccessMode {
        self.mode
    }

    #[must_use]
    pub const fn usage(self) -> ResourceUsage {
        self.usage
    }
}

/// Exact logical resource range affected by an access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessTarget {
    Buffer {
        buffer: BufferId,
        range: BufferRange,
    },
    Image {
        image: ImageId,
        subresources: ImageSubresourceRange,
    },
    Queries {
        pool: QueryPoolId,
        range: QueryRange,
    },
}

/// An explicit read or write emitted by a frontend operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceAccess {
    target: AccessTarget,
    scope: AccessScope,
}

impl ResourceAccess {
    #[must_use]
    pub const fn new(target: AccessTarget, scope: AccessScope) -> Self {
        Self { target, scope }
    }

    #[must_use]
    pub const fn target(self) -> AccessTarget {
        self.target
    }

    #[must_use]
    pub const fn scope(self) -> AccessScope {
        self.scope
    }
}

/// A barrier-defined usage and visibility transition for one exact range.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceTransition {
    target: AccessTarget,
    before: AccessScope,
    after: AccessScope,
}

impl ResourceTransition {
    pub fn new(
        target: AccessTarget,
        before: AccessScope,
        after: AccessScope,
    ) -> Result<Self, AccessDescriptionError> {
        if before == after {
            return Err(AccessDescriptionError::RedundantTransition);
        }
        Ok(Self {
            target,
            before,
            after,
        })
    }

    #[must_use]
    pub const fn target(self) -> AccessTarget {
        self.target
    }

    #[must_use]
    pub const fn before(self) -> AccessScope {
        self.before
    }

    #[must_use]
    pub const fn after(self) -> AccessScope {
        self.after
    }
}

/// Pointer-free identity retained by an operation independently of byte access.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResourceDependency {
    Allocation(GpuAllocationId),
    Buffer(BufferId),
    Image(ImageId),
    Sampler(SamplerId),
    Shader(ShaderId),
    Pipeline(PipelineId),
    DescriptorTable(DescriptorTableId),
    RenderPass(RenderPassId),
    QueryPool(QueryPoolId),
}

/// Failure to define an access or transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccessDescriptionError {
    EmptyBufferRange,
    BufferRangeOverflow,
    EmptyQueryRange,
    QueryRangeOverflow,
    EmptyPipelineStages,
    UsageModeMismatch {
        usage: ResourceUsage,
        mode: AccessMode,
    },
    RedundantTransition,
}

impl Display for AccessDescriptionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBufferRange => formatter.write_str("buffer access range is empty"),
            Self::BufferRangeOverflow => formatter.write_str("buffer access range overflows"),
            Self::EmptyQueryRange => formatter.write_str("query access range is empty"),
            Self::QueryRangeOverflow => formatter.write_str("query access range overflows"),
            Self::EmptyPipelineStages => formatter.write_str("access has no pipeline stages"),
            Self::UsageModeMismatch { usage, mode } => write!(
                formatter,
                "resource usage does not permit access direction: usage={usage:?} mode={mode:?}"
            ),
            Self::RedundantTransition => {
                formatter.write_str("resource transition has identical before and after scopes")
            }
        }
    }
}

impl std::error::Error for AccessDescriptionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_reject_empty_and_overflowing_values() {
        assert_eq!(
            BufferRange::new(0, 0),
            Err(AccessDescriptionError::EmptyBufferRange)
        );
        assert_eq!(
            BufferRange::new(u64::MAX, 2),
            Err(AccessDescriptionError::BufferRangeOverflow)
        );
        assert_eq!(
            QueryRange::new(u32::MAX, 2),
            Err(AccessDescriptionError::QueryRangeOverflow)
        );
    }

    #[test]
    fn scopes_reject_impossible_usage_directions() {
        assert_eq!(
            AccessScope::new(
                PipelineStages::VERTEX_INPUT,
                AccessMode::Write,
                ResourceUsage::VertexBuffer,
            ),
            Err(AccessDescriptionError::UsageModeMismatch {
                usage: ResourceUsage::VertexBuffer,
                mode: AccessMode::Write,
            })
        );
        assert!(
            AccessScope::new(
                PipelineStages::COMPUTE_SHADER,
                AccessMode::ReadWrite,
                ResourceUsage::StorageBuffer,
            )
            .is_ok()
        );
    }

    #[test]
    fn transitions_preserve_both_explicit_scopes() {
        let target = AccessTarget::Buffer {
            buffer: BufferId::new(1),
            range: BufferRange::new(0, 64).unwrap(),
        };
        let before = AccessScope::new(
            PipelineStages::COPY,
            AccessMode::Write,
            ResourceUsage::TransferDestination,
        )
        .unwrap();
        let after = AccessScope::new(
            PipelineStages::VERTEX_INPUT,
            AccessMode::Read,
            ResourceUsage::VertexBuffer,
        )
        .unwrap();
        let transition = ResourceTransition::new(target, before, after).unwrap();
        assert_eq!(transition.target(), target);
        assert_eq!(transition.before(), before);
        assert_eq!(transition.after(), after);
        assert_eq!(
            ResourceTransition::new(target, before, before),
            Err(AccessDescriptionError::RedundantTransition)
        );
    }
}
