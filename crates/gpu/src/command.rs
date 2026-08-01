//! Immutable backend-independent GPU operations and submission ordering.

use std::fmt::{Display, Formatter};

use crate::{
    AccessMode, AccessScope, AccessTarget, BackendFeatures, BufferId, BufferRange,
    CapabilityRequirement, CapabilityRequirements, DescriptorTableId, FrontendSubmissionId,
    ImageExtent, ImageFormat, ImageId, ImageKind, ImageSubresourceRange, PipelineId,
    PipelineStages, QueryKind, QueryPoolId, QueryRange, RenderPassDescription, RenderPassId,
    ResourceAccess, ResourceDependency, ResourceTransition, ResourceUsage, SampleCount,
};

/// One buffer and exact byte range named by an operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferRegion {
    pub buffer: BufferId,
    pub range: BufferRange,
}

impl BufferRegion {
    #[must_use]
    pub const fn target(self) -> AccessTarget {
        AccessTarget::Buffer {
            buffer: self.buffer,
            range: self.range,
        }
    }
}

/// Origin of an image operation in texels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageOrigin {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// One image subresource range and texel box.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageRegion {
    pub image: ImageId,
    pub subresources: ImageSubresourceRange,
    pub origin: ImageOrigin,
    pub extent: ImageExtent,
}

impl ImageRegion {
    #[must_use]
    pub const fn target(self) -> AccessTarget {
        AccessTarget::Image {
            image: self.image,
            subresources: self.subresources,
        }
    }
}

/// A typed transfer between logical resource ranges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CopyOperation {
    BufferToBuffer {
        source: BufferRegion,
        destination: BufferRegion,
    },
    BufferToImage {
        source: BufferRegion,
        destination: ImageRegion,
    },
    ImageToBuffer {
        source: ImageRegion,
        destination: BufferRegion,
    },
    ImageToImage {
        source: ImageRegion,
        destination: ImageRegion,
    },
}

impl CopyOperation {
    pub fn buffer_to_buffer(
        source: BufferRegion,
        destination: BufferRegion,
    ) -> Result<Self, CommandDescriptionError> {
        if source.range.size() != destination.range.size() {
            return Err(CommandDescriptionError::CopySizeMismatch);
        }
        Ok(Self::BufferToBuffer {
            source,
            destination,
        })
    }
}

/// Neutral clear payload retaining the exact requested value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
    Buffer(u32),
    Color([f32; 4]),
    DepthStencil { depth: f32, stencil: u8 },
}

impl ClearValue {
    fn is_finite(self) -> bool {
        match self {
            Self::Buffer(_) => true,
            Self::Color(value) => value.iter().all(|component| component.is_finite()),
            Self::DepthStencil { depth, .. } => depth.is_finite(),
        }
    }
}

/// A typed clear of a complete logical range.
#[derive(Clone, Debug, PartialEq)]
pub enum ClearOperation {
    Buffer {
        target: BufferRegion,
        value: ClearValue,
    },
    Image {
        target: ImageRegion,
        kind: ImageKind,
        format: ImageFormat,
        samples: SampleCount,
        value: ClearValue,
    },
}

impl ClearOperation {
    pub fn buffer(target: BufferRegion, value: u32) -> Result<Self, CommandDescriptionError> {
        Ok(Self::Buffer {
            target,
            value: ClearValue::Buffer(value),
        })
    }

    pub fn image(
        target: ImageRegion,
        kind: ImageKind,
        format: ImageFormat,
        samples: SampleCount,
        value: ClearValue,
    ) -> Result<Self, CommandDescriptionError> {
        let value_matches = matches!(
            (kind, value),
            (ImageKind::Color, ClearValue::Color(_))
                | (ImageKind::DepthStencil, ClearValue::DepthStencil { .. })
        );
        if !value_matches || format.is_depth_stencil() != (kind == ImageKind::DepthStencil) {
            return Err(CommandDescriptionError::ClearValueMismatch);
        }
        if !value.is_finite() {
            return Err(CommandDescriptionError::NonFiniteClearValue);
        }
        Ok(Self::Image {
            target,
            kind,
            format,
            samples,
            value,
        })
    }
}

/// Primitive assembly mode expected by a graphics draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PrimitiveTopology {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Patches,
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

impl DrawArguments {
    const fn is_empty(self) -> bool {
        match self {
            Self::NonIndexed {
                vertex_count,
                instance_count,
                ..
            } => vertex_count == 0 || instance_count == 0,
            Self::Indexed {
                index_count,
                instance_count,
                ..
            } => index_count == 0 || instance_count == 0,
        }
    }

    const fn is_indexed(self) -> bool {
        matches!(self, Self::Indexed { .. })
    }
}

/// One fully specified direct draw dependency set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DrawOperation {
    pub pipeline: PipelineId,
    pub render_pass: RenderPassId,
    pub topology: PrimitiveTopology,
    pub descriptor_tables: Box<[DescriptorTableId]>,
    pub vertex_buffers: Box<[BufferRegion]>,
    pub index_buffer: Option<(BufferRegion, IndexType)>,
    pub arguments: DrawArguments,
}

impl DrawOperation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipeline: PipelineId,
        render_pass: RenderPassId,
        topology: PrimitiveTopology,
        descriptor_tables: Vec<DescriptorTableId>,
        vertex_buffers: Vec<BufferRegion>,
        index_buffer: Option<(BufferRegion, IndexType)>,
        arguments: DrawArguments,
    ) -> Result<Self, CommandDescriptionError> {
        if arguments.is_empty() {
            return Err(CommandDescriptionError::EmptyDraw);
        }
        if arguments.is_indexed() != index_buffer.is_some() {
            return Err(CommandDescriptionError::IndexBufferMismatch);
        }
        Ok(Self {
            pipeline,
            render_pass,
            topology,
            descriptor_tables: descriptor_tables.into_boxed_slice(),
            vertex_buffers: vertex_buffers.into_boxed_slice(),
            index_buffer,
            arguments,
        })
    }
}

/// One compute dispatch with complete direct workgroup counts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DispatchOperation {
    pub pipeline: PipelineId,
    pub descriptor_tables: Box<[DescriptorTableId]>,
    pub workgroups: [u32; 3],
}

impl DispatchOperation {
    pub fn new(
        pipeline: PipelineId,
        descriptor_tables: Vec<DescriptorTableId>,
        workgroups: [u32; 3],
    ) -> Result<Self, CommandDescriptionError> {
        if workgroups.contains(&0) {
            return Err(CommandDescriptionError::EmptyDispatch);
        }
        Ok(Self {
            pipeline,
            descriptor_tables: descriptor_tables.into_boxed_slice(),
            workgroups,
        })
    }
}

/// One barrier containing all resource transitions committed together.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BarrierOperation {
    transitions: Box<[ResourceTransition]>,
}

impl BarrierOperation {
    pub fn new(transitions: Vec<ResourceTransition>) -> Result<Self, CommandDescriptionError> {
        if transitions.is_empty() {
            return Err(CommandDescriptionError::EmptyBarrier);
        }
        Ok(Self {
            transitions: transitions.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn transitions(&self) -> &[ResourceTransition] {
        &self.transitions
    }
}

/// Query lifecycle and resolve operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueryOperation {
    Reset {
        pool: QueryPoolId,
        kind: QueryKind,
        range: QueryRange,
    },
    Begin {
        pool: QueryPoolId,
        kind: QueryKind,
        query: u32,
    },
    End {
        pool: QueryPoolId,
        kind: QueryKind,
        query: u32,
    },
    WriteTimestamp {
        pool: QueryPoolId,
        query: u32,
    },
    Resolve {
        pool: QueryPoolId,
        kind: QueryKind,
        range: QueryRange,
        destination: BufferRegion,
    },
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

/// One render-pass attachment and its observable load/store behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderAttachment {
    pub image: ImageId,
    pub subresources: ImageSubresourceRange,
    pub kind: ImageKind,
    pub format: ImageFormat,
    pub samples: SampleCount,
    pub load: AttachmentLoad,
    pub store: AttachmentStore,
}

impl RenderAttachment {
    fn validate(self) -> Result<(), CommandDescriptionError> {
        if self.format.is_depth_stencil() != (self.kind == ImageKind::DepthStencil) {
            return Err(CommandDescriptionError::AttachmentKindMismatch);
        }
        if let AttachmentLoad::Clear(value) = self.load {
            let matches = matches!(
                (self.kind, value),
                (ImageKind::Color, ClearValue::Color(_))
                    | (ImageKind::DepthStencil, ClearValue::DepthStencil { .. })
            );
            if !matches {
                return Err(CommandDescriptionError::ClearValueMismatch);
            }
            if !value.is_finite() {
                return Err(CommandDescriptionError::NonFiniteClearValue);
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn target(self) -> AccessTarget {
        AccessTarget::Image {
            image: self.image,
            subresources: self.subresources,
        }
    }
}

/// Explicit render-pass boundaries. Backends must not infer pass lifetime from
/// draws or attachment identity.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderPassOperation {
    Begin {
        render_pass: RenderPassId,
        attachments: Box<[RenderAttachment]>,
    },
    End {
        render_pass: RenderPassId,
    },
}

impl RenderPassOperation {
    pub fn begin(
        render_pass: RenderPassId,
        description: RenderPassDescription,
        attachments: Vec<RenderAttachment>,
    ) -> Result<Self, CommandDescriptionError> {
        if attachments.len() != description.attachments().len() {
            return Err(CommandDescriptionError::AttachmentCountMismatch);
        }
        for (index, attachment) in attachments.iter().enumerate() {
            attachment.validate()?;
            if attachments[index + 1..]
                .iter()
                .any(|other| attachment.target() == other.target())
            {
                return Err(CommandDescriptionError::DuplicateAttachment);
            }
        }
        if attachments
            .iter()
            .zip(description.attachments())
            .any(|(attachment, expected)| {
                attachment.kind != expected.kind
                    || attachment.format != expected.format
                    || attachment.samples != expected.samples
            })
        {
            return Err(CommandDescriptionError::AttachmentDescriptionMismatch);
        }
        Ok(Self::Begin {
            render_pass,
            attachments: attachments.into_boxed_slice(),
        })
    }

    #[must_use]
    pub const fn end(render_pass: RenderPassId) -> Self {
        Self::End { render_pass }
    }
}

/// The semantic kind of one neutral GPU command.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuCommand {
    Copy(CopyOperation),
    Clear(ClearOperation),
    Draw(DrawOperation),
    Dispatch(DispatchOperation),
    Barrier(BarrierOperation),
    Query(QueryOperation),
    RenderPass(RenderPassOperation),
}

/// One immutable command plus all dependencies, accesses and capabilities
/// which are not implicit in its typed operands.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuOperation {
    command: GpuCommand,
    accesses: Box<[ResourceAccess]>,
    dependencies: Box<[ResourceDependency]>,
    requirements: CapabilityRequirements,
}

impl GpuOperation {
    /// Completes one operation without mutating a frontend or backend.
    /// Mandatory accesses, dependencies and feature requirements are derived
    /// from the command and merged with shader/descriptor-specific inputs.
    #[must_use]
    pub fn new(
        command: GpuCommand,
        additional_accesses: impl IntoIterator<Item = ResourceAccess>,
        additional_dependencies: impl IntoIterator<Item = ResourceDependency>,
        additional_requirements: CapabilityRequirements,
    ) -> Self {
        let mut accesses = command.accesses();
        extend_unique(&mut accesses, additional_accesses);
        let mut dependencies = command.dependencies();
        extend_unique(&mut dependencies, additional_dependencies);
        let requirements = command
            .capability_requirements()
            .merged(&additional_requirements);
        Self {
            command,
            accesses: accesses.into_boxed_slice(),
            dependencies: dependencies.into_boxed_slice(),
            requirements,
        }
    }

    #[must_use]
    pub const fn command(&self) -> &GpuCommand {
        &self.command
    }

    #[must_use]
    pub fn accesses(&self) -> &[ResourceAccess] {
        &self.accesses
    }

    #[must_use]
    pub fn dependencies(&self) -> &[ResourceDependency] {
        &self.dependencies
    }

    #[must_use]
    pub const fn capability_requirements(&self) -> &CapabilityRequirements {
        &self.requirements
    }
}

impl GpuCommand {
    fn accesses(&self) -> Vec<ResourceAccess> {
        match self {
            Self::Copy(copy) => copy_accesses(copy),
            Self::Clear(clear) => clear_accesses(clear),
            Self::Draw(draw) => draw_accesses(draw),
            Self::Dispatch(_) => Vec::new(),
            Self::Barrier(_) => Vec::new(),
            Self::Query(query) => query_accesses(query),
            Self::RenderPass(pass) => render_pass_accesses(pass),
        }
    }

    fn dependencies(&self) -> Vec<ResourceDependency> {
        let mut dependencies = Vec::new();
        match self {
            Self::Copy(_) | Self::Clear(_) => {
                for access in self.accesses() {
                    push_target_dependency(&mut dependencies, access.target());
                }
            }
            Self::Draw(draw) => {
                dependencies.push(ResourceDependency::Pipeline(draw.pipeline));
                dependencies.push(ResourceDependency::RenderPass(draw.render_pass));
                extend_unique(
                    &mut dependencies,
                    draw.descriptor_tables
                        .iter()
                        .copied()
                        .map(ResourceDependency::DescriptorTable),
                );
                for access in draw_accesses(draw) {
                    push_target_dependency(&mut dependencies, access.target());
                }
            }
            Self::Dispatch(dispatch) => {
                dependencies.push(ResourceDependency::Pipeline(dispatch.pipeline));
                extend_unique(
                    &mut dependencies,
                    dispatch
                        .descriptor_tables
                        .iter()
                        .copied()
                        .map(ResourceDependency::DescriptorTable),
                );
            }
            Self::Barrier(barrier) => {
                for transition in barrier.transitions() {
                    push_target_dependency(&mut dependencies, transition.target());
                }
            }
            Self::Query(query) => {
                for access in query_accesses(query) {
                    push_target_dependency(&mut dependencies, access.target());
                }
            }
            Self::RenderPass(pass) => {
                let render_pass = match pass {
                    RenderPassOperation::Begin { render_pass, .. }
                    | RenderPassOperation::End { render_pass } => *render_pass,
                };
                dependencies.push(ResourceDependency::RenderPass(render_pass));
                for access in render_pass_accesses(pass) {
                    push_target_dependency(&mut dependencies, access.target());
                }
            }
        }
        dependencies
    }

    fn capability_requirements(&self) -> CapabilityRequirements {
        let mut requirements = vec![CapabilityRequirement::Features(match self {
            Self::Copy(_) => BackendFeatures::COPY,
            Self::Clear(_) => BackendFeatures::CLEAR,
            Self::Draw(draw) if draw.arguments.is_indexed() => {
                BackendFeatures::DRAW.union(BackendFeatures::INDEXED_DRAW)
            }
            Self::Draw(_) => BackendFeatures::DRAW,
            Self::Dispatch(_) => BackendFeatures::DISPATCH,
            Self::Barrier(_) => BackendFeatures::BARRIER,
            Self::Query(_) => BackendFeatures::QUERY,
            Self::RenderPass(_) => BackendFeatures::RENDER_PASS,
        })];
        match self {
            Self::Clear(ClearOperation::Image {
                format, samples, ..
            }) => {
                requirements.push(CapabilityRequirement::ImageFormat(*format));
                requirements.push(CapabilityRequirement::SampleCount(*samples));
            }
            Self::Dispatch(dispatch) => requirements.push(
                CapabilityRequirement::ComputeWorkgroups(dispatch.workgroups),
            ),
            Self::Query(query) => {
                requirements.push(CapabilityRequirement::QueryKind(query_kind(query)))
            }
            Self::RenderPass(RenderPassOperation::Begin { attachments, .. }) => {
                requirements.push(CapabilityRequirement::ColorAttachments(
                    attachments
                        .iter()
                        .filter(|attachment| attachment.kind == ImageKind::Color)
                        .count() as u8,
                ));
                for attachment in attachments {
                    requirements.push(CapabilityRequirement::ImageFormat(attachment.format));
                    requirements.push(CapabilityRequirement::SampleCount(attachment.samples));
                }
            }
            _ => {}
        }
        CapabilityRequirements::new(requirements)
    }
}

/// A complete neutral operation sequence and its frontend ordering edges.
#[derive(Clone, Debug, PartialEq)]
pub struct OperationSubmission {
    id: FrontendSubmissionId,
    predecessors: Box<[FrontendSubmissionId]>,
    operations: Box<[GpuOperation]>,
}

impl OperationSubmission {
    pub fn new(
        id: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
        operations: Vec<GpuOperation>,
    ) -> Result<Self, CommandDescriptionError> {
        if operations.is_empty() {
            return Err(CommandDescriptionError::EmptySubmission);
        }
        if predecessors.contains(&id) {
            return Err(CommandDescriptionError::SelfDependency);
        }
        for (index, predecessor) in predecessors.iter().enumerate() {
            if predecessors[index + 1..].contains(predecessor) {
                return Err(CommandDescriptionError::DuplicateSubmissionDependency);
            }
        }
        Ok(Self {
            id,
            predecessors: predecessors.into_boxed_slice(),
            operations: operations.into_boxed_slice(),
        })
    }

    #[must_use]
    pub const fn id(&self) -> FrontendSubmissionId {
        self.id
    }

    #[must_use]
    pub fn predecessors(&self) -> &[FrontendSubmissionId] {
        &self.predecessors
    }

    #[must_use]
    pub fn operations(&self) -> &[GpuOperation] {
        &self.operations
    }

    #[must_use]
    pub fn capability_requirements(&self) -> Vec<CapabilityRequirements> {
        self.operations
            .iter()
            .map(|operation| operation.capability_requirements().clone())
            .collect()
    }
}

/// Failure to construct an immutable neutral operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandDescriptionError {
    CopySizeMismatch,
    ClearValueMismatch,
    NonFiniteClearValue,
    EmptyDraw,
    IndexBufferMismatch,
    EmptyDispatch,
    EmptyBarrier,
    AttachmentKindMismatch,
    AttachmentCountMismatch,
    AttachmentDescriptionMismatch,
    DuplicateAttachment,
    EmptySubmission,
    SelfDependency,
    DuplicateSubmissionDependency,
}

impl Display for CommandDescriptionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CopySizeMismatch => {
                formatter.write_str("buffer copy source and destination sizes differ")
            }
            Self::ClearValueMismatch => {
                formatter.write_str("clear value does not match its resource kind")
            }
            Self::NonFiniteClearValue => formatter.write_str("clear value is not finite"),
            Self::EmptyDraw => formatter.write_str("draw has no vertices, indices, or instances"),
            Self::IndexBufferMismatch => {
                formatter.write_str("draw arguments and index-buffer presence disagree")
            }
            Self::EmptyDispatch => formatter.write_str("dispatch has a zero workgroup dimension"),
            Self::EmptyBarrier => formatter.write_str("barrier has no resource transitions"),
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
            Self::SelfDependency => formatter.write_str("submission depends on itself"),
            Self::DuplicateSubmissionDependency => {
                formatter.write_str("submission contains a duplicate ordering dependency")
            }
        }
    }
}

impl std::error::Error for CommandDescriptionError {}

fn copy_accesses(copy: &CopyOperation) -> Vec<ResourceAccess> {
    let (source, destination) = match copy {
        CopyOperation::BufferToBuffer {
            source,
            destination,
        } => (source.target(), destination.target()),
        CopyOperation::BufferToImage {
            source,
            destination,
        } => (source.target(), destination.target()),
        CopyOperation::ImageToBuffer {
            source,
            destination,
        } => (source.target(), destination.target()),
        CopyOperation::ImageToImage {
            source,
            destination,
        } => (source.target(), destination.target()),
    };
    vec![
        ResourceAccess::new(
            source,
            scope(
                PipelineStages::COPY,
                AccessMode::Read,
                ResourceUsage::TransferSource,
            ),
        ),
        ResourceAccess::new(
            destination,
            scope(
                PipelineStages::COPY,
                AccessMode::Write,
                ResourceUsage::TransferDestination,
            ),
        ),
    ]
}

fn clear_accesses(clear: &ClearOperation) -> Vec<ResourceAccess> {
    let target = match clear {
        ClearOperation::Buffer { target, .. } => target.target(),
        ClearOperation::Image { target, .. } => target.target(),
    };
    vec![ResourceAccess::new(
        target,
        scope(
            PipelineStages::COPY,
            AccessMode::Write,
            ResourceUsage::TransferDestination,
        ),
    )]
}

fn draw_accesses(draw: &DrawOperation) -> Vec<ResourceAccess> {
    let mut accesses = draw
        .vertex_buffers
        .iter()
        .map(|buffer| {
            ResourceAccess::new(
                buffer.target(),
                scope(
                    PipelineStages::VERTEX_INPUT,
                    AccessMode::Read,
                    ResourceUsage::VertexBuffer,
                ),
            )
        })
        .collect::<Vec<_>>();
    if let Some((buffer, _)) = draw.index_buffer {
        accesses.push(ResourceAccess::new(
            buffer.target(),
            scope(
                PipelineStages::VERTEX_INPUT,
                AccessMode::Read,
                ResourceUsage::IndexBuffer,
            ),
        ));
    }
    accesses
}

fn query_accesses(query: &QueryOperation) -> Vec<ResourceAccess> {
    let (pool, range, mode) = match query {
        QueryOperation::Reset { pool, range, .. } => (*pool, *range, AccessMode::Write),
        QueryOperation::Begin { pool, query, .. }
        | QueryOperation::End { pool, query, .. }
        | QueryOperation::WriteTimestamp { pool, query } => (
            *pool,
            QueryRange::new(*query, 1).expect("one query is a valid range"),
            AccessMode::Write,
        ),
        QueryOperation::Resolve {
            pool,
            range,
            destination,
            ..
        } => {
            return vec![
                ResourceAccess::new(
                    AccessTarget::Queries {
                        pool: *pool,
                        range: *range,
                    },
                    scope(
                        PipelineStages::QUERY,
                        AccessMode::Read,
                        ResourceUsage::Query,
                    ),
                ),
                ResourceAccess::new(
                    destination.target(),
                    scope(
                        PipelineStages::COPY,
                        AccessMode::Write,
                        ResourceUsage::QueryResolveDestination,
                    ),
                ),
            ];
        }
    };
    vec![ResourceAccess::new(
        AccessTarget::Queries { pool, range },
        scope(PipelineStages::QUERY, mode, ResourceUsage::Query),
    )]
}

fn render_pass_accesses(pass: &RenderPassOperation) -> Vec<ResourceAccess> {
    let RenderPassOperation::Begin { attachments, .. } = pass else {
        return Vec::new();
    };
    attachments
        .iter()
        .map(|attachment| {
            let mode = if attachment.load == AttachmentLoad::Load {
                AccessMode::ReadWrite
            } else {
                AccessMode::Write
            };
            let (stages, usage) = match attachment.kind {
                ImageKind::Color => (PipelineStages::COLOR_OUTPUT, ResourceUsage::ColorAttachment),
                ImageKind::DepthStencil => (
                    PipelineStages::EARLY_DEPTH_STENCIL.union(PipelineStages::LATE_DEPTH_STENCIL),
                    ResourceUsage::DepthStencilAttachment,
                ),
            };
            ResourceAccess::new(attachment.target(), scope(stages, mode, usage))
        })
        .collect()
}

fn query_kind(query: &QueryOperation) -> QueryKind {
    match query {
        QueryOperation::Reset { kind, .. }
        | QueryOperation::Begin { kind, .. }
        | QueryOperation::End { kind, .. }
        | QueryOperation::Resolve { kind, .. } => *kind,
        QueryOperation::WriteTimestamp { .. } => QueryKind::Timestamp,
    }
}

fn push_target_dependency(dependencies: &mut Vec<ResourceDependency>, target: AccessTarget) {
    let dependency = match target {
        AccessTarget::Buffer { buffer, .. } => ResourceDependency::Buffer(buffer),
        AccessTarget::Image { image, .. } => ResourceDependency::Image(image),
        AccessTarget::Queries { pool, .. } => ResourceDependency::QueryPool(pool),
    };
    if !dependencies.contains(&dependency) {
        dependencies.push(dependency);
    }
}

fn scope(stages: PipelineStages, mode: AccessMode, usage: ResourceUsage) -> AccessScope {
    AccessScope::new(stages, mode, usage).expect("command-defined access scope must be valid")
}

fn extend_unique<T: PartialEq>(target: &mut Vec<T>, values: impl IntoIterator<Item = T>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendCapabilities, BackendLimits};

    fn buffer(id: u64, offset: u64, size: u64) -> BufferRegion {
        BufferRegion {
            buffer: BufferId::new(id),
            range: BufferRange::new(offset, size).unwrap(),
        }
    }

    fn image(id: u64) -> ImageRegion {
        ImageRegion {
            image: ImageId::new(id),
            subresources: ImageSubresourceRange {
                plane: 0,
                mip_level: 0,
                base_layer: 0,
                layer_count: 1,
            },
            origin: ImageOrigin { x: 0, y: 0, z: 0 },
            extent: ImageExtent::new(16, 16, 1).unwrap(),
        }
    }

    #[test]
    fn copies_derive_ordered_read_and_write_accesses() {
        let copy = CopyOperation::buffer_to_buffer(buffer(1, 0, 64), buffer(2, 32, 64)).unwrap();
        let operation = GpuOperation::new(
            GpuCommand::Copy(copy),
            [],
            [],
            CapabilityRequirements::none(),
        );
        assert_eq!(operation.accesses().len(), 2);
        assert_eq!(operation.accesses()[0].scope().mode(), AccessMode::Read);
        assert_eq!(operation.accesses()[1].scope().mode(), AccessMode::Write);
        assert_eq!(
            operation.dependencies(),
            &[
                ResourceDependency::Buffer(BufferId::new(1)),
                ResourceDependency::Buffer(BufferId::new(2))
            ]
        );
    }

    #[test]
    fn draw_and_dispatch_reject_incomplete_direct_arguments() {
        assert_eq!(
            DrawOperation::new(
                PipelineId::new(1),
                RenderPassId::new(1),
                PrimitiveTopology::Triangles,
                vec![],
                vec![],
                None,
                DrawArguments::Indexed {
                    first_index: 0,
                    index_count: 3,
                    vertex_offset: 0,
                    first_instance: 0,
                    instance_count: 1,
                },
            ),
            Err(CommandDescriptionError::IndexBufferMismatch)
        );
        assert_eq!(
            DispatchOperation::new(PipelineId::new(2), vec![], [1, 0, 1]),
            Err(CommandDescriptionError::EmptyDispatch)
        );
    }

    #[test]
    fn render_pass_validates_all_attachments_before_construction() {
        let color = RenderAttachment {
            image: ImageId::new(1),
            subresources: image(1).subresources,
            kind: ImageKind::Color,
            format: ImageFormat::Rgba8Unorm,
            samples: SampleCount::One,
            load: AttachmentLoad::Clear(ClearValue::Color([0.0, 0.0, 0.0, 1.0])),
            store: AttachmentStore::Store,
        };
        let pass = RenderPassOperation::begin(
            RenderPassId::new(1),
            RenderPassDescription::new(vec![crate::RenderPassAttachmentDescription {
                kind: ImageKind::Color,
                format: ImageFormat::Rgba8Unorm,
                samples: SampleCount::One,
            }])
            .unwrap(),
            vec![color],
        )
        .unwrap();
        let operation = GpuOperation::new(
            GpuCommand::RenderPass(pass),
            [],
            [],
            CapabilityRequirements::none(),
        );
        assert_eq!(operation.accesses().len(), 1);
        assert_eq!(
            operation.accesses()[0].scope().usage(),
            ResourceUsage::ColorAttachment
        );

        let invalid = RenderAttachment {
            format: ImageFormat::Depth32Float,
            ..color
        };
        assert_eq!(
            RenderPassOperation::begin(
                RenderPassId::new(1),
                RenderPassDescription::new(vec![crate::RenderPassAttachmentDescription {
                    kind: ImageKind::Color,
                    format: ImageFormat::Rgba8Unorm,
                    samples: SampleCount::One,
                }])
                .unwrap(),
                vec![invalid]
            ),
            Err(CommandDescriptionError::AttachmentKindMismatch)
        );

        assert_eq!(
            RenderPassOperation::begin(
                RenderPassId::new(1),
                RenderPassDescription::new(vec![crate::RenderPassAttachmentDescription {
                    kind: ImageKind::Color,
                    format: ImageFormat::Bgra8Unorm,
                    samples: SampleCount::One,
                }])
                .unwrap(),
                vec![color]
            ),
            Err(CommandDescriptionError::AttachmentDescriptionMismatch)
        );
    }

    #[test]
    fn operation_capabilities_are_negotiated_as_one_sequence() {
        let clear = ClearOperation::image(
            image(1),
            ImageKind::Color,
            ImageFormat::Rgba8Unorm,
            SampleCount::One,
            ClearValue::Color([0.0; 4]),
        )
        .unwrap();
        let clear = GpuOperation::new(
            GpuCommand::Clear(clear),
            [],
            [],
            CapabilityRequirements::none(),
        );
        let dispatch = GpuOperation::new(
            GpuCommand::Dispatch(
                DispatchOperation::new(PipelineId::new(2), vec![], [2, 1, 1]).unwrap(),
            ),
            [],
            [],
            CapabilityRequirements::none(),
        );
        let submission = OperationSubmission::new(
            FrontendSubmissionId::new(3),
            vec![FrontendSubmissionId::new(2)],
            vec![clear, dispatch],
        )
        .unwrap();
        let capabilities = BackendCapabilities::new(
            BackendFeatures::CLEAR,
            [ImageFormat::Rgba8Unorm],
            [SampleCount::One],
            [],
            [],
            BackendLimits {
                max_color_attachments: 1,
                max_descriptor_bindings: 0,
                max_compute_workgroups: [0; 3],
            },
        );
        let error = capabilities
            .negotiate_all(&submission.capability_requirements())
            .unwrap_err();
        assert_eq!(error.operation_index(), 1);
        assert_eq!(submission.predecessors(), &[FrontendSubmissionId::new(2)]);
    }

    #[test]
    fn submission_ordering_rejects_self_and_duplicate_edges() {
        let copy = GpuOperation::new(
            GpuCommand::Copy(
                CopyOperation::buffer_to_buffer(buffer(1, 0, 4), buffer(2, 0, 4)).unwrap(),
            ),
            [],
            [],
            CapabilityRequirements::none(),
        );
        assert_eq!(
            OperationSubmission::new(
                FrontendSubmissionId::new(1),
                vec![FrontendSubmissionId::new(1)],
                vec![copy.clone()]
            ),
            Err(CommandDescriptionError::SelfDependency)
        );
        assert_eq!(
            OperationSubmission::new(
                FrontendSubmissionId::new(3),
                vec![FrontendSubmissionId::new(1), FrontendSubmissionId::new(1)],
                vec![copy]
            ),
            Err(CommandDescriptionError::DuplicateSubmissionDependency)
        );
    }
}
