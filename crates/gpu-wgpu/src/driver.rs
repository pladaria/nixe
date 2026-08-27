//! Persistent `wgpu` resource ownership, command lowering, and coherence.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nixe_gpu::{
    AcceptedBackendSubmission, AlphaCompareOperation, AlphaTest, AttachmentLoad, AttachmentStore,
    BackendDriver, BackendDriverError, BackendResourceCreateInfo, BackendResourceHandle,
    BackendSubmissionToken, BackingView, BlockLinearLayout, ClearOperation, ClearValue,
    CopyOperation, DepthCompareOperation, DepthState, DrawArguments, DrawOperation,
    GpuCacheConfiguration, GpuCommand, ImageDescription, ImageDimension, ImageFormat,
    ImageMemoryLayout, ImageOrigin, ImageRegion, ImageSubresourceRange, IndexType,
    PipelineDescription, PipelineKind, PresentationImageFormat, PresentationImageRequest,
    PrimitiveTopology, RenderAttachment, RenderPassOperation, ResidentImage,
    ResolvedBackendResources, ResourceDependency, SampleCount, ShaderStage, TriangleRasterization,
    VertexBufferLayout, VertexFormat, VertexStepMode, ViewportTransform,
};
use nixe_memory::{
    CanonicalCpuWriteOverlap, CanonicalCpuWriteRange, CanonicalPageId, ContentGeneration,
    CpuVisibilityRequest, VisibilityState,
};
use wgpu::util::StagingBelt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferDescriptor,
    BufferSize, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder,
    CommandEncoderDescriptor, CompareFunction, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, DepthStencilState, Device, ErrorFilter, ErrorScopeGuard, Extent3d,
    FragmentState, FrontFace, IndexFormat, LoadOp, MapMode, MultisampleState, Operations, Origin3d,
    PipelineCache, PipelineCompilationOptions, PolygonMode, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    StencilState, StoreOp, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo,
    Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, TextureViewDimension, VertexAttribute as WgpuVertexAttribute,
    VertexBufferLayout as WgpuVertexBufferLayout, VertexFormat as WgpuVertexFormat, VertexState,
    VertexStepMode as WgpuVertexStepMode,
};

use crate::{
    PIPELINE_CACHE_MAGIC, WgpuExecutionContext, WgpuQueueAccess, WgpuVisibilityCoordinator,
};

// WebGPU and Maxwell expose at most eight simultaneous color attachments.
const MAX_COLOR_ATTACHMENTS: usize = 8;

enum Resource {
    Allocation,
    Buffer {
        buffer: Buffer,
        view: Option<nixe_gpu::BufferView>,
    },
    Image {
        texture: Texture,
        description: ImageDescription,
        view: Option<nixe_gpu::ImageView>,
        attachment_views: HashMap<ImageSubresourceRange, wgpu::TextureView>,
    },
    Sampler {
        sampler: wgpu::Sampler,
    },
    Shader {
        module: ShaderModule,
        neutral: nixe_gpu::ShaderBackendModule,
    },
    Pipeline {
        description: PipelineDescription,
        render: RenderPipelineCache,
    },
    DescriptorTable {
        bindings: Box<[nixe_gpu::DescriptorTableBinding]>,
        bind_groups: HashMap<(u64, u32), CachedBindGroup>,
    },
    RenderPass,
    QueryPool,
}

struct ResourceRecord {
    immutable: BackendResourceCreateInfo,
    host: Option<Resource>,
    content: Option<ResourceContent>,
    last_use: Option<ResourceUse>,
    retired: bool,
    resident_bytes: u64,
}

struct WgpuResourceSlot {
    handle: BackendResourceHandle,
    record: ResourceRecord,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResourceUse {
    serial: u64,
    submission: BackendSubmissionToken,
}

#[derive(Clone, Copy, Default)]
struct UploadMark {
    epoch: u64,
    handle: Option<BackendResourceHandle>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PresentationImageKey {
    allocation: nixe_gpu::GpuAllocationId,
    allocation_offset: u64,
    width: u32,
    height: u32,
    format: PresentationImageFormat,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PresentationImportKey {
    image: PresentationImageKey,
    layout: ImageMemoryLayout,
    row_pitch: u32,
    source_size: u64,
}

impl From<&PresentationImageRequest> for PresentationImageKey {
    fn from(request: &PresentationImageRequest) -> Self {
        Self {
            allocation: request.backing.allocation(),
            allocation_offset: request.backing.allocation_offset(),
            width: request.width,
            height: request.height,
            format: request.format,
        }
    }
}

impl From<&PresentationImageRequest> for PresentationImportKey {
    fn from(request: &PresentationImageRequest) -> Self {
        Self {
            image: PresentationImageKey::from(request),
            layout: request.layout,
            row_pitch: request.row_pitch,
            source_size: request.backing.size(),
        }
    }
}

struct PresentationImport {
    source: Buffer,
    texture: Texture,
    bind_group: BindGroup,
    uploaded: Vec<ContentGeneration>,
    cpu_writes: Option<nixe_memory::CanonicalCpuWriteDependency>,
    initialized: bool,
    last_used: u64,
    resident_bytes: u64,
}

struct ResourceContent {
    uploaded: Vec<ContentGeneration>,
    initialized: bool,
    device_writes: Vec<DeviceWrite>,
}

struct HostSubmission {
    index: wgpu::SubmissionIndex,
    completed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceWrite {
    region: DeviceWriteRegion,
    serial: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceWriteRegion {
    Buffer(TransferRange),
    /// Images require layout conversion, so one immutable backing binding is
    /// the smallest exact transfer domain currently exposed by this backend.
    ImageBinding(usize),
}

#[derive(Clone, Copy)]
enum ResidencyCandidate {
    Resource(BackendResourceHandle),
    Presentation(PresentationImportKey),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TransferRange {
    offset: u64,
    size: u64,
}

const MAX_REUSABLE_READBACK_BUFFERS: usize = 8;
const MAX_REUSABLE_READBACK_BYTES: u64 = 64 * 1024 * 1024;
const UPLOAD_STAGING_CHUNK_BYTES: u64 = 4 * 1024 * 1024;
const MAX_UPLOAD_BYTES_PER_SUBMISSION: u64 = MAX_RESIDENT_RESOURCE_BYTES;
const MAX_DEVICE_WRITE_REGIONS: usize = 256;
const MAX_RESIDENT_RESOURCE_COUNT: usize = 4_096;
const MAX_RESIDENT_RESOURCE_BYTES: u64 = 512 * 1024 * 1024;

impl ResourceContent {
    fn new(info: &BackendResourceCreateInfo) -> Option<Self> {
        let uploaded = match info {
            BackendResourceCreateInfo::Buffer {
                view: Some(view), ..
            } => view
                .backing()
                .range()
                .segments()
                .iter()
                .map(|segment| segment.current_content_generation())
                .collect(),
            BackendResourceCreateInfo::Image {
                view: Some(view), ..
            } => view
                .bindings()
                .iter()
                .flat_map(|binding| binding.backing().range().segments())
                .map(|segment| segment.current_content_generation())
                .collect(),
            _ => return None,
        };
        Some(Self {
            uploaded,
            initialized: false,
            device_writes: Vec::new(),
        })
    }

    fn has_device_writes(&self) -> bool {
        !self.device_writes.is_empty()
    }
}

#[cfg(debug_assertions)]
impl RenderPipelineKey {
    fn new(
        vertex: BackendResourceHandle,
        fragment: BackendResourceHandle,
        color_format: ImageFormat,
        depth_format: Option<ImageFormat>,
        draw: &DrawOperation,
    ) -> Self {
        Self {
            vertex,
            fragment,
            topology: draw.prepared.topology,
            triangle_rasterization: draw.prepared.triangle_rasterization,
            alpha_test: draw.prepared.alpha_test,
            color_format,
            depth_format,
            depth_state: draw.prepared.depth_state,
            vertex_buffers: draw
                .prepared
                .vertex_buffers
                .iter()
                .map(VertexPipelineLayoutKey::new)
                .collect(),
        }
    }

    fn matches(
        &self,
        vertex: BackendResourceHandle,
        fragment: BackendResourceHandle,
        color_format: ImageFormat,
        depth_format: Option<ImageFormat>,
        draw: &DrawOperation,
    ) -> bool {
        self.vertex == vertex
            && self.fragment == fragment
            && self.topology == draw.prepared.topology
            && self.triangle_rasterization == draw.prepared.triangle_rasterization
            && self.alpha_test == draw.prepared.alpha_test
            && self.color_format == color_format
            && self.depth_format == depth_format
            && self.depth_state == draw.prepared.depth_state
            && self.vertex_buffers.len() == draw.prepared.vertex_buffers.len()
            && self
                .vertex_buffers
                .iter()
                .zip(draw.prepared.vertex_buffers.iter())
                .all(|(cached, current)| cached.matches(current))
    }
}

struct RenderPipelineFingerprintInput<'a> {
    vertex: BackendResourceHandle,
    fragment: BackendResourceHandle,
    topology: PrimitiveTopology,
    triangle_rasterization: TriangleRasterization,
    alpha_test: Option<AlphaTest>,
    color_format: ImageFormat,
    depth_format: Option<ImageFormat>,
    depth_state: DepthState,
    vertex_buffers: &'a [VertexBufferLayout],
}

impl Hash for RenderPipelineFingerprintInput<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex.hash(state);
        self.fragment.hash(state);
        self.topology.hash(state);
        self.triangle_rasterization.hash(state);
        self.alpha_test.hash(state);
        self.color_format.hash(state);
        self.depth_format.hash(state);
        self.depth_state.hash(state);
        self.vertex_buffers.len().hash(state);
        for layout in self.vertex_buffers {
            let pulled = layout
                .attributes
                .iter()
                .any(|attribute| attribute.format.requires_vertex_pulling());
            pulled.then_some(layout.buffer.range.offset()).hash(state);
            layout.array_stride.hash(state);
            layout.step_mode.hash(state);
            layout.attributes.hash(state);
        }
    }
}

fn render_pipeline_fingerprint(
    vertex: BackendResourceHandle,
    fragment: BackendResourceHandle,
    color_format: ImageFormat,
    depth_format: Option<ImageFormat>,
    draw: &DrawOperation,
) -> u128 {
    nixe_gpu::cache_fingerprint(&RenderPipelineFingerprintInput {
        vertex,
        fragment,
        topology: draw.prepared.topology,
        triangle_rasterization: draw.prepared.triangle_rasterization,
        alpha_test: draw.prepared.alpha_test,
        color_format,
        depth_format,
        depth_state: draw.prepared.depth_state,
        vertex_buffers: &draw.prepared.vertex_buffers,
    })
}

#[cfg(debug_assertions)]
impl VertexPipelineLayoutKey {
    fn new(layout: &VertexBufferLayout) -> Self {
        let pulled = layout
            .attributes
            .iter()
            .any(|attribute| attribute.format.requires_vertex_pulling());
        Self {
            pulled_buffer_offset: pulled.then_some(layout.buffer.range.offset()),
            array_stride: layout.array_stride,
            step_mode: layout.step_mode,
            attributes: layout.attributes.clone(),
        }
    }

    fn matches(&self, layout: &VertexBufferLayout) -> bool {
        let pulled = layout
            .attributes
            .iter()
            .any(|attribute| attribute.format.requires_vertex_pulling());
        self.pulled_buffer_offset == pulled.then_some(layout.buffer.range.offset())
            && self.array_stride == layout.array_stride
            && self.step_mode == layout.step_mode
            && self.attributes == layout.attributes
    }
}

#[cfg(debug_assertions)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RenderPipelineKey {
    vertex: BackendResourceHandle,
    fragment: BackendResourceHandle,
    topology: PrimitiveTopology,
    triangle_rasterization: TriangleRasterization,
    alpha_test: Option<AlphaTest>,
    color_format: ImageFormat,
    depth_format: Option<ImageFormat>,
    depth_state: DepthState,
    vertex_buffers: Box<[VertexPipelineLayoutKey]>,
}

struct CachedRenderPipeline {
    identity: PreparedPipelineIdentity,
    #[cfg(debug_assertions)]
    key: RenderPipelineKey,
    pipeline: RenderPipeline,
    serial: u64,
    last_used: u64,
    vertex_pull_bind_groups: HashMap<u128, CachedVertexPullBindGroup>,
}

struct PreparedPipelineIdentity {
    draw: Arc<nixe_gpu::PreparedDraw>,
    vertex: BackendResourceHandle,
    fragment: BackendResourceHandle,
    color_format: ImageFormat,
    depth_format: Option<ImageFormat>,
}

impl PreparedPipelineIdentity {
    fn matches(
        &self,
        vertex: BackendResourceHandle,
        fragment: BackendResourceHandle,
        color_format: ImageFormat,
        depth_format: Option<ImageFormat>,
        draw: &DrawOperation,
    ) -> bool {
        Arc::ptr_eq(&self.draw, &draw.prepared)
            && self.vertex == vertex
            && self.fragment == fragment
            && self.color_format == color_format
            && self.depth_format == depth_format
    }
}

#[derive(Clone, Copy)]
struct RenderPipelineLocation {
    pipeline: BackendResourceHandle,
    vertex: BackendResourceHandle,
    fragment: BackendResourceHandle,
    color_format: ImageFormat,
    depth_format: Option<ImageFormat>,
    fingerprint: u128,
}

#[derive(Clone)]
struct PreparedRenderPipeline {
    location: RenderPipelineLocation,
    pipeline: RenderPipeline,
    serial: u64,
}

struct CurrentRenderPipeline {
    fingerprint: u128,
    record: CachedRenderPipeline,
}

#[derive(Default)]
struct RenderPipelineCache {
    records: HashMap<u128, CachedRenderPipeline>,
    current: Option<CurrentRenderPipeline>,
}

impl RenderPipelineCache {
    fn current_fingerprint(
        &self,
        vertex: BackendResourceHandle,
        fragment: BackendResourceHandle,
        color_format: ImageFormat,
        depth_format: Option<ImageFormat>,
        draw: &DrawOperation,
    ) -> Option<u128> {
        let current = self.current.as_ref()?;
        current
            .record
            .identity
            .matches(vertex, fragment, color_format, depth_format, draw)
            .then_some(current.fingerprint)
    }

    fn touch(&mut self, fingerprint: u128, last_used: u64) -> Option<(RenderPipeline, u64)> {
        if let Some(current) = self.current.as_mut()
            && current.fingerprint == fingerprint
        {
            current.record.last_used = last_used;
            return Some((current.record.pipeline.clone(), current.record.serial));
        }
        if let Some(current) = self.current.take() {
            self.records.insert(current.fingerprint, current.record);
        }
        let mut record = self.records.remove(&fingerprint)?;
        record.last_used = last_used;
        let result = (record.pipeline.clone(), record.serial);
        let current = CurrentRenderPipeline {
            fingerprint,
            record,
        };
        self.current = Some(current);
        Some(result)
    }

    #[cfg(debug_assertions)]
    fn current_record(&self) -> Option<&CachedRenderPipeline> {
        self.current.as_ref().map(|current| &current.record)
    }

    fn record_mut(&mut self, fingerprint: u128) -> Option<&mut CachedRenderPipeline> {
        if self
            .current
            .as_ref()
            .is_some_and(|current| current.fingerprint == fingerprint)
        {
            return self.current.as_mut().map(|current| &mut current.record);
        }
        self.records.get_mut(&fingerprint)
    }

    fn insert(
        &mut self,
        fingerprint: u128,
        pipeline: CachedRenderPipeline,
        capacity: usize,
    ) -> Option<(u128, u64)> {
        assert!(
            self.current.is_none() && !self.records.contains_key(&fingerprint),
            "duplicate WGPU pipeline fingerprint insertion"
        );
        let evicted = if self.records.len() == capacity {
            let evicted_fingerprint = self
                .records
                .iter()
                .min_by_key(|(_, pipeline)| pipeline.last_used)
                .map(|(fingerprint, _)| *fingerprint)
                .expect("configured WGPU pipeline cache is non-empty");
            let pipeline = self
                .records
                .remove(&evicted_fingerprint)
                .expect("selected WGPU pipeline LRU entry remains present");
            Some((evicted_fingerprint, pipeline.serial))
        } else {
            None
        };
        let current = CurrentRenderPipeline {
            fingerprint,
            record: pipeline,
        };
        self.current = Some(current);
        evicted
    }
}

fn least_recent_key<K, V>(records: &HashMap<K, V>, last_used: impl Fn(&V) -> u64) -> Option<K>
where
    K: Copy + Eq + Hash,
{
    records
        .iter()
        .min_by_key(|(_, value)| last_used(value))
        .map(|(key, _)| *key)
}

struct CachedBindGroup {
    bind_group: BindGroup,
    last_used: u64,
}

struct CachedVertexPullBindGroup {
    #[cfg(debug_assertions)]
    key: Box<[(u32, BackendResourceHandle)]>,
    bind_group: BindGroup,
    last_used: u64,
}

#[cfg(debug_assertions)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct VertexPipelineLayoutKey {
    pulled_buffer_offset: Option<u64>,
    array_stride: u64,
    step_mode: VertexStepMode,
    attributes: Box<[nixe_gpu::VertexAttribute]>,
}

enum PendingWriteback {
    Buffer {
        staging: Buffer,
        page: CanonicalPageId,
        page_offset: usize,
        staging_offset: usize,
        size: usize,
    },
    Image {
        staging: Buffer,
        backing: BackingView,
        host_row_pitch: u32,
        canonical_layout: ImageMemoryLayout,
        bytes_per_texel: usize,
        width: u32,
        height: u32,
        depth_or_layers: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DemandedBufferWriteback {
    handle: BackendResourceHandle,
    serial: u64,
    range: TransferRange,
    page_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DemandedWriteback {
    Buffer(DemandedBufferWriteback),
    Image {
        handle: BackendResourceHandle,
        binding: usize,
        serial: u64,
    },
}

impl DemandedWriteback {
    fn serial(self) -> u64 {
        match self {
            Self::Buffer(writeback) => writeback.serial,
            Self::Image { serial, .. } => serial,
        }
    }

    fn handle(self) -> BackendResourceHandle {
        match self {
            Self::Buffer(writeback) => writeback.handle,
            Self::Image { handle, .. } => handle,
        }
    }

    fn order_offset(self) -> u64 {
        match self {
            Self::Buffer(writeback) => writeback.range.offset,
            Self::Image { binding, .. } => u64::try_from(binding).unwrap_or(u64::MAX),
        }
    }
}

fn alpha_test_entry_point(alpha_test: Option<AlphaTest>) -> &'static str {
    match alpha_test.map(|test| test.comparison) {
        None => "main",
        Some(AlphaCompareOperation::Never) => "nixe_alpha_never",
        Some(AlphaCompareOperation::Less) => "nixe_alpha_less",
        Some(AlphaCompareOperation::Equal) => "nixe_alpha_equal",
        Some(AlphaCompareOperation::LessEqual) => "nixe_alpha_less_equal",
        Some(AlphaCompareOperation::Greater) => "nixe_alpha_greater",
        Some(AlphaCompareOperation::NotEqual) => "nixe_alpha_not_equal",
        Some(AlphaCompareOperation::GreaterEqual) => "nixe_alpha_greater_equal",
        Some(AlphaCompareOperation::Always) => "nixe_alpha_always",
    }
}

fn presentation_image_key(info: &BackendResourceCreateInfo) -> Option<PresentationImageKey> {
    let BackendResourceCreateInfo::Image {
        description,
        view: Some(view),
        ..
    } = info
    else {
        return None;
    };
    if description.dimension() != ImageDimension::Two || description.samples() != SampleCount::One {
        return None;
    }
    view.bindings()
        .iter()
        .enumerate()
        .find_map(|(binding, _)| presentation_binding_key(info, binding))
}

fn presentation_binding_key(
    info: &BackendResourceCreateInfo,
    binding: usize,
) -> Option<PresentationImageKey> {
    let BackendResourceCreateInfo::Image {
        description,
        view: Some(view),
        ..
    } = info
    else {
        return None;
    };
    let binding = view.bindings().get(binding)?;
    let subresources = binding.subresources();
    if subresources.plane != 0
        || subresources.mip_level != 0
        || subresources.base_layer != 0
        || subresources.layer_count == 0
    {
        return None;
    }
    let extent = description.extent();
    Some(PresentationImageKey {
        allocation: binding.backing().allocation(),
        allocation_offset: binding.backing().allocation_offset(),
        width: extent.width,
        height: extent.height,
        format: presentation_format(description.format())?,
    })
}

fn direct_presentation_key(request: &PresentationImageRequest) -> Option<PresentationImageKey> {
    let mut key = PresentationImageKey::from(request);
    key.format = match request.format {
        PresentationImageFormat::Rgba8 | PresentationImageFormat::Rgbx8 => {
            PresentationImageFormat::Rgba8
        }
        PresentationImageFormat::Bgra8 => PresentationImageFormat::Bgra8,
        PresentationImageFormat::Rgb565 | PresentationImageFormat::Rgba4444 => return None,
    };
    Some(key)
}

const fn presentation_bytes_per_texel(format: PresentationImageFormat) -> u32 {
    match format {
        PresentationImageFormat::Rgba8
        | PresentationImageFormat::Rgbx8
        | PresentationImageFormat::Bgra8 => 4,
        PresentationImageFormat::Rgb565 | PresentationImageFormat::Rgba4444 => 2,
    }
}

const fn presentation_format_code(format: PresentationImageFormat) -> u32 {
    match format {
        PresentationImageFormat::Rgba8 => 0,
        PresentationImageFormat::Rgbx8 => 1,
        PresentationImageFormat::Bgra8 => 2,
        PresentationImageFormat::Rgb565 => 3,
        PresentationImageFormat::Rgba4444 => 4,
    }
}

fn validate_presentation_request(
    request: &PresentationImageRequest,
) -> Result<(), BackendDriverError> {
    if request.width == 0 || request.height == 0 {
        return Err(unsupported("empty presentation image"));
    }
    let row_bytes = u64::from(request.width)
        .checked_mul(u64::from(presentation_bytes_per_texel(request.format)))
        .ok_or_else(|| unsupported("presentation row size overflow"))?;
    if u64::from(request.row_pitch) < row_bytes {
        return Err(unsupported(
            "presentation row pitch is smaller than one row",
        ));
    }
    let required = match request.layout {
        ImageMemoryLayout::PitchLinear {
            row_pitch,
            layer_stride,
        } => {
            if row_pitch != u64::from(request.row_pitch) {
                return Err(unsupported("presentation pitch-linear layouts disagree"));
            }
            let required = u64::from(request.height - 1)
                .checked_mul(row_pitch)
                .and_then(|offset| offset.checked_add(row_bytes))
                .ok_or_else(|| unsupported("presentation pitch-linear size overflow"))?;
            if layer_stride < required {
                return Err(unsupported("presentation pitch-linear layer is truncated"));
            }
            required
        }
        ImageMemoryLayout::BlockLinear(blocks) => {
            if blocks.block_width_log2 != 0
                || blocks.block_depth_log2 != 0
                || blocks.block_height_log2 > 5
                || !request.row_pitch.is_multiple_of(64)
            {
                return Err(unsupported("presentation block-linear layout"));
            }
            let block_height_gobs = 1_u64 << blocks.block_height_log2;
            let block_rows = 8 * block_height_gobs;
            let required = u64::from(request.height)
                .div_ceil(block_rows)
                .checked_mul(u64::from(request.row_pitch) / 64)
                .and_then(|blocks| blocks.checked_mul(512))
                .and_then(|bytes| bytes.checked_mul(block_height_gobs))
                .ok_or_else(|| unsupported("presentation block-linear size overflow"))?;
            if blocks.layer_stride < required {
                return Err(unsupported("presentation block-linear layer is truncated"));
            }
            required
        }
    };
    if required > request.backing.size() {
        return Err(unsupported("presentation backing is truncated"));
    }
    Ok(())
}

const fn presentation_format(format: ImageFormat) -> Option<PresentationImageFormat> {
    match format {
        ImageFormat::Rgba8Unorm | ImageFormat::Rgba8Srgb => Some(PresentationImageFormat::Rgba8),
        ImageFormat::Bgra8Unorm | ImageFormat::Bgra8Srgb => Some(PresentationImageFormat::Bgra8),
        _ => None,
    }
}

fn vertex_entry_point(
    uses_vertex_pulling: bool,
    triangle_rasterization: TriangleRasterization,
) -> &'static str {
    match (uses_vertex_pulling, triangle_rasterization) {
        (false, TriangleRasterization::Fill) => "main",
        (false, TriangleRasterization::FillRectangle) => "nixe_fill_rectangle",
        (true, TriangleRasterization::Fill) => "nixe_vertex_pull",
        (true, TriangleRasterization::FillRectangle) => "nixe_vertex_pull_fill_rectangle",
    }
}

/// Accelerated implementation retained behind [`nixe_gpu::Backend`].
pub(crate) struct WgpuBackendDriver {
    backend: nixe_gpu::BackendInstanceId,
    device: Device,
    queue: Queue,
    queue_access: WgpuQueueAccess,
    visibility: Arc<WgpuVisibilityCoordinator>,
    resources: Vec<Option<WgpuResourceSlot>>,
    presentation_images: HashMap<PresentationImageKey, Vec<BackendResourceHandle>>,
    presentation_imports: HashMap<PresentationImportKey, PresentationImport>,
    presentation_import_pipeline: Option<ComputePipeline>,
    submissions: HashMap<BackendSubmissionToken, HostSubmission>,
    completion_sender: std::sync::mpsc::Sender<BackendSubmissionToken>,
    completion_receiver: std::sync::mpsc::Receiver<BackendSubmissionToken>,
    next_use: u64,
    next_cache_use: u64,
    next_pipeline_serial: u64,
    upload_staging: StagingBelt,
    upload_bytes: u64,
    upload_canonical: Vec<u8>,
    upload_linear: Vec<u8>,
    cpu_dirty_ranges: Vec<CanonicalCpuWriteRange>,
    transfer_ranges: Vec<TransferRange>,
    uploaded_generations: Vec<ContentGeneration>,
    dirty_image_bindings: Vec<usize>,
    vertex_pull_binding_key: Vec<(u32, BackendResourceHandle)>,
    draw_bind_groups: Vec<Vec<BindGroup>>,
    draw_pipelines: Vec<PreparedRenderPipeline>,
    render_attachment_views: Vec<wgpu::TextureView>,
    uploaded_inputs: Vec<UploadMark>,
    upload_epoch: u64,
    readback_pool: Vec<Buffer>,
    readback_pool_bytes: u64,
    resident_resources: usize,
    resident_resource_bytes: u64,
    device_loss: Arc<Mutex<Option<Box<str>>>>,
    backend_error: Arc<Mutex<Option<Box<str>>>>,
    pipeline_cache: Option<PipelineCache>,
    pipeline_cache_path: Option<PathBuf>,
    cache_configuration: GpuCacheConfiguration,
    torn_down: bool,
}

impl WgpuBackendDriver {
    pub(crate) fn new(
        backend: nixe_gpu::BackendInstanceId,
        execution: WgpuExecutionContext,
        visibility: Arc<WgpuVisibilityCoordinator>,
        pipeline_cache: Option<PipelineCache>,
        pipeline_cache_path: Option<PathBuf>,
        cache_configuration: GpuCacheConfiguration,
    ) -> Self {
        let WgpuExecutionContext {
            device,
            queue,
            queue_access,
        } = execution;
        let device_loss = Arc::new(Mutex::new(None));
        let callback_state = Arc::clone(&device_loss);
        device.set_device_lost_callback(move |reason, message| {
            let mut state = callback_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *state = Some(format!("{reason:?}: {message}").into());
        });
        let backend_error = Arc::new(Mutex::new(None));
        let error_state = Arc::clone(&backend_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            *error_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(error.to_string().into_boxed_str());
        }));
        let (completion_sender, completion_receiver) = std::sync::mpsc::channel();
        let upload_staging = StagingBelt::new(device.clone(), UPLOAD_STAGING_CHUNK_BYTES);
        Self {
            backend,
            device,
            queue,
            queue_access,
            visibility,
            resources: Vec::new(),
            presentation_images: HashMap::new(),
            presentation_imports: HashMap::new(),
            presentation_import_pipeline: None,
            submissions: HashMap::new(),
            completion_sender,
            completion_receiver,
            next_use: 1,
            next_cache_use: 1,
            next_pipeline_serial: 1,
            upload_staging,
            upload_bytes: 0,
            upload_canonical: Vec::new(),
            upload_linear: Vec::new(),
            cpu_dirty_ranges: Vec::new(),
            transfer_ranges: Vec::new(),
            uploaded_generations: Vec::new(),
            dirty_image_bindings: Vec::new(),
            vertex_pull_binding_key: Vec::new(),
            draw_bind_groups: Vec::new(),
            draw_pipelines: Vec::new(),
            render_attachment_views: Vec::new(),
            uploaded_inputs: Vec::new(),
            upload_epoch: 0,
            readback_pool: Vec::new(),
            readback_pool_bytes: 0,
            resident_resources: 0,
            resident_resource_bytes: 0,
            device_loss,
            backend_error,
            pipeline_cache,
            pipeline_cache_path,
            cache_configuration,
            torn_down: false,
        }
    }

    fn require_device(&mut self) -> Result<(), BackendDriverError> {
        let loss = self
            .device_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(reason) = loss {
            self.clear_owned_state();
            return Err(BackendDriverError::device_lost(reason));
        }
        if let Some(error) = self
            .backend_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
        {
            return Err(BackendDriverError::failure(format!(
                "wgpu asynchronous validation failed: {error}"
            )));
        }
        if self.torn_down {
            Err(BackendDriverError::failure("wgpu backend is torn down"))
        } else {
            Ok(())
        }
    }

    fn drain_completions(&mut self) {
        while let Ok(token) = self.completion_receiver.try_recv() {
            if let Some(submission) = self.submissions.get_mut(&token) {
                submission.completed = true;
            }
        }
    }

    fn clear_owned_state(&mut self) {
        self.resources.clear();
        self.presentation_images.clear();
        self.presentation_imports.clear();
        self.presentation_import_pipeline = None;
        self.submissions.clear();
        self.readback_pool.clear();
        self.uploaded_inputs.clear();
        self.upload_epoch = 0;
        self.upload_staging = StagingBelt::new(self.device.clone(), UPLOAD_STAGING_CHUNK_BYTES);
        self.upload_canonical = Vec::new();
        self.upload_linear = Vec::new();
        self.cpu_dirty_ranges = Vec::new();
        self.transfer_ranges = Vec::new();
        self.uploaded_generations = Vec::new();
        self.dirty_image_bindings = Vec::new();
        self.vertex_pull_binding_key = Vec::new();
        self.draw_bind_groups = Vec::new();
        self.draw_pipelines = Vec::new();
        self.render_attachment_views = Vec::new();
        self.readback_pool_bytes = 0;
        self.resident_resources = 0;
        self.resident_resource_bytes = 0;
    }

    fn persist_pipeline_cache(&self) -> Result<(), BackendDriverError> {
        let (Some(cache), Some(path)) = (&self.pipeline_cache, &self.pipeline_cache_path) else {
            return Ok(());
        };
        let Some(data) = cache.get_data() else {
            return Ok(());
        };
        if data.len() as u64 > self.cache_configuration.persistent_pipeline_cache_bytes() {
            return Err(BackendDriverError::failure(format!(
                "wgpu pipeline cache exceeds its configured {} byte storage bound",
                self.cache_configuration.persistent_pipeline_cache_bytes()
            )));
        }
        let parent = path.parent().ok_or_else(|| {
            BackendDriverError::failure("wgpu pipeline cache path has no parent directory")
        })?;
        std::fs::create_dir_all(parent).map_err(|error| {
            BackendDriverError::failure(format!(
                "cannot create pipeline cache directory {}: {error}",
                parent.display()
            ))
        })?;
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = std::fs::File::create(&temporary).map_err(|error| {
            BackendDriverError::failure(format!(
                "cannot create pipeline cache {}: {error}",
                temporary.display()
            ))
        })?;
        use std::io::Write;
        file.write_all(PIPELINE_CACHE_MAGIC)
            .and_then(|()| file.write_all(&data))
            .and_then(|()| file.sync_all())
            .map_err(|error| {
                BackendDriverError::failure(format!(
                    "cannot write pipeline cache {}: {error}",
                    temporary.display()
                ))
            })?;
        #[cfg(target_os = "windows")]
        if let Err(error) = std::fs::remove_file(path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(BackendDriverError::failure(format!(
                "cannot replace pipeline cache {}: {error}",
                path.display()
            )));
        }
        std::fs::rename(&temporary, path).map_err(|error| {
            BackendDriverError::failure(format!(
                "cannot publish pipeline cache {}: {error}",
                path.display()
            ))
        })?;
        log::info!(
            "saved WGPU pipeline cache: path={} bytes={}",
            path.display(),
            data.len()
        );
        Ok(())
    }

    fn take_cache_use(&mut self) -> Result<u64, BackendDriverError> {
        let use_serial = self.next_cache_use;
        self.next_cache_use = self
            .next_cache_use
            .checked_add(1)
            .ok_or_else(|| BackendDriverError::failure("WGPU cache LRU sequence exhausted"))?;
        Ok(use_serial)
    }

    fn take_resource_use(&mut self) -> Result<u64, BackendDriverError> {
        let use_serial = self.next_use;
        self.next_use = self
            .next_use
            .checked_add(1)
            .ok_or_else(|| BackendDriverError::failure("wgpu resource-use timeline exhausted"))?;
        Ok(use_serial)
    }

    fn capture_error_scope(&self, scope: ErrorScopeGuard) -> Result<(), BackendDriverError> {
        if let Some(error) = pollster::block_on(scope.pop()) {
            if let Some(reason) = self
                .device_loss
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
            {
                return Err(BackendDriverError::device_lost(reason));
            }
            return Err(BackendDriverError::failure(format!(
                "wgpu validation failed: {error}"
            )));
        }
        Ok(())
    }

    fn upload_inputs(
        &mut self,
        accepted: &AcceptedBackendSubmission<'_>,
        dependencies: &ResolvedBackendResources,
        encoder: &mut CommandEncoder,
    ) -> Result<(), BackendDriverError> {
        self.upload_epoch = match self.upload_epoch.checked_add(1) {
            Some(epoch) => epoch,
            None => {
                self.uploaded_inputs.fill(UploadMark::default());
                1
            }
        };
        for operation in accepted.submission().operations() {
            for access in operation.accesses() {
                if !access.scope().mode().reads() {
                    continue;
                }
                match access.target() {
                    nixe_gpu::AccessTarget::Buffer { buffer, .. } => {
                        let handle =
                            dependency_handle(dependencies, ResourceDependency::Buffer(buffer))?;
                        if self.mark_input_for_upload(handle)? {
                            self.upload_buffer(handle, encoder)?;
                        }
                    }
                    nixe_gpu::AccessTarget::Image { image, .. } => {
                        let handle =
                            dependency_handle(dependencies, ResourceDependency::Image(image))?;
                        if self.mark_input_for_upload(handle)? {
                            self.upload_image(handle, encoder)?;
                        }
                    }
                    nixe_gpu::AccessTarget::Queries { .. } => {}
                }
            }
        }
        Ok(())
    }

    fn mark_input_for_upload(
        &mut self,
        handle: BackendResourceHandle,
    ) -> Result<bool, BackendDriverError> {
        let slot = usize::try_from(handle.slot()).map_err(|_| missing(handle))?;
        if self.uploaded_inputs.len() <= slot {
            self.uploaded_inputs.resize(slot + 1, UploadMark::default());
        }
        let mark = &mut self.uploaded_inputs[slot];
        if mark.epoch == self.upload_epoch && mark.handle == Some(handle) {
            return Ok(false);
        }
        *mark = UploadMark {
            epoch: self.upload_epoch,
            handle: Some(handle),
        };
        Ok(true)
    }

    fn upload_buffer(
        &mut self,
        handle: BackendResourceHandle,
        encoder: &mut CommandEncoder,
    ) -> Result<(), BackendDriverError> {
        let (buffer, view, upload_all) = {
            let record = self.resource_record(handle)?;
            let Resource::Buffer {
                buffer,
                view: Some(view),
                ..
            } = record.host.as_ref().ok_or_else(|| missing(handle))?
            else {
                return Ok(());
            };
            let content = record
                .content
                .as_ref()
                .expect("a canonically backed buffer has content state");
            let range = view.backing().range();
            if range.segments().len() != content.uploaded.len() {
                return Err(BackendDriverError::failure(
                    "wgpu buffer content record does not match its immutable backing",
                ));
            }
            (buffer.clone(), view.clone(), !content.initialized)
        };
        let mut uploaded_generations = std::mem::take(&mut self.uploaded_generations);
        uploaded_generations.clear();
        uploaded_generations.extend_from_slice(
            &self
                .resource_record(handle)?
                .content
                .as_ref()
                .expect("a canonically backed buffer has content state")
                .uploaded,
        );
        self.transfer_ranges.clear();
        if upload_all {
            self.transfer_ranges.push(TransferRange {
                offset: 0,
                size: view.size(),
            });
        } else {
            collect_dirty_buffer_ranges(
                view.backing().range(),
                &uploaded_generations,
                view.buffer_offset(),
                &mut self.cpu_dirty_ranges,
                &mut self.transfer_ranges,
            )?;
        }
        self.uploaded_generations = uploaded_generations;
        if self.transfer_ranges.is_empty() {
            return Ok(());
        }
        let ranges = std::mem::take(&mut self.transfer_ranges);
        let mut upload_canonical = std::mem::take(&mut self.upload_canonical);
        for range in &ranges {
            let size = usize_from_u64(range.size, "buffer upload size")?;
            upload_canonical.resize(size, 0);
            view.backing()
                .range()
                .read(range.offset, &mut upload_canonical)
                .map_err(|error| BackendDriverError::failure(error.to_string()))?;
            self.stage_buffer_upload(
                encoder,
                &buffer,
                view.buffer_offset() + range.offset,
                upload_canonical.as_slice(),
            )?;
        }
        self.upload_canonical = upload_canonical;
        let record = self.resource_record_mut(handle)?;
        let content = record
            .content
            .as_mut()
            .expect("a canonically backed buffer has content state");
        record_uploaded([view.backing().range()], &mut content.uploaded);
        content.initialized = true;
        for range in &ranges {
            subtract_buffer_write_range(&mut content.device_writes, *range, None);
        }
        self.transfer_ranges = ranges;
        Ok(())
    }

    fn upload_image(
        &mut self,
        handle: BackendResourceHandle,
        encoder: &mut CommandEncoder,
    ) -> Result<(), BackendDriverError> {
        let (texture, description, view, initialized) = {
            let record = self.resource_record(handle)?;
            let Resource::Image {
                texture,
                description,
                view: Some(view),
                ..
            } = record.host.as_ref().ok_or_else(|| missing(handle))?
            else {
                return Ok(());
            };
            let content = record
                .content
                .as_ref()
                .expect("a canonically backed image has content state");
            (
                texture.clone(),
                *description,
                view.clone(),
                content.initialized,
            )
        };
        let mut uploaded_generations = std::mem::take(&mut self.uploaded_generations);
        uploaded_generations.clear();
        uploaded_generations.extend_from_slice(
            &self
                .resource_record(handle)?
                .content
                .as_ref()
                .expect("a canonically backed image has content state")
                .uploaded,
        );
        let mut dirty_bindings = std::mem::take(&mut self.dirty_image_bindings);
        dirty_bindings.clear();
        collect_dirty_image_bindings(
            &view,
            &uploaded_generations,
            initialized,
            &mut dirty_bindings,
        )?;
        self.uploaded_generations = uploaded_generations;
        if dirty_bindings.is_empty() {
            self.dirty_image_bindings = dirty_bindings;
            return Ok(());
        }
        for binding_index in dirty_bindings.iter().copied() {
            let binding = &view.bindings()[binding_index];
            let subresources = binding.subresources();
            let extent = description
                .mip_extent(subresources.mip_level)
                .ok_or_else(|| unsupported("invalid image upload mip"))?;
            self.upload_canonical.resize(
                usize_from_u64(binding.backing().size(), "image upload size")?,
                0,
            );
            binding
                .backing()
                .range()
                .read(0, &mut self.upload_canonical)
                .map_err(|error| BackendDriverError::failure(error.to_string()))?;
            let bytes_per_texel = usize::from(
                description
                    .format()
                    .plane_bytes_per_texel(subresources.plane)
                    .ok_or_else(|| unsupported("image plane format"))?,
            );
            let host_row_pitch = align_u32(
                extent
                    .width
                    .checked_mul(u32::try_from(bytes_per_texel).unwrap())
                    .ok_or_else(|| unsupported("image upload row size"))?,
                wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
            )?;
            linearize_canonical_image_into(
                &self.upload_canonical,
                &mut self.upload_linear,
                binding.layout(),
                ImageCopyShape {
                    width: extent.width,
                    height: extent.height,
                    layers: u32::from(subresources.layer_count),
                    bytes_per_texel,
                    host_row_pitch,
                },
            )?;
            let upload_linear = std::mem::take(&mut self.upload_linear);
            self.stage_texture_upload(
                encoder,
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: u32::from(subresources.mip_level),
                    origin: Origin3d {
                        x: 0,
                        y: 0,
                        z: u32::from(subresources.base_layer),
                    },
                    aspect: TextureAspect::All,
                },
                upload_linear.as_slice(),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(host_row_pitch),
                    rows_per_image: Some(extent.height),
                },
                Extent3d {
                    width: extent.width,
                    height: extent.height,
                    depth_or_array_layers: u32::from(subresources.layer_count),
                },
            )?;
            self.upload_linear = upload_linear;
        }
        let record = self.resource_record_mut(handle)?;
        let content = record
            .content
            .as_mut()
            .expect("a canonically backed image has content state");
        for binding in dirty_bindings.iter().copied() {
            record_image_binding_uploaded(&view, binding, &mut content.uploaded);
        }
        content.initialized = true;
        content.device_writes.retain(|write| {
            let DeviceWriteRegion::ImageBinding(binding) = write.region else {
                return true;
            };
            !dirty_bindings.contains(&binding)
        });
        self.dirty_image_bindings = dirty_bindings;
        Ok(())
    }

    fn encode_submission(
        &mut self,
        accepted: &AcceptedBackendSubmission<'_>,
        dependencies: &ResolvedBackendResources,
        mut encoder: CommandEncoder,
    ) -> Result<CommandEncoder, BackendDriverError> {
        let operations = accepted.submission().operations();
        let mut index = 0;
        while index < operations.len() {
            match operations[index].command() {
                GpuCommand::Copy(copy) => self.encode_copy(&mut encoder, dependencies, copy)?,
                GpuCommand::Clear(clear) => self.encode_clear(&mut encoder, dependencies, clear)?,
                GpuCommand::RenderPass(RenderPassOperation::Begin { .. }) => {
                    let end = operations[index + 1..]
                        .iter()
                        .position(|operation| {
                            matches!(
                                operation.command(),
                                GpuCommand::RenderPass(RenderPassOperation::End { .. })
                            )
                        })
                        .map(|offset| index + 1 + offset)
                        .ok_or_else(|| unsupported("unterminated render pass"))?;
                    self.encode_render_pass(&mut encoder, dependencies, operations, index, end)?;
                    index = end;
                }
                GpuCommand::RenderPass(RenderPassOperation::End { .. }) => {
                    return Err(unsupported("render-pass end without begin"));
                }
                GpuCommand::Barrier(_) | GpuCommand::CacheMaintenance(_) => {
                    // `wgpu` tracks usages and inserts host barriers. Keeping these
                    // commands in sequence preserves the neutral ordering boundary.
                }
                GpuCommand::Draw(_) => return Err(unsupported("draw outside render pass")),
                GpuCommand::Dispatch(_) => {
                    return Err(unsupported("compute dispatch pipeline binding"));
                }
                GpuCommand::Query(_) => return Err(unsupported("query command")),
            }
            index += 1;
        }
        Ok(encoder)
    }

    fn stage_buffer_upload(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), BackendDriverError> {
        let size = u64::try_from(bytes.len()).map_err(|_| unsupported("buffer upload size"))?;
        self.reserve_upload_bytes(size)?;
        let size = BufferSize::new(size).ok_or_else(|| unsupported("empty buffer upload"))?;
        let mut staging = self
            .upload_staging
            .write_buffer(encoder, target, offset, size);
        staging.copy_from_slice(bytes);
        Ok(())
    }

    fn stage_texture_upload(
        &mut self,
        encoder: &mut CommandEncoder,
        target: TexelCopyTextureInfo<'_>,
        bytes: &[u8],
        layout: TexelCopyBufferLayout,
        extent: Extent3d,
    ) -> Result<(), BackendDriverError> {
        let size = u64::try_from(bytes.len()).map_err(|_| unsupported("image upload size"))?;
        self.reserve_upload_bytes(size)?;
        let size = BufferSize::new(size).ok_or_else(|| unsupported("empty image upload"))?;
        let staging = self.upload_staging.allocate(
            size,
            BufferSize::new(256).expect("WGPU texture-copy alignment is non-zero"),
        );
        staging
            .get_mapped_range_mut()
            .map_err(|error| BackendDriverError::failure(error.to_string()))?
            .copy_from_slice(bytes);
        encoder.copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: staging.buffer(),
                layout: TexelCopyBufferLayout {
                    offset: staging.offset(),
                    ..layout
                },
            },
            target,
            extent,
        );
        Ok(())
    }

    fn reserve_upload_bytes(&mut self, size: u64) -> Result<(), BackendDriverError> {
        self.upload_bytes = self
            .upload_bytes
            .checked_add(size)
            .filter(|bytes| *bytes <= MAX_UPLOAD_BYTES_PER_SUBMISSION)
            .ok_or_else(|| {
                BackendDriverError::failure("one submission exceeds upload storage budget")
            })?;
        Ok(())
    }

    fn encode_copy(
        &self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources,
        copy: &CopyOperation,
    ) -> Result<(), BackendDriverError> {
        match copy {
            CopyOperation::BufferToBuffer {
                source,
                destination,
            } => {
                let source_buffer = self.buffer(dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(source.buffer),
                )?)?;
                let destination_buffer = self.buffer(dependency_handle(
                    dependencies,
                    ResourceDependency::Buffer(destination.buffer),
                )?)?;
                encoder.copy_buffer_to_buffer(
                    source_buffer,
                    source.range.offset(),
                    destination_buffer,
                    destination.range.offset(),
                    source.range.size(),
                );
                Ok(())
            }
            CopyOperation::BufferToImage { .. }
            | CopyOperation::ImageToBuffer { .. }
            | CopyOperation::ImageToImage { .. } => {
                Err(unsupported("non-buffer neutral copy layout"))
            }
        }
    }

    fn encode_clear(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources,
        clear: &ClearOperation,
    ) -> Result<(), BackendDriverError> {
        match clear {
            ClearOperation::Buffer { target, value } => {
                let buffer = self
                    .buffer(dependency_handle(
                        dependencies,
                        ResourceDependency::Buffer(target.buffer),
                    )?)?
                    .clone();
                let ClearValue::Buffer(value) = value else {
                    unreachable!();
                };
                if *value == 0 {
                    encoder.clear_buffer(&buffer, target.range.offset(), Some(target.range.size()));
                } else {
                    if !target.range.size().is_multiple_of(4) {
                        return Err(unsupported("unaligned non-zero buffer clear"));
                    }
                    let mut pattern = std::mem::take(&mut self.upload_canonical);
                    pattern.resize(usize_from_u64(target.range.size(), "buffer clear size")?, 0);
                    for word in pattern.chunks_exact_mut(4) {
                        word.copy_from_slice(&value.to_le_bytes());
                    }
                    self.stage_buffer_upload(encoder, &buffer, target.range.offset(), &pattern)?;
                    self.upload_canonical = pattern;
                }
                Ok(())
            }
            ClearOperation::Image {
                target,
                kind,
                value,
                ..
            } => {
                let handle =
                    dependency_handle(dependencies, ResourceDependency::Image(target.image))?;
                let Resource::Image {
                    texture,
                    description,
                    ..
                } = self.resource(handle)?
                else {
                    return Err(kind_mismatch(handle));
                };
                require_full_image_region(*description, *target)?;
                let view = texture.create_view(&texture_view_descriptor(target.subresources));
                match (kind, value) {
                    (nixe_gpu::ImageKind::Color, ClearValue::Color(color)) => {
                        let attachments = [Some(RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(color_value(*color)),
                                store: StoreOp::Store,
                            },
                            depth_slice: None,
                        })];
                        encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Some("Nixe image clear"),
                            color_attachments: &attachments,
                            ..Default::default()
                        });
                    }
                    (nixe_gpu::ImageKind::DepthStencil, value) => {
                        let (depth_ops, stencil_ops) = match value {
                            ClearValue::Depth(depth) => (
                                Some(Operations {
                                    load: LoadOp::Clear(*depth),
                                    store: StoreOp::Store,
                                }),
                                None,
                            ),
                            ClearValue::Stencil(stencil) => (
                                None,
                                Some(Operations {
                                    load: LoadOp::Clear(u32::from(*stencil)),
                                    store: StoreOp::Store,
                                }),
                            ),
                            ClearValue::DepthStencil { depth, stencil } => (
                                Some(Operations {
                                    load: LoadOp::Clear(*depth),
                                    store: StoreOp::Store,
                                }),
                                Some(Operations {
                                    load: LoadOp::Clear(u32::from(*stencil)),
                                    store: StoreOp::Store,
                                }),
                            ),
                            _ => return Err(unsupported("depth-stencil clear value")),
                        };
                        encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Some("Nixe depth-stencil clear"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                                view: &view,
                                depth_ops,
                                stencil_ops,
                            }),
                            ..Default::default()
                        });
                    }
                    _ => return Err(unsupported("image clear value")),
                }
                Ok(())
            }
        }
    }

    fn encode_render_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources,
        operations: &[nixe_gpu::GpuOperation],
        begin: usize,
        end: usize,
    ) -> Result<(), BackendDriverError> {
        let GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) =
            operations[begin].command()
        else {
            unreachable!();
        };
        let mut draw_pipelines = std::mem::take(&mut self.draw_pipelines);
        draw_pipelines.clear();
        for (operation_index, operation) in operations[begin + 1..end].iter().enumerate() {
            let operation_index = begin + 1 + operation_index;
            if let GpuCommand::Draw(draw) = operation.command() {
                let location = self.render_pipeline_location(
                    dependencies,
                    operation_index,
                    attachments,
                    draw,
                )?;
                draw_pipelines.push(self.ensure_render_pipeline(
                    dependencies,
                    operation_index,
                    draw,
                    location,
                )?);
            } else if !matches!(operation.command(), GpuCommand::Barrier(_)) {
                return Err(unsupported("non-draw command inside render pass"));
            }
        }

        let mut views = std::mem::take(&mut self.render_attachment_views);
        views.clear();
        for attachment in attachments.iter() {
            views.push(self.attachment_view(dependencies, *attachment)?);
        }
        let mut draw_bind_groups = std::mem::take(&mut self.draw_bind_groups);
        {
            let mut color_attachments: [Option<RenderPassColorAttachment<'_>>;
                MAX_COLOR_ATTACHMENTS] = std::array::from_fn(|_| None);
            let mut color_attachment_count = 0;
            for (attachment, view) in attachments.iter().zip(&views) {
                if attachment.kind == nixe_gpu::ImageKind::Color {
                    let slot = color_attachments
                        .get_mut(color_attachment_count)
                        .ok_or_else(|| unsupported("too many color attachments"))?;
                    *slot = Some(RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: color_operations(attachment)?,
                        depth_slice: None,
                    });
                    color_attachment_count += 1;
                }
            }
            let depth_index = attachments
                .iter()
                .position(|attachment| attachment.kind == nixe_gpu::ImageKind::DepthStencil);
            let depth_attachment = depth_index
                .map(|index| depth_operations(&views[index], &attachments[index]))
                .transpose()?;
            let mut draw_count = 0;
            for operation in &operations[begin + 1..end] {
                let GpuCommand::Draw(draw) = operation.command() else {
                    continue;
                };
                if draw_count == draw_bind_groups.len() {
                    draw_bind_groups.push(Vec::new());
                }
                draw_bind_groups[draw_count].clear();
                self.create_draw_bind_groups(
                    dependencies,
                    draw,
                    &draw_pipelines[draw_count],
                    &mut draw_bind_groups[draw_count],
                )?;
                draw_count += 1;
            }
            draw_bind_groups.truncate(draw_count);
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Nixe neutral render pass"),
                color_attachments: &color_attachments[..color_attachment_count],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });
            let mut draw_index = 0;
            for operation in &operations[begin + 1..end] {
                let GpuCommand::Draw(draw) = operation.command() else {
                    continue;
                };
                pass.set_pipeline(&draw_pipelines[draw_index].pipeline);
                for (slot, layout) in draw.prepared.vertex_buffers.iter().enumerate() {
                    if !layout
                        .attributes
                        .iter()
                        .any(|attribute| !attribute.format.requires_vertex_pulling())
                    {
                        continue;
                    }
                    let buffer = self.buffer(dependency_handle(
                        dependencies,
                        ResourceDependency::Buffer(layout.buffer.buffer),
                    )?)?;
                    pass.set_vertex_buffer(
                        u32::try_from(slot)
                            .map_err(|_| unsupported("vertex buffer slot overflow"))?,
                        buffer.slice(layout.buffer.range.offset()..layout.buffer.range.end()),
                    );
                }
                for (group, bind_group) in draw_bind_groups[draw_index].iter().enumerate() {
                    pass.set_bind_group(
                        u32::try_from(group)
                            .map_err(|_| unsupported("descriptor-table group overflow"))?,
                        bind_group,
                        &[],
                    );
                }
                if let Some(viewport) = draw.prepared.viewport_transform {
                    let viewport = webgpu_viewport(viewport)?;
                    pass.set_viewport(
                        viewport.x,
                        viewport.y,
                        viewport.width,
                        viewport.height,
                        viewport.min_depth,
                        viewport.max_depth,
                    );
                }
                match draw.arguments {
                    DrawArguments::NonIndexed {
                        first_vertex,
                        vertex_count,
                        first_instance,
                        instance_count,
                    } => {
                        let (first_vertex, vertex_count) =
                            match draw.prepared.triangle_rasterization {
                                TriangleRasterization::Fill => (first_vertex, vertex_count),
                                TriangleRasterization::FillRectangle => (
                                    first_vertex.checked_mul(2).ok_or_else(|| {
                                        unsupported("fill-rectangle first vertex overflow")
                                    })?,
                                    vertex_count.checked_mul(2).ok_or_else(|| {
                                        unsupported("fill-rectangle vertex count overflow")
                                    })?,
                                ),
                            };
                        pass.draw(
                            first_vertex..first_vertex + vertex_count,
                            first_instance..first_instance + instance_count,
                        );
                    }
                    DrawArguments::Indexed {
                        first_index,
                        index_count,
                        vertex_offset,
                        first_instance,
                        instance_count,
                    } => {
                        let (region, index_type) = draw
                            .prepared
                            .index_buffer
                            .ok_or_else(|| unsupported("missing index buffer"))?;
                        let buffer = self.buffer(dependency_handle(
                            dependencies,
                            ResourceDependency::Buffer(region.buffer),
                        )?)?;
                        let format = match index_type {
                            IndexType::Uint16 => IndexFormat::Uint16,
                            IndexType::Uint32 => IndexFormat::Uint32,
                            IndexType::Uint8 => return Err(unsupported("8-bit index buffer")),
                        };
                        pass.set_index_buffer(
                            buffer.slice(region.range.offset()..region.range.end()),
                            format,
                        );
                        pass.draw_indexed(
                            first_index..first_index + index_count,
                            vertex_offset,
                            first_instance..first_instance + instance_count,
                        );
                    }
                }
                draw_index += 1;
            }
            drop(pass);
        }
        self.render_attachment_views = views;
        for groups in &mut draw_bind_groups {
            groups.clear();
        }
        self.draw_bind_groups = draw_bind_groups;
        self.draw_pipelines = draw_pipelines;
        Ok(())
    }

    fn create_draw_bind_groups(
        &mut self,
        dependencies: &ResolvedBackendResources,
        draw: &DrawOperation,
        prepared: &PreparedRenderPipeline,
        groups: &mut Vec<BindGroup>,
    ) -> Result<(), BackendDriverError> {
        let pipeline_handle = prepared.location.pipeline;
        let pipeline_fingerprint = prepared.location.fingerprint;
        let pipeline = &prepared.pipeline;
        let pipeline_serial = prepared.serial;
        groups.reserve(draw.prepared.descriptor_tables.len() + 1);
        for (group, table) in draw.prepared.descriptor_tables.iter().enumerate() {
            let group =
                u32::try_from(group).map_err(|_| unsupported("descriptor-table group overflow"))?;
            let table_handle =
                dependency_handle(dependencies, ResourceDependency::DescriptorTable(*table))?;
            let cache_use = self.take_cache_use()?;
            let bind_group_capacity = self.cache_configuration.bind_groups_per_descriptor_table();
            let cached = {
                let record = self.resource_record_mut(table_handle)?;
                let Some(Resource::DescriptorTable { bind_groups, .. }) = record.host.as_mut()
                else {
                    return Err(kind_mismatch(table_handle));
                };
                bind_groups
                    .get_mut(&(pipeline_serial, group))
                    .map(|cached| {
                        cached.last_used = cache_use;
                        cached.bind_group.clone()
                    })
            };
            if let Some(cached) = cached {
                groups.push(cached);
                continue;
            }
            log::debug!(
                "WGPU bind-group cache miss: descriptor={table_handle} pipeline-serial={pipeline_serial} group={group}"
            );
            let bindings = {
                let record = self.resource_record(table_handle)?;
                let Some(Resource::DescriptorTable { bindings, .. }) = record.host.as_ref() else {
                    return Err(kind_mismatch(table_handle));
                };
                bindings.clone()
            };
            let image_views = bindings
                .iter()
                .filter_map(|binding| {
                    let ResourceDependency::Image(image) = binding.resource else {
                        return None;
                    };
                    Some(
                        dependency_handle(dependencies, ResourceDependency::Image(image)).and_then(
                            |handle| match self.resource(handle)? {
                                Resource::Image {
                                    texture,
                                    description,
                                    ..
                                } => Ok((
                                    binding.binding,
                                    texture.create_view(&sampled_texture_view_descriptor(
                                        *description,
                                    )),
                                )),
                                _ => Err(kind_mismatch(handle)),
                            },
                        ),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let entries = bindings
                .iter()
                .map(|binding| {
                    let handle = dependency_handle(dependencies, binding.resource)?;
                    let resource = match self.resource(handle)? {
                        Resource::Buffer { buffer, .. } => buffer.as_entire_binding(),
                        Resource::Image { .. } => BindingResource::TextureView(
                            &image_views
                                .iter()
                                .find(|(candidate, _)| *candidate == binding.binding)
                                .ok_or_else(|| unsupported("missing sampled-image view"))?
                                .1,
                        ),
                        Resource::Sampler { sampler } => BindingResource::Sampler(sampler),
                        _ => return Err(kind_mismatch(handle)),
                    };
                    Ok(BindGroupEntry {
                        binding: u32::from(binding.binding),
                        resource,
                    })
                })
                .collect::<Result<Vec<_>, BackendDriverError>>()?;
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Nixe neutral descriptor table"),
                layout: &pipeline.get_bind_group_layout(group),
                entries: &entries,
            });
            let record = self.resource_record_mut(table_handle)?;
            let Some(Resource::DescriptorTable { bind_groups, .. }) = record.host.as_mut() else {
                return Err(kind_mismatch(table_handle));
            };
            if bind_groups.len() == bind_group_capacity {
                let key = least_recent_key(bind_groups, |cached| cached.last_used)
                    .expect("configured WGPU bind-group cache is non-empty");
                bind_groups
                    .remove(&key)
                    .expect("selected WGPU bind-group LRU entry remains present");
                log::debug!(
                    "WGPU bind-group cache evicted LRU entry: descriptor={table_handle} pipeline-serial={} group={}",
                    key.0,
                    key.1
                );
            }
            bind_groups.insert(
                (pipeline_serial, group),
                CachedBindGroup {
                    bind_group: bind_group.clone(),
                    last_used: cache_use,
                },
            );
            groups.push(bind_group);
        }
        if draw.prepared.vertex_buffers.iter().any(|layout| {
            layout
                .attributes
                .iter()
                .any(|attribute| attribute.format.requires_vertex_pulling())
        }) {
            let group = u32::try_from(draw.prepared.descriptor_tables.len())
                .map_err(|_| unsupported("vertex-pull bind group overflow"))?;
            let mut key = std::mem::take(&mut self.vertex_pull_binding_key);
            key.clear();
            for (slot, layout) in
                draw.prepared
                    .vertex_buffers
                    .iter()
                    .enumerate()
                    .filter(|(_, layout)| {
                        layout
                            .attributes
                            .iter()
                            .any(|attribute| attribute.format.requires_vertex_pulling())
                    })
            {
                key.push((
                    u32::try_from(slot).map_err(|_| unsupported("vertex-pull binding overflow"))?,
                    dependency_handle(
                        dependencies,
                        ResourceDependency::Buffer(layout.buffer.buffer),
                    )?,
                ));
            }
            let fingerprint = nixe_gpu::cache_fingerprint(&key);
            let cache_use = self.take_cache_use()?;
            let cached = {
                let record = self.resource_record_mut(pipeline_handle)?;
                let Some(Resource::Pipeline { render, .. }) = record.host.as_mut() else {
                    return Err(kind_mismatch(pipeline_handle));
                };
                let cached = render
                    .record_mut(pipeline_fingerprint)
                    .expect("compiled render pipeline remains resident")
                    .vertex_pull_bind_groups
                    .get_mut(&fingerprint);
                if let Some(cached) = cached {
                    cached.last_used = cache_use;
                    #[cfg(debug_assertions)]
                    assert_eq!(
                        cached.key.as_ref(),
                        key.as_slice(),
                        "XXH3-128 collision or incomplete vertex-pull bind-group key"
                    );
                    Some(cached.bind_group.clone())
                } else {
                    None
                }
            };
            if let Some(bind_group) = cached {
                groups.push(bind_group);
                self.vertex_pull_binding_key = key;
                return Ok(());
            }
            log::debug!(
                "WGPU vertex-pull bind-group cache miss: pipeline-serial={pipeline_serial} fingerprint={fingerprint:032x}"
            );
            let entries = key
                .iter()
                .map(|(binding, handle)| {
                    Ok(BindGroupEntry {
                        binding: *binding,
                        resource: self.buffer(*handle)?.as_entire_binding(),
                    })
                })
                .collect::<Result<Vec<_>, BackendDriverError>>()?;
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Nixe vertex-pull buffers"),
                layout: &pipeline.get_bind_group_layout(group),
                entries: &entries,
            });
            let capacity = self.cache_configuration.bind_groups_per_descriptor_table();
            let record = self.resource_record_mut(pipeline_handle)?;
            let Some(Resource::Pipeline { render, .. }) = record.host.as_mut() else {
                return Err(kind_mismatch(pipeline_handle));
            };
            let cache = &mut render
                .record_mut(pipeline_fingerprint)
                .expect("compiled render pipeline remains resident")
                .vertex_pull_bind_groups;
            if cache.len() == capacity {
                let evicted = least_recent_key(cache, |cached| cached.last_used)
                    .expect("configured vertex-pull bind-group cache is non-empty");
                cache
                    .remove(&evicted)
                    .expect("selected vertex-pull LRU entry remains present");
                log::debug!(
                    "WGPU vertex-pull bind-group cache evicted LRU entry: pipeline-serial={pipeline_serial} fingerprint={evicted:032x}"
                );
            }
            cache.insert(
                fingerprint,
                CachedVertexPullBindGroup {
                    #[cfg(debug_assertions)]
                    key: key.clone().into_boxed_slice(),
                    bind_group: bind_group.clone(),
                    last_used: cache_use,
                },
            );
            groups.push(bind_group);
            self.vertex_pull_binding_key = key;
        }
        Ok(())
    }

    fn render_pipeline_location(
        &self,
        dependencies: &ResolvedBackendResources,
        operation: usize,
        attachments: &[RenderAttachment],
        draw: &DrawOperation,
    ) -> Result<RenderPipelineLocation, BackendDriverError> {
        let pipeline = dependency_handle(
            dependencies,
            ResourceDependency::Pipeline(draw.prepared.pipeline),
        )?;
        let vertex = shader_handle_for_stage(dependencies, operation, ShaderStage::Vertex)?;
        let fragment = shader_handle_for_stage(dependencies, operation, ShaderStage::Fragment)?;
        let color_format = attachments
            .iter()
            .find(|attachment| attachment.kind == nixe_gpu::ImageKind::Color)
            .ok_or_else(|| unsupported("graphics draw without color attachment"))?
            .format;
        let depth_format = attachments
            .iter()
            .find(|attachment| attachment.kind == nixe_gpu::ImageKind::DepthStencil)
            .map(|attachment| attachment.format);
        let fingerprint = match self.resource(pipeline)? {
            Resource::Pipeline { render, .. } => render
                .current_fingerprint(vertex, fragment, color_format, depth_format, draw)
                .unwrap_or_else(|| {
                    render_pipeline_fingerprint(vertex, fragment, color_format, depth_format, draw)
                }),
            _ => return Err(kind_mismatch(pipeline)),
        };
        Ok(RenderPipelineLocation {
            pipeline,
            vertex,
            fragment,
            color_format,
            depth_format,
            fingerprint,
        })
    }

    fn ensure_render_pipeline(
        &mut self,
        dependencies: &ResolvedBackendResources,
        operation: usize,
        draw: &DrawOperation,
        location: RenderPipelineLocation,
    ) -> Result<PreparedRenderPipeline, BackendDriverError> {
        let pipeline_handle = location.pipeline;
        let vertex_handle = location.vertex;
        let fragment_handle = location.fragment;
        let Resource::Pipeline { description, .. } = self.resource(pipeline_handle)? else {
            return Err(kind_mismatch(pipeline_handle));
        };
        if description.kind != PipelineKind::Graphics {
            return Err(unsupported("compute pipeline used for draw"));
        }
        let fingerprint = location.fingerprint;
        let cache_use = self.take_cache_use()?;
        let pipeline_variant_capacity = self.cache_configuration.pipeline_variants_per_resource();
        let cached = {
            let record = self.resource_record_mut(pipeline_handle)?;
            let Some(Resource::Pipeline { render, .. }) = record.host.as_mut() else {
                return Err(kind_mismatch(pipeline_handle));
            };
            let cached = render.touch(fingerprint, cache_use);
            #[cfg(debug_assertions)]
            if cached.is_some()
                && let Some(cached) = render.current_record()
            {
                assert!(
                    cached.key.matches(
                        vertex_handle,
                        fragment_handle,
                        location.color_format,
                        location.depth_format,
                        draw,
                    ),
                    "XXH3-128 collision or incomplete WGPU pipeline cache key"
                );
            }
            cached
        };
        if let Some((pipeline, serial)) = cached {
            return Ok(PreparedRenderPipeline {
                location,
                pipeline,
                serial,
            });
        }
        log::debug!(
            "WGPU pipeline cache miss; compiling host pipeline: neutral={pipeline_handle} fingerprint={fingerprint:032x}"
        );
        let (_, vertex, vertex_ir) =
            self.shader_for_stage(dependencies, operation, ShaderStage::Vertex)?;
        let (_, fragment, _) =
            self.shader_for_stage(dependencies, operation, ShaderStage::Fragment)?;
        let color_format = location.color_format;
        let depth_format = location.depth_format;
        #[cfg(debug_assertions)]
        let key = RenderPipelineKey::new(
            vertex_handle,
            fragment_handle,
            color_format,
            depth_format,
            draw,
        );
        let target = ColorTargetState {
            format: texture_format(color_format)
                .ok_or_else(|| unsupported("color attachment format"))?,
            blend: None,
            write_mask: ColorWrites::ALL,
        };
        let targets = [Some(target)];
        let depth_stencil = depth_format
            .map(|format| {
                Ok(DepthStencilState {
                    format: texture_format(format)
                        .ok_or_else(|| unsupported("depth attachment format"))?,
                    depth_write_enabled: Some(
                        draw.prepared.depth_state.test_enabled
                            && draw.prepared.depth_state.write_enabled,
                    ),
                    depth_compare: Some(if draw.prepared.depth_state.test_enabled {
                        compare_function(draw.prepared.depth_state.compare)
                    } else {
                        CompareFunction::Always
                    }),
                    stencil: StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                })
            })
            .transpose()?;
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let uses_vertex_pulling = draw.prepared.vertex_buffers.iter().any(|layout| {
            layout
                .attributes
                .iter()
                .any(|attribute| attribute.format.requires_vertex_pulling())
        });
        let vertex = if uses_vertex_pulling {
            let group = u32::try_from(draw.prepared.descriptor_tables.len())
                .map_err(|_| unsupported("vertex-pull bind group overflow"))?;
            let module = nixe_gpu::lower_shader_ir_to_wgsl_with_vertex_pulling(
                vertex_ir.ir(),
                &draw.prepared.vertex_buffers,
                group,
            )
            .map_err(|error| {
                BackendDriverError::failure(format!("vertex-input pulling failed: {error}"))
            })?;
            self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Nixe vertex-pulling shader"),
                source: ShaderSource::Wgsl(module.source().into()),
            })
        } else {
            vertex
        };
        let attribute_storage = draw
            .prepared
            .vertex_buffers
            .iter()
            .map(|layout| {
                layout
                    .attributes
                    .iter()
                    .filter(|attribute| !attribute.format.requires_vertex_pulling())
                    .map(|attribute| WgpuVertexAttribute {
                        format: vertex_format(attribute.format)
                            .expect("pulled formats were filtered before native vertex lowering"),
                        offset: attribute.offset,
                        shader_location: attribute.shader_location,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let vertex_buffers = draw
            .prepared
            .vertex_buffers
            .iter()
            .zip(&attribute_storage)
            .map(|(layout, attributes)| {
                (!attributes.is_empty()).then_some(WgpuVertexBufferLayout {
                    array_stride: layout.array_stride,
                    step_mode: match layout.step_mode {
                        VertexStepMode::Vertex => WgpuVertexStepMode::Vertex,
                        VertexStepMode::Instance => WgpuVertexStepMode::Instance,
                    },
                    attributes,
                })
            })
            .collect::<Vec<_>>();
        let alpha_constants = draw.prepared.alpha_test.map(|test| {
            [(
                "nixe_alpha_reference",
                f64::from(f32::from_bits(test.reference_bits)),
            )]
        });
        let pipeline = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Nixe neutral graphics pipeline"),
                layout: None,
                vertex: VertexState {
                    module: &vertex,
                    entry_point: Some(vertex_entry_point(
                        uses_vertex_pulling,
                        draw.prepared.triangle_rasterization,
                    )),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &vertex_buffers,
                },
                primitive: PrimitiveState {
                    topology: primitive_topology(draw.prepared.topology)?,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &fragment,
                    entry_point: Some(alpha_test_entry_point(draw.prepared.alpha_test)),
                    compilation_options: PipelineCompilationOptions {
                        constants: alpha_constants.as_ref().map_or(&[], |values| values),
                        ..PipelineCompilationOptions::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: self.pipeline_cache.as_ref(),
            });
        self.capture_error_scope(scope)?;
        let serial = self.next_pipeline_serial;
        self.next_pipeline_serial = self.next_pipeline_serial.checked_add(1).ok_or_else(|| {
            BackendDriverError::failure("wgpu compiled-pipeline identity exhausted")
        })?;
        let ResourceRecord {
            host: Some(Resource::Pipeline { render, .. }),
            ..
        } = self.resource_record_mut(pipeline_handle)?
        else {
            return Err(kind_mismatch(pipeline_handle));
        };
        let prepared_pipeline = pipeline.clone();
        let evicted = render.insert(
            fingerprint,
            CachedRenderPipeline {
                identity: PreparedPipelineIdentity {
                    draw: Arc::clone(&draw.prepared),
                    vertex: vertex_handle,
                    fragment: fragment_handle,
                    color_format,
                    depth_format,
                },
                #[cfg(debug_assertions)]
                key,
                pipeline,
                serial,
                last_used: cache_use,
                vertex_pull_bind_groups: HashMap::new(),
            },
            pipeline_variant_capacity,
        );
        if let Some((evicted_fingerprint, evicted_serial)) = evicted {
            log::debug!(
                "WGPU pipeline cache evicted LRU variant: neutral={pipeline_handle} serial={evicted_serial} fingerprint={evicted_fingerprint:032x}"
            );
        }
        Ok(PreparedRenderPipeline {
            location,
            pipeline: prepared_pipeline,
            serial,
        })
    }

    fn shader_for_stage(
        &self,
        dependencies: &ResolvedBackendResources,
        operation: usize,
        stage: ShaderStage,
    ) -> Result<
        (
            BackendResourceHandle,
            ShaderModule,
            nixe_gpu::ShaderBackendModule,
        ),
        BackendDriverError,
    > {
        let handle = shader_handle_for_stage(dependencies, operation, stage)?;
        let Resource::Shader { module, neutral } = self.resource(handle)? else {
            return Err(kind_mismatch(handle));
        };
        Ok((handle, module.clone(), neutral.clone()))
    }

    fn encode_buffer_writeback(
        &mut self,
        encoder: &mut CommandEncoder,
        writeback: DemandedBufferWriteback,
        page: CanonicalPageId,
        output: &mut Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        let (buffer, buffer_size, view) = match self.resource(writeback.handle)? {
            Resource::Buffer {
                buffer,
                view: Some(view),
                ..
            } => (buffer.clone(), buffer.size(), view.clone()),
            _ => return Ok(()),
        };
        let source = view
            .buffer_offset()
            .checked_add(writeback.range.offset)
            .ok_or_else(|| unsupported("buffer writeback offset overflow"))?;
        let source_end = source
            .checked_add(writeback.range.size)
            .ok_or_else(|| unsupported("buffer writeback range overflow"))?;
        let aligned_source = source / 4 * 4;
        let aligned_end = align_u64(source_end, 4)?;
        if aligned_end > buffer_size {
            return Err(unsupported(
                "buffer writeback cannot satisfy WGPU copy alignment",
            ));
        }
        if aligned_source >= aligned_end {
            return Err(unsupported("empty aligned buffer writeback"));
        }
        let copy_size = aligned_end - aligned_source;
        let staging = self.take_readback_buffer(copy_size, "Nixe buffer readback");
        encoder.copy_buffer_to_buffer(&buffer, aligned_source, &staging, 0, copy_size);
        output.push(PendingWriteback::Buffer {
            staging,
            page,
            page_offset: usize_from_u64(writeback.page_offset, "page writeback offset")?,
            staging_offset: usize_from_u64(source - aligned_source, "staging writeback offset")?,
            size: usize_from_u64(writeback.range.size, "buffer writeback size")?,
        });
        Ok(())
    }

    fn encode_image_writeback(
        &mut self,
        encoder: &mut CommandEncoder,
        handle: BackendResourceHandle,
        binding_index: usize,
        output: &mut Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        let (texture, description, view) = match self.resource(handle)? {
            Resource::Image {
                texture,
                description,
                view: Some(view),
                ..
            } => (texture.clone(), *description, view.clone()),
            _ => return Ok(()),
        };
        let binding = view
            .bindings()
            .get(binding_index)
            .ok_or_else(|| unsupported("missing image writeback binding"))?;
        let subresources = binding.subresources();
        let extent = description
            .mip_extent(subresources.mip_level)
            .ok_or_else(|| unsupported("invalid image writeback mip"))?;
        let bytes_per_texel = usize::from(
            description
                .format()
                .plane_bytes_per_texel(subresources.plane)
                .ok_or_else(|| unsupported("image plane format"))?,
        );
        let width_bytes = usize::try_from(extent.width)
            .ok()
            .and_then(|width| width.checked_mul(bytes_per_texel))
            .ok_or_else(|| unsupported("image row size overflow"))?;
        let host_row_pitch = align_u32(
            u32::try_from(width_bytes).map_err(|_| unsupported("image row size"))?,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        )?;
        let layers = u32::from(subresources.layer_count);
        let size = u64::from(host_row_pitch)
            .checked_mul(u64::from(extent.height))
            .and_then(|value| value.checked_mul(u64::from(layers)))
            .ok_or_else(|| unsupported("image writeback size overflow"))?;
        let staging = self.take_readback_buffer(size, "Nixe image readback");
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: u32::from(subresources.mip_level),
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: u32::from(subresources.base_layer),
                },
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(host_row_pitch),
                    rows_per_image: Some(extent.height),
                },
            },
            Extent3d {
                width: extent.width,
                height: extent.height,
                depth_or_array_layers: layers,
            },
        );
        output.push(PendingWriteback::Image {
            staging,
            backing: binding.backing().clone(),
            host_row_pitch,
            canonical_layout: binding.layout(),
            bytes_per_texel,
            width: extent.width,
            height: extent.height,
            depth_or_layers: layers,
        });
        Ok(())
    }

    fn finish_writebacks(
        &mut self,
        writebacks: Vec<PendingWriteback>,
    ) -> Result<(), BackendDriverError> {
        if writebacks.is_empty() {
            return Ok(());
        }
        let mut receivers = Vec::with_capacity(writebacks.len());
        for writeback in &writebacks {
            let staging = match &writeback {
                PendingWriteback::Buffer { staging, .. }
                | PendingWriteback::Image { staging, .. } => staging,
            };
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            staging.map_async(MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
            receivers.push(receiver);
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| BackendDriverError::device_lost(error.to_string()))?;
        for (writeback, receiver) in writebacks.into_iter().zip(receivers) {
            let staging = match &writeback {
                PendingWriteback::Buffer { staging, .. }
                | PendingWriteback::Image { staging, .. } => staging,
            };
            receiver
                .recv()
                .map_err(|_| BackendDriverError::device_lost("wgpu map callback was lost"))?
                .map_err(|error| BackendDriverError::failure(error.to_string()))?;
            let mapped = staging.get_mapped_range(..).map_err(|error| {
                BackendDriverError::failure(format!("wgpu readback mapping failed: {error}"))
            })?;
            match &writeback {
                PendingWriteback::Buffer {
                    page,
                    page_offset,
                    staging_offset,
                    size,
                    ..
                } => {
                    let end = staging_offset
                        .checked_add(*size)
                        .ok_or_else(|| unsupported("mapped buffer writeback range overflow"))?;
                    let bytes = mapped
                        .get(*staging_offset..end)
                        .ok_or_else(|| unsupported("mapped buffer writeback exceeds staging"))?;
                    self.visibility
                        .write_page_range(*page, *page_offset, bytes)
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                }
                PendingWriteback::Image {
                    backing,
                    host_row_pitch,
                    canonical_layout,
                    bytes_per_texel,
                    width,
                    height,
                    depth_or_layers,
                    ..
                } => {
                    let mut canonical =
                        vec![0; usize_from_u64(backing.size(), "image backing size",)?];
                    self.visibility
                        .read_backing(backing, &mut canonical)
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                    write_linear_image_to_canonical(
                        &mapped,
                        &mut canonical,
                        *canonical_layout,
                        ImageCopyShape {
                            width: *width,
                            height: *height,
                            layers: *depth_or_layers,
                            bytes_per_texel: *bytes_per_texel,
                            host_row_pitch: *host_row_pitch,
                        },
                    )?;
                    self.visibility
                        .write_backing(backing, &canonical)
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                }
            }
            drop(mapped);
            staging.unmap();
            let staging = match writeback {
                PendingWriteback::Buffer { staging, .. }
                | PendingWriteback::Image { staging, .. } => staging,
            };
            self.recycle_readback_buffer(staging);
        }
        Ok(())
    }

    fn take_readback_buffer(&mut self, size: u64, label: &'static str) -> Buffer {
        if let Some(index) = self
            .readback_pool
            .iter()
            .position(|buffer| buffer.size() == size)
        {
            let buffer = self.readback_pool.swap_remove(index);
            self.readback_pool_bytes -= size;
            return buffer;
        }
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn recycle_readback_buffer(&mut self, buffer: Buffer) {
        let size = buffer.size();
        if self.readback_pool.len() < MAX_REUSABLE_READBACK_BUFFERS
            && self
                .readback_pool_bytes
                .checked_add(size)
                .is_some_and(|bytes| bytes <= MAX_REUSABLE_READBACK_BYTES)
        {
            self.readback_pool.push(buffer);
            self.readback_pool_bytes += size;
        }
    }

    fn attachment_view(
        &mut self,
        dependencies: &ResolvedBackendResources,
        attachment: RenderAttachment,
    ) -> Result<wgpu::TextureView, BackendDriverError> {
        let handle = dependency_handle(dependencies, ResourceDependency::Image(attachment.image))?;
        let ResourceRecord {
            host:
                Some(Resource::Image {
                    texture,
                    attachment_views,
                    ..
                }),
            ..
        } = self.resource_record_mut(handle)?
        else {
            return Err(kind_mismatch(handle));
        };
        Ok(attachment_views
            .entry(attachment.subresources)
            .or_insert_with(|| {
                texture.create_view(&texture_view_descriptor(attachment.subresources))
            })
            .clone())
    }

    fn buffer(&self, handle: BackendResourceHandle) -> Result<&Buffer, BackendDriverError> {
        let Resource::Buffer { buffer, .. } = self.resource(handle)? else {
            return Err(kind_mismatch(handle));
        };
        Ok(buffer)
    }

    fn resource(&self, handle: BackendResourceHandle) -> Result<&Resource, BackendDriverError> {
        self.resource_record(handle)?
            .host
            .as_ref()
            .ok_or_else(|| missing(handle))
    }

    fn resource_record(
        &self,
        handle: BackendResourceHandle,
    ) -> Result<&ResourceRecord, BackendDriverError> {
        let index = usize::try_from(handle.slot()).map_err(|_| missing(handle))?;
        self.resources
            .get(index)
            .and_then(Option::as_ref)
            .filter(|slot| slot.handle == handle)
            .map(|slot| &slot.record)
            .ok_or_else(|| missing(handle))
    }

    fn resource_record_mut(
        &mut self,
        handle: BackendResourceHandle,
    ) -> Result<&mut ResourceRecord, BackendDriverError> {
        let index = usize::try_from(handle.slot()).map_err(|_| missing(handle))?;
        self.resources
            .get_mut(index)
            .and_then(Option::as_mut)
            .filter(|slot| slot.handle == handle)
            .map(|slot| &mut slot.record)
            .ok_or_else(|| missing(handle))
    }

    fn index_presentable_image(
        &mut self,
        handle: BackendResourceHandle,
        info: &BackendResourceCreateInfo,
    ) {
        let Some(key) = presentation_image_key(info) else {
            return;
        };
        self.presentation_images
            .entry(key)
            .or_default()
            .push(handle);
    }

    fn remove_resource_record(&mut self, handle: BackendResourceHandle) -> Option<ResourceRecord> {
        let index = usize::try_from(handle.slot()).ok()?;
        let slot = self.resources.get_mut(index)?.as_ref()?;
        if slot.handle != handle {
            return None;
        }
        let record = self.resources[index]
            .take()
            .expect("validated WGPU resource slot")
            .record;
        if let Some(key) = presentation_image_key(&record.immutable)
            && let Some(handles) = self.presentation_images.get_mut(&key)
        {
            handles.retain(|candidate| *candidate != handle);
            if handles.is_empty() {
                self.presentation_images.remove(&key);
            }
        }
        if record.host.is_some() {
            self.resident_resources -= 1;
            self.resident_resource_bytes -= record.resident_bytes;
        }
        Some(record)
    }

    fn remove_presentation_import(
        &mut self,
        key: PresentationImportKey,
    ) -> Option<PresentationImport> {
        let import = self.presentation_imports.remove(&key)?;
        self.resident_resources -= 1;
        self.resident_resource_bytes -= import.resident_bytes;
        Some(import)
    }

    fn acquire_presentable(
        &mut self,
        request: PresentationImageRequest,
    ) -> Result<ResidentImage, BackendDriverError> {
        self.require_device()?;
        validate_presentation_request(&request)?;
        let import_key = PresentationImportKey::from(&request);
        let cpu_contents_unchanged = self
            .presentation_imports
            .get(&import_key)
            .filter(|import| import.initialized)
            .and_then(|import| import.cpu_writes.as_ref())
            .map_or_else(
                || request.cpu_writes.remains_current(),
                nixe_memory::CanonicalCpuWriteDependency::remains_current,
            );
        if cpu_contents_unchanged
            && let Some(key) = direct_presentation_key(&request)
            && let Some((description, texture)) = self
                .presentation_images
                .get(&key)
                .into_iter()
                .flatten()
                .filter_map(|handle| {
                    let record = self.resource_record(*handle).ok()?;
                    let content = record.content.as_ref()?;
                    let newest_write = content
                        .device_writes
                        .iter()
                        .filter_map(|write| match write.region {
                            DeviceWriteRegion::ImageBinding(binding)
                                if presentation_binding_key(&record.immutable, binding)
                                    == Some(key) =>
                            {
                                Some(write.serial)
                            }
                            _ => None,
                        })
                        .max()?;
                    let Resource::Image {
                        texture,
                        description,
                        ..
                    } = record.host.as_ref()?
                    else {
                        return None;
                    };
                    Some((newest_write, *description, texture.clone()))
                })
                .max_by_key(|candidate| candidate.0)
                .map(|(_, description, texture)| (description, texture))
        {
            return Ok(ResidentImage::new(
                self.backend,
                description,
                Arc::new(crate::WgpuResidentImage::new(texture)),
            ));
        }

        let mut import = match self.presentation_imports.remove(&import_key) {
            Some(import) => import,
            None => self.create_presentation_import(&request)?,
        };
        let result = self.refresh_presentation_import(&request, &mut import);
        self.presentation_imports.insert(import_key, import);
        let texture = result?;
        let description = ImageDescription::new(
            ImageDimension::Two,
            nixe_gpu::ImageExtent {
                width: request.width,
                height: request.height,
                depth: 1,
            },
            ImageFormat::Rgba8Unorm,
            nixe_gpu::ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
        Ok(ResidentImage::new(
            self.backend,
            description,
            Arc::new(crate::WgpuResidentImage::new(texture)),
        ))
    }

    fn create_presentation_import(
        &mut self,
        request: &PresentationImageRequest,
    ) -> Result<PresentationImport, BackendDriverError> {
        let source_size = align_u64(request.backing.size(), 4)?;
        let output_size = u64::from(request.width)
            .checked_mul(u64::from(request.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| unsupported("presentation import size overflow"))?;
        let resident_bytes = source_size
            .checked_add(output_size)
            .and_then(|bytes| bytes.checked_add(32))
            .ok_or_else(|| unsupported("presentation import size overflow"))?;
        self.ensure_residency_budget(1, resident_bytes, None)?;
        let source = self.device.create_buffer(&BufferDescriptor {
            label: Some("Nixe presentation canonical source"),
            size: source_size,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Nixe imported presentation image"),
            size: Extent3d {
                width: request.width,
                height: request.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let (layout, block_height_log2) = match request.layout {
            ImageMemoryLayout::PitchLinear { .. } => (0, 0),
            ImageMemoryLayout::BlockLinear(blocks) => (1, u32::from(blocks.block_height_log2)),
        };
        let parameters = [
            request.width,
            request.height,
            request.row_pitch,
            u32::try_from(request.backing.size())
                .map_err(|_| unsupported("presentation source size"))?,
            presentation_format_code(request.format),
            layout,
            block_height_log2,
            presentation_bytes_per_texel(request.format),
        ];
        let mut parameter_bytes = Vec::with_capacity(parameters.len() * 4);
        for parameter in parameters {
            parameter_bytes.extend_from_slice(&parameter.to_ne_bytes());
        }
        let parameter_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Nixe presentation import parameters"),
            size: parameter_bytes.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        {
            let _queue_access = self.queue_access.lock();
            self.queue
                .write_buffer(&parameter_buffer, 0, &parameter_bytes);
        }
        let pipeline = self.presentation_import_pipeline()?;
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Nixe presentation import bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: source.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: parameter_buffer.as_entire_binding(),
                },
            ],
        });
        self.resident_resources += 1;
        self.resident_resource_bytes += resident_bytes;
        Ok(PresentationImport {
            source,
            texture,
            bind_group,
            uploaded: request
                .backing
                .range()
                .segments()
                .iter()
                .map(|segment| segment.current_content_generation())
                .collect(),
            cpu_writes: nixe_memory::CanonicalCpuWriteDependency::capture(request.backing.range()),
            initialized: false,
            last_used: 0,
            resident_bytes,
        })
    }

    fn presentation_import_pipeline(&mut self) -> Result<ComputePipeline, BackendDriverError> {
        if let Some(pipeline) = &self.presentation_import_pipeline {
            return Ok(pipeline.clone());
        }
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Nixe presentation import shader"),
            source: ShaderSource::Wgsl(include_str!("presentation_import.wgsl").into()),
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Nixe presentation import pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: self.pipeline_cache.as_ref(),
            });
        self.capture_error_scope(scope)?;
        self.presentation_import_pipeline = Some(pipeline.clone());
        Ok(pipeline)
    }

    fn refresh_presentation_import(
        &mut self,
        request: &PresentationImageRequest,
        import: &mut PresentationImport,
    ) -> Result<Texture, BackendDriverError> {
        import.last_used = self.take_resource_use()?;
        if import.initialized
            && import
                .cpu_writes
                .as_ref()
                .is_some_and(nixe_memory::CanonicalCpuWriteDependency::remains_current)
        {
            return Ok(import.texture.clone());
        }
        for segment in request.backing.range().segments() {
            match segment.visibility_state() {
                VisibilityState::Clean | VisibilityState::CpuNewer => {}
                VisibilityState::GpuNewer { .. } => {
                    return Err(unsupported(
                        "device-authored presentation source has no compatible resident image",
                    ));
                }
                VisibilityState::Conflicting => {
                    return Err(unsupported(
                        "presentation source has conflicting authorities",
                    ));
                }
                VisibilityState::Invalid => {
                    return Err(unsupported("presentation source visibility is invalid"));
                }
            }
        }
        self.transfer_ranges.clear();
        if !import.initialized {
            self.transfer_ranges.push(TransferRange {
                offset: 0,
                size: align_u64(request.backing.size(), 4)?,
            });
        } else if request.backing.size().is_multiple_of(4) {
            collect_dirty_buffer_ranges(
                request.backing.range(),
                &import.uploaded,
                0,
                &mut self.cpu_dirty_ranges,
                &mut self.transfer_ranges,
            )?;
        } else if ranges_need_upload([request.backing.range()], &import.uploaded)? {
            self.transfer_ranges.push(TransferRange {
                offset: 0,
                size: align_u64(request.backing.size(), 4)?,
            });
        }
        if self.transfer_ranges.is_empty() {
            import.cpu_writes =
                nixe_memory::CanonicalCpuWriteDependency::capture(request.backing.range());
            return Ok(import.texture.clone());
        }

        self.upload_bytes = 0;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Nixe presentation import"),
            });
        let ranges = std::mem::take(&mut self.transfer_ranges);
        let mut upload = std::mem::take(&mut self.upload_canonical);
        for range in &ranges {
            let size = usize_from_u64(range.size, "presentation upload size")?;
            upload.clear();
            upload.resize(size, 0);
            let available = request.backing.size().saturating_sub(range.offset);
            let read_size = range.size.min(available);
            request
                .backing
                .range()
                .read(
                    range.offset,
                    &mut upload[..usize_from_u64(read_size, "presentation source range")?],
                )
                .map_err(|error| BackendDriverError::failure(error.to_string()))?;
            self.stage_buffer_upload(&mut encoder, &import.source, range.offset, &upload)?;
        }
        self.upload_canonical = upload;
        let pipeline = self.presentation_import_pipeline()?;
        {
            let mut compute = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Nixe presentation image conversion"),
                timestamp_writes: None,
            });
            compute.set_pipeline(&pipeline);
            compute.set_bind_group(0, &import.bind_group, &[]);
            compute.dispatch_workgroups(request.width.div_ceil(8), request.height.div_ceil(8), 1);
        }
        self.upload_staging.finish_and_recall_on_submit(&encoder);
        {
            let _queue_access = self.queue_access.lock();
            self.queue.submit([encoder.finish()]);
        }
        record_uploaded([request.backing.range()], &mut import.uploaded);
        import.cpu_writes =
            nixe_memory::CanonicalCpuWriteDependency::capture(request.backing.range());
        import.initialized = true;
        self.transfer_ranges = ranges;
        Ok(import.texture.clone())
    }

    fn create_host_resource(
        &self,
        info: &BackendResourceCreateInfo,
    ) -> Result<Resource, BackendDriverError> {
        Ok(match info {
            BackendResourceCreateInfo::Allocation { .. } => Resource::Allocation,
            BackendResourceCreateInfo::Buffer {
                description, view, ..
            } => Resource::Buffer {
                buffer: self.device.create_buffer(&BufferDescriptor {
                    label: Some("Nixe neutral buffer"),
                    size: description.size(),
                    usage: BufferUsages::COPY_SRC
                        | BufferUsages::COPY_DST
                        | BufferUsages::VERTEX
                        | BufferUsages::INDEX
                        | BufferUsages::UNIFORM
                        | BufferUsages::STORAGE
                        | BufferUsages::INDIRECT
                        | BufferUsages::QUERY_RESOLVE,
                    mapped_at_creation: false,
                }),
                view: view.clone(),
            },
            BackendResourceCreateInfo::Image {
                description, view, ..
            } => {
                let plan = image_texture_plan(description.format(), view.is_some())?;
                Resource::Image {
                    texture: self.device.create_texture(&TextureDescriptor {
                        label: Some("Nixe neutral image"),
                        size: texture_extent(*description),
                        mip_level_count: u32::from(description.mip_levels()),
                        sample_count: description.samples() as u32,
                        dimension: texture_dimension(description.dimension()),
                        format: plan.format,
                        usage: plan.usages,
                        view_formats: &[],
                    }),
                    description: *description,
                    view: view.clone(),
                    attachment_views: HashMap::new(),
                }
            }
            BackendResourceCreateInfo::Sampler { description, .. } => Resource::Sampler {
                sampler: self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("Nixe neutral sampler"),
                    address_mode_u: address_mode(description.address_modes[0]),
                    address_mode_v: address_mode(description.address_modes[1]),
                    address_mode_w: address_mode(description.address_modes[2]),
                    mag_filter: filter_mode(description.mag_filter),
                    min_filter: filter_mode(description.min_filter),
                    mipmap_filter: mip_filter_mode(description.mip_filter),
                    lod_min_clamp: description.lod_min,
                    lod_max_clamp: description.lod_max,
                    anisotropy_clamp: description.max_anisotropy as u16,
                    ..Default::default()
                }),
            },
            BackendResourceCreateInfo::Shader { module, .. } => Resource::Shader {
                module: self.device.create_shader_module(ShaderModuleDescriptor {
                    label: Some("Nixe translated shader"),
                    source: ShaderSource::Wgsl(module.source().into()),
                }),
                neutral: module.clone(),
            },
            BackendResourceCreateInfo::Pipeline { description, .. } => Resource::Pipeline {
                description: *description,
                render: RenderPipelineCache::default(),
            },
            BackendResourceCreateInfo::DescriptorTable { bindings, .. } => {
                Resource::DescriptorTable {
                    bindings: bindings.clone(),
                    bind_groups: HashMap::new(),
                }
            }
            BackendResourceCreateInfo::RenderPass { .. } => Resource::RenderPass,
            BackendResourceCreateInfo::QueryPool { .. } => Resource::QueryPool,
        })
    }

    fn ensure_residency_budget(
        &mut self,
        additional_count: usize,
        additional_bytes: u64,
        protected: Option<&ResolvedBackendResources>,
    ) -> Result<(), BackendDriverError> {
        if additional_bytes > MAX_RESIDENT_RESOURCE_BYTES {
            return Err(BackendDriverError::failure(
                "one backend resource exceeds the residency budget",
            ));
        }
        while self.resident_resources + additional_count > MAX_RESIDENT_RESOURCE_COUNT
            || self
                .resident_resource_bytes
                .checked_add(additional_bytes)
                .is_none_or(|bytes| bytes > MAX_RESIDENT_RESOURCE_BYTES)
        {
            let resource_candidate = self
                .resources
                .iter()
                .flatten()
                .filter(|slot| {
                    let handle = slot.handle;
                    let record = &slot.record;
                    record.host.is_some()
                        && !protected.is_some_and(|resources| {
                            resources.values().any(|protected| *protected == handle)
                        })
                        && record
                            .content
                            .as_ref()
                            .is_none_or(|content| !content.has_device_writes())
                })
                .min_by_key(|slot| slot.record.last_use.map_or(0, |last_use| last_use.serial))
                .map(|slot| {
                    (
                        slot.record.last_use.map_or(0, |last_use| last_use.serial),
                        ResidencyCandidate::Resource(slot.handle),
                    )
                });
            let presentation_candidate = self
                .presentation_imports
                .iter()
                .min_by_key(|(_, import)| import.last_used)
                .map(|(key, import)| (import.last_used, ResidencyCandidate::Presentation(*key)));
            let candidate = match (resource_candidate, presentation_candidate) {
                (Some(resource), Some(presentation)) => {
                    if resource.0 <= presentation.0 {
                        resource.1
                    } else {
                        presentation.1
                    }
                }
                (Some(resource), None) => resource.1,
                (None, Some(presentation)) => presentation.1,
                (None, None) => {
                    return Err(BackendDriverError::failure(
                        "resident resource budget is exhausted by active device contents",
                    ));
                }
            };
            match candidate {
                ResidencyCandidate::Resource(handle) => {
                    let resident_bytes = {
                        let record = self
                            .resource_record_mut(handle)
                            .expect("residency candidate came from the resource table");
                        record.host = None;
                        if let Some(content) = record.content.as_mut() {
                            content.initialized = false;
                        }
                        record.resident_bytes
                    };
                    self.resident_resources -= 1;
                    self.resident_resource_bytes -= resident_bytes;
                }
                ResidencyCandidate::Presentation(key) => {
                    self.remove_presentation_import(key);
                }
            }
        }
        Ok(())
    }

    fn ensure_resident(
        &mut self,
        resources: &ResolvedBackendResources,
    ) -> Result<(), BackendDriverError> {
        for handle in resources.values().copied() {
            let record = self.resource_record(handle)?;
            if record.host.is_some() {
                continue;
            }
            let info = record.immutable.clone();
            let resident_bytes = record.resident_bytes;
            self.ensure_residency_budget(1, resident_bytes, Some(resources))?;
            let host = self.create_host_resource(&info)?;
            self.resource_record_mut(handle)
                .expect("logical resource remains live while becoming resident")
                .host = Some(host);
            self.resident_resources += 1;
            self.resident_resource_bytes += resident_bytes;
        }
        Ok(())
    }

    fn materialize_cpu_page(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendDriverError> {
        let mut demanded = Vec::new();
        for slot in self.resources.iter().flatten() {
            collect_demanded_writebacks(slot.handle, &slot.record, request.page, &mut demanded)?;
        }
        prepare_demanded_writebacks(&mut demanded);
        if !demanded.is_empty() {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Nixe demanded canonical visibility"),
                });
            let mut writebacks = Vec::with_capacity(demanded.len());
            for demanded_writeback in &demanded {
                match *demanded_writeback {
                    DemandedWriteback::Buffer(buffer) => self.encode_buffer_writeback(
                        &mut encoder,
                        buffer,
                        request.page,
                        &mut writebacks,
                    )?,
                    DemandedWriteback::Image {
                        handle, binding, ..
                    } => {
                        self.encode_image_writeback(&mut encoder, handle, binding, &mut writebacks)?
                    }
                }
            }
            {
                let _queue_access = self.queue_access.lock();
                self.queue.submit([encoder.finish()]);
            }
            self.finish_writebacks(writebacks)?;
            for writeback in &demanded {
                let record = self
                    .resource_record_mut(writeback.handle())
                    .expect("demanded read-back resource remains live");
                complete_demanded_writeback(record, *writeback);
            }
            for writeback in &demanded {
                if let DemandedWriteback::Image {
                    handle, binding, ..
                } = *writeback
                {
                    let BackendResourceCreateInfo::Image {
                        view: Some(view), ..
                    } = &self
                        .resource_record(handle)
                        .expect("demanded image remains live")
                        .immutable
                    else {
                        continue;
                    };
                    self.visibility
                        .mark_backing_completed(
                            view.bindings()[binding].backing(),
                            request.visible_at,
                        )
                        .map_err(|error| BackendDriverError::failure(error.to_string()))?;
                }
            }
            let mut retired = Vec::new();
            for writeback in demanded {
                let handle = writeback.handle();
                if !retired.contains(&handle)
                    && self.resource_record(handle).is_ok_and(|record| {
                        record.retired
                            && record
                                .content
                                .as_ref()
                                .is_none_or(|content| !content.has_device_writes())
                    })
                {
                    retired.push(handle);
                }
            }
            for handle in retired {
                self.remove_resource_record(handle);
            }
        }
        // The neutral visibility contract is currently page-conservative. If
        // no exact device-write region overlapped this page, its prepared
        // mirror is already the newest content once the runtime has waited for
        // `visible_at`; no host transfer is necessary.
        self.visibility
            .mark_page_completed(request.page, request.visible_at)
            .map_err(|error| BackendDriverError::failure(error.to_string()))?;
        self.visibility
            .take_completed_page(request)
            .map_err(|error| BackendDriverError::failure(error.to_string()))
    }
}

impl BackendDriver for WgpuBackendDriver {
    fn create_resource(
        &mut self,
        handle: BackendResourceHandle,
        info: &BackendResourceCreateInfo,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        if let BackendResourceCreateInfo::Image {
            view: Some(view), ..
        } = info
            && view.swizzle() != nixe_gpu::Swizzle::IDENTITY
        {
            return Err(unsupported("non-identity image component swizzle"));
        }
        if let BackendResourceCreateInfo::Sampler { description, .. } = info {
            if description
                .address_modes
                .contains(&nixe_gpu::AddressMode::ClampToBorder)
            {
                return Err(unsupported(
                    "clamp-to-border sampler without a neutral border color",
                ));
            }
            if description.max_anisotropy.fract() != 0.0 || description.max_anisotropy > 16.0 {
                return Err(unsupported("sampler anisotropy outside exact wgpu range"));
            }
            if description.max_anisotropy > 1.0
                && (description.min_filter != nixe_gpu::FilterMode::Linear
                    || description.mag_filter != nixe_gpu::FilterMode::Linear
                    || description.mip_filter != nixe_gpu::FilterMode::Linear)
            {
                return Err(unsupported("anisotropic sampler with non-linear filtering"));
            }
        }
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let resident_bytes = estimated_resident_bytes(info)?;
        self.ensure_residency_budget(1, resident_bytes, None)?;
        let resource = self.create_host_resource(info)?;
        self.capture_error_scope(scope)?;
        let index = usize::try_from(handle.slot())
            .map_err(|_| BackendDriverError::failure("WGPU resource slot does not fit usize"))?;
        if index >= self.resources.len() {
            self.resources
                .try_reserve(index + 1 - self.resources.len())
                .map_err(|_| BackendDriverError::failure("WGPU resource table exhausted"))?;
            self.resources.resize_with(index + 1, || None);
        }
        if self.resources[index].is_some() {
            return Err(BackendDriverError::failure(format!(
                "WGPU resource slot is already occupied: {handle}"
            )));
        }
        self.resources[index] = Some(WgpuResourceSlot {
            handle,
            record: ResourceRecord {
                content: ResourceContent::new(info),
                immutable: info.clone(),
                host: Some(resource),
                last_use: None,
                retired: false,
                resident_bytes,
            },
        });
        self.index_presentable_image(handle, info);
        self.resident_resources += 1;
        self.resident_resource_bytes = self
            .resident_resource_bytes
            .checked_add(resident_bytes)
            .ok_or_else(|| BackendDriverError::failure("resident byte budget overflow"))?;
        Ok(())
    }

    fn destroy_resource(
        &mut self,
        handle: BackendResourceHandle,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        if let Some(BackendResourceCreateInfo::Allocation { id, .. }) = self
            .resource_record(handle)
            .ok()
            .map(|record| &record.immutable)
        {
            let imports = self
                .presentation_imports
                .keys()
                .copied()
                .filter(|key| key.image.allocation == *id)
                .collect::<Vec<_>>();
            for key in imports {
                self.remove_presentation_import(key);
            }
        }
        if let Ok(record) = self.resource_record_mut(handle) {
            record.retired = true;
            if record
                .content
                .as_ref()
                .is_some_and(ResourceContent::has_device_writes)
            {
                return Ok(());
            }
        }
        self.remove_resource_record(handle);
        Ok(())
    }

    fn submit(
        &mut self,
        accepted: &AcceptedBackendSubmission<'_>,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        let use_serial = self.take_resource_use()?;
        let dependencies = accepted.resources();
        self.ensure_resident(dependencies)?;
        self.upload_bytes = 0;
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Nixe neutral submission"),
            });
        self.upload_inputs(accepted, dependencies, &mut encoder)?;
        let encoder = self.encode_submission(accepted, dependencies, encoder)?;
        self.upload_staging.finish_and_recall_on_submit(&encoder);
        let submission_index = {
            let _queue_access = self.queue_access.lock();
            self.queue.submit([encoder.finish()])
        };
        let token = accepted.token();
        self.submissions.insert(
            token,
            HostSubmission {
                index: submission_index,
                completed: false,
            },
        );
        let completed = self.completion_sender.clone();
        self.queue.on_submitted_work_done(move || {
            let _ = completed.send(token);
        });
        for handle in dependencies.values() {
            if let Ok(record) = self.resource_record_mut(*handle) {
                record.last_use = Some(ResourceUse {
                    serial: use_serial,
                    submission: token,
                });
            }
        }
        for operation in accepted.submission().operations() {
            for access in operation.accesses() {
                if !access.scope().mode().writes() {
                    continue;
                }
                let handle = dependency_handle(dependencies, access.target().dependency())?;
                if let Ok(record) = self.resource_record_mut(handle) {
                    record_device_write(record, access.target(), use_serial)?;
                }
            }
        }
        Ok(())
    }

    fn has_completed(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<bool, BackendDriverError> {
        self.require_device()?;
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| BackendDriverError::device_lost(error.to_string()))?;
        self.drain_completions();
        self.require_device()?;
        Ok(self
            .submissions
            .get(&submission)
            .is_some_and(|submission| submission.completed))
    }

    fn wait_for_completion(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        let index = self
            .submissions
            .get(&submission)
            .map(|submission| submission.index.clone())
            .ok_or_else(|| BackendDriverError::failure("unknown wgpu submission token"))?;
        let status = self
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .map_err(|error| BackendDriverError::device_lost(error.to_string()))?;
        if !status.wait_finished() {
            return Err(BackendDriverError::failure(
                "wgpu directed submission wait did not finish",
            ));
        }
        self.require_device()?;
        self.submissions
            .get_mut(&submission)
            .ok_or_else(|| BackendDriverError::failure("unknown wgpu submission token"))?
            .completed = true;
        self.drain_completions();
        Ok(())
    }

    fn release_submission(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        self.submissions.remove(&submission);
        Ok(())
    }

    fn bind_visibility_requester(
        &mut self,
        requester: Arc<dyn nixe_gpu::BackendVisibilityRequester>,
    ) -> Result<(), BackendDriverError> {
        self.require_device()?;
        self.visibility
            .bind_requester(requester)
            .map_err(|error| BackendDriverError::failure(error.to_string()))
    }

    fn make_cpu_visible(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, BackendDriverError> {
        self.require_device()?;
        if request.device != self.visibility.device() {
            return Err(BackendDriverError::failure(
                "CPU visibility request targets another device",
            ));
        }
        self.materialize_cpu_page(request)
    }

    fn acquire_presentable_image(
        &mut self,
        request: PresentationImageRequest,
    ) -> Result<ResidentImage, BackendDriverError> {
        self.acquire_presentable(request)
    }

    fn teardown(&mut self) -> Result<(), BackendDriverError> {
        if self.torn_down {
            return Ok(());
        }
        let cache_result = self.persist_pipeline_cache();
        self.clear_owned_state();
        self.torn_down = true;
        cache_result
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WebGpuViewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

fn webgpu_viewport(transform: ViewportTransform) -> Result<WebGpuViewport, BackendDriverError> {
    let scale = transform.scale();
    let offset = transform.offset();
    let [min_depth, max_depth] = transform.depth_range();

    // WebGPU NDC has its origin at bottom-left while framebuffer coordinates
    // start at top-left. Consequently its positive viewport height already
    // implements the negative Y coefficient programmed by normal Maxwell
    // draws. See https://www.w3.org/TR/webgpu/#coordinate-systems.
    if scale[0] <= 0.0 {
        return Err(unsupported("non-positive Maxwell X viewport scale"));
    }
    if scale[1] >= 0.0 {
        return Err(unsupported("non-negative Maxwell Y viewport scale"));
    }
    if scale[2] < 0.0 {
        return Err(unsupported("reversed Maxwell depth viewport scale"));
    }
    if !(0.0..=1.0).contains(&min_depth)
        || !(0.0..=1.0).contains(&max_depth)
        || min_depth > max_depth
    {
        return Err(unsupported(
            "viewport depth range cannot be represented by WebGPU",
        ));
    }

    let viewport = WebGpuViewport {
        x: offset[0] - scale[0],
        y: offset[1] + scale[1],
        width: scale[0] * 2.0,
        height: scale[1] * -2.0,
        min_depth,
        max_depth,
    };
    if ![
        viewport.x,
        viewport.y,
        viewport.width,
        viewport.height,
        viewport.min_depth,
        viewport.max_depth,
    ]
    .into_iter()
    .all(f32::is_finite)
    {
        return Err(unsupported("Maxwell viewport conversion overflow"));
    }
    Ok(viewport)
}

pub(crate) const fn texture_format(format: ImageFormat) -> Option<TextureFormat> {
    Some(match format {
        ImageFormat::R8Unorm => TextureFormat::R8Unorm,
        ImageFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
        ImageFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        ImageFormat::Rgba8Srgb => TextureFormat::Rgba8UnormSrgb,
        ImageFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        ImageFormat::Bgra8Srgb => TextureFormat::Bgra8UnormSrgb,
        ImageFormat::R16Float => TextureFormat::R16Float,
        ImageFormat::Rg16Float => TextureFormat::Rg16Float,
        ImageFormat::Rgba16Float => TextureFormat::Rgba16Float,
        ImageFormat::R32Float => TextureFormat::R32Float,
        ImageFormat::Rg32Float => TextureFormat::Rg32Float,
        ImageFormat::Rgba32Float => TextureFormat::Rgba32Float,
        ImageFormat::Depth16Unorm => TextureFormat::Depth16Unorm,
        // WebGPU deliberately leaves the physical depth representation opaque.
        // It is therefore valid as an attachment but not as canonical D24S8
        // storage. See https://www.w3.org/TR/webgpu/#texture-formats
        ImageFormat::Depth24UnormStencil8Uint => TextureFormat::Depth24PlusStencil8,
        ImageFormat::Depth32Float => TextureFormat::Depth32Float,
        // This format requires an optional `wgpu` feature which the initial
        // device intentionally does not request.
        ImageFormat::Depth32FloatStencil8Uint => return None,
    })
}

const fn compare_function(compare: DepthCompareOperation) -> CompareFunction {
    match compare {
        DepthCompareOperation::Never => CompareFunction::Never,
        DepthCompareOperation::Less => CompareFunction::Less,
        DepthCompareOperation::Equal => CompareFunction::Equal,
        DepthCompareOperation::LessEqual => CompareFunction::LessEqual,
        DepthCompareOperation::Greater => CompareFunction::Greater,
        DepthCompareOperation::NotEqual => CompareFunction::NotEqual,
        DepthCompareOperation::GreaterEqual => CompareFunction::GreaterEqual,
        DepthCompareOperation::Always => CompareFunction::Always,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ImageTexturePlan {
    format: TextureFormat,
    usages: TextureUsages,
}

fn image_texture_plan(
    format: ImageFormat,
    has_canonical_backing: bool,
) -> Result<ImageTexturePlan, BackendDriverError> {
    let host = texture_format(format).ok_or_else(|| unsupported("image format"))?;
    if format == ImageFormat::Depth24UnormStencil8Uint {
        if has_canonical_backing {
            return Err(unsupported(
                "canonical D24S8 transfer without an explicit depth/stencil conversion",
            ));
        }
        return Ok(ImageTexturePlan {
            format: host,
            usages: TextureUsages::RENDER_ATTACHMENT,
        });
    }
    Ok(ImageTexturePlan {
        format: host,
        usages: required_texture_usages(format),
    })
}

pub(crate) fn required_texture_usages(format: ImageFormat) -> TextureUsages {
    if format == ImageFormat::Depth24UnormStencil8Uint {
        TextureUsages::RENDER_ATTACHMENT
    } else {
        TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
    }
}

const fn vertex_format(format: VertexFormat) -> Option<WgpuVertexFormat> {
    Some(match format {
        VertexFormat::Uint8x2 => WgpuVertexFormat::Uint8x2,
        VertexFormat::Uint8x4 => WgpuVertexFormat::Uint8x4,
        VertexFormat::Sint8x2 => WgpuVertexFormat::Sint8x2,
        VertexFormat::Sint8x4 => WgpuVertexFormat::Sint8x4,
        VertexFormat::Unorm8x2 => WgpuVertexFormat::Unorm8x2,
        VertexFormat::Unorm8x4 => WgpuVertexFormat::Unorm8x4,
        VertexFormat::Snorm8x2 => WgpuVertexFormat::Snorm8x2,
        VertexFormat::Snorm8x4 => WgpuVertexFormat::Snorm8x4,
        VertexFormat::Uint16x2 => WgpuVertexFormat::Uint16x2,
        VertexFormat::Uint16x4 => WgpuVertexFormat::Uint16x4,
        VertexFormat::Sint16x2 => WgpuVertexFormat::Sint16x2,
        VertexFormat::Sint16x4 => WgpuVertexFormat::Sint16x4,
        VertexFormat::Unorm16x2 => WgpuVertexFormat::Unorm16x2,
        VertexFormat::Unorm16x4 => WgpuVertexFormat::Unorm16x4,
        VertexFormat::Snorm16x2 => WgpuVertexFormat::Snorm16x2,
        VertexFormat::Snorm16x4 => WgpuVertexFormat::Snorm16x4,
        VertexFormat::Float16x2 => WgpuVertexFormat::Float16x2,
        VertexFormat::Float16x4 => WgpuVertexFormat::Float16x4,
        VertexFormat::Float32 => WgpuVertexFormat::Float32,
        VertexFormat::Float32x2 => WgpuVertexFormat::Float32x2,
        VertexFormat::Float32x3 => WgpuVertexFormat::Float32x3,
        VertexFormat::Float32x4 => WgpuVertexFormat::Float32x4,
        VertexFormat::Uint32 => WgpuVertexFormat::Uint32,
        VertexFormat::Uint32x2 => WgpuVertexFormat::Uint32x2,
        VertexFormat::Uint32x3 => WgpuVertexFormat::Uint32x3,
        VertexFormat::Uint32x4 => WgpuVertexFormat::Uint32x4,
        VertexFormat::Sint32 => WgpuVertexFormat::Sint32,
        VertexFormat::Sint32x2 => WgpuVertexFormat::Sint32x2,
        VertexFormat::Sint32x3 => WgpuVertexFormat::Sint32x3,
        VertexFormat::Sint32x4 => WgpuVertexFormat::Sint32x4,
        VertexFormat::Unorm10_10_10_2 => WgpuVertexFormat::Unorm10_10_10_2,
        VertexFormat::Uscaled { .. } | VertexFormat::Sscaled { .. } => return None,
    })
}

fn texture_extent(description: ImageDescription) -> Extent3d {
    Extent3d {
        width: description.extent().width,
        height: description.extent().height,
        depth_or_array_layers: match description.dimension() {
            ImageDimension::Three => description.extent().depth,
            _ => u32::from(description.array_layers()),
        },
    }
}

const fn texture_dimension(dimension: ImageDimension) -> TextureDimension {
    match dimension {
        ImageDimension::One => TextureDimension::D1,
        ImageDimension::Two | ImageDimension::Cube => TextureDimension::D2,
        ImageDimension::Three => TextureDimension::D3,
    }
}

fn primitive_topology(
    topology: PrimitiveTopology,
) -> Result<wgpu::PrimitiveTopology, BackendDriverError> {
    Ok(match topology {
        PrimitiveTopology::Points => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::Lines => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveTopology::Triangles => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        PrimitiveTopology::TriangleFan => return Err(unsupported("triangle fan topology")),
        PrimitiveTopology::Patches => return Err(unsupported("patch topology")),
    })
}

fn texture_view_descriptor(range: ImageSubresourceRange) -> TextureViewDescriptor<'static> {
    TextureViewDescriptor {
        label: Some("Nixe neutral image view"),
        base_mip_level: u32::from(range.mip_level),
        mip_level_count: Some(1),
        base_array_layer: u32::from(range.base_layer),
        array_layer_count: Some(u32::from(range.layer_count)),
        aspect: TextureAspect::All,
        ..Default::default()
    }
}

fn sampled_texture_view_descriptor(
    description: ImageDescription,
) -> TextureViewDescriptor<'static> {
    let dimension = match description.dimension() {
        ImageDimension::One => TextureViewDimension::D1,
        ImageDimension::Two if description.array_layers() == 1 => TextureViewDimension::D2,
        ImageDimension::Two => TextureViewDimension::D2Array,
        ImageDimension::Three => TextureViewDimension::D3,
        ImageDimension::Cube if description.array_layers() == 6 => TextureViewDimension::Cube,
        ImageDimension::Cube => TextureViewDimension::CubeArray,
    };
    TextureViewDescriptor {
        label: Some("Nixe neutral sampled image view"),
        dimension: Some(dimension),
        base_mip_level: 0,
        mip_level_count: Some(u32::from(description.mip_levels())),
        base_array_layer: 0,
        array_layer_count: Some(u32::from(description.array_layers())),
        aspect: TextureAspect::All,
        ..Default::default()
    }
}

fn require_full_image_region(
    description: ImageDescription,
    region: ImageRegion,
) -> Result<(), BackendDriverError> {
    let extent = description
        .mip_extent(region.subresources.mip_level)
        .ok_or_else(|| unsupported("invalid image region mip"))?;
    if region.origin != (ImageOrigin { x: 0, y: 0, z: 0 })
        || region.extent != extent
        || region.subresources.layer_count != 1
    {
        return Err(unsupported("partial image clear"));
    }
    Ok(())
}

fn color_operations(
    attachment: &RenderAttachment,
) -> Result<Operations<Color>, BackendDriverError> {
    let load = match attachment.load {
        AttachmentLoad::Load => LoadOp::Load,
        AttachmentLoad::Discard => LoadOp::Clear(Color::TRANSPARENT),
        AttachmentLoad::Clear(ClearValue::Color(color)) => LoadOp::Clear(color_value(color)),
        AttachmentLoad::Clear(_) => return Err(unsupported("color attachment clear value")),
    };
    Ok(Operations {
        load,
        store: store_operation(attachment.store),
    })
}

fn depth_operations<'a>(
    view: &'a wgpu::TextureView,
    attachment: &RenderAttachment,
) -> Result<RenderPassDepthStencilAttachment<'a>, BackendDriverError> {
    let (depth, stencil) = match attachment.load {
        AttachmentLoad::Load => (Some(LoadOp::Load), Some(LoadOp::Load)),
        AttachmentLoad::Discard => (Some(LoadOp::Clear(1.0)), Some(LoadOp::Clear(0))),
        AttachmentLoad::Clear(ClearValue::Depth(depth)) => (Some(LoadOp::Clear(depth)), None),
        AttachmentLoad::Clear(ClearValue::Stencil(stencil)) => {
            (None, Some(LoadOp::Clear(u32::from(stencil))))
        }
        AttachmentLoad::Clear(ClearValue::DepthStencil { depth, stencil }) => (
            Some(LoadOp::Clear(depth)),
            Some(LoadOp::Clear(u32::from(stencil))),
        ),
        AttachmentLoad::Clear(_) => return Err(unsupported("depth attachment clear value")),
    };
    Ok(RenderPassDepthStencilAttachment {
        view,
        depth_ops: depth.map(|load| Operations {
            load,
            store: store_operation(attachment.store),
        }),
        stencil_ops: stencil.map(|load| Operations {
            load,
            store: store_operation(attachment.store),
        }),
    })
}

const fn store_operation(store: AttachmentStore) -> StoreOp {
    match store {
        AttachmentStore::Store => StoreOp::Store,
        AttachmentStore::Discard => StoreOp::Discard,
    }
}

fn color_value(value: [f32; 4]) -> Color {
    Color {
        r: f64::from(value[0]),
        g: f64::from(value[1]),
        b: f64::from(value[2]),
        a: f64::from(value[3]),
    }
}

fn dependency_handle(
    dependencies: &ResolvedBackendResources,
    dependency: ResourceDependency,
) -> Result<BackendResourceHandle, BackendDriverError> {
    dependencies.get(&dependency).copied().ok_or_else(|| {
        BackendDriverError::failure(format!("missing resolved resource: {dependency:?}"))
    })
}

fn shader_handle_for_stage(
    dependencies: &ResolvedBackendResources,
    operation: usize,
    stage: ShaderStage,
) -> Result<BackendResourceHandle, BackendDriverError> {
    dependencies.shader(operation, stage).ok_or_else(|| {
        BackendDriverError::failure(format!(
            "operation {operation} has no resolved {stage:?} shader"
        ))
    })
}

fn record_device_write(
    record: &mut ResourceRecord,
    target: nixe_gpu::AccessTarget,
    serial: u64,
) -> Result<(), BackendDriverError> {
    let Some(content) = record.content.as_mut() else {
        return Ok(());
    };
    content.initialized = true;
    match (target, &record.immutable) {
        (
            nixe_gpu::AccessTarget::Buffer { range, .. },
            BackendResourceCreateInfo::Buffer {
                view: Some(view), ..
            },
        ) => {
            let offset = range
                .offset()
                .checked_sub(view.buffer_offset())
                .ok_or_else(|| unsupported("buffer device-write range precedes its backing"))?;
            record_buffer_device_write(
                &mut content.device_writes,
                TransferRange {
                    offset,
                    size: range.size(),
                },
                serial,
            )?;
        }
        (
            nixe_gpu::AccessTarget::Image { subresources, .. },
            BackendResourceCreateInfo::Image {
                view: Some(view), ..
            },
        ) => {
            for (index, binding) in view.bindings().iter().enumerate() {
                if !image_subresources_overlap(binding.subresources(), subresources) {
                    continue;
                }
                content
                    .device_writes
                    .retain(|write| write.region != DeviceWriteRegion::ImageBinding(index));
                if content.device_writes.len() == MAX_DEVICE_WRITE_REGIONS {
                    return Err(BackendDriverError::failure(
                        "device-write region budget is exhausted",
                    ));
                }
                content.device_writes.push(DeviceWrite {
                    region: DeviceWriteRegion::ImageBinding(index),
                    serial,
                });
            }
        }
        _ => {}
    }
    Ok(())
}

fn record_buffer_device_write(
    writes: &mut Vec<DeviceWrite>,
    new: TransferRange,
    serial: u64,
) -> Result<(), BackendDriverError> {
    let new_end = new
        .offset
        .checked_add(new.size)
        .ok_or_else(|| unsupported("buffer device-write range overflow"))?;
    let resulting = writes.iter().try_fold(1_usize, |count, write| {
        let DeviceWriteRegion::Buffer(old) = write.region else {
            return Ok(count + 1);
        };
        let old_end = old.offset + old.size;
        if old_end <= new.offset || new_end <= old.offset {
            return Ok(count + 1);
        }
        Ok(count + usize::from(old.offset < new.offset) + usize::from(old_end > new_end))
    })?;
    if resulting > MAX_DEVICE_WRITE_REGIONS {
        return Err(BackendDriverError::failure(
            "device-write region budget is exhausted",
        ));
    }
    let mut index = 0;
    while index < writes.len() {
        let DeviceWriteRegion::Buffer(old) = writes[index].region else {
            index += 1;
            continue;
        };
        let old_end = old.offset + old.size;
        if old_end <= new.offset || new_end <= old.offset {
            index += 1;
            continue;
        }
        let old_serial = writes[index].serial;
        let left = (old.offset < new.offset).then_some(TransferRange {
            offset: old.offset,
            size: new.offset - old.offset,
        });
        let right = (old_end > new_end).then_some(TransferRange {
            offset: new_end,
            size: old_end - new_end,
        });
        writes.swap_remove(index);
        if let Some(left) = left {
            writes.push(DeviceWrite {
                region: DeviceWriteRegion::Buffer(left),
                serial: old_serial,
            });
        }
        if let Some(right) = right {
            writes.push(DeviceWrite {
                region: DeviceWriteRegion::Buffer(right),
                serial: old_serial,
            });
        }
    }
    writes.push(DeviceWrite {
        region: DeviceWriteRegion::Buffer(new),
        serial,
    });
    Ok(())
}

fn image_subresources_overlap(left: ImageSubresourceRange, right: ImageSubresourceRange) -> bool {
    left.plane == right.plane
        && left.mip_level == right.mip_level
        && u32::from(left.base_layer) < u32::from(right.base_layer) + u32::from(right.layer_count)
        && u32::from(right.base_layer) < u32::from(left.base_layer) + u32::from(left.layer_count)
}

fn ranges_need_upload<'a>(
    ranges: impl IntoIterator<Item = &'a nixe_memory::CanonicalBackingRange>,
    uploaded: &[ContentGeneration],
) -> Result<bool, BackendDriverError> {
    let mut index = 0;
    let mut dirty = false;
    for range in ranges {
        for segment in range.segments() {
            let baseline = uploaded.get(index).copied().ok_or_else(|| {
                BackendDriverError::failure(
                    "wgpu resource content record does not match its immutable backing",
                )
            })?;
            index += 1;
            if baseline == segment.current_content_generation() {
                continue;
            }
            match segment
                .cpu_write_overlap_since(baseline)
                .map_err(|error| BackendDriverError::failure(error.to_string()))?
            {
                CanonicalCpuWriteOverlap::No => {}
                CanonicalCpuWriteOverlap::Yes | CanonicalCpuWriteOverlap::Unknown => dirty = true,
            }
        }
    }
    if index != uploaded.len() {
        return Err(BackendDriverError::failure(
            "wgpu resource content record does not match its immutable backing",
        ));
    }
    Ok(dirty)
}

fn collect_dirty_image_bindings(
    view: &nixe_gpu::ImageView,
    uploaded: &[ContentGeneration],
    initialized: bool,
    output: &mut Vec<usize>,
) -> Result<(), BackendDriverError> {
    let mut generation = 0_usize;
    for (binding, image_binding) in view.bindings().iter().enumerate() {
        let count = image_binding.backing().range().segments().len();
        let end = generation
            .checked_add(count)
            .ok_or_else(|| unsupported("image content generation range overflow"))?;
        let baselines = uploaded.get(generation..end).ok_or_else(|| {
            BackendDriverError::failure(
                "wgpu resource content record does not match its immutable backing",
            )
        })?;
        if !initialized || ranges_need_upload([image_binding.backing().range()], baselines)? {
            // A canonical byte interval cannot be copied directly into a host
            // texture until the pitch/block-linear layout is converted. One
            // immutable image binding is therefore the exact transfer domain.
            output.push(binding);
        }
        generation = end;
    }
    if generation != uploaded.len() {
        return Err(BackendDriverError::failure(
            "wgpu resource content record does not match its immutable backing",
        ));
    }
    Ok(())
}

fn record_image_binding_uploaded(
    view: &nixe_gpu::ImageView,
    binding: usize,
    uploaded: &mut [ContentGeneration],
) {
    let first = view.bindings()[..binding]
        .iter()
        .map(|binding| binding.backing().range().segments().len())
        .sum::<usize>();
    let segments = view.bindings()[binding].backing().range().segments();
    for (baseline, segment) in uploaded[first..first + segments.len()]
        .iter_mut()
        .zip(segments)
    {
        *baseline = segment.current_content_generation();
    }
}

fn collect_dirty_buffer_ranges(
    range: &nixe_memory::CanonicalBackingRange,
    uploaded: &[ContentGeneration],
    buffer_offset: u64,
    page_dirty: &mut Vec<CanonicalCpuWriteRange>,
    output: &mut Vec<TransferRange>,
) -> Result<(), BackendDriverError> {
    if range.segments().len() != uploaded.len() {
        return Err(BackendDriverError::failure(
            "wgpu resource content record does not match its immutable backing",
        ));
    }
    let view_end = buffer_offset
        .checked_add(range.size())
        .ok_or_else(|| unsupported("buffer upload range overflow"))?;
    let mut logical_offset = 0_u64;
    for (segment, baseline) in range.segments().iter().zip(uploaded) {
        if *baseline == segment.current_content_generation() {
            logical_offset += segment.size();
            continue;
        }
        page_dirty.clear();
        segment
            .append_cpu_write_ranges_since(*baseline, page_dirty)
            .map_err(|error| BackendDriverError::failure(error.to_string()))?;
        for dirty in page_dirty.iter().copied() {
            let dirty_start = buffer_offset
                .checked_add(logical_offset)
                .and_then(|offset| offset.checked_add(dirty.offset()))
                .ok_or_else(|| unsupported("buffer dirty range overflow"))?;
            let dirty_end = dirty_start
                .checked_add(dirty.size())
                .ok_or_else(|| unsupported("buffer dirty range overflow"))?;
            let aligned_start = dirty_start / 4 * 4;
            let aligned_end = align_u64(dirty_end, 4)?;
            if aligned_start < buffer_offset || aligned_end > view_end {
                if !buffer_offset.is_multiple_of(4) || !range.size().is_multiple_of(4) {
                    return Err(unsupported("unaligned canonically backed buffer upload"));
                }
                output.clear();
                output.push(TransferRange {
                    offset: 0,
                    size: range.size(),
                });
                return Ok(());
            }
            output.push(TransferRange {
                offset: aligned_start - buffer_offset,
                size: aligned_end - aligned_start,
            });
        }
        logical_offset = logical_offset
            .checked_add(segment.size())
            .ok_or_else(|| unsupported("buffer upload range overflow"))?;
    }
    normalize_transfer_ranges(output);
    Ok(())
}

fn normalize_transfer_ranges(ranges: &mut Vec<TransferRange>) {
    ranges.sort_unstable_by_key(|range| range.offset);
    let mut output = 0_usize;
    for input in 0..ranges.len() {
        let current = ranges[input];
        if output != 0 {
            let previous = &mut ranges[output - 1];
            let previous_end = previous.offset + previous.size;
            if current.offset <= previous_end {
                previous.size = previous_end.max(current.offset + current.size) - previous.offset;
                continue;
            }
        }
        ranges[output] = current;
        output += 1;
    }
    ranges.truncate(output);
}

fn record_uploaded<'a>(
    ranges: impl IntoIterator<Item = &'a nixe_memory::CanonicalBackingRange>,
    uploaded: &mut [ContentGeneration],
) {
    let mut index = 0;
    for range in ranges {
        for segment in range.segments() {
            uploaded[index] = segment.current_content_generation();
            index += 1;
        }
    }
    debug_assert_eq!(index, uploaded.len());
}

fn collect_demanded_writebacks(
    handle: BackendResourceHandle,
    record: &ResourceRecord,
    page: CanonicalPageId,
    output: &mut Vec<DemandedWriteback>,
) -> Result<(), BackendDriverError> {
    let Some(content) = record.content.as_ref() else {
        return Ok(());
    };
    match &record.immutable {
        BackendResourceCreateInfo::Buffer {
            view: Some(view), ..
        } => {
            for write in &content.device_writes {
                let DeviceWriteRegion::Buffer(dirty) = write.region else {
                    continue;
                };
                let dirty_end = dirty.offset + dirty.size;
                let mut logical_offset = 0_u64;
                for segment in view.backing().range().segments() {
                    let segment_end = logical_offset + segment.size();
                    if segment.page() == page {
                        let start = dirty.offset.max(logical_offset);
                        let end = dirty_end.min(segment_end);
                        if start < end {
                            output.push(DemandedWriteback::Buffer(DemandedBufferWriteback {
                                handle,
                                serial: write.serial,
                                range: TransferRange {
                                    offset: start,
                                    size: end - start,
                                },
                                page_offset: segment.offset() + start - logical_offset,
                            }));
                        }
                    }
                    logical_offset = segment_end;
                }
            }
        }
        BackendResourceCreateInfo::Image {
            view: Some(view), ..
        } => {
            for write in &content.device_writes {
                let DeviceWriteRegion::ImageBinding(binding) = write.region else {
                    continue;
                };
                if view.bindings()[binding]
                    .backing()
                    .range()
                    .segments()
                    .iter()
                    .any(|segment| segment.page() == page)
                {
                    output.push(DemandedWriteback::Image {
                        handle,
                        binding,
                        serial: write.serial,
                    });
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn coalesce_demanded_buffer_writebacks(writebacks: &mut Vec<DemandedWriteback>) {
    let mut output = 0_usize;
    for input in 0..writebacks.len() {
        let current = writebacks[input];
        if output != 0
            && let (DemandedWriteback::Buffer(previous), DemandedWriteback::Buffer(current_buffer)) =
                (&mut writebacks[output - 1], current)
            && previous.handle == current_buffer.handle
            && previous.serial == current_buffer.serial
            && previous.range.offset + previous.range.size == current_buffer.range.offset
            && previous.page_offset + previous.range.size == current_buffer.page_offset
        {
            previous.range.size += current_buffer.range.size;
            continue;
        }
        writebacks[output] = current;
        output += 1;
    }
    writebacks.truncate(output);
}

fn prepare_demanded_writebacks(writebacks: &mut Vec<DemandedWriteback>) {
    writebacks.sort_unstable_by_key(|writeback| {
        (
            writeback.serial(),
            writeback.handle(),
            writeback.order_offset(),
        )
    });
    coalesce_demanded_buffer_writebacks(writebacks);
}

fn complete_demanded_writeback(record: &mut ResourceRecord, completed: DemandedWriteback) {
    let Some(content) = record.content.as_mut() else {
        return;
    };
    match completed {
        DemandedWriteback::Buffer(completed) => {
            subtract_completed_buffer_write(
                &mut content.device_writes,
                completed.range,
                completed.serial,
            );
        }
        DemandedWriteback::Image {
            binding, serial, ..
        } => content.device_writes.retain(|write| {
            write.serial != serial || write.region != DeviceWriteRegion::ImageBinding(binding)
        }),
    }
}

fn subtract_completed_buffer_write(
    writes: &mut Vec<DeviceWrite>,
    completed: TransferRange,
    serial: u64,
) {
    subtract_buffer_write_range(writes, completed, Some(serial));
}

fn subtract_buffer_write_range(
    writes: &mut Vec<DeviceWrite>,
    completed: TransferRange,
    serial: Option<u64>,
) {
    let completed_end = completed.offset + completed.size;
    let mut index = 0;
    while index < writes.len() {
        let DeviceWriteRegion::Buffer(current) = writes[index].region else {
            index += 1;
            continue;
        };
        let current_end = current.offset + current.size;
        if serial.is_some_and(|serial| writes[index].serial != serial)
            || current_end <= completed.offset
            || completed_end <= current.offset
        {
            index += 1;
            continue;
        }
        let left = (current.offset < completed.offset).then_some(TransferRange {
            offset: current.offset,
            size: completed.offset - current.offset,
        });
        let right = (current_end > completed_end).then_some(TransferRange {
            offset: completed_end,
            size: current_end - completed_end,
        });
        let current_serial = writes[index].serial;
        writes.swap_remove(index);
        if let Some(left) = left {
            writes.push(DeviceWrite {
                region: DeviceWriteRegion::Buffer(left),
                serial: current_serial,
            });
        }
        if let Some(right) = right {
            writes.push(DeviceWrite {
                region: DeviceWriteRegion::Buffer(right),
                serial: current_serial,
            });
        }
    }
}

fn estimated_resident_bytes(info: &BackendResourceCreateInfo) -> Result<u64, BackendDriverError> {
    match info {
        BackendResourceCreateInfo::Buffer { description, .. } => Ok(description.size()),
        BackendResourceCreateInfo::Image { description, .. } => {
            let bytes_per_texel =
                (0..description.format().plane_count()).try_fold(0_u64, |total, plane| {
                    total
                        .checked_add(u64::from(
                            description
                                .format()
                                .plane_bytes_per_texel(plane)
                                .ok_or_else(|| unsupported("image plane format"))?,
                        ))
                        .ok_or_else(|| unsupported("image residency size overflow"))
                })?;
            let mut texels = 0_u64;
            for mip in 0..description.mip_levels() {
                let extent = description
                    .mip_extent(mip)
                    .ok_or_else(|| unsupported("image residency mip"))?;
                let layers = match description.dimension() {
                    ImageDimension::Three => 1,
                    _ => u64::from(description.array_layers()),
                };
                let mip_texels = u64::from(extent.width)
                    .checked_mul(u64::from(extent.height))
                    .and_then(|value| value.checked_mul(u64::from(extent.depth)))
                    .and_then(|value| value.checked_mul(layers))
                    .ok_or_else(|| unsupported("image residency size overflow"))?;
                texels = texels
                    .checked_add(mip_texels)
                    .ok_or_else(|| unsupported("image residency size overflow"))?;
            }
            texels
                .checked_mul(bytes_per_texel)
                .and_then(|value| value.checked_mul(description.samples() as u64))
                .ok_or_else(|| unsupported("image residency size overflow"))
        }
        BackendResourceCreateInfo::Shader { module, .. } => {
            Ok(module.source().len().try_into().unwrap_or(u64::MAX))
        }
        _ => Ok(0),
    }
}

#[derive(Clone, Copy)]
struct ImageCopyShape {
    width: u32,
    height: u32,
    layers: u32,
    bytes_per_texel: usize,
    host_row_pitch: u32,
}

#[cfg(test)]
fn linearize_canonical_image(
    canonical: &[u8],
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
) -> Result<Vec<u8>, BackendDriverError> {
    let mut output = Vec::new();
    linearize_canonical_image_into(canonical, &mut output, layout, shape)?;
    Ok(output)
}

fn linearize_canonical_image_into(
    canonical: &[u8],
    output: &mut Vec<u8>,
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
) -> Result<(), BackendDriverError> {
    let output_size = u64::from(shape.host_row_pitch)
        .checked_mul(u64::from(shape.height))
        .and_then(|size| size.checked_mul(u64::from(shape.layers)))
        .ok_or_else(|| unsupported("image upload size overflow"))?;
    output.resize(usize_from_u64(output_size, "image upload size")?, 0);
    copy_image_layout(canonical, output, layout, shape, false)?;
    Ok(())
}

fn write_linear_image_to_canonical(
    linear: &[u8],
    canonical: &mut [u8],
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
) -> Result<(), BackendDriverError> {
    copy_image_layout(linear, canonical, layout, shape, true)
}

fn copy_image_layout(
    source: &[u8],
    destination: &mut [u8],
    layout: ImageMemoryLayout,
    shape: ImageCopyShape,
    to_canonical: bool,
) -> Result<(), BackendDriverError> {
    let row_bytes = u64::from(shape.width)
        .checked_mul(
            u64::try_from(shape.bytes_per_texel).map_err(|_| unsupported("image texel size"))?,
        )
        .ok_or_else(|| unsupported("image row size overflow"))?;
    match layout {
        ImageMemoryLayout::PitchLinear {
            row_pitch,
            layer_stride,
        } => {
            for layer in 0..shape.layers {
                for y in 0..shape.height {
                    let linear = image_copy_row_offset(
                        layer,
                        y,
                        u64::from(shape.host_row_pitch),
                        u64::from(shape.host_row_pitch) * u64::from(shape.height),
                    )?;
                    let canonical = image_copy_row_offset(layer, y, row_pitch, layer_stride)?;
                    copy_image_bytes(
                        source,
                        destination,
                        linear,
                        canonical,
                        usize_from_u64(row_bytes, "image row size")?,
                        to_canonical,
                    )?;
                }
            }
        }
        ImageMemoryLayout::BlockLinear(blocks) => {
            if blocks.block_width_log2 != 0 || blocks.block_depth_log2 != 0 {
                return Err(unsupported("wide or deep block-linear image layout"));
            }
            let width_in_gobs = align_u64(row_bytes, 64)? / 64;
            let block_height_gobs = 1_u64 << blocks.block_height_log2;
            for layer in 0..shape.layers {
                for y in 0..shape.height {
                    let linear = image_copy_row_offset(
                        layer,
                        y,
                        u64::from(shape.host_row_pitch),
                        u64::from(shape.host_row_pitch) * u64::from(shape.height),
                    )?;
                    let mut byte_x = 0_u64;
                    while byte_x < row_bytes {
                        let chunk = (16 - byte_x % 16).min(row_bytes - byte_x);
                        let canonical = block_linear_byte_offset(
                            blocks,
                            width_in_gobs,
                            block_height_gobs,
                            layer,
                            y,
                            byte_x,
                        )?;
                        copy_image_bytes(
                            source,
                            destination,
                            linear
                                .checked_add(usize_from_u64(byte_x, "linear image offset")?)
                                .ok_or_else(|| unsupported("linear image offset"))?,
                            canonical,
                            usize_from_u64(chunk, "image copy size")?,
                            to_canonical,
                        )?;
                        byte_x += chunk;
                    }
                }
            }
        }
    }
    Ok(())
}

fn image_copy_row_offset(
    layer: u32,
    y: u32,
    row_pitch: u64,
    layer_stride: u64,
) -> Result<usize, BackendDriverError> {
    let offset = u64::from(layer)
        .checked_mul(layer_stride)
        .and_then(|offset| {
            u64::from(y)
                .checked_mul(row_pitch)
                .and_then(|row| offset.checked_add(row))
        })
        .ok_or_else(|| unsupported("image row offset"))?;
    usize_from_u64(offset, "canonical image offset")
}

fn block_linear_byte_offset(
    blocks: BlockLinearLayout,
    width_in_gobs: u64,
    block_height_gobs: u64,
    layer: u32,
    y: u32,
    byte_x: u64,
) -> Result<usize, BackendDriverError> {
    // Tegra's generic 16Bx2 GOB addressing, also used by pinned libnx
    // framebuffer conversion:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c
    let y = u64::from(y);
    let block_rows = 8_u64
        .checked_mul(block_height_gobs)
        .ok_or_else(|| unsupported("block-linear block height"))?;
    let terms = [
        u64::from(layer)
            .checked_mul(blocks.layer_stride)
            .ok_or_else(|| unsupported("block-linear layer offset"))?,
        (y / block_rows)
            .checked_mul(512)
            .and_then(|value| value.checked_mul(block_height_gobs))
            .and_then(|value| value.checked_mul(width_in_gobs))
            .ok_or_else(|| unsupported("block-linear Y offset"))?,
        (byte_x / 64)
            .checked_mul(512)
            .and_then(|value| value.checked_mul(block_height_gobs))
            .ok_or_else(|| unsupported("block-linear X offset"))?,
        ((y % block_rows) / 8) * 512,
        ((byte_x % 64) / 32) * 256,
        ((y % 8) / 2) * 64,
        ((byte_x % 32) / 16) * 32,
        (y % 2) * 16,
        byte_x % 16,
    ];
    let offset = terms.into_iter().try_fold(0_u64, |offset, term| {
        offset
            .checked_add(term)
            .ok_or_else(|| unsupported("block-linear image offset"))
    })?;
    usize_from_u64(offset, "canonical image offset")
}

fn copy_image_bytes(
    source: &[u8],
    destination: &mut [u8],
    linear: usize,
    canonical: usize,
    size: usize,
    to_canonical: bool,
) -> Result<(), BackendDriverError> {
    let linear_end = linear
        .checked_add(size)
        .ok_or_else(|| unsupported("linear image range"))?;
    let canonical_end = canonical
        .checked_add(size)
        .ok_or_else(|| unsupported("canonical image range"))?;
    let (from, to) = if to_canonical {
        (
            source
                .get(linear..linear_end)
                .ok_or_else(|| unsupported("linear image source exceeds backing"))?,
            canonical,
        )
    } else {
        (
            source
                .get(canonical..canonical_end)
                .ok_or_else(|| unsupported("canonical image source exceeds backing"))?,
            linear,
        )
    };
    let to_end = to
        .checked_add(size)
        .ok_or_else(|| unsupported("image destination range"))?;
    destination
        .get_mut(to..to_end)
        .ok_or_else(|| unsupported("image destination exceeds backing"))?
        .copy_from_slice(from);
    Ok(())
}

fn align_u64(value: u64, alignment: u64) -> Result<u64, BackendDriverError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| unsupported("aligned image size overflow"))
}

fn align_u32(value: u32, alignment: u32) -> Result<u32, BackendDriverError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| unsupported("aligned image row pitch overflow"))
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize, BackendDriverError> {
    usize::try_from(value).map_err(|_| unsupported(label))
}

fn filter_mode(mode: nixe_gpu::FilterMode) -> wgpu::FilterMode {
    match mode {
        nixe_gpu::FilterMode::Nearest => wgpu::FilterMode::Nearest,
        nixe_gpu::FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn mip_filter_mode(mode: nixe_gpu::FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        nixe_gpu::FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        nixe_gpu::FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

fn address_mode(mode: nixe_gpu::AddressMode) -> wgpu::AddressMode {
    match mode {
        nixe_gpu::AddressMode::Repeat => wgpu::AddressMode::Repeat,
        nixe_gpu::AddressMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        nixe_gpu::AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        nixe_gpu::AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

fn unsupported(semantic: &str) -> BackendDriverError {
    BackendDriverError::failure(format!(
        "wgpu backend cannot represent neutral semantic: {semantic}"
    ))
}

fn missing(handle: BackendResourceHandle) -> BackendDriverError {
    BackendDriverError::failure(format!("missing wgpu resource: {handle}"))
}

fn kind_mismatch(handle: BackendResourceHandle) -> BackendDriverError {
    BackendDriverError::failure(format!("wgpu resource kind mismatch: {handle}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nixe_gpu::{
        BackendInstanceId, BackendResourceHandle, BackendResourceKind, BlockLinearLayout, BufferId,
        BufferRange, BufferRegion, DepthCompareOperation, DepthState, ImageDescription,
        ImageDimension, ImageExtent, ImageFormat, ImageKind, ImageMemoryLayout, PrimitiveTopology,
        SampleCount, TriangleRasterization, VertexAttribute, VertexBufferLayout, VertexFormat,
        VertexStepMode, ViewportTransform,
    };
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};
    use wgpu::{CompareFunction, TextureFormat, TextureUsages, TextureViewDimension};

    use super::{
        DemandedBufferWriteback, DemandedWriteback, DeviceWrite, DeviceWriteRegion, ImageCopyShape,
        MAX_DEVICE_WRITE_REGIONS, TransferRange, WebGpuViewport, collect_dirty_buffer_ranges,
        compare_function, image_texture_plan, least_recent_key, linearize_canonical_image,
        prepare_demanded_writebacks, record_buffer_device_write, sampled_texture_view_descriptor,
        subtract_completed_buffer_write, vertex_entry_point, webgpu_viewport,
        write_linear_image_to_canonical,
    };

    #[test]
    fn bounded_backend_caches_select_the_least_recently_used_entry() {
        let records = HashMap::from([(11_u32, 40_u64), (22, 10), (33, 30)]);

        assert_eq!(least_recent_key(&records, |last_used| *last_used), Some(22));
    }

    #[test]
    fn pipeline_vertex_layout_ignores_buffer_identity_but_keeps_shader_specialization() {
        let layout = |buffer, offset, format| {
            VertexBufferLayout::new(
                BufferRegion {
                    buffer: BufferId::new(buffer),
                    range: BufferRange::new(offset, 64).unwrap(),
                },
                8,
                VertexStepMode::Vertex,
                vec![VertexAttribute {
                    format,
                    offset: 0,
                    shader_location: 0,
                }],
            )
            .unwrap()
        };
        let native = super::VertexPipelineLayoutKey::new(&layout(1, 0, VertexFormat::Float32x2));
        assert!(native.matches(&layout(2, 16, VertexFormat::Float32x2)));

        let pulled = super::VertexPipelineLayoutKey::new(&layout(1, 4, VertexFormat::Uint8x2));
        assert!(pulled.matches(&layout(2, 4, VertexFormat::Uint8x2)));
        assert!(!pulled.matches(&layout(2, 8, VertexFormat::Uint8x2)));

        let shader = |slot| {
            BackendResourceHandle::new(
                BackendInstanceId::new(1),
                slot,
                1,
                BackendResourceKind::Shader,
            )
        };
        let fingerprint = |layout: &VertexBufferLayout| {
            nixe_gpu::cache_fingerprint(&super::RenderPipelineFingerprintInput {
                vertex: shader(1),
                fragment: shader(2),
                topology: PrimitiveTopology::Triangles,
                triangle_rasterization: TriangleRasterization::Fill,
                alpha_test: None,
                color_format: ImageFormat::Rgba8Unorm,
                depth_format: None,
                depth_state: DepthState::DISABLED,
                vertex_buffers: std::slice::from_ref(layout),
            })
        };
        assert_eq!(
            fingerprint(&layout(1, 0, VertexFormat::Float32x2)),
            fingerprint(&layout(2, 16, VertexFormat::Float32x2))
        );
        assert_eq!(
            fingerprint(&layout(1, 4, VertexFormat::Uint8x2)),
            fingerprint(&layout(2, 4, VertexFormat::Uint8x2))
        );
        assert_ne!(
            fingerprint(&layout(1, 4, VertexFormat::Uint8x2)),
            fingerprint(&layout(2, 8, VertexFormat::Uint8x2))
        );
    }

    #[test]
    fn buffer_upload_tracking_selects_only_modified_aligned_bytes() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let uploaded = range
            .segments()
            .iter()
            .map(|segment| segment.current_content_generation())
            .collect::<Vec<_>>();
        let mut page_dirty = Vec::new();
        let mut dirty = Vec::new();
        collect_dirty_buffer_ranges(&range, &uploaded, 0, &mut page_dirty, &mut dirty).unwrap();
        assert!(dirty.is_empty());

        allocation.write(0x1101, &[1, 2]).unwrap();
        collect_dirty_buffer_ranges(&range, &uploaded, 0, &mut page_dirty, &mut dirty).unwrap();
        assert_eq!(
            dirty,
            [TransferRange {
                offset: 0x1100,
                size: 4
            }]
        );
    }

    #[test]
    fn newer_device_writes_replace_only_overlapping_bytes() {
        let mut writes = Vec::new();
        record_buffer_device_write(
            &mut writes,
            TransferRange {
                offset: 0,
                size: 100,
            },
            1,
        )
        .unwrap();
        record_buffer_device_write(
            &mut writes,
            TransferRange {
                offset: 40,
                size: 20,
            },
            2,
        )
        .unwrap();
        writes.sort_unstable_by_key(|write| match write.region {
            DeviceWriteRegion::Buffer(range) => range.offset,
            DeviceWriteRegion::ImageBinding(_) => u64::MAX,
        });

        assert_eq!(
            writes,
            [
                DeviceWrite {
                    region: DeviceWriteRegion::Buffer(TransferRange {
                        offset: 0,
                        size: 40
                    }),
                    serial: 1
                },
                DeviceWrite {
                    region: DeviceWriteRegion::Buffer(TransferRange {
                        offset: 40,
                        size: 20
                    }),
                    serial: 2
                },
                DeviceWrite {
                    region: DeviceWriteRegion::Buffer(TransferRange {
                        offset: 60,
                        size: 40
                    }),
                    serial: 1
                }
            ]
        );

        subtract_completed_buffer_write(
            &mut writes,
            TransferRange {
                offset: 44,
                size: 8,
            },
            2,
        );
        let device_newer_bytes = writes
            .iter()
            .map(|write| match write.region {
                DeviceWriteRegion::Buffer(range) => range.size,
                DeviceWriteRegion::ImageBinding(_) => 0,
            })
            .sum::<u64>();
        assert_eq!(device_newer_bytes, 92);
    }

    #[test]
    fn device_write_provenance_is_strictly_bounded() {
        let mut writes = Vec::new();
        for offset in 0..MAX_DEVICE_WRITE_REGIONS {
            record_buffer_device_write(
                &mut writes,
                TransferRange {
                    offset: (offset * 2) as u64,
                    size: 1,
                },
                offset as u64 + 1,
            )
            .unwrap();
        }
        assert_eq!(writes.len(), MAX_DEVICE_WRITE_REGIONS);
        assert!(
            record_buffer_device_write(
                &mut writes,
                TransferRange {
                    offset: (MAX_DEVICE_WRITE_REGIONS * 2) as u64,
                    size: 1,
                },
                MAX_DEVICE_WRITE_REGIONS as u64 + 1,
            )
            .is_err()
        );
    }

    #[test]
    fn demanded_alias_writebacks_keep_last_writer_order_and_coalesce() {
        let instance = BackendInstanceId::new(1);
        let older = BackendResourceHandle::new(instance, 0, 1, BackendResourceKind::Buffer);
        let newer = BackendResourceHandle::new(instance, 1, 1, BackendResourceKind::Buffer);
        let mut demanded = vec![
            DemandedWriteback::Buffer(DemandedBufferWriteback {
                handle: newer,
                serial: 2,
                range: TransferRange { offset: 4, size: 4 },
                page_offset: 4,
            }),
            DemandedWriteback::Buffer(DemandedBufferWriteback {
                handle: older,
                serial: 1,
                range: TransferRange { offset: 0, size: 4 },
                page_offset: 0,
            }),
            DemandedWriteback::Buffer(DemandedBufferWriteback {
                handle: older,
                serial: 1,
                range: TransferRange { offset: 4, size: 4 },
                page_offset: 4,
            }),
        ];

        prepare_demanded_writebacks(&mut demanded);

        assert_eq!(demanded.len(), 2);
        assert_eq!(demanded[0].serial(), 1);
        assert_eq!(demanded[1].serial(), 2);
        assert_eq!(
            demanded[0],
            DemandedWriteback::Buffer(DemandedBufferWriteback {
                handle: older,
                serial: 1,
                range: TransferRange { offset: 0, size: 8 },
                page_offset: 0,
            })
        );
    }

    #[test]
    fn vertex_entry_points_compose_pulling_and_rectangle_expansion() {
        assert_eq!(
            vertex_entry_point(false, TriangleRasterization::Fill),
            "main"
        );
        assert_eq!(
            vertex_entry_point(false, TriangleRasterization::FillRectangle),
            "nixe_fill_rectangle"
        );
        assert_eq!(
            vertex_entry_point(true, TriangleRasterization::Fill),
            "nixe_vertex_pull"
        );
        assert_eq!(
            vertex_entry_point(true, TriangleRasterization::FillRectangle),
            "nixe_vertex_pull_fill_rectangle"
        );
    }

    #[test]
    fn two_dimensional_image_arrays_create_explicit_array_views() {
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(64, 32, 1).unwrap(),
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            1,
            7,
            SampleCount::One,
        )
        .unwrap();

        let view = sampled_texture_view_descriptor(description);
        assert_eq!(view.dimension, Some(TextureViewDimension::D2Array));
        assert_eq!(view.array_layer_count, Some(7));
    }

    #[test]
    fn neutral_depth_comparisons_map_exactly_to_wgpu() {
        let cases = [
            (DepthCompareOperation::Never, CompareFunction::Never),
            (DepthCompareOperation::Less, CompareFunction::Less),
            (DepthCompareOperation::Equal, CompareFunction::Equal),
            (DepthCompareOperation::LessEqual, CompareFunction::LessEqual),
            (DepthCompareOperation::Greater, CompareFunction::Greater),
            (DepthCompareOperation::NotEqual, CompareFunction::NotEqual),
            (
                DepthCompareOperation::GreaterEqual,
                CompareFunction::GreaterEqual,
            ),
            (DepthCompareOperation::Always, CompareFunction::Always),
        ];
        for (neutral, wgpu) in cases {
            assert_eq!(compare_function(neutral), wgpu);
        }
    }

    #[test]
    fn transient_d24s8_uses_the_opaque_wgpu_attachment_format() {
        let plan = image_texture_plan(ImageFormat::Depth24UnormStencil8Uint, false).unwrap();

        assert_eq!(plan.format, TextureFormat::Depth24PlusStencil8);
        assert_eq!(plan.usages, TextureUsages::RENDER_ATTACHMENT);
    }

    #[test]
    fn canonical_d24s8_requires_an_explicit_conversion() {
        let error = image_texture_plan(ImageFormat::Depth24UnormStencil8Uint, true).unwrap_err();

        assert!(error.to_string().contains("canonical D24S8 transfer"));
    }

    #[test]
    fn ordinary_images_keep_the_conservative_transfer_usage_set() {
        let plan = image_texture_plan(ImageFormat::Rgba8Unorm, true).unwrap();

        assert_eq!(plan.format, TextureFormat::Rgba8Unorm);
        assert!(plan.usages.contains(TextureUsages::COPY_SRC));
        assert!(plan.usages.contains(TextureUsages::COPY_DST));
        assert!(plan.usages.contains(TextureUsages::TEXTURE_BINDING));
        assert!(plan.usages.contains(TextureUsages::RENDER_ATTACHMENT));
    }

    #[test]
    fn maxwell_negative_y_viewport_maps_exactly_to_webgpu_top_left_coordinates() {
        let transform =
            ViewportTransform::new([32.0, -16.0, 0.5], [32.0, 16.0, 0.5], [0.0, 1.0]).unwrap();

        assert_eq!(
            webgpu_viewport(transform).unwrap(),
            WebGpuViewport {
                x: 0.0,
                y: 0.0,
                width: 64.0,
                height: 32.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        );
    }

    #[test]
    fn viewport_axis_signs_without_an_exact_webgpu_mapping_remain_typed_failures() {
        let flipped_x =
            ViewportTransform::new([-32.0, -16.0, 0.5], [32.0, 16.0, 0.5], [0.0, 1.0]).unwrap();
        let flipped_y =
            ViewportTransform::new([32.0, 16.0, 0.5], [32.0, 16.0, 0.5], [0.0, 1.0]).unwrap();

        assert!(webgpu_viewport(flipped_x).is_err());
        assert!(webgpu_viewport(flipped_y).is_err());
    }

    #[test]
    fn maxwell_zero_to_one_depth_range_is_not_inferred_from_scale_and_offset() {
        let transform =
            ViewportTransform::new([640.0, -360.0, 1.0], [640.0, 360.0, 0.0], [0.0, 1.0]).unwrap();

        let viewport = webgpu_viewport(transform).unwrap();
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn unrepresentable_depth_ranges_fail_before_wgpu_validation() {
        let transform =
            ViewportTransform::new([32.0, -16.0, 1.0], [32.0, 16.0, 0.0], [-1.0, 1.0]).unwrap();

        assert!(webgpu_viewport(transform).is_err());
    }

    #[test]
    fn block_linear_image_round_trips_through_host_rows() {
        let layout = ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            layer_stride: 512,
        });
        let mut host = vec![0_u8; 256 * 8];
        for y in 0..8_usize {
            for x in 0..8_usize {
                let offset = y * 256 + x * 4;
                host[offset..offset + 4].copy_from_slice(&[
                    u8::try_from(x).unwrap(),
                    u8::try_from(y).unwrap(),
                    0x5a,
                    0xff,
                ]);
            }
        }
        let mut canonical = vec![0_u8; 512];
        let shape = ImageCopyShape {
            width: 8,
            height: 8,
            layers: 1,
            bytes_per_texel: 4,
            host_row_pitch: 256,
        };
        write_linear_image_to_canonical(&host, &mut canonical, layout, shape).unwrap();

        assert_eq!(
            linearize_canonical_image(&canonical, layout, shape).unwrap(),
            host
        );
    }

    #[test]
    fn block_linear_bulk_copy_matches_independent_texel_addressing() {
        let blocks = BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: 2,
            block_depth_log2: 0,
            layer_stride: 0x1000,
        };
        let layout = ImageMemoryLayout::BlockLinear(blocks);
        let shape = ImageCopyShape {
            width: 19,
            height: 17,
            layers: 2,
            bytes_per_texel: 4,
            host_row_pitch: 256,
        };
        let reference_offset = |layer: u32, x: u32, y: u32| {
            let byte_x = u64::from(x) * 4;
            let block_height_gobs = 4_u64;
            usize::try_from(
                u64::from(layer) * blocks.layer_stride
                    + (u64::from(y) / (8 * block_height_gobs)) * 512 * block_height_gobs * 2
                    + (byte_x / 64) * 512 * block_height_gobs
                    + ((u64::from(y) % (8 * block_height_gobs)) / 8) * 512
                    + ((byte_x % 64) / 32) * 256
                    + ((u64::from(y) % 8) / 2) * 64
                    + ((byte_x % 32) / 16) * 32
                    + (u64::from(y) % 2) * 16
                    + byte_x % 16,
            )
            .unwrap()
        };
        let linear_offset = |layer: u32, x: u32, y: u32| {
            usize::try_from(u64::from(layer * shape.height + y) * 256 + u64::from(x) * 4).unwrap()
        };
        let canonical = (0..0x2000)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect::<Vec<_>>();
        let mut expected_host = vec![0_u8; 256 * 17 * 2];
        for layer in 0..shape.layers {
            for y in 0..shape.height {
                for x in 0..shape.width {
                    let canonical_offset = reference_offset(layer, x, y);
                    let linear_offset = linear_offset(layer, x, y);
                    expected_host[linear_offset..linear_offset + 4]
                        .copy_from_slice(&canonical[canonical_offset..canonical_offset + 4]);
                }
            }
        }

        assert_eq!(
            linearize_canonical_image(&canonical, layout, shape).unwrap(),
            expected_host
        );

        let mut expected_canonical = vec![0xa5_u8; canonical.len()];
        for layer in 0..shape.layers {
            for y in 0..shape.height {
                for x in 0..shape.width {
                    let canonical_offset = reference_offset(layer, x, y);
                    let linear_offset = linear_offset(layer, x, y);
                    expected_canonical[canonical_offset..canonical_offset + 4]
                        .copy_from_slice(&expected_host[linear_offset..linear_offset + 4]);
                }
            }
        }
        let mut observed_canonical = vec![0xa5_u8; canonical.len()];
        write_linear_image_to_canonical(&expected_host, &mut observed_canonical, layout, shape)
            .unwrap();
        assert_eq!(observed_canonical, expected_canonical);
    }

    #[test]
    fn pitch_image_writeback_preserves_canonical_row_padding() {
        let layout = ImageMemoryLayout::PitchLinear {
            row_pitch: 16,
            layer_stride: 32,
        };
        let host = [
            1_u8, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let mut padded_host = vec![0_u8; 256 * 2];
        padded_host[..8].copy_from_slice(&host[..8]);
        padded_host[256..264].copy_from_slice(&host[16..24]);
        let mut canonical = vec![0xaa_u8; 32];
        write_linear_image_to_canonical(
            &padded_host,
            &mut canonical,
            layout,
            ImageCopyShape {
                width: 2,
                height: 2,
                layers: 1,
                bytes_per_texel: 4,
                host_row_pitch: 256,
            },
        )
        .unwrap();
        assert_eq!(&canonical[..8], &host[..8]);
        assert_eq!(&canonical[16..24], &host[16..24]);
    }
}
