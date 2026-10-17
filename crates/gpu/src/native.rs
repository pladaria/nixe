//! Immutable guest facts at the native execution boundary. No host handles,
//! resource-creation transactions, pass commands or aggregate access plans.
use crate::*;
use nixe_memory::CanonicalBackingRange;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Framebuffer source legalized by the console frontend. No guest tiling or
/// logical allocation IDs cross the native presentation boundary.
#[derive(Clone, Debug)]
pub struct NativePresentationImage {
    pub planes: Arc<[GuestImagePlane]>,
    pub extent: [u32; 2],
    pub format: crate::PresentationImageFormat,
    pub allow_canonical_import: bool,
}

#[derive(Clone, Debug)]
pub struct GuestImage {
    pub planes: Arc<[GuestImagePlane]>,
    pub description: ImageDescription,
    pub swizzle: Swizzle,
    /// Frontend-defined sample grid in each canonical plane. Native raster
    /// coordinates stay in pixels; byte authority covers every stored sample.
    pub sample_grid: [u32; 2],
    /// Opaque compression/MSAA bytes cannot be imported as ordinary texels.
    pub canonical_texels: bool,
}
impl GuestImage {
    pub fn same_view(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.planes, &other.planes)
            && self.description == other.description
            && self.swizzle == other.swizzle
            && self.sample_grid == other.sample_grid
            && self.canonical_texels == other.canonical_texels
    }
}
#[derive(Clone, Debug)]
pub enum GuestBinding {
    Buffer(CanonicalBackingRange),
    Image {
        image: GuestImage,
        /// Shader view dimensionality remains arrayed for a single layer.
        arrayed: bool,
    },
    Sampler(SamplerDescription),
}
#[derive(Clone, Debug)]
pub struct GuestVertexBuffer {
    pub binding: u32,
    pub backing: CanonicalBackingRange,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestVertexLayout {
    pub binding: u32,
    pub stride: u32,
    pub step: VertexStepMode,
    pub attributes: Box<[VertexAttribute]>,
}
#[derive(Clone, Debug)]
pub struct GuestShaderCode {
    pub identity: ssa::artifact::ContentId,
    pub functions: Box<[Arc<ssa::Function>]>,
}
#[derive(Clone, Debug)]
pub struct GuestProgram {
    pub identity: PreparedStateId,
    pub shader: Arc<GuestShaderCode>,
    /// Sorted logical SSA resource identities; Vulkan reflection assigns dense bindings.
    pub bindings: Box<[(ssa::ResourceId, GuestBinding)]>,
    pub shared_memory_bytes: u32,
}
#[derive(Clone, Debug)]
pub struct GuestDrawState {
    pub identity: PreparedStateId,
    pub program: Arc<GuestProgram>,
    pub vertices: Box<[GuestVertexBuffer]>,
    pub index: Option<(CanonicalBackingRange, IndexType)>,
    pub colors: Box<[GuestImage]>,
    pub depth_stencil: Option<GuestImage>,
    pub parameters: Arc<GuestDrawParameters>,
}
/// Validated draw semantics independent of current physical buffer/descriptor
/// bindings. Address changes retain this identity and its compiled preparation.
#[derive(Clone, Debug)]
pub struct GuestDrawParameters {
    pub identity: PreparedStateId,
    pub vertices: Box<[GuestVertexLayout]>,
    pub color_formats: Box<[ImageFormat]>,
    pub depth_stencil_format: Option<ImageFormat>,
    pub samples: SampleCount,
    pub extent: [u32; 2],
    /// Attachment load/store bounds, independent of the viewport and dynamic
    /// scissor. The allocated image can include texels outside this region.
    pub render_area: ScissorRect,
    pub topology: PrimitiveTopology,
    pub restart: bool,
    pub provoking_first: bool,
    pub raster: TriangleRasterization,
    pub line: Option<LineRasterization>,
    pub front_face: FrontFace,
    pub cull: CullMode,
    pub depth_bias: Option<DepthBias>,
    pub viewport: Option<ViewportTransform>,
    pub position_transform: Option<VertexPositionTransform>,
    pub scissor: ScissorRect,
    pub depth: DepthState,
    pub stencil: Option<StencilTestState>,
    pub color_outputs: Box<[ColorOutputState]>,
    pub blend_constants: [f32; 4],
    pub alpha_test: Option<AlphaTest>,
    pub tessellation: Option<TessellationState>,
    pub sample_mask: u64,
    pub alpha_to_coverage: bool,
}
impl GuestDrawParameters {
    /// Hash the consumed state, excluding its interned identity.
    pub fn hash_state<H: Hasher>(&self, state: &mut H) {
        self.vertices.hash(state);
        self.color_formats.hash(state);
        self.depth_stencil_format.hash(state);
        self.samples.hash(state);
        self.extent.hash(state);
        self.render_area.hash(state);
        self.topology.hash(state);
        self.restart.hash(state);
        self.provoking_first.hash(state);
        self.raster.hash(state);
        self.line.hash(state);
        self.front_face.hash(state);
        self.cull.hash(state);
        self.depth_bias.hash(state);
        self.viewport.hash(state);
        self.position_transform.hash(state);
        self.scissor.hash(state);
        self.depth.hash(state);
        self.stencil.hash(state);
        self.color_outputs.hash(state);
        self.alpha_test.hash(state);
        self.tessellation.hash(state);
        self.sample_mask.hash(state);
        self.alpha_to_coverage.hash(state);
        self.blend_constants.map(f32::to_bits).hash(state);
    }
    /// Compare consumed semantics when register revisions changed and then
    /// restored a prior value. Identity/provenance is not pipeline state.
    pub fn same_state(&self, other: &Self) -> bool {
        self.vertices == other.vertices
            && self.color_formats == other.color_formats
            && self.depth_stencil_format == other.depth_stencil_format
            && self.samples == other.samples
            && self.extent == other.extent
            && self.render_area == other.render_area
            && self.topology == other.topology
            && self.restart == other.restart
            && self.provoking_first == other.provoking_first
            && self.raster == other.raster
            && self.line == other.line
            && self.front_face == other.front_face
            && self.cull == other.cull
            && self.depth_bias == other.depth_bias
            && self.viewport == other.viewport
            && self.position_transform == other.position_transform
            && self.scissor == other.scissor
            && self.depth == other.depth
            && self.stencil == other.stencil
            && self.color_outputs == other.color_outputs
            && self.blend_constants.map(f32::to_bits) == other.blend_constants.map(f32::to_bits)
            && self.alpha_test == other.alpha_test
            && self.tessellation == other.tessellation
            && self.sample_mask == other.sample_mask
            && self.alpha_to_coverage == other.alpha_to_coverage
    }
}
#[derive(Clone, Debug)]
pub enum NativeCommand {
    Sequence(Box<[NativeCommand]>),
    /// Orders the device stream without waiting for it on the host.
    WaitForIdle,
    /// Ordered command-processor write. Inline writes to CPU-owned storage are
    /// committed at this recording position after prior input snapshots. Release
    /// writes and device-owned targets require actual producer completion for
    /// CPU visibility, including when the payload is already known.
    WriteMemory {
        target: CanonicalBackingRange,
        offset: u64,
        bytes: Arc<[u8]>,
        after_prior_work: bool,
    },
    Draw {
        state: Arc<GuestDrawState>,
        arguments: DrawArguments,
    },
    Dispatch {
        program: Arc<GuestProgram>,
        groups: [u32; 3],
    },
    Clear {
        image: GuestImage,
        rectangle: ScissorRect,
        value: ClearValue,
    },
    Blit {
        source: GuestImage,
        destination: GuestImage,
        source_rectangle: ScissorRect,
        destination_rectangle: ScissorRect,
        linear: bool,
        resolve: bool,
        sampling: Option<([i64; 2], [i64; 2])>,
    },
    Ordering(CacheMaintenanceOperation),
    Barrier(BarrierOperation),
    ConvertNv12 {
        images: Box<[GuestImage; 3]>,
        transform: Nv12ColorTransform,
    },
    UploadPlanes {
        images: Box<[GuestImage; 2]>,
        bytes: [Arc<[u8]>; 2],
        widths: [u32; 2],
    },
    ImportDecodedPlanes {
        images: Box<[GuestImage; 2]>,
        frame: DeviceImageFrame,
    },
}
#[derive(Clone, Debug)]
pub struct NativeWork {
    pub frontend: FrontendSubmissionId,
    pub command: NativeCommand,
    /// Cached coverage is consumed only at command-processor write boundaries.
    pub canonical_accesses: Arc<[BackingView]>,
}
impl NativeWork {
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + command_bytes(&self.command)
    }
}
fn command_bytes(command: &NativeCommand) -> usize {
    match command {
        NativeCommand::WriteMemory { bytes, .. } => bytes.len(),
        NativeCommand::UploadPlanes { bytes, .. } => bytes.iter().map(|b| b.len()).sum(),
        NativeCommand::Sequence(commands) => {
            std::mem::size_of_val(commands.as_ref())
                + commands.iter().map(command_bytes).sum::<usize>()
        }
        _ => 0,
    }
}
pub struct NativeSubmission {
    pub frontend: FrontendSubmissionId,
    pub segment: FrontendSubmissionSegment,
    pub final_segment: bool,
    pub commands: Vec<NativeCommand>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GuestImageAspect {
    Color,
    Depth,
    Stencil,
}
#[derive(Clone, Debug)]
pub struct GuestImagePlane {
    pub mip_level: u32,
    pub array_layer: u32,
    pub aspect: GuestImageAspect,
    pub backing: CanonicalBackingRange,
    pub row_pitch: u64,
    pub bytes_per_block: u32,
}
