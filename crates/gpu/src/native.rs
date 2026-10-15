//! Immutable guest facts at the native execution boundary. No host handles,
//! resource-creation transactions, pass commands or aggregate access plans.
use crate::*;
use nixe_memory::CanonicalBackingRange;
use std::sync::Arc;

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
#[derive(Clone, Debug)]
pub enum GuestBinding {
    Buffer(CanonicalBackingRange),
    Image(GuestImage),
    Sampler(SamplerDescription),
}
#[derive(Clone, Debug)]
pub struct GuestVertexBuffer {
    pub binding: u32,
    pub backing: CanonicalBackingRange,
    pub stride: u32,
    pub step: VertexStepMode,
    pub attributes: Box<[VertexAttribute]>,
}
#[derive(Clone, Debug)]
pub struct GuestShaderCode {
    pub identity: ssa::artifact::ContentId,
    pub functions: Box<[ssa::Function]>,
}
#[derive(Clone, Debug)]
pub struct GuestProgram {
    pub identity: PreparedStateId,
    pub shader: Arc<GuestShaderCode>,
    /// Sorted logical SSA resource identities; Vulkan reflection assigns dense bindings.
    pub bindings: Box<[(ssa::ResourceId, GuestBinding)]>,
    pub shared_memory_bytes: u32,
    pub local_memory_bytes: u32,
}
#[derive(Clone, Debug)]
pub struct GuestDrawState {
    pub program: GuestProgram,
    pub vertices: Box<[GuestVertexBuffer]>,
    pub index: Option<(CanonicalBackingRange, IndexType)>,
    pub colors: Box<[GuestImage]>,
    pub depth_stencil: Option<GuestImage>,
    pub extent: [u32; 2],
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
#[derive(Clone, Debug)]
pub enum NativeCommand {
    Sequence(Box<[NativeCommand]>),
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
