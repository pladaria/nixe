//! Immutable guest facts at the native execution boundary. No host handles,
//! resource-creation transactions, pass commands or aggregate access plans.
use crate::physical::{Footprint, PhysicalMapping};
use crate::*;
use nixe_memory::AllocationRange;
use std::{
    cell::RefCell,
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Framebuffer source legalized by the console frontend. No guest tiling or
/// logical allocation IDs cross the native presentation boundary.
#[derive(Clone, Debug)]
pub struct NativePresentationImage {
    pub inputs: Vec<AllocationRange>,
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
    /// An explicitly unbound buffer: reads return zero and writes are discarded.
    NullBuffer,
    Buffer(Arc<PhysicalMapping>),
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
    pub backing: Arc<PhysicalMapping>,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestVertexLayout {
    pub binding: u32,
    pub stride: u32,
    pub step: VertexStepMode,
    pub attributes: Box<[VertexAttribute]>,
}
pub type GuestShaderCode = crate::shader_compilation::ShaderInputs;
#[derive(Clone, Debug)]
pub struct GuestProgram {
    pub identity: PreparedStateId,
    pub shader: Arc<GuestShaderCode>,
    /// Sorted semantic resource identities; artifact metadata assigns dense bindings.
    pub bindings: Box<[(crate::shader_module::ResourceId, GuestBinding)]>,
    pub shared_memory_bytes: u32,
}
/// Index bytes come either from guest memory or from ordered inline methods.
#[derive(Clone, Debug)]
pub enum GuestIndexBuffer {
    Canonical(Arc<PhysicalMapping>, IndexType),
    Inline(Arc<[u8]>),
}
#[derive(Clone, Debug)]
pub struct GuestDrawState {
    pub identity: PreparedStateId,
    pub program: Arc<GuestProgram>,
    pub vertices: Box<[GuestVertexBuffer]>,
    pub index: Option<GuestIndexBuffer>,
    pub colors: Box<[GuestImage]>,
    pub depth_stencil: Option<GuestImage>,
    pub parameters: Arc<GuestDrawParameters>,
}
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GuestShaderState {
    pub base_instance: u32,
    pub point_size_bits: Option<u32>,
    pub y_negate: bool,
    pub force_early_z: bool,
}
/// Validated draw semantics independent of current physical buffer/descriptor
/// bindings. Address changes retain this identity and its compiled preparation.
#[derive(Clone, Debug)]
pub struct GuestDrawParameters {
    pub shader_state: GuestShaderState,
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
        self.shader_state.hash(state);
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
        self.shader_state == other.shader_state
            && self.vertices == other.vertices
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
    /// Ordered report payload. A fence commits preceding buffer publications.
    /// Ordinary inline writes execute directly on the command owner.
    ReportMemory {
        target: Vec<AllocationRange>,
        offset: u64,
        bytes: Arc<[u8]>,
        fence: bool,
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
}
pub struct NativeSubmission {
    pub frontend: FrontendSubmissionId,
    pub segment: FrontendSubmissionSegment,
    pub final_segment: bool,
    pub commands: Vec<NativeCommand>,
    // Only a suspended preparation acquires RAM ownership. Warm consumption
    // remains direct, and weak prepared-state caches cannot retain these pins.
    suspended_inputs: RefCell<Option<Vec<AllocationRange>>>,
}
impl NativeSubmission {
    pub fn new(
        frontend: FrontendSubmissionId,
        segment: FrontendSubmissionSegment,
        final_segment: bool,
        commands: Vec<NativeCommand>,
    ) -> Self {
        Self {
            frontend,
            segment,
            final_segment,
            commands,
            suspended_inputs: RefCell::new(None),
        }
    }
    /// Retain the original physical inputs before servicing control during a
    /// cold compile wait. Repeated polls acquire no additional pins. The batch
    /// owns them through actual consumption; cache entries remain weak.
    pub fn retain_suspended_inputs(&self) -> Result<(), nixe_memory::MemoryError> {
        let mut retained = self.suspended_inputs.borrow_mut();
        if retained.is_none() {
            let mut inputs = Vec::new();
            for command in &self.commands {
                command.retain_inputs(&mut inputs)?;
            }
            *retained = Some(inputs);
        }
        Ok(())
    }
}
impl NativeCommand {
    fn retain_inputs(
        &self,
        inputs: &mut Vec<AllocationRange>,
    ) -> Result<(), nixe_memory::MemoryError> {
        fn buffer(
            input: &PhysicalMapping,
            inputs: &mut Vec<AllocationRange>,
        ) -> Result<(), nixe_memory::MemoryError> {
            inputs.extend(input.pin(0, input.bytes())?);
            Ok(())
        }
        fn image(
            input: &GuestImage,
            inputs: &mut Vec<AllocationRange>,
        ) -> Result<(), nixe_memory::MemoryError> {
            for plane in input.planes.iter() {
                let range = plane.footprint.bounds();
                inputs.extend(
                    plane
                        .footprint
                        .mapping()
                        .pin(range.start, range.end - range.start)?,
                );
            }
            Ok(())
        }
        fn program(
            input: &GuestProgram,
            inputs: &mut Vec<AllocationRange>,
        ) -> Result<(), nixe_memory::MemoryError> {
            for (_, binding) in &input.bindings {
                match binding {
                    GuestBinding::Buffer(input) => buffer(input, inputs)?,
                    GuestBinding::Image { image: input, .. } => image(input, inputs)?,
                    GuestBinding::NullBuffer | GuestBinding::Sampler(_) => {}
                }
            }
            Ok(())
        }
        match self {
            Self::Sequence(commands) => {
                for command in commands {
                    command.retain_inputs(inputs)?;
                }
            }
            Self::Draw { state, .. } => {
                program(&state.program, inputs)?;
                for vertex in &state.vertices {
                    buffer(&vertex.backing, inputs)?;
                }
                if let Some(GuestIndexBuffer::Canonical(input, _)) = &state.index {
                    buffer(input, inputs)?;
                }
                for input in state.colors.iter().chain(state.depth_stencil.iter()) {
                    image(input, inputs)?;
                }
            }
            Self::Dispatch { program: input, .. } => program(input, inputs)?,
            Self::Clear { image: input, .. } => image(input, inputs)?,
            Self::Blit {
                source,
                destination,
                ..
            } => {
                image(source, inputs)?;
                image(destination, inputs)?;
            }
            Self::ConvertNv12 { images, .. } => {
                for input in images.iter() {
                    image(input, inputs)?;
                }
            }
            Self::UploadPlanes { images, .. } | Self::ImportDecodedPlanes { images, .. } => {
                for input in images.iter() {
                    image(input, inputs)?;
                }
            }
            // ReportMemory already owns its exact AllocationRange values.
            Self::ReportMemory { .. }
            | Self::WaitForIdle
            | Self::Ordering(_)
            | Self::Barrier(_) => {}
        }
        Ok(())
    }
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
    pub footprint: Arc<Footprint>,
    pub row_pitch: u64,
    pub bytes_per_block: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{Allocation, Machine, PAGE_SIZE};

    fn program(backing: Arc<PhysicalMapping>) -> Arc<GuestProgram> {
        Arc::new(GuestProgram {
            identity: PreparedStateId::new(1, 1),
            shader: Arc::new(GuestShaderCode {
                identity: crate::shader_identity::ContentId::of(b"suspended-input-test", &[]),
                stages: Box::new([]),
            }),
            bindings: Box::new([(
                crate::shader_module::ResourceId(0),
                GuestBinding::Buffer(backing),
            )]),
            shared_memory_bytes: 0,
        })
    }
    fn input(allocation: &Allocation, start: u64, bytes: u64) -> Arc<PhysicalMapping> {
        PhysicalMapping::new(&[AllocationRange::new(
            allocation.pin_range(start, bytes).unwrap(),
            start,
            bytes,
        )
        .unwrap()])
        .unwrap()
    }
    fn batch(program: Arc<GuestProgram>) -> NativeSubmission {
        NativeSubmission::new(
            FrontendSubmissionId::new(1),
            FrontendSubmissionSegment::FIRST,
            true,
            vec![NativeCommand::Dispatch {
                program,
                groups: [1, 1, 1],
            }],
        )
    }

    #[test]
    fn suspended_input_keeps_its_original_page_until_consumption_and_caches_remain_weak() {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let account = machine.account(3 * PAGE_SIZE).unwrap();
        let allocation = account.allocate(3 * PAGE_SIZE).unwrap();
        let mapping = input(&allocation, PAGE_SIZE + 17, 8);
        let cached = program(Arc::clone(&mapping));
        let work = batch(Arc::clone(&cached));
        work.retain_suspended_inputs().unwrap();
        for _ in 0..8 {
            work.retain_suspended_inputs().unwrap();
        }
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        let recycled = account.allocate(2 * PAGE_SIZE).unwrap();
        machine
            .write_resolved(&mapping.pin(0, 8).unwrap(), &[0x5a; 8])
            .unwrap();
        let mut bytes = [0; 8];
        machine
            .capture_stream(&mapping.pin(0, 8).unwrap(), &mut bytes)
            .unwrap();
        assert_eq!(bytes, [0x5a; 8]);
        drop(work);
        assert_eq!(machine.statistics().allocated_bytes, 2 * PAGE_SIZE);
        assert!(
            mapping.pin(0, 8).is_err(),
            "prepared/cache references must not own RAM"
        );
        drop((cached, recycled));
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn warm_metadata_and_a_failed_suspension_capture_cannot_retain_guest_ram() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let first = account.allocate(PAGE_SIZE).unwrap();
        let retired = account.allocate(PAGE_SIZE).unwrap();
        let retired_mapping = input(&retired, 0, 4);
        let unused = batch(program(Arc::clone(&retired_mapping)));
        drop(retired);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        let work = NativeSubmission::new(
            FrontendSubmissionId::new(2),
            FrontendSubmissionSegment::FIRST,
            true,
            vec![NativeCommand::Sequence(Box::new([
                NativeCommand::Dispatch {
                    program: program(input(&first, 0, 4)),
                    groups: [1, 1, 1],
                },
                NativeCommand::Dispatch {
                    program: program(retired_mapping),
                    groups: [1, 1, 1],
                },
            ]))],
        );
        assert!(work.retain_suspended_inputs().is_err());
        drop(first);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        drop((unused, work));
    }
}
