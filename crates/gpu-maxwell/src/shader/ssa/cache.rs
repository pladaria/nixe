//! Native shader identity and capture. No old renderer hashing/budget policy.
use super::super::{binary::MaxwellShaderMetadata, source};
use super::*;
use nixe_gpu::ssa::artifact::{ContentId, ShaderNamespace};

pub const MAXWELL_SHADER_NAMESPACE: ShaderNamespace = ShaderNamespace {
    guest: *b"nixe-switch-gpu1",
    // Versioned Maxwell decoder dialect; SPH SASS versions remain source data.
    isa: 1,
};

/// Capture on a shader-state or code dependency change, then move this immutable
/// value to a compilation worker. A prepared reference bypasses capture/hash.
#[derive(Clone, Debug)]
pub struct MaxwellShaderSnapshot {
    inputs: source::MaxwellShaderSnapshot,
    identity: ContentId,
}
impl MaxwellShaderSnapshot {
    pub fn capture(
        state: &crate::MaxwellThreeDState,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
    ) -> Result<Self, MaxwellShaderTranslationError> {
        let source = source::prepare_maxwell_shader_translation_source(state)?;
        Self::capture_source(&source, state.shader_bindings(), space, writes)
    }
    pub(crate) fn capture_source(
        source: &source::MaxwellShaderTranslationSourceKey,
        bindings: &crate::MaxwellThreeDShaderBindingState,
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
    ) -> Result<Self, MaxwellShaderTranslationError> {
        let inputs = source::capture_maxwell_shader_snapshot(
            &source.materialize(),
            bindings,
            space,
            writes,
        )?;
        let identity = snapshot_identity(&inputs);
        Ok(Self { inputs, identity })
    }
    pub fn capture_compute(
        space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        request: MaxwellSsaRequest,
        binding: impl Fn(u8) -> Result<(u64, u64), &'static str>,
    ) -> Result<Self, MaxwellShaderTranslationError> {
        if request.stage != MaxwellShaderStage::Compute {
            return Err(malformed(
                request.stage,
                0,
                0,
                "compute capture requires a QMD compute stage",
            ));
        }
        let mut binary = super::read_requested_binary(space, writes, &request)?;
        super::super::indirect::capture_compute(
            &mut binary,
            &super::super::binary::MaxwellShaderMemoryView::new(space, writes),
            binding,
        )?;
        let inputs = source::MaxwellShaderSnapshot {
            programs: Box::new([std::sync::Arc::new(
                source::MaxwellShaderProgramTranslationInput {
                    pipeline: 0,
                    register_count: request.register_count,
                    effective_group: None,
                    texture_constant_buffer_slot: None,
                    y_negate: false,
                    vertex_input_types: Box::new([]),
                    binary,
                },
            )]),
        };
        let identity = snapshot_identity(&inputs);
        Ok(Self { inputs, identity })
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.inputs
            .programs
            .iter()
            .map(|p| {
                p.binary.bundles.len()
                    * std::mem::size_of::<super::super::binary::MaxwellShaderInstructionBundle>()
                    + p.binary
                        .indirect_tables
                        .iter()
                        .map(|t| std::mem::size_of_val(t) + std::mem::size_of_val(&t.targets[..]))
                        .sum::<usize>()
            })
            .sum()
    }
    pub fn identity(&self) -> ContentId {
        self.identity
    }
    pub fn source_is_current(&self, space: &MaxwellGpuAddressSpace) -> bool {
        self.inputs.source_is_current(space)
    }
    pub fn staged_writes_are_irrelevant(&self, writes: &CanonicalWriteBatch) -> bool {
        self.inputs.staged_writes_are_irrelevant(writes)
    }
    pub fn indirect_bindings_current(
        &self,
        bindings: &crate::MaxwellThreeDShaderBindingState,
    ) -> bool {
        self.inputs.indirect_bindings_current(bindings)
    }
    pub fn compute_indirect_bindings_current(
        &self,
        binding: impl Fn(u8) -> Result<(u64, u64), &'static str>,
    ) -> bool {
        self.inputs.programs.iter().all(|program| {
            program
                .binary
                .indirect_tables
                .iter()
                .all(|table| table.compute_binding_current(&binding))
        })
    }
    pub fn compile(self) -> Result<MaxwellSsaPipeline, MaxwellShaderTranslationError> {
        super::program::compile_snapshot(self.inputs)
    }
}

pub(super) fn snapshot_identity(snapshot: &source::MaxwellShaderSnapshot) -> ContentId {
    // Stable explicit encodings, without virtual addresses, mappings/CPU epochs
    // or Rust's platform-dependent Hash representation. Dependencies prove
    // freshness; identical code at another mapping reuses the compiled artifact.
    let mut bytes = Vec::new();
    let word = |bytes: &mut Vec<u8>, value: u32| bytes.extend_from_slice(&value.to_le_bytes());
    word(&mut bytes, snapshot.programs.len() as u32);
    for p in &snapshot.programs {
        bytes.push(match p.binary.stage() {
            MaxwellShaderStage::VertexCullBeforeFetch => 0,
            MaxwellShaderStage::Vertex => 1,
            MaxwellShaderStage::TessellationInit => 2,
            MaxwellShaderStage::Tessellation => 3,
            MaxwellShaderStage::Geometry => 4,
            MaxwellShaderStage::Pixel => 5,
            MaxwellShaderStage::Compute => 6,
        });
        bytes.extend([
            p.register_count,
            p.effective_group.unwrap_or(255),
            p.texture_constant_buffer_slot.unwrap_or(255),
            p.y_negate as u8,
        ]);
        match p.binary.metadata {
            MaxwellShaderMetadata::Graphics(header) => {
                bytes.push(0);
                for value in header.words() {
                    word(&mut bytes, *value);
                }
            }
            MaxwellShaderMetadata::Compute {
                workgroup_size,
                register_count,
            } => {
                bytes.extend([1, register_count]);
                for value in workgroup_size {
                    word(&mut bytes, value);
                }
            }
        }
        word(&mut bytes, p.vertex_input_types.len() as u32);
        for &(location, ty) in &p.vertex_input_types {
            let index = match location {
                nixe_gpu::ShaderIoLocation::Generic(index) => index,
                _ => unreachable!("captured vertex format location is generic"),
            };
            bytes.extend([
                index,
                match ty {
                    nixe_gpu::ShaderScalarType::Float32 => 0,
                    nixe_gpu::ShaderScalarType::Unsigned32 => 1,
                    nixe_gpu::ShaderScalarType::Signed32 => 2,
                    nixe_gpu::ShaderScalarType::Unsigned64 => 3,
                    nixe_gpu::ShaderScalarType::Signed64 => 4,
                    nixe_gpu::ShaderScalarType::Float64 => 5,
                },
            ]);
        }
        word(&mut bytes, p.binary.bundles.len() as u32);
        for bundle in &p.binary.bundles {
            word(&mut bytes, bundle.offset);
            bytes.extend_from_slice(&bundle.control.to_le_bytes());
            for instruction in bundle.instructions {
                bytes.extend_from_slice(&instruction.to_le_bytes());
            }
        }
        word(&mut bytes, p.binary.indirect_tables.len() as u32);
        for table in &p.binary.indirect_tables {
            word(&mut bytes, table.offset);
            bytes.push(table.register);
            word(&mut bytes, table.load_offset);
            word(&mut bytes, table.read_end);
            word(&mut bytes, table.targets.len() as u32);
            for &(value, target) in &table.targets {
                word(&mut bytes, value);
                word(&mut bytes, target.byte_offset());
            }
        }
    }
    ContentId::of(b"nixe-maxwell-source-v1", &[&bytes])
}

/// Reusable semantic code. Freshness belongs to the current capture, not the
/// cached artifact: moving identical code must never retain old mapping proofs.
#[derive(Debug)]
pub struct MaxwellShaderCode {
    pub translation_ns: u64,
    pub programs: Box<[MaxwellSsaProgram]>,
    pub native: std::sync::Arc<nixe_gpu::native::GuestShaderCode>,
    pub(crate) uniform_control: Option<nixe_gpu::ssa::UniformControl>,
}
pub struct MaxwellShaderCompiler {
    pool: std::sync::Arc<nixe_gpu::ssa::compile::CompilePool>,
    cache: nixe_gpu::ssa::compile::CompilationCache<ContentId, MaxwellShaderCode>,
    bytes: usize,
}
impl MaxwellShaderCompiler {
    pub fn new(
        pool: std::sync::Arc<nixe_gpu::ssa::compile::CompilePool>,
        entries: usize,
        bytes: usize,
    ) -> Self {
        Self {
            pool,
            bytes,
            cache: nixe_gpu::ssa::compile::CompilationCache::new(entries, bytes),
        }
    }
    pub fn request(
        &self,
        snapshot: MaxwellShaderSnapshot,
    ) -> Result<
        nixe_gpu::ssa::compile::Pending<MaxwellShaderCode>,
        nixe_gpu::ssa::compile::CompileError,
    > {
        let weight = snapshot
            .inputs
            .programs
            .iter()
            .map(|p| p.binary.bundles.len() * 32)
            .sum();
        if weight > self.bytes {
            return Err(nixe_gpu::ssa::compile::CompileError::Failed(
                "shader capture exceeds the Maxwell compilation byte budget".into(),
            ));
        }
        self.cache.request_sized(
            &self.pool,
            snapshot.identity,
            weight,
            |code| {
                code.programs
                    .iter()
                    .map(|p| {
                        let f = &p.function;
                        f.values.len() * std::mem::size_of::<nixe_gpu::ssa::Value>()
                            + f.blocks
                                .iter()
                                .map(|b| {
                                    std::mem::size_of::<nixe_gpu::ssa::Block>()
                                        + b.instructions
                                            .iter()
                                            .map(|i| {
                                                std::mem::size_of_val(i)
                                                    + i.arguments.len() * 4
                                                    + i.results.len() * 4
                                            })
                                            .sum::<usize>()
                                        + b.phis
                                            .iter()
                                            .map(|phi| {
                                                std::mem::size_of_val(phi) + phi.incoming.len() * 8
                                            })
                                            .sum::<usize>()
                                })
                                .sum::<usize>()
                    })
                    .sum()
            },
            move |_| {
                let start = std::time::Instant::now();
                let identity = snapshot.identity();
                #[cfg(feature = "performance-counters")]
                if nixe_gpu::compilation_capture::enabled() {
                    nixe_gpu::compilation_capture::Key::new(&identity).detail(
                        "identity",
                        usize::MAX,
                        0,
                        &format!(
                            "source={} programs={:?}",
                            identity.hex(),
                            snapshot
                                .inputs
                                .programs
                                .iter()
                                .map(|p| (
                                    p.binary.stage(),
                                    p.binary.address,
                                    p.binary.bundles.len()
                                ))
                                .collect::<Vec<_>>()
                        ),
                    );
                }
                let mut code = snapshot
                    .compile()
                    .map_err(|e| {
                        nixe_gpu::ssa::compile::CompileError::Failed(e.to_string().into())
                    })?
                    .programs;
                // Source addresses are a property of the caller's fresh snapshot.
                // PC bases/stages stay in the reusable code for source diagnostics.
                for p in &mut code {
                    for source in &mut p.sources {
                        source.address = 0;
                    }
                }
                let translation_ns = start.elapsed().as_nanos() as u64;
                #[cfg(feature = "performance-counters")]
                if nixe_gpu::compilation_capture::enabled() {
                    nixe_gpu::compilation_capture::Key::new(&identity).detail(
                        "ssa",
                        usize::MAX,
                        translation_ns,
                        &format!(
                            "{:?}",
                            code.iter()
                                .map(|p| (
                                    p.stage,
                                    p.function.blocks.len(),
                                    p.function.values.len(),
                                    p.function
                                        .blocks
                                        .iter()
                                        .map(|b| b.instructions.len())
                                        .sum::<usize>()
                                ))
                                .collect::<Vec<_>>()
                        ),
                    );
                }
                nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::ShaderTranslationMisses, 1);
                nixe_gpu::metrics::record(
                    nixe_gpu::metrics::Counter::ShaderTranslationNanoseconds,
                    translation_ns,
                );
                let uniform_control = (code.len() == 1
                    && code[0].stage == MaxwellShaderStage::Compute)
                    .then(|| nixe_gpu::ssa::UniformControl::new(&code[0].function));
                Ok(MaxwellShaderCode {
                    uniform_control,
                    translation_ns,
                    native: std::sync::Arc::new(nixe_gpu::native::GuestShaderCode {
                        identity,
                        functions: code
                            .iter()
                            .map(|p| std::sync::Arc::clone(&p.function))
                            .collect(),
                    }),
                    programs: code,
                })
            },
        )
    }
    pub fn usage(&self) -> (usize, usize) {
        self.cache.usage()
    }
}

impl MaxwellShaderSnapshot {
    /// Resolve cached code PCs against this capture's current guest mappings.
    pub fn locate_source(
        &self,
        program: &MaxwellSsaProgram,
        location: ShaderSourceLocation,
    ) -> Option<(MaxwellShaderStage, u64, u32)> {
        let (stage, _, offset) = program.locate_source(location)?;
        let input = self
            .inputs
            .programs
            .iter()
            .find(|p| p.binary.stage() == stage)?;
        Some((stage, input.binary.address, offset))
    }
}
