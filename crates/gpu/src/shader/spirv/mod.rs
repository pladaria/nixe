//! Device-independent native shader emission. No guest ISA or Vulkan ownership.
//!
//! Registers become scalar SSA values, never a GPU register-file interpreter.
//! Predicated side effects use structured selections and definitions merge with
//! OpPhi. Unsupported guest branches stop until CFG structurization is available.
//! https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

use std::collections::{BTreeMap, HashMap};

use rspirv::{binary::Assemble, dr::Builder, spirv as spv};

use super::*;
use crate::{
    TessellationDomain, TessellationMode, TessellationOutput, TessellationSpacing,
    TessellationWinding,
};

mod default_control;
mod pipeline;
pub use pipeline::{
    SpirvPipelineBinding, SpirvTessellationOptions, SpirvTessellationShaders,
    lower_raster_shaders_to_spirv, lower_tessellation_shaders_to_spirv,
};
mod float;
mod half;
pub use default_control::{
    SpirvDefaultControlOptions, SpirvDefaultControlShader,
    lower_default_tessellation_control_to_spirv,
};
mod interface;
mod operations;
mod resources;
#[cfg(test)]
mod tests;

/// Enabled host numerical guarantees, not guesses based on GPU vendor. Native
/// device creation must query/enable these before supplying them to the emitter.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SpirvFloat32Capabilities {
    pub denorm_preserve: bool,
    pub rounding_mode_rte: bool,
    pub signed_zero_inf_nan_preserve: bool,
    /// VK_KHR_shader_fma with shaderFmaFloat32 enabled. GLSL.std.450 Fma alone
    /// does not guarantee a fused, correctly rounded result.
    pub fused_multiply_add: bool,
}

/// Optional wide arithmetic for exact repair of float32 underflow. No float64
/// subnormal support is required: all repair intermediates are normal or zero.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SpirvFloat64Capabilities {
    pub enabled: bool,
    pub rounding_mode_rte: bool,
    pub signed_zero_inf_nan_preserve: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpirvShaderOptions {
    /// Convert only the final pre-raster position, after guest output reads.
    pub depth_clip_negative_one_to_one: bool,
    /// Incoming patch cardinality: draw patch size for TCS, TCS output size for TES.
    /// Zero for stages without arrayed patch inputs.
    pub input_control_points: u32,
    /// Required only for evaluation shaders. Part of the native shader cache key.
    pub tessellation_mode: Option<TessellationMode>,
    pub float32: SpirvFloat32Capabilities,
    pub float64: SpirvFloat64Capabilities,
}

/// SPIR-V 1.3 module targeting Vulkan 1.1 plus declared extensions. Native
/// integration must still link interfaces and apply its final position convention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpirvShaderModule {
    words: Box<[u32]>,
    bindings: Box<[ShaderResourceAccess]>,
}

impl SpirvShaderModule {
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Set-zero bindings actually emitted after dead-code elimination, sorted
    /// by binding number. Do not build native layouts from unused IR declarations.
    pub fn bindings(&self) -> &[ShaderResourceAccess] {
        &self.bindings
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpirvShaderError {
    Stage(ShaderStage),
    Options(&'static str),
    Interface(ShaderIoLocation),
    StageInterface(ShaderStageInterfaceError),
    BindingConflict {
        binding: u8,
    },
    DefaultTessellationLevels {
        required: u8,
        defined: u8,
    },
    Instruction {
        source: ShaderSourceLocation,
        reason: &'static str,
    },
    Construction(String),
}

impl Display for SpirvShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SpirvShaderError {}
impl From<rspirv::dr::Error> for SpirvShaderError {
    fn from(error: rspirv::dr::Error) -> Self {
        Self::Construction(error.to_string())
    }
}
type Result<T> = std::result::Result<T, SpirvShaderError>;

/// Emits the supported native graphics subset. No device-side validation,
/// subprocess, or per-draw work is performed here; call on a shader-cache miss.
pub fn lower_shader_ir_to_spirv(
    shader: &VerifiedShaderIr,
    options: SpirvShaderOptions,
) -> Result<SpirvShaderModule> {
    let ir = shader.ir();
    if !matches!(
        ir.stage,
        ShaderStage::Vertex
            | ShaderStage::TessellationControl
            | ShaderStage::TessellationEvaluation
            | ShaderStage::Fragment
    ) {
        return Err(SpirvShaderError::Stage(ir.stage));
    }
    let patch_stage = matches!(
        ir.stage,
        ShaderStage::TessellationControl | ShaderStage::TessellationEvaluation
    );
    if patch_stage == (options.input_control_points == 0) {
        return Err(SpirvShaderError::Options(
            "patch stages require a nonzero input cardinality; other stages require zero",
        ));
    }
    if (ir.stage == ShaderStage::TessellationEvaluation) != options.tessellation_mode.is_some() {
        return Err(SpirvShaderError::Options(
            "only evaluation shaders require a tessellation mode",
        ));
    }
    // Resources are consumed by operations below, so unused declarations do not
    // create descriptors or force unnecessary backend features.
    let live = liveness::live_instructions(ir)
        .map_err(|(source, reason)| SpirvShaderError::Instruction { source, reason })?;
    let mut emitter = Emitter::new(options);
    emitter.emit_interfaces(ir)?;
    let bindings = emitter.emit_resources(ir, &live)?;
    let void = emitter.b.type_void();
    let function_type = emitter.b.type_function(void, []);
    let entry = emitter
        .b
        .begin_function(void, None, spv::FunctionControl::NONE, function_type)?;
    emitter.entry = entry;
    emitter.configure_stage(ir)?;
    emitter.block = emitter.b.begin_block(None)?;
    for (instruction, live) in ir.instructions().iter().zip(live) {
        if !live {
            continue;
        }
        emitter.source = instruction.source;
        // Liveness has already rejected conditional exits and excluded Never.
        if matches!(instruction.operation, ShaderOperation::Exit) {
            break;
        }
        if instruction.predicate == ShaderPredicate::Always {
            emitter.operation(&instruction.operation)?;
        } else {
            emitter.predicated(instruction)?;
        }
    }
    if options.depth_clip_negative_one_to_one {
        let z = emitter.load_interface(false, ShaderIoLocation::Position, 2, None)?;
        let w = emitter.load_interface(false, ShaderIoLocation::Position, 3, None)?;
        let z = emitter.b.bitcast(emitter.float, None, z)?;
        let w = emitter.b.bitcast(emitter.float, None, w)?;
        let half = emitter.constant(0.5_f32.to_bits());
        let half = emitter.b.bitcast(emitter.float, None, half)?;
        let z = emitter.b.f_mul(emitter.float, None, z, half)?;
        let w = emitter.b.f_mul(emitter.float, None, w, half)?;
        emitter.b.decorate(z, spv::Decoration::NoContraction, []);
        emitter.b.decorate(w, spv::Decoration::NoContraction, []);
        let z = emitter.b.f_add(emitter.float, None, z, w)?;
        emitter.b.decorate(z, spv::Decoration::NoContraction, []);
        let z = emitter.b.bitcast(emitter.uint, None, z)?;
        emitter.store_interface(ShaderIoLocation::Position, 2, None, z)?;
    }
    emitter.b.ret()?;
    emitter.b.end_function()?;
    Ok(SpirvShaderModule {
        words: emitter.b.module().assemble().into_boxed_slice(),
        bindings,
    })
}

struct Emitter {
    b: Builder,
    options: SpirvShaderOptions,
    uint: u32,
    int: u32,
    float: u32,
    boolean: u32,
    undefined_uint: u32,
    undefined_bool: u32,
    true_value: u32,
    false_value: u32,
    constants: HashMap<u32, u32>,
    entry: u32,
    block: u32,
    source: ShaderSourceLocation,
    registers: Vec<Option<u32>>,
    predicates: [Option<u32>; 8],
    interfaces: BTreeMap<(bool, ShaderIoLocation, u8), interface::Element>,
    constant_buffers: [Option<u32>; 256],
    constant_word_pointer: Option<u32>,
    variables: Vec<u32>,
    float_modes: bool,
    repair_float: Option<u32>,
    fma_extension: bool,
}

impl Emitter {
    fn new(options: SpirvShaderOptions) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spv::Capability::Shader);
        b.memory_model(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
        let uint = b.type_int(32, 0);
        let int = b.type_int(32, 1);
        let float = b.type_float(32, None);
        let boolean = b.type_bool();
        // Global undef IDs dominate both selection predecessors. Creating undef
        // in the merge would violate SSA dominance and the leading-phi rule.
        let undefined_uint = b.undef(uint, None);
        let undefined_bool = b.undef(boolean, None);
        let true_value = b.constant_true(boolean);
        let false_value = b.constant_false(boolean);
        Self {
            b,
            options,
            uint,
            int,
            float,
            boolean,
            undefined_uint,
            undefined_bool,
            true_value,
            false_value,
            constants: HashMap::new(),
            entry: 0,
            block: 0,
            source: ShaderSourceLocation::new(0),
            registers: Vec::new(),
            predicates: [None; 8],
            interfaces: BTreeMap::new(),
            constant_buffers: [None; 256],
            constant_word_pointer: None,
            variables: Vec::new(),
            float_modes: false,
            repair_float: None,
            fma_extension: false,
        }
    }

    fn unsupported(&self, reason: &'static str) -> SpirvShaderError {
        SpirvShaderError::Instruction {
            source: self.source,
            reason,
        }
    }

    fn configure_stage(&mut self, ir: &ShaderIr) -> Result<()> {
        use spv::ExecutionMode as M;
        let model = match ir.stage {
            ShaderStage::Vertex => spv::ExecutionModel::Vertex,
            ShaderStage::Fragment => {
                self.b.execution_mode(self.entry, M::OriginUpperLeft, []);
                if ir
                    .outputs
                    .iter()
                    .any(|e| e.location == ShaderIoLocation::FragmentDepth)
                {
                    self.b.execution_mode(self.entry, M::DepthReplacing, []);
                }
                spv::ExecutionModel::Fragment
            }
            ShaderStage::TessellationControl => {
                self.b.capability(spv::Capability::Tessellation);
                self.b.execution_mode(
                    self.entry,
                    M::OutputVertices,
                    [ir.tessellation_control_points.unwrap()],
                );
                spv::ExecutionModel::TessellationControl
            }
            ShaderStage::TessellationEvaluation => {
                self.b.capability(spv::Capability::Tessellation);
                let mode = self.options.tessellation_mode.unwrap();
                let domain = match mode.domain {
                    TessellationDomain::Triangles => M::Triangles,
                    TessellationDomain::Quads => M::Quads,
                    TessellationDomain::Isolines => M::Isolines,
                };
                let spacing = match mode.spacing {
                    TessellationSpacing::Equal => M::SpacingEqual,
                    TessellationSpacing::FractionalEven => M::SpacingFractionalEven,
                    TessellationSpacing::FractionalOdd => M::SpacingFractionalOdd,
                };
                self.b.execution_mode(self.entry, domain, []);
                self.b.execution_mode(self.entry, spacing, []);
                match (mode.domain, mode.output) {
                    (_, TessellationOutput::Points) => {
                        self.b.execution_mode(self.entry, M::PointMode, [])
                    }
                    (TessellationDomain::Isolines, TessellationOutput::Lines) => {}
                    (
                        TessellationDomain::Triangles | TessellationDomain::Quads,
                        TessellationOutput::Triangles(winding),
                    ) => {
                        self.b.execution_mode(
                            self.entry,
                            match winding {
                                TessellationWinding::Clockwise => M::VertexOrderCw,
                                TessellationWinding::CounterClockwise => M::VertexOrderCcw,
                            },
                            [],
                        );
                    }
                    _ => {
                        return Err(SpirvShaderError::Options(
                            "domain and output topology are incompatible",
                        ));
                    }
                }
                spv::ExecutionModel::TessellationEvaluation
            }
            _ => unreachable!("stage validated before emission"),
        };
        self.b
            .entry_point(model, self.entry, "main", &self.variables);
        Ok(())
    }

    fn constant(&mut self, bits: u32) -> u32 {
        // rspirv interns types, but not constants. Reusing constant IDs also
        // allows equal-length array types to share their declaration.
        *self
            .constants
            .entry(bits)
            .or_insert_with(|| self.b.constant_bit32(self.uint, bits))
    }
    fn read(&self, register: ShaderRegister) -> Result<u32> {
        self.registers
            .get(usize::from(register.index()))
            .copied()
            .flatten()
            .ok_or_else(|| self.unsupported("missing SSA register definition"))
    }
    fn write(&mut self, register: ShaderRegister, id: u32) {
        let index = usize::from(register.index());
        if self.registers.len() <= index {
            self.registers.resize(index + 1, None);
        }
        self.registers[index] = Some(id);
    }
    fn predicate(&mut self, predicate: ShaderPredicate) -> Result<u32> {
        Ok(match predicate {
            ShaderPredicate::Always => self.true_value,
            ShaderPredicate::Never => self.false_value,
            ShaderPredicate::Register { register, inverted } => {
                let value = self.predicates[usize::from(register)]
                    .ok_or_else(|| self.unsupported("missing SSA predicate definition"))?;
                if inverted {
                    self.b.logical_not(self.boolean, None, value)?
                } else {
                    value
                }
            }
        })
    }

    fn predicated(&mut self, instruction: &ShaderInstruction) -> Result<()> {
        let condition = self.predicate(instruction.predicate)?;
        let header = self.block;
        let body = self.b.id();
        let merge = self.b.id();
        let mut definitions = Vec::new();
        instruction.operation.visit_destination_registers(|r| {
            if !definitions.iter().any(|(existing, _)| *existing == r) {
                definitions.push((
                    r,
                    self.registers
                        .get(usize::from(r.index()))
                        .copied()
                        .flatten(),
                ));
            }
        });
        let old_predicates = self.predicates;
        self.b.selection_merge(merge, spv::SelectionControl::NONE)?;
        self.b.branch_conditional(condition, body, merge, [])?;
        self.block = self.b.begin_block(Some(body))?;
        self.operation(&instruction.operation)?;
        let body_end = self.block;
        self.b.branch(merge)?;
        self.block = self.b.begin_block(Some(merge))?;
        for (register, previous) in definitions {
            let new = self.read(register)?;
            let old = previous.unwrap_or(self.undefined_uint);
            let merged = self
                .b
                .phi(self.uint, None, [(new, body_end), (old, header)])?;
            self.write(register, merged);
        }
        for (index, old) in old_predicates.into_iter().enumerate() {
            if self.predicates[index] != old {
                let previous = old.unwrap_or(self.undefined_bool);
                self.predicates[index] = Some(self.b.phi(
                    self.boolean,
                    None,
                    [
                        (self.predicates[index].unwrap(), body_end),
                        (previous, header),
                    ],
                )?);
            }
        }
        Ok(())
    }
}
