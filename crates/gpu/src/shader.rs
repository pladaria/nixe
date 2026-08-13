//! Backend-neutral, guest-ISA-independent shader representation.
//!
//! Frontends translate guest machine code into these values. Backends consume
//! only verified programs and never need to understand Maxwell, SPH headers,
//! guest virtual addresses, or console command streams.

use std::{collections::BTreeSet, fmt::Display};

use crate::ShaderStage;

/// Stable location within the original guest shader byte stream.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShaderSourceLocation {
    byte_offset: u32,
}

impl ShaderSourceLocation {
    #[must_use]
    pub const fn new(byte_offset: u32) -> Self {
        Self { byte_offset }
    }

    #[must_use]
    pub const fn byte_offset(self) -> u32 {
        self.byte_offset
    }
}

/// Scalar numeric domain preserved from guest execution semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderScalarType {
    Unsigned32,
    Signed32,
    Float32,
    Unsigned64,
    Signed64,
    Float64,
}

/// One virtual scalar register in verified shader IR.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShaderRegister(u16);

impl ShaderRegister {
    #[must_use]
    pub const fn new(index: u16) -> Self {
        Self(index)
    }

    #[must_use]
    pub const fn index(self) -> u16 {
        self.0
    }
}

/// Predicate applied to one instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderPredicate {
    Always,
    Never,
    Register { register: u8, inverted: bool },
}

/// Semantic stage-interface location independent from a host shader language.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ShaderIoLocation {
    Position,
    PointSize,
    VertexId,
    InstanceId,
    Generic(u8),
    Color(u8),
    FragmentDepth,
    SampleMask,
}

/// Guest interpolation behavior for one fragment input.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderInterpolation {
    Constant,
    Perspective,
    ScreenLinear,
}

/// IEEE-754 behavior which may affect an operation's observable result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderFloatControl {
    rounding: ShaderRoundingMode,
    nan_mode: ShaderNanMode,
    flush_denormals_to_zero: bool,
    denormals_are_zero: bool,
    saturate: bool,
}

impl ShaderFloatControl {
    pub const PRECISE: Self = Self {
        rounding: ShaderRoundingMode::NearestEven,
        nan_mode: ShaderNanMode::Propagate,
        flush_denormals_to_zero: false,
        denormals_are_zero: false,
        saturate: false,
    };

    #[must_use]
    pub const fn new(
        rounding: ShaderRoundingMode,
        nan_mode: ShaderNanMode,
        flush_denormals_to_zero: bool,
        denormals_are_zero: bool,
        saturate: bool,
    ) -> Self {
        Self {
            rounding,
            nan_mode,
            flush_denormals_to_zero,
            denormals_are_zero,
            saturate,
        }
    }

    #[must_use]
    pub const fn rounding(self) -> ShaderRoundingMode {
        self.rounding
    }

    #[must_use]
    pub const fn nan_mode(self) -> ShaderNanMode {
        self.nan_mode
    }

    #[must_use]
    pub const fn flush_denormals_to_zero(self) -> bool {
        self.flush_denormals_to_zero
    }

    #[must_use]
    pub const fn denormals_are_zero(self) -> bool {
        self.denormals_are_zero
    }

    #[must_use]
    pub const fn saturate(self) -> bool {
        self.saturate
    }
}

/// Guest floating-point rounding direction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderRoundingMode {
    NearestEven,
    TowardNegative,
    TowardPositive,
    TowardZero,
}

/// Observable treatment of NaN results.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderNanMode {
    /// Preserve a NaN result; payload bits are not part of the contract.
    Propagate,
    /// Replace any NaN result with the canonical quiet NaN bit pattern.
    Canonicalize,
}

/// Accuracy contract carried by reciprocal operations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderReciprocalAccuracy {
    /// IEEE operation rounded according to the accompanying float control.
    Exact,
    /// Guest hardware permits an implementation-defined approximation.
    Approximate,
}

/// Descriptor-like resource class visible to one shader.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ShaderResourceKind {
    ConstantBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    Sampler,
}

/// Explicit guest resource use retained for validation and backend binding.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShaderResourceAccess {
    binding: u8,
    kind: ShaderResourceKind,
    readable: bool,
    writable: bool,
}

impl ShaderResourceAccess {
    pub fn new(
        binding: u8,
        kind: ShaderResourceKind,
        readable: bool,
        writable: bool,
    ) -> Result<Self, ShaderIrConstructionError> {
        if !readable && !writable {
            return Err(ShaderIrConstructionError::EmptyResourceAccess { binding });
        }
        Ok(Self {
            binding,
            kind,
            readable,
            writable,
        })
    }

    #[must_use]
    pub const fn binding(self) -> u8 {
        self.binding
    }

    #[must_use]
    pub const fn kind(self) -> ShaderResourceKind {
        self.kind
    }

    #[must_use]
    pub const fn readable(self) -> bool {
        self.readable
    }

    #[must_use]
    pub const fn writable(self) -> bool {
        self.writable
    }
}

/// One declared stage input or output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderInterfaceElement {
    location: ShaderIoLocation,
    component: u8,
    scalar_type: ShaderScalarType,
    interpolation: Option<ShaderInterpolation>,
}

impl ShaderInterfaceElement {
    pub fn new(
        location: ShaderIoLocation,
        component: u8,
        scalar_type: ShaderScalarType,
        interpolation: Option<ShaderInterpolation>,
    ) -> Result<Self, ShaderIrConstructionError> {
        if component > 3 {
            return Err(ShaderIrConstructionError::InvalidInterfaceComponent {
                location,
                component,
            });
        }
        Ok(Self {
            location,
            component,
            scalar_type,
            interpolation,
        })
    }

    #[must_use]
    pub const fn location(self) -> ShaderIoLocation {
        self.location
    }

    #[must_use]
    pub const fn component(self) -> u8 {
        self.component
    }

    #[must_use]
    pub const fn scalar_type(self) -> ShaderScalarType {
        self.scalar_type
    }

    #[must_use]
    pub const fn interpolation(self) -> Option<ShaderInterpolation> {
        self.interpolation
    }
}

/// Minimal operation vocabulary shared by frontend translation and backends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderOperation {
    Undefined32 {
        destination: ShaderRegister,
    },
    MoveImmediate32 {
        destination: ShaderRegister,
        bits: u32,
        scalar_type: ShaderScalarType,
    },
    LoadInput {
        destinations: Box<[ShaderRegister]>,
        location: ShaderIoLocation,
        first_component: u8,
        scalar_type: ShaderScalarType,
    },
    StoreOutput {
        sources: Box<[ShaderRegister]>,
        location: ShaderIoLocation,
        first_component: u8,
        scalar_type: ShaderScalarType,
    },
    Multiply32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        scalar_type: ShaderScalarType,
        float_control: ShaderFloatControl,
    },
    Reciprocal32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        accuracy: ShaderReciprocalAccuracy,
        float_control: ShaderFloatControl,
    },
    InterpolateInput {
        destination: ShaderRegister,
        location: ShaderIoLocation,
        component: u8,
        interpolation: ShaderInterpolation,
        perspective_reciprocal: Option<ShaderRegister>,
    },
    LoadConstantBuffer32 {
        destination: ShaderRegister,
        binding: u8,
        byte_offset: u32,
        scalar_type: ShaderScalarType,
    },
    Branch {
        target: ShaderSourceLocation,
    },
    Exit,
}

/// One operation with stable guest diagnostic provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderInstruction {
    source: ShaderSourceLocation,
    predicate: ShaderPredicate,
    operation: ShaderOperation,
}

impl ShaderInstruction {
    #[must_use]
    pub const fn new(
        source: ShaderSourceLocation,
        predicate: ShaderPredicate,
        operation: ShaderOperation,
    ) -> Self {
        Self {
            source,
            predicate,
            operation,
        }
    }

    #[must_use]
    pub const fn source(&self) -> ShaderSourceLocation {
        self.source
    }

    #[must_use]
    pub const fn predicate(&self) -> ShaderPredicate {
        self.predicate
    }

    #[must_use]
    pub const fn operation(&self) -> &ShaderOperation {
        &self.operation
    }
}

/// Unverified translation product. Construction alone grants no backend use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderIr {
    stage: ShaderStage,
    inputs: Box<[ShaderInterfaceElement]>,
    outputs: Box<[ShaderInterfaceElement]>,
    resources: Box<[ShaderResourceAccess]>,
    instructions: Box<[ShaderInstruction]>,
}

impl ShaderIr {
    #[must_use]
    pub fn new(
        stage: ShaderStage,
        inputs: Vec<ShaderInterfaceElement>,
        outputs: Vec<ShaderInterfaceElement>,
        resources: Vec<ShaderResourceAccess>,
        instructions: Vec<ShaderInstruction>,
    ) -> Self {
        Self {
            stage,
            inputs: inputs.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            resources: resources.into_boxed_slice(),
            instructions: instructions.into_boxed_slice(),
        }
    }

    #[must_use]
    pub const fn stage(&self) -> ShaderStage {
        self.stage
    }

    #[must_use]
    pub fn inputs(&self) -> &[ShaderInterfaceElement] {
        &self.inputs
    }

    #[must_use]
    pub fn outputs(&self) -> &[ShaderInterfaceElement] {
        &self.outputs
    }

    #[must_use]
    pub fn resources(&self) -> &[ShaderResourceAccess] {
        &self.resources
    }

    #[must_use]
    pub fn instructions(&self) -> &[ShaderInstruction] {
        &self.instructions
    }
}

/// Invalid IR construction input detected before verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderIrConstructionError {
    InvalidInterfaceComponent {
        location: ShaderIoLocation,
        component: u8,
    },
    EmptyResourceAccess {
        binding: u8,
    },
}

/// Shader IR whose control flow, data flow, interfaces, and resources passed
/// host-independent validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedShaderIr(ShaderIr);

impl VerifiedShaderIr {
    pub fn verify(ir: ShaderIr) -> Result<Self, ShaderVerificationError> {
        verify_interface_set(ir.stage, &ir.inputs, true)?;
        verify_interface_set(ir.stage, &ir.outputs, false)?;
        verify_resource_set(&ir.resources)?;
        verify_instructions(&ir)?;
        Ok(Self(ir))
    }

    #[must_use]
    pub const fn ir(&self) -> &ShaderIr {
        &self.0
    }
}

/// Inputs supplied to the small backend-neutral reference evaluator.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderEvaluationInputs {
    interface: std::collections::BTreeMap<(ShaderIoLocation, u8), u32>,
    constant_buffers: std::collections::BTreeMap<(u8, u32), u32>,
}

impl ShaderEvaluationInputs {
    #[must_use]
    pub fn with_interface_bits(
        mut self,
        location: ShaderIoLocation,
        component: u8,
        bits: u32,
    ) -> Self {
        self.interface.insert((location, component), bits);
        self
    }

    #[must_use]
    pub fn with_constant_buffer_bits(mut self, binding: u8, byte_offset: u32, bits: u32) -> Self {
        self.constant_buffers.insert((binding, byte_offset), bits);
        self
    }
}

/// Observable result of one reference-evaluator invocation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderEvaluationResult {
    outputs: std::collections::BTreeMap<(ShaderIoLocation, u8), u32>,
}

impl ShaderEvaluationResult {
    #[must_use]
    pub fn output_bits(&self, location: ShaderIoLocation, component: u8) -> Option<u32> {
        self.outputs.get(&(location, component)).copied()
    }
}

/// Reference-evaluator failure. Verified IR prevents data-flow failures; input
/// values and evaluator limits remain invocation-specific.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderEvaluationError {
    MissingInterfaceInput {
        location: ShaderIoLocation,
        component: u8,
    },
    MissingConstantBufferWord {
        binding: u8,
        byte_offset: u32,
    },
    UndefinedRegister(ShaderRegister),
    UnsupportedPredicateRegister(u8),
    UnsupportedRoundingMode(ShaderRoundingMode),
    ApproximateOperation(ShaderSourceLocation),
    StepLimitExceeded,
    MissingExit,
}

impl Display for ShaderEvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ShaderEvaluationError {}

/// Evaluates verified IR with a bounded instruction count.
pub fn evaluate_shader_ir(
    shader: &VerifiedShaderIr,
    inputs: &ShaderEvaluationInputs,
    step_limit: usize,
) -> Result<ShaderEvaluationResult, ShaderEvaluationError> {
    let ir = shader.ir();
    let targets = ir
        .instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.source, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut registers = vec![None; 256];
    let mut result = ShaderEvaluationResult::default();
    let mut pc = 0_usize;
    let mut steps = 0_usize;
    while let Some(instruction) = ir.instructions.get(pc) {
        if steps >= step_limit {
            return Err(ShaderEvaluationError::StepLimitExceeded);
        }
        steps += 1;
        pc += 1;
        match instruction.predicate {
            ShaderPredicate::Never => continue,
            ShaderPredicate::Register { register, .. } => {
                return Err(ShaderEvaluationError::UnsupportedPredicateRegister(
                    register,
                ));
            }
            ShaderPredicate::Always => {}
        }
        match instruction.operation() {
            ShaderOperation::Undefined32 { destination } => {
                registers[destination.index() as usize] = Some(0);
            }
            ShaderOperation::MoveImmediate32 {
                destination, bits, ..
            } => registers[destination.index() as usize] = Some(*bits),
            ShaderOperation::LoadInput {
                destinations,
                location,
                first_component,
                ..
            } => {
                for (index, destination) in destinations.iter().enumerate() {
                    let component = first_component.saturating_add(index as u8);
                    registers[destination.index() as usize] = Some(
                        inputs
                            .interface
                            .get(&(*location, component))
                            .copied()
                            .ok_or(ShaderEvaluationError::MissingInterfaceInput {
                                location: *location,
                                component,
                            })?,
                    );
                }
            }
            ShaderOperation::StoreOutput {
                sources,
                location,
                first_component,
                ..
            } => {
                for (index, source) in sources.iter().enumerate() {
                    let bits = register_bits(&registers, *source)?;
                    result.outputs.insert(
                        (*location, first_component.saturating_add(index as u8)),
                        bits,
                    );
                }
            }
            ShaderOperation::Multiply32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => {
                let left = register_bits(&registers, *left)?;
                let right = register_bits(&registers, *right)?;
                let value = match scalar_type {
                    ShaderScalarType::Float32 => {
                        evaluate_float_binary(left, right, *float_control, |a, b| a * b)?
                    }
                    ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 => {
                        left.wrapping_mul(right)
                    }
                    _ => unreachable!("Multiply32 only admits 32-bit scalar types"),
                };
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::Reciprocal32 {
                destination,
                source,
                accuracy,
                float_control,
            } => {
                if *accuracy == ShaderReciprocalAccuracy::Approximate {
                    return Err(ShaderEvaluationError::ApproximateOperation(
                        instruction.source,
                    ));
                }
                let source = register_bits(&registers, *source)?;
                let value = evaluate_float_unary(source, *float_control, |value| value.recip())?;
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::InterpolateInput {
                destination,
                location,
                component,
                perspective_reciprocal,
                ..
            } => {
                let mut bits = inputs
                    .interface
                    .get(&(*location, *component))
                    .copied()
                    .ok_or(ShaderEvaluationError::MissingInterfaceInput {
                        location: *location,
                        component: *component,
                    })?;
                if let Some(reciprocal) = perspective_reciprocal {
                    bits = evaluate_float_binary(
                        bits,
                        register_bits(&registers, *reciprocal)?,
                        ShaderFloatControl::PRECISE,
                        |a, b| a * b,
                    )?;
                }
                registers[destination.index() as usize] = Some(bits);
            }
            ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset,
                ..
            } => {
                registers[destination.index() as usize] = Some(
                    inputs
                        .constant_buffers
                        .get(&(*binding, *byte_offset))
                        .copied()
                        .ok_or(ShaderEvaluationError::MissingConstantBufferWord {
                            binding: *binding,
                            byte_offset: *byte_offset,
                        })?,
                );
            }
            ShaderOperation::Branch { target } => {
                pc = targets[target];
            }
            ShaderOperation::Exit => return Ok(result),
        }
    }
    Err(ShaderEvaluationError::MissingExit)
}

fn register_bits(
    registers: &[Option<u32>],
    register: ShaderRegister,
) -> Result<u32, ShaderEvaluationError> {
    registers[register.index() as usize].ok_or(ShaderEvaluationError::UndefinedRegister(register))
}

fn evaluate_float_binary(
    mut left: u32,
    mut right: u32,
    control: ShaderFloatControl,
    operation: impl FnOnce(f32, f32) -> f32,
) -> Result<u32, ShaderEvaluationError> {
    if control.rounding != ShaderRoundingMode::NearestEven {
        return Err(ShaderEvaluationError::UnsupportedRoundingMode(
            control.rounding,
        ));
    }
    if control.denormals_are_zero {
        left = flush_denormal_bits(left);
        right = flush_denormal_bits(right);
    }
    let result = operation(f32::from_bits(left), f32::from_bits(right));
    finish_float(result.to_bits(), control)
}

fn evaluate_float_unary(
    mut source: u32,
    control: ShaderFloatControl,
    operation: impl FnOnce(f32) -> f32,
) -> Result<u32, ShaderEvaluationError> {
    if control.rounding != ShaderRoundingMode::NearestEven {
        return Err(ShaderEvaluationError::UnsupportedRoundingMode(
            control.rounding,
        ));
    }
    if control.denormals_are_zero {
        source = flush_denormal_bits(source);
    }
    let result = operation(f32::from_bits(source));
    finish_float(result.to_bits(), control)
}

fn finish_float(mut bits: u32, control: ShaderFloatControl) -> Result<u32, ShaderEvaluationError> {
    if control.nan_mode == ShaderNanMode::Canonicalize && f32::from_bits(bits).is_nan() {
        bits = f32::NAN.to_bits();
    }
    if control.flush_denormals_to_zero {
        bits = flush_denormal_bits(bits);
    }
    if control.saturate && !f32::from_bits(bits).is_nan() {
        bits = f32::from_bits(bits).clamp(0.0, 1.0).to_bits();
    }
    Ok(bits)
}

const fn flush_denormal_bits(bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0 {
        bits & 0x8000_0000
    } else {
        bits
    }
}

/// Portable backend shader representation emitted after neutral verification.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderBackendLanguage {
    Wgsl,
}

/// Stable relation between one backend source line and guest shader offset.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderBackendSourceMapEntry {
    backend_line: u32,
    source: ShaderSourceLocation,
}

impl ShaderBackendSourceMapEntry {
    #[must_use]
    pub const fn backend_line(self) -> u32 {
        self.backend_line
    }

    #[must_use]
    pub const fn source(self) -> ShaderSourceLocation {
        self.source
    }
}

/// Backend-consumable module with no guest-ISA objects or addresses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderBackendModule {
    stage: ShaderStage,
    language: ShaderBackendLanguage,
    source: Box<str>,
    source_map: Box<[ShaderBackendSourceMapEntry]>,
}

impl ShaderBackendModule {
    #[must_use]
    pub const fn stage(&self) -> ShaderStage {
        self.stage
    }

    #[must_use]
    pub const fn language(&self) -> ShaderBackendLanguage {
        self.language
    }

    #[must_use]
    pub const fn source(&self) -> &str {
        &self.source
    }

    #[must_use]
    pub fn source_map(&self) -> &[ShaderBackendSourceMapEntry] {
        &self.source_map
    }
}

/// Neutral-to-backend lowering limitation, distinct from malformed guest code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderBackendLoweringError {
    UnsupportedStage(ShaderStage),
    UnsupportedInterface(ShaderIoLocation),
    InconsistentInterfaceType(ShaderIoLocation),
    PredicateRegister(u8),
    ControlFlow(ShaderSourceLocation),
    ResourceAccess(ShaderSourceLocation),
    NumericControl(ShaderSourceLocation),
}

impl Display for ShaderBackendLoweringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ShaderBackendLoweringError {}

#[derive(Clone, Copy)]
struct InterfaceGroup {
    components: u8,
    scalar_type: ShaderScalarType,
    interpolation: Option<ShaderInterpolation>,
}

/// Lowers verified neutral IR to a standalone WGSL module.
pub fn lower_shader_ir_to_wgsl(
    shader: &VerifiedShaderIr,
) -> Result<ShaderBackendModule, ShaderBackendLoweringError> {
    let ir = shader.ir();
    if !matches!(ir.stage, ShaderStage::Vertex | ShaderStage::Fragment) {
        return Err(ShaderBackendLoweringError::UnsupportedStage(ir.stage));
    }
    let input_groups = interface_groups(&ir.inputs)?;
    let output_groups = interface_groups(&ir.outputs)?;
    let mut source = String::new();
    emit_interface_struct(&mut source, "ShaderInput", ir.stage, true, &input_groups)?;
    emit_interface_struct(&mut source, "ShaderOutput", ir.stage, false, &output_groups)?;
    source.push_str(match ir.stage {
        ShaderStage::Vertex => "@vertex\n",
        ShaderStage::Fragment => "@fragment\n",
        _ => unreachable!(),
    });
    source.push_str("fn main(input: ShaderInput) -> ShaderOutput {\n");
    source.push_str("  var registers: array<u32, 256>;\n");
    source.push_str("  var output: ShaderOutput;\n");
    let mut source_map = Vec::new();
    for instruction in &ir.instructions {
        let line = source.lines().count() as u32 + 1;
        source_map.push(ShaderBackendSourceMapEntry {
            backend_line: line,
            source: instruction.source,
        });
        source.push_str(&format!(
            "  // Maxwell code byte offset 0x{:x}\n",
            instruction.source.byte_offset()
        ));
        match instruction.predicate {
            ShaderPredicate::Never => continue,
            ShaderPredicate::Register { register, .. } => {
                return Err(ShaderBackendLoweringError::PredicateRegister(register));
            }
            ShaderPredicate::Always => {}
        }
        emit_wgsl_operation(&mut source, instruction)?;
    }
    source.push_str("}\n");
    Ok(ShaderBackendModule {
        stage: ir.stage,
        language: ShaderBackendLanguage::Wgsl,
        source: source.into_boxed_str(),
        source_map: source_map.into_boxed_slice(),
    })
}

fn interface_groups(
    elements: &[ShaderInterfaceElement],
) -> Result<std::collections::BTreeMap<ShaderIoLocation, InterfaceGroup>, ShaderBackendLoweringError>
{
    let mut groups = std::collections::BTreeMap::new();
    for element in elements {
        let group = groups.entry(element.location).or_insert(InterfaceGroup {
            components: 0,
            scalar_type: element.scalar_type,
            interpolation: element.interpolation,
        });
        if group.scalar_type != element.scalar_type || group.interpolation != element.interpolation
        {
            return Err(ShaderBackendLoweringError::InconsistentInterfaceType(
                element.location,
            ));
        }
        group.components = group.components.max(element.component + 1);
    }
    Ok(groups)
}

fn emit_interface_struct(
    source: &mut String,
    name: &str,
    stage: ShaderStage,
    input: bool,
    groups: &std::collections::BTreeMap<ShaderIoLocation, InterfaceGroup>,
) -> Result<(), ShaderBackendLoweringError> {
    source.push_str(&format!("struct {name} {{\n"));
    for (location, group) in groups {
        let attribute = wgsl_interface_attribute(stage, input, *location, group.interpolation)?;
        source.push_str(&format!(
            "  {attribute} {}: {},\n",
            wgsl_field_name(*location),
            wgsl_type(*location, group.scalar_type, group.components)
        ));
    }
    source.push_str("};\n\n");
    Ok(())
}

fn wgsl_interface_attribute(
    stage: ShaderStage,
    input: bool,
    location: ShaderIoLocation,
    interpolation: Option<ShaderInterpolation>,
) -> Result<String, ShaderBackendLoweringError> {
    let base = match location {
        ShaderIoLocation::Position => "@builtin(position)".to_owned(),
        ShaderIoLocation::VertexId if stage == ShaderStage::Vertex && input => {
            "@builtin(vertex_index)".to_owned()
        }
        ShaderIoLocation::InstanceId if stage == ShaderStage::Vertex && input => {
            "@builtin(instance_index)".to_owned()
        }
        ShaderIoLocation::Generic(index) | ShaderIoLocation::Color(index) => {
            format!("@location({index})")
        }
        ShaderIoLocation::FragmentDepth if stage == ShaderStage::Fragment && !input => {
            "@builtin(frag_depth)".to_owned()
        }
        ShaderIoLocation::SampleMask => "@builtin(sample_mask)".to_owned(),
        other => return Err(ShaderBackendLoweringError::UnsupportedInterface(other)),
    };
    let interpolation = match interpolation {
        Some(ShaderInterpolation::Constant) => " @interpolate(flat)",
        Some(ShaderInterpolation::Perspective) => " @interpolate(perspective, center)",
        Some(ShaderInterpolation::ScreenLinear) => " @interpolate(linear, center)",
        None => "",
    };
    Ok(format!("{base}{interpolation}"))
}

fn wgsl_field_name(location: ShaderIoLocation) -> String {
    match location {
        ShaderIoLocation::Position => "position".to_owned(),
        ShaderIoLocation::PointSize => "point_size".to_owned(),
        ShaderIoLocation::VertexId => "vertex_id".to_owned(),
        ShaderIoLocation::InstanceId => "instance_id".to_owned(),
        ShaderIoLocation::Generic(index) => format!("generic_{index}"),
        ShaderIoLocation::Color(index) => format!("color_{index}"),
        ShaderIoLocation::FragmentDepth => "fragment_depth".to_owned(),
        ShaderIoLocation::SampleMask => "sample_mask".to_owned(),
    }
}

fn wgsl_type(location: ShaderIoLocation, scalar_type: ShaderScalarType, components: u8) -> String {
    let scalar = match scalar_type {
        ShaderScalarType::Unsigned32 => "u32",
        ShaderScalarType::Signed32 => "i32",
        ShaderScalarType::Float32 => "f32",
        ShaderScalarType::Unsigned64 => "u64",
        ShaderScalarType::Signed64 => "i64",
        ShaderScalarType::Float64 => "f64",
    };
    if matches!(
        location,
        ShaderIoLocation::Position | ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_)
    ) {
        format!("vec4<{scalar}>")
    } else if components == 1 {
        scalar.to_owned()
    } else {
        format!("vec{components}<{scalar}>")
    }
}

fn emit_wgsl_operation(
    source: &mut String,
    instruction: &ShaderInstruction,
) -> Result<(), ShaderBackendLoweringError> {
    match instruction.operation() {
        ShaderOperation::Undefined32 { destination } => source.push_str(&format!(
            "  registers[{}] = 0u; // deterministic choice for undefined guest bits\n",
            destination.index()
        )),
        ShaderOperation::MoveImmediate32 {
            destination, bits, ..
        } => source.push_str(&format!(
            "  registers[{}] = 0x{bits:08x}u;\n",
            destination.index()
        )),
        ShaderOperation::LoadInput {
            destinations,
            location,
            first_component,
            scalar_type,
        } => {
            for (index, destination) in destinations.iter().enumerate() {
                source.push_str(&format!(
                    "  registers[{}] = {};\n",
                    destination.index(),
                    wgsl_pack_expression(
                        *scalar_type,
                        &format!(
                            "input.{}{}",
                            wgsl_field_name(*location),
                            wgsl_component(first_component.saturating_add(index as u8))
                        )
                    )
                ));
            }
        }
        ShaderOperation::StoreOutput {
            sources,
            location,
            first_component,
            scalar_type,
        } => {
            for (index, register) in sources.iter().enumerate() {
                source.push_str(&format!(
                    "  output.{}{} = {};\n",
                    wgsl_field_name(*location),
                    wgsl_component(first_component.saturating_add(index as u8)),
                    wgsl_unpack_expression(*scalar_type, register.index())
                ));
            }
        }
        ShaderOperation::Multiply32 {
            destination,
            left,
            right,
            scalar_type,
            float_control,
        } => {
            require_precise_float(instruction.source, *float_control)?;
            let expression = match scalar_type {
                ShaderScalarType::Float32 => format!(
                    "bitcast<u32>(bitcast<f32>(registers[{}]) * bitcast<f32>(registers[{}]))",
                    left.index(),
                    right.index()
                ),
                ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 => {
                    format!("registers[{}] * registers[{}]", left.index(), right.index())
                }
                _ => {
                    return Err(ShaderBackendLoweringError::NumericControl(
                        instruction.source,
                    ));
                }
            };
            source.push_str(&format!(
                "  registers[{}] = {expression};\n",
                destination.index()
            ));
        }
        ShaderOperation::Reciprocal32 {
            destination,
            source: operand,
            float_control,
            ..
        } => {
            require_precise_float(instruction.source, *float_control)?;
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>(1.0 / bitcast<f32>(registers[{}]));\n",
                destination.index(),
                operand.index()
            ));
        }
        ShaderOperation::InterpolateInput {
            destination,
            location,
            component,
            perspective_reciprocal,
            ..
        } => {
            let input = format!(
                "input.{}{}",
                wgsl_field_name(*location),
                wgsl_component(*component)
            );
            let expression = perspective_reciprocal.map_or(input.clone(), |reciprocal| {
                format!("{input} * bitcast<f32>(registers[{}])", reciprocal.index())
            });
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>({expression});\n",
                destination.index()
            ));
        }
        ShaderOperation::LoadConstantBuffer32 { .. } => {
            return Err(ShaderBackendLoweringError::ResourceAccess(
                instruction.source,
            ));
        }
        ShaderOperation::Branch { .. } => {
            return Err(ShaderBackendLoweringError::ControlFlow(instruction.source));
        }
        ShaderOperation::Exit => source.push_str("  return output;\n"),
    }
    Ok(())
}

fn require_precise_float(
    source: ShaderSourceLocation,
    control: ShaderFloatControl,
) -> Result<(), ShaderBackendLoweringError> {
    if control == ShaderFloatControl::PRECISE {
        Ok(())
    } else {
        Err(ShaderBackendLoweringError::NumericControl(source))
    }
}

fn wgsl_component(component: u8) -> &'static str {
    match component {
        0 => ".x",
        1 => ".y",
        2 => ".z",
        3 => ".w",
        _ => unreachable!("verified interface component"),
    }
}

fn wgsl_pack_expression(scalar_type: ShaderScalarType, expression: &str) -> String {
    match scalar_type {
        ShaderScalarType::Unsigned32 => expression.to_owned(),
        ShaderScalarType::Signed32 | ShaderScalarType::Float32 => {
            format!("bitcast<u32>({expression})")
        }
        _ => unreachable!("32-bit interface lowering"),
    }
}

fn wgsl_unpack_expression(scalar_type: ShaderScalarType, register: u16) -> String {
    match scalar_type {
        ShaderScalarType::Unsigned32 => format!("registers[{register}]"),
        ShaderScalarType::Signed32 => format!("bitcast<i32>(registers[{register}])"),
        ShaderScalarType::Float32 => format!("bitcast<f32>(registers[{register}])"),
        _ => unreachable!("32-bit interface lowering"),
    }
}

/// Typed reason why translated IR cannot be consumed by any backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderVerificationError {
    EmptyProgram,
    MissingExit,
    DuplicateInterfaceElement {
        input: bool,
        location: ShaderIoLocation,
        component: u8,
    },
    InvalidInterpolation {
        input: bool,
        location: ShaderIoLocation,
    },
    DuplicateResourceBinding {
        binding: u8,
    },
    UndefinedRegister {
        source: ShaderSourceLocation,
        register: ShaderRegister,
    },
    UnsupportedPredicate {
        source: ShaderSourceLocation,
        register: u8,
    },
    UndeclaredInterfaceAccess {
        source: ShaderSourceLocation,
        input: bool,
        location: ShaderIoLocation,
        component: u8,
    },
    UndeclaredResourceAccess {
        source: ShaderSourceLocation,
        binding: u8,
    },
    InvalidBranchTarget {
        source: ShaderSourceLocation,
        target: ShaderSourceLocation,
    },
}

impl Display for ShaderVerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyProgram => formatter.write_str("shader IR contains no instructions"),
            Self::MissingExit => formatter.write_str("shader IR has no reachable structured exit"),
            Self::DuplicateInterfaceElement {
                input,
                location,
                component,
            } => write!(
                formatter,
                "shader IR has duplicate {} interface element {location:?}.{component}",
                if *input { "input" } else { "output" }
            ),
            Self::InvalidInterpolation { input, location } => write!(
                formatter,
                "shader IR has invalid interpolation on {} {location:?}",
                if *input { "input" } else { "output" }
            ),
            Self::DuplicateResourceBinding { binding } => {
                write!(
                    formatter,
                    "shader IR has duplicate resource binding {binding}"
                )
            }
            Self::UndefinedRegister { source, register } => write!(
                formatter,
                "shader IR reads undefined r{} at byte offset 0x{:x}",
                register.index(),
                source.byte_offset()
            ),
            Self::UnsupportedPredicate { source, register } => write!(
                formatter,
                "shader IR uses unsupported predicate register p{register} at byte offset 0x{:x}",
                source.byte_offset()
            ),
            Self::UndeclaredInterfaceAccess {
                source,
                input,
                location,
                component,
            } => write!(
                formatter,
                "shader IR accesses undeclared {} {location:?}.{component} at byte offset 0x{:x}",
                if *input { "input" } else { "output" },
                source.byte_offset()
            ),
            Self::UndeclaredResourceAccess { source, binding } => write!(
                formatter,
                "shader IR accesses undeclared resource {binding} at byte offset 0x{:x}",
                source.byte_offset()
            ),
            Self::InvalidBranchTarget { source, target } => write!(
                formatter,
                "shader IR branch at byte offset 0x{:x} targets non-instruction offset 0x{:x}",
                source.byte_offset(),
                target.byte_offset()
            ),
        }
    }
}

impl std::error::Error for ShaderVerificationError {}

fn verify_interface_set(
    stage: ShaderStage,
    elements: &[ShaderInterfaceElement],
    input: bool,
) -> Result<(), ShaderVerificationError> {
    let mut seen = BTreeSet::new();
    for element in elements {
        if !seen.insert((element.location, element.component)) {
            return Err(ShaderVerificationError::DuplicateInterfaceElement {
                input,
                location: element.location,
                component: element.component,
            });
        }
        let interpolation_is_valid = if input && stage == ShaderStage::Fragment {
            match element.location {
                ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_) => {
                    element.interpolation.is_some()
                }
                _ => element.interpolation.is_none(),
            }
        } else {
            element.interpolation.is_none()
        };
        if !interpolation_is_valid {
            return Err(ShaderVerificationError::InvalidInterpolation {
                input,
                location: element.location,
            });
        }
    }
    Ok(())
}

fn verify_resource_set(resources: &[ShaderResourceAccess]) -> Result<(), ShaderVerificationError> {
    let mut bindings = BTreeSet::new();
    for resource in resources {
        if !bindings.insert(resource.binding) {
            return Err(ShaderVerificationError::DuplicateResourceBinding {
                binding: resource.binding,
            });
        }
    }
    Ok(())
}

fn verify_instructions(ir: &ShaderIr) -> Result<(), ShaderVerificationError> {
    if ir.instructions.is_empty() {
        return Err(ShaderVerificationError::EmptyProgram);
    }
    let locations = ir
        .instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.source, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut incoming = vec![None::<BTreeSet<ShaderRegister>>; ir.instructions.len()];
    incoming[0] = Some(BTreeSet::new());
    let mut work = std::collections::VecDeque::from([0_usize]);
    let mut has_reachable_exit = false;
    while let Some(index) = work.pop_front() {
        let instruction = &ir.instructions[index];
        let mut definitions = incoming[index]
            .clone()
            .expect("queued instruction has incoming definitions");
        match instruction.predicate {
            ShaderPredicate::Never => {
                enqueue_shader_successor(
                    index.checked_add(1),
                    &definitions,
                    &mut incoming,
                    &mut work,
                )?;
                continue;
            }
            ShaderPredicate::Register { register, .. } => {
                return Err(ShaderVerificationError::UnsupportedPredicate {
                    source: instruction.source,
                    register,
                });
            }
            ShaderPredicate::Always => {}
        }
        match &instruction.operation {
            ShaderOperation::Undefined32 { destination }
            | ShaderOperation::MoveImmediate32 { destination, .. }
            | ShaderOperation::LoadConstantBuffer32 { destination, .. }
            | ShaderOperation::Reciprocal32 { destination, .. }
            | ShaderOperation::Multiply32 { destination, .. }
            | ShaderOperation::InterpolateInput { destination, .. } => {
                for source in operation_sources(&instruction.operation) {
                    require_definition(instruction.source, source, &definitions)?;
                }
                definitions.insert(*destination);
            }
            ShaderOperation::LoadInput {
                destinations,
                location,
                first_component,
                ..
            } => {
                verify_interface_range(
                    ir,
                    instruction.source,
                    true,
                    *location,
                    *first_component,
                    destinations.len(),
                )?;
                definitions.extend(destinations.iter().copied());
            }
            ShaderOperation::StoreOutput {
                sources,
                location,
                first_component,
                ..
            } => {
                verify_interface_range(
                    ir,
                    instruction.source,
                    false,
                    *location,
                    *first_component,
                    sources.len(),
                )?;
                for source in sources.iter().copied() {
                    require_definition(instruction.source, source, &definitions)?;
                }
            }
            ShaderOperation::Branch { target } => {
                let target_index = locations.get(target).copied().ok_or(
                    ShaderVerificationError::InvalidBranchTarget {
                        source: instruction.source,
                        target: *target,
                    },
                )?;
                enqueue_shader_successor(
                    Some(target_index),
                    &definitions,
                    &mut incoming,
                    &mut work,
                )?;
                continue;
            }
            ShaderOperation::Exit => {
                has_reachable_exit = true;
                continue;
            }
        }
        if let ShaderOperation::LoadConstantBuffer32 { binding, .. } = instruction.operation
            && !ir.resources.iter().any(|resource| {
                resource.binding == binding
                    && resource.kind == ShaderResourceKind::ConstantBuffer
                    && resource.readable
            })
        {
            return Err(ShaderVerificationError::UndeclaredResourceAccess {
                source: instruction.source,
                binding,
            });
        }
        enqueue_shader_successor(index.checked_add(1), &definitions, &mut incoming, &mut work)?;
    }
    if !has_reachable_exit {
        return Err(ShaderVerificationError::MissingExit);
    }
    Ok(())
}

fn enqueue_shader_successor(
    successor: Option<usize>,
    definitions: &BTreeSet<ShaderRegister>,
    incoming: &mut [Option<BTreeSet<ShaderRegister>>],
    work: &mut std::collections::VecDeque<usize>,
) -> Result<(), ShaderVerificationError> {
    let Some(successor) = successor.filter(|index| *index < incoming.len()) else {
        return Err(ShaderVerificationError::MissingExit);
    };
    match &mut incoming[successor] {
        None => {
            incoming[successor] = Some(definitions.clone());
            work.push_back(successor);
        }
        Some(existing) => {
            let merged = existing
                .intersection(definitions)
                .copied()
                .collect::<BTreeSet<_>>();
            if merged != *existing {
                *existing = merged;
                work.push_back(successor);
            }
        }
    }
    Ok(())
}

fn operation_sources(operation: &ShaderOperation) -> Vec<ShaderRegister> {
    match operation {
        ShaderOperation::Multiply32 { left, right, .. } => vec![*left, *right],
        ShaderOperation::Reciprocal32 { source, .. } => vec![*source],
        ShaderOperation::InterpolateInput {
            perspective_reciprocal,
            ..
        } => perspective_reciprocal.iter().copied().collect(),
        _ => Vec::new(),
    }
}

fn require_definition(
    source: ShaderSourceLocation,
    register: ShaderRegister,
    definitions: &BTreeSet<ShaderRegister>,
) -> Result<(), ShaderVerificationError> {
    if definitions.contains(&register) {
        Ok(())
    } else {
        Err(ShaderVerificationError::UndefinedRegister { source, register })
    }
}

fn verify_interface_range(
    ir: &ShaderIr,
    source: ShaderSourceLocation,
    input: bool,
    location: ShaderIoLocation,
    first_component: u8,
    count: usize,
) -> Result<(), ShaderVerificationError> {
    let interface = if input { &ir.inputs } else { &ir.outputs };
    for index in 0..count {
        let component = first_component.saturating_add(index as u8);
        if !interface
            .iter()
            .any(|element| element.location == location && element.component == component)
        {
            return Err(ShaderVerificationError::UndeclaredInterfaceAccess {
                source,
                input,
                location,
                component,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interface_components_are_bounded_to_one_vector_lane() {
        assert!(
            ShaderInterfaceElement::new(
                ShaderIoLocation::Generic(0),
                3,
                ShaderScalarType::Float32,
                Some(ShaderInterpolation::Perspective),
            )
            .is_ok()
        );
        assert_eq!(
            ShaderInterfaceElement::new(
                ShaderIoLocation::Position,
                4,
                ShaderScalarType::Float32,
                None,
            ),
            Err(ShaderIrConstructionError::InvalidInterfaceComponent {
                location: ShaderIoLocation::Position,
                component: 4,
            })
        );
    }

    #[test]
    fn ir_retains_numeric_types_predicates_and_source_locations() {
        let ir = ShaderIr::new(
            ShaderStage::Vertex,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![ShaderInstruction::new(
                ShaderSourceLocation::new(8),
                ShaderPredicate::Register {
                    register: 2,
                    inverted: true,
                },
                ShaderOperation::MoveImmediate32 {
                    destination: ShaderRegister::new(3),
                    bits: 0x3f80_0000,
                    scalar_type: ShaderScalarType::Float32,
                },
            )],
        );
        assert_eq!(ir.stage(), ShaderStage::Vertex);
        assert_eq!(ir.instructions()[0].source().byte_offset(), 8);
        assert_eq!(
            ir.instructions()[0].predicate(),
            ShaderPredicate::Register {
                register: 2,
                inverted: true,
            }
        );
    }

    #[test]
    fn verifier_rejects_undefined_registers_and_bad_branch_targets() {
        let store = ShaderInstruction::new(
            ShaderSourceLocation::new(8),
            ShaderPredicate::Always,
            ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(2)].into_boxed_slice(),
                location: ShaderIoLocation::Position,
                first_component: 0,
                scalar_type: ShaderScalarType::Float32,
            },
        );
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        assert_eq!(
            VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Vertex,
                Vec::new(),
                vec![output],
                Vec::new(),
                vec![
                    store,
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::Exit,
                    )
                ],
            )),
            Err(ShaderVerificationError::UndefinedRegister {
                source: ShaderSourceLocation::new(8),
                register: ShaderRegister::new(2),
            })
        );

        assert_eq!(
            VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Vertex,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(8),
                        ShaderPredicate::Always,
                        ShaderOperation::Branch {
                            target: ShaderSourceLocation::new(40),
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::Exit,
                    ),
                ],
            )),
            Err(ShaderVerificationError::InvalidBranchTarget {
                source: ShaderSourceLocation::new(8),
                target: ShaderSourceLocation::new(40),
            })
        );
    }

    #[test]
    fn verifier_intersects_definitions_at_control_flow_joins() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Unsigned32,
            None,
        )
        .unwrap();
        let result = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            Vec::new(),
            vec![output],
            Vec::new(),
            vec![
                ShaderInstruction::new(
                    ShaderSourceLocation::new(8),
                    ShaderPredicate::Always,
                    ShaderOperation::Branch {
                        target: ShaderSourceLocation::new(24),
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 1,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(0)].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ));
        assert_eq!(
            result,
            Err(ShaderVerificationError::UndefinedRegister {
                source: ShaderSourceLocation::new(24),
                register: ShaderRegister::new(0),
            })
        );
    }

    #[test]
    fn wgsl_preserves_pass_mul_w_interpolation() {
        let input = ShaderInterfaceElement::new(
            ShaderIoLocation::Generic(0),
            0,
            ShaderScalarType::Float32,
            Some(ShaderInterpolation::Perspective),
        )
        .unwrap();
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![input],
            Vec::new(),
            Vec::new(),
            vec![
                ShaderInstruction::new(
                    ShaderSourceLocation::new(8),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 0.5_f32.to_bits(),
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::InterpolateInput {
                        destination: ShaderRegister::new(1),
                        location: ShaderIoLocation::Generic(0),
                        component: 0,
                        interpolation: ShaderInterpolation::Perspective,
                        perspective_reciprocal: Some(ShaderRegister::new(0)),
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ))
        .unwrap();
        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(
            module
                .source()
                .contains("input.generic_0.x * bitcast<f32>(registers[0])")
        );
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn reference_evaluator_preserves_bits_and_float_controls() {
        let input = ShaderInterfaceElement::new(
            ShaderIoLocation::Generic(0),
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            vec![input],
            vec![output],
            Vec::new(),
            vec![
                ShaderInstruction::new(
                    ShaderSourceLocation::new(8),
                    ShaderPredicate::Always,
                    ShaderOperation::LoadInput {
                        destinations: vec![ShaderRegister::new(0)].into_boxed_slice(),
                        location: ShaderIoLocation::Generic(0),
                        first_component: 0,
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(1),
                        bits: 2.0_f32.to_bits(),
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::Multiply32 {
                        destination: ShaderRegister::new(2),
                        left: ShaderRegister::new(0),
                        right: ShaderRegister::new(1),
                        scalar_type: ShaderScalarType::Float32,
                        float_control: ShaderFloatControl::PRECISE,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(2)].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: 0,
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(40),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ))
        .unwrap();
        let result = evaluate_shader_ir(
            &shader,
            &ShaderEvaluationInputs::default().with_interface_bits(
                ShaderIoLocation::Generic(0),
                0,
                (-0.0_f32).to_bits(),
            ),
            32,
        )
        .unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some((-0.0_f32).to_bits())
        );

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        naga::front::wgsl::parse_str(module.source()).unwrap();
        assert_eq!(module.language(), ShaderBackendLanguage::Wgsl);
        assert_eq!(
            module.source_map().last().map(|entry| entry.source()),
            Some(ShaderSourceLocation::new(40))
        );
    }

    #[test]
    fn verified_control_flow_evaluates_only_the_selected_path() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Unsigned32,
            None,
        )
        .unwrap();
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            Vec::new(),
            vec![output],
            Vec::new(),
            vec![
                ShaderInstruction::new(
                    ShaderSourceLocation::new(8),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 1,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::Branch {
                        target: ShaderSourceLocation::new(32),
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 2,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(0)].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(40),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ))
        .unwrap();
        let result = evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(result.output_bits(ShaderIoLocation::Position, 0), Some(1));
    }

    #[test]
    fn evaluator_distinguishes_exact_and_guest_approximate_reciprocals() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let make_shader = |accuracy| {
            VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Vertex,
                Vec::new(),
                vec![output],
                Vec::new(),
                vec![
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(8),
                        ShaderPredicate::Always,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(0),
                            bits: f32::INFINITY.to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::Reciprocal32 {
                            destination: ShaderRegister::new(1),
                            source: ShaderRegister::new(0),
                            accuracy,
                            float_control: ShaderFloatControl::PRECISE,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(24),
                        ShaderPredicate::Always,
                        ShaderOperation::StoreOutput {
                            sources: vec![ShaderRegister::new(1)].into_boxed_slice(),
                            location: ShaderIoLocation::Position,
                            first_component: 0,
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(32),
                        ShaderPredicate::Always,
                        ShaderOperation::Exit,
                    ),
                ],
            ))
            .unwrap()
        };
        let exact = make_shader(ShaderReciprocalAccuracy::Exact);
        let result = evaluate_shader_ir(&exact, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some(0.0_f32.to_bits())
        );

        let approximate = make_shader(ShaderReciprocalAccuracy::Approximate);
        assert_eq!(
            evaluate_shader_ir(&approximate, &ShaderEvaluationInputs::default(), 16),
            Err(ShaderEvaluationError::ApproximateOperation(
                ShaderSourceLocation::new(16)
            ))
        );

        let canonicalizing = ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Canonicalize,
            false,
            false,
            false,
        );
        assert_eq!(
            finish_float(0x7fc1_2345, canonicalizing).unwrap(),
            f32::NAN.to_bits()
        );
    }
}
