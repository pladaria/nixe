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

/// Boolean comparison used by backend-neutral floating-point predicate writes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderFloatComparison {
    OrderedLess,
    OrderedEqual,
    OrderedLessOrEqual,
    OrderedGreater,
    OrderedNotEqual,
    OrderedGreaterOrEqual,
    IsNumber,
    IsNan,
    UnorderedLess,
    UnorderedEqual,
    UnorderedLessOrEqual,
    UnorderedGreater,
    UnorderedNotEqual,
    UnorderedGreaterOrEqual,
}

/// Boolean operation combining a comparison with an existing predicate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderPredicateSetOperation {
    And,
    Or,
    Xor,
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

/// Accuracy contract carried by guest scalar math operations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderMathAccuracy {
    /// IEEE operation rounded according to the accompanying float control.
    Exact,
    /// Guest hardware permits an implementation-defined approximation.
    Approximate,
}

/// Backend-neutral scalar special function selected by a guest math unit.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderSpecialFunction {
    Cosine,
    Sine,
    Exp2,
    Log2,
    SquareRoot,
}

/// Descriptor-like resource class visible to one shader.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ShaderResourceKind {
    ConstantBuffer,
    StorageBuffer,
    /// Two-dimensional sampled image with exactly one visible array layer.
    SampledImage,
    /// Two-dimensional sampled image whose array layer is selected by the shader.
    SampledImage2DArray,
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

/// One selected component produced by a neutral texture sample.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ShaderTextureSampleOutput {
    destination: ShaderRegister,
    component: u8,
}

impl ShaderTextureSampleOutput {
    pub fn new(
        destination: ShaderRegister,
        component: u8,
    ) -> Result<Self, ShaderIrConstructionError> {
        if component > 3 {
            return Err(ShaderIrConstructionError::InvalidTextureComponent { component });
        }
        Ok(Self {
            destination,
            component,
        })
    }

    #[must_use]
    pub const fn destination(self) -> ShaderRegister {
        self.destination
    }

    #[must_use]
    pub const fn component(self) -> u8 {
        self.component
    }
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
        let vector_location = matches!(
            location,
            ShaderIoLocation::Position | ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_)
        );
        if component > 3 || (!vector_location && component != 0) {
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
    Move32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        scalar_type: ShaderScalarType,
    },
    FloatAbsolute32 {
        destination: ShaderRegister,
        source: ShaderRegister,
    },
    FloatNegate32 {
        destination: ShaderRegister,
        source: ShaderRegister,
    },
    ConvertIntegerToFloat32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        source_type: ShaderScalarType,
    },
    RoundFloat32ToIntegral {
        destination: ShaderRegister,
        source: ShaderRegister,
        rounding: ShaderRoundingMode,
        flush_denormals_to_zero: bool,
    },
    ConvertFloat32ToInteger {
        destination: ShaderRegister,
        source: ShaderRegister,
        destination_type: ShaderScalarType,
        destination_bits: u8,
        rounding: ShaderRoundingMode,
        flush_denormals_to_zero: bool,
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
    Add32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        scalar_type: ShaderScalarType,
        float_control: ShaderFloatControl,
    },
    ShiftLeft32 {
        destination: ShaderRegister,
        value: ShaderRegister,
        amount: ShaderRegister,
        wrap: bool,
    },
    FloatMinMax32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        minimum: ShaderPredicate,
        float_control: ShaderFloatControl,
    },
    FusedMultiplyAdd32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        addend: ShaderRegister,
        float_control: ShaderFloatControl,
    },
    Reciprocal32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        accuracy: ShaderMathAccuracy,
        float_control: ShaderFloatControl,
    },
    ReciprocalSqrt32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        accuracy: ShaderMathAccuracy,
        float_control: ShaderFloatControl,
    },
    SpecialFunction32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        function: ShaderSpecialFunction,
        accuracy: ShaderMathAccuracy,
        float_control: ShaderFloatControl,
    },
    SetPredicateFloat32 {
        destination: u8,
        left: ShaderRegister,
        right: ShaderRegister,
        comparison: ShaderFloatComparison,
        accumulator: ShaderPredicate,
        set_operation: ShaderPredicateSetOperation,
        flush_denormals_to_zero: bool,
    },
    InterpolateInput {
        destination: ShaderRegister,
        location: ShaderIoLocation,
        component: u8,
        interpolation: ShaderInterpolation,
    },
    LoadConstantBuffer32 {
        destination: ShaderRegister,
        binding: u8,
        byte_offset: u32,
        scalar_type: ShaderScalarType,
    },
    LoadConstantBufferIndexed32 {
        destination: ShaderRegister,
        binding: u8,
        base_byte_offset: i32,
        dynamic_byte_offset: ShaderRegister,
        scalar_type: ShaderScalarType,
    },
    SampleTexture2D {
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 2],
        image_binding: u8,
        sampler_binding: u8,
    },
    SampleTexture2DArray {
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 2],
        /// Register whose low 16 bits hold the unsigned Maxwell array layer.
        array_index: ShaderRegister,
        image_binding: u8,
        sampler_binding: u8,
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
    InvalidTextureComponent {
        component: u8,
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
    UndefinedPredicate(u8),
    UnsupportedRoundingMode(ShaderRoundingMode),
    ApproximateOperation(ShaderSourceLocation),
    TextureSampling(ShaderSourceLocation),
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
    let targets = shader_instruction_entry_points(ir);
    let mut registers = vec![None; 256];
    let mut predicates = [None; 7];
    let mut result = ShaderEvaluationResult::default();
    let mut pc = 0_usize;
    let mut steps = 0_usize;
    while let Some(instruction) = ir.instructions.get(pc) {
        if steps >= step_limit {
            return Err(ShaderEvaluationError::StepLimitExceeded);
        }
        steps += 1;
        pc += 1;
        if !evaluate_shader_predicate(instruction.predicate, &predicates)? {
            continue;
        }
        match instruction.operation() {
            ShaderOperation::Undefined32 { destination } => {
                registers[destination.index() as usize] = Some(0);
            }
            ShaderOperation::MoveImmediate32 {
                destination, bits, ..
            } => registers[destination.index() as usize] = Some(*bits),
            ShaderOperation::Move32 {
                destination,
                source,
                ..
            } => {
                registers[destination.index() as usize] = Some(register_bits(&registers, *source)?);
            }
            ShaderOperation::FloatAbsolute32 {
                destination,
                source,
            } => {
                registers[destination.index() as usize] =
                    Some(register_bits(&registers, *source)? & 0x7fff_ffff);
            }
            ShaderOperation::FloatNegate32 {
                destination,
                source,
            } => {
                registers[destination.index() as usize] =
                    Some(register_bits(&registers, *source)? ^ 0x8000_0000);
            }
            ShaderOperation::ConvertIntegerToFloat32 {
                destination,
                source,
                source_type,
            } => {
                let bits = register_bits(&registers, *source)?;
                let value = match source_type {
                    ShaderScalarType::Signed32 => (bits as i32) as f32,
                    ShaderScalarType::Unsigned32 => bits as f32,
                    _ => unreachable!("verified integer-to-float source is 32-bit integer"),
                };
                registers[destination.index() as usize] = Some(value.to_bits());
            }
            ShaderOperation::RoundFloat32ToIntegral {
                destination,
                source,
                rounding,
                flush_denormals_to_zero,
            } => {
                let mut bits = register_bits(&registers, *source)?;
                if *flush_denormals_to_zero {
                    bits = flush_denormal_bits(bits);
                }
                let value = f32::from_bits(bits);
                let rounded = match rounding {
                    ShaderRoundingMode::TowardNegative => value.floor(),
                    ShaderRoundingMode::TowardPositive => value.ceil(),
                    ShaderRoundingMode::TowardZero => value.trunc(),
                    ShaderRoundingMode::NearestEven => {
                        return Err(ShaderEvaluationError::UnsupportedRoundingMode(*rounding));
                    }
                };
                registers[destination.index() as usize] = Some(rounded.to_bits());
            }
            ShaderOperation::ConvertFloat32ToInteger {
                destination,
                source,
                destination_type,
                destination_bits,
                rounding,
                flush_denormals_to_zero,
            } => {
                let mut bits = register_bits(&registers, *source)?;
                if *flush_denormals_to_zero {
                    bits = flush_denormal_bits(bits);
                }
                let value = f32::from_bits(bits);
                let rounded = match rounding {
                    ShaderRoundingMode::NearestEven => value.round_ties_even(),
                    ShaderRoundingMode::TowardNegative => value.floor(),
                    ShaderRoundingMode::TowardPositive => value.ceil(),
                    ShaderRoundingMode::TowardZero => value.trunc(),
                };
                let converted = if rounded.is_nan() {
                    0
                } else if *destination_type == ShaderScalarType::Signed32 {
                    let shift = 32 - u32::from(*destination_bits);
                    let minimum = -(1_i64 << (u32::from(*destination_bits) - 1));
                    let maximum = (1_i64 << (u32::from(*destination_bits) - 1)) - 1;
                    let integer = (rounded as i64).clamp(minimum, maximum) as i32;
                    ((integer << shift) >> shift) as u32
                } else {
                    let maximum = if *destination_bits == 32 {
                        u32::MAX as u64
                    } else {
                        (1_u64 << u32::from(*destination_bits)) - 1
                    };
                    (rounded as u64).min(maximum) as u32
                };
                registers[destination.index() as usize] = Some(converted);
            }
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
            ShaderOperation::Add32 {
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
                        evaluate_float_binary(left, right, *float_control, |a, b| a + b)?
                    }
                    ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 => {
                        left.wrapping_add(right)
                    }
                    _ => unreachable!("Add32 only admits 32-bit scalar types"),
                };
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::ShiftLeft32 {
                destination,
                value,
                amount,
                wrap,
            } => {
                let value = register_bits(&registers, *value)?;
                let amount = register_bits(&registers, *amount)?;
                let shifted = if *wrap {
                    value.wrapping_shl(amount & 31)
                } else if amount < 32 {
                    value << amount
                } else {
                    0
                };
                registers[destination.index() as usize] = Some(shifted);
            }
            ShaderOperation::FloatMinMax32 {
                destination,
                left,
                right,
                minimum,
                float_control,
            } => {
                let minimum = evaluate_shader_predicate(*minimum, &predicates)?;
                let value = evaluate_float_binary(
                    register_bits(&registers, *left)?,
                    register_bits(&registers, *right)?,
                    *float_control,
                    |left, right| {
                        if minimum {
                            left.min(right)
                        } else {
                            left.max(right)
                        }
                    },
                )?;
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } => {
                let left = register_bits(&registers, *left)?;
                let right = register_bits(&registers, *right)?;
                let addend = register_bits(&registers, *addend)?;
                let value =
                    evaluate_float_ternary(left, right, addend, *float_control, f32::mul_add)?;
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::Reciprocal32 {
                destination,
                source,
                accuracy,
                float_control,
            } => {
                if *accuracy == ShaderMathAccuracy::Approximate {
                    return Err(ShaderEvaluationError::ApproximateOperation(
                        instruction.source,
                    ));
                }
                let source = register_bits(&registers, *source)?;
                let value = evaluate_float_unary(source, *float_control, |value| value.recip())?;
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::ReciprocalSqrt32 {
                destination,
                source,
                accuracy,
                float_control,
            } => {
                if *accuracy == ShaderMathAccuracy::Approximate {
                    return Err(ShaderEvaluationError::ApproximateOperation(
                        instruction.source,
                    ));
                }
                let source = register_bits(&registers, *source)?;
                let value =
                    evaluate_float_unary(source, *float_control, |value| value.sqrt().recip())?;
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::SpecialFunction32 {
                destination,
                source,
                function,
                accuracy,
                float_control,
            } => {
                if *accuracy == ShaderMathAccuracy::Approximate {
                    return Err(ShaderEvaluationError::ApproximateOperation(
                        instruction.source,
                    ));
                }
                let source = register_bits(&registers, *source)?;
                let value = evaluate_float_unary(source, *float_control, |value| match function {
                    ShaderSpecialFunction::Cosine => value.cos(),
                    ShaderSpecialFunction::Sine => value.sin(),
                    ShaderSpecialFunction::Exp2 => value.exp2(),
                    ShaderSpecialFunction::Log2 => value.log2(),
                    ShaderSpecialFunction::SquareRoot => value.sqrt(),
                })?;
                registers[destination.index() as usize] = Some(value);
            }
            ShaderOperation::SetPredicateFloat32 {
                destination,
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            } => {
                let mut left = register_bits(&registers, *left)?;
                let mut right = register_bits(&registers, *right)?;
                if *flush_denormals_to_zero {
                    left = flush_denormal_bits(left);
                    right = flush_denormal_bits(right);
                }
                let compared = evaluate_float_comparison(
                    f32::from_bits(left),
                    f32::from_bits(right),
                    *comparison,
                );
                let accumulated = evaluate_shader_predicate(*accumulator, &predicates)?;
                predicates[usize::from(*destination)] = Some(match set_operation {
                    ShaderPredicateSetOperation::And => compared && accumulated,
                    ShaderPredicateSetOperation::Or => compared || accumulated,
                    ShaderPredicateSetOperation::Xor => compared ^ accumulated,
                });
            }
            ShaderOperation::InterpolateInput {
                destination,
                location,
                component,
                ..
            } => {
                let bits = inputs
                    .interface
                    .get(&(*location, *component))
                    .copied()
                    .ok_or(ShaderEvaluationError::MissingInterfaceInput {
                        location: *location,
                        component: *component,
                    })?;
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
            ShaderOperation::LoadConstantBufferIndexed32 {
                destination,
                binding,
                base_byte_offset,
                dynamic_byte_offset,
                ..
            } => {
                let dynamic_byte_offset = register_bits(&registers, *dynamic_byte_offset)?;
                let byte_offset = dynamic_byte_offset.wrapping_add(*base_byte_offset as u32) & !3;
                registers[destination.index() as usize] = Some(
                    inputs
                        .constant_buffers
                        .get(&(*binding, byte_offset))
                        .copied()
                        .ok_or(ShaderEvaluationError::MissingConstantBufferWord {
                            binding: *binding,
                            byte_offset,
                        })?,
                );
            }
            ShaderOperation::SampleTexture2D { .. }
            | ShaderOperation::SampleTexture2DArray { .. } => {
                return Err(ShaderEvaluationError::TextureSampling(instruction.source));
            }
            ShaderOperation::Branch { target } => {
                pc = targets[target];
            }
            ShaderOperation::Exit => return Ok(result),
        }
    }
    Err(ShaderEvaluationError::MissingExit)
}

fn evaluate_shader_predicate(
    predicate: ShaderPredicate,
    predicates: &[Option<bool>; 7],
) -> Result<bool, ShaderEvaluationError> {
    match predicate {
        ShaderPredicate::Always => Ok(true),
        ShaderPredicate::Never => Ok(false),
        ShaderPredicate::Register { register, inverted } => predicates
            .get(usize::from(register))
            .and_then(|value| *value)
            .map(|value| value ^ inverted)
            .ok_or(ShaderEvaluationError::UndefinedPredicate(register)),
    }
}

fn evaluate_float_comparison(left: f32, right: f32, comparison: ShaderFloatComparison) -> bool {
    let unordered = left.is_nan() || right.is_nan();
    match comparison {
        ShaderFloatComparison::OrderedLess => !unordered && left < right,
        ShaderFloatComparison::OrderedEqual => !unordered && left == right,
        ShaderFloatComparison::OrderedLessOrEqual => !unordered && left <= right,
        ShaderFloatComparison::OrderedGreater => !unordered && left > right,
        ShaderFloatComparison::OrderedNotEqual => !unordered && left != right,
        ShaderFloatComparison::OrderedGreaterOrEqual => !unordered && left >= right,
        ShaderFloatComparison::IsNumber => !unordered,
        ShaderFloatComparison::IsNan => unordered,
        ShaderFloatComparison::UnorderedLess => unordered || left < right,
        ShaderFloatComparison::UnorderedEqual => unordered || left == right,
        ShaderFloatComparison::UnorderedLessOrEqual => unordered || left <= right,
        ShaderFloatComparison::UnorderedGreater => unordered || left > right,
        ShaderFloatComparison::UnorderedNotEqual => unordered || left != right,
        ShaderFloatComparison::UnorderedGreaterOrEqual => unordered || left >= right,
    }
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

fn evaluate_float_ternary(
    mut left: u32,
    mut right: u32,
    mut addend: u32,
    control: ShaderFloatControl,
    operation: impl FnOnce(f32, f32, f32) -> f32,
) -> Result<u32, ShaderEvaluationError> {
    if control.rounding != ShaderRoundingMode::NearestEven {
        return Err(ShaderEvaluationError::UnsupportedRoundingMode(
            control.rounding,
        ));
    }
    if control.denormals_are_zero {
        left = flush_denormal_bits(left);
        right = flush_denormal_bits(right);
        addend = flush_denormal_bits(addend);
    }
    finish_float(
        operation(
            f32::from_bits(left),
            f32::from_bits(right),
            f32::from_bits(addend),
        )
        .to_bits(),
        control,
    )
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
    let mut input_groups = interface_groups(&ir.inputs)?;
    if ir.stage == ShaderStage::Vertex {
        input_groups
            .entry(ShaderIoLocation::VertexId)
            .or_insert(InterfaceGroup {
                components: 1,
                scalar_type: ShaderScalarType::Unsigned32,
                interpolation: None,
            });
    }
    let output_groups = interface_groups(&ir.outputs)?;
    let mut source = String::new();
    emit_wgsl_resources(&mut source, ir)?;
    source.push_str(
        "fn nixe_flush_denormal(bits: u32) -> u32 {\n\
           let magnitude = bits & 0x7fffffffu;\n\
           return select(bits, bits & 0x80000000u, magnitude > 0u && magnitude < 0x00800000u);\n\
         }\n\n\
         fn nixe_round_ties_even(value: f32) -> f32 {\n\
           if (value != value) { return value; }\n\
           let lower = floor(value);\n\
           let fraction = value - lower;\n\
           if (fraction < 0.5) { return lower; }\n\
           if (fraction > 0.5) { return lower + 1.0; }\n\
           return select(lower, lower + 1.0, (i32(lower) & 1) != 0);\n\
         }\n\n",
    );
    if !input_groups.is_empty() {
        emit_interface_struct(&mut source, "ShaderInput", ir.stage, true, &input_groups)?;
    }
    emit_interface_struct(&mut source, "ShaderOutput", ir.stage, false, &output_groups)?;
    let entry_point = match ir.stage {
        ShaderStage::Vertex => "nixe_guest_vertex",
        ShaderStage::Fragment => "nixe_guest_fragment",
        _ => unreachable!("unsupported stages returned above"),
    };
    if input_groups.is_empty() {
        source.push_str(&format!("fn {entry_point}() -> ShaderOutput {{\n"));
    } else {
        source.push_str(&format!(
            "fn {entry_point}(input: ShaderInput) -> ShaderOutput {{\n"
        ));
    }
    source.push_str("  var registers: array<u32, 256>;\n");
    source.push_str("  var predicates: array<bool, 7>;\n");
    source.push_str("  var output: ShaderOutput;\n");
    let mut source_map = Vec::new();
    if ir
        .instructions
        .iter()
        .any(|instruction| matches!(instruction.operation, ShaderOperation::Branch { .. }))
    {
        emit_wgsl_control_flow(&mut source, ir, &mut source_map)?;
        source.push_str("}\n");
    } else {
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
            let conditional = match instruction.predicate {
                ShaderPredicate::Never => continue,
                ShaderPredicate::Register { register, inverted } => {
                    source.push_str(&format!(
                        "  if ({}predicates[{register}]) {{\n",
                        if inverted { "!" } else { "" }
                    ));
                    true
                }
                ShaderPredicate::Always => false,
            };
            emit_wgsl_operation(&mut source, instruction)?;
            if conditional {
                source.push_str("  }\n");
            }
        }
        source.push_str("}\n");
    }
    if ir.stage == ShaderStage::Vertex
        && output_groups
            .get(&ShaderIoLocation::Position)
            .is_some_and(|position| position.scalar_type == ShaderScalarType::Float32)
    {
        emit_wgsl_vertex_entry_points(&mut source, &output_groups)?;
    } else if ir.stage == ShaderStage::Vertex {
        source.push_str(
            "\n@vertex\nfn main(input: ShaderInput) -> ShaderOutput {\n  return nixe_guest_vertex(input);\n}\n",
        );
    } else {
        emit_wgsl_fragment_entry_points(&mut source, &input_groups, &output_groups);
    }
    Ok(ShaderBackendModule {
        stage: ir.stage,
        language: ShaderBackendLanguage::Wgsl,
        source: source.into_boxed_str(),
        source_map: source_map.into_boxed_slice(),
    })
}

fn emit_wgsl_fragment_entry_points(
    source: &mut String,
    inputs: &std::collections::BTreeMap<ShaderIoLocation, InterfaceGroup>,
    outputs: &std::collections::BTreeMap<ShaderIoLocation, InterfaceGroup>,
) {
    let parameter = if inputs.is_empty() {
        ""
    } else {
        "input: ShaderInput"
    };
    let argument = if inputs.is_empty() { "" } else { "input" };
    source.push_str(&format!(
        "\n@fragment\nfn main({parameter}) -> ShaderOutput {{\n  return nixe_guest_fragment({argument});\n}}\n"
    ));
    if !outputs
        .get(&ShaderIoLocation::Color(0))
        .is_some_and(|color| {
            color.scalar_type == ShaderScalarType::Float32 && color.components == 4
        })
    {
        return;
    }
    source.push_str("\noverride nixe_alpha_reference: f32 = 0.0;\n");
    for (name, comparison) in [
        ("never", None),
        ("less", Some("<")),
        ("equal", Some("==")),
        ("less_equal", Some("<=")),
        ("greater", Some(">")),
        ("not_equal", Some("!=")),
        ("greater_equal", Some(">=")),
        ("always", Some("always")),
    ] {
        source.push_str(&format!(
            "\n@fragment\nfn nixe_alpha_{name}({parameter}) -> ShaderOutput {{\n  let output = nixe_guest_fragment({argument});\n"
        ));
        match comparison {
            None => source.push_str("  discard;\n"),
            Some("always") => {}
            Some(operator) => source.push_str(&format!(
                "  if (!(output.color_0.w {operator} nixe_alpha_reference)) {{ discard; }}\n"
            )),
        }
        source.push_str("  return output;\n}\n");
    }
}

fn emit_wgsl_vertex_entry_points(
    source: &mut String,
    outputs: &std::collections::BTreeMap<ShaderIoLocation, InterfaceGroup>,
) -> Result<(), ShaderBackendLoweringError> {
    source.push_str(
        "\n@vertex\nfn main(input: ShaderInput) -> ShaderOutput {\n  return nixe_guest_vertex(input);\n}\n\n\
         @vertex\nfn nixe_fill_rectangle(input: ShaderInput) -> ShaderOutput {\n\
           let host_vertex = input.vertex_id;\n\
           let guest_base = (host_vertex / 6u) * 3u;\n\
           var input0 = input;\n  var input1 = input;\n  var input2 = input;\n\
           input0.vertex_id = guest_base;\n\
           input1.vertex_id = guest_base + 1u;\n\
           input2.vertex_id = guest_base + 2u;\n\
           let vertex0 = nixe_guest_vertex(input0);\n\
           let vertex1 = nixe_guest_vertex(input1);\n\
           let vertex2 = nixe_guest_vertex(input2);\n\
           let point0 = vertex0.position.xy / vertex0.position.w;\n\
           let point1 = vertex1.position.xy / vertex1.position.w;\n\
           let point2 = vertex2.position.xy / vertex2.position.w;\n\
           let lower = min(point0, min(point1, point2));\n\
           let upper = max(point0, max(point1, point2));\n\
           let corners = array<vec2<f32>, 6>(\n\
             vec2<f32>(lower.x, lower.y), vec2<f32>(upper.x, lower.y),\n\
             vec2<f32>(upper.x, upper.y), vec2<f32>(lower.x, lower.y),\n\
             vec2<f32>(upper.x, upper.y), vec2<f32>(lower.x, upper.y));\n\
           let corner = corners[host_vertex % 6u];\n\
           let denominator = (point1.y - point2.y) * (point0.x - point2.x) +\n\
             (point2.x - point1.x) * (point0.y - point2.y);\n\
           let weight0 = ((point1.y - point2.y) * (corner.x - point2.x) +\n\
             (point2.x - point1.x) * (corner.y - point2.y)) / denominator;\n\
           let weight1 = ((point2.y - point0.y) * (corner.x - point2.x) +\n\
             (point0.x - point2.x) * (corner.y - point2.y)) / denominator;\n\
           let weight2 = 1.0 - weight0 - weight1;\n\
           let reciprocal_w = weight0 / vertex0.position.w +\n\
             weight1 / vertex1.position.w + weight2 / vertex2.position.w;\n\
           let rectangle_w = 1.0 / reciprocal_w;\n\
           var output = vertex0;\n",
    );
    for (location, group) in outputs {
        let field = wgsl_field_name(*location);
        match location {
            ShaderIoLocation::Position => source.push_str(
                "  let rectangle_z = weight0 * vertex0.position.z / vertex0.position.w +\n\
                   weight1 * vertex1.position.z / vertex1.position.w +\n\
                   weight2 * vertex2.position.z / vertex2.position.w;\n\
                   output.position = vec4<f32>(corner * rectangle_w, rectangle_z * rectangle_w, rectangle_w);\n",
            ),
            ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_) => match group.interpolation {
                Some(ShaderInterpolation::Constant) => {
                    source.push_str(&format!("  output.{field} = vertex2.{field};\n"));
                }
                Some(ShaderInterpolation::ScreenLinear) => {
                    if group.scalar_type != ShaderScalarType::Float32 {
                        return Err(ShaderBackendLoweringError::InconsistentInterfaceType(*location));
                    }
                    source.push_str(&format!(
                        "  output.{field} = weight0 * vertex0.{field} + weight1 * vertex1.{field} + weight2 * vertex2.{field};\n"
                    ));
                }
                Some(ShaderInterpolation::Perspective) => {
                    if group.scalar_type != ShaderScalarType::Float32 {
                        return Err(ShaderBackendLoweringError::InconsistentInterfaceType(*location));
                    }
                    source.push_str(&format!(
                        "  output.{field} = (weight0 * vertex0.{field} / vertex0.position.w + weight1 * vertex1.{field} / vertex1.position.w + weight2 * vertex2.{field} / vertex2.position.w) * rectangle_w;\n"
                    ));
                }
                // An unconsumed vertex output has no linked interpolation
                // contract. Its value is irrelevant to the fragment stage.
                None => {}
            },
            other => return Err(ShaderBackendLoweringError::UnsupportedInterface(*other)),
        }
    }
    source.push_str("  return output;\n}\n");
    Ok(())
}

fn emit_wgsl_control_flow(
    source: &mut String,
    ir: &ShaderIr,
    source_map: &mut Vec<ShaderBackendSourceMapEntry>,
) -> Result<(), ShaderBackendLoweringError> {
    let locations = shader_instruction_entry_points(ir);
    let mut leaders = BTreeSet::from([0_usize]);
    for (index, instruction) in ir.instructions.iter().enumerate() {
        if let ShaderOperation::Branch { target } = instruction.operation {
            leaders.insert(locations[&target]);
            if index + 1 < ir.instructions.len() {
                leaders.insert(index + 1);
            }
        } else if matches!(instruction.operation, ShaderOperation::Exit)
            && index + 1 < ir.instructions.len()
        {
            leaders.insert(index + 1);
        }
    }
    let leaders = leaders.into_iter().collect::<Vec<_>>();
    let mut instruction_blocks = vec![0_usize; ir.instructions.len()];
    for (block, start) in leaders.iter().copied().enumerate() {
        let end = leaders
            .get(block + 1)
            .copied()
            .unwrap_or(ir.instructions.len());
        instruction_blocks[start..end].fill(block);
    }

    source.push_str("  var nixe_block = 0u;\n");
    source.push_str("  loop {\n");
    source.push_str("    switch nixe_block {\n");
    for (block, start) in leaders.iter().copied().enumerate() {
        let end = leaders
            .get(block + 1)
            .copied()
            .unwrap_or(ir.instructions.len());
        source.push_str(&format!("      case {block}u: {{\n"));
        let mut terminated = false;
        for (index, instruction) in ir.instructions[start..end].iter().enumerate() {
            let instruction_index = start + index;
            let line = source.lines().count() as u32 + 1;
            source_map.push(ShaderBackendSourceMapEntry {
                backend_line: line,
                source: instruction.source,
            });
            source.push_str(&format!(
                "        // Maxwell code byte offset 0x{:x}\n",
                instruction.source.byte_offset()
            ));
            match instruction.operation() {
                ShaderOperation::Branch { target } => {
                    let target_block = instruction_blocks[locations[target]];
                    let fallthrough = instruction_blocks.get(instruction_index + 1).copied();
                    emit_wgsl_block_transfer(
                        source,
                        instruction.predicate,
                        target_block,
                        fallthrough,
                    );
                    terminated = true;
                }
                ShaderOperation::Exit => {
                    emit_wgsl_block_exit(
                        source,
                        instruction.predicate,
                        instruction_blocks.get(instruction_index + 1).copied(),
                    );
                    terminated = true;
                }
                _ => emit_wgsl_nested_operation(source, instruction)?,
            }
        }
        if !terminated {
            if let Some(next) = leaders.get(block + 1) {
                source.push_str(&format!(
                    "        nixe_block = {}u;\n        continue;\n",
                    instruction_blocks[*next]
                ));
            } else {
                source.push_str("        return output;\n");
            }
        }
        source.push_str("      }\n");
    }
    source.push_str("      default: { return output; }\n");
    source.push_str("    }\n");
    source.push_str("  }\n");
    source.push_str("  return output;\n");
    Ok(())
}

fn emit_wgsl_nested_operation(
    source: &mut String,
    instruction: &ShaderInstruction,
) -> Result<(), ShaderBackendLoweringError> {
    if instruction.predicate == ShaderPredicate::Never {
        return Ok(());
    }
    let conditional =
        if let ShaderPredicate::Register { register, inverted } = instruction.predicate {
            source.push_str(&format!(
                "        if ({}predicates[{register}]) {{\n",
                if inverted { "!" } else { "" }
            ));
            true
        } else {
            false
        };
    let mut operation = String::new();
    emit_wgsl_operation(&mut operation, instruction)?;
    for line in operation.lines() {
        source.push_str("      ");
        source.push_str(line);
        source.push('\n');
    }
    if conditional {
        source.push_str("        }\n");
    }
    Ok(())
}

fn emit_wgsl_block_transfer(
    source: &mut String,
    predicate: ShaderPredicate,
    target: usize,
    fallthrough: Option<usize>,
) {
    match predicate {
        ShaderPredicate::Always => source.push_str(&format!(
            "        nixe_block = {target}u;\n        continue;\n"
        )),
        ShaderPredicate::Never => emit_wgsl_fallthrough(source, fallthrough),
        ShaderPredicate::Register { .. } => {
            let condition = wgsl_predicate_expression(predicate);
            source.push_str(&format!(
                "        if ({condition}) {{\n          nixe_block = {target}u;\n        }} else {{\n"
            ));
            if let Some(fallthrough) = fallthrough {
                source.push_str(&format!("          nixe_block = {fallthrough}u;\n"));
            } else {
                source.push_str("          return output;\n");
            }
            source.push_str("        }\n        continue;\n");
        }
    }
}

fn emit_wgsl_block_exit(
    source: &mut String,
    predicate: ShaderPredicate,
    fallthrough: Option<usize>,
) {
    match predicate {
        ShaderPredicate::Always => source.push_str("        return output;\n"),
        ShaderPredicate::Never => emit_wgsl_fallthrough(source, fallthrough),
        ShaderPredicate::Register { .. } => {
            let condition = wgsl_predicate_expression(predicate);
            source.push_str(&format!(
                "        if ({condition}) {{\n          return output;\n        }}\n"
            ));
            emit_wgsl_fallthrough(source, fallthrough);
        }
    }
}

fn emit_wgsl_fallthrough(source: &mut String, fallthrough: Option<usize>) {
    if let Some(fallthrough) = fallthrough {
        source.push_str(&format!(
            "        nixe_block = {fallthrough}u;\n        continue;\n"
        ));
    } else {
        source.push_str("        return output;\n");
    }
}

fn emit_wgsl_resources(
    source: &mut String,
    ir: &ShaderIr,
) -> Result<(), ShaderBackendLoweringError> {
    for resource in &ir.resources {
        if !resource.readable || resource.writable {
            return Err(ShaderBackendLoweringError::ResourceAccess(
                ir.instructions[0].source,
            ));
        }
        match resource.kind {
            ShaderResourceKind::ConstantBuffer => source.push_str(&format!(
                "@group(0) @binding({}) var<storage, read> constant_buffer_{}: array<u32>;\n",
                resource.binding, resource.binding
            )),
            ShaderResourceKind::SampledImage => source.push_str(&format!(
                "@group(0) @binding({}) var sampled_image_{}: texture_2d<f32>;\n",
                resource.binding, resource.binding
            )),
            ShaderResourceKind::SampledImage2DArray => source.push_str(&format!(
                "@group(0) @binding({}) var sampled_image_{}: texture_2d_array<f32>;\n",
                resource.binding, resource.binding
            )),
            ShaderResourceKind::Sampler => source.push_str(&format!(
                "@group(0) @binding({}) var sampler_{}: sampler;\n",
                resource.binding, resource.binding
            )),
            ShaderResourceKind::StorageBuffer | ShaderResourceKind::StorageImage => {
                return Err(ShaderBackendLoweringError::ResourceAccess(
                    ir.instructions[0].source,
                ));
            }
        }
    }
    if !ir.resources.is_empty() {
        source.push('\n');
    }
    Ok(())
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
        ShaderOperation::Move32 {
            destination,
            source: operand,
            ..
        } => source.push_str(&format!(
            "  registers[{}] = registers[{}];\n",
            destination.index(),
            operand.index()
        )),
        ShaderOperation::FloatAbsolute32 {
            destination,
            source: operand,
        } => source.push_str(&format!(
            "  registers[{}] = registers[{}] & 0x7fffffffu;\n",
            destination.index(),
            operand.index()
        )),
        ShaderOperation::FloatNegate32 {
            destination,
            source: operand,
        } => source.push_str(&format!(
            "  registers[{}] = registers[{}] ^ 0x80000000u;\n",
            destination.index(),
            operand.index()
        )),
        ShaderOperation::ConvertIntegerToFloat32 {
            destination,
            source: operand,
            source_type,
        } => {
            let integer = match source_type {
                ShaderScalarType::Signed32 => {
                    format!("bitcast<i32>(registers[{}])", operand.index())
                }
                ShaderScalarType::Unsigned32 => format!("registers[{}]", operand.index()),
                _ => {
                    return Err(ShaderBackendLoweringError::NumericControl(
                        instruction.source,
                    ));
                }
            };
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>(f32({integer}));\n",
                destination.index()
            ));
        }
        ShaderOperation::RoundFloat32ToIntegral {
            destination,
            source: operand,
            rounding,
            flush_denormals_to_zero,
        } => {
            let bits = if *flush_denormals_to_zero {
                format!("nixe_flush_denormal(registers[{}])", operand.index())
            } else {
                format!("registers[{}]", operand.index())
            };
            let function = match rounding {
                ShaderRoundingMode::TowardNegative => "floor",
                ShaderRoundingMode::TowardPositive => "ceil",
                ShaderRoundingMode::TowardZero => "trunc",
                ShaderRoundingMode::NearestEven => {
                    return Err(ShaderBackendLoweringError::NumericControl(
                        instruction.source,
                    ));
                }
            };
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>({function}(bitcast<f32>({bits})));\n",
                destination.index()
            ));
        }
        ShaderOperation::ConvertFloat32ToInteger {
            destination,
            source: operand,
            destination_type,
            destination_bits,
            rounding,
            flush_denormals_to_zero,
        } => {
            let bits = if *flush_denormals_to_zero {
                format!("nixe_flush_denormal(registers[{}])", operand.index())
            } else {
                format!("registers[{}]", operand.index())
            };
            let value = format!("bitcast<f32>({bits})");
            let rounded = match rounding {
                ShaderRoundingMode::NearestEven => format!("nixe_round_ties_even({value})"),
                ShaderRoundingMode::TowardNegative => format!("floor({value})"),
                ShaderRoundingMode::TowardPositive => format!("ceil({value})"),
                ShaderRoundingMode::TowardZero => format!("trunc({value})"),
            };
            let local = format!("nixe_f2i_{:x}", instruction.source.byte_offset());
            source.push_str(&format!("  let {local} = {rounded};\n"));
            let clean = format!("{local}_clean");
            source.push_str(&format!(
                "  let {clean} = select({local}, 0.0, {local} != {local});\n"
            ));
            let (minimum, maximum, converted) = match destination_type {
                ShaderScalarType::Signed32 => {
                    let minimum = -(1_i64 << (u32::from(*destination_bits) - 1));
                    let maximum = (1_i64 << (u32::from(*destination_bits) - 1)) - 1;
                    let conversion_maximum = if *destination_bits == 32 {
                        2_147_483_520_i64
                    } else {
                        maximum
                    };
                    (
                        format!("{minimum}.0"),
                        format!("{}.0", maximum + 1),
                        format!(
                            "bitcast<u32>(i32(clamp({clean}, {minimum}.0, {conversion_maximum}.0)))"
                        ),
                    )
                }
                ShaderScalarType::Unsigned32 => {
                    let maximum = 1_u64 << u32::from(*destination_bits);
                    let conversion_maximum = if *destination_bits == 32 {
                        4_294_967_040_u64
                    } else {
                        maximum - 1
                    };
                    (
                        "0.0".to_owned(),
                        format!("{maximum}.0"),
                        format!("u32(clamp({clean}, 0.0, {conversion_maximum}.0))"),
                    )
                }
                _ => {
                    return Err(ShaderBackendLoweringError::NumericControl(
                        instruction.source,
                    ));
                }
            };
            let minimum_bits = if *destination_type == ShaderScalarType::Signed32 {
                1_u32 << (u32::from(*destination_bits) - 1)
            } else {
                0
            };
            let minimum_bits =
                if *destination_type == ShaderScalarType::Signed32 && *destination_bits < 32 {
                    minimum_bits | (!0_u32 << u32::from(*destination_bits))
                } else {
                    minimum_bits
                };
            let maximum_bits = if *destination_type == ShaderScalarType::Signed32 {
                (1_u32 << (u32::from(*destination_bits) - 1)).wrapping_sub(1)
            } else if *destination_bits == 32 {
                u32::MAX
            } else {
                (1_u32 << u32::from(*destination_bits)) - 1
            };
            source.push_str(&format!(
                "  registers[{}] = select(select({converted}, 0x{maximum_bits:08x}u, {clean} >= {maximum}), 0x{minimum_bits:08x}u, {clean} <= {minimum});\n",
                destination.index()
            ));
            source.push_str(&format!(
                "  registers[{}] = select(registers[{}], 0u, {local} != {local});\n",
                destination.index(),
                destination.index()
            ));
        }
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
                            wgsl_interface_component(
                                *location,
                                first_component.saturating_add(index as u8),
                            )
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
                    wgsl_interface_component(
                        *location,
                        first_component.saturating_add(index as u8),
                    ),
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
            let expression = match scalar_type {
                ShaderScalarType::Float32 => wgsl_float_multiply_expression(
                    instruction.source,
                    left.index(),
                    right.index(),
                    *float_control,
                )?,
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
        ShaderOperation::Add32 {
            destination,
            left,
            right,
            scalar_type,
            float_control,
        } => {
            let expression = match scalar_type {
                ShaderScalarType::Float32 => wgsl_float_add_expression(
                    instruction.source,
                    left.index(),
                    right.index(),
                    *float_control,
                )?,
                ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 => {
                    format!("registers[{}] + registers[{}]", left.index(), right.index())
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
        ShaderOperation::ShiftLeft32 {
            destination,
            value,
            amount,
            wrap,
        } => {
            let shifted = format!(
                "registers[{}] << (registers[{}] & 31u)",
                value.index(),
                amount.index()
            );
            let expression = if *wrap {
                shifted
            } else {
                format!("select(0u, {shifted}, registers[{}] < 32u)", amount.index())
            };
            source.push_str(&format!(
                "  registers[{}] = {expression};\n",
                destination.index()
            ));
        }
        ShaderOperation::FloatMinMax32 {
            destination,
            left,
            right,
            minimum,
            float_control,
        } => {
            let expression = wgsl_float_min_max_expression(
                instruction.source,
                left.index(),
                right.index(),
                *minimum,
                *float_control,
            )?;
            source.push_str(&format!(
                "  registers[{}] = {expression};\n",
                destination.index()
            ));
        }
        ShaderOperation::FusedMultiplyAdd32 {
            destination,
            left,
            right,
            addend,
            float_control,
        } => {
            let expression = wgsl_float_fma_expression(
                instruction.source,
                left.index(),
                right.index(),
                addend.index(),
                *float_control,
            )?;
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
        ShaderOperation::ReciprocalSqrt32 {
            destination,
            source: operand,
            float_control,
            ..
        } => {
            require_precise_float(instruction.source, *float_control)?;
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>(inverseSqrt(bitcast<f32>(registers[{}])));\n",
                destination.index(),
                operand.index()
            ));
        }
        ShaderOperation::SpecialFunction32 {
            destination,
            source: operand,
            function,
            float_control,
            ..
        } => {
            require_precise_float(instruction.source, *float_control)?;
            let function = match function {
                ShaderSpecialFunction::Cosine => "cos",
                ShaderSpecialFunction::Sine => "sin",
                ShaderSpecialFunction::Exp2 => "exp2",
                ShaderSpecialFunction::Log2 => "log2",
                ShaderSpecialFunction::SquareRoot => "sqrt",
            };
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>({function}(bitcast<f32>(registers[{}])));\n",
                destination.index(),
                operand.index()
            ));
        }
        ShaderOperation::SetPredicateFloat32 {
            destination,
            left,
            right,
            comparison,
            accumulator,
            set_operation,
            flush_denormals_to_zero,
        } => {
            let operand = |register: ShaderRegister| {
                let bits = format!("registers[{}]", register.index());
                if *flush_denormals_to_zero {
                    format!("bitcast<f32>(nixe_flush_denormal({bits}))")
                } else {
                    format!("bitcast<f32>({bits})")
                }
            };
            let left = operand(*left);
            let right = operand(*right);
            // WGSL has no scalar isNan builtin; IEEE NaN is the only value
            // which compares unequal to itself.
            let unordered = format!("(({left} != {left}) || ({right} != {right}))");
            let compared = match comparison {
                ShaderFloatComparison::OrderedLess => format!("(!{unordered} && {left} < {right})"),
                ShaderFloatComparison::OrderedEqual => {
                    format!("(!{unordered} && {left} == {right})")
                }
                ShaderFloatComparison::OrderedLessOrEqual => {
                    format!("(!{unordered} && {left} <= {right})")
                }
                ShaderFloatComparison::OrderedGreater => {
                    format!("(!{unordered} && {left} > {right})")
                }
                ShaderFloatComparison::OrderedNotEqual => {
                    format!("(!{unordered} && {left} != {right})")
                }
                ShaderFloatComparison::OrderedGreaterOrEqual => {
                    format!("(!{unordered} && {left} >= {right})")
                }
                ShaderFloatComparison::IsNumber => format!("!{unordered}"),
                ShaderFloatComparison::IsNan => unordered,
                ShaderFloatComparison::UnorderedLess => {
                    format!("({unordered} || {left} < {right})")
                }
                ShaderFloatComparison::UnorderedEqual => {
                    format!("({unordered} || {left} == {right})")
                }
                ShaderFloatComparison::UnorderedLessOrEqual => {
                    format!("({unordered} || {left} <= {right})")
                }
                ShaderFloatComparison::UnorderedGreater => {
                    format!("({unordered} || {left} > {right})")
                }
                ShaderFloatComparison::UnorderedNotEqual => {
                    format!("({unordered} || {left} != {right})")
                }
                ShaderFloatComparison::UnorderedGreaterOrEqual => {
                    format!("({unordered} || {left} >= {right})")
                }
            };
            let accumulator = wgsl_predicate_expression(*accumulator);
            let operator = match set_operation {
                ShaderPredicateSetOperation::And => "&&",
                ShaderPredicateSetOperation::Or => "||",
                ShaderPredicateSetOperation::Xor => "!=",
            };
            source.push_str(&format!(
                "  predicates[{destination}] = ({compared}) {operator} ({accumulator});\n"
            ));
        }
        ShaderOperation::InterpolateInput {
            destination,
            location,
            component,
            ..
        } => {
            source.push_str(&format!(
                "  registers[{}] = bitcast<u32>(input.{}{});\n",
                destination.index(),
                wgsl_field_name(*location),
                wgsl_component(*component)
            ));
        }
        ShaderOperation::LoadConstantBuffer32 {
            destination,
            binding,
            byte_offset,
            ..
        } => source.push_str(&format!(
            "  registers[{}] = constant_buffer_{}[{}u];\n",
            destination.index(),
            binding,
            byte_offset / 4
        )),
        ShaderOperation::LoadConstantBufferIndexed32 {
            destination,
            binding,
            base_byte_offset,
            dynamic_byte_offset,
            ..
        } => source.push_str(&format!(
            "  registers[{}] = constant_buffer_{}[(registers[{}] + 0x{:08x}u) >> 2u];\n",
            destination.index(),
            binding,
            dynamic_byte_offset.index(),
            *base_byte_offset as u32,
        )),
        ShaderOperation::SampleTexture2D {
            outputs,
            coordinates,
            image_binding,
            sampler_binding,
        } => {
            let sample = format!("nixe_sample_{:x}", instruction.source.byte_offset());
            source.push_str(&format!(
                "  let {sample} = textureSample(sampled_image_{image_binding}, sampler_{sampler_binding}, vec2<f32>(bitcast<f32>(registers[{}]), bitcast<f32>(registers[{}])));\n",
                coordinates[0].index(),
                coordinates[1].index(),
            ));
            for output in outputs {
                source.push_str(&format!(
                    "  registers[{}] = bitcast<u32>({sample}{});\n",
                    output.destination().index(),
                    wgsl_component(output.component()),
                ));
            }
        }
        ShaderOperation::SampleTexture2DArray {
            outputs,
            coordinates,
            array_index,
            image_binding,
            sampler_binding,
        } => {
            let sample = format!("nixe_sample_{:x}", instruction.source.byte_offset());
            source.push_str(&format!(
                "  let {sample} = textureSample(sampled_image_{image_binding}, sampler_{sampler_binding}, vec2<f32>(bitcast<f32>(registers[{}]), bitcast<f32>(registers[{}])), i32(registers[{}] & 0xffffu));\n",
                coordinates[0].index(),
                coordinates[1].index(),
                array_index.index(),
            ));
            for output in outputs {
                source.push_str(&format!(
                    "  registers[{}] = bitcast<u32>({sample}{});\n",
                    output.destination().index(),
                    wgsl_component(output.component()),
                ));
            }
        }
        ShaderOperation::Branch { .. } => {
            return Err(ShaderBackendLoweringError::ControlFlow(instruction.source));
        }
        ShaderOperation::Exit => source.push_str("  return output;\n"),
    }
    Ok(())
}

fn wgsl_predicate_expression(predicate: ShaderPredicate) -> String {
    match predicate {
        ShaderPredicate::Always => "true".to_owned(),
        ShaderPredicate::Never => "false".to_owned(),
        ShaderPredicate::Register { register, inverted } => {
            format!("{}predicates[{register}]", if inverted { "!" } else { "" })
        }
    }
}

fn wgsl_float_multiply_expression(
    source: ShaderSourceLocation,
    left: u16,
    right: u16,
    control: ShaderFloatControl,
) -> Result<String, ShaderBackendLoweringError> {
    if control.rounding() != ShaderRoundingMode::NearestEven
        || control.nan_mode() != ShaderNanMode::Propagate
        || control.saturate()
    {
        return Err(ShaderBackendLoweringError::NumericControl(source));
    }
    let operand = |register: u16| {
        if control.denormals_are_zero() {
            format!("nixe_flush_denormal(registers[{register}])")
        } else {
            format!("registers[{register}]")
        }
    };
    let result = format!(
        "bitcast<u32>(bitcast<f32>({}) * bitcast<f32>({}))",
        operand(left),
        operand(right)
    );
    Ok(if control.flush_denormals_to_zero() {
        format!("nixe_flush_denormal({result})")
    } else {
        result
    })
}

fn wgsl_float_add_expression(
    source: ShaderSourceLocation,
    left: u16,
    right: u16,
    control: ShaderFloatControl,
) -> Result<String, ShaderBackendLoweringError> {
    if control.rounding() != ShaderRoundingMode::NearestEven
        || control.nan_mode() != ShaderNanMode::Propagate
        || control.saturate()
    {
        return Err(ShaderBackendLoweringError::NumericControl(source));
    }
    let operand = |register: u16| {
        if control.denormals_are_zero() {
            format!("nixe_flush_denormal(registers[{register}])")
        } else {
            format!("registers[{register}]")
        }
    };
    let result = format!(
        "bitcast<u32>(bitcast<f32>({}) + bitcast<f32>({}))",
        operand(left),
        operand(right)
    );
    Ok(if control.flush_denormals_to_zero() {
        format!("nixe_flush_denormal({result})")
    } else {
        result
    })
}

fn wgsl_float_min_max_expression(
    source: ShaderSourceLocation,
    left: u16,
    right: u16,
    minimum: ShaderPredicate,
    control: ShaderFloatControl,
) -> Result<String, ShaderBackendLoweringError> {
    if control.rounding() != ShaderRoundingMode::NearestEven
        || control.nan_mode() != ShaderNanMode::Propagate
        || control.saturate()
    {
        return Err(ShaderBackendLoweringError::NumericControl(source));
    }
    let operand = |register: u16| {
        if control.denormals_are_zero() {
            format!("nixe_flush_denormal(registers[{register}])")
        } else {
            format!("registers[{register}]")
        }
    };
    let left = format!("bitcast<f32>({})", operand(left));
    let right = format!("bitcast<f32>({})", operand(right));
    let selected = format!(
        "select(max({left}, {right}), min({left}, {right}), {})",
        wgsl_predicate_expression(minimum)
    );
    let result = format!("bitcast<u32>({selected})");
    Ok(if control.flush_denormals_to_zero() {
        format!("nixe_flush_denormal({result})")
    } else {
        result
    })
}

fn wgsl_float_fma_expression(
    source: ShaderSourceLocation,
    left: u16,
    right: u16,
    addend: u16,
    control: ShaderFloatControl,
) -> Result<String, ShaderBackendLoweringError> {
    if control.rounding() != ShaderRoundingMode::NearestEven
        || control.nan_mode() != ShaderNanMode::Propagate
        || control.saturate()
    {
        return Err(ShaderBackendLoweringError::NumericControl(source));
    }
    let operand = |register: u16| {
        if control.denormals_are_zero() {
            format!("nixe_flush_denormal(registers[{register}])")
        } else {
            format!("registers[{register}]")
        }
    };
    let result = format!(
        "bitcast<u32>(fma(bitcast<f32>({}), bitcast<f32>({}), bitcast<f32>({})))",
        operand(left),
        operand(right),
        operand(addend)
    );
    Ok(if control.flush_denormals_to_zero() {
        format!("nixe_flush_denormal({result})")
    } else {
        result
    })
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

fn wgsl_interface_component(location: ShaderIoLocation, component: u8) -> &'static str {
    if matches!(
        location,
        ShaderIoLocation::Position | ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_)
    ) {
        wgsl_component(component)
    } else {
        debug_assert_eq!(component, 0, "scalar interface location component");
        ""
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
    UndefinedPredicate {
        source: ShaderSourceLocation,
        register: u8,
    },
    InvalidPredicateRegister {
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
    InvalidTextureSample {
        source: ShaderSourceLocation,
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
            Self::UndefinedPredicate { source, register } => write!(
                formatter,
                "shader IR reads undefined predicate p{register} at byte offset 0x{:x}",
                source.byte_offset()
            ),
            Self::InvalidPredicateRegister { source, register } => write!(
                formatter,
                "shader IR uses invalid predicate p{register} at byte offset 0x{:x}",
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
            Self::InvalidTextureSample { source } => write!(
                formatter,
                "shader IR has an invalid texture sample at byte offset 0x{:x}",
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
        } else if !input && stage == ShaderStage::Vertex {
            match element.location {
                // Frontends may leave an unconsumed output unlinked (`None`)
                // or attach the interpolation contract selected by the next
                // graphics stage.
                ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_) => true,
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
    let locations = shader_instruction_entry_points(ir);
    let mut incoming = vec![None::<ShaderDefinitions>; ir.instructions.len()];
    incoming[0] = Some(ShaderDefinitions::default());
    let mut work = std::collections::VecDeque::from([0_usize]);
    let mut has_reachable_exit = false;
    while let Some(index) = work.pop_front() {
        let instruction = &ir.instructions[index];
        let mut definitions = incoming[index]
            .clone()
            .expect("queued instruction has incoming definitions");
        let conditional = matches!(instruction.predicate, ShaderPredicate::Register { .. });
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
                validate_predicate_register(instruction.source, register)?;
                require_predicate_definition(instruction.source, register, &definitions.predicates)?
            }
            ShaderPredicate::Always => {}
        }
        match &instruction.operation {
            ShaderOperation::Undefined32 { destination }
            | ShaderOperation::MoveImmediate32 { destination, .. }
            | ShaderOperation::Move32 { destination, .. }
            | ShaderOperation::FloatAbsolute32 { destination, .. }
            | ShaderOperation::FloatNegate32 { destination, .. }
            | ShaderOperation::ConvertIntegerToFloat32 { destination, .. }
            | ShaderOperation::RoundFloat32ToIntegral { destination, .. }
            | ShaderOperation::ConvertFloat32ToInteger { destination, .. }
            | ShaderOperation::LoadConstantBuffer32 { destination, .. }
            | ShaderOperation::LoadConstantBufferIndexed32 { destination, .. }
            | ShaderOperation::Reciprocal32 { destination, .. }
            | ShaderOperation::ReciprocalSqrt32 { destination, .. }
            | ShaderOperation::SpecialFunction32 { destination, .. }
            | ShaderOperation::Multiply32 { destination, .. }
            | ShaderOperation::Add32 { destination, .. }
            | ShaderOperation::ShiftLeft32 { destination, .. }
            | ShaderOperation::FusedMultiplyAdd32 { destination, .. }
            | ShaderOperation::InterpolateInput { destination, .. } => {
                for source in operation_sources(&instruction.operation) {
                    require_definition(instruction.source, source, &definitions.registers)?;
                }
                if !conditional {
                    definitions.registers.insert(*destination);
                }
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
                if !conditional {
                    definitions.registers.extend(destinations.iter().copied());
                }
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
                    require_definition(instruction.source, source, &definitions.registers)?;
                }
            }
            ShaderOperation::FloatMinMax32 {
                destination,
                minimum,
                ..
            } => {
                for source in operation_sources(&instruction.operation) {
                    require_definition(instruction.source, source, &definitions.registers)?;
                }
                if let ShaderPredicate::Register { register, .. } = minimum {
                    validate_predicate_register(instruction.source, *register)?;
                    require_predicate_definition(
                        instruction.source,
                        *register,
                        &definitions.predicates,
                    )?;
                }
                if !conditional {
                    definitions.registers.insert(*destination);
                }
            }
            ShaderOperation::SetPredicateFloat32 {
                destination,
                accumulator,
                ..
            } => {
                validate_predicate_register(instruction.source, *destination)?;
                for source in operation_sources(&instruction.operation) {
                    require_definition(instruction.source, source, &definitions.registers)?;
                }
                if let ShaderPredicate::Register { register, .. } = accumulator {
                    validate_predicate_register(instruction.source, *register)?;
                    require_predicate_definition(
                        instruction.source,
                        *register,
                        &definitions.predicates,
                    )?;
                }
                if !conditional {
                    definitions.predicates.insert(*destination);
                }
            }
            ShaderOperation::SampleTexture2D {
                outputs,
                coordinates,
                image_binding,
                sampler_binding,
            } => {
                for coordinate in coordinates {
                    require_definition(instruction.source, *coordinate, &definitions.registers)?;
                }
                let valid_outputs = !outputs.is_empty()
                    && outputs.len() <= 4
                    && outputs
                        .iter()
                        .map(|output| output.component())
                        .collect::<BTreeSet<_>>()
                        .len()
                        == outputs.len()
                    && outputs
                        .iter()
                        .map(|output| output.destination())
                        .collect::<BTreeSet<_>>()
                        .len()
                        == outputs.len();
                if ir.stage != ShaderStage::Fragment || !valid_outputs {
                    return Err(ShaderVerificationError::InvalidTextureSample {
                        source: instruction.source,
                    });
                }
                for (binding, kind) in [
                    (*image_binding, ShaderResourceKind::SampledImage),
                    (*sampler_binding, ShaderResourceKind::Sampler),
                ] {
                    if !ir.resources.iter().any(|resource| {
                        resource.binding == binding
                            && resource.kind == kind
                            && resource.readable
                            && !resource.writable
                    }) {
                        return Err(ShaderVerificationError::UndeclaredResourceAccess {
                            source: instruction.source,
                            binding,
                        });
                    }
                }
                if !conditional {
                    definitions
                        .registers
                        .extend(outputs.iter().map(|output| output.destination()));
                }
            }
            ShaderOperation::SampleTexture2DArray {
                outputs,
                coordinates,
                array_index,
                image_binding,
                sampler_binding,
            } => {
                for source in [coordinates[0], coordinates[1], *array_index] {
                    require_definition(instruction.source, source, &definitions.registers)?;
                }
                let valid_outputs = !outputs.is_empty()
                    && outputs.len() <= 4
                    && outputs
                        .iter()
                        .map(|output| output.component())
                        .collect::<BTreeSet<_>>()
                        .len()
                        == outputs.len()
                    && outputs
                        .iter()
                        .map(|output| output.destination())
                        .collect::<BTreeSet<_>>()
                        .len()
                        == outputs.len();
                if ir.stage != ShaderStage::Fragment || !valid_outputs {
                    return Err(ShaderVerificationError::InvalidTextureSample {
                        source: instruction.source,
                    });
                }
                for (binding, kind) in [
                    (*image_binding, ShaderResourceKind::SampledImage2DArray),
                    (*sampler_binding, ShaderResourceKind::Sampler),
                ] {
                    if !ir.resources.iter().any(|resource| {
                        resource.binding == binding
                            && resource.kind == kind
                            && resource.readable
                            && !resource.writable
                    }) {
                        return Err(ShaderVerificationError::UndeclaredResourceAccess {
                            source: instruction.source,
                            binding,
                        });
                    }
                }
                if !conditional {
                    definitions
                        .registers
                        .extend(outputs.iter().map(|output| output.destination()));
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
                if !conditional {
                    continue;
                }
            }
            ShaderOperation::Exit => {
                has_reachable_exit = true;
                if !conditional {
                    continue;
                }
            }
        }
        if let ShaderOperation::LoadConstantBuffer32 { binding, .. }
        | ShaderOperation::LoadConstantBufferIndexed32 { binding, .. } = instruction.operation
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

fn shader_instruction_entry_points(
    ir: &ShaderIr,
) -> std::collections::BTreeMap<ShaderSourceLocation, usize> {
    let mut locations = std::collections::BTreeMap::new();
    for (index, instruction) in ir.instructions.iter().enumerate() {
        // One guest instruction may expand into several adjacent neutral
        // operations with identical provenance. Guest control-flow targets
        // enter before the complete expansion, never in its middle.
        locations.entry(instruction.source).or_insert(index);
    }
    locations
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ShaderDefinitions {
    registers: BTreeSet<ShaderRegister>,
    predicates: BTreeSet<u8>,
}

fn enqueue_shader_successor(
    successor: Option<usize>,
    definitions: &ShaderDefinitions,
    incoming: &mut [Option<ShaderDefinitions>],
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
            let merged = ShaderDefinitions {
                registers: existing
                    .registers
                    .intersection(&definitions.registers)
                    .copied()
                    .collect(),
                predicates: existing
                    .predicates
                    .intersection(&definitions.predicates)
                    .copied()
                    .collect(),
            };
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
        ShaderOperation::Move32 { source, .. } => vec![*source],
        ShaderOperation::Add32 { left, right, .. } => vec![*left, *right],
        ShaderOperation::ShiftLeft32 { value, amount, .. } => vec![*value, *amount],
        ShaderOperation::FloatMinMax32 { left, right, .. } => vec![*left, *right],
        ShaderOperation::FloatAbsolute32 { source, .. }
        | ShaderOperation::FloatNegate32 { source, .. }
        | ShaderOperation::ConvertIntegerToFloat32 { source, .. }
        | ShaderOperation::RoundFloat32ToIntegral { source, .. } => vec![*source],
        ShaderOperation::ConvertFloat32ToInteger { source, .. } => vec![*source],
        ShaderOperation::FusedMultiplyAdd32 {
            left,
            right,
            addend,
            ..
        } => vec![*left, *right, *addend],
        ShaderOperation::Reciprocal32 { source, .. } => vec![*source],
        ShaderOperation::ReciprocalSqrt32 { source, .. } => vec![*source],
        ShaderOperation::SpecialFunction32 { source, .. } => vec![*source],
        ShaderOperation::SetPredicateFloat32 { left, right, .. } => vec![*left, *right],
        ShaderOperation::LoadConstantBufferIndexed32 {
            dynamic_byte_offset,
            ..
        } => vec![*dynamic_byte_offset],
        ShaderOperation::SampleTexture2D { coordinates, .. } => coordinates.to_vec(),
        ShaderOperation::SampleTexture2DArray {
            coordinates,
            array_index,
            ..
        } => vec![coordinates[0], coordinates[1], *array_index],
        _ => Vec::new(),
    }
}

fn validate_predicate_register(
    source: ShaderSourceLocation,
    predicate: u8,
) -> Result<(), ShaderVerificationError> {
    if predicate < 7 {
        Ok(())
    } else {
        Err(ShaderVerificationError::InvalidPredicateRegister {
            source,
            register: predicate,
        })
    }
}

fn require_predicate_definition(
    source: ShaderSourceLocation,
    predicate: u8,
    definitions: &BTreeSet<u8>,
) -> Result<(), ShaderVerificationError> {
    if definitions.contains(&predicate) {
        Ok(())
    } else {
        Err(ShaderVerificationError::UndefinedPredicate {
            source,
            register: predicate,
        })
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
        assert_eq!(
            ShaderInterfaceElement::new(
                ShaderIoLocation::VertexId,
                1,
                ShaderScalarType::Unsigned32,
                None,
            ),
            Err(ShaderIrConstructionError::InvalidInterfaceComponent {
                location: ShaderIoLocation::VertexId,
                component: 1,
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
    fn branch_control_flow_evaluates_and_lowers_to_valid_wgsl() {
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
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(1),
                        bits: 0,
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::SetPredicateFloat32 {
                        destination: 0,
                        left: ShaderRegister::new(1),
                        right: ShaderRegister::new(1),
                        comparison: ShaderFloatComparison::OrderedEqual,
                        accumulator: ShaderPredicate::Always,
                        set_operation: ShaderPredicateSetOperation::And,
                        flush_denormals_to_zero: false,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Register {
                        register: 0,
                        inverted: true,
                    },
                    ShaderOperation::Branch {
                        target: ShaderSourceLocation::new(48),
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(40),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 2,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(48),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(0)].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(56),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ))
        .unwrap();
        let evaluated =
            evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(
            evaluated.output_bits(ShaderIoLocation::Position, 0),
            Some(2)
        );

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("var nixe_block = 0u"));
        assert!(module.source().contains("nixe_block = 2u"));
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn branches_enter_the_first_neutral_operation_for_one_guest_instruction() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Unsigned32,
            None,
        )
        .unwrap();
        let expanded_source = ShaderSourceLocation::new(32);
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
                        bits: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::Branch {
                        target: expanded_source,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::Undefined32 {
                        destination: ShaderRegister::new(1),
                    },
                ),
                ShaderInstruction::new(
                    expanded_source,
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(1),
                        bits: 42,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    expanded_source,
                    ShaderPredicate::Always,
                    ShaderOperation::Move32 {
                        destination: ShaderRegister::new(2),
                        source: ShaderRegister::new(1),
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(40),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(2)].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(48),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ))
        .unwrap();

        let result = evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(result.output_bits(ShaderIoLocation::Position, 0), Some(42));
        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("registers[1] = 0x0000002au"));
        assert!(module.source().contains("registers[2] = registers[1]"));
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn float_predicate_writes_drive_predicated_evaluation_and_wgsl() {
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
                        bits: 0.0_f32.to_bits(),
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
                    ShaderOperation::SetPredicateFloat32 {
                        destination: 0,
                        left: ShaderRegister::new(0),
                        right: ShaderRegister::new(1),
                        comparison: ShaderFloatComparison::OrderedLess,
                        accumulator: ShaderPredicate::Always,
                        set_operation: ShaderPredicateSetOperation::And,
                        flush_denormals_to_zero: true,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Register {
                        register: 0,
                        inverted: false,
                    },
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(1)].into_boxed_slice(),
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

        let result = evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 8).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some(2.0_f32.to_bits())
        );
        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("predicates[0] ="));
        assert!(module.source().contains("if (predicates[0])"));
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn register_moves_preserve_bits_under_predication() {
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
                        bits: 0xdead_beef,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(1),
                        bits: 1.0_f32.to_bits(),
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(2),
                        bits: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Always,
                    ShaderOperation::SetPredicateFloat32 {
                        destination: 0,
                        left: ShaderRegister::new(1),
                        right: ShaderRegister::new(1),
                        comparison: ShaderFloatComparison::OrderedEqual,
                        accumulator: ShaderPredicate::Always,
                        set_operation: ShaderPredicateSetOperation::And,
                        flush_denormals_to_zero: false,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(40),
                    ShaderPredicate::Register {
                        register: 0,
                        inverted: false,
                    },
                    ShaderOperation::Move32 {
                        destination: ShaderRegister::new(2),
                        source: ShaderRegister::new(0),
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(48),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(2)].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(56),
                    ShaderPredicate::Always,
                    ShaderOperation::Exit,
                ),
            ],
        ))
        .unwrap();

        let result = evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 8).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some(0xdead_beef)
        );
        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("registers[2] = registers[0]"));
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn wgsl_uses_logical_perspective_input_without_reapplying_mul_w() {
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
                    ShaderOperation::InterpolateInput {
                        destination: ShaderRegister::new(1),
                        location: ShaderIoLocation::Generic(0),
                        component: 0,
                        interpolation: ShaderInterpolation::Perspective,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
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
                .contains("registers[1] = bitcast<u32>(input.generic_0.x)")
        );
        assert!(!module.source().contains("input.generic_0.x *"));
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
    fn fused_multiply_add_preserves_single_rounding_in_evaluator_and_wgsl() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let left = f32::from_bits(0x3f80_0001);
        let right = f32::from_bits(0x3f80_0001);
        let addend = f32::from_bits(0xbf80_0002);
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            Vec::new(),
            vec![output],
            Vec::new(),
            vec![(0, left), (1, right), (2, addend)]
                .into_iter()
                .map(|(register, value)| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(8 + register * 8),
                        ShaderPredicate::Always,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(register as u16),
                            bits: value.to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        },
                    )
                })
                .chain([
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(32),
                        ShaderPredicate::Always,
                        ShaderOperation::FusedMultiplyAdd32 {
                            destination: ShaderRegister::new(3),
                            left: ShaderRegister::new(0),
                            right: ShaderRegister::new(1),
                            addend: ShaderRegister::new(2),
                            float_control: ShaderFloatControl::PRECISE,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(40),
                        ShaderPredicate::Always,
                        ShaderOperation::StoreOutput {
                            sources: vec![ShaderRegister::new(3)].into_boxed_slice(),
                            location: ShaderIoLocation::Position,
                            first_component: 0,
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(48),
                        ShaderPredicate::Always,
                        ShaderOperation::Exit,
                    ),
                ])
                .collect(),
        ))
        .unwrap();
        let evaluated =
            evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16).unwrap();
        let fused = left.mul_add(right, addend);
        assert_ne!(fused.to_bits(), (left * right + addend).to_bits());
        assert_eq!(
            evaluated.output_bits(ShaderIoLocation::Position, 0),
            Some(fused.to_bits())
        );

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("bitcast<u32>(fma("));
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn float_add_flushes_a_subnormal_result_in_evaluator_and_wgsl() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let control = ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            true,
            false,
            false,
        );
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
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(1),
                        bits: 0,
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::Add32 {
                        destination: ShaderRegister::new(2),
                        left: ShaderRegister::new(0),
                        right: ShaderRegister::new(1),
                        scalar_type: ShaderScalarType::Float32,
                        float_control: control,
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
        let evaluated =
            evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(
            evaluated.output_bits(ShaderIoLocation::Position, 0),
            Some(0)
        );

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("nixe_flush_denormal(bitcast<u32>"));
        naga::front::wgsl::parse_str(module.source()).unwrap();
    }

    #[test]
    fn float_absolute_and_negate_preserve_payload_bits() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let bits = 0xffc1_2345;
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
                        bits,
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::FloatAbsolute32 {
                        destination: ShaderRegister::new(1),
                        source: ShaderRegister::new(0),
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::FloatNegate32 {
                        destination: ShaderRegister::new(2),
                        source: ShaderRegister::new(1),
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
        let evaluated =
            evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(
            evaluated.output_bits(ShaderIoLocation::Position, 0),
            Some(bits)
        );

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(module.source().contains("& 0x7fffffffu"));
        assert!(module.source().contains("^ 0x80000000u"));
        naga::front::wgsl::parse_str(module.source()).unwrap();
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
        let exact = make_shader(ShaderMathAccuracy::Exact);
        let result = evaluate_shader_ir(&exact, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some(0.0_f32.to_bits())
        );

        let approximate = make_shader(ShaderMathAccuracy::Approximate);
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

    #[test]
    fn evaluator_distinguishes_exact_and_guest_approximate_reciprocal_sqrt() {
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
                            bits: 4.0_f32.to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::ReciprocalSqrt32 {
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
        let exact = make_shader(ShaderMathAccuracy::Exact);
        let result = evaluate_shader_ir(&exact, &ShaderEvaluationInputs::default(), 16).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some(0.5_f32.to_bits())
        );
        let module = lower_shader_ir_to_wgsl(&exact).unwrap();
        assert!(module.source().contains("inverseSqrt"));
        naga::front::wgsl::parse_str(module.source()).unwrap();

        let approximate = make_shader(ShaderMathAccuracy::Approximate);
        assert_eq!(
            evaluate_shader_ir(&approximate, &ShaderEvaluationInputs::default(), 16),
            Err(ShaderEvaluationError::ApproximateOperation(
                ShaderSourceLocation::new(16)
            ))
        );
    }

    #[test]
    fn scalar_special_functions_evaluate_exactly_and_lower_to_valid_wgsl() {
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let make_shader = |function, accuracy| {
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
                            bits: 4.0_f32.to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::SpecialFunction32 {
                            destination: ShaderRegister::new(1),
                            source: ShaderRegister::new(0),
                            function,
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

        let exact = make_shader(ShaderSpecialFunction::SquareRoot, ShaderMathAccuracy::Exact);
        let result = evaluate_shader_ir(&exact, &ShaderEvaluationInputs::default(), 8).unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Position, 0),
            Some(2.0_f32.to_bits())
        );

        for function in [
            ShaderSpecialFunction::Cosine,
            ShaderSpecialFunction::Sine,
            ShaderSpecialFunction::Exp2,
            ShaderSpecialFunction::Log2,
            ShaderSpecialFunction::SquareRoot,
        ] {
            let shader = make_shader(function, ShaderMathAccuracy::Approximate);
            assert_eq!(
                evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 8),
                Err(ShaderEvaluationError::ApproximateOperation(
                    ShaderSourceLocation::new(16)
                ))
            );
            let module = lower_shader_ir_to_wgsl(&shader).unwrap();
            naga::front::wgsl::parse_str(module.source()).unwrap();
        }
    }

    #[test]
    fn texture_sample_2d_verifies_lowers_to_naga_and_stays_out_of_scalar_evaluator() {
        let source = ShaderSourceLocation::new(24);
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            Vec::new(),
            (0..4)
                .map(|component| {
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        component,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap()
                })
                .collect(),
            vec![
                ShaderResourceAccess::new(32, ShaderResourceKind::SampledImage, true, false)
                    .unwrap(),
                ShaderResourceAccess::new(33, ShaderResourceKind::Sampler, true, false).unwrap(),
            ],
            vec![
                ShaderInstruction::new(
                    ShaderSourceLocation::new(8),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 0.25_f32.to_bits(),
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(1),
                        bits: 0.75_f32.to_bits(),
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    source,
                    ShaderPredicate::Always,
                    ShaderOperation::SampleTexture2D {
                        outputs: (0..4)
                            .map(|component| {
                                ShaderTextureSampleOutput::new(
                                    ShaderRegister::new(2 + u16::from(component)),
                                    component,
                                )
                                .unwrap()
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        coordinates: [ShaderRegister::new(0), ShaderRegister::new(1)],
                        image_binding: 32,
                        sampler_binding: 33,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(32),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: (2..6).map(ShaderRegister::new).collect(),
                        location: ShaderIoLocation::Color(0),
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

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(
            module
                .source()
                .contains("textureSample(sampled_image_32, sampler_33")
        );
        let naga_module = naga::front::wgsl::parse_str(module.source()).unwrap();
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&naga_module).unwrap();
        assert_eq!(
            evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 8),
            Err(ShaderEvaluationError::TextureSampling(source))
        );
    }

    #[test]
    fn float_min_max_selects_minimum_or_maximum_and_lowers_to_valid_wgsl() {
        let make_shader = |minimum| {
            VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Vertex,
                Vec::new(),
                vec![
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Position,
                        0,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap(),
                ],
                Vec::new(),
                vec![
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(8),
                        ShaderPredicate::Always,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(0),
                            bits: (-2.0_f32).to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(1),
                            bits: 3.0_f32.to_bits(),
                            scalar_type: ShaderScalarType::Float32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(24),
                        ShaderPredicate::Always,
                        ShaderOperation::FloatMinMax32 {
                            destination: ShaderRegister::new(2),
                            left: ShaderRegister::new(0),
                            right: ShaderRegister::new(1),
                            minimum,
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
            .unwrap()
        };

        for (minimum, expected) in [
            (ShaderPredicate::Always, -2.0_f32),
            (ShaderPredicate::Never, 3.0_f32),
        ] {
            let shader = make_shader(minimum);
            assert_eq!(
                evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 8)
                    .unwrap()
                    .output_bits(ShaderIoLocation::Position, 0),
                Some(expected.to_bits())
            );
            let module = lower_shader_ir_to_wgsl(&shader).unwrap();
            assert!(module.source().contains("select(max("));
            let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&parsed)
            .unwrap();
        }
    }

    #[test]
    fn shift_left_preserves_clamped_and_wrapped_maxwell_count_semantics() {
        let make_shader = |amount, wrap| {
            VerifiedShaderIr::verify(ShaderIr::new(
                ShaderStage::Fragment,
                Vec::new(),
                vec![
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Color(0),
                        0,
                        ShaderScalarType::Unsigned32,
                        None,
                    )
                    .unwrap(),
                ],
                Vec::new(),
                vec![
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(8),
                        ShaderPredicate::Always,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(0),
                            bits: 3,
                            scalar_type: ShaderScalarType::Unsigned32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(16),
                        ShaderPredicate::Always,
                        ShaderOperation::MoveImmediate32 {
                            destination: ShaderRegister::new(1),
                            bits: amount,
                            scalar_type: ShaderScalarType::Unsigned32,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(24),
                        ShaderPredicate::Always,
                        ShaderOperation::ShiftLeft32 {
                            destination: ShaderRegister::new(2),
                            value: ShaderRegister::new(0),
                            amount: ShaderRegister::new(1),
                            wrap,
                        },
                    ),
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(32),
                        ShaderPredicate::Always,
                        ShaderOperation::StoreOutput {
                            sources: vec![ShaderRegister::new(2)].into_boxed_slice(),
                            location: ShaderIoLocation::Color(0),
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
            .unwrap()
        };

        for (amount, wrap, expected) in [(4, false, 48), (36, false, 0), (36, true, 48)] {
            let shader = make_shader(amount, wrap);
            assert_eq!(
                evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 8)
                    .unwrap()
                    .output_bits(ShaderIoLocation::Color(0), 0),
                Some(expected)
            );
            let module = lower_shader_ir_to_wgsl(&shader).unwrap();
            let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&parsed)
            .unwrap();
        }
    }

    #[test]
    fn indexed_constant_buffer_load_uses_wrapping_byte_addressing() {
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            Vec::new(),
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Color(0),
                    0,
                    ShaderScalarType::Unsigned32,
                    None,
                )
                .unwrap(),
            ],
            vec![
                ShaderResourceAccess::new(1, ShaderResourceKind::ConstantBuffer, true, false)
                    .unwrap(),
            ],
            vec![
                ShaderInstruction::new(
                    ShaderSourceLocation::new(8),
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: ShaderRegister::new(0),
                        bits: 0x40,
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(16),
                    ShaderPredicate::Always,
                    ShaderOperation::LoadConstantBufferIndexed32 {
                        destination: ShaderRegister::new(1),
                        binding: 1,
                        base_byte_offset: -0x10,
                        dynamic_byte_offset: ShaderRegister::new(0),
                        scalar_type: ShaderScalarType::Unsigned32,
                    },
                ),
                ShaderInstruction::new(
                    ShaderSourceLocation::new(24),
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![ShaderRegister::new(1)].into_boxed_slice(),
                        location: ShaderIoLocation::Color(0),
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
        ))
        .unwrap();
        let result = evaluate_shader_ir(
            &shader,
            &ShaderEvaluationInputs::default().with_constant_buffer_bits(1, 0x30, 0x1234_5678),
            8,
        )
        .unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::Color(0), 0),
            Some(0x1234_5678)
        );

        let module = lower_shader_ir_to_wgsl(&shader).unwrap();
        assert!(
            module
                .source()
                .contains("constant_buffer_1[(registers[0] + 0xfffffff0u) >> 2u]")
        );
        let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .unwrap();
    }
}
