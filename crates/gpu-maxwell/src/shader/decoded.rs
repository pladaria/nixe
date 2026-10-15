//! Maxwell decoder/legalizer instruction protocol. Mutable ISA names exist
//! only here, before shared SSA construction. Native compilation never consumes
//! the oracle's ShaderIr/VerifiedShaderIr. The export conversion below belongs
//! exclusively to the retiring renderer and is deleted with that consumer.
use nixe_gpu::*;

// Hardware GPR fields are checked as eight-bit Maxwell names; legalization
// temporaries use an independent, dense u32 namespace with no old-IR limit.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct Register(u32);
impl Register {
    pub(super) const fn new(index: u32) -> Self {
        Self(index)
    }
    pub(super) const fn index(self) -> u32 {
        self.0
    }
    fn oracle(self) -> nixe_gpu::ShaderRegister {
        nixe_gpu::ShaderRegister::new(
            u16::try_from(self.0).expect("oracle temporary width was checked before export"),
        )
    }
}
#[cfg(test)]
impl From<nixe_gpu::ShaderRegister> for Register {
    fn from(value: nixe_gpu::ShaderRegister) -> Self {
        Self(u32::from(value.index()))
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TextureOutput {
    destination: Register,
    component: u8,
}
impl TextureOutput {
    pub(super) fn new(
        destination: Register,
        component: u8,
    ) -> Result<Self, nixe_gpu::ShaderIrConstructionError> {
        if component > 3 {
            return Err(nixe_gpu::ShaderIrConstructionError::InvalidTextureComponent { component });
        }
        Ok(Self {
            destination,
            component,
        })
    }
    pub(super) fn destination(self) -> Register {
        self.destination
    }
    pub(super) fn component(self) -> u8 {
        self.component
    }
    fn oracle(self) -> nixe_gpu::ShaderTextureSampleOutput {
        nixe_gpu::ShaderTextureSampleOutput::new(self.destination.oracle(), self.component)
            .expect("decoded texture component is bounded")
    }
}
#[cfg(test)]
impl From<nixe_gpu::ShaderTextureSampleOutput> for TextureOutput {
    fn from(value: nixe_gpu::ShaderTextureSampleOutput) -> Self {
        Self {
            destination: value.destination().into(),
            component: value.component(),
        }
    }
}
use Register as ShaderRegister;
use TextureOutput as ShaderTextureSampleOutput;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Operation {
    /// Retains an instruction location with no per-invocation data effect.
    Nop,
    QuadShuffle32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        operation: ShaderQuadOperation,
    },
    FloatQuadSwizzleAdd32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        swizzle: u8,
        float_control: ShaderFloatControl,
    },
    LoadComputeBuiltin32 {
        destination: ShaderRegister,
        builtin: ShaderComputeBuiltin,
        component: u8,
    },
    /// Word-aligned storage access; the index is an unsigned 32-bit word index.
    LoadStorageBuffer32 {
        destination: ShaderRegister,
        binding: u32,
        word_index: ShaderRegister,
    },
    StoreStorageBuffer32 {
        source: ShaderRegister,
        binding: u32,
        word_index: ShaderRegister,
    },
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
    /// Convert the selected IEEE binary16 lane to binary32, preserving subnormals.
    UnpackHalf32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        high: bool,
    },
    /// Round binary32 to IEEE binary16, ties to even, in the low sixteen bits.
    PackHalf32 {
        destination: ShaderRegister,
        source: ShaderRegister,
    },
    /// Fused binary16 arithmetic. Operands are exact binary16 values represented
    /// as F32 register bits; the low 16 result bits contain the rounded half.
    FusedMultiplyAddHalf32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        addend: ShaderRegister,
        saturate: bool,
    },
    ConvertIntegerToFloat32 {
        destination: ShaderRegister,
        source: ShaderRegister,
        source_type: ShaderScalarType,
        rounding: ShaderRoundingMode,
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
    /// Arrayed per-vertex I/O. `output` selects a TCS output read; otherwise
    /// this reads a TCS/TES input. Vertex indices are unsigned register values.
    LoadControlPoint {
        destination: ShaderRegister,
        vertex: ShaderRegister,
        output: bool,
        location: ShaderIoLocation,
        component: u8,
    },
    StoreControlPoint {
        source: ShaderRegister,
        vertex: ShaderRegister,
        location: ShaderIoLocation,
        component: u8,
    },
    /// TCS per-patch output read, including tessellation levels.
    LoadPatchOutput {
        destination: ShaderRegister,
        location: ShaderIoLocation,
        component: u8,
    },
    /// Rendezvous of all control-stage invocations in one patch. Orders output
    /// memory writes before subsequent output reads in those invocations.
    /// https://docs.vulkan.org/spec/latest/chapters/tessellation.html
    PatchBarrier,
    Multiply32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        scalar_type: ShaderScalarType,
        float_control: ShaderFloatControl,
    },
    /// Floating multiplication where either signed zero absorbs every operand
    /// (including NaN/infinity), producing positive zero. Input denormal
    /// handling precedes this test; output controls follow multiplication.
    /// https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/compiler/nir/nir_opcodes.py
    FloatMultiplyZero32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        float_control: ShaderFloatControl,
    },
    Add32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        scalar_type: ShaderScalarType,
        float_control: ShaderFloatControl,
    },
    /// Unsigned addition with an optional low-bit carry input and a 0/1
    /// carry output. Both results consume the original operands, even when
    /// registers alias. This does not represent ISA-specific flag registers.
    AddCarry32 {
        destination: ShaderRegister,
        carry_out: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        carry_in: Option<ShaderRegister>,
    },
    ShiftLeft32 {
        destination: ShaderRegister,
        value: ShaderRegister,
        amount: ShaderRegister,
        wrap: bool,
    },
    ShiftRight32 {
        destination: ShaderRegister,
        value: ShaderRegister,
        amount: ShaderRegister,
        wrap: bool,
        signed: bool,
        reverse: bool,
    },
    /// Extracts a field, extending the source sign bit when a signed field
    /// extends beyond bit 31. A zero count always returns zero.
    ExtractBitfield32 {
        destination: ShaderRegister,
        value: ShaderRegister,
        offset: ShaderRegister,
        count: ShaderRegister,
        signed: bool,
        reverse: bool,
    },
    Bitwise32 {
        destination: ShaderRegister,
        left: ShaderRegister,
        right: ShaderRegister,
        operation: ShaderBitwiseOperation,
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
    /// Select raw register bits using an integer condition (zero is false).
    Select32 {
        destination: ShaderRegister,
        condition: ShaderRegister,
        when_true: ShaderRegister,
        when_false: ShaderRegister,
    },
    CompareInteger32 {
        destination: ShaderRegister,
        float_result: bool,
        left: ShaderRegister,
        right: ShaderRegister,
        signed: bool,
        comparison: ShaderIntegerComparison,
        accumulator: ShaderPredicate,
        set_operation: ShaderPredicateSetOperation,
    },
    CompareFloat32 {
        destination: ShaderRegister,
        float_result: bool,
        left: ShaderRegister,
        right: ShaderRegister,
        comparison: ShaderFloatComparison,
        accumulator: ShaderPredicate,
        set_operation: ShaderPredicateSetOperation,
        flush_denormals_to_zero: bool,
    },
    /// Set selected predicates from a contiguous range of integer bits.
    SetPredicatesFromBits {
        source: ShaderRegister,
        first_bit: u8,
        write_mask: u8,
    },
    /// Snapshot all inputs before writing either result. The second result
    /// complements the first input, before either boolean combination.
    SetPredicateBoolean {
        destinations: [Option<u8>; 2],
        inputs: [ShaderPredicate; 3],
        operations: [ShaderPredicateSetOperation; 2],
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
    /// Both results use the old accumulator value. The second comparison is
    /// complemented before combining with the accumulator, not afterward.
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions-setp
    SetPredicateInteger32 {
        destinations: [Option<u8>; 2],
        left: ShaderRegister,
        right: ShaderRegister,
        signed: bool,
        comparison: ShaderIntegerComparison,
        accumulator: ShaderPredicate,
        set_operation: ShaderPredicateSetOperation,
    },
    InterpolateInput {
        destination: ShaderRegister,
        location: ShaderIoLocation,
        component: u8,
        interpolation: ShaderInterpolation,
    },
    LoadConstantBuffer32 {
        destination: ShaderRegister,
        binding: u32,
        byte_offset: u32,
        scalar_type: ShaderScalarType,
    },
    LoadConstantBufferIndexed32 {
        destination: ShaderRegister,
        binding: u32,
        base_byte_offset: i32,
        dynamic_byte_offset: ShaderRegister,
        scalar_type: ShaderScalarType,
    },
    SampleDepthCompare2D {
        offset: Option<[i32; 2]>,
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 2],
        array_index: Option<ShaderRegister>,
        reference: ShaderRegister,
        image_binding: u32,
        sampler_binding: u32,
        level_zero: bool,
    },
    SampleTexture2D {
        offset: Option<[i32; 2]>,
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 2],
        image_binding: u32,
        lod: Option<ShaderRegister>,
        sampler_binding: u32,
    },
    /// Unfiltered texel read at integer coordinates and a fixed mip level.
    LoadTexture2D {
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 2],
        image_binding: u32,
        mip_level: u32,
    },
    QueryTextureLevels {
        destination: ShaderRegister,
        image_binding: u32,
    },
    SampleTexture2DArray {
        offset: Option<[i32; 2]>,
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 2],
        /// Register whose low 16 bits hold the unsigned Maxwell array layer.
        array_index: ShaderRegister,
        image_binding: u32,
        lod: Option<ShaderRegister>,
        sampler_binding: u32,
    },
    SampleTextureCube {
        outputs: Box<[ShaderTextureSampleOutput]>,
        coordinates: [ShaderRegister; 3],
        image_binding: u32,
        lod: Option<ShaderRegister>,
        sampler_binding: u32,
    },
    Branch {
        target: ShaderSourceLocation,
    },
    BranchTable {
        value: ShaderRegister,
        targets: Box<[(u32, ShaderSourceLocation)]>,
    },
    /// Terminates a fragment invocation without attachment output.
    Discard,
    Exit,
}

fn visit_operation_destinations(operation: &Operation, mut visit: impl FnMut(ShaderRegister)) {
    match operation {
        Operation::AddCarry32 {
            destination,
            carry_out,
            ..
        } => {
            visit(*destination);
            visit(*carry_out);
        }
        Operation::FloatQuadSwizzleAdd32 { destination, .. }
        | Operation::QuadShuffle32 { destination, .. }
        | Operation::Undefined32 { destination }
        | Operation::LoadComputeBuiltin32 { destination, .. }
        | Operation::LoadStorageBuffer32 { destination, .. }
        | Operation::MoveImmediate32 { destination, .. }
        | Operation::Move32 { destination, .. }
        | Operation::Select32 { destination, .. }
        | Operation::FloatAbsolute32 { destination, .. }
        | Operation::FloatNegate32 { destination, .. }
        | Operation::UnpackHalf32 { destination, .. }
        | Operation::PackHalf32 { destination, .. }
        | Operation::FusedMultiplyAddHalf32 { destination, .. }
        | Operation::QueryTextureLevels { destination, .. }
        | Operation::ConvertIntegerToFloat32 { destination, .. }
        | Operation::RoundFloat32ToIntegral { destination, .. }
        | Operation::ConvertFloat32ToInteger { destination, .. }
        | Operation::LoadControlPoint { destination, .. }
        | Operation::LoadPatchOutput { destination, .. }
        | Operation::Multiply32 { destination, .. }
        | Operation::FloatMultiplyZero32 { destination, .. }
        | Operation::Add32 { destination, .. }
        | Operation::ShiftLeft32 { destination, .. }
        | Operation::ShiftRight32 { destination, .. }
        | Operation::ExtractBitfield32 { destination, .. }
        | Operation::Bitwise32 { destination, .. }
        | Operation::CompareInteger32 { destination, .. }
        | Operation::CompareFloat32 { destination, .. }
        | Operation::FloatMinMax32 { destination, .. }
        | Operation::FusedMultiplyAdd32 { destination, .. }
        | Operation::Reciprocal32 { destination, .. }
        | Operation::ReciprocalSqrt32 { destination, .. }
        | Operation::SpecialFunction32 { destination, .. }
        | Operation::InterpolateInput { destination, .. }
        | Operation::LoadConstantBuffer32 { destination, .. }
        | Operation::LoadConstantBufferIndexed32 { destination, .. } => visit(*destination),
        Operation::LoadInput { destinations, .. } => destinations.iter().copied().for_each(visit),
        Operation::SampleDepthCompare2D { outputs, .. }
        | Operation::SampleTexture2D { outputs, .. }
        | Operation::LoadTexture2D { outputs, .. }
        | Operation::SampleTexture2DArray { outputs, .. }
        | Operation::SampleTextureCube { outputs, .. } => outputs
            .iter()
            .map(|output| output.destination())
            .for_each(visit),
        _ => {}
    }
}
fn operation_sources(operation: &Operation) -> Vec<ShaderRegister> {
    match operation {
        Operation::Select32 {
            condition,
            when_true,
            when_false,
            ..
        } => vec![*condition, *when_true, *when_false],
        Operation::BranchTable { value, .. } => vec![*value],
        Operation::StoreOutput { sources, .. } => sources.to_vec(),
        Operation::LoadStorageBuffer32 { word_index, .. } => vec![*word_index],
        Operation::StoreStorageBuffer32 {
            source, word_index, ..
        } => vec![*source, *word_index],
        Operation::LoadControlPoint { vertex, .. } => vec![*vertex],
        Operation::StoreControlPoint { source, vertex, .. } => vec![*source, *vertex],
        Operation::FloatQuadSwizzleAdd32 { left, right, .. }
        | Operation::Multiply32 { left, right, .. }
        | Operation::FloatMultiplyZero32 { left, right, .. } => vec![*left, *right],
        Operation::SetPredicatesFromBits { source, .. }
        | Operation::QuadShuffle32 { source, .. }
        | Operation::Move32 { source, .. } => {
            vec![*source]
        }
        Operation::Add32 { left, right, .. } => vec![*left, *right],
        Operation::AddCarry32 {
            left,
            right,
            carry_in,
            ..
        } => {
            let mut sources = vec![*left, *right];
            sources.extend(carry_in);
            sources
        }
        Operation::Bitwise32 { left, right, .. } => vec![*left, *right],
        Operation::ExtractBitfield32 {
            value,
            offset,
            count,
            ..
        } => vec![*value, *offset, *count],
        Operation::ShiftLeft32 { value, amount, .. }
        | Operation::ShiftRight32 { value, amount, .. } => vec![*value, *amount],
        Operation::FloatMinMax32 { left, right, .. } => vec![*left, *right],
        Operation::FloatAbsolute32 { source, .. }
        | Operation::FloatNegate32 { source, .. }
        | Operation::UnpackHalf32 { source, .. }
        | Operation::PackHalf32 { source, .. }
        | Operation::ConvertIntegerToFloat32 { source, .. }
        | Operation::RoundFloat32ToIntegral { source, .. } => vec![*source],
        Operation::ConvertFloat32ToInteger { source, .. } => vec![*source],
        Operation::FusedMultiplyAdd32 {
            left,
            right,
            addend,
            ..
        }
        | Operation::FusedMultiplyAddHalf32 {
            left,
            right,
            addend,
            ..
        } => vec![*left, *right, *addend],
        Operation::Reciprocal32 { source, .. } => vec![*source],
        Operation::ReciprocalSqrt32 { source, .. } => vec![*source],
        Operation::SpecialFunction32 { source, .. } => vec![*source],
        Operation::CompareInteger32 { left, right, .. }
        | Operation::CompareFloat32 { left, right, .. }
        | Operation::SetPredicateFloat32 { left, right, .. }
        | Operation::SetPredicateInteger32 { left, right, .. } => vec![*left, *right],
        Operation::LoadConstantBufferIndexed32 {
            dynamic_byte_offset,
            ..
        } => vec![*dynamic_byte_offset],
        Operation::SampleDepthCompare2D {
            coordinates,
            array_index,
            reference,
            ..
        } => {
            let mut operands = coordinates.to_vec();
            operands.extend(array_index);
            operands.push(*reference);
            operands
        }
        Operation::SampleTexture2D {
            coordinates, lod, ..
        } => coordinates
            .iter()
            .copied()
            .chain(lod.iter().copied())
            .collect(),
        Operation::LoadTexture2D { coordinates, .. } => coordinates.to_vec(),
        Operation::SampleTexture2DArray {
            coordinates,
            array_index,
            lod,
            ..
        } => coordinates
            .iter()
            .copied()
            .chain([*array_index])
            .chain(lod.iter().copied())
            .collect(),
        Operation::SampleTextureCube {
            coordinates, lod, ..
        } => coordinates
            .iter()
            .copied()
            .chain(lod.iter().copied())
            .collect(),
        _ => Vec::new(),
    }
}

impl Operation {
    pub(super) fn source_registers(&self) -> Vec<ShaderRegister> {
        operation_sources(self)
    }
    pub(super) fn visit_destination_registers(&self, visit: impl FnMut(ShaderRegister)) {
        visit_operation_destinations(self, visit);
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Instruction {
    source: ShaderSourceLocation,
    predicate: ShaderPredicate,
    operation: Operation,
}
impl Instruction {
    pub(super) const fn new(
        source: ShaderSourceLocation,
        predicate: ShaderPredicate,
        operation: Operation,
    ) -> Self {
        Self {
            source,
            predicate,
            operation,
        }
    }
    pub(super) const fn source(&self) -> ShaderSourceLocation {
        self.source
    }
    pub(super) const fn predicate(&self) -> ShaderPredicate {
        self.predicate
    }
    pub(super) const fn operation(&self) -> &Operation {
        &self.operation
    }
}
impl From<Operation> for nixe_gpu::ShaderOperation {
    fn from(op: Operation) -> Self {
        match op {
            Operation::Nop => Self::Nop,
            Operation::QuadShuffle32 {
                destination,
                source,
                operation,
            } => Self::QuadShuffle32 {
                destination: destination.oracle(),
                source: source.oracle(),
                operation,
            },
            Operation::FloatQuadSwizzleAdd32 {
                destination,
                left,
                right,
                swizzle,
                float_control,
            } => Self::FloatQuadSwizzleAdd32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                swizzle,
                float_control,
            },
            Operation::LoadComputeBuiltin32 {
                destination,
                builtin,
                component,
            } => Self::LoadComputeBuiltin32 {
                destination: destination.oracle(),
                builtin,
                component,
            },
            Operation::LoadStorageBuffer32 {
                destination,
                binding,
                word_index,
            } => Self::LoadStorageBuffer32 {
                destination: destination.oracle(),
                binding: u8::try_from(binding)
                    .expect("oracle resource width was checked before export"),
                word_index: word_index.oracle(),
            },
            Operation::StoreStorageBuffer32 {
                source,
                binding,
                word_index,
            } => Self::StoreStorageBuffer32 {
                source: source.oracle(),
                binding: u8::try_from(binding)
                    .expect("oracle resource width was checked before export"),
                word_index: word_index.oracle(),
            },
            Operation::Undefined32 { destination } => Self::Undefined32 {
                destination: destination.oracle(),
            },
            Operation::MoveImmediate32 {
                destination,
                bits,
                scalar_type,
            } => Self::MoveImmediate32 {
                destination: destination.oracle(),
                bits,
                scalar_type,
            },
            Operation::Move32 {
                destination,
                source,
                scalar_type,
            } => Self::Move32 {
                destination: destination.oracle(),
                source: source.oracle(),
                scalar_type,
            },
            Operation::FloatAbsolute32 {
                destination,
                source,
            } => Self::FloatAbsolute32 {
                destination: destination.oracle(),
                source: source.oracle(),
            },
            Operation::FloatNegate32 {
                destination,
                source,
            } => Self::FloatNegate32 {
                destination: destination.oracle(),
                source: source.oracle(),
            },
            Operation::UnpackHalf32 {
                destination,
                source,
                high,
            } => Self::UnpackHalf32 {
                destination: destination.oracle(),
                source: source.oracle(),
                high,
            },
            Operation::PackHalf32 {
                destination,
                source,
            } => Self::PackHalf32 {
                destination: destination.oracle(),
                source: source.oracle(),
            },
            Operation::FusedMultiplyAddHalf32 {
                destination,
                left,
                right,
                addend,
                saturate,
            } => Self::FusedMultiplyAddHalf32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                addend: addend.oracle(),
                saturate,
            },
            Operation::ConvertIntegerToFloat32 {
                destination,
                source,
                source_type,
                rounding,
            } => Self::ConvertIntegerToFloat32 {
                destination: destination.oracle(),
                source: source.oracle(),
                source_type,
                rounding,
            },
            Operation::RoundFloat32ToIntegral {
                destination,
                source,
                rounding,
                flush_denormals_to_zero,
            } => Self::RoundFloat32ToIntegral {
                destination: destination.oracle(),
                source: source.oracle(),
                rounding,
                flush_denormals_to_zero,
            },
            Operation::ConvertFloat32ToInteger {
                destination,
                source,
                destination_type,
                destination_bits,
                rounding,
                flush_denormals_to_zero,
            } => Self::ConvertFloat32ToInteger {
                destination: destination.oracle(),
                source: source.oracle(),
                destination_type,
                destination_bits,
                rounding,
                flush_denormals_to_zero,
            },
            Operation::LoadInput {
                destinations,
                location,
                first_component,
                scalar_type,
            } => Self::LoadInput {
                destinations: destinations.into_iter().map(|v| v.oracle()).collect(),
                location,
                first_component,
                scalar_type,
            },
            Operation::StoreOutput {
                sources,
                location,
                first_component,
                scalar_type,
            } => Self::StoreOutput {
                sources: sources.into_iter().map(|v| v.oracle()).collect(),
                location,
                first_component,
                scalar_type,
            },
            Operation::LoadControlPoint {
                destination,
                vertex,
                output,
                location,
                component,
            } => Self::LoadControlPoint {
                destination: destination.oracle(),
                vertex: vertex.oracle(),
                output,
                location,
                component,
            },
            Operation::StoreControlPoint {
                source,
                vertex,
                location,
                component,
            } => Self::StoreControlPoint {
                source: source.oracle(),
                vertex: vertex.oracle(),
                location,
                component,
            },
            Operation::LoadPatchOutput {
                destination,
                location,
                component,
            } => Self::LoadPatchOutput {
                destination: destination.oracle(),
                location,
                component,
            },
            Operation::PatchBarrier => Self::PatchBarrier,
            Operation::Multiply32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => Self::Multiply32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                scalar_type,
                float_control,
            },
            Operation::FloatMultiplyZero32 {
                destination,
                left,
                right,
                float_control,
            } => Self::FloatMultiplyZero32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                float_control,
            },
            Operation::Add32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => Self::Add32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                scalar_type,
                float_control,
            },
            Operation::AddCarry32 {
                destination,
                carry_out,
                left,
                right,
                carry_in,
            } => Self::AddCarry32 {
                destination: destination.oracle(),
                carry_out: carry_out.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                carry_in: carry_in.map(|v| v.oracle()),
            },
            Operation::ShiftLeft32 {
                destination,
                value,
                amount,
                wrap,
            } => Self::ShiftLeft32 {
                destination: destination.oracle(),
                value: value.oracle(),
                amount: amount.oracle(),
                wrap,
            },
            Operation::ShiftRight32 {
                destination,
                value,
                amount,
                wrap,
                signed,
                reverse,
            } => Self::ShiftRight32 {
                destination: destination.oracle(),
                value: value.oracle(),
                amount: amount.oracle(),
                wrap,
                signed,
                reverse,
            },
            Operation::ExtractBitfield32 {
                destination,
                value,
                offset,
                count,
                signed,
                reverse,
            } => Self::ExtractBitfield32 {
                destination: destination.oracle(),
                value: value.oracle(),
                offset: offset.oracle(),
                count: count.oracle(),
                signed,
                reverse,
            },
            Operation::Bitwise32 {
                destination,
                left,
                right,
                operation,
            } => Self::Bitwise32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                operation,
            },
            Operation::FloatMinMax32 {
                destination,
                left,
                right,
                minimum,
                float_control,
            } => Self::FloatMinMax32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                minimum,
                float_control,
            },
            Operation::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } => Self::FusedMultiplyAdd32 {
                destination: destination.oracle(),
                left: left.oracle(),
                right: right.oracle(),
                addend: addend.oracle(),
                float_control,
            },
            Operation::Reciprocal32 {
                destination,
                source,
                accuracy,
                float_control,
            } => Self::Reciprocal32 {
                destination: destination.oracle(),
                source: source.oracle(),
                accuracy,
                float_control,
            },
            Operation::ReciprocalSqrt32 {
                destination,
                source,
                accuracy,
                float_control,
            } => Self::ReciprocalSqrt32 {
                destination: destination.oracle(),
                source: source.oracle(),
                accuracy,
                float_control,
            },
            Operation::SpecialFunction32 {
                destination,
                source,
                function,
                accuracy,
                float_control,
            } => Self::SpecialFunction32 {
                destination: destination.oracle(),
                source: source.oracle(),
                function,
                accuracy,
                float_control,
            },
            Operation::Select32 {
                destination,
                condition,
                when_true,
                when_false,
            } => Self::Select32 {
                destination: destination.oracle(),
                condition: condition.oracle(),
                when_true: when_true.oracle(),
                when_false: when_false.oracle(),
            },
            Operation::CompareInteger32 {
                destination,
                float_result,
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
            } => Self::CompareInteger32 {
                destination: destination.oracle(),
                float_result,
                left: left.oracle(),
                right: right.oracle(),
                signed,
                comparison,
                accumulator,
                set_operation,
            },
            Operation::CompareFloat32 {
                destination,
                float_result,
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            } => Self::CompareFloat32 {
                destination: destination.oracle(),
                float_result,
                left: left.oracle(),
                right: right.oracle(),
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            },
            Operation::SetPredicatesFromBits {
                source,
                first_bit,
                write_mask,
            } => Self::SetPredicatesFromBits {
                source: source.oracle(),
                first_bit,
                write_mask,
            },
            Operation::SetPredicateBoolean {
                destinations,
                inputs,
                operations,
            } => Self::SetPredicateBoolean {
                destinations,
                inputs,
                operations,
            },
            Operation::SetPredicateFloat32 {
                destination,
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            } => Self::SetPredicateFloat32 {
                destination,
                left: left.oracle(),
                right: right.oracle(),
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            },
            Operation::SetPredicateInteger32 {
                destinations,
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
            } => Self::SetPredicateInteger32 {
                destinations,
                left: left.oracle(),
                right: right.oracle(),
                signed,
                comparison,
                accumulator,
                set_operation,
            },
            Operation::InterpolateInput {
                destination,
                location,
                component,
                interpolation,
            } => Self::InterpolateInput {
                destination: destination.oracle(),
                location,
                component,
                interpolation,
            },
            Operation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset,
                scalar_type,
            } => Self::LoadConstantBuffer32 {
                destination: destination.oracle(),
                binding: u8::try_from(binding)
                    .expect("oracle resource width was checked before export"),
                byte_offset,
                scalar_type,
            },
            Operation::LoadConstantBufferIndexed32 {
                destination,
                binding,
                base_byte_offset,
                dynamic_byte_offset,
                scalar_type,
            } => Self::LoadConstantBufferIndexed32 {
                destination: destination.oracle(),
                binding: u8::try_from(binding)
                    .expect("oracle resource width was checked before export"),
                base_byte_offset,
                dynamic_byte_offset: dynamic_byte_offset.oracle(),
                scalar_type,
            },
            Operation::SampleDepthCompare2D {
                offset,
                outputs,
                coordinates,
                array_index,
                reference,
                image_binding,
                sampler_binding,
                level_zero,
            } => Self::SampleDepthCompare2D {
                offset,
                outputs: outputs.into_iter().map(|v| v.oracle()).collect(),
                coordinates: coordinates.map(|v| v.oracle()),
                array_index: array_index.map(|v| v.oracle()),
                reference: reference.oracle(),
                image_binding: u8::try_from(image_binding)
                    .expect("oracle resource width was checked before export"),
                sampler_binding: u8::try_from(sampler_binding)
                    .expect("oracle resource width was checked before export"),
                level_zero,
            },
            Operation::SampleTexture2D {
                offset,
                outputs,
                coordinates,
                image_binding,
                lod,
                sampler_binding,
            } => Self::SampleTexture2D {
                offset,
                outputs: outputs.into_iter().map(|v| v.oracle()).collect(),
                coordinates: coordinates.map(|v| v.oracle()),
                image_binding: u8::try_from(image_binding)
                    .expect("oracle resource width was checked before export"),
                lod: lod.map(|v| v.oracle()),
                sampler_binding: u8::try_from(sampler_binding)
                    .expect("oracle resource width was checked before export"),
            },
            Operation::LoadTexture2D {
                outputs,
                coordinates,
                image_binding,
                mip_level,
            } => Self::LoadTexture2D {
                outputs: outputs.into_iter().map(|v| v.oracle()).collect(),
                coordinates: coordinates.map(|v| v.oracle()),
                image_binding: u8::try_from(image_binding)
                    .expect("oracle resource width was checked before export"),
                mip_level,
            },
            Operation::QueryTextureLevels {
                destination,
                image_binding,
            } => Self::QueryTextureLevels {
                destination: destination.oracle(),
                image_binding: u8::try_from(image_binding)
                    .expect("oracle resource width was checked before export"),
            },
            Operation::SampleTexture2DArray {
                offset,
                outputs,
                coordinates,
                array_index,
                image_binding,
                lod,
                sampler_binding,
            } => Self::SampleTexture2DArray {
                offset,
                outputs: outputs.into_iter().map(|v| v.oracle()).collect(),
                coordinates: coordinates.map(|v| v.oracle()),
                array_index: array_index.oracle(),
                image_binding: u8::try_from(image_binding)
                    .expect("oracle resource width was checked before export"),
                lod: lod.map(|v| v.oracle()),
                sampler_binding: u8::try_from(sampler_binding)
                    .expect("oracle resource width was checked before export"),
            },
            Operation::SampleTextureCube {
                outputs,
                coordinates,
                image_binding,
                lod,
                sampler_binding,
            } => Self::SampleTextureCube {
                outputs: outputs.into_iter().map(|v| v.oracle()).collect(),
                coordinates: coordinates.map(|v| v.oracle()),
                image_binding: u8::try_from(image_binding)
                    .expect("oracle resource width was checked before export"),
                lod: lod.map(|v| v.oracle()),
                sampler_binding: u8::try_from(sampler_binding)
                    .expect("oracle resource width was checked before export"),
            },
            Operation::Branch { target } => Self::Branch { target },
            Operation::BranchTable { value, targets } => Self::BranchTable {
                value: value.oracle(),
                targets,
            },
            Operation::Discard => Self::Discard,
            Operation::Exit => Self::Exit,
        }
    }
}
impl From<Instruction> for nixe_gpu::ShaderInstruction {
    fn from(i: Instruction) -> Self {
        Self::new(i.source, i.predicate, i.operation.into())
    }
}

#[cfg(test)]
impl From<nixe_gpu::ShaderOperation> for Operation {
    fn from(op: nixe_gpu::ShaderOperation) -> Self {
        match op {
            nixe_gpu::ShaderOperation::Nop => Self::Nop,
            nixe_gpu::ShaderOperation::QuadShuffle32 {
                destination,
                source,
                operation,
            } => Self::QuadShuffle32 {
                destination: destination.into(),
                source: source.into(),
                operation,
            },
            nixe_gpu::ShaderOperation::FloatQuadSwizzleAdd32 {
                destination,
                left,
                right,
                swizzle,
                float_control,
            } => Self::FloatQuadSwizzleAdd32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                swizzle,
                float_control,
            },
            nixe_gpu::ShaderOperation::LoadComputeBuiltin32 {
                destination,
                builtin,
                component,
            } => Self::LoadComputeBuiltin32 {
                destination: destination.into(),
                builtin,
                component,
            },
            nixe_gpu::ShaderOperation::LoadStorageBuffer32 {
                destination,
                binding,
                word_index,
            } => Self::LoadStorageBuffer32 {
                destination: destination.into(),
                binding: u32::from(binding),
                word_index: word_index.into(),
            },
            nixe_gpu::ShaderOperation::StoreStorageBuffer32 {
                source,
                binding,
                word_index,
            } => Self::StoreStorageBuffer32 {
                source: source.into(),
                binding: u32::from(binding),
                word_index: word_index.into(),
            },
            nixe_gpu::ShaderOperation::Undefined32 { destination } => Self::Undefined32 {
                destination: destination.into(),
            },
            nixe_gpu::ShaderOperation::MoveImmediate32 {
                destination,
                bits,
                scalar_type,
            } => Self::MoveImmediate32 {
                destination: destination.into(),
                bits,
                scalar_type,
            },
            nixe_gpu::ShaderOperation::Move32 {
                destination,
                source,
                scalar_type,
            } => Self::Move32 {
                destination: destination.into(),
                source: source.into(),
                scalar_type,
            },
            nixe_gpu::ShaderOperation::FloatAbsolute32 {
                destination,
                source,
            } => Self::FloatAbsolute32 {
                destination: destination.into(),
                source: source.into(),
            },
            nixe_gpu::ShaderOperation::FloatNegate32 {
                destination,
                source,
            } => Self::FloatNegate32 {
                destination: destination.into(),
                source: source.into(),
            },
            nixe_gpu::ShaderOperation::UnpackHalf32 {
                destination,
                source,
                high,
            } => Self::UnpackHalf32 {
                destination: destination.into(),
                source: source.into(),
                high,
            },
            nixe_gpu::ShaderOperation::PackHalf32 {
                destination,
                source,
            } => Self::PackHalf32 {
                destination: destination.into(),
                source: source.into(),
            },
            nixe_gpu::ShaderOperation::FusedMultiplyAddHalf32 {
                destination,
                left,
                right,
                addend,
                saturate,
            } => Self::FusedMultiplyAddHalf32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                addend: addend.into(),
                saturate,
            },
            nixe_gpu::ShaderOperation::ConvertIntegerToFloat32 {
                destination,
                source,
                source_type,
                rounding,
            } => Self::ConvertIntegerToFloat32 {
                destination: destination.into(),
                source: source.into(),
                source_type,
                rounding,
            },
            nixe_gpu::ShaderOperation::RoundFloat32ToIntegral {
                destination,
                source,
                rounding,
                flush_denormals_to_zero,
            } => Self::RoundFloat32ToIntegral {
                destination: destination.into(),
                source: source.into(),
                rounding,
                flush_denormals_to_zero,
            },
            nixe_gpu::ShaderOperation::ConvertFloat32ToInteger {
                destination,
                source,
                destination_type,
                destination_bits,
                rounding,
                flush_denormals_to_zero,
            } => Self::ConvertFloat32ToInteger {
                destination: destination.into(),
                source: source.into(),
                destination_type,
                destination_bits,
                rounding,
                flush_denormals_to_zero,
            },
            nixe_gpu::ShaderOperation::LoadInput {
                destinations,
                location,
                first_component,
                scalar_type,
            } => Self::LoadInput {
                destinations: destinations.into_iter().map(Into::into).collect(),
                location,
                first_component,
                scalar_type,
            },
            nixe_gpu::ShaderOperation::StoreOutput {
                sources,
                location,
                first_component,
                scalar_type,
            } => Self::StoreOutput {
                sources: sources.into_iter().map(Into::into).collect(),
                location,
                first_component,
                scalar_type,
            },
            nixe_gpu::ShaderOperation::LoadControlPoint {
                destination,
                vertex,
                output,
                location,
                component,
            } => Self::LoadControlPoint {
                destination: destination.into(),
                vertex: vertex.into(),
                output,
                location,
                component,
            },
            nixe_gpu::ShaderOperation::StoreControlPoint {
                source,
                vertex,
                location,
                component,
            } => Self::StoreControlPoint {
                source: source.into(),
                vertex: vertex.into(),
                location,
                component,
            },
            nixe_gpu::ShaderOperation::LoadPatchOutput {
                destination,
                location,
                component,
            } => Self::LoadPatchOutput {
                destination: destination.into(),
                location,
                component,
            },
            nixe_gpu::ShaderOperation::PatchBarrier => Self::PatchBarrier,
            nixe_gpu::ShaderOperation::Multiply32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => Self::Multiply32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                scalar_type,
                float_control,
            },
            nixe_gpu::ShaderOperation::FloatMultiplyZero32 {
                destination,
                left,
                right,
                float_control,
            } => Self::FloatMultiplyZero32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                float_control,
            },
            nixe_gpu::ShaderOperation::Add32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => Self::Add32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                scalar_type,
                float_control,
            },
            nixe_gpu::ShaderOperation::AddCarry32 {
                destination,
                carry_out,
                left,
                right,
                carry_in,
            } => Self::AddCarry32 {
                destination: destination.into(),
                carry_out: carry_out.into(),
                left: left.into(),
                right: right.into(),
                carry_in: carry_in.map(Into::into),
            },
            nixe_gpu::ShaderOperation::ShiftLeft32 {
                destination,
                value,
                amount,
                wrap,
            } => Self::ShiftLeft32 {
                destination: destination.into(),
                value: value.into(),
                amount: amount.into(),
                wrap,
            },
            nixe_gpu::ShaderOperation::ShiftRight32 {
                destination,
                value,
                amount,
                wrap,
                signed,
                reverse,
            } => Self::ShiftRight32 {
                destination: destination.into(),
                value: value.into(),
                amount: amount.into(),
                wrap,
                signed,
                reverse,
            },
            nixe_gpu::ShaderOperation::ExtractBitfield32 {
                destination,
                value,
                offset,
                count,
                signed,
                reverse,
            } => Self::ExtractBitfield32 {
                destination: destination.into(),
                value: value.into(),
                offset: offset.into(),
                count: count.into(),
                signed,
                reverse,
            },
            nixe_gpu::ShaderOperation::Bitwise32 {
                destination,
                left,
                right,
                operation,
            } => Self::Bitwise32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                operation,
            },
            nixe_gpu::ShaderOperation::FloatMinMax32 {
                destination,
                left,
                right,
                minimum,
                float_control,
            } => Self::FloatMinMax32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                minimum,
                float_control,
            },
            nixe_gpu::ShaderOperation::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } => Self::FusedMultiplyAdd32 {
                destination: destination.into(),
                left: left.into(),
                right: right.into(),
                addend: addend.into(),
                float_control,
            },
            nixe_gpu::ShaderOperation::Reciprocal32 {
                destination,
                source,
                accuracy,
                float_control,
            } => Self::Reciprocal32 {
                destination: destination.into(),
                source: source.into(),
                accuracy,
                float_control,
            },
            nixe_gpu::ShaderOperation::ReciprocalSqrt32 {
                destination,
                source,
                accuracy,
                float_control,
            } => Self::ReciprocalSqrt32 {
                destination: destination.into(),
                source: source.into(),
                accuracy,
                float_control,
            },
            nixe_gpu::ShaderOperation::SpecialFunction32 {
                destination,
                source,
                function,
                accuracy,
                float_control,
            } => Self::SpecialFunction32 {
                destination: destination.into(),
                source: source.into(),
                function,
                accuracy,
                float_control,
            },
            nixe_gpu::ShaderOperation::Select32 {
                destination,
                condition,
                when_true,
                when_false,
            } => Self::Select32 {
                destination: destination.into(),
                condition: condition.into(),
                when_true: when_true.into(),
                when_false: when_false.into(),
            },
            nixe_gpu::ShaderOperation::CompareInteger32 {
                destination,
                float_result,
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
            } => Self::CompareInteger32 {
                destination: destination.into(),
                float_result,
                left: left.into(),
                right: right.into(),
                signed,
                comparison,
                accumulator,
                set_operation,
            },
            nixe_gpu::ShaderOperation::CompareFloat32 {
                destination,
                float_result,
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            } => Self::CompareFloat32 {
                destination: destination.into(),
                float_result,
                left: left.into(),
                right: right.into(),
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            },
            nixe_gpu::ShaderOperation::SetPredicatesFromBits {
                source,
                first_bit,
                write_mask,
            } => Self::SetPredicatesFromBits {
                source: source.into(),
                first_bit,
                write_mask,
            },
            nixe_gpu::ShaderOperation::SetPredicateBoolean {
                destinations,
                inputs,
                operations,
            } => Self::SetPredicateBoolean {
                destinations,
                inputs,
                operations,
            },
            nixe_gpu::ShaderOperation::SetPredicateFloat32 {
                destination,
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            } => Self::SetPredicateFloat32 {
                destination,
                left: left.into(),
                right: right.into(),
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
            },
            nixe_gpu::ShaderOperation::SetPredicateInteger32 {
                destinations,
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
            } => Self::SetPredicateInteger32 {
                destinations,
                left: left.into(),
                right: right.into(),
                signed,
                comparison,
                accumulator,
                set_operation,
            },
            nixe_gpu::ShaderOperation::InterpolateInput {
                destination,
                location,
                component,
                interpolation,
            } => Self::InterpolateInput {
                destination: destination.into(),
                location,
                component,
                interpolation,
            },
            nixe_gpu::ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset,
                scalar_type,
            } => Self::LoadConstantBuffer32 {
                destination: destination.into(),
                binding: u32::from(binding),
                byte_offset,
                scalar_type,
            },
            nixe_gpu::ShaderOperation::LoadConstantBufferIndexed32 {
                destination,
                binding,
                base_byte_offset,
                dynamic_byte_offset,
                scalar_type,
            } => Self::LoadConstantBufferIndexed32 {
                destination: destination.into(),
                binding: u32::from(binding),
                base_byte_offset,
                dynamic_byte_offset: dynamic_byte_offset.into(),
                scalar_type,
            },
            nixe_gpu::ShaderOperation::SampleDepthCompare2D {
                offset,
                outputs,
                coordinates,
                array_index,
                reference,
                image_binding,
                sampler_binding,
                level_zero,
            } => Self::SampleDepthCompare2D {
                offset,
                outputs: outputs.into_iter().map(Into::into).collect(),
                coordinates: coordinates.map(Into::into),
                array_index: array_index.map(Into::into),
                reference: reference.into(),
                image_binding: u32::from(image_binding),
                sampler_binding: u32::from(sampler_binding),
                level_zero,
            },
            nixe_gpu::ShaderOperation::SampleTexture2D {
                offset,
                outputs,
                coordinates,
                image_binding,
                lod,
                sampler_binding,
            } => Self::SampleTexture2D {
                offset,
                outputs: outputs.into_iter().map(Into::into).collect(),
                coordinates: coordinates.map(Into::into),
                image_binding: u32::from(image_binding),
                lod: lod.map(Into::into),
                sampler_binding: u32::from(sampler_binding),
            },
            nixe_gpu::ShaderOperation::LoadTexture2D {
                outputs,
                coordinates,
                image_binding,
                mip_level,
            } => Self::LoadTexture2D {
                outputs: outputs.into_iter().map(Into::into).collect(),
                coordinates: coordinates.map(Into::into),
                image_binding: u32::from(image_binding),
                mip_level,
            },
            nixe_gpu::ShaderOperation::QueryTextureLevels {
                destination,
                image_binding,
            } => Self::QueryTextureLevels {
                destination: destination.into(),
                image_binding: u32::from(image_binding),
            },
            nixe_gpu::ShaderOperation::SampleTexture2DArray {
                offset,
                outputs,
                coordinates,
                array_index,
                image_binding,
                lod,
                sampler_binding,
            } => Self::SampleTexture2DArray {
                offset,
                outputs: outputs.into_iter().map(Into::into).collect(),
                coordinates: coordinates.map(Into::into),
                array_index: array_index.into(),
                image_binding: u32::from(image_binding),
                lod: lod.map(Into::into),
                sampler_binding: u32::from(sampler_binding),
            },
            nixe_gpu::ShaderOperation::SampleTextureCube {
                outputs,
                coordinates,
                image_binding,
                lod,
                sampler_binding,
            } => Self::SampleTextureCube {
                outputs: outputs.into_iter().map(Into::into).collect(),
                coordinates: coordinates.map(Into::into),
                image_binding: u32::from(image_binding),
                lod: lod.map(Into::into),
                sampler_binding: u32::from(sampler_binding),
            },
            nixe_gpu::ShaderOperation::Branch { target } => Self::Branch { target },
            nixe_gpu::ShaderOperation::BranchTable { value, targets } => Self::BranchTable {
                value: value.into(),
                targets,
            },
            nixe_gpu::ShaderOperation::Discard => Self::Discard,
            nixe_gpu::ShaderOperation::Exit => Self::Exit,
        }
    }
}
#[cfg(test)]
impl From<nixe_gpu::ShaderInstruction> for Instruction {
    fn from(i: nixe_gpu::ShaderInstruction) -> Self {
        Self::new(i.source(), i.predicate(), i.operation().clone().into())
    }
}

/// Frontend-local resource identities have no portable-renderer width limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResourceAccess {
    binding: u32,
    kind: ShaderResourceKind,
    readable: bool,
    writable: bool,
}
impl ResourceAccess {
    pub(super) fn new(
        binding: u32,
        kind: ShaderResourceKind,
        readable: bool,
        writable: bool,
    ) -> Result<Self, nixe_gpu::ssa::Error> {
        if !readable && !writable {
            return Err(nixe_gpu::ssa::Error {
                stage: None,
                source: ShaderSourceLocation::new(0),
                detail: "empty frontend resource access".into(),
            });
        }
        Ok(Self {
            binding,
            kind,
            readable,
            writable,
        })
    }
    pub(super) fn binding(self) -> u32 {
        self.binding
    }
    pub(super) fn kind(self) -> ShaderResourceKind {
        self.kind
    }
    pub(super) fn oracle(
        self,
    ) -> Result<nixe_gpu::ShaderResourceAccess, super::error::MaxwellShaderTranslationError> {
        let binding = u8::try_from(self.binding)
            .map_err(|_| super::error::MaxwellShaderTranslationError::ResourceBindingExhausted)?;
        Ok(
            nixe_gpu::ShaderResourceAccess::new(binding, self.kind, self.readable, self.writable)
                .expect("decoded resource has a read/write use"),
        )
    }
}

#[cfg(test)]
impl From<Register> for nixe_gpu::ShaderRegister {
    fn from(value: Register) -> Self {
        value.oracle()
    }
}
#[cfg(test)]
impl From<TextureOutput> for nixe_gpu::ShaderTextureSampleOutput {
    fn from(value: TextureOutput) -> Self {
        value.oracle()
    }
}
