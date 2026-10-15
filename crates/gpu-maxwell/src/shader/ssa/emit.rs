//! Maxwell-specific word/predicate legalization. The shared IR receives typed
//! values and semantic operations, never ISA registers, predicates or flags.
use super::*;
use crate::shader::decoded::Register as ShaderRegister;
use crate::shader::decoded::{Instruction as DecodedInstruction, Operation as D};
use nixe_gpu::ShaderIntegerComparison;
use nixe_gpu::ssa::{MathFunction, SampleMode};

impl Emitter {
    pub(super) fn read(&self, register: ShaderRegister) -> Operand {
        self.registers[register.index() as usize].into()
    }
    fn write(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        register: ShaderRegister,
        value: Operand,
    ) {
        self.builder
            .assign(block, pc, self.registers[register.index() as usize], value);
    }
    fn value(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        op: Op,
        args: Vec<Operand>,
    ) -> Operand {
        self.builder.emit(block, pc, op, args)[0].into()
    }
    fn constant(&mut self, block: BlockId, pc: ShaderSourceLocation, bits: u32) -> Operand {
        self.value(block, pc, Op::Constant(Constant::u32(bits)), vec![])
    }
    fn boolean_constant(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        value: bool,
    ) -> Operand {
        self.value(block, pc, Op::Constant(Constant::boolean(value)), vec![])
    }
    fn cast(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        value: Operand,
        from: Type,
        to: Type,
    ) -> Operand {
        if from == to {
            value
        } else {
            self.value(block, pc, Op::Bitcast { from, to }, vec![value])
        }
    }
    fn float(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        register: ShaderRegister,
    ) -> Operand {
        self.cast(block, pc, self.read(register), Type::U32, Type::F32)
    }
    fn write_float(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        register: ShaderRegister,
        value: Operand,
    ) {
        let bits = self.cast(block, pc, value, Type::F32, Type::U32);
        self.write(block, pc, register, bits);
    }
    fn binary(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        operation: Binary,
        ty: Type,
        a: Operand,
        b: Operand,
    ) -> Operand {
        self.value(
            block,
            pc,
            Op::Binary {
                op: operation,
                ty,
                float_control: ShaderFloatControl::PRECISE,
            },
            vec![a, b],
        )
    }
    fn compare(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        comparison: ShaderIntegerComparison,
        a: Operand,
        b: Operand,
    ) -> Operand {
        self.value(
            block,
            pc,
            Op::IntegerCompare {
                ty: Type::U32,
                comparison,
            },
            vec![a, b],
        )
    }
    fn select(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        ty: Type,
        condition: Operand,
        yes: Operand,
        no: Operand,
    ) -> Operand {
        self.value(block, pc, Op::Select(ty), vec![condition, yes, no])
    }
    fn not(&mut self, block: BlockId, pc: ShaderSourceLocation, value: Operand) -> Operand {
        self.value(
            block,
            pc,
            Op::Unary {
                op: Unary::Not,
                ty: Type::Bool,
            },
            vec![value],
        )
    }
    fn boolean(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        operation: ShaderPredicateSetOperation,
        a: Operand,
        b: Operand,
    ) -> Operand {
        self.binary(
            block,
            pc,
            match operation {
                ShaderPredicateSetOperation::And => Binary::And,
                ShaderPredicateSetOperation::Or => Binary::Or,
                ShaderPredicateSetOperation::Xor => Binary::Xor,
            },
            Type::Bool,
            a,
            b,
        )
    }
    pub(super) fn predicate(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        predicate: ShaderPredicate,
    ) -> Operand {
        match predicate {
            ShaderPredicate::Always => self.boolean_constant(block, pc, true),
            ShaderPredicate::Never => self.boolean_constant(block, pc, false),
            ShaderPredicate::Register { register, inverted } => {
                let value = self.snapshots[register as usize]
                    .unwrap_or_else(|| self.predicates[register as usize].into());
                if inverted {
                    self.not(block, pc, value)
                } else {
                    value
                }
            }
        }
    }
    pub(super) fn snapshot_predicates(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        instructions: &[DecodedInstruction],
    ) {
        // SASS snapshots predicate sources/guard before any destination write.
        // In particular a predicated multi-result opcode cannot disable itself
        // halfway through legalization, or read its newly written accumulator.
        let mut mask = 0u8;
        let mut visit = |p| {
            if let ShaderPredicate::Register { register, .. } = p {
                mask |= 1 << register;
            }
        };
        for i in instructions {
            visit(i.predicate());
            match i.operation() {
                D::FloatMinMax32 { minimum, .. } => visit(*minimum),
                D::CompareInteger32 { accumulator, .. }
                | D::CompareFloat32 { accumulator, .. }
                | D::SetPredicateFloat32 { accumulator, .. }
                | D::SetPredicateInteger32 { accumulator, .. } => visit(*accumulator),
                D::SetPredicateBoolean { inputs, .. } => {
                    for &p in inputs {
                        visit(p);
                    }
                }
                _ => {}
            }
        }
        self.snapshots.fill(None);
        for index in 0..7 {
            if mask & (1 << index) != 0 {
                self.snapshots[index] = Some(self.value(
                    block,
                    pc,
                    Op::Copy(Type::Bool),
                    vec![self.predicates[index].into()],
                ));
            }
        }
    }
    pub(super) fn operation(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        op: &D,
    ) -> Result<(), MaxwellShaderTranslationError> {
        match op {
            D::Nop => {}
            D::Undefined32 { destination } => {
                let value = self.value(block, pc, Op::Undefined(Type::U32), vec![]);
                self.write(block, pc, *destination, value);
            }
            D::MoveImmediate32 {
                destination, bits, ..
            } => {
                let value = self.constant(block, pc, *bits);
                self.write(block, pc, *destination, value);
            }
            D::Move32 {
                destination,
                source,
                ..
            } => self.write(block, pc, *destination, self.read(*source)),
            D::FloatAbsolute32 {
                destination,
                source,
            }
            | D::FloatNegate32 {
                destination,
                source,
            } => {
                let mask = self.constant(
                    block,
                    pc,
                    if matches!(op, D::FloatAbsolute32 { .. }) {
                        0x7fff_ffff
                    } else {
                        0x8000_0000
                    },
                );
                let value = self.binary(
                    block,
                    pc,
                    if matches!(op, D::FloatAbsolute32 { .. }) {
                        Binary::And
                    } else {
                        Binary::Xor
                    },
                    Type::U32,
                    self.read(*source),
                    mask,
                );
                self.write(block, pc, *destination, value);
            }
            D::UnpackHalf32 {
                destination,
                source,
                high,
            } => {
                let mut bits = self.read(*source);
                if *high {
                    let amount = self.constant(block, pc, 16);
                    bits = self.binary(block, pc, Binary::ShiftRight, Type::U32, bits, amount);
                }
                let value = self.value(block, pc, Op::UnpackHalf, vec![bits]);
                self.write_float(block, pc, *destination, value);
            }
            D::PackHalf32 {
                destination,
                source,
            } => {
                let source = self.float(block, pc, *source);
                let value = self.value(block, pc, Op::PackHalf, vec![source]);
                self.write(block, pc, *destination, value);
            }
            D::FusedMultiplyAddHalf32 {
                destination,
                left,
                right,
                addend,
                saturate,
            } => {
                let a = self.float(block, pc, *left);
                let b = self.float(block, pc, *right);
                let c = self.float(block, pc, *addend);
                let value = self.value(
                    block,
                    pc,
                    Op::HalfFma {
                        saturate: *saturate,
                    },
                    vec![a, b, c],
                );
                self.write(block, pc, *destination, value);
            }
            D::Multiply32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            }
            | D::Add32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => {
                let ty = scalar_type32(*scalar_type, pc)?;
                let a = self.cast(block, pc, self.read(*left), Type::U32, ty);
                let b = self.cast(block, pc, self.read(*right), Type::U32, ty);
                let value = self.value(
                    block,
                    pc,
                    Op::Binary {
                        op: if matches!(op, D::Multiply32 { .. }) {
                            Binary::Multiply
                        } else {
                            Binary::Add
                        },
                        ty,
                        float_control: *float_control,
                    },
                    vec![a, b],
                );
                let bits = self.cast(block, pc, value, ty, Type::U32);
                self.write(block, pc, *destination, bits);
            }
            D::FloatMultiplyZero32 {
                destination,
                left,
                right,
                float_control,
            } => {
                let mask = self.constant(block, pc, 0x7fff_ffff);
                let threshold = self.constant(
                    block,
                    pc,
                    if float_control.denormals_are_zero() {
                        0x7f_ffff
                    } else {
                        0
                    },
                );
                let a = self.binary(block, pc, Binary::And, Type::U32, self.read(*left), mask);
                let b = self.binary(block, pc, Binary::And, Type::U32, self.read(*right), mask);
                let a = self.compare(
                    block,
                    pc,
                    ShaderIntegerComparison::LessOrEqual,
                    a,
                    threshold,
                );
                let b = self.compare(
                    block,
                    pc,
                    ShaderIntegerComparison::LessOrEqual,
                    b,
                    threshold,
                );
                let absorbing = self.binary(block, pc, Binary::Or, Type::Bool, a, b);
                let zero = self.constant(block, pc, 0);
                let a = self.select(block, pc, Type::U32, absorbing, zero, self.read(*left));
                let b = self.select(block, pc, Type::U32, absorbing, zero, self.read(*right));
                let a = self.cast(block, pc, a, Type::U32, Type::F32);
                let b = self.cast(block, pc, b, Type::U32, Type::F32);
                let value = self.value(
                    block,
                    pc,
                    Op::Binary {
                        op: Binary::Multiply,
                        ty: Type::F32,
                        float_control: *float_control,
                    },
                    vec![a, b],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } => {
                let a = self.float(block, pc, *left);
                let b = self.float(block, pc, *right);
                let c = self.float(block, pc, *addend);
                let value = self.value(
                    block,
                    pc,
                    Op::Fma {
                        ty: Type::F32,
                        float_control: *float_control,
                    },
                    vec![a, b, c],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::AddCarry32 {
                destination,
                carry_out,
                left,
                right,
                carry_in,
            } => {
                let c = if let Some(r) = carry_in {
                    self.read(*r)
                } else {
                    self.constant(block, pc, 0)
                };
                let results = self.builder.emit(
                    block,
                    pc,
                    Op::AddCarry,
                    vec![self.read(*left), self.read(*right), c],
                );
                self.write(block, pc, *destination, results[0].into());
                self.write(block, pc, *carry_out, results[1].into());
            }
            D::Bitwise32 {
                destination,
                left,
                right,
                operation,
            } => {
                let value = self.binary(
                    block,
                    pc,
                    match operation {
                        nixe_gpu::ShaderBitwiseOperation::And => Binary::And,
                        nixe_gpu::ShaderBitwiseOperation::Or => Binary::Or,
                        nixe_gpu::ShaderBitwiseOperation::Xor => Binary::Xor,
                    },
                    Type::U32,
                    self.read(*left),
                    self.read(*right),
                );
                self.write(block, pc, *destination, value);
            }
            D::Select32 {
                destination,
                condition,
                when_true,
                when_false,
            } => {
                let zero = self.constant(block, pc, 0);
                let test = self.compare(
                    block,
                    pc,
                    ShaderIntegerComparison::NotEqual,
                    self.read(*condition),
                    zero,
                );
                let value = self.select(
                    block,
                    pc,
                    Type::U32,
                    test,
                    self.read(*when_true),
                    self.read(*when_false),
                );
                self.write(block, pc, *destination, value);
            }
            D::ShiftLeft32 { .. } | D::ShiftRight32 { .. } => self.shift(block, pc, op),
            D::ExtractBitfield32 { .. } => self.extract(block, pc, op),
            D::FloatMinMax32 {
                destination,
                left,
                right,
                minimum,
                float_control,
            } => {
                let a = self.float(block, pc, *left);
                let b = self.float(block, pc, *right);
                let minimum = self.predicate(block, pc, *minimum);
                let value = self.value(
                    block,
                    pc,
                    Op::FloatMinMax {
                        float_control: *float_control,
                    },
                    vec![a, b, minimum],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::Reciprocal32 {
                destination,
                source,
                accuracy,
                float_control,
            }
            | D::ReciprocalSqrt32 {
                destination,
                source,
                accuracy,
                float_control,
            } => {
                let source = self.float(block, pc, *source);
                let function = if matches!(op, D::Reciprocal32 { .. }) {
                    MathFunction::Reciprocal
                } else {
                    MathFunction::ReciprocalSqrt
                };
                let value = self.value(
                    block,
                    pc,
                    Op::Math {
                        function,
                        accuracy: *accuracy,
                        float_control: *float_control,
                    },
                    vec![source],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::SpecialFunction32 {
                destination,
                source,
                function,
                accuracy,
                float_control,
            } => {
                let source = self.float(block, pc, *source);
                let value = self.value(
                    block,
                    pc,
                    Op::Math {
                        function: MathFunction::Special(*function),
                        accuracy: *accuracy,
                        float_control: *float_control,
                    },
                    vec![source],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::ConvertIntegerToFloat32 {
                destination,
                source,
                source_type,
                rounding,
            } => {
                let ty = scalar_type32(*source_type, pc)?;
                let source = self.cast(block, pc, self.read(*source), Type::U32, ty);
                let value = self.value(
                    block,
                    pc,
                    Op::Convert {
                        from: ty,
                        to: Type::F32,
                        rounding: *rounding,
                        flush_denormals: false,
                    },
                    vec![source],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::RoundFloat32ToIntegral {
                destination,
                source,
                rounding,
                flush_denormals_to_zero,
            } => {
                let source = self.float(block, pc, *source);
                let value = self.value(
                    block,
                    pc,
                    Op::RoundIntegral {
                        rounding: *rounding,
                        flush_denormals: *flush_denormals_to_zero,
                    },
                    vec![source],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::ConvertFloat32ToInteger {
                destination,
                source,
                destination_type,
                destination_bits,
                rounding,
                flush_denormals_to_zero,
            } => {
                let ty = scalar_type32(*destination_type, pc)?;
                let source = self.float(block, pc, *source);
                let value = self.value(
                    block,
                    pc,
                    Op::FloatToInteger {
                        to: ty,
                        bits: u32::from(*destination_bits),
                        rounding: *rounding,
                        flush_denormals: *flush_denormals_to_zero,
                    },
                    vec![source],
                );
                let bits = self.cast(block, pc, value, ty, Type::U32);
                self.write(block, pc, *destination, bits);
            }
            D::CompareInteger32 {
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
                ..
            }
            | D::SetPredicateInteger32 {
                destinations: _,
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
            } => {
                let ty = if *signed { Type::I32 } else { Type::U32 };
                let a = self.cast(block, pc, self.read(*left), Type::U32, ty);
                let b = self.cast(block, pc, self.read(*right), Type::U32, ty);
                let compared = self.value(
                    block,
                    pc,
                    Op::IntegerCompare {
                        ty,
                        comparison: *comparison,
                    },
                    vec![a, b],
                );
                self.set_comparison(block, pc, op, compared, *accumulator, *set_operation);
            }
            D::CompareFloat32 {
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
                ..
            }
            | D::SetPredicateFloat32 {
                left,
                right,
                comparison,
                accumulator,
                set_operation,
                flush_denormals_to_zero,
                ..
            } => {
                let a = self.float(block, pc, *left);
                let b = self.float(block, pc, *right);
                let compared = self.value(
                    block,
                    pc,
                    Op::FloatCompare {
                        ty: Type::F32,
                        comparison: *comparison,
                        flush_denormals: *flush_denormals_to_zero,
                    },
                    vec![a, b],
                );
                self.set_comparison(block, pc, op, compared, *accumulator, *set_operation);
            }
            D::SetPredicatesFromBits {
                source,
                first_bit,
                write_mask,
            } => {
                let zero = self.constant(block, pc, 0);
                for index in 0..7 {
                    if write_mask & (1 << index) != 0 {
                        let mask = self.constant(block, pc, 1 << (u32::from(*first_bit) + index));
                        let bits = self.binary(
                            block,
                            pc,
                            Binary::And,
                            Type::U32,
                            self.read(*source),
                            mask,
                        );
                        let value =
                            self.compare(block, pc, ShaderIntegerComparison::NotEqual, bits, zero);
                        self.builder
                            .assign(block, pc, self.predicates[index as usize], value);
                    }
                }
            }
            D::SetPredicateBoolean {
                destinations,
                inputs,
                operations,
            } => {
                let inputs = inputs.map(|p| self.predicate(block, pc, p));
                let inverse = self.not(block, pc, inputs[0]);
                let results = [inputs[0], inverse].map(|a| {
                    let a = self.boolean(block, pc, operations[0], a, inputs[1]);
                    self.boolean(block, pc, operations[1], a, inputs[2])
                });
                for (destination, value) in destinations.iter().zip(results) {
                    if let Some(destination) = destination {
                        self.builder.assign(
                            block,
                            pc,
                            self.predicates[*destination as usize],
                            value,
                        );
                    }
                }
            }
            D::LoadComputeBuiltin32 {
                destination,
                builtin,
                component,
            } => {
                let value = self.value(
                    block,
                    pc,
                    Op::Builtin {
                        builtin: *builtin,
                        component: u32::from(*component),
                    },
                    vec![],
                );
                self.write(block, pc, *destination, value);
            }
            D::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset,
                ..
            } => {
                let address = self.constant(block, pc, *byte_offset);
                let value = self.load(block, pc, *binding, MemorySpace::Constant, address);
                self.write(block, pc, *destination, value);
            }
            D::LoadConstantBufferIndexed32 {
                destination,
                binding,
                base_byte_offset,
                dynamic_byte_offset,
                ..
            } => {
                let base = self.constant(block, pc, *base_byte_offset as u32);
                let address = self.binary(
                    block,
                    pc,
                    Binary::Add,
                    Type::U32,
                    self.read(*dynamic_byte_offset),
                    base,
                );
                let mask = self.constant(block, pc, !3u32);
                let address = self.binary(block, pc, Binary::And, Type::U32, address, mask);
                let value = self.load(block, pc, *binding, MemorySpace::Constant, address);
                self.write(block, pc, *destination, value);
            }
            D::LoadStorageBuffer32 {
                destination,
                binding,
                word_index,
            } => {
                let amount = self.constant(block, pc, 2);
                let address = self.binary(
                    block,
                    pc,
                    Binary::ShiftLeft,
                    Type::U32,
                    self.read(*word_index),
                    amount,
                );
                let value = self.load(block, pc, *binding, MemorySpace::Storage, address);
                self.write(block, pc, *destination, value);
            }
            D::StoreStorageBuffer32 {
                source,
                binding,
                word_index,
            } => {
                let amount = self.constant(block, pc, 2);
                let address = self.binary(
                    block,
                    pc,
                    Binary::ShiftLeft,
                    Type::U32,
                    self.read(*word_index),
                    amount,
                );
                self.builder.emit(
                    block,
                    pc,
                    Op::Store {
                        address_type: Type::U32,
                        space: MemorySpace::Storage,
                        resource: Some(ResourceId::new(*binding)),
                        ty: Type::U32,
                    },
                    vec![address, self.read(*source)],
                );
            }
            D::LoadInput {
                destinations,
                location,
                first_component,
                scalar_type,
            } => {
                let ty = scalar_type32(*scalar_type, pc)?;
                let values: Vec<_> = (0..destinations.len())
                    .map(|i| {
                        let value = self.value(
                            block,
                            pc,
                            Op::Input {
                                location: *location,
                                component: u32::from(*first_component) + i as u32,
                                ty,
                            },
                            vec![],
                        );
                        self.cast(block, pc, value, ty, Type::U32)
                    })
                    .collect();
                for (&destination, value) in destinations.iter().zip(values) {
                    self.write(block, pc, destination, value);
                }
            }
            D::StoreOutput {
                sources,
                location,
                first_component,
                scalar_type,
            } => {
                let ty = scalar_type32(*scalar_type, pc)?;
                for (i, &source) in sources.iter().enumerate() {
                    let value = self.cast(block, pc, self.read(source), Type::U32, ty);
                    self.builder.emit(
                        block,
                        pc,
                        Op::Output {
                            location: *location,
                            component: u32::from(*first_component) + i as u32,
                            ty,
                        },
                        vec![value],
                    );
                }
            }
            D::InterpolateInput {
                destination,
                location,
                component,
                interpolation,
            } => {
                let value = self.value(
                    block,
                    pc,
                    Op::Interpolate {
                        location: *location,
                        component: u32::from(*component),
                        interpolation: *interpolation,
                    },
                    vec![],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::LoadControlPoint {
                destination,
                vertex,
                output,
                location,
                component,
            } => {
                let value = self.value(
                    block,
                    pc,
                    Op::ControlPointLoad {
                        location: *location,
                        component: u32::from(*component),
                        output: *output,
                    },
                    vec![self.read(*vertex)],
                );
                self.write(block, pc, *destination, value);
            }
            D::StoreControlPoint {
                source,
                vertex,
                location,
                component,
            } => {
                self.builder.emit(
                    block,
                    pc,
                    Op::ControlPointStore {
                        location: *location,
                        component: u32::from(*component),
                    },
                    vec![self.read(*vertex), self.read(*source)],
                );
            }
            D::LoadPatchOutput {
                destination,
                location,
                component,
            } => {
                let value = self.value(
                    block,
                    pc,
                    Op::PatchOutputLoad {
                        location: *location,
                        component: u32::from(*component),
                    },
                    vec![],
                );
                self.write(block, pc, *destination, value);
            }
            D::PatchBarrier => {
                self.builder.emit(
                    block,
                    pc,
                    Op::Barrier {
                        spaces: nixe_gpu::ssa::BarrierMemory::OUTPUTS,
                        execution: nixe_gpu::ssa::BarrierScope::Patch,
                        memory: nixe_gpu::ssa::BarrierScope::Invocation,
                    },
                    vec![],
                );
            }
            D::QuadShuffle32 {
                destination,
                source,
                operation,
            } => {
                let value = self.value(block, pc, Op::Quad(*operation), vec![self.read(*source)]);
                self.write(block, pc, *destination, value);
            }
            D::FloatQuadSwizzleAdd32 {
                destination,
                left,
                right,
                swizzle,
                float_control,
            } => {
                let a = self.float(block, pc, *left);
                let b = self.float(block, pc, *right);
                let value = self.value(
                    block,
                    pc,
                    Op::QuadSwizzleAdd {
                        swizzle: u32::from(*swizzle),
                        float_control: *float_control,
                    },
                    vec![a, b],
                );
                self.write_float(block, pc, *destination, value);
            }
            D::QueryTextureLevels {
                destination,
                image_binding,
            } => {
                let value = self.value(
                    block,
                    pc,
                    Op::TextureLevels {
                        image: ResourceId::new(*image_binding),
                    },
                    vec![],
                );
                self.write(block, pc, *destination, value);
            }
            D::SampleTexture2D { .. }
            | D::SampleTexture2DArray { .. }
            | D::SampleTextureCube { .. }
            | D::LoadTexture2D { .. }
            | D::SampleDepthCompare2D { .. } => self.sample(block, pc, op)?,
            D::Branch { .. } | D::BranchTable { .. } | D::Exit | D::Discard => {
                unreachable!("CFG terminators are lowered by the frontend")
            }
        }
        Ok(())
    }
    fn set_comparison(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        op: &D,
        compared: Operand,
        accumulator: ShaderPredicate,
        operation: ShaderPredicateSetOperation,
    ) {
        let accumulator = self.predicate(block, pc, accumulator);
        let inverse = self.not(block, pc, compared);
        let primary = self.boolean(block, pc, operation, compared, accumulator);
        let secondary = self.boolean(block, pc, operation, inverse, accumulator);
        match op {
            D::CompareInteger32 {
                destination,
                float_result,
                ..
            }
            | D::CompareFloat32 {
                destination,
                float_result,
                ..
            } => {
                let one =
                    self.constant(block, pc, if *float_result { 1.0_f32.to_bits() } else { 1 });
                let zero = self.constant(block, pc, 0);
                let value = self.select(block, pc, Type::U32, primary, one, zero);
                self.write(block, pc, *destination, value);
            }
            D::SetPredicateFloat32 { destination, .. } => {
                self.builder
                    .assign(block, pc, self.predicates[*destination as usize], primary)
            }
            D::SetPredicateInteger32 { destinations, .. } => {
                for (dest, value) in destinations.iter().zip([primary, secondary]) {
                    if let Some(dest) = dest {
                        self.builder
                            .assign(block, pc, self.predicates[*dest as usize], value);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    fn load(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        binding: u32,
        space: MemorySpace,
        address: Operand,
    ) -> Operand {
        self.value(
            block,
            pc,
            Op::Load {
                address_type: Type::U32,
                space,
                resource: Some(ResourceId::new(binding)),
                ty: Type::U32,
            },
            vec![address],
        )
    }
    fn shift(&mut self, block: BlockId, pc: ShaderSourceLocation, op: &D) {
        let (destination, source, amount, right, wrap, signed, reverse) = match *op {
            D::ShiftLeft32 {
                destination,
                value,
                amount,
                wrap,
            } => (destination, value, amount, false, wrap, false, false),
            D::ShiftRight32 {
                destination,
                value,
                amount,
                wrap,
                signed,
                reverse,
            } => (destination, value, amount, true, wrap, signed, reverse),
            _ => unreachable!("non-shift decoder operation"),
        };
        let mut base = self.read(source);
        if reverse {
            base = self.value(
                block,
                pc,
                Op::Unary {
                    op: Unary::BitReverse,
                    ty: Type::U32,
                },
                vec![base],
            );
        }
        let mask = self.constant(block, pc, 31);
        let count = self.binary(block, pc, Binary::And, Type::U32, self.read(amount), mask);
        let ty = if signed { Type::I32 } else { Type::U32 };
        let base_typed = self.cast(block, pc, base, Type::U32, ty);
        let count_typed = self.cast(block, pc, count, Type::U32, ty);
        let bits = self.binary(
            block,
            pc,
            if right {
                Binary::ShiftRight
            } else {
                Binary::ShiftLeft
            },
            ty,
            base_typed,
            count_typed,
        );
        let mut bits = self.cast(block, pc, bits, ty, Type::U32);
        if !wrap {
            let limit = self.constant(block, pc, 32);
            let inside = self.compare(
                block,
                pc,
                ShaderIntegerComparison::Less,
                self.read(amount),
                limit,
            );
            let fallback = if signed {
                let count = self.cast(block, pc, mask, Type::U32, Type::I32);
                let sign = self.binary(block, pc, Binary::ShiftRight, Type::I32, base_typed, count);
                self.cast(block, pc, sign, Type::I32, Type::U32)
            } else {
                self.constant(block, pc, 0)
            };
            bits = self.select(block, pc, Type::U32, inside, bits, fallback);
        }
        self.write(block, pc, destination, bits);
    }
    fn extract(&mut self, block: BlockId, pc: ShaderSourceLocation, op: &D) {
        let D::ExtractBitfield32 {
            destination,
            value: source,
            offset,
            count,
            signed,
            reverse,
        } = *op
        else {
            unreachable!("non-extract decoder operation")
        };
        let mut base = self.read(source);
        if reverse {
            base = self.value(
                block,
                pc,
                Op::Unary {
                    op: Unary::BitReverse,
                    ty: Type::U32,
                },
                vec![base],
            );
        }
        let zero = self.constant(block, pc, 0);
        let one = self.constant(block, pc, 1);
        let thirtyone = self.constant(block, pc, 31);
        let width = self.constant(block, pc, 32);
        let inside = self.compare(
            block,
            pc,
            ShaderIntegerComparison::Less,
            self.read(offset),
            width,
        );
        let position = self.select(block, pc, Type::U32, inside, self.read(offset), thirtyone);
        let minus = self.constant(block, pc, u32::MAX);
        let negative = self.binary(block, pc, Binary::Multiply, Type::U32, position, minus);
        let remaining = self.binary(block, pc, Binary::Add, Type::U32, width, negative);
        let shorter = self.compare(
            block,
            pc,
            ShaderIntegerComparison::Less,
            self.read(count),
            remaining,
        );
        let size = self.select(block, pc, Type::U32, shorter, self.read(count), remaining);
        let size_masked = self.binary(block, pc, Binary::And, Type::U32, size, thirtyone);
        let field_mask = self.binary(block, pc, Binary::ShiftLeft, Type::U32, one, size_masked);
        let field_mask = self.binary(block, pc, Binary::Add, Type::U32, field_mask, minus);
        let full = self.compare(block, pc, ShaderIntegerComparison::Equal, size, width);
        let field_mask = self.select(block, pc, Type::U32, full, minus, field_mask);
        let mut value = self.binary(block, pc, Binary::ShiftRight, Type::U32, base, position);
        value = self.binary(block, pc, Binary::And, Type::U32, value, field_mask);
        if signed {
            let neg_size = self.binary(block, pc, Binary::Multiply, Type::U32, size, minus);
            let shift = self.binary(block, pc, Binary::Add, Type::U32, width, neg_size);
            let shift = self.binary(block, pc, Binary::And, Type::U32, shift, thirtyone);
            let extended = self.binary(block, pc, Binary::ShiftLeft, Type::U32, value, shift);
            let extended = self.cast(block, pc, extended, Type::U32, Type::I32);
            let shift = self.cast(block, pc, shift, Type::U32, Type::I32);
            let extended = self.binary(block, pc, Binary::ShiftRight, Type::I32, extended, shift);
            value = self.cast(block, pc, extended, Type::I32, Type::U32);
            let base = self.cast(block, pc, base, Type::U32, Type::I32);
            let sign_count = self.cast(block, pc, thirtyone, Type::U32, Type::I32);
            let sign = self.binary(block, pc, Binary::ShiftRight, Type::I32, base, sign_count);
            let sign = self.cast(block, pc, sign, Type::I32, Type::U32);
            value = self.select(block, pc, Type::U32, inside, value, sign);
        } else {
            value = self.select(block, pc, Type::U32, inside, value, zero);
        }
        let empty = self.compare(
            block,
            pc,
            ShaderIntegerComparison::Equal,
            self.read(count),
            zero,
        );
        value = self.select(block, pc, Type::U32, empty, zero, value);
        self.write(block, pc, destination, value);
    }
    fn sample(
        &mut self,
        block: BlockId,
        pc: ShaderSourceLocation,
        op: &D,
    ) -> Result<(), MaxwellShaderTranslationError> {
        let (outputs, coordinates, image, sampler, lod, layer, offset, mode) = match op {
            D::SampleTexture2D {
                outputs,
                coordinates,
                image_binding,
                sampler_binding,
                lod,
                offset,
            } => (
                outputs.as_ref(),
                coordinates.as_slice(),
                *image_binding,
                Some(*sampler_binding),
                *lod,
                None,
                *offset,
                if lod.is_some() {
                    SampleMode::Explicit
                } else {
                    SampleMode::Implicit
                },
            ),
            D::SampleTexture2DArray {
                outputs,
                coordinates,
                image_binding,
                sampler_binding,
                lod,
                array_index,
                offset,
            } => (
                outputs.as_ref(),
                coordinates.as_slice(),
                *image_binding,
                Some(*sampler_binding),
                *lod,
                Some(*array_index),
                *offset,
                if lod.is_some() {
                    SampleMode::Explicit
                } else {
                    SampleMode::Implicit
                },
            ),
            D::SampleTextureCube {
                outputs,
                coordinates,
                image_binding,
                sampler_binding,
                lod,
            } => (
                outputs.as_ref(),
                coordinates.as_slice(),
                *image_binding,
                Some(*sampler_binding),
                *lod,
                None,
                None,
                if lod.is_some() {
                    SampleMode::Explicit
                } else {
                    SampleMode::Implicit
                },
            ),
            D::LoadTexture2D {
                outputs,
                coordinates,
                image_binding,
                mip_level,
            } => (
                outputs.as_ref(),
                coordinates.as_slice(),
                *image_binding,
                None,
                None,
                None,
                None,
                SampleMode::Fetch(*mip_level),
            ),
            D::SampleDepthCompare2D {
                outputs,
                coordinates,
                image_binding,
                sampler_binding,
                reference,
                array_index,
                offset,
                level_zero,
            } => (
                outputs.as_ref(),
                coordinates.as_slice(),
                *image_binding,
                Some(*sampler_binding),
                Some(*reference),
                *array_index,
                *offset,
                if *level_zero {
                    SampleMode::CompareLevelZero
                } else {
                    SampleMode::Compare
                },
            ),
            _ => unreachable!("non-texture decoder operation"),
        };
        let mut arguments: Vec<_> = coordinates
            .iter()
            .map(|&r| {
                if matches!(mode, SampleMode::Fetch(_)) {
                    self.cast(block, pc, self.read(r), Type::U32, Type::I32)
                } else {
                    self.float(block, pc, r)
                }
            })
            .collect();
        if let Some(layer) = layer {
            let mask = self.constant(block, pc, 0xffff);
            arguments.push(self.binary(block, pc, Binary::And, Type::U32, self.read(layer), mask));
        }
        if let Some(lod) = lod {
            arguments.push(self.float(block, pc, lod));
        }
        let kind = self.resource_kinds.get(&image).copied().ok_or_else(|| {
            MaxwellShaderTranslationError::Ssa(nixe_gpu::ssa::Error {
                stage: None,
                source: pc,
                detail: "sampled image has no decoder resource declaration".into(),
            })
        })?;
        let values = self.builder.emit(
            block,
            pc,
            Op::TextureSample {
                image: ResourceId::new(image),
                sampler: sampler.map(ResourceId::new),
                kind,
                mode,
                offset,
            },
            arguments,
        );
        for output in outputs {
            self.write_float(
                block,
                pc,
                output.destination(),
                values[output.component() as usize].into(),
            );
        }
        Ok(())
    }
}
fn scalar_type32(
    ty: ShaderScalarType,
    pc: ShaderSourceLocation,
) -> Result<Type, MaxwellShaderTranslationError> {
    Ok(match ty {
        ShaderScalarType::Unsigned32 => Type::U32,
        ShaderScalarType::Signed32 => Type::I32,
        ShaderScalarType::Float32 => Type::F32,
        _ => {
            return Err(MaxwellShaderTranslationError::Ssa(nixe_gpu::ssa::Error {
                stage: None,
                source: pc,
                detail: "decoded operation has unsupported scalar width".into(),
            }));
        }
    })
}
