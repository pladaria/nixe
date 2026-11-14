//! Portable warp matrix arithmetic. One helper per shape avoids duplicating the
//! collective at each instruction site and bounds register pressure for K=16.
use super::*;
use crate::ir::Program;
use spirv::{Decoration, FunctionControl, GlslStd450Op as G, LoopControl, StorageClass};

impl EmitContext<'_> {
    pub(super) fn matrix_instruction(&mut self, ir: &Ir, id: InstId) -> Result<Option<Id>> {
        let inst = ir.inst(id)?;
        let function = self.matrix_functions[matrix_variant(inst.flags)?];
        if function == Id::NONE {
            return Err(invariant("Matrix helper was not defined"));
        }
        let mut operands = [function, Id::NONE, Id::NONE, Id::NONE];
        for (index, operand) in operands[1..].iter_mut().enumerate() {
            *operand = self.def(ir, inst.arg(index)?)?;
        }
        self.op(Op::FunctionCall, self.types.u32[1], &operands)
            .map(Some)
    }

    pub(in crate::spirv) fn define_matrix_functions(&mut self, ir: &Ir, p: &Program) -> Result<()> {
        let mut needed = [false; 2];
        for &block in &p.blocks {
            for &id in ir.block(block)?.instructions() {
                let inst = ir.inst(id)?;
                if inst.opcode() == O::MatrixMultiplyF16 {
                    needed[matrix_variant(inst.flags)?] = true;
                }
            }
        }
        if needed.iter().any(|&v| v)
            && (!self.profile.supports_subgroup_stage(self.stage)
                || self.profile.has_broken_spirv_subgroup_shuffle)
        {
            return Err(unsupported(
                "Matrix arithmetic requires subgroup shuffle support",
            ));
        }
        for (variant, needed) in needed.into_iter().enumerate() {
            if needed {
                self.builder
                    .capability(spirv::Capability::GroupNonUniform)?;
                self.builder
                    .capability(spirv::Capability::GroupNonUniformShuffle)?;
                self.matrix_functions[variant] = self.define_matrix_function(variant)?;
            }
        }
        Ok(())
    }

    fn matrix_u32_imm(&mut self, op: Op, value: Id, constant: u32) -> Result<Id> {
        let constant = self.const_u32(constant)?;
        self.op(op, self.types.u32[0], &[value, constant])
    }

    fn matrix_round_half(&mut self, value: Id) -> Result<Id> {
        // Explicit round-to-nearest-even at each K=8 boundary. An unpack(pack())
        // round trip can be optimized away by drivers. Subnormals use the 2^-24 grid.
        let u = self.types.u32[0];
        let f = self.types.f32[0];
        let bits = self.op(Op::Bitcast, u, &[value])?;
        let magnitude = self.matrix_u32_imm(Op::BitwiseAnd, bits, 0x7fffffff)?;
        let tie = self.matrix_u32_imm(Op::ShiftRightLogical, magnitude, 13)?;
        let tie = self.matrix_u32_imm(Op::BitwiseAnd, tie, 1)?;
        let rounded = self.matrix_u32_imm(Op::IAdd, magnitude, 0xfff)?;
        let rounded = self.op(Op::IAdd, u, &[rounded, tie])?;
        let rounded = self.matrix_u32_imm(Op::BitwiseAnd, rounded, !0x1fff)?;
        let limit = self.const_u32(0x47800000)?;
        let infinity = self.const_u32(0x7f800000)?;
        let overflow = self.op(Op::UGreaterThanEqual, self.types.bool, &[rounded, limit])?;
        let normal = self.op(Op::Select, u, &[overflow, infinity, rounded])?;
        let magnitude_float = self.op(Op::Bitcast, f, &[magnitude])?;
        let scale = self.const_f32(16777216.0)?;
        let scaled = self.op(Op::FMul, f, &[magnitude_float, scale])?;
        let rounded_float = self.ext(G::RoundEven, f, &[scaled])?;
        let scale = self.const_f32(1.0 / 16777216.0)?;
        let subnormal = self.op(Op::FMul, f, &[rounded_float, scale])?;
        let subnormal = self.op(Op::Bitcast, u, &[subnormal])?;
        let min_normal = self.const_u32(0x38800000)?;
        let is_normal = self.op(
            Op::UGreaterThanEqual,
            self.types.bool,
            &[magnitude, min_normal],
        )?;
        let finite = self.op(Op::Select, u, &[is_normal, normal, subnormal])?;
        let non_finite = self.op(
            Op::UGreaterThanEqual,
            self.types.bool,
            &[magnitude, infinity],
        )?;
        let result = self.op(Op::Select, u, &[non_finite, magnitude, finite])?;
        let sign = self.matrix_u32_imm(Op::BitwiseAnd, bits, 0x80000000)?;
        let result = self.op(Op::BitwiseOr, u, &[result, sign])?;
        self.op(Op::Bitcast, f, &[result])
    }

    fn matrix_group(
        &mut self,
        inputs: &[Id; 6],
        accumulators: &[Id; 4],
        a_base: Id,
        b_base: Id,
        upper: Option<Id>,
    ) -> Result<()> {
        let u = self.types.u32[0];
        let f = self.types.f32[0];
        let scope = self.const_u32(3)?;
        let pick = |ctx: &mut Self, low, high| match upper {
            Some(upper) => ctx.op(Op::Select, u, &[upper, high, low]),
            None => Ok(low),
        };
        let b = pick(self, inputs[4], inputs[5])?;
        let a = [
            pick(self, inputs[0], inputs[2])?,
            pick(self, inputs[1], inputs[3])?,
        ];
        for component in 0..4 {
            let mut bs = [Id::NONE; 2];
            for (column, bs) in bs.iter_mut().enumerate() {
                let lane = self.matrix_u32_imm(Op::IAdd, b_base, component + column as u32 * 4)?;
                let shuffled = self.op(Op::GroupNonUniformShuffle, u, &[scope, b, lane])?;
                *bs = self.ext(G::UnpackHalf2x16, self.types.f32[1], &[shuffled])?;
            }
            let lane = self.matrix_u32_imm(Op::IAdd, a_base, component)?;
            for row in 0..2 {
                let shuffled = self.op(Op::GroupNonUniformShuffle, u, &[scope, a[row], lane])?;
                let a = self.ext(G::UnpackHalf2x16, self.types.f32[1], &[shuffled])?;
                for column in 0..2 {
                    let accumulator = accumulators[row * 2 + column];
                    let mut value = self.op(Op::Load, f, &[accumulator])?;
                    for half in 0..2 {
                        let a = self.extract(f, a, half)?;
                        let b = self.extract(f, bs[column], half)?;
                        value = self.ext(G::Fma, f, &[a, b, value])?;
                        self.builder
                            .decorate(value, Decoration::NoContraction, &[])?;
                    }
                    self.builder
                        .instruction(Op::Store, &[accumulator.0, value.0])?;
                }
            }
        }
        for &accumulator in accumulators {
            let value = self.op(Op::Load, f, &[accumulator])?;
            let rounded = self.matrix_round_half(value)?;
            self.builder
                .instruction(Op::Store, &[accumulator.0, rounded.0])?;
        }
        Ok(())
    }

    fn define_matrix_function(&mut self, variant: usize) -> Result<Id> {
        // HMMA.1688/16816.F16: four half accumulators per lane. Each K=8
        // group uses four row providers, with two half products per provider.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
        let u = self.types.u32[0];
        let f = self.types.f32[0];
        let ft = self.builder.type_function(
            self.types.u32[1],
            &[self.types.u32[3], self.types.u32[1], self.types.u32[1]],
        )?;
        let function = self.builder.result(
            Op::Function,
            self.types.u32[1],
            &[FunctionControl::DONT_INLINE.bits(), ft.0],
        )?;
        let a = self.builder.parameter(self.types.u32[3])?;
        let b = self.builder.parameter(self.types.u32[1])?;
        let c = self.builder.parameter(self.types.u32[1])?;
        self.builder.new_label()?;
        let up = self.builder.type_pointer(StorageClass::Function, u)?;
        let fp = self.builder.type_pointer(StorageClass::Function, f)?;
        let counter = self.builder.variable(up, StorageClass::Function, None)?;
        let mut accumulators = [Id::NONE; 4];
        for accumulator in &mut accumulators {
            *accumulator = self.builder.variable(fp, StorageClass::Function, None)?;
        }
        let mut inputs = [Id::NONE; 6];
        for (index, input) in inputs.iter_mut().enumerate() {
            *input = self.extract(u, if index < 4 { a } else { b }, (index % 4) as u32)?;
        }
        for row in 0..2 {
            let packed = self.extract(u, c, row)?;
            let unpacked = self.ext(G::UnpackHalf2x16, self.types.f32[1], &[packed])?;
            for column in 0..2 {
                let value = self.extract(f, unpacked, column)?;
                self.builder.instruction(
                    Op::Store,
                    &[accumulators[(row * 2 + column) as usize].0, value.0],
                )?;
            }
        }
        self.builder
            .instruction(Op::Store, &[counter.0, self.constants.u32_zero.0])?;
        let lane = self.op(Op::Load, u, &[self.objects.subgroup_local_invocation_id])?;
        let warp = self.matrix_u32_imm(Op::BitwiseAnd, lane, !31)?;
        let row = self.matrix_u32_imm(Op::BitwiseAnd, lane, 28)?;
        let column = self.matrix_u32_imm(Op::BitwiseAnd, lane, 3)?;
        let column = self.matrix_u32_imm(Op::ShiftLeftLogical, column, 3)?;
        let a_base = self.op(Op::IAdd, u, &[warp, row])?;
        let b_base = self.op(Op::IAdd, u, &[warp, column])?;
        if variant == 0 {
            self.matrix_group(&inputs, &accumulators, a_base, b_base, None)?;
        } else {
            let header = self.builder.reserve_id()?;
            let body = self.builder.reserve_id()?;
            let next = self.builder.reserve_id()?;
            let end = self.builder.reserve_id()?;
            self.builder.instruction(Op::Branch, &[header.0])?;
            self.builder.label(header)?;
            let group = self.op(Op::Load, u, &[counter])?;
            let two = self.const_u32(2)?;
            let more = self.op(Op::ULessThan, self.types.bool, &[group, two])?;
            self.builder.instruction(
                Op::LoopMerge,
                &[end.0, next.0, LoopControl::DONT_UNROLL.bits()],
            )?;
            self.builder
                .instruction(Op::BranchConditional, &[more.0, body.0, end.0])?;
            self.builder.label(body)?;
            let one = self.const_u32(1)?;
            let upper = self.op(Op::IEqual, self.types.bool, &[group, one])?;
            self.matrix_group(&inputs, &accumulators, a_base, b_base, Some(upper))?;
            self.builder.instruction(Op::Branch, &[next.0])?;
            self.builder.label(next)?;
            let increment = self.op(Op::IAdd, u, &[group, one])?;
            self.builder
                .instruction(Op::Store, &[counter.0, increment.0])?;
            self.builder.instruction(Op::Branch, &[header.0])?;
            self.builder.label(end)?;
        }
        let mut packed = [Id::NONE; 2];
        for (row, packed) in packed.iter_mut().enumerate() {
            let a = self.op(Op::Load, f, &[accumulators[row * 2]])?;
            let b = self.op(Op::Load, f, &[accumulators[row * 2 + 1]])?;
            let pair = self.op(Op::CompositeConstruct, self.types.f32[1], &[a, b])?;
            *packed = self.ext(G::PackHalf2x16, u, &[pair])?;
        }
        let result = self.op(Op::CompositeConstruct, self.types.u32[1], &packed)?;
        self.builder.instruction(Op::ReturnValue, &[result.0])?;
        self.builder.end_function()?;
        Ok(function)
    }
}

fn matrix_variant(depth: u32) -> Result<usize> {
    match depth {
        8 => Ok(0),
        16 => Ok(1),
        _ => Err(invariant("Invalid matrix multiply depth")),
    }
}
