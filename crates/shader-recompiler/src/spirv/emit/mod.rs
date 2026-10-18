use super::{EmitContext, Id};
use crate::{
    CompileError, Result,
    arena::InstId,
    diagnostics::FailureCategory,
    ir::{Ir, Opcode as O, Value},
};
use spirv::{GlslStd450Op, Op};
mod arithmetic;
mod atomic;
mod attributes;
mod buffers;
mod convert;
mod direct;
mod float;
mod image;
mod image_access;
mod image_operands;
mod route;
mod shared;
mod special;
mod warp;
pub use route::{EmissionRoute, emission_route};

fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::UnsupportedReferenceCase, message)
}
fn invariant(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Internal, message)
}
fn literal(ir: &Ir, value: Value) -> Result<u32> {
    match ir.resolve(value)? {
        Value::U32(v) => Ok(v),
        _ => Err(invariant("SPIR-V literal operand must be an immediate U32")),
    }
}
impl EmitContext<'_> {
    pub(super) fn op(&mut self, op: Op, ty: Id, args: &[Id]) -> Result<Id> {
        let mut words = [0; 8];
        if args.len() > words.len() {
            return Err(invariant("Too many SPIR-V operation operands"));
        }
        for (word, id) in words.iter_mut().zip(args) {
            *word = id.0;
        }
        self.builder.result(op, ty, &words[..args.len()])
    }
    pub(super) fn ext(&mut self, op: GlslStd450Op, ty: Id, args: &[Id]) -> Result<Id> {
        self.builder.glsl(ty, op, args)
    }
    pub(super) fn extract(&mut self, ty: Id, composite: Id, index: u32) -> Result<Id> {
        self.builder
            .result(Op::CompositeExtract, ty, &[composite.0, index])
    }
    pub(super) fn pseudo(
        &mut self,
        ir: &mut Ir,
        producer: InstId,
        opcode: O,
        value: Id,
    ) -> Result<()> {
        if let Some(id) = ir.associated_pseudo(producer, opcode)? {
            ir.inst_mut(id)?.definition = value.0;
            ir.invalidate(id)?;
        }
        Ok(())
    }
    pub fn emit_instruction(&mut self, ir: &mut Ir, id: InstId) -> Result<()> {
        let opcode = ir.inst(id)?.opcode();
        use EmissionRoute as R;
        let result = match emission_route(opcode) {
            R::Noop => return Ok(()),
            R::Invariant => {
                return Err(invariant(format!(
                    "Unlowered or unconsumed IR operation {} reached SPIR-V emission",
                    opcode.name()
                )));
            }
            R::Unsupported => {
                return Err(unsupported(format!(
                    "Unsupported SPIR-V operation {}",
                    opcode.name()
                )));
            }
            R::Identity => {
                let value = self.def(ir, ir.inst(id)?.arg(0)?)?;
                if value == Id::NONE {
                    return Err(unsupported("Forward identity declaration"));
                }
                value
            }
            R::Phi => {
                let inst = ir.inst(id)?;
                if !matches!(inst.ty(), crate::ir::Type::U1 | crate::ir::Type::U32) {
                    return Err(unsupported("SPIR-V phi supports only U1 and U32"));
                }
                let mut parents = Vec::with_capacity(inst.num_args());
                for i in 0..inst.num_args() {
                    parents.push(Id(ir.block(inst.phi_block(i)?)?.definition));
                }
                if parents.contains(&Id::NONE) {
                    return Err(invariant("Phi predecessor label is not defined"));
                }
                let phi = self
                    .builder
                    .deferred_phi(self.types.get(inst.ty())?, &parents)?;
                let result = phi.id;
                self.deferred_phis.push((id, phi));
                result
            }
            R::Direct => {
                let direct = direct::direct(opcode)
                    .ok_or_else(|| invariant("Missing direct emitter route"))?;
                let mut args = [Id::NONE; 5];
                for (dest, &(index, is_literal)) in args.iter_mut().zip(direct.args) {
                    let value = ir.inst(id)?.arg(index)?;
                    *dest = if is_literal {
                        Id(literal(ir, value)?)
                    } else {
                        self.def(ir, value)?
                    };
                }
                let ty = self.types.get(direct.ty)?;
                let result = match direct.operation {
                    direct::Operation::Core(Op::Undef) => self.builder.undef(ty)?,
                    direct::Operation::Core(op) => self.op(op, ty, &args[..direct.args.len()])?,
                    direct::Operation::Extended(op) => {
                        self.ext(op, ty, &args[..direct.args.len()])?
                    }
                };
                if direct.decorate
                    && crate::ir::FpControl::from_bits(ir.inst(id)?.flags)?.no_contraction
                {
                    self.builder
                        .decorate(result, spirv::Decoration::NoContraction, &[])?;
                }
                result
            }
            route => {
                let value = match route {
                    R::Arithmetic => self.arithmetic(ir, id)?,
                    R::Buffer => self.buffer_instruction(ir, id)?,
                    R::Shared => self.shared_instruction(ir, id)?,
                    R::Attribute => self.attribute_instruction(ir, id)?,
                    R::Special => self.special_instruction(ir, id)?,
                    R::Atomic => self.atomic_instruction(ir, id)?,
                    R::Warp => self.warp_instruction(ir, id)?,
                    R::Image => self.image_instruction(ir, id)?,
                    _ => unreachable!(),
                };
                value.ok_or_else(|| {
                    invariant(format!("Missing emitter route for {}", opcode.name()))
                })?
            }
        };
        ir.inst_mut(id)?.definition = result.0;
        Ok(())
    }
    pub fn patch_phis(&mut self, ir: &Ir) -> Result<()> {
        for (id, phi) in std::mem::take(&mut self.deferred_phis) {
            let inst = ir.inst(id)?;
            let mut values = Vec::with_capacity(inst.num_args());
            for i in 0..inst.num_args() {
                values.push(self.def(ir, inst.arg(i)?)?);
            }
            self.builder.patch_phi(phi, &values)?;
        }
        Ok(())
    }
    pub fn assemble(&self) -> Result<Vec<u32>> {
        let mut words = self.builder.assemble()?;
        if !self.opcode_rewrites.is_empty() {
            let rewrites: std::collections::HashMap<_, _> = self
                .opcode_rewrites
                .iter()
                .map(|(id, op)| (id.0, *op))
                .collect();
            let mut offset = 5;
            while offset < words.len() {
                let len = (words[offset] >> 16) as usize;
                let op = words[offset] & 0xffff;
                if len >= 3
                    && (op == Op::GroupNonUniformShuffleXor as u32
                        || op == Op::GroupNonUniformQuadBroadcast as u32)
                    && let Some(replacement) = rewrites.get(&words[offset + 2])
                {
                    words[offset] = (words[offset] & 0xffff0000) | *replacement as u32;
                }
                offset += len;
            }
        }
        Ok(words)
    }
}
