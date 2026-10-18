mod atomic;
mod convert;
mod dispatch;
mod float;
mod half;
mod integer;
mod lut;
mod memory;
mod predicate;
mod special;
mod stage;
mod surface;
mod texture;
mod texture_swizzled;
mod video;

use super::{
    bits::extract,
    decode::{DecodedInstruction, Opcode},
    instruction::Instruction,
    operands::Operands,
};
use crate::{
    CompileError, Result,
    arena::BlockId,
    diagnostics::FailureCategory,
    ir::{Emitter, Ir, Pred, Value},
};

fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Unsupported, message)
}

#[derive(Clone, Copy)]
enum Source {
    Reg0,
    Reg20,
    Reg39,
    Cbuf,
    Imm20,
    Imm32,
    Imm16,
}

struct Visitor<'a> {
    metadata: &'a crate::environment::EnvironmentMetadata,
    emit: Emitter<'a>,
    instruction: Instruction,
}

impl Visitor<'_> {
    fn field<const START: u32, const WIDTH: u32>(&self) -> u32 {
        extract::<START, WIDTH>(self.instruction.0) as u32
    }
    fn bit<const BIT: u32>(&self) -> bool {
        self.field::<BIT, 1>() != 0
    }
    fn source(&mut self, source: Source) -> Result<Value> {
        let mut operands = Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        };
        match source {
            Source::Reg0 => operands.read_u32(self.instruction.reg0()),
            Source::Reg20 => operands.reg20(),
            Source::Reg39 => operands.reg39(),
            Source::Cbuf => operands.cbuf(),
            Source::Imm20 => Ok(operands.imm20()),
            Source::Imm32 => Ok(operands.imm32()),
            Source::Imm16 => Ok(Value::U32(self.field::<20, 16>())),
        }
    }
    fn a(&mut self) -> Result<Value> {
        self.emit.get_reg(self.instruction.reg8())
    }
    fn write(&mut self, value: Value) -> Result<()> {
        self.emit.set_reg(self.instruction.reg0(), value)?;
        Ok(())
    }
    fn predicate<const START: u32, const NEG: u32>(&mut self) -> Result<Value> {
        self.emit.get_pred(
            Pred::new(self.field::<START, 3>() as u8)?,
            self.bit::<NEG>(),
        )
    }
    fn set_predicate<const START: u32>(&mut self, value: Value) -> Result<()> {
        self.emit
            .set_pred(Pred::new(self.field::<START, 3>() as u8)?, value)
    }
    fn flags_zs(&mut self, value: Value, pseudo: bool) -> Result<()> {
        let z = if pseudo {
            self.emit.get_zero_from_op(value)?
        } else {
            self.emit.i_equal(value, Value::U32(0))?
        };
        let s = if pseudo {
            self.emit.get_sign_from_op(value)?
        } else {
            self.emit.i_less_than(value, Value::U32(0), true)?
        };
        self.emit.set_z_flag(z)?;
        self.emit.set_s_flag(s)?;
        Ok(())
    }
    fn clear_co(&mut self) -> Result<()> {
        self.emit.set_c_flag(Value::U1(false))?;
        self.emit.set_o_flag(Value::U1(false))?;
        Ok(())
    }
    fn flags_arithmetic(&mut self, value: Value) -> Result<()> {
        self.flags_zs(value, true)?;
        let c = self.emit.get_carry_from_op(value)?;
        let o = self.emit.get_overflow_from_op(value)?;
        self.emit.set_c_flag(c)?;
        self.emit.set_o_flag(o)?;
        Ok(())
    }
}

/// Translate a decoded instruction inside its already structured predicate path.
pub fn translate_instruction(
    metadata: &crate::environment::EnvironmentMetadata,
    ir: &mut Ir,
    block: BlockId,
    decoded: DecodedInstruction,
) -> Result<()> {
    let mut visitor = Visitor {
        metadata,
        emit: Emitter::new(ir, block)?,
        instruction: decoded.instruction,
    };
    visitor.dispatch(decoded.opcode)
}

pub struct MaxwellTranslator;

impl<E: crate::Environment> super::asl::CodeTranslator<E> for MaxwellTranslator {
    fn translate(
        &mut self,
        environment: &mut E,
        ir: &mut Ir,
        block: BlockId,
        begin: super::location::Location,
        end: super::location::Location,
    ) -> Result<()> {
        let mut location = begin;
        while location < end {
            let decoded = super::decode::fetch(environment, location)?;
            translate_instruction(environment.metadata(), ir, block, decoded).map_err(|error| {
                error.at(
                    environment.metadata().stage,
                    location.address(),
                    Some(decoded.instruction.0),
                )
            })?;
            location = location.next();
        }
        Ok(())
    }
}
