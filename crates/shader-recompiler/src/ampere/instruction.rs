//! Native 128-bit encoding and semantic control flow.
//! Encoding reference: https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h
use crate::{
    CompileError, Environment, Result,
    diagnostics::FailureCategory,
    ir::{Condition, FlowTest, Pred},
    types::ProgramOffset,
};

#[derive(Clone, Copy, Debug)]
pub struct Instruction(pub u128);
impl Instruction {
    pub fn fetch(env: &mut impl Environment, pc: ProgramOffset) -> Result<Self> {
        if pc.0 & 15 != 0 {
            return Err(unsupported("Unaligned native shader instruction"));
        }
        let high =
            pc.0.checked_add(8)
                .ok_or_else(|| unsupported("Shader address overflow"))?;
        Ok(Self(
            u128::from(env.read_instruction(pc)?)
                | (u128::from(env.read_instruction(ProgramOffset(high))?) << 64),
        ))
    }
    pub fn bits(self, offset: u32, count: u32) -> u32 {
        ((self.0 >> offset) & ((1u128 << count) - 1)) as u32
    }
    pub fn opcode(self) -> u32 {
        self.bits(0, 12)
    }
    pub fn condition(self) -> Result<Condition> {
        // Reconvergence membership does not predicate an invocation effect.
        if matches!(self.opcode(), 0x941 | 0x942 | 0x945) {
            return Ok(Condition::always(true));
        }
        // R2UR's lane predicate selects the broadcast source; it must not
        // exclude the other active lanes from the subgroup operation.
        if self.opcode() == 0x3c2 {
            return Ok(Condition::always(true));
        }
        let index = self.bits(12, 3) as u8;
        Ok(Condition {
            flow_test: FlowTest::T,
            predicate: if matches!(
                self.opcode(),
                0xc82
                    | 0x882
                    | 0x88c
                    | 0x28c
                    | 0x290
                    | 0x890
                    | 0x292
                    | 0x892
                    | 0x2a4
                    | 0x4a4
                    | 0x8a4
                    | 0x2a5
                    | 0x899
                    | 0xab9
                    | 0xabb
            ) {
                Pred::uniform(index)?
            } else {
                Pred::new(index)?
            },
            negated: self.bits(15, 1) != 0,
        })
    }
    pub fn branch_target(self, pc: ProgramOffset) -> Result<ProgramOffset> {
        // Relative 50-bit byte displacement, from the following instruction.
        if self.bits(32, 2) != 0
            || self.bits(82, 5)
                != if matches!(self.opcode(), 0x944 | 0x950) {
                    16
                } else {
                    0
                }
            || (self.opcode() != 0x947 && self.bits(87, 4) != 7)
            || self.bits(91, 1) != 0
        {
            return Err(unsupported(
                "Native branch convergence/extra predicate mode",
            ));
        }
        let raw = ((self.0 >> 32) & ((1u128 << 50) - 1)) as u64;
        let relative = ((raw << 14) as i64) >> 14;
        let target = i64::from(pc.0) + 16 + relative;
        if target < 0 || target > i64::from(u32::MAX) || target & 15 != 0 {
            return Err(unsupported("Native branch target outside program offsets"));
        }
        Ok(ProgramOffset(target as u32))
    }
    /// Native BRA has an instruction predicate and a separate branch predicate.
    /// Use the existing predicate-before block for their conjunction, without
    /// inventing a temporary predicate register or rewriting shader instructions.
    pub fn branch_predicates(self) -> Result<(Condition, Condition)> {
        let primary = self.condition()?;
        let secondary = if self.opcode() == 0x947 {
            Condition {
                flow_test: FlowTest::T,
                predicate: Pred::new(self.bits(87, 3) as u8)?,
                negated: self.bits(90, 1) != 0,
            }
        } else {
            Condition::always(true)
        };
        Ok(
            if primary == Condition::always(false) || secondary == Condition::always(false) {
                (Condition::always(true), Condition::always(false))
            } else if secondary == Condition::always(true) {
                (Condition::always(true), primary)
            } else if primary == Condition::always(true) {
                (Condition::always(true), secondary)
            } else {
                (primary, secondary)
            },
        )
    }
}
pub(crate) fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Unsupported, message)
}
