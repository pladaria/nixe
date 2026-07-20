//! A32-to-IR translation.

use crate::{decode::DecodedOpcode, ir::builder::IrBuilder, location::DecodedInstruction};

use super::block::{LiftOutcome, direct_branch_target};

pub(crate) fn lift(
    _builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    match decoded.instruction.pattern().name {
        "nop" => LiftOutcome::Continue,
        "b" => LiftOutcome::Terminate(super::block::direct_branch(
            direct_branch_target(decoded)
                .expect("validated A32 branch displacement always produces an aligned target"),
        )),
        _ => LiftOutcome::Interpret(decoded.instruction.coverage_id()),
    }
}
