//! Shared guest-value storage and deterministic SSA operand layout for both tiers.

use crate::abi::GuestValue;
use crate::analysis::StateSet;
use crate::jit_error::Error;
use cranelift_codegen::ir::{self, types};

#[derive(Default)]
pub(crate) struct Values {
    pub registers: [Option<ir::Value>; 32],
    pub vectors: [Option<ir::Value>; 32],
    pub system: [Option<ir::Value>; 3],
}

impl Values {
    pub(crate) fn get(&self, guest: GuestValue) -> Result<ir::Value, Error> {
        let value = match guest {
            GuestValue::Vector(index) => self.vectors[usize::from(index)],
            GuestValue::Fpcr | GuestValue::TpidrEl0 | GuestValue::TpidrroEl0 => {
                self.system[system_index(guest)]
            }
            _ => self.registers[register_index(guest)],
        };
        value.ok_or_else(|| {
            Error::internal(format!(
                "native input {guest:?} missing from shared liveness"
            ))
        })
    }

    /// Bind an ingress/phi or a lowered definition. The caller owns may-dirty
    /// state; binding a block parameter is not an architectural register write.
    pub(crate) fn bind(&mut self, guest: GuestValue, value: ir::Value) {
        match guest {
            GuestValue::Vector(index) => self.vectors[usize::from(index)] = Some(value),
            GuestValue::Fpcr | GuestValue::TpidrEl0 | GuestValue::TpidrroEl0 => {
                self.system[system_index(guest)] = Some(value)
            }
            _ => self.registers[register_index(guest)] = Some(value),
        }
    }
}

pub(crate) fn register_operands(state: StateSet) -> Vec<GuestValue> {
    (0..31)
        .filter(|&index| state.integer.x[index])
        .map(|index| GuestValue::General(index as u8))
        .chain(state.integer.sp.then_some(GuestValue::Sp))
        .chain(
            (0..32)
                .filter(|&index| state.vector[index])
                .map(|index| GuestValue::Vector(index as u8)),
        )
        .chain(state.fpcr.then_some(GuestValue::Fpcr))
        .chain(state.tpidr_el0.then_some(GuestValue::TpidrEl0))
        .chain(state.tpidrro_el0.then_some(GuestValue::TpidrroEl0))
        .collect()
}
pub(crate) fn guest_type(guest: GuestValue) -> ir::Type {
    match guest {
        GuestValue::Vector(_) => types::I8X16,
        GuestValue::Fpcr => types::I32,
        _ => types::I64,
    }
}
pub(crate) fn system_index(guest: GuestValue) -> usize {
    match guest {
        GuestValue::Fpcr => 0,
        GuestValue::TpidrEl0 => 1,
        GuestValue::TpidrroEl0 => 2,
        _ => unreachable!(),
    }
}
pub(crate) fn register_index(guest: GuestValue) -> usize {
    match guest {
        GuestValue::General(index) => usize::from(index),
        GuestValue::Sp => 31,
        _ => unreachable!(),
    }
}
