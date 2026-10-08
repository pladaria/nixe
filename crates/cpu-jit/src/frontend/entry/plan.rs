//! Compilation-time contracts copied from an incoming native edge. These are
//! allocation preferences, never callable roots or publication dependencies.

use crate::abi::{
    ExitStateMap, GuestValue, LazyFlags, NzcvLocation, RegisterClass, ValueBinding, ValueLocation,
};
use crate::analysis::StateSet;
use cranelift_codegen::nixe::EntryConstraint;

#[derive(Clone, Debug, Default)]
pub(crate) struct Plan {
    bindings: Vec<ValueBinding>,
    flags: Option<(u8, ValueLocation)>,
}

impl Plan {
    pub(crate) fn without(mut self, discard: StateSet) -> Self {
        self.bindings.retain(|binding| {
            binding
                .value
                .state()
                .unwrap()
                .intersection(discard)
                .is_empty()
        });
        if let Some((mask, location)) = self.flags {
            let mask = mask & !discard.nzcv;
            self.flags = (mask != 0).then_some((mask, location));
        }
        self
    }

    pub(crate) fn from_exit(source: &ExitStateMap) -> Self {
        let bindings = source
            .bindings
            .iter()
            .copied()
            .filter(|binding| {
                matches!(binding.location, ValueLocation::Register { .. })
                    && !source
                        .dirty_live
                        .intersection(binding.value.state().unwrap())
                        .is_empty()
                    && !matches!(
                        binding.value,
                        GuestValue::Fpsr | GuestValue::Fpcr | GuestValue::TpidrroEl0
                    )
            })
            .collect();
        let flags = match source.nzcv {
            NzcvLocation::Packed(location)
            | NzcvLocation::Deferred(
                LazyFlags::Packed(location) | LazyFlags::Canonical(location),
            ) if matches!(location, ValueLocation::Register { .. })
                && source.dirty_live.nzcv != 0 =>
            {
                Some((source.dirty_live.nzcv, location))
            }
            _ => None,
        };
        Self { bindings, flags }
    }

    pub(crate) fn carry(&self) -> StateSet {
        let mut state = StateSet::default();
        for binding in &self.bindings {
            state = state.union(binding.value.state().unwrap());
        }
        state.nzcv = self.flags.map_or(0, |(mask, _)| mask);
        state
    }

    pub(crate) fn len(&self) -> usize {
        self.bindings.len() + usize::from(self.flags.is_some())
    }

    pub(crate) fn constraints(&self, inputs: &[GuestValue], flags: bool) -> Vec<EntryConstraint> {
        // Exit maps may alias guest values (e.g. MOV X1,X0). Independent entry
        // definitions cannot share a physical register: constrain one, let the
        // allocator choose the others and let the ordinary bridge copy aliases.
        let mut occupied = [0_u32; 2];
        let mut constraint = |location: Option<ValueLocation>| {
            if let Some(ValueLocation::Register { class, index }) = location {
                let vector = class == RegisterClass::Vector;
                let bank = &mut occupied[usize::from(vector)];
                if *bank & (1 << index) == 0 {
                    *bank |= 1 << index;
                    return EntryConstraint::Register { index, vector };
                }
            }
            EntryConstraint::Any
        };
        let mut result = Vec::with_capacity(inputs.len() + usize::from(flags));
        for guest in inputs {
            result.push(constraint(
                self.bindings
                    .iter()
                    .find(|binding| binding.value == *guest)
                    .map(|binding| binding.location),
            ));
        }
        if flags {
            result.push(constraint(self.flags.map(|(_, location)| location)));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{CodeVersion, ExitSiteKey, HostAbi};

    #[test]
    fn aliased_exit_values_and_flags_do_not_create_overlapping_entry_definitions() {
        let register = ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 3,
        };
        let mut live = StateSet::default();
        live.integer.x.insert(0);
        live.integer.x.insert(1);
        live.nzcv = crate::analysis::NZCV;
        let source = ExitStateMap {
            site: ExitSiteKey {
                source: CodeVersion::new(1).unwrap(),
                state_map: 0,
            },
            abi: HostAbi::X86_64,
            live,
            dirty_live: live,
            bindings: vec![
                ValueBinding {
                    value: GuestValue::General(0),
                    location: register,
                },
                ValueBinding {
                    value: GuestValue::General(1),
                    location: register,
                },
            ]
            .into(),
            nzcv: NzcvLocation::Deferred(LazyFlags::Packed(register)),
            host_fpsr_pending: false,
        };
        let plan = Plan::from_exit(&source);
        assert_eq!(plan.carry(), live);
        assert_eq!(
            plan.constraints(&[GuestValue::General(0), GuestValue::General(1)], true),
            vec![
                EntryConstraint::Register {
                    index: 3,
                    vector: false
                },
                EntryConstraint::Any,
                EntryConstraint::Any,
            ]
        );
        // Pruned input zero must not reserve its register against input one.
        assert_eq!(
            plan.constraints(&[GuestValue::General(1)], false),
            vec![EntryConstraint::Register {
                index: 3,
                vector: false
            }]
        );
    }
}
