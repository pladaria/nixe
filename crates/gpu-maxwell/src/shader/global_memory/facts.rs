//! Must-facts qualified by one hardware predicate. A write forgets unrelated
//! guards (their truth may overlap), and predicate replacement retires its old
//! qualifications. This is bounded by the ISA's seven predicate registers;
//! neither path combinations nor executable variants are enumerated.
use super::*;
type Guard = Option<(u8, bool)>;

fn guard(predicate: ShaderPredicate) -> Guard {
    match predicate {
        ShaderPredicate::Register { register, inverted } => Some((register, inverted)),
        ShaderPredicate::Always => None,
        ShaderPredicate::Never => unreachable!("never-executed operations have no transfer"),
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct Facts(BTreeMap<(ShaderRegister, Guard), Value>);
impl Facts {
    pub fn get(&self, register: ShaderRegister, predicate: ShaderPredicate) -> Option<Value> {
        self.0
            .get(&(register, guard(predicate)))
            .or_else(|| self.0.get(&(register, None)))
            .copied()
    }
    pub fn forget(&mut self, register: ShaderRegister) {
        let keys: Vec<_> = self
            .0
            .range((register, None)..=(register, Some((6, true))))
            .map(|(&key, _)| key)
            .collect();
        for key in keys {
            self.0.remove(&key);
        }
    }
    pub fn write(
        &mut self,
        register: ShaderRegister,
        predicate: ShaderPredicate,
        value: Option<Value>,
    ) {
        let condition = guard(predicate);
        let opposite = condition.map(|(p, inverted)| (p, !inverted));
        let preserved = opposite.and_then(|(register_index, inverted)| {
            self.get(
                register,
                ShaderPredicate::Register {
                    register: register_index,
                    inverted,
                },
            )
        });
        self.forget(register);
        if condition.is_some()
            && let Some(value) = value
            && Some(value) == preserved
        {
            self.0.insert((register, None), value);
        } else {
            if let Some(value) = preserved {
                self.0.insert((register, opposite), value);
            }
            if let Some(value) = value {
                self.0.insert((register, condition), value);
            }
        }
    }
    pub fn intersect(&mut self, other: &Self) {
        let keys: Vec<_> = self.0.keys().chain(other.0.keys()).copied().collect();
        let mut result = BTreeMap::new();
        for (register, condition) in keys {
            let predicate = condition.map_or(ShaderPredicate::Always, |(register, inverted)| {
                ShaderPredicate::Register { register, inverted }
            });
            if let Some(value) = self.get(register, predicate)
                && other.get(register, predicate) == Some(value)
            {
                result.insert((register, condition), value);
            }
        }
        self.0 = result;
    }
    fn forget_predicate(&mut self, predicate: u8) {
        self.0
            .retain(|&(_, guard), _| guard.is_none_or(|(p, _)| p != predicate));
    }
    pub fn invalidate_predicate_writes(&mut self, operation: &ShaderOperation) {
        match operation {
            ShaderOperation::SetPredicateFloat32 { destination, .. } => {
                self.forget_predicate(*destination)
            }
            ShaderOperation::SetPredicateBoolean { destinations, .. }
            | ShaderOperation::SetPredicateInteger32 { destinations, .. } => {
                for &p in destinations.iter().flatten() {
                    self.forget_predicate(p);
                }
            }
            ShaderOperation::SetPredicatesFromBits { write_mask, .. } => {
                for p in 0..7 {
                    if write_mask & (1 << p) != 0 {
                        self.forget_predicate(p);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const R: ShaderRegister = ShaderRegister::new(0);
    const V: Value = Value::BaseWord(0, 0x100);
    const P: ShaderPredicate = ShaderPredicate::Register {
        register: 0,
        inverted: false,
    };
    const N: ShaderPredicate = ShaderPredicate::Register {
        register: 0,
        inverted: true,
    };
    #[test]
    fn guarded_definitions_never_authorize_the_opposite_or_unconditional_access() {
        let mut facts = Facts::default();
        facts.write(R, P, Some(V));
        assert_eq!(facts.get(R, P), Some(V));
        assert_eq!(facts.get(R, N), None);
        assert_eq!(facts.get(R, ShaderPredicate::Always), None);
        facts.forget_predicate(0);
        assert_eq!(facts.get(R, P), None);
    }
    #[test]
    fn conditional_clobbers_preserve_only_the_opposite_arm() {
        let mut facts = Facts::default();
        facts.write(R, ShaderPredicate::Always, Some(V));
        facts.write(R, P, None);
        assert_eq!(facts.get(R, N), Some(V));
        assert_eq!(facts.get(R, P), None);
        facts.write(R, P, Some(V));
        assert_eq!(facts.get(R, ShaderPredicate::Always), Some(V));
    }
    #[test]
    fn unrelated_guards_may_overlap_and_cannot_preserve_a_clobbered_fact() {
        let mut facts = Facts::default();
        facts.write(R, P, Some(V));
        facts.write(
            R,
            ShaderPredicate::Register {
                register: 1,
                inverted: false,
            },
            None,
        );
        assert_eq!(facts.get(R, P), None);
    }
    #[test]
    fn joins_keep_only_facts_proven_on_both_incoming_edges() {
        let mut unqualified = Facts::default();
        unqualified.write(R, ShaderPredicate::Always, Some(V));
        let mut qualified = Facts::default();
        qualified.write(R, P, Some(V));
        let mut a = unqualified.clone();
        a.intersect(&qualified);
        qualified.intersect(&unqualified);
        assert_eq!(a, qualified);
        assert_eq!(a.get(R, P), Some(V));
        assert_eq!(a.get(R, ShaderPredicate::Always), None);
    }
}
