//! Reshape-only discovery evidence. Retain weak identities, including inputs
//! later dropped by graph trimming, not another set of code snapshots/words.

use super::*;
use crate::hcq::Graph;
use crate::lifetime::unit::reshape::negative::{Owner, SelectionPage};

pub(crate) struct Inspection {
    key: BlockKey,
    version: ReachabilityVersion,
    unit: unit::UnitHandle,
    instructions: usize,
    leaders: std::ops::Range<usize>,
}

impl Demanded {
    pub fn inspection(&self, extent: &Extent, leaders: &mut Vec<u16>) -> Inspection {
        debug_assert!(extent.instructions <= self.unit.instructions.len());
        let start = leaders.len();
        leaders.extend(extent.leaders.iter().map(|key| {
            u16::try_from((key.pc.get() - self.key.pc.get()) / 4)
                .expect("validated LCQ image ordinal")
        }));
        Inspection {
            key: self.key,
            version: self.version,
            unit: self.unit.registered_handle().unwrap(),
            instructions: extent.instructions,
            leaders: start..leaders.len(),
        }
    }
}

/// A third family's immutable membership, not a temporary compiler claim.
/// The family unit's existing retirement/cutover watch revokes this evidence.
pub(crate) struct Blocked {
    key: crate::abi::InstructionKey,
    unit: unit::UnitHandle,
}

impl Blocked {
    pub(super) fn capture(state: &State, key: crate::abi::InstructionKey) -> Result<Self, Error> {
        Ok(Self {
            key,
            unit: state
                .units
                .active_family_owner(key)
                .ok_or(Error::StalePublication)?,
        })
    }
}

pub(crate) struct DiscoveryEvidence {
    inspected: Vec<Inspection>,
    blocked: Vec<Blocked>,
    // One packed worker-only vector, not an allocation per inspected input.
    leaders: Vec<u16>,
    // Cap-excluded inputs are known evidence; unavailable frontier inputs are
    // not. A no-op additionally requires all eligible words inside its graph.
    complete_inputs: bool,
    // Actual vector capacity; storage is destroyed before returning the charge.
    _charge: MetadataLease,
}

impl Work<'_> {
    /// Short-lived pins for checking structural results against executable
    /// memory, including images discarded by trimming. Never put these pins in
    /// the resident negative index. Allocate/charge and destroy outside state.
    pub fn capture_discovery_inputs(
        &self,
        evidence: &DiscoveryEvidence,
    ) -> Result<Accounted<Vec<unit::Snapshot>>, Error> {
        self.capacity()?;
        let mut seen = std::collections::HashSet::new();
        let inputs: Vec<_> = evidence
            .inspected
            .iter()
            .filter(|input| seen.insert(input.unit))
            .collect();
        let pins = Vec::with_capacity(inputs.len());
        let bytes = pins.capacity() * size_of::<unit::Snapshot>();
        let mut pins = self.process.cache.account(pins, bytes, Tier::Hcq)?;
        let state = self.process.lock();
        self.validate(&state)?;
        let allowed = self.allowed_families();
        evidence.validate_inputs(&state, true, |key| {
            state.units.instruction_available(key, allowed)
        })?;
        for input in &inputs {
            let slot = state
                .dispatch
                .get(*state.keys.get(&input.key).unwrap())
                .unwrap();
            let pin = state
                .units
                .pin_inspected_lcq(
                    slot.owners[0].unwrap(),
                    input.key,
                    slot.snapshot().lcq().unwrap(),
                    input.unit,
                )
                .ok_or(Error::StalePublication)?;
            pins.value.push(pin);
        }
        drop(state);
        Ok(pins)
    }

    pub fn check_discovery(&self, evidence: &DiscoveryEvidence) -> Result<(), Error> {
        self.capacity()?;
        let state = self.process.lock();
        self.validate_discovery_locked(&state, evidence)
    }

    pub(super) fn validate_discovery_locked(
        &self,
        state: &State,
        evidence: &DiscoveryEvidence,
    ) -> Result<(), Error> {
        self.validate(state)?;
        let allowed = self.allowed_families();
        evidence.validate_inputs(state, true, |key| {
            state.units.instruction_available(key, allowed)
        })
    }

    /// Discovery builds this vector as worker scratch. Charge its actual
    /// capacity when transferring it into the captured graph, outside JIT state.
    pub fn discovery_evidence(
        &self,
        inspected: Vec<Inspection>,
        blocked: Vec<Blocked>,
        leaders: Vec<u16>,
        complete_inputs: bool,
    ) -> Result<DiscoveryEvidence, Error> {
        let bytes = inspected
            .capacity()
            .checked_mul(size_of::<Inspection>())
            .and_then(|bytes| {
                blocked
                    .capacity()
                    .checked_mul(size_of::<Blocked>())
                    .and_then(|frontiers| bytes.checked_add(frontiers))
            })
            .and_then(|bytes| {
                leaders
                    .capacity()
                    .checked_mul(size_of::<u16>())
                    .and_then(|leaders| bytes.checked_add(leaders))
            })
            .ok_or(Error::Capacity("discovery evidence size overflow"))?;
        let charge = self.process.cache.charge_metadata(bytes, Tier::Hcq)?;
        Ok(DiscoveryEvidence {
            inspected,
            blocked,
            leaders,
            complete_inputs,
            _charge: charge,
        })
    }
}

impl DiscoveryEvidence {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inspected.len()
    }

    pub(in crate::lifetime) fn owner_capacity(&self) -> usize {
        self.inspected.len() * 2 + self.blocked.len()
    }

    pub(super) fn selection_capacity(&self) -> usize {
        self.inspected
            .iter()
            .map(|input| SelectionPage::span(input.key, input.instructions).len())
            .sum()
    }

    pub(super) fn append_selection(&self, owners: &mut Vec<Owner>) {
        for input in &self.inspected {
            owners.extend(
                SelectionPage::span(input.key, input.instructions).map(Owner::SelectionPage),
            );
        }
    }

    pub(super) fn append_entry_pages(&self, owners: &mut Vec<Owner>) {
        for input in &self.inspected {
            owners.extend(SelectionPage::span(input.key, input.instructions).map(Owner::EntryPage));
        }
    }

    pub(super) fn validate_backend(
        &self,
        state: &State,
        allowed: [Option<crate::sampling::FamilyIdentity>; 2],
    ) -> Result<(), Error> {
        if !self.complete_inputs {
            return Err(Error::StalePublication);
        }
        self.validate_inputs(state, true, |key| {
            state.units.instruction_available(key, allowed)
        })
    }

    /// Called under the final result guard, independently of retained graph
    /// inputs. Every eligible prefix must be within the unchanged family;
    /// foreign membership cuts are justified by named, still-live blockers.
    pub(in crate::lifetime) fn validate_no_op(
        &self,
        state: &State,
        graph: &Graph,
    ) -> Result<(), Error> {
        if !self.complete_inputs {
            return Err(Error::StalePublication);
        }
        self.validate_inputs(state, false, |key| graph.contains(key))
    }

    fn validate_inputs(
        &self,
        state: &State,
        selection: bool,
        mut eligible: impl FnMut(crate::abi::InstructionKey) -> bool,
    ) -> Result<(), Error> {
        for blocked in &self.blocked {
            if state.units.active_family_owner(blocked.key) != Some(blocked.unit) {
                return Err(Error::StalePublication);
            }
        }
        for input in &self.inspected {
            let slot = state
                .keys
                .get(&input.key)
                .and_then(|h| state.dispatch.get(*h))
                .ok_or(Error::StalePublication)?;
            let payload = slot.snapshot();
            let (Some(owner), Some(entry)) = (slot.owners[0], payload.lcq()) else {
                return Err(Error::StalePublication);
            };
            if payload.reachability() != input.version {
                return Err(Error::StalePublication);
            }
            let instructions = state
                .units
                .inspected_lcq(owner, input.key, entry, input.unit)
                .ok_or(Error::StalePublication)?;
            // No-op requires membership in its unchanged graph; structural
            // evidence instead checks discovery's allowed ownership. Neither
            // validation silently drops an acquired, subsequently trimmed input.
            let mut leaders = self.leaders[input.leaders.clone()]
                .iter()
                .copied()
                .peekable();
            for (index, word) in instructions.iter().take(input.instructions).enumerate() {
                if !eligible(word.key) {
                    return Err(Error::StalePublication);
                }
                if selection {
                    let captured = leaders
                        .peek()
                        .is_some_and(|&leader| usize::from(leader) == index);
                    let demanded = state
                        .keys
                        .get(&word.key.block_key())
                        .and_then(|handle| state.dispatch.get(*handle))
                        .is_some_and(|slot| slot.snapshot().lcq().is_some());
                    if captured != demanded {
                        return Err(Error::StalePublication);
                    }
                    if captured {
                        leaders.next();
                    }
                }
            }
            debug_assert!(!selection || leaders.next().is_none());
        }
        Ok(())
    }

    /// Caller has validated this evidence under the same guard. The caller's
    /// vector was sized before locking; deduplication happens outside state.
    pub(in crate::lifetime) fn append_owners(&self, state: &State, owners: &mut Vec<Owner>) {
        for input in &self.inspected {
            owners.extend([
                Owner::Unit(input.unit),
                Owner::Dispatch(*state.keys.get(&input.key).unwrap()),
            ]);
        }
        owners.extend(self.blocked.iter().map(|blocked| Owner::Unit(blocked.unit)));
    }
}
