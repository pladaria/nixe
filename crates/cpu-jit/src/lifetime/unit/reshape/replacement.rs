//! Frozen predecessor evidence for the existing replacement publisher. These
//! worker-local snapshots retain storage; validation still rejects retired code.

use super::*;
use crate::abi::ReachabilityVersion;
use crate::hcq::Graph;

pub(in crate::lifetime) struct Fallback {
    pub key: BlockKey,
    pub slot: Handle<DispatchSlot>,
    pub previous: UnitEntry,
    pub reachability: ReachabilityVersion,
    pub baseline: Snapshot,
}

pub(in crate::lifetime) struct Replacement {
    pub predecessors: [Option<Snapshot>; 2],
    pub fallbacks: Vec<Fallback>,
}

impl Replacement {
    /// Preserve public labels still covered by this candidate. Looking up the
    /// current slot owner costs at most two identity comparisons, not an entry scan.
    pub fn retains_entry(&self, slot: &DispatchSlot) -> bool {
        slot.owners[1].is_some_and(|owner| {
            self.predecessors
                .iter()
                .flatten()
                .any(|previous| previous.registered_handle() == Some(owner.unit))
        })
    }

    pub fn has_predecessors(&self) -> bool {
        self.predecessors.iter().any(Option::is_some)
    }

    /// Allocate immutable fallback payloads before publication. Their exact
    /// baselines and previous HCQ owners are checked again at the commit point.
    /// No destructor taking the cache mutex may run under JIT state.
    pub fn prepare_payloads(&self, process: &Lifetime) -> Result<Option<StagedPayloads>, Error> {
        if self.fallbacks.is_empty() {
            return Ok(None);
        }
        let values = Vec::with_capacity(self.fallbacks.len());
        let bytes = values.capacity() * size_of::<DispatchPayload>();
        let mut values = process.cache.account(values, bytes, Tier::Hcq)?;
        {
            let mut state = process.lock();
            state.running()?;
            self.validate(&state)?;
            for fallback in &self.fallbacks {
                let result = state.reachabilities.next_id();
                let identity = process.publication_identity(&mut state, result, Tier::Hcq)?;
                let old = state.dispatch.get(fallback.slot).unwrap().snapshot();
                values
                    .value
                    .push(DispatchPayload::new(identity, old.lcq(), None));
            }
        }
        let payloads = values
            .iter()
            .map(|value| {
                process
                    .cache
                    .account(
                        value.clone(),
                        size_of::<Accounted<DispatchPayload>>(),
                        Tier::Hcq,
                    )
                    .map(|value| Some(Box::new(value)))
            })
            .collect::<Result<Box<[_]>, _>>()?;
        let bytes = size_of_val(&*payloads);
        Ok(Some(process.cache.account(payloads, bytes, Tier::Hcq)?))
    }

    /// Called after all fallible publication checks and before new membership.
    /// Work validation has checked these exact live, reserved participants.
    pub fn retire(&self, state: &mut State, sequence: Option<MaintenanceSequence>, graph: &Graph) {
        for predecessor in self.predecessors.iter().flatten() {
            let handle = predecessor.registered_handle().unwrap().0;
            let units = &mut state.units;
            let family = units.records.get(handle).unwrap().family.unwrap();
            let mut previous = None;
            for instruction in predecessor.instructions.iter() {
                if graph.contains(instruction.key) {
                    continue; // Transfer selected membership in place below.
                }
                assert!(units.family_owners.remove(instruction.key, family));
                let page = negative::SelectionPage::of(instruction.key.block_key());
                if previous != Some(page) {
                    units.negatives.invalidate_selection(page);
                    previous = Some(page);
                }
            }
            let record = units.records.get_mut(handle).unwrap();
            record.lifecycle = Lifecycle::Superseded;
            record.queue_retirement(
                handle,
                &mut units.retirements,
                &mut units.negatives,
                Reason::TierCutover,
                sequence.expect("replacement registered before reachable mutation"),
            );
        }
    }

    pub fn publish_fallbacks(&self, state: &mut State, payloads: Option<&mut StagedPayloads>) {
        let Some(payloads) = payloads else {
            debug_assert!(self.fallbacks.is_empty());
            return;
        };
        for (fallback, payload) in self.fallbacks.iter().zip(payloads.value.iter_mut()) {
            state.units.invalidate_entry_negatives(fallback.key);
            state
                .units
                .negatives
                .invalidate(negative::Owner::Dispatch(fallback.slot));
            let slot = state.dispatch.get_mut(fallback.slot).unwrap();
            // No new unit references this slot: predecessor retirement retains
            // its existing slot count until actual reclamation. The LCQ root
            // is still owned by its original baseline record.
            let owners = [slot.owners[0], None];
            *payload = Some(slot.replace(payload.take().unwrap(), owners));
        }
    }

    pub fn new(predecessors: [Option<Snapshot>; 2]) -> Self {
        // At most the two participant entry sets; no maximum-size allocation
        // and no additional code/state-map copies. Prepare outside JIT state.
        let capacity = predecessors
            .iter()
            .flatten()
            .map(|unit| unit.entries.len())
            .sum();
        Self {
            predecessors,
            fallbacks: Vec::with_capacity(capacity),
        }
    }

    pub fn capture_fallbacks(
        &mut self,
        state: &State,
        selected: impl Fn(BlockKey) -> bool,
    ) -> Result<(), Error> {
        for predecessor in self.predecessors.iter().flatten() {
            let unit = predecessor
                .registered_handle()
                .ok_or(Error::StalePublication)?;
            for (index, entry) in predecessor.entries.iter().enumerate() {
                if selected(entry.key) {
                    continue; // The selected label already owns a captured graph input.
                }
                let slot_handle = *state.keys.get(&entry.key).ok_or(Error::StalePublication)?;
                let slot = state
                    .dispatch
                    .get(slot_handle)
                    .ok_or(Error::StalePublication)?;
                if !slot.owners[1].is_some_and(|owner| owner.unit == unit && owner.index == index) {
                    return Err(Error::StalePublication);
                }
                let payload = slot.snapshot();
                let owner = slot.owners[0].ok_or(Error::StalePublication)?;
                let record = state
                    .units
                    .lcq_record(
                        owner,
                        entry.key,
                        payload.lcq().ok_or(Error::StalePublication)?,
                    )
                    .ok_or(Error::StalePublication)?;
                self.fallbacks.push(Fallback {
                    key: entry.key,
                    slot: slot_handle,
                    previous: UnitEntry { unit, index },
                    reachability: payload.reachability(),
                    baseline: Snapshot::retain(&record.code),
                });
            }
        }
        Ok(())
    }

    pub fn validate(&self, state: &State) -> Result<(), Error> {
        // The candidate validates participant identities first. These baseline
        // entries can lie wholly outside the new graph and need their own check.
        for fallback in &self.fallbacks {
            if state.keys.get(&fallback.key) != Some(&fallback.slot) {
                return Err(Error::StalePublication);
            }
            let slot = state
                .dispatch
                .get(fallback.slot)
                .ok_or(Error::StalePublication)?;
            let payload = slot.snapshot();
            let (Some(owner), Some(entry)) = (slot.owners[0], payload.lcq()) else {
                return Err(Error::StalePublication);
            };
            if payload.reachability() != fallback.reachability
                || !slot.owners[1].is_some_and(|owner| {
                    owner.unit == fallback.previous.unit && owner.index == fallback.previous.index
                })
                || !state
                    .units
                    .matches_lcq(owner, fallback.key, entry, &fallback.baseline)
            {
                return Err(Error::StalePublication);
            }
        }
        Ok(())
    }

    pub fn unchanged(&self, graph: &Graph, entries: &[usize]) -> bool {
        let [Some(previous), None] = &self.predecessors else {
            return false; // An initial region or a merge is always a new partition.
        };
        // Instructions are unique; membership lookup is indexed. Entry order
        // (including a different logical root) is not a partition change.
        previous.instructions.len() == graph.instructions.len()
            && previous
                .instructions
                .iter()
                .all(|word| graph.contains(word.key))
            && previous.entries.len() == entries.len()
            && self.fallbacks.is_empty()
    }
}
