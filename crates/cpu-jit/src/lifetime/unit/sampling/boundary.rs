//! Classify a real transfer using current point-lookup identities only. No
//! successor discovery, slot creation, guest reads or family/unit scans.

use super::*;
use crate::sampling::{BoundaryKey, FamilyIdentity};

pub(in crate::lifetime::unit) struct Endpoint {
    pub(in crate::lifetime::unit) payload: DispatchPayload,
    pub(in crate::lifetime::unit) family: Option<FamilyIdentity>,
    pub(in crate::lifetime::unit) unit: UnitHandle,
}

impl Lifetime {
    /// `block` is the actual logical source block, not the invocation entry or
    /// necessarily an HCQ public entry. Both endpoint reachabilities must come
    /// from existing demanded dispatch slots; absent identities are not invented.
    /// The caller protects the source unit and supplies an executed terminal.
    pub(crate) fn sample_transfer(
        &self,
        unit: &CodeUnit,
        block: BlockKey,
        instruction: InstructionKey,
        samples: &mut Samples,
        edge: ObservedEdge,
    ) -> Result<(), Error> {
        let Some(state) = self.sample_state()? else {
            return Ok(());
        };
        let handle = unit
            .registered_handle()
            .ok_or(Error::InvalidUnit("sample source is unpublished"))?;
        if handle.1 != self.identity {
            return Err(Error::InvalidUnit(
                "sample source belongs to another process",
            ));
        }
        let Some(record) = state.units.records.get(handle.0).filter(|record| {
            live(record) && record.code.id == unit.id && record.code.version == unit.version
        }) else {
            return Ok(());
        };
        if block.at(instruction.block_key().pc) != Some(instruction.block_key()) {
            return Err(Error::InvalidUnit("sample source mixes execution contexts"));
        }
        let Some(source) = endpoint(&state, block) else {
            return Ok(());
        };
        let Some(source_family) = family(&state, instruction) else {
            return Ok(());
        };
        match unit.tier {
            Tier::Lcq => {
                if unit.entries.first().map(|entry| entry.key) != Some(block)
                    || !source
                        .payload
                        .lcq()
                        .is_some_and(|entry| entry.unit == unit.id && entry.version == unit.version)
                {
                    return Ok(());
                }
                // LCQ images are contiguous; verify the actual instruction in
                // constant time rather than searching retained code images.
                let present = instruction
                    .block_key()
                    .pc
                    .get()
                    .checked_sub(block.pc.get())
                    .and_then(|offset| usize::try_from(offset / 4).ok())
                    .and_then(|index| unit.instructions.get(index))
                    .is_some_and(|image| image.key == instruction);
                if !present {
                    return Err(Error::InvalidUnit(
                        "sample instruction is absent from LCQ source",
                    ));
                }
            }
            Tier::Hcq => {
                let Some(owner) = record
                    .family
                    .and_then(|owner| state.units.families.get(owner))
                else {
                    return Ok(());
                };
                let identity = Some(FamilyIdentity {
                    id: owner.id,
                    version: owner.version,
                });
                if source_family != identity || source.family != identity {
                    return Ok(());
                }
            }
        }
        let target = block
            .at(edge.destination)
            .and_then(|key| endpoint(&state, key).map(|entry| (key, entry)));
        if let Some((key, target)) = target {
            // A native edge between public entries of the same HCQ family is
            // not a reshape boundary. A retained LCQ entry is different, even
            // when its instruction is already owned by that same family.
            if unit.tier == Tier::Hcq
                && target.payload.hcq().is_some()
                && source_family == target.family
            {
                return Ok(());
            }
            if unit.tier == Tier::Hcq || source_family.is_some() || target.family.is_some() {
                let boundary = BoundaryKey {
                    source: instruction,
                    target: InstructionKey::new(key).unwrap(),
                    source_version: source.payload.reachability(),
                    target_version: target.payload.reachability(),
                    source_family,
                    target_family: target.family,
                };
                let queue = state.background_queue.upgrade();
                // Carry only verified value identities across admission. Its
                // capacity/queue/state try-locks must not nest under this guard.
                drop(state);
                if let Some(snapshot) = samples.boundary(boundary, queue.is_some())
                    && let Some(queue) = queue
                {
                    self.admit_reshape(&queue, samples, block, snapshot)?;
                }
                return Ok(());
            }
        }
        if unit.tier == Tier::Lcq
            && source_family.is_none()
            && source.family.is_none()
            && source.payload.hcq().is_none()
        {
            // An undemanded/misaligned target can still be an observed seed
            // successor, but cannot supply a versioned reshape endpoint.
            let queue = state.background_queue.upgrade();
            let version = source.payload.reachability();
            drop(state);
            self.sample_seed(queue, samples, block, version, Some(edge))?;
        }
        Ok(())
    }
}

fn live(record: &UnitRecord) -> bool {
    record.lifecycle == Lifecycle::Published && record.retirement.is_none()
}

// Outer None means an ownership entry is no longer eligible; it must never be
// confused with Some(None), which proves there is no current family owner.
pub(in crate::lifetime::unit) fn family(
    state: &State,
    instruction: InstructionKey,
) -> Option<Option<FamilyIdentity>> {
    let Some(handle) = state.units.family_owners.get(instruction) else {
        return Some(None);
    };
    let family = state.units.families.get(handle)?;
    let record = state
        .units
        .records
        .get(family.unit.registered_handle()?.0)?;
    (live(record)
        && record.family == Some(handle)
        && record.code.id == family.unit.id
        && record.code.version == family.unit.version)
        .then_some(Some(FamilyIdentity {
            id: family.id,
            version: family.version,
        }))
}

pub(in crate::lifetime::unit) fn endpoint(state: &State, key: BlockKey) -> Option<Endpoint> {
    let slot = state.dispatch.get(*state.keys.get(&key)?)?;
    let payload = slot.snapshot();
    let entry = payload.preferred()?;
    let owner = slot.owners[usize::from(payload.hcq().is_some())]?;
    let record = state.units.records.get(owner.unit.0)?;
    if !live(record)
        || record.code.id != entry.unit
        || record.code.version != entry.version
        || record.code.entries.get(owner.index)?.key != key
    {
        return None;
    }
    let family = family(state, InstructionKey::new(key)?)?;
    if let Some(hcq) = payload.hcq()
        && family
            != Some(FamilyIdentity {
                id: hcq.family,
                version: hcq.family_version,
            })
    {
        return None;
    }
    Some(Endpoint {
        payload,
        family,
        unit: owner.unit,
    })
}

#[cfg(test)]
mod tests;
