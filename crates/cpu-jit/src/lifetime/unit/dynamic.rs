//! Cold preparation of an exact indirect transfer. Preparation pins immutable
//! source/target contracts, but does not make a native address callable in a PIC.
//! Cache insertion must revalidate under state before exposing the bridge.

use super::*;
use crate::abi::{AdmissionEpoch, BlockKey, ExitSiteKey, ReachabilityVersion};
use crate::lifetime::{NativeSuspension, State};

pub(in crate::lifetime) mod pic;

/// A different source map/version or target specialization/version is a
/// different transfer even when the guest target PC happens to be identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BridgeKey {
    pub source: ExitSiteKey,
    pub target: BlockKey,
    pub reachability: ReachabilityVersion,
    pub target_version: CodeVersion,
}

pub(crate) struct PreparedBridge<'p> {
    process: &'p Lifetime,
    admission: AdmissionEpoch,
    source: UnitHandle,
    target: UnitHandle,
    source_code: Arc<Accounted<CodeUnit>>,
    target_code: Arc<Accounted<CodeUnit>>,
    target_entry: usize,
    key: BridgeKey,
}

/// Unpublished executable owner. Dropping it returns the actual transfer span
/// and both strong unit references. Empty transfers retain the same ownership
/// contract but use the target fast ingress directly, without an allocation.
pub(crate) struct PreparedTransfer<'p> {
    prepared: PreparedBridge<'p>,
    code: Option<Box<Installed>>,
}

impl NativeSuspension<'_> {
    /// Resolve/install for subsequent native hits. This cold call returns only
    /// canonical ingress: System-ABI dispatch cannot retain source registers.
    pub(crate) fn resolve_bridge(
        &mut self,
        source: UnitHandle,
        state_map: u32,
        target: BlockKey,
    ) -> Result<Option<PublishedEntry>, Error> {
        let process = Arc::clone(&self.reader.process);
        let Some(prepared) = process.prepare_dynamic_bridge(source, state_map, target)? else {
            return Ok(None);
        };
        let entry = prepared
            .target_code
            .entry(&prepared.target_code.entries[prepared.target_entry]);
        match self.cache_bridge(prepared) {
            Ok(()) => Ok(Some(entry)),
            // A bridge is optional reachability, not permission to exceed the
            // hard budget. The epoch still protects canonical ingress; do not
            // wait for reclamation while holding that epoch on a cache miss.
            Err(Error::Capacity(_)) => Ok(Some(entry)),
            Err(error) => Err(error),
        }
    }
}

impl Lifetime {
    /// Resolve a BR/BLR/RET destination in canonical mode, without waiting,
    /// compilation or guest-address-to-host-pointer conversion. Admission and
    /// both strong acquisitions share one lock interval. A missing/withdrawing
    /// target remains a cold miss, not a cached negative result.
    pub(crate) fn prepare_dynamic_bridge(
        &self,
        source: UnitHandle,
        state_map: u32,
        target: BlockKey,
    ) -> Result<Option<PreparedBridge<'_>>, Error> {
        let state = self.lock();
        let admission = state.open()?;
        let from = links::eligible(&state, self, source)?;
        let map = from
            .code
            .states
            .get(state_map as usize)
            .ok_or(Error::InvalidUnit("dynamic source map is absent"))?;
        if !map
            .transfer
            .as_ref()
            .is_some_and(|transfer| transfer.static_target.is_none())
            || !map.exit.is_some_and(|exit| {
                matches!(
                    exit.kind,
                    EdgeKind::Indirect | EdgeKind::Call | EdgeKind::Return
                )
            })
            || from.code.instructions[0].key.block_key().at(target.pc) != Some(target)
        {
            return Err(Error::InvalidUnit(
                "dynamic source or target execution key does not match",
            ));
        }
        let Some(slot) = state
            .keys
            .get(&target)
            .and_then(|slot| state.dispatch.get(*slot))
        else {
            return Ok(None);
        };
        let payload = slot.snapshot();
        if payload.preferred().is_none() {
            return Ok(None);
        }
        let owner = slot.owners[usize::from(payload.hcq().is_some())]
            .ok_or(Error::InvalidUnit("dynamic target has no registered owner"))?;
        let to = match links::eligible(&state, self, owner.unit) {
            Ok(to) => to,
            Err(Error::StaleUnit) => return Ok(None),
            Err(error) => return Err(error),
        };
        if from.code.code.metadata.abi != to.code.code.metadata.abi {
            return Err(Error::InvalidUnit("dynamic link host ABIs differ"));
        }
        Ok(Some(PreparedBridge {
            process: self,
            admission,
            source,
            target: owner.unit,
            source_code: Arc::clone(&from.code),
            target_code: Arc::clone(&to.code),
            target_entry: owner.index,
            key: BridgeKey {
                source: map.state.site,
                target,
                reachability: payload.reachability(),
                target_version: to.code.version,
            },
        }))
    }
}

impl<'p> PreparedBridge<'p> {
    pub(crate) fn key(&self) -> BridgeKey {
        self.key
    }

    fn validate(&self, state: &State) -> Result<(), Error> {
        if state.open()? != self.admission {
            return Err(Error::StalePublication);
        }
        let from = links::eligible(state, self.process, self.source)?;
        let to = links::eligible(state, self.process, self.target)?;
        if !Arc::ptr_eq(&from.code, &self.source_code)
            || !Arc::ptr_eq(&to.code, &self.target_code)
            || links::target_payload(state, to, self.target_entry)?.reachability()
                != self.key.reachability
        {
            return Err(Error::StalePublication);
        }
        Ok(())
    }

    /// Emission and W^X installation run outside JIT state. No PIC entry owns
    /// this preparation yet; closure/retirement may win while it is emitted.
    pub(crate) fn emit(self) -> Result<PreparedTransfer<'p>, Error> {
        self.validate(&self.process.lock())?;
        let code = bridge::install(
            self.process,
            &self.source_code,
            self.key.source.state_map,
            &self.target_code,
            self.target_entry,
            bridge::Tail::DynamicInline,
        )?;
        Ok(PreparedTransfer {
            prepared: self,
            code,
        })
    }
}

impl PreparedTransfer<'_> {
    // This validation is deliberately separate from emission. The eventual
    // PIC insertion must hold this same guard through attaching the root and
    // writing the native record; validating and unlocking first is not enough.
    pub(in crate::lifetime) fn validate(&self, state: &State) -> Result<(), Error> {
        self.prepared.validate(state)
    }

    pub(crate) fn address(&self) -> usize {
        self.code.as_ref().map_or_else(
            || {
                let target = &self.prepared.target_code;
                target.code.allocation.address()
                    + target.entries[self.prepared.target_entry].fast_offset as usize
            },
            |code| code.allocation.address(),
        )
    }
}

#[cfg(test)]
mod tests;
