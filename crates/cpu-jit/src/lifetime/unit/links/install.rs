//! Installation of static edges and synchronized safety unlink.
//! Native writes never hold JIT state; the exclusive Closed transition prevents
//! another installer/retirer from mutating the graph in the write interval.

use super::*;
use crate::executable::Write;
use crate::native::link;

impl PreparedLink<'_> {
    pub(super) fn validate(&self, state: &State) -> Result<(), Error> {
        if state.admission != self.admission {
            return Err(Error::StalePublication);
        }
        let from = eligible(state, self.process, self.source)?;
        let to = eligible(state, self.process, self.target)?;
        if !Arc::ptr_eq(&from.code, &self.source_code)
            || !Arc::ptr_eq(&to.code, &self.target_code)
            || target_payload(state, to, self.target_entry)?.reachability() != self.reachability
        {
            return Err(Error::StalePublication);
        }
        Ok(())
    }
}

impl<'p> Transition<'p> {
    /// Drain safety work first, then consume the pending FIFO within this
    /// stop's remaining installation budget. True means the queue is empty;
    /// false means valid fallbacks/requests remain for a later stop. Callers
    /// acknowledge/defer through the existing Batch protocol; a joined
    /// safety request still prevents reopening until its records are drained.
    pub(crate) fn drain_links(&mut self) -> Result<bool, Error> {
        loop {
            // Safety unlinks/cancellations never consume the performance quota,
            // including requests arriving after the last installation attempt.
            self.drain_retirements()?;
            let handle = {
                let state = self.process.lock();
                self.require_closed(&state)?;
                let Some(handle) = state.units.links.head else {
                    return Ok(true);
                };
                if state.link_install_attempts == INSTALL_LIMIT {
                    return Ok(false);
                }
                LinkHandle(handle, self.process.identity)
            };
            self.install_link(handle)?;
        }
    }

    /// True means newly installed. False means already installed, or a stale
    /// pending target discarded without modifying its source branch, or budget
    /// exhausted (the source retains its valid old link or safe fallback).
    pub(crate) fn install_link(&mut self, handle: LinkHandle) -> Result<bool, Error> {
        let process = self.process;
        let (prepared, previous) = {
            let mut state = process.lock();
            self.require_closed(&state)?;
            if handle.1 != process.identity {
                return Err(Error::StaleUnit);
            }
            let record = state
                .units
                .links
                .records
                .get(handle.0)
                .ok_or(Error::StaleUnit)?;
            if record.installed {
                return Ok(false);
            }
            if state.link_install_attempts == INSTALL_LIMIT {
                return Ok(false);
            }
            let prepared = PreparedLink {
                process,
                admission: state.admission,
                source: record.source,
                target: record.target,
                source_code: Arc::clone(&record.source_code),
                target_code: Arc::clone(&record.target_code),
                state_map: record.state_map,
                target_entry: record.target_entry,
                island: record.island,
                reachability: record.reachability,
            };
            // Count consumed records, including stale/failed preparations, not
            // just successful writes. Retrying failures cannot make one stop
            // unbounded. Duplicate already-installed handles cost no attempt.
            state.link_install_attempts += 1;
            if let Err(error) = prepared.validate(&state) {
                if matches!(error, Error::StaleUnit | Error::StalePublication) {
                    let removed = state.units.remove_uninstalled_link(handle.0).unwrap();
                    drop(state);
                    drop(removed);
                    return Ok(false);
                }
                return Err(error);
            }
            let previous = state
                .units
                .records
                .get(prepared.source.0)
                .unwrap()
                .static_sites[prepared.island]
                .callable;
            (prepared, previous)
        };
        if let Some(previous) = previous {
            // Publication retained the old callable edge alongside this pending
            // successor. Restore/synchronize its fallback before releasing the
            // old target/bridge, even if the old baseline remains published.
            self.unlink_link(LinkHandle(previous, self.process.identity))?;
        }
        // Preparation/allocation stay off the generated edge. The transfer
        // must include selective canonical writeback,
        // not just compare the two sets of physical input bindings.
        let bridge = super::super::bridge::install(
            process,
            &prepared.source_code,
            prepared.state_map,
            &prepared.target_code,
            prepared.target_entry,
            super::super::bridge::Tail::StaticIsland,
        )?;
        let allocation = &prepared.source_code.code.allocation;
        let map = &prepared.source_code.states[prepared.state_map as usize];
        let target = &prepared.target_code.entries[prepared.target_entry];
        let target_address =
            prepared.target_code.code.allocation.address() + target.fast_offset as usize;
        let abi = prepared.source_code.code.metadata.abi;
        let branch = link::emit(
            abi,
            (allocation.address() + map.native_offset as usize) as u64,
            bridge
                .as_ref()
                .map_or(target_address, |bridge| bridge.allocation.address()) as u64,
            allocation.island_address(prepared.island).unwrap() as u64,
        )
        .map_err(Error::InvalidUnit)?;
        {
            let mut state = process.lock();
            self.require_closed(&state)?;
            if let Err(error) = prepared.validate(&state) {
                if matches!(error, Error::StaleUnit | Error::StalePublication) {
                    let removed = state.units.remove_uninstalled_link(handle.0).unwrap();
                    drop(state);
                    drop(removed);
                    return Ok(false);
                }
                return Err(error);
            }
            // Roots/backlinks already exist. Conservatively mark callable
            // before changing bytes; errors cannot permit pending-only removal.
            let record = state.units.links.records.get_mut(handle.0).unwrap();
            record.bridge = bridge;
            record.installed = true;
            let site = &mut state
                .units
                .records
                .get_mut(prepared.source.0)
                .unwrap()
                .static_sites
                .value[prepared.island];
            debug_assert!(site.callable.is_none());
            site.callable = Some(handle.0);
        }
        let code = Write::Code {
            offset: map.native_offset as usize,
            bytes: branch.patch(),
        };
        // Invalidation may queue after revalidation, but cannot retire either
        // owner or reopen this transition. Its safety drain observes this link.
        let result = unsafe {
            if let Some(bytes) = &branch.island {
                self.patch_unit(
                    prepared.source,
                    &[
                        Write::Island {
                            index: prepared.island,
                            bytes,
                        },
                        code,
                    ],
                )
            } else {
                self.patch_unit(prepared.source, &[code])
            }
        };
        if let Err(error) = result {
            process.fail(&mut process.lock(), error);
            return Err(error);
        }
        let mut state = process.lock();
        self.require_closed(&state)?;
        state.units.links.unqueue(handle.0);
        Ok(true)
    }

    /// Shared by link replacement and every unit-retirement path. Do not release
    /// roots, detach the source map, or mark the target retired before this call.
    pub(crate) fn unlink_link(&mut self, handle: LinkHandle) -> Result<(), Error> {
        let process = self.process;
        if handle.1 != process.identity {
            return Err(Error::StaleUnit);
        }
        let handle = handle.0;
        let installed = {
            let mut state = process.lock();
            self.require_closed(&state)?;
            let record = state
                .units
                .links
                .records
                .get(handle)
                .ok_or(Error::StaleUnit)?;
            if record.installed {
                Some((
                    record.source,
                    Arc::clone(&record.source_code),
                    record.state_map,
                    record.island,
                ))
            } else {
                let removed = state.units.remove_uninstalled_link(handle).unwrap();
                drop(state);
                drop(removed);
                None
            }
        };
        let Some((source, code, state_map, island)) = installed else {
            return Ok(());
        };
        let map = &code.states[state_map as usize];
        let allocation = &code.code.allocation;
        let result = (|| {
            let fallback = map.transfer.as_ref().unwrap().fallback_offset;
            let branch = link::emit(
                code.code.metadata.abi,
                (allocation.address() + map.native_offset as usize) as u64,
                (allocation.address() + fallback as usize) as u64,
                allocation.island_address(island).unwrap() as u64,
            )
            .map_err(Error::InvalidUnit)?;
            if branch.island.is_some() {
                return Err(Error::InvalidUnit(
                    "source-local fallback unexpectedly requires an island",
                ));
            }
            // Restore to this same retained unit; metadata/ABI stay unchanged.
            // Target references are still in the registered link throughout RW
            // closure and instruction/pipeline synchronization.
            unsafe {
                self.patch_unit(
                    source,
                    &[Write::Code {
                        offset: map.native_offset as usize,
                        bytes: branch.patch(),
                    }],
                )
            }
        })();
        if let Err(error) = result {
            process.fail(&mut process.lock(), error);
            return Err(error);
        }
        let removed = {
            let mut state = process.lock();
            self.require_closed(&state)?;
            state.units.links.records.get_mut(handle).unwrap().installed = false;
            state.units.remove_uninstalled_link(handle).unwrap()
        };
        drop(removed);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
