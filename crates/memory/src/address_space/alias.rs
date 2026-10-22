//! Aliasing changes source permissions and destination mappings in one admission.
use super::*;
#[cfg(test)]
mod tests;

impl AddressSpace {
    pub fn map_alias(
        &self,
        source: GuestVirtualAddress,
        destination: GuestVirtualAddress,
        bytes: u64,
        source_permissions: MemoryPermissions,
        alias_permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let end = self.pages(source, bytes)?;
        let destination_end = self.pages(destination, bytes)?;
        if source.get() < destination_end && destination.get() < end {
            return Err(MemoryError::Invalid("virtual alias ranges overlap"));
        }
        let mut aliases = self.machine().aliases();
        let writer = self.quiescence.writer()?;
        let mut source_changed = false;
        let mut source_executable = source_permissions.contains(MemoryPermissions::EXECUTE);
        for offset in (0..bytes).step_by(PAGE_SIZE as usize) {
            let entry = unsafe { self.entry(source.get() + offset) };
            if !entry.present() || unsafe { self.entry(destination.get() + offset) }.present() {
                return Err(MemoryError::Invalid(
                    "alias source is unmapped or destination is already mapped",
                ));
            }
            source_changed |= entry.permissions != source_permissions;
            source_executable |= entry.permissions.contains(MemoryPermissions::EXECUTE);
        }
        let generation = self
            .next_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MemoryError::Invalid("mapping identity exhausted"))?;
        let mut captured = Vec::new();
        let mut address = source.get();
        let mappings = unsafe { &*self.mappings.get() };
        while address < end {
            let (&base, mapping) = mappings
                .range(..=address)
                .next_back()
                .expect("validated alias source mapping");
            let stop = (base + mapping.bytes).min(end);
            let offset = mapping.offset + address - base;
            captured.push((
                GuestVirtualAddress::new(destination.get() + address - source.get()),
                Mapping {
                    generation,
                    bytes: stop - address,
                    allocation: mapping.allocation.view(offset, stop - address)?,
                    offset,
                    aliases: Vec::new(),
                },
                AllocationRange::new(
                    mapping.allocation.pin_range(offset, stop - address)?,
                    offset,
                    stop - address,
                )?,
            ));
            address = stop;
        }
        let transition = self.quiescence.begin_transition(writer)?;
        transition.wait_quiescent();
        let result = (|| {
            for (address, mut mapping, range) in captured {
                unsafe {
                    self.map_physical_entries(address, &range, alias_permissions)?;
                }
                mapping.aliases = self.register_aliases(&mut aliases, address.get(), &range)?;
                unsafe { &mut *self.mappings.get() }.insert(address.get(), mapping);
            }
            if source_changed {
                unsafe {
                    self.update_page_permissions(source.get(), bytes, |mut entry| {
                        entry.permissions = source_permissions;
                        entry
                    })?;
                }
                unsafe {
                    self.update_mapping_generations(&mut aliases, source.get(), end, generation)?;
                }
                self.invalidate_mapping_observers(source, bytes);
            }
            if source_changed && source_executable {
                self.quiescence.invalidate_code(source, bytes);
            }
            if alias_permissions.contains(MemoryPermissions::EXECUTE) {
                self.quiescence.invalidate_code(destination, bytes);
            }
            self.invalidate_mapping_observers(destination, bytes);
            Ok(())
        })()
        .map_err(|error| transition.fail(error));
        if result.is_ok() {
            transition.publish();
        } else {
            drop(transition);
        }
        drop(aliases);
        result
    }

    pub fn unmap_alias(
        &self,
        source: GuestVirtualAddress,
        destination: GuestVirtualAddress,
        bytes: u64,
        source_permissions: MemoryPermissions,
    ) -> Result<(), MemoryError> {
        let end = self.pages(source, bytes)?;
        let destination_end = self.pages(destination, bytes)?;
        if source.get() < destination_end && destination.get() < end {
            return Err(MemoryError::Invalid("virtual alias ranges overlap"));
        }
        let mut aliases = self.machine().aliases();
        let writer = self.quiescence.writer()?;
        let mut changed = false;
        let mut source_executable = source_permissions.contains(MemoryPermissions::EXECUTE);
        let mut alias_executable = false;
        for offset in (0..bytes).step_by(PAGE_SIZE as usize) {
            let original = unsafe { self.entry(source.get() + offset) };
            let alias = unsafe { self.entry(destination.get() + offset) };
            if !original.present()
                || !alias.present()
                || original.physical_page != alias.physical_page
            {
                return Err(MemoryError::AliasMismatch);
            }
            changed |= original.permissions != source_permissions;
            source_executable |= original.permissions.contains(MemoryPermissions::EXECUTE);
            alias_executable |= alias.permissions.contains(MemoryPermissions::EXECUTE);
        }
        let generation = self
            .next_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MemoryError::Invalid("mapping identity exhausted"))?;
        let transition = self.quiescence.begin_transition(writer)?;
        transition.wait_quiescent();
        let mut removed = Vec::new();
        let result = (|| {
            unsafe {
                self.arena.unmap(destination, bytes)?;
            }
            for address in (destination.get()..destination_end).step_by(PAGE_SIZE as usize) {
                unsafe {
                    self.set_entry(address, MappedPage::EMPTY);
                }
            }
            removed =
                unsafe { self.remove_mappings(&mut aliases, destination.get(), destination_end)? };
            if changed {
                unsafe {
                    self.update_page_permissions(source.get(), end - source.get(), |mut entry| {
                        entry.permissions = source_permissions;
                        entry
                    })?;
                }
                unsafe {
                    self.update_mapping_generations(&mut aliases, source.get(), end, generation)?;
                }
                self.invalidate_mapping_observers(source, bytes);
            }
            if changed && source_executable {
                self.quiescence.invalidate_code(source, bytes);
            }
            if alias_executable {
                self.quiescence.invalidate_code(destination, bytes);
            }
            self.invalidate_mapping_observers(destination, bytes);
            Ok(())
        })()
        .map_err(|error| transition.fail(error));
        if result.is_ok() {
            transition.publish();
        } else {
            drop(transition);
        }
        drop(aliases);
        drop(removed);
        result
    }
}
