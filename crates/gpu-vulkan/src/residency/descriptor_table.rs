//! Channel-owned descriptor bytes and translations shared across pipeline users.
use super::*;
use texture_bindings::{
    BoundDescriptor, DescriptorMapping, DescriptorSource, DescriptorTable, TextureChannel,
};

impl Residency<'_> {
    #[cfg(feature = "performance-counters")]
    pub(crate) fn note_input_read(&mut self) {
        self.runtime.statistics.mutable_input_reads += 1;
    }
    #[cfg(feature = "performance-counters")]
    pub(crate) fn note_input_location(&mut self, reused: bool) {
        if reused {
            self.runtime.statistics.mutable_input_span_reuses += 1;
        } else {
            self.runtime.statistics.mutable_input_relocations += 1;
        }
    }
    #[cfg(feature = "performance-counters")]
    pub(crate) fn note_sampled_slot(&mut self, kept: bool) {
        if kept {
            self.runtime.statistics.sampled_slot_keeps += 1;
        } else {
            self.runtime.statistics.sampled_slot_resolves += 1;
        }
    }
    /// Read live bytes every time. Only the translation and interpretation are
    /// retained: stable registers never make descriptor memory immutable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_descriptor<'r, M, T: BoundDescriptor + 'r>(
        &'r mut self,
        memory: &mut M,
        index: u32,
        table: impl for<'b> Fn(&'b mut TextureChannel) -> &'b mut DescriptorTable<T>,
        identity: impl Fn(&M, &nixe_memory::device_memory::DeviceMemory) -> DescriptorMapping,
        mut locate: impl FnMut(
            &mut M,
            &nixe_memory::device_memory::DeviceMemory,
            u64,
        ) -> Result<DescriptorSource, VulkanError>,
        mut read_virtual: impl FnMut(&mut Self, &mut M, u64, &mut [u8; 32]) -> Result<(), VulkanError>,
    ) -> Result<(&'r [u8; 32], u64), VulkanError> {
        #[cfg(feature = "performance-counters")]
        {
            self.runtime.statistics.descriptor_reads += 1;
        }
        loop {
            let mut mapping = identity(memory, &self.runtime.device_memory);
            let table_ref = table(&mut self.texture_cache.bindings.channel);
            // Pool/base/limit changes clear the records. Every populated slot
            // has already passed admission at this index.
            let cached = table_ref
                .entries
                .get(index as usize)
                .filter(|entry| entry.mapping == mapping)
                .and_then(|entry| entry.source.clone());
            let relocated = cached.is_none();
            let source = match cached {
                Some(source) => source,
                None => {
                    let base = table_ref.table.ok_or(VulkanError::InvalidResource(
                        "unprogrammed descriptor table",
                    ))?;
                    if index > table_ref.maximum {
                        return Err(VulkanError::InvalidResource(
                            "descriptor index exceeds pool limit",
                        ));
                    }
                    let address = base
                        .checked_add(u64::from(index) * 32)
                        .ok_or(VulkanError::InvalidResource("descriptor address overflow"))?;
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_relocations += 1;
                    }
                    let source = locate(memory, &self.runtime.device_memory, address)?;
                    if matches!(mapping, DescriptorMapping::Guarded(_)) {
                        // Locating lazy native backing can replace the address-space guard.
                        mapping = identity(memory, &self.runtime.device_memory);
                    }
                    if table_ref.entries.len() <= index as usize {
                        table_ref
                            .entries
                            .resize_with(index as usize + 1, Default::default);
                    }
                    let entry = &mut table_ref.entries[index as usize];
                    entry.mapping = mapping;
                    entry.source = Some(source.clone());
                    source
                }
            };
            let bytes = match source {
                DescriptorSource::Contiguous { device, host } => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_contiguous_reads += 1;
                    }
                    self.flush_read_span(device, 32)?;
                    // GSV's guard pins its device mappings. On Maxwell the texture
                    // cache lock holds off physical retirement until this read ends.
                    unsafe { host.read_array() }
                }
                DescriptorSource::Captured(capture) => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_fragmented_reads += 1;
                    }
                    let mut bytes = [0; 32];
                    self.read_captured_memory(&capture, &mut bytes)?;
                    bytes
                }
                DescriptorSource::Virtual(address) => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_fragmented_reads += 1;
                    }
                    let mut bytes = [0; 32];
                    read_virtual(self, memory, address, &mut bytes)?;
                    bytes
                }
            };
            // A concurrent remap during location, coherence or copying makes
            // these bytes obsolete. The old generation also forces relocation.
            if matches!(mapping, DescriptorMapping::Concurrent { .. })
                && identity(memory, &self.runtime.device_memory) != mapping
            {
                #[cfg(feature = "performance-counters")]
                {
                    self.runtime.statistics.descriptor_mapping_retries += 1;
                }
                continue;
            }
            let table = table(&mut self.texture_cache.bindings.channel);
            let changes = table.changes;
            let entry = &mut table.entries[index as usize];
            let changed = entry.revision == 0 || bytes != entry.bytes;
            if changed {
                #[cfg(feature = "performance-counters")]
                {
                    self.runtime.statistics.descriptor_changes += 1;
                }
                entry.bytes = bytes;
                entry.value = None;
            } else if (relocated || entry.changes != changes)
                && let Some(value) = &mut entry.value
            {
                #[cfg(feature = "performance-counters")]
                {
                    self.runtime.statistics.descriptor_selection_invalidations += 1;
                }
                value.invalidate_selection();
            }
            if changed || relocated || entry.changes != changes {
                table.next_revision = table
                    .next_revision
                    .checked_add(1)
                    .expect("descriptor revision exhausted");
                entry.revision = table.next_revision;
            }
            entry.changes = changes;
            return Ok((&entry.bytes, entry.revision));
        }
    }
}

#[cfg(test)]
mod tests;
