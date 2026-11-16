//! Channel-owned descriptor bytes and translations shared across pipeline users.
use super::*;
use texture_bindings::{BoundDescriptor, DescriptorSource, DescriptorTable, TextureChannel};

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
    pub(crate) fn read_descriptor<M, T: BoundDescriptor>(
        &mut self,
        memory: &mut M,
        index: u32,
        table: impl for<'b> Fn(&'b mut TextureChannel) -> &'b mut DescriptorTable<T>,
        identity: impl Fn(&M, &nixe_memory::device_memory::DeviceMemory) -> [u64; 3],
        mut locate: impl FnMut(
            &mut M,
            &nixe_memory::device_memory::DeviceMemory,
            u64,
        ) -> Result<DescriptorSource, VulkanError>,
        mut read_virtual: impl FnMut(&mut Self, &mut M, u64, &mut [u8; 32]) -> Result<(), VulkanError>,
    ) -> Result<([u8; 32], u64), VulkanError> {
        #[cfg(feature = "performance-counters")]
        {
            self.runtime.statistics.descriptor_reads += 1;
        }
        loop {
            let mapping = identity(memory, &self.runtime.device_memory);
            let (address, source, relocated) = {
                let table = table(&mut self.texture_cache.bindings.channel);
                let base = table.table.ok_or(VulkanError::InvalidResource(
                    "unprogrammed descriptor table",
                ))?;
                if index > table.maximum {
                    return Err(VulkanError::InvalidResource(
                        "descriptor index exceeds pool limit",
                    ));
                }
                let address = base
                    .checked_add(u64::from(index) * 32)
                    .ok_or(VulkanError::InvalidResource("descriptor address overflow"))?;
                let entry = table.entry(index);
                let relocated =
                    entry.address != address || entry.mapping != mapping || entry.source.is_none();
                (
                    address,
                    (!relocated).then(|| entry.source.as_ref().unwrap().clone()),
                    relocated,
                )
            };
            let (source, mapping) = match source {
                Some(source) => (source, mapping),
                None => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_relocations += 1;
                    }
                    let source = locate(memory, &self.runtime.device_memory, address)?;
                    let mapping = identity(memory, &self.runtime.device_memory);
                    let entry = table(&mut self.texture_cache.bindings.channel).entry(index);
                    entry.address = address;
                    entry.mapping = mapping;
                    entry.source = Some(source.clone());
                    (source, mapping)
                }
            };
            let mut bytes = [0; 32];
            match &source {
                DescriptorSource::Contiguous { device, host } => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_contiguous_reads += 1;
                    }
                    self.flush_read_span(*device, 32)?;
                    // Coherence can replace physical resources. Never dereference
                    // a saved pointer after its mapping identity has changed.
                    if identity(memory, &self.runtime.device_memory) != mapping {
                        #[cfg(feature = "performance-counters")]
                        {
                            self.runtime.statistics.descriptor_mapping_retries += 1;
                        }
                        table(&mut self.texture_cache.bindings.channel)
                            .entry(index)
                            .source = None;
                        continue;
                    }
                    // The consuming memory owner retains the checked mapping.
                    // Raw instructions form no Rust reference to guest memory.
                    unsafe {
                        host.read_into(&mut bytes);
                    }
                }
                DescriptorSource::Captured(capture) => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_fragmented_reads += 1;
                    }
                    self.read_captured_memory(capture, &mut bytes)?
                }
                DescriptorSource::Virtual => {
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.descriptor_fragmented_reads += 1;
                    }
                    read_virtual(self, memory, address, &mut bytes)?
                }
            }
            if identity(memory, &self.runtime.device_memory) != mapping {
                #[cfg(feature = "performance-counters")]
                {
                    self.runtime.statistics.descriptor_mapping_retries += 1;
                }
                table(&mut self.texture_cache.bindings.channel)
                    .entry(index)
                    .source = None;
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
            return Ok((entry.bytes, entry.revision));
        }
    }
}
