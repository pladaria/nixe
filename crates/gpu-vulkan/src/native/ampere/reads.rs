//! Persistent locations for mutable inputs. Values are read on every use;
//! mapping identity alone never makes guest contents immutable.
use super::*;

pub(super) struct ReadSlot<const N: usize> {
    address: u64,
    mapping: Option<[u64; 3]>,
    pointer: Option<nixe_gpu::native::ReadSpan>,
    capture: Option<nixe_memory::CapturedMemory>,
    pub bytes: [u8; N],
}
impl<const N: usize> Default for ReadSlot<N> {
    fn default() -> Self {
        Self {
            address: 0,
            mapping: None,
            pointer: None,
            capture: None,
            bytes: [0; N],
        }
    }
}
impl<const N: usize> ReadSlot<N> {
    #[cfg_attr(feature = "performance-counters", track_caller)]
    pub fn update(
        &mut self,
        residency: &mut Residency<'_>,
        memory: &mut impl RenderMemory,
        address: u64,
    ) -> Result<bool, VulkanError> {
        #[cfg(feature = "performance-counters")]
        residency.note_input_read();
        loop {
            let mapping = [
                memory.image_space_id(),
                memory.mapping_generation(),
                residency.device_memory().generation(),
            ];
            let relocated = self.address != address || self.mapping != Some(mapping);
            let reusable = self.mapping == Some(mapping)
                && self
                    .pointer
                    .is_some_and(|span| span.at(address, N as u64).is_some());
            #[cfg(feature = "performance-counters")]
            if relocated {
                residency.note_input_location(reusable);
            }
            if relocated && !reusable {
                self.pointer = memory.read_span(address, N as u64).map_err(compilation)?;
                self.capture = if self.pointer.is_none() {
                    Some(
                        memory
                            .capture_memory(address, N as u64)
                            .map_err(compilation)?,
                    )
                } else {
                    None
                };
                self.address = address;
                self.mapping = Some([
                    memory.image_space_id(),
                    memory.mapping_generation(),
                    residency.device_memory().generation(),
                ]);
            }
            self.address = address;
            let mut bytes = [0; N];
            if let Some(span) = self.pointer {
                let (device, host) = span.at(address, N as u64).expect("retained input extent");
                residency.flush_read_span(device, N as u64)?;
                if self.mapping
                    != Some([
                        memory.image_space_id(),
                        memory.mapping_generation(),
                        residency.device_memory().generation(),
                    ])
                {
                    self.mapping = None;
                    continue;
                }
                // The consuming mapping owner retains RAM. Coherence was resolved
                // and the physical translation revalidated before this raw read.
                unsafe {
                    host.read_into(&mut bytes);
                }
            } else {
                residency.read_captured_memory(
                    self.capture.as_ref().expect("fragmented input"),
                    &mut bytes,
                )?;
            }
            if self.mapping
                != Some([
                    memory.image_space_id(),
                    memory.mapping_generation(),
                    residency.device_memory().generation(),
                ])
            {
                self.mapping = None;
                continue;
            }
            let changed = relocated || bytes != self.bytes;
            self.bytes = bytes;
            return Ok(changed);
        }
    }
}
