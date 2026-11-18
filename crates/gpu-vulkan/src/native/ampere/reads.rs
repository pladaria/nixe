//! Persistent locations for mutable inputs. Values are read on every use;
//! mapping identity alone never makes guest contents immutable.
use super::*;

pub(super) struct ReadSlot<const N: usize> {
    address: u64,
    mapping: Option<[u64; 2]>,
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
        let mapping = [memory.image_space_id(), memory.mapping_generation()];
        let same_mapping = self.mapping == Some(mapping);
        let relocated = !same_mapping || self.address != address;
        let mut direct = if same_mapping {
            self.pointer.and_then(|span| span.at(address, N as u64))
        } else {
            None
        };
        #[cfg(feature = "performance-counters")]
        if relocated {
            residency.note_input_location(direct.is_some());
        }

        if direct.is_none() && (relocated || self.capture.is_none()) {
            let span = memory.read_span(address, N as u64).map_err(compilation)?;
            let capture = if let Some(span) = span {
                direct = Some(span.at(address, N as u64).expect("resolved input extent"));
                None
            } else {
                Some(
                    memory
                        .capture_memory(address, N as u64)
                        .map_err(compilation)?,
                )
            };
            self.pointer = span;
            self.capture = capture;
            // Lazy commitment may reacquire the guard. Take the generation that
            // owns the resolved span; its device pins live until the next map edit.
            self.mapping = Some([mapping[0], memory.mapping_generation()]);
        }
        self.address = address;
        let bytes = if let Some((device, host)) = direct {
            residency.flush_read_span(device, N as u64)?;
            // Coherence only publishes contents under the retained mapping.
            unsafe { host.read_array() }
        } else {
            let mut bytes = [0; N];
            residency.read_captured_memory(
                self.capture.as_ref().expect("fragmented input"),
                &mut bytes,
            )?;
            bytes
        };
        let changed = relocated || self.bytes != bytes;
        self.bytes = bytes;
        Ok(changed)
    }
}
