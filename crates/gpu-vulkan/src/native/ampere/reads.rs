//! Persistent locations for mutable inputs. Values are read on every use;
//! mapping identity alone never makes guest contents immutable.
use super::*;

pub(super) struct ReadSlot<const N: usize> {
    address: u64,
    mapping: Option<[u64; 3]>,
    pointer: Option<(u64, nixe_memory::HostAddress)>,
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
    pub fn update(
        &mut self,
        residency: &mut Residency<'_>,
        memory: &mut impl RenderMemory,
        address: u64,
    ) -> Result<bool, VulkanError> {
        let mapping = [
            memory.image_space_id(),
            memory.mapping_generation(),
            residency.device_memory().generation(),
        ];
        let relocated = self.address != address || self.mapping != Some(mapping);
        if relocated {
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
        let mut bytes = [0; N];
        if let Some((device, host)) = self.pointer {
            residency.flush_read_span(device, N as u64)?;
            // The consuming view retains shared mapping ownership. Both virtual
            // and physical generations were checked before using this address.
            unsafe {
                host.read_into(&mut bytes);
            }
        } else {
            residency.read_captured_memory(
                self.capture.as_ref().expect("fragmented input"),
                &mut bytes,
            )?;
        }
        let changed = relocated || bytes != self.bytes;
        self.bytes = bytes;
        Ok(changed)
    }
}
