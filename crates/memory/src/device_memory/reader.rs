//! ReadBlockUnsafe owns the reference's four translation slots on the actual
//! reader. The shared device table owns mappings, never a second read cache.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc#L461-L498
use super::*;
use std::cell::Cell;
#[derive(Clone, Copy, Debug, Default)]
struct Translation {
    device_page: u64,
    host: usize,
}
/// Raw device RAM consumption by one GPU/decoder owner. Coherence is resolved
/// by that owner's caches before this copy; it never flushes or admits RAM.
/// Reader state is Send, not Sync, and is not cloned across execution owners.
#[derive(Debug)]
pub struct DeviceMemoryReader {
    memory: DeviceMemory,
    generation: Cell<u64>,
    slots: [Cell<Translation>; 4],
    cursor: Cell<usize>,
}
impl DeviceMemoryReader {
    pub(super) fn new(memory: DeviceMemory) -> Self {
        Self {
            memory,
            generation: Cell::new(0),
            slots: std::array::from_fn(|_| Cell::new(Translation::default())),
            cursor: Cell::new(0),
        }
    }
    /// MemoryManager::ReadBlockImpl copies a known physically contiguous GPU
    /// page through GetPointer, without consulting ReadBlockUnsafe's t_slot.
    pub fn read_contiguous(&self, address: u64, output: &mut [u8]) -> Result<(), MemoryError> {
        if output.is_empty() {
            return Ok(());
        }
        let source = self
            .memory
            .pointer(address, output.len() as u64)
            .ok_or(MemoryError::Invalid("contiguous device read exceeds RAM"))?;
        // SAFETY: the checked physical span belongs to the machine's retained
        // arena. Copy instructions create no Rust references to guest RAM.
        unsafe {
            self.memory.machine().copy_ram_to_host(source, output);
        }
        Ok(())
    }
    pub fn read_block_unsafe(&self, address: u64, output: &mut [u8]) -> Result<(), MemoryError> {
        let end = address
            .checked_add(output.len() as u64)
            .filter(|&end| end <= 1 << self.memory.0.address_bits)
            .ok_or(MemoryError::Invalid(
                "device block read exceeds address space",
            ))?;
        if output.is_empty() {
            return Ok(());
        }
        // Map/Unmap clear Eden's t_slot. Shared live mapping handles publish an
        // epoch instead, so each actual reader clears its own four slots.
        let generation = self.memory.generation();
        if self.generation.get() != generation {
            for slot in &self.slots {
                slot.set(Translation::default());
            }
            self.generation.set(generation);
            self.cursor.set(0);
        }
        let within = address & (PAGE_SIZE - 1);
        if output.len() as u64 <= PAGE_SIZE - within {
            let device_page = address & !(PAGE_SIZE - 1);
            for slot in &self.slots {
                let slot = slot.get();
                if slot.device_page == device_page && slot.host != 0 {
                    // SAFETY: physical arena RAM stays mapped for the reader's
                    // machine lifetime. Raw instruction copies permit guest
                    // tearing; no Rust references to RAM are created.
                    unsafe {
                        self.memory.machine().copy_ram_to_host(
                            crate::HostAddress::from_pointer(
                                (slot.host + within as usize) as *mut u8,
                            ),
                            output,
                        );
                    }
                    return Ok(());
                }
            }
            let page = self.memory.0.tracked.entry(address).page();
            if page.physical != 0 {
                let source = self
                    .memory
                    .machine()
                    .arena()
                    .host_address(
                        crate::ArenaOffset::new(u64::from(page.physical - 1) * PAGE_SIZE),
                        PAGE_SIZE,
                    )
                    .ok_or(MemoryError::Invalid("device PTE physical page exceeds RAM"))?;
                let cursor = self.cursor.get();
                self.slots[cursor].set(Translation {
                    device_page,
                    host: source.as_ptr() as usize,
                });
                self.cursor.set((cursor + 1) & 3);
                unsafe {
                    self.memory.machine().copy_ram_to_host(
                        crate::HostAddress::from_pointer(source.as_ptr().add(within as usize)),
                        output,
                    );
                }
                return Ok(());
            }
        }
        let mut at = address;
        while at < end {
            let page = self.memory.0.tracked.entry(at).page();
            let within = at & (PAGE_SIZE - 1);
            let count = ((u64::from(page.continuity) + 1) * PAGE_SIZE - within).min(end - at);
            let target = &mut output[(at - address) as usize..(at - address + count) as usize];
            if page.physical == 0 {
                target.fill(0);
            } else {
                let source = self
                    .memory
                    .machine()
                    .arena()
                    .host_address(
                        crate::ArenaOffset::new(u64::from(page.physical - 1) * PAGE_SIZE + within),
                        count,
                    )
                    .ok_or(MemoryError::Invalid("device PTE physical page exceeds RAM"))?;
                unsafe {
                    self.memory.machine().copy_ram_to_host(source, target);
                }
            }
            at += count;
        }
        Ok(())
    }
}
