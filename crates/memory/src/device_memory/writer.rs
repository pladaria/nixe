//! Device RAM stores use the same compressed translation as GetPointer and
//! WalkBlock. Shader/resource invalidation belongs to the consuming cache owner.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.inc
use super::*;
use crate::WriteSource;
impl DeviceMemory {
    pub fn write_contiguous(
        &self,
        address: u64,
        source: WriteSource<'_>,
    ) -> Result<(), MemoryError> {
        if source.is_empty() {
            return Ok(());
        }
        let physical = self.physical_address(address).ok_or(MemoryError::Invalid(
            "contiguous GPU write has no device RAM backing",
        ))?;
        let destination = self
            .machine()
            .arena()
            .host_address(physical, source.len() as u64)
            .ok_or(MemoryError::Invalid("device write exceeds physical RAM"))?;
        self.write_command_pages(address, source.len() as u64);
        // The machine retains its stable physical arena; cache owners resolve
        // coherence. Raw instruction copies expose no Rust references to RAM.
        unsafe {
            match source {
                WriteSource::Bytes(bytes) => crate::raw::write(destination.as_ptr(), bytes),
                WriteSource::Ram {
                    address,
                    offset,
                    bytes,
                } => crate::raw::copy(address.as_ptr().add(offset), destination.as_ptr(), bytes),
            }
        }
        Ok(())
    }
    pub fn write_block_unsafe(
        &self,
        address: u64,
        source: WriteSource<'_>,
    ) -> Result<(), MemoryError> {
        let end = address
            .checked_add(source.len() as u64)
            .filter(|&end| end <= 1 << self.0.address_bits)
            .ok_or(MemoryError::Invalid(
                "device block write exceeds address space",
            ))?;
        let mut at = address;
        while at < end {
            let page = self.0.tracked.entry(at).page();
            let bytes = ((u64::from(page.continuity) + 1) * PAGE_SIZE - (at & (PAGE_SIZE - 1)))
                .min(end - at);
            if page.physical != 0 {
                let begin = (at - address) as usize;
                self.write_contiguous(at, source.slice(begin..begin + bytes as usize))?;
            }
            at += bytes;
        }
        Ok(())
    }
}
