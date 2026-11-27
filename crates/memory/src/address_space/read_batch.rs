//! Bounded host admission for streams of small guest reads and comparisons.
use super::*;
use crate::transition::HostAdmission;

const BYTE_BUDGET: usize = 16 * 1024;
const MIN_READ_CHARGE: usize = 256;

/// An ordinary-memory read series. Callers release admission before invoking
/// callbacks or taking locks that can lead to another guest memory access.
pub struct ReadBatch<'a> {
    space: &'a AddressSpace,
    admission: Option<HostAdmission<'a>>,
    remaining: usize,
}

impl<'a> ReadBatch<'a> {
    pub(super) fn new(space: &'a AddressSpace) -> Self {
        Self {
            space,
            admission: None,
            remaining: 0,
        }
    }

    pub fn release(&mut self) {
        self.admission = None;
        self.remaining = 0;
    }

    fn admit(&mut self, bytes: usize) -> Result<(), MemoryError> {
        let charge = bytes.max(MIN_READ_CHARGE);
        if self.remaining < charge {
            self.release();
            self.admission = Some(self.space.quiescence.enter_host()?);
            self.remaining = BYTE_BUDGET;
        }
        // Even small accesses eventually give mapping writers a chance to run.
        self.remaining = self.remaining.saturating_sub(charge);
        Ok(())
    }

    #[inline]
    fn direct_entry(&self, address: GuestVirtualAddress) -> Result<Option<u64>, MemoryError> {
        // Tracking can change a mapped page's ownership during host admission.
        // Reload the entry on each access so a newly cached page takes the
        // coherent path. Admission protects its mapping and physical lifetime.
        let raw = unsafe { self.space.raw_entry(address.get()) };
        if raw & READ_MARKED != 0 || raw == 0 {
            require_entry(raw, address, MemoryPermissions::READ)?;
            if entry_cached(raw) {
                return Ok(None);
            }
        }
        Ok(Some(raw))
    }

    pub fn read_into(
        &mut self,
        start: GuestVirtualAddress,
        destination: &mut [u8],
    ) -> Result<Access<()>, MemoryError> {
        let bytes = destination.len();
        if bytes == 0 {
            return Ok(Access::Ready(()));
        }
        self.space.range(start, bytes as u64)?;
        if bytes as u64 <= PAGE_SIZE - start.get() % PAGE_SIZE {
            self.admit(bytes)?;
            if let Some(entry) = self.direct_entry(start)? {
                unsafe { crate::raw::capture(entry_pointer(entry, start.get()), destination) };
                return Ok(Access::Ready(()));
            }
        }
        self.release();
        self.space.read_into(start, destination)
    }

    /// Compare a retained snapshot against ordinary RAM. Cached pages and
    /// unaligned ranges require the caller's coherent read path.
    pub fn compare_words(
        &mut self,
        start: GuestVirtualAddress,
        words: &[u64],
    ) -> Result<DirectWordsComparison, MemoryError> {
        if !start.get().is_multiple_of(8) {
            return Ok(DirectWordsComparison::NeedsCoherentRead);
        }
        let bytes = size_of_val(words);
        let end = self.space.range(start, bytes as u64)?;
        self.admit(bytes)?;
        if bytes != 0 && bytes as u64 <= PAGE_SIZE - start.get() % PAGE_SIZE {
            let Some(entry) = self.direct_entry(start)? else {
                return Ok(DirectWordsComparison::NeedsCoherentRead);
            };
            return Ok(
                if unsafe { crate::raw::matches_words(entry_pointer(entry, start.get()), words) } {
                    DirectWordsComparison::Equal
                } else {
                    DirectWordsComparison::Different
                },
            );
        }
        let mut at = start.get();
        while at < end {
            let Some(entry) = self.direct_entry(GuestVirtualAddress::new(at))? else {
                return Ok(DirectWordsComparison::NeedsCoherentRead);
            };
            let n = ((PAGE_SIZE - at % PAGE_SIZE).min(end - at) / 8) as usize;
            let i = ((at - start.get()) / 8) as usize;
            if !unsafe { crate::raw::matches_words(entry_pointer(entry, at), &words[i..i + n]) } {
                return Ok(DirectWordsComparison::Different);
            }
            at += n as u64 * 8;
        }
        Ok(DirectWordsComparison::Equal)
    }
}
