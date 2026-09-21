//! Closed-authorized mutation of an existing unit's code and reserved islands.
//! The cache mutex serializes aliases; the borrowed coordinator capability
//! prevents execution from reopening through the complete write/sync interval.

use super::*;
use crate::lifetime::unit::patch::ClosedCode;

pub(crate) enum Write<'a> {
    Code {
        offset: usize,
        bytes: &'a [u8],
    },
    Island {
        index: usize,
        bytes: &'a [u8; ISLAND_SLOT_BYTES],
    },
}

impl Cache {
    pub(crate) fn patch(
        &self,
        permit: &ClosedCode<'_, '_, '_>,
        writes: &[Write<'_>],
    ) -> Result<(), Error> {
        let allocation = permit.allocation();
        if !std::ptr::eq(self, allocation.cache.as_ref()) {
            return Err(Error::Output(
                "patch allocation belongs to another cache".into(),
            ));
        }
        let state = self.lock()?;
        if state.segments[allocation.segment].generation != Some(allocation.generation) {
            return Err(Error::Output(
                "patch allocation has a stale segment generation".into(),
            ));
        }
        if writes.is_empty() {
            return Ok(());
        }
        let backing = state.backing.as_ref().ok_or(Error::Closed)?;
        let (offset, bytes) = backing.write_window(
            allocation.segment * SEGMENT_BYTES + allocation.span.start,
            allocation.span.len,
        );
        backing.rw.as_ref().ok_or(Error::Poisoned)?.protect(
            offset,
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
        )?;
        let mut window = Window {
            state,
            offset,
            bytes,
            finished: false,
        };
        for write in writes {
            let (rx, bytes): (usize, &[u8]) = match write {
                Write::Code { offset, bytes } => {
                    if offset
                        .checked_add(bytes.len())
                        .is_none_or(|end| end > allocation.len())
                    {
                        return Err(Error::Output(
                            "patch lies outside its owned code span".into(),
                        ));
                    }
                    (allocation.address() + offset, bytes)
                }
                Write::Island { index, bytes } => (
                    allocation
                        .island_address(*index)
                        .ok_or_else(|| Error::Output("patch names an unreserved island".into()))?,
                    bytes.as_slice(),
                ),
            };
            let rw = window.state.backing.as_ref().unwrap().rw.as_ref().unwrap();
            // The permit retains this allocation; validation above confines
            // both aliases to its code span or one of its owned island slots.
            unsafe { linux::copy(rw.base.as_ptr().add(rx - self.base), rx as *const u8, bytes) };
        }
        window.finish()
    }
}

struct Window<'a> {
    state: MutexGuard<'a, State>,
    offset: usize,
    bytes: usize,
    finished: bool,
}
impl Window<'_> {
    fn finish(&mut self) -> Result<(), Error> {
        self.state
            .backing
            .as_ref()
            .unwrap()
            .rw
            .as_ref()
            .unwrap()
            .protect(self.offset, self.bytes, libc::PROT_NONE)?;
        // Cache lines were synchronized using RX addresses by copy(). Other
        // cores must discard old fetched instructions before admission reopens.
        wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().map_err(|error| {
            Error::Host {
                operation: "synchronize patched executable pipelines",
                error,
            }
        })?;
        self.finished = true;
        Ok(())
    }
}
impl Drop for Window<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Partial mutation, protection/synchronization error, or unwind:
            // disable the entire write alias and further cache use. RX owners
            // stay alive, but the coordinator must never admit execution again.
            self.state.failed = true;
            drop(self.state.backing.as_mut().unwrap().rw.take());
        }
    }
}

#[cfg(test)]
mod tests;
