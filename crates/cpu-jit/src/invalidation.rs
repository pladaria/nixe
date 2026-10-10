//! Mapping authority stays in ExecutionMemory; compiled ranges stay in Dynarmic.

use crate::{JitProcess, JitThread, Native};
use nixe_memory::{
    GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidation, MemoryInvalidationCursor,
    MemoryInvalidationError, MemoryInvalidationKind, MemoryInvalidationSource,
};

pub struct Invalidations {
    cursor: MemoryInvalidationCursor,
    records: Vec<MemoryInvalidation>,
    physical: Vec<GuestPhysicalPageId>,
    ranges: Vec<(GuestVirtualAddress, usize)>,
}

impl Invalidations {
    pub fn new(cursor: MemoryInvalidationCursor) -> Self {
        Self {
            cursor,
            records: Vec::new(),
            physical: Vec::new(),
            ranges: Vec::new(),
        }
    }

    pub fn cursor(&self) -> MemoryInvalidationCursor {
        self.cursor
    }

    /// Called only on a stopped native core, under its memory execution lease.
    pub fn synchronize(&mut self, process: &JitProcess, native: &Native) -> Result<(), Box<str>> {
        self.records.clear();
        let latest = match process
            .memory
            .read_invalidations_since(self.cursor, &mut self.records)
        {
            Ok(latest) => latest,
            Err(MemoryInvalidationError::HistoryLost { latest, .. }) => {
                JitThread::invalidate(native, 0, 0)?;
                self.cursor = latest;
                return Ok(());
            }
            Err(error) => return Err(error.to_string().into()),
        };
        if self.records.is_empty() {
            self.cursor = latest;
            return Ok(());
        }
        self.physical.clear();
        self.ranges.clear();
        let space = process.cpu.address_space_id();
        for record in &self.records {
            match record.kind {
                MemoryInvalidationKind::Mapping {
                    address_space,
                    start,
                    size,
                } if address_space == space => {
                    // Historical virtual ranges also cover aliases which have
                    // disappeared or changed physical identity since compilation.
                    self.ranges.push((start, size as usize));
                }
                MemoryInvalidationKind::InstructionCache { address_space }
                    if address_space == space =>
                {
                    JitThread::invalidate(native, 0, 0)?;
                    self.cursor = latest;
                    return Ok(());
                }
                MemoryInvalidationKind::ExecutableContent { first, second } => {
                    self.physical.push(first);
                    self.physical.extend(second);
                }
                _ => {}
            }
        }
        self.physical.sort_unstable();
        self.physical.dedup();
        process
            .memory
            .append_mapped_alias_ranges(space, &self.physical, &mut self.ranges);
        self.ranges.sort_unstable();
        self.ranges.dedup();
        // Like the upstream adapters, delegate block ownership and interval
        // union to Dynarmic; only duplicate notifications are removed here.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/core/arm/dynarmic/arm_dynarmic_64.cpp
        for &(start, size) in &self.ranges {
            JitThread::invalidate(native, start.get(), size)?;
        }
        // Public Dynarmic invalidations are queued, not flushed synchronously.
        // Run/Step apply them before looking up any block, including native
        // links/RSB/fast dispatch. Nixe never clears CacheInvalidation. A stopped
        // core may acknowledge accepted work without executing a guest to flush
        // it; failed notification must leave this cursor unacknowledged.
        self.cursor = latest;
        Ok(())
    }
}
