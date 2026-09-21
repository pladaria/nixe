//! One cold snapshot at terminal shutdown; no hot-path statistics.

use super::*;
use crate::executable::Cache;

#[derive(Default)]
pub(in crate::lifetime) struct ShutdownSummary {
    lcq_units: usize,
    hcq_units: usize,
    hcq_published: usize,
    lcq_native_bytes: usize,
    hcq_native_bytes: usize,
}

impl Units {
    pub(in crate::lifetime) fn shutdown_summary(&self) -> ShutdownSummary {
        let mut summary = ShutdownSummary::default();
        // Count distinct resident units, not entry labels or historical IDs.
        // Retired units awaiting reclamation still occupy native storage.
        for record in self.records.values() {
            let bytes = record.code.code.allocation.len();
            match record.code.tier {
                Tier::Lcq => {
                    summary.lcq_units += 1;
                    summary.lcq_native_bytes += bytes;
                }
                Tier::Hcq => {
                    summary.hcq_units += 1;
                    summary.hcq_native_bytes += bytes;
                    summary.hcq_published += usize::from(record.lifecycle == Lifecycle::Published);
                }
            }
        }
        summary
    }
}

impl ShutdownSummary {
    pub(in crate::lifetime) fn log(self, process: u64, cache: &Cache) {
        log::debug!(
            "JIT shutdown units: jit_process={} lcq_resident={} hcq_resident={} hcq_published={} lcq_native_bytes={} hcq_native_bytes={}",
            process,
            self.lcq_units,
            self.hcq_units,
            self.hcq_published,
            self.lcq_native_bytes,
            self.hcq_native_bytes,
        );
        // Separate snapshot: do not nest cache/JIT locks to make a diagnostic
        // transactional. Workers may still release staging/metadata on shutdown.
        // Committed segments include free capacity; metadata includes charged
        // staging/registries. Neither this total nor native bytes represent RSS.
        match cache.usage() {
            Ok(usage) => log::debug!(
                "JIT shutdown cache: jit_process={} committed_bytes={} metadata_bytes={} total_bytes={}",
                process,
                usage.committed,
                usage.metadata,
                usage.total(),
            ),
            Err(error) => log::debug!(
                "JIT shutdown cache: jit_process={} usage_unavailable={}",
                process,
                error,
            ),
        }
    }
}
