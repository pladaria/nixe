//! Low-overhead aggregate measurements for the direct JIT.

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nixe_cpu::memory::DirectMemoryMetrics;
use nixe_cpu_direct_memory::{FaultRuntimeMetrics, NativeMemoryAccessKind};
use nixe_memory::{
    CpuMemoryBackend, MemoryInvalidation, MemoryInvalidationKind, MemoryInvalidationOrigin,
};

const REPORT_VERSION: u32 = 7;
const EXIT_REASON_COUNT: usize = 10;
const DIRECT_ACCESS_WIDTH_COUNT: usize = 5;

#[derive(Clone, Copy)]
pub(super) enum ExitReason {
    Dispatch = 0,
    Budget = 1,
    Control = 2,
    Architectural = 3,
    Unsupported = 4,
    DataFault = 5,
    Scheduled = 6,
    LoaderReturn = 7,
    Internal = 8,
    Reconcile = 9,
}

#[derive(Default)]
pub(super) struct InvalidationMetrics {
    records: u64,
    mapping: u64,
    instruction_cache: u64,
    device_write: u64,
    host_write: u64,
    cache_maintenance: u64,
    unknown: u64,
    relevant: u64,
    irrelevant: u64,
}

impl InvalidationMetrics {
    pub(super) fn record(&mut self, invalidation: MemoryInvalidation, relevant: bool) {
        self.records += 1;
        if relevant {
            self.relevant += 1;
        } else {
            self.irrelevant += 1;
        }
        match invalidation.kind {
            MemoryInvalidationKind::Mapping { .. } => self.mapping += 1,
            MemoryInvalidationKind::InstructionCache { .. } => self.instruction_cache += 1,
            MemoryInvalidationKind::ExecutableContent { .. } => match invalidation.origin {
                MemoryInvalidationOrigin::DeviceWrite => self.device_write += 1,
                MemoryInvalidationOrigin::HostWrite => self.host_write += 1,
                MemoryInvalidationOrigin::CacheMaintenance => self.cache_maintenance += 1,
                MemoryInvalidationOrigin::Mapping | MemoryInvalidationOrigin::Unknown => {
                    self.unknown += 1;
                }
            },
        }
    }
}

pub(super) struct Performance {
    path: PathBuf,
    written: AtomicBool,
    memory_backend: OnceLock<CpuMemoryBackend>,
    memory_backend_reason: OnceLock<Box<str>>,
    regions_discovered: AtomicU64,
    guest_blocks_discovered: AtomicU64,
    regions_compiled: AtomicU64,
    region_entry_points: AtomicU64,
    secondary_entry_hits: AtomicU64,
    compiled_guest_instructions: AtomicU64,
    unique_guest_instructions: AtomicU64,
    overlapping_guest_instructions: AtomicU64,
    lookup_hits: AtomicU64,
    lookup_misses: AtomicU64,
    guest_instructions: AtomicU64,
    clif_instructions: AtomicU64,
    native_bytes: AtomicU64,
    compile_time_ns: AtomicU64,
    native_time_ns: AtomicU64,
    compiled_direct_reads: [AtomicU64; DIRECT_ACCESS_WIDTH_COUNT],
    compiled_direct_writes: [AtomicU64; DIRECT_ACCESS_WIDTH_COUNT],
    exit_reasons: [AtomicU64; EXIT_REASON_COUNT],
    invalidations: AtomicU64,
    invalidation_mapping: AtomicU64,
    invalidation_instruction_cache: AtomicU64,
    invalidation_device_write: AtomicU64,
    invalidation_host_write: AtomicU64,
    invalidation_cache_maintenance: AtomicU64,
    invalidation_unknown: AtomicU64,
    invalidation_relevant: AtomicU64,
    invalidation_irrelevant: AtomicU64,
    invalidation_history_lost: AtomicU64,
    regions_retired: AtomicU64,
    region_recompilations: AtomicU64,
    fault_baseline: FaultRuntimeMetrics,
    direct_mmap_calls: AtomicU64,
    direct_mmap_pages: AtomicU64,
    direct_mprotect_calls: AtomicU64,
    direct_mprotect_pages: AtomicU64,
    direct_replaced_pages: AtomicU64,
    direct_writable_alias_pages_armed: AtomicU64,
    direct_writable_alias_pages_revoked: AtomicU64,
    direct_host_failures: AtomicU64,
    direct_peak_mapped_pages: AtomicU64,
    direct_control_reserved_bytes: AtomicU64,
    direct_baseline_vma_count: AtomicU64,
    direct_peak_vma_growth: AtomicU64,
    direct_vma_samples: AtomicU64,
    transition_exclusive_acquisitions: AtomicU64,
    transition_exclusive_wait_ns: AtomicU64,
    transition_peak_shared_holders: AtomicU64,
    transition_safepoint_notifications: AtomicU64,
}

impl Performance {
    pub(super) fn new(directory: &Path, title: &str) -> Result<Self, Box<str>> {
        fs::create_dir_all(directory).map_err(|error| {
            format!(
                "cannot create JIT performance report directory {}: {error}",
                directory.display()
            )
            .into_boxed_str()
        })?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = directory.join(format!(
            "{}-{timestamp}-{}.toml",
            file_stem(title),
            std::process::id()
        ));
        Ok(Self {
            path,
            written: AtomicBool::new(false),
            memory_backend: OnceLock::new(),
            memory_backend_reason: OnceLock::new(),
            regions_discovered: AtomicU64::new(0),
            guest_blocks_discovered: AtomicU64::new(0),
            regions_compiled: AtomicU64::new(0),
            region_entry_points: AtomicU64::new(0),
            secondary_entry_hits: AtomicU64::new(0),
            compiled_guest_instructions: AtomicU64::new(0),
            unique_guest_instructions: AtomicU64::new(0),
            overlapping_guest_instructions: AtomicU64::new(0),
            lookup_hits: AtomicU64::new(0),
            lookup_misses: AtomicU64::new(0),
            guest_instructions: AtomicU64::new(0),
            clif_instructions: AtomicU64::new(0),
            native_bytes: AtomicU64::new(0),
            compile_time_ns: AtomicU64::new(0),
            native_time_ns: AtomicU64::new(0),
            compiled_direct_reads: std::array::from_fn(|_| AtomicU64::new(0)),
            compiled_direct_writes: std::array::from_fn(|_| AtomicU64::new(0)),
            exit_reasons: std::array::from_fn(|_| AtomicU64::new(0)),
            invalidations: AtomicU64::new(0),
            invalidation_mapping: AtomicU64::new(0),
            invalidation_instruction_cache: AtomicU64::new(0),
            invalidation_device_write: AtomicU64::new(0),
            invalidation_host_write: AtomicU64::new(0),
            invalidation_cache_maintenance: AtomicU64::new(0),
            invalidation_unknown: AtomicU64::new(0),
            invalidation_relevant: AtomicU64::new(0),
            invalidation_irrelevant: AtomicU64::new(0),
            invalidation_history_lost: AtomicU64::new(0),
            regions_retired: AtomicU64::new(0),
            region_recompilations: AtomicU64::new(0),
            fault_baseline: nixe_cpu_direct_memory::metrics(),
            direct_mmap_calls: AtomicU64::new(0),
            direct_mmap_pages: AtomicU64::new(0),
            direct_mprotect_calls: AtomicU64::new(0),
            direct_mprotect_pages: AtomicU64::new(0),
            direct_replaced_pages: AtomicU64::new(0),
            direct_writable_alias_pages_armed: AtomicU64::new(0),
            direct_writable_alias_pages_revoked: AtomicU64::new(0),
            direct_host_failures: AtomicU64::new(0),
            direct_peak_mapped_pages: AtomicU64::new(0),
            direct_control_reserved_bytes: AtomicU64::new(0),
            direct_baseline_vma_count: AtomicU64::new(0),
            direct_peak_vma_growth: AtomicU64::new(0),
            direct_vma_samples: AtomicU64::new(0),
            transition_exclusive_acquisitions: AtomicU64::new(0),
            transition_exclusive_wait_ns: AtomicU64::new(0),
            transition_peak_shared_holders: AtomicU64::new(0),
            transition_safepoint_notifications: AtomicU64::new(0),
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn record_discovery(&self, guest_blocks: usize) {
        add(&self.regions_discovered, 1);
        add(&self.guest_blocks_discovered, guest_blocks as u128);
    }

    pub(super) fn record_memory_backend(&self, backend: CpuMemoryBackend, reason: Box<str>) {
        let _ = self.memory_backend.set(backend);
        let _ = self.memory_backend_reason.set(reason);
    }

    pub(super) fn record_compilation(
        &self,
        clif_instructions: usize,
        native_bytes: usize,
        elapsed: Duration,
    ) {
        add(&self.regions_compiled, 1);
        add(&self.clif_instructions, clif_instructions as u128);
        add(&self.native_bytes, native_bytes as u128);
        add(&self.compile_time_ns, elapsed.as_nanos());
    }

    pub(super) fn record_direct_access(&self, kind: NativeMemoryAccessKind, width: u8) {
        let Some(index) = direct_access_width_index(width) else {
            return;
        };
        let counters = match kind {
            NativeMemoryAccessKind::Read => &self.compiled_direct_reads,
            NativeMemoryAccessKind::Write => &self.compiled_direct_writes,
        };
        add(&counters[index], 1);
    }

    pub(super) fn record_lookup(&self, hit: bool) {
        add(
            if hit {
                &self.lookup_hits
            } else {
                &self.lookup_misses
            },
            1,
        );
    }

    pub(super) fn record_secondary_entry_hit(&self) {
        add(&self.secondary_entry_hits, 1);
    }

    pub(super) fn record_region_coverage(
        &self,
        entry_points: usize,
        instructions: usize,
        unique_instructions: usize,
    ) {
        add(&self.region_entry_points, entry_points as u128);
        add(&self.compiled_guest_instructions, instructions as u128);
        add(&self.unique_guest_instructions, unique_instructions as u128);
        add(
            &self.overlapping_guest_instructions,
            instructions.saturating_sub(unique_instructions) as u128,
        );
    }

    pub(super) fn record_guest_instructions(&self, instructions: u64) {
        add(&self.guest_instructions, u128::from(instructions));
    }

    pub(super) fn record_native_time(&self, elapsed: Duration) {
        add(&self.native_time_ns, elapsed.as_nanos());
    }

    pub(super) fn record_exit(&self, reason: ExitReason) {
        add(&self.exit_reasons[reason as usize], 1);
    }

    pub(super) fn record_invalidations(&self, metrics: &InvalidationMetrics) {
        add(&self.invalidations, u128::from(metrics.records));
        add(&self.invalidation_mapping, u128::from(metrics.mapping));
        add(
            &self.invalidation_instruction_cache,
            u128::from(metrics.instruction_cache),
        );
        add(
            &self.invalidation_device_write,
            u128::from(metrics.device_write),
        );
        add(
            &self.invalidation_host_write,
            u128::from(metrics.host_write),
        );
        add(
            &self.invalidation_cache_maintenance,
            u128::from(metrics.cache_maintenance),
        );
        add(&self.invalidation_unknown, u128::from(metrics.unknown));
        add(&self.invalidation_relevant, u128::from(metrics.relevant));
        add(
            &self.invalidation_irrelevant,
            u128::from(metrics.irrelevant),
        );
    }

    pub(super) fn record_history_lost(&self) {
        add(&self.invalidations, 1);
        add(&self.invalidation_history_lost, 1);
    }

    pub(super) fn record_regions_retired(&self, count: usize) {
        add(&self.regions_retired, count as u128);
    }

    pub(super) fn record_region_recompilation(&self) {
        add(&self.region_recompilations, 1);
    }

    pub(super) fn record_direct_memory(&self, metrics: DirectMemoryMetrics) {
        for (destination, value) in [
            (&self.direct_mmap_calls, metrics.arena.mmap_calls),
            (&self.direct_mmap_pages, metrics.arena.mmap_pages),
            (&self.direct_mprotect_calls, metrics.arena.mprotect_calls),
            (&self.direct_mprotect_pages, metrics.arena.mprotect_pages),
            (&self.direct_replaced_pages, metrics.arena.replaced_pages),
            (
                &self.direct_writable_alias_pages_armed,
                metrics.arena.writable_alias_pages_armed,
            ),
            (
                &self.direct_writable_alias_pages_revoked,
                metrics.arena.writable_alias_pages_revoked,
            ),
            (&self.direct_host_failures, metrics.arena.host_failures),
            (
                &self.direct_peak_mapped_pages,
                metrics.arena.peak_mapped_pages,
            ),
            (
                &self.direct_control_reserved_bytes,
                metrics.arena.control_reserved_bytes,
            ),
            (
                &self.direct_baseline_vma_count,
                metrics.arena.baseline_vma_count,
            ),
            (&self.direct_peak_vma_growth, metrics.arena.peak_vma_growth),
            (&self.direct_vma_samples, metrics.arena.vma_samples),
            (
                &self.transition_exclusive_acquisitions,
                metrics.execution_gate.exclusive_acquisitions,
            ),
            (
                &self.transition_exclusive_wait_ns,
                metrics.execution_gate.exclusive_wait_ns,
            ),
            (
                &self.transition_peak_shared_holders,
                metrics.execution_gate.peak_shared_holders,
            ),
            (
                &self.transition_safepoint_notifications,
                metrics.execution_gate.safepoint_notifications,
            ),
        ] {
            destination.fetch_max(value, Ordering::Relaxed);
        }
    }

    pub(super) fn write(&self, slow_memory_calls: u64) -> Result<(), Box<str>> {
        if self.written.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let mut report = String::new();
        writeln!(report, "version = {REPORT_VERSION}").unwrap();
        let backend = match self.memory_backend.get() {
            Some(CpuMemoryBackend::Checked) => "checked",
            Some(CpuMemoryBackend::LinuxDirect) => "linux_direct",
            None => "unbound",
        };
        let reason = self
            .memory_backend_reason
            .get()
            .map_or("memory binding was not recorded", Box::as_ref);
        writeln!(report, "memory_backend = {backend:?}").unwrap();
        writeln!(report, "memory_backend_reason = {reason:?}").unwrap();
        metric(&mut report, "regions_discovered", &self.regions_discovered);
        metric(
            &mut report,
            "guest_blocks_discovered",
            &self.guest_blocks_discovered,
        );
        metric(&mut report, "regions_compiled", &self.regions_compiled);
        metric(
            &mut report,
            "region_entry_points",
            &self.region_entry_points,
        );
        metric(
            &mut report,
            "secondary_entry_hits",
            &self.secondary_entry_hits,
        );
        metric(
            &mut report,
            "compiled_guest_instructions",
            &self.compiled_guest_instructions,
        );
        metric(
            &mut report,
            "unique_guest_instructions",
            &self.unique_guest_instructions,
        );
        metric(
            &mut report,
            "overlapping_guest_instructions",
            &self.overlapping_guest_instructions,
        );
        metric(&mut report, "lookup_hits", &self.lookup_hits);
        metric(&mut report, "lookup_misses", &self.lookup_misses);
        metric(&mut report, "guest_instructions", &self.guest_instructions);
        metric(&mut report, "clif_instructions", &self.clif_instructions);
        metric(&mut report, "native_bytes", &self.native_bytes);
        metric(&mut report, "compile_time_ns", &self.compile_time_ns);
        metric(&mut report, "native_time_ns", &self.native_time_ns);
        writeln!(report, "slow_memory_calls = {slow_memory_calls}").unwrap();
        metric(&mut report, "invalidations", &self.invalidations);
        let faults = nixe_cpu_direct_memory::metrics();
        writeln!(report, "\n[direct_faults]").unwrap();
        for (name, value) in [
            (
                "captured",
                faults.captured.saturating_sub(self.fault_baseline.captured),
            ),
            (
                "retries",
                faults.retries.saturating_sub(self.fault_baseline.retries),
            ),
            (
                "escapes",
                faults.escapes.saturating_sub(self.fault_baseline.escapes),
            ),
            (
                "fatal_dispatches",
                faults
                    .fatal_dispatches
                    .saturating_sub(self.fault_baseline.fatal_dispatches),
            ),
            (
                "unattributed",
                faults
                    .unattributed
                    .saturating_sub(self.fault_baseline.unattributed),
            ),
            (
                "nested",
                faults.nested.saturating_sub(self.fault_baseline.nested),
            ),
            (
                "sigbus",
                faults.sigbus.saturating_sub(self.fault_baseline.sigbus),
            ),
            (
                "checked_reads",
                faults
                    .checked_reads
                    .saturating_sub(self.fault_baseline.checked_reads),
            ),
            (
                "tracking_writes",
                faults
                    .tracking_writes
                    .saturating_sub(self.fault_baseline.tracking_writes),
            ),
            (
                "mmio_reads",
                faults
                    .mmio_reads
                    .saturating_sub(self.fault_baseline.mmio_reads),
            ),
            (
                "mmio_writes",
                faults
                    .mmio_writes
                    .saturating_sub(self.fault_baseline.mmio_writes),
            ),
            (
                "jit_faults",
                faults
                    .jit_faults
                    .saturating_sub(self.fault_baseline.jit_faults),
            ),
            (
                "interpreter_faults",
                faults
                    .interpreter_faults
                    .saturating_sub(self.fault_baseline.interpreter_faults),
            ),
            (
                "checked_writes",
                faults
                    .checked_writes
                    .saturating_sub(self.fault_baseline.checked_writes),
            ),
            (
                "semantic_faults",
                faults
                    .guest_faults
                    .saturating_sub(self.fault_baseline.guest_faults),
            ),
        ] {
            writeln!(report, "{name} = {value}").unwrap();
        }
        writeln!(report, "\n[compiled_direct_accesses]").unwrap();
        for (index, width) in [1_u8, 2, 4, 8, 16].into_iter().enumerate() {
            writeln!(
                report,
                "read_{width} = {}",
                self.compiled_direct_reads[index].load(Ordering::Relaxed)
            )
            .unwrap();
            writeln!(
                report,
                "write_{width} = {}",
                self.compiled_direct_writes[index].load(Ordering::Relaxed)
            )
            .unwrap();
        }
        writeln!(report, "\n[direct_memory]").unwrap();
        for (name, value) in [
            ("mmap_calls", &self.direct_mmap_calls),
            ("mmap_pages", &self.direct_mmap_pages),
            ("mprotect_calls", &self.direct_mprotect_calls),
            ("mprotect_pages", &self.direct_mprotect_pages),
            ("replaced_pages", &self.direct_replaced_pages),
            (
                "writable_alias_pages_armed",
                &self.direct_writable_alias_pages_armed,
            ),
            (
                "writable_alias_pages_revoked",
                &self.direct_writable_alias_pages_revoked,
            ),
            ("host_failures", &self.direct_host_failures),
            ("peak_mapped_pages", &self.direct_peak_mapped_pages),
            (
                "control_reserved_bytes",
                &self.direct_control_reserved_bytes,
            ),
            ("baseline_vma_count", &self.direct_baseline_vma_count),
            ("peak_vma_growth", &self.direct_peak_vma_growth),
            ("vma_samples", &self.direct_vma_samples),
            (
                "transition_exclusive_acquisitions",
                &self.transition_exclusive_acquisitions,
            ),
            (
                "transition_exclusive_wait_ns",
                &self.transition_exclusive_wait_ns,
            ),
            (
                "transition_peak_shared_holders",
                &self.transition_peak_shared_holders,
            ),
            (
                "transition_safepoint_notifications",
                &self.transition_safepoint_notifications,
            ),
        ] {
            metric(&mut report, name, value);
        }
        writeln!(report, "\n[invalidation_details]").unwrap();
        for (name, value) in [
            ("mapping", &self.invalidation_mapping),
            ("instruction_cache", &self.invalidation_instruction_cache),
            ("device_write", &self.invalidation_device_write),
            ("host_write", &self.invalidation_host_write),
            ("cache_maintenance", &self.invalidation_cache_maintenance),
            ("unknown", &self.invalidation_unknown),
            ("relevant", &self.invalidation_relevant),
            ("irrelevant", &self.invalidation_irrelevant),
            ("history_lost", &self.invalidation_history_lost),
            ("regions_retired", &self.regions_retired),
            ("region_recompilations", &self.region_recompilations),
        ] {
            metric(&mut report, name, value);
        }
        writeln!(report, "\n[exit_reasons]").unwrap();
        for (name, value) in [
            ("dispatch", ExitReason::Dispatch),
            ("budget", ExitReason::Budget),
            ("control", ExitReason::Control),
            ("architectural", ExitReason::Architectural),
            ("unsupported", ExitReason::Unsupported),
            ("data_fault", ExitReason::DataFault),
            ("scheduled", ExitReason::Scheduled),
            ("loader_return", ExitReason::LoaderReturn),
            ("internal", ExitReason::Internal),
            ("reconcile", ExitReason::Reconcile),
        ] {
            writeln!(
                report,
                "{name} = {}",
                self.exit_reasons[value as usize].load(Ordering::Relaxed)
            )
            .unwrap();
        }

        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .and_then(|mut file| file.write_all(report.as_bytes()))
            .map_err(|error| {
                format!(
                    "cannot write JIT performance report {}: {error}",
                    self.path.display()
                )
                .into_boxed_str()
            })
    }
}

fn metric(output: &mut String, name: &str, value: &AtomicU64) {
    writeln!(output, "{name} = {}", value.load(Ordering::Relaxed)).unwrap();
}

const fn direct_access_width_index(width: u8) -> Option<usize> {
    match width {
        1 => Some(0),
        2 => Some(1),
        4 => Some(2),
        8 => Some(3),
        16 => Some(4),
        _ => None,
    }
}

fn add(counter: &AtomicU64, value: u128) {
    counter.fetch_add(value.min(u128::from(u64::MAX)) as u64, Ordering::Relaxed);
}

fn file_stem(title: &str) -> String {
    const MAX_CHARACTERS: usize = 80;

    let mut stem = String::new();
    let mut separator = false;
    for character in title.trim().chars() {
        if character.is_alphanumeric() {
            if separator && !stem.is_empty() {
                stem.push('-');
            }
            separator = false;
            stem.extend(character.to_lowercase());
        } else {
            separator = true;
        }
        if stem.chars().count() >= MAX_CHARACTERS {
            break;
        }
    }
    if stem.is_empty() { "nixe".into() } else { stem }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_file_stem_is_safe_and_bounded() {
        assert_eq!(file_stem("es2gears"), "es2gears");
        assert_eq!(file_stem("  Super Mario™ Odyssey  "), "super-mario-odyssey");
        assert_eq!(file_stem("///"), "nixe");
        assert!(file_stem(&"a".repeat(100)).chars().count() <= 80);
    }

    #[test]
    fn report_counts_compiled_direct_accesses_by_kind_and_width() {
        let directory = tempfile::tempdir().unwrap();
        let performance = Performance::new(directory.path(), "access-counts").unwrap();
        performance.record_direct_access(NativeMemoryAccessKind::Read, 1);
        performance.record_direct_access(NativeMemoryAccessKind::Read, 8);
        performance.record_direct_access(NativeMemoryAccessKind::Write, 4);
        performance.record_direct_access(NativeMemoryAccessKind::Write, 16);
        performance.write(0).unwrap();

        let report: toml::Value = toml::from_str(
            &std::fs::read_to_string(performance.path()).expect("performance report was written"),
        )
        .unwrap();
        let accesses = &report["compiled_direct_accesses"];
        assert_eq!(accesses["read_1"].as_integer(), Some(1));
        assert_eq!(accesses["read_8"].as_integer(), Some(1));
        assert_eq!(accesses["write_4"].as_integer(), Some(1));
        assert_eq!(accesses["write_8"].as_integer(), Some(0));
        assert_eq!(accesses["write_16"].as_integer(), Some(1));
    }
}
