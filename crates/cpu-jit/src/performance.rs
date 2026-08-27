//! Low-overhead aggregate measurements for the direct JIT.

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nixe_memory::{MemoryInvalidation, MemoryInvalidationKind, MemoryInvalidationOrigin};

const REPORT_VERSION: u32 = 4;
const EXIT_REASON_COUNT: usize = 10;

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
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn record_discovery(&self, guest_blocks: usize) {
        add(&self.regions_discovered, 1);
        add(&self.guest_blocks_discovered, guest_blocks as u128);
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

    pub(super) fn write(&self, slow_memory_calls: u64) -> Result<(), Box<str>> {
        if self.written.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let mut report = String::new();
        writeln!(report, "version = {REPORT_VERSION}").unwrap();
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
}
