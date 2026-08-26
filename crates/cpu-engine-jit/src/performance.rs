//! Opt-in aggregate JIT performance counters.

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nixe_cpu_engine::{EngineDomainId, EngineExecutorId};

use crate::abi::{
    EXIT_ARCHITECTURAL, EXIT_BUDGET_EXHAUSTED, EXIT_DATA_FAULT, EXIT_DISPATCH, EXIT_INTERNAL,
    EXIT_INTERPRET_ONE, EXIT_LOADER_RETURN, EXIT_NONE, EXIT_PENDING_EVENT, EXIT_SAFEPOINT,
    EXIT_SCHEDULED, EXIT_UNSUPPORTED,
};
use crate::compilation_pool::CompilationPoolSnapshot;

const REPORT_VERSION: u32 = 6;
const EXIT_KIND_COUNT: usize = EXIT_INTERNAL as usize + 1;

pub(crate) struct JitPerformanceReport {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl JitPerformanceReport {
    pub(crate) fn new(path: &Path) -> Result<Self, Box<str>> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "cannot create JIT performance report directory {}: {error}",
                    parent.display()
                )
                .into_boxed_str()
            })?;
        }
        let started_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let header = format!(
            "nixe_jit_performance_version={REPORT_VERSION}\nprocess_id={}\nstarted_unix_ms={started_unix_ms}\nhost_debug_assertions={}\ncranelift_opt_level=\"none\"\ncranelift_opt_level_source=\"default\"\ncold_tier=\"interpreter\"\nhot_promotion_visits=3\nasynchronous_compilation=true\ncompilation_workers_policy=\"max(1,host_logical_cores/2)\"\nnative_poll_fast_path=true\n",
            std::process::id(),
            cfg!(debug_assertions),
        );
        fs::write(path, header).map_err(|error| {
            format!(
                "cannot initialize JIT performance report {}: {error}",
                path.display()
            )
            .into_boxed_str()
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            write_lock: Mutex::new(()),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    fn append_domain(
        &self,
        domain: EngineDomainId,
        elapsed: Duration,
        executors: &[(EngineExecutorId, ExecutorPerformance)],
        cache: Option<&CachePerformanceSnapshot>,
        compilation_pool: Option<&CompilationPoolSnapshot>,
    ) -> Result<(), Box<str>> {
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut total = ExecutorPerformance::default();
        for (_, counters) in executors {
            total.merge(counters);
        }
        if let Some(pool) = compilation_pool {
            total.codegen_completed = total
                .codegen_completed
                .saturating_add(pool.completed.saturating_sub(pool.failed));
            total.codegen_ns = total.codegen_ns.saturating_add(pool.codegen_ns);
            total.compiled_guest_instructions = total
                .compiled_guest_instructions
                .saturating_add(pool.compiled_guest_instructions);
            total.compiled_ir_operations = total
                .compiled_ir_operations
                .saturating_add(pool.compiled_ir_operations);
            total.compiled_native_bytes = total
                .compiled_native_bytes
                .saturating_add(pool.compiled_native_bytes);
            total.compiled_native_named_operations = total
                .compiled_native_named_operations
                .saturating_add(pool.compiled_native_named_operations);
            total.compiled_semantic_helper_callsites = total
                .compiled_semantic_helper_callsites
                .saturating_add(pool.compiled_semantic_helper_callsites);
        }
        let mut output = String::new();
        writeln!(output).unwrap();
        writeln!(output, "[domain.{}]", domain.get()).unwrap();
        writeln!(output, "elapsed_ns={}", duration_ns(elapsed)).unwrap();
        writeln!(output, "executors={}", executors.len()).unwrap();
        render_counters(&mut output, &total);
        if let Some(cache) = cache {
            render_cache_counters(&mut output, cache);
        }
        if let Some(pool) = compilation_pool {
            render_compilation_pool_counters(&mut output, pool);
        }
        for (executor, counters) in executors {
            writeln!(output).unwrap();
            writeln!(
                output,
                "[domain.{}.executor.{}]",
                domain.get(),
                executor.get()
            )
            .unwrap();
            render_counters(&mut output, counters);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|error| {
                format!(
                    "cannot open JIT performance report {}: {error}",
                    self.path.display()
                )
                .into_boxed_str()
            })?;
        file.write_all(output.as_bytes()).map_err(|error| {
            format!(
                "cannot append JIT performance report {}: {error}",
                self.path.display()
            )
            .into_boxed_str()
        })
    }
}

pub(crate) struct DomainPerformance {
    report: Arc<JitPerformanceReport>,
    domain: EngineDomainId,
    started: Instant,
    executors: Mutex<Vec<(EngineExecutorId, ExecutorPerformance)>>,
}

impl DomainPerformance {
    pub(crate) fn new(report: Arc<JitPerformanceReport>, domain: EngineDomainId) -> Self {
        Self {
            report,
            domain,
            started: Instant::now(),
            executors: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn record_executor(
        &self,
        executor: EngineExecutorId,
        counters: ExecutorPerformance,
    ) {
        let mut executors = self
            .executors
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((_, previous)) = executors
            .iter_mut()
            .find(|(recorded, _)| *recorded == executor)
        {
            previous.merge(&counters);
        } else {
            executors.push((executor, counters));
            executors.sort_unstable_by_key(|(executor, _)| *executor);
        }
    }

    pub(crate) fn write(
        &self,
        cache: Option<&CachePerformanceSnapshot>,
        compilation_pool: Option<&CompilationPoolSnapshot>,
    ) -> Result<(), Box<str>> {
        let executors = self
            .executors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        self.report.append_domain(
            self.domain,
            self.started.elapsed(),
            &executors,
            cache,
            compilation_pool,
        )
    }

    pub(crate) fn report_path(&self) -> &Path {
        self.report.path()
    }
}

#[derive(Clone, Default)]
pub(crate) struct ExecutorPerformance {
    pub(crate) run_slices: u64,
    pub(crate) instructions_reported: u64,
    pub(crate) local_cache_hits: u64,
    pub(crate) local_cache_misses: u64,
    pub(crate) cache_miss_resolution_calls: u64,
    pub(crate) cache_miss_resolution_ns: u64,
    pub(crate) cold_interpreter_steps: u64,
    pub(crate) hot_promotions: u64,
    pub(crate) compilation_attempts: u64,
    pub(crate) codegen_completed: u64,
    pub(crate) frontend_ns: u64,
    pub(crate) codegen_ns: u64,
    pub(crate) compiled_guest_instructions: u64,
    pub(crate) compiled_ir_operations: u64,
    pub(crate) compiled_native_bytes: u64,
    pub(crate) compiled_native_named_operations: u64,
    pub(crate) compiled_semantic_helper_callsites: u64,
    pub(crate) native_entries: u64,
    pub(crate) native_ns: u64,
    pub(crate) native_instructions: u64,
    pub(crate) link_attempts: u64,
    pub(crate) link_successes: u64,
    pub(crate) link_stale: u64,
    pub(crate) invalidation_batches: u64,
    pub(crate) invalidation_records: u64,
    pub(crate) invalidation_history_lost: u64,
    pub(crate) filtered_invalidation_polls: u64,
    pub(crate) relevant_invalidation_polls: u64,
    pub(crate) helper_memory_reads: u64,
    pub(crate) helper_memory_writes: u64,
    pub(crate) helper_atomics: u64,
    pub(crate) helper_exclusives: u64,
    pub(crate) helper_semantics: u64,
    pub(crate) helper_system_polls: u64,
    pub(crate) helper_system_other: u64,
    exit_kinds: [u64; EXIT_KIND_COUNT],
}

impl ExecutorPerformance {
    pub(crate) fn add_duration(target: &mut u64, elapsed: Duration) {
        *target = target.saturating_add(duration_ns(elapsed));
    }

    pub(crate) fn record_native_exit(&mut self, kind: u32, instructions: u64) {
        self.native_instructions = self.native_instructions.saturating_add(instructions);
        if let Some(count) = self.exit_kinds.get_mut(kind as usize) {
            *count = count.saturating_add(1);
        }
    }

    fn merge(&mut self, other: &Self) {
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {
                $(self.$field = self.$field.saturating_add(other.$field);)+
            };
        }
        add!(
            run_slices,
            instructions_reported,
            local_cache_hits,
            local_cache_misses,
            cache_miss_resolution_calls,
            cache_miss_resolution_ns,
            cold_interpreter_steps,
            hot_promotions,
            compilation_attempts,
            codegen_completed,
            frontend_ns,
            codegen_ns,
            compiled_guest_instructions,
            compiled_ir_operations,
            compiled_native_bytes,
            compiled_native_named_operations,
            compiled_semantic_helper_callsites,
            native_entries,
            native_ns,
            native_instructions,
            link_attempts,
            link_successes,
            link_stale,
            invalidation_batches,
            invalidation_records,
            invalidation_history_lost,
            filtered_invalidation_polls,
            relevant_invalidation_polls,
            helper_memory_reads,
            helper_memory_writes,
            helper_atomics,
            helper_exclusives,
            helper_semantics,
            helper_system_polls,
            helper_system_other,
        );
        for (target, source) in self.exit_kinds.iter_mut().zip(other.exit_kinds) {
            *target = target.saturating_add(source);
        }
    }
}

fn render_counters(output: &mut String, counters: &ExecutorPerformance) {
    macro_rules! field {
        ($name:literal, $value:expr) => {
            writeln!(output, "{}={}", $name, $value).unwrap()
        };
    }
    field!("run_slices", counters.run_slices);
    field!("instructions_reported", counters.instructions_reported);
    field!("local_cache_hits", counters.local_cache_hits);
    field!("local_cache_misses", counters.local_cache_misses);
    field!(
        "cache_miss_resolution_calls",
        counters.cache_miss_resolution_calls
    );
    field!(
        "cache_miss_resolution_ns",
        counters.cache_miss_resolution_ns
    );
    field!("cold_interpreter_steps", counters.cold_interpreter_steps);
    field!("hot_promotions", counters.hot_promotions);
    field!("compilation_attempts", counters.compilation_attempts);
    field!("codegen_completed", counters.codegen_completed);
    field!("frontend_ns", counters.frontend_ns);
    field!("codegen_ns", counters.codegen_ns);
    field!(
        "compiled_guest_instructions",
        counters.compiled_guest_instructions
    );
    field!("compiled_ir_operations", counters.compiled_ir_operations);
    field!("compiled_native_bytes", counters.compiled_native_bytes);
    field!(
        "compiled_native_named_operations",
        counters.compiled_native_named_operations
    );
    field!(
        "compiled_semantic_helper_callsites",
        counters.compiled_semantic_helper_callsites
    );
    field!("native_entries", counters.native_entries);
    field!("native_ns", counters.native_ns);
    field!("native_instructions", counters.native_instructions);
    let native_ips = if counters.native_ns == 0 {
        0.0
    } else {
        counters.native_instructions as f64 * 1_000_000_000.0 / counters.native_ns as f64
    };
    writeln!(output, "native_ips={native_ips:.3}").unwrap();
    field!("link_attempts", counters.link_attempts);
    field!("link_successes", counters.link_successes);
    field!("link_stale", counters.link_stale);
    field!("invalidation_batches", counters.invalidation_batches);
    field!("invalidation_records", counters.invalidation_records);
    field!(
        "invalidation_history_lost",
        counters.invalidation_history_lost
    );
    field!(
        "filtered_invalidation_polls",
        counters.filtered_invalidation_polls
    );
    field!(
        "relevant_invalidation_polls",
        counters.relevant_invalidation_polls
    );
    field!("helper_memory_reads", counters.helper_memory_reads);
    field!("helper_memory_writes", counters.helper_memory_writes);
    field!("helper_atomics", counters.helper_atomics);
    field!("helper_exclusives", counters.helper_exclusives);
    field!("helper_semantics", counters.helper_semantics);
    field!("helper_system_polls", counters.helper_system_polls);
    field!("helper_system_other", counters.helper_system_other);
    for (kind, name) in [
        (EXIT_NONE, "none"),
        (EXIT_INTERPRET_ONE, "interpret_one"),
        (EXIT_BUDGET_EXHAUSTED, "budget_exhausted"),
        (EXIT_SAFEPOINT, "safepoint"),
        (EXIT_PENDING_EVENT, "pending_event"),
        (EXIT_LOADER_RETURN, "loader_return"),
        (EXIT_DISPATCH, "dispatch"),
        (EXIT_ARCHITECTURAL, "architectural"),
        (EXIT_UNSUPPORTED, "unsupported"),
        (EXIT_DATA_FAULT, "data_fault"),
        (EXIT_SCHEDULED, "scheduled"),
        (EXIT_INTERNAL, "internal"),
    ] {
        writeln!(output, "exit_{name}={}", counters.exit_kinds[kind as usize]).unwrap();
    }
}

fn render_compilation_pool_counters(output: &mut String, counters: &CompilationPoolSnapshot) {
    macro_rules! field {
        ($name:literal, $value:expr) => {
            writeln!(output, "compilation_pool_{}={}", $name, $value).unwrap()
        };
    }
    field!("workers", counters.workers);
    field!("queue_capacity", counters.queue_capacity);
    field!("enqueued", counters.enqueued);
    field!("duplicate_requests", counters.duplicate_requests);
    field!("queued_discarded", counters.queued_discarded);
    field!("started", counters.started);
    field!("completed", counters.completed);
    field!("failed", counters.failed);
    field!("completed_discarded", counters.completed_discarded);
    field!("peak_queued", counters.peak_queued);
    field!("peak_running", counters.peak_running);
    field!("codegen_ns", counters.codegen_ns);
    field!(
        "compiled_guest_instructions",
        counters.compiled_guest_instructions
    );
    field!("compiled_ir_operations", counters.compiled_ir_operations);
    field!("compiled_native_bytes", counters.compiled_native_bytes);
    field!(
        "compiled_native_named_operations",
        counters.compiled_native_named_operations
    );
    field!(
        "compiled_semantic_helper_callsites",
        counters.compiled_semantic_helper_callsites
    );
}

#[derive(Clone, Default)]
pub(crate) struct CachePerformanceSnapshot {
    pub(crate) global_ready_hits: u64,
    pub(crate) global_waits: u64,
    pub(crate) global_builds: u64,
    pub(crate) build_stale: u64,
    pub(crate) build_failures: u64,
    pub(crate) regions_published: u64,
    pub(crate) publish_stale: u64,
    pub(crate) unique_region_keys: u64,
    pub(crate) recompiled_region_keys: u64,
    pub(crate) repeat_builds: u64,
    pub(crate) regions_evicted_capacity: u64,
    pub(crate) regions_retired_invalidation: u64,
    pub(crate) links_unlinked_capacity: u64,
    pub(crate) links_unlinked_invalidation: u64,
    pub(crate) invalidation_batches: u64,
    pub(crate) invalidation_executable_content: u64,
    pub(crate) invalidation_mapping: u64,
    pub(crate) invalidation_instruction_cache: u64,
    pub(crate) invalidation_history_lost: u64,
    pub(crate) fast_irrelevant_invalidations: u64,
    pub(crate) peak_live_regions: u64,
    pub(crate) peak_live_mapped_bytes: u64,
    pub(crate) peak_cache_slots: u64,
}

fn render_cache_counters(output: &mut String, counters: &CachePerformanceSnapshot) {
    macro_rules! field {
        ($name:literal, $value:expr) => {
            writeln!(output, "{}={}", $name, $value).unwrap()
        };
    }
    field!("global_cache_ready_hits", counters.global_ready_hits);
    field!("global_cache_waits", counters.global_waits);
    field!("global_cache_builds", counters.global_builds);
    field!("cache_build_stale", counters.build_stale);
    field!("cache_build_failures", counters.build_failures);
    field!("regions_published", counters.regions_published);
    field!("publish_stale", counters.publish_stale);
    field!("unique_region_keys", counters.unique_region_keys);
    field!("recompiled_region_keys", counters.recompiled_region_keys);
    field!("repeat_builds", counters.repeat_builds);
    field!(
        "regions_evicted_capacity",
        counters.regions_evicted_capacity
    );
    field!(
        "regions_retired_invalidation",
        counters.regions_retired_invalidation
    );
    field!("links_unlinked_capacity", counters.links_unlinked_capacity);
    field!(
        "links_unlinked_invalidation",
        counters.links_unlinked_invalidation
    );
    field!("cache_invalidation_batches", counters.invalidation_batches);
    field!(
        "invalidation_executable_content",
        counters.invalidation_executable_content
    );
    field!("invalidation_mapping", counters.invalidation_mapping);
    field!(
        "invalidation_instruction_cache",
        counters.invalidation_instruction_cache
    );
    field!(
        "cache_invalidation_history_lost",
        counters.invalidation_history_lost
    );
    field!(
        "cache_fast_irrelevant_invalidations",
        counters.fast_irrelevant_invalidations
    );
    field!("peak_live_regions", counters.peak_live_regions);
    field!("peak_live_mapped_bytes", counters.peak_live_mapped_bytes);
    field!("peak_cache_slots", counters.peak_cache_slots);
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
