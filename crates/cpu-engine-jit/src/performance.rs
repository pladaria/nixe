//! Opt-in aggregate JIT performance counters.

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use nixe_cpu::ir::region::IrRegion;
use nixe_cpu_engine::{EngineDomainId, EngineExecutorId};

use crate::abi::{
    EXIT_ARCHITECTURAL, EXIT_BUDGET_EXHAUSTED, EXIT_DATA_FAULT, EXIT_DISPATCH, EXIT_INTERNAL,
    EXIT_LOADER_RETURN, EXIT_NONE, EXIT_PENDING_EVENT, EXIT_SAFEPOINT, EXIT_SCHEDULED,
    EXIT_UNSUPPORTED,
};
use crate::compilation_pool::CompilationPoolSnapshot;
use crate::compiler::{CompilationMetrics, CompiledRegion};
use crate::{cache::LinkOutcome, links::LinkKind};

const REPORT_VERSION: u32 = 10;
const EXIT_KIND_COUNT: usize = EXIT_INTERNAL as usize + 1;

pub(crate) struct JitPerformanceReport {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl JitPerformanceReport {
    pub(crate) fn new(directory: &Path) -> Result<Self, Box<str>> {
        fs::create_dir_all(directory).map_err(|error| {
            format!(
                "cannot create JIT performance report directory {}: {error}",
                directory.display()
            )
            .into_boxed_str()
        })?;
        let started = SystemTime::now();
        let started_unix_ms = started
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let timestamp: DateTime<Utc> = started.into();
        let path = directory.join(format!(
            "jit-performance-{}-{}.toml",
            timestamp.format("%Y%m%dT%H%M%S%.3fZ"),
            std::process::id()
        ));
        let header = format!(
            "nixe_jit_performance_version={REPORT_VERSION}\nprocess_id={}\nstarted_unix_ms={started_unix_ms}\nhost_debug_assertions={}\nlight_tier=\"cranelift-none-single-pass-synchronous\"\nlight_cranelift_verifier={}\noptimized_tier=\"cranelift-speed-backtracking-asynchronous\"\noptimized_cranelift_verifier=true\nhot_promotion_entries=100\npromotion_queue=\"lifo-deduplicated-direct-publication\"\ncompilation_workers_policy=\"max(1,host_logical_cores/2)\"\ncranelift_pass_timing=true\nnative_poll_fast_path=true\nlink_pic_policy=\"four-way-round-robin-replacement\"\nguest_invalidation_prefilter=true\ninstruction_boundary=\"entry-hoisted-control-ssa-budget-shared-exit\"\n",
            std::process::id(),
            cfg!(debug_assertions),
            cfg!(debug_assertions),
        );
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .and_then(|mut report| report.write_all(header.as_bytes()))
            .map_err(|error| {
                format!(
                    "cannot initialize JIT performance report {}: {error}",
                    path.display()
                )
                .into_boxed_str()
            })?;
        Ok(Self {
            path,
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
            total.codegen_completed = total.codegen_completed.saturating_add(pool.published);
            total.worker_compilation_ns = total
                .worker_compilation_ns
                .saturating_add(pool.worker_total_ns);
            total.nixe_ir_verify_ns = total
                .nixe_ir_verify_ns
                .saturating_add(pool.nixe_ir_verify_ns);
            total.state_validation_ns = total
                .state_validation_ns
                .saturating_add(pool.state_validation_ns);
            total.lowering_ns = total.lowering_ns.saturating_add(pool.lowering_ns);
            total.cranelift_compile_ns = total
                .cranelift_compile_ns
                .saturating_add(pool.cranelift_compile_ns);
            total.cranelift_verifier_ns = total
                .cranelift_verifier_ns
                .saturating_add(pool.cranelift_verifier_ns);
            total.cranelift_optimize_ns = total
                .cranelift_optimize_ns
                .saturating_add(pool.cranelift_optimize_ns);
            total.cranelift_vcode_lower_ns = total
                .cranelift_vcode_lower_ns
                .saturating_add(pool.cranelift_vcode_lower_ns);
            total.cranelift_regalloc_ns = total
                .cranelift_regalloc_ns
                .saturating_add(pool.cranelift_regalloc_ns);
            total.cranelift_emit_ns = total
                .cranelift_emit_ns
                .saturating_add(pool.cranelift_emit_ns);
            total.cranelift_other_ns = total
                .cranelift_other_ns
                .saturating_add(pool.cranelift_other_ns);
            total.publication_total_ns = total
                .publication_total_ns
                .saturating_add(pool.publication_total_ns);
            total.publication_lock_wait_ns = total
                .publication_lock_wait_ns
                .saturating_add(pool.publication_lock_wait_ns);
            total.publication_allocation_ns = total
                .publication_allocation_ns
                .saturating_add(pool.publication_allocation_ns);
            total.publication_zero_copy_ns = total
                .publication_zero_copy_ns
                .saturating_add(pool.publication_zero_copy_ns);
            total.publication_protection_ns = total
                .publication_protection_ns
                .saturating_add(pool.publication_protection_ns);
            total.publication_instruction_cache_ns = total
                .publication_instruction_cache_ns
                .saturating_add(pool.publication_instruction_cache_ns);
            total.diagnostics_ns = total.diagnostics_ns.saturating_add(pool.diagnostics_ns);
            total.pending_region_ns = total
                .pending_region_ns
                .saturating_add(pool.pending_region_ns);
            total.compiled_guest_instructions = total
                .compiled_guest_instructions
                .saturating_add(pool.compiled_guest_instructions);
            total.compiled_ir_operations = total
                .compiled_ir_operations
                .saturating_add(pool.compiled_ir_operations);
            total.compiled_clif_instructions = total
                .compiled_clif_instructions
                .saturating_add(pool.compiled_clif_instructions);
            total.compiled_clif_blocks = total
                .compiled_clif_blocks
                .saturating_add(pool.compiled_clif_blocks);
            total.compiled_native_code_bytes = total
                .compiled_native_code_bytes
                .saturating_add(pool.compiled_native_code_bytes);
            total.compiled_native_mapped_bytes = total
                .compiled_native_mapped_bytes
                .saturating_add(pool.compiled_native_mapped_bytes);
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
    pub(crate) hot_promotions: u64,
    pub(crate) compilation_attempts: u64,
    pub(crate) codegen_completed: u64,
    pub(crate) frontend_ns: u64,
    pub(crate) worker_compilation_ns: u64,
    pub(crate) nixe_ir_verify_ns: u64,
    pub(crate) state_validation_ns: u64,
    pub(crate) lowering_ns: u64,
    pub(crate) cranelift_compile_ns: u64,
    pub(crate) cranelift_verifier_ns: u64,
    pub(crate) cranelift_optimize_ns: u64,
    pub(crate) cranelift_vcode_lower_ns: u64,
    pub(crate) cranelift_regalloc_ns: u64,
    pub(crate) cranelift_emit_ns: u64,
    pub(crate) cranelift_other_ns: u64,
    pub(crate) publication_total_ns: u64,
    pub(crate) publication_lock_wait_ns: u64,
    pub(crate) publication_allocation_ns: u64,
    pub(crate) publication_zero_copy_ns: u64,
    pub(crate) publication_protection_ns: u64,
    pub(crate) publication_instruction_cache_ns: u64,
    pub(crate) diagnostics_ns: u64,
    pub(crate) pending_region_ns: u64,
    pub(crate) compiled_guest_instructions: u64,
    pub(crate) compiled_ir_operations: u64,
    pub(crate) compiled_clif_instructions: u64,
    pub(crate) compiled_clif_blocks: u64,
    pub(crate) compiled_native_code_bytes: u64,
    pub(crate) compiled_native_mapped_bytes: u64,
    pub(crate) compiled_native_named_operations: u64,
    pub(crate) compiled_semantic_helper_callsites: u64,
    pub(crate) native_entries: u64,
    pub(crate) native_ns: u64,
    pub(crate) native_instructions: u64,
    pub(crate) link_direct_published: u64,
    pub(crate) link_direct_already_present: u64,
    pub(crate) link_direct_pic_full: u64,
    pub(crate) link_indirect_published: u64,
    pub(crate) link_indirect_already_present: u64,
    pub(crate) link_indirect_pic_full: u64,
    pub(crate) link_stale: u64,
    pub(crate) invalidation_batches: u64,
    pub(crate) invalidation_records: u64,
    pub(crate) invalidation_history_lost: u64,
    pub(crate) invalidation_checks_filtered: u64,
    pub(crate) invalidation_checks_relevant: u64,
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

    pub(crate) fn record_light_compilation(
        &mut self,
        region: &IrRegion,
        compiled: &CompiledRegion,
        metrics: CompilationMetrics,
    ) {
        self.codegen_completed = self.codegen_completed.saturating_add(1);
        macro_rules! add {
            ($field:ident, $value:expr) => {
                self.$field = self.$field.saturating_add($value)
            };
        }
        add!(nixe_ir_verify_ns, metrics.nixe_ir_verify_ns);
        add!(state_validation_ns, metrics.state_validation_ns);
        add!(lowering_ns, metrics.lowering_ns);
        add!(cranelift_compile_ns, metrics.cranelift_compile_ns);
        add!(cranelift_verifier_ns, metrics.cranelift_verifier_ns);
        add!(cranelift_optimize_ns, metrics.cranelift_optimize_ns);
        add!(cranelift_vcode_lower_ns, metrics.cranelift_vcode_lower_ns);
        add!(cranelift_regalloc_ns, metrics.cranelift_regalloc_ns);
        add!(cranelift_emit_ns, metrics.cranelift_emit_ns);
        add!(cranelift_other_ns, metrics.cranelift_other_ns);
        add!(publication_total_ns, metrics.publication.total_ns);
        add!(publication_lock_wait_ns, metrics.publication.lock_wait_ns);
        add!(publication_allocation_ns, metrics.publication.allocation_ns);
        add!(publication_zero_copy_ns, metrics.publication.zero_copy_ns);
        add!(publication_protection_ns, metrics.publication.protection_ns);
        add!(
            publication_instruction_cache_ns,
            metrics.publication.instruction_cache_ns
        );
        add!(
            compiled_guest_instructions,
            u64::from(region.metadata.guest_instruction_count)
        );
        add!(
            compiled_ir_operations,
            u64::from(region.metadata.ir_operation_count)
        );
        add!(compiled_clif_instructions, metrics.clif_instructions);
        add!(compiled_clif_blocks, metrics.clif_blocks);
        add!(compiled_native_code_bytes, metrics.native_code_bytes);
        add!(compiled_native_mapped_bytes, metrics.native_mapped_bytes);
        add!(
            compiled_native_named_operations,
            compiled.metadata.native_named_operations
        );
        add!(
            compiled_semantic_helper_callsites,
            compiled.metadata.semantic_calls.len() as u64
        );
    }

    pub(crate) fn record_native_exit(&mut self, kind: u32, instructions: u64) {
        self.native_instructions = self.native_instructions.saturating_add(instructions);
        if let Some(count) = self.exit_kinds.get_mut(kind as usize) {
            *count = count.saturating_add(1);
        }
    }

    pub(crate) fn record_link(&mut self, kind: LinkKind, outcome: LinkOutcome) {
        let counter = match (kind, outcome) {
            (LinkKind::Direct, LinkOutcome::Published) => &mut self.link_direct_published,
            (LinkKind::Direct, LinkOutcome::AlreadyPresent) => {
                &mut self.link_direct_already_present
            }
            (LinkKind::Direct, LinkOutcome::PicFull) => &mut self.link_direct_pic_full,
            (LinkKind::Indirect, LinkOutcome::Published) => &mut self.link_indirect_published,
            (LinkKind::Indirect, LinkOutcome::AlreadyPresent) => {
                &mut self.link_indirect_already_present
            }
            (LinkKind::Indirect, LinkOutcome::PicFull) => &mut self.link_indirect_pic_full,
        };
        *counter = counter.saturating_add(1);
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
            hot_promotions,
            compilation_attempts,
            codegen_completed,
            frontend_ns,
            worker_compilation_ns,
            nixe_ir_verify_ns,
            state_validation_ns,
            lowering_ns,
            cranelift_compile_ns,
            cranelift_verifier_ns,
            cranelift_optimize_ns,
            cranelift_vcode_lower_ns,
            cranelift_regalloc_ns,
            cranelift_emit_ns,
            cranelift_other_ns,
            publication_total_ns,
            publication_lock_wait_ns,
            publication_allocation_ns,
            publication_zero_copy_ns,
            publication_protection_ns,
            publication_instruction_cache_ns,
            diagnostics_ns,
            pending_region_ns,
            compiled_guest_instructions,
            compiled_ir_operations,
            compiled_clif_instructions,
            compiled_clif_blocks,
            compiled_native_code_bytes,
            compiled_native_mapped_bytes,
            compiled_native_named_operations,
            compiled_semantic_helper_callsites,
            native_entries,
            native_ns,
            native_instructions,
            link_direct_published,
            link_direct_already_present,
            link_direct_pic_full,
            link_indirect_published,
            link_indirect_already_present,
            link_indirect_pic_full,
            link_stale,
            invalidation_batches,
            invalidation_records,
            invalidation_history_lost,
            invalidation_checks_filtered,
            invalidation_checks_relevant,
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
    field!("hot_promotions", counters.hot_promotions);
    field!("compilation_attempts", counters.compilation_attempts);
    field!("codegen_completed", counters.codegen_completed);
    field!("frontend_ns", counters.frontend_ns);
    field!("worker_compilation_ns", counters.worker_compilation_ns);
    field!("nixe_ir_verify_ns", counters.nixe_ir_verify_ns);
    field!("state_validation_ns", counters.state_validation_ns);
    field!("lowering_ns", counters.lowering_ns);
    field!("cranelift_compile_ns", counters.cranelift_compile_ns);
    field!("cranelift_verifier_ns", counters.cranelift_verifier_ns);
    field!("cranelift_optimize_ns", counters.cranelift_optimize_ns);
    field!(
        "cranelift_vcode_lower_ns",
        counters.cranelift_vcode_lower_ns
    );
    field!("cranelift_regalloc_ns", counters.cranelift_regalloc_ns);
    field!("cranelift_emit_ns", counters.cranelift_emit_ns);
    field!("cranelift_other_ns", counters.cranelift_other_ns);
    field!("publication_total_ns", counters.publication_total_ns);
    field!(
        "publication_lock_wait_ns",
        counters.publication_lock_wait_ns
    );
    field!(
        "publication_allocation_ns",
        counters.publication_allocation_ns
    );
    field!(
        "publication_zero_copy_ns",
        counters.publication_zero_copy_ns
    );
    field!(
        "publication_protection_ns",
        counters.publication_protection_ns
    );
    field!(
        "publication_instruction_cache_ns",
        counters.publication_instruction_cache_ns
    );
    field!("diagnostics_ns", counters.diagnostics_ns);
    field!("pending_region_ns", counters.pending_region_ns);
    field!(
        "compiled_guest_instructions",
        counters.compiled_guest_instructions
    );
    field!("compiled_ir_operations", counters.compiled_ir_operations);
    field!(
        "compiled_clif_instructions",
        counters.compiled_clif_instructions
    );
    field!("compiled_clif_blocks", counters.compiled_clif_blocks);
    field!(
        "compiled_native_code_bytes",
        counters.compiled_native_code_bytes
    );
    field!(
        "compiled_native_mapped_bytes",
        counters.compiled_native_mapped_bytes
    );
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
    field!("link_direct_published", counters.link_direct_published);
    field!(
        "link_direct_already_present",
        counters.link_direct_already_present
    );
    field!("link_direct_pic_full", counters.link_direct_pic_full);
    field!("link_indirect_published", counters.link_indirect_published);
    field!(
        "link_indirect_already_present",
        counters.link_indirect_already_present
    );
    field!("link_indirect_pic_full", counters.link_indirect_pic_full);
    field!("link_stale", counters.link_stale);
    field!("invalidation_batches", counters.invalidation_batches);
    field!("invalidation_records", counters.invalidation_records);
    field!(
        "invalidation_history_lost",
        counters.invalidation_history_lost
    );
    field!(
        "invalidation_checks_filtered",
        counters.invalidation_checks_filtered
    );
    field!(
        "invalidation_checks_relevant",
        counters.invalidation_checks_relevant
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
    field!("published", counters.published);
    field!("stale", counters.stale);
    field!("failed", counters.failed);
    field!("peak_queued", counters.peak_queued);
    field!("peak_running", counters.peak_running);
    field!("worker_total_ns", counters.worker_total_ns);
    field!("nixe_ir_verify_ns", counters.nixe_ir_verify_ns);
    field!("state_validation_ns", counters.state_validation_ns);
    field!("lowering_ns", counters.lowering_ns);
    field!("cranelift_compile_ns", counters.cranelift_compile_ns);
    field!("cranelift_verifier_ns", counters.cranelift_verifier_ns);
    field!("cranelift_optimize_ns", counters.cranelift_optimize_ns);
    field!(
        "cranelift_vcode_lower_ns",
        counters.cranelift_vcode_lower_ns
    );
    field!("cranelift_regalloc_ns", counters.cranelift_regalloc_ns);
    field!("cranelift_emit_ns", counters.cranelift_emit_ns);
    field!("cranelift_other_ns", counters.cranelift_other_ns);
    field!("publication_total_ns", counters.publication_total_ns);
    field!(
        "publication_lock_wait_ns",
        counters.publication_lock_wait_ns
    );
    field!(
        "publication_allocation_ns",
        counters.publication_allocation_ns
    );
    field!(
        "publication_zero_copy_ns",
        counters.publication_zero_copy_ns
    );
    field!(
        "publication_protection_ns",
        counters.publication_protection_ns
    );
    field!(
        "publication_instruction_cache_ns",
        counters.publication_instruction_cache_ns
    );
    field!("diagnostics_ns", counters.diagnostics_ns);
    field!("pending_region_ns", counters.pending_region_ns);
    field!(
        "compiled_guest_instructions",
        counters.compiled_guest_instructions
    );
    field!("compiled_ir_operations", counters.compiled_ir_operations);
    field!(
        "compiled_clif_instructions",
        counters.compiled_clif_instructions
    );
    field!("compiled_clif_blocks", counters.compiled_clif_blocks);
    field!(
        "compiled_native_code_bytes",
        counters.compiled_native_code_bytes
    );
    field!(
        "compiled_native_mapped_bytes",
        counters.compiled_native_mapped_bytes
    );
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
    pub(crate) tier_upgrades: u64,
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
    field!("tier_upgrades", counters.tier_upgrades);
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
