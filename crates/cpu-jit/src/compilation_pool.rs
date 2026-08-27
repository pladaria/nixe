//! Domain-owned asynchronous Cranelift compilation workers.

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cranelift_codegen::isa::OwnedTargetIsa;
use nixe_cpu::profile::GuestCpuProfile;

use crate::cache::{
    CacheError, CodeTier, PendingRegion, ProcessCodeCache, PromotionCell, PromotionRequest,
    RegionKey,
};
use crate::compiler::{CompilationMetrics, CompilerContext};
use crate::diagnostics::JitDiagnostics;
use crate::executable_memory::SharedExecutableMemory;

const TASKS_PER_WORKER: usize = 64;

pub(crate) fn host_compilation_workers() -> usize {
    thread::available_parallelism()
        .map(|parallelism| parallelism.get() / 2)
        .unwrap_or(1)
        .max(1)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompletedMetrics {
    pub(crate) worker_total_ns: u64,
    pub(crate) diagnostics_ns: u64,
    pub(crate) pending_region_ns: u64,
    pub(crate) guest_instructions: u64,
    pub(crate) ir_operations: u64,
    pub(crate) native_named_operations: u64,
    pub(crate) semantic_helper_callsites: u64,
    pub(crate) compilation: CompilationMetrics,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompilationPoolSnapshot {
    pub(crate) workers: u64,
    pub(crate) queue_capacity: u64,
    pub(crate) enqueued: u64,
    pub(crate) duplicate_requests: u64,
    pub(crate) queued_discarded: u64,
    pub(crate) started: u64,
    pub(crate) completed: u64,
    pub(crate) published: u64,
    pub(crate) stale: u64,
    pub(crate) failed: u64,
    pub(crate) peak_queued: u64,
    pub(crate) peak_running: u64,
    pub(crate) worker_total_ns: u64,
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
}

struct PoolState {
    stopping: bool,
    queue: VecDeque<PromotionRequest>,
    known: HashMap<RegionKey, Arc<PromotionCell>>,
    running: usize,
    snapshot: CompilationPoolSnapshot,
}

struct SharedPool {
    state: Mutex<PoolState>,
    ready: Condvar,
    queue_capacity: usize,
}

struct WorkerEnvironment {
    isa: OwnedTargetIsa,
    executable_memory: SharedExecutableMemory,
    diagnostics: Option<Arc<JitDiagnostics>>,
    profile: GuestCpuProfile,
    collect_performance: bool,
    code_cache: Arc<ProcessCodeCache>,
}

#[derive(Clone)]
pub(crate) struct CompilationPoolHandle {
    shared: Arc<SharedPool>,
}

impl CompilationPoolHandle {
    pub(crate) fn enqueue(&self, task: PromotionRequest) -> bool {
        let mut state = lock(&self.shared.state);
        if state.stopping {
            return false;
        }
        if state.known.contains_key(&task.key) {
            state.snapshot.duplicate_requests = state.snapshot.duplicate_requests.saturating_add(1);
            return false;
        }
        if state.queue.len() == self.shared.queue_capacity
            && let Some(discarded) = state.queue.pop_front()
        {
            state.known.remove(&discarded.key);
            discarded.promotion.rearm();
            state.snapshot.queued_discarded = state.snapshot.queued_discarded.saturating_add(1);
        }
        state.known.insert(task.key, Arc::clone(&task.promotion));
        // Ryujinx's rejit queue is deliberately a deduplicated stack so that
        // recently hot code is optimized before older, potentially cold work:
        // https://www.git.axenov.dev/Museum/ryujinx/src/commit/a23d8cb92f3f1bb8dc144f4d9fb3fddee749feae/src/ARMeilleure/Translation/TranslatorQueue.cs
        state.queue.push_back(task);
        state.snapshot.enqueued = state.snapshot.enqueued.saturating_add(1);
        state.snapshot.peak_queued = state.snapshot.peak_queued.max(state.queue.len() as u64);
        self.shared.ready.notify_one();
        true
    }

    pub(crate) fn snapshot(&self) -> CompilationPoolSnapshot {
        lock(&self.shared.state).snapshot
    }
}

pub(crate) struct CompilationPool {
    handle: CompilationPoolHandle,
    workers: Vec<JoinHandle<()>>,
}

impl CompilationPool {
    pub(crate) fn new(
        worker_count: usize,
        isa: OwnedTargetIsa,
        executable_memory: SharedExecutableMemory,
        code_cache: Arc<ProcessCodeCache>,
        diagnostics: Option<Arc<JitDiagnostics>>,
        profile: GuestCpuProfile,
        collect_performance: bool,
    ) -> Result<Self, Box<str>> {
        let worker_count = worker_count.max(1);
        let queue_capacity = worker_count.saturating_mul(TASKS_PER_WORKER).max(1);
        let shared = Arc::new(SharedPool {
            state: Mutex::new(PoolState {
                stopping: false,
                queue: VecDeque::new(),
                known: HashMap::new(),
                running: 0,
                snapshot: CompilationPoolSnapshot {
                    workers: worker_count as u64,
                    queue_capacity: queue_capacity as u64,
                    ..CompilationPoolSnapshot::default()
                },
            }),
            ready: Condvar::new(),
            queue_capacity,
        });
        let environment = Arc::new(WorkerEnvironment {
            isa,
            executable_memory,
            diagnostics,
            profile,
            collect_performance,
            code_cache,
        });
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let worker_shared = Arc::clone(&shared);
            let worker_environment = Arc::clone(&environment);
            match thread::Builder::new()
                .name(format!("nixe-jit-compiler-{index}"))
                .spawn(move || worker_loop(worker_shared, worker_environment))
            {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    stop_shared(&shared);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(
                        format!("cannot create JIT compilation worker: {error}").into_boxed_str()
                    );
                }
            }
        }
        Ok(Self {
            handle: CompilationPoolHandle { shared },
            workers,
        })
    }

    pub(crate) fn handle(&self) -> CompilationPoolHandle {
        self.handle.clone()
    }

    pub(crate) fn snapshot(&self) -> CompilationPoolSnapshot {
        self.handle.snapshot()
    }

    pub(crate) fn begin_shutdown(&self) {
        stop_shared(&self.handle.shared);
    }

    pub(crate) fn shutdown(&mut self) {
        self.begin_shutdown();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for CompilationPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn stop_shared(shared: &SharedPool) {
    let mut state = lock(&shared.state);
    if state.stopping {
        return;
    }
    state.stopping = true;
    state.snapshot.queued_discarded = state
        .snapshot
        .queued_discarded
        .saturating_add(state.queue.len() as u64);
    for task in &state.queue {
        task.promotion.mark_stale();
    }
    for promotion in state.known.values() {
        promotion.mark_stale();
    }
    state.queue.clear();
    state.known.clear();
    shared.ready.notify_all();
}

fn worker_loop(shared: Arc<SharedPool>, environment: Arc<WorkerEnvironment>) {
    let mut compiler = CompilerContext::new(
        Arc::clone(&environment.isa),
        environment.collect_performance,
    );
    loop {
        let task = {
            let mut state = lock(&shared.state);
            while state.queue.is_empty() && !state.stopping {
                state = shared
                    .ready
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if state.stopping {
                return;
            }
            let task = state.queue.pop_back().expect("a ready worker owns a task");
            state.running += 1;
            state.snapshot.started = state.snapshot.started.saturating_add(1);
            state.snapshot.peak_running = state.snapshot.peak_running.max(state.running as u64);
            task
        };

        if !task.promotion.begin_compilation() {
            let mut state = lock(&shared.state);
            state.running = state.running.saturating_sub(1);
            state.known.remove(&task.key);
            continue;
        }

        let started = environment.collect_performance.then(Instant::now);
        let guest_instructions = u64::from(task.region.metadata.guest_instruction_count);
        let ir_operations = u64::from(task.region.metadata.ir_operation_count);
        let mut metrics = CompletedMetrics {
            guest_instructions,
            ir_operations,
            ..CompletedMetrics::default()
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            let compiled = compiler.compile(
                &task.region,
                &environment.executable_memory,
                task.promotion.cancellation(),
                None,
                &mut metrics.compilation,
            )?;
            if let Some(diagnostics) = &environment.diagnostics {
                let diagnostics_started = environment.collect_performance.then(Instant::now);
                diagnostics
                    .dump_region(&environment.profile, &task.region, &compiled)
                    .map_err(CacheError::Internal)?;
                if let Some(started) = diagnostics_started {
                    metrics.diagnostics_ns = duration_ns(started.elapsed());
                }
            }
            metrics.native_named_operations = compiled.metadata.native_named_operations;
            metrics.semantic_helper_callsites = compiled.metadata.semantic_calls.len() as u64;
            let pending_started = environment.collect_performance.then(Instant::now);
            let pending = PendingRegion::new(
                task.address_space,
                task.mode,
                Arc::clone(&task.region),
                compiled,
                CodeTier::Optimized,
                None,
            )?;
            if let Some(started) = pending_started {
                metrics.pending_region_ns = duration_ns(started.elapsed());
            }
            environment.code_cache.publish_upgrade(&task, pending)
        }))
        .unwrap_or_else(|_| {
            Err(CacheError::Internal(
                format!(
                    "panic was contained during asynchronous JIT compilation for {:?}",
                    task.key
                )
                .into_boxed_str(),
            ))
        });
        if let Some(started) = started {
            metrics.worker_total_ns = duration_ns(started.elapsed());
        }
        let hard_failure = match &result {
            Err(CacheError::Stale | CacheError::Cancelled) => None,
            Err(error) => Some(error.clone()),
            Ok(_) => None,
        };
        if let Some(error) = hard_failure {
            task.promotion.mark_stale();
            environment.code_cache.record_background_failure(error);
        }
        let mut state = lock(&shared.state);
        state.running = state.running.saturating_sub(1);
        state.known.remove(&task.key);
        state.snapshot.completed = state.snapshot.completed.saturating_add(1);
        macro_rules! add_metric {
            ($field:ident, $value:expr) => {
                state.snapshot.$field = state.snapshot.$field.saturating_add($value);
            };
        }
        add_metric!(worker_total_ns, metrics.worker_total_ns);
        add_metric!(nixe_ir_verify_ns, metrics.compilation.nixe_ir_verify_ns);
        add_metric!(state_validation_ns, metrics.compilation.state_validation_ns);
        add_metric!(lowering_ns, metrics.compilation.lowering_ns);
        add_metric!(
            cranelift_compile_ns,
            metrics.compilation.cranelift_compile_ns
        );
        add_metric!(
            cranelift_verifier_ns,
            metrics.compilation.cranelift_verifier_ns
        );
        add_metric!(
            cranelift_optimize_ns,
            metrics.compilation.cranelift_optimize_ns
        );
        add_metric!(
            cranelift_vcode_lower_ns,
            metrics.compilation.cranelift_vcode_lower_ns
        );
        add_metric!(
            cranelift_regalloc_ns,
            metrics.compilation.cranelift_regalloc_ns
        );
        add_metric!(cranelift_emit_ns, metrics.compilation.cranelift_emit_ns);
        add_metric!(cranelift_other_ns, metrics.compilation.cranelift_other_ns);
        add_metric!(
            publication_total_ns,
            metrics.compilation.publication.total_ns
        );
        add_metric!(
            publication_lock_wait_ns,
            metrics.compilation.publication.lock_wait_ns
        );
        add_metric!(
            publication_allocation_ns,
            metrics.compilation.publication.allocation_ns
        );
        add_metric!(
            publication_zero_copy_ns,
            metrics.compilation.publication.zero_copy_ns
        );
        add_metric!(
            publication_protection_ns,
            metrics.compilation.publication.protection_ns
        );
        add_metric!(
            publication_instruction_cache_ns,
            metrics.compilation.publication.instruction_cache_ns
        );
        add_metric!(diagnostics_ns, metrics.diagnostics_ns);
        add_metric!(pending_region_ns, metrics.pending_region_ns);
        state.snapshot.compiled_guest_instructions = state
            .snapshot
            .compiled_guest_instructions
            .saturating_add(metrics.guest_instructions);
        state.snapshot.compiled_ir_operations = state
            .snapshot
            .compiled_ir_operations
            .saturating_add(metrics.ir_operations);
        add_metric!(
            compiled_clif_instructions,
            metrics.compilation.clif_instructions
        );
        add_metric!(compiled_clif_blocks, metrics.compilation.clif_blocks);
        add_metric!(
            compiled_native_code_bytes,
            metrics.compilation.native_code_bytes
        );
        add_metric!(
            compiled_native_mapped_bytes,
            metrics.compilation.native_mapped_bytes
        );
        state.snapshot.compiled_native_named_operations = state
            .snapshot
            .compiled_native_named_operations
            .saturating_add(metrics.native_named_operations);
        state.snapshot.compiled_semantic_helper_callsites = state
            .snapshot
            .compiled_semantic_helper_callsites
            .saturating_add(metrics.semantic_helper_callsites);
        match result {
            Ok(_) => {
                state.snapshot.published = state.snapshot.published.saturating_add(1);
            }
            Err(CacheError::Stale | CacheError::Cancelled) => {
                state.snapshot.stale = state.snapshot.stale.saturating_add(1);
            }
            Err(_) => {
                state.snapshot.failed = state.snapshot.failed.saturating_add(1);
            }
        }
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use nixe_cpu::ir::region::{IrRegion, RegionMetadata};
    use nixe_cpu::location::{ExecutionState, LocationDescriptor};
    use nixe_memory::{
        AddressSpaceId, ContentGeneration, GuestPhysicalPageId, GuestVirtualAddress,
        MappingGeneration,
    };

    use super::*;
    use crate::cache::TranslationMode;

    fn request(pc: u64) -> PromotionRequest {
        let address_space = AddressSpaceId::new(7);
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(pc),
            ExecutionState::A64,
            GuestCpuProfile::SWITCH_1_ID,
        );
        let dependency = nixe_cpu::memory::CodePageDependency {
            page: GuestPhysicalPageId::new(pc >> 12),
            generation: ContentGeneration::new(1),
            mapping_generation: MappingGeneration::new(1),
        };
        PromotionRequest {
            key: RegionKey::new(
                address_space,
                location,
                TranslationMode::Baseline,
                dependency,
            ),
            baseline_id: pc,
            address_space,
            mode: TranslationMode::Baseline,
            region: Arc::new(IrRegion::new(
                RegionMetadata {
                    start: location,
                    guest_byte_count: 4,
                    guest_instruction_count: 1,
                    ir_operation_count: 1,
                    entries: Box::new([]),
                    exits: Box::new([]),
                    code_dependencies: Box::new([dependency]),
                    safepoints: Box::new([]),
                },
                Vec::new(),
            )),
            promotion: Arc::new(PromotionCell::new()),
        }
    }

    #[test]
    fn worker_count_is_half_the_host_parallelism_with_one_as_the_floor() {
        let host = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        assert_eq!(host_compilation_workers(), (host / 2).max(1));
    }

    #[test]
    fn bounded_queue_discards_the_oldest_request_and_deduplicates_keys() {
        let shared = Arc::new(SharedPool {
            state: Mutex::new(PoolState {
                stopping: false,
                queue: VecDeque::new(),
                known: HashMap::new(),
                running: 0,
                snapshot: CompilationPoolSnapshot::default(),
            }),
            ready: Condvar::new(),
            queue_capacity: 2,
        });
        let handle = CompilationPoolHandle {
            shared: Arc::clone(&shared),
        };
        let oldest = request(0x1000);
        let middle = request(0x2000);
        let newest = request(0x3000);
        let newest_key = newest.key;
        let middle_key = middle.key;

        assert!(handle.enqueue(oldest));
        assert!(handle.enqueue(middle));
        assert!(handle.enqueue(newest));
        assert!(!handle.enqueue(request(0x3000)));

        let mut state = lock(&shared.state);
        assert_eq!(state.queue.pop_back().unwrap().key, newest_key);
        assert_eq!(state.queue.pop_back().unwrap().key, middle_key);
        assert_eq!(state.snapshot.queued_discarded, 1);
        assert_eq!(state.snapshot.duplicate_requests, 1);
    }
}
