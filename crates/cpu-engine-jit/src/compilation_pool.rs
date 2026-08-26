//! Domain-owned asynchronous Cranelift compilation workers.

use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cranelift_codegen::isa::OwnedTargetIsa;
use nixe_cpu::ir::region::IrRegion;
use nixe_cpu::profile::GuestCpuProfile;
use nixe_memory::AddressSpaceId;

use crate::cache::{
    CacheError, CompilationCancellation, PendingRegion, RegionKey, TranslationMode,
};
use crate::compiler::CompilerContext;
use crate::diagnostics::JitDiagnostics;
use crate::executable_memory::SharedExecutableMemory;

const TASKS_PER_WORKER: usize = 64;

pub(crate) fn host_compilation_workers() -> usize {
    thread::available_parallelism()
        .map(|parallelism| parallelism.get() / 2)
        .unwrap_or(1)
        .max(1)
}

pub(crate) struct CompilationTask {
    pub(crate) key: RegionKey,
    pub(crate) address_space: AddressSpaceId,
    pub(crate) mode: TranslationMode,
    pub(crate) region: IrRegion,
}

pub(crate) struct CompletedCompilation {
    pub(crate) result: Result<PendingRegion, CacheError>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompletedMetrics {
    pub(crate) codegen_ns: u64,
    pub(crate) guest_instructions: u64,
    pub(crate) ir_operations: u64,
    pub(crate) native_bytes: u64,
    pub(crate) native_named_operations: u64,
    pub(crate) semantic_helper_callsites: u64,
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
    pub(crate) failed: u64,
    pub(crate) completed_discarded: u64,
    pub(crate) peak_queued: u64,
    pub(crate) peak_running: u64,
    pub(crate) codegen_ns: u64,
    pub(crate) compiled_guest_instructions: u64,
    pub(crate) compiled_ir_operations: u64,
    pub(crate) compiled_native_bytes: u64,
    pub(crate) compiled_native_named_operations: u64,
    pub(crate) compiled_semantic_helper_callsites: u64,
}

struct PoolState {
    stopping: bool,
    queue: VecDeque<CompilationTask>,
    known: HashSet<RegionKey>,
    completed: HashMap<RegionKey, CompletedCompilation>,
    completion_order: VecDeque<RegionKey>,
    running: usize,
    snapshot: CompilationPoolSnapshot,
}

struct SharedPool {
    state: Mutex<PoolState>,
    ready: Condvar,
    queue_capacity: usize,
    completed_capacity: usize,
}

struct WorkerEnvironment {
    isa: OwnedTargetIsa,
    executable_memory: SharedExecutableMemory,
    diagnostics: Option<Arc<JitDiagnostics>>,
    profile: GuestCpuProfile,
}

#[derive(Clone)]
pub(crate) struct CompilationPoolHandle {
    shared: Arc<SharedPool>,
}

impl CompilationPoolHandle {
    pub(crate) fn enqueue(&self, task: CompilationTask) -> bool {
        let mut state = lock(&self.shared.state);
        if state.stopping {
            return false;
        }
        if !state.known.insert(task.key) {
            state.snapshot.duplicate_requests = state.snapshot.duplicate_requests.saturating_add(1);
            return false;
        }
        if state.queue.len() == self.shared.queue_capacity
            && let Some(discarded) = state.queue.pop_front()
        {
            state.known.remove(&discarded.key);
            state.snapshot.queued_discarded = state.snapshot.queued_discarded.saturating_add(1);
        }
        state.queue.push_back(task);
        state.snapshot.enqueued = state.snapshot.enqueued.saturating_add(1);
        state.snapshot.peak_queued = state.snapshot.peak_queued.max(state.queue.len() as u64);
        self.shared.ready.notify_one();
        true
    }

    pub(crate) fn contains(&self, key: RegionKey) -> bool {
        lock(&self.shared.state).known.contains(&key)
    }

    pub(crate) fn take_completed(&self, key: RegionKey) -> Option<CompletedCompilation> {
        let mut state = lock(&self.shared.state);
        let completed = state.completed.remove(&key)?;
        state.known.remove(&key);
        if let Some(index) = state
            .completion_order
            .iter()
            .position(|candidate| *candidate == key)
        {
            state.completion_order.remove(index);
        }
        Some(completed)
    }

    pub(crate) fn snapshot(&self) -> CompilationPoolSnapshot {
        lock(&self.shared.state).snapshot
    }

    #[cfg(test)]
    pub(crate) fn wait_for_completion(&self, key: RegionKey) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let state = lock(&self.shared.state);
            if state.completed.contains_key(&key) {
                return true;
            }
            if state.stopping || !state.known.contains(&key) {
                return false;
            }
            drop(state);
            assert!(
                Instant::now() < deadline,
                "asynchronous JIT compilation timed out in test"
            );
            thread::yield_now();
        }
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
        diagnostics: Option<Arc<JitDiagnostics>>,
        profile: GuestCpuProfile,
    ) -> Result<Self, Box<str>> {
        let worker_count = worker_count.max(1);
        let queue_capacity = worker_count.saturating_mul(TASKS_PER_WORKER).max(1);
        let shared = Arc::new(SharedPool {
            state: Mutex::new(PoolState {
                stopping: false,
                queue: VecDeque::new(),
                known: HashSet::new(),
                completed: HashMap::new(),
                completion_order: VecDeque::new(),
                running: 0,
                snapshot: CompilationPoolSnapshot {
                    workers: worker_count as u64,
                    queue_capacity: queue_capacity as u64,
                    ..CompilationPoolSnapshot::default()
                },
            }),
            ready: Condvar::new(),
            queue_capacity,
            completed_capacity: queue_capacity,
        });
        let environment = Arc::new(WorkerEnvironment {
            isa,
            executable_memory,
            diagnostics,
            profile,
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
    state.snapshot.completed_discarded = state
        .snapshot
        .completed_discarded
        .saturating_add(state.completed.len() as u64);
    state.queue.clear();
    state.completed.clear();
    state.completion_order.clear();
    state.known.clear();
    shared.ready.notify_all();
}

fn worker_loop(shared: Arc<SharedPool>, environment: Arc<WorkerEnvironment>) {
    let mut compiler = CompilerContext::new(Arc::clone(&environment.isa));
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
            let task = state.queue.pop_front().expect("a ready worker owns a task");
            state.running += 1;
            state.snapshot.started = state.snapshot.started.saturating_add(1);
            state.snapshot.peak_running = state.snapshot.peak_running.max(state.running as u64);
            task
        };

        let started = Instant::now();
        let guest_instructions = u64::from(task.region.metadata.guest_instruction_count);
        let ir_operations = u64::from(task.region.metadata.ir_operation_count);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let compiled = compiler.compile(
                &task.region,
                &environment.executable_memory,
                CompilationCancellation::active(),
            )?;
            if let Some(diagnostics) = &environment.diagnostics {
                diagnostics
                    .dump_region(&environment.profile, &task.region, &compiled)
                    .map_err(CacheError::Internal)?;
            }
            let metrics = CompletedMetrics {
                codegen_ns: duration_ns(started.elapsed()),
                guest_instructions,
                ir_operations,
                native_bytes: compiled.mapped_len() as u64,
                native_named_operations: compiled.metadata.native_named_operations,
                semantic_helper_callsites: compiled.metadata.semantic_calls.len() as u64,
            };
            let pending =
                PendingRegion::new(task.address_space, task.mode, &task.region, compiled)?;
            Ok::<_, CacheError>((pending, metrics))
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

        let (result, metrics) = match result {
            Ok((pending, metrics)) => (Ok(pending), metrics),
            Err(error) => (
                Err(error),
                CompletedMetrics {
                    codegen_ns: duration_ns(started.elapsed()),
                    guest_instructions,
                    ir_operations,
                    ..CompletedMetrics::default()
                },
            ),
        };
        let mut state = lock(&shared.state);
        state.running = state.running.saturating_sub(1);
        state.snapshot.completed = state.snapshot.completed.saturating_add(1);
        state.snapshot.codegen_ns = state.snapshot.codegen_ns.saturating_add(metrics.codegen_ns);
        state.snapshot.compiled_guest_instructions = state
            .snapshot
            .compiled_guest_instructions
            .saturating_add(metrics.guest_instructions);
        state.snapshot.compiled_ir_operations = state
            .snapshot
            .compiled_ir_operations
            .saturating_add(metrics.ir_operations);
        state.snapshot.compiled_native_bytes = state
            .snapshot
            .compiled_native_bytes
            .saturating_add(metrics.native_bytes);
        state.snapshot.compiled_native_named_operations = state
            .snapshot
            .compiled_native_named_operations
            .saturating_add(metrics.native_named_operations);
        state.snapshot.compiled_semantic_helper_callsites = state
            .snapshot
            .compiled_semantic_helper_callsites
            .saturating_add(metrics.semantic_helper_callsites);
        if result.is_err() {
            state.snapshot.failed = state.snapshot.failed.saturating_add(1);
        }
        if state.stopping {
            state.known.remove(&task.key);
            state.snapshot.completed_discarded =
                state.snapshot.completed_discarded.saturating_add(1);
            continue;
        }
        if state.completed.len() == shared.completed_capacity
            && let Some(discarded) = state.completion_order.pop_front()
        {
            state.completed.remove(&discarded);
            state.known.remove(&discarded);
            state.snapshot.completed_discarded =
                state.snapshot.completed_discarded.saturating_add(1);
        }
        state
            .completed
            .insert(task.key, CompletedCompilation { result });
        state.completion_order.push_back(task.key);
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
    use super::*;

    #[test]
    fn worker_count_is_half_the_host_parallelism_with_one_as_the_floor() {
        let host = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        assert_eq!(host_compilation_workers(), (host / 2).max(1));
    }
}
