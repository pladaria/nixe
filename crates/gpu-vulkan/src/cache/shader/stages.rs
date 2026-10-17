//! Synchronous leaf reuse, before SPIR-V generation. A hit shares the compiled
//! stage retained by program/prepared owners. Weak entries do not create a
//! second SPIR-V store or bypass the existing shader byte budget.
use super::*;
use nixe_gpu::ssa::artifact::StageKey;
use rustc_hash::FxHashMap as HashMap;
use std::sync::{Condvar, Mutex, Weak};

enum Value {
    Empty,
    Compiling,
    Ready(Weak<CompiledStage>),
    Failed(CompileError),
}
struct Cell {
    value: Mutex<Value>,
    ready: Condvar,
}
struct State {
    entries: HashMap<StageKey, (Arc<Cell>, u64)>,
    clock: u64,
}
pub(super) struct Stages {
    maximum: usize,
    state: Mutex<State>,
}
impl Stages {
    pub(super) fn new(maximum: usize) -> Self {
        Self {
            maximum,
            state: Mutex::new(State {
                entries: HashMap::default(),
                clock: 0,
            }),
        }
    }
    pub(super) fn get(
        &self,
        key: StageKey,
        emit: impl FnOnce() -> Result<Arc<CompiledStage>, CompileError>,
    ) -> Result<Arc<CompiledStage>, CompileError> {
        let cell = {
            let mut state = self.state.lock().unwrap();
            state.clock = state.clock.wrapping_add(1);
            let clock = state.clock;
            if let Some((cell, touched)) = state.entries.get_mut(&key) {
                *touched = clock;
                Arc::clone(cell)
            } else {
                if self.maximum == 0 {
                    return Err(CompileError::Budget);
                }
                if state.entries.len() >= self.maximum {
                    // Only cache-owned cells are eligible. A caller retains its
                    // cell through publication, so in-flight work cannot split.
                    let victim = state
                        .entries
                        .iter()
                        .filter(|(_, (cell, _))| Arc::strong_count(cell) == 1)
                        .min_by_key(|(_, (_, touched))| *touched)
                        .map(|(key, _)| *key)
                        .ok_or(CompileError::Budget)?;
                    state.entries.remove(&victim);
                    #[cfg(feature = "performance-counters")]
                    nixe_gpu::compilation_capture::Key::new(&victim).event("evict", usize::MAX, 0);
                }
                let cell = Arc::new(Cell {
                    value: Mutex::new(Value::Empty),
                    ready: Condvar::new(),
                });
                state.entries.insert(key, (Arc::clone(&cell), clock));
                cell
            }
        };
        let mut value = cell.value.lock().unwrap();
        loop {
            match &*value {
                Value::Ready(stage) => {
                    if let Some(stage) = stage.upgrade() {
                        #[cfg(feature = "performance-counters")]
                        nixe_gpu::compilation_capture::Key::new(&key).event(
                            "hit_ready",
                            usize::MAX,
                            stage.module.words.len() as u64,
                        );
                        return Ok(stage);
                    }
                }
                Value::Failed(error)
                    if !matches!(error, CompileError::Budget | CompileError::Backpressure) =>
                {
                    return Err(error.clone());
                }
                Value::Compiling => {
                    value = cell.ready.wait(value).unwrap();
                    continue;
                }
                _ => {}
            }
            break;
        }
        *value = Value::Compiling;
        drop(value);
        #[cfg(feature = "performance-counters")]
        let start = Instant::now();
        #[cfg(feature = "performance-counters")]
        nixe_gpu::compilation_capture::Key::new(&key).event("begin", usize::MAX, 0);
        // No queued dependency: the current worker completes this leaf itself.
        // Unwinding must also publish a terminal result to duplicate callers.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(emit))
            .unwrap_or_else(|_| Err(CompileError::Failed("stage emitter panicked".into())));
        let mut value = cell.value.lock().unwrap();
        *value = match &result {
            Ok(stage) => Value::Ready(Arc::downgrade(stage)),
            Err(error) => Value::Failed(error.clone()),
        };
        cell.ready.notify_all();
        #[cfg(feature = "performance-counters")]
        nixe_gpu::compilation_capture::Key::new(&key).detail(
            if result.is_ok() { "end" } else { "failed" },
            usize::MAX,
            start.elapsed().as_nanos() as u64,
            &result
                .as_ref()
                .map(|s| format!("stage={:?} identity={}", s.stage, s.identity.hex()))
                .unwrap_or_else(ToString::to_string),
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    fn key(value: u8) -> StageKey {
        StageKey(ContentId::of(b"stage-test", &[&[value]]))
    }
    fn stage() -> Arc<CompiledStage> {
        Arc::new(CompiledStage::new(
            ShaderStage::Compute,
            spirv::Module {
                words: vec![0x07230203, 0x10500, 0, 1, 0].into_boxed_slice(),
                bindings: Box::new([]),
                push_constant_bytes: 0,
            },
            Some([1, 1, 1]),
            0,
            None,
        ))
    }
    #[test]
    fn concurrent_duplicates_share_one_leaf_and_weak_entries_do_not_retain_code() {
        let cache = Arc::new(Stages::new(8));
        let calls = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let calls = Arc::clone(&calls);
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    gate.wait();
                    cache
                        .get(key(1), || {
                            calls.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(20));
                            Ok(stage())
                        })
                        .unwrap()
                })
            })
            .collect();
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(values.iter().all(|s| Arc::ptr_eq(s, &values[0])));
        let weak = Arc::downgrade(&values[0]);
        drop(values);
        assert!(weak.upgrade().is_none());
        cache
            .get(key(1), || {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(stage())
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
    #[test]
    fn eviction_drops_only_lookup_ownership_and_panics_publish_terminal_errors() {
        let cache = Stages::new(1);
        let first = cache.get(key(1), || Ok(stage())).unwrap();
        let second = cache.get(key(2), || Ok(stage())).unwrap();
        assert_eq!(cache.state.lock().unwrap().entries.len(), 1);
        assert_eq!(first.module.words, second.module.words);
        let error = cache
            .get(key(3), || panic!("fixture emitter panic"))
            .unwrap_err();
        assert!(matches!(error, CompileError::Failed(_)));
        assert!(
            cache
                .get(key(3), || panic!(
                    "a terminal failure must not compile again"
                ))
                .is_err()
        );
    }
}
