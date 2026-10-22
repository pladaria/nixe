//! Bounded cold compilation with one publication per key. No queue ownership.
use rustc_hash::FxHashMap as HashMap;
use std::{
    hash::Hash,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    Backpressure,
    Budget,
    Stopped,
    Failed(Box<str>),
    Source {
        stage: Option<crate::ShaderStage>,
        source: Option<nixe_shader_recompiler::types::ProgramOffset>,
        detail: Box<str>,
    },
}
impl From<nixe_shader_recompiler::CompileError> for CompileError {
    fn from(error: nixe_shader_recompiler::CompileError) -> Self {
        Self::Source {
            stage: error.stage.map(crate::shader_compilation::neutral_stage),
            source: error.pc,
            detail: error.to_string().into(),
        }
    }
}
impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backpressure => {
                f.write_str("compilation queue is full; service visibility/control and retry")
            }
            Self::Budget => f.write_str("compilation cache budget is occupied by pending work"),
            Self::Stopped => f.write_str("compilation workers stopped"),
            Self::Failed(e) => f.write_str(e),
            Self::Source {
                stage,
                source,
                detail,
            } => write!(f, "shader {stage:?} at PC {source:?}: {detail}"),
        }
    }
}
impl std::error::Error for CompileError {}

type Result<T> = std::result::Result<Arc<T>, Arc<CompileError>>;
struct Cell<T> {
    result: OnceLock<Result<T>>,
    notification: Mutex<()>,
    ready: Condvar,
}
pub struct Pending<T>(Arc<Cell<T>>);
impl<T> Clone for Pending<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
impl<T> Pending<T> {
    pub fn poll(&self) -> Option<Result<T>> {
        self.0.result.get().cloned()
    }
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Result<T>> {
        if let Some(result) = self.poll() {
            return Some(result);
        }
        let lock = self.0.notification.lock().unwrap();
        let _wait = self
            .0
            .ready
            .wait_timeout_while(lock, timeout, |_| self.0.result.get().is_none())
            .unwrap();
        self.poll()
    }
}
impl<T> Cell<T> {
    fn publish(&self, result: Result<T>) {
        let _lock = self.notification.lock().unwrap();
        assert!(
            self.result.set(result).is_ok(),
            "compilation completed twice"
        );
        self.ready.notify_all();
    }
}

type Job = Box<dyn FnOnce(usize) + Send + 'static>;
type Wakeup = Arc<Mutex<Option<Arc<dyn Fn() + Send + Sync>>>>;
pub struct CompilePool {
    sender: Option<mpsc::SyncSender<Job>>,
    threads: Vec<thread::JoinHandle<()>>,
    queued: Arc<AtomicUsize>,
    capacity: usize,
    wakeup: Wakeup,
}
impl CompilePool {
    pub fn new(workers: usize, queue: usize) -> std::io::Result<Self> {
        Self::with_name(workers, queue, "shader-compile")
    }
    pub fn with_name(workers: usize, queue: usize, name: &str) -> std::io::Result<Self> {
        if workers == 0 || queue == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "nonzero worker/queue limits required",
            ));
        }
        let (tx, rx) = mpsc::sync_channel::<Job>(queue);
        let rx = Arc::new(Mutex::new(rx));
        let queued = Arc::new(AtomicUsize::new(0));
        let mut pool = Self {
            sender: Some(tx),
            threads: Vec::new(),
            queued: Arc::clone(&queued),
            capacity: queue,
            wakeup: Arc::new(Mutex::new(None)),
        };
        for index in 0..workers {
            let rx = Arc::clone(&rx);
            let queued = Arc::clone(&queued);
            let wakeup = Arc::clone(&pool.wakeup);
            pool.threads.push(
                thread::Builder::new()
                    .name(format!("{name}-{index}"))
                    .spawn(move || {
                        loop {
                            let job = rx.lock().unwrap().recv();
                            let Ok(job) = job else {
                                break;
                            };
                            queued.fetch_sub(1, Ordering::Relaxed);
                            job(index);
                            let wake = wakeup.lock().unwrap().clone();
                            if let Some(wake) = wake {
                                wake();
                            }
                        }
                    })?,
            );
        }
        Ok(pool)
    }
    pub fn bind_wakeup(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        // Install before the consumer checks readiness and starts parking.
        // Every later job publishes its result before reading this callback.
        *self.wakeup.lock().unwrap() = Some(wake);
    }
    pub fn workers(&self) -> usize {
        self.threads.len()
    }
    pub fn submit(
        &self,
        job: impl FnOnce(usize) + Send + 'static,
    ) -> std::result::Result<(), CompileError> {
        let sender = self.sender.as_ref().ok_or(CompileError::Stopped)?;
        self.queued.fetch_add(1, Ordering::Relaxed);
        let result = sender.try_send(Box::new(job)).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => CompileError::Backpressure,
            mpsc::TrySendError::Disconnected(_) => CompileError::Stopped,
        });
        if result.is_err() {
            self.queued.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }
}
impl Drop for CompilePool {
    fn drop(&mut self) {
        self.sender.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

struct Entry<T> {
    pending: Pending<T>,
    touched: u64,
    weight: usize,
}
struct State<K, T> {
    entries: HashMap<K, Entry<T>>,
    clock: u64,
    weight: usize,
}
/// Entry/byte limits include pending work. Eviction only drops reconstructible
/// cache ownership; prepared/batch references retain published values safely.
pub struct CompilationCache<K, T> {
    state: Arc<Mutex<State<K, T>>>,
    entries: usize,
    bytes: usize,
}
impl<K: Eq + Hash + Clone + Send + 'static, T: Send + Sync + 'static> CompilationCache<K, T> {
    pub fn new(entries: usize, bytes: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                entries: HashMap::default(),
                clock: 0,
                weight: 0,
            })),
            entries,
            bytes,
        }
    }
    pub fn request(
        &self,
        pool: &CompilePool,
        key: K,
        weight: usize,
        compile: impl FnOnce(usize) -> std::result::Result<T, CompileError> + Send + 'static,
    ) -> std::result::Result<Pending<T>, CompileError> {
        self.request_sized(pool, key, weight, move |_| weight, compile)
    }
    pub fn request_sized(
        &self,
        pool: &CompilePool,
        key: K,
        weight: usize,
        weigh: impl FnOnce(&T) -> usize + Send + 'static,
        compile: impl FnOnce(usize) -> std::result::Result<T, CompileError> + Send + 'static,
    ) -> std::result::Result<Pending<T>, CompileError> {
        #[cfg(feature = "performance-counters")]
        let capture_key = crate::compilation_capture::Key::new(&key);
        let mut state = self.state.lock().unwrap();
        state.clock = state.clock.wrapping_add(1);
        let clock = state.clock;
        if let Some(entry) = state.entries.get_mut(&key) {
            if !entry.pending.poll().is_some_and(|r| {
                r.is_err_and(|e| matches!(*e, CompileError::Budget | CompileError::Backpressure))
            }) {
                #[cfg(feature = "performance-counters")]
                capture_key.event(
                    if entry.pending.poll().is_some() {
                        "hit_ready"
                    } else {
                        "hit_pending"
                    },
                    usize::MAX,
                    entry.weight as u64,
                );
                entry.touched = clock;
                return Ok(entry.pending.clone());
            }
            let old = state.entries.remove(&key).unwrap();
            state.weight -= old.weight;
        }
        if weight > self.bytes || self.entries == 0 {
            return Err(CompileError::Budget);
        }
        // Avoid allocating pending cells or evicting useful entries on each
        // readiness poll while the queue has no room. The bounded channel still
        // arbitrates racing producers; cache hits remain available at pressure.
        if pool.queued.load(Ordering::Relaxed) >= pool.capacity {
            return Err(CompileError::Backpressure);
        }
        while state.entries.len() >= self.entries || state.weight > self.bytes - weight {
            let candidate = state
                .entries
                .iter()
                .filter(|(_, e)| e.pending.poll().is_some())
                .min_by_key(|(_, e)| e.touched)
                .map(|(k, _)| k.clone());
            let Some(candidate) = candidate else {
                return Err(CompileError::Budget);
            };
            #[cfg(feature = "performance-counters")]
            crate::compilation_capture::Key::new(&candidate).event("evict", usize::MAX, 0);
            let old = state.entries.remove(&candidate).unwrap();
            state.weight -= old.weight;
        }
        let pending = Pending(Arc::new(Cell {
            result: OnceLock::new(),
            notification: Mutex::new(()),
            ready: Condvar::new(),
        }));
        let publish = pending.clone();
        let state_owner = Arc::clone(&self.state);
        let publish_key = key.clone();
        let byte_budget = self.bytes;
        #[cfg(feature = "performance-counters")]
        let queued = std::time::Instant::now();
        #[cfg(feature = "performance-counters")]
        capture_key.event("miss", usize::MAX, weight as u64);
        pool.submit(move |worker| {
            #[cfg(feature = "performance-counters")]
            capture_key.event("begin", worker, queued.elapsed().as_nanos() as u64);
            #[cfg(feature = "performance-counters")]
            let started = std::time::Instant::now();
            // A compiler bug must resolve its pending request too, rather than
            // strand every duplicate caller indefinitely.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile(worker)))
                .unwrap_or_else(|_| Err(CompileError::Failed("compilation worker panicked".into())))
                .map(Arc::new)
                .map_err(Arc::new);
            let mut result = result;
            let mut state = state_owner.lock().unwrap();
            if let Ok(value) = &result {
                let actual = weigh(value);
                let old = state.entries.get(&publish_key).map_or(0, |e| e.weight);
                while actual <= byte_budget && state.weight - old > byte_budget - actual {
                    let candidate = state
                        .entries
                        .iter()
                        .filter(|(k, e)| *k != &publish_key && e.pending.poll().is_some())
                        .min_by_key(|(_, e)| e.touched)
                        .map(|(k, _)| k.clone());
                    let Some(candidate) = candidate else {
                        break;
                    };
                    #[cfg(feature = "performance-counters")]
                    crate::compilation_capture::Key::new(&candidate).event("evict", worker, 0);
                    let entry = state.entries.remove(&candidate).unwrap();
                    state.weight -= entry.weight;
                }
                if actual > byte_budget
                    || state.weight - old > byte_budget - actual.min(byte_budget)
                {
                    result = Err(Arc::new(CompileError::Budget));
                } else if let Some(entry) = state.entries.get_mut(&publish_key) {
                    entry.weight = actual;
                    state.weight = state.weight - old + actual;
                }
            }
            #[cfg(feature = "performance-counters")]
            capture_key.detail(
                if result.is_ok() { "end" } else { "failed" },
                worker,
                started.elapsed().as_nanos() as u64,
                &result
                    .as_ref()
                    .err()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
            );
            publish.0.publish(result);
        })?;
        state.weight += weight;
        state.entries.insert(
            key,
            Entry {
                pending: pending.clone(),
                touched: clock,
                weight,
            },
        );
        Ok(pending)
    }
    pub fn usage(&self) -> (usize, usize) {
        let s = self.state.lock().unwrap();
        (s.entries.len(), s.weight)
    }
    pub fn clear_ready(&self) {
        let mut s = self.state.lock().unwrap();
        s.entries.retain(|_, e| e.pending.poll().is_none());
        s.weight = s.entries.values().map(|e| e.weight).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn ready<T>(p: Pending<T>) -> Arc<T> {
        p.wait_timeout(Duration::from_secs(10))
            .expect("compiler must publish")
            .unwrap()
    }
    #[test]
    fn duplicate_misses_compile_once_and_publish_the_same_value() {
        let pool = CompilePool::new(4, 16).unwrap();
        let cache = Arc::new(CompilationCache::new(16, 4096));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pending = Vec::new();
        for _ in 0..64 {
            let calls = Arc::clone(&calls);
            pending.push(
                cache
                    .request(&pool, 7u32, 4, move |_| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok(42u32)
                    })
                    .unwrap(),
            );
        }
        let first = ready(pending.remove(0));
        for p in pending {
            assert!(Arc::ptr_eq(&first, &ready(p)));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn queue_pressure_does_not_drop_a_required_or_duplicate_job() {
        let pool = CompilePool::new(1, 1).unwrap();
        let cache = CompilationCache::new(3, 12);
        let warm = ready(cache.request(&pool, 0, 4, |_| Ok(0u32)).unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first = cache
            .request(&pool, 1, 4, move |_| {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(1u32)
            })
            .unwrap();
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let second = cache.request(&pool, 2, 4, |_| Ok(2)).unwrap();
        assert!(matches!(
            cache.request(&pool, 3, 4, |_| Ok(3)),
            Err(CompileError::Backpressure)
        ));
        let retained = cache
            .request(&pool, 0, 4, |_| {
                panic!("pressure evicted a useful warm entry")
            })
            .expect("queue pressure must preserve available cached results");
        assert!(Arc::ptr_eq(&warm, &ready(retained)));
        let duplicate = cache
            .request(&pool, 2, 4, |_| panic!("duplicate must not compile"))
            .unwrap();
        release_tx.send(()).unwrap();
        assert_eq!(*ready(first), 1);
        let two = ready(second);
        assert!(Arc::ptr_eq(&two, &ready(duplicate)));
    }
    #[test]
    fn actual_byte_accounting_evicts_reconstructible_entries_and_retries_pressure() {
        let pool = CompilePool::new(1, 4).unwrap();
        let cache = CompilationCache::new(4, 32);
        let value = ready(
            cache
                .request_sized(&pool, 1, 1, Vec::len, |_| Ok(vec![7u8; 24]))
                .unwrap(),
        );
        assert_eq!(cache.usage(), (1, 24));
        let other = ready(
            cache
                .request_sized(&pool, 2, 1, Vec::len, |_| Ok(vec![9u8; 24]))
                .unwrap(),
        );
        assert_eq!(cache.usage(), (1, 24));
        assert_eq!(value[0], 7);
        assert_eq!(other[0], 9);
        let too_large = cache
            .request_sized(&pool, 3, 1, Vec::len, |_| Ok(vec![0; 33]))
            .unwrap();
        assert_eq!(
            *too_large
                .wait_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap_err(),
            CompileError::Budget
        );
        assert_eq!(
            *ready(
                cache
                    .request_sized(&pool, 3, 1, Vec::len, |_| Ok(vec![11; 8]))
                    .unwrap()
            ),
            vec![11; 8]
        );
    }
    #[test]
    fn failed_compilation_and_panics_are_published_without_a_success_fallback() {
        let pool = CompilePool::new(1, 4).unwrap();
        let cache = CompilationCache::<u32, u32>::new(4, 16);
        let p = cache
            .request(&pool, 1, 4, |_| panic!("test compiler bug"))
            .unwrap();
        assert!(matches!(
            &*p.wait_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap_err(),
            CompileError::Failed(_)
        ));
        assert_eq!(*ready(cache.request(&pool, 2, 4, |_| Ok(12)).unwrap()), 12);
    }
}
