//! Bounded cold compilation with one publication per key. No queue ownership.
use rustc_hash::FxHashMap as HashMap;
use std::{
    hash::Hash,
    sync::{Arc, Condvar, Mutex, mpsc},
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
        source: crate::ShaderSourceLocation,
        detail: Box<str>,
    },
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
            } => write!(
                f,
                "shader {stage:?} at PC {:#x}: {detail}",
                source.byte_offset()
            ),
        }
    }
}
impl std::error::Error for CompileError {}

impl From<super::Error> for CompileError {
    fn from(e: super::Error) -> Self {
        Self::Source {
            stage: e.stage,
            source: e.source,
            detail: e.detail.into_boxed_str(),
        }
    }
}

type Result<T> = std::result::Result<Arc<T>, Arc<CompileError>>;
struct Cell<T> {
    result: Mutex<Option<Result<T>>>,
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
        self.0.result.lock().unwrap().clone()
    }
    /// Bounded wait leaves servicing of unrelated control/visibility to the owner.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Result<T>> {
        let result = self.0.result.lock().unwrap();
        self.0
            .ready
            .wait_timeout_while(result, timeout, |r| r.is_none())
            .unwrap()
            .0
            .clone()
    }
}

type Job = Box<dyn FnOnce(usize) + Send + 'static>;
pub struct CompilePool {
    sender: Option<mpsc::SyncSender<Job>>,
    threads: Vec<thread::JoinHandle<()>>,
}
impl CompilePool {
    pub fn new(workers: usize, queue: usize) -> std::io::Result<Self> {
        if workers == 0 || queue == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "nonzero worker/queue limits required",
            ));
        }
        let (tx, rx) = mpsc::sync_channel::<Job>(queue);
        let rx = Arc::new(Mutex::new(rx));
        let mut pool = Self {
            sender: Some(tx),
            threads: Vec::new(),
        };
        for index in 0..workers {
            let rx = Arc::clone(&rx);
            pool.threads.push(
                thread::Builder::new()
                    .name(format!("vulkan-compile-{index}"))
                    .spawn(move || {
                        loop {
                            let job = rx.lock().unwrap().recv();
                            let Ok(job) = job else {
                                break;
                            };
                            job(index);
                        }
                    })?,
            );
        }
        Ok(pool)
    }
    pub fn workers(&self) -> usize {
        self.threads.len()
    }
    fn submit(&self, job: Job) -> std::result::Result<(), CompileError> {
        self.sender
            .as_ref()
            .ok_or(CompileError::Stopped)?
            .try_send(job)
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => CompileError::Backpressure,
                mpsc::TrySendError::Disconnected(_) => CompileError::Stopped,
            })
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
        let mut state = self.state.lock().unwrap();
        state.clock = state.clock.wrapping_add(1);
        let clock = state.clock;
        if let Some(entry) = state.entries.get_mut(&key) {
            if !entry.pending.poll().is_some_and(|r| {
                r.is_err_and(|e| matches!(*e, CompileError::Budget | CompileError::Backpressure))
            }) {
                entry.touched = clock;
                return Ok(entry.pending.clone());
            }
            let old = state.entries.remove(&key).unwrap();
            state.weight -= old.weight;
        }
        if weight > self.bytes || self.entries == 0 {
            return Err(CompileError::Budget);
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
            let old = state.entries.remove(&candidate).unwrap();
            state.weight -= old.weight;
        }
        let pending = Pending(Arc::new(Cell {
            result: Mutex::new(None),
            ready: Condvar::new(),
        }));
        let publish = pending.clone();
        let state_owner = Arc::clone(&self.state);
        let publish_key = key.clone();
        let byte_budget = self.bytes;
        pool.submit(Box::new(move |worker| {
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
            *publish.0.result.lock().unwrap() = Some(result);
            publish.0.ready.notify_all();
        }))?;
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
        let cache = CompilationCache::new(8, 64);
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
