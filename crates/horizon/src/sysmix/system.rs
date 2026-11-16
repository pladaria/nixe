//! One native mixer worker services all owned streams at a 5 ms cadence.
use super::{
    User,
    stream::{Stream, StreamSession},
};
use crate::{HorizonIpcResult as R, audio_device::AudioDeviceState};
use nixe_audio::AudioBackend;
use nixe_memory::{AccessResult, PendingAccess};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Debug, Default)]
struct Registry {
    users: BTreeMap<u64, Weak<User>>,
    streams: BTreeMap<u32, Weak<Stream>>,
    next: u32,
}
#[derive(Debug, Default)]
struct Flags {
    stop: bool,
    paused: bool,
    parked: bool,
    finished: bool,
}
#[derive(Debug, Default)]
pub(super) struct Clock {
    flags: Mutex<Flags>,
    changed: Condvar,
}
impl Clock {
    pub(super) fn wake(&self) {
        self.changed.notify_all();
    }
    pub(super) fn paused(&self) -> bool {
        self.flags.lock().unwrap().paused
    }
    pub(super) fn cancel(&self, cancelled: &AtomicBool) {
        let _flags = self.flags.lock().unwrap();
        cancelled.store(true, Ordering::Release);
        self.wake();
    }
    pub(super) fn wait(
        self: &Arc<Self>,
        mut pending: PendingAccess,
        cancelled: &AtomicBool,
    ) -> Result<AccessResult, R> {
        let clock = self.clone();
        pending.set_waker(Arc::new(move || {
            let _flags = clock.flags.lock().unwrap();
            clock.changed.notify_all();
        }));
        let mut flags = self.flags.lock().unwrap();
        loop {
            if flags.stop || cancelled.load(Ordering::Acquire) {
                return Err(R::AUDIO_OPERATION_FAILED);
            }
            if let Some(result) = pending.poll() {
                return result.map_err(|error| {
                    log::error!("native audio memory access failed: {error}");
                    R::AUDIO_INVALID_ADDRESS_INFO
                });
            }
            flags = self.changed.wait(flags).unwrap();
        }
    }
}
#[derive(Debug, Default)]
pub(crate) struct MixerSystem {
    registry: Arc<Mutex<Registry>>,
    clock: Arc<Clock>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
struct Finished(Arc<Clock>);
impl Drop for Finished {
    fn drop(&mut self) {
        self.0
            .flags
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .finished = true;
        self.0.wake();
    }
}
impl MixerSystem {
    pub(crate) fn open(&self, process: &nixe_runtime::RunnableProcess) -> Option<Arc<User>> {
        let mut registry = self.registry.lock().unwrap();
        let key = process.process_id();
        if registry.users.get(&key).and_then(Weak::upgrade).is_some() {
            return None;
        }
        registry.users.retain(|_, user| user.strong_count() != 0);
        let user = Arc::new(User {
            memory: process.memory().clone(),
        });
        registry.users.insert(key, Arc::downgrade(&user));
        Some(user)
    }
    fn start_worker(&self) -> Result<(), R> {
        let mut worker = self.worker.lock().unwrap();
        if worker.is_some() {
            return Ok(());
        }
        let registry = self.registry.clone();
        let clock = self.clock.clone();
        *worker = Some(
            thread::Builder::new()
                .name("nixe-sysmix".into())
                .spawn(move || {
                    let _finished = Finished(clock.clone());
                    let quantum = Duration::from_millis(5);
                    let mut next = Instant::now();
                    let mut live = Vec::new();
                    loop {
                        let mut flags = clock.flags.lock().unwrap();
                        if flags.stop {
                            break;
                        }
                        if flags.paused {
                            drop(flags);
                            Self::collect(&registry, &mut live);
                            for stream in &live {
                                stream.pause(true);
                            }
                            live.clear();
                            flags = clock.flags.lock().unwrap();
                            flags.parked = true;
                            clock.wake();
                            while flags.paused && !flags.stop {
                                flags = clock.changed.wait(flags).unwrap();
                            }
                            flags.parked = false;
                            if flags.stop {
                                break;
                            }
                            drop(flags);
                            Self::collect(&registry, &mut live);
                            for stream in &live {
                                stream.pause(false);
                            }
                            live.clear();
                            next = Instant::now();
                            continue;
                        }
                        let now = Instant::now();
                        if now < next {
                            drop(clock.changed.wait_timeout(flags, next - now).unwrap());
                            continue;
                        }
                        drop(flags);
                        Self::collect(&registry, &mut live);
                        for stream in &live {
                            stream.pump();
                        }
                        live.clear();
                        next += quantum;
                        let now = Instant::now();
                        if next < now {
                            next = now + quantum;
                        }
                    }
                })
                .map_err(|error| {
                    log::error!("cannot start native audio worker: {error}");
                    R::AUDIO_OPERATION_FAILED
                })?,
        );
        Ok(())
    }
    fn collect(registry: &Mutex<Registry>, live: &mut Vec<Arc<Stream>>) {
        registry.lock().unwrap().streams.retain(|_, stream| {
            if let Some(stream) = stream.upgrade() {
                live.push(stream);
                true
            } else {
                false
            }
        });
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create(
        &self,
        user: Arc<User>,
        backend: Arc<dyn AudioBackend>,
        devices: Arc<AudioDeviceState>,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        address: u64,
        size: u64,
        flags: u32,
    ) -> Result<Arc<StreamSession>, R> {
        self.start_worker()?;
        let id = {
            let mut registry = self.registry.lock().unwrap();
            registry.next = registry
                .next
                .checked_add(1)
                .ok_or(R::AUDIO_OUT_OF_SESSIONS)?;
            registry.next
        };
        let stream = Stream::new(
            id,
            user,
            backend,
            devices,
            kernel,
            address,
            size,
            flags,
            self.clock.clone(),
        )?;
        self.registry
            .lock()
            .unwrap()
            .streams
            .insert(id, Arc::downgrade(&stream));
        self.clock.wake();
        Ok(Arc::new(StreamSession { stream }))
    }
    pub(crate) fn engage(
        &self,
        memory: &Arc<nixe_runtime::memory::ProcessMemory>,
        id: u32,
        engaged: bool,
    ) -> Result<(), R> {
        let stream = self
            .registry
            .lock()
            .unwrap()
            .streams
            .get(&id)
            .and_then(Weak::upgrade)
            .ok_or(R::AUDIO_NOT_FOUND)?;
        if !Arc::ptr_eq(memory, &stream.user.memory) {
            return Err(R::AUDIO_INVALID_HANDLE);
        }
        stream.engage(engaged)
    }
    pub(crate) fn set_host_paused(&self, paused: bool) {
        let worker = self.worker.lock().unwrap();
        let mut flags = self.clock.flags.lock().unwrap();
        flags.paused = paused;
        self.clock.wake();
        while worker.is_some() && paused && !flags.parked && !flags.finished {
            flags = self.clock.changed.wait(flags).unwrap();
        }
    }
}
impl Drop for MixerSystem {
    fn drop(&mut self) {
        self.clock.flags.lock().unwrap().stop = true;
        self.clock.wake();
        if let Some(worker) = self.worker.get_mut().unwrap().take() {
            worker.join().expect("native audio worker panicked");
        }
    }
}
