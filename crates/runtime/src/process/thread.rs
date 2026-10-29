use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use crate::GuestCpuState;
use nixe_memory::GuestVirtualAddress;
use nixe_scheduler::{CoreSet, GuestThreadId, VirtualCpuId};

use crate::{EventObject, ReadableEventObject, ThreadExit};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadCreateRequest {
    pub entry: GuestVirtualAddress,
    pub argument: u64,
    pub fpcr: u32,
    pub fpsr: u32,
    pub stack_top: GuestVirtualAddress,
    pub priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadCreation {
    pub id: GuestThreadId,
    pub handle: u32,
}

#[derive(Clone, Debug)]
pub enum ThreadCreateError {
    InvalidPriority(i32),
    InvalidVirtualCpu(VirtualCpuId),
    OutOfHandles,
    OutOfMemory,
    Memory(Arc<nixe_memory::MemoryError>),
    Terminating,
    IdentityExhausted,
    Internal,
}

impl Display for ThreadCreateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "guest thread creation failed: {self:?}")
    }
}

impl Error for ThreadCreateError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadStartError {
    InvalidState,
    Internal,
}

/// The current kernel wait owns its accepted resources and cancellation.
/// Termination cancels it independently of the entry which stopped the thread.
/// https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_thread.cpp
pub trait ThreadWait: Send + Sync {
    fn cancel(&self);
}

/// The canonical kernel thread retained by handles and active execution.
/// Scheduling fields belong to the same generational ID in SchedulerState.
pub struct ThreadObject {
    id: AtomicU64,
    registers: Mutex<Option<GuestCpuState>>,
    exit: Mutex<Option<ThreadExit>>,
    pub stack_bottom: GuestVirtualAddress,
    pub stack_top: GuestVirtualAddress,
    pub tls_base: GuestVirtualAddress,
    pub loader_return: Option<GuestVirtualAddress>,
    completion: EventObject,
    wait: Mutex<Option<Weak<dyn ThreadWait>>>,
    // The process closes its handles/running references on termination; exported
    // threads retain their parent until final destruction.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp
    owner: OnceLock<Arc<super::RunnableProcess>>,
    retirement: OnceLock<(nixe_scheduler::ProcessId, crate::ExternalEventSender)>,
    pub(super) _allocation: ThreadLocalRegion,
}
impl std::fmt::Debug for ThreadObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadObject")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}
/// One kernel-owned TLS region. It is released with the last retained thread,
/// after its native context and running reference have been relinquished.
pub(super) struct ThreadLocalRegion {
    pub pages: Arc<Mutex<super::thread_local::ThreadLocalPages>>,
    pub memory: Arc<crate::memory::ProcessMemory>,
    pub address: GuestVirtualAddress,
}
impl ThreadLocalRegion {
    pub(super) fn allocate(
        pages: Arc<Mutex<super::thread_local::ThreadLocalPages>>,
        memory: Arc<crate::memory::ProcessMemory>,
    ) -> Result<Self, crate::memory::MappingError> {
        // Reserve metadata under the pool lock; map/initialize outside it, as
        // CreateThreadLocalRegion does for a new KThreadLocalPage.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_process.cpp
        let page = {
            let mut state = pages.lock().unwrap();
            if state.closed {
                return Err(crate::memory::MappingError::StateMismatch);
            }
            if let Some(address) = state.reserve_existing() {
                return Ok(Self {
                    pages: pages.clone(),
                    memory,
                    address,
                });
            }
            state.reserve_page()?
        };
        if let Err(error) =
            memory.map_zeroed(page, nixe_memory::PAGE_SIZE, super::thread_local::MAPPING)
        {
            pages.lock().unwrap().recycle_page(page);
            return Err(error);
        }
        let mut state = pages.lock().unwrap();
        if state.closed {
            return Err(crate::memory::MappingError::StateMismatch);
        }
        state.insert_page(page);
        drop(state);
        Ok(Self {
            pages,
            memory,
            address: page,
        })
    }
}
impl Drop for ThreadLocalRegion {
    fn drop(&mut self) {
        let page = {
            let mut pages = self.pages.lock().unwrap();
            if pages.closed {
                return;
            }
            pages.release_region(self.address)
        };
        if let Some(page) = page {
            // The page leaves the allocation index before unmapping. Its VA is
            // reusable only after the mapping has been released.
            self.memory
                .unmap(page, nixe_memory::PAGE_SIZE, super::thread_local::MAPPING)
                .expect("the final TLS region retains its kernel page mapping");
            self.pages.lock().unwrap().recycle_page(page);
        }
    }
}
impl ThreadObject {
    pub(super) fn new(
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        id: GuestThreadId,
        state: GuestCpuState,
        stack_bottom: GuestVirtualAddress,
        stack_top: GuestVirtualAddress,
        loader_return: Option<GuestVirtualAddress>,
        allocation: ThreadLocalRegion,
    ) -> Self {
        Self {
            id: AtomicU64::new(id.get()),
            registers: Mutex::new(Some(state)),
            exit: Mutex::new(None),
            stack_bottom,
            stack_top,
            tls_base: allocation.address,
            loader_return,
            completion: EventObject::new(kernel),
            wait: Mutex::new(None),
            retirement: OnceLock::new(),
            owner: OnceLock::new(),
            _allocation: allocation,
        }
    }
    pub fn id(&self) -> GuestThreadId {
        GuestThreadId::new(self.thread_id())
    }
    pub fn thread_id(&self) -> u64 {
        self.id.load(Ordering::Acquire)
    }
    pub(crate) fn assign_thread_id(&self, id: u64) {
        self.id.store(id, Ordering::Release);
    }
    pub(crate) fn bind_retirement(
        &self,
        process: nixe_scheduler::ProcessId,
        events: crate::ExternalEventSender,
    ) {
        self.retirement
            .set((process, events))
            .expect("thread is registered once");
    }
    pub(crate) fn bind_owner(&self, process: &Arc<super::RunnableProcess>) {
        self.owner.set(process.clone()).expect("one thread owner");
    }
    pub fn owner(&self) -> Option<&Arc<super::RunnableProcess>> {
        self.owner.get()
    }
    /// Starts the retained kernel object and takes its running reference once.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L1152-L1184
    pub fn start(self: &Arc<Self>) -> Result<(), ThreadStartError> {
        let process = self.owner().ok_or(ThreadStartError::InvalidState)?;
        let id = self.id();
        let mut state = process.state.lock().unwrap();
        if state.lifecycle != nixe_scheduler::ProcessLifecycle::Running {
            return Err(ThreadStartError::InvalidState);
        }
        process.threads.start(self)?;
        state.num_running_threads += 1;
        drop(state);
        match process.scheduler().start_thread(id) {
            Ok(()) => Ok(()),
            Err(error) => {
                let mut state = process.state.lock().unwrap();
                if process.threads.finish(id) {
                    state.num_running_threads -= 1;
                }
                match error {
                    nixe_scheduler::SchedulerError::InvalidThreadState { .. } => {
                        Err(ThreadStartError::InvalidState)
                    }
                    _ => Err(ThreadStartError::Internal),
                }
            }
        }
    }
    pub fn is_signalled(&self) -> bool {
        self.completion.is_signalled()
    }
    pub(crate) fn signal(&self) {
        self.cancel_wait();
        self.completion.signal();
    }
    pub fn set_wait(&self, wait: Weak<dyn ThreadWait>) {
        let mut current = self.wait.lock().unwrap();
        assert!(current.is_none(), "one kernel wait per thread");
        *current = Some(wait);
    }
    pub fn clear_wait(&self) {
        self.wait.lock().unwrap().take();
    }
    pub(crate) fn cancel_wait(&self) {
        let wait = self
            .wait
            .lock()
            .unwrap()
            .take()
            .and_then(|wait| wait.upgrade());
        if let Some(wait) = wait {
            wait.cancel();
        }
    }
    pub fn readable_event(&self) -> ReadableEventObject {
        self.completion.readable()
    }

    pub fn exit(&self) -> Option<ThreadExit> {
        *self.exit.lock().unwrap()
    }
    pub(crate) fn set_exit(&self, exit: ThreadExit) {
        *self.exit.lock().unwrap() = Some(exit);
    }
    pub fn registers(&self) -> ThreadRegisters<'_> {
        ThreadRegisters(self.registers.lock().unwrap())
    }
    pub(crate) fn take_state(&self) -> Option<GuestCpuState> {
        self.registers.lock().unwrap().take()
    }
    pub(crate) fn restore_state(&self, state: GuestCpuState) {
        assert!(
            self.registers.lock().unwrap().replace(state).is_none(),
            "one register owner"
        );
    }

    pub fn snapshot(&self) -> Option<nixe_cpu::state::ThreadCpuState> {
        self.registers
            .lock()
            .unwrap()
            .as_ref()
            .filter(|state| state.is_available())
            .map(|state| state.snapshot())
    }
    pub(crate) fn materialize(&self, leave_core: bool) {
        if let Some(state) = self.registers.lock().unwrap().as_mut() {
            if leave_core {
                state.leave_core();
            } else {
                state.materialize();
            }
        }
    }
}
impl Drop for ThreadObject {
    fn drop(&mut self) {
        if let Some((process, events)) = self.retirement.get() {
            events.retire_thread(
                *process,
                self.id(),
                Arc::downgrade(self.owner.get().expect("registered thread has an owner")),
            );
        }
    }
}
/// A short register borrow for stopped-thread diagnostics and initialization.
/// Exception dispatch moves the register owner out, so it holds no lock while
/// accessing guest memory, scheduling, or devices.
pub struct ThreadRegisters<'a>(MutexGuard<'a, Option<GuestCpuState>>);
impl std::ops::Deref for ThreadRegisters<'_> {
    type Target = GuestCpuState;
    fn deref(&self) -> &Self::Target {
        self.0
            .as_ref()
            .expect("registers belong to the running core")
    }
}
impl std::ops::DerefMut for ThreadRegisters<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
            .as_mut()
            .expect("registers belong to the running core")
    }
}
struct Entry {
    object: Weak<ThreadObject>,
    running: Option<Arc<ThreadObject>>,
}
/// Process membership only; all mutable thread fields live in ThreadObject.
#[derive(Default)]
pub struct ThreadTable {
    entries: Mutex<BTreeMap<GuestThreadId, Entry>>,
}
impl ThreadTable {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(
        &self,
        thread: &Arc<ThreadObject>,
        started: bool,
    ) -> Result<GuestThreadId, ThreadTableError> {
        let id = thread.id();
        let mut entries = self.entries.lock().unwrap();
        if entries.contains_key(&id) {
            return Err(ThreadTableError::Duplicate(id));
        }
        entries.insert(
            id,
            Entry {
                object: Arc::downgrade(thread),
                running: started.then(|| thread.clone()),
            },
        );
        Ok(id)
    }
    pub fn get(&self, id: GuestThreadId) -> Option<Arc<ThreadObject>> {
        self.entries.lock().unwrap().get(&id)?.object.upgrade()
    }
    pub fn remove(&self, id: GuestThreadId) -> Result<(), ThreadTableError> {
        let entry = self
            .entries
            .lock()
            .unwrap()
            .remove(&id)
            .ok_or(ThreadTableError::Missing(id))?;
        drop(entry);
        Ok(())
    }
    pub(super) fn start(&self, thread: &Arc<ThreadObject>) -> Result<(), ThreadStartError> {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries
            .get_mut(&thread.id())
            .ok_or(ThreadStartError::Internal)?;
        if entry.running.is_some() {
            return Err(ThreadStartError::InvalidState);
        }
        entry.running = Some(Arc::clone(thread));
        Ok(())
    }
    pub(super) fn finish(&self, id: GuestThreadId) -> bool {
        let running = self
            .entries
            .lock()
            .unwrap()
            .get_mut(&id)
            .and_then(|entry| entry.running.take());
        let was_running = running.is_some();
        drop(running);
        was_running
    }
    pub(super) fn release_running(&self) {
        let running: Vec<_> = self
            .entries
            .lock()
            .unwrap()
            .values_mut()
            .filter_map(|entry| entry.running.take())
            .collect();
        drop(running);
    }
    pub(crate) fn rekey(
        &self,
        old: GuestThreadId,
        new: GuestThreadId,
    ) -> Result<(), ThreadTableError> {
        let mut entries = self.entries.lock().unwrap();
        if entries.contains_key(&new) {
            return Err(ThreadTableError::Duplicate(new));
        }
        let entry = entries.remove(&old).ok_or(ThreadTableError::Missing(old))?;
        entries.insert(new, entry);
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = (GuestThreadId, Arc<ThreadObject>)> + use<> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(id, entry)| Some((*id, entry.object.upgrade()?)))
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub(crate) fn ids(&self) -> Vec<GuestThreadId> {
        self.entries.lock().unwrap().keys().copied().collect()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadTableError {
    Duplicate(GuestThreadId),
    Missing(GuestThreadId),
}
impl Display for ThreadTableError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "kernel thread registry: {self:?}")
    }
}
impl Error for ThreadTableError {}
