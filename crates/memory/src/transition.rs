//! Address-space quiescence. Stable native execution never takes a reader lock.
use crate::{AddressSpace, GuestVirtualAddress, MemoryError};
use std::{
    cell::Cell,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
};

const HOST_CLOSED: usize = 1 << (usize::BITS - 1);

struct CoreSlot {
    active: AtomicBool,
    acknowledged: AtomicU64,
    interrupt: Arc<dyn Fn() + Send + Sync>,
    code_pending: AtomicBool,
    code_ranges: Mutex<Vec<CodeInvalidation>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodeInvalidation {
    pub start: GuestVirtualAddress,
    pub bytes: u64,
}
#[derive(Default)]
pub(crate) struct Quiescence {
    writer: Mutex<()>,
    sequence: AtomicU64,
    host_accesses: AtomicUsize,
    cores: Mutex<Vec<Weak<CoreSlot>>>,
    notification: Mutex<()>,
    changed: Condvar,
    failure: OnceLock<Arc<MemoryError>>,
    closed: AtomicU8,
}
pub struct CoreContext {
    writes: Arc<crate::cpu_writes::CpuWriteCollector>,
    space: Arc<AddressSpace>,
    slot: Arc<CoreSlot>,
    // Contexts may move while idle, but cannot be shared between OS threads.
    _not_sync: PhantomData<Cell<()>>,
}
pub struct NativeAdmission<'a> {
    context: &'a CoreContext,
    active: Cell<bool>,
    _not_send: PhantomData<Rc<()>>,
}
pub(crate) struct HostAdmission<'a>(&'a Quiescence);
pub(crate) struct MappingTransition<'a> {
    space: &'a Quiescence,
    _writer: MutexGuard<'a, ()>,
    published: bool,
    cores: Vec<Arc<CoreSlot>>,
    requested: u64,
}
impl Quiescence {
    pub(crate) fn fail_execution(&self, error: MemoryError) -> MemoryError {
        let _ = self.failure.set(Arc::new(error));
        let cores: Vec<_> = self
            .cores
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for core in cores {
            if core.active.load(Ordering::Acquire) {
                (core.interrupt)();
            }
        }
        MemoryError::TransitionFailed(Arc::clone(self.failure.get().unwrap()))
    }
    fn check(&self) -> Result<(), MemoryError> {
        self.failure.get().map_or_else(
            || {
                if self.closed.load(Ordering::Acquire) != 0 {
                    Err(MemoryError::Invalid("address space is closed"))
                } else {
                    Ok(())
                }
            },
            |error| Err(MemoryError::TransitionFailed(Arc::clone(error))),
        )
    }
    pub fn register(
        &self,
        owner: Arc<AddressSpace>,
        interrupt: Arc<dyn Fn() + Send + Sync>,
    ) -> CoreContext {
        let slot = Arc::new(CoreSlot {
            active: AtomicBool::new(false),
            acknowledged: AtomicU64::new(0),
            interrupt,
            code_pending: AtomicBool::new(false),
            code_ranges: Mutex::new(Vec::new()),
        });
        self.cores.lock().unwrap().push(Arc::downgrade(&slot));
        CoreContext {
            writes: owner.machine().cpu_writes().register(),
            space: owner,
            slot,
            _not_sync: PhantomData,
        }
    }
    /// Caller holds the space writer and has drained native execution. Native
    /// cache mutation is deferred to that core's own thread, outside C++ Run.
    pub fn invalidate_code(&self, start: GuestVirtualAddress, bytes: u64) {
        let mut cores = self.cores.lock().unwrap();
        cores.retain(|core| core.strong_count() != 0);
        for core in cores.iter().filter_map(Weak::upgrade) {
            debug_assert!(!core.active.load(Ordering::Acquire));
            let mut ranges = core.code_ranges.lock().unwrap();
            let end = start.get() + bytes;
            if let Some(last) = ranges.last_mut().filter(|last| {
                start.get() <= last.start.get() + last.bytes && last.start.get() <= end
            }) {
                let begin = last.start.get().min(start.get());
                last.bytes = (last.start.get() + last.bytes).max(end) - begin;
                last.start = GuestVirtualAddress::new(begin);
            } else {
                ranges.push(CodeInvalidation { start, bytes });
            }
            core.code_pending.store(true, Ordering::Release);
        }
    }
    fn notify_exit(&self) {
        if self.sequence.load(Ordering::Acquire) & 1 != 0
            || self.host_accesses.load(Ordering::Acquire) & HOST_CLOSED != 0
        {
            // Serialize notification with the writer's recheck/park window.
            let _notification = self.notification.lock().unwrap();
            self.changed.notify_all();
        }
    }
    fn wait_published(&self, native: bool) -> Result<(), MemoryError> {
        let mut notification = self.notification.lock().unwrap();
        while self.sequence.load(Ordering::Acquire) & 1 != 0
            || (!native && self.host_accesses.load(Ordering::Acquire) & HOST_CLOSED != 0)
        {
            notification = self.changed.wait(notification).unwrap();
        }
        self.check()
    }
    pub fn enter_host(&self) -> Result<HostAdmission<'_>, MemoryError> {
        loop {
            self.check()?;
            let sequence = self.sequence.load(Ordering::Acquire);
            if sequence & 1 != 0
                || self
                    .host_accesses
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                        (count & HOST_CLOSED == 0).then_some(count + 1)
                    })
                    .is_err()
            {
                self.wait_published(false)?;
                continue;
            }
            if self.sequence.load(Ordering::Acquire) == sequence {
                if let Err(error) = self.check() {
                    self.host_accesses.fetch_sub(1, Ordering::Release);
                    self.notify_exit();
                    return Err(error);
                }
                return Ok(HostAdmission(self));
            }
            self.host_accesses.fetch_sub(1, Ordering::Release);
            self.notify_exit();
        }
    }
    /// The writer serializes validation and pin capture before requesting an
    /// interruption. It never waits with allocator, scheduler or page locks held.
    pub fn writer(&self) -> Result<MutexGuard<'_, ()>, MemoryError> {
        let writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.check()?;
        Ok(writer)
    }
    pub fn begin_transition<'a>(
        &'a self,
        writer: MutexGuard<'a, ()>,
    ) -> Result<MappingTransition<'a>, MemoryError> {
        self.check()?;
        self.begin_transition_inner(writer, true)
    }
    pub fn begin_tracking<'a>(
        &'a self,
        writer: MutexGuard<'a, ()>,
        interrupt_native: bool,
    ) -> Result<MappingTransition<'a>, MemoryError> {
        self.check()?;
        self.begin_transition_inner(writer, interrupt_native)
    }
    pub fn begin_close(&self) -> Result<Option<MappingTransition<'_>>, MemoryError> {
        // Teardown must also release mappings after an earlier transition or
        // native invalidation failure. No new admission may enter this space.
        let writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.closed.load(Ordering::Acquire) {
            2 => return Ok(None),
            1 => {
                self.check()?;
                return Err(MemoryError::Invalid("address-space closure was abandoned"));
            }
            _ => {}
        }
        self.closed.store(1, Ordering::Release);
        self.begin_transition_inner(writer, true).map(Some)
    }
    pub fn finish_close(&self) {
        self.closed.store(2, Ordering::Release);
    }
    fn begin_transition_inner<'a>(
        &'a self,
        writer: MutexGuard<'a, ()>,
        interrupt_native: bool,
    ) -> Result<MappingTransition<'a>, MemoryError> {
        let requested = if interrupt_native {
            let before = self.sequence.load(Ordering::Relaxed);
            let published = before.checked_add(2).ok_or(MemoryError::Invalid(
                "mapping transition generation exhausted",
            ))?;
            debug_assert_eq!(before & 1, 0);
            self.sequence.store(published - 1, Ordering::Release);
            published - 1
        } else {
            0
        };
        let previous = self.host_accesses.fetch_or(HOST_CLOSED, Ordering::AcqRel);
        debug_assert_eq!(previous & HOST_CLOSED, 0);
        let cores: Vec<_> = if interrupt_native {
            let mut registered = self.cores.lock().unwrap();
            registered.retain(|slot| slot.strong_count() != 0);
            registered.iter().filter_map(Weak::upgrade).collect()
        } else {
            Vec::new()
        };
        // Establish the RAII failure/publication boundary before external halt
        // callbacks; an unwinding callback must never leave the space closed.
        let transition = MappingTransition {
            space: self,
            _writer: writer,
            published: false,
            cores,
            requested,
        };
        for core in &transition.cores {
            if core.active.load(Ordering::Acquire) {
                (core.interrupt)();
            }
        }

        Ok(transition)
    }
}
impl CoreContext {
    /// A failed native invocation/cache update cannot publish a reusable
    /// context. Preserve its cause and stop every core using this address space.
    pub fn fail(&self, error: MemoryError) -> MemoryError {
        self.space.quiescence.fail_execution(error)
    }
    fn activate(&self) -> Result<(), MemoryError> {
        loop {
            self.space.quiescence.check()?;
            let sequence = self.space.quiescence.sequence.load(Ordering::Acquire);
            if sequence & 1 != 0 {
                self.space.quiescence.wait_published(true)?;
                continue;
            }
            self.slot
                .active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| MemoryError::Invalid("native core context is already active"))?;
            if self.space.quiescence.sequence.load(Ordering::Acquire) == sequence {
                if let Err(error) = self.space.quiescence.check() {
                    self.deactivate();
                    return Err(error);
                }
                return Ok(());
            }
            self.deactivate();
        }
    }
    fn deactivate(&self) {
        self.slot.active.store(false, Ordering::Release);
        self.slot.acknowledged.store(
            self.space.quiescence.sequence.load(Ordering::Acquire),
            Ordering::Release,
        );
        self.space.quiescence.notify_exit();
    }
    pub fn enter_native(&self) -> Result<NativeAdmission<'_>, MemoryError> {
        self.activate()?;
        Ok(NativeAdmission {
            context: self,
            active: Cell::new(true),
            _not_send: PhantomData,
        })
    }
}
impl Drop for CoreContext {
    fn drop(&mut self) {
        self.deactivate();
    }
}
impl NativeAdmission<'_> {
    pub(crate) fn cpu_writes(&self) -> &crate::cpu_writes::CpuWriteCollector {
        &self.context.writes
    }
    /// Call after admission and before invoking native code. Admission prevents
    /// a writer from queuing another invalidation between this drain and Run.
    /// The unchanged path is one atomic test, with no queue/mutex participation.
    pub fn synchronize_code(
        &self,
        mut invalidate: impl FnMut(CodeInvalidation) -> Result<(), MemoryError>,
    ) -> Result<(), MemoryError> {
        if !self.active.get() {
            return Err(MemoryError::Invalid(
                "code synchronization requires active native admission",
            ));
        }
        let slot = &self.context.slot;
        if !slot.code_pending.load(Ordering::Acquire) {
            return Ok(());
        }
        let ranges = {
            let mut ranges = slot.code_ranges.lock().unwrap();
            slot.code_pending.store(false, Ordering::Release);
            std::mem::take(&mut *ranges)
        };
        for range in ranges {
            if let Err(error) = invalidate(range) {
                // The local native cache is now only partially invalidated.
                // Do not allow this or another core to resume the space.
                self.suspend();
                return Err(self.context.fail(error));
            }
        }
        Ok(())
    }
    pub(crate) fn address_space(&self) -> &AddressSpace {
        &self.context.space
    }
    pub(crate) fn is_active(&self) -> bool {
        self.active.get()
    }
    /// Capture/pin the old physical operation before suspending. The native
    /// invocation must have returned first: a paused C++ run/compilation frame
    /// is not a mapping or code-invalidation safepoint. Recovery may then wait
    /// on graphics without preventing its mapping/protection writer.
    pub fn suspend(&self) {
        if self.active.replace(false) {
            self.context.deactivate();
        }
    }
    pub fn resume(&self) -> Result<(), MemoryError> {
        if !self.active.get() {
            self.context.activate()?;
            self.active.set(true);
        }
        Ok(())
    }
}
impl Drop for NativeAdmission<'_> {
    fn drop(&mut self) {
        self.suspend();
    }
}
impl Drop for HostAdmission<'_> {
    fn drop(&mut self) {
        self.0.host_accesses.fetch_sub(1, Ordering::Release);
        self.0.notify_exit();
    }
}
impl MappingTransition<'_> {
    pub fn wait_quiescent(&self) {
        let mut notification = self.space.notification.lock().unwrap();
        while self.space.host_accesses.load(Ordering::Acquire) & !HOST_CLOSED != 0
            || self.cores.iter().any(|core| {
                core.active.load(Ordering::Acquire)
                    && core.acknowledged.load(Ordering::Acquire) < self.requested
            })
        {
            notification = self.space.changed.wait(notification).unwrap();
        }
    }

    pub fn fail(&self, error: MemoryError) -> MemoryError {
        self.space.fail_execution(error)
    }
    pub fn publish(mut self) {
        self.published = true;
    }
}
impl Drop for MappingTransition<'_> {
    fn drop(&mut self) {
        if !self.published {
            self.space.fail_execution(MemoryError::Invalid(
                "address-space transition abandoned before publication",
            ));
        }
        let _notification = self.space.notification.lock().unwrap();
        if self.requested != 0 {
            self.space.sequence.fetch_add(1, Ordering::Release);
        }
        self.space
            .host_accesses
            .fetch_and(!HOST_CLOSED, Ordering::Release);
        self.space.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time::Duration};
    #[test]
    fn data_tracking_drains_host_access_without_closing_native_admission() {
        let space = AddressSpace::new(
            crate::Machine::new(crate::PAGE_SIZE).unwrap(),
            crate::AddressWidth::Bits32,
        )
        .unwrap();
        let core = space.register_core(Arc::new(|| panic!("data protection interrupted a core")));
        let native = core.enter_native().unwrap();
        let host = space.quiescence.enter_host().unwrap();
        let transition = space
            .quiescence
            .begin_tracking(space.quiescence.writer().unwrap(), false)
            .unwrap();
        let (entered, native_entered) = std::sync::mpsc::channel();
        let running = Arc::clone(&space);
        let newcomer = thread::spawn(move || {
            let core = running.register_core(Arc::new(|| {
                panic!("data protection interrupted a new core")
            }));
            let _native = core.enter_native().unwrap();
            entered.send(()).unwrap();
        });
        let arrived = native_entered.recv_timeout(Duration::from_secs(1));
        let (entered, host_entered) = std::sync::mpsc::channel();
        let pending = Arc::clone(&space);
        let new_host = thread::spawn(move || {
            let _host = pending.quiescence.enter_host().unwrap();
            entered.send(()).unwrap();
        });
        assert!(
            host_entered
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );
        drop(host);
        transition.wait_quiescent();
        transition.publish();
        host_entered.recv_timeout(Duration::from_secs(1)).unwrap();
        newcomer.join().unwrap();
        new_host.join().unwrap();
        arrived.expect("data protection closed native admission");
        assert!(native.is_active());
    }
    #[test]
    fn writer_waits_for_native_and_host_access_then_publishes_before_resume() {
        let space = AddressSpace::new(
            crate::Machine::new(crate::PAGE_SIZE).unwrap(),
            crate::AddressWidth::Bits32,
        )
        .unwrap();
        let (halt, requested) = std::sync::mpsc::channel();
        let context = space.register_core(Arc::new(move || {
            halt.send(()).unwrap();
        }));
        let (running, entered) = std::sync::mpsc::channel();
        let (resume, resumed) = std::sync::mpsc::channel();
        let (host_release, host_may_release) = std::sync::mpsc::channel();
        let data = Arc::new(AtomicU64::new(0));
        let observed = Arc::clone(&data);
        let core = thread::spawn(move || {
            let native = context.enter_native().unwrap();
            running.send(()).unwrap();
            requested.recv().unwrap();
            native.suspend();
            native.resume().unwrap();
            resume.send(observed.load(Ordering::Acquire)).unwrap();
        });
        let host_space = Arc::clone(&space);
        let (held, host_entered) = std::sync::mpsc::channel();
        let host = thread::spawn(move || {
            let _access = host_space.quiescence.enter_host().unwrap();
            held.send(()).unwrap();
            host_may_release.recv().unwrap();
        });
        entered.recv().unwrap();
        host_entered.recv().unwrap();
        let update = Arc::clone(&space);
        let (ready, ready_rx) = std::sync::mpsc::channel();
        let writer = thread::spawn(move || {
            let transition = update
                .quiescence
                .begin_transition(update.quiescence.writer().unwrap())
                .unwrap();
            transition.wait_quiescent();
            data.store(73, Ordering::Release);
            transition.publish();
            ready.send(()).unwrap();
        });
        assert!(ready_rx.recv_timeout(Duration::from_millis(20)).is_err());
        host_release.send(()).unwrap();
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(resumed.recv_timeout(Duration::from_secs(1)).unwrap(), 73);
        core.join().unwrap();
        host.join().unwrap();
        writer.join().unwrap();
    }
    #[test]
    fn failed_transition_unblocks_waiters_with_the_original_failure() {
        let space = AddressSpace::new(
            crate::Machine::new(crate::PAGE_SIZE).unwrap(),
            crate::AddressWidth::Bits32,
        )
        .unwrap();
        let transition = space
            .quiescence
            .begin_transition(space.quiescence.writer().unwrap())
            .unwrap();
        let error = transition.fail(MemoryError::Invalid("injected mapping syscall error"));
        drop(transition);
        assert!(error.to_string().contains("injected mapping syscall error"));
        assert!(
            space
                .quiescence
                .enter_host()
                .err()
                .unwrap()
                .to_string()
                .contains("injected mapping syscall error")
        );
        let context = space.register_core(Arc::new(|| {}));
        assert!(context.enter_native().is_err());
    }
    #[test]
    fn closure_drains_running_core_and_releases_ram_retained_by_stopped_context() {
        use crate::{AddressWidth, MemoryPermissions, PAGE_SIZE};
        let machine = crate::Machine::new(PAGE_SIZE).unwrap();
        let account = machine.account(PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                account.allocate(PAGE_SIZE).unwrap(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let captured = space.capture(base, 4, MemoryPermissions::WRITE).unwrap();
        let (halt, halted) = std::sync::mpsc::channel();
        let context = space.register_core(Arc::new(move || halt.send(()).unwrap()));
        let (started, running) = std::sync::mpsc::channel();
        let core = thread::spawn(move || {
            let native = context.enter_native().unwrap();
            started.send(()).unwrap();
            halted.recv_timeout(Duration::from_secs(5)).unwrap();
            native.suspend();
            assert!(native.resume().is_err());
            drop(native);
            context
        });
        running.recv_timeout(Duration::from_secs(5)).unwrap();
        space.close().unwrap();
        let retained_context = core.join().unwrap();
        assert!(retained_context.enter_native().is_err());
        assert!(space.translate(base, MemoryPermissions::READ).is_err());
        assert_eq!(account.used_bytes(), 0);
        assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
        assert!(matches!(
            machine.apply_resolved_access(
                &captured,
                &crate::AccessOperation::Write(Box::new(9u32.to_le_bytes()))
            ),
            Ok(crate::AccessResult::Written)
        ));
        drop(captured);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        space.close().unwrap();
        assert!(retained_context.enter_native().is_err());
    }

    #[test]
    fn closure_after_native_failure_releases_mappings_and_preserves_original_cause() {
        use crate::{AddressWidth, MemoryPermissions, PAGE_SIZE};
        let machine = crate::Machine::new(PAGE_SIZE).unwrap();
        let account = machine.account(PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(PAGE_SIZE);
        space
            .map(
                base,
                account.allocate(PAGE_SIZE).unwrap(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let context = space.register_core(Arc::new(|| {}));
        space
            .protect(base, PAGE_SIZE, MemoryPermissions::READ_EXECUTE)
            .unwrap();
        let native = context.enter_native().unwrap();
        let MemoryError::TransitionFailed(original) = native
            .synchronize_code(|_| Err(MemoryError::Invalid("native invalidation failed")))
            .unwrap_err()
        else {
            panic!("expected original failure")
        };
        drop(native);
        space.close().unwrap();
        space.close().unwrap();
        assert_eq!(account.used_bytes(), 0);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        let MemoryError::TransitionFailed(later) = context.enter_native().err().unwrap() else {
            panic!("closure replaced original failure")
        };
        assert!(Arc::ptr_eq(&original, &later));
    }
    #[test]
    fn code_invalidation_is_consumed_on_the_core_after_publication() {
        use crate::{AddressWidth, GuestVirtualAddress, MemoryPermissions, PAGE_SIZE};
        let machine = crate::Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x1000);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let (halt, halted) = std::sync::mpsc::channel();
        let context = space.register_core(Arc::new(move || {
            halt.send(()).unwrap();
        }));
        let (started, running) = std::sync::mpsc::channel();
        let core = thread::spawn(move || {
            let native = context.enter_native().unwrap();
            native
                .synchronize_code(|_| panic!("ordinary RAM unexpectedly invalidated code"))
                .unwrap();
            started.send(()).unwrap();
            halted.recv().unwrap();
            native.suspend();
            native.resume().unwrap();
            let mut invalidations = Vec::new();
            let own_thread = thread::current().id();
            native
                .synchronize_code(|range| {
                    assert_eq!(thread::current().id(), own_thread);
                    invalidations.push(range);
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                invalidations,
                [CodeInvalidation {
                    start: base,
                    bytes: PAGE_SIZE
                }]
            );
            native
                .synchronize_code(|_| panic!("consumed code invalidation was repeated"))
                .unwrap();
        });
        running.recv().unwrap();
        space
            .protect(base, PAGE_SIZE, MemoryPermissions::READ_EXECUTE)
            .unwrap();
        core.join().unwrap();
        assert!(space.translate(base, MemoryPermissions::EXECUTE).is_ok());
    }
    #[test]
    fn native_code_invalidation_failure_prevents_partial_cache_reuse() {
        use crate::{AddressWidth, GuestVirtualAddress, MemoryPermissions, PAGE_SIZE};
        let machine = crate::Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x1000);
        space
            .map(
                base,
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let context = space.register_core(Arc::new(|| {}));
        space
            .protect(base, PAGE_SIZE, MemoryPermissions::READ_EXECUTE)
            .unwrap();
        let native = context.enter_native().unwrap();
        let failure = native
            .synchronize_code(|_| {
                Err(MemoryError::Invalid(
                    "injected native cache invalidation failure",
                ))
            })
            .unwrap_err();
        assert!(!native.is_active());
        assert!(failure.to_string().contains("injected native cache"));
        assert!(
            context
                .enter_native()
                .err()
                .unwrap()
                .to_string()
                .contains("injected native cache")
        );
        assert!(
            space
                .translate(base, MemoryPermissions::READ)
                .unwrap_err()
                .to_string()
                .contains("injected native cache")
        );
    }
}
