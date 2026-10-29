//! Deadline dispatch independent of CPU returns. Queue callbacks run unlocked.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/core_timing.cpp
use super::*;
use std::{
    cmp::{Ordering as Comparison, Reverse},
    collections::BinaryHeap,
    future::Future,
    pin::Pin,
    sync::{
        OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
    thread::JoinHandle,
    time::Duration,
};

#[cfg(feature = "performance-counters")]
use std::sync::atomic::AtomicU64;

#[cfg(feature = "performance-counters")]
#[derive(Clone, Copy, Debug, Default)]
pub struct TimerStatistics {
    pub expirations: u64,
    pub lateness_ns: u64,
    pub maximum_lateness_ns: u64,
    pub host_wakeups: u64,
    pub compactions: u64,
}
#[cfg(feature = "performance-counters")]
#[derive(Debug, Default)]
struct Statistics {
    expirations: AtomicU64,
    lateness_ns: AtomicU64,
    maximum_lateness_ns: AtomicU64,
    host_wakeups: AtomicU64,
    compactions: AtomicU64,
}

#[derive(Debug)]
enum Action {
    Wait(Weak<KernelWait>),
    Alarm(Weak<Alarm>),
    Preempt(u64),
}

impl Action {
    fn live(&self) -> bool {
        match self {
            Self::Wait(wait) => wait.upgrade().is_some_and(|wait| wait.outcome().is_none()),
            Self::Alarm(alarm) => alarm
                .upgrade()
                .is_some_and(|alarm| alarm.state.load(Ordering::Acquire) == 0),
            Self::Preempt(_) => true,
        }
    }
}

#[derive(Debug)]
struct Entry {
    deadline: u64,
    sequence: u64,
    action: Action,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        (self.deadline, self.sequence) == (other.deadline, other.sequence)
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Comparison> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Comparison {
        (self.deadline, self.sequence).cmp(&(other.deadline, other.sequence))
    }
}

#[derive(Debug, Default)]
struct Queue {
    heap: BinaryHeap<Reverse<Entry>>,
    sequence: u64,
}

#[derive(Debug)]
struct Shared {
    queue: Mutex<Queue>,
    advance: Mutex<()>,
    live: Arc<AtomicUsize>,
    entries: AtomicUsize,
    compact_pending: AtomicBool,
    stopped: AtomicBool,
    preemption: bool,
    #[cfg(feature = "performance-counters")]
    statistics: Statistics,
    notification: OnceLock<Waker>,
    notification_pending: AtomicBool,
    failed: AtomicBool,
}

#[derive(Debug)]
pub(super) struct Timers {
    shared: Arc<Shared>,
    clock: crate::VirtualClock,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Timers {
    pub(super) fn new(clock: crate::VirtualClock, preemption: bool) -> Self {
        Self {
            shared: Arc::new(Shared {
                queue: Mutex::default(),
                advance: Mutex::new(()),
                live: Arc::new(AtomicUsize::new(0)),
                entries: AtomicUsize::new(0),
                compact_pending: AtomicBool::new(false),
                stopped: AtomicBool::new(false),
                preemption,
                #[cfg(feature = "performance-counters")]
                statistics: Statistics::default(),
                notification: OnceLock::new(),
                notification_pending: AtomicBool::new(false),
                failed: AtomicBool::new(false),
            }),
            clock,
            worker: Mutex::new(None),
        }
    }

    pub(super) fn start(
        &self,
        kernel: Weak<SchedulerState>,
        interval: Option<u64>,
        notification: Waker,
    ) -> std::io::Result<()> {
        let mut owner = self.worker.lock().expect("timer worker lock poisoned");
        if owner.is_some() || self.shared.stopped.load(Ordering::Acquire) {
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }
        if self.shared.notification.set(notification).is_err() {
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }
        if let Some(interval) = interval {
            self.insert(
                self.clock.scheduler_time_ns().saturating_add(interval),
                Action::Preempt(interval),
            );
        }
        if self.clock.mode() != crate::VirtualClockMode::Realtime {
            return Ok(());
        }
        let shared = self.shared.clone();
        let clock = self.clock.clone();
        let worker = std::thread::Builder::new()
            .name("nixe-timer".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    loop {
                        let epoch = clock.timer_epoch();
                        if shared.stopped.load(Ordering::Acquire) {
                            break;
                        }
                        let next = Self::next(&shared);
                        let now = clock.scheduler_time_ns();
                        if !clock.is_paused() && next.is_some_and(|next| next <= now) {
                            drop(epoch);
                            let Some(kernel) = kernel.upgrade() else {
                                break;
                            };
                            kernel.timers.advance(&kernel);
                            continue;
                        }
                        let timeout = if !clock.is_paused() {
                            next.map(|next| Duration::from_nanos(next.saturating_sub(now)))
                        } else {
                            None
                        };
                        drop(clock.wait_for_timer_change(epoch, timeout));
                        #[cfg(feature = "performance-counters")]
                        shared
                            .statistics
                            .host_wakeups
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }));
                if result.is_err() {
                    shared.failed.store(true, Ordering::Release);
                    shared.notify();
                }
            })?;
        *owner = Some(worker);
        Ok(())
    }

    fn compact(shared: &Shared, queue: &mut Queue) {
        shared.compact_pending.store(false, Ordering::Release);
        let live = shared.live.load(Ordering::Acquire) + usize::from(shared.preemption);
        if queue.heap.len() > 1024 && queue.heap.len() > live.saturating_mul(2) {
            queue.heap.retain(|entry| entry.0.action.live());
            #[cfg(feature = "performance-counters")]
            shared
                .statistics
                .compactions
                .fetch_add(1, Ordering::Relaxed);
        }
        shared.entries.store(queue.heap.len(), Ordering::Release);
    }
    fn next(shared: &Shared) -> Option<u64> {
        let mut queue = shared.queue.lock().expect("timer queue lock poisoned");
        if shared.compact_pending.load(Ordering::Acquire) {
            Self::compact(shared, &mut queue);
        }
        while queue
            .heap
            .peek()
            .is_some_and(|entry| !entry.0.action.live())
        {
            queue.heap.pop();
        }
        shared.entries.store(queue.heap.len(), Ordering::Release);
        queue.heap.peek().map(|entry| entry.0.deadline)
    }
    pub(super) fn invalidate(&self) {
        let entries = self.shared.entries.load(Ordering::Acquire);
        let live = self.shared.live.load(Ordering::Acquire) + usize::from(self.shared.preemption);
        if entries > 1024
            && entries > live.saturating_mul(2)
            && !self.shared.compact_pending.swap(true, Ordering::AcqRel)
        {
            if self.clock.mode() == crate::VirtualClockMode::Realtime {
                self.clock.wake_timers();
            } else {
                Self::compact(
                    &self.shared,
                    &mut self.shared.queue.lock().expect("timer queue lock poisoned"),
                );
            }
        }
    }

    fn insert(&self, deadline: u64, action: Action) -> bool {
        let mut queue = self.shared.queue.lock().expect("timer queue lock poisoned");
        if self.shared.stopped.load(Ordering::Acquire) {
            return false;
        }
        let previous = queue.heap.peek().map(|entry| entry.0.deadline);
        Self::compact(&self.shared, &mut queue);
        let sequence = queue.sequence;
        queue.sequence = sequence
            .checked_add(1)
            .expect("kernel deadline sequence exhausted");
        queue.heap.push(Reverse(Entry {
            deadline,
            sequence,
            action,
        }));
        self.shared
            .entries
            .store(queue.heap.len(), Ordering::Release);
        let changed = previous.is_none_or(|previous| deadline < previous);
        drop(queue);
        if changed {
            self.clock.wake_timers();
        }
        true
    }

    pub(super) fn arm_wait(&self, wait: &Arc<KernelWait>, deadline: u64) {
        if wait.arm_timer(&self.shared.live)
            && !self.insert(deadline, Action::Wait(Arc::downgrade(wait)))
            && let Some(kernel) = wait.kernel.upgrade()
        {
            kernel.resolve_wait(wait.token(), WaitOutcome::Terminated);
        }
    }

    pub(super) fn advance(&self, kernel: &SchedulerState) -> usize {
        let _advance = self
            .shared
            .advance
            .lock()
            .expect("timer advance lock poisoned");
        if self.clock.is_paused() {
            return 0;
        }
        let mut count = 0;
        loop {
            let now = self.clock.scheduler_time_ns();
            let entry = {
                let mut queue = self.shared.queue.lock().expect("timer queue lock poisoned");
                if queue.heap.peek().is_none_or(|entry| entry.0.deadline > now) {
                    break;
                }
                let entry = queue.heap.pop().expect("due timer").0;
                self.shared
                    .entries
                    .store(queue.heap.len(), Ordering::Release);
                entry
            };
            #[cfg(feature = "performance-counters")]
            if entry.action.live() {
                let lateness = now.saturating_sub(entry.deadline);
                self.shared
                    .statistics
                    .expirations
                    .fetch_add(1, Ordering::Relaxed);
                self.shared
                    .statistics
                    .lateness_ns
                    .fetch_add(lateness, Ordering::Relaxed);
                self.shared
                    .statistics
                    .maximum_lateness_ns
                    .fetch_max(lateness, Ordering::Relaxed);
            }
            match entry.action {
                Action::Wait(wait) => {
                    if let Some(wait) = wait.upgrade()
                        && kernel.resolve_wait(wait.token(), WaitOutcome::TimedOut)
                    {
                        count += 1;
                    }
                }
                Action::Alarm(alarm) => {
                    if let Some(alarm) = alarm.upgrade() {
                        alarm.finish(1);
                    }
                }
                Action::Preempt(interval) => {
                    if kernel.rotate() {
                        self.shared.notify();
                    }
                    let next = now.saturating_add(interval - (now - entry.deadline) % interval);
                    self.insert(next, Action::Preempt(interval));
                }
            }
        }
        count
    }

    pub(super) fn stop(&self, kernel: &SchedulerState) -> std::thread::Result<()> {
        if !self.shared.stopped.swap(true, Ordering::AcqRel) {
            self.clock.wake_timers();
        }
        let worker = self
            .worker
            .lock()
            .expect("timer worker lock poisoned")
            .take();
        let joined = if let Some(worker) = worker
            && worker.thread().id() != std::thread::current().id()
        {
            worker.join()
        } else {
            Ok(())
        };
        let _advance = self
            .shared
            .advance
            .lock()
            .expect("timer advance lock poisoned");
        let entries = {
            let mut queue = self.shared.queue.lock().expect("timer queue lock poisoned");
            self.shared.entries.store(0, Ordering::Release);
            std::mem::take(&mut queue.heap)
        };
        for entry in entries {
            match entry.0.action {
                Action::Wait(wait) => {
                    if let Some(wait) = wait.upgrade() {
                        kernel.resolve_wait(wait.token(), WaitOutcome::Terminated);
                    }
                }
                Action::Alarm(alarm) => {
                    if let Some(alarm) = alarm.upgrade() {
                        alarm.finish(2);
                    }
                }
                Action::Preempt(_) => {}
            }
        }
        if self.shared.failed.load(Ordering::Acquire) {
            Err(Box::new("timer worker failed"))
        } else {
            joined
        }
    }
}

impl Shared {
    fn notify(&self) {
        if !self.notification_pending.swap(true, Ordering::AcqRel)
            && let Some(notification) = self.notification.get()
        {
            notification.wake_by_ref();
        }
    }
}

#[derive(Debug)]
struct Alarm {
    state: AtomicU8,
    waker: Mutex<Option<Waker>>,
    live: Arc<AtomicUsize>,
}

impl Alarm {
    fn finish(&self, result: u8) {
        if self
            .state
            .compare_exchange(0, result, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.live.fetch_sub(1, Ordering::Release);
            let waker = self.waker.lock().expect("timer alarm lock poisoned").take();
            if let Some(waker) = waker {
                waker.wake();
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimerStopped;

/// A service deadline uses the kernel heap and does not occupy a host worker.
pub struct Deadline {
    kernel: Weak<SchedulerState>,
    deadline: u64,
    alarm: Option<Arc<Alarm>>,
}

impl Future for Deadline {
    type Output = Result<(), TimerStopped>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(kernel) = this.kernel.upgrade() else {
            return Poll::Ready(Err(TimerStopped));
        };
        if let Some(alarm) = &this.alarm {
            match alarm.state.load(Ordering::Acquire) {
                1 => return Poll::Ready(Ok(())),
                2 => return Poll::Ready(Err(TimerStopped)),
                _ => {}
            }
        } else {
            if !kernel.clock.is_paused() && kernel.clock.scheduler_time_ns() >= this.deadline {
                return Poll::Ready(Ok(()));
            }
            kernel.timers.shared.live.fetch_add(1, Ordering::Relaxed);
            let alarm = Arc::new(Alarm {
                state: AtomicU8::new(0),
                waker: Mutex::new(None),
                live: kernel.timers.shared.live.clone(),
            });
            if !kernel
                .timers
                .insert(this.deadline, Action::Alarm(Arc::downgrade(&alarm)))
            {
                alarm.finish(2);
            }
            this.alarm = Some(alarm);
        }
        let alarm = this.alarm.as_ref().expect("registered timer alarm");
        let next = context.waker().clone();
        let previous = alarm
            .waker
            .lock()
            .expect("timer alarm lock poisoned")
            .replace(next);
        drop(previous);
        match alarm.state.load(Ordering::Acquire) {
            0 => Poll::Pending,
            1 => Poll::Ready(Ok(())),
            _ => Poll::Ready(Err(TimerStopped)),
        }
    }
}

impl Drop for Deadline {
    fn drop(&mut self) {
        if let Some(alarm) = &self.alarm {
            let cancelled = alarm
                .state
                .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
            if cancelled {
                alarm.live.fetch_sub(1, Ordering::Release);
            }
            let waker = alarm
                .waker
                .lock()
                .expect("timer alarm lock poisoned")
                .take();
            drop(waker);
            if cancelled && let Some(kernel) = self.kernel.upgrade() {
                kernel.timers.invalidate();
            }
        }
    }
}

impl SchedulerState {
    #[cfg(feature = "performance-counters")]
    pub fn timing_statistics(&self) -> TimerStatistics {
        let statistics = &self.timers.shared.statistics;
        TimerStatistics {
            expirations: statistics.expirations.load(Ordering::Relaxed),
            lateness_ns: statistics.lateness_ns.load(Ordering::Relaxed),
            maximum_lateness_ns: statistics.maximum_lateness_ns.load(Ordering::Relaxed),
            host_wakeups: statistics.host_wakeups.load(Ordering::Relaxed),
            compactions: statistics.compactions.load(Ordering::Relaxed),
        }
    }

    pub fn clock(&self) -> &crate::VirtualClock {
        &self.clock
    }

    pub fn deadline(self: &Arc<Self>, deadline: u64) -> Deadline {
        Deadline {
            kernel: Arc::downgrade(self),
            deadline,
            alarm: None,
        }
    }

    pub fn arm_deadline(&self, wait: &Arc<KernelWait>, deadline: u64) {
        self.timers.arm_wait(wait, deadline);
    }

    pub fn advance_time(&self, nanoseconds: u64) -> usize {
        self.clock.advance_scheduler_to(nanoseconds);
        self.timers.advance(self)
    }

    pub fn start_timers(self: &Arc<Self>, notification: Waker) -> std::io::Result<()> {
        self.timers.start(
            Arc::downgrade(self),
            self.profile.preemption_interval_ns(),
            notification,
        )
    }

    pub fn acknowledge_timer_notification(&self) -> bool {
        self.timers
            .shared
            .notification_pending
            .store(false, Ordering::Release);
        self.timers.shared.failed.load(Ordering::Acquire)
    }

    pub fn deadline_count(&self) -> usize {
        self.timers.shared.live.load(Ordering::Acquire)
    }

    pub fn stop_timers(&self) -> std::thread::Result<()> {
        let result = self.timers.stop(self);
        let retired = self
            .edit(|state| {
                state.closed = true;
                let mut retired = Vec::new();
                for slot in 0..state.slots.entries.len() {
                    let Some(thread) = state.slots.entries[slot].thread.as_ref() else {
                        continue;
                    };
                    if thread.wait.is_none() {
                        continue;
                    }
                    let index = Index::new(slot as u32 + 1).expect("kernel thread slot");
                    state.resolve_wait(index, WaitOutcome::Terminated, &self.profile);
                    let wait = state.slots[index].wait.take();
                    let nodes = std::mem::take(&mut state.slots[index].wait_nodes);
                    retired.push((wait, nodes));
                }
                Ok(retired)
            })
            .expect("kernel wait shutdown");
        drop(retired);
        result
    }
}

impl Drop for SchedulerState {
    fn drop(&mut self) {
        let _ = self.timers.stop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PriorityRange, VirtualClock, VirtualClockMode, VirtualCpuDescriptor};
    use std::sync::mpsc;
    use std::task::Wake;

    fn kernel(mode: VirtualClockMode) -> Arc<SchedulerState> {
        let kernel = SchedulerState::new(
            MachineSchedulerProfile::new(
                vec![VirtualCpuDescriptor::new(VirtualCpuId::new(0), 0)],
                PriorityRange::new(0, 63).unwrap(),
            )
            .unwrap(),
            VirtualClock::new(mode),
        );
        kernel.start_timers(Waker::noop().clone()).unwrap();
        kernel
    }

    fn blocked(kernel: &Arc<SchedulerState>) -> Arc<KernelWait> {
        let thread = kernel
            .create_thread(ScheduledThreadConfig {
                process: ProcessId::new(1),
                base_priority: 44,
                ideal_vcpu: Some(VirtualCpuId::new(0)),
                affinity: kernel.profile().all_cores(),
            })
            .unwrap();
        kernel.make_ready(thread).unwrap();
        let lease = kernel.claim(VirtualCpuId::new(0)).unwrap().unwrap();
        kernel.complete(lease, Completion::Waiting).unwrap();
        kernel.register_wait(thread).unwrap()
    }

    struct Notify(mpsc::Sender<()>);
    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            let _ = self.0.send(());
        }
    }

    #[test]
    fn timer_resolves_a_registered_continuation_without_a_cpu_completion() {
        let kernel = kernel(VirtualClockMode::Realtime);
        let wait = blocked(&kernel);
        let (send, receive) = mpsc::channel();
        let waker = Waker::from(Arc::new(Notify(send)));
        let mut future = std::pin::pin!(wait.wait());
        assert!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        kernel.arm_deadline(&wait, kernel.clock().scheduler_time_ns() + 2_000_000);
        receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(wait.outcome(), Some(WaitOutcome::TimedOut));
        assert_eq!(
            future.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Ready(WaitOutcome::TimedOut)
        );
        // The result owner must install its return before guest execution.
        assert_eq!(
            kernel.thread(wait.token().thread).unwrap().lifecycle,
            ThreadLifecycle::Waiting
        );
        kernel.finish_wait(wait.token()).unwrap();
        assert_eq!(
            kernel.thread(wait.token().thread).unwrap().lifecycle,
            ThreadLifecycle::Ready
        );
    }

    #[test]
    fn registration_has_one_terminal_winner_and_never_reenters_waiting() {
        let kernel = kernel(VirtualClockMode::Fixed { unix_seconds: 0 });
        let wait = blocked(&kernel);
        assert_eq!(wait.phase(), WaitPhase::Registering);
        let outcomes = [
            WaitOutcome::Signalled(3),
            WaitOutcome::TimedOut,
            WaitOutcome::Cancelled,
        ];
        let wins = std::thread::scope(|scope| {
            let threads: Vec<_> = outcomes
                .into_iter()
                .map(|outcome| {
                    let kernel = kernel.clone();
                    let token = wait.token();
                    scope.spawn(move || kernel.resolve_wait(token, outcome))
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .filter(|won| *won)
                .count()
        });
        assert_eq!(wins, 1);
        assert!(!wait.commit());
        assert!(wait.outcome().is_some());
        assert!(!kernel.resolve_wait(wait.token(), WaitOutcome::Ready));
        kernel.finish_wait(wait.token()).unwrap();
    }

    #[test]
    fn earlier_realtime_head_interrupts_the_existing_sleep() {
        let kernel = kernel(VirtualClockMode::Realtime);
        let mut later =
            std::pin::pin!(kernel.deadline(kernel.clock().scheduler_time_ns() + 5_000_000_000));
        assert!(
            later
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        // Allow the worker to start waiting for the original head.
        std::thread::sleep(Duration::from_millis(10));
        let (send, receive) = mpsc::channel();
        let waker = Waker::from(Arc::new(Notify(send)));
        let mut earlier =
            std::pin::pin!(kernel.deadline(kernel.clock().scheduler_time_ns() + 2_000_000));
        assert!(
            earlier
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(
            earlier.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Ready(Ok(()))
        );
        assert!(
            later
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        kernel.stop_timers().unwrap();
    }

    #[test]
    fn paused_realtime_deadlines_retain_their_remaining_guest_time() {
        let kernel = kernel(VirtualClockMode::Realtime);
        kernel.clock().set_paused(true);
        let paused = kernel.clock().scheduler_time_ns();
        let (send, receive) = mpsc::channel();
        let waker = Waker::from(Arc::new(Notify(send)));
        let mut deadline = std::pin::pin!(kernel.deadline(paused + 15_000_000));
        assert!(
            deadline
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        assert!(receive.recv_timeout(Duration::from_millis(30)).is_err());
        assert_eq!(kernel.clock().scheduler_time_ns(), paused);
        kernel.clock().set_paused(false);
        receive.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(
            deadline.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Ready(Ok(()))
        );
        assert!(kernel.clock().scheduler_time_ns() >= paused + 15_000_000);
        kernel.stop_timers().unwrap();
    }

    #[test]
    fn equal_deadlines_preserve_insertion_order_and_bulk_cancellation_compacts() {
        struct OrderedNotify(usize, mpsc::Sender<usize>);
        impl Wake for OrderedNotify {
            fn wake(self: Arc<Self>) {
                self.1.send(self.0).unwrap();
            }
        }
        let kernel = kernel(VirtualClockMode::Fixed { unix_seconds: 0 });
        let (send, receive) = mpsc::channel();
        let mut alarms = Vec::new();
        for index in 0..8 {
            let waker = Waker::from(Arc::new(OrderedNotify(index, send.clone())));
            let mut alarm = Box::pin(kernel.deadline(100));
            assert!(
                alarm
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending()
            );
            alarms.push(alarm);
        }
        kernel.advance_time(100);
        assert_eq!(
            receive.try_iter().collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        alarms.clear();
        for _ in 0..2200 {
            let mut alarm = Box::pin(kernel.deadline(1_000_000_000_000));
            assert!(
                alarm
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            alarms.push(alarm);
        }
        assert_eq!(kernel.deadline_count(), 2200);
        drop(alarms);
        assert_eq!(kernel.deadline_count(), 0);
        assert!(kernel.timers.shared.queue.lock().unwrap().heap.len() <= 1024);
        kernel.stop_timers().unwrap();
    }

    #[test]
    fn stale_far_future_deadlines_compact_and_do_not_delay_a_new_head() {
        let kernel = kernel(VirtualClockMode::Fixed { unix_seconds: 0 });
        for _ in 0..2200 {
            let wait = blocked(&kernel);
            kernel.arm_deadline(&wait, 1_000_000_000_000);
            kernel.finish_wait(wait.token()).unwrap();
            kernel.terminate(wait.token().thread, false).unwrap();
            kernel.unregister(wait.token().thread).unwrap();
        }
        assert!(kernel.timers.shared.queue.lock().unwrap().heap.len() <= 1024);
        let wait = blocked(&kernel);
        kernel.arm_deadline(&wait, 100);
        assert_eq!(kernel.advance_time(99), 0);
        assert_eq!(wait.outcome(), None);
        assert_eq!(kernel.advance_time(100), 1);
        assert_eq!(wait.outcome(), Some(WaitOutcome::TimedOut));
        kernel.finish_wait(wait.token()).unwrap();
        assert_eq!(kernel.timers.shared.live.load(Ordering::Acquire), 0);
    }
}
