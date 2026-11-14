//! Shared syncpoint pages use retained physical memory and event-driven wakeups.
use nixe_memory::{Access, AccessResult, Machine, MemoryError};
use nixe_runtime::{
    ReadableEventObject, RunnableProcess, TransferMemoryObject, WritableEventObject,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, JoinHandle},
};

#[derive(Debug)]
pub(crate) struct SyncptMirror {
    pub counter_page: Arc<TransferMemoryObject>,
    pub device_counter_page: Arc<super::gpu::PhysicalMap>,
    pub notification: WritableEventObject,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

async fn read(
    machine: &Machine,
    memory: &TransferMemoryObject,
    offset: u64,
    size: usize,
) -> Result<Vec<u8>, Arc<MemoryError>> {
    match machine
        .read_captured_owned(memory.ranges(), offset, vec![0; size])
        .map_err(Arc::new)?
    {
        Access::Ready(bytes) => Ok(bytes),
        Access::Pending(pending) => {
            pending.retain_owner(Arc::new(memory.clone()));
            match pending.await? {
                AccessResult::Read(bytes) => Ok(bytes),
                _ => Err(Arc::new(MemoryError::Invalid("GSV mirror read completion"))),
            }
        }
    }
}
pub(crate) async fn write(
    machine: &Machine,
    memory: &TransferMemoryObject,
    offset: u64,
    bytes: Vec<u8>,
) -> Result<(), Arc<MemoryError>> {
    match machine
        .write_captured_owned(memory.ranges(), offset, bytes)
        .map_err(Arc::new)?
    {
        Access::Ready(_) => Ok(()),
        Access::Pending(pending) => {
            pending.retain_owner(Arc::new(memory.clone()));
            match pending.await? {
                AccessResult::Written(_) => Ok(()),
                _ => Err(Arc::new(MemoryError::Invalid(
                    "GSV mirror write completion",
                ))),
            }
        }
    }
}
impl SyncptMirror {
    pub(crate) async fn create(
        owner: &Arc<RunnableProcess>,
        device_memory: &nixe_memory::device_memory::DeviceMemory,
        rw: Arc<TransferMemoryObject>,
        ro: Arc<TransferMemoryObject>,
        count: u32,
        request: ReadableEventObject,
        notify: WritableEventObject,
    ) -> Result<Self, Box<str>> {
        let machine = owner.memory().space().machine().clone();
        // @SPECULATIVE: this shared page layout and base counter 0 follow the
        // reference's SDK observations; the ABI is absent from Switchbrew.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
        let header: [u32; 8] = [1, 0, count, 0x40, 0, 0x1040, 0x100, 0x1100];
        let mut initial = vec![0; 0x2000];
        for (word, value) in initial.chunks_exact_mut(4).zip(header) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        write(&machine, &ro, 0, initial)
            .await
            .map_err(|error| error.to_string().into_boxed_str())?;
        let device_counter_page = Arc::new(
            super::gpu::PhysicalMap::new(
                device_memory,
                ro.ranges()
                    .capture(0, 0x2000)
                    .map_err(|e| e.to_string().into_boxed_str())?,
            )
            .map_err(|e| e.to_string().into_boxed_str())?,
        );
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let process = Arc::downgrade(owner);
        let counter_page = ro.clone();
        let notification = notify.clone();
        let worker = thread::Builder::new()
            .name("nixe-gsv-syncpt".into())
            .spawn(move || {
                let task = async move {
                    loop {
                        request.wait_async().await;
                        request.clear();
                        // The client rechecks the published counter after acknowledgement.
                        // No counter is advanced by a request or by elapsed host time.
                        let armed =
                            read(&machine, &rw, 0x100, count.div_ceil(64) as usize * 8).await?;
                        let summary = read(&machine, &rw, 0, 8).await?;
                        write(&machine, &ro, 0x1100, armed).await?;
                        write(&machine, &ro, 0x1040, summary).await?;
                        notify.signal();
                    }
                    #[allow(unreachable_code)]
                    Ok::<(), Arc<MemoryError>>(())
                };
                run(task, &stopping, &process);
            })
            .map_err(|error| error.to_string().into_boxed_str())?;
        Ok(Self {
            counter_page,
            device_counter_page,
            notification,
            stop,
            worker: Some(worker),
        })
    }
}
fn run(
    task: impl Future<Output = Result<(), Arc<MemoryError>>>,
    stop: &AtomicBool,
    process: &Weak<RunnableProcess>,
) {
    let mut task = std::pin::pin!(task);
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    while !stop.load(Ordering::Acquire) {
        match Pin::as_mut(&mut task).poll(&mut context) {
            Poll::Pending => thread::park(),
            Poll::Ready(result) => {
                if let Err(error) = result
                    && let Some(process) = process.upgrade()
                {
                    process.report_device_failure(format!("GSV syncpoint mirror: {error}"));
                }
                break;
            }
        }
    }
}
impl Drop for SyncptMirror {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            worker.join().expect("GSV syncpoint mirror panicked");
        }
    }
}

/// Native counter state and event registrations share one publication lock.
/// The event is signalled only after both CPU/GPU counter pages are written.
#[derive(Debug)]
pub(crate) struct Counter {
    pub timeline: nixe_gpu::GuestTimeline,
    pub value: u64,
    pub consumed_value: u64,
    slots: std::collections::BTreeMap<usize, Weak<std::sync::Mutex<Slot>>>,
    waiters: std::collections::BTreeMap<u64, (u32, Waker)>,
    next_waiter: u64,
}
#[derive(Debug)]
pub(crate) struct Slot {
    pub event: nixe_runtime::EventObject,
    pending: Option<(u64, u64, bool)>,
}
impl Slot {
    pub fn new(event: nixe_runtime::EventObject) -> Self {
        Self {
            event,
            pending: None,
        }
    }
    pub fn cancel(&mut self) {
        self.pending = None;
        self.event.signal();
    }
}
impl Counter {
    pub fn new(timeline: nixe_gpu::GuestTimeline, value: u64) -> Self {
        Self {
            timeline,
            value,
            consumed_value: value,
            slots: Default::default(),
            waiters: Default::default(),
            next_waiter: 0,
        }
    }
    pub fn has_waiters(&self) -> bool {
        !self.waiters.is_empty()
            || self.slots.values().any(|slot| {
                slot.upgrade().is_some_and(|slot| {
                    slot.lock()
                        .unwrap()
                        .pending
                        .is_some_and(|(id, _, _)| id == self.timeline.instance().get())
                })
            })
    }
    fn reached(&self, threshold: u64, wide: bool) -> bool {
        if wide {
            self.value >= threshold
        } else {
            (self.value as u32).wrapping_sub(threshold as u32) as i32 >= 0
        }
    }
    pub fn arm(
        &mut self,
        index: usize,
        slot: &Arc<std::sync::Mutex<Slot>>,
        threshold: u64,
        wide: bool,
    ) {
        let mut state = slot.lock().unwrap();
        state.event.clear();
        state.pending = None;
        if self.reached(threshold, wide) {
            state.event.signal();
            self.slots.remove(&index);
        } else {
            state.pending = Some((self.timeline.instance().get(), threshold, wide));
            self.slots.insert(index, Arc::downgrade(slot));
        }
    }
    pub fn notify(&mut self) {
        let current = self.value as u32;
        self.waiters.retain(|_, (threshold, wake)| {
            let reached = current.wrapping_sub(*threshold) as i32 >= 0;
            if reached {
                wake.wake_by_ref();
            }
            !reached
        });
        let instance = self.timeline.instance().get();
        let value = self.value;
        self.slots.retain(|_, slot| {
            let Some(slot) = slot.upgrade() else {
                return false;
            };
            let mut slot = slot.lock().unwrap();
            let Some((id, threshold, wide)) = slot.pending else {
                return false;
            };
            if id != instance {
                return false;
            }
            let reached = if wide {
                value >= threshold
            } else {
                (value as u32).wrapping_sub(threshold as u32) as i32 >= 0
            };
            if reached {
                slot.pending = None;
                slot.event.signal();
            }
            !reached
        });
    }
}

/// Service waits use the same counter publication as guest event slots. Dropping
/// the request unregisters its wakeup; no timer or polling worker is involved.
pub(crate) struct CounterWait {
    counter: Arc<std::sync::Mutex<Counter>>,
    threshold: u32,
    registration: Option<u64>,
}
impl CounterWait {
    pub fn new(counter: Arc<std::sync::Mutex<Counter>>, threshold: u32) -> Self {
        Self {
            counter,
            threshold,
            registration: None,
        }
    }
}
impl Future for CounterWait {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut counter = this.counter.lock().unwrap();
        if counter.reached(u64::from(this.threshold), false) {
            if let Some(id) = this.registration.take() {
                counter.waiters.remove(&id);
            }
            return Poll::Ready(());
        }
        let id = *this.registration.get_or_insert_with(|| {
            counter.next_waiter = counter
                .next_waiter
                .checked_add(1)
                .expect("counter wait identity overflow");
            counter.next_waiter
        });
        match counter.waiters.entry(id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((this.threshold, context.waker().clone()));
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if !entry.get().1.will_wake(context.waker()) {
                    entry.get_mut().1.clone_from(context.waker());
                }
            }
        }
        Poll::Pending
    }
}
impl Drop for CounterWait {
    fn drop(&mut self) {
        if let Some(id) = self.registration {
            self.counter.lock().unwrap().waiters.remove(&id);
        }
    }
}

impl super::SyncptContext {
    pub fn allocate(&mut self, owner: u64) -> Result<Arc<std::sync::Mutex<Counter>>, Box<str>> {
        let id = (1..super::SYNCPOINT_COUNT)
            .find(|id| {
                self.timelines
                    .get(id)
                    .is_none_or(|t| Arc::strong_count(t) == 1 && !t.lock().unwrap().has_waiters())
            })
            .ok_or("native counters exhausted")?;
        let value = self
            .timelines
            .get(&id)
            .map_or(0, |t| t.lock().unwrap().value);
        self.next_timeline = self
            .next_timeline
            .checked_add(1)
            .ok_or("native timeline lifetime overflow")?;
        let counter = Arc::new(std::sync::Mutex::new(Counter::new(
            nixe_gpu::GuestTimeline::new(
                nixe_gpu::GuestSyncpointId::new(id),
                nixe_gpu::TimelineInstanceId::new(self.next_timeline),
                nixe_gpu::TimelineOwnerId::new(owner),
                nixe_gpu::GuestSyncpointValue::new(value as u32),
            ),
            value,
        )));
        self.timelines.insert(id, counter.clone());
        Ok(counter)
    }
    pub fn publication(&self) -> Result<super::execution::Publication, nixe_memory::MemoryError> {
        let gpu = self
            .gpu_values
            .as_ref()
            .map(|a| {
                nixe_memory::CapturedMemory::new(vec![nixe_memory::AllocationRange::new(
                    a.pin_range(0, nixe_memory::PAGE_SIZE)?,
                    0,
                    nixe_memory::PAGE_SIZE,
                )?])
            })
            .transpose()?;
        Ok(super::execution::Publication {
            gpu,
            cpu: self.mirror.as_ref().map(|m| {
                (
                    m.counter_page.clone(),
                    m.notification.clone(),
                    m.device_counter_page.clone(),
                )
            }),
        })
    }
}
