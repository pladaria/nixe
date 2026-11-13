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
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let process = Arc::downgrade(owner);
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
