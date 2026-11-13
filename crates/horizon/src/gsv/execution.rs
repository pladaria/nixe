//! Native command/MME owner and ordered graphics-preparation owner. IPC only
//! admits retained inputs; neither worker retains an accepted kernel request.
use super::gpu::{GpuAddressSpace, GpuCapture};
use nixe_gpu::native::ImageAddressSpace;
use nixe_gpu_ampere::{self as ampere, FrontendHost, PreparationHost, Result, Source};
use nixe_memory::{Access, AccessResult, Machine, PendingAccess};
use nixe_runtime::RunnableProcess;
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Poll, Waker},
    thread::{self, JoinHandle},
};
mod control;
mod frontend;
mod semaphore;

/// One physical cache and native owner are selected for the application. The
/// startup closure erases its type once; command preparation stays monomorphic.
pub struct AmpereGraphics {
    pub(crate) memory: nixe_memory::device_memory::DeviceMemory,
    owner: Mutex<Owner>,
}
type Initialize = Box<dyn FnOnce(&Arc<RunnableProcess>) -> Result<Arc<Executor>> + Send>;
struct Owner {
    initialize: Option<Initialize>,
    executor: Option<Arc<Executor>>,
}
impl std::fmt::Debug for AmpereGraphics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmpereGraphics").finish_non_exhaustive()
    }
}
impl AmpereGraphics {
    pub fn new<E: ampere::Rasterizer>(
        memory: nixe_memory::device_memory::DeviceMemory,
        initialize: impl FnOnce() -> std::result::Result<E, nixe_gpu::BackendRuntimeError>
        + Send
        + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            memory,
            owner: Mutex::new(Owner {
                executor: None,
                initialize: Some(Box::new(move |process| Executor::new(process, initialize))),
            }),
        })
    }
    pub(crate) fn executor(&self, process: &Arc<RunnableProcess>) -> Result<Arc<Executor>> {
        let mut owner = self.owner.lock().unwrap();
        if let Some(executor) = &owner.executor {
            return Ok(executor.clone());
        }
        let initialize = owner
            .initialize
            .take()
            .ok_or("native GPU initialization already failed")?;
        let executor = initialize(process)?;
        owner.executor = Some(executor.clone());
        Ok(executor)
    }
}

const MAX_INPUT: usize = 64 * 1024 * 1024;

pub(crate) struct Submission {
    pub id: u64,
    pub channel: u64,
    pub space_id: u64,
    pub space: Arc<Mutex<GpuAddressSpace>>,
    pub entries: Vec<ampere::GpfifoEntry>,
    pub timeline: Arc<Mutex<crate::gsv::syncpt::Counter>>,
    pub completion: Option<nixe_gpu::ReservedTimelinePoint>,
    pub(crate) publication: Option<Publication>,
    _permit: Permit,
}
struct Permit {
    bytes: usize,
    budget: Arc<AtomicUsize>,
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.budget.fetch_sub(self.bytes, Ordering::Release);
    }
}
#[derive(Debug)]
pub(crate) struct Stopped;
struct ReplyState {
    result: Option<std::result::Result<(), Stopped>>,
    wake: Option<Waker>,
    finished: bool,
}
pub(crate) struct Completion(Arc<Mutex<ReplyState>>);
impl Completion {
    fn new() -> (Reply, Self) {
        let state = Arc::new(Mutex::new(ReplyState {
            result: None,
            wake: None,
            finished: false,
        }));
        (Reply(state.clone()), Self(state))
    }
    pub async fn wait(self) -> std::result::Result<(), Stopped> {
        std::future::poll_fn(|cx| {
            let mut state = self.0.lock().unwrap();
            if let Some(result) = state.result.take() {
                Poll::Ready(result)
            } else {
                state.wake = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await
    }
}
struct Reply(Arc<Mutex<ReplyState>>);
impl Reply {
    fn finish(self) {
        let mut state = self.0.lock().unwrap();
        state.finished = true;
        state.result = Some(Ok(()));
        let wake = state.wake.take();
        drop(state);
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
impl Drop for Reply {
    fn drop(&mut self) {
        let mut state = self.0.lock().unwrap();
        if !state.finished {
            state.result = Some(Err(Stopped));
        }
        let wake = state.wake.take();
        drop(state);
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
enum Ingress {
    Submit(Arc<Submission>),
    Memory(MemoryJob, Reply),
    Retire(u64, Reply),
    Stop,
}
enum Control {
    Failure(Box<str>),
    Begin(Arc<Submission>),
    End,
    Memory(MemoryJob, Reply),
    Retire(u64, Reply),
}
#[derive(Debug)]
pub(crate) struct Executor {
    input: mpsc::SyncSender<Ingress>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    frontend_thread: thread::Thread,
    budget: Arc<AtomicUsize>,
    next: std::sync::atomic::AtomicU64,
    endpoint: Arc<control::Endpoint>,
}
/// Counter pages are retained separately from the service context. Workers do
/// not re-enter a session lock during completion or process teardown.
pub(crate) struct Publication {
    pub cpu: Option<(
        Arc<nixe_runtime::TransferMemoryObject>,
        nixe_runtime::WritableEventObject,
        Arc<super::gpu::PhysicalMap>,
    )>,
    pub gpu: Option<nixe_memory::CapturedMemory>,
}
impl Executor {
    pub fn new<E: ampere::Rasterizer>(
        process: &Arc<RunnableProcess>,
        initialize: impl FnOnce() -> std::result::Result<E, nixe_gpu::BackendRuntimeError>
        + Send
        + 'static,
    ) -> Result<Arc<Self>> {
        let (input, receive) = mpsc::sync_channel(128);
        let (send, mut recv) = nixe_gpu::ordered::channel(1 << 17);
        let stop = Arc::new(AtomicBool::new(false));
        let machine = process.memory().space().machine().clone();
        let clock = process.scheduler().clock().clone();
        let weak = Arc::downgrade(process);
        let endpoint = Arc::new(control::Endpoint::default());
        let route: Arc<dyn nixe_memory::DeviceAccessEndpoint> = endpoint.clone();
        machine.set_device_endpoint(&route);
        let back_endpoint = endpoint.clone();
        let waiters = Arc::new(semaphore::Waiters::default());
        let front_waiters = waiters.clone();
        let front_stop = stop.clone();
        let front_machine = machine.clone();
        let front_process = weak.clone();
        let front = thread::Builder::new()
            .name("nixe-ampere-mme".into())
            .spawn(move || {
                let result =
                    frontend::run(receive, send, front_machine, &front_stop, &front_waiters);
                if let Err(e) = result
                    && !front_stop.swap(true, Ordering::AcqRel)
                    && let Some(p) = front_process.upgrade()
                {
                    log::error!("Ampere command execution failed: {e}");
                    p.report_device_failure(e);
                }
            })
            .map_err(|e| e.to_string().into_boxed_str())?;
        let back_stop = stop.clone();
        let back = match thread::Builder::new()
            .name("nixe-ampere-gpu".into())
            .spawn(move || {
                let result = (|| {
                    let mut renderer = initialize().map_err(|e| e.to_string().into_boxed_str())?;
                    back_endpoint.bind(&mut renderer)?;
                    let result = preparation(
                        &mut recv,
                        machine,
                        clock,
                        &back_stop,
                        &waiters,
                        &back_endpoint,
                        &mut renderer,
                    );
                    if let Err(error) = &result
                        && !back_stop.swap(true, Ordering::AcqRel)
                    {
                        log::error!("Ampere graphics preparation failed: {error}");
                        if let Some(process) = weak.upgrade() {
                            process.report_device_failure(error.clone());
                        }
                    }
                    back_endpoint.stop();
                    let teardown = renderer
                        .teardown()
                        .map_err(|e| e.to_string().into_boxed_str());
                    result.and(teardown)
                })();
                back_endpoint.stop();
                if let Err(e) = result
                    && !back_stop.swap(true, Ordering::AcqRel)
                    && let Some(p) = weak.upgrade()
                {
                    log::error!("Ampere native owner failed: {e}");
                    p.report_device_failure(e);
                }
            }) {
            Ok(t) => t,
            Err(e) => {
                stop.store(true, Ordering::Release);
                let _ = input.send(Ingress::Stop);
                let _ = front.join();
                return Err(e.to_string().into());
            }
        };
        Ok(Arc::new(Self {
            input,
            stop,
            frontend_thread: front.thread().clone(),
            threads: vec![back, front],
            budget: Arc::new(AtomicUsize::new(0)),
            next: std::sync::atomic::AtomicU64::new(1),
            endpoint,
        }))
    }
    pub fn submission(
        &self,
        space: Arc<Mutex<GpuAddressSpace>>,
        entries: Vec<ampere::GpfifoEntry>,
        timeline: Arc<Mutex<crate::gsv::syncpt::Counter>>,
        completion: Option<nixe_gpu::ReservedTimelinePoint>,
        publication: Option<Publication>,
    ) -> Result<Submission> {
        let channel = timeline.lock().unwrap().timeline.instance().get();
        let space_id = space.lock().unwrap().image_space_id();
        let bytes = entries
            .iter()
            .try_fold(0_usize, |sum, e| sum.checked_add(e.words as usize * 4))
            .ok_or("GPU command size overflow")?;
        self.budget
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&n| n <= MAX_INPUT)
            })
            .map_err(|_| "native GPU ingress budget exhausted")?;
        let permit = Permit {
            bytes,
            budget: self.budget.clone(),
        };
        Ok(Submission {
            id: self.next.fetch_add(1, Ordering::Relaxed),
            channel,
            space_id,
            space,
            entries,
            timeline,
            completion,
            publication,
            _permit: permit,
        })
    }
    pub fn present(&self, frame: crate::gsv::display::Frame) -> Result<Completion> {
        if self.stopped() {
            return Err("native GPU owner is stopping".into());
        }
        Ok(self.endpoint.present(frame))
    }
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }
    pub fn submit(&self, submission: Submission) -> Result<()> {
        if self.stopped() {
            return Err("native GPU owner is stopping".into());
        }
        self.input
            .try_send(Ingress::Submit(Arc::new(submission)))
            .map_err(|e| format!("native GPU ingress: {e}").into_boxed_str())?;
        self.frontend_thread.unpark();
        Ok(())
    }
    pub fn memory(&self, job: MemoryJob) -> Result<Completion> {
        let (reply, completion) = Completion::new();
        self.input
            .try_send(Ingress::Memory(job, reply))
            .map_err(|e| format!("native GPU memory admission: {e}").into_boxed_str())?;
        self.frontend_thread.unpark();
        Ok(completion)
    }
    pub fn retire(&self, channel: u64) -> Result<Completion> {
        let (reply, completion) = Completion::new();
        self.input
            .try_send(Ingress::Retire(channel, reply))
            .map_err(|e| format!("native GPU retirement admission: {e}").into_boxed_str())?;
        self.frontend_thread.unpark();
        Ok(completion)
    }
}
impl Drop for Executor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.endpoint.stop();
        let _ = self.input.try_send(Ingress::Stop);
        for t in &self.threads {
            t.thread().unpark();
        }
        for t in self.threads.drain(..) {
            if t.join().is_err() {
                log::error!("native GPU worker panicked during shutdown");
            }
        }
    }
}
struct Memory<'a> {
    machine: &'a Machine,
    space: &'a Arc<Mutex<GpuAddressSpace>>,
    stop: &'a AtomicBool,
    waiters: &'a semaphore::Waiters,
}
impl Memory<'_> {
    fn pending(&self, mut pending: PendingAccess) -> Result<AccessResult> {
        let thread = thread::current();
        pending.set_waker(Arc::new(move || thread.unpark()));
        loop {
            if let Some(result) = pending.poll() {
                return result.map_err(|e| e.to_string().into());
            }
            if self.stop.load(Ordering::Acquire) {
                return Err("GPU memory access cancelled".into());
            }
            thread::park();
        }
    }
    fn captured_read(&self, source: &Arc<GpuCapture>, bytes: Vec<u8>) -> Result<Vec<u8>> {
        match self
            .machine
            .read_captured_owned(&source.ranges, 0, bytes)
            .map_err(|e| e.to_string().into_boxed_str())?
        {
            Access::Ready(b) => Ok(b),
            Access::Pending(p) => {
                p.retain_owner(source.clone());
                match self.pending(p)? {
                    AccessResult::Read(b) => Ok(b),
                    _ => Err("invalid command read completion".into()),
                }
            }
        }
    }
    fn read(&self, address: u64, bytes: usize) -> Result<Vec<u8>> {
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let capture = Arc::new(
            self.space
                .lock()
                .unwrap()
                .capture(address, bytes as u64)
                .map_err(|e| {
                    format!("GPU read address={address:#x} bytes={bytes:#x}: {e}").into_boxed_str()
                })?,
        );
        self.captured_read(&capture, vec![0; bytes])
    }
}
struct Front<'a> {
    writer: &'a mut ampere::ordered::Writer<Control>,
    memory: Memory<'a>,
    writers: &'a mut bool,
}
impl FrontendHost for Front<'_> {
    fn method(&mut self, source: Source) -> Result<()> {
        // Ordinary register writes cannot change command/MME input memory.
        // Effects are ordered before the next input fetch; no per-word waits.
        if ampere::writes_memory(source) {
            *self.writers = true;
        }
        self.writer.method(source)
    }
    fn barrier(&mut self) -> Result<()> {
        self.writer.ring.flush();
        self.writer
            .ring
            .wait_until(self.writer.ring.position())
            .map_err(|_| "native preparation stopped before coherent dependency")?;
        *self.writers = false;
        Ok(())
    }
    fn read(&mut self, address: u64, bytes: usize) -> Result<Vec<u8>> {
        self.memory.read(address, bytes)
    }
    fn stopped(&self) -> bool {
        self.memory.stop.load(Ordering::Acquire)
    }
}
struct Back<'a, E> {
    renderer: &'a mut E,
    source: Source,
    memory: Memory<'a>,
    clock: &'a nixe_scheduler::VirtualClock,
}
impl<E: ampere::Rasterizer> PreparationHost for Back<'_, E> {
    fn blit(&mut self, source: Source, operation: &ampere::blit::Blit) -> Result<()> {
        self.renderer
            .blit(source, operation, &mut *self.memory.space.lock().unwrap())
    }
    fn read(&mut self, address: u64, bytes: usize) -> Result<Vec<u8>> {
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let capture = self
            .memory
            .space
            .lock()
            .unwrap()
            .capture(address, bytes as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        read_owned(self.renderer, &capture.ranges, bytes)
    }
    fn write(&mut self, address: u64, bytes: Vec<u8>) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let capture = self
            .memory
            .space
            .lock()
            .unwrap()
            .capture(address, bytes.len() as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        let ranges = capture
            .ranges
            .capture(0, bytes.len() as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        let written = match self
            .renderer
            .access_memory(ranges, nixe_memory::AccessOperation::Write(bytes))
            .map_err(|e| e.to_string().into_boxed_str())?
        {
            AccessResult::Written(bytes) => bytes,
            _ => return Err("invalid native GPU write completion".into()),
        };
        self.memory
            .waiters
            .written(&capture.ranges, 0, &written, |source| {
                read_owned(self.renderer, &source.ranges, 4)
            })
    }
    fn report(&mut self, address: u64, bytes: Vec<u8>) -> Result<()> {
        let capture = self
            .memory
            .space
            .lock()
            .unwrap()
            .capture(address, bytes.len() as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        let ranges = capture
            .ranges
            .capture(0, bytes.len() as u64)
            .map_err(|e| e.to_string().into_boxed_str())?;
        self.renderer
            .report_memory(
                nixe_gpu::FrontendSubmissionId::new(self.source.submission),
                ranges,
                bytes.into(),
                false,
            )
            .map_err(|e| e.to_string().into_boxed_str())?;
        // Physical pins alone do not retain the client's CPU sharing lease.
        self.renderer
            .signal_fence(Some(Box::new(move |result| {
                drop(capture);
                result
            })))
            .map_err(|e| e.to_string().into())
    }
    fn barrier(&mut self) -> Result<()> {
        self.renderer
            .order_device(nixe_gpu::FrontendSubmissionId::new(self.source.submission))
            .map_err(|e| e.to_string().into())
    }
    fn fragment_barrier(&mut self, before_depth: bool) -> Result<()> {
        self.renderer.fragment_barrier(self.source, before_depth)
    }
    fn invalidate_texture_data(&mut self) -> Result<()> {
        self.renderer
            .invalidate_gpu_cache()
            .map_err(|e| e.to_string().into_boxed_str())?;
        self.barrier()
    }
    fn timestamp(&self) -> u64 {
        // @SPECULATIVE: retain the current report-clock scale until GA10F's
        // guest timestamp conversion is documented. Scheduler time stays native.
        (u128::from(self.clock.scheduler_time_ns()) * 384 / 625 / 256) as u64
    }
    fn stopped(&self) -> bool {
        self.memory.stop.load(Ordering::Acquire)
    }
    fn dispatch(
        &mut self,
        launch: &ampere::compute::Launch,
        _registers: &ampere::Registers,
    ) -> Result<()> {
        let code = self.read(launch.program, 16)?;
        Err(format!("Ampere shader compilation not implemented: QMD={:#x} program={:#x} grid={:?} block={:?} shared_bytes={} first_instruction={:02x?}; dispatch and releases not executed", launch.address, launch.program, launch.grid, launch.block, launch.shared_bytes, code).into())
    }
    fn clear(
        &mut self,
        source: Source,
        regs: &ampere::Registers,
        state: &mut ampere::GraphicsState,
    ) -> Result<()> {
        self.renderer
            .clear(source, regs, state, &mut *self.memory.space.lock().unwrap())
    }
    fn draw(
        &mut self,
        source: Source,
        regs: &ampere::Registers,
        state: &mut ampere::GraphicsState,
    ) -> Result<()> {
        self.renderer
            .draw(source, regs, state, &mut *self.memory.space.lock().unwrap())
    }
}
fn read_owned(
    renderer: &mut impl nixe_gpu::OwnerMemory,
    capture: &nixe_memory::CapturedMemory,
    bytes: usize,
) -> Result<Vec<u8>> {
    let ranges = capture
        .capture(0, bytes as u64)
        .map_err(|e| e.to_string().into_boxed_str())?;
    match renderer
        .access_memory(ranges, nixe_memory::AccessOperation::Read(vec![0; bytes]))
        .map_err(|e| e.to_string().into_boxed_str())?
    {
        AccessResult::Read(bytes) => Ok(bytes),
        _ => Err("invalid owner memory read completion".into()),
    }
}
fn preparation<E: ampere::Rasterizer>(
    input: &mut nixe_gpu::ordered::Receiver<Control>,
    machine: Machine,
    clock: nixe_scheduler::VirtualClock,
    stop: &AtomicBool,
    waiters: &semaphore::Waiters,
    endpoint: &control::Endpoint,
    renderer: &mut E,
) -> Result<()> {
    let mut current = None::<Arc<Submission>>;
    let mut channels = BTreeMap::<u64, ampere::Preparation>::new();
    let mut schedulers = BTreeMap::<u64, ampere::compute::Scheduler>::new();
    let mut source = Source::default();
    while !stop.load(Ordering::Acquire) {
        while let Some(message) = input.try_recv() {
            match message {
                nixe_gpu::ordered::Read::Control(Control::Failure(error)) => return Err(error),
                nixe_gpu::ordered::Read::Control(Control::Begin(job)) => {
                    source.channel = job.channel;
                    source.submission = job.id;
                    if let std::collections::btree_map::Entry::Vacant(entry) =
                        channels.entry(job.channel)
                    {
                        renderer.create_channel(job.channel);
                        entry.insert(ampere::Preparation::default());
                    }
                    renderer.bind_channel(job.channel);
                    renderer.begin_frame_work();
                    current = Some(job);
                }
                nixe_gpu::ordered::Read::Control(Control::End) => {
                    let job = current.take().ok_or("submission end without beginning")?;
                    // A PCAS request with unresolved dependencies remains in
                    // the address-space scheduler; it is not a running dispatch.
                    // Retiring this command stream must not retire that QMD or
                    // publish its releases. Subsequent PCAS can make it runnable.
                    renderer
                        .end_command_list()
                        .map_err(|e| e.to_string().into_boxed_str())?;
                    if let Some(point) = &job.completion {
                        for _ in 0..point.increments() {
                            let (value, wide_value) = {
                                let mut timeline = job.timeline.lock().unwrap();
                                let value = timeline
                                    .timeline
                                    .consume_increment(point.owner(), point)
                                    .map_err(|e| e.to_string().into_boxed_str())?;
                                timeline.consumed_value = timeline
                                    .consumed_value
                                    .checked_add(1)
                                    .ok_or("native counter overflow")?;
                                (value, timeline.consumed_value)
                            };
                            if let Some(publication) = &job.publication {
                                let id = u64::from(value.syncpoint().get());
                                if let Some(gpu) = &publication.gpu {
                                    renderer
                                        .report_memory(
                                            nixe_gpu::FrontendSubmissionId::new(job.id),
                                            gpu.capture(id * 4, 4)
                                                .map_err(|e| e.to_string().into_boxed_str())?,
                                            Arc::from(value.value().get().to_le_bytes()),
                                            false,
                                        )
                                        .map_err(|e| e.to_string().into_boxed_str())?;
                                }
                                if let Some((cpu, _, _)) = &publication.cpu {
                                    renderer
                                        .report_memory(
                                            nixe_gpu::FrontendSubmissionId::new(job.id),
                                            cpu.ranges()
                                                .capture(0x40 + id * 8, 8)
                                                .map_err(|e| e.to_string().into_boxed_str())?,
                                            Arc::from(wide_value.to_le_bytes()),
                                            false,
                                        )
                                        .map_err(|e| e.to_string().into_boxed_str())?;
                                }
                            }
                            let job = job.clone();
                            renderer
                                .signal_fence(Some(Box::new(move |result| {
                                    result?;
                                    let point = job.completion.as_ref().unwrap();
                                    let mut timeline = job.timeline.lock().unwrap();
                                    timeline
                                        .timeline
                                        .increment_reserved(point.owner(), point)
                                        .map_err(|e| {
                                            Arc::new(nixe_memory::MemoryError::Execution(
                                                e.to_string().into(),
                                            ))
                                        })?;
                                    timeline.value = wide_value;
                                    timeline.notify();
                                    drop(timeline);
                                    if let Some((_, notify, _)) =
                                        job.publication.as_ref().and_then(|p| p.cpu.as_ref())
                                    {
                                        notify.signal();
                                    }
                                    Ok(())
                                })))
                                .map_err(|e| e.to_string().into_boxed_str())?;
                        }
                    }
                }
                nixe_gpu::ordered::Read::Control(Control::Memory(job, reply)) => {
                    job.apply(|space, address, size, compute| {
                        renderer.invalidate_space(space, address, size);
                        if compute && let Some(scheduler) = schedulers.get_mut(&space) {
                            scheduler.invalidate(address, size)?;
                        }
                        Ok(())
                    })?;
                    reply.finish();
                }
                nixe_gpu::ordered::Read::Control(Control::Retire(channel, reply)) => {
                    channels.remove(&channel);
                    renderer.close_channel(channel);
                    reply.finish();
                }
                nixe_gpu::ordered::Read::Words(mut words) => {
                    let job = current.as_ref().ok_or("methods outside submission")?;
                    let channel = channels.entry(job.channel).or_default();
                    let scheduler = schedulers.entry(job.space_id).or_default();
                    let mut host = Back {
                        renderer,
                        source,
                        memory: Memory {
                            machine: &machine,
                            space: &job.space,
                            stop,
                            waiters,
                        },
                        clock: &clock,
                    };
                    ampere::ordered::consume(&mut words, &mut source, |source| {
                        host.source = source;
                        channel.execute(source, scheduler, &mut host)
                    })?;
                }
            }
            if endpoint.requested.load(Ordering::Acquire) {
                endpoint.progress(renderer)?;
            }
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
        }
        if !input.wait(
            || stop.load(Ordering::Acquire),
            || endpoint.requested.load(Ordering::Acquire),
            || {
                renderer
                    .flush_deferred()
                    .map_err(|e| e.to_string().into_boxed_str())?;
                endpoint.progress(renderer)
            },
        )? {
            break;
        }
    }
    Ok(())
}

/// A service-visible memory job travels through the same ingress as commands.
/// Captured mappings/sharing leases survive until ordered mutation completes.
pub(crate) enum MemoryJob {
    Remap {
        space: Arc<Mutex<GpuAddressSpace>>,
        tiles: Vec<super::gpu::TileMapping>,
    },
    NativeRegion {
        space: Arc<Mutex<GpuAddressSpace>>,
        address: u64,
        bytes: u64,
        account: Option<nixe_memory::MemoryAccount>,
    },
    Kind {
        space: Arc<Mutex<GpuAddressSpace>>,
        address: u64,
        bytes: u64,
        kind: u8,
    },
    Map {
        space: Arc<Mutex<GpuAddressSpace>>,
        address: u64,
        bytes: u64,
        handle: u32,
        source: Arc<nixe_runtime::memory::GpuMemoryLease>,
        offset: u64,
        kind: u8,
    },
    Retire {
        spaces: Vec<Arc<Mutex<GpuAddressSpace>>>,
        handle: u32,
        _source: Arc<nixe_runtime::memory::GpuMemoryLease>,
    },
}
impl MemoryJob {
    pub fn apply(
        self,
        mut invalidate: impl FnMut(u64, u64, u64, bool) -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Remap { space, tiles } => {
                let mut space = space.lock().unwrap();
                for tile in tiles {
                    invalidate(space.image_space_id(), tile.address, tile.bytes, true)?;
                    space
                        .remap(tile)
                        .map_err(|e| e.to_string().into_boxed_str())?;
                }
                Ok(())
            }
            Self::NativeRegion {
                space,
                address,
                bytes,
                account,
            } => {
                let mut space = space.lock().unwrap();
                invalidate(space.image_space_id(), address, bytes, true)?;
                if let Some(account) = account {
                    space.create_region(address, bytes, account)
                } else {
                    space.destroy_region(address, bytes)
                }
                .map_err(|e| e.to_string().into())
            }
            Self::Kind {
                space,
                address,
                bytes,
                kind,
            } => {
                let mut space = space.lock().unwrap();
                // Kind has no compute scheduler effect; invalidate image selection.
                invalidate(space.image_space_id(), address, bytes, false)?;
                space
                    .modify_kind(address, bytes, kind)
                    .map_err(|e| e.to_string().into())
            }
            Self::Map {
                space,
                address,
                bytes,
                handle,
                source,
                offset,
                kind,
            } => {
                let mut space = space.lock().unwrap();
                invalidate(space.image_space_id(), address, bytes, true)?;
                space
                    .map(address, bytes, handle, source, offset, kind)
                    .map_err(|e| e.to_string().into())
            }
            Self::Retire { spaces, handle, .. } => {
                for space in spaces {
                    let mut space = space.lock().unwrap();
                    let ranges: Vec<_> = space
                        .mappings
                        .iter()
                        .filter(|(_, m)| m.memory_handle == Some(handle))
                        .map(|(&a, m)| (a, m.end - a))
                        .collect();
                    for (address, bytes) in ranges {
                        invalidate(space.image_space_id(), address, bytes, true)?;
                        space
                            .unmap(address, bytes)
                            .map_err(|e| e.to_string().into_boxed_str())?;
                    }
                }
                Ok(())
            }
        }
    }
}
