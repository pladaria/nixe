//! Packet/MME owner. Native resources are accessed only through ordered requests.
use super::*;
use nixe_gpu_maxwell::{
    MaxwellCommandFetch, MaxwellCommandProcessor, MaxwellFrontendDispatchError, MaxwellMethodSink,
    MaxwellSubmissionExecutionError,
};
use std::collections::BTreeMap;
#[cfg(feature = "performance-counters")]
use std::time::Duration;
use std::time::Instant;

#[cfg(feature = "performance-counters")]
#[derive(Clone, Copy, Default)]
pub(super) struct Statistics {
    pub active_ns: u64,
    pub mme_ns: u64,
    pub read_ns: u64,
    pub read_count: u64,
    pub queue_ns: u64,
    pub batches: u64,
    pub words: u64,
    pub occupancy: u64,
    pub peak: u64,
    pub entries: u64,
}
struct Link<'a> {
    sender: nixe_gpu::ordered::Sender<GpuMessage>,
    receiver: &'a mpsc::Receiver<GpuMessage>,
    budget: &'a GpuWorkBudget,
    memories:
        BTreeMap<nixe_gpu_maxwell::MaxwellAddressSpaceId, nixe_gpu_maxwell::MaxwellCommandMemory>,
    current_memory: Option<(
        nixe_gpu_maxwell::MaxwellAddressSpaceId,
        nixe_gpu_maxwell::MaxwellCommandMemory,
    )>,
    writer_position: u64,
    sync_position: u64,
    synced_writer_position: u64,
    host_position: u64,
    synced_host_position: u64,
    memory_position: u64,
    #[cfg(feature = "performance-counters")]
    stats: Statistics,
    current: FrontendSubmissionId,
    selected: Option<FrontendSubmissionId>,
    fetch_state: FetchState,
    read_reply: CommandReplySender,
    read_response: mpsc::Receiver<(FrontendSubmissionId, CommandReadReply)>,
    replies: BTreeMap<FrontendSubmissionId, CommandReadReply>,
    acquire: Option<(u64, bool)>,
}
#[derive(Default)]
enum FetchState {
    #[default]
    New,
    Native {
        tick: Option<u64>,
        sync: Option<u64>,
    },
    Owner,
}
struct Job {
    dispatch: MaxwellFrontendDispatch,
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    space: nixe_gpu_maxwell::MaxwellAddressSpaceId,
    stream: nixe_gpu_maxwell::MaxwellCommandStream,
    fetch: FetchState,
    acquire: Option<(u64, bool)>,
    #[cfg(feature = "performance-counters")]
    waiting_since: Option<Instant>,
}
struct Channel {
    processor: MaxwellCommandProcessor,
    deliveries: VecDeque<GpuQueuedSubmission>,
    job: Option<Job>,
    storage: Vec<u8>,
    closing: bool,
}
impl Channel {
    fn pending(&self) -> bool {
        self.job.is_some() || !self.deliveries.is_empty()
    }
    fn start(&mut self, link: &mut Link<'_>) -> Result<(), MaxwellSubmissionExecutionError> {
        if self.job.is_some() {
            return Ok(());
        }
        let mut queued = self.deliveries.pop_front().expect("scheduled channel");
        queued.started = Some(Instant::now());
        let submission = &queued.submission;
        let dispatch = submission.dispatch.clone();
        let stream = nixe_gpu_maxwell::MaxwellCommandStream::new(
            &dispatch,
            submission.address_space,
            std::mem::take(&mut self.storage),
        );
        self.job = Some(Job {
            dispatch,
            descriptor: submission.descriptor,
            request: submission.request,
            space: submission.address_space,
            stream,
            fetch: FetchState::New,
            acquire: None,
            #[cfg(feature = "performance-counters")]
            waiting_since: None,
        });
        link.send(GpuMessage::Submission(queued))
    }
}
#[derive(Default)]
struct SpaceWork {
    deliveries: usize,
    closing: bool,
}
impl Link<'_> {
    fn close_space(
        &mut self,
        id: nixe_gpu_maxwell::MaxwellAddressSpaceId,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        if self
            .current_memory
            .as_ref()
            .is_some_and(|(space, _)| *space == id)
        {
            self.current_memory = None;
        } else {
            self.memories.remove(&id);
        }
        self.send(GpuMessage::CloseAddressSpace(id))
    }
    fn select_memory(&mut self, space: nixe_gpu_maxwell::MaxwellAddressSpaceId) {
        if self
            .current_memory
            .as_ref()
            .is_some_and(|(id, _)| *id == space)
        {
            return;
        }
        let memory = self
            .memories
            .remove(&space)
            .expect("registered command memory");
        if let Some((id, previous)) = self.current_memory.replace((space, memory)) {
            self.memories.insert(id, previous);
        }
    }
    fn send(&mut self, message: GpuMessage) -> Result<(), MaxwellSubmissionExecutionError> {
        let writes_memory = matches!(
            message,
            GpuMessage::MemoryAccess(_)
                | GpuMessage::GuestMemoryReleased { .. }
                | GpuMessage::PhysicalRetired { .. }
                | GpuMessage::ConvertNv12 { .. }
        );
        self.sender.send(message).map_err(|_| {
            MaxwellSubmissionExecutionError::Native(
                "graphics preparation owner disconnected".into(),
            )
        })?;
        if writes_memory {
            self.memory_position = self.sender.position();
        }
        Ok(())
    }
    fn select(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
        if self.selected != Some(self.current) {
            self.send(GpuMessage::SelectSubmission(self.current))?;
            self.selected = Some(self.current);
        }
        Ok(())
    }
    #[cfg(feature = "performance-counters")]
    fn snapshot(&self) -> Statistics {
        Statistics {
            queue_ns: self.sender.wait.as_nanos() as u64,
            occupancy: self.sender.occupancy,
            peak: self.sender.peak,
            entries: self.sender.publications,
            batches: self.sender.batches,
            ..self.stats
        }
    }
}
impl MaxwellMethodSink for Link<'_> {
    fn acquire(
        &mut self,
        source: nixe_gpu_maxwell::MaxwellMethodSource,
        address: u64,
        payload: u32,
        operation: u8,
    ) -> Result<bool, MaxwellFrontendDispatchError> {
        let (target, pending) = *self.acquire.get_or_insert((self.sender.position(), false));
        if !self.sender.consumed(target) {
            return Ok(false);
        }
        let word = if pending {
            while let Ok((id, result)) = self.read_response.try_recv() {
                self.replies.insert(id, result);
            }
            let Some(reply) = self.replies.remove(&self.current) else {
                return Ok(false);
            };
            let CommandReadReply::Semaphore(result) = reply else {
                unreachable!("semaphore reply")
            };
            result
        } else {
            let (space, memory) = self
                .current_memory
                .as_ref()
                .expect("active semaphore space");
            match memory.read_semaphore(address) {
                Ok(Some(word)) => Ok(word),
                Ok(None) => {
                    self.send(GpuMessage::SemaphoreRead {
                        frontend: self.current,
                        space: *space,
                        address,
                        reply: self.read_reply.clone(),
                    })
                    .map_err(|e| MaxwellFrontendDispatchError::Execution(Box::new(e)))?;
                    self.acquire = Some((target, true));
                    return Ok(false);
                }
                Err(error) => Err(error),
            }
        }
        .map_err(|error| {
            MaxwellFrontendDispatchError::Execution(Box::new(
                MaxwellSubmissionExecutionError::InlineAddress { source, error },
            ))
        })?;
        let matches = match operation {
            1 => word == payload,
            4 => word >= payload,
            8 => word & payload != 0,
            _ => unreachable!("admitted acquire"),
        };
        if matches {
            // Successful acquires can expose CPU writes even when the read needed no back request.
            self.current_memory
                .as_ref()
                .expect("active semaphore space")
                .1
                .sync_guest_memory();
        }
        self.acquire = if matches {
            None
        } else {
            Some((self.sender.position(), false))
        };
        Ok(matches)
    }
    fn write(
        &mut self,
        prefix: &[u64],
        bytes: &[u8],
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        #[cfg(feature = "performance-counters")]
        {
            self.stats.words += prefix.len() as u64 + bytes.len().div_ceil(8) as u64;
        }
        self.select()?;
        self.sender.write(prefix, bytes).map_err(|_| {
            MaxwellSubmissionExecutionError::Native(
                "graphics preparation owner disconnected".into(),
            )
        })
    }
    fn flush(&mut self) {
        self.sender.flush();
    }
    fn note_sync(&mut self) {
        self.sync_position = self.sender.position();
        self.synced_writer_position = self.writer_position;
        self.synced_host_position = self.host_position;
    }
    fn note_gpu_writer(&mut self) {
        self.writer_position = self.sender.position();
    }
    fn note_host_method(&mut self) {
        self.host_position = self.sender.position();
    }
    fn is_stopped(&self) -> bool {
        self.budget.is_stopped()
    }
    #[cfg(feature = "performance-counters")]
    fn blocked_time(&self) -> Duration {
        self.sender.wait + Duration::from_nanos(self.stats.read_ns)
    }
    #[cfg(feature = "performance-counters")]
    fn mme_time(&mut self, active: Duration) {
        self.stats.mme_ns += active.as_nanos() as u64;
    }
    fn fetch(
        &mut self,
        request: &mut MaxwellCommandFetch,
    ) -> Result<bool, MaxwellFrontendDispatchError> {
        #[cfg(feature = "performance-counters")]
        let start = Instant::now();
        let result = self.fetch_commands(request);
        #[cfg(feature = "performance-counters")]
        {
            self.stats.read_ns += start.elapsed().as_nanos() as u64;
        }
        result
    }
}
impl Link<'_> {
    fn fetch_commands(
        &mut self,
        request: &mut MaxwellCommandFetch,
    ) -> Result<bool, MaxwellFrontendDispatchError> {
        if matches!(self.fetch_state, FetchState::Owner) {
            while let Ok((id, result)) = self.read_response.try_recv() {
                self.replies.insert(id, result);
            }
            let Some(result) = self.replies.remove(&self.current) else {
                return Ok(false);
            };
            let CommandReadReply::Commands(result) = result else {
                unreachable!("command reply")
            };
            *request = result?;
            self.fetch_state = FetchState::New;
            return Ok(true);
        }
        let (space, memory) = self
            .current_memory
            .as_mut()
            .expect("active command address space");
        debug_assert_eq!(*space, request.space);
        if matches!(self.fetch_state, FetchState::New) {
            if request.entries[request.range.start].sync()
                == nixe_gpu_maxwell::MaxwellGpfifoSyncMode::Wait
            {
                self.sync_position = self.sender.position();
                self.synced_writer_position = self.writer_position;
                self.synced_host_position = self.host_position;
            }
            // SYNC_WAIT orders Host execution, not completion of every engine
            // method. Only producer dependencies constrain engine overlap.
            // https://github.com/NVIDIA/open-gpu-doc/blob/ab27fc22db5de0d02a4cabe08e555663b62db4d4/manuals/volta/gv100/dev_pbdma.ref.txt#L104-L138
            let mut target = self
                .synced_writer_position
                .max(self.synced_host_position)
                .max(self.memory_position);
            // No producer can be discovered by walking pages if all known
            // compute/copy methods have already run. The reader still registers
            // physical command pages and checks their actual coherence below.
            let pending_writer = !self.sender.consumed(self.writer_position);
            let written = pending_writer && memory.gpu_written(request);
            if written {
                target = target.max(self.writer_position);
            }
            let waited = !self.sender.consumed(target);
            self.sender.wait_until(target).map_err(|_| {
                MaxwellFrontendDispatchError::Execution(Box::new(
                    MaxwellSubmissionExecutionError::Native(
                        "graphics preparation owner disconnected".into(),
                    ),
                ))
            })?;
            let remapped = memory.mappings_changed();
            if remapped {
                memory.refresh_mappings();
            }
            // A synchronized producer can write a page for the first time while
            // we wait. Recheck only when there is still a later producer to wait
            // for; a second walk without either condition proves nothing new.
            if !written
                && (waited || remapped)
                && !self.sender.consumed(self.writer_position)
                && memory.gpu_written(request)
            {
                target = target.max(self.writer_position);
                self.sender.wait_until(target).map_err(|_| {
                    MaxwellFrontendDispatchError::Execution(Box::new(
                        MaxwellSubmissionExecutionError::Native(
                            "graphics preparation owner disconnected".into(),
                        ),
                    ))
                })?;
            }
            // Once its producers ran, a suspended read never acquires dependencies
            // on unrelated work submitted by another channel in the meantime.
            self.fetch_state = FetchState::Native {
                tick: None,
                sync: (target < self.sync_position).then_some(self.sync_position),
            };
        }
        let FetchState::Native { mut tick, mut sync } = self.fetch_state else {
            unreachable!("native command read")
        };
        if let Some(pending) = tick {
            if !memory.command_tick_done(pending).map_err(|error| {
                MaxwellFrontendDispatchError::Execution(Box::new(
                    MaxwellSubmissionExecutionError::Native(error.to_string().into()),
                ))
            })? {
                return Ok(false);
            }
            tick = None;
            self.fetch_state = FetchState::Native { tick, sync };
        }
        loop {
            match memory
                .read_clean(request, sync.is_none())
                .map_err(|e| MaxwellFrontendDispatchError::CommandSource(Box::new(e)))?
            {
                nixe_gpu_maxwell::CommandRead::Copied => {
                    self.fetch_state = FetchState::New;
                    return Ok(true);
                }
                nixe_gpu_maxwell::CommandRead::Pending(tick) => {
                    self.fetch_state = FetchState::Native {
                        tick: Some(tick),
                        sync,
                    };
                    return Ok(false);
                }
                nixe_gpu_maxwell::CommandRead::NeedsSync => {
                    self.sender
                        .wait_until(sync.take().expect("pending host sync"))
                        .map_err(|_| {
                            MaxwellFrontendDispatchError::Execution(Box::new(
                                MaxwellSubmissionExecutionError::Native(
                                    "graphics preparation owner disconnected".into(),
                                ),
                            ))
                        })?;
                    self.fetch_state = FetchState::Native { tick, sync };
                }
                nixe_gpu_maxwell::CommandRead::Synced => {
                    unreachable!("RAM is copied by the command reader")
                }
                nixe_gpu_maxwell::CommandRead::Unknown => {
                    self.send(GpuMessage::CommandFetch {
                        request: request.take_storage(),
                        reply: self.read_reply.clone(),
                    })
                    .map_err(|e| MaxwellFrontendDispatchError::Execution(Box::new(e)))?;
                    self.fetch_state = FetchState::Owner;
                    #[cfg(feature = "performance-counters")]
                    {
                        self.stats.read_count += 1;
                    }
                    return Ok(false);
                }
            }
        }
    }
}

fn memory_error(error: impl Display) -> Arc<MemoryError> {
    Arc::new(MemoryError::Execution(error.to_string().into()))
}

pub(super) fn run(
    receiver: mpsc::Receiver<GpuMessage>,
    sender: nixe_gpu::ordered::Sender<GpuMessage>,
    budget: &GpuWorkBudget,
    mme_jit: bool,
    failure: &Mutex<Option<GpuExecutorFailure>>,
) -> Result<(), GpuExecutorFailure> {
    budget.command_thread.get_or_init(std::thread::current);
    let mut channels = BTreeMap::<nixe_gpu_maxwell::MaxwellChannelId, Channel>::new();
    let mut spaces = BTreeMap::<nixe_gpu_maxwell::MaxwellAddressSpaceId, SpaceWork>::new();
    let mut pending = VecDeque::new();
    let (read_reply, read_response) = mpsc::channel();
    let mut link = Link {
        sender,
        receiver: &receiver,
        budget,
        memories: BTreeMap::new(),
        current_memory: None,
        writer_position: 0,
        sync_position: 0,
        synced_writer_position: 0,
        host_position: 0,
        synced_host_position: 0,
        memory_position: 0,
        current: FrontendSubmissionId::new(0),
        selected: None,
        fetch_state: FetchState::New,
        #[cfg(feature = "performance-counters")]
        stats: Default::default(),
        read_reply,
        read_response,
        replies: BTreeMap::new(),
        acquire: None,
    };
    let native_error = |error: MaxwellSubmissionExecutionError| GpuExecutorFailure {
        frontend: FrontendSubmissionId::new(0),
        detail: error.to_string().into(),
        memory_cause: None,
        boundary: None,
    };
    let mut waiting_since = Instant::now();
    let result: Result<(), GpuExecutorFailure> = (|| loop {
        if budget.is_stopped() {
            return Ok(());
        }
        let message = match receiver.try_recv() {
            Ok(message) => Some(message),
            Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            Err(mpsc::TryRecvError::Empty) => None,
        };
        let mut progressed = message.is_some();
        if let Some(message) = message {
            match message {
                GpuMessage::CreateAddressSpace(space) => {
                    spaces.insert(space.id(), SpaceWork::default());
                    link.memories.insert(
                        space.id(),
                        nixe_gpu_maxwell::MaxwellCommandMemory::new(space.memory().clone()),
                    );
                    link.send(GpuMessage::CreateAddressSpace(space))
                        .map_err(native_error)?;
                }
                message @ GpuMessage::AddressSpaceIoctl { .. } => {
                    link.send(message).map_err(native_error)?;
                    link.sender
                        .wait_until(link.sender.position())
                        .map_err(|_| {
                            native_error(MaxwellSubmissionExecutionError::Native(
                                "address-space owner disconnected".into(),
                            ))
                        })?;
                }
                GpuMessage::CloseAddressSpace(id) => {
                    let space = spaces.get_mut(&id).expect("registered address space");
                    if space.deliveries != 0 {
                        space.closing = true;
                    } else {
                        spaces.remove(&id);
                        link.close_space(id).map_err(native_error)?;
                    }
                }
                GpuMessage::CreateChannel(configuration) => {
                    channels.insert(
                        configuration.id(),
                        Channel {
                            processor: MaxwellCommandProcessor::new(configuration, mme_jit),
                            deliveries: VecDeque::new(),
                            job: None,
                            storage: Vec::new(),
                            closing: false,
                        },
                    );
                    link.send(GpuMessage::CreateChannel(configuration))
                        .map_err(native_error)?;
                }
                GpuMessage::CloseChannel(id) => {
                    let channel = channels.get_mut(&id).expect("registered command channel");
                    if channel.pending() {
                        channel.closing = true;
                    } else {
                        channels.remove(&id);
                        link.send(GpuMessage::CloseChannel(id))
                            .map_err(native_error)?;
                    }
                }
                GpuMessage::Submission(queued) => {
                    let submission = &queued.submission;
                    spaces
                        .get_mut(&submission.address_space)
                        .expect("registered address space")
                        .deliveries += 1;
                    let id = submission.channel.id();
                    let channel = channels.get_mut(&id).expect("registered command channel");
                    if !channel.pending() {
                        pending.push_back(id);
                    }
                    channel.deliveries.push_back(queued);
                }
                message => link.send(message).map_err(native_error)?,
            }
        }
        for _ in 0..pending.len() {
            let channel_id = pending.pop_front().unwrap();
            let channel = channels.get_mut(&channel_id).unwrap();
            channel.start(&mut link).map_err(native_error)?;
            let job = channel.job.as_mut().unwrap();
            let id = job.dispatch.scheduled().frontend();
            link.current = id;
            link.select_memory(job.space);
            link.fetch_state = std::mem::take(&mut job.fetch);
            link.acquire = job.acquire.take();
            let before = link.sender.position();
            #[cfg(feature = "performance-counters")]
            if let Some(start) = job.waiting_since.take() {
                link.stats.read_ns += start.elapsed().as_nanos() as u64;
            }
            #[cfg(feature = "performance-counters")]
            let (start, blocked) = (Instant::now(), link.blocked_time());
            let done = job
                .stream
                .run(&mut channel.processor, &mut link)
                .map_err(|failure| GpuExecutorFailure {
                    memory_cause: None,
                    frontend: id,
                    detail: failure.to_string().into(),
                    boundary: Some((
                        job.descriptor,
                        job.request,
                        Box::new(MaxwellFrontendDispatchBoundary {
                            dispatch: Box::new(job.dispatch.clone()),
                            failure,
                        }),
                    )),
                })?;
            #[cfg(feature = "performance-counters")]
            {
                link.stats.active_ns += start
                    .elapsed()
                    .saturating_sub(link.blocked_time().saturating_sub(blocked))
                    .as_nanos() as u64;
            }
            progressed |= done || link.sender.position() != before;
            if done {
                link.select().map_err(native_error)?;
                #[cfg(feature = "performance-counters")]
                link.send(GpuMessage::FrontMetrics(link.snapshot()))
                    .map_err(native_error)?;
                link.send(GpuMessage::EndSubmission).map_err(native_error)?;
                link.selected = None;
                let job = channel.job.take().unwrap();
                channel.storage = job.stream.into_storage();
                let space = spaces.get_mut(&job.space).unwrap();
                space.deliveries -= 1;
                if space.deliveries == 0 && space.closing {
                    spaces.remove(&job.space);
                    link.close_space(job.space).map_err(native_error)?;
                }
            } else {
                job.fetch = std::mem::take(&mut link.fetch_state);
                job.acquire = link.acquire.take();
                #[cfg(feature = "performance-counters")]
                {
                    job.waiting_since = Some(Instant::now());
                }
            }
            if channel.pending() {
                pending.push_back(channel_id);
            } else if channel.closing {
                channels.remove(&channel_id);
                link.send(GpuMessage::CloseChannel(channel_id))
                    .map_err(native_error)?;
            }
        }
        if progressed {
            waiting_since = Instant::now();
            continue;
        }
        if pending.is_empty() {
            let start = Instant::now();
            // An ingress sender records a token after publishing its message.
            std::thread::park();
            link.send(GpuMessage::FrontIdle {
                start,
                end: Instant::now(),
            })
            .map_err(native_error)?;
        } else {
            // GPU completion and CPU semaphore writes do not have an ingress
            // message. Poll pending channels only, with bounded idle backoff.
            if budget
                .progress_requested
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
                && link.sender.needs_idle_hook()
            {
                budget.request_hook(GpuWorkBudget::HOST_PROGRESS);
            }
            if waiting_since.elapsed() < std::time::Duration::from_micros(500) {
                std::thread::yield_now();
            } else {
                std::thread::park_timeout(
                    if waiting_since.elapsed() < std::time::Duration::from_secs(2) {
                        std::time::Duration::from_micros(50)
                    } else {
                        std::time::Duration::from_millis(1)
                    },
                );
            }
        }
    })();
    if let Err(error) = &result
        && !budget.is_stopped()
    {
        failure.lock().unwrap().get_or_insert_with(|| error.clone());
        budget.stop();
    }
    result
}

impl Drop for Link<'_> {
    fn drop(&mut self) {
        let cause = memory_error("GPU command processor stopped");
        for message in self.receiver.try_iter() {
            match message {
                GpuMessage::MemoryAccess(request) => request.finish(Err(Arc::clone(&cause))),
                GpuMessage::ConvertNv12 { reply, .. } => {
                    let _ = reply.send(Err("GPU command processor stopped".into()));
                }
                GpuMessage::RequestComposite { registration, .. } => {
                    registration.complete(Err("GPU command processor stopped".into()))
                }
                _ => {}
            }
        }
    }
}
