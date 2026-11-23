//! Host control reaches the preparation owner independently of a frontend blocked
//! on a coherent read. Presentation enters after its producer fence has completed;
//! guest methods still travel exclusively through the ordered-method ring.
use super::*;
use nixe_memory::{
    AllocationId, DeviceAccessEndpoint, DeviceAccessRequest, Extent, MachineId, MemoryError,
};
use std::collections::VecDeque;

enum Request {
    Access(DeviceAccessRequest),
    Present(Box<crate::gsv::display::Frame>, Reply),
    Nvn(nvn::Request),
    Released(MachineId, AllocationId),
    Retired(MachineId, AllocationId, Box<[Extent]>),
    Pressure(MachineId, u64),
}
#[derive(Default)]
struct Inbox {
    stopped: bool,
    requests: VecDeque<Request>,
}
#[derive(Default)]
pub(super) struct Endpoint {
    inbox: Mutex<Inbox>,
    presenter: Option<nixe_video::FrameMailbox>,
    frame_sequence: std::sync::atomic::AtomicU64,
    pub requested: AtomicBool,
    submit: AtomicBool,
    thread: std::sync::OnceLock<thread::Thread>,
    nvn_budget: Arc<AtomicUsize>,
}
impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint").finish_non_exhaustive()
    }
}
impl Endpoint {
    pub fn new(presenter: Option<nixe_video::FrameMailbox>) -> Self {
        Self {
            presenter,
            ..Self::default()
        }
    }
    pub fn wake(&self) {
        self.requested.store(true, Ordering::Release);
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
    }
    fn send(&self, request: Request) {
        let mut inbox = self.inbox.lock().unwrap();
        if inbox.stopped {
            drop(inbox);
            if let Request::Access(request) = request {
                request.finish(Err(Arc::new(MemoryError::Execution(
                    "native GPU owner stopped".into(),
                ))));
            }
            return;
        }
        inbox.requests.push_back(request);
        drop(inbox);
        self.wake();
    }
    pub fn present(&self, frame: crate::gsv::display::Frame) -> Completion {
        let (reply, completion) = Completion::new();
        self.send(Request::Present(Box::new(frame), reply));
        completion
    }
    pub(super) fn nvn_work(&self, work: crate::nvn_backend::Work) -> Result<()> {
        let (request, reply) = nvn::Request::new(work, &self.nvn_budget);
        let mut inbox = self.inbox.lock().unwrap();
        if inbox.stopped {
            return Err("native GPU owner stopped".into());
        }
        inbox.requests.push_back(Request::Nvn(request));
        drop(inbox);
        self.wake();
        if let Some(reply) = reply {
            reply.recv().map_err(|_| "native GPU owner stopped")?
        } else {
            Ok(())
        }
    }
    pub(super) fn nvn_present(
        &self,
        queue: u64,
        texture: nixe_nvn::Texture,
        interval: u32,
        flip: bool,
        crop: [u32; 4],
        receipt: nixe_nvn::display::PresentReceipt,
    ) -> Result<()> {
        let sequence = self.frame_sequence.fetch_add(1, Ordering::Relaxed);
        self.nvn_work(crate::nvn_backend::Work::Present {
            queue,
            texture,
            interval,
            flip,
            crop,
            mailbox: self.presenter.clone(),
            sequence,
            receipt: Some(receipt),
        })
    }
    pub fn stop(&self) {
        let mut inbox = self.inbox.lock().unwrap();
        inbox.stopped = true;
        let requests = std::mem::take(&mut inbox.requests);
        drop(inbox);
        for request in requests {
            if let Request::Access(request) = request {
                request.finish(Err(Arc::new(MemoryError::Execution(
                    "native GPU owner stopped".into(),
                ))));
            }
        }
        self.wake();
    }
    pub fn bind<E: ampere::Rasterizer>(self: &Arc<Self>, renderer: &mut E) -> Result<()> {
        let _ = self.thread.set(thread::current());
        let wake = self.clone();
        let submit = self.clone();
        renderer
            .bind_host_wakeup(
                Arc::new(move || wake.wake()),
                Arc::new(move || {
                    submit.submit.store(true, Ordering::Release);
                    submit.wake();
                }),
            )
            .map_err(|e| e.to_string().into())
    }
    pub fn progress<E: ampere::Rasterizer + nixe_gpu::GpuPresentation>(
        &self,
        renderer: &mut E,
    ) -> Result<()> {
        self.requested.store(false, Ordering::Release);
        loop {
            let Some(request) = self.inbox.lock().unwrap().requests.pop_front() else {
                break;
            };
            match request {
                Request::Access(request) => renderer.queue_device_access(request),
                Request::Present(frame, reply) => {
                    let image = ampere::Rasterizer::composite(
                        renderer,
                        frame.image.clone(),
                        frame.interval,
                    )?;
                    if let Some(presenter) = &self.presenter {
                        let sequence = self.frame_sequence.fetch_add(1, Ordering::Relaxed);
                        presenter.publish(Arc::new(nixe_video::PresentationFrame::new(
                            image, sequence,
                        )));
                    }
                    // Composition retains its source through submission. The
                    // published host image owns independent native storage.
                    renderer.submit(Some(Box::new(move |result| {
                        result?;
                        drop(frame);
                        reply.finish();
                        Ok(())
                    })))
                }
                Request::Nvn(request) => request.run(renderer),
                Request::Released(machine, allocation) => {
                    renderer.release_guest_memory(machine, allocation)
                }
                Request::Retired(machine, allocation, extents) => {
                    renderer.retire_physical_memory(machine, allocation, &extents)
                }
                Request::Pressure(machine, bytes) => renderer.allocation_pressure(machine, bytes),
            }
            .map_err(|e| e.to_string().into_boxed_str())?;
        }
        if self.submit.swap(false, Ordering::AcqRel) {
            renderer
                .flush()
                .map_err(|e| e.to_string().into_boxed_str())?;
        }
        renderer.progress().map_err(|e| e.to_string().into())
    }
}
impl DeviceAccessEndpoint for Endpoint {
    fn submit(&self, request: DeviceAccessRequest) {
        self.send(Request::Access(request));
    }
    fn cancel(&self, _: nixe_memory::AccessRequestId) {
        self.wake();
    }
    fn allocation_released(&self, machine: MachineId, allocation: AllocationId) {
        self.send(Request::Released(machine, allocation));
    }
    fn physical_retired(
        &self,
        machine: MachineId,
        allocation: AllocationId,
        extents: Box<[Extent]>,
    ) {
        self.send(Request::Retired(machine, allocation, extents));
    }
    fn allocation_pressure(&self, machine: MachineId, bytes: u64) {
        self.send(Request::Pressure(machine, bytes));
    }
}
