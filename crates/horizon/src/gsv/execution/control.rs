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
    pub requested: AtomicBool,
    submit: AtomicBool,
    thread: std::sync::OnceLock<thread::Thread>,
}
impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint").finish_non_exhaustive()
    }
}
impl Endpoint {
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
    pub fn progress<E: ampere::Rasterizer>(&self, renderer: &mut E) -> Result<()> {
        self.requested.store(false, Ordering::Release);
        loop {
            let Some(request) = self.inbox.lock().unwrap().requests.pop_front() else {
                break;
            };
            match request {
                Request::Access(request) => renderer.queue_device_access(request),
                Request::Present(frame, reply) => {
                    let image = renderer.composite(
                        frame.image.clone(),
                        frame.interval,
                        nixe_gpu::frame_metrics::FramePacing {
                            queued: frame.queued,
                            vsync: frame.queued,
                        },
                    )?;
                    renderer.submit(Some(Box::new(move |result| {
                        result?;
                        drop(image);
                        drop(frame);
                        reply.finish();
                        Ok(())
                    })))
                }
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
