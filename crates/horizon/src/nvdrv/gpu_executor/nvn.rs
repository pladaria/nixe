use super::*;
use crate::nvn_backend::Work;
use nixe_gpu::nvn_fence_action;
use nixe_nvn::{Fence, Submission, Texture};
pub(super) struct Request {
    work: Work,
    reply: Option<mpsc::SyncSender<nixe_nvn::Result<()>>>,
    permit: Option<GpuFrontendPermit>,
}
impl Request {
    pub(super) fn run(
        self,
        backend: &mut impl nixe_gpu::GpuPresentation,
    ) -> Result<(), nixe_gpu::BackendRuntimeError> {
        self.work.run(backend, self.permit, self.reply)
    }
}
struct Client {
    backend: std::sync::Weak<NvDrvGpuExecutor>,
    mailbox: Option<nixe_video::FrameMailbox>,
    sequence: std::sync::atomic::AtomicU64,
}
pub(in crate::nvdrv) fn client(
    backend: &Arc<NvDrvGpuExecutor>,
    mailbox: Option<nixe_video::FrameMailbox>,
) -> Arc<dyn nixe_nvn::Backend> {
    Arc::new(Client {
        backend: Arc::downgrade(backend),
        mailbox,
        sequence: 0.into(),
    })
}
impl Client {
    fn execute(&self, work: Work) -> nixe_nvn::Result<()> {
        let backend = self.backend.upgrade().ok_or("GPU owner stopped")?;
        backend.require_healthy().map_err(|e| e.to_string())?;
        // Bound queued work, including small control packets. When full, let
        // this caller wait behind its own request without overtaking older lists.
        let charge = work.allocation_bytes().max(MAX_GPU_INGRESS_BYTES / 64);
        let permit = match backend.budget.reserve(charge).ok_or("GPU owner stopped")? {
            GpuSubmissionAdmission::Ready(permit) => Some(permit),
            GpuSubmissionAdmission::Pending(_) => None,
        };
        let wait = permit.is_none() || matches!(work, Work::Drain);
        let (reply, done) = if wait {
            let (tx, rx) = mpsc::sync_channel(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        backend
            .sender
            .lock()
            .unwrap()
            .as_ref()
            .ok_or("GPU owner stopped")?
            .send(GpuMessage::Nvn(Request {
                work,
                reply,
                permit,
            }))
            .map_err(|_| "GPU owner stopped")?;
        if let Some(done) = done {
            done.recv()
                .map_err(|_| "GPU owner stopped before NVN execution")?
        } else {
            Ok(())
        }
    }
}
impl nixe_nvn::Backend for Client {
    fn submit(&self, queue: u64, ops: Submission) -> nixe_nvn::Result<()> {
        self.execute(Work::Submit(queue, ops))
    }
    fn drain(&self) -> nixe_nvn::Result<()> {
        self.execute(Work::Drain)
    }
    fn finalize_queue(&self, queue: u64) -> nixe_nvn::Result<()> {
        self.execute(Work::FinalizeQueue(queue))
    }
    fn acquire_fence(&self, f: Arc<Fence>) -> nixe_nvn::Result<()> {
        self.execute(Work::AcquireFence(nvn_fence_action(f)))
    }
    fn fence(&self, queue: u64, f: Arc<Fence>, flush_for_cpu: bool) -> nixe_nvn::Result<()> {
        self.execute(Work::Fence(queue, nvn_fence_action(f), flush_for_cpu))
    }
    fn flush(&self, queue: u64) -> nixe_nvn::Result<()> {
        self.execute(Work::Flush(queue))
    }
    fn present(
        &self,
        queue: u64,
        texture: Texture,
        interval: u32,
        flip_vertical: bool,
        crop: [u32; 4],
    ) -> nixe_nvn::Result<()> {
        let seq = self
            .sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        self.execute(Work::Present {
            queue,
            texture,
            interval,
            flip: flip_vertical,
            crop,
            mailbox: self.mailbox.clone(),
            sequence: seq,
            receipt: None,
        })
    }
}
