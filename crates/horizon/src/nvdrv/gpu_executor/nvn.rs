use super::*;
use nixe_nvn::{Fence, Submission, Texture};
pub(super) enum Work {
    Submit(u64, Submission),
    Flush,
    Drain,
    FinalizeQueue(u64),
    Fence(nixe_gpu::FenceAction),
    AcquireFence(nixe_gpu::FenceAction),
    Present(Texture, u32, bool, Option<nixe_video::FrameMailbox>, u64),
}
pub(super) struct Request {
    work: Work,
    reply: Option<mpsc::SyncSender<nixe_nvn::Result<()>>>,
    _permit: Option<GpuFrontendPermit>,
}
impl Request {
    pub(super) fn run(
        self,
        backend: &mut impl nixe_gpu::GpuPresentation,
    ) -> Result<(), nixe_gpu::BackendRuntimeError> {
        let result = match self.work {
            Work::Submit(queue, commands) => backend.nvn_submit(queue, commands),
            Work::Flush => backend.flush(),
            Work::Drain => Ok(()),
            Work::FinalizeQueue(queue) => backend.nvn_finalize_queue(queue),
            Work::Fence(action) => backend.submit(Some(action)),
            Work::AcquireFence(action) => backend.nvn_acquire_fence(action),
            Work::Present(t, interval, flip_vertical, mailbox, sequence) => backend
                .nvn_present(t, interval, flip_vertical)
                .map(|image| {
                    if let Some(mailbox) = mailbox {
                        mailbox.publish(Arc::new(nixe_video::PresentationFrame::new(
                            image, sequence,
                        )))
                    }
                }),
        };
        if let Some(reply) = self.reply {
            let _ = reply.send(result.as_ref().copied().map_err(|e| e.to_string().into()));
        }
        result
    }
}
/// The owner may stop before recording the fence, or before its timeline callback.
struct PendingSignal(Option<Arc<Fence>>);
impl PendingSignal {
    fn finish(&mut self, result: nixe_nvn::Result<()>) {
        if let Some(fence) = self.0.take() {
            fence.signal(result);
        }
    }
}
impl Drop for PendingSignal {
    fn drop(&mut self) {
        if let Some(fence) = self.0.take() {
            fence.signal(Err("GPU owner stopped before NVN fence completion".into()));
        }
    }
}
fn signal(fence: Arc<Fence>) -> nixe_gpu::FenceAction {
    let mut pending = PendingSignal(Some(fence));
    Box::new(move |result| {
        pending.finish(result.as_ref().copied().map_err(|e| e.to_string().into()));
        result
    })
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
        let charge = match &work {
            Work::Submit(_, ops) => ops
                .operations
                .capacity()
                .saturating_mul(size_of::<nixe_nvn::Operation>())
                .saturating_add(
                    ops.draws
                        .capacity()
                        .saturating_mul(size_of::<nixe_nvn::Draw>()),
                )
                .saturating_add(ops.update_bytes),
            _ => 0,
        }
        .max(MAX_GPU_INGRESS_BYTES / 64);
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
                _permit: permit,
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
        self.execute(Work::AcquireFence(signal(f)))
    }
    fn fence(&self, f: Arc<Fence>) -> nixe_nvn::Result<()> {
        self.execute(Work::Fence(signal(f)))
    }
    fn flush(&self) -> nixe_nvn::Result<()> {
        self.execute(Work::Flush)
    }
    fn present(&self, t: Texture, interval: u32, flip_vertical: bool) -> nixe_nvn::Result<()> {
        let seq = self
            .sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        self.execute(Work::Present(
            t,
            interval,
            flip_vertical,
            self.mailbox.clone(),
            seq,
        ))
    }
}
