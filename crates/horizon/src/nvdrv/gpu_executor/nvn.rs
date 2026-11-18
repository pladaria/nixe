use super::*;
use nixe_nvn::{Fence, Operation, Texture};
pub(super) enum Work {
    Submit(Vec<Operation>),
    Flush,
    Fence(Arc<Fence>),
    Present(Texture, u32, Option<nixe_video::FrameMailbox>, u64),
}
pub(super) struct Request {
    work: Work,
    reply: mpsc::SyncSender<nixe_nvn::Result<()>>,
}
impl Request {
    pub(super) fn run(self, backend: &mut impl nixe_gpu::GpuPresentation) {
        let result = match self.work {
            Work::Submit(commands) => backend.nvn_submit(commands),
            Work::Flush => backend.flush(),
            Work::Fence(fence) => backend.submit(Some(Box::new(move |result| {
                fence.signal(
                    result
                        .as_ref()
                        .map(|_| ())
                        .map_err(|e| e.to_string().into()),
                );
                result
            }))),
            Work::Present(t, interval, mailbox, sequence) => {
                backend.nvn_present(t, interval).map(|image| {
                    if let Some(mailbox) = mailbox {
                        mailbox.publish(Arc::new(nixe_video::PresentationFrame::new(
                            image, sequence,
                        )))
                    }
                })
            }
        };
        let _ = self.reply.send(result.map_err(|e| e.to_string().into()));
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
        let (reply, done) = mpsc::sync_channel(1);
        backend
            .sender
            .lock()
            .unwrap()
            .as_ref()
            .ok_or("GPU owner stopped")?
            .send(GpuMessage::Nvn(Request { work, reply }))
            .map_err(|_| "GPU owner stopped")?;
        done.recv()
            .map_err(|_| "GPU owner stopped before NVN execution")?
    }
}
impl nixe_nvn::Backend for Client {
    fn submit(&self, ops: Vec<Operation>) -> nixe_nvn::Result<()> {
        self.execute(Work::Submit(ops))
    }
    fn fence(&self, f: Arc<Fence>) -> nixe_nvn::Result<()> {
        self.execute(Work::Fence(f))
    }
    fn flush(&self) -> nixe_nvn::Result<()> {
        self.execute(Work::Flush)
    }
    fn present(&self, t: Texture, interval: u32) -> nixe_nvn::Result<()> {
        let seq = self
            .sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        self.execute(Work::Present(t, interval, self.mailbox.clone(), seq))
    }
}
