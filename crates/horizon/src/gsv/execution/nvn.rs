//! NVN submissions enter the preparation owner directly, bypassing PFIFO and MME.
use super::{MAX_INPUT, Permit, control::Endpoint};
use crate::nvn_backend::Work;
use nixe_gpu::nvn_fence_action;
use nixe_nvn::display::DisplayHistory;
use nixe_nvn::{Backend, Fence, Result, Submission, Texture};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

pub(super) struct Request {
    work: Work,
    reply: Option<mpsc::SyncSender<Result<()>>>,
    permit: Option<Permit>,
}

impl Request {
    pub(super) fn new(
        work: Work,
        budget: &Arc<AtomicUsize>,
    ) -> (Self, Option<mpsc::Receiver<Result<()>>>) {
        let bytes = work.allocation_bytes().max(MAX_INPUT / 64);
        let permit = budget
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&next| next <= MAX_INPUT)
            })
            .is_ok()
            .then(|| Permit {
                bytes,
                budget: budget.clone(),
            });
        let (reply, done) = if permit.is_none() || matches!(work, Work::Drain) {
            let (sender, receiver) = mpsc::sync_channel(1);
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        (
            Self {
                work,
                reply,
                permit,
            },
            done,
        )
    }

    pub(super) fn run(
        self,
        renderer: &mut impl nixe_gpu::GpuPresentation,
    ) -> std::result::Result<(), nixe_gpu::BackendRuntimeError> {
        self.work.run(renderer, self.permit, self.reply)
    }
}

struct Client {
    owner: Weak<Endpoint>,
    display: Arc<DisplayHistory>,
}

pub(super) fn client(owner: &Arc<Endpoint>, display: Arc<DisplayHistory>) -> Arc<dyn Backend> {
    Arc::new(Client {
        owner: Arc::downgrade(owner),
        display,
    })
}

impl Client {
    fn owner(&self) -> Result<Arc<Endpoint>> {
        self.owner
            .upgrade()
            .ok_or_else(|| "native GPU owner stopped".into())
    }

    fn execute(&self, work: Work) -> Result<()> {
        self.owner()?.nvn_work(work)
    }
}

impl Backend for Client {
    fn submit(&self, queue: u64, commands: Submission) -> Result<()> {
        self.execute(Work::Submit(queue, commands))
    }
    fn drain(&self) -> Result<()> {
        self.execute(Work::Drain)
    }
    fn finalize_queue(&self, queue: u64) -> Result<()> {
        self.execute(Work::FinalizeQueue(queue))
    }
    fn acquire_fence(&self, fence: Arc<Fence>) -> Result<()> {
        self.execute(Work::AcquireFence(nvn_fence_action(fence)))
    }
    fn fence(&self, queue: u64, fence: Arc<Fence>, flush_for_cpu: bool) -> Result<()> {
        self.execute(Work::Fence(queue, nvn_fence_action(fence), flush_for_cpu))
    }
    fn flush(&self, queue: u64) -> Result<()> {
        self.execute(Work::Flush(queue))
    }
    fn present(
        &self,
        queue: u64,
        texture: Texture,
        interval: u32,
        flip: bool,
        crop: [u32; 4],
    ) -> Result<()> {
        self.owner()?
            .nvn_present(queue, texture, interval, flip, crop, self.display.queue())
    }
}
