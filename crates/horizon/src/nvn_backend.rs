//! Work shared by the NVN graphics owners.
use nixe_gpu::{BackendRuntimeError, GpuPresentation, NvnQueueWork};
use nixe_nvn::{Submission, Texture};
use std::sync::{Arc, mpsc};

pub(crate) enum Work {
    Submit(u64, Submission),
    Flush(u64),
    Drain,
    FinalizeQueue(u64),
    Fence(u64, nixe_gpu::FenceAction, bool),
    AcquireFence(nixe_gpu::FenceAction),
    Present {
        queue: u64,
        texture: Texture,
        interval: u32,
        flip: bool,
        crop: [u32; 4],
        mailbox: Option<nixe_video::FrameMailbox>,
        sequence: u64,
        receipt: Option<nixe_nvn::display::PresentReceipt>,
    },
}
impl Work {
    pub(crate) fn run(
        self,
        backend: &mut impl GpuPresentation,
        hold: impl Send + 'static,
        reply: Option<mpsc::SyncSender<nixe_nvn::Result<()>>>,
    ) -> Result<(), BackendRuntimeError> {
        let result = match self {
            Self::Drain => {
                if let Some(reply) = reply {
                    backend.nvn_drain(Box::new(move || {
                        let _ = reply.send(Ok(()));
                    }));
                }
                return Ok(());
            }
            Self::AcquireFence(action) => backend.nvn_acquire_fence(action),
            work => {
                let (queue, work) = match work {
                    Self::Submit(queue, commands) => (queue, NvnQueueWork::Submit(commands)),
                    Self::FinalizeQueue(queue) => (queue, NvnQueueWork::Finalize),
                    Self::Flush(queue) => (
                        queue,
                        NvnQueueWork::Flush {
                            flush_for_cpu: false,
                            action: None,
                        },
                    ),
                    Self::Fence(queue, action, flush_for_cpu) => (
                        queue,
                        NvnQueueWork::Flush {
                            flush_for_cpu,
                            action: Some(action),
                        },
                    ),
                    Self::Present {
                        queue,
                        texture,
                        interval,
                        flip,
                        crop,
                        mailbox,
                        sequence,
                        receipt,
                    } => (
                        queue,
                        NvnQueueWork::Present {
                            texture,
                            interval,
                            flip,
                            crop,
                            publish: Box::new(move |image| {
                                if let Some(mailbox) = mailbox {
                                    mailbox.publish(Arc::new(nixe_video::PresentationFrame::new(
                                        image, sequence,
                                    )));
                                }
                            }),
                            complete: receipt.map(|receipt| -> nixe_gpu::FenceAction {
                                Box::new(move |result| {
                                    result?;
                                    receipt.complete();
                                    Ok(())
                                })
                            }),
                        },
                    ),
                    _ => unreachable!(),
                };
                backend.nvn_queue_work(queue, work, hold)
            }
        };
        if let Some(reply) = reply {
            let _ = reply.send(result.as_ref().copied().map_err(|e| e.to_string().into()));
        }
        result
    }

    pub(crate) fn allocation_bytes(&self) -> usize {
        match self {
            Self::Submit(_, commands) => commands.allocation_bytes(),
            _ => 0,
        }
    }
}
