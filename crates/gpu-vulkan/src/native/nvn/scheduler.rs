//! Replay independent NVN queues on the native recording owner. An unresolved
//! event retains only its queue's suffix and lets other queues issue producers.
use super::*;
use nixe_gpu::{GpuPresentation, NvnQueueWork};
use nixe_nvn::{Submission, SubmissionCursor};

type Retained = Box<dyn Send>;
type Drain = Box<dyn FnOnce() + Send>;

#[derive(Default)]
pub(in crate::native) struct Scheduler {
    parked: VecDeque<ParkedQueue>,
    drains: VecDeque<(u64, Drain)>,
    sequence: u64,
    retained_bytes: usize,
}

struct ParkedQueue {
    id: u64,
    head: Blocked,
    suffix: VecDeque<Queued>,
}

struct Blocked {
    wait: nixe_nvn::event::EventWait,
    commands: Submission,
    cursor: SubmissionCursor,
    sequence: u64,
    _hold: Retained,
}

struct Queued {
    work: NvnQueueWork,
    sequence: u64,
    hold: Retained,
}

fn work_bytes(work: &NvnQueueWork) -> usize {
    match work {
        NvnQueueWork::Submit(commands) => commands.allocation_bytes().max(1024),
        _ => 1024,
    }
}

impl Scheduler {
    pub(in crate::native) fn has_waiters(&self) -> bool {
        !self.parked.is_empty()
    }

    pub(in crate::native) fn drain(&mut self, finish: Drain) {
        if self.has_waiters() {
            self.drains.push_back((self.sequence, finish));
        } else {
            finish();
        }
    }

    fn retain(&mut self, bytes: usize) -> Result<u64, BackendRuntimeError> {
        // Parked work retains the ingress permit if it has one. The slow
        // admission path has no permit because it waits for the owner reply.
        // Bound that work too without standing in the way of other queues.
        self.retained_bytes = self
            .retained_bytes
            .checked_add(bytes)
            .filter(|&size| size <= 64 * 1024 * 1024)
            .ok_or_else(|| {
                runtime(error(
                    "parked NVN command queues exceed their memory budget",
                ))
            })?;
        self.sequence += 1;
        Ok(self.sequence)
    }

    fn retire_drains(&mut self) {
        let oldest = self.parked.iter().map(|q| q.head.sequence).min();
        while self
            .drains
            .front()
            .is_some_and(|&(limit, _)| oldest.is_none_or(|seq| seq > limit))
        {
            let (_, finish) = self.drains.pop_front().unwrap();
            finish();
        }
    }
}

impl NativeGpuExecution {
    pub(in crate::native) fn queue_nvn_work(
        &mut self,
        queue: u64,
        work: NvnQueueWork,
        hold: impl Send + 'static,
    ) -> Result<(), BackendRuntimeError> {
        let scheduler = &mut self.state.nvn.work;
        if scheduler.has_waiters()
            && let Some(index) = scheduler.parked.iter().position(|q| q.id == queue)
        {
            let sequence = scheduler.retain(work_bytes(&work))?;
            scheduler.parked[index].suffix.push_back(Queued {
                work,
                sequence,
                hold: Box::new(hold),
            });
            return Ok(());
        }
        let wake_waiters = scheduler.has_waiters();
        match work {
            NvnQueueWork::Submit(mut commands) => {
                let mut cursor = SubmissionCursor::default();
                if let Some(wait) = self.run_nvn_submission(queue, &mut commands, &mut cursor)? {
                    let scheduler = &mut self.state.nvn.work;
                    let sequence = scheduler.retain(commands.allocation_bytes().max(1024))?;
                    scheduler.parked.push_back(ParkedQueue {
                        id: queue,
                        head: Blocked {
                            wait,
                            commands,
                            cursor,
                            sequence,
                            _hold: Box::new(hold),
                        },
                        suffix: VecDeque::new(),
                    });
                }
            }
            work => self.issue_nvn_queue_control(queue, work)?,
        }
        if wake_waiters {
            self.progress_nvn_waits()?;
        }
        Ok(())
    }

    pub(in crate::native) fn progress_nvn_waits(&mut self) -> Result<(), BackendRuntimeError> {
        if !self.state.nvn.work.has_waiters() {
            return Ok(());
        }
        nixe_gpu::OwnerMemory::invalidate_gpu_cache(self).map_err(BackendRuntimeError::Memory)?;
        loop {
            let mut progressed = false;
            let queues = self.state.nvn.work.parked.len();
            for _ in 0..queues {
                let mut pending = self.state.nvn.work.parked.pop_front().unwrap();
                if !self.state.nvn.events.ready(&pending.head.wait) {
                    self.state.nvn.work.parked.push_back(pending);
                    continue;
                }
                progressed = true;
                self.execution()
                    .residency
                    .order_nvn(0x100)
                    .map_err(runtime)?;
                if let Some(wait) = self.run_nvn_submission(
                    pending.id,
                    &mut pending.head.commands,
                    &mut pending.head.cursor,
                )? {
                    pending.head.wait = wait;
                    self.state.nvn.work.parked.push_back(pending);
                    continue;
                }
                self.state.nvn.work.retained_bytes -=
                    pending.head.commands.allocation_bytes().max(1024);
                log::trace!("NVN queue {:#x} resumed", pending.id);
                let ParkedQueue {
                    id,
                    head,
                    mut suffix,
                } = pending;
                drop(head);
                while let Some(Queued {
                    work,
                    sequence,
                    hold,
                }) = suffix.pop_front()
                {
                    let bytes = work_bytes(&work);
                    match work {
                        NvnQueueWork::Submit(mut commands) => {
                            let mut cursor = SubmissionCursor::default();
                            if let Some(wait) =
                                self.run_nvn_submission(id, &mut commands, &mut cursor)?
                            {
                                self.state.nvn.work.parked.push_back(ParkedQueue {
                                    id,
                                    head: Blocked {
                                        wait,
                                        commands,
                                        cursor,
                                        sequence,
                                        _hold: hold,
                                    },
                                    suffix,
                                });
                                break;
                            }
                        }
                        work => self.issue_nvn_queue_control(id, work)?,
                    }
                    self.state.nvn.work.retained_bytes -= bytes;
                }
            }
            self.state.nvn.work.retire_drains();
            if !progressed || !self.state.nvn.work.has_waiters() {
                return Ok(());
            }
        }
    }

    fn issue_nvn_queue_control(
        &mut self,
        queue: u64,
        work: NvnQueueWork,
    ) -> Result<(), BackendRuntimeError> {
        match work {
            NvnQueueWork::Submit(_) => unreachable!("lists use resumable replay"),
            NvnQueueWork::Flush {
                flush_for_cpu,
                action,
            } => self.nvn_flush(flush_for_cpu, action),
            NvnQueueWork::Present {
                texture,
                interval,
                flip,
                crop,
                publish,
                complete,
            } => {
                let image = self.composite_image(
                    super::presentation(texture, flip, crop).map_err(runtime)?,
                    interval,
                    None,
                )?;
                publish(image);
                if let Some(complete) = complete {
                    self.nvn_acquire_fence(complete)?;
                }
                Ok(())
            }
            NvnQueueWork::Finalize => {
                self.state.nvn.finalize_queue(queue);
                Ok(())
            }
        }
    }
}
