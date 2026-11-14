//! A suspended acquire blocks its channel, while other channels can produce
//! its release. Mapping changes follow the consumed prefix, not a suspended
//! channel's unfetched tail; otherwise a producer needing a new map deadlocks.
use super::*;
use std::collections::{BTreeSet, VecDeque};

struct Work {
    job: Arc<Submission>,
    entry: usize,
    offset: usize,
    wait: Option<Arc<semaphore::Wait>>,
    fetch: commands::Fetch,
    started: Option<std::time::Instant>,
}
enum Pending {
    Submit(Work),
    Memory(MemoryJob, Reply),
    Retire(u64, Reply),
}
fn admit(message: Ingress, pending: &mut VecDeque<Pending>) -> bool {
    pending.push_back(match message {
        Ingress::Submit(job) => Pending::Submit(Work {
            job,
            entry: 0,
            offset: 0,
            wait: None,
            fetch: Default::default(),
            started: None,
        }),
        Ingress::Memory(job, reply) => Pending::Memory(job, reply),
        Ingress::Retire(channel, reply) => Pending::Retire(channel, reply),
        Ingress::Stop => return false,
    });
    true
}

pub(super) fn run(
    input: mpsc::Receiver<Ingress>,
    output: nixe_gpu::ordered::Sender<Control>,
    machine: Machine,
    stop: &AtomicBool,
    waiters: &semaphore::Waiters,
    commands: Arc<dyn nixe_gpu::command_memory::CommandMemory>,
) -> Result<()> {
    let mut writer = ampere::ordered::Writer::new(output);
    let mut channels = BTreeMap::<u64, ampere::Channel>::new();
    let mut pending = VecDeque::new();
    let mut dependencies = dependencies::Dependencies::default();
    let result: Result<()> = (|| {
        while !stop.load(Ordering::Acquire) {
            if pending.is_empty() {
                let Ok(message) = input.recv() else {
                    break;
                };
                if !admit(message, &mut pending) {
                    break;
                }
            }
            // Keep ingress bounded even if a channel cannot make progress.
            while pending.len() < 128 {
                match input.try_recv() {
                    Ok(message) => {
                        if !admit(message, &mut pending) {
                            return Ok(());
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
                }
            }
            let mut blocked = BTreeSet::new();
            let mut selected = None;
            for (index, item) in pending.iter_mut().enumerate() {
                let work = match item {
                    Pending::Memory(..) => {
                        selected = Some(index);
                        break;
                    }
                    Pending::Retire(channel, _) => {
                        if blocked.contains(channel) {
                            continue;
                        }
                        selected = Some(index);
                        break;
                    }
                    Pending::Submit(work) => work,
                };
                if !blocked.insert(work.job.channel) {
                    continue;
                }
                let memory = Memory {
                    machine: &machine,
                    space: &work.job.space,
                    stop,
                    waiters,
                };
                if let Some(wait) = &work.wait {
                    if !wait
                        .ready(&memory)
                        .map_err(|e| wait.acquire.source.error(e))?
                    {
                        continue;
                    }
                    waiters.remove(work.job.channel);
                    work.wait = None;
                }
                if !work.fetch.ready(&*commands)? {
                    continue;
                }
                selected = Some(index);
                break;
            }
            let Some(index) = selected else {
                // CPU stores have no device notification; poll only while an
                // acquire is outstanding. GPU writes and new jobs wake early.
                thread::park_timeout(std::time::Duration::from_micros(100));
                continue;
            };
            match pending.remove(index).unwrap() {
                Pending::Memory(job, reply) => {
                    writer
                        .ring
                        .send(Control::Memory(job, reply))
                        .map_err(|_| "native preparation stopped")?;
                    writer
                        .ring
                        .wait_until(writer.ring.position())
                        .map_err(|_| "native mapping operation failed")?;
                    dependencies.pending_draw = false;
                }
                Pending::Retire(channel, reply) => {
                    channels.remove(&channel);
                    waiters.remove(channel);
                    writer
                        .ring
                        .send(Control::Retire(channel, reply))
                        .map_err(|_| "native preparation stopped")?;
                }
                Pending::Submit(mut work) => {
                    let job = &work.job;
                    writer
                        .ring
                        .send(Control::Begin(
                            job.clone(),
                            *work.started.get_or_insert_with(std::time::Instant::now),
                        ))
                        .map_err(|_| "native preparation stopped")?;
                    writer.reset();
                    let channel = channels.entry(job.channel).or_default();
                    let mut host = Front {
                        writer: &mut writer,
                        memory: Memory {
                            machine: &machine,
                            space: &job.space,
                            stop,
                            waiters,
                        },
                        dependencies: &mut dependencies,
                        pending_write: None,
                        commands: &*commands,
                    };
                    while work.entry < job.entries.len() {
                        let entry = &job.entries[work.entry];
                        if work.offset == entry.words as usize {
                            work.entry += 1;
                            work.offset = 0;
                            continue;
                        }
                        if entry.sync_wait && !work.fetch.started() {
                            host.finish_draws()?;
                            host.dependencies.sync(host.writer.ring.position());
                        }
                        // Re-fetch only the unconsumed tail after an acquire:
                        // a CPU release may publish new command words there.
                        let source = Source {
                            channel: job.channel,
                            submission: job.id,
                            entry: work.entry as u32,
                            pushbuffer: entry.address,
                            offset: work.offset as u32,
                            ..Source::default()
                        };
                        let Some(data) = host
                            .read_commands(
                                &mut work.fetch,
                                entry.address + work.offset as u64 * 4,
                                (entry.words as usize - work.offset) * 4,
                            )
                            .map_err(|error| source.error(error))?
                        else {
                            break;
                        };
                        let progress = channel.execute(source, &data, &mut host)?;
                        work.offset += progress.words;
                        if let Some(acquire) = progress.acquire {
                            let wait = waiters.register(acquire, &host.memory)?;
                            // Arm before draining: matching releases in that
                            // prefix must remain latched even if overwritten.
                            host.barrier()?;
                            if wait
                                .ready(&host.memory)
                                .map_err(|e| acquire.source.error(e))?
                            {
                                waiters.remove(job.channel);
                                continue;
                            }
                            work.wait = Some(wait);
                            break;
                        }
                    }
                    host.finish_writes();
                    if work.wait.is_some() || work.fetch.started() {
                        pending.insert(index, Pending::Submit(work));
                    } else {
                        host.writer
                            .ring
                            .send(Control::End)
                            .map_err(|_| "native preparation stopped")?;
                        host.dependencies.pending_draw = false;
                        if job.completion.is_some() {
                            host.dependencies.any(host.writer.ring.position());
                        }
                    }
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        writer
            .ring
            .send(Control::Failure(error))
            .map_err(|_| "native preparation stopped")?;
        let _ = writer.ring.wait_until(writer.ring.position());
    }
    Ok(())
}
