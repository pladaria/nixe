//! Packet decoding owns its worker, output leases and completion counter.
//! @SPECULATIVE: queue limits, fence/output layouts and ready notifications follow
//! native SDK observations; Switchbrew names these operations without their ABI.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
use super::{execution::Publication, syncpt::Counter, video_image::Image};
use crate::video_codec::{Decoder, Frame};
use nixe_memory::{Access, AccessResult, CapturedMemory, Machine};
use nixe_runtime::{EventObject, RunnableProcess};
use std::{
    collections::{BTreeSet, VecDeque},
    future::Future,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, JoinHandle},
};

type Result<T> = std::result::Result<T, Box<str>>;
pub(crate) const BUSY: u32 = (3603 << 9) | 111;
pub(crate) const EMPTY: u32 = 0x1c286f;
#[derive(Debug)]
pub(crate) enum WorkMemory {
    Transfer {
        _memory: Arc<nixe_runtime::TransferMemoryObject>,
    },
    Remote {
        _process: Arc<RunnableProcess>,
    },
}
pub(crate) struct Session {
    pub memory: Mutex<Option<WorkMemory>>,
    inner: Arc<Inner>,
    worker: Option<JoinHandle<()>>,
    timer: Option<nixe_scheduler::LoopingEvent>,
}
impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoderSession")
            .finish_non_exhaustive()
    }
}
struct Inner {
    state: Mutex<State>,
    stop: AtomicBool,
    event: EventObject,
    process: Weak<RunnableProcess>,
    machine: Machine,
    device: Option<[u8; 16]>,
    counter: Arc<Mutex<Counter>>,
    publication: Publication,
}
#[derive(Default)]
struct State {
    config: Option<Config>,
    started: bool,
    inputs: VecDeque<Input>,
    free: VecDeque<Target>,
    outputs: VecDeque<Output>,
    // Registration tracks identities. Only queued work retains image storage.
    registered: BTreeSet<u32>,
    busy: BTreeSet<u32>,
    submitted: u64,
    end_queued: bool,
    reset: Option<(u64, Reset)>,
    reset_id: u64,
    reset_done: u64,
    reset_wake: Option<Waker>,
    error: Option<Box<str>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reset {
    Flush,
    Stop,
    Finalize,
}
#[derive(Clone, Copy)]
pub(crate) struct Config {
    pub codec: u32,
    pub width: u32,
    pub height: u32,
}
struct Input {
    bytes: Vec<u8>,
    pts: i64,
    end: bool,
}
struct Decoded {
    frame: Option<Frame>,
    pts: i64,
    end: bool,
}
struct Target {
    image: Arc<Image>,
    release: Option<(Arc<Mutex<Counter>>, u32)>,
    point: nixe_gpu::ReservedTimelinePoint,
}
pub(crate) struct Output {
    pub image: u32,
    pub width: u32,
    pub height: u32,
    pub pts: i64,
    pub end: bool,
}
struct Tick(Weak<Inner>);
impl Wake for Tick {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        if let Some(inner) = self.0.upgrade() {
            let s = inner.state.lock().unwrap();
            if s.started && !s.end_queued && s.inputs.len() < 8 {
                inner.event.signal();
            }
        }
    }
}
struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
impl Session {
    pub fn new(
        process: &Arc<RunnableProcess>,
        counter: Arc<Mutex<Counter>>,
        publication: Publication,
        device: Option<[u8; 16]>,
    ) -> Result<Self> {
        let inner = Arc::new(Inner {
            state: Mutex::new(State::default()),
            stop: AtomicBool::new(false),
            event: EventObject::new(process.scheduler()),
            process: Arc::downgrade(process),
            machine: process.memory().space().machine().clone(),
            device,
            counter,
            publication,
        });
        let thread_inner = inner.clone();
        let worker = thread::Builder::new()
            .name("nixe-gsv-video".into())
            .spawn(move || {
                if let Err(error) = thread_inner.run()
                    && !thread_inner.stop.load(Ordering::Acquire)
                {
                    let mut state = thread_inner.state.lock().unwrap();
                    state.error = Some(error.clone());
                    if let Some(wake) = state.reset_wake.take() {
                        wake.wake();
                    }
                    drop(state);
                    if let Some(p) = thread_inner.process.upgrade() {
                        p.report_device_failure(format!("GSV video decoding failed: {error}"));
                    }
                }
            })
            .map_err(|e| e.to_string().into_boxed_str())?;
        let mut session = Self {
            memory: Mutex::new(None),
            inner,
            worker: Some(worker),
            timer: None,
        };
        let period = std::num::NonZeroU64::new(4_000_000).unwrap();
        session.timer = Some(
            process
                .scheduler()
                .schedule_looping_event(
                    process
                        .scheduler()
                        .clock()
                        .scheduler_time_ns()
                        .saturating_add(period.get()),
                    period,
                    Waker::from(Arc::new(Tick(Arc::downgrade(&session.inner)))),
                )
                .map_err(|_| Box::<str>::from("video ready timer is stopped"))?,
        );
        Ok(session)
    }
    pub fn event(&self) -> nixe_runtime::ReadableEventObject {
        self.inner.event.readable()
    }
    pub fn configure(&self, config: Config) -> Result<()> {
        if self.memory.lock().unwrap().is_none() {
            return Err("decoder is not initialized".into());
        }
        let mut s = self.inner.state.lock().unwrap();
        if s.config.is_some()
            || s.started
            || config.width == 0
            || config.height == 0
            || config.width > 16384
            || config.height > 16384
        {
            return Err("invalid decoder configuration lifecycle/dimensions".into());
        }
        s.config = Some(config);
        drop(s);
        self.worker.as_ref().unwrap().thread().unpark();
        Ok(())
    }
    pub fn start(&self) -> Result<()> {
        let mut s = self.inner.state.lock().unwrap();
        if s.config.is_none() {
            return Err("decoder not configured".into());
        }
        s.started = true;
        drop(s);
        self.inner.event.signal();
        self.worker.as_ref().unwrap().thread().unpark();
        Ok(())
    }
    pub fn submit(
        &self,
        bytes: Vec<u8>,
        pts: i64,
        end: bool,
    ) -> Result<std::result::Result<u64, u32>> {
        let mut s = self.inner.state.lock().unwrap();
        if !s.started || pts.checked_mul(1000).is_none() || (!end && bytes.is_empty()) {
            return Err(format!(
                "invalid decoder input: started={}, end_queued={}, pts={pts:#x}, end={end}, bytes={}",
                s.started, s.end_queued, bytes.len(),
            ).into());
        }
        if s.inputs.len() >= 24 && !end {
            return Ok(Err(BUSY));
        }
        s.submitted = s
            .submitted
            .checked_add(1)
            .ok_or("decoder input sequence overflow")?;
        let sequence = s.submitted;
        s.inputs.push_back(Input { bytes, pts, end });
        // @SPECULATIVE: EOS ends one compressed stream, not the session. A
        // client may submit its next stream without Stop/Start or Flush. The
        // worker drains and resets the codec at the queued boundary; existing
        // output images retain their original fence and ordering.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/host1x/nx2_video_decoder.cpp
        s.end_queued = end;
        drop(s);
        self.worker.as_ref().unwrap().thread().unpark();
        Ok(Ok(sequence))
    }
    pub fn bind(
        &self,
        image: Arc<Image>,
        release: Option<(Arc<Mutex<Counter>>, u32)>,
        register: bool,
    ) -> Result<[u8; 16]> {
        let mut s = self.inner.state.lock().unwrap();
        if s.busy.contains(&image.id) {
            return Err("decoder image is still queued".into());
        }
        if register {
            if !s.registered.insert(image.id) {
                return Err("decoder image already registered".into());
            }
        } else {
            if !s.registered.contains(&image.id) {
                return Err("decoder image is not registered".into());
            }
            // Clients may consume images through their fence ring without
            // requesting output metadata. Releasing an image retires either form.
            s.outputs.retain(|output| output.image != image.id);
        }
        let mut c = self.inner.counter.lock().unwrap();
        let owner = c.timeline.owner();
        let point = c
            .timeline
            .reserve(owner, 1)
            .map_err(|e| e.to_string().into_boxed_str())?;
        let mut result = [0; 16];
        result[..4].copy_from_slice(&(!c.timeline.syncpoint().get()).to_le_bytes());
        result[8..].copy_from_slice(&u64::from(point.point().value().get()).to_le_bytes());
        s.busy.insert(image.id);
        s.free.push_back(Target {
            image,
            release,
            point,
        });
        drop(c);
        drop(s);
        self.worker.as_ref().unwrap().thread().unpark();
        Ok(result)
    }
    pub fn unbind(&self, id: u32) -> Result<()> {
        let mut s = self.inner.state.lock().unwrap();
        if s.busy.contains(&id) {
            return Err("decoder image is still in flight".into());
        }
        if !s.registered.remove(&id) {
            return Err("unknown decoder image".into());
        }
        Ok(())
    }
    pub fn output(&self) -> Option<Output> {
        let mut s = self.inner.state.lock().unwrap();
        s.outputs.pop_front()
    }
    pub async fn reset(&self, operation: Reset) -> Result<()> {
        let id = {
            let mut s = self.inner.state.lock().unwrap();
            if s.reset.is_some() {
                return Err("overlapping decoder reset".into());
            }
            s.reset_id = s.reset_id.checked_add(1).ok_or("decoder reset overflow")?;
            let id = s.reset_id;
            s.reset = Some((id, operation));
            id
        };
        self.worker.as_ref().unwrap().thread().unpark();
        std::future::poll_fn(|cx| {
            let mut s = self.inner.state.lock().unwrap();
            if let Some(e) = &s.error {
                return Poll::Ready(Err(e.clone()));
            }
            if s.reset_done >= id {
                Poll::Ready(Ok(()))
            } else {
                s.reset_wake = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await?;
        if operation == Reset::Finalize {
            self.memory.lock().unwrap().take();
        }
        Ok(())
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if let Some(t) = self.timer.take() {
            t.cancel();
        }
        self.inner.stop.store(true, Ordering::Release);
        self.worker.as_ref().unwrap().thread().unpark();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::error!("GSV video worker panicked during shutdown");
        }
    }
}
impl Inner {
    fn wait<F: Future>(&self, future: F) -> Result<F::Output> {
        let mut future = std::pin::pin!(future);
        let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
        let mut cx = Context::from_waker(&waker);
        loop {
            if self.stop.load(Ordering::Acquire) {
                return Err("video operation cancelled".into());
            }
            if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
                return Ok(value);
            }
            thread::park();
        }
    }
    fn write(&self, ranges: &CapturedMemory, offset: u64, bytes: Vec<u8>) -> Result<()> {
        match self
            .machine
            .write_captured_owned(ranges, offset, bytes)
            .map_err(|e| e.to_string().into_boxed_str())?
        {
            Access::Ready(_) => Ok(()),
            Access::Pending(p) => {
                match self.wait(p)?.map_err(|e| e.to_string().into_boxed_str())? {
                    AccessResult::Written(_) => Ok(()),
                    _ => Err("invalid video write completion".into()),
                }
            }
        }
    }
    fn complete(&self, target: &Target) -> Result<()> {
        let (id, value) = {
            let mut c = self.counter.lock().unwrap();
            let owner = c.timeline.owner();
            c.timeline
                .consume_increment(owner, &target.point)
                .map_err(|e| e.to_string().into_boxed_str())?;
            (
                c.timeline.syncpoint().get(),
                c.value
                    .checked_add(1)
                    .ok_or("video completion counter overflow")?,
            )
        };
        if let Some(gpu) = &self.publication.gpu {
            self.write(
                gpu,
                u64::from(id) * 4,
                (value as u32).to_le_bytes().to_vec(),
            )?;
        }
        if let Some((cpu, _, _)) = &self.publication.cpu {
            self.write(
                cpu.ranges(),
                0x40 + u64::from(id) * 8,
                value.to_le_bytes().to_vec(),
            )?;
        }
        let mut c = self.counter.lock().unwrap();
        let owner = c.timeline.owner();
        c.timeline
            .increment_reserved(owner, &target.point)
            .map_err(|e| e.to_string().into_boxed_str())?;
        c.value = value;
        c.consumed_value = value;
        c.notify();
        drop(c);
        if let Some((_, event, _)) = &self.publication.cpu {
            event.signal();
        }
        Ok(())
    }
    fn publish(&self, target: &Target, frame: &Frame) -> Result<()> {
        let l = target.image.pool.layout;
        if frame.width != l.width as usize || frame.height != l.height as usize {
            return Err("decoded frame dimensions disagree with output image".into());
        }
        let uv = frame.chroma();
        for (src, stride, width, rows, offset) in [
            (
                frame.luma(),
                frame.luma_pitch(),
                frame.width,
                frame.height,
                0,
            ),
            (
                uv.as_ref(),
                frame.chroma_pitch(),
                frame.width.div_ceil(2) * 2,
                frame.height.div_ceil(2),
                l.chroma,
            ),
        ] {
            let mut bytes = vec![0; l.pitch * rows];
            for y in 0..rows {
                bytes[y * l.pitch..y * l.pitch + width]
                    .copy_from_slice(&src[y * stride..y * stride + width]);
            }
            self.write(
                &target.image.pool.backing.ranges,
                target.image.offset() + offset as u64,
                bytes,
            )?;
        }
        Ok(())
    }
    fn run(&self) -> Result<()> {
        let mut decoder: Option<Decoder> = None;
        let mut decoded = VecDeque::<Decoded>::new();
        loop {
            if self.stop.load(Ordering::Acquire) {
                return Ok(());
            }
            let mut s = self.state.lock().unwrap();
            if let Some((id, operation)) = s.reset.take() {
                let stop = operation != Reset::Flush;
                s.inputs.clear();
                s.end_queued = false;
                if stop {
                    s.started = false;
                    s.outputs.clear();
                    s.busy.clear();
                }
                if operation == Reset::Finalize {
                    s.config = None;
                    s.registered.clear();
                }
                let free = if stop {
                    std::mem::take(&mut s.free)
                } else {
                    VecDeque::new()
                };
                drop(s);
                if operation == Reset::Finalize {
                    decoder = None;
                } else if let Some(d) = decoder.as_mut() {
                    d.flush();
                }
                decoded.clear();
                for target in free {
                    self.complete(&target)?;
                }
                let mut s = self.state.lock().unwrap();
                s.reset_done = id;
                if let Some(w) = s.reset_wake.take() {
                    w.wake();
                }
                continue;
            }
            if decoder.is_none()
                && let Some(c) = s.config
            {
                drop(s);
                decoder = Some(Decoder::new(c.codec, self.device).map_err(|e| e.into_boxed_str())?);
                continue;
            }
            if !decoded.is_empty() && !s.free.is_empty() {
                let target = s.free.pop_front().unwrap();
                let frame = decoded.pop_front().unwrap();
                drop(s);
                if let Some((counter, value)) = &target.release {
                    let future = super::syncpt::CounterWait::new(counter.clone(), *value);
                    let mut future = std::pin::pin!(future);
                    let completed = self.wait(std::future::poll_fn(|cx| {
                        if self.state.lock().unwrap().reset.is_some() {
                            return Poll::Ready(false);
                        }
                        future.as_mut().poll(cx).map(|()| true)
                    }))?;
                    if !completed {
                        self.state.lock().unwrap().free.push_front(target);
                        decoded.push_front(frame);
                        continue;
                    }
                }
                if let Some(pixels) = &frame.frame {
                    self.publish(&target, pixels)?;
                }
                let l = target.image.pool.layout;
                self.state.lock().unwrap().outputs.push_back(Output {
                    image: target.image.id,
                    width: l.width,
                    height: l.height,
                    pts: frame.pts,
                    end: frame.end,
                });
                self.state.lock().unwrap().busy.remove(&target.image.id);
                self.complete(&target)?;
                continue;
            }
            if s.started
                && decoder.is_some()
                && decoded.len() < 2
                && let Some(input) = s.inputs.pop_front()
            {
                drop(s);
                let d = decoder.as_mut().unwrap();
                if !input.bytes.is_empty() {
                    d.send(&input.bytes, input.pts as u64)
                        .map_err(|e| e.into_boxed_str())?;
                }
                while let Some((origin, frame)) = d.receive().map_err(|e| e.into_boxed_str())? {
                    decoded.push_back(Decoded {
                        frame: Some(frame),
                        pts: origin as i64,
                        end: false,
                    });
                }
                if input.end {
                    d.finish().map_err(|e| e.into_boxed_str())?;
                    while let Some((origin, frame)) = d.receive().map_err(|e| e.into_boxed_str())? {
                        decoded.push_back(Decoded {
                            frame: Some(frame),
                            pts: origin as i64,
                            end: false,
                        });
                    }
                    if let Some(last) = decoded.back_mut() {
                        last.end = true;
                    } else {
                        decoded.push_back(Decoded {
                            frame: None,
                            pts: input.pts,
                            end: true,
                        });
                    }
                    // FFmpeg's drained state rejects further packets until a
                    // flush. Owned frames remain valid across the codec reset.
                    d.flush();
                }
                self.event.signal();
                continue;
            }
            drop(s);
            thread::park();
        }
    }
}
