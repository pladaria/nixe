use super::*;

use nixe_scheduler::{EventObject, ReadableEventObject, SchedulerState};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[derive(Debug, Default)]
struct FenceState {
    result: Option<Result<()>>,
    event: Option<EventObject>,
}

#[derive(Debug, Default)]
pub struct Fence(Mutex<FenceState>);

impl Fence {
    pub fn signal(&self, result: Result<()>) {
        let event = {
            let mut state = self.0.lock().unwrap();
            state.result = Some(result);
            state.event.take()
        };
        if let Some(event) = event {
            event.signal();
        }
    }

    fn result(&self) -> Option<Result<()>> {
        self.0.lock().unwrap().result.clone()
    }

    /// Allocate a kernel event only when the calling guest thread must park.
    pub fn event(&self, scheduler: &Arc<SchedulerState>) -> Option<ReadableEventObject> {
        let mut state = self.0.lock().unwrap();
        if state.result.is_some() {
            return None;
        }
        Some(
            state
                .event
                .get_or_insert_with(|| EventObject::new(scheduler))
                .readable(),
        )
    }
}

#[derive(Debug)]
pub enum PendingWait {
    Acquire(Duration),
    Fence(Arc<Fence>, u64),
}

impl PendingWait {
    pub fn complete(self, timed_out: bool) -> Result<u64> {
        match self {
            Self::Acquire(_) => Ok(0),
            Self::Fence(_, _) if timed_out => Ok(2),
            Self::Fence(f, _) => f
                .result()
                .ok_or("fence woke before GPU completion")?
                .map(|()| 1),
        }
    }
}
#[derive(Default)]
pub(crate) struct Queues {
    state: HashMap<u64, Arc<Queue>>,
    syncs: HashMap<u64, Arc<Fence>>,
    windows: HashMap<u64, Window>,
}
struct Queue {
    upper_left: bool,
    minus_one_to_one: bool,
    capture: Mutex<Option<command::Capture>>,
}
fn queue(c: &Call) -> Result<Arc<Queue>> {
    c.session
        .queues
        .lock()
        .unwrap()
        .state
        .get(&c.args[0])
        .cloned()
        .ok_or_else(|| "unknown queue".into())
}
impl Queue {
    fn ordered<T>(&self, work: impl FnOnce(&mut command::Capture) -> Result<T>) -> Result<T> {
        let mut capture = self.capture.lock().unwrap();
        work(capture.as_mut().ok_or("queue finalized")?)
    }
}
struct Window {
    textures: Vec<u64>,
    next: usize,
    interval: u32,
    deadline: Option<Instant>,
}

fn native_present_interval(requested: u32) -> u32 {
    // The SDK keeps the requested signed interval in the window object.
    // Its native window paces and queues buffers with a minimum of one tick.
    (requested as i32).max(1) as u32
}

pub fn initialize(c: &Call) -> Result<u64> {
    let device: u64 = c.read(c.args[1])?;
    let [origin, depth]: [u32; 2] = c.read(device + 4)?;
    let queue = Arc::new(Queue {
        upper_left: origin == 1,
        minus_one_to_one: depth == 0,
        capture: Mutex::new(Some(command::Capture::default())),
    });
    c.session
        .queues
        .lock()
        .unwrap()
        .state
        .insert(c.args[0], queue);
    Ok(1)
}

pub fn finalize(c: &Call) -> Result<u64> {
    let queue = c
        .session
        .queues
        .lock()
        .unwrap()
        .state
        .remove(&c.args[0])
        .ok_or("unknown queue")?;
    let mut capture = queue.capture.lock().unwrap();
    capture.take();
    c.session.graphics()?.backend.finalize_queue(c.args[0])?;
    Ok(0)
}

pub fn window_initialize(c: &Call) -> Result<u64> {
    let b: WindowBuilder = c.read(c.args[1])?;
    let textures = b
        .textures
        .get(
            ..if b.active == 0 {
                b.count
            } else {
                b.active.min(b.count)
            } as usize,
        )
        .filter(|s| !s.is_empty())
        .ok_or("invalid window textures")?
        .to_vec();
    c.write(c.args[0], &[b.interval, 0, 0, 0, 0])?;
    c.session.queues.lock().unwrap().windows.insert(
        c.args[0],
        Window {
            textures,
            next: 0,
            interval: native_present_interval(b.interval),
            deadline: None,
        },
    );
    Ok(1)
}
pub fn sync_initialize(c: &Call) -> Result<u64> {
    c.session.queues.lock().unwrap().syncs.remove(&c.args[0]);
    Ok(1)
}
pub fn fence(c: &Call) -> Result<u64> {
    if c.args[2] != 0 || c.args[3] & !1 != 0 {
        return Err(format!(
            "unsupported fence condition {} or flags {:#x}",
            c.args[2], c.args[3]
        )
        .into());
    }
    queue(c)?.ordered(|_| {
        let fence = Arc::new(Fence::default());
        c.session
            .graphics()?
            .backend
            .fence(fence.clone(), c.args[3] & 1 != 0)?;
        c.session
            .queues
            .lock()
            .unwrap()
            .syncs
            .insert(c.args[1], fence);
        Ok(0)
    })
}
pub fn wait(c: &Call) -> Result<u64> {
    let f = c
        .session
        .queues
        .lock()
        .unwrap()
        .syncs
        .get(&c.args[0])
        .cloned();
    let Some(f) = f else { return Ok(0) };
    if let Some(result) = f.result() {
        return result.map(|()| 0);
    }
    if c.args[1] == 0 {
        return Ok(2);
    }
    c.pending.set(Some(PendingWait::Fence(f, c.args[1])));
    Ok(0)
}
pub fn flush(c: &Call) -> Result<u64> {
    queue(c)?.ordered(|_| c.session.graphics()?.backend.flush())?;
    Ok(0)
}
pub fn acquire(c: &Call) -> Result<u64> {
    acquire_slot(c, c.args[1])
}
pub fn window_acquire(c: &Call) -> Result<u64> {
    if c.args[1] != 0 {
        let ready = Arc::new(Fence::default());
        c.session.graphics()?.backend.acquire_fence(ready.clone())?;
        c.session
            .queues
            .lock()
            .unwrap()
            .syncs
            .insert(c.args[1], ready);
    }
    acquire_slot(c, c.args[0])
}
fn acquire_slot(c: &Call, window: u64) -> Result<u64> {
    let interval = native_present_interval(c.read(window)?);
    let (index, deadline) = {
        let mut q = c.session.queues.lock().unwrap();
        let w = q.windows.get_mut(&window).ok_or("unknown window")?;
        // Present submits the copy to an independently leased image on the same
        // graphics queue. Later writes to this slot already follow that copy.
        let index = w.next;
        w.next = (w.next + 1) % w.textures.len();
        // @SIMPLIFICATION Pace the default 60 Hz window at acquire. The host
        // compositor owns its independent copy and can finish it asynchronously.
        let now = Instant::now();
        if w.interval != interval {
            w.interval = interval;
            w.deadline = None;
        }
        let t = w.deadline.unwrap_or(now).max(now);
        w.deadline = Some(t + Duration::from_nanos(16_666_667 * u64::from(w.interval)));
        (index, t)
    };
    if let Some(delay) = deadline.checked_duration_since(Instant::now()) {
        c.pending.set(Some(PendingWait::Acquire(delay)));
    }
    c.write(c.args[2], &(index as i32))?;
    Ok(0)
}
pub fn submit(c: &Call) -> Result<u64> {
    let q = queue(c)?;
    q.ordered(|capture| {
        let bounds = c.session.pools.lock().unwrap().snapshot();
        let mut submission = Submission::new(q.upper_left, q.minus_one_to_one, bounds);
        let span = nixe_memory::trace::span("nvn_list_capture", c.args[1], 0, 0);
        for i in 0..c.args[1] as u32 {
            capture.list(c, c.read(c.args[2] + u64::from(i) * 8)?, &mut submission)?;
        }
        #[cfg(feature = "execution-trace")]
        capture.trace_submit(submission.block_count());
        drop(span);
        c.session.graphics()?.backend.submit(c.args[0], submission)
    })?;
    Ok(0)
}
pub fn present(c: &Call) -> Result<u64> {
    let [interval, x, y, width, height]: [u32; 5] = c.read(c.args[1])?;
    let (texture, queue) = {
        let q = c.session.queues.lock().unwrap();
        let w = q.windows.get(&c.args[1]).ok_or("unknown window")?;
        (
            *w.textures
                .get(c.args[2] as u32 as usize)
                .ok_or("invalid window texture index")?,
            q.state.get(&c.args[0]).cloned().ok_or("unknown queue")?,
        )
    };
    queue.ordered(|_| {
        c.session.graphics()?.backend.present(
            c.read(texture)?,
            native_present_interval(interval),
            !queue.upper_left,
            [x, y, width, height],
        )
    })?;
    Ok(0)
}
