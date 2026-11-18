use super::*;
use bytemuck::{Pod, Zeroable};
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
    state: HashMap<u64, command::State>,
    syncs: HashMap<u64, Arc<Fence>>,
    windows: HashMap<u64, Window>,
}
struct Window {
    textures: Vec<u64>,
    next: usize,
    interval: u32,
    deadline: Option<Instant>,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct WindowBuilder {
    device: u64,
    native: u64,
    count: u32,
    interval: u32,
    textures: [u64; 4],
}
pub fn set_device(c: &Call) -> Result<u64> {
    c.write(c.args[0], &c.args[1])?;
    Ok(0)
}
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &0u32)?;
    Ok(0)
}
pub fn initialize(c: &Call) -> Result<u64> {
    c.session
        .queues
        .lock()
        .unwrap()
        .state
        .insert(c.args[0], Default::default());
    Ok(1)
}
pub fn window_defaults(c: &Call) -> Result<u64> {
    let b = WindowBuilder {
        interval: 1,
        ..Default::default()
    };
    c.write_bytes(c.args[0] + 8, &bytemuck::bytes_of(&b)[8..])?;
    Ok(0)
}
pub fn native_window(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &c.args[1])?;
    Ok(0)
}
pub fn textures(c: &Call) -> Result<u64> {
    let n = c.args[1] as u32 as usize;
    if !(1..=4).contains(&n) {
        return Err(format!("unsupported window texture count {n}").into());
    }
    let mut tex = [0u64; 4];
    c.read_bytes(c.args[2], bytemuck::cast_slice_mut(&mut tex[..n]))?;
    c.write(c.args[0] + 24, &tex)?;
    c.write(c.args[0] + 16, &(n as u32))?;
    Ok(0)
}
pub fn window_initialize(c: &Call) -> Result<u64> {
    let b: WindowBuilder = c.read(c.args[1])?;
    let textures = b
        .textures
        .get(..b.count as usize)
        .filter(|s| !s.is_empty())
        .ok_or("invalid window textures")?
        .to_vec();
    c.session.queues.lock().unwrap().windows.insert(
        c.args[0],
        Window {
            textures,
            next: 0,
            interval: b.interval,
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
    if c.args[2] != 0 || c.args[3] != 0 {
        return Err(format!(
            "unsupported fence condition {} or flags {:#x}",
            c.args[2], c.args[3]
        )
        .into());
    }
    let f = Arc::new(Fence::default());
    c.session.graphics()?.backend.fence(f.clone())?;
    c.session.queues.lock().unwrap().syncs.insert(c.args[1], f);
    Ok(0)
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
    c.session.graphics()?.backend.flush()?;
    Ok(0)
}
pub fn acquire(c: &Call) -> Result<u64> {
    let (index, deadline) = {
        let mut q = c.session.queues.lock().unwrap();
        let w = q.windows.get_mut(&c.args[1]).ok_or("unknown window")?;
        // Present submits the copy to an independently leased image on the same
        // graphics queue. Later writes to this slot already follow that copy.
        let index = w.next;
        w.next = (w.next + 1) % w.textures.len();
        // @SIMPLIFICATION Pace the default 60 Hz window at acquire. The host
        // compositor owns its independent copy and can finish it asynchronously.
        let now = Instant::now();
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
    let mut ops = Vec::new();
    {
        let mut queues = c.session.queues.lock().unwrap();
        let q = queues.state.get_mut(&c.args[0]).ok_or("unknown queue")?;
        for i in 0..c.args[1] as u32 {
            q.replay(c, c.read(c.args[2] + u64::from(i) * 8)?, &mut ops)?
        }
    }
    c.session.graphics()?.backend.submit(ops)?;
    Ok(0)
}
pub fn present(c: &Call) -> Result<u64> {
    let (texture, interval) = {
        let q = c.session.queues.lock().unwrap();
        let w = q.windows.get(&c.args[1]).ok_or("unknown window")?;
        (
            *w.textures
                .get(c.args[2] as u32 as usize)
                .ok_or("invalid window texture index")?,
            w.interval,
        )
    };
    c.session
        .graphics()?
        .backend
        .present(c.read(texture)?, interval)?;
    Ok(0)
}
