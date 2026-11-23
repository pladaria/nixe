//! Frame history for the NVN2 display, written by the ordered presentation owner.
use super::*;
use nixe_scheduler::VirtualClock;

const HISTORY_ENTRIES: usize = 8;
type FrameInfo = [u64; 5];

#[derive(Default)]
struct History {
    latest: u64,
    frames: [FrameInfo; HISTORY_ENTRIES],
}

pub struct DisplayHistory {
    clock: VirtualClock,
    history: Mutex<History>,
}

impl DisplayHistory {
    pub fn new(clock: VirtualClock) -> Self {
        Self {
            clock,
            history: Mutex::default(),
        }
    }

    fn tick(&self) -> u64 {
        (self.clock.elapsed_ns() / 32).max(1)
    }

    pub fn queue(self: &Arc<Self>) -> PresentReceipt {
        let mut history = self.history.lock().unwrap();
        history.latest += 1;
        let number = history.latest;
        history.frames[slot(number)] = [1, number, self.tick(), 0, 0];
        PresentReceipt {
            display: self.clone(),
            number,
        }
    }
}

fn slot(number: u64) -> usize {
    number as usize % HISTORY_ENTRIES
}

pub struct PresentReceipt {
    display: Arc<DisplayHistory>,
    number: u64,
}

impl PresentReceipt {
    pub fn complete(self) {
        let mut history = self.display.history.lock().unwrap();
        let frame = &mut history.frames[slot(self.number)];
        if frame[1] == self.number {
            // @SIMPLIFICATION Both display checkpoints use the completion tick
            // of the virtual compositor, which owns the guest presentation.
            let tick = self.display.tick();
            frame[0] = 2;
            frame[3] = tick;
            frame[4] = tick;
        }
    }
}

fn history<'a>(c: &Call<'a>) -> Result<&'a DisplayHistory> {
    c.session
        .graphics()?
        .display
        .as_deref()
        .ok_or_else(|| "NVN2 display history is not attached".into())
}

// Both SDK VI exports query the same native display controller. The Layer
// argument is ignored by the SDK wrappers for this controller.
pub fn latest(c: &Call) -> Result<u64> {
    let number = history(c)?.history.lock().unwrap().latest;
    c.write(c.args[0], &number)?;
    Ok(0)
}

pub fn list(c: &Call) -> Result<u64> {
    let capacity = c.args[1] as i32;
    if c.args[0] == 0 || capacity <= 0 {
        return Ok(HISTORY_ENTRIES as u64);
    }
    // FrameInfo consists of status, frame number, the queue tick and two
    // display checkpoints. A completed compositor handoff supplies both
    // checkpoints even if the host window never shows the image.
    let mut frames = [[0u64; 5]; HISTORY_ENTRIES];
    let count = {
        let history = history(c)?.history.lock().unwrap();
        let count = history
            .latest
            .min(HISTORY_ENTRIES as u64)
            .min(capacity as u64) as usize;
        for (i, frame) in frames[..count].iter_mut().enumerate() {
            *frame = history.frames[slot(history.latest - i as u64)];
        }
        count
    };
    c.write_bytes(c.args[0], bytemuck::cast_slice(&frames[..count]))?;
    Ok(count as u64)
}
