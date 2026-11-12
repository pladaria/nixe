//! Owner-local in-flight measurements; only completed samples cross threads.
use super::timing::{Calibration, Interval};
use nixe_gpu::frame_metrics::{FrameMetrics, FramePacing, FrameTiming};
use std::time::{Duration, Instant};

const PENDING: usize = 128;
const MAX_INTERVALS: usize = 4096;

struct Idle {
    start: Instant,
    end: Instant,
    submitted: u64,
    recorded: Option<Instant>,
}

#[derive(Default)]
struct Timeline {
    idle: Vec<Idle>,
    completions: Vec<(u64, u64)>,
}

struct Frame {
    id: u64,
    start: Instant,
    cpu_end: Option<Instant>,
    producer: u64,
    swap_interval: u32,
    outstanding: usize,
    gpu_available: bool,
    first_gpu: Option<u64>,
    end_gpu: u64,
    union_end: u64,
    union_ticks: u64,
    period: f64,
    mask: u64,
    pacing: Option<FramePacing>,
    timeline: Timeline,
    overflow: bool,
}

pub(crate) struct Recorder {
    pub history: FrameMetrics,
    origin: Instant,
    frames: Box<[Option<Frame>]>,
    timelines: Vec<Timeline>,
    next: u64,
    current: Option<u64>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            history: FrameMetrics::default(),
            origin: Instant::now(),
            frames: (0..PENDING).map(|_| None).collect(),
            timelines: Vec::new(),
            next: 1,
            current: None,
        }
    }
    pub(super) fn is_active(&self) -> bool {
        self.current.is_some()
    }
    pub fn begin_cpu(&mut self) {
        self.begin_cpu_at(Instant::now());
    }
    pub fn begin_cpu_at(&mut self, start: Instant) {
        if self.current.is_some() {
            return;
        }
        let id = self.next;
        self.next += 1;
        let slot = &mut self.frames[id as usize % PENDING];
        let mut timeline = if let Some(frame) = slot.take() {
            self.history.record_dropped();
            frame.timeline
        } else {
            self.timelines.pop().unwrap_or_default()
        };
        timeline.idle.clear();
        timeline.completions.clear();
        *slot = Some(Frame {
            id,
            start,
            cpu_end: None,
            producer: 0,
            swap_interval: 0,
            outstanding: 0,
            gpu_available: true,
            first_gpu: None,
            end_gpu: 0,
            union_end: 0,
            union_ticks: 0,
            period: 0.0,
            mask: 0,
            pacing: None,
            timeline,
            overflow: false,
        });
        self.current = Some(id);
    }
    fn frame(&mut self, id: u64) -> Option<&mut Frame> {
        self.frames[id as usize % PENDING]
            .as_mut()
            .filter(|f| f.id == id)
    }
    pub fn idle(
        &mut self,
        start: Instant,
        end: Instant,
        submitted: u64,
        recorded: Option<Instant>,
    ) {
        let Some(frame) = self.current.and_then(|id| self.frame(id)) else {
            return;
        };
        if frame.timeline.idle.len() == MAX_INTERVALS {
            frame.overflow = true;
            return;
        }
        frame.timeline.idle.push(Idle {
            start,
            end,
            submitted,
            recorded,
        });
    }
    pub fn composite(&mut self, pacing: FramePacing) {
        self.begin_cpu();
        self.frame(self.current.unwrap()).unwrap().pacing = Some(pacing);
    }
    pub fn begin_submission(&mut self) -> Option<u64> {
        let id = self.current?;
        self.frame(id)?.outstanding += 1;
        Some(id)
    }
    pub(super) fn end_cpu(
        &mut self,
        tick: u64,
        swap_interval: u32,
        calibration: Option<&Calibration>,
    ) {
        let Some(id) = self.current.take() else {
            return;
        };
        let frame = self.frame(id).unwrap();
        frame.cpu_end = Some(Instant::now());
        frame.producer = tick;
        frame.swap_interval = swap_interval;
        self.publish_ready(id, calibration);
    }
    pub(super) fn retire(
        &mut self,
        id: u64,
        tick: u64,
        interval: Option<Interval>,
        calibration: Option<&Calibration>,
    ) {
        let Some(frame) = self.frame(id) else { return };
        frame.outstanding -= 1;
        if let Some(interval) = interval {
            let first = *frame.first_gpu.get_or_insert(interval.start);
            let start = interval.start.wrapping_sub(first) & interval.mask;
            let end = interval.end.wrapping_sub(first) & interval.mask;
            // Queue order permits overlapping TOP/BOTTOM intervals. Count their
            // union, retaining idle gaps between submissions as gaps.
            frame.union_ticks += end.saturating_sub(start.max(frame.union_end));
            if end >= frame.union_end {
                frame.union_end = end;
                frame.end_gpu = interval.end;
            }
            if frame.timeline.completions.len() == MAX_INTERVALS {
                frame.overflow = true;
            } else {
                frame.timeline.completions.push((tick, frame.end_gpu));
            }
            frame.period = interval.period;
            frame.mask = interval.mask;
        } else {
            frame.gpu_available = false;
        }
        self.publish_ready(id, calibration);
    }
    fn publish_ready(&mut self, id: u64, calibration: Option<&Calibration>) {
        let Some(frame) = self.frame(id) else { return };
        if frame.cpu_end.is_none() || frame.outstanding != 0 {
            return;
        }
        let frame = self.frames[id as usize % PENDING].take().unwrap();
        if frame.overflow {
            self.history.record_dropped();
            self.timelines.push(frame.timeline);
            return;
        }
        let cpu_end = frame.cpu_end.unwrap();
        let measured = frame.gpu_available
            && frame.first_gpu.is_some()
            // A narrow device clock may have wrapped more than once while the
            // owner was stalled. Such intervals cannot be reconstructed exactly.
            && (frame.mask == u64::MAX
                || (frame.start.elapsed().as_nanos() as f64)
                    < (frame.mask as f64 + 1.0) * frame.period);
        let completion = measured
            .then(|| {
                calibration.and_then(|c| c.completion(frame.end_gpu, frame.period, frame.mask))
            })
            .flatten();
        let latency =
            completion.and_then(|(end, _)| end.max(cpu_end).checked_duration_since(frame.start));
        let mut cpu_vsync = Duration::ZERO;
        let mut total_vsync = latency.map(|_| Duration::ZERO);
        if let Some(pacing) = frame.pacing {
            let mut completions = frame.timeline.completions.iter().peekable();
            let mut prefix_end = None;
            for idle in &frame.timeline.idle {
                while let Some(&&(tick, end)) = completions.peek() {
                    if tick > idle.submitted {
                        break;
                    }
                    prefix_end = Some(end);
                    completions.next();
                }
                let start = idle.start.max(frame.start).max(pacing.queued);
                let end = idle.end.min(pacing.vsync);
                let Some(wait) = end.checked_duration_since(start) else {
                    continue;
                };
                cpu_vsync += wait;
                if wait.is_zero() {
                    continue;
                }
                // A queued native job still recording at the end of this idle
                // interval keeps the interval on the critical path.
                let Some(recorded) = idle.recorded else {
                    continue;
                };
                if let (Some(excluded), Some((gpu_end, error))) = (&mut total_vsync, completion) {
                    let ready = if let Some(prefix) = prefix_end {
                        let ticks = frame.end_gpu.wrapping_sub(prefix) & frame.mask;
                        gpu_end
                            .checked_sub(Duration::from_nanos((ticks as f64 * frame.period) as u64))
                            .and_then(|ready| ready.checked_add(error))
                    } else {
                        Some(frame.start)
                    };
                    if let Some(ready) = ready {
                        // Keep worker backlog, submission waits and GPU work
                        // that overlaps the owner's deliberate display wait.
                        *excluded += end.saturating_duration_since(start.max(ready).max(recorded));
                    } else {
                        total_vsync = None;
                    }
                }
            }
        }
        self.history.publish(FrameTiming {
            sequence: 0,
            frame: id,
            producer: frame.producer,
            swap_interval: frame.swap_interval,
            started: frame.start.duration_since(self.origin),
            cpu: cpu_end
                .duration_since(frame.start)
                .saturating_sub(cpu_vsync),
            cpu_vsync,
            gpu: measured
                .then(|| Duration::from_nanos((frame.union_ticks as f64 * frame.period) as u64)),
            total: latency
                .zip(total_vsync)
                .and_then(|(latency, pacing)| latency.checked_sub(pacing)),
            total_vsync,
            latency,
            clock_error: completion.map(|(_, error)| error),
        });
        self.timelines.push(frame.timeline);
    }
}
