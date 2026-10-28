//! Owner-local in-flight measurements; only completed samples cross threads.
use super::timing::{Calibration, Interval};
use nixe_gpu::frame_metrics::{FrameMetrics, FrameTiming};
use std::time::{Duration, Instant};

const PENDING: usize = 128;

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
}

pub(crate) struct Recorder {
    pub history: FrameMetrics,
    origin: Instant,
    frames: Box<[Option<Frame>]>,
    next: u64,
    current: Option<u64>,
    pub pending_tick: Option<u64>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            history: FrameMetrics::default(),
            origin: Instant::now(),
            frames: (0..PENDING).map(|_| None).collect(),
            next: 1,
            current: None,
            pending_tick: None,
        }
    }
    pub fn begin_cpu(&mut self) {
        if self.current.is_some() {
            return;
        }
        let id = self.next;
        self.next += 1;
        let slot = &mut self.frames[id as usize % PENDING];
        if slot.is_some() {
            self.history.record_dropped();
        }
        *slot = Some(Frame {
            id,
            start: Instant::now(),
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
        });
        self.current = Some(id);
    }
    fn frame(&mut self, id: u64) -> Option<&mut Frame> {
        self.frames[id as usize % PENDING]
            .as_mut()
            .filter(|f| f.id == id)
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
        self.pending_tick = Some(tick);
        self.publish_ready(id, calibration);
    }
    pub(super) fn retire(
        &mut self,
        id: u64,
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
        self.history.publish(FrameTiming {
            sequence: 0,
            frame: id,
            producer: frame.producer,
            swap_interval: frame.swap_interval,
            started: frame.start.duration_since(self.origin),
            cpu: cpu_end.duration_since(frame.start),
            gpu: measured
                .then(|| Duration::from_nanos((frame.union_ticks as f64 * frame.period) as u64)),
            total: completion.and_then(|(end, _)| end.checked_duration_since(frame.start)),
            clock_error: completion.map(|(_, error)| error),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn interval(start: u64, end: u64) -> Interval {
        Interval {
            start,
            end,
            period: 1_000_000.0,
            mask: 255,
        }
    }
    #[test]
    fn completion_is_asynchronous_and_overlapping_intervals_are_unioned_across_wrap() {
        let mut recorder = Recorder::new();
        recorder.begin_cpu();
        let first = recorder.begin_submission().unwrap();
        let second = recorder.begin_submission().unwrap();
        recorder.retire(first, Some(interval(250, 10)), None);
        recorder.end_cpu(2, 1, None);
        assert!(recorder.history.read_since(0, 10).samples.is_empty());
        recorder.retire(second, Some(interval(5, 20)), None);
        let sample = recorder.history.read_since(0, 10).samples[0];
        assert_eq!(sample.gpu, Some(Duration::from_millis(26)));
        assert_eq!(sample.total, None);
        assert_eq!(sample.clock_error, None);
    }
    #[test]
    fn completion_before_cpu_seal_and_unsupported_timestamps_keep_cpu_measurements() {
        let mut recorder = Recorder::new();
        recorder.begin_cpu();
        let id = recorder.begin_submission().unwrap();
        recorder.retire(id, None, None);
        assert!(recorder.history.read_since(0, 10).samples.is_empty());
        recorder.end_cpu(1, 0, None);
        let samples = recorder.history.read_since(0, 10).samples;
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].gpu, None);
        assert_eq!(samples[0].total, None);
        assert_eq!(samples[0].swap_interval, 0);
    }
    #[test]
    fn stalled_gpu_cannot_grow_storage_or_overwrite_a_new_frame_with_an_old_result() {
        let mut recorder = Recorder::new();
        for tick in 1..=PENDING as u64 + 1 {
            recorder.begin_cpu();
            recorder.begin_submission();
            recorder.end_cpu(tick, 1, None);
        }
        recorder.retire(1, Some(interval(1, 2)), None);
        assert!(recorder.history.read_since(0, 1).samples.is_empty());
        recorder.retire(PENDING as u64 + 1, Some(interval(20, 30)), None);
        let batch = recorder.history.read_since(0, 1);
        assert_eq!(batch.dropped, 1);
        assert_eq!(batch.samples[0].frame, PENDING as u64 + 1);
        assert_eq!(batch.samples[0].gpu, Some(Duration::from_millis(10)));
    }

    #[test]
    fn separate_frames_keep_their_submissions_and_gpu_idle_gaps_are_excluded() {
        let mut recorder = Recorder::new();
        recorder.begin_cpu();
        let a = recorder.begin_submission().unwrap();
        recorder.begin_cpu();
        assert_eq!(recorder.begin_submission(), Some(a));
        recorder.end_cpu(2, 1, None);
        assert_eq!(recorder.begin_submission(), None);
        recorder.begin_cpu();
        let b = recorder.begin_submission().unwrap();
        recorder.end_cpu(3, 2, None);
        recorder.retire(a, Some(interval(10, 20)), None);
        recorder.retire(a, Some(interval(30, 40)), None);
        recorder.retire(b, Some(interval(50, 55)), None);
        let batch = recorder.history.read_since(0, 10);
        assert_eq!(batch.samples.len(), 2);
        assert_eq!(batch.samples[0].gpu, Some(Duration::from_millis(20)));
        assert_eq!(batch.samples[0].producer, 2);
        assert_eq!(batch.samples[0].swap_interval, 1);
        assert_eq!(batch.samples[1].gpu, Some(Duration::from_millis(5)));
        assert_eq!(batch.samples[1].producer, 3);
        assert_eq!(batch.samples[1].swap_interval, 2);
    }

    #[test]
    fn a_clock_that_may_have_wrapped_more_than_once_is_reported_as_unavailable() {
        let mut recorder = Recorder::new();
        recorder.origin -= Duration::from_secs(1);
        recorder.begin_cpu();
        let id = recorder.begin_submission().unwrap();
        recorder.frame(id).unwrap().start = recorder.origin;
        recorder.retire(id, Some(interval(250, 10)), None);
        recorder.end_cpu(1, 0, None);
        let sample = recorder.history.read_since(0, 1).samples[0];
        assert!(sample.cpu >= Duration::from_secs(1));
        assert_eq!(sample.gpu, None);
        assert_eq!(sample.total, None);
    }
}
