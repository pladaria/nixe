//! Bounded, non-destructive history of completed render measurements.
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

/// About 68 seconds at 60 frames per second. Recording never grows this storage.
pub const FRAME_METRICS_CAPACITY: usize = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameTiming {
    /// Publication sequence, starting at one; use this as the reader's cursor.
    pub sequence: u64,
    /// Backend-local render identity; independent of completion/publication order.
    pub frame: u64,
    /// Native producer tick of the immutable resident image, scoped to this history.
    pub producer: u64,
    /// Guest-requested display refresh interval for this frame, not host WSI.
    pub swap_interval: u32,
    /// Monotonic time since this recorder was created.
    pub started: Duration,
    /// Elapsed graphics preparation time, including dependency waits, across all
    /// command lists consumed for this image. This is not summed thread CPU usage.
    pub cpu: Duration,
    /// Union of native submission timestamp intervals, including uploads and
    /// device dependencies. Not a hardware utilization or shader occupancy count.
    pub gpu: Option<Duration>,
    /// CPU start to the GPU timestamp of the resident image's completion.
    /// CPU/GPU overlap is preserved; this is not `cpu + gpu`. Excludes WSI/scanout.
    pub total: Option<Duration>,
    /// Upper uncertainty of GPU/CPU clock correlation, when available.
    pub clock_error: Option<Duration>,
}

/// Readers own their returned batch; no renderer or GPU resource is retained.
#[derive(Debug)]
pub struct FrameTimingBatch {
    pub samples: Vec<FrameTiming>,
    /// Published samples overwritten since the requested cursor.
    pub overwritten: u64,
    /// In-flight measurements discarded rather than delaying execution.
    pub dropped: u64,
}

struct History {
    samples: Box<[FrameTiming]>,
    published: u64,
    dropped: u64,
}

/// Clone this handle before moving the renderer to its owner thread. Publication
/// locks once per completed frame; reads never reset counters or consume history.
#[derive(Clone)]
pub struct FrameMetrics(Arc<Mutex<History>>);

impl Default for FrameMetrics {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(History {
            samples: vec![FrameTiming::default(); FRAME_METRICS_CAPACITY].into_boxed_slice(),
            published: 0,
            dropped: 0,
        })))
    }
}

impl FrameMetrics {
    pub fn publish(&self, mut sample: FrameTiming) {
        let mut history = self.0.lock().unwrap();
        history.published += 1;
        sample.sequence = history.published;
        let index = (sample.sequence - 1) as usize % FRAME_METRICS_CAPACITY;
        history.samples[index] = sample;
    }

    pub fn record_dropped(&self) {
        self.0.lock().unwrap().dropped += 1;
    }

    /// Copy only the latest completed sample, in O(1), without allocating or
    /// consuming history. No GPU operation or completion wait is performed.
    pub fn latest(&self) -> Option<FrameTiming> {
        let history = self.0.lock().unwrap();
        let sequence = history.published.checked_sub(1)?;
        Some(history.samples[sequence as usize % FRAME_METRICS_CAPACITY])
    }

    /// Return at most `limit` samples newer than `after`, oldest first. Advance
    /// the cursor to the last returned sequence. Zero starts at retained history.
    /// Allocation happens on the reader before acquiring the publication lock.
    pub fn read_since(&self, after: u64, limit: usize) -> FrameTimingBatch {
        let mut samples = Vec::with_capacity(limit.min(FRAME_METRICS_CAPACITY));
        let history = self.0.lock().unwrap();
        let oldest = history
            .published
            .saturating_sub(FRAME_METRICS_CAPACITY as u64);
        let first = after.max(oldest).min(history.published);
        let count = (history.published - first).min(limit.min(FRAME_METRICS_CAPACITY) as u64);
        for sequence in first + 1..=first + count {
            samples.push(history.samples[(sequence - 1) as usize % FRAME_METRICS_CAPACITY]);
        }
        FrameTimingBatch {
            samples,
            overwritten: oldest.saturating_sub(after),
            dropped: history.dropped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_readers_keep_cursors_across_wrap_and_report_overwritten_samples() {
        let metrics = FrameMetrics::default();
        assert_eq!(metrics.latest(), None);
        for frame in 1..=FRAME_METRICS_CAPACITY as u64 + 7 {
            metrics.publish(FrameTiming {
                frame,
                ..FrameTiming::default()
            });
        }
        let batch = metrics.read_since(0, 3);
        assert_eq!(batch.overwritten, 7);
        assert_eq!(
            batch.samples.iter().map(|s| s.frame).collect::<Vec<_>>(),
            [8, 9, 10]
        );
        let next = metrics.clone().read_since(10, usize::MAX);
        assert_eq!(next.samples.len(), FRAME_METRICS_CAPACITY - 3);
        assert_eq!(
            next.samples.last().unwrap().sequence,
            FRAME_METRICS_CAPACITY as u64 + 7
        );
        assert_eq!(metrics.read_since(0, 3).samples, batch.samples);
        assert_eq!(metrics.latest(), next.samples.last().copied());
        assert!(metrics.read_since(u64::MAX, 10).samples.is_empty());
        assert!(metrics.read_since(0, 0).samples.is_empty());
    }

    #[test]
    fn concurrent_readers_observe_whole_samples_without_consuming_history() {
        let metrics = FrameMetrics::default();
        let writer = metrics.clone();
        let thread = std::thread::spawn(move || {
            for frame in 1..=10_000 {
                writer.publish(FrameTiming {
                    frame,
                    cpu: Duration::from_nanos(frame),
                    ..FrameTiming::default()
                });
            }
        });
        while !thread.is_finished() {
            if let Some(sample) = metrics.latest() {
                assert_eq!(sample.sequence, sample.frame);
                assert_eq!(sample.cpu.as_nanos(), sample.frame as u128);
            }
            for sample in metrics.read_since(0, 64).samples {
                assert_eq!(sample.sequence, sample.frame);
                assert_eq!(sample.cpu.as_nanos(), sample.frame as u128);
            }
        }
        thread.join().unwrap();
        assert_eq!(metrics.read_since(9_999, 1).samples[0].frame, 10_000);
    }
}
