//! Optional aggregate counter capture, excluded from production builds.
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

pub(super) struct Capture {
    stop: mpsc::Sender<()>,
    worker: thread::JoinHandle<io::Result<()>>,
}
impl Capture {
    pub(super) fn start(memory: nixe_memory::Machine) -> io::Result<Option<Self>> {
        let Some(path) = std::env::var_os("NIXE_PERFORMANCE_COUNTERS") else {
            return Ok(None);
        };
        let started = Instant::now();
        nixe_memory::trace::event(
            "performance_clock_sync",
            started.elapsed().as_nanos() as u64,
            0,
            0,
        );
        if let Some(path) = std::env::var_os("NIXE_PERFORMANCE_FRAME_TIMINGS") {
            nixe_video_winit::metrics::start_frame_capture(std::path::Path::new(&path), started)?;
        }
        let page_report =
            std::env::var_os("NIXE_PERFORMANCE_MEMORY_REPORT").map(std::path::PathBuf::from);
        let mut output = BufWriter::new(File::create(path)?);
        let (stop, receiver) = mpsc::channel();
        let counters = move || {
            let state = memory.statistics();
            let tracking = memory.tracking_statistics();
            let mut values = vec![
                ("PhysicalReservedBytes", state.physical_capacity),
                ("PhysicalCommittedBytes", memory.arena().committed_bytes()?),
                ("VirtualReservedBytes", state.virtual_reserved_bytes),
                ("PageTableReservedBytes", state.page_table_reserved_bytes),
                ("PhysicalAllocatedBytes", state.allocated_bytes),
                ("GuestOwnedBytes", state.guest_owned_bytes),
                ("RetainedPhysicalBytes", state.retained_bytes),
                ("LivePhysicalAllocations", state.live_allocations as u64),
                ("TrackingBitmapBytes", tracking.bitmap_bytes as u64),
                ("TrackingDirtyQueueBytes", tracking.dirty_queue_bytes as u64),
                ("ObserverPages", tracking.observer_pages as u64),
                (
                    "ObserverRegistrations",
                    tracking.observer_registrations as u64,
                ),
                ("ProtectionCalls", tracking.protection_calls),
                ("CapturedPhysicalBytes", tracking.captured_bytes),
            ];
            values.extend(nixe_gpu::metrics::snapshot());
            values.extend(nixe_cpu_jit::metrics::snapshot());
            values.extend(nixe_audio::metrics::snapshot());
            values.extend(nixe_video_winit::metrics::snapshot());
            Ok::<_, io::Error>(values)
        };
        write!(output, "seconds")?;
        for (name, _) in counters()? {
            write!(output, ",{name}")?;
        }
        for bin in 0..nixe_video_winit::metrics::histogram().len() {
            write!(output, ",FrameIntervalBin{bin}")?;
        }
        writeln!(output)?;
        let worker = thread::spawn(move || {
            loop {
                let stop = !matches!(
                    receiver.recv_timeout(Duration::from_secs(1)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );
                write!(output, "{:.6}", started.elapsed().as_secs_f64())?;
                for (_, value) in counters()? {
                    write!(output, ",{value}")?;
                }
                for value in nixe_video_winit::metrics::histogram() {
                    write!(output, ",{value}")?;
                }
                writeln!(output)?;
                if stop {
                    output.flush()?;
                    return page_report.map_or(Ok(()), |path| {
                        let mut report = BufWriter::new(File::create(path)?);
                        writeln!(report, "{{")?;
                        let values = counters()?;
                        for (index, (name, value)) in values.iter().enumerate() {
                            writeln!(
                                report,
                                "  \"{name}\": {value}{}",
                                if index + 1 == values.len() { "" } else { "," }
                            )?;
                        }
                        writeln!(report, "}}")?;
                        report.flush()
                    });
                }
            }
        });
        Ok(Some(Self { stop, worker }))
    }
    pub(super) fn finish(self) -> io::Result<()> {
        let frames = nixe_video_winit::metrics::finish_frame_capture();
        let _ = self.stop.send(());
        let counters = self
            .worker
            .join()
            .map_err(|_| io::Error::other("performance capture worker panicked"))?;
        frames?;
        nixe_gpu::compilation_capture::finish()?;
        counters
    }
}
