//! Optional, asynchronous GPU samples for local performance investigations.
use super::*;
use std::io::{BufWriter, Write};
use std::time::Instant;

pub(super) struct Timings {
    output: Arc<Mutex<BufWriter<std::fs::File>>>,
    started: Instant,
    period: f64,
    sequence: u64,
    pending: Vec<Sample>,
}

pub(super) struct Sample {
    queries: wgpu::QuerySet,
    resolve: Buffer,
    readback: Buffer,
    seconds: f64,
    kind: &'static str,
    extent: Option<(u32, u32)>,
}

impl Timings {
    pub(super) fn new(device: &Device, queue: &Queue) -> Option<Self> {
        let path = std::env::var_os("NIXE_GPU_TIMINGS")?;
        let features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(features) {
            log::warn!("GPU timestamps are unavailable on this device");
            return None;
        }
        let file = match std::fs::File::create(path) {
            Ok(file) => file,
            Err(error) => {
                log::warn!("cannot create GPU timing output: {error}");
                return None;
            }
        };
        let mut output = BufWriter::new(file);
        writeln!(
            output,
            "seconds,kind,upload_ns,commands_ns,width,height,format"
        )
        .ok()?;
        Some(Self {
            output: Arc::new(Mutex::new(output)),
            started: Instant::now(),
            period: f64::from(queue.get_timestamp_period()),
            sequence: 0,
            pending: Vec::new(),
        })
    }

    pub(super) fn begin(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        submission: &nixe_gpu::OperationSubmission,
    ) -> Option<Sample> {
        self.sequence += 1;
        let kind = if submission
            .operations()
            .iter()
            .any(|operation| matches!(operation.command(), GpuCommand::ConvertNv12(_)))
        {
            "nv12"
        } else if submission
            .operations()
            .iter()
            .any(|operation| matches!(operation.command(), GpuCommand::UploadImage { .. }))
        {
            "image-upload"
        } else {
            // Bound profiling overhead for the many small Maxwell segments.
            if !self.sequence.is_multiple_of(16) {
                return None;
            }
            "graphics-sample"
        };
        let extent = submission.operations().iter().find_map(|operation| {
            let extent = match operation.command() {
                GpuCommand::ConvertNv12(conversion) => conversion.destination.extent,
                GpuCommand::UploadImage { destination, .. } => destination.extent,
                _ => return None,
            };
            Some((extent.width, extent.height))
        });
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Nixe GPU timing sample"),
            ty: wgpu::QueryType::Timestamp,
            count: 3,
        });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("Nixe GPU timing resolve"),
            size: 24,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("Nixe GPU timing readback"),
            size: 24,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.write_timestamp(&queries, 0);
        Some(Sample {
            queries,
            resolve,
            readback,
            seconds: self.started.elapsed().as_secs_f64(),
            kind,
            extent,
        })
    }

    pub(super) fn finish(&mut self, sample: Sample, encoder: &mut CommandEncoder) {
        encoder.write_timestamp(&sample.queries, 2);
        encoder.resolve_query_set(&sample.queries, 0..3, &sample.resolve, 0);
        encoder.copy_buffer_to_buffer(&sample.resolve, 0, &sample.readback, 0, 24);
        self.pending.push(sample);
    }

    pub(super) fn submitted(&mut self) {
        for sample in self.pending.drain(..) {
            let output = Arc::clone(&self.output);
            let period = self.period;
            let mapped = sample.readback.clone();
            sample.readback.map_async(MapMode::Read, .., move |result| {
                if let Err(error) = result {
                    log::warn!("GPU timing map failed: {error}");
                    return;
                }
                let bytes = mapped.get_mapped_range(..).expect("completed timing map");
                let ticks = bytes
                    .chunks_exact(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let mut output = output.lock().unwrap();
                let _ = writeln!(
                    output,
                    "{:.6},{},{:.0},{:.0},{},{},",
                    sample.seconds,
                    sample.kind,
                    ticks[1].wrapping_sub(ticks[0]) as f64 * period,
                    ticks[2].wrapping_sub(ticks[1]) as f64 * period,
                    sample
                        .extent
                        .map(|extent| extent.0.to_string())
                        .unwrap_or_default(),
                    sample
                        .extent
                        .map(|extent| extent.1.to_string())
                        .unwrap_or_default(),
                );
                drop(bytes);
                mapped.unmap();
            });
        }
    }

    pub(super) fn image_readback(&self, description: ImageDescription) {
        let _ = writeln!(
            self.output.lock().unwrap(),
            "{:.6},image-readback,,,{},{},{:?}",
            self.started.elapsed().as_secs_f64(),
            description.extent().width,
            description.extent().height,
            description.format()
        );
    }

    pub(super) fn readback_request(&self, caller: &'static std::panic::Location<'static>) {
        let _ = writeln!(
            self.output.lock().unwrap(),
            "{:.6},visibility-request,,,,,{}:{}",
            self.started.elapsed().as_secs_f64(),
            caller.file(),
            caller.line()
        );
    }
}

impl Sample {
    pub(super) fn uploaded(&self, encoder: &mut CommandEncoder) {
        encoder.write_timestamp(&self.queries, 1);
    }
}
