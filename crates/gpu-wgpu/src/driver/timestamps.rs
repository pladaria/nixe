//! Optional bounded device intervals. Recording never waits for a query or slot.
use super::*;
const SLOTS: u32 = 128;
struct Slot {
    index: u32,
    readback: Buffer,
}
#[derive(Clone, Copy)]
struct Calibration {
    device: u64,
    host: u64,
}
pub(super) struct Timestamps {
    queries: wgpu::QuerySet,
    resolve: Buffer,
    available: Arc<Mutex<Vec<Slot>>>,
    calibration: Option<(u64, Option<Calibration>)>,
}
pub(super) struct Stamp {
    queries: wgpu::QuerySet,
    resolve: Buffer,
    slot: Slot,
    available: Arc<Mutex<Vec<Slot>>>,
    period: f64,
    calibration: Option<Calibration>,
    epoch: u64,
    id: u64,
    segment: u64,
}
impl Timestamps {
    pub(super) fn new(device: &Device) -> Self {
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Nixe timeline queries"),
            ty: wgpu::QueryType::Timestamp,
            count: SLOTS * 2,
        });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("Nixe timeline resolution"),
            size: u64::from(SLOTS) * 256,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let available = (0..SLOTS)
            .map(|index| Slot {
                index,
                readback: device.create_buffer(&BufferDescriptor {
                    label: Some("Nixe timeline readback"),
                    size: 16,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            })
            .collect();
        Self {
            queries,
            resolve,
            available: Arc::new(Mutex::new(available)),
            calibration: None,
        }
    }
    pub(super) fn begin(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        id: u64,
        segment: u64,
    ) -> Option<Stamp> {
        let Some(slot) = self.available.lock().unwrap().pop() else {
            nixe_trace::event("gpu.timestamp_slots_full", 0, 1);
            return None;
        };
        let epoch = nixe_trace::epoch();
        if self.calibration.is_none_or(|(old, _)| old != epoch) {
            self.calibration = Some((epoch, calibrate(device)));
        }
        encoder.write_timestamp(&self.queries, slot.index * 2);
        Some(Stamp {
            queries: self.queries.clone(),
            resolve: self.resolve.clone(),
            slot,
            available: self.available.clone(),
            period: f64::from(queue.get_timestamp_period()),
            calibration: self.calibration.unwrap().1,
            epoch,
            id,
            segment,
        })
    }
}
impl Stamp {
    pub(super) fn end(&self, encoder: &mut CommandEncoder) {
        encoder.write_timestamp(&self.queries, self.slot.index * 2 + 1);
        encoder.resolve_query_set(
            &self.queries,
            self.slot.index * 2..self.slot.index * 2 + 2,
            &self.resolve,
            u64::from(self.slot.index) * 256,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            u64::from(self.slot.index) * 256,
            &self.slot.readback,
            0,
            16,
        );
    }
    pub(super) fn map(self) {
        let readback = self.slot.readback.clone();
        readback.map_async(MapMode::Read, .., move |result| {
            if result.is_ok() {
                if let Ok(bytes) = self.slot.readback.get_mapped_range(..) {
                    let from = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
                    let to = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());
                    let start = self.calibration.and_then(|c| {
                        let delta = (i128::from(from) - i128::from(c.device)) as f64 * self.period;
                        let host = c.host as f64 + delta;
                        (host >= 0.0 && host <= u64::MAX as f64).then_some(host as u64)
                    });
                    nixe_trace::device_interval(
                        self.epoch,
                        self.id,
                        self.segment,
                        start,
                        (to.wrapping_sub(from) as f64 * self.period) as u64,
                    );
                }
                self.slot.readback.unmap();
            } else {
                nixe_trace::event("gpu.timestamp_map_failed", self.id, 1);
            }
            self.available.lock().unwrap().push(self.slot);
        });
    }
}
#[cfg(not(target_os = "macos"))]
fn calibrate(device: &Device) -> Option<Calibration> {
    // SAFETY: the HAL guard retains the imported device and its instance; this
    // read-only query does not access a queue or call back into wgpu.
    unsafe {
        let hal = device.as_hal::<wgpu::hal::api::Vulkan>()?;
        if !hal
            .enabled_device_extensions()
            .contains(&ash::ext::calibrated_timestamps::NAME)
        {
            return None;
        }
        let extension = ash::ext::calibrated_timestamps::Device::new(
            hal.shared_instance().raw_instance(),
            hal.raw_device(),
        );
        let before = nixe_trace::clock_ns();
        let (ticks, deviation) = extension
            .get_calibrated_timestamps(&[ash::vk::CalibratedTimestampInfoEXT::default()
                .time_domain(ash::vk::TimeDomainEXT::DEVICE)])
            .ok()?;
        let after = nixe_trace::clock_ns();
        nixe_trace::event(
            "gpu.calibration_error_ns",
            0,
            deviation.saturating_add((after - before).div_ceil(2)),
        );
        Some(Calibration {
            device: ticks[0],
            host: before + (after - before) / 2,
        })
    }
}
#[cfg(target_os = "macos")]
fn calibrate(_device: &Device) -> Option<Calibration> {
    None
}
