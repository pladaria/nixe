//! Submission timestamps retire with their command slot. No GPU waits are added.
use super::*;
use nixe_gpu::metrics::{Counter, record};

#[derive(Clone, Copy)]
pub(super) struct Interval {
    pub start: u64,
    pub end: u64,
    pub period: f64,
    pub mask: u64,
}

/// One clock bridge per device queue, shared by all timestamp pools.
pub(super) struct Calibration {
    device: ash::ext::calibrated_timestamps::Device,
    instant: Instant,
    monotonic_ns: u64,
    error: Duration,
}
impl Calibration {
    pub fn new(owner: &DeviceOwner) -> Result<Option<Self>, VulkanError> {
        if !owner.calibrated_timestamps {
            return Ok(None);
        }
        let instance = ash::ext::calibrated_timestamps::Instance::new(
            &owner.instance.entry,
            &owner.instance.instance,
        );
        let domains =
            unsafe { instance.get_physical_device_calibrateable_time_domains(owner.physical) }
                .map_err(|e| VulkanError::vk("calibrated timestamp domains", e))?;
        if !domains.contains(&vk::TimeDomainEXT::CLOCK_MONOTONIC) {
            return Ok(None);
        }
        let before = Instant::now();
        let Some(monotonic_ns) = monotonic_ns() else {
            return Ok(None);
        };
        let after = Instant::now();
        Ok(Some(Self {
            device: ash::ext::calibrated_timestamps::Device::new(
                &owner.instance.instance,
                &owner.device,
            ),
            instant: before + after.duration_since(before) / 2,
            monotonic_ns,
            error: after.duration_since(before),
        }))
    }
    fn sample(&self) -> Result<([u64; 2], u64), VulkanError> {
        // Fixed arrays avoid the allocating ash convenience wrapper. Both
        // clock domains are supported and belong to this live logical device.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkGetCalibratedTimestampsKHR.html
        let domains = [
            vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE),
            vk::CalibratedTimestampInfoEXT::default()
                .time_domain(vk::TimeDomainEXT::CLOCK_MONOTONIC),
        ];
        let mut sample = [0; 2];
        let mut deviation = 0;
        unsafe {
            (self.device.fp().get_calibrated_timestamps_ext)(
                self.device.device(),
                2,
                domains.as_ptr(),
                sample.as_mut_ptr(),
                &mut deviation,
            )
            .result()
            .map_err(|e| VulkanError::vk("calibrated queue timestamps", e))?;
        }
        Ok((sample, deviation))
    }
    fn host_ns(tick: u64, sample: [u64; 2], period: f64, mask: u64) -> Option<u64> {
        let age = sample[0].wrapping_sub(tick) & mask;
        sample[1].checked_sub((age as f64 * period) as u64)
    }
    pub fn completion(&self, tick: u64, period: f64, mask: u64) -> Option<(Instant, Duration)> {
        let (sample, deviation) = self.sample().ok()?;
        let host = Self::host_ns(tick, sample, period, mask)?;
        let instant = if host >= self.monotonic_ns {
            self.instant
                .checked_add(Duration::from_nanos(host - self.monotonic_ns))?
        } else {
            self.instant
                .checked_sub(Duration::from_nanos(self.monotonic_ns - host))?
        };
        Some((
            instant,
            self.error
                + Duration::from_nanos(deviation)
                + Duration::from_nanos(period.ceil() as u64),
        ))
    }
}

fn monotonic_ns() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, time.as_mut_ptr()) } != 0 {
            return None;
        }
        let time = unsafe { time.assume_init() };
        Some(time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

pub(super) struct Queries {
    owner: Arc<DeviceOwner>,
    pool: vk::QueryPool,
    period: f64,
    mask: u64,
    pending: bool,
}
impl Queries {
    pub fn new(owner: Arc<DeviceOwner>, family: u32) -> Result<Option<Self>, VulkanError> {
        let families = unsafe {
            owner
                .instance
                .instance
                .get_physical_device_queue_family_properties(owner.physical)
        };
        let bits = families[family as usize].timestamp_valid_bits;
        if bits == 0 {
            return Ok(None);
        }
        let properties = unsafe {
            owner
                .instance
                .instance
                .get_physical_device_properties(owner.physical)
        };
        let pool = unsafe {
            owner.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("queue timing query pool", e))?;
        Ok(Some(Self {
            owner,
            pool,
            period: f64::from(properties.limits.timestamp_period),
            mask: u64::MAX >> (64 - bits),
            pending: false,
        }))
    }
    pub fn begin(&mut self, dispatch: &mut crate::dispatch::Dispatch, enabled: bool) {
        self.pending = enabled;
        if enabled {
            dispatch.begin_submission_timing(self.pool);
        }
    }
    pub fn enable(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        if !self.pending {
            self.begin(dispatch, true);
        }
    }
    pub fn end(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        if !self.pending {
            return;
        }
        unsafe {
            dispatch.cmd_write_timestamp(vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, 1);
        }
    }
    pub fn retire(
        &mut self,
        tick: u64,
        calibration: Option<&Calibration>,
    ) -> Result<Option<Interval>, VulkanError> {
        if !self.pending {
            return Ok(None);
        }
        let mut ticks = [0u64; 2];
        // The slot timeline signal proves both writes completed. Never use WAIT.
        unsafe {
            self.owner.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .map_err(|e| VulkanError::vk("completed queue timestamps", e))?;
        let elapsed = ticks[1].wrapping_sub(ticks[0]) & self.mask;
        record(
            Counter::HostQueueGpuNanoseconds,
            (elapsed as f64 * self.period) as u64,
        );
        record(Counter::HostQueueTimedSubmissions, 1);
        if nixe_gpu::trace::active()
            && let Some(calibration) = calibration
        {
            // Correlate completed device queries with CLOCK_MONOTONIC, retaining
            // the driver's uncertainty. No queue wait or global idle is added.
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkGetCalibratedTimestampsKHR.html
            let (sample, deviation) = calibration.sample()?;
            if let [Some(start), Some(end)] =
                ticks.map(|value| Calibration::host_ns(value, sample, self.period, self.mask))
            {
                nixe_gpu::trace::event("gpu_execution", tick, start, end);
                nixe_gpu::trace::event("gpu_calibration_error", tick, deviation, 0);
            }
        }
        self.pending = false;
        Ok(Some(Interval {
            start: ticks[0],
            end: ticks[1],
            period: self.period,
            mask: self.mask,
        }))
    }
}
impl Drop for Queries {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_query_pool(self.pool, None);
        }
    }
}
