use std::{
    arch::x86_64::{__cpuid, _mm_lfence, _rdtsc},
    sync::OnceLock,
    time::Duration,
};

const NANOS_PER_SECOND: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug)]
pub(super) struct Scale {
    frequency: u64,
    nanoseconds_per_tick: u64,
}

impl Scale {
    pub(super) fn host() -> Option<Self> {
        static HOST: OnceLock<Option<Scale>> = OnceLock::new();
        *HOST.get_or_init(Self::detect)
    }

    fn detect() -> Option<Self> {
        // Linux qualifies the TSC across CPUs before using it as its clocksource.
        // Invariance alone only describes the rate on each CPU.
        // https://www.kernel.org/doc/Documentation/virt/kvm/x86/timekeeping.rst
        let clocksource = std::fs::read_to_string(
            "/sys/devices/system/clocksource/clocksource0/current_clocksource",
        )
        .ok()?;
        if clocksource.trim() != "tsc" {
            return None;
        }
        let cpu = __cpuid(1);
        let extended = __cpuid(0x8000_0000).eax;
        if cpu.edx & (1 << 4) == 0
            || cpu.ecx & (1 << 31) != 0
            || extended < 0x8000_0007
            || __cpuid(0x8000_0007).edx & (1 << 8) == 0
        {
            return None;
        }
        let frequency = crystal_frequency().or_else(measure_frequency)?;
        Some(Self {
            frequency,
            nanoseconds_per_tick: multiplier(NANOS_PER_SECOND, frequency)?,
        })
    }

    pub(super) fn multiplier(self, frequency: u64) -> Option<u64> {
        multiplier(frequency, self.frequency)
    }

    pub(super) fn nanoseconds(self, ticks: u64) -> u64 {
        ((u128::from(ticks) * u128::from(self.nanoseconds_per_tick)) >> 64) as u64
    }
}

fn multiplier(output_frequency: u64, input_frequency: u64) -> Option<u64> {
    let scale = (u128::from(output_frequency) << 64) / u128::from(input_frequency);
    u64::try_from(scale).ok().filter(|scale| *scale != 0)
}

fn crystal_frequency() -> Option<u64> {
    if __cpuid(0).eax < 0x15 {
        return None;
    }
    let ratio = __cpuid(0x15);
    let crystal = u64::from(ratio.ecx);
    if crystal == 0 || ratio.ebx == 0 || ratio.eax == 0 {
        return None;
    }
    Some(crystal * u64::from(ratio.ebx) / u64::from(ratio.eax))
}

fn measure_frequency() -> Option<u64> {
    let (first_ns, first_tsc) = sample_reference()?;
    std::thread::sleep(Duration::from_millis(25));
    let (last_ns, last_tsc) = sample_reference()?;
    let nanos = last_ns.checked_sub(first_ns)?;
    let ticks = last_tsc.checked_sub(first_tsc)?;
    let scaled = u128::from(ticks) * u128::from(NANOS_PER_SECOND);
    u64::try_from(scaled.checked_div(u128::from(nanos))?).ok()
}

fn sample_reference() -> Option<(u64, u64)> {
    let mut best = None;
    let mut narrowest = u64::MAX;
    for _ in 0..8 {
        let mut reference = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let start = read();
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut reference) } != 0 {
            return None;
        }
        let width = read().checked_sub(start)?;
        if width < narrowest {
            narrowest = width;
            let nanos = reference.tv_sec as u64 * NANOS_PER_SECOND + reference.tv_nsec as u64;
            best = Some((nanos, start + width / 2));
        }
    }
    best
}

#[inline]
pub(super) fn read() -> u64 {
    // Use the same ordered sample in the scheduler and the native emitter.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/x64/rdtsc.h
    unsafe {
        _mm_lfence();
        let ticks = _rdtsc();
        _mm_lfence();
        ticks
    }
}
