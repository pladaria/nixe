use nixe_gpu::cache_counters::CacheCounters;
use std::fmt::Write;

/// Periodic host diagnostics retain counters only, never native resources.
pub struct CacheMonitor {
    caches: [(&'static str, CacheCounters); 5],
    previous: [(u64, u64); 5],
}

impl CacheMonitor {
    pub(crate) fn new(caches: [(&'static str, CacheCounters); 5]) -> Self {
        Self {
            caches,
            previous: [(0, 0); 5],
        }
    }

    /// `new` and `evicted` count work since this monitor's previous sample.
    /// A miss may be a first use; it is not proof of a recompilation.
    pub fn sample(&mut self) -> String {
        let mut line = String::from("GPU caches: ");
        for (index, ((name, counters), previous)) in
            self.caches.iter().zip(&mut self.previous).enumerate()
        {
            let now = counters.snapshot();
            if index != 0 {
                line.push_str("; ");
            }
            write!(line, "{name}: {}", now.entries).unwrap();
            if now.entry_limit != 0 {
                write!(line, "/{}", now.entry_limit).unwrap();
            }
            if now.byte_limit != 0 {
                write!(
                    line,
                    " ({:.1}/{:.1} MiB)",
                    now.bytes as f64 / (1024. * 1024.),
                    now.byte_limit as f64 / (1024. * 1024.),
                )
                .unwrap();
            }
            write!(
                line,
                " [new={}, evicted={}]",
                now.misses.wrapping_sub(previous.0),
                now.evictions.wrapping_sub(previous.1),
            )
            .unwrap();
            *previous = (now.misses, now.evictions);
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_show_current_occupancy_and_interval_work() {
        let counters = CacheCounters::new(4096, 64 * 1024 * 1024);
        let mut monitor =
            CacheMonitor::new(std::array::from_fn(|_| ("shader cache", counters.clone())));
        counters.publish_usage(123, 2 * 1024 * 1024);
        counters.miss();
        counters.evict(3);
        let first = monitor.sample();
        assert!(
            first.starts_with(
                "GPU caches: shader cache: 123/4096 (2.0/64.0 MiB) [new=1, evicted=3]"
            )
        );
        let second = monitor.sample();
        assert!(second.contains("shader cache: 123/4096 (2.0/64.0 MiB) [new=0, evicted=0]"));
        assert!(!second.contains('\n'));
    }
}
