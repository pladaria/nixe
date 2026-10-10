//! Standalone allocation/retention workload; measurements stay outside production.
use nixe_memory::{
    CanonicalBackingPage, CanonicalBackingStore, ContentGeneration, GuestPhysicalPageId,
};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "startup".into());
    let count: u64 = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "8192".into())
        .parse()?;
    if !["startup", "churn"].contains(&mode.as_str()) || count == 0 {
        return Err("expected startup|churn positive-count".into());
    }
    let start = Instant::now();
    let store = CanonicalBackingStore::allocate()?;
    let mut retained = Vec::new();
    for index in 0..count {
        let page = CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(index + 1),
            4096,
            ContentGeneration::INITIAL,
        )?;
        let host = page.direct_backing()?;
        // The workload owns these bytes exclusively; no guest or GPU executes.
        unsafe {
            let first = host.base() as *mut u8;
            assert_eq!(first.read_volatile(), 0);
            assert_eq!(first.add(4095).read_volatile(), 0);
            first.write_volatile(0x5a);
            first.add(4095).write_volatile(0xa5);
        }
        if mode == "startup" {
            retained.push(page);
        }
    }
    let elapsed = start.elapsed().as_nanos();
    println!("mode={mode} count={count} elapsed_ns={elapsed}");
    for line in std::fs::read_to_string("/proc/self/status")?.lines() {
        if line.starts_with("VmRSS:") || line.starts_with("VmSize:") {
            println!("{line}");
        }
    }
    println!(
        "vmas={}",
        std::fs::read_to_string("/proc/self/maps")?.lines().count()
    );
    #[cfg(feature = "performance-counters")]
    for (name, value) in nixe_memory::metrics::snapshot() {
        if name.starts_with("Backing") {
            println!("{name}={value}");
        }
    }
    std::hint::black_box(&retained);
    Ok(())
}
