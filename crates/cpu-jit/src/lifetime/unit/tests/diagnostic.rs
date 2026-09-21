use super::*;
use std::cell::RefCell;

// Capture only this test's thread; unrelated parallel publications are ignored.
thread_local! {
    static CAPTURE: RefCell<Option<(Arc<Lifetime>, Vec<String>)>> = const { RefCell::new(None) };
}

struct Logger;
impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() == log::Level::Debug
    }

    fn log(&self, record: &log::Record<'_>) {
        CAPTURE.with_borrow_mut(|capture| {
            if let Some((process, lines)) = capture {
                assert!(process.state.try_lock().is_ok(), "logging under JIT state");
                assert!(
                    process.cache.try_usage().unwrap().is_some(),
                    "logging under cache lock"
                );
                assert_eq!(record.level(), log::Level::Debug);
                lines.push(record.args().to_string());
            }
        });
    }

    fn flush(&self) {}
}

#[test]
fn hcq_promotion_diagnostic_is_once_per_successful_unit_outside_state() {
    logger();
    let process = process();
    CAPTURE.set(Some((process.clone(), Vec::new())));
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let publications = [
        process.reserve(key(0)).unwrap(),
        process.reserve(key(4)).unwrap(),
    ];
    let first = process
        .prepare_unit(&publications, input(&process, &[0, 4], Tier::Hcq), &cursor)
        .unwrap();
    let stale = process
        .prepare_unit(&publications, input(&process, &[0, 4], Tier::Hcq), &cursor)
        .unwrap();
    let family = first.family.as_ref().unwrap();
    let expected = format!(
        "HCQ promotion: family={} version={} address_space=1 seed=0x0 instructions=2 entries=2 native_bytes=16",
        family.id.get(),
        family.version.get()
    );
    first.publish().unwrap();
    assert!(matches!(stale.publish(), Err(Error::StalePublication)));
    let (_, lines) = CAPTURE.take().unwrap();
    assert_eq!(lines, [expected]); // Neither LCQ nor failed HCQ logged a promotion.
}

fn logger() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        log::set_logger(&Logger).unwrap();
        log::set_max_level(log::LevelFilter::Debug);
    });
}

#[test]
fn shutdown_reports_resident_units_and_cache_once_before_reclamation() {
    logger();
    for populated in [false, true] {
        let process = process();
        if populated {
            let cursor = AtomicU64::new(0);
            publish(&process, &cursor, &[0, 4], Tier::Lcq);
            publish(&process, &cursor, &[0, 4], Tier::Hcq);
            process.try_service_links().unwrap();
        }
        let usage = process.cache.usage().unwrap();
        CAPTURE.set(Some((process.clone(), Vec::new())));
        process.request_shutdown().unwrap();
        process.request_shutdown().unwrap();
        assert!(process.try_shutdown().unwrap());
        assert!(process.try_shutdown().unwrap());
        let (_, lines) = CAPTURE.take().unwrap();
        let count = usize::from(populated);
        assert_eq!(
            lines,
            [
                format!(
                    "JIT shutdown units: jit_process={} lcq_resident={count} hcq_resident={count} hcq_published={count} lcq_native_bytes={} hcq_native_bytes={}",
                    process.identity,
                    count * 16,
                    count * 16
                ),
                format!(
                    "JIT shutdown cache: jit_process={} committed_bytes={} metadata_bytes={} total_bytes={}",
                    process.identity,
                    usage.committed,
                    usage.metadata,
                    usage.total()
                ),
            ]
        );
        assert_eq!(process.cache.usage().unwrap().committed, 0);
    }
}

#[test]
fn shutdown_includes_retired_but_retained_hcq_without_counting_it_as_published() {
    logger();
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0], Tier::Lcq);
    let hcq = publish(&process, &cursor, &[0], Tier::Hcq);
    process.try_service_links().unwrap();
    let retained = process.snapshot(hcq).unwrap();
    process.retire_unit(hcq).unwrap();
    let mut stop = process.try_transition().unwrap().unwrap();
    stop.wait_closed().unwrap();
    stop.drain_retirements().unwrap();
    stop.batch().unwrap().complete().unwrap();
    assert!(stop.try_reopen().unwrap());
    drop(stop);
    assert_eq!(process.reclaim_units().unwrap(), 0);
    CAPTURE.set(Some((process.clone(), Vec::new())));
    assert!(!process.try_shutdown().unwrap());
    drop(retained);
    assert!(process.try_shutdown().unwrap());
    let (_, lines) = CAPTURE.take().unwrap();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        lines[0],
        format!(
            "JIT shutdown units: jit_process={} lcq_resident=1 hcq_resident=1 hcq_published=0 lcq_native_bytes=16 hcq_native_bytes=16",
            process.identity,
        )
    );
}
