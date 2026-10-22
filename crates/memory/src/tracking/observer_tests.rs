use super::*;
use crate::{AccessOperation, AccessResult};

fn input(pages: u64) -> (Machine, AllocationRange) {
    let bytes = pages * PAGE_SIZE;
    let machine = Machine::new(bytes).unwrap();
    let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, bytes).unwrap();
    (machine, range)
}

fn capture(machine: &Machine, range: &AllocationRange) -> Observation {
    let observation = Observation::default();
    machine
        .access_captured(
            std::slice::from_ref(range),
            AccessOperation::ReadObserved(observation.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    observation
}

fn inspections(machine: &Machine) -> usize {
    machine
        .tracking()
        .observers
        .lock()
        .unwrap()
        .pages
        .values()
        .map(|page| page.registration_inspections)
        .sum()
}

#[test]
fn overlapping_live_registrations_have_linear_inspection_cost_and_all_invalidate() {
    let (machine, range) = input(1);
    let field = range.slice(7, 4).unwrap();
    let observations: Vec<_> = (0..4096).map(|_| capture(&machine, &field)).collect();
    assert_eq!(
        machine.tracking_statistics().observer_registrations,
        observations.len()
    );
    assert!(inspections(&machine) < 2 * observations.len());
    assert!(
        observations
            .iter()
            .all(|observation| !observation.is_invalid())
    );
    machine.write_resolved(&[field], &[7; 4]).unwrap();
    assert!(observations.iter().all(Observation::is_invalid));
    assert_eq!(machine.tracking_statistics().observer_registrations, 0);
}

#[test]
fn dropped_and_externally_invalidated_observers_remain_bounded_during_churn() {
    let (machine, range) = input(1);
    let field = range.slice(0, 1).unwrap();
    let live: Vec<_> = (0..128).map(|_| capture(&machine, &field)).collect();
    // Keep invalidated flags alive too: pruning must not rely only on weak count.
    let mut invalid = Vec::new();
    for index in 0..10_000 {
        let observation = capture(&machine, &field);
        if index % 2 == 0 {
            observation.invalidate();
            invalid.push(observation);
        }
        assert!(machine.tracking_statistics().observer_registrations <= 2 * live.len());
    }
    assert!(inspections(&machine) < 3 * 10_128);
    assert!(live.iter().all(|observation| !observation.is_invalid()));
    machine.write_resolved(&[field], &[1]).unwrap();
    assert!(live.iter().all(Observation::is_invalid));
    assert_eq!(machine.tracking_statistics().observer_pages, 0);
}

#[test]
fn pruning_recovers_after_a_live_peak_and_retirement_releases_quiet_pages() {
    let (machine, range) = input(2);
    let field = range.slice(PAGE_SIZE, 1).unwrap();
    let live: Vec<_> = (0..1024).map(|_| capture(&machine, &field)).collect();
    drop(live);
    // More than a full sweep interval since the peak; no writes are needed
    // to reclaim dead registrations during repeated create/drop cycles.
    for _ in 0..2048 {
        drop(capture(&machine, &field));
    }
    assert!(machine.tracking_statistics().observer_registrations <= 16);
    assert!(inspections(&machine) < 3 * 3072);
    let last = capture(&machine, &field);
    drop(field);
    drop(range);
    assert!(last.is_invalid());
    assert_eq!(machine.tracking_statistics().observer_pages, 0);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn pruning_preserves_cpu_device_and_shared_observation_domains() {
    let (machine, range) = input(2);
    let both = capture(&machine, &range);
    let first = range.slice(0, 1).unwrap();
    let second = range.slice(PAGE_SIZE, 1).unwrap();
    let AccessResult::NativeRead {
        area: Some(device), ..
    } = machine
        .read_native_resolved(std::slice::from_ref(&second), Some(&second))
        .unwrap()
    else {
        panic!("expected retained native area")
    };
    let copy = PhysicalCapture {
        source: second.physical_extents().next().unwrap(),
        destination: 0,
    };
    let cpu = machine
        .capture_clean_scattered(std::slice::from_ref(&second), &[copy], &mut [0])
        .unwrap();
    for _ in 0..128 {
        drop(capture(&machine, &range));
    }
    assert!(!both.is_invalid());
    assert!(!cpu.is_invalid());
    assert!(!device.observation().is_invalid());
    machine.write_resolved(&[first], &[1]).unwrap();
    assert!(both.is_invalid());
    assert!(!cpu.is_invalid());
    assert!(!device.observation().is_invalid());
    machine
        .publish_device_write(std::slice::from_ref(&second), || Ok(()))
        .unwrap();
    assert!(device.observation().is_invalid());
    assert!(!cpu.is_invalid());
    machine.write_resolved(&[second], &[2]).unwrap();
    assert!(cpu.is_invalid());
    assert_eq!(machine.tracking_statistics().observer_registrations, 0);
}
