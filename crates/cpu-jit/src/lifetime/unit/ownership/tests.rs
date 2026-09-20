use super::*;
use crate::abi::FpSpecialization;
use crate::lifetime::unit::tests::{input, key, process, publish};

fn instruction(pc: u64) -> InstructionKey {
    InstructionKey::new(key(pc)).unwrap()
}

fn retire(process: &Lifetime, unit: UnitHandle) {
    process.retire_unit(unit).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn ownership_includes_interior_instructions_without_creating_hcq_entries() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0, 4, 8], Tier::Lcq);
    assert!(process.lock().units.family_owners.entries.is_empty());
    let mut candidate = input(&process, &[0, 4, 8], Tier::Hcq);
    candidate.entries = candidate.entries.into_vec().into_iter().take(1).collect();
    let prepared = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], candidate, &cursor)
        .unwrap();
    // Staged output owns no instruction until its publication point.
    assert!(process.lock().units.family_owners.entries.is_empty());
    let unit = prepared.publish().unwrap();
    let state = process.lock();
    let owner = state.units.records.get(unit.0).unwrap().family.unwrap();
    for pc in [0, 4, 8] {
        assert_eq!(state.units.family_owners.get(instruction(pc)), Some(owner));
        let payload = state
            .dispatch
            .get(*state.keys.get(&key(pc)).unwrap())
            .unwrap()
            .snapshot();
        assert_eq!(payload.hcq().is_some(), pc == 0);
    }
    assert_eq!(state.units.family_owners.get(instruction(12)), None);
    let specialized = InstructionKey::new(crate::abi::BlockKey {
        fp: FpSpecialization::Exact(0),
        ..key(4)
    })
    .unwrap();
    assert_eq!(state.units.family_owners.get(specialized), None);
    assert!(state.units.family_owner_storage.is_some());
}

#[test]
fn unlink_removes_ownership_even_when_old_code_is_pinned_and_replacement_is_live() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0, 4], Tier::Lcq);
    let old = publish(&process, &cursor, &[0, 4], Tier::Hcq);
    let pin = process.snapshot(old).unwrap();
    let old_owner = process
        .lock()
        .units
        .family_owners
        .get(instruction(4))
        .unwrap();
    retire(&process, old);
    assert!(process.lock().units.family_owners.entries.is_empty());
    let replacement = publish(&process, &cursor, &[0, 4], Tier::Hcq);
    let owner = process
        .lock()
        .units
        .records
        .get(replacement.0)
        .unwrap()
        .family
        .unwrap();
    assert_ne!(owner, old_owner);
    // A stale generational removal cannot erase the new family's membership.
    assert!(
        !process
            .lock()
            .units
            .family_owners
            .remove(instruction(4), old_owner)
    );
    drop(pin);
    process.reclaim_units().unwrap();
    assert_eq!(
        process.lock().units.family_owners.get(instruction(4)),
        Some(owner)
    );
    retire(&process, replacement);
    process.reclaim_units().unwrap();
    assert!(process.lock().units.family_owners.entries.is_empty());
}

#[test]
fn ownership_growth_preserves_point_lookups_and_reuses_empty_capacity() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut owners = Vec::new();
    for pc in (0..256).step_by(8) {
        publish(&process, &cursor, &[pc, pc + 4], Tier::Lcq);
        let hcq = publish(&process, &cursor, &[pc, pc + 4], Tier::Hcq);
        owners.push((pc, hcq));
    }
    let capacity = {
        let state = process.lock();
        assert_eq!(state.units.family_owners.entries.len(), 64);
        for &(pc, hcq) in &owners {
            let family = state.units.records.get(hcq.0).unwrap().family;
            assert_eq!(state.units.family_owners.get(instruction(pc)), family);
            assert_eq!(state.units.family_owners.get(instruction(pc + 4)), family);
        }
        state.units.family_owners.entries.capacity()
    };
    for &(_, hcq) in &owners {
        retire(&process, hcq);
    }
    process.reclaim_units().unwrap();
    assert!(process.lock().units.family_owners.entries.is_empty());
    for (pc, _) in owners {
        publish(&process, &cursor, &[pc, pc + 4], Tier::Hcq);
    }
    let state = process.lock();
    assert_eq!(state.units.family_owners.entries.len(), 64);
    assert_eq!(state.units.family_owners.entries.capacity(), capacity);
}

#[test]
fn discarded_or_stale_preparation_never_publishes_membership() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[0, 4], Tier::Lcq);
    for discard in [true, false] {
        let publications = [
            process.reserve(key(0)).unwrap(),
            process.reserve(key(4)).unwrap(),
        ];
        let prepared = process
            .prepare_unit(&publications, input(&process, &[0, 4], Tier::Hcq), &cursor)
            .unwrap();
        if discard {
            drop(prepared);
        } else {
            cursor.store(1, Ordering::Release);
            assert_eq!(prepared.publish(), Err(Error::StalePublication));
        }
        assert!(process.lock().units.family_owners.entries.is_empty());
        assert!(process.lock().units.families.is_empty());
    }
}
