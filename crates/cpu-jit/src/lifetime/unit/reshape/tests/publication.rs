use super::discovery::{NOP, RET, owned, owned_entries, reshape};
use super::*;
use crate::hcq::Graph;
use crate::lifetime::background::Frozen;
use crate::lifetime::unit::dynamic::pic::tests::cache;
use crate::lifetime::unit::{Input, dynamic, links, tests::publish_words};

fn output(process: &Lifetime, frozen: &Frozen<'_, '_>) -> Input {
    let pcs: Vec<_> = frozen
        .graph()
        .instructions
        .iter()
        .map(|word| word.instruction.key.block_key().pc.get())
        .collect();
    let mut result = input(process, &pcs, Tier::Hcq);
    for (word, captured) in result
        .instructions
        .iter_mut()
        .zip(&frozen.graph().instructions)
    {
        *word = captured.instruction;
    }
    let mut entries = result.entries.into_vec();
    result.entries = frozen
        .entries()
        .iter()
        .map(|&index| {
            let key = frozen.graph().blocks[index].key;
            let index = entries.iter().position(|entry| entry.key == key).unwrap();
            entries.swap_remove(index)
        })
        .collect();
    result.dependencies = frozen.dependencies().into();
    result
}

#[test]
fn replacement_publishes_zero_one_or_two_participants_and_survives_their_cleanup() {
    for count in 0..=2 {
        let process = process();
        let cursor = AtomicU64::new(0);
        publish_words(&process, 0, &[NOP, 0x14000003]); // B 16.
        publish_words(&process, 16, &[NOP, RET]);
        let mut old = Vec::new();
        if count >= 1 {
            old.push(owned(&process, &[(16, NOP), (20, RET)]));
        }
        if count == 2 {
            old.push(owned(&process, &[(0, NOP), (4, 0x14000003)]));
        }
        let work = reshape(&process, 0, 4, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let replacement = frozen
            .prepare(output(&process, &frozen), &cursor)
            .unwrap()
            .publish()
            .unwrap();
        let successor = {
            let state = process.lock();
            let family = state
                .units
                .records
                .get(replacement.0)
                .unwrap()
                .family
                .unwrap();
            for pc in [0, 4, 16, 20] {
                assert_eq!(state.units.family_owners.get(instruction(pc)), Some(family));
            }
            for pc in [0, 16] {
                let slot = state
                    .dispatch
                    .get(*state.keys.get(&key(pc)).unwrap())
                    .unwrap();
                assert_eq!(slot.owners[1].unwrap().unit, replacement);
                assert_eq!(
                    slot.snapshot().hcq().unwrap().entry.unit,
                    state.units.records.get(replacement.0).unwrap().code.id
                );
            }
            for predecessor in &old {
                let record = state.units.records.get(predecessor.0).unwrap();
                assert_eq!(record.lifecycle, Lifecycle::Superseded);
                assert_eq!(record.retirement.unwrap().0, Reason::TierCutover);
            }
            family
        };
        if count != 0 {
            assert!(process.try_service_links().unwrap());
            let state = process.lock();
            for predecessor in &old {
                assert!(matches!(
                    state.units.records.get(predecessor.0).unwrap().lifecycle,
                    Lifecycle::Retired(_)
                ));
            }
        }
        // Frozen snapshots/compiler ownership delay actual reclamation, not
        // cutover. Cleanup must neither withdraw successor entries nor ownership.
        process.reclaim_units().unwrap();
        assert_eq!(
            process.lock().units.family_owners.get(instruction(16)),
            Some(successor)
        );
        drop(frozen);
        drop(work);
        assert!(process.try_service_links().unwrap());
        for predecessor in old {
            assert!(process.lock().units.records.get(predecessor.0).is_none());
        }
        assert_eq!(
            process.lock().units.family_owners.get(instruction(16)),
            Some(successor)
        );
        assert!(process.try_shutdown().unwrap());
    }
}

#[test]
fn replacement_restores_discarded_entries_and_withdraws_unselected_membership() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[NOP, RET]);
    let baseline = publish_words(&process, 20, &[RET]);
    let outside = publish_words(&process, 64, &[RET]);
    let old = owned_entries(&process, &[(16, NOP), (20, RET), (64, RET)], 3);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.replacement().fallbacks.len(), 1);
    let successor = frozen
        .prepare(output(&process, &frozen), &cursor)
        .unwrap()
        .publish()
        .unwrap();
    {
        let state = process.lock();
        let retained = state
            .dispatch
            .get(*state.keys.get(&key(20)).unwrap())
            .unwrap();
        assert_eq!(retained.owners[1].unwrap().unit, successor);
        assert_eq!(retained.owners[0].unwrap().unit, baseline);
        let slot = state
            .dispatch
            .get(*state.keys.get(&key(64)).unwrap())
            .unwrap();
        assert!(slot.snapshot().hcq().is_none());
        assert!(slot.owners[1].is_none());
        assert_eq!(slot.owners[0].unwrap().unit, outside);
        assert_eq!(slot.snapshot().preferred(), slot.snapshot().lcq());
        assert!(state.units.family_owners.get(instruction(64)).is_none());
        assert_eq!(
            state.units.family_owners.get(instruction(20)),
            state.units.records.get(successor.0).unwrap().family
        );
        assert_eq!(
            state.units.records.get(old.0).unwrap().lifecycle,
            Lifecycle::Superseded
        );
    }
    assert!(process.try_service_links().unwrap());
    {
        let state = process.lock();
        assert_eq!(
            state.units.family_owners.get(instruction(20)),
            state.units.records.get(successor.0).unwrap().family
        );
    }
    drop(frozen);
    drop(work);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn replacement_cuts_installed_static_and_pic_roots_before_retiring_predecessor() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    let baseline = publish_words(&process, 64, &[RET]);
    let old = owned_entries(&process, &[(16, RET), (64, RET)], 2);
    let selected = links::tests::source(&process, &cursor, 128, 16);
    let dropped = links::tests::source(&process, &cursor, 132, 64);
    let indirect = dynamic::tests::source(&process, &cursor, 136, EdgeKind::Indirect);
    // The source helper deliberately defers installations. Register a fresh
    // LinkPatch stop and install both real native patches before replacement.
    process.request(Reason::LinkPatch).unwrap();
    assert!(process.try_service_links().unwrap());
    let mut reader = process.register().unwrap();
    for (map, pc) in [(0, 16), (1, 64)] {
        cache(
            &mut reader,
            process
                .prepare_dynamic_bridge(indirect, map, key(pc))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    }
    {
        let state = process.lock();
        assert!(
            state
                .units
                .records
                .get(old.0)
                .unwrap()
                .pic_incoming
                .is_some()
        );
    }
    for source in [selected, dropped] {
        links::tests::assert_callable_target(&process, source, old);
    }
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let successor = frozen
        .prepare(output(&process, &frozen), &cursor)
        .unwrap()
        .publish()
        .unwrap();
    {
        let state = process.lock();
        // Dispatch now names the replacement/baseline; old native roots still
        // pin the predecessor and its fault records until the rendezvous.
        assert!(
            state
                .units
                .records
                .get(old.0)
                .unwrap()
                .pic_incoming
                .is_some()
        );
    }
    for source in [selected, dropped] {
        links::tests::assert_callable_target(&process, source, old);
    }
    assert!(process.try_service_links().unwrap());
    {
        let state = process.lock();
        let record = state.units.records.get(old.0).unwrap();
        assert!(matches!(record.lifecycle, Lifecycle::Retired(_)));
        assert!(record.incoming.is_none());
        assert!(record.pic_incoming.is_none());
    }
    for (source, target) in [(selected, successor), (dropped, baseline)] {
        links::tests::assert_callable_target(&process, source, target);
    }
    drop(frozen);
    drop(work);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn replacement_staged_output_refreshes_directory_and_registry_capacity_without_reemission() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    owned(&process, &[(16, RET)]);
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let prepared = frozen.prepare(output(&process, &frozen), &cursor).unwrap();
    let code = Arc::clone(prepared.unit.as_ref().unwrap());
    // Unrelated publishers consume both registry reservations and replace the
    // segment directory. Keep Open authority and the captured inputs intact.
    let mut pc = 0x10000;
    while process.lock().units.families.has_space() {
        publish_words(&process, pc, &[RET]);
        owned(&process, &[(pc, RET)]);
        pc += 4;
    }
    while process.lock().units.records.has_space() {
        publish_words(&process, pc, &[RET]);
        pc += 4;
    }
    let capacities = {
        let state = process.lock();
        (
            state.units.records.capacity(),
            state.units.families.capacity(),
        )
    };
    frozen.check().unwrap();
    let successor = prepared.publish().unwrap();
    assert!(Arc::ptr_eq(
        &process.lock().units.records.get(successor.0).unwrap().code,
        &code
    ));
    {
        let state = process.lock();
        assert!(state.units.records.capacity() > capacities.0);
        assert!(state.units.families.capacity() > capacities.1);
    }
    assert!(process.try_service_links().unwrap());
    drop(frozen);
    drop(work);
    drop(code);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn replacement_reuses_predecessor_membership_capacity() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut left = [NOP; 32];
    left[31] = 0x14000021; // B from 124 to 256.
    let mut right = [NOP; 32];
    right[31] = RET;
    publish_words(&process, 0, &left);
    publish_words(&process, 256, &right);
    for (start, words) in [(0, left), (256, right)] {
        owned(
            &process,
            &words
                .into_iter()
                .enumerate()
                .map(|(i, bits)| (start + i as u64 * 4, bits))
                .collect::<Vec<_>>(),
        );
    }
    let before = process.lock().units.family_owners.entries.capacity();
    let work = reshape(&process, 0, 124, 256);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.graph().instructions.len(), 64);
    frozen
        .prepare(output(&process, &frozen), &cursor)
        .unwrap()
        .publish()
        .unwrap();
    assert!(process.try_service_links().unwrap());
    {
        let state = process.lock();
        assert_eq!(state.units.family_owners.entries.len(), 64);
        assert_eq!(state.units.family_owners.entries.capacity(), before);
    }
    drop(frozen);
    drop(work);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn abandoned_or_stale_replacement_keeps_old_family_and_does_not_restore_old_baseline() {
    for stale in [false, true] {
        let process = process();
        let cursor = AtomicU64::new(0);
        publish_words(&process, 0, &[0x14000004]);
        publish_words(&process, 16, &[RET]);
        let baseline = publish_words(&process, 64, &[RET]);
        let old = owned_entries(&process, &[(16, RET), (64, RET)], 2);
        let work = reshape(&process, 0, 0, 16);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let prepared = frozen.prepare(output(&process, &frozen), &cursor).unwrap();
        let expected = if stale {
            let next = publish_words(&process, 64, &[RET]);
            assert_eq!(prepared.publish(), Err(Error::StalePublication));
            next
        } else {
            drop(prepared);
            baseline
        };
        {
            let state = process.lock();
            let record = state.units.records.get(old.0).unwrap();
            assert_eq!(record.lifecycle, Lifecycle::Published);
            assert!(record.retirement.is_none());
            assert_eq!(
                state.units.family_owners.get(instruction(64)),
                record.family
            );
            let slot = state
                .dispatch
                .get(*state.keys.get(&key(64)).unwrap())
                .unwrap();
            assert_eq!(slot.owners[0].unwrap().unit, expected);
            assert_eq!(slot.owners[1].unwrap().unit, old);
            assert_eq!(state.units.families.values().count(), 1);
        }
        drop(frozen);
        drop(work);
        assert!(process.try_shutdown().unwrap());
    }
}

#[test]
fn replacement_reuses_native_span_and_registry_storage_only_after_fault_reader_grace() {
    use crate::abi::{NativeFrame, PollBudget};
    use nixe_cpu::state::a64::A64State;

    let process = process();
    let cursor = AtomicU64::new(0);
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    publish_words(&process, 64, &[RET]);
    let old = owned(&process, &[(16, RET)]);
    let pin = process.snapshot(old).unwrap();
    let address = pin.code.allocation.address();
    let old_id = pin.id;
    let old_family = process
        .lock()
        .units
        .records
        .get(old.0)
        .unwrap()
        .family
        .unwrap();
    let work = reshape(&process, 0, 0, 16);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let successor = frozen
        .prepare(output(&process, &frozen), &cursor)
        .unwrap()
        .publish()
        .unwrap();
    assert!(process.try_service_links().unwrap());
    drop(frozen);
    drop(work);
    assert_eq!(process.reclaim_units().unwrap(), 0); // Compiler pin still owns old bytes.
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = NativeFrame::new(&mut cpu, PollBudget::new(4096, 100).unwrap());
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let fault = invocation.fault(address + 12).unwrap();
    assert_eq!(fault.unit.id, old_id);
    drop(pin);
    // A newer invocation borrowed the old directory after retirement. Detach
    // its entry now, but keep its bytes/metadata until this second grace period.
    assert_eq!(process.reclaim_units().unwrap(), 0);
    assert!(invocation.fault(address + 12).is_none());
    assert_eq!(fault.unit.id, old_id);
    assert!(process.lock().units.records.get(old.0).is_some());
    drop(invocation);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    let (unit_capacity, family_capacity, next_unit, next_family) = {
        let state = process.lock();
        assert!(state.units.records.get(old.0).is_none());
        assert!(state.units.families.get(old_family).is_none());
        (
            state.units.records.capacity(),
            state.units.families.capacity(),
            state.units.records.next_handle().unwrap(),
            state.units.families.next_handle().unwrap(),
        )
    };
    let reused = owned(&process, &[(64, RET)]);
    let code = process.snapshot(reused).unwrap();
    assert_eq!(code.code.allocation.address(), address);
    assert_ne!(code.id, old_id);
    {
        let state = process.lock();
        assert_eq!(reused.0, next_unit);
        assert_eq!(
            state.units.records.get(reused.0).unwrap().family,
            Some(next_family)
        );
        assert_eq!(state.units.records.capacity(), unit_capacity);
        assert_eq!(state.units.families.capacity(), family_capacity);
        assert_eq!(
            state.units.family_owners.get(instruction(16)),
            state.units.records.get(successor.0).unwrap().family
        );
    }
    assert_eq!(process.retire_unit(old).err(), Some(Error::StaleUnit));
    let invocation = unsafe { reader.admit(&mut frame, key(64)) }
        .unwrap()
        .unwrap();
    assert_eq!(invocation.fault(address + 12).unwrap().unit.id, code.id);
    drop(invocation);
    drop(code);
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn alternating_interior_observations_stabilize_membership_entries_and_storage() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[0x14000004]); // B 32.
    publish_words(&process, 32, &[RET]);
    let mut current = owned(&process, &[(16, 0x14000004), (32, RET)]);
    let mut steady = None;
    for round in 0..32 {
        let (root, target) = if round % 2 == 0 { (0, 16) } else { (16, 32) };
        let work = reshape(&process, root, root, target);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(frozen.graph().instructions.len(), 3);
        if round < 2 {
            assert!(!frozen.unchanged());
            let grown = frozen
                .prepare(output(&process, &frozen), &cursor)
                .unwrap()
                .publish()
                .unwrap();
            drop(frozen);
            drop(work);
            process.try_service_links().unwrap();
            assert!(matches!(process.snapshot(current), Err(Error::StaleUnit)));
            current = grown;
            continue;
        }
        assert!(frozen.unchanged());
        assert_eq!(frozen.entries().len(), 3);
        assert!(frozen.prepare_unchanged().unwrap().install().unwrap());
        drop(frozen);
        drop(work);
        let boundary = boundary(&process, root, root, target);
        let negative = crate::lifetime::unit::reshape::negative::Key {
            source: key(root),
            boundary,
        };
        assert_suppressed(&process, key(root), boundary);
        let capacities = {
            let state = process.lock();
            assert!(state.units.negatives.get(negative).is_some());
            assert_eq!(state.units.records.values().count(), 4); // Three baselines, one HCQ.
            assert_eq!(state.units.families.values().count(), 1);
            assert_eq!(state.units.family_owners.entries.len(), 3);
            (
                state.units.records.capacity(),
                state.units.families.capacity(),
                state.units.family_owners.entries.capacity(),
            )
        };
        let usage = process.cache.usage().unwrap();
        if round >= 3 {
            let measured = (capacities, usage.metadata, usage.committed);
            if let Some(previous) = steady {
                assert_eq!(measured, previous, "no-op storage grew at round {round}");
            } else {
                steady = Some(measured);
            }
        }

        process.snapshot(current).unwrap();
        // Remove only this weak negative so the next observation can be admitted;
        // the positive family itself never changes after its entries converge.
        let removed = {
            let mut state = process.lock();
            state.units.negatives.invalidate_unit(current);
            state.units.negatives.take_removed()
        };
        drop(removed);
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.cache.usage().unwrap().committed, 0);
}

#[test]
fn queued_running_and_staged_reshapes_survive_unrelated_closed_maintenance() {
    for count in 0..=2 {
        for running in [false, true] {
            let process = process();
            let cursor = AtomicU64::new(0);
            publish_words(&process, 0, &[0x14000004]);
            publish_words(&process, 16, &[RET]);
            if count >= 1 {
                owned(&process, &[(16, RET)]);
            }
            if count == 2 {
                owned(&process, &[(0, 0x14000004)]);
            }
            let queue = Queue::new(1, &process).unwrap().unwrap();
            let mut samples = Samples::new();
            let snapshot = heat(&mut samples, boundary(&process, 0, 0, 16));
            assert_eq!(
                process
                    .admit_reshape(&queue, &mut samples, key(0), snapshot)
                    .unwrap(),
                Outcome::Queued
            );
            let mut work = running.then(|| {
                process
                    .accept_background(queue.pop().unwrap().unwrap())
                    .unwrap()
                    .unwrap()
            });
            process.request(Reason::LinkPatch).unwrap();
            let mut stop = process.try_transition().unwrap().unwrap();
            stop.wait_closed().unwrap();
            if !running {
                work = Some(
                    process
                        .accept_background(queue.pop().unwrap().unwrap())
                        .unwrap()
                        .unwrap(),
                );
            }
            let work = work.unwrap();
            work.check().unwrap();
            let frozen = work
                .reserve_candidate(Graph::discover(&work).unwrap())
                .unwrap()
                .freeze()
                .unwrap();
            let prepared = frozen.prepare(output(&process, &frozen), &cursor).unwrap();
            let code = Arc::clone(prepared.unit.as_ref().unwrap());
            stop.batch().unwrap().complete().unwrap();
            assert!(stop.try_reopen().unwrap());
            drop(stop);
            // Reopening changes admission but not the reserved identities.
            assert_eq!(
                process
                    .admit_reshape(&queue, &mut samples, key(0), snapshot)
                    .unwrap(),
                Outcome::Deferred
            );
            let successor = prepared.publish().unwrap();
            assert!(Arc::ptr_eq(
                &process.lock().units.records.get(successor.0).unwrap().code,
                &code
            ));
            process.try_service_links().unwrap();
            drop(code);
            drop(frozen);
            drop(work);
            assert!(process.try_shutdown().unwrap());
        }
    }
}
