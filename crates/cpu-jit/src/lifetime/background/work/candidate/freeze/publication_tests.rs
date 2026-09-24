use super::super::tests::work;
use super::*;
use crate::lifetime::background::tests::setup;
use crate::lifetime::unit::tests::{input, key, publish_words};

const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

#[test]
fn frozen_optimizer_rejection_is_version_local_and_releases_compiler_claims() {
    use crate::lifetime::background::tests::snapshot;
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    let job = work(&process, &queue, &mut samples, 0);
    let frozen = job
        .reserve_candidate(Graph::discover(&job).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    frozen.reject().unwrap();
    drop(frozen);
    drop(job);
    {
        let state = process.lock();
        assert!(state.candidates.entries.is_empty());
        assert_eq!(state.compilers, 0);
        let slot = state
            .dispatch
            .get(*state.keys.get(&key(0)).unwrap())
            .unwrap();
        assert!(!slot.optimization.pinned());
        assert!(slot.snapshot().lcq().is_some());
        assert!(slot.snapshot().hcq().is_none());
    }
    process.request(Reason::LinkPatch).unwrap();
    process.try_service_links().unwrap();
    assert_eq!(
        process
            .admit_seed(&queue, &mut samples, snapshot(&process, 0))
            .unwrap(),
        Outcome::Duplicate
    );
    publish_words(&process, 0, &[NOP, RET]);
    process.try_service_links().unwrap();
    assert_eq!(
        process
            .admit_seed(&queue, &mut samples, snapshot(&process, 0))
            .unwrap(),
        Outcome::Queued
    );
}

#[test]
fn obsolete_optimizer_rejection_cannot_disable_the_current_seed() {
    use crate::lifetime::background::tests::snapshot;
    for change in 0..3 {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[NOP, NOP, RET]);
        publish_words(&process, 4, &[NOP, RET]);
        let job = work(&process, &queue, &mut samples, 0);
        let frozen = job
            .reserve_candidate(Graph::discover(&job).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        match change {
            0 => {
                let mut state = process.lock();
                let key = InstructionKey::new(key(8)).unwrap();
                let claim = *state.candidates.get(key).unwrap();
                state.candidates.remove(key, claim.token);
            }
            1 => {
                publish_words(&process, 4, &[NOP, RET]);
            }
            2 => {
                publish_words(&process, 0, &[NOP, NOP, RET]);
            }
            _ => unreachable!(),
        }
        assert_eq!(frozen.reject(), Err(Error::StalePublication));
        drop(frozen);
        drop(job);
        process.try_service_links().unwrap();
        assert_eq!(
            process
                .admit_seed(&queue, &mut samples, snapshot(&process, 0))
                .unwrap(),
            Outcome::Queued
        );
    }
}

// Synthetic machine code is sufficient here: these tests isolate the final
// candidate authority transaction, not HCQ lowering or native execution.
fn output(frozen: &Frozen<'_, '_>) -> unit::Input {
    let pcs: Vec<_> = frozen
        .graph()
        .instructions
        .iter()
        .map(|word| word.instruction.key.block_key().pc.get())
        .collect();
    let mut output = input(frozen.candidate.work.process, &pcs, Tier::Hcq);
    output.instructions = frozen
        .graph()
        .instructions
        .iter()
        .map(|word| word.instruction)
        .collect();
    let mut entries = output.entries.into_vec();
    output.entries = frozen
        .entries()
        .iter()
        .map(|&index| {
            let key = frozen.graph().blocks[index].key;
            let position = entries.iter().position(|entry| entry.key == key).unwrap();
            entries.remove(position)
        })
        .collect();
    output.dependencies = frozen.dependencies().into();
    output
}

#[test]
fn frozen_publication_binds_words_entries_and_dependencies_before_preparation() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    let job = work(&process, &queue, &mut samples, 0);
    let frozen = job
        .reserve_candidate(Graph::discover(&job).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    for case in 0..5 {
        let mut candidate = output(&frozen);
        match case {
            0 => candidate.instructions[1].bits = 0,
            1 => candidate.instructions.swap(0, 1),
            2 => candidate.entries[0].key = key(4),
            3 => candidate.dependencies = Box::new([]),
            4 => candidate.tier = Tier::Lcq,
            _ => unreachable!(),
        }
        assert!(matches!(
            frozen.prepare(candidate, &AtomicU64::new(0)),
            Err(Error::InvalidUnit(
                "HCQ output differs from its frozen candidate"
            ))
        ));
        frozen.check().unwrap();
    }
    let state = process.lock();
    assert!(
        state
            .dispatch
            .get(*state.keys.get(&key(0)).unwrap())
            .unwrap()
            .snapshot()
            .hcq()
            .is_none()
    );
}

#[test]
fn frozen_publication_rechecks_claims_and_nonentry_inputs_after_preparation() {
    for case in 0..5 {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[NOP, NOP, RET]);
        publish_words(&process, 4, &[NOP, RET]);
        let work = work(&process, &queue, &mut samples, 0);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(frozen.entries(), &[0]);
        let cursor = AtomicU64::new(0);
        let candidate = output(&frozen);
        let address = candidate.code.allocation.address();
        let prepared = frozen.prepare(candidate, &cursor).unwrap();
        // All earlier phase checks succeeded. None authorizes this later commit.
        match case {
            0 => {
                let mut state = process.lock();
                let key = InstructionKey::new(key(8)).unwrap();
                let claim = *state.candidates.get(key).unwrap();
                state.candidates.remove(key, claim.token);
            }
            1 => {
                publish_words(&process, 4, &[NOP, RET]);
            }
            2 => {
                publish_words(&process, 0, &[NOP, NOP, RET]);
                process.try_service_links().unwrap();
            }
            3 => {
                process.request_shutdown().unwrap();
            }
            4 => {
                process
                    .invalidate_memory(&[nixe_memory::MemoryInvalidationKind::InstructionCache {
                        address_space: key(0).address_space,
                    }])
                    .unwrap();
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            prepared.publish(),
            Err(Error::StalePublication | Error::Shutdown)
        ));
        let state = process.lock();
        let slot = state
            .dispatch
            .get(*state.keys.get(&key(0)).unwrap())
            .unwrap();
        assert!(slot.snapshot().lcq().is_some());
        assert!(slot.snapshot().hcq().is_none());
        assert!(
            state
                .units
                .instruction_available(InstructionKey::new(key(0)).unwrap(), [None; 2])
        );
        drop(state);
        if case == 0 {
            // Failed commit returns its actual code span; retained input
            // snapshots do not keep unpublished HCQ executable storage alive.
            let replacement = output(&frozen);
            assert_eq!(replacement.code.allocation.address(), address);
        }
        drop(frozen);
        assert!(process.lock().candidates.entries.is_empty());
    }
}

#[test]
fn frozen_publication_pins_exact_captures_without_exporting_coverage_or_internal_leaders() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let cursor = AtomicU64::new(0);
    let prepared = frozen.prepare(output(&frozen), &cursor).unwrap();
    let captures: Vec<_> = frozen
        .graph()
        .units
        .iter()
        .map(|unit| unit.registered_handle().unwrap())
        .collect();
    let handle = prepared.publish().unwrap();
    for capture in captures {
        assert!(matches!(
            process.retire_unit(capture),
            Err(Error::PinnedBaseline)
        ));
    }
    let published = process.snapshot(handle).unwrap();
    assert_eq!(published.entries.len(), 1);
    assert_eq!(published.instructions.len(), 3);
    let state = process.lock();
    assert!(!state.keys.contains_key(&key(8)));
    let interior = state
        .dispatch
        .get(*state.keys.get(&key(4)).unwrap())
        .unwrap()
        .snapshot();
    assert!(interior.lcq().is_some());
    assert!(interior.hcq().is_none());
    assert_eq!(state.candidates.entries.len(), 3);
    drop(state);
    drop(frozen);
    assert!(process.lock().candidates.entries.is_empty());
}

#[test]
fn frozen_publication_survives_unrelated_epochs_cursors_and_directory_publications() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let cursor = AtomicU64::new(0);
    let candidate = output(&frozen);
    let version = candidate.identity.version();
    let address = candidate.code.allocation.address();
    process.request(Reason::LinkPatch).unwrap();
    // Cold preparation is allowed even while execution admission is closed.
    let prepared = frozen.prepare(candidate, &cursor).unwrap();
    process.try_service_links().unwrap();
    cursor.store(1, Ordering::Release); // Unrelated memory history, not input provenance.
    publish_words(&process, 0x1000, &[NOP, RET]);
    let other_job = super::super::tests::work(&process, &queue, &mut samples, 0x1000);
    let other_frozen = other_job
        .reserve_candidate(Graph::discover(&other_job).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let other = other_frozen
        .prepare(output(&other_frozen), &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let published = prepared.publish().unwrap();
    let snapshot = process.snapshot(published).unwrap();
    assert_eq!(snapshot.version, version);
    assert_eq!(snapshot.code.allocation.address(), address);
    let other = process.snapshot(other).unwrap();
    assert_eq!(
        snapshot.code.allocation.segment,
        other.code.allocation.segment
    );
    process.try_service_links().unwrap();
    let mut reader = process.register().unwrap();
    let mut guest = nixe_cpu::state::a64::A64State::default();
    let mut frame =
        crate::abi::NativeFrame::new(&mut guest, crate::abi::PollBudget::new(100, 100).unwrap());
    let _active = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    assert_eq!(
        unsafe { process.directory.unit(address) }.unwrap().version,
        version
    );
    assert_eq!(
        unsafe { process.directory.unit(other.code.allocation.address()) }
            .unwrap()
            .version,
        other.version
    );
}

#[test]
fn frozen_publication_waits_for_reopen_and_shutdown_cancels_the_wait() {
    use std::sync::mpsc;
    use std::time::Duration;
    for shutdown in [false, true] {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[NOP, RET]);
        let work = work(&process, &queue, &mut samples, 0);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let cursor = AtomicU64::new(0);
        let prepared = frozen.prepare(output(&frozen), &cursor).unwrap();
        process.request(Reason::LinkPatch).unwrap();
        let mut stop = process.try_transition().unwrap().unwrap();
        stop.wait_closed().unwrap();
        let (started, ready) = mpsc::channel();
        let (finished, result) = mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                started.send(()).unwrap();
                finished.send(prepared.publish()).unwrap();
            });
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(matches!(
                result.recv_timeout(Duration::from_millis(20)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            if shutdown {
                process.request_shutdown().unwrap();
                assert!(matches!(
                    result.recv_timeout(Duration::from_secs(5)).unwrap(),
                    Err(Error::StalePublication | Error::Shutdown)
                ));
            } else {
                stop.drain_links().unwrap();
                stop.batch().unwrap().complete().unwrap();
                assert!(stop.try_reopen().unwrap());
                assert!(result.recv_timeout(Duration::from_secs(5)).unwrap().is_ok());
            }
        });
    }
}
