use super::*;
use crate::hcq::{Exit, Target};
use crate::lifetime::background::tests::{setup, snapshot};
use crate::lifetime::unit::tests::{key, publish_words};

const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

pub(super) fn work<'p>(
    process: &'p Lifetime,
    queue: &Queue,
    samples: &mut Samples,
    pc: u64,
) -> Work<'p> {
    assert_eq!(
        process
            .admit_seed(queue, samples, snapshot(process, pc))
            .unwrap(),
        Outcome::Queued
    );
    process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap()
}

#[test]
fn successor_collision_trims_the_prefix_without_rejecting_the_seed() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    let a = work(&process, &queue, &mut samples, 0);
    let b = work(&process, &queue, &mut samples, 4);
    let winner = b.reserve_candidate(Graph::discover(&b).unwrap()).unwrap();
    let prefix = a.reserve_candidate(Graph::discover(&a).unwrap()).unwrap();
    assert_eq!(prefix.graph().instructions.len(), 1);
    assert_eq!(prefix.graph().inputs.len(), 1);
    assert_eq!(prefix.graph().units.len(), 1);
    assert_eq!(prefix.graph().inputs[0].instructions, 1);
    assert_eq!(
        prefix.graph().blocks[0].exit,
        Exit::Fallthrough(Target::External(key(4)))
    );
    assert_eq!(process.lock().candidates.entries.len(), 3);
    prefix.check().unwrap();
    winner.check().unwrap();
    drop(prefix);
    winner.check().unwrap();
    drop(winner);
    // The same seed job remains usable; no REJECTED token was written.
    let next = a.reserve_candidate(Graph::discover(&a).unwrap()).unwrap();
    assert_eq!(next.graph().instructions.len(), 3);
    next.check().unwrap();
    drop(next);
    assert!(process.lock().candidates.entries.is_empty());
}

#[test]
fn interior_collision_drops_only_the_disconnected_tail_of_a_diamond() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[0x54000080]); // B.EQ 16.
    publish_words(&process, 4, &[NOP, NOP, 0x14000005]); // B 32.
    publish_words(&process, 16, &[RET]);
    publish_words(&process, 32, &[RET]);
    let a = work(&process, &queue, &mut samples, 0);
    let graph = Graph::discover(&a).unwrap();
    // A later demand introduces a collision inside the old canonical block.
    publish_words(&process, 8, &[NOP, 0x14000005]);
    let b = work(&process, &queue, &mut samples, 8);
    let winner = b.reserve_candidate(Graph::discover(&b).unwrap()).unwrap();
    let candidate = a.reserve_candidate(graph).unwrap();
    let graph = candidate.graph();
    assert_eq!(
        graph
            .instructions
            .iter()
            .map(|w| w.instruction.key.block_key().pc.get())
            .collect::<Vec<_>>(),
        [0, 4, 16]
    );
    assert_eq!(
        graph.blocks[0].exit,
        Exit::Conditional {
            fallthrough: Target::Internal(1),
            taken: Target::Internal(2)
        }
    );
    assert_eq!(
        graph.blocks[1].exit,
        Exit::Fallthrough(Target::External(key(8)))
    );
    assert_eq!(graph.inputs.len(), 3);
    assert_eq!(graph.units.len(), 3);
    candidate.check().unwrap();
    winner.check().unwrap();
}

#[test]
fn trim_keeps_a_tail_reachable_from_an_independent_branch() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[0x54000060]); // B.EQ 12.
    publish_words(&process, 4, &[NOP, NOP, RET]);
    publish_words(&process, 12, &[RET]);
    let a = work(&process, &queue, &mut samples, 0);
    let graph = Graph::discover(&a).unwrap();
    // Claim just 8: the other branch still reaches the demanded leader at 12.
    {
        let mut state = process.lock();
        state.candidates = Index::new(8);
        state.candidates.insert(Claim {
            key: InstructionKey::new(key(8)).unwrap(),
            token: u64::MAX,
        });
    }
    let candidate = a.reserve_candidate(graph).unwrap();
    assert_eq!(
        candidate
            .graph()
            .instructions
            .iter()
            .map(|w| w.instruction.key.block_key().pc.get())
            .collect::<Vec<_>>(),
        [0, 4, 12]
    );
    assert_eq!(
        candidate.graph().blocks[1].exit,
        Exit::Fallthrough(Target::External(key(8)))
    );
    assert_eq!(
        candidate.graph().blocks[0].exit,
        Exit::Conditional {
            fallthrough: Target::Internal(1),
            taken: Target::Internal(2)
        }
    );
    candidate.check().unwrap();
    drop(candidate);
    assert_eq!(process.lock().candidates.entries.len(), 1);
}

#[test]
fn promotion_racing_discovery_trims_its_changed_successor_payload() {
    use crate::lifetime::unit::tests::publish;
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let graph = Graph::discover(&work).unwrap();
    publish(&process, &AtomicU64::new(0), &[4], Tier::Hcq);
    process.try_service_links().unwrap();
    let candidate = work.reserve_candidate(graph).unwrap();
    assert_eq!(candidate.graph().instructions.len(), 1);
    assert_eq!(candidate.graph().inputs.len(), 1);
    candidate.check().unwrap();
}

#[test]
fn a_second_collision_defers_without_partial_claims_or_retrimming() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    publish_words(&process, 8, &[RET]);
    let a = work(&process, &queue, &mut samples, 0);
    let b = work(&process, &queue, &mut samples, 8);
    let winner = b.reserve_candidate(Graph::discover(&b).unwrap()).unwrap();
    let (graph, needed, capacity) = a.trim_candidate(Graph::discover(&a).unwrap()).unwrap();
    assert_eq!(graph.instructions.len(), 2);
    let c = work(&process, &queue, &mut samples, 4);
    let newer = c.reserve_candidate(Graph::discover(&c).unwrap()).unwrap();
    assert!(matches!(
        a.reserve_trimmed_candidate(graph, needed, capacity, false),
        Err(CompileError::Deferred)
    ));
    assert!(
        process
            .lock()
            .candidates
            .get(InstructionKey::new(key(0)).unwrap())
            .is_none()
    );
    newer.check().unwrap();
    winner.check().unwrap();
    assert_eq!(process.lock().candidates.entries.len(), 2);
}

#[test]
fn trim_preserves_only_samples_from_a_reachable_indirect_terminal() {
    use crate::sampling::Successor;
    for (collision, expected) in [(4, vec![0]), (16, vec![0, 4, 32])] {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[NOP, 0xd61f0000]); // BR X0.
        publish_words(&process, 16, &[RET]);
        publish_words(&process, 32, &[RET]);
        let mut observed = snapshot(&process, 0);
        for (slot, pc) in observed.successors.iter_mut().zip([16, 32]) {
            *slot = Some(Successor {
                target: key(pc),
                count: 8,
                sequence: 1,
            });
        }
        process.admit_seed(&queue, &mut samples, observed).unwrap();
        let a = process
            .accept_background(queue.wait().unwrap().unwrap())
            .unwrap()
            .unwrap();
        let graph = Graph::discover(&a).unwrap();
        // An interior BR collision must disconnect both observed successors.
        if collision == 4 {
            publish_words(&process, 4, &[0xd61f0000]);
        }
        let b = work(&process, &queue, &mut samples, collision);
        let winner = b.reserve_candidate(Graph::discover(&b).unwrap()).unwrap();
        let candidate = a.reserve_candidate(graph).unwrap();
        assert_eq!(
            candidate
                .graph()
                .instructions
                .iter()
                .map(|w| w.instruction.key.block_key().pc.get())
                .collect::<Vec<_>>(),
            expected
        );
        let frozen = candidate.freeze().unwrap();
        assert_eq!(frozen.entries().len(), if collision == 4 { 1 } else { 2 });
        frozen.analyze().unwrap();
        winner.check().unwrap();
    }
}

#[test]
fn candidate_validates_nonseed_inputs_before_reserving_any_instruction() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    publish_words(&process, 4, &[RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let graph = Graph::discover(&work).unwrap();
    publish_words(&process, 4, &[RET]);
    assert!(matches!(
        work.reserve_candidate(graph),
        Err(CompileError::Cancelled)
    ));
    assert!(process.lock().candidates.entries.is_empty());
}

#[test]
fn candidate_rechecks_nonseed_versions_after_acquisition() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    publish_words(&process, 4, &[RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    publish_words(&process, 4, &[RET]);
    assert_eq!(candidate.check(), Err(Error::StalePublication));
    drop(candidate);
    assert!(process.lock().candidates.entries.is_empty());
}

#[test]
fn candidate_claims_survive_maintenance_until_their_owner_releases_them() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    publish_words(&process, 4, &[RET]);
    let old_work = work(&process, &queue, &mut samples, 0);
    let old = old_work
        .reserve_candidate(Graph::discover(&old_work).unwrap())
        .unwrap();
    process.request(Reason::LinkPatch).unwrap();
    process.try_service_links().unwrap();
    old.check().unwrap();
    let current_work = work(&process, &queue, &mut samples, 4);
    assert!(matches!(
        current_work.reserve_candidate(Graph::discover(&current_work).unwrap()),
        Err(CompileError::Deferred)
    ));
    assert_eq!(process.lock().candidates.entries.len(), 2);
    drop(old);
    assert!(process.lock().candidates.entries.is_empty());
    let current = current_work
        .reserve_candidate(Graph::discover(&current_work).unwrap())
        .unwrap();
    assert_eq!(process.lock().candidates.entries.len(), 1);
    current.check().unwrap();
}

#[test]
fn candidate_unwind_releases_keys_and_reuses_accounted_index_capacity() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let first = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    let capacity = process.lock().candidates.entries.capacity();
    let charged = process.cache.usage().unwrap().metadata;
    assert!(process.lock().candidates.charge.is_some());
    drop(first);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _candidate = work
                .reserve_candidate(Graph::discover(&work).unwrap())
                .unwrap();
            panic!("compiler unwind");
        }))
        .is_err()
    );
    assert!(process.lock().candidates.entries.is_empty());
    assert_eq!(process.lock().candidates.entries.capacity(), capacity);
    assert_eq!(process.cache.usage().unwrap().metadata, charged);
    drop(work);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.lock().candidates.entries.capacity(), 0);
    assert!(process.lock().candidates.charge.is_none());
}

#[test]
fn candidate_index_growth_preserves_other_live_claims() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[RET]);
    let mut words = vec![NOP; 64];
    words.push(RET);
    publish_words(&process, 16, &words);
    let a = work(&process, &queue, &mut samples, 0);
    let b = work(&process, &queue, &mut samples, 16);
    let first = a.reserve_candidate(Graph::discover(&a).unwrap()).unwrap();
    let capacity = process.lock().candidates.entries.capacity();
    let second = b.reserve_candidate(Graph::discover(&b).unwrap()).unwrap();
    assert!(process.lock().candidates.entries.capacity() > capacity);
    assert_eq!(process.lock().candidates.entries.len(), 66);
    first.check().unwrap();
    second.check().unwrap();
    drop(second);
    first.check().unwrap();
    assert_eq!(process.lock().candidates.entries.len(), 1);
    drop(first);
    assert!(process.lock().candidates.entries.is_empty());
}

#[test]
fn candidate_keeps_shutdown_pending_until_compiler_protection_is_released() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    assert!(!process.try_shutdown().unwrap());
    assert!(candidate.check().is_err());
    assert_eq!(process.lock().candidates.entries.len(), 1);
    drop(candidate);
    assert!(process.lock().candidates.entries.is_empty());
    assert!(!process.try_shutdown().unwrap());
    drop(work);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.lock().candidates.entries.capacity(), 0);
}

#[test]
fn independent_candidates_remain_claimed_concurrently_without_holding_jit_state() {
    use std::sync::mpsc;
    use std::time::Duration;
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[RET]);
    publish_words(&process, 16, &[RET]);
    let jobs: Vec<_> = [0, 16]
        .into_iter()
        .map(|pc| {
            process
                .admit_seed(&queue, &mut samples, snapshot(&process, pc))
                .unwrap();
            queue.wait().unwrap().unwrap()
        })
        .collect();
    std::thread::scope(|scope| {
        let (ready, received) = mpsc::channel();
        let mut releases = Vec::new();
        for job in jobs {
            let process = &process;
            let ready = ready.clone();
            let (release, wait) = mpsc::channel();
            releases.push(release);
            scope.spawn(move || {
                let work = process.accept_background(job).unwrap().unwrap();
                let candidate = work
                    .reserve_candidate(Graph::discover(&work).unwrap())
                    .unwrap();
                ready.send(()).unwrap();
                wait.recv().unwrap();
                candidate.check().unwrap();
            });
        }
        for _ in 0..2 {
            received.recv_timeout(Duration::from_secs(10)).unwrap();
        }
        assert_eq!(process.lock().candidates.entries.len(), 2);
        for release in releases {
            release.send(()).unwrap();
        }
    });
    assert!(process.lock().candidates.entries.is_empty());
}
