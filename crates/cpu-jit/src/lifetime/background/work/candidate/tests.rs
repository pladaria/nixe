use super::*;
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
fn candidate_batch_is_all_or_none_and_conflict_does_not_reject_the_seed() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    let a = work(&process, &queue, &mut samples, 0);
    let b = work(&process, &queue, &mut samples, 4);
    let winner = b.reserve_candidate(Graph::discover(&b).unwrap()).unwrap();
    assert!(matches!(
        a.reserve_candidate(Graph::discover(&a).unwrap()),
        Err(CompileError::Deferred)
    ));
    assert_eq!(process.lock().candidates.entries.len(), 2);
    assert!(
        process
            .lock()
            .candidates
            .get(InstructionKey::new(key(0)).unwrap())
            .is_none()
    );
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
