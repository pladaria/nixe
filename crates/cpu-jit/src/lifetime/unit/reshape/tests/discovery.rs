use super::*;
use crate::hcq::{DiscoveryError, Exit, Graph, StructuralReason, Target};
use crate::lifetime::background::{Work, workers::CompileError};
use crate::lifetime::unit::tests::publish_words;

pub(super) const NOP: u32 = 0xd503201f;
pub(super) const RET: u32 = 0xd65f03c0;

mod rejection;

pub(super) fn owned(process: &Lifetime, words: &[(u64, u32)]) -> UnitHandle {
    owned_entries(process, words, 1)
}

pub(super) fn owned_entries(
    process: &Lifetime,
    words: &[(u64, u32)],
    entries: usize,
) -> UnitHandle {
    let pcs: Vec<_> = words.iter().map(|word| word.0).collect();
    let mut candidate = input(process, &pcs, Tier::Hcq);
    for (instruction, &(_, bits)) in candidate.instructions.iter_mut().zip(words) {
        instruction.bits = bits;
    }
    candidate.entries = candidate
        .entries
        .into_vec()
        .into_iter()
        .take(entries)
        .collect();
    let publications: Vec<_> = pcs[..entries]
        .iter()
        .map(|pc| process.reserve(key(*pc)).unwrap())
        .collect();
    let handle = process
        .prepare_unit(&publications, candidate, &AtomicU64::new(0))
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    handle
}

pub(super) fn reshape(process: &Lifetime, block: u64, source: u64, target: u64) -> Work<'_> {
    let queue = Queue::new(1, process).unwrap().unwrap();
    let mut samples = Samples::new();
    let snapshot = heat(&mut samples, boundary(process, block, source, target));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(block), snapshot)
            .unwrap(),
        Outcome::Queued
    );
    process
        .accept_background(queue.pop().unwrap().unwrap())
        .unwrap()
        .unwrap()
}

#[test]
fn reshape_discovery_uses_zero_one_or_two_families_and_stops_at_a_third() {
    for owners in 0..=2 {
        let process = process();
        let source = publish_words(&process, 0, &[NOP, 0x14000003]); // B 16.
        let target = publish_words(&process, 16, &[NOP, 0x14000003]); // B 32.
        publish_words(&process, 32, &[NOP, RET]);
        owned(&process, &[(32, NOP), (36, RET)]);
        if owners >= 1 {
            owned(&process, &[(16, NOP), (20, 0x14000003)]);
        }
        if owners == 2 {
            owned(&process, &[(0, NOP), (4, 0x14000003)]);
        }
        let work = reshape(&process, 0, 4, 16);
        let slots = process.lock().keys.len();
        let graph = Graph::discover(&work).unwrap();
        assert_eq!(
            graph
                .instructions
                .iter()
                .map(|w| w.instruction.key.block_key().pc.get())
                .collect::<Vec<_>>(),
            [0, 4, 16, 20]
        );
        assert_eq!(graph.units.len(), 2);
        assert!(graph.units.iter().all(|unit| unit.tier == Tier::Lcq));
        assert!(
            graph
                .units
                .iter()
                .any(|unit| unit.registered_handle() == Some(source))
        );
        assert!(
            graph
                .units
                .iter()
                .any(|unit| unit.registered_handle() == Some(target))
        );
        assert_eq!(
            graph.blocks.last().unwrap().exit,
            Exit::Jump(Target::External(key(32)))
        );
        assert_eq!(process.lock().keys.len(), slots); // Source PC 4 is not demanded.
        assert!(!process.lock().keys.contains_key(&key(4)));
        let repeated = Graph::discover(&work).unwrap();
        assert_eq!(
            graph
                .blocks
                .iter()
                .map(|b| (&b.key, &b.instructions, &b.exit))
                .collect::<Vec<_>>(),
            repeated
                .blocks
                .iter()
                .map(|b| (&b.key, &b.instructions, &b.exit))
                .collect::<Vec<_>>()
        );
        let candidate = work.reserve_candidate(graph).unwrap();
        candidate.check().unwrap();
        assert_eq!(candidate.graph().instructions.len(), 4);
        drop(candidate);
        // Exact instruction claims are released, not permanent seed rejection.
        work.reserve_candidate(repeated).unwrap().check().unwrap();
    }
}

#[test]
fn reshape_discovery_accepts_a_retained_interior_entry_of_the_same_family() {
    let process = process();
    publish_words(&process, 0, &[0xd61f0000]); // BR X0 observed to PC 16.
    publish_words(&process, 16, &[NOP, RET]);
    owned(&process, &[(0, 0xd61f0000), (16, NOP), (20, RET)]);
    let work = reshape(&process, 0, 0, 16);
    let graph = Graph::discover(&work).unwrap();
    assert_eq!(graph.instructions.len(), 3);
    assert_eq!(graph.blocks[0].exit, Exit::Indirect);
    assert_eq!(graph.blocks[1].key, key(16));
    assert!(graph.units.iter().all(|unit| unit.tier == Tier::Lcq));
    work.reserve_candidate(graph).unwrap().check().unwrap();
}

#[test]
fn reshape_uses_the_previous_root_but_repartitions_if_it_cannot_reach_the_boundary() {
    for dynamic in [false, true] {
        let process = process();
        let root = if dynamic { 0xd61f0000 } else { 0x14000004 };
        publish_words(&process, 0, &[root]); // BR X0 or B 16.
        publish_words(&process, 16, &[0x14000004]); // B 32.
        publish_words(&process, 32, &[RET]);
        owned_entries(&process, &[(0, root), (16, 0x14000004), (32, RET)], 2);
        let work = reshape(&process, 16, 16, 32);
        let graph = Graph::discover(&work).unwrap();
        assert_eq!(graph.blocks[0].key, key(if dynamic { 16 } else { 0 }));
        assert_eq!(graph.instructions.len(), if dynamic { 2 } else { 3 });
        let frozen = work.reserve_candidate(graph).unwrap().freeze().unwrap();
        assert_eq!(frozen.replacement().fallbacks.len(), usize::from(dynamic));
        frozen.check().unwrap();
    }
}

fn seed(process: &Lifetime, pc: u64) -> Work<'_> {
    let queue = Queue::new(1, process).unwrap().unwrap();
    let observed = crate::sampling::AdmissionSnapshot {
        key: key(pc),
        version: process.reserve(key(pc)).unwrap().reachability,
        sequence: 8,
        last_edge: None,
        successors: [None; 4],
    };
    assert_eq!(
        process
            .admit_seed(&queue, &mut Samples::new(), observed)
            .unwrap(),
        Outcome::Queued
    );
    process
        .accept_background(queue.pop().unwrap().unwrap())
        .unwrap()
        .unwrap()
}

#[test]
fn reshape_claims_exclude_a_seed_at_an_unowned_required_endpoint() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[RET]);
    let first = reshape(&process, 0, 0, 16);
    let second = seed(&process, 16);
    let second_graph = Graph::discover(&second).unwrap();
    assert!(second_graph.discovery.is_none()); // Seeds have no negative evidence allocation.
    let candidate = first
        .reserve_candidate(Graph::discover(&first).unwrap())
        .unwrap();
    assert!(matches!(
        second.reserve_candidate(second_graph),
        Err(CompileError::Deferred)
    ));
    candidate.check().unwrap();
    drop(candidate);
    second
        .reserve_candidate(Graph::discover(&second).unwrap())
        .unwrap()
        .check()
        .unwrap();
}

#[test]
fn competing_zero_family_reshapes_never_reserve_a_partial_batch() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[RET]);
    publish_words(&process, 32, &[0x17fffffc]); // B 16.
    let first = reshape(&process, 0, 0, 16);
    let second = reshape(&process, 32, 32, 16);
    let graph = Graph::discover(&second).unwrap();
    let candidate = first
        .reserve_candidate(Graph::discover(&first).unwrap())
        .unwrap();
    assert!(matches!(
        second.reserve_candidate(graph),
        Err(CompileError::Deferred)
    ));
    // The failed reshape must not claim even its free root. A seed can use it,
    // with the shared occupied target still external.
    let independent = seed(&process, 32);
    let prefix = independent
        .reserve_candidate(Graph::discover(&independent).unwrap())
        .unwrap();
    assert_eq!(prefix.graph().instructions.len(), 1);
    candidate.check().unwrap();
    drop(prefix);
    drop(candidate);
    second
        .reserve_candidate(Graph::discover(&second).unwrap())
        .unwrap()
        .check()
        .unwrap();
}

#[test]
fn reshape_keeps_both_participants_when_an_optional_successor_is_claimed() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]); // B 16.
    publish_words(&process, 16, &[0x14000004]); // B 32.
    publish_words(&process, 32, &[RET]);
    owned(&process, &[(0, 0x14000004)]);
    owned(&process, &[(16, 0x14000004)]);
    let work = reshape(&process, 0, 0, 16);
    let graph = Graph::discover(&work).unwrap();
    let tail = seed(&process, 32);
    let claimed = tail
        .reserve_candidate(Graph::discover(&tail).unwrap())
        .unwrap();
    let candidate = work.reserve_candidate(graph).unwrap();
    assert_eq!(candidate.graph().instructions.len(), 2);
    assert_eq!(
        candidate.graph().blocks[1].exit,
        Exit::Jump(Target::External(key(32)))
    );
    candidate.check().unwrap();
    claimed.check().unwrap();
    drop(candidate);
    claimed.check().unwrap();
}

#[test]
fn reshape_cannot_continue_after_a_participant_is_retired() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    let owner = owned(&process, &[(16, RET)]);
    let work = reshape(&process, 0, 0, 16);
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    process.retire_unit(owner).unwrap();
    assert_eq!(candidate.check(), Err(Error::StalePublication));
    drop(candidate);
    drop(work);
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    let fresh = reshape(&process, 0, 0, 16);
    fresh
        .reserve_candidate(Graph::discover(&fresh).unwrap())
        .unwrap()
        .check()
        .unwrap();
}

#[test]
fn reshape_candidate_and_owner_reservation_survive_unrelated_maintenance() {
    let process = process();
    publish_words(&process, 0, &[0x14000004]);
    publish_words(&process, 16, &[RET]);
    owned(&process, &[(16, RET)]);
    let old = reshape(&process, 0, 0, 16);
    let candidate = old
        .reserve_candidate(Graph::discover(&old).unwrap())
        .unwrap();
    process.request(Reason::LinkPatch).unwrap();
    process.try_service_links().unwrap();
    candidate.check().unwrap();
    let queue = Queue::new(1, &process).unwrap().unwrap();
    let mut samples = Samples::new();
    let snapshot = heat(&mut samples, boundary(&process, 0, 0, 16));
    assert_eq!(
        process
            .admit_reshape(&queue, &mut samples, key(0), snapshot)
            .unwrap(),
        Outcome::Deferred
    );
    drop(candidate);
    drop(old);
    let current = reshape(&process, 0, 0, 16);
    current.check().unwrap();
    current
        .reserve_candidate(Graph::discover(&current).unwrap())
        .unwrap()
        .check()
        .unwrap();
}

#[test]
fn reshape_discovery_does_not_connect_calls_returns_or_unobserved_edges() {
    for bits in [0x94000004, RET, 0xd4000001, 0x14000008] {
        let process = process();
        publish_words(&process, 0, &[bits]);
        publish_words(&process, 16, &[RET]);
        let work = reshape(&process, 0, 0, 16);
        let result = Graph::discover(&work);
        if bits == 0x14000008 {
            // Its actual successor has no LCQ input; that remains retryable.
            assert!(matches!(
                result,
                Err(DiscoveryError::Interrupted(CompileError::Deferred))
            ));
        } else {
            assert!(matches!(result, Err(DiscoveryError::Structural(result))
                if result.reason() == StructuralReason::Disconnected));
        }
    }
}

#[test]
fn reshape_discovery_rejects_a_mandatory_source_disconnected_from_the_root() {
    let process = process();
    publish_words(&process, 0, &[RET]);
    publish_words(&process, 16, &[0x14000004]); // B 32.
    publish_words(&process, 32, &[RET]);
    // The source is a valid member of this family, but the supplied root cannot
    // reach it. Mandatory worklist inclusion must not fabricate connectivity.
    owned(&process, &[(0, RET), (16, 0x14000004)]);
    let work = reshape(&process, 0, 16, 32);
    assert!(matches!(
        Graph::discover(&work),
        Err(DiscoveryError::Structural(result)) if result.reason() == StructuralReason::Disconnected
    ));
}

#[test]
fn reshape_indirect_observation_belongs_to_the_actual_terminal_not_the_root() {
    for connected in [false, true] {
        let process = process();
        let root = if connected { 0x14000004 } else { 0xd61f0000 }; // B 16 or BR X0.
        publish_words(&process, 0, &[root]);
        publish_words(&process, 16, &[0xd61f0020]); // Actual observed BR X1 -> 32.
        publish_words(&process, 32, &[0x17fffffc]); // B 16.
        owned(&process, &[(0, root), (16, 0xd61f0020)]);
        let work = reshape(&process, 0, 16, 32);
        let result = Graph::discover(&work);
        if connected {
            let graph = result.unwrap();
            assert_eq!(graph.instructions.len(), 3);
            assert_eq!(graph.blocks[1].exit, Exit::Indirect);
            assert_eq!(graph.blocks[2].exit, Exit::Jump(Target::Internal(1)));
        } else {
            // Attaching 16's observation to root BR X0 would fabricate a path
            // 0 -> 32 -> 16 and incorrectly accept this disconnected graph.
            assert!(matches!(result, Err(DiscoveryError::Structural(result))
                if result.reason() == StructuralReason::Disconnected));
        }
    }
}

#[test]
fn reshape_discovery_prioritizes_the_required_target_under_the_shared_ceiling() {
    let process = process();
    publish_words(&process, 0, &[0x54008000]); // B.EQ 0x1000; fallthrough 4.
    publish_words(&process, 0x1000, &[RET]);
    for (pc, next) in [
        (4u64, 0x2000u64),
        (0x2000, 0x3000),
        (0x3000, 0x4000),
        (0x4000, 0x5000),
    ] {
        let mut words = vec![NOP; 512];
        let last = pc + 511 * 4;
        words[511] = 0x14000000 | (((next.wrapping_sub(last) / 4) as u32) & 0x03ff_ffff);
        publish_words(&process, pc, &words);
    }
    let work = reshape(&process, 0, 0, 0x1000);
    let graph = Graph::discover(&work).unwrap();
    assert!(graph.contains(instruction(0x1000)));
    assert_eq!(graph.instructions.len(), 1538); // The fourth full block cannot fit.
    assert!(!graph.contains(instruction(0x4000)));
    assert_eq!(graph.inputs.len(), 5);
    assert_eq!(graph.discovery.as_ref().unwrap().len(), 6); // Includes the input that did not fit.
    assert_eq!(
        graph
            .discovery
            .as_ref()
            .unwrap()
            .validate_no_op(&process.lock(), &graph),
        Err(Error::StalePublication)
    );
}
