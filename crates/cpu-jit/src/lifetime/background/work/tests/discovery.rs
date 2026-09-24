use super::*;
use crate::hcq::{Exit, Graph, Target};
use crate::lifetime::unit::tests::publish_words;
use crate::sampling::Successor;

const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

#[test]
fn hcq_discovery_follows_a_demanded_diamond_and_loop_without_allocating_missing_slots() {
    let (process, queue, mut samples) = setup(0);
    for (pc, bits) in [
        (0, 0x54000040),
        (4, 0x14000002),
        (8, 0x14000001),
        (12, 0x17fffffd),
    ] {
        publish_words(&process, pc, &[bits]);
    }
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let slots = process.lock().keys.len();
    let graph = Graph::discover(&work).unwrap();
    assert_eq!(graph.instructions.len(), 4);
    assert_eq!(graph.inputs.len(), 4);
    assert_eq!(
        graph.blocks[0].exit,
        Exit::Conditional {
            fallthrough: Target::Internal(1),
            taken: Target::Internal(2)
        }
    );
    assert_eq!(graph.blocks[3].exit, Exit::Jump(Target::Internal(0)));
    assert_eq!(process.lock().keys.len(), slots);
}

#[test]
fn hcq_discovery_stops_before_foreign_interior_membership() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, NOP, RET]);
    publish_words(&process, 8, &[NOP, RET]);
    publish(&process, &AtomicU64::new(0), &[8], Tier::Hcq);
    process.try_service_links().unwrap();
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let input = work.lcq(key(0)).unwrap().unwrap();
    assert_eq!(work.extent(&input).unwrap().instructions, 2);
    let graph = Graph::discover(&work).unwrap();
    assert_eq!(graph.instructions.len(), 2);
    assert_eq!(graph.inputs[0].instructions, 2);
    assert_eq!(
        graph.blocks[0].exit,
        Exit::Fallthrough(Target::External(key(8)))
    );
}

#[test]
fn hcq_discovery_does_not_pull_callees_or_return_continuations_from_samples() {
    for branch in [0x94000002, RET] {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[branch]);
        publish_words(&process, 4, &[RET]);
        publish_words(&process, 8, &[RET]);
        let mut observed = snapshot(&process, 0);
        observed.successors[0] = Some(Successor {
            target: key(8),
            count: 8,
            sequence: 1,
        });
        observed.successors[1] = Some(Successor {
            target: key(4),
            count: 8,
            sequence: 2,
        });
        assert_eq!(
            process.admit_seed(&queue, &mut samples, observed).unwrap(),
            Outcome::Queued
        );
        let work = process
            .accept_background(queue.wait().unwrap().unwrap())
            .unwrap()
            .unwrap();
        let graph = Graph::discover(&work).unwrap();
        assert_eq!(graph.instructions.len(), 1);
        assert_eq!(graph.units.len(), 1);
    }
}

#[test]
fn hcq_discovery_sample_priority_selects_whole_blocks_under_the_ceiling() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[0xd61f0000]); // BR X0: sampled targets are eligible.
    let mut bits = [NOP; 512];
    for pc in [0x1000, 0x2000, 0x3000, 0x4000] {
        bits[511] = if pc == 0x3000 { 0x14000601 } else { RET }; // B 0x5000.
        publish_words(&process, pc, &bits);
    }
    publish_words(&process, 0x5000, &[RET]);
    let mut observed = snapshot(&process, 0);
    for (slot, (pc, count, sequence)) in observed.successors.iter_mut().zip([
        (0x2000, 10, 5),
        (0x1000, 10, 5),
        (0x4000, 20, 1),
        (0x3000, 10, 20),
    ]) {
        *slot = Some(Successor {
            target: key(pc),
            count,
            sequence,
        });
    }
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let graph = Graph::discover(&work).unwrap();
    // A 512-word miss must not stop the queue: the later one-word target fits.
    assert_eq!(graph.instructions.len(), 1538);
    assert_eq!(
        graph
            .inputs
            .iter()
            .map(|i| i.key.pc.get())
            .collect::<Vec<_>>(),
        [0, 0x1000, 0x3000, 0x4000, 0x5000]
    );
}

#[test]
fn hcq_discovery_revalidates_a_captured_input_before_scanning_its_extent() {
    let (process, queue, mut samples) = setup(2);
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let old = work.lcq(key(4)).unwrap().unwrap();
    publish_words(&process, 4, &[RET]);
    assert!(matches!(work.extent(&old), Err(Error::StalePublication)));
}

#[test]
fn hcq_discovery_leaves_an_undemanded_successor_external_without_creating_a_slot() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[0x14000080]); // B 0x200, never demanded.
    enqueue(&process, &queue, &mut samples, 0);
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let graph = Graph::discover(&work).unwrap();
    assert_eq!(graph.instructions.len(), 1);
    assert_eq!(
        graph.blocks[0].exit,
        Exit::Jump(Target::External(key(0x200)))
    );
    assert_eq!(process.lock().keys.len(), 1);
}

#[test]
fn hcq_discovery_merges_interior_demands_but_cancels_conflicting_captured_words() {
    for conflict in [false, true] {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[NOP, NOP, RET]);
        publish_words(&process, 4, &[NOP, if conflict { NOP } else { RET }]);
        enqueue(&process, &queue, &mut samples, 0);
        let work = process
            .accept_background(queue.wait().unwrap().unwrap())
            .unwrap()
            .unwrap();
        let result = Graph::discover(&work);
        if conflict {
            assert!(matches!(
                result,
                Err(crate::hcq::DiscoveryError::Interrupted(
                    crate::lifetime::background::workers::CompileError::Cancelled
                ))
            ));
        } else {
            let graph = result.unwrap();
            assert_eq!(graph.instructions.len(), 3);
            assert_eq!(graph.inputs.len(), 2);
            assert_eq!(graph.blocks.len(), 2);
        }
    }
}

#[test]
fn hcq_discovery_accepts_exactly_2048_distinct_words_and_leaves_the_next_edge_external() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[0xd61f0000]);
    for pc in [0x1000, 0x2000, 0x3000, 0x4000] {
        let mut bits = vec![NOP; if pc == 0x1000 { 511 } else { 512 }];
        *bits.last_mut().unwrap() = if pc == 0x4000 { 0x14000201 } else { RET }; // B 0x5000.
        publish_words(&process, pc, &bits);
    }
    publish_words(&process, 0x5000, &[RET]);
    let mut observed = snapshot(&process, 0);
    for (slot, pc) in observed
        .successors
        .iter_mut()
        .zip([0x4000, 0x3000, 0x2000, 0x1000])
    {
        *slot = Some(Successor {
            target: key(pc),
            count: 1,
            sequence: 1,
        });
    }
    assert_eq!(
        process.admit_seed(&queue, &mut samples, observed).unwrap(),
        Outcome::Queued
    );
    let work = process
        .accept_background(queue.wait().unwrap().unwrap())
        .unwrap()
        .unwrap();
    let graph = Graph::discover(&work).unwrap();
    assert_eq!(graph.instructions.len(), 2048);
    assert_eq!(graph.inputs.len(), 5);
    assert_eq!(
        graph.blocks.last().unwrap().exit,
        Exit::Jump(Target::External(key(0x5000)))
    );
}
