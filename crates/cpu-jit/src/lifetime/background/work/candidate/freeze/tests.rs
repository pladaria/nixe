use super::super::tests::work;
use super::*;
use crate::lifetime::background::tests::{setup, snapshot};
use crate::lifetime::unit::{
    EdgeKind,
    links::tests::{source, source_input},
    tests::{key, publish_words},
};
use crate::sampling::Successor;

const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

#[test]
fn freeze_does_not_export_demanded_internal_leaders_or_create_coverage_slots() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, NOP, RET]);
    publish_words(&process, 4, &[NOP, RET]);
    let work = work(&process, &queue, &mut samples, 0);
    let slots = process.lock().keys.len();
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    assert_eq!(candidate.graph().blocks.len(), 2);
    let frozen = candidate.freeze().unwrap();
    assert_eq!(frozen.entries(), &[0]);
    assert_eq!(frozen.graph().units.len(), 2);
    assert_eq!(
        frozen.dependencies(),
        &*frozen.graph().units[0].dependencies
    );
    let mut foreign = key(0);
    foreign.address_space = nixe_memory::AddressSpaceId::new(2);
    assert!(
        !frozen
            .graph()
            .contains(InstructionKey::new(foreign).unwrap())
    );
    assert_eq!(process.lock().keys.len(), slots);
    assert!(!process.lock().keys.contains_key(&key(8)));
    frozen.validate_locked(&process.lock()).unwrap();
    let analysis = frozen.analyze().unwrap();
    assert_eq!(analysis.blocks.len(), frozen.graph().blocks.len());
    assert_eq!(
        analysis.instructions.len(),
        frozen.graph().instructions.len()
    );
}

#[test]
fn freeze_exports_uninstalled_external_static_sources_but_not_internal_sources() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 8, &[RET]);
    let mut input = source_input(&process, 0, 8);
    input.instructions[0].bits = 0x14000002; // B +8; existing shared decoder semantics.
    process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input,
            &AtomicU64::new(0),
        )
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    let first = work(&process, &queue, &mut samples, 0);
    let frozen = first
        .reserve_candidate(Graph::discover(&first).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.entries(), &[0]);
    drop(frozen);
    drop(first);
    source(&process, &AtomicU64::new(0), 128, 8); // Leaves its static link pending.
    let next = work(&process, &queue, &mut samples, 0);
    let frozen = next
        .reserve_candidate(Graph::discover(&next).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.entries(), &[0, 1]);
    assert_eq!(frozen.graph().blocks[1].key, key(8));
}

#[test]
fn freeze_dynamic_samples_use_the_captured_terminal_not_the_first_split_block() {
    for dynamic in [false, true] {
        let (process, queue, mut samples) = setup(0);
        let branch = if dynamic { 0xd61f0000 } else { 0x14000003 }; // BR X0 or B +12.
        publish_words(&process, 0, &[NOP, branch]);
        publish_words(&process, 4, &[branch]);
        publish_words(&process, 16, &[RET]);
        let mut observed = snapshot(&process, 0);
        observed.successors[0] = Some(Successor {
            target: key(16),
            count: 8,
            sequence: 1,
        });
        process.admit_seed(&queue, &mut samples, observed).unwrap();
        let work = process
            .accept_background(queue.wait().unwrap().unwrap())
            .unwrap()
            .unwrap();
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let entries: Vec<_> = frozen
            .entries()
            .iter()
            .map(|&i| frozen.graph().blocks[i].key.pc.get())
            .collect();
        assert_eq!(entries, if dynamic { vec![0, 16] } else { vec![0] });
    }
}

#[test]
fn freeze_sees_registered_indirect_and_return_roots() {
    for kind in [EdgeKind::Indirect, EdgeKind::Return] {
        let (process, queue, mut samples) = setup(0);
        publish_words(&process, 0, &[NOP, RET]);
        publish_words(&process, 4, &[RET]);
        let src =
            crate::lifetime::unit::dynamic::tests::source(&process, &AtomicU64::new(0), 128, kind);
        let mut reader = process.register().unwrap();
        reader
            .cache_bridge(
                process
                    .prepare_dynamic_bridge(src, 0, key(4))
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
        let work = work(&process, &queue, &mut samples, 0);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert_eq!(frozen.entries(), &[0, 1]);
    }
}

#[test]
fn freeze_never_reselects_late_incoming_links_and_rejects_replaced_inputs() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    publish_words(&process, 4, &[RET]);
    let src = crate::lifetime::unit::dynamic::tests::source(
        &process,
        &AtomicU64::new(0),
        128,
        EdgeKind::Return,
    );
    let mut reader = process.register().unwrap();
    let work = work(&process, &queue, &mut samples, 0);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.entries(), &[0]);
    reader
        .cache_bridge(
            process
                .prepare_dynamic_bridge(src, 0, key(4))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    // A new demand not affecting any captured input cannot change this list.
    publish_words(&process, 16, &[RET]);
    frozen.check().unwrap();
    frozen.validate_locked(&process.lock()).unwrap();
    assert_eq!(frozen.entries(), &[0]);
    publish_words(&process, 4, &[RET]);
    assert_eq!(frozen.check(), Err(Error::StalePublication));
}

#[test]
fn freeze_cancels_a_new_external_interior_entry_before_backend_work() {
    let (process, queue, mut samples) = setup(0);
    publish_words(&process, 0, &[NOP, RET]);
    let src = crate::lifetime::unit::dynamic::tests::source(
        &process,
        &AtomicU64::new(0),
        128,
        EdgeKind::Indirect,
    );
    let mut reader = process.register().unwrap();
    let work = work(&process, &queue, &mut samples, 0);
    let candidate = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap();
    assert_eq!(candidate.graph().blocks.len(), 1);
    publish_words(&process, 4, &[RET]);
    reader
        .cache_bridge(
            process
                .prepare_dynamic_bridge(src, 0, key(4))
                .unwrap()
                .unwrap(),
        )
        .unwrap();
    assert!(matches!(candidate.freeze(), Err(CompileError::Cancelled)));
    assert!(process.lock().candidates.entries.is_empty());
    work.check().unwrap();
}

#[test]
fn freeze_uses_the_source_instruction_not_partial_baseline_unit_membership() {
    let (process, queue, mut samples) = setup(0);
    let mut input = source_input(&process, 0, 4);
    input.instructions = [NOP, NOP, NOP, 0x17fffffe]
        .into_iter()
        .enumerate()
        .map(|(i, bits)| unit::Instruction {
            key: InstructionKey::new(key(i as u64 * 4)).unwrap(),
            bits,
        })
        .collect();
    input.states[0].exit.as_mut().unwrap().pc = key(12).pc;
    input.states[0].transfer.as_mut().unwrap().completed = 4;
    process
        .prepare_unit(
            &[process.reserve(key(0)).unwrap()],
            input,
            &AtomicU64::new(0),
        )
        .unwrap()
        .publish()
        .unwrap();
    publish_words(&process, 4, &[NOP, NOP, 0x17fffffe]);
    process.try_service_links().unwrap();
    publish_words(&process, 8, &[NOP, 0x17fffffe]);
    crate::lifetime::unit::tests::publish(&process, &AtomicU64::new(0), &[8], Tier::Hcq);
    process.try_service_links().unwrap();
    let work = work(&process, &queue, &mut samples, 0);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.graph().instructions.len(), 2);
    assert_eq!(frozen.entries(), &[0, 1]);
    assert_eq!(frozen.graph().blocks[1].key, key(4));
}

#[test]
fn freeze_dependency_union_is_unique_deterministic_and_preserves_mapping_identities() {
    use nixe_memory::{GuestPhysicalPageId, MappingGeneration};
    let dep = |page, generation| CodePageDependency {
        page: GuestPhysicalPageId::new(page),
        mapping_generation: MappingGeneration::new(generation),
    };
    let a = [dep(2, 1), dep(1, 4)];
    let b = [dep(2, 1), dep(1, 5)];
    let expected = vec![dep(1, 4), dep(1, 5), dep(2, 1)];
    assert_eq!(
        dependency_union([a.as_slice(), b.as_slice()].into_iter()),
        expected
    );
    assert_eq!(
        dependency_union([b.as_slice(), a.as_slice()].into_iter()),
        expected
    );
}
