use super::*;
use crate::hcq::tests::{NOP, key, words};

#[test]
fn seed_samples_counts_recency_and_pc_precede_direct_and_conditional_edges() {
    let mut queue = Worklist::new(key(100));
    for (pc, count, sequence) in [(16, 3, 9), (12, 3, 10), (8, 3, 10), (20, 4, 1)] {
        queue.sample(Successor {
            target: key(pc),
            count,
            sequence,
        });
    }
    queue.successors(&Exit::Conditional {
        fallthrough: Target::External(key(4)),
        taken: Target::External(key(0)),
    });
    queue.successors(&Exit::Jump(Target::External(key(24))));
    let result: Vec<_> = std::iter::from_fn(|| queue.pop())
        .map(|k| k.pc.get())
        .collect();
    assert_eq!(result, [100, 20, 8, 12, 16, 24, 4, 0]);
}

#[test]
fn queue_deduplicates_priorities_and_never_accepts_foreign_execution_keys() {
    let mut queue = Worklist::new(key(0));
    queue.push(key(4), 2, 0, 0);
    queue.sample(Successor {
        target: key(4),
        count: 1,
        sequence: 1,
    });
    queue.push(
        BlockKey {
            fp: crate::abi::FpSpecialization::Exact(0),
            ..key(8)
        },
        1,
        0,
        0,
    );
    assert_eq!(queue.pop(), Some(key(0)));
    assert_eq!(queue.pop(), Some(key(4)));
    queue.push(key(4), 0, 0, 0);
    assert_eq!(queue.pop(), None);
}

#[test]
fn sample_filter_does_not_follow_calls_returns_or_semantic_boundaries() {
    for exit in [
        Exit::Call(Some(key(4))),
        Exit::Call(None),
        Exit::Return,
        Exit::Boundary(End::Architectural),
    ] {
        assert!(!permits_sample(&exit, key(4)));
        let mut queue = Worklist::new(key(0));
        queue.successors(&exit);
        assert_eq!(queue.pop(), Some(key(0)));
        assert_eq!(queue.pop(), None);
    }
    assert!(permits_sample(&Exit::Indirect, key(4)));
    assert!(permits_sample(
        &Exit::Jump(Target::External(key(4))),
        key(4)
    ));
    assert!(!permits_sample(
        &Exit::Jump(Target::External(key(4))),
        key(8)
    ));
}

#[test]
fn whole_block_selection_accepts_2048_and_refuses_2049_without_partial_mutation() {
    let mut builder = Builder::new(key(0));
    builder.merge(key(0), &words(0, &vec![NOP; 1536])).unwrap();
    assert_eq!(
        select_prefix(
            &builder,
            key(0x10000),
            &words(0x10000, &[NOP; 512]),
            MAX_INSTRUCTIONS
        ),
        Ok(512)
    );
    assert_eq!(
        select_prefix(
            &builder,
            key(0x10000),
            &words(0x10000, &[NOP; 513]),
            MAX_INSTRUCTIONS
        ),
        Ok(0)
    );
    assert_eq!(builder.words.len(), 1536);
}

#[test]
fn an_interior_leader_allows_a_complete_prefix_but_never_an_arbitrary_slice() {
    let mut builder = Builder::new(key(0));
    builder.merge(key(0), &words(0, &vec![NOP; 1792])).unwrap();
    let incoming = words(0x10000, &[NOP; 512]);
    assert_eq!(
        select_prefix(&builder, key(0x10000), &incoming, MAX_INSTRUCTIONS),
        Ok(0)
    );
    builder.leader(key(0x10400)).unwrap();
    assert_eq!(
        select_prefix(&builder, key(0x10000), &incoming, MAX_INSTRUCTIONS),
        Ok(256)
    );
}

#[test]
fn identical_overlap_costs_zero_but_conflicting_words_cancel_selection() {
    let mut builder = Builder::new(key(0));
    builder.merge(key(0), &words(0, &[NOP; 512])).unwrap();
    assert_eq!(
        select_prefix(&builder, key(4), &words(4, &[NOP; 512]), 513),
        Ok(512)
    );
    assert_eq!(
        select_prefix(&builder, key(4), &words(4, &[NOP; 512]), 512),
        Ok(0)
    );
    assert_eq!(
        select_prefix(&builder, key(4), &words(4, &[0xd65f03c0]), 2048),
        Err(Error::StaleCapture)
    );
}
