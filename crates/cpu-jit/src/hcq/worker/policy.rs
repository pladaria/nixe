//! Cold profitability estimates in sampled executions. The compiler cost proxy
//! is region size; benefit is new coverage, entries or a merged boundary. This
//! is deliberately bounded: a persistently hot small change eventually qualifies.

use crate::hcq::{Graph, Target};

pub(super) fn required_samples(
    graph: &Graph,
    predecessor_words: usize,
    predecessor_entries: usize,
    predecessors: usize,
    entries: usize,
) -> u16 {
    if predecessors == 0 {
        let internal = graph.blocks.iter().any(|block| {
            block
                .successors()
                .iter()
                .any(|target| matches!(target, Target::Internal(_)))
        });
        return if internal || graph.instructions.len() >= 8 {
            8
        } else if graph.instructions.len() > 1 {
            32
        } else {
            128
        };
    }
    let growth = graph.instructions.len().saturating_sub(predecessor_words);
    let new_entries = entries.saturating_sub(predecessor_entries);
    let benefit = growth.div_ceil(16) + new_entries + predecessors.saturating_sub(1);
    let work = predecessor_words + graph.instructions.len();
    work.div_ceil(8 * benefit.max(1)).clamp(8, 128) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hcq::flow::tests::graph;

    #[test]
    fn a_tiny_leaf_requires_more_evidence_than_an_internal_loop() {
        let leaf = graph(&[(0, &[0xd65f03c0])]);
        let loop_body = graph(&[(0, &[0x14000000])]);
        assert_eq!(required_samples(&leaf, 0, 0, 0, 1), 128);
        assert_eq!(required_samples(&loop_body, 0, 0, 0, 1), 8);
    }

    #[test]
    fn small_entry_growth_is_batched_but_hot_changes_are_never_starved() {
        let body = vec![0xd503201f; 512];
        let region = graph(&[(0, &body)]);
        let one_entry = required_samples(&region, 512, 1, 1, 2);
        let gathered_entries = required_samples(&region, 512, 1, 1, 9);
        assert_eq!(one_entry, 128);
        assert!(gathered_entries < one_entry);
        assert!(required_samples(&region, 512, 1, 1, 1) <= 128);
        assert!(required_samples(&region, 16, 1, 1, 2) < one_entry);
    }
}
