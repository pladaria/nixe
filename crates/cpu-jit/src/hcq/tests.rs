use super::*;
use crate::abi::{FpSpecialization, InstructionKey};
use nixe_cpu::{platform::TargetPlatform, profile::ProcessCpuContext};
use nixe_memory::AddressSpaceId;

pub(super) const NOP: u32 = 0xd503201f;
const RET: u32 = 0xd65f03c0;

pub(super) fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1)),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}

pub(super) fn words(pc: u64, bits: &[u32]) -> Vec<Instruction> {
    bits.iter()
        .enumerate()
        .map(|(index, &bits)| Instruction {
            key: InstructionKey::new(key(pc.wrapping_add(index as u64 * 4))).unwrap(),
            bits,
        })
        .collect()
}

fn graph(seed: u64, inputs: &[(u64, &[u32])]) -> (Vec<Word>, Vec<Block>) {
    let mut builder = Builder::new(key(seed));
    for &(pc, bits) in inputs {
        builder.merge(key(pc), words(pc, bits)).unwrap();
    }
    builder.finish().unwrap()
}

fn shape(blocks: &[Block]) -> Vec<(BlockKey, Range<usize>, Exit)> {
    blocks
        .iter()
        .map(|b| (b.key, b.instructions.clone(), b.exit.clone()))
        .collect()
}

#[test]
fn overlapping_interior_inputs_have_one_copy_and_stable_seed_first_blocks() {
    let (first, a) = graph(4, &[(0, &[NOP, NOP, RET]), (4, &[NOP, RET])]);
    let (second, b) = graph(4, &[(4, &[NOP, RET]), (0, &[NOP, NOP, RET])]);
    assert_eq!(first.len(), 3);
    assert_eq!(second.len(), 3);
    assert_eq!(shape(&a), shape(&b));
    assert_eq!(a[0].key, key(4));
    assert_eq!(a[0].instructions, 1..3);
    assert_eq!(a[0].exit, Exit::Return);
    assert_eq!(a[1].instructions, 0..1);
    assert_eq!(a[1].exit, Exit::Fallthrough(Target::Internal(0)));
}

#[test]
fn diamond_and_backedge_resolve_to_unique_canonical_blocks() {
    // B.EQ 8; B 12; B 12; B 0.
    let (_, blocks) = graph(
        0,
        &[
            (12, &[0x17fffffd]),
            (8, &[0x14000001]),
            (0, &[0x54000040]),
            (4, &[0x14000002]),
        ],
    );
    assert_eq!(blocks.len(), 4);
    assert_eq!(
        blocks[0].exit,
        Exit::Conditional {
            fallthrough: Target::Internal(1),
            taken: Target::Internal(2),
        }
    );
    assert_eq!(blocks[1].exit, Exit::Jump(Target::Internal(3)));
    assert_eq!(blocks[2].exit, Exit::Jump(Target::Internal(3)));
    assert_eq!(blocks[3].exit, Exit::Jump(Target::Internal(0)));
}

#[test]
fn absent_successors_remain_external_without_invented_words() {
    let (instructions, blocks) = graph(0, &[(0, &[0x54000040])]);
    assert_eq!(instructions.len(), 1);
    assert_eq!(
        blocks[0].exit,
        Exit::Conditional {
            fallthrough: Target::External(key(4)),
            taken: Target::External(key(8)),
        }
    );
    let (_, blocks) = graph(0, &[(0, &[NOP]), (16, &[RET])]);
    assert_eq!(blocks[0].exit, Exit::Fallthrough(Target::External(key(4))));
}

#[test]
fn calls_and_returns_never_become_internal_edges() {
    let (_, blocks) = graph(0, &[(0, &[0x94000002]), (4, &[RET]), (8, &[0xd63f0000])]);
    assert_eq!(blocks[0].exit, Exit::Call(Some(key(8))));
    assert_eq!(blocks[1].exit, Exit::Return);
    assert_eq!(blocks[2].exit, Exit::Call(None));
    let (_, blocks) = graph(0, &[(0, &[0xd61f0000])]);
    assert_eq!(blocks[0].exit, Exit::Indirect);
}

#[test]
fn compare_and_test_branches_share_signed_target_semantics() {
    for bits in [0xb4ffffe0, 0x36ffffe0] {
        // CBZ X0, -4; TBZ W0, #31, -4.
        let (_, blocks) = graph(4, &[(4, &[bits])]);
        assert_eq!(
            blocks[0].exit,
            Exit::Conditional {
                fallthrough: Target::External(key(8)),
                taken: Target::External(key(0)),
            }
        );
    }
    let (_, blocks) = graph(0, &[(0, &[0x17ffffff])]);
    assert_eq!(
        blocks[0].exit,
        Exit::Jump(Target::External(key(u64::MAX - 3)))
    );
}

#[test]
fn semantic_stops_reuse_lcq_boundary_classification() {
    for (bits, expected) in [
        (0xd4000001, End::Architectural),
        (0xd4200000, End::Architectural),
        (0xd51b4400, End::FpMode),
        (0, End::Invalid),
    ] {
        let (_, blocks) = graph(0, &[(0, &[bits]), (4, &[RET])]);
        assert_eq!(blocks[0].exit, Exit::Boundary(expected));
    }
}

#[test]
fn observed_interior_targets_split_without_becoming_public_entries() {
    let mut builder = Builder::new(key(0));
    builder.merge(key(0), words(0, &[NOP, NOP, RET])).unwrap();
    builder.leader(key(4)).unwrap();
    builder.leader(key(100)).unwrap();
    let (instructions, blocks) = builder.finish().unwrap();
    assert_eq!(instructions.len(), 3);
    assert_eq!(blocks.len(), 2); // Absent observation does not invent coverage.
    assert_eq!(blocks[0].exit, Exit::Fallthrough(Target::Internal(1)));
}

#[test]
fn direct_backedge_splits_an_interior_leader_without_an_extra_lcq_root() {
    let (instructions, blocks) = graph(0, &[(0, &[NOP, NOP, 0x17ffffff])]);
    assert_eq!(instructions.len(), 3);
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].instructions, 0..1);
    assert_eq!(blocks[0].exit, Exit::Fallthrough(Target::Internal(1)));
    assert_eq!(blocks[1].instructions, 1..3);
    assert_eq!(blocks[1].exit, Exit::Jump(Target::Internal(1)));
}

#[test]
fn conflicting_bytes_or_execution_context_are_stale() {
    let mut builder = Builder::new(key(0));
    builder.merge(key(0), words(0, &[NOP, RET])).unwrap();
    assert_eq!(
        builder.merge(key(4), words(4, &[NOP])),
        Err(Error::StaleCapture)
    );
    let foreign = BlockKey {
        fp: FpSpecialization::Exact(0),
        ..key(0)
    };
    assert_eq!(builder.leader(foreign), Err(Error::StaleCapture));
    assert_eq!(
        builder.merge(
            key(0),
            [Instruction {
                key: InstructionKey::new(foreign).unwrap(),
                bits: NOP,
            }]
        ),
        Err(Error::StaleCapture)
    );
}

#[test]
fn malformed_input_and_empty_seed_are_not_fabricated_as_code() {
    let mut builder = Builder::new(key(0));
    assert_eq!(
        builder.merge(key(0), words(4, &[NOP])),
        Err(Error::InvalidInput(
            "HCQ LCQ image is not contiguous from its root"
        ))
    );
    assert!(matches!(
        Builder::new(key(0)).finish(),
        Err(Error::EmptySeed)
    ));
}

#[test]
fn overlap_count_is_distinct_guest_words_not_fragment_lengths() {
    let mut builder = Builder::new(key(0));
    let bits = [NOP; 512];
    for pc in [0, 512, 1024, 1536] {
        builder.merge(key(pc), words(pc, &bits)).unwrap();
    }
    let (instructions, _) = builder.finish().unwrap();
    assert_eq!(instructions.len(), 896); // Four overlapping 512-word inputs.
}
