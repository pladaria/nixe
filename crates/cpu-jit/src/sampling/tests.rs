use super::*;
use crate::abi::FpSpecialization;
use nixe_cpu::{platform::TargetPlatform, profile::ProcessCpuContext};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1)),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}

fn version(value: u64) -> ReachabilityVersion {
    ReachabilityVersion::new(value).unwrap()
}

fn edge(pc: u64) -> Option<ObservedEdge> {
    Some(ObservedEdge {
        destination: GuestVirtualAddress::new(pc),
        kind: EdgeKind::Indirect,
    })
}

fn boundary(pc: u64) -> BoundaryKey {
    BoundaryKey {
        source: InstructionKey::new(key(pc)).unwrap(),
        target: InstructionKey::new(key(pc + 4)).unwrap(),
        source_version: version(1),
        target_version: version(1),
        source_family: Some(FamilyIdentity {
            id: HcqFamilyId::new(1).unwrap(),
            version: FamilyVersion::new(1).unwrap(),
        }),
        target_family: None,
    }
}

fn seed_record(samples: &Samples, key: BlockKey) -> Seed {
    *samples.seeds[samples.seed_set(key)]
        .iter()
        .flatten()
        .find(|entry| entry.snapshot.key == key)
        .unwrap()
}

#[test]
fn thresholds_deferral_and_owned_snapshots() {
    let mut samples = Samples::new();
    for _ in 0..7 {
        assert!(samples.seed(key(0), version(1), edge(4), true).is_none());
    }
    let seed = samples.seed(key(0), version(1), edge(4), true).unwrap();
    assert_eq!(seed.sequence, 8);
    assert_eq!(seed.successors[0].unwrap().count, 8);
    samples.defer_seed(seed);
    assert_eq!(seed_record(&samples, key(0)).score, 7);
    let newer = samples.seed(key(0), version(1), edge(8), true).unwrap();
    samples.defer_seed(seed); // An old response cannot cool a newer observation.
    assert_eq!(seed_record(&samples, key(0)).score, 8);
    assert_eq!(seed.last_edge, edge(4));
    assert_eq!(seed.successors[1], None);
    assert_eq!(newer.successors[1].unwrap().target, key(8));

    let key = boundary(0);
    for _ in 0..3 {
        assert!(samples.boundary(key, true).is_none());
    }
    let snapshot = samples.boundary(key, true).unwrap();
    samples.defer_boundary(snapshot);
    let set = samples.boundary_set(key);
    assert_eq!(samples.boundaries[set][0].unwrap().score, 3);
    let newer = samples.boundary(key, true).unwrap();
    samples.defer_boundary(snapshot);
    assert_eq!(samples.boundaries[set][0].unwrap().score, 4);
    assert_eq!(newer.sequence, snapshot.sequence + 1);
    assert_eq!(snapshot.key, key);
}

#[test]
fn disabled_admission_saturates_without_requests() {
    let mut samples = Samples::new();
    for _ in 0..600 {
        assert!(samples.seed(key(0), version(1), edge(4), false).is_none());
        assert!(samples.boundary(boundary(0), false).is_none());
    }
    let seed = seed_record(&samples, key(0));
    assert_eq!(seed.score, 8);
    assert_eq!(seed.snapshot.successors[0].unwrap().count, 255);
    let set = samples.boundary_set(boundary(0));
    assert_eq!(samples.boundaries[set][0].unwrap().score, 4);
    assert_eq!(samples.seeds.len(), 256);
    assert_eq!(samples.boundaries.len(), 64);
}

#[test]
fn every_replacement_tie_break_is_explicit() {
    let rank = |entry: &(u8, u64)| *entry;
    assert_eq!(victim(&[Some((0, 0)), None, None], rank), 1);
    assert_eq!(
        victim(&[Some((3, 1)), Some((2, 99)), Some((4, 0))], rank),
        1
    );
    assert_eq!(victim(&[Some((2, 9)), Some((2, 8)), Some((2, 9))], rank), 1);
    assert_eq!(victim(&[Some((2, 8)); 4], rank), 0);
}

#[test]
fn seed_collisions_replace_lowest_score_then_oldest() {
    let mut samples = Samples::new();
    let set = samples.seed_set(key(0));
    let keys: Vec<_> = (0..100_000)
        .map(|i| key(i * 4))
        .filter(|key| samples.seed_set(*key) == set)
        .take(6)
        .collect();
    assert_eq!(keys.len(), 6);
    for key in &keys[..4] {
        samples.seed(*key, version(1), None, false);
    }
    samples.seed(keys[0], version(1), None, false);
    samples.seed(keys[4], version(1), None, false);
    assert_eq!(samples.seeds[set][1].unwrap().snapshot.key, keys[4]);
    samples.seed(keys[5], version(1), None, false);
    assert_eq!(samples.seeds[set][2].unwrap().snapshot.key, keys[5]);
    assert_eq!(samples.seeds[set][0].unwrap().snapshot.key, keys[0]);
}

#[test]
fn boundary_collisions_use_same_bounded_replacement() {
    let mut samples = Samples::new();
    let set = samples.boundary_set(boundary(0));
    let keys: Vec<_> = (0..100_000)
        .map(|i| boundary(i * 4))
        .filter(|key| samples.boundary_set(*key) == set)
        .take(4)
        .collect();
    assert_eq!(keys.len(), 4);
    samples.boundary(keys[0], false);
    samples.boundary(keys[1], false);
    samples.boundary(keys[0], false);
    samples.boundary(keys[2], false);
    assert_eq!(samples.boundaries[set][0].unwrap().snapshot.key, keys[0]);
    assert_eq!(samples.boundaries[set][1].unwrap().snapshot.key, keys[2]);
    samples.boundary(keys[2], false);
    samples.boundary(keys[3], false);
    assert_eq!(samples.boundaries[set][0].unwrap().snapshot.key, keys[3]);
}

#[test]
fn successors_replace_lowest_count_then_oldest() {
    let mut samples = Samples::new();
    for pc in [4, 8, 12, 16, 4, 20, 24] {
        samples.seed(key(0), version(1), edge(pc), false);
    }
    let snapshot = samples.seed(key(0), version(1), None, true).unwrap();
    assert_eq!(snapshot.last_edge, None);
    assert_eq!(
        snapshot.successors.map(|s| s.unwrap().target.pc.get()),
        [4, 20, 24, 16]
    );
    assert_eq!(snapshot.successors[0].unwrap().count, 2);
}

#[test]
fn invalid_destination_is_observed_but_never_becomes_a_successor() {
    let mut samples = Samples::new();
    for _ in 0..7 {
        samples.seed(key(0), version(1), None, false);
    }
    let snapshot = samples.seed(key(0), version(1), edge(3), true).unwrap();
    assert_eq!(snapshot.last_edge, edge(3));
    assert_eq!(snapshot.successors, [None; 4]);
}

#[test]
fn full_execution_keys_never_share_heat() {
    let mut samples = Samples::new();
    let base = key(0);
    let other_platform = BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch2, AddressSpaceId::new(1)),
        base.pc,
        base.fp,
    )
    .unwrap();
    let keys = [
        base,
        BlockKey {
            address_space: AddressSpaceId::new(2),
            ..base
        },
        BlockKey {
            fp: FpSpecialization::Exact(0),
            ..base
        },
        other_platform,
    ];
    for key in keys {
        for _ in 0..7 {
            assert!(samples.seed(key, version(1), None, true).is_none());
        }
    }
    for key in keys {
        let snapshot = samples.seed(key, version(1), edge(4), true).unwrap();
        assert_eq!(snapshot.key, key);
        assert_eq!(
            snapshot.successors[0].unwrap().target,
            key.at(GuestVirtualAddress::new(4)).unwrap()
        );
    }
}

#[test]
fn changed_seed_version_resets_heat_and_successors() {
    let mut samples = Samples::new();
    for _ in 0..8 {
        samples.seed(key(0), version(1), edge(4), false);
    }
    assert!(samples.seed(key(0), version(2), None, true).is_none());
    let seed = seed_record(&samples, key(0));
    assert_eq!(seed.score, 1);
    assert_eq!(seed.snapshot.version, version(2));
    assert_eq!(seed.snapshot.successors, [None; 4]);
}

#[test]
fn every_boundary_identity_change_resets_heat() {
    let base = boundary(0);
    let owner = FamilyIdentity {
        id: HcqFamilyId::new(2).unwrap(),
        version: FamilyVersion::new(2).unwrap(),
    };
    for changed in [
        BoundaryKey {
            source_version: version(2),
            ..base
        },
        BoundaryKey {
            target_version: version(2),
            ..base
        },
        BoundaryKey {
            source_family: Some(owner),
            ..base
        },
        BoundaryKey {
            source_family: Some(FamilyIdentity {
                version: FamilyVersion::new(2).unwrap(),
                ..base.source_family.unwrap()
            }),
            ..base
        },
        BoundaryKey {
            source_family: None,
            ..base
        },
        BoundaryKey {
            target_family: Some(owner),
            ..base
        },
    ] {
        let mut samples = Samples::new();
        for _ in 0..4 {
            samples.boundary(base, false);
        }
        assert!(samples.boundary(changed, true).is_none());
        let ways = &samples.boundaries[samples.boundary_set(changed)];
        assert_eq!(ways.iter().flatten().count(), 1);
        assert_eq!(ways[0].unwrap().score, 1);
        assert_eq!(ways[0].unwrap().snapshot.key, changed);
    }
}

#[test]
fn owner_invalidation_clears_matching_seeds_and_both_boundary_endpoints() {
    let mut samples = Samples::new();
    for pc in [0, 4, 8] {
        samples.seed(key(pc), version(1), None, false);
        samples.boundary(boundary(pc), false);
    }
    samples.invalidate(key(4));
    assert_eq!(samples.seeds.iter().flatten().flatten().count(), 2);
    let remaining: Vec<_> = samples.boundaries.iter().flatten().flatten().collect();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].snapshot.key, boundary(8));
    assert!(samples.seed(key(4), version(1), None, true).is_none());
    assert_eq!(seed_record(&samples, key(4)).score, 1);
}

#[test]
fn overflow_clears_both_tables_and_restarts_at_one_on_either_path() {
    for use_seed in [false, true] {
        let mut samples = Samples::new();
        samples.seed(key(0), version(1), edge(4), false);
        samples.boundary(boundary(0), false);
        let seeds_ptr = samples.seeds.as_ptr();
        let boundaries_ptr = samples.boundaries.as_ptr();
        samples.sequence = u64::MAX - 1;
        samples.seed(key(0), version(1), None, false);
        assert_eq!(samples.sequence, u64::MAX);
        if use_seed {
            assert!(samples.seed(key(0), version(1), None, true).is_none());
            assert!(samples.boundaries.iter().flatten().all(Option::is_none));
            assert_eq!(seed_record(&samples, key(0)).score, 1);
            assert_eq!(seed_record(&samples, key(0)).snapshot.successors, [None; 4]);
        } else {
            assert!(samples.boundary(boundary(0), true).is_none());
            assert!(samples.seeds.iter().flatten().all(Option::is_none));
        }
        assert_eq!(samples.sequence, 1);
        assert_eq!(seeds_ptr, samples.seeds.as_ptr());
        assert_eq!(boundaries_ptr, samples.boundaries.as_ptr());
    }
}
