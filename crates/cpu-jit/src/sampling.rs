//! Bounded, exclusively vCPU-owned observations. Callers validate current
//! lifetime identities before recording; no lookup, allocation or locking here.

use crate::abi::{BlockKey, FamilyVersion, HcqFamilyId, InstructionKey, ReachabilityVersion};
use crate::lifetime::unit::EdgeKind;
use nixe_memory::GuestVirtualAddress;
use std::hash::{BuildHasher, RandomState};

const SEED_SETS: usize = 256;
const BOUNDARY_SETS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ObservedEdge {
    pub destination: GuestVirtualAddress,
    pub kind: EdgeKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Successor {
    pub target: BlockKey,
    pub count: u8,
    pub sequence: u64,
}

/// Owned queue input, never a pointer into the sampling table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionSnapshot {
    pub key: BlockKey,
    pub version: ReachabilityVersion,
    pub sequence: u64,
    pub last_edge: Option<ObservedEdge>,
    pub successors: [Option<Successor>; 4],
}

#[derive(Clone, Copy)]
struct Seed {
    snapshot: AdmissionSnapshot,
    score: u8,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct FamilyIdentity {
    pub id: HcqFamilyId,
    pub version: FamilyVersion,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BoundaryKey {
    pub source: InstructionKey,
    pub target: InstructionKey,
    pub source_version: ReachabilityVersion,
    pub target_version: ReachabilityVersion,
    // Both may be absent: reshape permits zero, one or two current families.
    pub source_family: Option<FamilyIdentity>,
    pub target_family: Option<FamilyIdentity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReshapeSnapshot {
    pub key: BoundaryKey,
    pub sequence: u64,
}

#[derive(Clone, Copy)]
struct Boundary {
    snapshot: ReshapeSnapshot,
    score: u8,
}

pub(crate) struct Samples {
    seeds: Box<[[Option<Seed>; 4]]>,
    boundaries: Box<[[Option<Boundary>; 2]]>,
    hash: RandomState,
    sequence: u64,
}

impl Samples {
    #[cfg(test)]
    pub(crate) fn same_boundary_set(&self, a: BoundaryKey, b: BoundaryKey) -> bool {
        self.boundary_set(a) == self.boundary_set(b)
    }

    #[cfg(test)]
    pub(crate) fn boundary_snapshot(
        &self,
        source: InstructionKey,
        target: InstructionKey,
    ) -> Option<(ReshapeSnapshot, u8)> {
        self.boundaries
            .iter()
            .flatten()
            .flatten()
            .find(|entry| {
                entry.snapshot.key.source == source && entry.snapshot.key.target == target
            })
            .map(|entry| (entry.snapshot, entry.score))
    }
    #[cfg(test)]
    pub(crate) fn seed_snapshot(&self, key: BlockKey) -> Option<(AdmissionSnapshot, u8)> {
        self.seeds[self.seed_set(key)]
            .iter()
            .flatten()
            .find(|seed| seed.snapshot.key == key)
            .map(|seed| (seed.snapshot, seed.score))
    }

    pub fn new() -> Self {
        Self {
            seeds: vec![[None; 4]; SEED_SETS].into_boxed_slice(),
            boundaries: vec![[None; 2]; BOUNDARY_SETS].into_boxed_slice(),
            hash: RandomState::new(),
            sequence: 0,
        }
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence = match self.sequence.checked_add(1) {
            Some(sequence) => sequence,
            None => {
                self.seeds.fill([None; 4]);
                self.boundaries.fill([None; 2]);
                1
            }
        };
        self.sequence
    }

    fn seed_set(&self, key: BlockKey) -> usize {
        self.hash.hash_one(key) as usize & (SEED_SETS - 1)
    }

    fn boundary_set(&self, key: BoundaryKey) -> usize {
        // Versions do not affect placement: an endpoint/owner version change
        // replaces old heat for these same full execution-context endpoints.
        self.hash.hash_one((key.source, key.target)) as usize & (BOUNDARY_SETS - 1)
    }

    /// `admission_enabled` stays false with zero workers and until the real
    /// consumer is connected. Heat can saturate without creating requests.
    pub fn seed(
        &mut self,
        key: BlockKey,
        version: ReachabilityVersion,
        edge: Option<ObservedEdge>,
        admission_enabled: bool,
    ) -> Option<AdmissionSnapshot> {
        let sequence = self.next_sequence();
        let set = self.seed_set(key);
        let ways = &mut self.seeds[set];
        let way = ways
            .iter()
            .position(|entry| entry.is_some_and(|seed| seed.snapshot.key == key))
            .unwrap_or_else(|| victim(ways, |seed| (seed.score, seed.snapshot.sequence)));
        let seed = ways[way].get_or_insert_with(|| Seed::new(key, version));
        if seed.snapshot.key != key || seed.snapshot.version != version {
            *seed = Seed::new(key, version);
        }
        seed.score = (seed.score + 1).min(8);
        seed.snapshot.sequence = sequence;
        seed.snapshot.last_edge = edge;
        if let Some(target) = edge.and_then(|edge| key.at(edge.destination)) {
            let slots = &mut seed.snapshot.successors;
            let slot = slots
                .iter()
                .position(|entry| entry.is_some_and(|entry| entry.target == target))
                .unwrap_or_else(|| victim(slots, |entry| (entry.count, entry.sequence)));
            let count = slots[slot]
                .filter(|entry| entry.target == target)
                .map_or(1, |entry| entry.count.saturating_add(1));
            slots[slot] = Some(Successor {
                target,
                count,
                sequence,
            });
        }
        (admission_enabled && seed.score == 8).then_some(seed.snapshot)
    }

    pub fn boundary(
        &mut self,
        key: BoundaryKey,
        admission_enabled: bool,
    ) -> Option<ReshapeSnapshot> {
        let sequence = self.next_sequence();
        let set = self.boundary_set(key);
        let ways = &mut self.boundaries[set];
        let way = ways
            .iter()
            .position(|entry| {
                entry.is_some_and(|entry| {
                    entry.snapshot.key.source == key.source
                        && entry.snapshot.key.target == key.target
                })
            })
            .unwrap_or_else(|| victim(ways, |entry| (entry.score, entry.snapshot.sequence)));
        let entry = ways[way].get_or_insert(Boundary {
            snapshot: ReshapeSnapshot { key, sequence },
            score: 0,
        });
        if entry.snapshot.key != key {
            entry.score = 0;
        }
        entry.snapshot = ReshapeSnapshot { key, sequence };
        entry.score = (entry.score + 1).min(4);
        (admission_enabled && entry.score == 4).then_some(entry.snapshot)
    }

    pub fn defer_seed(&mut self, snapshot: AdmissionSnapshot) {
        let set = self.seed_set(snapshot.key);
        for seed in self.seeds[set].iter_mut().flatten() {
            if seed.snapshot == snapshot {
                seed.score = 7;
                break;
            }
        }
    }

    pub fn defer_boundary(&mut self, snapshot: ReshapeSnapshot) {
        let set = self.boundary_set(snapshot.key);
        for boundary in self.boundaries[set].iter_mut().flatten() {
            if boundary.snapshot == snapshot {
                boundary.score = 3;
                break;
            }
        }
    }
}

impl Seed {
    fn new(key: BlockKey, version: ReachabilityVersion) -> Self {
        Self {
            snapshot: AdmissionSnapshot {
                key,
                version,
                sequence: 0,
                last_edge: None,
                successors: [None; 4],
            },
            score: 0,
        }
    }
}

fn victim<T>(slots: &[Option<T>], rank: impl Fn(&T) -> (u8, u64)) -> usize {
    slots
        .iter()
        .enumerate()
        .min_by_key(|(index, slot)| match slot {
            None => (false, 0, 0, *index),
            Some(entry) => {
                let (score, sequence) = rank(entry);
                (true, score, sequence, *index)
            }
        })
        .expect("sampling sets have a fixed nonzero width")
        .0
}

#[cfg(test)]
mod tests;
