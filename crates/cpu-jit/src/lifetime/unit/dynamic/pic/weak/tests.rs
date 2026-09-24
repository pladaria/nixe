use super::super::tests::{cache, install};
use super::*;
use crate::abi::{GuestValue, RegisterClass, ValueBinding};
use crate::lifetime::Reader;
use crate::lifetime::unit::dynamic::tests::source;
use crate::lifetime::unit::tests::{input, key, process, publish};
use crate::native::pic::WAYS;

#[test]
fn weak_bridge_collision_replacement_rotates_without_releasing_pic_roots() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let handles = [
        install(&mut reader, &process, source, 0, 4),
        install(&mut reader, &process, source, 1, 4),
        install(&mut reader, &process, source, 0, 8),
    ];
    // Exercise the bucket policy with a forced collision among three real,
    // live bridge handles, independent of the process's randomized hasher.
    let mut bucket = Set::default();
    for handle in handles {
        bucket.insert(WeakBridge {
            site: handle.site,
            generation: handle.generation,
        });
    }
    assert_eq!(bucket.ways[0].unwrap().generation, handles[2].generation);
    assert_eq!(bucket.ways[1].unwrap().generation, handles[1].generation);
    assert_eq!(bucket.replace, 1);
    bucket.insert(WeakBridge {
        site: handles[2].site,
        generation: handles[2].generation,
    });
    assert_eq!(bucket.replace, 1); // Same bridge only refreshes its weak anchor.
    let state = process.lock();
    for handle in handles {
        assert!(
            state
                .weak_bridge(WeakBridge {
                    site: handle.site,
                    generation: handle.generation
                })
                .is_some()
        );
    }
}

#[test]
fn weak_bridge_concurrent_preparations_share_the_published_winner() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let first = process.register().unwrap();
    let second = process.register().unwrap();
    let barrier = std::sync::Barrier::new(2);
    let (first, second) = std::thread::scope(|scope| {
        let run = |mut reader: Reader| {
            let prepared = process
                .prepare_dynamic_bridge(source, 0, key(4))
                .unwrap()
                .unwrap();
            barrier.wait();
            let handle = cache(&mut reader, prepared).unwrap();
            (reader, handle)
        };
        let a = scope.spawn(move || run(first));
        let b = scope.spawn(move || run(second));
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(first.1.generation, second.1.generation);
    {
        let state = process.lock();
        let a = state
            .readers
            .get(first.1.site.reader)
            .unwrap()
            .pic
            .way(first.1.site.slot)
            .bridge
            .as_ref()
            .unwrap();
        let b = state
            .readers
            .get(second.1.site.reader)
            .unwrap()
            .pic
            .way(second.1.site.slot)
            .bridge
            .as_ref()
            .unwrap();
        assert!(Arc::ptr_eq(a, b));
        assert_eq!(Arc::strong_count(a), 2);
    }
    drop((first, second));
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn weak_bridge_hit_shares_nonempty_code_without_allocating_or_emitting() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let mut candidate = input(&process, &[4], Tier::Lcq);
    candidate.entries[0].contract.live_in.integer.x.insert(0);
    candidate.entries[0].contract.bindings = std::sync::Arc::from([ValueBinding {
        value: GuestValue::General(0),
        location: ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 0,
        },
    }]);
    let target = process
        .prepare_unit(&[process.reserve(key(4)).unwrap()], candidate, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut first = process.register().unwrap();
    let mut second = process.register().unwrap();
    let a = install(&mut first, &process, source, 0, 4);
    let before = process.cache.usage().unwrap();
    // Even exhausted generation allocation cannot interfere with reuse.
    process.lock().bridge_generations = CheckedCounter::exhausted();
    let b = install(&mut second, &process, source, 0, 4);
    assert_eq!(a.generation, b.generation);
    assert_eq!(process.cache.usage().unwrap(), before);
    let bridge_key = {
        let state = process.lock();
        let a = state
            .readers
            .get(a.site.reader)
            .unwrap()
            .pic
            .way(a.site.slot)
            .bridge
            .as_ref()
            .unwrap();
        let b = state
            .readers
            .get(b.site.reader)
            .unwrap()
            .pic
            .way(b.site.slot)
            .bridge
            .as_ref()
            .unwrap();
        assert!(Arc::ptr_eq(a, b));
        assert_eq!(Arc::strong_count(a), 2);
        assert!(a._code.is_some());
        a.key
    };
    drop(first);
    process.retire_unit(target).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.drain_retirements().unwrap();
    assert!(process.lock().find_weak_bridge(bridge_key).is_none());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn weak_bridge_stale_way_does_not_resurrect_reused_slot_or_retain_metadata() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4, 8196, 16388], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let a = install(&mut reader, &process, source, 0, 4);
    let old_key = process
        .prepare_dynamic_bridge(source, 0, key(4))
        .unwrap()
        .unwrap()
        .key();
    install(&mut reader, &process, source, 0, 8196);
    let before = process.cache.usage().unwrap();
    let c = install(&mut reader, &process, source, 0, 16388);
    assert_eq!(a.site.slot, c.site.slot);
    assert_ne!(a.generation, c.generation);
    assert_eq!(process.cache.usage().unwrap(), before);
    let mut state = process.lock();
    assert!(
        state
            .weak_bridge(WeakBridge {
                site: a.site,
                generation: a.generation
            })
            .is_none()
    );
    assert!(state.find_weak_bridge(old_key).is_none());
    let (shard, set) = state.weak_set(old_key).unwrap();
    assert!(
        state.readers.get(shard).unwrap().pic.weak.sets[set]
            .ways
            .iter()
            .all(|way| way.is_none_or(|weak| weak.generation != a.generation))
    );
}

#[test]
fn weak_bridge_full_key_mismatch_never_shares_a_transfer() {
    let process = process();
    let cursor = AtomicU64::new(0);
    publish(&process, &cursor, &[4], Tier::Lcq);
    let source = source(&process, &cursor, 0, EdgeKind::Indirect);
    let mut reader = process.register().unwrap();
    let a = install(&mut reader, &process, source, 0, 4);
    let base = process
        .prepare_dynamic_bridge(source, 0, key(4))
        .unwrap()
        .unwrap()
        .key();
    for field in 0..5 {
        let mut changed = base;
        match field {
            0 => changed.source.state_map += 1,
            1 => changed.source.source = CodeVersion::new(base.source.source.get() + 100).unwrap(),
            2 => changed.target.fp = crate::abi::FpSpecialization::Exact(0),
            3 => {
                changed.reachability =
                    ReachabilityVersion::new(base.reachability.get() + 100).unwrap()
            }
            _ => {
                changed.target_version = CodeVersion::new(base.target_version.get() + 100).unwrap()
            }
        }
        let mut state = process.lock();
        // Force a bucket collision with a valid but different bridge. Checking
        // only the PC, slot generation or hash would incorrectly accept it.
        let (shard, set) = state.weak_set(changed).unwrap();
        state.readers.get_mut(shard).unwrap().pic.weak.sets[set].ways[0] = Some(WeakBridge {
            site: a.site,
            generation: a.generation,
        });
        assert!(state.find_weak_bridge(changed).is_none());
    }
}

#[test]
fn weak_bridge_shards_follow_active_vcpus_and_release_real_storage() {
    let process = process();
    drop(process.register().unwrap()); // Prime reusable selector/reader storage.
    let before = process.cache.usage().unwrap();
    let mut readers = Vec::new();
    for _ in 0..19 {
        readers.push(process.register().unwrap());
    }
    {
        let state = process.lock();
        assert_eq!(state.weak_shards.len(), 19);
        for (index, handle) in state.weak_shards.iter().enumerate() {
            let pic = &state.readers.get(*handle).unwrap().pic;
            assert_eq!(pic.shard_index, index);
            assert_eq!(pic.weak.sets.len() * 2, WAYS);
        }
    }
    drop(readers.swap_remove(3));
    {
        let state = process.lock();
        assert_eq!(state.weak_shards.len(), 18);
        for (index, handle) in state.weak_shards.iter().enumerate() {
            assert_eq!(state.readers.get(*handle).unwrap().pic.shard_index, index);
        }
    }
    drop(readers);
    let state = process.lock();
    assert!(state.weak_shards.is_empty());
    assert!(state.readers.is_empty());
    // Only the grown reusable registry/selector capacity remains, not 19
    // orphaned weak tables. Both allocations are explicitly budgeted.
    assert!(process.cache.usage().unwrap().metadata - before.metadata < 32 * 1024);
}
