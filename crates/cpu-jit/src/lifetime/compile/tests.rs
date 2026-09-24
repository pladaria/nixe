use super::*;
use crate::abi::FpSpecialization;
use crate::executable::Cache;
use crate::lifetime::Reason;
use nixe_cpu::{platform::TargetPlatform, profile::ProcessCpuContext};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

fn process() -> Arc<Lifetime> {
    Arc::new(Lifetime::new(Cache::new().unwrap()).unwrap())
}
fn key(pc: u64) -> BlockKey {
    BlockKey::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, AddressSpaceId::new(1)),
        GuestVirtualAddress::new(pc),
        FpSpecialization::Dynamic,
    )
    .unwrap()
}
fn owner(request: Request<'_>) -> Claim<'_> {
    match request {
        Request::Owner(claim) => claim,
        _ => panic!("expected owner"),
    }
}
fn reopen(process: &Lifetime) {
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
}

#[test]
fn terminal_close_wakes_waiters_but_drains_the_cancelled_compile_owner() {
    let process = process();
    let mut reader = process.register().unwrap();
    let claim = owner(reader.claim(key(0)).unwrap());
    let (ready, started) = mpsc::channel();
    let waiting_process = process.clone();
    let waiter = std::thread::spawn(move || {
        let Request::Wait(waiter) = waiting_process.claim(key(0)).unwrap() else {
            panic!("expected existing owner");
        };
        ready.send(()).unwrap();
        waiter.wait()
    });
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    process.request_shutdown().unwrap();
    process.request_shutdown().unwrap();
    assert_eq!(waiter.join().unwrap(), Err(Error::Shutdown));
    assert_eq!(claim.validate(), Err(Error::Shutdown));
    assert!(!process.try_shutdown().unwrap());
    assert_eq!(process.lock().compilers, 1);
    drop(claim);
    assert!(process.try_shutdown().unwrap());
    assert!(process.try_shutdown().unwrap());
    assert!(matches!(process.register(), Err(Error::Shutdown)));
    assert_eq!(process.lock().dispatch.capacity(), 0);
    assert_eq!(process.lock().readers.capacity(), 0);
}

#[test]
fn shutdown_counts_old_claims_even_after_pressure_reuses_their_dispatch_slots() {
    let process = process();
    let mut first_reader = process.register().unwrap();
    let mut second_reader = process.register().unwrap();
    let old = owner(first_reader.claim(key(0)).unwrap());
    process.request(Reason::Eviction).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    transition.relieve_pressure(0, Tier::Lcq).unwrap();
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    let new = owner(second_reader.claim(key(0)).unwrap());
    assert_ne!(old.publication.slot, new.publication.slot);
    assert_eq!(process.lock().compilers, 2);
    assert!(!process.try_shutdown().unwrap());
    drop(new);
    assert!(!process.try_shutdown().unwrap());
    drop(old);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.lock().compilers, 0);
}

#[test]
fn one_owner_per_key_and_abandonment_reuses_actual_slots() {
    let process = process();
    let mut reader = process.register().unwrap();
    for _ in 0..1000 {
        let claim = owner(reader.claim(key(0)).unwrap());
        assert!(matches!(process.claim(key(0)).unwrap(), Request::Wait(_)));
        drop(claim);
        assert!(process.lock().keys.is_empty());
        assert_eq!(process.lock().dispatch.values().count(), 0);
    }
    assert_eq!(process.lock().dispatch.capacity(), 16);
}

#[test]
fn same_key_waits_but_different_keys_have_simultaneous_owners() {
    let process = process();
    let barrier = Arc::new(Barrier::new(4));
    std::thread::scope(|scope| {
        for pc in [0, 4, 8, 12] {
            let process = &process;
            let barrier = &barrier;
            scope.spawn(move || {
                let mut reader = process.register().unwrap();
                let claim = owner(reader.claim(key(pc)).unwrap());
                let identity = claim.begin_unit().unwrap();
                barrier.wait(); // No compiler is holding the process mutex.
                assert!(identity.version().get() > 0);
                assert!(matches!(process.claim(key(pc)).unwrap(), Request::Wait(_)));
                barrier.wait();
            });
        }
    });
    assert!(process.lock().keys.is_empty());
}

#[test]
fn cancellation_wakes_waiters_and_stale_drop_cannot_release_new_owner() {
    let process = process();
    let old = owner(process.claim(key(0)).unwrap());
    let (started, ready) = mpsc::channel();
    let (finished, result) = mpsc::channel();
    std::thread::scope(|scope| {
        let process = &process;
        scope.spawn(move || {
            let mut reader = process.register().unwrap();
            let Request::Wait(wait) = reader.claim(key(0)).unwrap() else {
                panic!()
            };
            started.send(()).unwrap();
            finished.send(wait.wait()).unwrap();
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        process.request(Reason::MappingChange).unwrap();
        assert_eq!(
            result.recv_timeout(Duration::from_secs(5)).unwrap(),
            Err(Error::Closed)
        );
    });
    reopen(&process);
    let fresh = owner(process.claim(key(0)).unwrap());
    assert_eq!(old.validate(), Err(Error::StalePublication));
    drop(old);
    fresh.validate().unwrap();
    assert!(matches!(process.claim(key(0)).unwrap(), Request::Wait(_)));
    drop(fresh);
    assert!(process.lock().keys.is_empty());
}

#[test]
fn capture_reacquires_a_new_claim_without_reviving_old_publication_tokens() {
    let process = process();
    let old = owner(process.claim(key(0)).unwrap());
    let publication = old.publication;
    let identity = old.identity;
    process.request(Reason::MappingChange).unwrap();
    reopen(&process);
    let fresh = old.after_capture().unwrap();
    assert!(fresh.identity != identity);
    fresh.validate().unwrap();
    assert_eq!(
        process.lock().validate(&publication),
        Err(Error::StalePublication)
    );
    assert!(matches!(process.claim(key(0)).unwrap(), Request::Wait(_)));
    drop(fresh);
    assert!(process.lock().keys.is_empty());
}

#[test]
fn capture_reacquisition_cannot_replace_a_current_owner_or_cross_closed_admission() {
    let process = process();
    let old = owner(process.claim(key(0)).unwrap());
    process.request(Reason::MappingChange).unwrap();
    assert!(matches!(old.after_capture(), Err(Error::Closed)));
    reopen(&process);
    let old = owner(process.claim(key(0)).unwrap());
    process.request(Reason::MappingChange).unwrap();
    reopen(&process);
    let winner = owner(process.claim(key(0)).unwrap());
    assert!(matches!(old.after_capture(), Err(Error::StalePublication)));
    winner.validate().unwrap();
    assert!(matches!(process.claim(key(0)).unwrap(), Request::Wait(_)));
    drop(winner);
    assert!(process.lock().keys.is_empty());
}

#[test]
fn owner_drop_wakes_waiter_without_following_replacement_claim() {
    let process = process();
    let old = owner(process.claim(key(0)).unwrap());
    let Request::Wait(wait) = process.claim(key(0)).unwrap() else {
        panic!()
    };
    drop(old);
    let fresh = owner(process.claim(key(0)).unwrap());
    wait.wait().unwrap();
    fresh.validate().unwrap();
}

#[test]
fn concurrent_same_key_has_exactly_one_winner() {
    let process = process();
    let barrier = Barrier::new(8);
    let winners = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let process = &process;
            let barrier = &barrier;
            let winners = &winners;
            scope.spawn(move || {
                let mut reader = process.register().unwrap();
                barrier.wait();
                let request = reader.claim(key(0)).unwrap();
                if matches!(request, Request::Owner(_)) {
                    winners.fetch_add(1, Ordering::Relaxed);
                }
                barrier.wait(); // Nobody releases the winner before all claims.
                match request {
                    Request::Wait(wait) => wait.wait().unwrap(),
                    Request::Owner(claim) => drop(claim),
                    Request::Ready => panic!("no native unit was published"),
                }
            });
        }
    });
    assert_eq!(winners.load(Ordering::Relaxed), 1);
    assert!(process.lock().keys.is_empty());
}

#[test]
fn publication_wakes_waiters_and_ready_requires_new_protected_lookup() {
    use crate::abi::{CodeUnitId, CodeVersion, PublishedEntry};
    use std::num::NonZeroUsize;
    let process = process();
    let claim = owner(process.claim(key(0)).unwrap());
    let Request::Wait(wait) = process.claim(key(0)).unwrap() else {
        panic!()
    };
    process
        .publish(
            claim.publication().unwrap(),
            Some(PublishedEntry {
                unit: CodeUnitId::new(1).unwrap(),
                version: CodeVersion::new(1).unwrap(),
                canonical: NonZeroUsize::new(16).unwrap(),
                fast: NonZeroUsize::new(32).unwrap(),
            }),
            None,
        )
        .unwrap();
    wait.wait().unwrap();
    drop(claim);
    assert!(matches!(process.claim(key(0)).unwrap(), Request::Ready));
}

#[test]
fn shutdown_and_failure_cancel_claims_and_waiters() {
    for shutdown in [false, true] {
        let process = process();
        let claim = owner(process.claim(key(0)).unwrap());
        let Request::Wait(wait) = process.claim(key(0)).unwrap() else {
            panic!()
        };
        if shutdown {
            process.request(Reason::Shutdown).unwrap();
        } else {
            process.fail(&mut process.lock(), Error::CacheFailed);
        }
        let expected = if shutdown {
            Error::Shutdown
        } else {
            Error::CacheFailed
        };
        assert_eq!(wait.wait(), Err(expected));
        assert_eq!(claim.validate(), Err(expected));
        drop(claim);
        assert!(process.lock().keys.is_empty());
    }
}

#[test]
fn closure_between_slot_reservation_and_claim_leaves_no_empty_slot() {
    let process = process();
    let publication = process.reserve(key(0)).unwrap();
    process.request(Reason::MappingChange).unwrap();
    assert!(matches!(
        process.claim_reserved(publication),
        Err(Error::Closed)
    ));
    assert!(process.lock().keys.is_empty());
    assert_eq!(process.lock().dispatch.values().count(), 0);
    reopen(&process);
    let fresh = owner(process.claim(key(0)).unwrap());
    // A late stale token cannot discard the current claim.
    assert!(matches!(
        process.claim_reserved(publication),
        Err(Error::StalePublication)
    ));
    fresh.validate().unwrap();
}
