use nixe_memory::{
    CanonicalBackingPage, CanonicalBackingStore, ContentGeneration, DirectArena, DirectMapRequest,
    DirectProtection, ExecutionGate, GuestPhysicalPageId,
};
use std::sync::{atomic::AtomicUsize, atomic::Ordering, mpsc};
use std::time::Duration;

#[test]
fn first_exclusive_activation_drains_every_native_alias_arena() {
    let first_gate = ExecutionGate::new();
    let second_gate = ExecutionGate::new();
    let store = CanonicalBackingStore::allocate_with_execution_gate(first_gate.clone()).unwrap();
    let page = CanonicalBackingPage::zeroed(
        &store,
        GuestPhysicalPageId::new(1),
        4096,
        ContentGeneration::INITIAL,
    )
    .unwrap();
    let backing = page.direct_backing().unwrap();
    let first = DirectArena::with_execution_gate(8192, first_gate).unwrap();
    let second = DirectArena::with_execution_gate(8192, second_gate.clone()).unwrap();
    for arena in [&first, &second] {
        arena
            .map_pages(&[DirectMapRequest {
                guest_address: 4096,
                backing: &backing,
                protection: DirectProtection::ReadWrite,
            }])
            .unwrap();
        page.register_direct_alias(arena, 4096, DirectProtection::ReadWrite)
            .unwrap();
    }
    let admitted = second_gate.acquire_shared();
    let (started, start) = mpsc::channel();
    let (finished, finish) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        started.send(()).unwrap();
        page.enable_exclusive_access().unwrap();
        finished.send(()).unwrap();
        page
    });
    start.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(
        finish.recv_timeout(Duration::from_millis(30)).is_err(),
        "an admitted writer in the second arena must drain before activation"
    );
    drop(admitted);
    finish.recv_timeout(Duration::from_secs(2)).unwrap();
    let _page = worker.join().unwrap();
    let entry = |arena: &DirectArena| {
        unsafe { &*((arena.view().exclusive_page_table as *const AtomicUsize).add(1)) }
            .load(Ordering::Acquire)
    };
    assert!(entry(&first) > 1);
    assert_eq!(
        entry(&first),
        entry(&second),
        "aliases must share one physical monitor"
    );
}
