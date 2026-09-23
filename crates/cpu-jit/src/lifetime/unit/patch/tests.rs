use super::*;
use crate::lifetime::Phase;
use crate::lifetime::unit::tests::{frame, input_with_islands, key, process, publish};
use nixe_cpu::state::a64::A64State;

fn replacement(value: u8) -> (usize, Vec<u8>) {
    if cfg!(target_arch = "x86_64") {
        (5, vec![value]) // Only MOV EAX's immediate; retain all instruction PCs.
    } else {
        (
            4,
            (0x52800000 | (u32::from(value) << 5))
                .to_le_bytes()
                .to_vec(),
        )
    }
}

fn execute(process: &Arc<Lifetime>) -> u32 {
    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    let call: unsafe extern "C" fn() -> u32 = unsafe { std::mem::transmute(address) };
    // The fixture is a System-ABI leaf. The real process epoch protects its
    // published code for the entire call, including return to the host.
    unsafe { call() }
}

#[test]
fn published_code_changes_only_after_closed_and_runs_after_reopening() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let input = input_with_islands(&process, &[0], Tier::Lcq, 1);
    let handle = process
        .prepare_unit(&[process.reserve(key(0)).unwrap()], input, &cursor)
        .unwrap()
        .publish()
        .unwrap();
    let retained = process.snapshot(handle).unwrap();
    assert_eq!(execute(&process), 42);

    let mut reader = process.register().unwrap();
    let mut cpu = A64State::default();
    let mut frame = frame(&mut cpu);
    let invocation = unsafe { reader.admit(&mut frame, key(0)) }
        .unwrap()
        .unwrap();
    let ticket = process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    let (offset, bytes) = replacement(77);
    let writes = [Write::Code {
        offset,
        bytes: &bytes,
    }];
    assert_eq!(
        unsafe { transition.patch_unit(handle, &writes) },
        Err(Error::Closed)
    );
    assert_eq!(process.lock().phase, Phase::Closing);
    assert!(
        !process
            .maintenance_complete(Reason::LinkPatch, ticket)
            .unwrap()
    );
    drop(invocation);
    transition.wait_closed().unwrap();

    // Initialize the reserved slot through the same protected write operation.
    // This low-level fixture does not make it callable; production registration
    // and root/backlink ownership are exercised by the linked lifecycle tests.
    let pc = retained.code.allocation.address() as u64;
    let island = crate::native::link::emit(
        retained.code.metadata.abi,
        pc + (1 << 32),
        pc,
        pc + (1 << 32) + 16,
    )
    .unwrap()
    .island
    .unwrap();
    unsafe {
        transition
            .patch_unit(
                handle,
                &[
                    Write::Island {
                        index: 0,
                        bytes: &island,
                    },
                    Write::Code {
                        offset,
                        bytes: &bytes,
                    },
                ],
            )
            .unwrap();
    }
    assert_eq!(process.lock().phase, Phase::Closed);
    let stored = unsafe {
        std::slice::from_raw_parts(
            retained.code.allocation.island_address(0).unwrap() as *const u8,
            16,
        )
    };
    assert_eq!(stored, island);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert_eq!(execute(&process), 77);
    assert_eq!(
        unsafe { transition.patch_unit(handle, &writes) },
        Err(Error::Closed)
    );

    // Rewrite a previously executed address, synchronizing before another
    // thread can fetch it. No extra code owner or append-only patch buffer.
    process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let (offset, bytes) = replacement(42);
    unsafe {
        transition
            .patch_unit(
                handle,
                &[Write::Code {
                    offset,
                    bytes: &bytes,
                }],
            )
            .unwrap();
    }
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    std::thread::scope(|scope| {
        assert_eq!(scope.spawn(|| execute(&process)).join().unwrap(), 42);
    });
}

#[test]
fn closed_patch_rejects_foreign_and_unlinked_handles() {
    let process = process();
    // Even sharing the same cache does not confer another process's authority.
    let other = Lifetime::new(Arc::clone(&process.cache)).unwrap();
    let cursor = AtomicU64::new(0);
    let own = publish(&process, &cursor, &[0], Tier::Lcq);
    let foreign = publish(&other, &cursor, &[0], Tier::Lcq);
    let retained = process.snapshot(own).unwrap();
    process.retire_unit(own).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    assert_eq!(
        unsafe { transition.patch_unit(foreign, &[]) },
        Err(Error::StaleUnit)
    );
    transition.drain_retirements().unwrap();
    assert_eq!(
        unsafe { transition.patch_unit(own, &[]) },
        Err(Error::StaleUnit)
    );
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(retained);
}

#[test]
fn partial_patch_error_permanently_blocks_reopening_and_acknowledgement() {
    for invalid_island in [false, true] {
        let process = process();
        let cursor = AtomicU64::new(0);
        let handle = publish(&process, &cursor, &[0], Tier::Lcq);
        let retained = process.snapshot(handle).unwrap();
        process.request(Reason::LinkPatch).unwrap();
        let mut transition = process.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        let (offset, bytes) = replacement(77);
        let invalid = if invalid_island {
            Write::Island {
                index: 0,
                bytes: &[0; 16],
            }
        } else {
            Write::Code {
                offset: usize::MAX,
                bytes: &[0],
            }
        };
        assert_eq!(
            unsafe {
                transition.patch_unit(
                    handle,
                    &[
                        Write::Code {
                            offset,
                            bytes: &bytes,
                        },
                        invalid,
                    ],
                )
            },
            Err(Error::CacheFailed)
        );
        // The first write really happened. Never execute this partially
        // updated unit; retained RX is only read here to check the failure path.
        assert_eq!(
            unsafe {
                std::slice::from_raw_parts(
                    (retained.code.allocation.address() + offset) as *const u8,
                    bytes.len(),
                )
            },
            bytes
        );
        assert!(matches!(
            process.cache.usage(),
            Err(crate::executable::Error::Poisoned)
        ));
        assert!(matches!(transition.batch(), Err(Error::CacheFailed)));
        assert_eq!(transition.try_reopen(), Err(Error::CacheFailed));
        assert_eq!(process.lock().phase, Phase::Closed);
        drop(transition);
        assert!(matches!(process.try_transition(), Err(Error::CacheFailed)));
        assert!(matches!(process.register(), Err(Error::CacheFailed)));
    }
}

#[test]
fn unwinding_a_write_permit_cannot_leave_a_reopenable_transition() {
    let process = process();
    let cursor = AtomicU64::new(0);
    let handle = publish(&process, &cursor, &[0], Tier::Lcq);
    let retained = process.snapshot(handle).unwrap();
    process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _permit = ClosedCode {
            allocation: &retained.code.allocation,
            transition: &mut transition,
            finished: false,
        };
        panic!("abandon registered code patch");
    }));
    assert!(result.is_err());
    assert_eq!(transition.try_reopen(), Err(Error::CacheFailed));
    assert!(matches!(transition.batch(), Err(Error::CacheFailed)));
}
