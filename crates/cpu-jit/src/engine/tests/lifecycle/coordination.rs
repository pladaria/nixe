use super::*;
use crate::lifetime::{Error, Reason};
use nixe_memory::{ExecutionMutationObserver, MemoryInvalidationKind};

#[test]
fn unaffected_chains_cannot_bypass_memory_holds_for_linking_or_shutdown() {
    for shutdown in [false, true] {
        let mut first = fixture();
        let process = first.process.clone();
        let mut units = vec![publish(&mut first, TARGET)];
        for pc in [0x1004, 0x1014, 0x1000, 0x1010] {
            units.push(publish(&mut first, GuestVirtualAddress::new(pc)));
        }
        assert!(process.lifetime.try_service_links().unwrap());
        let mut second = JitThread::new(process.clone()).unwrap();
        for thread in [&mut first, &mut second] {
            for offset in [0, 16] {
                warm(thread, offset, 1);
            }
        }
        // This memory-authority interval affects no compiled instruction. It
        // still owns the process stop; a cache with nothing to unlink is not
        // permission for either execution or optional linking to reopen it.
        let hold = process
            .lifetime
            .clone()
            .begin(&[MemoryInvalidationKind::Mapping {
                address_space: SPACE,
                start: GuestVirtualAddress::new(0x5000),
                size: 4096,
            }])
            .unwrap();
        let committed = process
            .lifetime
            .executable_cache()
            .usage()
            .unwrap()
            .committed;
        process.lifetime.request(Reason::LinkPatch).unwrap();
        assert!(!process.lifetime.try_service_links().unwrap());
        let mut transition = process.lifetime.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        assert!(transition.drain_links().unwrap());
        assert_eq!(
            transition.batch().unwrap().complete_with_links_deferred(),
            Err(Error::MaintenancePending)
        );
        assert!(!transition.try_reopen().unwrap());
        drop(transition);
        for thread in [&mut first, &mut second] {
            let mut state = initial(0);
            let before = state.clone();
            assert!(matches!(
                thread.invoke(
                    &mut ReturnStack::default(),
                    &mut NativeWorker::default(),
                    &mut state,
                    PollBudget::new(4096, 64).unwrap(),
                    &VcpuEventState::default()
                ),
                Err(invocation::Error::Lifetime(Error::Closed))
            ));
            assert_eq!(state, before);
        }
        assert_eq!(process.lifetime.reclaim_units().unwrap(), 0);
        assert_eq!(
            process
                .lifetime
                .executable_cache()
                .usage()
                .unwrap()
                .committed,
            committed
        );
        if shutdown {
            assert!(!process.try_shutdown().unwrap());
            assert_eq!(
                process
                    .lifetime
                    .executable_cache()
                    .usage()
                    .unwrap()
                    .committed,
                committed
            );
        }
        drop(hold);
        if !shutdown {
            assert!(process.lifetime.try_service_links().unwrap());
            for unit in units {
                assert!(process.lifetime.snapshot(unit).is_ok());
            }
            // No unnecessary flush: both original PICs and static patches
            // remain usable without installing fresh bridges via the resolver.
            for thread in [&mut first, &mut second] {
                for offset in [0, 16] {
                    let mut state = initial(offset);
                    let (returned, budget) = fallback::without_resolver(
                        &mut ReturnStack::default(),
                        thread,
                        &mut state,
                        PollBudget::new(1, 64).unwrap(),
                        &VcpuEventState::default(),
                    );
                    assert_eq!(returned.reason, NativeExitReason::Architectural);
                    assert_eq!(budget.slice_remaining, 61);
                    assert_eq!(state.general_register_storage_mut()[0], 1);
                }
            }
        }
        assert!(process.try_shutdown().unwrap());
        assert_eq!(
            process
                .lifetime
                .executable_cache()
                .usage()
                .unwrap()
                .committed,
            0
        );
    }
}
