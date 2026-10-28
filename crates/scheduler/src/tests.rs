use super::*;
use crate::{PriorityRange, VirtualClock, VirtualClockMode, VirtualCpuDescriptor};

fn scheduler(cores: u32) -> SchedulerState {
    SchedulerState::new(
        MachineSchedulerProfile::new(
            (0..cores)
                .map(|core| {
                    VirtualCpuDescriptor::new(VirtualCpuId::new(core), 0)
                        .with_rotation_priority(if core == 3 { 63 } else { 59 })
                })
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap()
        .with_preemption(10_000_000, 2)
        .unwrap(),
        VirtualClock::new(VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}

fn add(
    s: &SchedulerState,
    process: u64,
    priority: i32,
    core: Option<u32>,
    affinity: &[u32],
) -> GuestThreadId {
    let thread = s
        .create_thread(ScheduledThreadConfig {
            process: ProcessId::new(process),
            base_priority: priority,
            ideal_vcpu: core.map(VirtualCpuId::new),
            affinity: s
                .profile()
                .core_set(affinity.iter().copied().map(VirtualCpuId::new))
                .unwrap(),
        })
        .unwrap();
    s.make_ready(thread).unwrap();
    thread
}

fn selected(s: &SchedulerState, core: u32) -> Option<GuestThreadId> {
    s.selection(VirtualCpuId::new(core)).unwrap().thread
}

#[test]
fn requested_suspension_waits_for_the_last_kernel_dependency_in_either_order() {
    for pause_first in [false, true] {
        let s = scheduler(1);
        let owner = add(&s, 1, 20, Some(0), &[0]);
        let peer = add(&s, 1, 30, Some(0), &[0]);
        let waiter = add(&s, 2, 10, Some(0), &[0]);
        let caller = s.claim(VirtualCpuId::new(0)).unwrap().unwrap();
        assert_eq!(caller.thread, waiter);
        s.complete(caller, Completion::Waiting).unwrap();
        if pause_first {
            s.set_activity(owner, true).unwrap();
        }
        s.add_waiter(
            ProcessId::new(1),
            LockKey::ThreadProperty(owner),
            owner,
            waiter,
            Some(waiter),
        )
        .unwrap();
        if !pause_first {
            s.set_activity(owner, true).unwrap();
        }
        let info = s.thread(owner).unwrap();
        assert!(info.paused);
        assert_eq!(info.lifecycle, ThreadLifecycle::Ready);
        assert_eq!(info.effective_priority, 10);
        assert_eq!(selected(&s, 0), Some(owner));
        assert!(s.remove_waiter(waiter));
        assert_eq!(
            s.thread(owner).unwrap().lifecycle,
            ThreadLifecycle::Suspended
        );
        assert_eq!(selected(&s, 0), Some(peer));
        s.set_activity(owner, false).unwrap();
        assert_eq!(selected(&s, 0), Some(owner));
    }
}

#[test]
fn kernel_handoff_transfers_suspend_deferral_to_the_new_owner() {
    let s = scheduler(1);
    let owner = add(&s, 1, 20, Some(0), &[0]);
    let peer = add(&s, 1, 40, Some(0), &[0]);
    let next = add(&s, 1, 10, Some(0), &[0]);
    let lease = s.claim(VirtualCpuId::new(0)).unwrap().unwrap();
    s.complete(lease, Completion::Waiting).unwrap();
    let next_wait = s.register_wait(next).unwrap();
    s.set_activity(next, true).unwrap();
    let last = add(&s, 1, 15, Some(0), &[0]);
    let lease = s.claim(VirtualCpuId::new(0)).unwrap().unwrap();
    s.complete(lease, Completion::Waiting).unwrap();
    let key = LockKey::ThreadProperty(owner);
    s.add_waiter(ProcessId::new(1), key, owner, next, None)
        .unwrap();
    s.add_waiter(ProcessId::new(1), key, owner, last, None)
        .unwrap();
    s.set_activity(owner, true).unwrap();
    assert_eq!(selected(&s, 0), Some(owner));
    let handoff = s.release_lock(ProcessId::new(1), key, owner).unwrap();
    assert_eq!(handoff.next, Some(next));
    assert!(handoff.more_waiters);
    assert_eq!(
        s.thread(owner).unwrap().lifecycle,
        ThreadLifecycle::Suspended
    );
    s.finish_wait(next_wait).unwrap();
    assert!(s.thread(next).unwrap().paused);
    assert_eq!(selected(&s, 0), Some(next));
    assert!(s.remove_waiter(last));
    assert_eq!(
        s.thread(next).unwrap().lifecycle,
        ThreadLifecycle::Suspended
    );
    assert_eq!(selected(&s, 0), Some(peer));
}

#[test]
fn user_mutex_waiters_do_not_override_requested_suspension() {
    let s = scheduler(1);
    let owner = add(&s, 1, 20, Some(0), &[0]);
    let peer = add(&s, 1, 30, Some(0), &[0]);
    let waiter = add(&s, 1, 10, Some(0), &[0]);
    let lease = s.claim(VirtualCpuId::new(0)).unwrap().unwrap();
    s.complete(lease, Completion::Waiting).unwrap();
    donate(&s, owner, waiter, 0x1000).unwrap();
    s.set_activity(owner, true).unwrap();
    assert_eq!(
        s.thread(owner).unwrap().lifecycle,
        ThreadLifecycle::Suspended
    );
    assert_eq!(selected(&s, 0), Some(peer));
}

fn lease(s: &SchedulerState, core: u32) -> Lease {
    s.claim(VirtualCpuId::new(core)).unwrap().unwrap()
}

fn donate(
    s: &SchedulerState,
    owner: GuestThreadId,
    waiter: GuestThreadId,
    key: u64,
) -> Result<(), SchedulerError> {
    let process = s.thread(owner).unwrap().process;
    s.add_waiter(process, LockKey::User(key), owner, waiter, None)
}

#[test]
fn priority_donation_requeues_only_the_actual_current_owner_at_the_front() {
    for current_is_owner in [true, false] {
        let s = scheduler(1);
        let owner = add(&s, 1, 44, Some(0), &[0]);
        let caller = lease(&s, 0);
        let peer = add(&s, 1, 30, Some(0), &[0]);
        let waiter = s
            .create_thread(ScheduledThreadConfig {
                process: ProcessId::new(1),
                base_priority: 30,
                ideal_vcpu: Some(VirtualCpuId::new(0)),
                affinity: s.profile().all_cores(),
            })
            .unwrap();
        let current = if current_is_owner { owner } else { peer };
        s.add_waiter(
            ProcessId::new(1),
            LockKey::User(0x1000),
            owner,
            waiter,
            Some(current),
        )
        .unwrap();
        assert_eq!(s.thread(owner).unwrap().effective_priority, 30);
        assert_eq!(selected(&s, 0), Some(current));
        assert!(s.remove_waiter(waiter));
        assert_eq!(s.thread(owner).unwrap().effective_priority, 44);
        assert_eq!(selected(&s, 0), Some(peer));
        s.complete(caller, Completion::Ready).unwrap();
    }
}

#[test]
fn native_boundaries_keep_queue_turn_and_generation() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    let b = add(&s, 1, 20, Some(0), &[0]);
    let generation = s.selection(VirtualCpuId::new(0)).unwrap().generation;
    for outcome in [
        Completion::Ready,
        Completion::KernelCall {
            interrupt_handled: false,
        },
    ] {
        let owner = lease(&s, 0);
        assert_eq!(owner.thread, a);
        s.complete(owner, outcome).unwrap();
        if matches!(outcome, Completion::KernelCall { .. }) {
            s.make_ready(a).unwrap();
        }
        assert_eq!(
            s.selection(VirtualCpuId::new(0)).unwrap().generation,
            generation
        );
    }
    s.yield_thread(a, YieldMigration::None).unwrap();
    assert_eq!(selected(&s, 0), Some(b));
}

#[test]
fn donor_balancing_invalidates_yield_cache_even_when_the_original_selection_returns() {
    let s = scheduler(2);
    let a = add(&s, 1, 10, Some(0), &[0]);
    s.yield_thread(a, YieldMigration::None).unwrap();
    let generation = s.selection(VirtualCpuId::new(0)).unwrap().generation;
    let b = add(&s, 2, 5, Some(0), &[0, 1]);
    assert_eq!((selected(&s, 0), selected(&s, 1)), (Some(a), Some(b)));
    assert_eq!(
        s.selection(VirtualCpuId::new(0)).unwrap().generation,
        generation + 2
    );
    let c = add(&s, 2, 10, Some(0), &[0]);
    s.yield_thread(a, YieldMigration::None).unwrap();
    assert_eq!(selected(&s, 0), Some(c));
}

#[test]
fn svc_exit_retains_ownership_until_its_racing_interrupt_is_handled() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    let owner = lease(&s, 0);
    add(&s, 1, 10, Some(0), &[0]);
    assert!(
        !s.complete(
            owner,
            Completion::KernelCall {
                interrupt_handled: false
            }
        )
        .unwrap()
    );
    assert_eq!(s.lease_for_vcpu(VirtualCpuId::new(0)), Some(owner));
    s.pin_thread(VirtualCpuId::new(0), a).unwrap();
    assert!(
        s.complete(
            owner,
            Completion::KernelCall {
                interrupt_handled: true
            }
        )
        .unwrap()
    );
    assert_eq!(s.lease_for_vcpu(VirtualCpuId::new(0)), None);
    assert_eq!(selected(&s, 0), Some(a));
}

#[test]
fn priority_masks_cover_both_ends_and_cross_priority_iteration() {
    let s = scheduler(1);
    let low = add(&s, 1, 63, Some(0), &[0]);
    let high = add(&s, 1, 0, Some(0), &[0]);
    let middle = add(&s, 1, 31, Some(0), &[0]);
    assert_eq!(selected(&s, 0), Some(high));
    s.terminate(high, false).unwrap();
    assert_eq!(selected(&s, 0), Some(middle));
    s.terminate(middle, false).unwrap();
    assert_eq!(selected(&s, 0), Some(low));
}

#[test]
fn first_readiness_uses_ideal_or_least_populated_allowed_core() {
    let s = scheduler(3);
    let a = add(&s, 1, 20, None, &[0, 1, 2]);
    let b = add(&s, 1, 20, None, &[0, 1, 2]);
    let c = add(&s, 1, 20, Some(2), &[0, 1, 2]);
    assert_eq!(
        (selected(&s, 0), selected(&s, 1), selected(&s, 2)),
        (Some(a), Some(b), Some(c))
    );
}

#[test]
fn idle_core_takes_non_top_before_a_donor_top() {
    let s = scheduler(2);
    let a = add(&s, 1, 20, Some(0), &[0, 1]);
    let b = add(&s, 1, 30, Some(0), &[0]);
    assert_eq!((selected(&s, 0), selected(&s, 1)), (Some(b), Some(a)));
}

#[test]
fn restricted_source_stops_scan_without_donor_fallback() {
    let s = scheduler(2);
    let high = add(&s, 1, 1, Some(0), &[0]);
    add(&s, 1, 5, Some(0), &[0, 1]);
    add(&s, 1, 10, Some(0), &[0, 1]);
    assert_eq!(selected(&s, 0), Some(high));
    assert_eq!(selected(&s, 1), None);
    s.set_priority(high, 2, None).unwrap();
    assert!(selected(&s, 1).is_some());
}

#[test]
fn pinned_override_and_kernel_waiter_precedence() {
    let s = scheduler(1);
    let high = add(&s, 1, 10, Some(0), &[0]);
    let pinned = add(&s, 1, 20, Some(0), &[0]);
    s.pin_thread(VirtualCpuId::new(0), pinned).unwrap();
    assert_eq!(selected(&s, 0), Some(pinned));
    let owner = lease(&s, 0);
    s.complete(owner, Completion::Waiting).unwrap();
    assert_eq!(selected(&s, 0), None);
    let waiter = add(&s, 2, 30, Some(0), &[0]);
    s.add_waiter(
        ProcessId::new(1),
        LockKey::ThreadProperty(high),
        high,
        waiter,
        None,
    )
    .unwrap();
    assert_eq!(selected(&s, 0), Some(high));
    assert!(s.remove_waiter(waiter));
    assert_eq!(selected(&s, 0), None);
    s.unpin_thread(pinned).unwrap();
    assert_eq!(selected(&s, 0), Some(high));
}

#[test]
fn migration_does_not_transfer_live_register_ownership() {
    let s = scheduler(2);
    let a = add(&s, 1, 10, Some(0), &[0, 1]);
    let owner = lease(&s, 0);
    s.migrate(
        a,
        IdealCore::Core(VirtualCpuId::new(1)),
        s.profile().core_set([VirtualCpuId::new(1)]).unwrap(),
    )
    .unwrap();
    assert_eq!(selected(&s, 1), Some(a));
    assert!(s.claim(VirtualCpuId::new(1)).unwrap().is_none());
    assert_eq!(s.lease_for_vcpu(VirtualCpuId::new(0)), Some(owner));
    s.complete(owner, Completion::Ready).unwrap();
    let migrated = lease(&s, 1);
    assert_eq!(migrated.thread, a);
    assert_eq!(
        s.complete(owner, Completion::Ready),
        Err(SchedulerError::StaleLease(owner))
    );
}

#[test]
fn yields_preserve_equal_priority_fifo_and_other_cores_suggestion_order() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    let b = add(&s, 1, 20, Some(0), &[0]);
    for migration in [
        YieldMigration::None,
        YieldMigration::Suggested,
        YieldMigration::Any,
    ] {
        s.yield_thread(a, migration).unwrap();
        assert_eq!(selected(&s, 0), Some(b));
        s.yield_thread(b, YieldMigration::None).unwrap();
        // Yield -2 leaves a in suggestions, so selection may await a donor.
        if migration != YieldMigration::Any {
            assert_eq!(selected(&s, 0), Some(a));
        }
    }
}

#[test]
fn no_effect_yield_caches_only_until_process_scheduling_changes() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    s.yield_thread(a, YieldMigration::None).unwrap();
    let generation = s.state.lock().unwrap().processes[&ProcessId::new(1)].generation;
    for _ in 0..100 {
        s.yield_thread(a, YieldMigration::None).unwrap();
    }
    assert_eq!(
        s.state.lock().unwrap().processes[&ProcessId::new(1)].generation,
        generation
    );
    let b = add(&s, 1, 20, Some(0), &[0]);
    s.yield_thread(a, YieldMigration::None).unwrap();
    assert_eq!(selected(&s, 0), Some(b));
}

#[test]
fn restricted_yield_suggestion_does_not_cache_a_no_effect_result() {
    let s = scheduler(2);
    add(&s, 1, 1, Some(0), &[0]);
    add(&s, 1, 5, Some(0), &[0, 1]);
    let caller = add(&s, 2, 20, Some(1), &[1]);
    s.yield_thread(caller, YieldMigration::Suggested).unwrap();
    let state = s.state.lock().unwrap();
    let index = state.slots.index(caller).unwrap();
    assert_eq!(state.slots[index].yield_generation, None);
}

#[test]
fn rotation_does_not_force_preemption_of_other_priorities() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    add(&s, 1, 20, Some(0), &[0]);
    let running = lease(&s, 0);
    s.take_reschedule_requests();
    s.rotate();
    assert_eq!(selected(&s, 0), Some(a));
    assert!(s.take_reschedule_requests().is_empty());
    s.complete(running, Completion::Ready).unwrap();
    s.set_priority(a, 59, None).unwrap();
    let b = selected(&s, 0).unwrap();
    s.set_priority(b, 59, None).unwrap();
    let before = selected(&s, 0);
    s.rotate();
    assert_ne!(selected(&s, 0), before);
}

#[test]
fn outgoing_selections_sample_the_shared_clock_and_preserve_frozen_ties() {
    for mode in [
        VirtualClockMode::Realtime,
        VirtualClockMode::Fixed { unix_seconds: 0 },
    ] {
        let clock = VirtualClock::new(mode);
        let s = SchedulerState::new(scheduler(2).profile().clone(), clock.clone());
        let local = add(&s, 1, 59, Some(0), &[0]);
        add(&s, 1, 59, Some(0), &[0]);
        let remote = add(&s, 1, 59, Some(1), &[1]);
        add(&s, 1, 59, Some(1), &[1]);
        if matches!(mode, VirtualClockMode::Fixed { .. }) {
            clock.advance_scheduler_to(10_000_000);
        }
        let before = clock.scheduler_time_ns();
        s.rotate();
        let after = clock.scheduler_time_ns();
        let state = s.state.lock().unwrap();
        let a = state.slots[state.slots.index(local).unwrap()].last_scheduled;
        let b = state.slots[state.slots.index(remote).unwrap()].last_scheduled;
        assert!((before..=after).contains(&a));
        assert!((a..=after).contains(&b));
        if matches!(mode, VirtualClockMode::Fixed { .. }) {
            assert_eq!((a, b), (10_000_000, 10_000_000));
        }
    }
}

#[test]
fn rotation_prefers_an_older_local_but_accepts_a_tied_suggestion() {
    for tied in [false, true] {
        let s = scheduler(2);
        let clock = s.state.lock().unwrap().clock.clone();
        let local = add(&s, 1, 59, Some(0), &[0]);
        add(&s, 1, 59, Some(0), &[0]);
        let remote = add(&s, 1, 59, Some(1), &[1]);
        add(&s, 1, 59, Some(1), &[1]);
        clock.advance_scheduler_to(100);
        s.yield_thread(local, YieldMigration::None).unwrap();
        if !tied {
            clock.advance_scheduler_to(200);
        }
        s.yield_thread(remote, YieldMigration::None).unwrap();
        s.migrate(remote, IdealCore::Any, s.profile().all_cores())
            .unwrap();
        s.rotate();
        assert_eq!(selected(&s, 0), Some(if tied { remote } else { local }));
    }
}

#[test]
#[cfg(feature = "performance-counters")]
fn empty_retirement_does_not_reselect_but_ready_retirement_does() {
    for state in [
        ThreadLifecycle::Created,
        ThreadLifecycle::Exited,
        ThreadLifecycle::Ready,
    ] {
        let s = scheduler(1);
        let ready = add(&s, 1, 20, Some(0), &[0]);
        let retiring = s
            .create_thread(ScheduledThreadConfig {
                process: ProcessId::new(2),
                base_priority: 10,
                ideal_vcpu: Some(VirtualCpuId::new(0)),
                affinity: s.profile().all_cores(),
            })
            .unwrap();
        if state != ThreadLifecycle::Created {
            s.make_ready(retiring).unwrap();
        }
        if state == ThreadLifecycle::Exited {
            s.terminate(retiring, false).unwrap();
        }
        s.take_reschedule_requests();
        let before = s.statistics().selection_passes;
        s.unregister(retiring).unwrap();
        assert_eq!(
            s.statistics().selection_passes - before,
            u64::from(state == ThreadLifecycle::Ready)
        );
        assert_eq!(selected(&s, 0), Some(ready));
        assert_eq!(
            s.take_reschedule_requests().is_empty(),
            state != ThreadLifecycle::Ready
        );
    }
}

#[test]
fn donation_propagates_only_affected_chain_and_restores_remaining_donors() {
    let s = scheduler(1);
    let a = add(&s, 1, 40, Some(0), &[0]);
    let b = add(&s, 1, 30, Some(0), &[0]);
    let c = add(&s, 1, 5, Some(0), &[0]);
    let d = add(&s, 1, 10, Some(0), &[0]);
    donate(&s, a, b, 0x100).unwrap();
    donate(&s, b, c, 0x200).unwrap();
    donate(&s, a, d, 0x300).unwrap();
    assert_eq!(s.thread(a).unwrap().effective_priority, 5);
    assert!(s.remove_waiter(c));
    assert_eq!(s.thread(a).unwrap().effective_priority, 10);
    assert_eq!(s.thread(b).unwrap().effective_priority, 30);
    s.set_priority(d, 45, None).unwrap();
    assert_eq!(s.thread(a).unwrap().effective_priority, 30);
    assert!(s.remove_waiter(b));
    assert_eq!(s.thread(a).unwrap().effective_priority, 40);
}

#[test]
fn donation_cycle_is_detected_without_corrupting_existing_links() {
    let s = scheduler(1);
    let a = add(&s, 1, 40, Some(0), &[0]);
    let b = add(&s, 1, 20, Some(0), &[0]);
    donate(&s, a, b, 1).unwrap();
    assert_eq!(donate(&s, b, a, 2), Err(SchedulerError::PriorityCycle(a)));
    assert!(s.remove_waiter(b));
    assert_eq!(s.thread(a).unwrap().effective_priority, 40);
    assert_eq!(s.priority_donation_count(), 0);
}

#[test]
fn wake_generation_and_reused_slot_reject_stale_completions() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    s.complete(
        lease(&s, 0),
        Completion::KernelCall {
            interrupt_handled: false,
        },
    )
    .unwrap();
    let token = s.register_wait(a).unwrap();
    s.finish_wait(token).unwrap();
    assert_eq!(s.finish_wait(token), Err(SchedulerError::StaleWake(token)));
    s.terminate(a, false).unwrap();
    s.unregister(a).unwrap();
    let b = add(&s, 1, 20, Some(0), &[0]);
    assert_ne!(a, b);
    assert!(s.thread(a).is_none());
    assert_eq!(s.finish_wait(token), Err(SchedulerError::StaleWake(token)));
    assert_eq!(selected(&s, 0), Some(b));
}

#[test]
fn paused_waiter_wakes_without_becoming_runnable_until_resumed() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    s.complete(
        lease(&s, 0),
        Completion::KernelCall {
            interrupt_handled: false,
        },
    )
    .unwrap();
    let token = s.register_wait(a).unwrap();
    s.set_activity(a, true).unwrap();
    s.finish_wait(token).unwrap();
    assert_eq!(selected(&s, 0), None);
    assert_eq!(s.thread(a).unwrap().lifecycle, ThreadLifecycle::Suspended);
    s.set_activity(a, false).unwrap();
    assert_eq!(selected(&s, 0), Some(a));
}

#[test]
fn selected_changes_produce_targeted_wake_masks() {
    let s = scheduler(2);
    add(&s, 1, 20, Some(0), &[0]);
    s.take_reschedule_requests();
    add(&s, 1, 10, Some(1), &[1]);
    assert_eq!(
        s.take_reschedule_requests().iter().collect::<Vec<_>>(),
        [VirtualCpuId::new(1)]
    );
    assert!(s.take_reschedule_requests().is_empty());
}

#[test]
fn simultaneous_claims_have_one_register_owner() {
    let s = std::sync::Arc::new(scheduler(1));
    add(&s, 1, 20, Some(0), &[0]);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let s = s.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                s.claim(VirtualCpuId::new(0))
            })
        })
        .collect();
    barrier.wait();
    let owners = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .filter(|result| result.as_ref().is_ok_and(|lease| lease.is_some()))
        .count();
    assert_eq!(owners, 1);
}

#[test]
fn a_restricted_non_top_candidate_does_not_use_previously_collected_donors() {
    let s = scheduler(3);
    let filler = add(&s, 1, 0, Some(2), &[2]);
    let donor = add(&s, 1, 3, Some(1), &[1, 2]);
    add(&s, 1, 40, Some(1), &[1]);
    add(&s, 1, 1, Some(0), &[0]);
    add(&s, 1, 5, Some(0), &[0, 2]);
    s.terminate(filler, false).unwrap();
    assert_eq!(selected(&s, 2), None);
    assert_eq!(selected(&s, 1), Some(donor));
}

#[test]
fn pinning_restricts_physical_membership_and_restores_requested_affinity() {
    let s = scheduler(2);
    let a = add(&s, 1, 20, Some(0), &[0, 1]);
    s.pin_thread(VirtualCpuId::new(0), a).unwrap();
    s.migrate(
        a,
        IdealCore::Core(VirtualCpuId::new(1)),
        s.profile().core_set([VirtualCpuId::new(1)]).unwrap(),
    )
    .unwrap();
    assert_eq!(selected(&s, 0), Some(a));
    assert_eq!(selected(&s, 1), None);
    s.unpin_thread(a).unwrap();
    assert_eq!(selected(&s, 0), None);
    assert_eq!(selected(&s, 1), Some(a));
}

#[test]
fn pending_suspend_applies_after_unpinning_and_terminal_threads_release_pins() {
    let s = scheduler(1);
    let a = add(&s, 1, 20, Some(0), &[0]);
    s.pin_thread(VirtualCpuId::new(0), a).unwrap();
    s.set_activity(a, true).unwrap();
    assert_eq!(selected(&s, 0), Some(a));
    s.unpin_thread(a).unwrap();
    assert_eq!(selected(&s, 0), None);
    s.set_activity(a, false).unwrap();
    s.pin_thread(VirtualCpuId::new(0), a).unwrap();
    let b = add(&s, 1, 10, Some(0), &[0]);
    assert_eq!(selected(&s, 0), Some(a));
    s.terminate(a, false).unwrap();
    assert_eq!(selected(&s, 0), Some(b));
}

#[test]
fn periodic_policy_rotates_all_four_designated_priorities_only() {
    for priority in [20, 59, 63] {
        let s = scheduler(4);
        let mut pairs = Vec::new();
        for core in 0..4 {
            let a = add(&s, 1, priority, Some(core), &[core]);
            let b = add(&s, 1, priority, Some(core), &[core]);
            pairs.push((a, b));
        }
        s.rotate();
        for (core, &(a, b)) in pairs.iter().enumerate() {
            assert_eq!(
                selected(&s, core as u32),
                Some(if priority == if core == 3 { 63 } else { 59 } {
                    b
                } else {
                    a
                })
            );
        }
    }
}

#[test]
fn priority_changes_only_keep_the_calling_thread_at_the_front() {
    let s = scheduler(2);
    let other_running = add(&s, 1, 10, Some(0), &[0]);
    let waiting = add(&s, 1, 20, Some(0), &[0]);
    let caller = add(&s, 1, 10, Some(1), &[1]);
    let same_priority = add(&s, 1, 20, Some(1), &[1]);
    let owner = lease(&s, 0);
    s.set_priority(other_running, 20, Some(caller)).unwrap();
    assert_eq!(selected(&s, 0), Some(waiting));
    s.set_priority(caller, 20, Some(caller)).unwrap();
    assert_eq!(selected(&s, 1), Some(caller));
    s.yield_thread(caller, YieldMigration::None).unwrap();
    assert_eq!(selected(&s, 1), Some(same_priority));
    s.complete(owner, Completion::Ready).unwrap();
}

#[test]
fn competing_wait_completions_claim_one_generation() {
    let s = std::sync::Arc::new(scheduler(1));
    let a = add(&s, 1, 20, Some(0), &[0]);
    s.complete(
        lease(&s, 0),
        Completion::KernelCall {
            interrupt_handled: false,
        },
    )
    .unwrap();
    let token = s.register_wait(a).unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let s = s.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                s.finish_wait(token)
            })
        })
        .collect();
    barrier.wait();
    let completed = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .filter(Result::is_ok)
        .count();
    assert_eq!(completed, 1);
    assert_eq!(selected(&s, 0), Some(a));
}

#[test]
fn affinity_without_ideal_replaces_a_disallowed_active_core_with_the_highest_allowed() {
    let s = scheduler(3);
    let thread = add(&s, 1, 20, Some(0), &[0]);
    s.migrate(
        thread,
        IdealCore::Any,
        s.profile()
            .core_set([VirtualCpuId::new(1), VirtualCpuId::new(2)])
            .unwrap(),
    )
    .unwrap();
    assert_eq!(selected(&s, 2), Some(thread));
}

#[test]
fn selection_loss_invalidates_the_previous_process_yield_cache() {
    let s = scheduler(1);
    let a = add(&s, 1, 10, Some(0), &[0]);
    s.yield_thread(a, YieldMigration::None).unwrap();
    let b = add(&s, 2, 5, Some(0), &[0]);
    let peer = add(&s, 2, 10, Some(0), &[0]);
    s.set_priority(b, 20, None).unwrap();
    assert_eq!(selected(&s, 0), Some(a));
    s.yield_thread(a, YieldMigration::None).unwrap();
    assert_eq!(selected(&s, 0), Some(peer));
}

#[test]
fn ideal_only_changes_and_same_mask_pin_roundtrips_preserve_fifo() {
    let s = scheduler(1);
    let a = add(&s, 1, 10, Some(0), &[0]);
    add(&s, 1, 10, Some(0), &[0]);
    s.migrate(a, IdealCore::Any, s.profile().all_cores())
        .unwrap();
    assert_eq!(selected(&s, 0), Some(a));
    s.pin_thread(VirtualCpuId::new(0), a).unwrap();
    s.unpin_thread(a).unwrap();
    assert_eq!(selected(&s, 0), Some(a));
}

#[test]
fn pinning_overrides_an_existing_suspend_without_resolving_a_real_wait() {
    let s = scheduler(1);
    let a = add(&s, 1, 10, Some(0), &[0]);
    s.set_activity(a, true).unwrap();
    s.pin_thread(VirtualCpuId::new(0), a).unwrap();
    assert_eq!(selected(&s, 0), Some(a));
    s.complete(lease(&s, 0), Completion::Waiting).unwrap();
    assert_eq!(selected(&s, 0), None);
    s.unpin_thread(a).unwrap();
    assert_eq!(selected(&s, 0), None);
}

#[test]
fn periodic_rotation_is_independent_of_transient_worker_ownership() {
    for owned in [false, true] {
        let s = scheduler(2);
        add(&s, 1, 20, Some(0), &[0]);
        let local = add(&s, 1, 60, Some(0), &[0]);
        add(&s, 1, 10, Some(1), &[1]);
        add(&s, 1, 30, Some(1), &[0, 1]);
        let mut owner = owned.then(|| lease(&s, 0));
        s.rotate();
        let owner = owner.take().unwrap_or_else(|| lease(&s, 0));
        s.complete(owner, Completion::Waiting).unwrap();
        assert_eq!(selected(&s, 0), Some(local));
    }
}

#[test]
#[cfg(feature = "performance-counters")]
fn no_effect_yields_and_masked_changes_do_not_select_again() {
    for migration in [
        YieldMigration::None,
        YieldMigration::Suggested,
        YieldMigration::Any,
    ] {
        let s = scheduler(1);
        let a = add(&s, 1, 10, Some(0), &[0]);
        let before = s.statistics().selection_passes;
        s.yield_thread(a, migration).unwrap();
        s.yield_thread(a, migration).unwrap();
        assert_eq!(s.statistics().selection_passes, before);
        assert_eq!(s.statistics().cached_yields, 1);
        s.pin_thread(VirtualCpuId::new(0), a).unwrap();
        let before = s.statistics().selection_passes;
        s.set_activity(a, true).unwrap();
        s.migrate(a, IdealCore::Any, s.profile().all_cores())
            .unwrap();
        assert_eq!(s.statistics().selection_passes, before);
    }
}

#[test]
fn lock_handoff_preserves_remaining_fifo_and_donations() {
    let s = scheduler(1);
    let owner = add(&s, 1, 40, Some(0), &[0]);
    let first = add(&s, 1, 10, Some(0), &[0]);
    let second = add(&s, 1, 10, Some(0), &[0]);
    let address = 0x100;
    let key = LockKey::User(address);
    donate(&s, owner, second, address).unwrap();
    donate(&s, owner, first, address).unwrap();
    let result = s.release_lock(ProcessId::new(1), key, owner).unwrap();
    assert_eq!(
        result,
        LockHandoff {
            next: Some(second),
            more_waiters: true
        }
    );
    assert_eq!(s.thread(owner).unwrap().effective_priority, 40);
    assert_eq!(
        s.release_lock(ProcessId::new(1), key, second).unwrap().next,
        Some(first)
    );
    assert_eq!(s.priority_donation_count(), 0);
}

#[test]
fn priority_inheritance_reorders_the_actual_condition_or_arbiter_node() {
    for kind in [
        OrderedWaitKind::ConditionVariable,
        OrderedWaitKind::AddressArbiter,
    ] {
        let s = scheduler(1);
        let a = add(&s, 1, 40, Some(0), &[0]);
        let b = add(&s, 1, 20, Some(0), &[0]);
        let donor = add(&s, 1, 5, Some(0), &[0]);
        s.register_ordered_wait(a, kind, 0x200).unwrap();
        s.register_ordered_wait(b, kind, 0x200).unwrap();
        assert_eq!(
            s.first_ordered_waiter(ProcessId::new(1), kind, 0x200),
            Some(b)
        );
        donate(&s, a, donor, 0x100).unwrap();
        assert_eq!(
            s.first_ordered_waiter(ProcessId::new(1), kind, 0x200),
            Some(a)
        );
        s.remove_waiter(donor);
        assert_eq!(
            s.first_ordered_waiter(ProcessId::new(1), kind, 0x200),
            Some(b)
        );
        s.terminate(b, false).unwrap();
        s.unregister(b).unwrap();
        assert_eq!(
            s.ordered_waiters(ProcessId::new(1), kind, 0x200, usize::MAX),
            [a]
        );
    }
}

#[path = "tests/property.rs"]
mod property;
