use super::*;
use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuExit, CpuFaultKind, TimerSnapshot, VcpuEventState,
};
use nixe_memory::MemoryInvalidationSource;

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 17,
            frequency: 19,
        }
    }
}

fn state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(PC.get());
    state
}

#[test]
fn target_first_publication_installs_real_static_exits_without_explicit_link_registration() {
    // Exercise B, BL and both outcomes of CBZ. Both successors are resident
    // before publishing the source; no test explicitly requests/registers links.
    for (branch, x0, taken) in [
        (0x14000003, 0, true),
        (0x94000003, 0, true),
        (0xb4000060, 0, true),
        (0xb4000060, 5, false),
    ] {
        let mut worker = NativeWorker::default();
        let mut thread = budget::setup(
            &[
                0xd4200000, branch, 0x91000821, 0xd40000e1, // fallthrough: ADD X1,#2; SVC #7
                0x91000400, 0xd40000e1, // taken: ADD X0,#1; SVC #7
            ],
            false,
        );
        let process = thread.process.clone();
        let mut source = None;
        for offset in [8, 16, 4] {
            let key = thread.key(PC.checked_add(offset).unwrap()).unwrap();
            let Request::Owner(claim) = thread.reader.claim(key).unwrap() else {
                panic!()
            };
            let unit = thread
                .compiler
                .publish(
                    Compilation::capture(claim, &*process.memory).unwrap(),
                    &process.lifetime,
                    process.lifetime.executable_cache(),
                    &*process.memory,
                )
                .unwrap();
            if offset == 4 {
                source = Some(unit);
            }
        }
        let mut state = state();
        state.set_pc(PC.get() + 4);
        state.general_register_storage_mut()[0] = x0;
        let report = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                64,
                &Timer,
                &VcpuEventState::default(),
            )
            .unwrap();
        assert!(
            matches!(report.stop, CpuExit::SupervisorCall { source, immediate: 7 }
            if source.pc.get() == PC.get() + if taken { 20 } else { 12 })
        );
        assert_eq!(report.progress, 3);
        assert_eq!(
            state.general_register_storage_mut()[0],
            x0 + u64::from(taken)
        );
        assert_eq!(
            state.general_register_storage_mut()[1],
            if taken { 0 } else { 2 }
        );
        assert_eq!(
            state.general_register_storage_mut()[30],
            if branch == 0x94000003 {
                PC.get() + 8
            } else {
                0
            }
        );
        let source = process.lifetime.snapshot(source.unwrap()).unwrap();
        for map in &source.states {
            let Some(transfer) = &map.transfer else {
                continue;
            };
            if transfer.static_target.is_none() {
                continue;
            }
            let address = source.code.allocation.address();
            let fallback = crate::native::link::emit(
                source.code.metadata.abi,
                (address + map.native_offset as usize) as u64,
                (address + transfer.fallback_offset as usize) as u64,
                0,
            )
            .unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    (address + map.native_offset as usize) as *const u8,
                    fallback.patch().len(),
                )
            };
            assert_ne!(actual, fallback.patch());
        }
        drop(source);
        assert!(process.lifetime.try_shutdown().unwrap());
    }
}

#[test]
fn source_first_demand_links_only_the_executed_conditional_destination() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(
        &[
            0xd4200000, // unused setup entry
            0xb4000060, // CBZ X0,+12
            0x91000821, 0xd40000e1, // fallthrough: ADD X1,#2; SVC #7
            0x91000400, 0xd40000e1, // taken: ADD X0,#1; SVC #7
        ],
        false,
    );
    let process = thread.process.clone();
    let Request::Owner(claim) = thread
        .reader
        .claim(thread.key(PC.checked_add(4).unwrap()).unwrap())
        .unwrap()
    else {
        panic!()
    };
    let source_handle = thread
        .compiler
        .publish(
            Compilation::capture(claim, &*process.memory).unwrap(),
            &process.lifetime,
            process.lifetime.executable_cache(),
            &*process.memory,
        )
        .unwrap();
    let mut state = state();
    for (iteration, x0) in [0, 5, 0, 5].into_iter().enumerate() {
        state.set_pc(PC.get() + 4);
        state.general_register_storage_mut()[0] = x0;
        state.general_register_storage_mut()[1] = 0;
        let report = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                64,
                &Timer,
                &VcpuEventState::default(),
            )
            .unwrap();
        assert_eq!(report.progress, 3);
        assert!(
            matches!(report.stop, CpuExit::SupervisorCall { source, immediate: 7 }
            if source.pc.get() == PC.get() + if x0 == 0 { 20 } else { 12 })
        );
        assert_eq!(
            state.general_register_storage_mut()[0],
            if x0 == 0 { 1 } else { 5 }
        );
        assert_eq!(
            state.general_register_storage_mut()[1],
            if x0 == 0 { 0 } else { 2 }
        );
        // Inspect the actual published source, without retaining a compiler
        // claim or explicitly registering/installing any link.
        let source = process.lifetime.snapshot(source_handle).unwrap();
        for map in &source.states {
            let Some(transfer) = &map.transfer else {
                continue;
            };
            let Some(target) = transfer.static_target else {
                continue;
            };
            let address = source.code.allocation.address();
            let fallback = crate::native::link::emit(
                source.code.metadata.abi,
                (address + map.native_offset as usize) as u64,
                (address + transfer.fallback_offset as usize) as u64,
                0,
            )
            .unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    (address + map.native_offset as usize) as *const u8,
                    fallback.patch().len(),
                )
            };
            assert_eq!(
                actual != fallback.patch(),
                iteration != 0 || target.pc.get() == PC.get() + 16
            );
        }
    }
    assert!(thread.process.lifetime.try_shutdown().unwrap());
}

#[test]
fn slice_services_pending_links_on_closed_admission_and_deferred_control_exit() {
    for (deferred, svc) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut worker = NativeWorker::default();
        // The entry used by setup is separate. Publish the real source/target
        // explicitly so this fixture retains their handles for registration.
        let mut thread = budget::setup(
            &[
                0xd4200000,                                // BRK #0 (unused setup entry)
                0x14000002,                                // B target
                0xd4200120,                                // BRK #9 (must not execute)
                0x91000400,                                // target: ADD X0,X0,#1
                if svc { 0xd40000e1 } else { 0xd4200020 }, // SVC #7 or BRK #1
            ],
            false,
        );
        let process = thread.process.clone();
        let mut units = Vec::new();
        for offset in [4, 12] {
            let key = thread.key(PC.checked_add(offset).unwrap()).unwrap();
            let Request::Owner(claim) = thread.reader.claim(key).unwrap() else {
                panic!()
            };
            units.push(
                thread
                    .compiler
                    .publish(
                        Compilation::capture(claim, &*process.memory).unwrap(),
                        &process.lifetime,
                        process.lifetime.executable_cache(),
                        &*process.memory,
                    )
                    .unwrap(),
            );
        }
        // Target publication now schedules the waiting source automatically.
        // Keep the fallback for this fixture's explicit Closed/deferred cases.
        if let Some(mut transition) = process.lifetime.try_transition().unwrap() {
            transition.wait_closed().unwrap();
            transition
                .batch()
                .unwrap()
                .complete_with_links_deferred()
                .unwrap();
            assert!(transition.try_reopen().unwrap());
        }
        let source = process.lifetime.snapshot(units[0]).unwrap();
        let site = source
            .states
            .iter()
            .find(|site| {
                site.transfer
                    .as_ref()
                    .is_some_and(|transfer| transfer.static_target.is_some())
            })
            .unwrap();
        let width = site.transfer.as_ref().unwrap().patch_bytes as usize;
        let patch = (source.code.allocation.address() + site.native_offset as usize) as *const u8;
        let fallback = unsafe { std::slice::from_raw_parts(patch, width) }.to_vec();
        let ticket = process
            .lifetime
            .request(lifetime::Reason::LinkPatch)
            .unwrap();
        let mut transition = process.lifetime.try_transition().unwrap().unwrap();
        transition.wait_closed().unwrap();
        transition
            .refresh_static_link(units[0], 0)
            .unwrap()
            .unwrap();
        if deferred {
            transition
                .batch()
                .unwrap()
                .complete_with_links_deferred()
                .unwrap();
            assert!(transition.try_reopen().unwrap());
            thread.sample_remaining = 1;
        }
        drop(transition);
        let events = VcpuEventState::default();
        let mut state = state();
        for expected in [1, 2] {
            state.set_pc(PC.get() + 4);
            let report = thread
                .run_slice(
                    &mut crate::ReturnStack::default(),
                    &mut worker,
                    &mut state,
                    64,
                    &Timer,
                    &events,
                )
                .unwrap();
            if svc {
                assert!(matches!(report.stop,
                    CpuExit::SupervisorCall { source, immediate: 7 }
                        if source.pc.get() == PC.get() + 16));
                assert_eq!(state.pc(), PC.get() + 16);
            } else {
                assert!(matches!(
                    report.stop,
                    CpuExit::ArchitecturalException {
                        syndrome: Some(1),
                        ..
                    }
                ));
            }
            assert_eq!(report.progress, 3); // B, ADD and owned SVC/BRK completion.
            assert_eq!(state.general_register_storage_mut()[0], expected);
            assert!(ticket.is_complete().unwrap());
            assert_ne!(
                unsafe { std::slice::from_raw_parts(patch, width) },
                fallback
            );
        }
        drop(source);
        assert!(process.lifetime.try_shutdown().unwrap());
    }
}

#[test]
fn recognized_unsupported_catalog_preserves_identity_and_pre_state_on_both_platforms() {
    use nixe_cpu::decode::{DecodeSupport, a64};

    let mut worker = NativeWorker::default();
    for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
        for pattern in a64::patterns()
            .iter()
            .filter(|pattern| pattern.decoder == DecodeSupport::RecognizedUnimplemented)
        {
            let bits = pattern
                .regression_fixture
                .expect("unsupported fixture")
                .encoding
                .bits();
            let memory = memory(DirectBackendPolicy::Required);
            memory
                .overwrite_mapped_ram(SPACE, PC, &bits.to_le_bytes())
                .unwrap();
            let process =
                Arc::new(JitProcess::new(ProcessCpuContext::new(platform, SPACE), memory).unwrap());
            let mut thread = JitThread::new(process.clone()).unwrap();
            let mut state = state();
            state
                .general_register_storage_mut()
                .fill(0x1234_5678_9abc_def0);
            let before = state.clone();
            let report = thread
                .run_slice(
                    &mut crate::ReturnStack::default(),
                    &mut worker,
                    &mut state,
                    100,
                    &Timer,
                    &VcpuEventState::default(),
                )
                .unwrap();
            assert!(
                matches!(report.stop,
                    CpuExit::UnsupportedSemantics { coverage_id, source, .. }
                        if coverage_id == pattern.coverage_id && source.pc == PC
                ),
                "{platform:?}: {}: {:?}",
                pattern.name,
                report.stop
            );
            assert_eq!(report.progress, 0);
            assert_eq!(state, before);
            drop(thread);
            assert!(process.try_shutdown().unwrap());
        }
    }
}

#[test]
fn empty_slice_control_and_events_stop_before_native_registration_or_demand() {
    let previous_stack = worker::signal_stack();
    let mut worker = NativeWorker::default();
    let process = Arc::new(JitProcess::new(cpu(), memory(DirectBackendPolicy::Required)).unwrap());
    let mut thread = JitThread::new(process).unwrap();
    let mut state = state();
    let before = state.clone();
    let events = VcpuEventState::default();
    let control = thread.control();
    control.request(ControlRequest::Preempt);
    events.post_interrupts(5);
    for (limit, expected) in [
        (0, CpuExit::BudgetExhausted),
        (10, CpuExit::Safepoint),
        (10, CpuExit::PendingEvent { mask: 5 }),
    ] {
        let report = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                limit,
                &Timer,
                &events,
            )
            .unwrap();
        assert_eq!(report.stop, expected);
        assert_eq!(report.progress, 0);
        assert_eq!(report.context, Some(state.register_context()));
        assert_eq!(state, before);
        assert_eq!(worker::signal_stack(), previous_stack);
    }
    let error = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            u64::MAX,
            &Timer,
            &events,
        )
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::InvalidRequest);
    assert_eq!(error.progress, 0);
    assert_eq!(worker::signal_stack(), previous_stack);
}

#[test]
fn slice_demands_call_return_and_system_continuations_then_executes_exit_stub() {
    let mut worker = NativeWorker::default();
    // BL +16; B stub; stub: SVC #7; padding; MRS X0,CNTPCT_EL0; ADD; RET.
    let mut thread = budget::setup(
        &[
            0x94000004, 0x14000001, 0xd40000e1, 0, 0xd53be020, 0x91000400, 0xd65f03c0,
        ],
        false,
    );
    let mut state = state();
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            100,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(
        matches!(report.stop, CpuExit::SupervisorCall { source, immediate: 7 } if source.pc.get() == PC.get() + 8)
    );
    assert_eq!(report.progress, 6);
    assert_eq!(state.general_register_storage_mut()[0], 18);
    assert_eq!(state.pc(), PC.get() + 8);
    assert_eq!(state.general_register_storage_mut()[30], PC.get() + 4);
    assert_eq!(thread.sample_remaining, 4090);
    assert_eq!(report.context, Some(state.register_context()));
}

#[test]
fn backedges_obey_small_slices_and_preserve_sample_phase_between_runs() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0x91000400, 0x17ffffff], false); // ADD X0,X0,#1; B -4.
    thread.sample_remaining = 1;
    let mut state = state();
    for (limit, progress, value, phase) in [(3, 4, 2, 4093), (1, 2, 3, 4091)] {
        let report = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                limit,
                &Timer,
                &VcpuEventState::default(),
            )
            .unwrap();
        assert_eq!(report.stop, CpuExit::BudgetExhausted);
        assert_eq!(report.progress, progress);
        assert_eq!(state.general_register_storage_mut()[0], value);
        assert_eq!(state.pc(), PC.get());
        assert_eq!(thread.sample_remaining, phase);
    }
}

#[test]
fn exhausted_prefix_finishes_only_its_cold_instruction_before_yielding() {
    let mut worker = NativeWorker::default();
    // NOP; MRS X0,CNTPCT_EL0; ADD X1,X1,#1; BRK #9.
    let mut thread = budget::setup(&[0xd503201f, 0xd53be020, 0x91000421, 0xd4200120], false);
    let mut state = state();
    let events = VcpuEventState::default();
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            1,
            &Timer,
            &events,
        )
        .unwrap();
    assert_eq!(report.stop, CpuExit::BudgetExhausted);
    assert_eq!(report.progress, 2);
    assert_eq!(state.pc(), PC.get() + 8);
    assert_eq!(state.general_register_storage_mut()[0], 17);
    assert_eq!(state.general_register_storage_mut()[1], 0);
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            1,
            &Timer,
            &events,
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            syndrome: Some(9),
            ..
        }
    ));
    assert_eq!(report.progress, 2);
    assert_eq!(state.general_register_storage_mut()[1], 1);
    assert_eq!(state.pc(), PC.get() + 12);
}

#[test]
fn control_posted_by_completion_prevents_the_next_native_entry_and_preserves_event() {
    let mut worker = NativeWorker::default();
    struct PreemptingTimer {
        control: CpuControl,
        events: VcpuEventState,
    }
    impl ArchitecturalTimer for PreemptingTimer {
        fn snapshot(&self) -> TimerSnapshot {
            self.control.request(ControlRequest::Preempt);
            self.events.post_interrupts(2);
            Timer.snapshot()
        }
    }
    let mut thread = budget::setup(&[0xd53be020, 0x91000421, 0xd4200000], false);
    let mut state = state();
    let events = VcpuEventState::default();
    let timer = PreemptingTimer {
        control: thread.control(),
        events: events.clone(),
    };
    for (expected, progress) in [
        (CpuExit::Safepoint, 1),
        (CpuExit::PendingEvent { mask: 2 }, 0),
    ] {
        let report = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                100,
                &timer,
                &events,
            )
            .unwrap();
        assert_eq!(report.stop, expected);
        assert_eq!(report.progress, progress);
        assert_eq!(state.pc(), PC.get() + 4);
        assert_eq!(state.general_register_storage_mut()[0], 17);
        assert_eq!(state.general_register_storage_mut()[1], 0);
    }
}

#[test]
fn slice_reports_precise_fetch_and_data_faults_with_completed_prefix_only() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0xd503201f, 0xf9400020], false); // NOP; LDR X0,[X1].
    let events = VcpuEventState::default();
    let mut state = state();
    state.general_register_storage_mut()[1] = 0x5000;
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            100,
            &Timer,
            &events,
        )
        .unwrap();
    assert!(
        matches!(report.stop, CpuExit::DataFault { source, fault } if source.pc.get() == PC.get() + 4 && fault.address.get() == 0x5000)
    );
    assert_eq!(report.progress, 1);
    assert_eq!(state.pc(), PC.get() + 4);
    assert_eq!(report.context, Some(state.register_context()));
    for (pc, reason) in [
        (0x2000, InstructionFetchFaultReason::Unmapped),
        (0x1001, InstructionFetchFaultReason::Misaligned),
    ] {
        state.set_pc(pc);
        let report = thread
            .run_slice(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                100,
                &Timer,
                &events,
            )
            .unwrap();
        assert!(
            matches!(report.stop, CpuExit::FetchFault { fault } if fault.address.get() == pc && fault.reason == reason)
        );
        assert_eq!(report.progress, 0);
        assert_eq!(state.pc(), pc);
    }
}

#[test]
fn invalidation_notification_acknowledges_canonical_state_then_demands_replaced_code() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0x14000000], false); // B .
    thread
        .process
        .memory
        .overwrite_mapped_ram(SPACE, PC, &0xd4200120_u32.to_le_bytes())
        .unwrap();
    // The memory authority already closed admission and unlinked the old loop.
    // A CPU notification alone must never be used to make that write safe.
    let control = thread.control();
    let cursor = thread.process.memory.invalidation_cursor();
    control.request_invalidation(cursor.get());
    let mut state = state();
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            10,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            syndrome: Some(9),
            ..
        }
    ));
    assert_eq!(report.progress, 1);
    assert!(control.acknowledged_invalidation(cursor.get()));
}

#[test]
fn closed_admission_yields_without_acknowledging_maintenance_and_shutdown_is_terminal() {
    let mut worker = NativeWorker::default();
    let mut thread = budget::setup(&[0x14000000], false); // B .
    let process = thread.process.clone();
    let ticket = process
        .lifetime
        .request(lifetime::Reason::MappingChange)
        .unwrap();
    let mut state = state();
    let before = state.clone();
    let events = VcpuEventState::default();
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            1,
            &Timer,
            &events,
        )
        .unwrap();
    assert_eq!(report.stop, CpuExit::Safepoint);
    assert_eq!(report.progress, 0);
    assert_eq!(state, before);
    assert!(!ticket.is_complete().unwrap());
    process
        .lifetime
        .request(lifetime::Reason::Shutdown)
        .unwrap();
    let error = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut worker,
            &mut state,
            1,
            &Timer,
            &events,
        )
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::Unavailable);
    assert_eq!(error.progress, 0);
    assert_eq!(state, before);
}
