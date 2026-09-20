use super::*;

#[test]
fn cold_completion_samples_exactly_its_own_deadline_after_prefix_exhaustion() {
    for word in [0xd53be020, 0xd503203f, 0xd53b4420] {
        // MRS counter; YIELD; MRS FPSR. None invents an outgoing region edge.
        for phase in [1, 2, 3] {
            let mut thread = budget::setup(&[0xd503201f, word], false);
            let mut state = state();
            let (exit, mut budget) = exit_at_sample(&mut thread, &mut state, phase);
            assert_eq!(exit.completion_sample().is_some(), phase == 2);
            let key = thread.key(PC).unwrap();
            let before = thread.samples.seed_snapshot(key);
            assert_eq!(before.is_some(), phase == 1);
            thread
                .complete(
                    exit,
                    &mut state,
                    &mut budget,
                    &Timer,
                    &VcpuEventState::default(),
                    1,
                )
                .unwrap();
            assert_eq!(budget.slice_remaining, -1);
            let after = thread.samples.seed_snapshot(key);
            match phase {
                1 => {
                    assert_eq!(after, before); // The prefix already consumed it.
                    assert_eq!(budget.sample_remaining, 4095);
                }
                2 => {
                    let (snapshot, score) = after.unwrap();
                    assert_eq!((snapshot.sequence, score), (1, 1));
                    assert_eq!(snapshot.last_edge, None);
                    assert_eq!(snapshot.successors, [None; 4]);
                    assert_eq!(budget.sample_remaining, 4096);
                }
                3 => {
                    assert!(after.is_none());
                    assert_eq!(budget.sample_remaining, 1);
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn trapping_cold_fp_does_not_charge_or_sample_the_pending_instruction() {
    for trap in [false, true] {
        let mut thread = budget::setup(&[0xd503201f, 0x1e622030], false);
        let mut state = state();
        state.set_vector(1, 0x7ff0000000000001); // FCMPE with signaling NaN.
        state.set_fpcr(if trap { 1 << 8 } else { 0 });
        let (exit, mut budget) = exit_at_sample(&mut thread, &mut state, 2);
        assert!(exit.completion_sample().is_some());
        let stop = thread
            .complete(
                exit,
                &mut state,
                &mut budget,
                &Timer,
                &VcpuEventState::default(),
                1,
            )
            .unwrap();
        let sample = thread.samples.seed_snapshot(thread.key(PC).unwrap());
        if trap {
            assert!(matches!(
                stop,
                Some(CpuExit::ArchitecturalException {
                    kind: ExceptionKind::FloatingPoint,
                    ..
                })
            ));
            assert_eq!(budget.sample_remaining, 1);
            assert_eq!(budget.slice_remaining, 0);
            assert!(sample.is_none());
        } else {
            assert!(stop.is_none());
            assert_eq!(budget.sample_remaining, 4096);
            assert_eq!(budget.slice_remaining, -1);
            assert_eq!(sample.unwrap().1, 1);
        }
    }
}

#[test]
fn later_unit_completion_retains_its_root_and_rejects_a_replaced_source() {
    for replace in [false, true] {
        // A: B B; B: NOP; MRS X0,FPSR.
        let mut thread = budget::setup(&[0x14000001, 0xd503201f, 0xd53b4420], false);
        let source = thread.key(PC.checked_add(4).unwrap()).unwrap();
        assert!(matches!(thread.demand(source.pc).unwrap(), Demand::Ready));
        assert!(thread.process.lifetime.try_service_links().unwrap());
        let mut worker = NativeWorker::default();
        let mut state = state();
        state.set_fpsr(1 << 27);
        let (exit, mut budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(3, 2).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap();
        let exit = exit.unwrap();
        assert!(exit.completion_sample().is_some());
        assert_eq!(budget.sample_remaining, 1);
        if replace {
            thread
                .process
                .memory
                .overwrite_mapped_ram(SPACE, source.pc, &0xd4200000_u32.to_le_bytes())
                .unwrap();
            assert!(matches!(thread.demand(source.pc).unwrap(), Demand::Ready));
            assert!(thread.process.lifetime.try_service_links().unwrap());
        }
        let stop = thread
            .complete(
                exit,
                &mut state,
                &mut budget,
                &Timer,
                &VcpuEventState::default(),
                2,
            )
            .unwrap();
        assert!(stop.is_none());
        assert_eq!(state.general_register_storage_mut()[0], 1 << 27);
        assert_eq!(budget.sample_remaining, 4096);
        assert_eq!(budget.slice_remaining, -1);
        assert!(
            thread
                .samples
                .seed_snapshot(thread.key(PC).unwrap())
                .is_none()
        );
        let sample = thread.samples.seed_snapshot(source);
        if replace {
            assert!(sample.is_none());
        } else {
            assert_eq!(sample.unwrap().1, 1);
        }
    }
}

#[test]
fn cache_completion_invalidating_its_source_cannot_heat_stale_code() {
    let mut thread = budget::setup(&[0xd503201f, 0xd50b7520], false); // IC IVAU,X0.
    let mut state = state();
    state.general_register_storage_mut()[0] = PC.get();
    let (exit, mut budget) = exit_at_sample(&mut thread, &mut state, 2);
    assert!(exit.completion_sample().is_some());
    assert!(
        thread
            .complete(
                exit,
                &mut state,
                &mut budget,
                &Timer,
                &VcpuEventState::default(),
                1
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(budget.sample_remaining, 4096);
    assert_eq!(budget.slice_remaining, -1);
    assert!(
        thread
            .samples
            .seed_snapshot(thread.key(PC).unwrap())
            .is_none()
    );
}
