use super::*;
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::execution::{
    ArchitecturalTimer, CpuExit, SchedulerRequest, TimerSnapshot, VcpuEventState,
};
use nixe_cpu::memory::{DataAccessFaultReason, MemoryAccess, MemoryValue, SyntheticMmio};
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn exit(thread: &mut JitThread, state: &mut A64State) -> (invocation::Exit, PollBudget) {
    let mut worker = NativeWorker::default();
    let (exit, budget) = thread
        .invoke(&mut worker, state, PollBudget::new(4096, 1).unwrap())
        .unwrap();
    (exit.unwrap(), budget)
}

#[test]
fn system_and_scheduling_completion_charge_even_after_native_prefix_exhaustion() {
    for word in [0xd53be020, 0xd503203f, 0xd53b4420] {
        // MRS X0,CNTPCT_EL0; YIELD; MRS X0,FPSR.
        let mut thread = budget::setup(&[0xd503201f, word], false);
        let mut state = state();
        state.set_fpsr(16);
        let (exit, mut budget) = exit(&mut thread, &mut state);
        assert_eq!(budget.slice_remaining, 0);
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
        if word == 0xd503203f {
            assert!(matches!(
                stop,
                Some(CpuExit::Scheduled {
                    request: SchedulerRequest::Yield,
                    ..
                })
            ));
        } else {
            assert!(stop.is_none());
            assert_eq!(
                state.general_register_storage_mut()[0],
                if word == 0xd53b4420 { 16 } else { 17 }
            );
        }
        assert_eq!(state.pc(), PC.get() + 8);
        assert_eq!(budget.slice_remaining, -1);
        assert_eq!(budget.sample_remaining, 4094);
    }
}

#[test]
fn ic_completion_can_unlink_its_source_and_faults_do_not_earn_work() {
    let mut worker = NativeWorker::default();
    for valid in [false, true] {
        let mut thread = budget::setup(&[0xd503201f, 0xd50b7520], false); // IC IVAU,X0.
        let mut state = state();
        state.general_register_storage_mut()[0] = if valid { PC.get() } else { 0x5000 };
        let (exit, mut budget) = exit(&mut thread, &mut state);
        let before = state.clone();
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
        if valid {
            assert!(stop.is_none());
            assert_eq!(state.pc(), PC.get() + 8);
            assert_eq!(budget.slice_remaining, -1);
            state.set_pc(PC.get());
            assert!(
                thread
                    .invoke(&mut worker, &mut state, PollBudget::new(4096, 1).unwrap())
                    .unwrap()
                    .0
                    .is_none()
            );
        } else {
            assert!(
                matches!(stop, Some(CpuExit::DataFault { fault, .. }) if fault.reason == DataAccessFaultReason::Unmapped)
            );
            assert_eq!(state, before);
            assert_eq!(budget.slice_remaining, 0);
        }
    }
}

#[test]
fn exact_fp_completion_charges_success_but_preserves_pre_state_and_budget_on_trap() {
    for trap in [false, true] {
        let mut thread = budget::setup(&[0xd503201f, 0x1e622030], false); // FCMPE D1,D2.
        let mut state = state();
        state.set_vector(1, 0x7ff0000000000001); // Signaling NaN forces exact completion.
        state.set_fpcr(if trap { 1 << 8 } else { 0 });
        let (exit, mut budget) = exit(&mut thread, &mut state);
        let before = state.clone();
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
        if trap {
            assert!(matches!(
                stop,
                Some(CpuExit::ArchitecturalException {
                    kind: ExceptionKind::FloatingPoint,
                    syndrome: Some(1),
                    ..
                })
            ));
            assert_eq!(state, before);
            assert_eq!(budget.slice_remaining, 0);
        } else {
            assert!(stop.is_none());
            assert_eq!(state.nzcv().bits(), 0x30000000);
            assert_eq!(state.fpsr() & 1, 1);
            assert_eq!(state.pc(), PC.get() + 8);
            assert_eq!(budget.slice_remaining, -1);
        }
    }
}

#[test]
fn invalid_instruction_diagnostic_uses_captured_bits_after_code_is_replaced() {
    let mut thread = budget::setup(&[0xd503201f, 0], false);
    let mut state = state();
    let (exit, mut budget) = exit(&mut thread, &mut state);
    thread
        .process
        .memory
        .overwrite_mapped_ram(
            SPACE,
            PC.checked_add(4).unwrap(),
            &0xd4200120_u32.to_le_bytes(),
        )
        .unwrap();
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
    assert!(matches!(stop, Some(CpuExit::UnallocatedEncoding { error })
        if error.instruction.encoding == nixe_cpu::location::InstructionEncoding::from_u32(0)));
    assert_eq!(state.pc(), PC.get() + 4);
    assert_eq!(budget.slice_remaining, 0);
}

#[test]
fn explicit_guest_exceptions_are_delivered_once_and_native_dispatch_is_not_recharged() {
    for word in [0xd4000121, 0xd4200120, 0x14000000] {
        // SVC #9; BRK #9; B .
        let mut thread = budget::setup(&[0xd503201f, word], false);
        let mut state = state();
        let (exit, mut budget) = exit(&mut thread, &mut state);
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
        match word {
            0xd4000121 => assert!(matches!(
                stop,
                Some(CpuExit::SupervisorCall { immediate: 9, .. })
            )),
            0xd4200120 => assert!(matches!(
                stop,
                Some(CpuExit::ArchitecturalException {
                    kind: ExceptionKind::Breakpoint,
                    syndrome: Some(9),
                    ..
                })
            )),
            _ => assert!(stop.is_none()),
        }
        assert_eq!(budget.slice_remaining, -1);
        assert_eq!(state.pc(), PC.get() + 4);
    }
}

#[test]
fn mmio_completion_runs_once_and_charges_only_a_successful_access() {
    struct Device {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }
    impl SyntheticMmio for Device {
        fn read(&mut self, _: u64, _: MemoryAccess) -> Result<MemoryValue, Box<str>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                Err("device refused read".into())
            } else {
                Ok(MemoryValue::U64(37))
            }
        }
        fn write(&mut self, _: u64, _: MemoryAccess, _: MemoryValue) -> Result<(), Box<str>> {
            panic!()
        }
    }
    for fail in [false, true] {
        let mut memory = memory(DirectBackendPolicy::Required);
        let calls = Arc::new(AtomicUsize::new(0));
        let setup = Arc::get_mut(&mut memory).unwrap();
        let device = GuestPhysicalPageId::new(2);
        assert!(setup.add_mmio_page(
            device,
            Device {
                calls: calls.clone(),
                fail
            }
        ));
        assert!(setup.map_page(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            device,
            MemoryPermissions::READ_WRITE
        ));
        setup
            .overwrite_mapped_ram(
                SPACE,
                PC,
                &[
                    0x1f, 0x20, 0x03, 0xd5, 0x20, 0, 0x40, 0xf9, 0, 0, 0x20, 0xd4,
                ],
            )
            .unwrap(); // NOP; LDR X0,[X1]; BRK.
        let mut thread = JitThread::new(Arc::new(JitProcess::new(cpu(), memory).unwrap())).unwrap();
        assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
        let mut state = state();
        state.general_register_storage_mut()[1] = 0x3000;
        let (exit, mut budget) = exit(&mut thread, &mut state);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let before = state.clone();
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
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        if fail {
            assert!(
                matches!(stop, Some(CpuExit::DataFault { fault, .. }) if matches!(fault.reason, DataAccessFaultReason::Device(ref detail) if &**detail == "device refused read"))
            );
            assert_eq!(state, before);
            assert_eq!(budget.slice_remaining, 0);
        } else {
            assert!(stop.is_none());
            assert_eq!(state.general_register_storage_mut()[0], 37);
            assert_eq!(state.pc(), PC.get() + 8);
            assert_eq!(budget.slice_remaining, -1);
        }
    }
}
