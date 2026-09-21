use super::*;
use crate::lcq::invocation::{Exit, MemoryExit};
use nixe_cpu::memory::{
    CpuMemory, MemoryAccess, MemoryAccessSize, MemoryAliasRequest, MemoryAttributes,
    MemoryMappingProperties, MemoryMappingPurpose, MemoryValue,
};

pub(super) fn setup(words: &[u32], writable: bool) -> JitThread {
    let memory = memory(DirectBackendPolicy::Required);
    memory
        .overwrite_mapped_ram(
            SPACE,
            PC,
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    if writable {
        let executable = MemoryMappingProperties::new(
            MemoryPermissions::READ_EXECUTE,
            MemoryMappingPurpose::Normal,
            MemoryAttributes::NONE,
        );
        memory
            .map_alias(MemoryAliasRequest {
                address_space: SPACE,
                source: PC,
                destination: GuestVirtualAddress::new(0x3000),
                size: 4096,
                source_before: executable,
                source_after: executable,
                destination_properties: MemoryMappingProperties::new(
                    MemoryPermissions::READ_WRITE,
                    MemoryMappingPurpose::Normal,
                    MemoryAttributes::NONE,
                ),
            })
            .unwrap();
    }
    // These tests assert baseline-specific fragment/poll behavior. Promotion
    // has separate execution tests with an explicitly selected nonzero pool.
    let process = Arc::new(JitProcess::with_workers(cpu(), memory, 0).unwrap());
    let mut thread = JitThread::new(process).unwrap();
    assert!(matches!(thread.demand(PC).unwrap(), Demand::Ready));
    thread
}

#[test]
fn cold_poll_observes_bound_requests_without_consuming_them() {
    use nixe_cpu::execution::ControlRequest;
    for interrupt in [false, true] {
        let mut worker = NativeWorker::default();
        let mut thread = setup(&[0xb1000400, 0x14000000], false); // ADDS X0,X0,#1; B .
        let events = VcpuEventState::default();
        if interrupt {
            events.post_interrupts(4);
        } else {
            thread.control.request(ControlRequest::Preempt);
        }
        let mut state = A64State::default();
        state.set_pc(PC.get());
        let (Some(Exit::Native { returned, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(1, 100).unwrap(),
                &events,
            )
            .unwrap()
        else {
            panic!("expected native control exit")
        };
        assert_eq!(returned.reason, crate::abi::NativeExitReason::Control);
        assert!(!returned.poll.sample && !returned.poll.exhausted);
        assert_eq!(budget.slice_remaining, 98);
        assert_eq!(budget.sample_remaining, 4095);
        assert_eq!(state.general_register_storage_mut()[0], 1);
        assert_eq!(state.pc(), PC.get() + 4);
        if interrupt {
            assert_eq!(events.take_pending_interrupts(), 4);
        } else {
            assert!(
                thread
                    .control
                    .take_pending()
                    .unwrap()
                    .contains(ControlRequest::Preempt)
            );
        }
    }
}

#[test]
fn bounded_straight_line_work_is_charged_once_with_slice_and_sample_overshoot() {
    let mut worker = NativeWorker::default();
    for count in [1, 17, 512] {
        let mut thread = setup(&vec![0xd503201f; count], false); // NOPs then invalid word, or ceiling.
        let mut state = A64State::default();
        state.set_pc(PC.get());
        let (Some(Exit::Native { returned, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(1, 1).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(budget.slice_remaining, 1 - count as i64);
        assert!(returned.poll.exhausted && returned.poll.sample);
        assert_eq!(budget.sample_remaining, 4097 - count as i64);
        assert_eq!(state.pc(), PC.get() + count as u64 * 4);
    }
}

#[test]
fn taken_and_untaken_branches_charge_the_branch_and_preserve_lazy_nzcv() {
    let mut worker = NativeWorker::default();
    let mut thread = setup(&[0xb1000400, 0x54000000], false); // ADDS X0,X0,#1; B.EQ .
    for value in [0, u64::MAX] {
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[0] = value;
        let (
            Some(Exit::Native {
                returned, guest, ..
            }),
            budget,
        ) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(4096, 1).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(budget.slice_remaining, -1);
        assert!(returned.poll.exhausted);
        assert_eq!(
            guest.kind,
            if value == 0 {
                EdgeKind::NotTaken
            } else {
                EdgeKind::Taken
            }
        );
        assert_eq!(state.pc(), PC.get() + if value == 0 { 8 } else { 4 });
        assert_eq!(state.nzcv().bits(), if value == 0 { 0 } else { 0x60000000 });
    }
}

#[test]
fn escaped_fault_charges_only_completed_prefix_and_retry_does_not_double_charge() {
    let mut worker = NativeWorker::default();
    // NOP; STR X0,[X1]; BRK. Writing the armed executable page repairs protection
    // and retries the identical store. An unmapped target instead escapes PRE.
    for valid in [false, true] {
        let mut thread = setup(&[0xd503201f, 0xf9000020, 0xd4200000], true);
        let mut state = A64State::default();
        state.set_pc(PC.get());
        state.general_register_storage_mut()[0] = 37;
        state.general_register_storage_mut()[1] = if valid { 0x3800 } else { 0x5000 };
        let (Some(exit), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(4096, 1).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        if valid {
            assert!(
                matches!(exit, Exit::Native { guest, .. } if guest.kind == EdgeKind::Breakpoint(0))
            );
            assert_eq!(budget.slice_remaining, -1);
            assert_eq!(
                thread
                    .process
                    .memory
                    .read(
                        SPACE,
                        GuestVirtualAddress::new(0x3800),
                        MemoryAccess::normal(MemoryAccessSize::Doubleword)
                    )
                    .unwrap()
                    .value,
                MemoryValue::U64(37)
            );
        } else {
            assert!(matches!(
                exit,
                Exit::Memory {
                    outcome: MemoryExit::Fault(_),
                    ..
                }
            ));
            assert_eq!(budget.slice_remaining, 0);
            assert_eq!(state.pc(), PC.get() + 4);
        }
    }
}

#[test]
fn pre_completion_exits_do_not_charge_the_pending_instruction() {
    let mut worker = NativeWorker::default();
    for word in [0xd4200000, 0xd4000001, 0xd53b4420] {
        // BRK; SVC; MRS X0,FPSR.
        let mut thread = setup(&[0xd503201f, word], false);
        let mut state = A64State::default();
        state.set_pc(PC.get());
        let (Some(Exit::Native { returned, .. }), budget) = thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                PollBudget::new(4096, 10).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(budget.slice_remaining, 9);
        assert_eq!(returned.reason, crate::abi::NativeExitReason::Architectural);
        assert_eq!(state.pc(), PC.get() + 4);
    }
}

#[test]
fn exhausted_budget_never_registers_a_worker_or_enters_native_code() {
    let previous_stack = worker::signal_stack();
    let mut worker = NativeWorker::default();
    let mut thread = setup(&[0x91000400, 0xd4200000], false);
    for remaining in [0, -1] {
        let mut state = A64State::default();
        state.set_pc(PC.get());
        let mut budget = PollBudget::new(4096, 1).unwrap();
        budget.reconcile(remaining, false).unwrap();
        assert!(matches!(
            thread.invoke(
                &mut crate::ReturnStack::default(),
                &mut worker,
                &mut state,
                budget,
                &VcpuEventState::default()
            ),
            Err(invocation::Error::Native(
                crate::native::NativeReturnError::Budget(crate::abi::BudgetError::ExhaustedSlice)
            ))
        ));
        assert_eq!(state.general_register_storage_mut()[0], 0);
        assert_eq!(state.pc(), PC.get());
        assert_eq!(worker::signal_stack(), previous_stack);
    }
}
