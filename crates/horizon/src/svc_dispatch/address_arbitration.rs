//! Horizon's process-local address arbiter (SVC 0x34/0x35).
//!
//! ABI, validation and signed comparison/update semantics:
//! https://switchbrew.org/wiki/SVC#WaitForAddress
//! https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/svc/kern_svc_address_arbiter.cpp
//! https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_address_arbiter.cpp

use super::synchronization::read_process_wide_key_word;
use super::*;
use nixe_cpu::memory::{MemoryAccessClass, MemoryAlignment, MemoryOrdering};
use nixe_runtime::AddressWaitResult;

// https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/include/mesosphere/kern_k_memory_layout.hpp
const KERNEL_ADDRESS_START: u64 = 0u64.wrapping_sub(1 << 39);
const KERNEL_ADDRESS_END: u64 = 0u64.wrapping_sub(0x20_0000);

impl HorizonSvcDispatcher {
    pub(super) fn wait_for_address(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = read_register(context.thread().state(), 0);
        let kind = read_register(context.thread().state(), 1) as u32;
        let value = read_register(context.thread().state(), 2) as u32 as i32;
        let timeout = read_register(context.thread().state(), 3) as i64;
        if let Some(code) = validate_address(address, kind) {
            result(context, code);
            return resume();
        }

        let thread = context.thread().id();
        let now = self.virtual_time_ns();
        let waits = context
            .process_mut()
            .address_waits_mut()
            .priority_waits_mut();
        waits.expire(now);
        if let Some(completion) = waits.take_result(thread) {
            self.finish_wait(thread.get(), 0x34);
            result(
                context,
                match completion {
                    AddressWaitResult::Signalled => HorizonKernelResult::SUCCESS,
                    AddressWaitResult::TimedOut => HorizonKernelResult::TIMED_OUT,
                },
            );
            return resume();
        }
        if waits.contains(thread) {
            // A retry is a continuation of the original wait: do not repeat
            // the comparison or decrement, or restart its deadline.
            return ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        }

        let can_wait = if kind == 1 {
            match decrement_if_less_than(context, address, value) {
                Ok(can_wait) => can_wait,
                Err(fault) => return memory_fault(context, 0x34, fault),
            }
        } else {
            let observed = match read_process_wide_key_word(context, address) {
                Ok(observed) => observed as i32,
                Err(fault) => return memory_fault(context, 0x34, fault),
            };
            if kind == 0 {
                observed < value
            } else {
                observed == value
            }
        };
        if !can_wait {
            result(context, HorizonKernelResult::INVALID_STATE);
            return resume();
        }
        // DecrementAndWaitIfLessThan updates even with a zero timeout.
        if timeout == 0 {
            result(context, HorizonKernelResult::TIMED_OUT);
            return resume();
        }
        let deadline = (timeout > 0).then(|| now.saturating_add(timeout as u64));
        // Routing and applying staged effects run under the coordinator's
        // exclusive ownership, before another SVC can signal this address.
        // Registration needs the scheduler's current effective priority.
        match self.queue_runtime_request(
            thread,
            PendingRuntimeRequest::RegisterAddressWait { address, deadline },
            "WaitForAddress",
        ) {
            Ok(()) => ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry),
            Err(fault) => ExceptionDispatchOutcome::Fault(fault),
        }
    }

    pub(super) fn signal_to_address(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let address = read_register(context.thread().state(), 0);
        let kind = read_register(context.thread().state(), 1) as u32;
        let value = read_register(context.thread().state(), 2) as u32 as i32;
        let count = read_register(context.thread().state(), 3) as u32 as i32;
        if let Some(code) = validate_address(address, kind) {
            result(context, code);
            return resume();
        }
        let now = self.virtual_time_ns();
        let waits = context
            .process_mut()
            .address_waits_mut()
            .priority_waits_mut();
        waits.expire(now);
        let waiting = if kind == 2 {
            waits.waiting_count(address)
        } else {
            0
        };
        if kind != 0 {
            let replacement = if kind == 1 || waiting == 0 {
                value.wrapping_add(1)
            } else if count <= 0 || waiting <= count as usize {
                value.wrapping_sub(1)
            } else {
                value
            };
            let equal = if replacement == value {
                match read_process_wide_key_word(context, address) {
                    Ok(observed) => observed as i32 == value,
                    Err(fault) => return memory_fault(context, 0x35, fault),
                }
            } else {
                match compare_exchange(context, address, value as u32, replacement as u32) {
                    Ok(observed) => observed == value as u32,
                    Err(fault) => return memory_fault(context, 0x35, fault),
                }
            };
            if !equal {
                result(context, HorizonKernelResult::INVALID_STATE);
                return resume();
            }
        }
        let count = if count <= 0 {
            usize::MAX
        } else {
            count as usize
        };
        context
            .process_mut()
            .address_waits_mut()
            .priority_waits_mut()
            .signal(address, count);
        result(context, HorizonKernelResult::SUCCESS);
        resume()
    }
}

fn validate_address(address: u64, kind: u32) -> Option<HorizonKernelResult> {
    if (KERNEL_ADDRESS_START..KERNEL_ADDRESS_END).contains(&address) {
        Some(HorizonKernelResult::INVALID_CURRENT_MEMORY)
    } else if !address.is_multiple_of(4) {
        Some(HorizonKernelResult::INVALID_ADDRESS)
    } else if kind > 2 {
        Some(HorizonKernelResult::INVALID_ENUM_VALUE)
    } else {
        None
    }
}

fn decrement_if_less_than(
    context: &ExceptionDispatchContext<'_>,
    address: u64,
    value: i32,
) -> Result<bool, DataAccessFault> {
    // Horizon's CanAccessAtomic validates writable, normal memory even when
    // the comparison fails. Consume that requirement without a dummy write.
    // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/include/mesosphere/arch/arm64/kern_cpu.hpp
    let process = context.process();
    let (region, mapping) = process
        .memory()
        .query_page(
            process.cpu().address_space_id(),
            GuestVirtualAddress::new(address),
        )
        .ok_or_else(|| {
            DataAccessFault::new(
                process.cpu().address_space_id(),
                GuestVirtualAddress::new(address),
                nixe_cpu::memory::DataAccessKind::Write,
                DataAccessFaultReason::Unmapped,
            )
        })?;
    if region != MemoryRegionKind::Ram
        || !mapping.permissions.contains(MemoryPermissions::READ_WRITE)
    {
        return Err(DataAccessFault::new(
            process.cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            nixe_cpu::memory::DataAccessKind::Write,
            DataAccessFaultReason::AtomicRegionUnsupported,
        ));
    }
    let mut observed = read_process_wide_key_word(context, address)?;
    loop {
        if observed as i32 >= value {
            return Ok(false);
        }
        let previous = compare_exchange(context, address, observed, observed.wrapping_sub(1))?;
        if previous == observed {
            return Ok(true);
        }
        observed = previous;
    }
}

pub(super) fn compare_exchange(
    context: &ExceptionDispatchContext<'_>,
    address: u64,
    expected: u32,
    replacement: u32,
) -> Result<u32, DataAccessFault> {
    let access = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        MemoryOrdering::AcquireRelease,
        MemoryAccessClass::Atomic,
    );
    context
        .process()
        .memory()
        .atomic_compare_exchange(
            context.process().cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            access,
            MemoryValue::U32(expected),
            MemoryValue::U32(replacement),
        )
        .map(|result| match result.previous {
            MemoryValue::U32(value) => value,
            _ => unreachable!("word compare-exchange returns a word"),
        })
}

fn memory_fault(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    fault: DataAccessFault,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    if matches!(
        fault.reason,
        DataAccessFaultReason::Unmapped
            | DataAccessFaultReason::ReadPermissionDenied
            | DataAccessFaultReason::WritePermissionDenied
            | DataAccessFaultReason::AddressOverflow
            | DataAccessFaultReason::AtomicRegionUnsupported
    ) {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        resume()
    } else {
        ExceptionDispatchOutcome::Fault(HorizonSvcFault::GuestMemory { immediate, fault })
    }
}
