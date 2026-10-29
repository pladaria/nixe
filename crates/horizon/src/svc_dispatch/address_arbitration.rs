//! Horizon's process-local address arbiter (SVC 0x34/0x35).
//!
//! ABI, validation and signed comparison/update semantics:
//! https://switchbrew.org/wiki/SVC#WaitForAddress
//! https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/svc/kern_svc_address_arbiter.cpp
//! https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_address_arbiter.cpp

use super::address_word::{AddressWord, address_word_fault};
use super::*;

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
        let permissions = if kind == 1 {
            MemoryPermissions::READ_WRITE
        } else {
            MemoryPermissions::READ
        };
        let word = match AddressWord::capture(context, address, permissions) {
            Ok(word) => word,
            Err(error) => return address_word_fault(context, 0x34, std::sync::Arc::new(error)),
        };
        let waits = context.process().address_waits().clone();
        let kernel = context.process().scheduler().clone();
        let thread = context.thread().id();
        let deadline = (timeout > 0).then(|| self.virtual_time_ns().saturating_add(timeout as u64));
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        let running = lifetime.clone();
        self.start_owned_svc(context, lifetime, None, async move {
            let turn = waits.address_operation(address, thread).await;
            running
                .begin()
                .map_err(|_| runtime_fault("WaitForAddress cancelled before its word operation"))?;
            let compared: Result<bool, std::sync::Arc<nixe_memory::MemoryError>> = async {
                let mut observed = word.read_owned().await?;
                loop {
                    if !(if kind == 2 {
                        observed as i32 == value
                    } else {
                        (observed as i32) < value
                    }) {
                        return Ok(false);
                    }
                    if kind != 1 {
                        return Ok(true);
                    }
                    let (previous, written) = word
                        .compare_exchange_owned(observed, observed.wrapping_sub(1))
                        .await?;
                    if written {
                        return Ok(true);
                    }
                    observed = previous;
                }
            }
            .await;
            let immediate = match compared {
                Ok(false) => Some(HorizonKernelResult::INVALID_STATE),
                Ok(true) if timeout == 0 => Some(HorizonKernelResult::TIMED_OUT),
                Ok(true) => None,
                Err(error) if error.is_access_fault() => {
                    Some(HorizonKernelResult::INVALID_CURRENT_MEMORY)
                }
                Err(fault) => {
                    return Err(HorizonSvcFault::MemoryAccess {
                        immediate: 0x34,
                        fault,
                    });
                }
            };
            if let Some(code) = immediate {
                return Ok(memory_wait::SvcCompletion::Return {
                    code,
                    output1: None,
                });
            }
            let wait = {
                let mut state = waits.lock();
                state
                    .enqueue_address_wait(address, thread)
                    .map_err(|fault| HorizonSvcFault::Scheduling {
                        operation: "WaitForAddress",
                        fault,
                    })?
            };
            if let Some(deadline) = deadline {
                kernel.arm_deadline(&wait, deadline);
            }
            drop(turn);
            let outcome = wait.wait().await;
            waits.lock().remove_address_wait(thread);
            let code = if outcome == nixe_scheduler::WaitOutcome::TimedOut {
                HorizonKernelResult::TIMED_OUT
            } else if outcome == nixe_scheduler::WaitOutcome::Terminated {
                HorizonKernelResult::CANCELLED
            } else {
                HorizonKernelResult::SUCCESS
            };
            Ok(memory_wait::SvcCompletion::Return {
                code,
                output1: None,
            })
        })
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
        let word = if kind == 0 {
            None
        } else {
            match AddressWord::capture(context, address, MemoryPermissions::READ) {
                Ok(word) => Some(word),
                Err(error) => return address_word_fault(context, 0x35, std::sync::Arc::new(error)),
            }
        };
        let waits = context.process().address_waits().clone();
        let thread = context.thread().id();
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        let running = lifetime.clone();
        self.start_owned_svc(context, lifetime, None, async move {
            let turn = waits.address_operation(address, thread).await;
            running.begin().map_err(|_| {
                runtime_fault("SignalToAddress cancelled before its word operation")
            })?;
            let limit = if count <= 0 {
                usize::MAX
            } else {
                count as usize
            };
            let (targets, more) = waits.lock().signal_targets(address, limit);
            let replacement = signal_replacement(kind, value, count, targets.is_empty(), more);
            let equal = if let Some(word) = word {
                if replacement == value {
                    word.read_owned()
                        .await
                        .map(|observed| observed == value as u32)
                } else {
                    word.compare_exchange_owned(value as u32, replacement as u32)
                        .await
                        .map(|(_, written)| written)
                }
            } else {
                Ok(true)
            };
            let code = match equal {
                Ok(true) => {
                    waits.lock().complete_signal(&targets);
                    HorizonKernelResult::SUCCESS
                }
                Ok(false) => HorizonKernelResult::INVALID_STATE,
                Err(error) if error.is_access_fault() => {
                    HorizonKernelResult::INVALID_CURRENT_MEMORY
                }
                Err(fault) => {
                    return Err(HorizonSvcFault::MemoryAccess {
                        immediate: 0x35,
                        fault,
                    });
                }
            };
            drop(turn);
            Ok(super::memory_wait::SvcCompletion::kernel(
                source,
                vcpu,
                move |_, context| {
                    result(context, code);
                    resume()
                },
            ))
        })
    }
}

// The unbounded-count case uses a distinct decrement from the bounded case.
// https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_address_arbiter.cpp
fn signal_replacement(kind: u32, value: i32, count: i32, empty: bool, more: bool) -> i32 {
    if kind == 0 {
        value
    } else if kind == 1 || empty {
        value.wrapping_add(1)
    } else if count <= 0 {
        value.wrapping_sub(2)
    } else if !more {
        value.wrapping_sub(1)
    } else {
        value
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

#[cfg(test)]
mod tests {
    use super::signal_replacement;
    #[test]
    fn conditional_signal_updates_match_bounded_and_unbounded_wait_counts() {
        for (kind, count, empty, more, expected) in [
            (0, 0, false, false, 9),
            (1, 1, false, false, 10),
            (2, 0, true, false, 10),
            (2, -1, false, false, 7),
            (2, 0, false, false, 7),
            (2, 1, false, false, 8),
            (2, 1, false, true, 9),
            (2, 4, false, false, 8),
        ] {
            assert_eq!(signal_replacement(kind, 9, count, empty, more), expected);
        }
        assert_eq!(signal_replacement(1, i32::MAX, 1, false, false), i32::MIN);
        assert_eq!(
            signal_replacement(2, i32::MIN, 0, false, false),
            i32::MAX - 1
        );
    }
}
