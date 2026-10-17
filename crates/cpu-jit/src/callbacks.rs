use crate::ffi;
use nixe_cpu::execution::ArchitecturalTimer;
use nixe_cpu::memory::*;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_memory::{CanonicalBackingPage, GuestVirtualAddress};
use std::collections::HashSet;
use std::ffi::c_void;

pub struct Context<'a> {
    pub memory: &'a ExecutionMemory,
    pub native_memory: NativeMemorySession<'a>,
    pub cpu: ProcessCpuContext,
    pub timer: &'a dyn ArchitecturalTimer,
    pub blocked_fetches: &'a mut HashSet<u64>,
    pub visibility: Option<CanonicalBackingPage>,
    pub exclusive_prepare: Option<CanonicalBackingPage>,
    pub device_access: bool,
    pub store_resume: Option<GuestVirtualAddress>,
    pub data_fault: Option<DataAccessFault>,
    pub fatal: Option<Box<str>>,
}
pub fn table() -> ffi::Callbacks {
    ffi::Callbacks {
        code,
        memory,
        counter,
        data_cache,
    }
}
unsafe extern "C" fn code(context: *mut c_void, address: u64, output: *mut u32) -> bool {
    crate::metrics::record(crate::metrics::Counter::CodeFetches, 1);
    let context = unsafe { &mut *context.cast::<Context<'_>>() };
    match context.memory.try_fetch32(
        context.cpu.address_space_id(),
        GuestVirtualAddress::new(address),
    ) {
        Ok(Some(word)) => {
            unsafe {
                *output = word.bits;
            }
            true
        }
        Ok(None) => {
            context.blocked_fetches.insert(address);
            false
        }
        // Translation can read ahead of the executed path. Report the actual
        // fetch fault only if Dynarmic executes its NoExecuteFault terminal.
        Err(_) => false,
    }
}
unsafe extern "C" fn counter(context: *mut c_void) -> u64 {
    unsafe { &*context.cast::<Context<'_>>() }
        .timer
        .snapshot()
        .counter
}
unsafe extern "C" fn data_cache(context: *mut c_void, address: u64) -> bool {
    let context = unsafe { &*context.cast::<Context<'_>>() };
    context.memory.try_native_data_cache_maintenance(
        context.cpu.address_space_id(),
        GuestVirtualAddress::new(address),
    )
}
unsafe extern "C" fn memory(
    context: *mut c_void,
    address: u64,
    bytes: u32,
    op: u32,
    value: *mut u64,
    expected: *const u64,
) -> bool {
    let context = unsafe { &mut *context.cast::<Context<'_>>() };
    crate::metrics::record(
        match op {
            0 => crate::metrics::Counter::ReadCallbacks,
            1 => crate::metrics::Counter::WriteCallbacks,
            _ => crate::metrics::Counter::AtomicCallbacks,
        },
        1,
    );
    nixe_memory::trace::detail_event("cpu_callback", address, u64::from(op), u64::from(bytes));
    let size = match bytes {
        1 => MemoryAccessSize::Byte,
        2 => MemoryAccessSize::Halfword,
        4 => MemoryAccessSize::Word,
        8 => MemoryAccessSize::Doubleword,
        16 => MemoryAccessSize::Quadword,
        _ => {
            context.fatal = Some("invalid Dynarmic memory width".into());
            return false;
        }
    };
    let address = GuestVirtualAddress::new(address);
    let space = context.cpu.address_space_id();
    let kind = if op == 0 {
        DataAccessKind::Read
    } else {
        DataAccessKind::Write
    };
    let access = if op == 2 {
        MemoryAccess::new(
            size,
            MemoryAlignment::Natural,
            MemoryOrdering::SequentiallyConsistent,
            MemoryAccessClass::Atomic,
        )
    } else {
        MemoryAccess {
            alignment: MemoryAlignment::Unaligned,
            ..MemoryAccess::normal(size)
        }
    };
    if matches!(op, 3 | 5) {
        let kind = if op == 3 {
            DataAccessKind::Read
        } else {
            DataAccessKind::Write
        };
        let access = MemoryAccess::new(
            size,
            MemoryAlignment::Natural,
            MemoryOrdering::SequentiallyConsistent,
            MemoryAccessClass::Exclusive,
        );
        match context
            .memory
            .native_exclusive_page(space, address, access, kind)
        {
            Ok(page) => {
                if op == 3 {
                    if matches!(
                        page.visibility_state(),
                        nixe_memory::VisibilityState::GpuNewer { .. }
                    ) {
                        context.visibility = Some(page);
                    } else {
                        context.exclusive_prepare = Some(page);
                    }
                    return false;
                }
                let (input, identity, version) = unsafe {
                    (
                        u128::from(*value) | (u128::from(*value.add(1)) << 64),
                        *expected,
                        *expected.add(1),
                    )
                };
                match page.try_exclusive_store(
                    address.get() as usize % nixe_memory::DIRECT_PAGE_SIZE,
                    bytes as usize,
                    Some(identity),
                    version,
                    input,
                ) {
                    Ok(Some(stored)) => {
                        unsafe {
                            *value = u64::from(stored);
                        }
                        return true;
                    }
                    Ok(None) => {
                        context.visibility = Some(page);
                        return false;
                    }
                    Err(error) => {
                        context.fatal = Some(error.to_string().into());
                        return false;
                    }
                }
            }
            Err(fault) => {
                context.data_fault = Some(fault);
                return false;
            }
        }
    }
    let input = unsafe { u128::from(*value) | (u128::from(*value.add(1)) << 64) };
    let result = if op == 2 {
        let expected = unsafe { u128::from(*expected) | (u128::from(*expected.add(1)) << 64) };
        match context.native_memory.compare_exchange(
            address,
            access,
            MemoryValue::from_bits(size, expected),
            MemoryValue::from_bits(size, input),
        ) {
            Ok(NativeAtomicResult::Complete(stored)) => Ok(u128::from(stored)),
            Ok(NativeAtomicResult::Visibility(page)) => {
                context.visibility = Some(page);
                return false;
            }
            Err(fault) => Err(fault),
        }
    } else {
        match context.native_memory.resolve_access(address, access, kind) {
            Ok(NativeMemoryAccess::Ram(mut page)) => {
                let mut data = input.to_le_bytes();
                let offset = address.get() as usize % nixe_memory::DIRECT_PAGE_SIZE;
                let copied = if op == 0 {
                    page.read(offset, &mut data[..bytes as usize])
                } else {
                    page.write_prepared(offset, &data[..bytes as usize])
                };
                copied.map(|_| u128::from_le_bytes(data)).map_err(|error| {
                    nixe_cpu::memory::DataAccessFault::new(
                        space,
                        address,
                        kind,
                        nixe_cpu::memory::DataAccessFaultReason::HostBacking(
                            error.to_string().into(),
                        ),
                    )
                })
            }
            Ok(NativeMemoryAccess::CheckedRam) => {
                let access = MemoryAccess {
                    alignment: MemoryAlignment::Unaligned,
                    ..MemoryAccess::normal(size)
                };
                if op == 0 {
                    context
                        .memory
                        .read(space, address, access)
                        .map(|r| r.value.bits())
                } else {
                    context
                        .memory
                        .write(space, address, access, MemoryValue::from_bits(size, input))
                        .map(|_| 0)
                }
            }
            Ok(NativeMemoryAccess::Fatal(message)) => {
                context.fatal = Some(message);
                return false;
            }
            Ok(NativeMemoryAccess::Visibility(page)) => {
                // Exit before the access commits. Device reconciliation runs
                // only after Run and its mapping lease have ended.
                context.visibility = Some(page);
                if op == 1 {
                    context.store_resume = Some(address);
                }
                return false;
            }
            Ok(NativeMemoryAccess::Device) => {
                context.device_access = true;
                if op == 1 {
                    context.store_resume = Some(address);
                }
                return false;
            }
            Err(fault) => {
                context.data_fault = Some(fault);
                return false;
            }
        }
    };
    match result {
        Ok(bits) => {
            unsafe {
                *value = bits as u64;
                *value.add(1) = (bits >> 64) as u64;
            }
            true
        }
        Err(fault) => {
            if context.data_fault.is_none() {
                context.data_fault = Some(fault);
            }
            false
        }
    }
}

pub fn maintain(
    memory: &ExecutionMemory,
    cpu: ProcessCpuContext,
    exit: &ffi::Exit,
) -> Result<(), DataAccessFault> {
    let address = GuestVirtualAddress::new(exit.value);
    if exit.kind == 5 {
        return memory.maintain_cache(
            cpu.address_space_id(),
            CacheMaintenanceKind::InstructionInvalidate,
            (exit.detail == 0).then_some(address),
        );
    }
    let kind = match exit.detail {
        0 | 1 => CacheMaintenanceKind::DataCleanAndInvalidate,
        2..=5 => CacheMaintenanceKind::DataClean,
        6 | 7 => CacheMaintenanceKind::DataInvalidate,
        8 => {
            let start = exit.value & !63;
            for offset in [0, 16, 32, 48] {
                memory.write(
                    cpu.address_space_id(),
                    GuestVirtualAddress::new(start + offset),
                    MemoryAccess::normal(MemoryAccessSize::Quadword),
                    MemoryValue::U128(0),
                )?;
            }
            return Ok(());
        }
        _ => unreachable!("bridge uses Dynarmic's closed cache operation enum"),
    };
    memory.maintain_cache(cpu.address_space_id(), kind, Some(address))
}
