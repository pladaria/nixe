use crate::ffi;
use nixe_cpu::execution::ArchitecturalTimer;
use nixe_cpu::memory::*;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_memory::{CanonicalBackingPage, GuestVirtualAddress};
use std::collections::HashSet;
use std::ffi::c_void;

pub struct Context<'a> {
    pub memory: &'a ExecutionMemory,
    pub cpu: ProcessCpuContext,
    pub timer: &'a dyn ArchitecturalTimer,
    pub blocked_fetches: &'a mut HashSet<u64>,
    pub visibility: Option<CanonicalBackingPage>,
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
    crate::metrics::record(crate::metrics::Counter::MemoryCallbacks, 1);
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
    let input = unsafe { u128::from(*value) | (u128::from(*value.add(1)) << 64) };
    let result = if op == 2 {
        let expected = unsafe { u128::from(*expected) | (u128::from(*expected.add(1)) << 64) };
        match context.memory.try_native_compare_exchange(
            space,
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
        let resolution = match context
            .memory
            .resolve_native_access(space, address, access, kind)
        {
            Ok(NativeMemoryAccess::Ram) => DirectFaultResolution::Retry,
            Ok(NativeMemoryAccess::CheckedRam) => DirectFaultResolution::Cold,
            Ok(NativeMemoryAccess::Fatal(message)) => {
                context.fatal = Some(message);
                return false;
            }
            Ok(NativeMemoryAccess::Visibility(page)) => {
                // MemoryAbort exits at the current guest PC before committing the
                // failed access. Resolve only after Run and its lease have ended.
                // vendor/dynarmic/src/dynarmic/backend/x64/a64_emit_x64_memory.cpp
                context.visibility = Some(page);
                if op == 1 {
                    context.store_resume = Some(address);
                }
                return false;
            }
            Ok(NativeMemoryAccess::Device) => {
                // Never perform an externally observable device access in an
                // instruction that may subsequently need a visibility restart.
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
        };
        match resolution {
            DirectFaultResolution::Retry => {
                // The execution lease keeps mappings stable; the resolver checks
                // the complete access and reconciles tracking/device ownership.
                let arena = context.memory.direct_address_space_view(space).unwrap();
                let pointer = (arena.base + address.get() as usize) as *mut u8;
                let mut data = input.to_le_bytes();
                unsafe {
                    if op == 0 {
                        std::ptr::copy_nonoverlapping(pointer, data.as_mut_ptr(), bytes as usize);
                    } else {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), pointer, bytes as usize);
                    }
                }
                Ok(u128::from_le_bytes(data))
            }
            DirectFaultResolution::Cold => {
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
            DirectFaultResolution::Fault(fault) => Err(fault),
            DirectFaultResolution::Fatal(message) => {
                context.fatal = Some(message);
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
