use super::*;

pub(super) fn get_info(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let info_type = read_register(context.thread().state(), 1) as u32;
    let handle = read_register(context.thread().state(), 2) as u32;
    let subtype = read_register(context.thread().state(), 3);
    log::trace!("GetInfo: type={info_type:#x} handle={handle:#x} subtype={subtype:#x}");
    // DebuggerAttached takes a zero handle rather than a process handle.
    // No guest debugger can attach to this runtime; Eden reports the same zero.
    // https://switchbrew.org/wiki/SVC#InfoType
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/kernel/svc/svc_info.cpp
    if info_type == 8 {
        if handle != INVALID_HANDLE {
            result(context, HorizonKernelResult::INVALID_HANDLE);
        } else if subtype != 0 {
            result(context, HorizonKernelResult::INVALID_COMBINATION);
        } else {
            result(context, HorizonKernelResult::SUCCESS);
            write_u64(context.thread_mut().state_mut(), 1, 0);
        }
        return resume();
    }
    // RandomEntropy validation and its four process-owned values follow the
    // public kernel implementation:
    // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_info.cpp#L230-L240
    if info_type == 11 {
        if handle != INVALID_HANDLE {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let Some(value) = usize::try_from(subtype)
            .ok()
            .and_then(|index| context.process().random_entropy(index))
        else {
            result(context, HorizonKernelResult::INVALID_COMBINATION);
            return resume();
        };
        result(context, HorizonKernelResult::SUCCESS);
        write_u64(context.thread_mut().state_mut(), 1, value);
        return resume();
    }
    if subtype != 0 {
        result(context, HorizonKernelResult::INVALID_COMBINATION);
        return resume();
    }
    // Retain the selected KProcess, then query it outside the handle-table lock.
    // The current-process pseudohandle already has the dispatch context's owner.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_info.cpp
    let retained;
    let process = if handle == CURRENT_PROCESS_HANDLE {
        context.process().as_ref()
    } else {
        retained = match context
            .process()
            .handles()
            .lookup_as::<nixe_runtime::RunnableProcess>(handle)
        {
            Some(process) => process,
            None => {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            }
        };
        retained.as_ref()
    };
    let layout = process.memory_layout();
    // Horizon derives these six physical-memory views from the same KProcess
    // counters. Runtime keeps their shared accounting incremental so queries
    // remain mutually consistent without scanning mappings here.
    // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_process.cpp
    let value = match info_type {
        // Use the same effective NPDM permissions as runtime thread validation.
        // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/svc/kern_svc_info.cpp
        0 | 1 => {
            let mask = if info_type == 0 {
                process.allowed_core_mask()
            } else {
                process.allowed_priority_mask()
            };
            let Some(mask) = mask else {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::UnsupportedSemantics {
                    immediate: 0x29,
                    documented_name: "GetInfo (process thread policy unavailable)",
                });
            };
            mask
        }
        2 => layout.alias().base().get(),
        3 => layout.alias().size(),
        4 => layout.heap().base().get(),
        5 => layout.heap().size(),
        12 => layout.aslr().base().get(),
        13 => layout.aslr().size(),
        14 => layout.stack().base().get(),
        15 => layout.stack().size(),
        6 => process.total_user_physical_memory_size(),
        7 => process.used_user_physical_memory_size(),
        16 => process.total_system_resource_size(),
        17 => process.used_system_resource_size(),
        // UserExceptionContextAddress is the process-local region, allocated
        // from kernel TLS pages rather than borrowed from a live thread.
        // https://switchbrew.org/wiki/SVC#InfoType
        20 => process.user_exception_context_address().get(),
        21 => process.total_non_system_user_physical_memory_size(),
        22 => process.used_non_system_user_physical_memory_size(),
        28 => 0,
        // https://switchbrew.org/wiki/SVC#InfoType
        29 | 30 if layout.remote().is_some() => {
            let region = layout.remote().unwrap();
            if info_type == 29 {
                region.base().get()
            } else {
                region.size()
            }
        }
        // @SPECULATIVE: usage/peak count the guest-owned physical allocations
        // charged to the remote account, including allocations by remote HLE
        // owners. The kernel's precise charging rules are not documented.
        31 if layout.remote().is_some() => process.remote_memory().used_bytes(),
        32 if layout.remote().is_some() => process.remote_memory().peak_bytes(),
        // @SPECULATIVE: expose our 4 KiB process granule. Switchbrew identifies
        // this as ProcessPageSize but does not specify S2 granule selection.
        33 if layout.remote().is_some() => nixe_memory::PAGE_SIZE,
        _ => {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::UnsupportedInfoType {
                info_type,
                handle,
                subtype,
            });
        }
    };
    result(context, HorizonKernelResult::SUCCESS);
    write_u64(context.thread_mut().state_mut(), 1, value);
    resume()
}

pub(super) fn terminate(
    scope: ExceptionTerminationScope,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    ExceptionDispatchOutcome::Terminate {
        scope,
        exit_code: 0,
        reason: ExceptionTerminationReason::Requested,
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn break_process(
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let reason = read_register(context.thread().state(), 0);
        let info = read_register(context.thread().state(), 1);
        let size = read_register(context.thread().state(), 2);
        if reason & 0x8000_0000 != 0 {
            result(context, HorizonKernelResult::SUCCESS);
            return resume();
        }
        let Some(count) = usize::try_from(size)
            .ok()
            .filter(|size| (1..=nixe_runtime::MAX_GUEST_BREAK_PAYLOAD_BYTES).contains(size))
        else {
            return break_termination(reason, info, size, None);
        };
        let mut bytes = [0; nixe_runtime::MAX_GUEST_BREAK_PAYLOAD_BYTES];
        match context
            .process()
            .memory()
            .space()
            .read_into(GuestVirtualAddress::new(info), &mut bytes[..count])
        {
            Ok(nixe_memory::Access::Ready(())) => break_termination(
                reason,
                info,
                size,
                nixe_runtime::GuestBreakPayload::new(&bytes[..count]),
            ),
            Err(error) if error.is_access_fault() => break_termination(reason, info, size, None),
            Err(error) => ExceptionDispatchOutcome::Fault(HorizonSvcFault::MemoryAccess {
                immediate: 0x26,
                fault: std::sync::Arc::new(error),
            }),
            Ok(nixe_memory::Access::Pending(access)) => {
                match super::access::wait_read(context, 0x26, access) {
                    Ok(bytes) => break_termination(
                        reason,
                        info,
                        size,
                        nixe_runtime::GuestBreakPayload::new(&bytes),
                    ),
                    Err(fault) => reject(context, fault),
                }
            }
        }
    }
}
fn break_termination(
    reason: u64,
    info: u64,
    size: u64,
    payload: Option<nixe_runtime::GuestBreakPayload>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    ExceptionDispatchOutcome::Terminate {
        scope: ExceptionTerminationScope::Process,
        exit_code: reason,
        reason: ExceptionTerminationReason::Break {
            reason,
            info,
            size,
            payload,
        },
    }
}

use nixe_runtime::memory::{Mapping, MappingError, MemoryKind};
use std::sync::Arc;

fn mapping(kind: MemoryKind, permissions: MemoryPermissions) -> Mapping {
    Mapping {
        kind,
        permissions,
        uncached: false,
        locked: false,
        permission_locked: false,
        gpu_sharable: false,
        gpu_shared: 0,
    }
}
fn mapping_result(
    context: &mut ExceptionDispatchContext<'_>,
    outcome: Result<(), MappingError>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    match outcome {
        Ok(()) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(
            context,
            HorizonSvcFault::MemoryMapping {
                fault: Arc::new(fault),
            },
        ),
    }
}
// Both SVCs operate on process-owned normal pages in the Alias reservation.
// https://switchbrew.org/wiki/SVC#MapPhysicalMemory
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/kernel/svc/svc_physical_memory.cpp
pub(super) fn physical_memory(
    context: &mut ExceptionDispatchContext<'_>,
    unmap: bool,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    if !start.is_aligned_to(nixe_memory::PAGE_SIZE) {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    if size == 0 || !size.is_multiple_of(nixe_memory::PAGE_SIZE) {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    let Some(end) = start.get().checked_add(size) else {
        result(context, HorizonKernelResult::INVALID_MEMORY_REGION);
        return resume();
    };
    if context.process().total_system_resource_size() == 0 {
        result(context, HorizonKernelResult::INVALID_STATE);
        return resume();
    }
    let alias = context.process().memory_layout().alias();
    if start.get() < alias.base().get()
        || end > alias.base().get() + alias.size()
        || end > context.process().memory().space().width().bytes()
    {
        result(context, HorizonKernelResult::INVALID_MEMORY_REGION);
        return resume();
    }
    let process = context.process().clone();
    let memory = process.memory();
    let completed = if unmap {
        super::access::wait_io(context, memory.unmap_physical(start, size))
    } else {
        super::access::wait_io(context, memory.map_physical(start, size))
    };
    mapping_result(context, completed)
}
// ABI: https://switchbrew.org/wiki/SVC (MapRemoteMemory / UnmapRemoteMemory).
pub(super) fn remote_memory(
    context: &mut ExceptionDispatchContext<'_>,
    unmap: bool,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    // @SPECULATIVE: resolve a normal process handle or current-process
    // pseudohandle. Validate ranges and report the same error classes as
    // MapPhysicalMemory; remote-specific access policy is not documented.
    let process = if handle == CURRENT_PROCESS_HANDLE {
        context.process().clone()
    } else {
        match context
            .process()
            .handles()
            .lookup_as::<nixe_runtime::RunnableProcess>(handle)
        {
            Some(process) => process,
            None => {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            }
        }
    };
    if !start.is_aligned_to(nixe_memory::PAGE_SIZE) {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    if size == 0 || !size.is_multiple_of(nixe_memory::PAGE_SIZE) {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if process.memory_layout().remote().is_none_or(|region| {
        start.get() < region.base().get()
            || start
                .get()
                .checked_add(size)
                .is_none_or(|end| end > region.base().get() + region.size())
    }) {
        result(context, HorizonKernelResult::INVALID_MEMORY_REGION);
        return resume();
    }
    let completed = if unmap {
        super::access::wait_io(context, process.unmap_remote_memory(start, size))
    } else {
        super::access::wait_io(context, process.map_remote_memory(start, size))
    };
    mapping_result(context, completed)
}

pub(super) fn set_memory_permission(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    let raw = read_register(context.thread().state(), 2) as u32;
    let permissions = match raw {
        0 => MemoryPermissions::NONE,
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        _ => return reject(context, HorizonSvcFault::InvalidMemoryPermission { raw }),
    };
    let outcome = context.process().memory().protect(start, size, permissions);
    mapping_result(context, outcome)
}
pub(super) fn map_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let (destination, source, size) = match memory_alias_arguments(context) {
        Ok(args) => args,
        Err(code) => {
            result(context, code);
            return resume();
        }
    };
    let before = match context.process().memory().query(source) {
        Ok(info) if info.base.get() + info.bytes >= source.get() + size => info.mapping,
        _ => None,
    };
    let Some(before) = before.filter(|before| {
        matches!(
            before.kind,
            MemoryKind::Heap | MemoryKind::MutableCode | MemoryKind::MutableModuleCode
        ) && before.permissions == MemoryPermissions::READ_WRITE
            && !before.locked
            && !before.uncached
            && !before.permission_locked
    }) else {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        return resume();
    };
    let after = Mapping {
        permissions: MemoryPermissions::NONE,
        locked: true,
        ..before
    };
    let outcome = context.process().memory().map_alias(
        source,
        destination,
        size,
        before,
        after,
        mapping(MemoryKind::Stack, MemoryPermissions::READ_WRITE),
    );
    mapping_result(context, outcome)
}
pub(super) fn unmap_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let (destination, source, size) = match memory_alias_arguments(context) {
        Ok(args) => args,
        Err(code) => {
            result(context, code);
            return resume();
        }
    };
    let before = match context.process().memory().query(source) {
        Ok(info) if info.base.get() + info.bytes >= source.get() + size => info.mapping,
        _ => None,
    };
    let Some(before) = before.filter(|before| {
        matches!(
            before.kind,
            MemoryKind::Heap | MemoryKind::MutableCode | MemoryKind::MutableModuleCode
        ) && before.permissions == MemoryPermissions::NONE
            && before.locked
            && !before.uncached
            && !before.permission_locked
    }) else {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        return resume();
    };
    let restored = Mapping {
        permissions: MemoryPermissions::READ_WRITE,
        locked: false,
        ..before
    };
    let outcome = context.process().memory().unmap_alias(
        source,
        destination,
        size,
        before,
        restored,
        mapping(MemoryKind::Stack, MemoryPermissions::READ_WRITE),
    );
    mapping_result(context, outcome)
}
fn memory_alias_arguments(
    context: &ExceptionDispatchContext<'_>,
) -> Result<(GuestVirtualAddress, GuestVirtualAddress, u64), HorizonKernelResult> {
    let destination = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let source = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    // Public SVC ABI validation and register order:
    // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_memory.cpp#L72-L128
    if !destination.is_aligned_to(USER_BUFFER_ALIGNMENT)
        || !source.is_aligned_to(USER_BUFFER_ALIGNMENT)
    {
        return Err(HorizonKernelResult::INVALID_ADDRESS);
    }
    if size == 0 || !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
        return Err(HorizonKernelResult::INVALID_SIZE);
    }
    let Some(source_end) = source.get().checked_add(size) else {
        return Err(HorizonKernelResult::INVALID_CURRENT_MEMORY);
    };
    let Some(destination_end) = destination.get().checked_add(size) else {
        return Err(HorizonKernelResult::INVALID_CURRENT_MEMORY);
    };
    if source_end > context.process().address_space_limit() {
        return Err(HorizonKernelResult::INVALID_CURRENT_MEMORY);
    }
    let stack = context.process().memory_layout().stack();
    let stack_end = stack
        .base()
        .get()
        .checked_add(stack.size())
        .expect("validated process stack region does not overflow");
    if destination.get() < stack.base().get() || destination_end > stack_end {
        return Err(HorizonKernelResult::INVALID_MEMORY_REGION);
    }
    Ok((destination, source, size))
}

pub(super) fn map_shared_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let raw = read_register(context.thread().state(), 3) as u32;
    let permissions = match raw {
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        _ => return reject(context, HorizonSvcFault::InvalidMemoryPermission { raw }),
    };
    let Some(shared) = context
        .process()
        .handles()
        .lookup_as::<SharedMemoryObject>(handle)
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if size == 0 || !size.is_multiple_of(USER_BUFFER_ALIGNMENT) || size != shared.size() as u64 {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !start.is_aligned_to(USER_BUFFER_ALIGNMENT)
        || start
            .get()
            .checked_add(size)
            .is_none_or(|end| end > context.process().address_space_limit())
    {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    if !shared.remote_permissions().contains(permissions) {
        result(context, HorizonKernelResult::INVALID_STATE);
        return resume();
    }
    let outcome = context.process().memory().map(
        start,
        shared.allocation().clone(),
        0,
        size,
        mapping(MemoryKind::Shared, permissions),
    );
    mapping_result(context, outcome)
}
pub(super) fn unmap_shared_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let Some(shared) = context
        .process()
        .handles()
        .lookup_as::<SharedMemoryObject>(handle)
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if size == 0 || !size.is_multiple_of(USER_BUFFER_ALIGNMENT) || size != shared.size() as u64 {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !start.is_aligned_to(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    let outcome = context.process().memory().query(start).and_then(|info| {
        let expected = info
            .mapping
            .filter(|mapping| mapping.kind == MemoryKind::Shared)
            .ok_or(MappingError::StateMismatch)?;
        context
            .process()
            .memory()
            .unmap_shared(start, shared.allocation(), size, expected)
    });
    mapping_result(context, outcome)
}
pub(super) fn create_transfer_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let raw = read_register(context.thread().state(), 3) as u32;
    let permissions = match raw {
        0 => MemoryPermissions::NONE,
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        _ => return reject(context, HorizonSvcFault::InvalidMemoryPermission { raw }),
    };
    if size == 0 || !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !start.is_aligned_to(USER_BUFFER_ALIGNMENT)
        || start
            .get()
            .checked_add(size)
            .is_none_or(|end| end > context.process().address_space_limit())
    {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    let transfer = match TransferMemoryObject::new(
        Arc::clone(context.process().memory()),
        start,
        size,
        permissions,
    ) {
        Ok(transfer) => transfer,
        Err(fault) => {
            return reject(
                context,
                HorizonSvcFault::MemoryMapping {
                    fault: Arc::new(fault),
                },
            );
        }
    };
    match context.process().handles().insert(transfer) {
        Ok(handle) => {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
        }
        Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
    }
    resume()
}
pub(super) fn set_memory_attribute(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    let mask = read_register(context.thread().state(), 2) as u32;
    let value = read_register(context.thread().state(), 3) as u32;
    let outcome = match (mask, value) {
        (8, 0 | 8) => context
            .process()
            .memory()
            .set_uncached(start, size, value != 0),
        (16, 16) => context.process().memory().lock_permissions(start, size),
        _ => {
            return reject(
                context,
                HorizonSvcFault::InvalidMemoryAttribute { mask, value },
            );
        }
    };
    mapping_result(context, outcome)
}

/// https://switchbrew.org/wiki/SVC#SetGpuMemoryAttribute
pub(super) fn set_gpu_memory_attribute(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    let mask = read_register(context.thread().state(), 2) as u32;
    let value = read_register(context.thread().state(), 3) as u32;
    if mask & !0x20 != 0 || value & !mask != 0 {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryAttribute { mask, value },
        );
    }
    let outcome = context.process().memory().set_gpu_sharable(
        start,
        size,
        (mask & 0x20 != 0).then_some(value & 0x20 != 0),
    );
    mapping_result(context, outcome)
}
