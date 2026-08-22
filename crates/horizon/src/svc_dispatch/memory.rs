use super::*;

pub(super) fn get_info(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let info_type = read_register(context.thread().state(), 1) as u32;
    let handle = read_register(context.thread().state(), 2) as u32;
    let subtype = read_register(context.thread().state(), 3);
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
    if handle != CURRENT_PROCESS_HANDLE {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    }
    let layout = context.process().memory_layout();
    let value = match info_type {
        2 => layout.alias().base().get(),
        3 => layout.alias().size(),
        4 => layout.heap().base().get(),
        5 => layout.heap().size(),
        12 => layout.aslr().base().get(),
        13 => layout.aslr().size(),
        14 => layout.stack().base().get(),
        15 => layout.stack().size(),
        6 => layout.memory_capacity(),
        7 => context.process().used_memory_size(),
        28 => 0,
        _ => {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::UnsupportedSemantics {
                immediate: 0x29,
                documented_name: "GetInfo",
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

pub(super) fn break_process(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let reason = read_register(context.thread().state(), 0);
    let info = read_register(context.thread().state(), 1);
    let size = read_register(context.thread().state(), 2);
    if reason & 0x8000_0000 != 0 {
        result(context, HorizonKernelResult::SUCCESS);
        return resume();
    }
    let payload = usize::try_from(size)
        .ok()
        .filter(|size| (1..=nixe_runtime::MAX_GUEST_BREAK_PAYLOAD_BYTES).contains(size))
        .and_then(|size| {
            let mut bytes = vec![0; size];
            match crate::ipc_wire::read_bytes(
                context.process(),
                GuestVirtualAddress::new(info),
                &mut bytes,
            ) {
                Ok(()) => nixe_runtime::GuestBreakPayload::new(&bytes),
                Err(error) => {
                    log::debug!(
                        "svcBreak payload could not be captured: info={info:#x} size={size:#x} error={error:?}"
                    );
                    None
                }
            }
        });
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
    let end = start.get().checked_add(size);
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    let valid_range = query.is_some_and(|query| {
        query.purpose.allows_reprotect()
            && query.base.get() <= start.get()
            && end.is_some_and(|end| query.base.get().saturating_add(query.size) >= end)
    });
    if !valid_range {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryState {
                immediate: 0x02,
                address: start,
                purpose: query.map_or(MemoryMappingPurpose::Normal, |query| query.purpose),
            },
        );
    }
    match context
        .process()
        .set_memory_permissions(start, size, permissions)
    {
        Ok(()) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryProtection { fault }),
    }
}

pub(super) fn map_shared_memory(
    context: &mut ExceptionDispatchContext<'_>,
    hid_system: &mut crate::HidSystem,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let raw_permissions = read_register(context.thread().state(), 3) as u32;
    let permissions = match raw_permissions {
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        _ => {
            return reject(
                context,
                HorizonSvcFault::InvalidMemoryPermission {
                    raw: raw_permissions,
                },
            );
        }
    };
    let Some(shared_memory) = context
        .process()
        .handles()
        .get_as::<SharedMemoryObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    // Public ABI validation and register order:
    // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#MapSharedMemory
    if size == 0
        || !size.is_multiple_of(USER_BUFFER_ALIGNMENT)
        || usize::try_from(size).ok() != Some(shared_memory.size())
    {
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
    if !shared_memory.remote_permissions().contains(permissions) {
        result(context, HorizonKernelResult::INVALID_STATE);
        return resume();
    }
    let mapping_permissions = if permissions == MemoryPermissions::READ {
        MemoryPermissions::READ_WRITE
    } else {
        permissions
    };
    match context.process().resize_memory_mapping(
        start,
        0,
        size,
        mapping_permissions,
        MemoryMappingPurpose::SharedMemory,
    ) {
        Ok(()) => {
            let mut backing = vec![0_u8; shared_memory.size()];
            if shared_memory.read(0, &mut backing).is_err() {
                let _ = context.process().resize_memory_mapping(
                    start,
                    size,
                    0,
                    mapping_permissions,
                    MemoryMappingPurpose::SharedMemory,
                );
                return reject(
                    context,
                    HorizonSvcFault::InternalRuntime {
                        operation: "reading a shared-memory backing at its declared size",
                    },
                );
            }
            for (offset, byte) in backing
                .into_iter()
                .enumerate()
                .filter(|(_, byte)| *byte != 0)
            {
                let Some(address) = start.checked_add(offset as u64) else {
                    unreachable!("validated shared-memory range contains every backing byte")
                };
                if let Err(fault) = context.process().memory().write(
                    context.process().cpu().address_space_id(),
                    address,
                    MemoryAccess::normal(MemoryAccessSize::Byte),
                    MemoryValue::U8(byte),
                ) {
                    let _ = context.process().resize_memory_mapping(
                        start,
                        size,
                        0,
                        mapping_permissions,
                        MemoryMappingPurpose::SharedMemory,
                    );
                    return reject(
                        context,
                        HorizonSvcFault::GuestMemory {
                            immediate: 0x13,
                            fault,
                        },
                    );
                }
            }
            if permissions != mapping_permissions
                && let Err(fault) =
                    context
                        .process()
                        .set_memory_permissions(start, size, permissions)
            {
                let _ = context.process().resize_memory_mapping(
                    start,
                    size,
                    0,
                    mapping_permissions,
                    MemoryMappingPurpose::SharedMemory,
                );
                return reject(context, HorizonSvcFault::MemoryProtection { fault });
            }
            log::debug!(
                "mapped temporary shared memory handle {handle:#x} at {start} ({size:#x} bytes)"
            );
            if hid_system.owns(&shared_memory) {
                hid_system.register_mapping(context.process().cpu().address_space_id(), start);
            }
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryMapping { fault }),
    }
}

pub(super) fn create_transfer_memory(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    // Public ABI and validation reference:
    // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#CreateTransferMemory
    let address = read_register(context.thread().state(), 1);
    let size = read_register(context.thread().state(), 2);
    let raw_permissions = read_register(context.thread().state(), 3) as u32;
    let permissions = match raw_permissions {
        0 => MemoryPermissions::NONE,
        1 => MemoryPermissions::READ,
        3 => MemoryPermissions::READ_WRITE,
        raw => return reject(context, HorizonSvcFault::InvalidMemoryPermission { raw }),
    };
    if size == 0 || !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !address.is_multiple_of(USER_BUFFER_ALIGNMENT)
        || address
            .checked_add(size)
            .is_none_or(|end| end > context.process().address_space_limit())
    {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    let start = GuestVirtualAddress::new(address);
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    if !query.is_some_and(|query| {
        query.base.get() <= address
            && query
                .base
                .get()
                .checked_add(query.size)
                .is_some_and(|end| end >= address.saturating_add(size))
    }) {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        return resume();
    }
    let backing = match context
        .process()
        .canonical_memory()
        .translate_canonical_range(
            context.process().cpu().address_space_id(),
            start,
            size,
            MemoryPermissions::NONE,
        ) {
        Ok(backing) => backing,
        Err(fault) => {
            return reject(
                context,
                HorizonSvcFault::CanonicalMemory {
                    immediate: 0x15,
                    fault,
                },
            );
        }
    };
    match context
        .process_mut()
        .handles_mut()
        .insert(TransferMemoryObject::new(start, size, permissions, backing))
    {
        Ok(handle) => {
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, u64::from(handle));
        }
        Err(_) => result(context, HorizonKernelResult::OUT_OF_HANDLES),
    }
    resume()
}

pub(super) fn unmap_shared_memory(
    context: &mut ExceptionDispatchContext<'_>,
    hid_system: &mut crate::HidSystem,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
    let size = read_register(context.thread().state(), 2);
    let Some(shared_memory) = context
        .process()
        .handles()
        .get_as::<SharedMemoryObject>(handle)
        .cloned()
    else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    if size == 0
        || !size.is_multiple_of(USER_BUFFER_ALIGNMENT)
        || usize::try_from(size).ok() != Some(shared_memory.size())
    {
        result(context, HorizonKernelResult::INVALID_SIZE);
        return resume();
    }
    if !start.is_aligned_to(USER_BUFFER_ALIGNMENT) {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    }
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    let Some(query) = query.filter(|mapping| {
        mapping.base == start
            && mapping.size == size
            && mapping.purpose == MemoryMappingPurpose::SharedMemory
    }) else {
        result(context, HorizonKernelResult::INVALID_ADDRESS);
        return resume();
    };
    match context.process().resize_memory_mapping(
        start,
        size,
        0,
        query.permissions,
        MemoryMappingPurpose::SharedMemory,
    ) {
        Ok(()) => {
            if hid_system.owns(&shared_memory) {
                hid_system.unregister_mapping(context.process().cpu().address_space_id(), start);
            }
            log::debug!(
                "unmapped temporary shared memory handle {handle:#x} from {start} ({size:#x} bytes)"
            );
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryMapping { fault }),
    }
}

pub(super) fn set_memory_attribute(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let start = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
    let size = read_register(context.thread().state(), 1);
    let raw_mask = read_register(context.thread().state(), 2) as u32;
    let raw_value = read_register(context.thread().state(), 3) as u32;
    let uncached = MemoryAttributes::UNCACHED.bits();
    let permission_locked = MemoryAttributes::PERMISSION_LOCKED.bits();
    let valid_update = (raw_mask == uncached && raw_value & !uncached == 0)
        || (raw_mask == permission_locked && raw_value == permission_locked);
    if !valid_update {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryAttribute {
                mask: raw_mask,
                value: raw_value,
            },
        );
    }
    let (Some(mask), Some(value)) = (
        MemoryAttributes::from_bits(raw_mask),
        MemoryAttributes::from_bits(raw_value),
    ) else {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryAttribute {
                mask: raw_mask,
                value: raw_value,
            },
        );
    };
    let end = start.get().checked_add(size);
    let query = context.process().memory().query_memory(
        context.process().cpu().address_space_id(),
        start,
        GuestVirtualAddress::new(context.process().address_space_limit()),
    );
    let valid_range = query.is_some_and(|query| {
        query.purpose.allows_attribute_change()
            && query.base.get() <= start.get()
            && end.is_some_and(|end| query.base.get().saturating_add(query.size) >= end)
    });
    if !valid_range {
        return reject(
            context,
            HorizonSvcFault::InvalidMemoryState {
                immediate: 0x03,
                address: start,
                purpose: query.map_or(MemoryMappingPurpose::Normal, |query| query.purpose),
            },
        );
    }
    match context
        .process()
        .set_memory_attributes(start, size, mask, value)
    {
        Ok(()) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => reject(context, HorizonSvcFault::MemoryProtection { fault }),
    }
}
