//! System shared objects use the same physical allocator as process memory.
use nixe_memory::{
    Access, AccessResult, AllocationRange, MemoryAccount, MemoryError, MemoryPermissions,
};
use nixe_runtime::SharedMemoryObject;
use std::sync::Arc;

pub(crate) fn create(
    account: &MemoryAccount,
    bytes: usize,
    initial: &[u8],
) -> Result<SharedMemoryObject, Arc<MemoryError>> {
    let allocation = account.allocate(bytes as u64).map_err(Arc::new)?;
    if !initial.is_empty() {
        let len = initial.len() as u64;
        let range = AllocationRange::new(allocation.pin_range(0, len).map_err(Arc::new)?, 0, len)
            .map_err(Arc::new)?;
        // The allocation is still private: no device or guest can own it.
        account
            .machine()
            .write_resolved(&[range], initial)
            .map_err(Arc::new)?;
    }
    SharedMemoryObject::new(allocation, bytes, MemoryPermissions::READ)
        .map_err(|error| Arc::new(MemoryError::Execution(error.to_string().into())))
}

pub(crate) async fn write(
    memory: &SharedMemoryObject,
    offset: usize,
    bytes: &[u8],
) -> Result<(), Arc<MemoryError>> {
    match memory.write(offset, bytes).map_err(Arc::new)? {
        Access::Ready(()) => Ok(()),
        Access::Pending(pending) => match pending.await? {
            AccessResult::Written(_) => Ok(()),
            _ => Err(Arc::new(MemoryError::Execution(
                "invalid system-memory write completion".into(),
            ))),
        },
    }
}

pub(crate) async fn read(
    memory: &SharedMemoryObject,
    offset: usize,
    bytes: usize,
) -> Result<Box<[u8]>, Arc<MemoryError>> {
    match memory.read(offset, bytes).map_err(Arc::new)? {
        Access::Ready(bytes) => Ok(bytes),
        Access::Pending(pending) => match pending.await? {
            AccessResult::Read(data) if data.len() == bytes => Ok(data.into_boxed_slice()),
            _ => Err(Arc::new(MemoryError::Execution(
                "invalid system-memory read completion".into(),
            ))),
        },
    }
}

pub(crate) async fn read_transfer_into(
    memory: &nixe_runtime::TransferMemoryObject,
    offset: u64,
    output: &mut Vec<u8>,
) -> Result<(), Arc<MemoryError>> {
    let storage = std::mem::take(output);
    let storage = match memory.ranges()[0]
        .allocation()
        .machine()
        .read_captured_owned(memory.ranges(), offset, storage)
        .map_err(Arc::new)?
    {
        Access::Ready(storage) => storage,
        Access::Pending(pending) => match pending.await? {
            AccessResult::Read(storage) => storage,
            _ => {
                return Err(Arc::new(MemoryError::Execution(
                    "invalid transfer-memory read completion".into(),
                )));
            }
        },
    };
    *output = storage;
    Ok(())
}

pub(crate) async fn write_transfer(
    memory: &nixe_runtime::TransferMemoryObject,
    offset: u64,
    bytes: &mut Vec<u8>,
) -> Result<(), Arc<MemoryError>> {
    let storage = std::mem::take(bytes);
    let storage = match memory.ranges()[0]
        .allocation()
        .machine()
        .write_captured_owned(memory.ranges(), offset, storage)
        .map_err(Arc::new)?
    {
        Access::Ready(storage) => storage,
        Access::Pending(pending) => match pending.await? {
            AccessResult::Written(storage) => storage,
            _ => {
                return Err(Arc::new(MemoryError::Execution(
                    "invalid transfer-memory write completion".into(),
                )));
            }
        },
    };
    *bytes = storage;
    Ok(())
}
