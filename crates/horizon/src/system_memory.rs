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
            AccessResult::Written => Ok(()),
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
            AccessResult::Read(data) if data.len() == bytes => Ok(data),
            _ => Err(Arc::new(MemoryError::Execution(
                "invalid system-memory read completion".into(),
            ))),
        },
    }
}

pub(crate) async fn read_transfer(
    memory: &nixe_runtime::TransferMemoryObject,
    offset: u64,
    bytes: usize,
) -> Result<Vec<u8>, Arc<MemoryError>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    if offset
        .checked_add(bytes as u64)
        .is_none_or(|end| end > memory.size())
    {
        return Err(Arc::new(MemoryError::Invalid(
            "transfer read exceeds its accepted memory",
        )));
    }
    let machine = memory.ranges()[0].allocation().machine();
    let mut output = vec![0; bytes];
    match machine
        .read_captured_into(memory.ranges(), offset, &mut output)
        .map_err(Arc::new)?
    {
        Access::Ready(()) => {}
        Access::Pending(pending) => match pending.await? {
            AccessResult::Read(bytes) if bytes.len() == output.len() => {
                output.copy_from_slice(&bytes)
            }
            _ => {
                return Err(Arc::new(MemoryError::Execution(
                    "invalid transfer-memory read completion".into(),
                )));
            }
        },
    }
    Ok(output)
}

pub(crate) async fn write_transfer(
    memory: &nixe_runtime::TransferMemoryObject,
    offset: u64,
    bytes: &[u8],
) -> Result<(), Arc<MemoryError>> {
    if bytes.is_empty() {
        return Ok(());
    }
    let machine = memory.ranges()[0].allocation().machine();
    match machine
        .write_captured(memory.ranges(), offset, bytes)
        .map_err(Arc::new)?
    {
        Access::Ready(()) => Ok(()),
        Access::Pending(pending) => match pending.await? {
            AccessResult::Written => Ok(()),
            _ => Err(Arc::new(MemoryError::Execution(
                "invalid transfer-memory write completion".into(),
            ))),
        },
    }
}
