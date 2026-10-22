//! A kernel word keeps its accepted physical identity across asynchronous access.
use super::*;
use nixe_memory::{
    Access, AccessOperation, AccessOrdering, AccessResult, AllocationRange, AtomicAccess,
    AtomicOperation, MemoryError,
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(super) struct AddressWord {
    range: nixe_memory::CapturedMemory,
    permissions: MemoryPermissions,
}
impl AddressWord {
    fn permission_fault(&self, required: MemoryPermissions) -> MemoryError {
        MemoryError::PermissionDenied {
            address: self.range[0]
                .mapping()
                .map_or(GuestVirtualAddress::MIN, |mapping| mapping.start),
            required,
        }
    }
    pub(super) fn from_range(range: AllocationRange) -> Self {
        debug_assert_eq!(range.bytes(), 4);
        Self {
            range: range.into(),
            permissions: MemoryPermissions::READ_WRITE,
        }
    }
    pub(super) fn range(&self) -> AllocationRange {
        self.range[0].clone()
    }
    pub(super) async fn read_owned(&self) -> Result<u32, Arc<MemoryError>> {
        match self.read().map_err(Arc::new)? {
            Access::Ready(value) => Ok(value),
            Access::Pending(pending) => match pending.await? {
                AccessResult::Read(bytes) if bytes.len() == 4 => {
                    Ok(u32::from_le_bytes(bytes[..].try_into().unwrap()))
                }
                _ => Err(Arc::new(MemoryError::Execution(
                    "kernel word read completion kind".into(),
                ))),
            },
        }
    }
    pub(super) async fn compare_exchange_owned(
        &self,
        expected: u32,
        replacement: u32,
    ) -> Result<(u32, bool), Arc<MemoryError>> {
        let result = match self
            .compare_exchange(expected, replacement)
            .map_err(Arc::new)?
        {
            Access::Ready(result) => result,
            Access::Pending(pending) => pending.await?,
        };
        match result {
            AccessResult::Atomic { previous, written } => Ok((previous as u32, written)),
            _ => Err(Arc::new(MemoryError::Execution(
                "kernel word CAS completion kind".into(),
            ))),
        }
    }
    pub(super) fn capture(
        context: &ExceptionDispatchContext<'_>,
        address: u64,
        permissions: MemoryPermissions,
    ) -> Result<Self, MemoryError> {
        let (range, permissions) = context.process().memory().space().capture_scalar(
            GuestVirtualAddress::new(address),
            4,
            permissions,
        )?;
        Ok(Self {
            range: range.into(),
            permissions,
        })
    }
    pub(super) fn read(&self) -> Result<Access<u32>, MemoryError> {
        if !self.permissions.contains(MemoryPermissions::READ) {
            return Err(self.permission_fault(MemoryPermissions::READ));
        }
        let mut bytes = [0; 4];
        match self.range[0]
            .allocation()
            .machine()
            .read_captured_into(&self.range, 0, &mut bytes)?
        {
            Access::Ready(()) => Ok(Access::Ready(u32::from_le_bytes(bytes))),
            Access::Pending(pending) => Ok(Access::Pending(pending)),
        }
    }
    pub(super) fn compare_exchange(
        &self,
        expected: u32,
        replacement: u32,
    ) -> Result<Access<AccessResult>, MemoryError> {
        if !self.permissions.contains(MemoryPermissions::READ_WRITE) {
            return Err(self.permission_fault(MemoryPermissions::READ_WRITE));
        }
        self.range[0].allocation().machine().access_captured(
            std::slice::from_ref(&self.range[0]),
            AccessOperation::Atomic(AtomicAccess {
                bytes: 4,
                ordering: AccessOrdering::AcquireRelease,
                operation: AtomicOperation::CompareExchange {
                    expected: u128::from(expected),
                    replacement: u128::from(replacement),
                },
            }),
        )
    }
    pub(super) async fn write(&self, value: u32) -> Result<(), Arc<MemoryError>> {
        if !self.permissions.contains(MemoryPermissions::WRITE) {
            return Err(Arc::new(self.permission_fault(MemoryPermissions::WRITE)));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        match self.range[0]
            .allocation()
            .machine()
            .write_captured(&self.range, 0, &value.to_le_bytes())
            .map_err(Arc::new)?
        {
            Access::Ready(()) => Ok(()),
            Access::Pending(pending) => match pending.await? {
                AccessResult::Written => Ok(()),
                _ => Err(Arc::new(MemoryError::Execution(
                    "kernel word write completion kind".into(),
                ))),
            },
        }
    }
}

pub(super) fn address_word_fault(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    fault: Arc<MemoryError>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    if fault.is_access_fault() {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        resume()
    } else {
        ExceptionDispatchOutcome::Fault(HorizonSvcFault::MemoryAccess { immediate, fault })
    }
}
