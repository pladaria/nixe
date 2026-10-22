//! Accepted IPC resources. Only this request's pinned descriptors supply wire
//! buffers; suspending a service never keeps a CPU context or a handle lock.
use super::{
    IpcWireError,
    message::{BufferMode, HipcRequest, ReceiveStatics},
};
use nixe_memory::{Access, AccessResult, GuestVirtualAddress, MemoryPermissions};
use nixe_runtime::{
    ExceptionDispatchContext, HandleObject, HandleReservation, HandleTable, HandleValue,
    ProcessMountNamespace,
};
use std::sync::Arc;

struct Buffer {
    address: GuestVirtualAddress,
    bytes: u64,
    permissions: MemoryPermissions,
    ranges: nixe_memory::CapturedMemory,
}
pub(crate) struct Request {
    thread: nixe_runtime::ThreadObject,
    target: HandleObject,
    process_id: u64,
    cpu: nixe_cpu::profile::ProcessCpuContext,
    memory: Arc<nixe_runtime::memory::ProcessMemory>,
    handles: Arc<HandleTable>,
    mounts: Arc<ProcessMountNamespace>,
    buffers: Vec<Buffer>,
    copied: Vec<HandleObject>,
    moved: Vec<Option<HandleObject>>,
    returned: Vec<HandleReservation>,
    lifetime: Arc<super::operation::CallLifetime>,
}
impl Request {
    pub fn new(context: &ExceptionDispatchContext<'_>, target: HandleObject) -> Self {
        let process = context.process();
        Self {
            thread: context.thread().object(),
            target,
            process_id: process.process_id(),
            cpu: process.cpu(),
            memory: Arc::clone(process.memory()),
            handles: Arc::clone(process.handles()),
            mounts: Arc::clone(process.mounts()),
            buffers: Vec::new(),
            copied: Vec::new(),
            moved: Vec::new(),
            returned: Vec::new(),
            lifetime: Arc::default(),
        }
    }
    pub fn lifetime(&self) -> Arc<super::operation::CallLifetime> {
        self.lifetime.clone()
    }
    pub fn begin_service(&self) -> Result<(), IpcWireError> {
        self.lifetime.begin()
    }
    pub fn target(&self) -> &HandleObject {
        &self.target
    }
    pub fn process_id(&self) -> u64 {
        self.process_id
    }
    pub fn cpu(&self) -> nixe_cpu::profile::ProcessCpuContext {
        self.cpu
    }
    pub fn memory(&self) -> &Arc<nixe_runtime::memory::ProcessMemory> {
        &self.memory
    }
    pub fn handles(&self) -> &Arc<HandleTable> {
        &self.handles
    }
    pub fn mounts(&self) -> &ProcessMountNamespace {
        &self.mounts
    }
    /// Nested service data pointers are accepted when their descriptor is
    /// consumed, before any asynchronous work reads the pointed-to bytes.
    pub fn accept_input(&mut self, address: u64, bytes: u64) -> Result<(), IpcWireError> {
        self.pin_buffer(address, bytes, MemoryPermissions::READ)
    }
    pub fn pin_command(
        &mut self,
        start: GuestVirtualAddress,
        bytes: usize,
    ) -> Result<(), IpcWireError> {
        self.pin_buffer(start.get(), bytes as u64, MemoryPermissions::READ_WRITE)
    }
    fn pin_buffer(
        &mut self,
        address: u64,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), IpcWireError> {
        if bytes == 0 {
            return Ok(());
        }
        let address = GuestVirtualAddress::new(address);
        let ranges = self
            .memory
            .space()
            .capture(address, bytes, permissions)
            .map_err(|error| Self::memory_error(address, bytes, Arc::new(error)))?;
        self.buffers.push(Buffer {
            address,
            bytes,
            permissions,
            ranges: nixe_memory::CapturedMemory::new(ranges)
                .map_err(|error| Self::memory_error(address, bytes, Arc::new(error)))?,
        });
        Ok(())
    }
    pub fn accept_descriptors(&mut self, hipc: &HipcRequest<'_>) -> Result<(), IpcWireError> {
        for descriptor in &hipc.send_statics {
            self.pin_buffer(
                descriptor.address,
                u64::from(descriptor.size),
                MemoryPermissions::READ,
            )?;
        }
        for (descriptors, permission) in [
            (&hipc.send_buffers, MemoryPermissions::READ),
            (&hipc.receive_buffers, MemoryPermissions::WRITE),
            (&hipc.exchange_buffers, MemoryPermissions::READ_WRITE),
        ] {
            for descriptor in descriptors {
                if descriptor.mode == BufferMode::Invalid {
                    return Err(IpcWireError::Malformed("invalid IPC buffer mode"));
                }
                self.pin_buffer(descriptor.address, descriptor.size, permission)?;
            }
        }
        if let ReceiveStatics::Entries(descriptors) = &hipc.receive_statics {
            for descriptor in descriptors {
                self.pin_buffer(
                    descriptor.address,
                    u64::from(descriptor.size),
                    MemoryPermissions::WRITE,
                )?;
            }
        }
        let real: Vec<_> = hipc
            .copy_handles
            .iter()
            .copied()
            .filter(|handle| {
                !matches!(
                    *handle,
                    crate::CURRENT_THREAD_HANDLE | crate::CURRENT_PROCESS_HANDLE
                )
            })
            .collect();
        let (copied, moved) = self
            .handles
            .capture_transfer(&real, &hipc.move_handles)
            .map_err(IpcWireError::Handle)?;
        let mut copied = copied.into_iter();
        self.copied = hipc
            .copy_handles
            .iter()
            .map(|&handle| match handle {
                crate::CURRENT_PROCESS_HANDLE => {
                    HandleObject::new(nixe_runtime::ProcessObject::new(self.process_id))
                }
                crate::CURRENT_THREAD_HANDLE => HandleObject::new(self.thread.clone()),
                _ => copied
                    .next()
                    .expect("real copy handles were captured in wire order"),
            })
            .collect();
        self.moved = moved.into_iter().map(Some).collect();
        Ok(())
    }
    pub fn copy_handle<T: std::any::Any + Send + Sync>(&self, index: usize) -> Option<Arc<T>> {
        self.copied.get(index)?.downcast()
    }
    pub fn return_object(
        &mut self,
        object: HandleObject,
    ) -> Result<u32, nixe_runtime::HandleError> {
        let reservation = self.handles.reserve_object(object)?;
        let handle = reservation.handle();
        self.returned.push(reservation);
        Ok(handle)
    }
    pub fn return_handle<T: HandleValue>(
        &mut self,
        object: T,
    ) -> Result<u32, nixe_runtime::HandleError> {
        self.return_object(HandleObject::new(object))
    }
    pub fn publish_handles(&mut self) -> Result<(), IpcWireError> {
        self.lifetime.check()?;
        self.handles
            .publish(&mut self.returned)
            .map_err(|_| IpcWireError::Internal("reply handle publication was cancelled"))
    }
    fn buffer(
        &self,
        address: GuestVirtualAddress,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<&Buffer, IpcWireError> {
        let end = address
            .get()
            .checked_add(bytes)
            .ok_or(IpcWireError::Malformed("IPC buffer range overflows"))?;
        self.buffers
            .iter()
            .find(|buffer| {
                buffer.permissions.contains(permissions)
                    && address.get() >= buffer.address.get()
                    && end <= buffer.address.get() + buffer.bytes
            })
            .ok_or(IpcWireError::Malformed(
                "access is outside accepted IPC descriptors",
            ))
    }
    pub fn validate_write(
        &self,
        address: GuestVirtualAddress,
        bytes: usize,
    ) -> Result<(), IpcWireError> {
        if bytes == 0 {
            return Ok(());
        }
        self.buffer(address, bytes as u64, MemoryPermissions::WRITE)
            .map(|_| ())
    }
    pub async fn read(
        &self,
        address: GuestVirtualAddress,
        output: &mut [u8],
    ) -> Result<(), IpcWireError> {
        if output.is_empty() {
            return Ok(());
        }
        let bytes = output.len() as u64;
        let buffer = self.buffer(address, bytes, MemoryPermissions::READ)?;
        match self
            .memory
            .account()
            .machine()
            .read_captured_into(&buffer.ranges, address.get() - buffer.address.get(), output)
            .map_err(|error| Self::memory_error(address, bytes, Arc::new(error)))?
        {
            Access::Ready(()) => Ok(()),
            Access::Pending(pending) => match pending
                .await
                .map_err(|error| Self::memory_error(address, bytes, error))?
            {
                AccessResult::Read(data) if data.len() == output.len() => {
                    output.copy_from_slice(&data);
                    Ok(())
                }
                _ => Err(IpcWireError::Internal("invalid owned IPC read completion")),
            },
        }
    }
    pub async fn write(
        &self,
        address: GuestVirtualAddress,
        bytes: &[u8],
    ) -> Result<(), IpcWireError> {
        self.lifetime.check()?;
        if bytes.is_empty() {
            return Ok(());
        }
        let count = bytes.len() as u64;
        let buffer = self.buffer(address, count, MemoryPermissions::WRITE)?;
        match self
            .memory
            .account()
            .machine()
            .write_captured(&buffer.ranges, address.get() - buffer.address.get(), bytes)
            .map_err(|error| Self::memory_error(address, count, Arc::new(error)))?
        {
            Access::Ready(()) => Ok(()),
            Access::Pending(pending) => match pending
                .await
                .map_err(|error| Self::memory_error(address, count, error))?
            {
                AccessResult::Written => Ok(()),
                _ => Err(IpcWireError::Internal("invalid owned IPC write completion")),
            },
        }
    }
    fn memory_error(
        address: GuestVirtualAddress,
        bytes: u64,
        fault: Arc<nixe_memory::MemoryError>,
    ) -> IpcWireError {
        IpcWireError::GuestAccess {
            address,
            bytes,
            fault,
        }
    }
}
