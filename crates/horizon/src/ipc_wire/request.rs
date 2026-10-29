//! Accepted IPC resources. Only this request's pinned descriptors supply wire
//! buffers; suspending a service never keeps a CPU context or a handle lock.
use super::{
    IpcWireError,
    message::{BufferMode, COMMAND_BUFFER_SIZE, HipcHeader, HipcRequest, ReceiveStatics},
};
use nixe_memory::{Access, AccessResult, GuestVirtualAddress, MemoryPermissions};
use nixe_runtime::{
    ExceptionDispatchContext, HandleObject, HandleReservation, HandleValue, ProcessMountNamespace,
};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct Buffer {
    pub address: u64,
    pub size: u64,
    ranges: nixe_memory::CapturedMemory,
}
pub(crate) struct Request {
    thread: Arc<nixe_runtime::ThreadObject>,
    target: HandleObject,
    process: Arc<nixe_runtime::RunnableProcess>,
    process_generation: u64,
    command: Buffer,
    buffers: Vec<Buffer>,
    offsets: [usize; 6],
    copied: Vec<Option<HandleObject>>,
    returned: Vec<HandleReservation>,
    lifetime: Arc<super::operation::CallLifetime>,
}
impl Request {
    pub fn new(
        context: &ExceptionDispatchContext<'_>,
        target: HandleObject,
        address: GuestVirtualAddress,
        bytes: usize,
    ) -> Result<Self, IpcWireError> {
        let process = context.process();
        let process_generation = process.generation();
        let command = Self::capture(
            process,
            address.get(),
            bytes as u64,
            MemoryPermissions::READ_WRITE,
        )?;
        Ok(Self {
            thread: context.thread().object().clone(),
            target,
            process: process.clone(),
            process_generation,
            command,
            offsets: [0; 6],
            buffers: Vec::new(),
            copied: Vec::new(),
            returned: Vec::new(),
            lifetime: Arc::default(),
        })
    }
    fn check_lifetime(&self) -> Result<(), IpcWireError> {
        if self.process_generation != self.process.generation() {
            return Err(IpcWireError::Cancelled);
        }
        self.lifetime.check()
    }
    pub fn lifetime(&self) -> Arc<super::operation::CallLifetime> {
        self.lifetime.clone()
    }
    pub fn begin_service(&self) -> Result<(), IpcWireError> {
        self.check_lifetime()?;
        self.lifetime.begin()
    }
    pub fn target(&self) -> &HandleObject {
        &self.target
    }
    pub fn kernel(&self) -> &Arc<nixe_scheduler::SchedulerState> {
        self.process.scheduler()
    }

    pub fn process_id(&self) -> u64 {
        self.process.process_id()
    }
    pub fn cpu(&self) -> nixe_cpu::profile::ProcessCpuContext {
        self.process.cpu()
    }
    pub fn memory(&self) -> &Arc<nixe_runtime::memory::ProcessMemory> {
        self.process.memory()
    }
    pub fn mounts(&self) -> &ProcessMountNamespace {
        self.process.mounts()
    }
    /// Nested service data pointers are accepted when their descriptor is
    /// consumed, before any asynchronous work reads the pointed-to bytes.
    pub fn capture_input(&self, address: u64, bytes: u64) -> Result<Buffer, IpcWireError> {
        Self::capture(&self.process, address, bytes, MemoryPermissions::READ)
    }
    pub fn command_size(&self) -> usize {
        self.command.size as usize
    }
    pub fn send_statics(&self) -> &[Buffer] {
        &self.buffers[self.offsets[0]..self.offsets[1]]
    }
    pub fn send_buffers(&self) -> &[Buffer] {
        &self.buffers[self.offsets[1]..self.offsets[2]]
    }
    pub fn receive_buffers(&self) -> &[Buffer] {
        &self.buffers[self.offsets[2]..self.offsets[3]]
    }
    pub fn receive_statics(&self) -> &[Buffer] {
        &self.buffers[self.offsets[4]..self.offsets[5]]
    }
    /// Keep the normal HLE command prefix, then capture only declared data and
    /// receive entries outside it. Every read uses the accepted physical input.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/hle_ipc.cpp#L255-L267
    pub async fn read_command(&self) -> Result<Vec<u8>, IpcWireError> {
        let capacity = self.command_size();
        let prefix = capacity.min(COMMAND_BUFFER_SIZE);
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(prefix).map_err(|_| {
            IpcWireError::HostResourceExhausted("allocating the IPC command buffer")
        })?;
        bytes.resize(prefix, 0);
        self.read(&self.command, 0, &mut bytes).await?;
        if prefix == capacity {
            return Ok(bytes);
        }
        let header =
            HipcHeader::decode(&bytes).map_err(|error| IpcWireError::Malformed(error.0))?;
        let inline_end = header
            .message_size(capacity)
            .map_err(|error| IpcWireError::Malformed(error.0))?;
        let receive = header
            .receive_list(inline_end, capacity)
            .map_err(|error| IpcWireError::Malformed(error.0))?;
        let size = inline_end.max(receive.end);
        if size <= prefix {
            bytes.truncate(size);
            return Ok(bytes);
        }
        bytes.try_reserve_exact(size - prefix).map_err(|_| {
            IpcWireError::HostResourceExhausted("allocating the IPC command buffer")
        })?;
        bytes.resize(size, 0);
        let data_end = if receive.start <= inline_end {
            size
        } else {
            inline_end
        };
        if data_end > prefix {
            self.read(&self.command, prefix as u64, &mut bytes[prefix..data_end])
                .await?;
        }
        let receive_start = receive.start.max(prefix).max(data_end);
        if receive_start < receive.end {
            self.read(
                &self.command,
                receive_start as u64,
                &mut bytes[receive_start..receive.end],
            )
            .await?;
        }
        Ok(bytes)
    }

    fn capture(
        process: &nixe_runtime::RunnableProcess,
        address: u64,
        bytes: u64,
        permissions: MemoryPermissions,
    ) -> Result<Buffer, IpcWireError> {
        let start = GuestVirtualAddress::new(address);
        let ranges = if bytes == 0 {
            Vec::new()
        } else {
            process
                .memory()
                .space()
                .capture(start, bytes, permissions)
                .map_err(|error| Self::memory_error(start, bytes, Arc::new(error)))?
        };
        Ok(Buffer {
            address,
            size: bytes,
            ranges: nixe_memory::CapturedMemory::new(ranges)
                .map_err(|error| Self::memory_error(start, bytes, Arc::new(error)))?,
        })
    }
    pub fn accept_descriptors(&mut self, hipc: &HipcRequest<'_>) -> Result<(), IpcWireError> {
        self.copied = super::handles::capture(
            &self.process,
            &self.thread,
            hipc.copy_handles.iter().copied(),
            hipc.move_handles.iter().copied(),
            false,
        )
        .map_err(IpcWireError::Kernel)?
        .copied;
        let count = hipc.send_statics.len()
            + hipc.send_buffers.len()
            + hipc.receive_buffers.len()
            + hipc.exchange_buffers.len()
            + match &hipc.receive_statics {
                ReceiveStatics::Entries(v) => v.len(),
                _ => 0,
            };
        self.buffers
            .try_reserve_exact(count)
            .map_err(|_| IpcWireError::HostResourceExhausted("accepting IPC descriptors"))?;
        // Bind by descriptor family/index, as HLERequestContext's A/X/B/C access.
        // Command and nested input captures never participate in descriptor selection.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/hle_ipc.cpp#L314-L448
        for descriptor in &hipc.send_statics {
            self.buffers.push(Self::capture(
                &self.process,
                descriptor.address,
                u64::from(descriptor.size),
                MemoryPermissions::READ,
            )?);
        }
        self.offsets[1] = self.buffers.len();
        for (group, (descriptors, permission)) in [
            (&hipc.send_buffers, MemoryPermissions::READ),
            (&hipc.receive_buffers, MemoryPermissions::WRITE),
            (&hipc.exchange_buffers, MemoryPermissions::READ_WRITE),
        ]
        .into_iter()
        .enumerate()
        {
            for descriptor in descriptors {
                if descriptor.mode == BufferMode::Invalid {
                    return Err(IpcWireError::Malformed("invalid IPC buffer mode"));
                }
                self.buffers.push(Self::capture(
                    &self.process,
                    descriptor.address,
                    descriptor.size,
                    permission,
                )?);
            }
            self.offsets[group + 2] = self.buffers.len();
        }
        if let ReceiveStatics::Entries(descriptors) = &hipc.receive_statics {
            for descriptor in descriptors {
                self.buffers.push(Self::capture(
                    &self.process,
                    descriptor.address,
                    u64::from(descriptor.size),
                    MemoryPermissions::WRITE,
                )?);
            }
        }
        self.offsets[5] = self.buffers.len();
        Ok(())
    }
    pub fn copy_handle<T: std::any::Any + Send + Sync>(&self, index: usize) -> Option<Arc<T>> {
        self.copied.get(index)?.as_ref()?.downcast()
    }
    pub fn return_object(
        &mut self,
        object: HandleObject,
    ) -> Result<u32, nixe_runtime::HandleError> {
        let reservation = self.process.handles().reserve(Some(object))?;
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
        self.check_lifetime()?;
        self.process
            .handles()
            .publish(&mut self.returned)
            .map_err(|_| IpcWireError::Internal("reply handle publication was cancelled"))
    }
    pub async fn write_response(&self, bytes: &[u8]) -> Result<(), IpcWireError> {
        if bytes.len() > self.command_size() {
            return Err(IpcWireError::Internal(
                "encoded response exceeds its accepted command buffer",
            ));
        }
        self.write(&self.command, 0, bytes).await
    }
    pub async fn read(
        &self,
        buffer: &Buffer,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), IpcWireError> {
        if output.is_empty() {
            return Ok(());
        }
        let bytes = output.len() as u64;
        let address = GuestVirtualAddress::new(buffer.address)
            .checked_add(offset)
            .ok_or(IpcWireError::Malformed("IPC input offset overflows"))?;
        match self
            .process
            .memory()
            .account()
            .machine()
            .read_captured_into(&buffer.ranges, offset, output)
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
        buffer: &Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), IpcWireError> {
        self.check_lifetime()?;
        if bytes.is_empty() {
            return Ok(());
        }
        let count = bytes.len() as u64;
        let address = GuestVirtualAddress::new(buffer.address)
            .checked_add(offset)
            .ok_or(IpcWireError::Malformed("IPC output offset overflows"))?;
        self.lifetime.begin_output()?;
        let result = match self.process.memory().account().machine().write_captured(
            &buffer.ranges,
            offset,
            bytes,
        ) {
            Err(error) => Err(Self::memory_error(address, count, Arc::new(error))),
            Ok(Access::Ready(())) => Ok(()),
            Ok(Access::Pending(pending)) => match pending.await {
                Ok(AccessResult::Written) => Ok(()),
                Ok(_) => Err(IpcWireError::Internal("invalid owned IPC write completion")),
                Err(error) => Err(Self::memory_error(address, count, error)),
            },
        };
        self.lifetime.end_output();
        result
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
