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
    command: Buffer,
    buffers: Vec<Buffer>,
    offsets: [usize; 6],
    copied: Vec<Option<HandleObject>>,
    returned: Vec<HandleReservation>,
    lifetime: Arc<super::operation::CallLifetime>,
    payload_budget: Arc<super::payload::PayloadBudget>,
    payload: std::sync::OnceLock<Arc<super::payload::PayloadPermit>>,
}
impl Request {
    pub fn new(
        context: &ExceptionDispatchContext<'_>,
        address: GuestVirtualAddress,
        bytes: usize,
        payload_budget: Arc<super::payload::PayloadBudget>,
    ) -> Result<Self, IpcWireError> {
        let process = context.process();
        let command = Self::capture(
            process,
            address.get(),
            bytes as u64,
            MemoryPermissions::READ_WRITE,
        )?;
        Ok(Self {
            command,
            offsets: [0; 6],
            buffers: Vec::new(),
            copied: Vec::new(),
            returned: Vec::new(),
            lifetime: Arc::new(super::operation::CallLifetime::for_client(context)),
            payload_budget,
            payload: std::sync::OnceLock::new(),
        })
    }
    pub async fn reserve_payload(&self, size: usize) -> Result<(), IpcWireError> {
        if size == 0 {
            return Ok(());
        }
        if self.payload.get().is_some() {
            return Err(IpcWireError::Internal(
                "IPC payload must be admitted once before effects",
            ));
        }
        let permit = self.payload_budget.reserve(size, &self.lifetime).await?;
        self.payload
            .set(Arc::new(permit))
            .map_err(|_| IpcWireError::Internal("duplicate IPC payload admission"))
    }
    fn check_lifetime(&self) -> Result<(), IpcWireError> {
        if self.lifetime.client().generation() != self.lifetime.client().process().generation() {
            return Err(IpcWireError::Cancelled);
        }
        self.lifetime.check()
    }
    pub fn lifetime(&self) -> &Arc<super::operation::CallLifetime> {
        &self.lifetime
    }
    pub fn begin_service(&self) -> Result<(), IpcWireError> {
        self.check_lifetime()?;
        self.lifetime.begin()
    }
    pub fn kernel(&self) -> &Arc<nixe_scheduler::SchedulerState> {
        self.lifetime.client().process().scheduler()
    }

    pub fn thread_id(&self) -> u64 {
        self.lifetime.client().thread().thread_id()
    }
    pub fn process_id(&self) -> u64 {
        self.lifetime.client().process().process_id()
    }
    pub fn cpu(&self) -> nixe_cpu::profile::ProcessCpuContext {
        self.lifetime.client().process().cpu()
    }
    pub fn memory(&self) -> &Arc<nixe_runtime::memory::ProcessMemory> {
        self.lifetime.client().process().memory()
    }
    pub fn mounts(&self) -> &ProcessMountNamespace {
        self.lifetime.client().process().mounts()
    }
    /// Nested service data pointers are accepted when their descriptor is
    /// consumed, before any asynchronous work reads the pointed-to bytes.
    pub fn capture_input(&self, address: u64, bytes: u64) -> Result<Buffer, IpcWireError> {
        Self::capture(
            self.lifetime.client().process(),
            address,
            bytes,
            MemoryPermissions::READ,
        )
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
    pub async fn read_command(&self, mut bytes: Vec<u8>) -> Result<Vec<u8>, IpcWireError> {
        let capacity = self.command_size();
        let prefix = capacity.min(COMMAND_BUFFER_SIZE);
        bytes.clear();
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
            self.read_inline(&self.command, prefix as u64, &mut bytes[prefix..data_end])
                .await?;
        }
        let receive_start = receive.start.max(prefix).max(data_end);
        if receive_start < receive.end {
            self.read_inline(
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
            self.lifetime.client().process(),
            self.lifetime.client().thread(),
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
                self.lifetime.client().process(),
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
                    self.lifetime.client().process(),
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
                    self.lifetime.client().process(),
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
        let reservation = self
            .lifetime
            .client()
            .process()
            .handles()
            .reserve(Some(object))?;
        let handle = reservation.handle();
        self.returned.push(reservation);
        Ok(handle)
    }
    pub fn return_service(
        &mut self,
        object: crate::HorizonIpcObject,
    ) -> Result<u32, nixe_runtime::HandleError> {
        self.return_handle(super::ServiceSession::new(object))
    }
    pub fn return_handle<T: HandleValue>(
        &mut self,
        object: T,
    ) -> Result<u32, nixe_runtime::HandleError> {
        self.return_object(HandleObject::new(object))
    }
    pub fn publish_handles(&mut self) -> Result<(), IpcWireError> {
        self.check_lifetime()?;
        self.lifetime.commit_reply()?;
        self.lifetime
            .client()
            .process()
            .handles()
            .publish(&mut self.returned)
            .map_err(|error| {
                if self.lifetime.client().generation()
                    != self.lifetime.client().process().generation()
                {
                    IpcWireError::Cancelled
                } else {
                    IpcWireError::Handle(error)
                }
            })
    }
    pub async fn write_response(&self, mut bytes: Vec<u8>) -> Result<(), IpcWireError> {
        let _reply = nixe_memory::trace::span("ipc_reply", self.thread_id(), bytes.len() as u64, 0);
        if bytes.len() > self.command_size() {
            return Err(IpcWireError::Internal(
                "encoded response exceeds its accepted command buffer",
            ));
        }
        self.write(&self.command, 0, &mut bytes).await
    }
    async fn read_inline(
        &self,
        buffer: &Buffer,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), IpcWireError> {
        let mut bytes = vec![0; output.len()];
        self.read(buffer, offset, &mut bytes).await?;
        output.copy_from_slice(&bytes);
        Ok(())
    }
    fn retain_payload(&self, pending: &nixe_memory::PendingAccess) {
        if let Some(permit) = self.payload.get() {
            pending.retain_owner(permit.clone());
        }
    }
    pub async fn read(
        &self,
        buffer: &Buffer,
        offset: u64,
        output: &mut Vec<u8>,
    ) -> Result<(), IpcWireError> {
        if output.is_empty() {
            return Ok(());
        }
        let bytes = output.len() as u64;
        nixe_memory::trace::event("ipc_input_copy", self.thread_id(), bytes, buffer.address);
        let address = GuestVirtualAddress::new(buffer.address)
            .checked_add(offset)
            .ok_or(IpcWireError::Malformed("IPC input offset overflows"))?;
        let storage = std::mem::take(output);
        let storage = match self
            .lifetime
            .client()
            .process()
            .memory()
            .account()
            .machine()
            .read_captured_owned(&buffer.ranges, offset, storage)
            .map_err(|error| Self::memory_error(address, bytes, Arc::new(error)))?
        {
            Access::Ready(bytes) => bytes,
            Access::Pending(pending) => {
                self.retain_payload(&pending);
                match pending
                    .await
                    .map_err(|error| Self::memory_error(address, bytes, error))?
                {
                    AccessResult::Read(data) => data,
                    _ => return Err(IpcWireError::Internal("invalid owned IPC read completion")),
                }
            }
        };
        *output = storage;
        Ok(())
    }
    pub async fn write(
        &self,
        buffer: &Buffer,
        offset: u64,
        bytes: &mut Vec<u8>,
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
        let storage = std::mem::take(bytes);
        let result = match self
            .lifetime
            .client()
            .process()
            .memory()
            .account()
            .machine()
            .write_captured_owned(&buffer.ranges, offset, storage)
        {
            Err(error) => Err(Self::memory_error(address, count, Arc::new(error))),
            Ok(Access::Ready(storage)) => Ok(storage),
            Ok(Access::Pending(pending)) => {
                self.retain_payload(&pending);
                match pending.await {
                    Ok(AccessResult::Written(storage)) => Ok(storage),
                    Ok(_) => Err(IpcWireError::Internal("invalid owned IPC write completion")),
                    Err(error) => Err(Self::memory_error(address, count, error)),
                }
            }
        };
        self.lifetime.end_output();
        *bytes = result?;
        Ok(())
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
