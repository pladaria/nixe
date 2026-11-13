//! GPU context and ordered memory jobs. Events become ready after the mutation.
use super::*;
use crate::gsv::{
    GsvInterface,
    gpu::{GpuAddressSpace, VA_END, VA_START},
};
use std::sync::Arc;

pub(super) fn dispatch(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed("unexpected GPU job descriptors"));
    }
    let mut state = session.state.lock().unwrap();
    let gpu = state
        .gpu
        .as_mut()
        .ok_or(IpcWireError::Malformed("GPU session missing"))?;
    let data = request.data;
    let mut output = Vec::new();
    let mut handles = Vec::new();
    let mut signal = None;
    match (session.interface, request.command_id, data.len()) {
        (GsvInterface::Gpu, 0, 8) => output.extend_from_slice(data),
        (GsvInterface::Gpu, 11, 0) => {
            output.extend_from_slice(&0_u32.to_le_bytes());
            output.extend_from_slice(&31_u32.to_le_bytes());
        }
        (GsvInterface::Gpu, 12, 4) => {
            let id = request_u32(data, 0).unwrap() as usize;
            let event = gpu
                .events
                .get_mut(id)
                .ok_or(IpcWireError::Malformed("GPU event out of range"))?
                .get_or_insert_with(|| nixe_runtime::EventObject::new(process.kernel()));
            handles.push(
                process
                    .return_handle(event.readable())
                    .map_err(IpcWireError::Handle)?,
            );
        }
        (GsvInterface::Gpu, 21, 4) => {
            let id = request_u32(data, 0).unwrap() as usize;
            signal = Some(
                gpu.events
                    .get(id)
                    .and_then(Option::as_ref)
                    .ok_or(IpcWireError::Malformed("GPU worker event not created"))?
                    .clone(),
            );
            gpu.worker_ready = true;
        }
        (GsvInterface::Gpu, 22, 0) if gpu.worker_ready => {
            handles.push(
                process
                    .return_service(HorizonIpcObject::Gsv(session.child(GsvInterface::Worker)))
                    .map_err(IpcWireError::Handle)?,
            );
        }
        (GsvInterface::Gpu, 41, 24) => {
            // @SPECULATIVE: the observed device name and zero flags describe
            // NvRmGpu. The third word is an optional opaque debug label.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            if request_u64(data, 0) != Some(u64::from_le_bytes(*b"NvRmGpu\0"))
                || request_u64(data, 8) != Some(0)
            {
                return Err(IpcWireError::Malformed(
                    "unsupported graphics device name/flags",
                ));
            }
            gpu.next_device = gpu
                .next_device
                .checked_add(1)
                .ok_or(IpcWireError::HostResourceExhausted("graphics device IDs"))?;
            gpu.devices.insert(gpu.next_device);
            output.extend_from_slice(&gpu.next_device.to_le_bytes());
        }
        (GsvInterface::Gpu, 42, 4) => {
            if !gpu.devices.remove(&request_u32(data, 0).unwrap()) {
                return Err(IpcWireError::Malformed("unknown graphics device"));
            }
        }
        // @SPECULATIVE: AS creation and partition-query layouts are based on
        // the reference SDK observations; partition bounds are inclusive here.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
        (GsvInterface::Gpu, 33, 4) => {
            if request_u32(data, 0) != Some(0) {
                return Err(IpcWireError::Malformed(
                    "unsupported GPU address-space flags",
                ));
            }
            gpu.next_space = gpu
                .next_space
                .checked_add(1)
                .ok_or(IpcWireError::HostResourceExhausted("GPU address space IDs"))?;
            let id = gpu.next_space;
            gpu.spaces.insert(
                id,
                GpuAddressSpace::new(process.memory().space().machine().clone()),
            );
            output.extend_from_slice(&id.to_le_bytes());
        }
        (GsvInterface::Gpu, 201, 4) | (GsvInterface::Gpu, 202, 4) => {
            let id = request_u32(data, 0).unwrap();
            if !gpu.spaces.contains_key(&id) {
                return Err(IpcWireError::Malformed("unknown GPU address space"));
            }
            let (first, last) = if request.command_id == 201 {
                (VA_START, VA_END - nixe_memory::PAGE_SIZE - 1)
            } else {
                (u64::MAX, 0)
            };
            output.extend_from_slice(&first.to_le_bytes());
            output.extend_from_slice(&last.to_le_bytes());
        }
        (GsvInterface::Gpu, 203, 4) => {
            // @SPECULATIVE: legacy SDK allocator description is
            // {kind=1, start, size}, as observed in the reference.
            if !gpu.spaces.contains_key(&request_u32(data, 0).unwrap()) {
                return Err(IpcWireError::Malformed("unknown GPU address space"));
            }
            for value in [1_u64, VA_START, VA_END - nixe_memory::PAGE_SIZE - VA_START] {
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        (GsvInterface::Worker, 101, 24) => {
            let start = request_u64(data, 0).unwrap();
            let end = request_u64(data, 8).unwrap();
            // TIPC serializes each u8 argument in its own padded word.
            // @SPECULATIVE: only the observed registration policy (0, 1)
            // is supported. Its bits are not sufficiently documented to
            // infer additional allocation or cache policies.
            if request_u32(data, 16) != Some(0) || request_u32(data, 20) != Some(1) {
                return Err(IpcWireError::Malformed("unsupported GPU memory flags"));
            }
            let bytes = end
                .checked_sub(start)
                .and_then(|size| size.checked_add(1))
                .ok_or(IpcWireError::Malformed("invalid GPU memory range"))?;
            let memory = process
                .memory()
                .share_gpu(GuestVirtualAddress::new(start), bytes)
                .map_err(|error| {
                    IpcWireError::GraphicsService(error.to_string().into_boxed_str())
                })?;
            gpu.next_memory = gpu
                .next_memory
                .checked_add(1)
                .ok_or(IpcWireError::HostResourceExhausted("GPU memory IDs"))?;
            let id = gpu.next_memory;
            gpu.memories.insert(id, Arc::new(memory));
            output.extend_from_slice(&id.to_le_bytes());
        }
        // @SPECULATIVE: packed job fields follow SDK observations in the reference.
        (GsvInterface::Worker, 211, 64) => {
            let space = request_u32(data, 0).unwrap();
            let first = request_u64(data, 4).unwrap();
            let last = request_u64(data, 12).unwrap();
            let handle = request_u32(data, 20).unwrap();
            let offset = request_u64(data, 24).unwrap();
            let event = request_u32(data, 60).unwrap() as usize;
            signal = Some(
                gpu.events
                    .get(event)
                    .and_then(Option::as_ref)
                    .ok_or(IpcWireError::Malformed("GPU map event not created"))?
                    .clone(),
            );
            let memory = gpu
                .memories
                .get(&handle)
                .ok_or(IpcWireError::Malformed("unknown GPU memory handle"))?
                .clone();
            let bytes = last
                .checked_sub(first)
                .and_then(|size| size.checked_add(1))
                .ok_or(IpcWireError::Malformed("GPU map range overflow"))?;
            gpu.spaces
                .get_mut(&space)
                .ok_or(IpcWireError::Malformed("unknown GPU address space"))?
                .map(first, bytes, handle, memory, offset)
                .map_err(|error| IpcWireError::SystemMemory(Arc::new(error)))?;
        }
        (GsvInterface::Worker, 102, 8) => {
            let handle = request_u32(data, 0).unwrap();
            let event = request_u32(data, 4).unwrap() as usize;
            signal = Some(
                gpu.events
                    .get(event)
                    .and_then(Option::as_ref)
                    .ok_or(IpcWireError::Malformed("GPU release event not created"))?
                    .clone(),
            );
            if !gpu.memories.contains_key(&handle) {
                return Err(IpcWireError::Malformed("unknown GPU memory handle"));
            }
            for space in gpu.spaces.values_mut() {
                let ranges: Vec<_> = space
                    .mappings
                    .iter()
                    .filter(|(_, mapping)| mapping.memory_handle == handle)
                    .map(|(&start, mapping)| (start, mapping.end - start))
                    .collect();
                for (start, bytes) in ranges {
                    space
                        .unmap(start, bytes)
                        .map_err(|error| IpcWireError::SystemMemory(Arc::new(error)))?;
                }
            }
            gpu.memories.remove(&handle);
        }
        _ => {
            return Err(IpcWireError::UnsupportedService(
                UnsupportedServiceOperation::Command {
                    service: if matches!(session.interface, GsvInterface::Worker) {
                        "GSV KUserWorkerSession"
                    } else {
                        "GSV GpuSession"
                    },
                    command_id: request.command_id,
                },
            ));
        }
    }
    drop(state);
    if let Some(event) = signal {
        event.signal();
    }
    Ok(encode_tipc_response(0, &output, &handles, &[])?)
}
