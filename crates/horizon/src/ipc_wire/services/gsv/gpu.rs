//! GPU context and ordered memory jobs. Events become ready after the mutation.
use super::*;
use crate::gsv::{
    GsvInterface,
    gpu::{GpuAddressSpace, VA_END, VA_START},
};
use std::sync::Arc;

pub(super) async fn dispatch(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if matches!(session.interface, GsvInterface::Worker) && request.command_id == 216 {
        return super::sparse::dispatch(process, session, request, hipc).await;
    }
    if has_ipc_descriptors(hipc) {
        return Err(IpcWireError::Malformed("unexpected GPU job descriptors"));
    }
    let (completion, signal, output, handles) = {
        let mut state = session.state.lock().unwrap();
        let remote_memory = state.process.as_ref().map(|p| p.remote_memory().clone());
        let gpu = state
            .gpu
            .as_mut()
            .ok_or(IpcWireError::Malformed("GPU session missing"))?;
        let data = request.data;
        let mut output = Vec::new();
        let mut handles = Vec::new();
        let mut signal = None;
        let mut memory_job = None;
        match (&session.interface, request.command_id, data.len()) {
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
                    Arc::new(std::sync::Mutex::new(GpuAddressSpace::new(
                        gpu.memory.clone(),
                    ))),
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
                if data[16] != 0 || data[20] != 1 {
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
            (GsvInterface::Worker, command @ (214 | 215), size)
                if size == if command == 214 { 40 } else { 28 } =>
            {
                // @SPECULATIVE: inclusive VA range, options and completion
                // event follow the reference's native SDK observations. Only
                // server-backed regions (flags 0xc) are accepted here; sparse
                // tile mapping has its own ownership contract.
                // https://switchbrew.org/wiki/Switch_2%3A_GSV_services#KUserWorkerSession
                // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
                let create = command == 214;
                if (create
                    && (request_u32(data, 20) != Some(0xc)
                        || data[24] != 1
                        || data[28] != 1
                        || data[32] != 1))
                    || (!create && data[20] != 1)
                {
                    return Err(IpcWireError::GraphicsService(
                        format!("unsupported native GPU region options: command={command} data={data:02x?}").into()));
                }
                let id = request_u32(data, 0).unwrap();
                let address = request_u64(data, 4).unwrap();
                let bytes = request_u64(data, 12)
                    .unwrap()
                    .checked_sub(address)
                    .and_then(|n| n.checked_add(1))
                    .ok_or(IpcWireError::Malformed("native GPU region overflow"))?;
                let event = request_u32(data, if create { 36 } else { 24 }).unwrap() as usize;
                signal = Some(
                    gpu.events
                        .get(event)
                        .and_then(Option::as_ref)
                        .ok_or(IpcWireError::Malformed(
                            "native GPU region event not created",
                        ))?
                        .clone(),
                );
                memory_job = Some(crate::gsv::execution::MemoryJob::NativeRegion {
                    space: gpu
                        .spaces
                        .get(&id)
                        .cloned()
                        .ok_or(IpcWireError::Malformed("unknown GPU address space"))?,
                    address,
                    bytes,
                    account: if create {
                        Some(
                            remote_memory
                                .clone()
                                .ok_or(IpcWireError::Malformed("GSV process missing"))?,
                        )
                    } else {
                        None
                    },
                });
                log::debug!(
                    "GSV native GPU region: create={create} address={address:#x} bytes={bytes:#x}"
                );
            }
            (GsvInterface::Worker, 211, 64) => {
                // @SPECULATIVE: native SDK flags and kind bytes follow the
                // observed job layout. Byte arguments occupy padded words;
                // their padding is uninitialized and is not an argument.
                if request_u32(data, 32) != Some(0)
                    || data[36] > 1
                    || request_u32(data, 48) != Some(0)
                    || data[52] != 1
                    || data[56] != 1
                {
                    return Err(IpcWireError::Malformed("unsupported GPU map options"));
                }
                let kind = crate::gsv::gpu::memory_kind(data[40], data[44])
                    .map_err(IpcWireError::GraphicsService)?;
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
                memory_job = Some(crate::gsv::execution::MemoryJob::Map {
                    space: gpu
                        .spaces
                        .get(&space)
                        .cloned()
                        .ok_or(IpcWireError::Malformed("unknown GPU address space"))?,
                    address: first,
                    bytes,
                    handle,
                    source: memory,
                    offset,
                    kind,
                });
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
                let source = gpu
                    .memories
                    .remove(&handle)
                    .ok_or(IpcWireError::Malformed("unknown GPU memory handle"))?;
                memory_job = Some(crate::gsv::execution::MemoryJob::Retire {
                    spaces: gpu.spaces.values().cloned().collect(),
                    handle,
                    _source: source,
                });
            }
            (GsvInterface::Worker, 213, 28) => {
                // NormalUnmapGpuMemory4K retires the VA mapping in command order.
                // Existing captures retain physical storage until their users finish.
                // https://switchbrew.org/wiki/Switch_2%3A_GSV_services#KUserWorkerSession
                // @SPECULATIVE: the inclusive range, padded policy byte and
                // event index follow the reference's SDK request observations.
                // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h#L594-L609
                if data[20] != 1 {
                    return Err(IpcWireError::GraphicsService(
                        format!("unsupported native unmap policy: data={data:02x?}").into(),
                    ));
                }
                let id = request_u32(data, 0).unwrap();
                let address = request_u64(data, 4).unwrap();
                let bytes = request_u64(data, 12)
                    .unwrap()
                    .checked_sub(address)
                    .and_then(|size| size.checked_add(1))
                    .ok_or(IpcWireError::Malformed("native GPU unmap range overflow"))?;
                let event = request_u32(data, 24).unwrap() as usize;
                signal = Some(
                    gpu.events
                        .get(event)
                        .and_then(Option::as_ref)
                        .ok_or(IpcWireError::Malformed("GPU unmap event not created"))?
                        .clone(),
                );
                memory_job = Some(crate::gsv::execution::MemoryJob::Unmap {
                    space: gpu
                        .spaces
                        .get(&id)
                        .cloned()
                        .ok_or(IpcWireError::Malformed("unknown GPU address space"))?,
                    address,
                    bytes,
                });
            }
            (GsvInterface::Worker, 212, 36) => {
                // @SPECULATIVE: native ModifyMemoryKind4K fields follow SDK
                // observations in the reference. Only its synchronous policy
                // and the verified pitch/generic kind families are consumed.
                // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
                // PTE kinds: https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/src/common/inc/swref/published/turing/tu102/dev_mmu.h
                let id = request_u32(data, 0).unwrap();
                let address = request_u64(data, 4).unwrap();
                let last = request_u64(data, 12).unwrap();
                let kind = crate::gsv::gpu::memory_kind(data[20], data[24])
                    .map_err(IpcWireError::GraphicsService)?;
                if data[28] != 1 {
                    return Err(IpcWireError::Malformed(
                        "unsupported GPU kind change policy",
                    ));
                }
                let event = request_u32(data, 32).unwrap() as usize;
                signal = Some(
                    gpu.events
                        .get(event)
                        .and_then(Option::as_ref)
                        .ok_or(IpcWireError::Malformed("GPU kind event not created"))?
                        .clone(),
                );
                let bytes = last
                    .checked_sub(address)
                    .and_then(|n| n.checked_add(1))
                    .ok_or(IpcWireError::Malformed("GPU kind range overflow"))?;
                memory_job = Some(crate::gsv::execution::MemoryJob::Kind {
                    space: gpu
                        .spaces
                        .get(&id)
                        .cloned()
                        .ok_or(IpcWireError::Malformed("unknown GPU address space"))?,
                    address,
                    bytes,
                    kind,
                });
                output.extend_from_slice(&u32::from(kind).to_le_bytes());
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
        // Admission remains under the service lock shared with submissions, but
        // asynchronous completion never retains that lock.
        let completion = if let Some(job) = memory_job {
            if let Some(executor) = &gpu.executor {
                Some(
                    executor
                        .memory(job)
                        .map_err(IpcWireError::GraphicsService)?,
                )
            } else {
                job.apply(|_, _, _, _| Ok(()))
                    .map_err(IpcWireError::GraphicsService)?;
                None
            }
        } else {
            None
        };
        (completion, signal, output, handles)
    };
    if let Some(completion) = completion {
        completion
            .wait()
            .await
            .map_err(|_| IpcWireError::Cancelled)?;
    }
    if let Some(event) = signal {
        event.signal();
    }
    Ok(encode_tipc_response(0, &output, &handles, &[])?)
}
