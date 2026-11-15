//! Native batch Opus IPC. Shared memory, ring ordering and decoders belong to hwacod.
use super::*;
use crate::hwacod::{self, AudioCodecSession, Config, Layout, Manager, Task};
use crate::object::DomainSession;
use std::{sync::Arc, time::Instant};

pub(in crate::ipc_wire) async fn dispatch_audio_codec(
    process: &mut Request,
    session: &AudioCodecSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let target = match &request.domain {
        Some(DomainRequest::Close { object_id }) => {
            return opus_error(
                &request,
                if session.domain.close_object(*object_id) {
                    HorizonIpcResult::SUCCESS
                } else {
                    HorizonIpcResult::CMIF_TARGET_NOT_FOUND
                },
            );
        }
        Some(DomainRequest::SendMessage {
            object_id,
            input_objects,
        }) => {
            if !input_objects.is_empty() {
                return opus_error(&request, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            if *object_id == 1 {
                Some(session.root.clone())
            } else {
                session.domain.object(*object_id)
            }
        }
        None if session.domain.is_domain() => {
            return Err(IpcWireError::Malformed(
                "domain audio codec request omitted its domain header",
            ));
        }
        None => Some(session.root.clone()),
    };
    let Some(target) = target else {
        return opus_error(&request, HorizonIpcResult::CMIF_TARGET_NOT_FOUND);
    };
    dispatch_object(process, &session.domain, &target, request, hipc).await
}

pub(super) fn child(
    process: &mut Request,
    domain: &DomainSession<OpusObject>,
    request: &CmifRequest<'_>,
    object: OpusObject,
    data: &[u8],
    copies: &[u32],
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if domain.is_domain() {
        let Some(id) = domain.insert_object(object) else {
            return opus_error(request, HorizonIpcResult::CMIF_OUT_OF_DOMAIN_ENTRIES);
        };
        semantic_success(request.token, true, data, copies, &[id], None)
    } else {
        let handle = process
            .return_service(HorizonIpcObject::AudioCodec(AudioCodecSession::new(object)))
            .map_err(IpcWireError::Handle)?;
        semantic_success(request.token, false, data, copies, &[], Some(handle))
    }
}

#[derive(Clone, Copy)]
enum Input {
    None,
    Config,
    Tasks,
}
fn shape(
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    scalar: usize,
    handles: usize,
    input: Input,
) -> bool {
    request.has_payload_size(scalar)
        && hipc.pid.is_none()
        && hipc.copy_handles.len() == handles
        && hipc.move_handles.is_empty()
        && hipc.receive_buffers.is_empty()
        && hipc.exchange_buffers.is_empty()
        && matches!(hipc.receive_statics, ReceiveStatics::None)
        && match input {
            Input::None => hipc.send_statics.is_empty() && hipc.send_buffers.is_empty(),
            Input::Config => {
                hipc.send_buffers.is_empty()
                    && hipc.send_statics.len() == 1
                    && hipc.send_statics[0].size == 0x120
            }
            Input::Tasks => {
                hipc.send_statics.len() <= 1
                    && hipc.send_buffers.len() <= 1
                    && (!hipc.send_statics.is_empty() || !hipc.send_buffers.is_empty())
            }
        }
}
async fn config(process: &Request) -> Result<Result<Config, HorizonIpcResult>, IpcWireError> {
    process.reserve_payload(0x120).await?;
    let mut bytes = vec![0; 0x120];
    process
        .read(&process.send_statics()[0], 0, &mut bytes)
        .await?;
    Ok(Config::parse(&bytes))
}

pub(super) async fn dispatch_object(
    process: &mut Request,
    domain: &DomainSession<OpusObject>,
    target: &OpusObject,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let mut data = Vec::new();
    let result = match (target, request.command_id) {
        (OpusObject::Decoder(decoder), _) => {
            return super::dispatch_decode(process, decoder, request, hipc).await;
        }
        (OpusObject::Client(_), 0) if shape(&request, hipc, 0, 0, Input::Config) => {
            match config(process)
                .await?
                .and_then(|config| config.required_bytes())
            {
                Ok(size) => {
                    data.extend(size.to_le_bytes());
                    Ok(())
                }
                Err(error) => Err(error),
            }
        }
        (OpusObject::Client(_), 1) if shape(&request, hipc, 4, 0, Input::None) => {
            match hwacod::manager_size(request_u32(request.data, 0).unwrap()) {
                Ok(size) => {
                    data.extend(size.to_le_bytes());
                    Ok(())
                }
                Err(error) => Err(error),
            }
        }
        (OpusObject::Client(tags), 10) if shape(&request, hipc, 0x80, 2, Input::None) => {
            let transfer = request_u64(request.data, 0x60).unwrap();
            let address = request_u64(request.data, 0x68).unwrap();
            let size = request_u64(request.data, 0x70).unwrap();
            let properties = request_u64(request.data, 0x78).unwrap();
            let layout =
                match Layout::parse(&request.data[..0x60], transfer, address, size, properties) {
                    Ok(layout) => layout,
                    Err(error) => return opus_error(&request, error),
                };
            let Some(work) = process.copy_handle::<TransferMemoryObject>(0) else {
                return opus_error(&request, hwacod::audio_result(1536));
            };
            let Some(owner) = process.copy_handle::<nixe_runtime::RunnableProcess>(1) else {
                return opus_error(&request, hwacod::audio_result(1536));
            };
            if !Arc::ptr_eq(owner.memory(), process.memory()) {
                return Err(IpcWireError::Malformed(
                    "batch audio manager belongs to another process",
                ));
            }
            if transfer > work.size()
                || hwacod::overlaps(work.address().get(), work.size(), address, size)
            {
                return opus_error(&request, hwacod::ADDRESS);
            }
            process
                .reserve_payload(64 + (layout.io_count as usize + 1) * 64)
                .await?;
            let io = match TransferMemoryObject::new(
                owner.memory().clone(),
                GuestVirtualAddress::new(address),
                size,
                nixe_memory::MemoryPermissions::READ_WRITE,
            ) {
                Ok(io) => Arc::new(io),
                Err(_) => return opus_error(&request, hwacod::ADDRESS),
            };
            let tag = match tags.acquire() {
                Ok(tag) => tag,
                Err(error) => return opus_error(&request, error),
            };
            let manager = Arc::new(Manager::new(
                process.process_id(),
                tag,
                layout,
                nixe_runtime::EventObject::new(process.kernel()),
                owner.memory().space().machine().clone(),
                io,
                work,
            ));
            manager
                .initialize()
                .await
                .map_err(IpcWireError::SystemMemory)?;
            log::debug!(
                "opened batch Opus manager: tag={} instances={} io={} input_stride={} output_stride={}",
                manager.tag.value,
                manager.layout.instance_count,
                manager.layout.io_count,
                manager.layout.input_stride,
                manager.layout.output_stride
            );
            let event = process
                .return_handle(manager.event.readable())
                .map_err(IpcWireError::Handle)?;
            let tag = u32::from(manager.tag.value).to_le_bytes();
            return child(
                process,
                domain,
                &request,
                OpusObject::Batch(manager),
                &tag,
                &[event],
            );
        }
        (OpusObject::Batch(manager), id @ (0 | 1 | 10)) => {
            if process.process_id() != manager.owner {
                return opus_error(&request, hwacod::INVALID);
            }
            let input = match id {
                0 => Input::Config,
                1 => Input::None,
                _ => Input::Tasks,
            };
            if !shape(&request, hipc, if id == 10 { 0 } else { 4 }, 0, input) {
                return opus_error(&request, hwacod::INVALID);
            }
            let turn = manager.publication.enter();
            (&turn).await;
            process.lifetime().check()?;
            match id {
                0 => match config(process).await? {
                    Ok(config) => {
                        manager.initialize_instance(request_u32(request.data, 0).unwrap(), config)
                    }
                    Err(error) => Err(error),
                },
                1 => match manager
                    .reclaim()
                    .await
                    .map_err(IpcWireError::SystemMemory)?
                {
                    Ok(()) => manager.finalize_instance(request_u32(request.data, 0).unwrap()),
                    Err(error) => Err(error),
                },
                _ => execute(process, manager, hipc).await?,
            }
        }
        (OpusObject::Client(_), 0 | 1 | 10) => Err(hwacod::INVALID),
        _ => return unsupported_service_command(target.name(), request.command_id),
    };
    if let Err(error) = result {
        return opus_error(&request, error);
    }
    semantic_success(
        request.token,
        request.domain.is_some(),
        &data,
        &[],
        &[],
        None,
    )
}

// The native client accepts a pointer, a map alias, or auto-select's paired
// descriptors. A nonempty transfer must have exactly one active side.
fn task_input<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
) -> Result<&'a Buffer, IpcWireError> {
    match (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice()) {
        ([_], []) => Ok(&process.send_statics()[0]),
        ([], [_]) => Ok(&process.send_buffers()[0]),
        ([_], [_]) => one_auto_select_input(process, hipc),
        _ => Err(IpcWireError::Malformed(
            "batch audio requires one task input",
        )),
    }
}

async fn execute(
    process: &Request,
    manager: &Arc<Manager>,
    hipc: &HipcRequest<'_>,
) -> Result<Result<(), HorizonIpcResult>, IpcWireError> {
    let input = task_input(process, hipc)?;
    if input.size % 16 != 0 || input.size / 16 > u64::from(manager.layout.io_count) {
        return Ok(Err(hwacod::INVALID));
    }
    // Admission precedes capture and codec effects; one PCM buffer is reused for the batch.
    let budget = input.size
        + input.size / 16 * u64::from(manager.layout.input_stride)
        + 2 * u64::from(manager.layout.output_stride);
    process
        .reserve_payload(
            usize::try_from(budget)
                .map_err(|_| IpcWireError::Malformed("audio batch payload too large"))?,
        )
        .await?;
    let mut bytes = vec![0; input.size as usize];
    process.read(input, 0, &mut bytes).await?;
    let tasks = match bytes
        .chunks_exact(16)
        .map(Task::parse)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(tasks) => tasks,
        Err(error) => return Ok(Err(error)),
    };
    if let Err(error) = manager
        .reclaim()
        .await
        .map_err(IpcWireError::SystemMemory)?
        .and_then(|()| manager.validate(&tasks))
    {
        return Ok(Err(error));
    }
    let mut packets = Vec::with_capacity(tasks.len());
    for &task in &tasks {
        packets.push(
            manager
                .read(manager.io_offset(task), task.input_bytes as usize)
                .await
                .map_err(IpcWireError::SystemMemory)?,
        );
    }
    if tasks.is_empty() {
        return Ok(Ok(()));
    }
    let mut pcm = vec![0_i16; manager.layout.output_stride as usize / 2];
    // If an accepted batch cannot finish publication, its codec history cannot
    // be replayed safely. Fault the manager and wake its consumer.
    struct Publishing<'a>(Option<&'a Manager>);
    impl Drop for Publishing<'_> {
        fn drop(&mut self) {
            if let Some(manager) = self.0 {
                manager.fault();
            }
        }
    }
    let mut publishing = Publishing(Some(manager));
    for (task, packet) in tasks.into_iter().zip(packets) {
        let start = Instant::now();
        let result = manager.decode(task, &packet, &mut pcm);
        let duration = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let (result, output_bytes) = match result {
            Ok(samples) => {
                let bytes: Vec<u8> = pcm[..samples]
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                let size = bytes.len() as u32;
                manager
                    .write(
                        manager.io_offset(task) + u64::from(manager.layout.input_stride),
                        bytes,
                    )
                    .await
                    .map_err(IpcWireError::SystemMemory)?;
                (HorizonIpcResult::SUCCESS, size)
            }
            Err(error) => {
                log::debug!(
                    "batch Opus decode failed: tag={} instance={} result={:#x}",
                    task.tag,
                    task.instance,
                    error.raw()
                );
                (error, 0)
            }
        };
        manager
            .commit(task, task.completion(result, output_bytes, duration))
            .await
            .map_err(IpcWireError::SystemMemory)?;
    }
    publishing.0 = None;
    Ok(Ok(()))
}
