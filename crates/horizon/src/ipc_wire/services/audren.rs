//! Audio-renderer configuration, updates and automatic execution.
//!
//! IPC and renderer state: https://switchbrew.org/wiki/Audio_services#audren:u
//! Parameter layout: https://github.com/switchbrew/libnx/blob/master/nx/source/services/audren.c

use super::prelude::*;

use crate::audio_renderer::update::{HEADER_SIZE, UpdateError, UpdateLayout};
use crate::audio_renderer::{PARAMETER_SIZE, RendererParameters};
use crate::{AudioRendererManagerSession, AudioRendererSession};

impl From<crate::audio_renderer::RenderError> for crate::ipc_wire::IpcWireError {
    fn from(error: crate::audio_renderer::RenderError) -> Self {
        use crate::audio_renderer::RenderError;
        match error {
            RenderError::Memory(error) => Self::SystemMemory(error),
            RenderError::Diagnostic(detail) => Self::AudioRenderer(detail),
            RenderError::Stopped => Self::Cancelled,
        }
    }
}

fn update_error(error: UpdateError, token: u32) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    match error {
        UpdateError::Invalid => cmif_error(token, HorizonIpcResult::AUDIO_INVALID_UPDATE_INFO),
        UpdateError::Unsupported(detail) => Err(IpcWireError::AudioRenderer(detail)),
        UpdateError::Mapping(error) => Err(IpcWireError::SystemMemory(std::sync::Arc::new(error))),
    }
}

fn update_outputs<'a>(
    process: &'a Request,
    hipc: &HipcRequest<'_>,
    auto: bool,
) -> Result<[&'a Buffer; 2], IpcWireError> {
    let [status, performance] = hipc.receive_buffers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "renderer update requires two output buffers",
        ));
    };
    let mut result = [&process.receive_buffers()[0], &process.receive_buffers()[1]];
    let pointers = match &hipc.receive_statics {
        ReceiveStatics::Entries(pointers) if auto && pointers.len() == 2 => Some(pointers),
        ReceiveStatics::None if !auto => None,
        _ => {
            return Err(IpcWireError::Malformed(
                "renderer update output pointer count is invalid",
            ));
        }
    };
    for (index, descriptor) in [status, performance].into_iter().enumerate() {
        let address = descriptor.address;
        let size = usize::try_from(descriptor.size)
            .map_err(|_| IpcWireError::Malformed("renderer output size overflows"))?;
        result[index] = if let Some(pointers) = pointers {
            let pointer = &pointers[index];
            match (
                pointer.address != 0 || pointer.size != 0,
                address != 0 || size != 0,
            ) {
                (true, false) if pointer.address != 0 => &process.receive_statics()[index],
                (false, true) if address != 0 => &process.receive_buffers()[index],
                (false, false) => &process.receive_buffers()[index],
                _ => {
                    return Err(IpcWireError::Malformed(
                        "renderer output has invalid auto-select descriptor sides",
                    ));
                }
            }
        } else {
            &process.receive_buffers()[index]
        };
    }
    Ok(result)
}

async fn dispatch_update(
    process: &Request,
    session: &AudioRendererSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let auto = request.command_id == 10;
    if hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || (!auto && !hipc.send_statics.is_empty())
    {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let input_buffer = if auto {
        one_auto_select_input(process, hipc)?
    } else {
        one_send_buffer(process, hipc)?
    };
    let input_size = input_buffer.size as usize;
    let [output_buffer, _performance] = update_outputs(process, hipc, auto)?;
    let output_size = output_buffer.size as usize;
    if input_size < HEADER_SIZE {
        return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_UPDATE_INFO);
    }
    let mut header = [0; HEADER_SIZE];
    process.read(input_buffer, 0, &mut header).await?;
    let layout = match UpdateLayout::parse(&session.0.parameters, &header, input_size) {
        Ok(layout) => layout,
        Err(error) => return update_error(error, request.token),
    };
    if output_size != layout.output_size {
        return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_UPDATE_INFO);
    }
    let mut input = Vec::new();
    input
        .try_reserve_exact(input_size)
        .map_err(|_| IpcWireError::HostResourceExhausted("reading an audio renderer update"))?;
    input.resize(input_size, 0);
    process.read(input_buffer, 0, &mut input).await?;
    let output = {
        let mut state = session.0.update.take().await;
        if let Some(error) = session.0.failure.get() {
            return Err(error.clone().into());
        }
        let output = match state.update(&session.0.parameters, &layout, &input, |address, size| {
            process.memory().space().capture(
                GuestVirtualAddress::new(address),
                size,
                nixe_memory::MemoryPermissions::NONE,
            )
        }) {
            Ok(output) => output,
            Err(error) => return update_error(error, request.token),
        };
        session.0.event.clear();
        output
    };
    process.write(output_buffer, 0, &output).await?;
    semantic_success(request.token, false, &[], &[], &[], None)
}

pub(in crate::ipc_wire) async fn dispatch_audio_renderer_manager(
    process: &mut Request,
    session: &AudioRendererManagerSession,
    backend: Option<&std::sync::Arc<dyn nixe_audio::AudioBackend>>,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let command = request.command_id;
    if matches!(command, 2 | 4) {
        let required_size = if command == 2 { 8 } else { 16 };
        if !request.has_payload_size(required_size) || has_ipc_descriptors(hipc) {
            return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
        }
        let (revision, aruid) = if command == 2 {
            (
                u32::from_le_bytes(*b"REV1"),
                request_u64(request.data, 0).unwrap(),
            )
        } else {
            (
                request_u32(request.data, 0).unwrap(),
                request_u64(request.data, 8).unwrap(),
            )
        };
        let Some(revision) = crate::audio_renderer::revision_number(revision) else {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_REVISION);
        };
        let backend = backend.ok_or_else(|| {
            IpcWireError::AudioBackend("IAudioDevice requires a host audio backend".into())
        })?;
        let device = session
            .device(revision, aruid, backend.clone())
            .map_err(|error| IpcWireError::AudioBackend(error.0.into_boxed_str()))?;
        let handle = process
            .return_handle(HorizonIpcObject::AudioDevice(device))
            .map_err(IpcWireError::Handle)?;
        log::debug!("opening audio device interface: revision={revision} aruid={aruid:#x}");
        return semantic_success(request.token, false, &[], &[], &[], Some(handle));
    }
    if !matches!(command, 0 | 1) {
        return unsupported_service_command("audren:u", request.command_id);
    }
    let opening = command == 0;
    let required_size = if opening { 72 } else { PARAMETER_SIZE };
    let invalid_descriptors = if opening {
        hipc.pid.is_none()
            || hipc.copy_handles.len() != 2
            || !hipc.move_handles.is_empty()
            || !hipc.send_buffers.is_empty()
            || !hipc.receive_buffers.is_empty()
            || !hipc.exchange_buffers.is_empty()
            || !hipc.send_statics.is_empty()
            || !matches!(hipc.receive_statics, ReceiveStatics::None)
    } else {
        has_ipc_descriptors(hipc)
    };
    if !request.has_payload_size(required_size) || invalid_descriptors {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let params = RendererParameters::decode(request.data[..PARAMETER_SIZE].try_into().unwrap());
    let Some(revision) = params.revision_number() else {
        return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_REVISION);
    };
    let Some(size) = params.work_buffer_size(revision) else {
        return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_ARGUMENT);
    };
    if opening {
        let Some(work_memory) = process.copy_handle::<TransferMemoryObject>(0) else {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_HANDLE);
        };
        let Some(owner) = process.copy_handle::<nixe_runtime::RunnableProcess>(1) else {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INVALID_HANDLE);
        };
        if owner.process_id() != process.process_id() {
            return unsupported_variant(
                "audren:u",
                command,
                "renderer for another process is not implemented",
            );
        }
        let work_size = request_u64(request.data, 56).unwrap();
        if work_size > work_memory.size() || work_size < size {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_INSUFFICIENT_BUFFER);
        }
        if params.execution_mode != 0 || params.rendering_device != 0 {
            return unsupported_variant(
                "audren:u",
                command,
                "manual execution or a non-ADSP rendering device is not implemented",
            );
        }
        // These are renderer constraints, not constraints on the sizing query.
        if params.sample_rate != 48000 || params.samples != 240 {
            return unsupported_variant(
                "audren:u",
                command,
                "renderer rates other than 48000 Hz / 240 samples are not implemented",
            );
        }
        let aruid = request_u64(request.data, 64).unwrap();
        log::debug!(
            "opening audio renderer: parameters={params:?} work-memory={work_size:#x} aruid={aruid:#x}"
        );
        let Some(renderer) = session
            .open(params, work_memory, work_size, aruid, process.process_id())
            .await
            .map_err(IpcWireError::SystemMemory)?
        else {
            return cmif_error(request.token, HorizonIpcResult::AUDIO_OUT_OF_SESSIONS);
        };
        let handle = process
            .return_handle(HorizonIpcObject::AudioRenderer(renderer))
            .map_err(IpcWireError::Handle)?;
        return semantic_success(request.token, false, &[], &[], &[], Some(handle));
    }
    log::debug!(
        "audio renderer work memory: sample-rate={} parameters={params:?} revision-number={revision} size={size:#x}",
        params.sample_rate,
    );
    semantic_success(request.token, false, &size.to_le_bytes(), &[], &[], None)
}

fn unsupported_variant<T>(
    service: &'static str,
    command_id: u32,
    detail: &'static str,
) -> Result<T, IpcWireError> {
    Err(IpcWireError::UnsupportedService(
        UnsupportedServiceOperation::CommandVariant {
            service,
            command_id,
            detail,
        },
    ))
}

pub(in crate::ipc_wire) async fn dispatch_audio_renderer(
    process: &mut Request,
    session: &AudioRendererSession,
    backend: Option<&std::sync::Arc<dyn nixe_audio::AudioBackend>>,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let command = request.command_id;
    if matches!(command, 4 | 10) {
        return dispatch_update(process, session, request, hipc).await;
    }
    if !matches!(command, 0..=3 | 5..=9) {
        return unsupported_service_command("IAudioRenderer", command);
    }
    if has_ipc_descriptors(hipc) || (command == 8 && !request.has_payload_size(4)) {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    let state = &session.0;
    if let Some(error) = state.failure.get() {
        return Err(error.clone().into());
    }
    let value = match command {
        0 => state.parameters.sample_rate,
        1 => state.parameters.samples,
        2 => state.parameters.mixes,
        3 => u32::from(!state.update.take().await.started),
        5 => {
            let backend = backend.ok_or_else(|| {
                IpcWireError::AudioRenderer("Start requires a host audio backend".into())
            })?;
            state
                .start(backend.as_ref())
                .await
                .map_err(IpcWireError::AudioRenderer)?;
            return semantic_success(request.token, false, &[], &[], &[], None);
        }
        6 => {
            state.stop();
            return semantic_success(request.token, false, &[], &[], &[], None);
        }
        7 => {
            let handle = process
                .return_handle(state.event.clone())
                .map_err(IpcWireError::Handle)?;
            return semantic_success(request.token, false, &[], &[handle], &[], None);
        }
        8 => {
            *state.rendering_time_limit.lock().unwrap() = request_u32(request.data, 0).unwrap();
            return semantic_success(request.token, false, &[], &[], &[], None);
        }
        9 => *state.rendering_time_limit.lock().unwrap(),
        _ => unreachable!(),
    };
    semantic_success(request.token, false, &value.to_le_bytes(), &[], &[], None)
}
