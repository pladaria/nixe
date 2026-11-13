use super::nv_ioctl::{nv_ioctl_buffers, nv_ioctl2_buffers, nv_ioctl3_buffers};
use super::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NvDrvCommand {
    Open,
    Ioctl,
    Close,
    Initialize,
    QueryEvent,
    DumpStatus,
    GetStatus,
    SetAruid,
    Ioctl2,
    Ioctl3,
    SetGraphicsFirmwareMemoryMarginEnabled,
}

impl NvDrvCommand {
    const fn decode(command_id: u32) -> Option<Self> {
        match command_id {
            0 => Some(Self::Open),
            1 => Some(Self::Ioctl),
            2 => Some(Self::Close),
            3 => Some(Self::Initialize),
            4 => Some(Self::QueryEvent),
            8 => Some(Self::SetAruid),
            6 => Some(Self::GetStatus),
            9 => Some(Self::DumpStatus),
            11 => Some(Self::Ioctl2),
            12 => Some(Self::Ioctl3),
            13 => Some(Self::SetGraphicsFirmwareMemoryMarginEnabled),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) async fn dispatch_nvdrv(
    process: &mut Request,
    session: &NvDrvSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    caller_thread_id: u64,
    clock: &nixe_runtime::VirtualClock,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let Some(command) = NvDrvCommand::decode(request.command_id) else {
        return Err(IpcWireError::UnsupportedNvDrv(
            crate::nvdrv::UnsupportedNvDrvOperation::ServiceCommand {
                command_id: request.command_id,
            },
        ));
    };
    match command {
        // GetStatus returns NvError followed by NvDrvStatus.
        // https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#GetStatus
        NvDrvCommand::GetStatus => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (status, error) = session.status();
            let mut data = [0; 36];
            data[..4].copy_from_slice(&error.to_le_bytes());
            data[4..].copy_from_slice(&status);
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                None,
            ))
        }

        // DumpStatus emits driver diagnostics and carries no input or output.
        // https://switchbrew.org/wiki/NV_services#DumpStatus
        NvDrvCommand::DumpStatus => {
            if !request.has_payload_size(0) || has_ipc_descriptors(hipc) {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.dump_status();
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }

        NvDrvCommand::Open => {
            let descriptor = one_send_buffer(process, hipc)?;
            let size = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("nvdrv device path is too large"))?;
            if size == 0 || size > 0x100 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            process.reserve_payload(size).await?;
            let mut path = vec![0_u8; size];
            process.read(descriptor, 0, &mut path).await?;
            let (fd, error) = match session.open(&path, process.process_id()) {
                Ok(fd) => (fd.raw(), crate::nvdrv::NV_SUCCESS),
                Err(crate::nvdrv::NvDrvCallError::GuestResult(error)) => (0, error),
                Err(crate::nvdrv::NvDrvCallError::Unsupported(operation)) => {
                    return Err(IpcWireError::UnsupportedNvDrv(operation));
                }
            };
            let mut data = Vec::with_capacity(8);
            data.extend_from_slice(&fd.to_le_bytes());
            data.extend_from_slice(&error.to_le_bytes());
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &data, None)?,
                None,
            ))
        }
        command @ (NvDrvCommand::Ioctl | NvDrvCommand::Ioctl2 | NvDrvCommand::Ioctl3) => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(ioctl) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let buffers = if command == NvDrvCommand::Ioctl2 {
                // libnx `nvIoctl2` uses command 11 and carries the GPFIFO
                // descriptor array in a second input auto-select buffer:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L172-L208
                nv_ioctl2_buffers(hipc, ioctl)?
            } else if command == NvDrvCommand::Ioctl3 {
                nv_ioctl3_buffers(hipc, ioctl)?
            } else {
                nv_ioctl_buffers(hipc, ioctl)?
            };
            let input_size = buffers
                .input
                .map(|descriptor| descriptor.size)
                .map_or(Ok(0), usize::try_from)
                .map_err(|_| IpcWireError::Malformed("nvdrv ioctl input is too large"))?;
            let additional_input_size = buffers
                .additional_input
                .map(|descriptor| descriptor.size)
                .map_or(Ok(0), usize::try_from)
                .map_err(|_| IpcWireError::Malformed("nvdrv Ioctl2 input is too large"))?;
            // Like Eden's NVDRV::Ioctl2, consume the entire inline descriptor.
            // GPFIFO submissions can contain more than 64 KiB of entries.
            // dev-tools/eden/src/core/hle/service/nvdrv/nvdrv_interface.cpp
            let output_size = buffers.output.map_or(0, |buffer| buffer.size)
                + buffers.additional_output.map_or(0, |buffer| buffer.size);
            let output_size = usize::try_from(output_size)
                .map_err(|_| IpcWireError::Malformed("nvdrv ioctl output size overflows"))?;
            process
                .reserve_payload(input_size + additional_input_size + output_size)
                .await?;
            let mut input = vec![0_u8; input_size];
            if buffers.input.is_some() {
                process
                    .read(&process.send_buffers()[0], 0, &mut input)
                    .await?;
            }
            log::trace!(
                "nvdrv ioctl input: fd={fd:#010x} request={ioctl:#010x} input-size={} inline-size={additional_input_size} input-prefix={:02x?}",
                input.len(),
                &input[..input.len().min(64)],
            );
            let mut additional_input = vec![0_u8; additional_input_size];
            if buffers.additional_input.is_some() {
                process
                    .read(&process.send_buffers()[1], 0, &mut additional_input)
                    .await?;
            }
            let mut response = loop {
                let response = session
                    .ioctl_outcome(crate::nvdrv::NvDrvIoctlRequest {
                        fd: NvDrvFileDescriptor::new(fd),
                        request: ioctl,
                        input: &input,
                        inline: if let Some(buffer) = buffers.additional_output {
                            crate::nvdrv::NvDrvInlineBuffer::Output(
                                usize::try_from(buffer.size).map_err(|_| {
                                    IpcWireError::Malformed("nvdrv inline output is too large")
                                })?,
                            )
                        } else if command == NvDrvCommand::Ioctl2 {
                            crate::nvdrv::NvDrvInlineBuffer::Input(&additional_input)
                        } else {
                            crate::nvdrv::NvDrvInlineBuffer::None
                        },
                        process_id: process.process_id(),
                        address_space: process.cpu().address_space_id(),
                        memory: process.memory(),
                        caller: crate::nvdrv::NvDrvIoctlCaller {
                            thread_id: caller_thread_id,
                            clock,
                        },
                    })
                    .map_err(IpcWireError::UnsupportedNvDrv)?;
                match response {
                    NvDrvIoctlOutcome::Complete(response) => break response,
                    NvDrvIoctlOutcome::PendingAddressSpace(pending) => {
                        break pending.await.map_err(IpcWireError::UnsupportedNvDrv)?;
                    }
                    NvDrvIoctlOutcome::PendingHost1x(capture) => {
                        break capture
                            .finish()
                            .await
                            .map_err(IpcWireError::UnsupportedNvDrv)?;
                    }
                    NvDrvIoctlOutcome::PendingSubmission(wait) => {
                        crate::ipc_wire::operation::ServiceEvent::new(
                            process.kernel(),
                            wait.wake_event(),
                            None,
                            process.lifetime(),
                        )
                        .await?;
                    }
                    NvDrvIoctlOutcome::PendingSyncpointWait(wait) => {
                        crate::ipc_wire::operation::ServiceEvent::new(
                            process.kernel(),
                            wait.wake_event(),
                            wait.remaining(),
                            process.lifetime(),
                        )
                        .await?;
                    }
                }
            };
            if response.driver_result != crate::nvdrv::NV_SUCCESS {
                // A driver status is guest-visible ABI, not an unsupported
                // emulator operation. In particular, asynchronous syncpoint
                // waits return Timeout while arming their completion event.
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/fence.c
                // EventWait/EventWaitAsync report Timeout immediately when the
                // syncpoint is not reached yet; completion is signalled through
                // the registered event. This is routine fence synchronization,
                // including zero-timeout polls, rather than an expired wait.
                let level = if response.driver_result == crate::nvdrv::NV_TIMEOUT
                    && matches!(ioctl, 0xc010_001d | 0xc010_001e)
                {
                    log::Level::Trace
                } else {
                    log::Level::Debug
                };
                log::log!(
                    level,
                    "nvdrv ioctl returned a non-success driver status: fd={fd:#010x} request={ioctl:#010x} result={:#x}{} input-size={} input-prefix={:02x?}",
                    response.driver_result,
                    if response.driver_result == crate::nvdrv::NV_TIMEOUT {
                        " (Timeout)"
                    } else {
                        ""
                    },
                    input.len(),
                    &input[..input.len().min(64)],
                );
            }
            if buffers.output.is_some() {
                write_descriptor_bytes(
                    process,
                    &process.receive_buffers()[0],
                    &mut response.output,
                )
                .await?;
            }
            if buffers.additional_output.is_some() {
                write_descriptor_bytes(
                    process,
                    &process.receive_buffers()[1],
                    &mut response.additional_output,
                )
                .await?;
            }
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &response.driver_result.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        NvDrvCommand::Close => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let mut close = session.close(NvDrvFileDescriptor::new(fd));
            let error = std::future::poll_fn(|cx| {
                if process.lifetime().check().is_err() {
                    close.cancel();
                }
                std::future::Future::poll(std::pin::Pin::new(&mut close), cx)
            })
            .await
            .map_err(IpcWireError::UnsupportedNvDrv)?;
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &error.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        NvDrvCommand::Initialize => {
            let Some(transfer_size) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.copy_handles.len() != 2
                || process
                    .copy_handle::<nixe_runtime::RunnableProcess>(0)
                    .is_none_or(|owner| owner.process_id() != process.process_id())
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let Some(transfer) = process.copy_handle::<TransferMemoryObject>(1) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if u64::from(transfer_size) > transfer.size() || transfer_size == 0 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.initialize(transfer_size);
            Ok((
                encode_response(
                    request.token,
                    HorizonIpcResult::SUCCESS,
                    &crate::nvdrv::NV_SUCCESS.to_le_bytes(),
                    None,
                )?,
                None,
            ))
        }
        NvDrvCommand::QueryEvent => {
            let Some(fd) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            let Some(event_id) = request_u32(request.data, 4) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.pid.is_some()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let (event, error) = session
                .query_event(NvDrvFileDescriptor::new(fd), event_id, process.process_id())
                .map_err(IpcWireError::UnsupportedNvDrv)?;
            let handle = if let Some(event) = event {
                Some(process.return_handle(event).map_err(IpcWireError::Handle)?)
            } else {
                None
            };
            Ok((
                encode_nvdrv_query_event_response(request.token, error, handle)?,
                handle,
            ))
        }
        NvDrvCommand::SetGraphicsFirmwareMemoryMarginEnabled => {
            // Command 13 takes a u64 and returns no driver-result payload.
            // Its effective boolean is gated by the system configuration
            // nv!nv_graphics_firmware_memory_margin. Nixe keeps that setting
            // disabled: Maxwell executes directly without an NVIDIA graphics
            // firmware heap, so every input leaves the margin disabled. There
            // is no per-client state to allocate or firmware memory to reserve.
            // https://switchbrew.org/w/index.php?title=NV_services#SetGraphicsFirmwareMemoryMarginEnabled
            if request_u64(request.data, 0).is_none()
                || !request.has_payload_size(8)
                || has_ipc_descriptors(hipc)
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
        NvDrvCommand::SetAruid => {
            // INvDrvServices::SetAruid carries the caller PID descriptor and
            // an AppletResourceUserId. Associate both with the shared nvdrv
            // session so cloned service handles observe the same identity:
            // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L104-L106
            let Some(applet_resource_user_id) = request_u64(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.pid.is_none()
                || !hipc.copy_handles.is_empty()
                || !hipc.move_handles.is_empty()
                || !hipc.send_statics.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            session.set_aruid(process.process_id(), applet_resource_user_id);
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
                None,
            ))
        }
    }
}

fn encode_nvdrv_query_event_response(
    token: u32,
    driver_result: u32,
    copy_handle: Option<u32>,
) -> Result<Vec<u8>, IpcWireError> {
    let copy_handles = copy_handle.as_slice();
    let response_data = driver_result.to_le_bytes();
    CmifResponse {
        token,
        result: HorizonIpcResult::SUCCESS.raw(),
        data: &response_data,
        pid: None,
        copy_handles,
        move_handles: &[],
        send_statics: &[],
        is_domain: false,
        domain_objects: &[],
    }
    .encode()
    .map_err(Into::into)
}
