use super::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NvDrvCommand {
    Open,
    Ioctl,
    Close,
    Initialize,
    QueryEvent,
    SetAruid,
    Ioctl2,
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
            11 => Some(Self::Ioctl2),
            _ => None,
        }
    }
}

pub(in crate::ipc_wire) fn dispatch_nvdrv(
    process: &mut ExceptionProcessContext<'_>,
    session: &NvDrvSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    caller_thread_id: u64,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    let service = NvDrvService::new(session);
    let Some(command) = NvDrvCommand::decode(request.command_id) else {
        return Err(IpcWireError::UnsupportedNvDrv(
            crate::nvdrv::UnsupportedNvDrvOperation::ServiceCommand {
                command_id: request.command_id,
            },
        ));
    };
    match command {
        NvDrvCommand::Open => {
            let descriptor = one_send_buffer(hipc)?;
            let size = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("nvdrv device path is too large"))?;
            if size == 0 || size > 0x100 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut path = vec![0_u8; size];
            read_bytes(
                process,
                GuestVirtualAddress::new(descriptor.address),
                &mut path,
            )?;
            let (fd, error) = match service.open(&path, process.process_id()) {
                Ok(fd) => (fd.raw(), crate::nvdrv::NV_SUCCESS),
                Err(NvDrvServiceError::DriverResult(error)) => (0, error),
                Err(NvDrvServiceError::Unsupported(operation)) => {
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
        command @ (NvDrvCommand::Ioctl | NvDrvCommand::Ioctl2) => {
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
            } else {
                nv_ioctl_buffers(hipc, ioctl)?
            };
            let input_size = buffers
                .input
                .map(|descriptor| descriptor.size)
                .map_or(Ok(0), usize::try_from)
                .map_err(|_| IpcWireError::Malformed("nvdrv ioctl input is too large"))?;
            if input_size > 0x1000 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut input = vec![0_u8; input_size];
            if let Some(input_descriptor) = buffers.input {
                read_bytes(
                    process,
                    GuestVirtualAddress::new(input_descriptor.address),
                    &mut input,
                )?;
            }
            let additional_input_size = buffers
                .additional_input
                .map(|descriptor| descriptor.size)
                .map_or(Ok(0), usize::try_from)
                .map_err(|_| IpcWireError::Malformed("nvdrv Ioctl2 input is too large"))?;
            if additional_input_size > 0x1_0000 {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            let mut additional_input = vec![0_u8; additional_input_size];
            if let Some(input_descriptor) = buffers.additional_input {
                read_bytes(
                    process,
                    GuestVirtualAddress::new(input_descriptor.address),
                    &mut additional_input,
                )?;
            }
            let response = service
                .ioctl(crate::nvdrv::NvDrvIoctlRequest {
                    fd: NvDrvFileDescriptor::new(fd),
                    request: ioctl,
                    input: &input,
                    additional_input: &additional_input,
                    process_id: process.process_id(),
                    address_space: process.cpu().address_space_id(),
                    translator: process.canonical_memory(),
                    thread_id: caller_thread_id,
                })
                .map_err(IpcWireError::UnsupportedNvDrv)?;
            let response = match response {
                NvDrvIoctlOutcome::Complete(response) => response,
                NvDrvIoctlOutcome::PendingSyncpointWait(wait) => {
                    return Err(IpcWireError::PendingNvDrv(wait));
                }
            };
            if let Some(output_descriptor) = buffers.output {
                write_descriptor_bytes(process, output_descriptor, &response.output)?;
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
            let error = service.close(NvDrvFileDescriptor::new(fd));
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
            let Some(_transfer_size) = request_u32(request.data, 0) else {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            };
            if hipc.copy_handles.len() != 2
                || hipc.copy_handles[0] != crate::CURRENT_PROCESS_HANDLE
                || process
                    .handles()
                    .get_as::<TransferMemoryObject>(hipc.copy_handles[1])
                    .is_none()
            {
                return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
            }
            service.initialize();
            Ok((
                encode_response(request.token, HorizonIpcResult::SUCCESS, &[], None)?,
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
            let (event, error) = service
                .query_event(NvDrvFileDescriptor::new(fd), event_id, process.process_id())
                .map_err(IpcWireError::UnsupportedNvDrv)?;
            let handle = if let Some(event) = event {
                Some(process.handles_mut().insert(event).map_err(|_| {
                    IpcWireError::HostResourceExhausted("installing an nvdrv event handle")
                })?)
            } else {
                None
            };
            Ok((
                encode_nvdrv_query_event_response(request.token, error, handle)?,
                handle,
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
            service.set_aruid(process.process_id(), applet_resource_user_id);
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

const NV_IOCTL_WRITE: u32 = 1 << 30;
const NV_IOCTL_READ: u32 = 1 << 31;
const NV_IOCTL_SIZE_MASK: u32 = 0x3fff;

// nvIoctl derives the two buffer presences and their common size directly
// from these Linux-style request fields before issuing CMIF command 1:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c#L137-L170

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvIoctlBuffers {
    input: Option<BufferDescriptor>,
    additional_input: Option<BufferDescriptor>,
    output: Option<BufferDescriptor>,
}

fn nv_ioctl_buffers(hipc: &HipcRequest<'_>, request: u32) -> Result<NvIoctlBuffers, IpcWireError> {
    nv_ioctl_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &hipc.receive_buffers,
        &hipc.receive_statics,
        request,
    )
}

fn nv_ioctl_buffer_descriptors(
    send_statics: &[SendStaticDescriptor],
    send_buffers: &[BufferDescriptor],
    receive_buffers: &[BufferDescriptor],
    receive_statics: &ReceiveStatics,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [input] = send_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one input auto-select buffer",
        ));
    };
    let [output] = receive_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one output auto-select buffer",
        ));
    };
    let [input_pointer] = send_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one input auto-select pointer",
        ));
    };
    let ReceiveStatics::Entries(output_pointers) = receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires one output auto-select pointer",
        ));
    };
    let [output_pointer] = output_pointers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl requires exactly one output auto-select pointer",
        ));
    };

    // Nixe reports a zero CMIF pointer-buffer size, so libnx selects the
    // map-alias side of each HipcAutoSelect pair and emits null pointer-side
    // placeholders. Keep this transport contract pinned to the implementation
    // used by the target homebrew:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h#L157-L177
    if input_pointer.address != 0
        || input_pointer.size != 0
        || output_pointer.address != 0
        || output_pointer.size != 0
    {
        return Err(IpcWireError::Malformed(
            "nvdrv ioctl auto-select pointer placeholders are not null",
        ));
    }

    let encoded_size = u64::from((request >> 16) & NV_IOCTL_SIZE_MASK);
    let select = |descriptor: BufferDescriptor,
                  present: bool,
                  present_error: &'static str,
                  absent_error: &'static str|
     -> Result<Option<BufferDescriptor>, IpcWireError> {
        if descriptor.mode == BufferMode::Invalid {
            return Err(IpcWireError::Malformed(present_error));
        }
        if present {
            if descriptor.size != encoded_size || (encoded_size != 0 && descriptor.address == 0) {
                return Err(IpcWireError::Malformed(present_error));
            }
            Ok(Some(descriptor))
        } else if descriptor.address == 0
            && descriptor.size == 0
            && descriptor.mode == BufferMode::Normal
        {
            Ok(None)
        } else {
            Err(IpcWireError::Malformed(absent_error))
        }
    };

    Ok(NvIoctlBuffers {
        input: select(
            *input,
            request & NV_IOCTL_WRITE != 0,
            "nvdrv ioctl input buffer does not match its encoded direction and size",
            "nvdrv ioctl without input carries a non-null input placeholder",
        )?,
        additional_input: None,
        output: select(
            *output,
            request & NV_IOCTL_READ != 0,
            "nvdrv ioctl output buffer does not match its encoded direction and size",
            "nvdrv ioctl without output carries a non-null output placeholder",
        )?,
    })
}

fn nv_ioctl2_buffers(hipc: &HipcRequest<'_>, request: u32) -> Result<NvIoctlBuffers, IpcWireError> {
    nv_ioctl2_buffer_descriptors(
        &hipc.send_statics,
        &hipc.send_buffers,
        &hipc.receive_buffers,
        &hipc.receive_statics,
        request,
    )
}

fn nv_ioctl2_buffer_descriptors(
    send_statics: &[SendStaticDescriptor],
    send_buffers: &[BufferDescriptor],
    receive_buffers: &[BufferDescriptor],
    receive_statics: &ReceiveStatics,
    request: u32,
) -> Result<NvIoctlBuffers, IpcWireError> {
    let [input, additional_input] = send_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly two input auto-select buffers",
        ));
    };
    let [output] = receive_buffers else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly one output auto-select buffer",
        ));
    };
    let [input_pointer, additional_input_pointer] = send_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly two input auto-select pointers",
        ));
    };
    let ReceiveStatics::Entries(output_pointers) = receive_statics else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires one output auto-select pointer",
        ));
    };
    let [output_pointer] = output_pointers.as_slice() else {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 requires exactly one output auto-select pointer",
        ));
    };
    if [input_pointer, additional_input_pointer]
        .into_iter()
        .any(|pointer| pointer.address != 0 || pointer.size != 0)
        || output_pointer.address != 0
        || output_pointer.size != 0
    {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 auto-select pointer placeholders are not null",
        ));
    }
    if additional_input.mode == BufferMode::Invalid
        || (additional_input.size != 0 && additional_input.address == 0)
    {
        return Err(IpcWireError::Malformed(
            "nvdrv Ioctl2 additional input buffer is invalid",
        ));
    }

    let ordinary = nv_ioctl_buffer_descriptors(
        &[*input_pointer],
        &[*input],
        &[*output],
        &ReceiveStatics::Entries(vec![*output_pointer]),
        request,
    )?;
    Ok(NvIoctlBuffers {
        input: ordinary.input,
        additional_input: Some(*additional_input),
        output: ordinary.output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_wire::message::ReceiveStaticDescriptor;

    fn buffer(address: u64, size: u64) -> BufferDescriptor {
        BufferDescriptor {
            address,
            size,
            mode: BufferMode::Normal,
        }
    }

    fn ioctl_descriptors(
        input: BufferDescriptor,
        output: BufferDescriptor,
        request: u32,
    ) -> Result<NvIoctlBuffers, IpcWireError> {
        nv_ioctl_buffer_descriptors(
            &[SendStaticDescriptor {
                address: 0,
                size: 0,
                index: 0,
            }],
            &[input],
            &[output],
            &ReceiveStatics::Entries(vec![ReceiveStaticDescriptor {
                address: 0,
                size: 0,
            }]),
            request,
        )
    }

    #[test]
    fn write_only_ioctl_accepts_the_libnx_null_output_placeholder() {
        assert_eq!(
            ioctl_descriptors(buffer(0x1000, 40), buffer(0, 0), 0x4028_4109),
            Ok(NvIoctlBuffers {
                input: Some(buffer(0x1000, 40)),
                additional_input: None,
                output: None,
            })
        );
    }

    #[test]
    fn ioctl_direction_rejects_a_non_null_inactive_side() {
        assert_eq!(
            ioctl_descriptors(buffer(0x1000, 40), buffer(0x2000, 40), 0x4028_4109),
            Err(IpcWireError::Malformed(
                "nvdrv ioctl without output carries a non-null output placeholder"
            ))
        );
    }

    #[test]
    fn query_event_response_uses_a_copy_handle() {
        let response =
            encode_nvdrv_query_event_response(9, crate::nvdrv::NV_SUCCESS, Some(0x55)).unwrap();
        let word = |offset| u32::from_le_bytes(response[offset..offset + 4].try_into().unwrap());

        assert_eq!(word(4) >> 31, 1);
        assert_eq!(word(8), 1 << 1);
        assert_eq!(word(12), 0x55);
        assert_eq!(word(24), HorizonIpcResult::SUCCESS.raw());
        assert_eq!(word(32), crate::nvdrv::NV_SUCCESS);
    }
}
