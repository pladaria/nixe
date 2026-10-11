//! `IFile` wire command decoding.

use nixe_memory::GuestVirtualAddress;
use nixe_runtime::ExceptionProcessContext;

use crate::ipc_wire::buffer::{one_receive_buffer, one_send_buffer};
use crate::ipc_wire::io::{read_bytes, request_u32, request_u64};
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::ipc_wire::{IpcWireError, UnsupportedServiceOperation};
use crate::{IpcRequest, MAX_IPC_READ_BYTES, SemanticIpcObject};

use super::commands::FileCommand;

pub(super) fn decode(
    process: &ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    let Some(command) = FileCommand::decode(request.command_id) else {
        return Ok(None);
    };
    match command {
        // IFile::Read input layout and map-alias output buffer:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L980-L994
        FileCommand::Read => {
            let offset = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                "file read request omits its offset",
            ))?;
            let requested = request_u64(request.data, 16)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(IpcWireError::Malformed(
                    "file read request size is out of range",
                ))?;
            let descriptor = one_receive_buffer(hipc)?;
            let capacity = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("file output buffer is too large"))?;
            Ok(Some(IpcRequest::ReadFile {
                offset,
                size: requested.min(capacity).min(MAX_IPC_READ_BYTES),
            }))
        }
        // IFile::Write carries option/padding/offset/size and one map-alias
        // input buffer:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L994-L1017
        FileCommand::Write => {
            if !matches!(object, SemanticIpcObject::HostFile(_)) {
                return Ok(None);
            }
            let option = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                "file write request omits its option",
            ))?;
            let offset = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                "file write request omits its offset",
            ))?;
            let requested = request_u64(request.data, 16)
                .ok_or(IpcWireError::Malformed("file write request omits its size"))?;
            if requested > MAX_IPC_READ_BYTES as u64 {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandSizeLimitExceeded {
                        service: "IFile",
                        command_id: request.command_id,
                        operation: "write",
                        requested,
                        limit: MAX_IPC_READ_BYTES as u64,
                    },
                ));
            }
            let requested = usize::try_from(requested)
                .map_err(|_| IpcWireError::Malformed("file write request size is out of range"))?;
            let descriptor = one_send_buffer(hipc)?;
            let capacity = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("file input buffer is too large"))?;
            if requested > capacity {
                return Err(IpcWireError::Malformed(
                    "file write size exceeds its input buffer",
                ));
            }
            let mut data = vec![0; requested];
            read_bytes(
                process,
                GuestVirtualAddress::new(descriptor.address),
                &mut data,
            )?;
            Ok(Some(IpcRequest::WriteFile {
                offset,
                data,
                flush: option & 1 != 0,
            }))
        }
        FileCommand::Flush => {
            if !matches!(object, SemanticIpcObject::HostFile(_)) {
                return Ok(None);
            }
            Ok(Some(IpcRequest::FlushFile))
        }
        FileCommand::SetSize => {
            if !matches!(object, SemanticIpcObject::HostFile(_)) {
                return Ok(None);
            }
            Ok(Some(IpcRequest::SetFileSize {
                size: request_u64(request.data, 0).ok_or(IpcWireError::Malformed(
                    "set-file-size request omits its size",
                ))?,
            }))
        }
        FileCommand::GetSize => Ok(Some(IpcRequest::GetFileSize)),
    }
}
