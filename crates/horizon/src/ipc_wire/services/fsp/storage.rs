//! `IStorage` wire command decoding.

use crate::ipc_wire::buffer::one_receive_buffer;
use crate::ipc_wire::io::{has_ipc_descriptors, request_u64};
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::ipc_wire::{IpcWireError, UnsupportedServiceOperation};
use crate::{IpcRequest, MAX_IPC_STORAGE_READ_BYTES};

use super::commands::StorageCommand;

pub(super) fn decode(
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    let Some(command) = StorageCommand::decode(request.command_id) else {
        return Ok(None);
    };
    match command {
        // IStorage::Read carries offset/size and exactly one map-alias output
        // buffer. Unlike IFile::Read, its response contains no byte-count
        // scalar:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L975-L983
        StorageCommand::Read => {
            let offset = request_u64(request.data, 0).ok_or(IpcWireError::Malformed(
                "storage read request omits its offset",
            ))?;
            let requested = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                "storage read request omits its size",
            ))?;
            if requested > MAX_IPC_STORAGE_READ_BYTES as u64 {
                return Err(IpcWireError::UnsupportedService(
                    UnsupportedServiceOperation::CommandSizeLimitExceeded {
                        service: "IStorage",
                        command_id: request.command_id,
                        operation: "read",
                        requested,
                        limit: MAX_IPC_STORAGE_READ_BYTES as u64,
                    },
                ));
            }
            let requested = usize::try_from(requested).map_err(|_| {
                IpcWireError::Malformed("storage read request size is out of range")
            })?;
            let descriptor = one_receive_buffer(hipc)?;
            let capacity = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("storage output buffer is too large"))?;
            if requested > capacity {
                return Err(IpcWireError::Malformed(
                    "storage read size exceeds its output buffer",
                ));
            }
            Ok(Some(IpcRequest::ReadStorage {
                offset,
                size: requested,
            }))
        }
        // IStorage::GetSize returns one signed 64-bit size scalar:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L1005-L1007
        StorageCommand::GetSize => {
            if has_ipc_descriptors(hipc) {
                return Ok(None);
            }
            Ok(Some(IpcRequest::GetStorageSize))
        }
    }
}
