//! `IFileSystem` wire command decoding.

use nixe_memory::GuestVirtualAddress;
use nixe_runtime::ExceptionProcessContext;

use crate::ipc_wire::IpcWireError;
use crate::ipc_wire::io::{read_bytes, request_u32, request_u64};
use crate::ipc_wire::message::{
    BufferDescriptor, BufferMode, CmifRequest, HipcRequest, SendStaticDescriptor,
};
use crate::{IpcRequest, MAX_IPC_PATH_BYTES, SemanticIpcObject};

use super::abi::FS_MAX_PATH;
use super::commands::FileSystemCommand;

pub(super) fn decode(
    process: &ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    let Some(command) = FileSystemCommand::decode(request.command_id) else {
        return Ok(None);
    };
    match command {
        // IFileSystem::CreateFile/CreateDirectory use the same bounded
        // input-pointer path as open operations:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L816-L840
        FileSystemCommand::CreateFile => {
            if !matches!(object, SemanticIpcObject::HostDirectoryFileSystem(_)) {
                return Ok(None);
            }
            let option = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                "create-file request omits its option",
            ))?;
            let size = request_u64(request.data, 8).ok_or(IpcWireError::Malformed(
                "create-file request omits its size",
            ))?;
            Ok(Some(IpcRequest::CreateFile {
                path: read_path(process, hipc)?,
                size,
                option,
            }))
        }
        FileSystemCommand::CreateDirectory => {
            if !matches!(object, SemanticIpcObject::HostDirectoryFileSystem(_)) {
                return Ok(None);
            }
            Ok(Some(IpcRequest::CreateDirectory {
                path: read_path(process, hipc)?,
            }))
        }
        // IFileSystem OpenFile/OpenDirectory use one input pointer path and a
        // u32 mode:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L878-L893
        FileSystemCommand::OpenFile => Ok(Some(IpcRequest::OpenFile {
            path: read_path(process, hipc)?,
            mode: request_u32(request.data, 0)
                .ok_or(IpcWireError::Malformed("open-file request omits its mode"))?,
        })),
        FileSystemCommand::OpenDirectory => Ok(Some(IpcRequest::OpenDirectory {
            path: read_path(process, hipc)?,
            mode: request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                "open-directory request omits its mode",
            ))?,
        })),
    }
}

fn read_path(
    process: &ExceptionProcessContext<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<String, IpcWireError> {
    enum InputDescriptor {
        Static(SendStaticDescriptor),
        Buffer(BufferDescriptor),
    }

    let descriptor = match (hipc.send_statics.as_slice(), hipc.send_buffers.as_slice()) {
        ([descriptor], []) => InputDescriptor::Static(*descriptor),
        ([], [descriptor]) => InputDescriptor::Buffer(*descriptor),
        _ => {
            return Err(IpcWireError::Malformed(
                "filesystem path requires exactly one input descriptor",
            ));
        }
    };
    let (address, size) = match descriptor {
        InputDescriptor::Static(descriptor) => (descriptor.address, usize::from(descriptor.size)),
        InputDescriptor::Buffer(descriptor) => {
            if descriptor.mode == BufferMode::Invalid {
                return Err(IpcWireError::Malformed(
                    "filesystem path buffer has an invalid mapping mode",
                ));
            }
            (
                descriptor.address,
                usize::try_from(descriptor.size)
                    .map_err(|_| IpcWireError::Malformed("filesystem path buffer is too large"))?,
            )
        }
    };
    if size == 0 || size > FS_MAX_PATH || size > MAX_IPC_PATH_BYTES + 1 {
        return Err(IpcWireError::Malformed(
            "filesystem path descriptor has an invalid size",
        ));
    }

    let mut bytes = vec![0; size];
    read_bytes(process, GuestVirtualAddress::new(address), &mut bytes)?;
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(IpcWireError::Malformed(
            "filesystem path is not null terminated",
        ))?;
    String::from_utf8(bytes[..nul].to_vec())
        .map_err(|_| IpcWireError::Malformed("filesystem path is not UTF-8"))
}
