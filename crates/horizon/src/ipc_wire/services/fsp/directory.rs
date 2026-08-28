//! `IDirectory` wire command decoding.

use crate::ipc_wire::IpcWireError;
use crate::ipc_wire::buffer::one_receive_buffer;
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::{IpcRequest, MAX_IPC_LIST_ENTRIES};

use super::abi::FS_DIRECTORY_ENTRY_SIZE;
use super::commands::DirectoryCommand;

pub(super) fn decode(
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    let Some(command) = DirectoryCommand::decode(request.command_id) else {
        return Ok(None);
    };
    match command {
        // IDirectory::Read returns fixed 0x310-byte FsDirectoryEntry records
        // through one map-alias output buffer:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L1043-L1051
        DirectoryCommand::Read => {
            let descriptor = one_receive_buffer(hipc)?;
            let capacity = usize::try_from(descriptor.size)
                .ok()
                .map(|size| size / FS_DIRECTORY_ENTRY_SIZE)
                .ok_or(IpcWireError::Malformed(
                    "directory output buffer is too large",
                ))?;
            Ok(Some(IpcRequest::ReadDirectory {
                max_entries: capacity.min(MAX_IPC_LIST_ENTRIES),
            }))
        }
        DirectoryCommand::GetEntryCount => Ok(Some(IpcRequest::GetDirectoryEntryCount)),
    }
}
