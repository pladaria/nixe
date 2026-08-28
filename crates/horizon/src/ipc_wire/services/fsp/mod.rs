//! Wire command decoding for `fsp-srv` and its child objects.

pub(super) mod abi;
mod commands;
mod directory;
mod file;
mod filesystem;
mod storage;

use nixe_runtime::ExceptionProcessContext;

use crate::ipc_wire::IpcWireError;
use crate::ipc_wire::io::has_ipc_descriptors;
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::{IpcRequest, SemanticIpcObject};

use commands::FileSystemProxyCommand;

pub(in crate::ipc_wire) fn object_name(object: &SemanticIpcObject) -> &'static str {
    match object {
        SemanticIpcObject::ReadOnlyFileSystem(_) => "IFileSystem(read-only)",
        SemanticIpcObject::ReadOnlyStorage(_) => "IStorage(read-only)",
        SemanticIpcObject::HostDirectoryFileSystem(_) => "IFileSystem(sd-card)",
        SemanticIpcObject::ReadOnlyFile(_) => "IFile(read-only)",
        SemanticIpcObject::HostFile(_) => "IFile(sd-card)",
        SemanticIpcObject::ReadOnlyDirectory(_) => "IDirectory",
    }
}

pub(super) fn decode_root_request(
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    let Some(command) = FileSystemProxyCommand::decode(request.command_id) else {
        return Ok(None);
    };
    match command {
        // libnx sends the current PID and a zero placeholder:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L75-L82
        FileSystemProxyCommand::SetCurrentProcess => {
            if hipc.pid.is_none() || request.data.len() < 8 {
                return Ok(None);
            }
            Ok(Some(IpcRequest::SetCurrentProcess))
        }
        FileSystemProxyCommand::OpenDataFileSystemByCurrentProcess => {
            Ok(Some(IpcRequest::OpenPrimaryFileSystem))
        }
        FileSystemProxyCommand::OpenSdCardFileSystem => Ok(Some(IpcRequest::OpenSdCardFileSystem)),
        // IFileSystemProxy::OpenDataStorageByCurrentProcess returns one
        // IStorage child object and carries no input payload:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/fs.c#L470-L472
        FileSystemProxyCommand::OpenDataStorageByCurrentProcess => {
            if has_ipc_descriptors(hipc) {
                return Ok(None);
            }
            Ok(Some(IpcRequest::OpenPrimaryStorage))
        }
        // ABI and scalar output type:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/cb4b882e3b176480ac57a1161a85ff175c3f162c/libraries/libstratosphere/include/stratosphere/fssrv/sf/fssrv_sf_i_file_system_proxy.hpp#L134-L136
        FileSystemProxyCommand::GetGlobalAccessLogMode => {
            if has_ipc_descriptors(hipc) {
                return Ok(None);
            }
            Ok(Some(IpcRequest::GetGlobalAccessLogMode))
        }
    }
}

pub(super) fn decode_object_request(
    process: &ExceptionProcessContext<'_>,
    object: &SemanticIpcObject,
    request: &CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Option<IpcRequest>, IpcWireError> {
    match object {
        SemanticIpcObject::ReadOnlyFileSystem(_)
        | SemanticIpcObject::HostDirectoryFileSystem(_) => {
            filesystem::decode(process, object, request, hipc)
        }
        SemanticIpcObject::ReadOnlyFile(_) | SemanticIpcObject::HostFile(_) => {
            file::decode(process, object, request, hipc)
        }
        SemanticIpcObject::ReadOnlyStorage(_) => storage::decode(request, hipc),
        SemanticIpcObject::ReadOnlyDirectory(_) => directory::decode(request, hipc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_wire::message::COMMAND_BUFFER_SIZE;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn decode_request(command_id: u32) -> (HipcRequest<'static>, CmifRequest<'static>) {
        let command = Box::leak(Box::new([0_u8; COMMAND_BUFFER_SIZE]));
        put_u32(command, 0, 4);
        put_u32(command, 4, 8);
        put_u32(command, 16, 0x4943_4653);
        put_u32(command, 24, command_id);
        let hipc = HipcRequest::decode(command).unwrap();
        let request = CmifRequest::decode(&hipc, false).unwrap();
        (hipc, request)
    }

    #[test]
    fn filesystem_proxy_decodes_the_sd_card_open_command() {
        let (hipc, request) = decode_request(18);
        assert_eq!(
            decode_root_request(&request, &hipc).unwrap(),
            Some(IpcRequest::OpenSdCardFileSystem)
        );
    }

    #[test]
    fn filesystem_proxy_decodes_global_access_log_mode_as_a_scalar_query() {
        let (hipc, request) = decode_request(1005);
        assert_eq!(
            decode_root_request(&request, &hipc).unwrap(),
            Some(IpcRequest::GetGlobalAccessLogMode)
        );
    }

    #[test]
    fn filesystem_proxy_decodes_current_process_data_storage_open() {
        let (hipc, request) = decode_request(200);
        assert_eq!(
            decode_root_request(&request, &hipc).unwrap(),
            Some(IpcRequest::OpenPrimaryStorage)
        );
    }
}
