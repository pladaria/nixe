//! Save-data filesystem opening and application identity validation.
use super::super::response::{encode_semantic_response, semantic_error};
use crate::ipc_wire::io::has_ipc_descriptors;
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::ipc_wire::{IpcWireError, UnsupportedServiceOperation};
use crate::{
    HorizonIpcObject, HorizonIpcResult, HostDirectoryFileSystem, IpcResponse, IpcSession,
    SaveDataSystem, SemanticIpcObject,
};
use nixe_runtime::ExceptionProcessContext;

pub(in crate::ipc_wire::services) fn open_save_data(
    process: &mut ExceptionProcessContext<'_>,
    session: &IpcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    saves: Option<&SaveDataSystem>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // Space ID followed by an aligned 0x40-byte SaveDataAttribute.
    // https://switchbrew.org/wiki/Filesystem_services#OpenSaveDataFileSystem
    // https://github.com/switchbrew/libnx/blob/master/nx/include/switch/services/fs.h
    if !request.has_payload_size(72)
        || has_ipc_descriptors(hipc)
        || request.data[1..8].iter().any(|b| *b != 0)
        || request.data[44..72].iter().any(|b| *b != 0)
    {
        return semantic_error(
            request.token,
            Some(session),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
        );
    }
    let unsupported = |detail| {
        IpcWireError::UnsupportedService(UnsupportedServiceOperation::CommandVariant {
            service: "fsp-srv",
            command_id: 51,
            detail,
        })
    };
    let saves =
        saves.ok_or_else(|| unsupported("application save-data namespace is unavailable"))?;
    let program = u64::from_le_bytes(request.data[8..16].try_into().unwrap());
    let user = u128::from_le_bytes(request.data[16..32].try_into().unwrap());
    let id = u64::from_le_bytes(request.data[32..40].try_into().unwrap());
    if request.data[0] != 1 || id != 0 || request.data[41..44].iter().any(|b| *b != 0) {
        return Err(unsupported(
            "save space, static ID, rank or index is unsupported",
        ));
    }
    if program != 0 && program != saves.program_id() {
        return Err(unsupported(
            "cross-application save-data access is unsupported",
        ));
    }
    let kind = match request.data[40] {
        1 if user != 0 => "account",
        2 if user == 0 => "device",
        _ => {
            return Err(unsupported(
                "save-data type or user identity is unsupported",
            ));
        }
    };
    let volume = match saves.open(kind, user) {
        Ok(volume) => volume,
        Err(error) => {
            let result = if error.kind() == std::io::ErrorKind::NotFound {
                HorizonIpcResult::FS_PATH_NOT_FOUND
            } else {
                log::error!("opening save data failed: {error}");
                HorizonIpcResult::FS_UNEXPECTED
            };
            return semantic_error(request.token, Some(session), result);
        }
    };
    let object =
        SemanticIpcObject::HostDirectoryFileSystem(HostDirectoryFileSystem::from_save(volume));
    let handle = process
        .handles_mut()
        .insert(HorizonIpcObject::SemanticObject(object))
        .map_err(|_| IpcWireError::Internal("save-data child handle allocation failed"))?;
    encode_semantic_response(
        process,
        Some(session),
        None,
        request,
        hipc,
        IpcResponse::Handle(handle),
    )
}
