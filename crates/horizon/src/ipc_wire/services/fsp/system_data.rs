//! OpenDataStorageByDataId for host-generated system PublicData archives.
use super::super::response::{encode_semantic_response, semantic_error};
use crate::ipc_wire::io::has_ipc_descriptors;
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::ipc_wire::{IpcWireError, UnsupportedServiceOperation};
use crate::{HorizonIpcResult, IpcDispatcher, IpcSession, SystemArchives};
use nixe_runtime::ExceptionProcessContext;

pub(in crate::ipc_wire::services) fn open_system_data(
    process: &mut ExceptionProcessContext<'_>,
    session: &IpcSession,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
    archives: &SystemArchives,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    // StorageId is a byte followed by seven alignment bytes and a u64 DataId.
    // PublicData access is exempt from the generic content-owner permission check.
    // https://switchbrew.org/wiki/Filesystem_services#OpenDataStorageByDataId
    if !request.has_payload_size(16) || has_ipc_descriptors(hipc) {
        return semantic_error(
            request.token,
            Some(session),
            HorizonIpcResult::CMIF_INVALID_IN_HEADER,
        );
    }
    let storage_id = request.data[0];
    let data_id = u64::from_le_bytes(request.data[8..16].try_into().unwrap());
    let unsupported = || {
        IpcWireError::UnsupportedService(UnsupportedServiceOperation::DataStorage {
            storage_id,
            data_id,
        })
    };
    // None searches content stores; BuiltInSystem addresses the system archive.
    if !matches!(storage_id, 0 | 3) {
        return Err(unsupported());
    }
    let storage = archives
        .open(data_id)
        .map_err(|e| IpcWireError::SystemArchive(e.into_boxed_str()))?
        .ok_or_else(unsupported)?;
    let (_, handles) = process.mounts_and_handles_mut();
    match IpcDispatcher::open_storage(handles, storage) {
        Ok(response) => {
            encode_semantic_response(process, Some(session), None, request, hipc, response)
        }
        Err(error) => semantic_error(
            request.token,
            Some(session),
            HorizonIpcResult::from_semantic(crate::IpcService::FileSystem, error),
        ),
    }
}
