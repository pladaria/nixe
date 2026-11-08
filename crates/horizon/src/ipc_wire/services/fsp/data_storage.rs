//! OpenDataStorageByDataId for installed add-ons and system PublicData archives.
use super::super::response::{encode_semantic_response, semantic_error};
use crate::ipc_wire::io::has_ipc_descriptors;
use crate::ipc_wire::message::{CmifRequest, HipcRequest};
use crate::ipc_wire::request::Request;
use crate::ipc_wire::{IpcWireError, UnsupportedServiceOperation};
use crate::{
    HorizonIpcResult, IpcResponse, IpcSession, ReadOnlyStorage, SemanticIpcObject, SystemArchives,
};

pub(in crate::ipc_wire::services) async fn open_data_storage(
    process: &mut Request,
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
    // Resolve actual Data content before synthesizing a system archive, as in
    // Eden's FSP_SRV::OpenDataStorageByDataId. The loader has already resolved
    // these add-on RomFS views and retains their physical source for the caller.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/filesystem/fsp/fsp_srv.cpp
    let installed = (storage_id == 0)
        .then(|| {
            process
                .mounts()
                .add_ons()
                .iter()
                .find(|content| content.title_id().get() == data_id)
                .and_then(|content| content.mounts().first())
                .map(|mount| mount.romfs().storage())
        })
        .flatten();
    let storage = match installed {
        Some(storage) => storage,
        None => archives
            .open(data_id)
            .map_err(|e| IpcWireError::SystemArchive(e.into_boxed_str()))?
            .ok_or_else(unsupported)?,
    };
    let response = IpcResponse::Object(SemanticIpcObject::ReadOnlyStorage(ReadOnlyStorage::new(
        storage,
    )));
    encode_semantic_response(process, Some(session), None, request, hipc, response).await
}
