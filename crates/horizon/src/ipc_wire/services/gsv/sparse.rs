//! Ordered client-backed tile mappings, with physical alias retention.
use super::*;
use crate::gsv::{execution::MemoryJob, gpu::TileMapping};

pub(super) async fn dispatch(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    // @SPECULATIVE: the command is documented, but its packed fields and
    // 40-byte request layout follow observed SDK calls in the reference.
    // https://switchbrew.org/wiki/Switch_2%3A_GSV_services#KUserWorkerSession
    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
    let data = request.data;
    if data.len() != 28
        || request_u64(data, 0) != Some(1)
        || request_u64(data, 12) != Some(40)
        || data[20] != 1
        || hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.exchange_buffers.is_empty()
        || !hipc.send_statics.is_empty()
        || !matches!(hipc.receive_statics, ReceiveStatics::None)
    {
        return Err(IpcWireError::Malformed(
            "unsupported GPU tile request layout/options",
        ));
    }
    let input = one_send_buffer(process, hipc)?;
    let output = one_receive_buffer(process, hipc)?;
    let size = usize::try_from(input.size)
        .map_err(|_| IpcWireError::Malformed("GPU tile list too large"))?;
    if !size.is_multiple_of(40) || output.size < (size / 40) as u64 {
        return Err(IpcWireError::Malformed("invalid GPU tile list/output size"));
    }
    process
        .reserve_payload(
            size.checked_add(size / 40)
                .ok_or(IpcWireError::Malformed("GPU tile payload overflow"))?,
        )
        .await?;
    let mut bytes = vec![0; size];
    process.read(input, 0, &mut bytes).await?;
    let mut kinds = Vec::with_capacity(size / 40);
    let (completion, event) = {
        let state = session.state.lock().unwrap();
        let gpu = state
            .gpu
            .as_ref()
            .ok_or(IpcWireError::Malformed("GPU context missing"))?;
        let id = request_u32(data, 8).unwrap();
        let space = gpu
            .spaces
            .get(&id)
            .cloned()
            .ok_or(IpcWireError::Malformed("unknown GPU address space"))?;
        let event = gpu
            .events
            .get(request_u32(data, 24).unwrap() as usize)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or(IpcWireError::Malformed("GPU tile event not created"))?;
        let mut tiles = Vec::with_capacity(size / 40);
        for tile in bytes.chunks_exact(40) {
            log::trace!("GSV GPU tile request: {tile:02x?}");
            if tile[16] != 0 || tile[17] > 1 || tile[18] != 0 || tile[21] != 0 {
                return Err(IpcWireError::GraphicsService(
                    format!("unsupported GPU tile protection/options: {tile:02x?}").into(),
                ));
            }
            let kind = crate::gsv::gpu::memory_kind(tile[19], tile[20])
                .map_err(IpcWireError::GraphicsService)?;
            let handle = request_u32(tile, 24).unwrap();
            let source = if handle == 0 {
                None
            } else {
                Some((
                    handle,
                    gpu.memories
                        .get(&handle)
                        .cloned()
                        .ok_or(IpcWireError::Malformed("unknown GPU tile memory handle"))?,
                ))
            };
            kinds.push(kind);
            let bytes = request_u64(tile, 8).unwrap();
            if bytes != 0 {
                tiles.push(TileMapping {
                    address: request_u64(tile, 0).unwrap(),
                    bytes,
                    source,
                    offset: request_u64(tile, 32).unwrap(),
                    kind,
                });
            }
        }
        let job = MemoryJob::Remap { space, tiles };
        let completion = if let Some(executor) = &gpu.executor {
            Some(
                executor
                    .memory(job)
                    .map_err(IpcWireError::GraphicsService)?,
            )
        } else {
            job.apply(|_, _, _, _| Ok(()))
                .map_err(IpcWireError::GraphicsService)?;
            None
        };
        (completion, event)
    };
    if let Some(completion) = completion {
        completion
            .wait()
            .await
            .map_err(|_| IpcWireError::Cancelled)?;
    }
    process.write(output, 0, &mut kinds).await?;
    event.signal();
    Ok(encode_tipc_response(
        0,
        &((size / 40) as u64).to_le_bytes(),
        &[],
        &[],
    )?)
}
