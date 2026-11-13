//! The random service shares its contract between both console platforms.
//! https://switchbrew.org/wiki/SPL_services#csrng
use super::prelude::*;

pub(in crate::ipc_wire) async fn dispatch_csrng(
    process: &mut Request,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if request.command_id != 0 {
        return unsupported_service_command("csrng", request.command_id);
    }
    if hipc.pid.is_some()
        || !hipc.copy_handles.is_empty()
        || !hipc.move_handles.is_empty()
        || !hipc.send_statics.is_empty()
        || !hipc.send_buffers.is_empty()
        || !hipc.exchange_buffers.is_empty()
    {
        return cmif_error(request.token, HorizonIpcResult::CMIF_INVALID_IN_HEADER);
    }
    // CMIF's trailing bytes include the output-pointer length table, not
    // arguments to GenerateRandomBytes. AutoSelect reserves both descriptor slots.
    let output = if hipc.receive_buffers.is_empty() {
        match process.receive_statics() {
            [output] => output,
            _ => return Err(IpcWireError::Malformed("random output requires one buffer")),
        }
    } else if matches!(hipc.receive_statics, ReceiveStatics::None) {
        // @SPECULATIVE: Switch 2 clients use a map-alias output and leave a
        // nonzero alignment tail. SPL's new buffer ABI is not documented yet;
        // the reference consumes the selected output with no scalar arguments.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/spl/spl_module.cpp
        if process.cpu().platform() != nixe_cpu::platform::TargetPlatform::Switch2 {
            return Err(IpcWireError::Malformed(
                "CSRNG map-alias ABI requires Switch 2",
            ));
        }
        one_receive_buffer(process, hipc)?
    } else {
        one_auto_select_output(process, hipc)?
    };
    let size = usize::try_from(output.size)
        .map_err(|_| IpcWireError::Malformed("random output size overflows"))?;
    process.reserve_payload(size).await?;
    let mut bytes = vec![0; size];
    getrandom::fill(&mut bytes).map_err(|_| {
        IpcWireError::HostResourceExhausted("host cryptographic random source failed")
    })?;
    process.write(output, 0, &mut bytes).await?;
    semantic_success(request.token, false, &[], &[], &[], None)
}
