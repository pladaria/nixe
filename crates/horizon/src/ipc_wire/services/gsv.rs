//! GSV uses TIPC directly; CMIF domains never wrap its sub-sessions.
use super::prelude::*;
mod display;
mod gpu;
mod syncpt;
use crate::ipc_wire::message::{TipcRequest, encode_tipc_response};

pub(in crate::ipc_wire) async fn dispatch_gsv(
    process: &mut Request,
    session: &crate::GsvSession,
    request: TipcRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<Vec<u8>, IpcWireError> {
    if process.process_id() != session.process_id {
        return Err(IpcWireError::Malformed(
            "GSV session belongs to another process",
        ));
    }
    match session.interface {
        crate::gsv::GsvInterface::User => {}
        crate::gsv::GsvInterface::Display => {
            return display::dispatch(process, session, request, hipc);
        }
        crate::gsv::GsvInterface::Gpu | crate::gsv::GsvInterface::Worker => {
            return gpu::dispatch(process, session, request, hipc);
        }
        crate::gsv::GsvInterface::Syncpt => {
            return syncpt::dispatch(process, session, request, hipc);
        }
    }
    match request.command_id {
        1 | 4 => {
            // https://switchbrew.org/wiki/Switch_2%3A_GSV_services#BootWithRemoteMemory
            let size = if request.command_id == 4 { 48 } else { 16 };
            if request.data.len() != size
                || hipc.copy_handles.len() != 1
                || hipc.pid.is_some()
                || !hipc.move_handles.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || request_u64(request.data, 0) != Some(process.process_id())
            {
                return Err(IpcWireError::Malformed("invalid GSV boot request"));
            }
            let boot = if request.command_id == 4 {
                let owner = process
                    .copy_handle::<nixe_runtime::RunnableProcess>(0)
                    .filter(|owner| owner.process_id() == session.process_id)
                    .ok_or(IpcWireError::Malformed(
                        "GSV boot requires its client's process handle",
                    ))?;
                let version = request_u64(request.data, 8).unwrap();
                if !matches!(version, 0x8000_0000_0000_0001 | 0x8000_0000_0000_0005)
                    || request.data[16..40].iter().any(|&byte| byte != 0)
                {
                    return Err(IpcWireError::UnsupportedService(
                        UnsupportedServiceOperation::CommandVariant {
                            service: "GSV UserSession",
                            command_id: request.command_id,
                            detail: "unverified BootWithRemoteMemory version/options",
                        },
                    ));
                }
                let usage = request_u64(request.data, 40).unwrap();
                if usage != owner.remote_memory().used_bytes()
                    && usage != owner.remote_memory().peak_bytes()
                {
                    return Err(IpcWireError::Malformed(
                        "GSV remote-memory usage disagrees with its owner",
                    ));
                }
                crate::gsv::GsvBoot::Remote {
                    account: owner.remote_memory().clone(),
                    version,
                }
            } else {
                let memory = process
                    .copy_handle::<nixe_runtime::TransferMemoryObject>(0)
                    .filter(|memory| Some(memory.size()) == request_u64(request.data, 8))
                    .ok_or(IpcWireError::Malformed(
                        "invalid GSV transfer-memory handle or size",
                    ))?;
                crate::gsv::GsvBoot::Transfer(memory)
            };
            let mut state = session.state.lock().unwrap();
            if state.boot.is_some() {
                return Err(IpcWireError::Malformed("GSV session already booted"));
            }
            state.boot = Some(boot);
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        101 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("GSV CreateContext has no inputs"));
            }
            let mut state = session.state.lock().unwrap();
            if state.process.is_none() || state.context_created {
                return Err(IpcWireError::Malformed("invalid GSV context lifecycle"));
            }
            state.context_created = true;
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        11 => {
            if !request.data.is_empty()
                || hipc.copy_handles.len() != 1
                || !hipc.move_handles.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
            {
                return Err(IpcWireError::Malformed(
                    "invalid GSV SetCurrentProcessHandle request",
                ));
            }
            let owner = process
                .copy_handle::<nixe_runtime::RunnableProcess>(0)
                .filter(|owner| owner.process_id() == session.process_id)
                .ok_or(IpcWireError::Malformed(
                    "GSV requires its client's process handle",
                ))?;
            let mut state = session.state.lock().unwrap();
            if state.boot.is_none() || state.process.is_some() {
                return Err(IpcWireError::Malformed(
                    "GSV process handle already registered",
                ));
            }
            state.process = Some(owner);
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        301 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("display context has no inputs"));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created || state.display.is_some() {
                return Err(IpcWireError::Malformed("invalid display context lifecycle"));
            }
            state.display = Some(crate::gsv::display::DisplayContext::default());
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        305 => {
            // @SPECULATIVE: 3.1 is documented; observed 3.2 clients use the
            // same native display session with additional commands.

            if request.data.len() != 8
                || !matches!(
                    request_u64(request.data, 0),
                    Some(0x300010000 | 0x300020000)
                )
                || has_ipc_descriptors(hipc)
            {
                return Err(IpcWireError::Malformed(
                    "unsupported display session version",
                ));
            }
            let mut state = session.state.lock().unwrap();
            let display = state
                .display
                .as_mut()
                .filter(|display| !display.initialized)
                .ok_or(IpcWireError::Malformed(
                    "display context not created or already initialized",
                ))?;
            let handle = process
                .return_service(HorizonIpcObject::Gsv(
                    session.child(crate::gsv::GsvInterface::Display),
                ))
                .map_err(IpcWireError::Handle)?;
            display.initialized = true;
            Ok(encode_tipc_response(0, &[], &[handle], &[])?)
        }
        1101 | 1102 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "graphics-frame context has no inputs",
                ));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created {
                return Err(IpcWireError::Malformed("GSV context missing"));
            }
            if request.command_id == 1101 {
                if state.frame.is_some() {
                    return Err(IpcWireError::Malformed(
                        "graphics-frame context already exists",
                    ));
                }
                state.frame = Some(crate::gsv::GraphicsFrame::default());
            } else {
                let frame = state
                    .frame
                    .as_mut()
                    .filter(|frame| !frame.initialized)
                    .ok_or(IpcWireError::Malformed(
                        "invalid graphics-frame initialization",
                    ))?;
                frame.initialized = true;
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        1111 => {
            if request.data.len() != 4 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("invalid frame-length mode"));
            }
            let mut state = session.state.lock().unwrap();
            let frame = state
                .frame
                .as_mut()
                .filter(|frame| frame.initialized)
                .ok_or(IpcWireError::Malformed(
                    "graphics-frame context not initialized",
                ))?;
            // The policy is retained for the display owner; setting it does not
            // submit work or signal a presentation completion.
            frame.length_mode = request_u32(request.data, 0).unwrap();
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        115 => {
            if request.data.len() != 8 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "invalid native memory boot request",
                ));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created || state.native_memory_features.is_some() {
                return Err(IpcWireError::Malformed("invalid native memory lifecycle"));
            }
            // @SPECULATIVE: bit 1 selects the native-memory mode requested by
            // observed SDKs. Only this mode is granted; other feature bits are
            // not implemented. The reference echoes all requested features.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nx2_gsv.h
            let granted = request_u64(request.data, 0).unwrap() & 2;
            state.native_memory_features = Some(granted);
            Ok(encode_tipc_response(0, &granted.to_le_bytes(), &[], &[])?)
        }
        497 => {
            // @SPECULATIVE: version 2 is documented for 19.0.0; observed newer
            // clients request version 4. The reference uses the same session
            // contracts for both, with newer job command IDs.

            if request.data.len() != 8
                || !matches!(request_u64(request.data, 0), Some(2 | 4))
                || has_ipc_descriptors(hipc)
            {
                return Err(IpcWireError::Malformed("unsupported GPU session version"));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created
                || state.gpu.is_some()
                || !matches!(state.boot, Some(crate::gsv::GsvBoot::Remote { .. }))
            {
                return Err(IpcWireError::Malformed("invalid GPU session lifecycle"));
            }
            let handle = process
                .return_service(HorizonIpcObject::Gsv(
                    session.child(crate::gsv::GsvInterface::Gpu),
                ))
                .map_err(IpcWireError::Handle)?;
            state.gpu = Some(crate::gsv::gpu::GpuContext::default());
            Ok(encode_tipc_response(0, &[], &[handle], &[])?)
        }
        401..=403 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("GPU context has no inputs"));
            }
            let mut state = session.state.lock().unwrap();
            let gpu = state
                .gpu
                .as_mut()
                .ok_or(IpcWireError::Malformed("GPU session missing"))?;
            match request.command_id {
                401 if !gpu.created => gpu.created = true,
                402 if gpu.created && !gpu.initialized => gpu.initialized = true,
                403 if gpu.created && gpu.memories.is_empty() && gpu.spaces.is_empty() => {
                    gpu.initialized = false;
                    gpu.created = false;
                }
                _ => return Err(IpcWireError::Malformed("invalid GPU context transition")),
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        434 => {
            if request.data.len() != 4 || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("invalid GPU address-space query"));
            }
            let id = request_u32(request.data, 0).unwrap();
            let state = session.state.lock().unwrap();
            if !state
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.spaces.contains_key(&id))
            {
                return Err(IpcWireError::Malformed("unknown GPU address space"));
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        204 => {
            if request.data.len() != 24
                || hipc.copy_handles.len() != 2
                || !hipc.move_handles.is_empty()
                || !hipc.send_buffers.is_empty()
                || !hipc.receive_buffers.is_empty()
                || !hipc.exchange_buffers.is_empty()
                || request_u32(request.data, 0) != Some(1)
            {
                return Err(IpcWireError::Malformed(
                    "invalid shared syncpoint initialization",
                ));
            }
            let rw = process
                .copy_handle::<TransferMemoryObject>(0)
                .filter(|memory| {
                    memory.size() == 0x1000
                        && memory.permissions() == nixe_memory::MemoryPermissions::READ_WRITE
                })
                .ok_or(IpcWireError::Malformed(
                    "invalid syncpoint request transfer memory",
                ))?;
            let ro = process
                .copy_handle::<TransferMemoryObject>(1)
                .filter(|memory| {
                    memory.size() >= 0x2000
                        && memory.permissions() == nixe_memory::MemoryPermissions::READ
                })
                .ok_or(IpcWireError::Malformed(
                    "invalid syncpoint value transfer memory",
                ))?;
            if request_u64(request.data, 8) != Some(rw.size())
                || request_u64(request.data, 16) != Some(ro.size())
            {
                return Err(IpcWireError::Malformed(
                    "syncpoint transfer-memory size mismatch",
                ));
            }
            let count = request_u32(request.data, 4)
                .unwrap()
                .min(crate::gsv::SYNCPOINT_COUNT);
            if count == 0 {
                return Err(IpcWireError::Malformed("empty syncpoint counter table"));
            }
            {
                let state = session.state.lock().unwrap();
                if !state
                    .syncpt
                    .as_ref()
                    .is_some_and(|context| context.initialized && context.mirror.is_none())
                {
                    return Err(IpcWireError::Malformed(
                        "invalid syncpoint mirror lifecycle",
                    ));
                }
            }
            let (request_write, request_read) =
                nixe_runtime::EventObject::create_pair(process.kernel());
            let (notify_write, notify_read) =
                nixe_runtime::EventObject::create_pair(process.kernel());
            let mirror = crate::gsv::SyncptMirror::create(
                process.lifetime().client().process(),
                rw,
                ro,
                count,
                request_read,
                notify_write,
            )
            .await
            .map_err(|detail| {
                IpcWireError::GraphicsService(
                    format!("GSV syncpoint initialization: {detail}").into_boxed_str(),
                )
            })?;
            let request_handle = process
                .return_handle(request_write)
                .map_err(IpcWireError::Handle)?;
            let notify_handle = process
                .return_handle(notify_read)
                .map_err(IpcWireError::Handle)?;
            session
                .state
                .lock()
                .unwrap()
                .syncpt
                .as_mut()
                .unwrap()
                .mirror = Some(mirror);
            Ok(encode_tipc_response(
                0,
                &[],
                &[request_handle, notify_handle],
                &[],
            )?)
        }
        201 | 202 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("syncpoint context has no inputs"));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created {
                return Err(IpcWireError::Malformed("GSV context not created"));
            }
            if request.command_id == 201 {
                if state.syncpt.is_some() {
                    return Err(IpcWireError::Malformed("syncpoint context already exists"));
                }
                state.syncpt = Some(crate::gsv::SyncptContext::default());
                Ok(encode_tipc_response(0, &[], &[], &[])?)
            } else {
                let syncpt = state
                    .syncpt
                    .as_mut()
                    .ok_or(IpcWireError::Malformed("syncpoint context missing"))?;
                if syncpt.initialized {
                    return Err(IpcWireError::Malformed(
                        "syncpoint context already initialized",
                    ));
                }
                let handle = process
                    .return_service(HorizonIpcObject::Gsv(
                        session.child(crate::gsv::GsvInterface::Syncpt),
                    ))
                    .map_err(IpcWireError::Handle)?;
                syncpt.initialized = true;
                Ok(encode_tipc_response(0, &[], &[handle], &[])?)
            }
        }
        1402 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "NvDrv2 remote session has no inputs",
                ));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created
                || !matches!(state.boot, Some(crate::gsv::GsvBoot::Remote { .. }))
            {
                return Err(IpcWireError::Malformed(
                    "NvDrv2 requires a remote GSV context",
                ));
            }
            if state.nvdrv.upgrade().is_some() {
                return Ok(encode_tipc_response(0x200f3, &[], &[], &[])?);
            }
            let client =
                std::sync::Arc::new(std::sync::Mutex::new(crate::gsv::NvDrv2State::default()));
            let handle = process
                .return_service(HorizonIpcObject::NvDrv2(crate::NvDrv2Session {
                    owner: session.clone(),
                    state: client.clone(),
                }))
                .map_err(IpcWireError::Handle)?;
            state.nvdrv = std::sync::Arc::downgrade(&client);
            Ok(encode_tipc_response(0, &[], &[handle], &[])?)
        }
        1501 | 1502 => {
            if !request.data.is_empty() || has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed(
                    "image-exchange context has no inputs",
                ));
            }
            let mut state = session.state.lock().unwrap();
            if !state.context_created {
                return Err(IpcWireError::Malformed("GSV context has not been created"));
            }
            if request.command_id == 1501 {
                if state.inter_module_images.is_some() {
                    return Err(IpcWireError::Malformed(
                        "image-exchange context already exists",
                    ));
                }
                state.inter_module_images = Some(crate::gsv::InterModuleImages::default());
            } else {
                let context = state
                    .inter_module_images
                    .as_mut()
                    .ok_or(IpcWireError::Malformed(
                        "image-exchange context does not exist",
                    ))?;
                if context.initialized {
                    return Err(IpcWireError::Malformed(
                        "image-exchange context already initialized",
                    ));
                }
                context.initialized = true;
            }
            Ok(encode_tipc_response(0, &[], &[], &[])?)
        }
        _ => Err(IpcWireError::UnsupportedService(
            UnsupportedServiceOperation::Command {
                service: "GSV UserSession",
                command_id: request.command_id,
            },
        )),
    }
}
