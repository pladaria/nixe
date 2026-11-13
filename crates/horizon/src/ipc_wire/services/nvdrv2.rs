//! NvDrv2's CMIF boundary. Device state belongs to the Switch 2 graphics owner.
//! https://switchbrew.org/wiki/Switch_2%3A_GSV_services#INvDrv2User
use super::prelude::*;
use crate::gsv::NvDrv2Device;
mod address_space;
mod channel;

pub(in crate::ipc_wire) async fn dispatch_nvdrv2(
    process: &mut Request,
    session: &crate::NvDrv2Session,
    request: CmifRequest<'_>,
    hipc: &HipcRequest<'_>,
) -> Result<(Vec<u8>, Option<u32>), IpcWireError> {
    if process.process_id() != session.owner.process_id {
        return Err(IpcWireError::Malformed("NvDrv2 client process mismatch"));
    }
    match request.command_id {
        0 => {
            let descriptor = one_send_buffer(process, hipc)?;
            let size = usize::try_from(descriptor.size)
                .map_err(|_| IpcWireError::Malformed("device path too long"))?;
            if size == 0 || size > 0x100 {
                return Err(IpcWireError::Malformed("invalid NvDrv2 device path length"));
            }
            process.reserve_payload(size).await?;
            let mut bytes = vec![0; size];
            process.read(descriptor, 0, &mut bytes).await?;
            let path = bytes.split(|&byte| byte == 0).next().unwrap();
            log::debug!("NvDrv2 opening {}", String::from_utf8_lossy(path));
            let device = match path {
                b"/dev/nvhost-ctrl-gpu" => NvDrv2Device::ControlGpu,
                b"/dev/nvhost-as-gpu" => NvDrv2Device::AddressSpaceGpu(std::sync::Arc::default()),
                b"/dev/nvhost-gpu" => NvDrv2Device::ChannelGpu(std::sync::Arc::new(
                    std::sync::Mutex::new(crate::gsv::channel::Channel::default()),
                )),
                b"/dev/nvhost-tsg-gpu" => NvDrv2Device::TsgGpu(std::sync::Arc::new(
                    std::sync::Mutex::new(crate::gsv::Tsg::default()),
                )),
                b"/dev/nvhost-ctrl" => NvDrv2Device::Control,
                _ => return Err(IpcWireError::Malformed("unsupported NvDrv2 device path")),
            };
            let mut state = session.state.lock().unwrap();
            state.next_fd =
                state
                    .next_fd
                    .checked_add(1)
                    .ok_or(IpcWireError::HostResourceExhausted(
                        "NvDrv2 file descriptors",
                    ))?;
            let fd = state.next_fd;
            state.files.insert(fd, device);
            log::debug!("NvDrv2 opened {} as {fd}", String::from_utf8_lossy(path));
            let mut data = Vec::from(fd.to_le_bytes());
            data.extend_from_slice(&0_u32.to_le_bytes());
            semantic_success(request.token, false, &data, &[], &[], None)
        }
        1 | 11 | 12 => {
            let fd = request_u32(request.data, 0)
                .ok_or(IpcWireError::Malformed("NvDrv2 ioctl descriptor missing"))?;
            let ioctl = request_u32(request.data, 4)
                .ok_or(IpcWireError::Malformed("NvDrv2 ioctl number missing"))?;
            let device = session
                .state
                .lock()
                .unwrap()
                .files
                .get(&fd)
                .cloned()
                .ok_or(IpcWireError::Malformed("invalid NvDrv2 file descriptor"))?;
            let buffers = match request.command_id {
                11 => super::nv_ioctl::nv_ioctl2_buffers(hipc, ioctl)?,
                12 => super::nv_ioctl::nv_ioctl3_buffers(hipc, ioctl)?,
                _ => super::nv_ioctl::nv_ioctl_buffers(hipc, ioctl)?,
            };
            let size = |buffer: Option<BufferDescriptor>| -> Result<usize, IpcWireError> {
                usize::try_from(buffer.map_or(0, |buffer| buffer.size))
                    .map_err(|_| IpcWireError::Malformed("NvDrv2 buffer too large"))
            };
            let (input_size, inline_size, output_size, extra_size) = (
                size(buffers.input)?,
                size(buffers.additional_input)?,
                size(buffers.output)?,
                size(buffers.additional_output)?,
            );
            let budget = input_size
                .checked_add(inline_size)
                .and_then(|n| n.checked_add(output_size))
                .and_then(|n| n.checked_add(extra_size))
                .and_then(|n| n.checked_add(4))
                .ok_or(IpcWireError::Malformed("NvDrv2 payload overflow"))?;
            process.reserve_payload(budget).await?;
            let mut input = vec![0; input_size];
            if buffers.input.is_some() {
                process
                    .read(&process.send_buffers()[0], 0, &mut input)
                    .await?;
            }
            log::trace!(
                "NvDrv2 ioctl: device={device:?} request={ioctl:#010x} \
                input={:02x?} output_size={output_size} \
                inline_size={inline_size} extra_size={extra_size}",
                &input[..input.len().min(80)]
            );
            let mut output = vec![0; output_size];
            let mut extra = vec![0; extra_size];
            match (&device, ioctl) {
                (NvDrv2Device::ControlGpu, 0x402c4703 | 0xc0344704)
                    if inline_size == 0
                        && extra_size == 0
                        && if ioctl == 0x402c4703 {
                            input.len() == 44 && output.is_empty()
                        } else {
                            input.len() == 52 && output.len() == 52
                        } =>
                {
                    // Identical ZBC table ABI for both console generations.
                    // Retain registrations and reference counts for query calls.
                    // https://switchbrew.org/wiki/NV_services#NVGPU_GPU_IOCTL_ZBC_SET_TABLE
                    let result = {
                        let mut state = session.state.lock().unwrap();
                        if ioctl == 0x402c4703 {
                            state.zbc_tables.set(&input)
                        } else {
                            state.zbc_tables.query(&input).map(|bytes| output = bytes)
                        }
                    };
                    match result {
                        Ok(()) => (),
                        Err(crate::nvdrv::NvDrvCallError::GuestResult(code)) => {
                            return semantic_success(
                                request.token,
                                false,
                                &code.to_le_bytes(),
                                &[],
                                &[],
                                None,
                            );
                        }
                        Err(error) => {
                            return Err(IpcWireError::GraphicsService(error.to_string().into()));
                        }
                    }
                }
                (NvDrv2Device::AddressSpaceGpu(binding), _) => {
                    address_space::ioctl(session, binding, ioctl, &input, &mut output, &mut extra)?
                }
                (NvDrv2Device::ChannelGpu(channel), 0xc018481b) if request.command_id == 11 => {
                    let mut entries = vec![0; inline_size];
                    process
                        .read(&process.send_buffers()[1], 0, &mut entries)
                        .await?;
                    channel::submit(process, session, fd, channel, &input, &entries, &mut output)
                        .await?;
                }
                (NvDrv2Device::ChannelGpu(channel), _) => {
                    channel::ioctl(session, fd, channel, ioctl, &input, &mut output)?
                }
                (NvDrv2Device::TsgGpu(tsg), 0xc0045401 | 0xc018540b)
                    if input.len() == output.len()
                        && input.len() == if ioctl == 0xc0045401 { 4 } else { 24 } =>
                {
                    // https://switchbrew.org/wiki/NV_services#NVGPU_TSG_IOCTL_BIND_CHANNEL_EX
                    let target = request_u32(&input, 0).unwrap();
                    let subcontext = if input.len() == 24 {
                        request_u32(&input, 4).unwrap()
                    } else {
                        0
                    };
                    if subcontext >= 64 {
                        return Err(IpcWireError::Malformed("TSG subcontext out of range"));
                    }
                    let files = session.state.lock().unwrap();
                    let Some(NvDrv2Device::ChannelGpu(channel)) = files.files.get(&target).cloned()
                    else {
                        return Err(IpcWireError::Malformed("TSG bind requires a channel fd"));
                    };
                    drop(files);
                    let mut channel = channel.lock().unwrap();
                    if channel.group.is_some() {
                        return Err(IpcWireError::Malformed("GPU channel already grouped"));
                    }
                    channel.group = Some((tsg.clone(), subcontext));
                    output.copy_from_slice(&input);
                }
                (NvDrv2Device::TsgGpu(tsg), 0x00005403 | 0x00005404)
                    if input.is_empty() && output.is_empty() =>
                {
                    tsg.lock().unwrap().enabled = ioctl == 0x00005403;
                }
                (NvDrv2Device::TsgGpu(tsg), 0xc0085407)
                    if input.len() == 8 && output.len() == 8 =>
                {
                    let level = request_u32(&input, 0).unwrap();
                    if level > 2 {
                        return Err(IpcWireError::Malformed("invalid TSG interleave level"));
                    }
                    tsg.lock().unwrap().interleave = level;
                    output.copy_from_slice(&input);
                }
                (NvDrv2Device::TsgGpu(tsg), 0xc0045408)
                    if input.len() == 4 && output.len() == 4 =>
                {
                    tsg.lock().unwrap().timeslice_us = request_u32(&input, 0).unwrap();
                    output.copy_from_slice(&input);
                }
                (NvDrv2Device::TsgGpu(tsg), 0x8004540a)
                    if input.is_empty() && output.len() == 4 =>
                {
                    output.copy_from_slice(&tsg.lock().unwrap().timeslice_us.to_le_bytes());
                }
                (NvDrv2Device::ControlGpu, 0x80044701) if input.is_empty() && output.len() == 4 => {
                    // @SPECULATIVE: use the reference's minimum ZCULL context
                    // requirement; host rendering does not store guest ZCULL RAM.
                    output.copy_from_slice(&1_u32.to_le_bytes());
                }
                (NvDrv2Device::ControlGpu, 0x80284702)
                    if input.is_empty() && output.len() == 40 =>
                {
                    // @SPECULATIVE: ZCULL geometry follows the reference driver
                    // profile; GA10F-specific compression geometry is unknown.
                    let geometry: [u32; 10] =
                        [0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10];
                    for (word, value) in output.chunks_exact_mut(4).zip(geometry) {
                        word.copy_from_slice(&value.to_le_bytes());
                    }
                }
                (NvDrv2Device::ControlGpu, 0xc0084713)
                    if input.len() == 8 && output.len() == 8 && extra.len() == 24 =>
                {
                    // @SPECULATIVE: the Switch 2 inline ABI extends the documented
                    // GPC/TPC byte pairs to twelve SMs, two per GA10F TPC.
                    // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
                    output.copy_from_slice(&input);
                    for (sm, pair) in extra.chunks_exact_mut(2).enumerate() {
                        pair.copy_from_slice(&[0, (sm / 2) as u8]);
                    }
                }
                (NvDrv2Device::ControlGpu, 0xc0184706)
                    if input.len() == 24 && output.len() == 24 =>
                {
                    // The mask follows the reported topology, not host hardware.
                    // https://switchbrew.org/wiki/NV_services#NVGPU_GPU_IOCTL_GET_TPC_MASKS
                    if request_u32(&input, 0) != Some(4) || (!extra.is_empty() && extra.len() != 4)
                    {
                        return Err(IpcWireError::Malformed("invalid TPC mask size"));
                    }
                    output.copy_from_slice(&input);
                    output[16..24].copy_from_slice(&0x3f_u64.to_le_bytes());
                    if !extra.is_empty() {
                        extra.copy_from_slice(&0x3f_u32.to_le_bytes());
                    }
                }
                (NvDrv2Device::ControlGpu, 0xc010471f)
                    if input.len() == 16 && output.len() == 16 =>
                {
                    // https://switchbrew.org/wiki/NV_services#NVGPU_GPU_IOCTL_GET_ENGINE_INFO
                    // @SPECULATIVE: GA10F runlist assignments follow the reference.
                    // Graphics/compute share runlist 0; async copy uses runlist 1.
                    let items: [u32; 8] = [0, 0, 0, 0, 2, 0, 1, 0];
                    let requested = request_u32(&input, 0).unwrap() as usize;
                    output.copy_from_slice(&input);
                    output[..4].copy_from_slice(&32_u32.to_le_bytes());
                    let count = requested.min(32);
                    if extra.len() < count {
                        return Err(IpcWireError::Malformed("GPU engine info buffer too small"));
                    }
                    let bytes: Vec<_> = items.into_iter().flat_map(u32::to_le_bytes).collect();
                    extra[..count].copy_from_slice(&bytes[..count]);
                }
                (NvDrv2Device::ControlGpu, 0xc0e04705)
                    if input.len() >= 16 && output.len() == input.len() =>
                {
                    let requested = request_u64(&input, 0).unwrap();
                    output.copy_from_slice(&input);
                    output[..8].copy_from_slice(&0xd0_u64.to_le_bytes());
                    let properties = crate::gsv::characteristics::encode();
                    let count = requested.min(properties.len() as u64) as usize;
                    let embedded = count.min(output.len() - 16);
                    output[16..16 + embedded].copy_from_slice(&properties[..embedded]);
                    if buffers.additional_output.is_some() {
                        if extra.len() < count {
                            return Err(IpcWireError::Malformed(
                                "GPU characteristics inline buffer too small",
                            ));
                        }
                        extra[..count].copy_from_slice(&properties[..count]);
                    }
                }
                _ => {
                    return Err(IpcWireError::GraphicsService(
                        format!(
                            "NvDrv2 unsupported ioctl: device={device:?} \
                request={ioctl:#010x} input_bytes={input_size} \
                output_bytes={output_size} inline_bytes={inline_size} \
                extra_bytes={extra_size}"
                        )
                        .into_boxed_str(),
                    ));
                }
            }
            if buffers.output.is_some() {
                write_descriptor_bytes(process, &process.receive_buffers()[0], &mut output).await?;
            }
            if buffers.additional_output.is_some() {
                write_descriptor_bytes(process, &process.receive_buffers()[1], &mut extra).await?;
            }
            semantic_success(request.token, false, &0_u32.to_le_bytes(), &[], &[], None)
        }
        2 => {
            let fd = request_u32(request.data, 0).ok_or(IpcWireError::Malformed(
                "NvDrv2 close requires a descriptor",
            ))?;
            let device = {
                let mut state = session.state.lock().unwrap();
                state.events.retain(|&(owner, _), _| owner != fd);
                state.files.remove(&fd)
            };
            let error = if device.is_some() {
                0_u32
            } else {
                crate::nvdrv::NV_BAD_PARAMETER
            };
            let completion = if let Some(NvDrv2Device::ChannelGpu(channel)) = device {
                let mut channel = channel.lock().unwrap();
                channel.closed = true;
                let owner = session.owner.state.lock().unwrap();
                match (
                    channel.timeline.as_ref(),
                    owner.gpu.as_ref().and_then(|g| g.executor.as_ref()),
                ) {
                    (Some(timeline), Some(executor)) => Some(
                        executor
                            .retire(timeline.lock().unwrap().timeline.instance().get())
                            .map_err(IpcWireError::GraphicsService)?,
                    ),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(completion) = completion {
                completion
                    .wait()
                    .await
                    .map_err(|_| IpcWireError::Cancelled)?;
            }
            semantic_success(request.token, false, &error.to_le_bytes(), &[], &[], None)
        }
        4 => {
            // https://switchbrew.org/wiki/NV_services#QueryEvent
            let fd = request_u32(request.data, 0)
                .ok_or(IpcWireError::Malformed("missing QueryEvent fd"))?;
            let id = request_u32(request.data, 4)
                .ok_or(IpcWireError::Malformed("missing QueryEvent ID"))?;
            if has_ipc_descriptors(hipc) {
                return Err(IpcWireError::Malformed("QueryEvent has descriptors"));
            }
            let mut state = session.state.lock().unwrap();
            let supported = match state.files.get(&fd) {
                Some(NvDrv2Device::ControlGpu) => matches!(id, 1 | 2),
                Some(NvDrv2Device::ChannelGpu(_)) => matches!(id, 1..=3 | 7),
                _ => false,
            };
            if !supported {
                return unsupported_service_command("NvDrv2 QueryEvent", request.command_id);
            }
            // These are persistent device notification events, initially idle.
            // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.cpp
            // @SPECULATIVE: Switch 2 also exposes channel event 7; its trigger
            // is undocumented. Keep its identity and initial unsignalled state,
            // as observed in nvhost_gpu.cpp. It is not a completion fence.
            let event = state
                .events
                .entry((fd, id))
                .or_insert_with(|| nixe_runtime::EventObject::new(process.kernel()));
            let handle = process
                .return_handle(event.readable())
                .map_err(IpcWireError::Handle)?;
            semantic_success(
                request.token,
                false,
                &0_u32.to_le_bytes(),
                &[handle],
                &[],
                None,
            )
        }
        13 => {
            let configuration = request_u64(request.data, 0)
                .ok_or(IpcWireError::Malformed("NvDrv2 configuration requires u64"))?;
            if configuration & !1 != 0 || has_ipc_descriptors(hipc) {
                return unsupported_service_command("NvDrv2 SetConfiguration", request.command_id);
            }
            session.state.lock().unwrap().configuration = configuration;
            semantic_success(request.token, false, &[], &[], &[], None)
        }
        _ => unsupported_service_command("NvDrv2", request.command_id),
    }
}
