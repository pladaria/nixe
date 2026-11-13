//! NvDrv2 channel setup; allocation does not execute or complete command buffers.
//! https://switchbrew.org/wiki/NV_services#/dev/nvhost-gpu
use super::*;
use std::sync::{Arc, Mutex};

/// Accept the hardware stream at the native channel boundary. Engine execution
/// is still unsupported: do not return an incremented fence or acknowledge work.
pub(super) async fn submit(
    process: &Request,
    session: &crate::NvDrv2Session,
    fd: u32,
    channel: &Arc<Mutex<crate::gsv::channel::Channel>>,
    input: &[u8],
    entries: &[u8],
) -> Result<(), IpcWireError> {
    let count =
        request_u32(input, 8).ok_or(IpcWireError::Malformed("missing GPFIFO entry count"))?;
    let flags =
        request_u32(input, 12).ok_or(IpcWireError::Malformed("missing GPFIFO submit flags"))?;
    if entries.len() != count as usize * 8 {
        return Err(IpcWireError::Malformed(
            "GPFIFO inline entry count mismatch",
        ));
    }
    // Hardware-format entries and the Switch driver increment-value extension.
    // https://switchbrew.org/wiki/NV_services#NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2
    if flags & !0x137 != 0 || flags & 4 == 0 || flags & 0x101 == 0x101 {
        return Err(IpcWireError::Malformed("unsupported GPFIFO submit flags"));
    }
    let space_id = {
        let channel = channel.lock().unwrap();
        if !channel.entries.is_some_and(|capacity| count <= capacity) || channel.timeline.is_none()
        {
            return Err(IpcWireError::Malformed(
                "GPFIFO channel not allocated or submission exceeds capacity",
            ));
        }
        channel
            .space
            .ok_or(IpcWireError::Malformed("GPU channel has no address space"))?
    };
    // Wait commands must enter the ordered engine scheduler, not block this
    // service worker or fetch commands before the dependency is satisfied.
    if flags & 1 != 0 {
        return Err(IpcWireError::GraphicsService(
            format!(
                "Ampere engine execution not implemented: channel={fd} \
                address_space={space_id} dependency=({}, {})",
                request_u32(input, 16).unwrap(),
                request_u32(input, 20).unwrap()
            )
            .into_boxed_str(),
        ));
    }
    let decoded =
        crate::gsv::channel::decode_entries(entries).map_err(IpcWireError::GraphicsService)?;
    let captured = {
        let owner = session.owner.state.lock().unwrap();
        let space = owner
            .gpu
            .as_ref()
            .and_then(|gpu| gpu.spaces.get(&space_id))
            .ok_or(IpcWireError::Malformed("GPU address space has retired"))?;
        decoded
            .iter()
            .map(|entry| {
                space
                    .capture(entry.address, u64::from(entry.words) * 4)
                    .map(Arc::new)
                    .map_err(|error| {
                        IpcWireError::GraphicsService(
                            format!(
                                "GPFIFO input: channel={fd} address_space={space_id} \
                address={:#x} words={} error={error}",
                                entry.address, entry.words
                            )
                            .into_boxed_str(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let Some((entry, source)) = decoded.first().zip(captured.first()) else {
        return Err(IpcWireError::GraphicsService(
            format!(
                "Ampere empty submission/fence execution not implemented: \
                channel={fd} address_space={space_id} flags={flags:#x}"
            )
            .into_boxed_str(),
        ));
    };
    let machine = process.memory().space().machine();
    let bytes = match machine
        .read_captured_owned(&source.ranges, 0, vec![0; 4])
        .map_err(|error| IpcWireError::SystemMemory(Arc::new(error)))?
    {
        nixe_memory::Access::Ready(bytes) => bytes,
        nixe_memory::Access::Pending(pending) => {
            pending.retain_owner(source.clone());
            process.retain_payload(&pending);
            match pending.await.map_err(IpcWireError::SystemMemory)? {
                nixe_memory::AccessResult::Read(bytes) => bytes,
                _ => return Err(IpcWireError::Internal("invalid GPU input completion")),
            }
        }
    };
    let header = u32::from_le_bytes(bytes.try_into().unwrap());
    Err(IpcWireError::GraphicsService(
        format!(
            "Ampere engine execution not implemented: channel={fd} \
                address_space={space_id} entries={count} flags={flags:#x} \
                gpfifo_entry=0 pushbuffer={:#x} words={} header={header:#010x} \
                subchannel={} method={:#x}; command memory resolved and read \
                coherently; no completion published",
            entry.address,
            entry.words,
            (header >> 13) & 7,
            (header & 0xfff) * 4
        )
        .into_boxed_str(),
    ))
}

pub(super) fn ioctl(
    session: &crate::NvDrv2Session,
    fd: u32,
    channel: &Arc<Mutex<crate::gsv::channel::Channel>>,
    ioctl: u32,
    input: &[u8],
    output: &mut [u8],
) -> Result<(), IpcWireError> {
    let mut channel = channel.lock().unwrap();
    match (ioctl, input.len(), output.len()) {
        (0x40044820, 4, 0) => {
            let runlist = request_u32(input, 0).unwrap();
            if runlist > 1 || channel.runlist.is_some() {
                return Err(IpcWireError::Malformed("invalid GPU channel runlist"));
            }
            channel.runlist = Some(runlist);
        }
        (0xc020481a, 32, 32) => {
            let entries = request_u32(input, 0).unwrap();
            if entries == 0
                || !entries.is_power_of_two()
                || channel.runlist.is_none()
                || channel.entries.is_some()
            {
                return Err(IpcWireError::Malformed("invalid GPFIFO allocation"));
            }
            // Bit 1 requests deterministic submission; num_jobs sizes the
            // driver's in-flight job pool. Neither flag completes guest work.
            // NVIDIA L4T UAPI, nvgpu_alloc_gpfifo_ex_args:
            // https://github.com/OE4T/linux-nvgpu/blob/patches-rel-36/include/uapi/linux/nvgpu.h
            let jobs = request_u32(input, 4).unwrap();
            let flags = request_u32(input, 8).unwrap();
            if flags & !2 != 0 || (flags & 2 != 0 && jobs == 0) {
                return Err(IpcWireError::Malformed(
                    "unsupported GPFIFO allocation flags/job pool",
                ));
            }
            let mut owner = session.owner.state.lock().unwrap();
            let sync = owner
                .syncpt
                .as_mut()
                .ok_or(IpcWireError::Malformed("syncpoint context missing"))?;
            let id = (1..crate::gsv::SYNCPOINT_COUNT)
                .find(|id| {
                    sync.timelines
                        .get(id)
                        .is_none_or(|t| Arc::strong_count(t) == 1)
                })
                .ok_or(IpcWireError::HostResourceExhausted("GPU channel counters"))?;
            let value = sync
                .timelines
                .get(&id)
                .map_or(nixe_gpu::GuestSyncpointValue::new(0), |t| {
                    t.lock().unwrap().current_point().value()
                });
            sync.next_timeline =
                sync.next_timeline
                    .checked_add(1)
                    .ok_or(IpcWireError::HostResourceExhausted(
                        "GPU timeline lifetimes",
                    ))?;
            let timeline = Arc::new(Mutex::new(nixe_gpu::GuestTimeline::new(
                nixe_gpu::GuestSyncpointId::new(id),
                nixe_gpu::TimelineInstanceId::new(sync.next_timeline),
                nixe_gpu::TimelineOwnerId::new(u64::from(fd)),
                value,
            )));
            sync.timelines.insert(id, timeline.clone());
            channel.timeline = Some(timeline);
            channel.entries = Some(entries);
            channel.inflight_jobs = jobs;
            channel.deterministic = flags & 2 != 0;
            output.copy_from_slice(input);
            output[12..16].copy_from_slice(&id.to_le_bytes());
            output[16..20].copy_from_slice(&value.get().to_le_bytes());
        }
        (0xc0104809, 16, 16) => {
            let class = request_u32(input, 0).unwrap();
            if !matches!(class, 0x902d | 0xc797 | 0xc7c0 | 0xc56f | 0xa140 | 0xc7b5)
                || request_u32(input, 4).unwrap() & !1 != 0
            {
                return Err(IpcWireError::Malformed(
                    "unsupported Ampere object class/flags",
                ));
            }
            if !channel.objects.insert(class) {
                return Err(IpcWireError::Malformed(
                    "GPU object context already allocated",
                ));
            }
            output.copy_from_slice(input);
            output[8..16].copy_from_slice(&u64::from(class).to_le_bytes());
        }
        (0x40044803, 4, 0) => channel.timeout_ms = request_u32(input, 0).unwrap(),
        (0xc004481d, 4, 4) => {
            channel.timeslice_us = request_u32(input, 0).unwrap();
            output.copy_from_slice(input);
        }
        (0x4004480d, 4, 0) => {
            let priority = request_u32(input, 0).unwrap();
            if !matches!(priority, 0x32 | 0x64 | 0x96) {
                return Err(IpcWireError::Malformed("unsupported GPU channel priority"));
            }
            channel.priority = priority;
        }
        (0x40084714, 8, 0) => channel.client_data = request_u64(input, 0).unwrap(),
        (0x80084715, 0, 8) => output.copy_from_slice(&channel.client_data.to_le_bytes()),
        (0xc018480c, 24, 24) => {
            channel.error_notifier = request_u32(input, 16).unwrap() != 0;
            output.copy_from_slice(input);
        }
        (0xc008481f, 8, 8) => {
            channel.preemption = [
                request_u32(input, 0).unwrap(),
                request_u32(input, 4).unwrap(),
            ];
            output.copy_from_slice(input);
        }
        (0xc010480b, 16, 16) => {
            let mode = request_u32(input, 8).unwrap();
            if mode > 3 {
                return Err(IpcWireError::Malformed("invalid ZCULL binding mode"));
            }
            channel.zcull = Some((request_u64(input, 0).unwrap(), mode));
            output.copy_from_slice(input);
        }
        _ => {
            return Err(IpcWireError::GraphicsService(
                format!(
                    "NvDrv2 unsupported channel ioctl: fd={fd} request={ioctl:#010x} \
                input_bytes={} output_bytes={}",
                    input.len(),
                    output.len()
                )
                .into_boxed_str(),
            ));
        }
    }
    Ok(())
}
