//! NvDrv2 channel allocation and native command admission.
//! https://switchbrew.org/wiki/NV_services#/dev/nvhost-gpu
use super::*;
use std::sync::{Arc, Mutex};

pub(super) async fn submit(
    process: &Request,
    session: &crate::NvDrv2Session,
    fd: u32,
    channel: &Arc<Mutex<crate::gsv::channel::Channel>>,
    input: &[u8],
    entries: &[u8],
    output: &mut [u8],
) -> Result<(), IpcWireError> {
    if input.len() != 24 || output.len() != 24 {
        return Err(IpcWireError::Malformed("GPFIFO ioctl size"));
    }
    let count = request_u32(input, 8).unwrap();
    let flags = request_u32(input, 12).unwrap();
    if entries.len() != count as usize * 8
        || flags & !0x137 != 0
        || flags & 4 == 0
        || flags & 0x101 == 0x101
    {
        return Err(IpcWireError::Malformed("invalid GPFIFO entries/flags"));
    }
    if flags & 1 != 0 {
        return Err(IpcWireError::GraphicsService(
            format!(
                "native GPFIFO dependency not implemented: channel={fd} syncpoint={} value={}",
                request_u32(input, 16).unwrap(),
                request_u32(input, 20).unwrap()
            )
            .into(),
        ));
    }
    let decoded =
        nixe_gpu_ampere::decode_entries(entries).map_err(IpcWireError::GraphicsService)?;
    let channel = channel.lock().unwrap();
    if channel.closed {
        return Err(IpcWireError::Malformed("GPU channel is closed"));
    }
    if !channel.entries.is_some_and(|capacity| count <= capacity) {
        return Err(IpcWireError::Malformed("GPFIFO allocation/capacity"));
    }
    let timeline = channel
        .timeline
        .clone()
        .ok_or(IpcWireError::Malformed("channel timeline missing"))?;
    let space_id = channel
        .space
        .ok_or(IpcWireError::Malformed("channel address space missing"))?;
    if channel.objects.is_empty() {
        return Err(IpcWireError::Malformed("GPU object context missing"));
    }
    let mut owner = session.owner.state.lock().unwrap();
    if owner
        .syncpt
        .as_ref()
        .is_some_and(|sync| sync.mirror_initializing)
    {
        return Err(IpcWireError::GraphicsService(
            "native submission overlaps unfinished counter mirror initialization".into(),
        ));
    }
    let publication = owner
        .syncpt
        .as_ref()
        .map(crate::gsv::SyncptContext::publication)
        .transpose()
        .map_err(|e| IpcWireError::SystemMemory(Arc::new(e)))?;
    let gpu = owner
        .gpu
        .as_mut()
        .ok_or(IpcWireError::Malformed("GPU session missing"))?;
    let space = gpu
        .spaces
        .get(&space_id)
        .cloned()
        .ok_or(IpcWireError::Malformed("GPU address space retired"))?;
    if gpu.executor.is_none() {
        gpu.executor = Some(
            session
                .owner
                .graphics
                .executor(process.lifetime().client().process())
                .map_err(IpcWireError::GraphicsService)?,
        );
    }
    let executor = gpu.executor.as_ref().unwrap().clone();
    let (completion, point) = {
        let mut counter = timeline.lock().unwrap();
        let t = &mut counter.timeline;
        // @SPECULATIVE: NvDrv2 appends increment_value increments, unlike the
        // Switch 1 in-stream extension. This follows the native driver in
        // Lichii nvhost_gpu.cpp; the native ABI is not publicly documented.
        let increments = if flags & 2 != 0 { 2_u32 } else { 0 }
            .checked_add(if flags & 0x100 != 0 {
                request_u32(input, 20).unwrap()
            } else {
                0
            })
            .ok_or(IpcWireError::Malformed("fence increment overflow"))?;
        let timeline_owner = t.owner();
        let completion = if increments == 0 {
            None
        } else {
            Some(
                t.reserve(timeline_owner, increments)
                    .map_err(|e| IpcWireError::GraphicsService(e.to_string().into()))?,
            )
        };
        (completion, t.latest_reserved_point())
    };
    let submission = executor
        .submission(space, decoded, timeline, completion, publication)
        .map_err(IpcWireError::GraphicsService)?;
    executor.submit(submission).map_err(|error| {
        if executor.stopped() {
            IpcWireError::Cancelled
        } else {
            IpcWireError::GraphicsService(error)
        }
    })?;
    drop(owner);
    output.copy_from_slice(input);
    output[12..16].fill(0);
    output[16..20].copy_from_slice(&point.syncpoint().get().to_le_bytes());
    output[20..24].copy_from_slice(&point.value().get().to_le_bytes());
    Ok(())
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
    if channel.closed {
        return Err(IpcWireError::Malformed("GPU channel is closed"));
    }
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
            let timeline = sync
                .allocate(u64::from(fd))
                .map_err(IpcWireError::GraphicsService)?;
            let (id, raw_value) = {
                let counter = timeline.lock().unwrap();
                (
                    counter.timeline.syncpoint().get(),
                    nixe_gpu::GuestSyncpointValue::new(counter.value as u32),
                )
            };
            channel.timeline = Some(timeline);
            channel.entries = Some(entries);
            channel.inflight_jobs = jobs;
            channel.deterministic = flags & 2 != 0;
            output.copy_from_slice(input);
            output[12..16].copy_from_slice(&id.to_le_bytes());
            output[16..20].copy_from_slice(&raw_value.get().to_le_bytes());
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
