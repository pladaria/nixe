//! Address-space file descriptions retain their identity across cloned sessions
//! and in-flight ioctls, independently of closing the numeric descriptor.
use super::*;
use std::sync::{Arc, Mutex};

pub(super) fn ioctl(
    session: &crate::NvDrv2Session,
    binding: &Arc<Mutex<Option<u32>>>,
    ioctl: u32,
    input: &[u8],
    output: &mut [u8],
    extra: &mut [u8],
) -> Result<(), IpcWireError> {
    let mut binding = binding.lock().unwrap();
    let id = *binding;
    match (id, ioctl) {
        (Some(id), 0x40044101) if input.len() == 4 && output.is_empty() => {
            let target = request_u32(input, 0).unwrap();
            let files = session.state.lock().unwrap();
            let Some(NvDrv2Device::ChannelGpu(channel)) = files.files.get(&target).cloned() else {
                return Err(IpcWireError::Malformed("AS bind requires a channel fd"));
            };
            drop(files);
            let mut channel = channel.lock().unwrap();
            if channel.space.is_some() {
                return Err(IpcWireError::Malformed(
                    "GPU channel already has an address space",
                ));
            }
            channel.space = Some(id);
        }
        (Some(_), 0xc0404108) if input.len() == 64 && output.len() == 64 => {
            // https://switchbrew.org/wiki/NV_services#NVGPU_AS_IOCTL_GET_VA_REGIONS
            // AllocAsEx selected a single 4-KiB partition. The final page
            // belongs to the driver syncpoint mapping, not user allocation.
            output.copy_from_slice(input);
            output[8..12].copy_from_slice(&24_u32.to_le_bytes());
            output[16..].fill(0);
            output[16..24].copy_from_slice(&crate::gsv::gpu::VA_START.to_le_bytes());
            output[24..28].copy_from_slice(&4096_u32.to_le_bytes());
            let pages =
                (crate::gsv::gpu::VA_END - crate::gsv::gpu::VA_START) / nixe_memory::PAGE_SIZE - 1;
            output[32..40].copy_from_slice(&pages.to_le_bytes());
            if !extra.is_empty() {
                if extra.len() < 24 {
                    return Err(IpcWireError::Malformed("VA regions output too small"));
                }
                extra[..24].copy_from_slice(&output[16..40]);
            }
        }
        (Some(id), 0x8010410b) if input.is_empty() && output.len() == 16 => {
            // https://switchbrew.org/wiki/NV_services#NVGPU_AS_IOCTL_GET_SYNC_RO_MAP
            // @SPECULATIVE: a page at the end of the VA partition exposes
            // 192 counters with a four-byte stride, as in the reference.
            // This is retained physical memory, not an address intercepted
            // with fabricated counter values. Engine completion must publish
            // counters here before it can release dependent GPU work.
            let mut owner = session.owner.state.lock().unwrap();
            if owner.gpu.as_ref().is_some_and(|gpu| gpu.executor.is_some())
                && owner
                    .syncpt
                    .as_ref()
                    .is_some_and(|sync| sync.gpu_values.is_none())
            {
                return Err(IpcWireError::GraphicsService("late GPU syncpoint page initialization requires an ordered counter snapshot and is not implemented".into()));
            }
            let account = match &owner.boot {
                Some(crate::gsv::GsvBoot::Remote { account, .. }) => account.clone(),
                _ => {
                    return Err(IpcWireError::Malformed(
                        "syncpoint GPU mapping needs remote memory",
                    ));
                }
            };
            let syncpt = owner
                .syncpt
                .as_mut()
                .filter(|s| s.initialized)
                .ok_or(IpcWireError::Malformed("syncpoint context missing"))?;
            if syncpt.gpu_values.is_none() {
                syncpt.gpu_values = Some(
                    account
                        .allocate(nixe_memory::PAGE_SIZE)
                        .map_err(|e| IpcWireError::SystemMemory(std::sync::Arc::new(e)))?,
                );
            }
            let values = syncpt.gpu_values.as_ref().unwrap();
            let ranges = vec![
                nixe_memory::AllocationRange::new(
                    values
                        .pin_range(0, nixe_memory::PAGE_SIZE)
                        .map_err(|e| IpcWireError::SystemMemory(std::sync::Arc::new(e)))?,
                    0,
                    nixe_memory::PAGE_SIZE,
                )
                .map_err(|e| IpcWireError::SystemMemory(std::sync::Arc::new(e)))?,
            ];
            let space = owner
                .gpu
                .as_mut()
                .and_then(|g| g.spaces.get_mut(&id))
                .ok_or(IpcWireError::Malformed("GPU address space missing"))?;
            let mut space = space.lock().unwrap();
            let address = crate::gsv::gpu::VA_END - nixe_memory::PAGE_SIZE;
            if space.sync_mapping.is_none() {
                space.sync_mapping = Some(
                    crate::gsv::gpu::PhysicalMap::new(&space.memory, ranges)
                        .map_err(|e| IpcWireError::SystemMemory(std::sync::Arc::new(e)))?,
                );
            }
            output[..8].copy_from_slice(&address.to_le_bytes());
            output[8..12].copy_from_slice(&4_u32.to_le_bytes());
            output[12..16].copy_from_slice(&crate::gsv::SYNCPOINT_COUNT.to_le_bytes());
        }
        (None, 0x40284109) if input.len() == 40 && output.is_empty() => {
            // https://switchbrew.org/wiki/NV_services#NVGPU_AS_IOCTL_ALLOC_AS_EX
            // @SPECULATIVE: with flags 5, Switch 2 reuses the as_fd word
            // for the GSV address-space ID (observed here as the ID
            // returned by CreateGpuAddressSpace). It does not create an
            // independent mapping table or broadcast maps to other ASes.
            let id = request_u32(input, 4).unwrap();
            let owner = session.owner.state.lock().unwrap();
            let gpu = owner
                .gpu
                .as_ref()
                .ok_or(IpcWireError::Malformed("GPU context missing"))?;
            if request_u32(input, 0) != Some(5)
                || request_u64(input, 8) != Some(0)
                || request_u64(input, 16) != Some(crate::gsv::gpu::VA_START)
                || request_u64(input, 24) != Some(crate::gsv::gpu::VA_END)
                || request_u64(input, 32) != Some(0)
                || !gpu.spaces.contains_key(&id)
            {
                return Err(IpcWireError::Malformed(
                    "unsupported NvDrv2 GPU address-space initialization",
                ));
            }
            drop(owner);
            *binding = Some(id);
        }
        _ => {
            return Err(IpcWireError::GraphicsService(
                format!(
                    "NvDrv2 unsupported address-space ioctl: binding={id:?} \
                request={ioctl:#010x} input_bytes={} output_bytes={}",
                    input.len(),
                    output.len()
                )
                .into_boxed_str(),
            ));
        }
    }
    Ok(())
}
