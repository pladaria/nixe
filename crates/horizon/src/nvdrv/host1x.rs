//! Host1x video channel control ABI. Engine execution is separate from ioctl state.
//!
//! Public ABI: https://switchbrew.org/wiki/NV_services#NVHOST_IOCTL_CHANNEL_SUBMIT
//! Cross-check: https://github.com/eden-emulator/mirror/blob/master/src/core/hle/service/nvdrv/devices/nvhost_nvdec_common.cpp

mod vic;
mod video;
mod worker;

use super::nvhost_ctrl::NvHostControl;
use super::nvmap::{NvMapHandle, NvMapObject, NvMapObjects, NvMapOwner};
use super::{
    NV_BAD_PARAMETER, NvDrvCallError, NvDrvDeviceDescriptor, NvDrvDeviceKind, NvDrvErrorContext,
    NvDrvFileDescriptor, NvDrvValidationReason, UnsupportedNvDrvOperation, input_u32,
    require_input_size, write_u32,
};
use nixe_gpu::{GuestSyncpointId, GuestSyncpointValue, GuestTimelinePoint};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct Channel {
    syncpoint: GuestSyncpointId,
    nvmap_fd: Option<NvDrvFileDescriptor>,
    timeout: u32,
    pins: BTreeMap<u64, u32>,
    packet: PacketState,
    registers: Vec<u32>,
}

#[derive(Debug, Default)]
pub(super) struct Host1x {
    channels: BTreeMap<NvDrvFileDescriptor, Channel>,
    mappings: video::Mappings,
    worker: Option<worker::Worker>,
}

impl Host1x {
    pub(super) fn open(&mut self, fd: NvDrvFileDescriptor, syncpoint: GuestSyncpointId) {
        self.channels.insert(
            fd,
            Channel {
                syncpoint,
                nvmap_fd: None,
                timeout: 0,
                pins: BTreeMap::new(),
                packet: PacketState::default(),
                registers: vec![0; 0x446],
            },
        );
    }
    pub(super) fn close(&mut self, fd: NvDrvFileDescriptor) -> Option<GuestSyncpointId> {
        let channel = self.channels.remove(&fd)?;
        if let Some(worker) = &self.worker {
            worker.close(fd);
        }
        for (object, count) in &channel.pins {
            Arc::make_mut(&mut self.mappings).retain(|_, mapping| {
                if mapping.object.id().raw() == *object {
                    mapping.references -= count;
                }
                mapping.references != 0
            });
        }
        Some(channel.syncpoint)
    }
    pub(super) fn clear(&mut self) {
        self.worker.take();
        self.channels.clear();
        Arc::make_mut(&mut self.mappings).clear();
    }
    pub(super) fn require_healthy(&self) -> Result<(), UnsupportedNvDrvOperation> {
        if let Some(worker) = &self.worker {
            worker.require_healthy()?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ioctl(
        &mut self,
        devices: &BTreeMap<NvDrvFileDescriptor, NvDrvDeviceDescriptor>,
        nvmap: &NvMapObjects,
        control: &Arc<Mutex<NvHostControl>>,
        gpu: Option<&Arc<super::gpu_executor::NvDrvGpuExecutor>>,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
    ) -> Result<Vec<u8>, NvDrvCallError> {
        self.require_healthy()
            .map_err(NvDrvCallError::Unsupported)?;
        let unsupported = || {
            NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Ioctl {
                context: NvDrvErrorContext::new(
                    descriptor.kind(),
                    request,
                    descriptor.fd(),
                    None,
                    NvDrvValidationReason::UnsupportedOperation,
                ),
            })
        };
        if request & 0xc000_ffff == 0xc000_0001 {
            return self.submit(nvmap, control, gpu, descriptor, request, input);
        }
        if matches!(request & 0xc000_ffff, 0xc000_0009 | 0xc000_000a) {
            return self.map_buffers(nvmap, descriptor, request, input);
        }
        let channel = self
            .channels
            .get_mut(&descriptor.fd())
            .ok_or_else(unsupported)?;
        let mut output = input.to_vec();
        match request {
            0x4004_4801 => {
                require_input_size(input, 4)?;
                let fd = NvDrvFileDescriptor::new(input_u32(input, 0)?);
                if !devices.get(&fd).is_some_and(|map| {
                    map.kind() == NvDrvDeviceKind::NvMap
                        && map.owner().process_id() == descriptor.owner().process_id()
                }) {
                    return Err(NV_BAD_PARAMETER.into());
                }
                channel.nvmap_fd = Some(fd);
            }
            0xc008_0002 => {
                require_input_size(input, 8)?;
                if input_u32(input, 0)? != 0 {
                    return Err(NV_BAD_PARAMETER.into());
                }
                write_u32(&mut output, 4, channel.syncpoint.get())?;
            }
            0xc008_0003 => {
                require_input_size(input, 8)?;
                // Tegra's channel waitbase zero is the public ABI's default waitbase.
                write_u32(&mut output, 4, 0)?;
            }
            0x4004_0007 => {
                require_input_size(input, 4)?;
                channel.timeout = input_u32(input, 0)?;
            }
            _ => return Err(unsupported()),
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct Mapping {
    object: NvMapObject,
    references: u32,
}

fn execution_error(
    descriptor: NvDrvDeviceDescriptor,
    request: u32,
    detail: impl Into<Box<str>>,
) -> NvDrvCallError {
    NvDrvCallError::Unsupported(UnsupportedNvDrvOperation::Host1xExecution {
        context: NvDrvErrorContext::new(
            descriptor.kind(),
            request,
            descriptor.fd(),
            None,
            NvDrvValidationReason::UnsupportedOperation,
        ),
        detail: detail.into(),
    })
}

#[derive(Debug, Default)]
struct PacketState {
    class: u32,
    offset: u32,
    count: u32,
    mask: u32,
    incrementing: bool,
    engine_method: u32,
    wait_value: u32,
}

impl PacketState {
    // Host1x CDMA opcodes, distinct from Maxwell GPFIFO packets:
    // https://github.com/eden-emulator/mirror/blob/master/src/video_core/cdma_pusher.h
    fn word(&mut self, word: u32) -> Result<Option<(u32, u32, u32)>, String> {
        if self.mask != 0 {
            let bit = self.mask.trailing_zeros();
            self.mask &= !(1 << bit);
            return Ok(Some((self.class, self.offset + bit, word)));
        }
        if self.count != 0 {
            self.count -= 1;
            let offset = self.offset;
            if self.incrementing {
                self.offset += 1;
            }
            return Ok(Some((self.class, offset, word)));
        }
        self.offset = word >> 16 & 0xfff;
        match word >> 28 {
            0 => {
                self.class = word >> 6 & 0x3ff;
                self.mask = word & 0x3f;
            }
            1 | 2 => {
                self.count = word & 0xffff;
                self.incrementing = word >> 28 == 1;
            }
            3 => self.mask = word & 0xffff,
            4 => return Ok(Some((self.class, self.offset, word & 0xfff))),
            opcode => {
                return Err(format!(
                    "unsupported CDMA opcode {opcode:#x}, word={word:#010x}"
                ));
            }
        }
        Ok(None)
    }
}

impl Host1x {
    fn submit(
        &mut self,
        nvmap: &NvMapObjects,
        control: &Arc<Mutex<NvHostControl>>,
        gpu: Option<&Arc<super::gpu_executor::NvDrvGpuExecutor>>,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
    ) -> Result<Vec<u8>, NvDrvCallError> {
        if input.len() < 16 {
            return Err(NV_BAD_PARAMETER.into());
        }
        let buffers = input_u32(input, 0)? as usize;
        let relocations = input_u32(input, 4)? as usize;
        let syncpoints = input_u32(input, 8)? as usize;
        let fences = input_u32(input, 12)? as usize;
        let buffer_end = buffers
            .checked_mul(12)
            .and_then(|v| v.checked_add(16))
            .ok_or(NV_BAD_PARAMETER)?;
        let reloc_end = relocations
            .checked_mul(16)
            .and_then(|v| v.checked_add(buffer_end))
            .ok_or(NV_BAD_PARAMETER)?;
        let shift_end = relocations
            .checked_mul(4)
            .and_then(|v| v.checked_add(reloc_end))
            .ok_or(NV_BAD_PARAMETER)?;
        let sync_end = syncpoints
            .checked_mul(20)
            .and_then(|v| v.checked_add(shift_end))
            .ok_or(NV_BAD_PARAMETER)?;
        let end = fences
            .checked_mul(4)
            .and_then(|v| v.checked_add(sync_end))
            .ok_or(NV_BAD_PARAMETER)?;
        require_input_size(input, end)?;
        if ((request >> 16) & 0x3fff) as usize != end || fences < syncpoints {
            return Err(NV_BAD_PARAMETER.into());
        }
        let owner = NvMapOwner::new(descriptor.owner().process_id());
        let channel = self
            .channels
            .get_mut(&descriptor.fd())
            .ok_or(NV_BAD_PARAMETER)?;
        if channel.nvmap_fd.is_none() {
            return Err(super::NV_INVALID_STATE.into());
        }
        let mut writes = nixe_memory::CanonicalWriteBatch::new();
        for index in 0..relocations {
            let offset = buffer_end + index * 16;
            let command = nvmap
                .object_snapshot_by_owned_handle(owner, NvMapHandle::new(input_u32(input, offset)?))
                .map_err(super::nvmap_driver_result)?;
            let target = nvmap
                .object_snapshot_by_owned_handle(
                    owner,
                    NvMapHandle::new(input_u32(input, offset + 8)?),
                )
                .map_err(super::nvmap_driver_result)?;
            let (&base, _) = self
                .mappings
                .iter()
                .find(|(_, mapping)| mapping.object.id() == target.id())
                .ok_or(NV_BAD_PARAMETER)?;
            let target_offset = input_u32(input, offset + 12)?;
            let shift = input_u32(input, reloc_end + index * 4)?;
            if target_offset >= target.size() || shift > 31 {
                return Err(NV_BAD_PARAMETER.into());
            }
            let address = base.checked_add(target_offset).ok_or(NV_BAD_PARAMETER)? >> shift;
            writes
                .stage(
                    command.backing().ok_or(super::NV_INVALID_STATE)?,
                    u64::from(input_u32(input, offset + 4)?),
                    &address.to_le_bytes(),
                )
                .map_err(|error| execution_error(descriptor, request, error.to_string()))?;
        }
        writes
            .commit_ordered()
            .map_err(|error| execution_error(descriptor, request, error.to_string()))?;
        let mut increments = BTreeMap::<u32, u32>::new();
        let mut operations = Vec::new();
        for index in 0..buffers {
            let offset = 16 + index * 12;
            let object = nvmap
                .object_snapshot_by_owned_handle(owner, NvMapHandle::new(input_u32(input, offset)?))
                .map_err(super::nvmap_driver_result)?;
            let count = input_u32(input, offset + 8)?;
            let length = (count as usize).checked_mul(4).ok_or(NV_BAD_PARAMETER)?;
            if count > i32::MAX as u32 || length > object.size() as usize {
                return Err(NV_BAD_PARAMETER.into());
            }
            let mut bytes = vec![0; length];
            object
                .backing()
                .ok_or(super::NV_INVALID_STATE)?
                .read(u64::from(input_u32(input, offset + 4)?), &mut bytes)
                .map_err(|error| execution_error(descriptor, request, error.to_string()))?;
            for word in bytes.chunks_exact(4) {
                let word = u32::from_le_bytes(word.try_into().unwrap());
                let Some((class, method, argument)) = channel
                    .packet
                    .word(word)
                    .map_err(|error| execution_error(descriptor, request, error))?
                else {
                    continue;
                };
                log::trace!(
                    "Host1x method: fd={} class={class:#x} method={method:#x} argument={argument:#010x}",
                    descriptor.fd()
                );
                if class == 1 {
                    match method {
                        0x4e => channel.packet.wait_value = argument,
                        0x8 | 0x50 => {
                            let point = GuestTimelinePoint::new(
                                GuestSyncpointId::new(argument),
                                GuestSyncpointValue::new(channel.packet.wait_value),
                            );
                            operations.push(worker::Operation::Wait(point));
                        }
                        _ => {
                            return Err(execution_error(
                                descriptor,
                                request,
                                format!("unsupported Host1x control method {method:#x}"),
                            ));
                        }
                    }
                    continue;
                }
                let expected = if descriptor.kind() == NvDrvDeviceKind::HostNvDec {
                    0xf0
                } else {
                    0x5d
                };
                if class != expected {
                    return Err(execution_error(
                        descriptor,
                        request,
                        format!("unsupported channel class {class:#x}"),
                    ));
                }
                match method {
                    0 => {
                        if argument & !0xffff != 0 || argument >> 8 > 2 {
                            return Err(execution_error(
                                descriptor,
                                request,
                                "unsupported syncpoint increment condition",
                            ));
                        }
                        let id = argument & 0xff;
                        let count = increments.entry(id).or_default();
                        *count = count.checked_add(1).ok_or(super::NV_OVERFLOW)?;
                    }
                    0x10 => channel.packet.engine_method = argument,
                    0x11 => {
                        let method = channel.packet.engine_method;
                        let register =
                            channel.registers.get_mut(method as usize).ok_or_else(|| {
                                execution_error(
                                    descriptor,
                                    request,
                                    format!("engine register out of range: {method:#x}"),
                                )
                            })?;
                        *register = argument;
                        log::trace!(
                            "Host1x engine: fd={} method={method:#x} argument={argument:#010x}",
                            descriptor.fd()
                        );
                        if method == 0xc0 {
                            operations.push(worker::Operation::Execute(channel.registers.clone()));
                        }
                    }
                    _ => {
                        return Err(execution_error(
                            descriptor,
                            request,
                            format!("unsupported THI method {method:#x} argument={argument:#x}"),
                        ));
                    }
                }
            }
        }
        let mut output = input.to_vec();
        let mut completions = Vec::new();
        for index in 0..syncpoints {
            let offset = shift_end + index * 20;
            let id = input_u32(input, offset)?;
            let count = input_u32(input, offset + 4)?;
            if id != channel.syncpoint.get() || increments.remove(&id).unwrap_or(0) != count {
                return Err(execution_error(
                    descriptor,
                    request,
                    "submission syncpoint increments do not match executed commands",
                ));
            }
        }
        if !increments.is_empty() {
            return Err(execution_error(
                descriptor,
                request,
                "unadvertised syncpoint increments",
            ));
        }
        let mut control_guard = control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for index in 0..syncpoints {
            let count = input_u32(input, shift_end + index * 20 + 4)?;
            let reservation = control_guard.reserve_channel_submission(
                descriptor,
                request,
                channel.syncpoint,
                count,
            )?;
            write_u32(
                &mut output,
                sync_end + index * 4,
                reservation.point().value().get(),
            )?;
            completions.push(reservation);
        }
        drop(control_guard);
        if self.worker.is_none() {
            self.worker = Some(worker::Worker::new(Arc::clone(control), gpu.cloned())?);
        }
        self.worker.as_ref().unwrap().submit(worker::Job {
            descriptor,
            request,
            operations,
            completions,
            mappings: Arc::clone(&self.mappings),
            timeout: channel.timeout,
        })?;
        Ok(output)
    }
}

impl Host1x {
    fn map_buffers(
        &mut self,
        nvmap: &NvMapObjects,
        descriptor: NvDrvDeviceDescriptor,
        request: u32,
        input: &[u8],
    ) -> Result<Vec<u8>, NvDrvCallError> {
        if input.len() < 12 {
            return Err(NV_BAD_PARAMETER.into());
        }
        let count = input_u32(input, 0)? as usize;
        let length = count
            .checked_mul(8)
            .and_then(|v| v.checked_add(12))
            .ok_or(NV_BAD_PARAMETER)?;
        require_input_size(input, length)?;
        if ((request >> 16) & 0x3fff) as usize != length {
            return Err(NV_BAD_PARAMETER.into());
        }
        let channel = self
            .channels
            .get_mut(&descriptor.fd())
            .ok_or(NV_BAD_PARAMETER)?;
        if channel.nvmap_fd.is_none() {
            return Err(super::NV_INVALID_STATE.into());
        }
        let unmap = request & 0xff == 10;
        let mut output = input.to_vec();
        for index in 0..count {
            let offset = 12 + index * 8;
            let object = nvmap
                .object_snapshot_by_owned_handle(
                    NvMapOwner::new(descriptor.owner().process_id()),
                    NvMapHandle::new(input_u32(input, offset)?),
                )
                .map_err(super::nvmap_driver_result)?;
            let id = object.id().raw();
            if object.backing().is_none() {
                return Err(super::NV_INVALID_STATE.into());
            }
            if unmap {
                let pin = channel.pins.get_mut(&id).ok_or(NV_BAD_PARAMETER)?;
                *pin -= 1;
                if *pin == 0 {
                    channel.pins.remove(&id);
                }
                Arc::make_mut(&mut self.mappings).retain(|_, mapping| {
                    if mapping.object.id().raw() == id {
                        mapping.references -= 1;
                    }
                    mapping.references != 0
                });
                output[offset..offset + 8].fill(0);
            } else {
                let existing = Arc::make_mut(&mut self.mappings)
                    .iter_mut()
                    .find(|(_, mapping)| mapping.object.id().raw() == id);
                let address = if let Some((&address, mapping)) = existing {
                    mapping.references = mapping
                        .references
                        .checked_add(1)
                        .ok_or(super::NV_OVERFLOW)?;
                    address
                } else {
                    // Host1x has a 32-bit IOVA, separate from Maxwell's GPU VA.
                    // Reuse unmapped holes; keep the object pinned while any channel owns it.
                    let size = u64::from(object.size()).next_multiple_of(4096);
                    let mut address = 4096_u64;
                    for (&base, mapping) in self.mappings.iter() {
                        if address + size <= u64::from(base) {
                            break;
                        }
                        address = u64::from(base)
                            + u64::from(mapping.object.size()).next_multiple_of(4096);
                    }
                    if address + size > 1_u64 << 32 {
                        return Err(super::NV_INSUFFICIENT_MEMORY.into());
                    }
                    let address = address as u32;
                    Arc::make_mut(&mut self.mappings).insert(
                        address,
                        Mapping {
                            object,
                            references: 1,
                        },
                    );
                    address
                };
                let pin = channel.pins.entry(id).or_default();
                *pin = pin.checked_add(1).ok_or(super::NV_OVERFLOW)?;
                write_u32(&mut output, offset + 4, address)?;
            }
        }
        if unmap {
            output[..12].fill(0);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::super::nvmap::{NvMapAllocationMetadata, NvMapCpuMapping};
    use super::super::{NvDrvDescriptorOwner, NvDrvPermissionProfile, NvDrvSessionId};
    use super::*;
    use nixe_memory::{
        AddressSpaceId, CanonicalAllocation, GuestVirtualAddress, MemoryPermissions,
    };

    fn descriptor(fd: u32, kind: NvDrvDeviceKind) -> NvDrvDeviceDescriptor {
        NvDrvDeviceDescriptor::open(
            NvDrvFileDescriptor::new(fd),
            kind,
            NvDrvDescriptorOwner::new(NvDrvSessionId::ROOT, 1),
            NvDrvPermissionProfile::Application,
        )
    }
    #[test]
    fn video_channels_share_pinned_objects_until_the_last_unmap_or_close() {
        let mut nvmap = NvMapObjects::default();
        let owner = NvMapOwner::new(1);
        let handle = nvmap.create(owner, 4096).unwrap();
        let allocation = CanonicalAllocation::zeroed(4096, 4096).unwrap();
        allocation.write(7, &[42]).unwrap();
        nvmap
            .allocate(
                owner,
                handle,
                NvMapAllocationMetadata::new(0, 0, 4096, 0),
                NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(0x1000)),
                allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
            )
            .unwrap();
        let devices = BTreeMap::from([(
            NvDrvFileDescriptor::new(1),
            descriptor(1, NvDrvDeviceKind::NvMap),
        )]);
        let mut host = Host1x::default();
        let control = Arc::new(Mutex::new(NvHostControl::default()));
        let mut map = [0; 20];
        write_u32(&mut map, 0, 1).unwrap();
        write_u32(&mut map, 12, handle.raw()).unwrap();
        for (fd, kind) in [
            (2, NvDrvDeviceKind::HostNvDec),
            (3, NvDrvDeviceKind::HostVic),
        ] {
            host.open(NvDrvFileDescriptor::new(fd), GuestSyncpointId::new(fd));
            host.ioctl(
                &devices,
                &nvmap,
                &control,
                None,
                descriptor(fd, kind),
                0x4004_4801,
                &1u32.to_le_bytes(),
            )
            .unwrap();
            let output = host
                .ioctl(
                    &devices,
                    &nvmap,
                    &control,
                    None,
                    descriptor(fd, kind),
                    0xc014_0009,
                    &map,
                )
                .unwrap();
            assert_eq!(input_u32(&output, 16).unwrap(), 4096);
        }
        assert_eq!(host.mappings[&4096].references, 2);
        let mut byte = [0];
        host.mappings[&4096]
            .object
            .backing()
            .unwrap()
            .read(7, &mut byte)
            .unwrap();
        assert_eq!(byte, [42]);
        host.close(NvDrvFileDescriptor::new(2));
        assert_eq!(host.mappings[&4096].references, 1);
        host.ioctl(
            &devices,
            &nvmap,
            &control,
            None,
            descriptor(3, NvDrvDeviceKind::HostVic),
            0xc014_000a,
            &map,
        )
        .unwrap();
        assert!(host.mappings.is_empty());
        assert!(
            host.ioctl(
                &devices,
                &nvmap,
                &control,
                None,
                descriptor(3, NvDrvDeviceKind::HostVic),
                0xc014_000a,
                &map
            )
            .is_err()
        );
        let output = host
            .ioctl(
                &devices,
                &nvmap,
                &control,
                None,
                descriptor(3, NvDrvDeviceKind::HostVic),
                0xc014_0009,
                &map,
            )
            .unwrap();
        assert_eq!(input_u32(&output, 16).unwrap(), 4096);
        host.clear();
        assert!(host.mappings.is_empty());
    }
}
