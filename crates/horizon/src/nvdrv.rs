//! Minimal `nvdrv` and `/dev/nvmap` state for CPU-produced display buffers.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const IOCTL_NVMAP_CREATE: u32 = 0xc008_0101;
const IOCTL_NVMAP_FROM_ID: u32 = 0xc008_0103;
const IOCTL_NVMAP_ALLOC: u32 = 0xc020_0104;
const IOCTL_NVMAP_FREE: u32 = 0xc018_0105;
const IOCTL_NVMAP_PARAM: u32 = 0xc00c_0109;
const IOCTL_NVMAP_GET_ID: u32 = 0xc008_010e;

pub(crate) const NV_SUCCESS: u32 = 0;
pub(crate) const NV_NOT_IMPLEMENTED: u32 = 1;
pub(crate) const NV_NOT_INITIALIZED: u32 = 3;
pub(crate) const NV_BAD_PARAMETER: u32 = 4;
pub(crate) const NV_INVALID_STATE: u32 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceKind {
    NvMap,
    NvHostCtrl,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvMapAllocation {
    pub handle: u32,
    pub id: u32,
    pub size: u32,
    pub alignment: u32,
    pub kind: u8,
    pub cpu_address: Option<u64>,
    pub flags: u32,
}

#[derive(Debug)]
struct NvDrvState {
    initialized: bool,
    next_fd: u32,
    next_handle: u32,
    next_id: u32,
    devices: BTreeMap<u32, DeviceKind>,
    allocations: BTreeMap<u32, NvMapAllocation>,
}

/// One cloneable `nvdrv` service connection sharing file descriptors and maps.
#[derive(Clone, Debug)]
pub struct NvDrvSession {
    state: Arc<Mutex<NvDrvState>>,
}

impl Default for NvDrvSession {
    fn default() -> Self {
        Self::new()
    }
}

impl NvDrvSession {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(NvDrvState {
                initialized: false,
                next_fd: 1,
                next_handle: 1,
                next_id: 1,
                devices: BTreeMap::new(),
                allocations: BTreeMap::new(),
            })),
        }
    }

    pub(crate) fn initialize(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .initialized = true;
    }

    pub(crate) fn open(&self, path: &[u8]) -> Result<u32, u32> {
        let kind = match path {
            b"/dev/nvmap" => DeviceKind::NvMap,
            b"/dev/nvhost-ctrl" => DeviceKind::NvHostCtrl,
            _ => return Err(NV_NOT_IMPLEMENTED),
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.initialized {
            return Err(NV_NOT_INITIALIZED);
        }
        let fd = state.next_fd;
        state.next_fd = state.next_fd.checked_add(1).ok_or(NV_INVALID_STATE)?;
        state.devices.insert(fd, kind);
        Ok(fd)
    }

    pub(crate) fn close(&self, fd: u32) -> u32 {
        if self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .devices
            .remove(&fd)
            .is_some()
        {
            NV_SUCCESS
        } else {
            NV_BAD_PARAMETER
        }
    }

    pub(crate) fn ioctl(&self, fd: u32, request: u32, input: &[u8]) -> (Vec<u8>, u32) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.devices.get(&fd) != Some(&DeviceKind::NvMap) {
            return (input.to_vec(), NV_NOT_IMPLEMENTED);
        }
        match ioctl_nvmap(&mut state, request, input) {
            Ok(output) => (output, NV_SUCCESS),
            Err(error) => (input.to_vec(), error),
        }
    }

    #[must_use]
    pub fn allocation_by_id(&self, id: u32) -> Option<NvMapAllocation> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .allocations
            .values()
            .find(|allocation| allocation.id == id)
            .cloned()
    }
}

fn ioctl_nvmap(state: &mut NvDrvState, request: u32, input: &[u8]) -> Result<Vec<u8>, u32> {
    match request {
        IOCTL_NVMAP_CREATE => {
            let size = input_u32(input, 0)?;
            if size == 0 {
                return Err(NV_BAD_PARAMETER);
            }
            let handle = state.next_handle;
            state.next_handle = state.next_handle.checked_add(1).ok_or(NV_INVALID_STATE)?;
            let id = state.next_id;
            state.next_id = state.next_id.checked_add(1).ok_or(NV_INVALID_STATE)?;
            state.allocations.insert(
                handle,
                NvMapAllocation {
                    handle,
                    id,
                    size,
                    alignment: 0,
                    kind: 0,
                    cpu_address: None,
                    flags: 0,
                },
            );
            let mut output = sized_output(input, 8);
            output[4..8].copy_from_slice(&handle.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_FROM_ID => {
            let id = input_u32(input, 0)?;
            let handle = state
                .allocations
                .values()
                .find(|allocation| allocation.id == id)
                .map(|allocation| allocation.handle)
                .ok_or(NV_BAD_PARAMETER)?;
            let mut output = sized_output(input, 8);
            output[4..8].copy_from_slice(&handle.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_ALLOC => {
            let handle = input_u32(input, 0)?;
            let flags = input_u32(input, 8)?;
            let alignment = input_u32(input, 12)?;
            let kind = *input.get(16).ok_or(NV_BAD_PARAMETER)?;
            let address = input_u64(input, 24)?;
            if address == 0 || alignment < 0x1000 || !alignment.is_power_of_two() {
                return Err(NV_BAD_PARAMETER);
            }
            let allocation = state.allocations.get_mut(&handle).ok_or(NV_BAD_PARAMETER)?;
            if allocation.cpu_address.is_some() {
                return Err(NV_INVALID_STATE);
            }
            allocation.flags = flags;
            allocation.alignment = alignment;
            allocation.kind = kind;
            allocation.cpu_address = Some(address);
            Ok(sized_output(input, 32))
        }
        IOCTL_NVMAP_FREE => {
            let handle = input_u32(input, 0)?;
            let allocation = state.allocations.remove(&handle).ok_or(NV_BAD_PARAMETER)?;
            let mut output = sized_output(input, 24);
            output[8..16].copy_from_slice(&0_u64.to_le_bytes());
            output[16..20].copy_from_slice(&allocation.size.to_le_bytes());
            output[20..24].copy_from_slice(&0_u32.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_PARAM => {
            let handle = input_u32(input, 0)?;
            let parameter = input_u32(input, 4)?;
            let allocation = state.allocations.get(&handle).ok_or(NV_BAD_PARAMETER)?;
            let value = match parameter {
                1 => allocation.size,
                2 => allocation.alignment,
                4 => 0,
                5 => u32::from(allocation.kind),
                _ => return Err(NV_BAD_PARAMETER),
            };
            let mut output = sized_output(input, 12);
            output[8..12].copy_from_slice(&value.to_le_bytes());
            Ok(output)
        }
        IOCTL_NVMAP_GET_ID => {
            let handle = input_u32(input, 4)?;
            let allocation = state.allocations.get(&handle).ok_or(NV_BAD_PARAMETER)?;
            let mut output = sized_output(input, 8);
            output[0..4].copy_from_slice(&allocation.id.to_le_bytes());
            Ok(output)
        }
        _ => Err(NV_NOT_IMPLEMENTED),
    }
}

fn sized_output(input: &[u8], size: usize) -> Vec<u8> {
    let mut output = vec![0_u8; size];
    let copied = input.len().min(size);
    output[..copied].copy_from_slice(&input[..copied]);
    output
}

fn input_u32(input: &[u8], offset: usize) -> Result<u32, u32> {
    Ok(u32::from_le_bytes(
        input
            .get(offset..offset + 4)
            .ok_or(NV_BAD_PARAMETER)?
            .try_into()
            .unwrap(),
    ))
}

fn input_u64(input: &[u8], offset: usize) -> Result<u64, u32> {
    Ok(u64::from_le_bytes(
        input
            .get(offset..offset + 8)
            .ok_or(NV_BAD_PARAMETER)?
            .try_into()
            .unwrap(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvmap_allocation_retains_guest_address_identity() {
        let session = NvDrvSession::new();
        session.initialize();
        let fd = session.open(b"/dev/nvmap").unwrap();
        let (created, error) = session.ioctl(fd, IOCTL_NVMAP_CREATE, &0x2000_u32.to_le_bytes());
        assert_eq!(error, NV_SUCCESS);
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = vec![0_u8; 32];
        allocate[0..4].copy_from_slice(&handle.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[16] = 0xfe;
        allocate[24..32].copy_from_slice(&0x1234_5000_u64.to_le_bytes());
        assert_eq!(
            session.ioctl(fd, IOCTL_NVMAP_ALLOC, &allocate).1,
            NV_SUCCESS
        );
        let id = session
            .state
            .lock()
            .unwrap()
            .allocations
            .get(&handle)
            .unwrap()
            .id;
        assert_eq!(
            session.allocation_by_id(id).unwrap().cpu_address,
            Some(0x1234_5000)
        );
    }
}
