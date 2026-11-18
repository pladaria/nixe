use super::*;
use bytemuck::{Pod, Zeroable};
use nixe_memory::{DeviceHeapMapping, GuestVirtualAddress, MemoryPermissions};
use std::collections::HashMap;

pub(crate) type Mappings = HashMap<u64, Arc<DeviceHeapMapping>>;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Builder {
    device: u64,
    storage: u64,
    bytes: u64,
    flags: u32,
    padding: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Pool {
    pub cpu: u64,
    pub gpu: u64,
    pub bytes: u64,
    pub flags: u32,
    padding: u32,
}
pub fn defaults(c: &Call) -> Result<u64> {
    c.write_bytes(
        c.args[0] + 8,
        &bytemuck::bytes_of(&Builder {
            flags: 0x21,
            ..Builder::zeroed()
        })[8..],
    )?;
    Ok(0)
}
pub fn set_device(c: &Call) -> Result<u64> {
    c.write(c.args[0], &c.args[1])?;
    Ok(0)
}
pub fn storage(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[c.args[1], c.args[2]])?;
    Ok(0)
}
pub fn flags(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 24, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn initialize(c: &Call) -> Result<u64> {
    let b: Builder = c.read(c.args[1])?;
    if b.flags & 0x200 != 0 {
        return Err("virtual memory pools are unsupported".into());
    }
    let address = GuestVirtualAddress::new(b.storage);
    let ranges = c
        .memory
        .capture(address, b.bytes, MemoryPermissions::READ)
        .map_err(|e| e.to_string())?;
    let mapping = c
        .session
        .graphics()?
        .heap
        .map(ranges.into(), b.bytes, Some((c.memory.identity(), address)))
        .map_err(|e| e.to_string())?;
    let pool = Pool {
        cpu: b.storage,
        gpu: mapping.address(),
        bytes: b.bytes,
        flags: b.flags,
        padding: 0,
    };
    c.write(c.args[0], &pool)?;
    c.session.pools.lock().unwrap().insert(c.args[0], mapping);
    Ok(1)
}
pub fn map(c: &Call) -> Result<u64> {
    let p: Pool = c.read(c.args[0])?;
    Ok(if p.flags & 1 != 0 { 0 } else { p.cpu })
}
pub fn address(c: &Call) -> Result<u64> {
    c.read(c.args[0] + 8)
}
pub fn flush(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION CPU writes already dirty the shared physical buffer/image caches. Submission
    // drains those invalidations before consuming the pool; mapping needs no upload.
    Ok(0)
}
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct DescriptorPool {
    pub base: u64,
    pub count: u32,
    padding: u32,
}
pub fn descriptor_pool(c: &Call) -> Result<u64> {
    let p: Pool = c.read(c.args[1])?;
    let off = c.args[2];
    let count = c.args[3] as u32;
    if off > p.bytes || u64::from(count) * 32 > p.bytes - off {
        return Err("descriptor pool exceeds memory pool".into());
    }
    c.write(
        c.args[0],
        &DescriptorPool {
            base: p.cpu + off,
            count,
            padding: 0,
        },
    )?;
    Ok(1)
}
impl DescriptorPool {
    pub fn new(base: u64, count: u32) -> Self {
        Self {
            base,
            count,
            padding: 0,
        }
    }
    pub fn slot(self, index: u32) -> Result<u64> {
        if index >= self.count {
            return Err(format!("descriptor {index} exceeds pool capacity {}", self.count).into());
        }
        Ok(self.base + u64::from(index) * 32)
    }
}
pub(crate) fn slot(c: &Call, pointer: u64, index: u32) -> Result<u64> {
    let pool: DescriptorPool = c.read(pointer)?;
    pool.slot(index)
}
pub fn finalize(c: &Call) -> Result<u64> {
    c.session.pools.lock().unwrap().remove(&c.args[0]);
    Ok(0)
}
