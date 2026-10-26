//! Physical audio-buffer attachments shared by voices and DSP effects.
use super::{
    update::{MemoryPool, UpdateError},
    worker::{RenderError, RenderMemory},
};
use nixe_memory::{Access, AccessResult, AllocationRange, MemoryError};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct Attachment {
    pub(super) address: u64,
    pub(super) ranges: nixe_memory::CapturedMemory,
    pub(super) bytes: u64,
    pub(super) pool: Option<usize>,
}
impl Attachment {
    pub(super) fn write(
        &self,
        memory: &RenderMemory,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RenderError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let machine = self.ranges[0].allocation().machine();
        match machine
            .write_captured(&self.ranges, offset, bytes)
            .map_err(Arc::new)?
        {
            Access::Ready(()) => Ok(()),
            Access::Pending(pending) => match memory.wait(pending)? {
                AccessResult::Written => Ok(()),
                _ => Err("invalid DSP memory write completion".into()),
            },
        }
    }
    pub(super) fn read(
        &self,
        memory: &RenderMemory,
        offset: u64,
        bytes: &mut [u8],
    ) -> Result<(), RenderError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let machine = self.ranges[0].allocation().machine();
        match machine
            .read_captured_into(&self.ranges, offset, bytes)
            .map_err(Arc::new)?
        {
            Access::Ready(()) => Ok(()),
            Access::Pending(pending) => match memory.wait(pending)? {
                AccessResult::Read(data) if data.len() == bytes.len() => {
                    bytes.copy_from_slice(&data);
                    Ok(())
                }
                _ => Err(Arc::new(MemoryError::Execution(
                    "invalid DSP memory read completion".into(),
                ))
                .into()),
            },
        }
    }
}
pub(super) fn attach(
    address: u64,
    size: u64,
    pools: &[Option<MemoryPool>],
    force: bool,
    translate: &impl Fn(u64, u64) -> Result<Vec<AllocationRange>, MemoryError>,
) -> Result<Option<Attachment>, UpdateError> {
    if address == 0 || size == 0 {
        return Ok(None);
    }
    let end = address.checked_add(size).ok_or(UpdateError::Invalid)?;
    for (i, pool) in pools.iter().enumerate() {
        let Some(pool) = pool else { continue };
        if address >= pool.address && end <= pool.address.saturating_add(pool.bytes) {
            let ranges = pool
                .ranges
                .capture(address - pool.address, size)
                .map_err(UpdateError::Mapping)?;
            return Ok(Some(Attachment {
                address,
                ranges: nixe_memory::CapturedMemory::new(ranges).map_err(UpdateError::Mapping)?,
                bytes: size,
                pool: Some(i),
            }));
        }
    }
    if force {
        return Ok(Some(Attachment {
            address,
            ranges: nixe_memory::CapturedMemory::new(
                translate(address, size).map_err(UpdateError::Mapping)?,
            )
            .map_err(UpdateError::Mapping)?,
            bytes: size,
            pool: None,
        }));
    }
    Err(UpdateError::Unsupported(format!("RequestUpdate: audio buffer {address:#x} ({size} bytes) is outside attached memory pools").into()))
}
