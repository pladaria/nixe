//! Canonical audio-buffer attachments shared by voices and DSP effects.
use super::update::{MemoryPool, UpdateError};
use nixe_memory::{CanonicalBackingRange, CanonicalRangeTranslationError};

#[derive(Debug)]
pub(super) struct Attachment {
    pub(super) address: u64,
    pub(super) backing: CanonicalBackingRange,
    pub(super) pool: Option<usize>,
}
impl Attachment {
    pub(super) fn read(&self, offset: u64, bytes: &mut [u8]) -> Result<(), Box<str>> {
        self.backing
            .read(offset, bytes)
            .map_err(|e| format!("rendering: audio buffer read failed: {e}").into())
    }
}
pub(super) fn attach(
    address: u64,
    size: u64,
    pools: &[Option<MemoryPool>],
    force: bool,
    translate: &impl Fn(u64, u64) -> Result<CanonicalBackingRange, CanonicalRangeTranslationError>,
) -> Result<Option<Attachment>, UpdateError> {
    if address == 0 || size == 0 {
        return Ok(None);
    }
    let end = address.checked_add(size).ok_or(UpdateError::Invalid)?;
    for (i, pool) in pools.iter().enumerate() {
        let Some(pool) = pool else { continue };
        if address >= pool.address && end <= pool.address.saturating_add(pool.backing.size()) {
            let backing = pool
                .backing
                .snapshot_subrange(address - pool.address, size)
                .map_err(|_| UpdateError::Invalid)?;
            return Ok(Some(Attachment {
                address,
                backing,
                pool: Some(i),
            }));
        }
    }
    if force {
        return Ok(Some(Attachment {
            address,
            backing: translate(address, size).map_err(UpdateError::Mapping)?,
            pool: None,
        }));
    }
    Err(UpdateError::Unsupported(format!("RequestUpdate: audio buffer {address:#x} ({size} bytes) is outside attached memory pools").into()))
}
