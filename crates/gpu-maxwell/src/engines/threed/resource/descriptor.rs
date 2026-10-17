//! Immutable descriptor bytes and their exact ordered validity proof.
//!
//! Descriptor identity is independent of draw-state revisions. Reuse the
//! observation instead of rebuilding page tracking for each resource set.
//! Independent reference mechanisms:
//! https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/texture_cache/texture_cache.h
//! https://github.com/yangjieshao/Ryujinx/blob/2505a1abcd5f31e4364bfb7f9313546f1840e5e1/src/Ryujinx.Graphics.Gpu/Image/TexturePool.cs
use super::*;

#[derive(Clone, Eq, Hash, PartialEq)]
struct Key {
    parent: nixe_memory::CanonicalByteMappingKey,
    offset: u64,
    size: u8,
}

pub(super) struct Cache {
    entries: nixe_gpu::WorkingSet<Key, Arc<Snapshot>>,
}
impl Default for Cache {
    fn default() -> Self {
        Self {
            entries: nixe_gpu::WorkingSet::new(4096, 64 * 1024 * 1024),
        }
    }
}
impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorCache")
            .field("usage", &self.entries.usage())
            .finish()
    }
}
impl Cache {
    pub(super) fn read(
        &mut self,
        parent: &CanonicalBackingRange,
        offset: u64,
        size: u8,
        writes: Option<&CanonicalWriteBatch>,
    ) -> Result<Arc<Snapshot>, MaxwellThreeDResourceError> {
        let key = Key {
            parent: parent.byte_mapping_key(),
            offset,
            size,
        };
        let range = if let Some(snapshot) = self.entries.get(&key) {
            if snapshot.remains_current(writes)? {
                return Ok(Arc::clone(snapshot));
            }
            // New bytes need an independent observation, not a new physical
            // topology. Old snapshots keep their own unchanged byte versions.
            snapshot.range.clone()
        } else {
            parent
                .snapshot_subrange(offset, u64::from(size))
                .map_err(MaxwellThreeDResourceError::Canonical)?
        };
        let snapshot = Arc::new(Snapshot::capture(range, size, writes)?);
        // Both parent identity and the narrow observed range retain metadata.
        // Charge them conservatively even when other consumers share it.
        let bytes = std::mem::size_of::<Snapshot>()
            + parent.metadata_bytes()
            + snapshot.range.metadata_bytes()
            + 512 * (1 + snapshot.range.pages().count());
        self.entries.insert(key, Arc::clone(&snapshot), bytes);
        Ok(snapshot)
    }
}

#[derive(Debug)]
pub(super) struct Snapshot {
    range: CanonicalBackingRange,
    pub(super) bytes: [u8; 32],
    size: u8,
    cpu_writes: CanonicalCpuWriteDependency,
    overlay_changed: bool,
}
impl Snapshot {
    pub(super) fn read(
        parent: &CanonicalBackingRange,
        offset: u64,
        size: u8,
        writes: Option<&CanonicalWriteBatch>,
    ) -> Result<Self, MaxwellThreeDResourceError> {
        let range = parent
            .snapshot_subrange(offset, u64::from(size))
            .map_err(MaxwellThreeDResourceError::Canonical)?;
        Self::capture(range, size, writes)
    }

    fn capture(
        range: CanonicalBackingRange,
        size: u8,
        writes: Option<&CanonicalWriteBatch>,
    ) -> Result<Self, MaxwellThreeDResourceError> {
        debug_assert!((1..=32).contains(&size));
        // Arm before reading. A concurrent write must not establish a clean
        // baseline for bytes already captured from an older version.
        let cpu_writes = CanonicalCpuWriteDependency::capture(&range)
            .map_err(MaxwellThreeDResourceError::CanonicalAccess)?;
        let mut bytes = [0; 32];
        read_backing_bytes(&range, 0, &mut bytes[..usize::from(size)], writes)?;
        let mut overlay_changed = false;
        if let Some(writes) = writes
            && writes
                .overlaps(&range, 0, u64::from(size))
                .map_err(MaxwellThreeDResourceError::StagedCanonicalAccess)?
        {
            let mut original = [0; 32];
            read_backing_bytes(&range, 0, &mut original[..usize::from(size)], None)?;
            overlay_changed = original[..usize::from(size)] != bytes[..usize::from(size)];
        }
        Ok(Self {
            range,
            bytes,
            size,
            cpu_writes,
            overlay_changed,
        })
    }

    pub(super) fn remains_current(
        &self,
        writes: Option<&CanonicalWriteBatch>,
    ) -> Result<bool, MaxwellThreeDResourceError> {
        // CPU-write epochs alone do not prove that an outstanding GPU producer
        // has left descriptor bytes unchanged. Reject without waiting; an actual
        // read on a miss resolves that producer through normal memory admission.
        if !self
            .range
            .subrange_is_cpu_visible(0, u64::from(self.size))
            .map_err(MaxwellThreeDResourceError::Canonical)?
        {
            return Ok(false);
        }
        if !self.cpu_writes.remains_current()
            && (self.overlay_changed
                || !self
                    .cpu_writes
                    .confirm_unchanged(&self.bytes[..usize::from(self.size)])
                    .map_err(MaxwellThreeDResourceError::CanonicalAccess)?)
        {
            return Ok(false);
        }
        let Some(writes) = writes else {
            return Ok(!self.overlay_changed);
        };
        if !writes
            .overlaps(&self.range, 0, u64::from(self.size))
            .map_err(MaxwellThreeDResourceError::StagedCanonicalAccess)?
        {
            return Ok(!self.overlay_changed);
        }
        let mut bytes = [0; 32];
        writes
            .read_overlay(&self.range, 0, &mut bytes[..usize::from(self.size)])
            .map_err(MaxwellThreeDResourceError::StagedCanonicalAccess)?;
        Ok(bytes[..usize::from(self.size)] == self.bytes[..usize::from(self.size)])
    }
}

#[cfg(test)]
mod tests;
