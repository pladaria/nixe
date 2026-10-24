//! Indexed descriptor slots with immutable bytes and monotonic invalidation.
//! Slot indexing and byte comparison follow Eden DescriptorTable:
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/descriptor_table.h
use super::*;
use nixe_gpu::physical::PhysicalMapping;
use nixe_memory::{AccessOperation, AccessResult, Observation};

#[derive(Clone, Eq, Hash, PartialEq)]
struct TableKey {
    parent: Arc<PhysicalMapping>,
    block: u64,
    lane: u8,
    size: u8,
}
const SLOTS: usize = 64;
type Slots = Box<[Option<Arc<Snapshot>>; SLOTS]>;
pub(super) struct Cache {
    tables: nixe_gpu::WorkingSet<TableKey, Slots>,
}
impl Default for Cache {
    fn default() -> Self {
        Self {
            tables: nixe_gpu::WorkingSet::new(4096, 64 * 1024 * 1024),
        }
    }
}
impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorCache")
            .field("usage", &self.tables.usage())
            .finish()
    }
}
impl Cache {
    pub(super) fn read(
        &mut self,
        parent: &Arc<PhysicalMapping>,
        offset: u64,
        size: u8,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Arc<Snapshot>, MaxwellThreeDResourceError> {
        // TIC/TSC indices select dense 32-byte slots. Short CB words and storage
        // descriptors use their byte lane in the same indexing scheme. Allocate
        // blocks only as consumed, without sizing storage to an unused pool limit.
        let key = TableKey {
            parent: Arc::clone(parent),
            block: offset / (32 * SLOTS as u64),
            lane: (offset % 32) as u8,
            size,
        };
        let index = (offset / 32 % SLOTS as u64) as usize;
        if let Some(snapshot) = self
            .tables
            .get(&key)
            .and_then(|slots| slots[index].as_ref())
            && snapshot.remains_current()
        {
            return Ok(Arc::clone(snapshot));
        }
        let snapshot = Arc::new(Snapshot::read(parent, offset, size, access)?);
        if let Some(slots) = self.tables.get_mut(&key) {
            slots[index] = Some(Arc::clone(&snapshot));
        } else {
            let mut slots: Slots = Box::new(std::array::from_fn(|_| None));
            slots[index] = Some(Arc::clone(&snapshot));
            let weight = SLOTS * (std::mem::size_of::<Snapshot>() + 512)
                + std::mem::size_of_val(slots.as_ref())
                + parent.metadata_bytes();
            self.tables.insert(key, slots, weight);
        }
        Ok(snapshot)
    }
}
#[derive(Debug)]
pub(super) struct Snapshot {
    pub(super) bytes: [u8; 32],
    pub(super) parent: Arc<PhysicalMapping>,
    pub(super) offset: u64,
    pub(super) size: u8,
    observation: Observation,
}
impl Snapshot {
    pub(super) fn read(
        parent: &Arc<PhysicalMapping>,
        offset: u64,
        size: u8,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Self, MaxwellThreeDResourceError> {
        assert!((1..=32).contains(&size));
        let ranges = parent
            .pin(offset, u64::from(size))
            .map_err(|e| MaxwellThreeDResourceError::Memory(Arc::new(e)))?;
        let observation = Observation::default();
        let AccessResult::Read(captured) =
            access(ranges, AccessOperation::ReadObserved(observation.clone()))
                .map_err(MaxwellThreeDResourceError::Memory)?
        else {
            unreachable!("read returns bytes")
        };
        let mut bytes = [0; 32];
        bytes[..usize::from(size)].copy_from_slice(&captured);
        Ok(Self {
            bytes,
            observation,
            parent: Arc::clone(parent),
            offset,
            size,
        })
    }
    pub(super) fn remains_current(&self) -> bool {
        !self.observation.is_invalid()
    }
}

#[cfg(test)]
mod tests;
