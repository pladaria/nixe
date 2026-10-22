//! Immutable descriptor bytes with one monotonic invalidation observation.
use super::*;
use nixe_gpu::physical::PhysicalMapping;
use nixe_memory::{AccessOperation, AccessResult, Observation};

#[derive(Clone, Eq, Hash, PartialEq)]
struct Key {
    parent: Arc<PhysicalMapping>,
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
        parent: &Arc<PhysicalMapping>,
        offset: u64,
        size: u8,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Arc<Snapshot>, MaxwellThreeDResourceError> {
        let key = Key {
            parent: Arc::clone(parent),
            offset,
            size,
        };
        if let Some(snapshot) = self.entries.get(&key)
            && snapshot.remains_current()
        {
            return Ok(Arc::clone(snapshot));
        }
        let snapshot = Arc::new(Snapshot::read(parent, offset, size, access)?);
        let weight = std::mem::size_of::<Snapshot>() + parent.metadata_bytes() + 512;
        self.entries.insert(key, Arc::clone(&snapshot), weight);
        Ok(snapshot)
    }
}
#[derive(Debug)]
pub(super) struct Snapshot {
    pub(super) bytes: [u8; 32],
    observation: Observation,
}
impl Snapshot {
    pub(super) fn read(
        parent: &PhysicalMapping,
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
        Ok(Self { bytes, observation })
    }
    pub(super) fn remains_current(&self) -> bool {
        !self.observation.is_invalid()
    }
}

#[cfg(test)]
mod tests;
