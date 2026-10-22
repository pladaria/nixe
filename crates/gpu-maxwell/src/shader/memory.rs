//! Owned shader observations captured on the command processor.
const MAXWELL_SHADER_READ_LIMIT: usize = 64 * 1024;
use crate::{MaxwellGpuAccessError, MaxwellGpuMemory, MaxwellShaderStage};
use nixe_gpu::physical::PhysicalMapping;
use nixe_memory::{AccessOperation, AccessResult, MemoryPermissions, Observation};
use std::sync::Arc;

pub struct MaxwellShaderMemoryCache {
    reads: nixe_gpu::WorkingSet<(crate::MaxwellAddressSpaceId, u64, usize), MaxwellShaderRead>,
}
impl Default for MaxwellShaderMemoryCache {
    fn default() -> Self {
        Self {
            reads: nixe_gpu::WorkingSet::new(4096, 32 * 1024 * 1024),
        }
    }
}
pub(super) struct MaxwellShaderMemoryView<'a, 'm> {
    address_space: &'a MaxwellGpuMemory,
    cache: &'a mut MaxwellShaderMemoryCache,
    access: &'a mut nixe_gpu::MemoryAccess<'m>,
}
#[derive(Clone, Debug)]
pub(super) struct MaxwellShaderRead {
    pub(super) address: u64,
    pub(super) stage: MaxwellShaderStage,
    pub(super) bytes: Arc<[u8]>,
    pub(super) physical: Arc<PhysicalMapping>,
    observation: Observation,
    pub(super) mappings: Arc<[crate::MaxwellGpuMapping]>,
    address_space: crate::MaxwellAddressSpaceId,
    mapping_generation: nixe_gpu::MappingGeneration,
}
impl MaxwellShaderRead {
    fn mapping_is_current(&self, space: &MaxwellGpuMemory) -> bool {
        self.address_space == space.id()
            && (self.mapping_generation == space.mapping_generation()
                || self
                    .mappings
                    .iter()
                    .all(|mapping| space.retained_mapping_is_current(mapping)))
    }
    pub(super) fn is_current(&self, space: &MaxwellGpuMemory) -> bool {
        !self.observation.is_invalid() && self.mapping_is_current(space)
    }
}
impl<'a, 'm> MaxwellShaderMemoryView<'a, 'm> {
    pub(super) fn new(
        address_space: &'a MaxwellGpuMemory,
        cache: &'a mut MaxwellShaderMemoryCache,
        access: &'a mut nixe_gpu::MemoryAccess<'m>,
    ) -> Self {
        Self {
            address_space,
            cache,
            access,
        }
    }
    pub(super) fn read(
        &mut self,
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
    ) -> Result<MaxwellShaderRead, ShaderCaptureError> {
        let key = (self.address_space.id(), address, size);
        if let Some(read) = self.cache.reads.get(&key)
            && read.is_current(self.address_space)
        {
            return Ok(read.clone());
        }
        if size > MAXWELL_SHADER_READ_LIMIT {
            return Err(ShaderCaptureError::ReadTooLarge {
                requested: size,
                limit: MAXWELL_SHADER_READ_LIMIT,
            });
        }
        let fail = |error| ShaderCaptureError::Memory {
            stage,
            address,
            error,
        };
        let previous = self
            .cache
            .reads
            .get(&key)
            .filter(|read| read.mapping_is_current(self.address_space));
        let (physical, mappings) = if let Some(previous) = previous {
            (
                Arc::clone(&previous.physical),
                Arc::clone(&previous.mappings),
            )
        } else {
            let base = self
                .address_space
                .address(address)
                .map_err(MaxwellGpuAccessError::Address)
                .map_err(fail)?;
            let resolved = self
                .address_space
                .resolve_range(base, size as u64, MemoryPermissions::READ)
                .map_err(fail)?;
            let inputs = self.address_space.capture(&resolved).map_err(fail)?;
            let physical = PhysicalMapping::new(&inputs)
                .map_err(|e| fail(MaxwellGpuAccessError::Backing(Arc::new(e))))?;
            let mut mappings = Vec::new();
            for segment in resolved.segments() {
                if !mappings
                    .iter()
                    .any(|m: &crate::MaxwellGpuMapping| m.id() == segment.mapping().id())
                {
                    mappings.push(segment.mapping().clone());
                }
            }
            (physical, mappings.into())
        };
        let inputs = physical
            .pin(0, size as u64)
            .map_err(|e| fail(MaxwellGpuAccessError::Backing(Arc::new(e))))?;
        let observation = Observation::default();
        let AccessResult::Read(bytes) =
            (self.access)(inputs, AccessOperation::ReadObserved(observation.clone()))
                .map_err(|e| fail(MaxwellGpuAccessError::Backing(e)))?
        else {
            unreachable!("read returns bytes")
        };
        let read = MaxwellShaderRead {
            address,
            stage,
            bytes: bytes.into(),
            physical,
            observation,
            mappings,
            address_space: self.address_space.id(),
            mapping_generation: self.address_space.mapping_generation(),
        };
        let weight = std::mem::size_of::<MaxwellShaderRead>()
            + read.bytes.len()
            + std::mem::size_of_val(read.mappings.as_ref())
            + read.physical.metadata_bytes()
            + 512;
        self.cache.reads.insert(key, read.clone(), weight);
        Ok(read)
    }
}

#[derive(Debug)]
pub(super) enum ShaderCaptureError {
    ReadTooLarge {
        requested: usize,
        limit: usize,
    },
    Memory {
        stage: MaxwellShaderStage,
        address: u64,
        error: MaxwellGpuAccessError,
    },
}
impl std::fmt::Display for ShaderCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadTooLarge { requested, limit } => write!(
                f,
                "shader capture requests {requested} bytes, limit {limit}"
            ),
            Self::Memory {
                stage,
                address,
                error,
            } => write!(f, "{stage:?} shader capture at {address:#x}: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shader::test_support::mapped_memory, test_memory};

    #[test]
    fn unchanged_shader_reads_share_owned_bytes_without_allocating_or_backend_calls() {
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        let (allocation, space, base) = mapped_memory();
        let mut cache = MaxwellShaderMemoryCache::default();
        let mut access = test_memory::access(space.memory().machine());
        let first = MaxwellShaderMemoryView::new(space.memory(), &mut cache, &mut access)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        let mut unexpected = |_, _| panic!("unchanged shader proof called the owner");
        for _ in 0..1000 {
            let repeated =
                MaxwellShaderMemoryView::new(space.memory(), &mut cache, &mut unexpected)
                    .read(MaxwellShaderStage::Compute, base + 8, 4)
                    .unwrap();
            assert!(Arc::ptr_eq(&first.bytes, &repeated.bytes));
        }
        assert_eq!(ALLOCATIONS.with(|n| n.replace(None).unwrap()), 0);
        test_memory::write(&allocation, 8, &[9]).unwrap();
        let changed = MaxwellShaderMemoryView::new(space.memory(), &mut cache, &mut access)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(changed.bytes[0], 9);
        assert_eq!(first.bytes[0], 0);
        assert!(!first.is_current(space.memory()));
        test_memory::write(&allocation, 8, &[0]).unwrap();
        assert!(!first.is_current(space.memory()));
        assert!(!changed.is_current(space.memory()));
    }

    #[test]
    fn device_publication_is_resolved_by_the_owner_before_shader_recapture() {
        let (allocation, space, base) = mapped_memory();
        let memory = space.memory().machine();
        let mut cache = MaxwellShaderMemoryCache::default();
        let first = MaxwellShaderMemoryView::new(
            space.memory(),
            &mut cache,
            &mut test_memory::access(memory),
        )
        .read(MaxwellShaderStage::Compute, base + 8, 4)
        .unwrap();
        memory
            .publish_device_write(&test_memory::ranges(&allocation), || Ok(()))
            .unwrap();
        assert!(!first.is_current(space.memory()));
        let mut readbacks = 0;
        let mut owner = |ranges: Vec<nixe_memory::AllocationRange>, operation| {
            readbacks += 1;
            memory.write_resolved(&ranges, &[7; 4]).map_err(Arc::new)?;
            memory
                .apply_resolved_access(&ranges, &operation)
                .map_err(Arc::new)
        };
        let next = MaxwellShaderMemoryView::new(space.memory(), &mut cache, &mut owner)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(&*next.bytes, &[7; 4]);
        assert_eq!(readbacks, 1);
        assert!(!first.is_current(space.memory()));
    }

    #[test]
    fn remapping_gpu_va_cannot_refresh_the_old_physical_shader_proof() {
        let (allocation, mut space, base) = mapped_memory();
        let memory = space.memory().machine().clone();
        let mut cache = MaxwellShaderMemoryCache::default();
        let mut access = test_memory::access(&memory);
        let first = MaxwellShaderMemoryView::new(space.memory(), &mut cache, &mut access)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        let address = space.address(base).unwrap();
        space.unmap(address).unwrap();
        assert!(!first.is_current(space.memory()));
        space.reserve(1, 0x1000, 1, base).unwrap();
        space
            .map(crate::MaxwellMapRequest {
                allocation: crate::MaxwellAllocationId::new(1),
                backing: test_memory::ranges(&allocation),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: Some(address),
            })
            .unwrap();
        assert!(!first.is_current(space.memory()));
        let current = MaxwellShaderMemoryView::new(space.memory(), &mut cache, &mut access)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(current.bytes, first.bytes);
        assert!(current.is_current(space.memory()));
    }
}
