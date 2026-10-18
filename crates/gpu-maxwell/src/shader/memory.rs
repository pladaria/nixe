//! Consumed canonical memory reads with retained freshness dependencies.
const MAXWELL_SHADER_READ_LIMIT: usize = 64 * 1024;
use crate::{MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellShaderStage};
use nixe_memory::{CanonicalBackingRange, CanonicalCpuWriteDependency, MemoryPermissions};
use std::sync::Arc;

/// Retained address proofs for capture-time constant and texture observations.
/// Like Eden's descriptor tables, reuse immutable mapping/observation storage
/// rather than allocating it for every specialization revalidation.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/descriptor_table.h
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

pub(super) struct MaxwellShaderMemoryView<'a> {
    address_space: &'a MaxwellGpuAddressSpace,
    cache: &'a mut MaxwellShaderMemoryCache,
}

#[derive(Clone, Debug)]
pub(super) struct MaxwellShaderRead {
    pub(super) address: u64,
    pub(super) bytes: Arc<[u8]>,
    pub(super) snapshot: CanonicalBackingRange,
    pub(super) cpu_writes: CanonicalCpuWriteDependency,
    pub(super) mappings: Arc<[crate::MaxwellGpuMapping]>,
    address_space: crate::MaxwellAddressSpaceId,
    mapping_generation: nixe_memory::MappingGeneration,
}

impl MaxwellShaderRead {
    fn mapping_is_current(&self, space: &MaxwellGpuAddressSpace) -> bool {
        self.address_space == space.id()
            && (self.mapping_generation == space.mapping_generation()
                || self
                    .mappings
                    .iter()
                    .all(|mapping| space.retained_mapping_is_current(mapping)))
    }

    pub(super) fn is_current(&self, space: &MaxwellGpuAddressSpace) -> bool {
        if !self.mapping_is_current(space) {
            return false;
        }
        if !self
            .snapshot
            .subrange_is_cpu_visible(0, self.bytes.len() as u64)
            .unwrap_or(false)
        {
            return false;
        }
        // Page dirtiness can come from unrelated words sharing this page.
        self.cpu_writes.remains_current()
            || self
                .cpu_writes
                .confirm_unchanged(&self.bytes)
                .unwrap_or(false)
    }
}

impl<'a> MaxwellShaderMemoryView<'a> {
    pub(super) const fn new(
        address_space: &'a MaxwellGpuAddressSpace,
        cache: &'a mut MaxwellShaderMemoryCache,
    ) -> Self {
        Self {
            address_space,
            cache,
        }
    }

    pub(super) fn read(
        &mut self,
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
    ) -> Result<MaxwellShaderRead, ShaderCaptureError> {
        let key = (self.address_space.id(), address, size);
        let previous = self.cache.reads.get(&key);
        if let Some(read) = previous
            && read.is_current(self.address_space)
        {
            return Ok(read.clone());
        }
        let previous = previous.cloned();
        let read = self.read_uncached(stage, address, size, previous.as_ref())?;
        let bytes = std::mem::size_of::<MaxwellShaderRead>()
            + read.bytes.len()
            + std::mem::size_of_val(read.mappings.as_ref())
            + read.snapshot.metadata_bytes()
            + 512 * read.snapshot.pages().count();
        self.cache.reads.insert(key, read.clone(), bytes);
        Ok(read)
    }

    fn read_uncached(
        &self,
        stage: MaxwellShaderStage,
        address: u64,
        size: usize,
        previous: Option<&MaxwellShaderRead>,
    ) -> Result<MaxwellShaderRead, ShaderCaptureError> {
        if size > MAXWELL_SHADER_READ_LIMIT {
            return Err(ShaderCaptureError::ReadTooLarge {
                requested: size,
                limit: MAXWELL_SHADER_READ_LIMIT,
            });
        }
        let size_u64 = size as u64;
        let previous = previous.filter(|read| read.mapping_is_current(self.address_space));
        let (snapshot, mappings) = if let Some(previous) = previous {
            (previous.snapshot.clone(), Arc::clone(&previous.mappings))
        } else {
            self.resolve(stage, address, size_u64)?
        };
        // Independent captured bytes retain the previous mapping proof; a
        // real CPU/device mutation requires a new observation baseline.
        let cpu_writes = if let Some(previous) = previous.filter(|read| {
            read.cpu_writes.remains_current()
                && read
                    .snapshot
                    .subrange_is_cpu_visible(0, size_u64)
                    .unwrap_or(false)
        }) {
            previous.cpu_writes.clone()
        } else {
            CanonicalCpuWriteDependency::capture(&snapshot).map_err(|error| {
                ShaderCaptureError::Memory {
                    stage,
                    address,
                    error: MaxwellGpuAccessError::Backing(error),
                }
            })?
        };
        let mut bytes = vec![0; size];
        snapshot
            .read(0, &mut bytes)
            .map_err(|error| ShaderCaptureError::Memory {
                stage,
                address,
                error: MaxwellGpuAccessError::Backing(error),
            })?;
        Ok(MaxwellShaderRead {
            address,
            bytes: bytes.into(),
            snapshot,
            cpu_writes,
            mappings,
            address_space: self.address_space.id(),
            mapping_generation: self.address_space.mapping_generation(),
        })
    }

    fn resolve(
        &self,
        stage: MaxwellShaderStage,
        address: u64,
        size_u64: u64,
    ) -> Result<(CanonicalBackingRange, Arc<[crate::MaxwellGpuMapping]>), ShaderCaptureError> {
        let gpu_address = self
            .address_space
            .address(address)
            .map_err(MaxwellGpuAccessError::Address)
            .map_err(|error| ShaderCaptureError::Memory {
                stage,
                address,
                error,
            })?;
        let resolved = self
            .address_space
            .resolve_range(gpu_address, size_u64, MemoryPermissions::READ)
            .map_err(|error| ShaderCaptureError::Memory {
                stage,
                address,
                error,
            })?;
        let mut canonical_segments = Vec::new();
        let mut mappings = Vec::new();
        for segment in resolved.segments() {
            segment
                .mapping()
                .backing()
                .snapshot_subrange_into(
                    segment.backing_offset(),
                    segment.size(),
                    &mut canonical_segments,
                )
                .map_err(|_| ShaderCaptureError::SourceChangedDuringRead { stage, address })?;
            if !mappings
                .iter()
                .any(|mapping: &crate::MaxwellGpuMapping| mapping.id() == segment.mapping().id())
            {
                mappings.push(segment.mapping().clone());
            }
        }
        let snapshot = CanonicalBackingRange::new(canonical_segments)
            .map_err(|_| ShaderCaptureError::SourceChangedDuringRead { stage, address })?;
        Ok((snapshot, mappings.into()))
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
    SourceChangedDuringRead {
        stage: MaxwellShaderStage,
        address: u64,
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
            Self::SourceChangedDuringRead { stage, address } => {
                write!(f, "{stage:?} shader changed during capture at {address:#x}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::test_support::{mapped_memory, write_shader_words};

    #[test]
    fn cached_reads_share_unchanged_proofs_without_allocating_and_keep_old_bytes() {
        use crate::engines::tests::register_benchmark::ALLOCATIONS;
        let (allocation, space, base) = mapped_memory();
        let mut cache = MaxwellShaderMemoryCache::default();

        let first = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        ALLOCATIONS.with(|n| n.set(Some(0)));
        for _ in 0..1000 {
            let current = MaxwellShaderMemoryView::new(&space, &mut cache)
                .read(MaxwellShaderStage::Compute, base + 8, 4)
                .unwrap();
            assert!(Arc::ptr_eq(&first.bytes, &current.bytes));
        }
        assert_eq!(ALLOCATIONS.with(|n| n.replace(None).unwrap()), 0);
        allocation.write(8, &[9]).unwrap();
        let next = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(next.bytes[0], 9);
        assert_eq!(first.bytes[0], 0);
        assert!(!first.is_current(&space));
        write_shader_words(&space, &[(base + 8, 7)]);
        let overlaid = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(overlaid.bytes[0], 7);
        assert!(overlaid.is_current(&space));
        assert!(!next.is_current(&space));
        allocation.write(8, &[9]).unwrap();
        let after = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(after.bytes[0], 9);
        assert!(next.is_current(&space));
    }

    #[test]
    fn cached_shader_reads_resolve_device_producers_before_reusing_cpu_bytes() {
        use nixe_memory::{
            CpuVisibilityRequest, DeviceAccessDeclaration, DeviceVisibilityPoint, NonCpuDeviceId,
            VisibilityCoordinator, VisibilityCoordinatorError,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Producer(AtomicUsize);
        impl VisibilityCoordinator for Producer {
            fn cache_cpu_page(
                &self,
                _: nixe_memory::DeviceVisibilityRequest,
                bytes: &[u8],
            ) -> Result<(), VisibilityCoordinatorError> {
                assert!(!bytes.is_empty());
                Ok(())
            }
            fn make_cpu_visible(
                &self,
                request: CpuVisibilityRequest,
            ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(vec![7; request.size].into_boxed_slice())
            }
        }
        let (allocation, space, base) = mapped_memory();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut cache = MaxwellShaderMemoryCache::default();

        let first = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        let producer = Arc::new(Producer(AtomicUsize::new(0)));
        CanonicalBackingRange::publish_device_writes(
            [(
                &range,
                DeviceAccessDeclaration::write(
                    NonCpuDeviceId::new(1),
                    DeviceVisibilityPoint::new(1),
                    DeviceVisibilityPoint::new(1),
                )
                .unwrap(),
            )],
            producer.clone(),
        )
        .unwrap();
        assert!(!first.is_current(&space));
        assert_eq!(producer.0.load(Ordering::Relaxed), 0);
        let next = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(next.bytes.as_ref(), &[7; 4]);
        assert_eq!(producer.0.load(Ordering::Relaxed), 1);
        assert!(!first.is_current(&space));
    }

    #[test]
    fn dirty_page_revalidates_consumed_bytes_and_keeps_real_changes_stale() {
        let (allocation, space, base) = mapped_memory();
        let mut cache = MaxwellShaderMemoryCache::default();
        allocation.write(8, &[1, 2, 3, 4]).unwrap();

        let read = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        allocation.write(0x800, &[5]).unwrap();
        assert!(!read.cpu_writes.remains_current());
        assert!(read.is_current(&space));
        assert!(read.cpu_writes.remains_current());
        allocation.write(8, &[9]).unwrap();
        assert!(!read.is_current(&space));
        assert!(!read.is_current(&space));
        allocation.write(8, &[1]).unwrap();
        assert!(read.is_current(&space));
    }

    #[test]
    fn committed_word_updates_preserve_unrelated_bytes_and_owned_versions() {
        let (allocation, space, base) = mapped_memory();
        let mut cache = MaxwellShaderMemoryCache::default();
        write_shader_words(&space, &[(base + 8, 7)]);
        let read = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 8)
            .unwrap();
        assert!(read.is_current(&space));
        allocation.write(0x800, &[1]).unwrap();
        assert!(read.is_current(&space));
        write_shader_words(&space, &[(base + 8, 9)]);
        assert!(!read.is_current(&space));
        assert_eq!(read.bytes.as_ref(), &[7, 0, 0, 0, 0, 0, 0, 0]);
        write_shader_words(&space, &[(base + 8, 7)]);
        assert!(read.is_current(&space));
        allocation.write(12, &[1]).unwrap();
        assert!(!read.is_current(&space));
    }

    #[test]
    fn mapping_epoch_fast_path_rejects_unmapped_and_replaced_shader_sources() {
        let (allocation, mut space, base) = mapped_memory();
        let mut cache = MaxwellShaderMemoryCache::default();

        let read = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert!(read.is_current(&space));
        let address = space.address(base).unwrap();
        assert_eq!(space.unmap(address).unwrap().len(), 1);
        assert!(!read.is_current(&space));
        space.reserve(1, 0x1000, 1, base).unwrap();
        space
            .map(crate::MaxwellMapRequest {
                allocation: crate::MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
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
        assert!(!read.is_current(&space));
        let current = MaxwellShaderMemoryView::new(&space, &mut cache)
            .read(MaxwellShaderStage::Compute, base + 8, 4)
            .unwrap();
        assert_eq!(current.bytes, read.bytes);
        assert!(current.is_current(&space));
    }
}
