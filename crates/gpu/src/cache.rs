//! Fast identities and application-selected bounds for derived GPU caches.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

use twox_hash::XxHash3_128;

/// Default number of entries retained by each frontend shader index.
pub const DEFAULT_SHADER_CACHE_ENTRIES: usize = 4_096;
/// Default number of neutral frontend pipelines retained in memory.
pub const DEFAULT_PIPELINE_CACHE_ENTRIES: usize = 8_192;
/// Default host-specialized variants retained for one neutral pipeline.
pub const DEFAULT_PIPELINE_VARIANTS_PER_RESOURCE: usize = 64;
/// Default host bind groups retained for one neutral descriptor table.
pub const DEFAULT_BIND_GROUPS_PER_DESCRIPTOR_TABLE: usize = 64;
/// Default maximum size of one persistent backend pipeline cache.
pub const DEFAULT_PERSISTENT_PIPELINE_CACHE_BYTES: u64 = 256 * 1024 * 1024;
/// Smallest shader working set capable of retaining one complete neutral
/// graphics or compute pipeline without self-eviction.
pub const MIN_SHADER_CACHE_ENTRIES: usize = 6;

/// Cache working-set policy selected by the application composition root.
///
/// These values are software budgets, not host hardware capabilities.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuCacheConfiguration {
    shader_entries: usize,
    pipeline_entries: usize,
    pipeline_variants_per_resource: usize,
    bind_groups_per_descriptor_table: usize,
    persistent_pipeline_cache_bytes: u64,
}

impl GpuCacheConfiguration {
    pub const fn new(
        shader_entries: usize,
        pipeline_entries: usize,
        pipeline_variants_per_resource: usize,
        bind_groups_per_descriptor_table: usize,
        persistent_pipeline_cache_bytes: u64,
    ) -> Result<Self, GpuCacheConfigurationError> {
        if shader_entries < MIN_SHADER_CACHE_ENTRIES {
            return Err(GpuCacheConfigurationError::BelowMinimum {
                field: "shader_entries",
                minimum: MIN_SHADER_CACHE_ENTRIES,
            });
        }
        if pipeline_entries == 0 {
            return Err(GpuCacheConfigurationError::BelowMinimum {
                field: "pipeline_entries",
                minimum: 1,
            });
        }
        if pipeline_variants_per_resource == 0 {
            return Err(GpuCacheConfigurationError::BelowMinimum {
                field: "pipeline_variants_per_resource",
                minimum: 1,
            });
        }
        if bind_groups_per_descriptor_table == 0 {
            return Err(GpuCacheConfigurationError::BelowMinimum {
                field: "bind_groups_per_descriptor_table",
                minimum: 1,
            });
        }
        if persistent_pipeline_cache_bytes == 0 {
            return Err(GpuCacheConfigurationError::BelowMinimum {
                field: "persistent_pipeline_cache_bytes",
                minimum: 1,
            });
        }
        Ok(Self {
            shader_entries,
            pipeline_entries,
            pipeline_variants_per_resource,
            bind_groups_per_descriptor_table,
            persistent_pipeline_cache_bytes,
        })
    }

    #[must_use]
    pub const fn shader_entries(self) -> usize {
        self.shader_entries
    }

    #[must_use]
    pub const fn pipeline_entries(self) -> usize {
        self.pipeline_entries
    }

    #[must_use]
    pub const fn pipeline_variants_per_resource(self) -> usize {
        self.pipeline_variants_per_resource
    }

    #[must_use]
    pub const fn bind_groups_per_descriptor_table(self) -> usize {
        self.bind_groups_per_descriptor_table
    }

    #[must_use]
    pub const fn persistent_pipeline_cache_bytes(self) -> u64 {
        self.persistent_pipeline_cache_bytes
    }
}

impl Default for GpuCacheConfiguration {
    fn default() -> Self {
        Self {
            shader_entries: DEFAULT_SHADER_CACHE_ENTRIES,
            pipeline_entries: DEFAULT_PIPELINE_CACHE_ENTRIES,
            pipeline_variants_per_resource: DEFAULT_PIPELINE_VARIANTS_PER_RESOURCE,
            bind_groups_per_descriptor_table: DEFAULT_BIND_GROUPS_PER_DESCRIPTOR_TABLE,
            persistent_pipeline_cache_bytes: DEFAULT_PERSISTENT_PIPELINE_CACHE_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCacheConfigurationError {
    BelowMinimum { field: &'static str, minimum: usize },
}

impl Display for GpuCacheConfigurationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BelowMinimum { field, minimum } => write!(
                formatter,
                "GPU cache capacity `{field}` must be at least {minimum}"
            ),
        }
    }
}

impl Error for GpuCacheConfigurationError {}

/// Computes the non-cryptographic 128-bit identity used by in-memory GPU
/// caches. It is intentionally not a persistent serialization format.
///
/// Optimized builds treat the result as the identity. Development builds must
/// additionally compare the exact semantic key on a hit so collisions and
/// incomplete key definitions fail immediately.
#[must_use]
#[inline]
pub fn cache_fingerprint<T: Hash + ?Sized>(value: &T) -> u128 {
    let mut hasher = CacheFingerprintHasher(XxHash3_128::new());
    value.hash(&mut hasher);
    hasher.0.finish_128()
}

struct CacheFingerprintHasher(XxHash3_128);

impl Hasher for CacheFingerprintHasher {
    fn finish(&self) -> u64 {
        self.0.finish_128() as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuCacheConfiguration, cache_fingerprint};

    #[test]
    fn fingerprint_is_stable_for_equal_semantic_inputs() {
        assert_eq!(
            cache_fingerprint(&(7_u32, &[1_u8, 2, 3][..])),
            cache_fingerprint(&(7_u32, &[1_u8, 2, 3][..]))
        );
        assert_ne!(
            cache_fingerprint(&(7_u32, &[1_u8, 2, 3][..])),
            cache_fingerprint(&(8_u32, &[1_u8, 2, 3][..]))
        );
    }

    #[test]
    fn cache_configuration_rejects_zero_capacities() {
        assert!(GpuCacheConfiguration::new(0, 1, 1, 1, 1).is_err());
        assert!(GpuCacheConfiguration::new(6, 0, 1, 1, 1).is_err());
        assert!(GpuCacheConfiguration::new(6, 1, 0, 1, 1).is_err());
        assert!(GpuCacheConfiguration::new(6, 1, 1, 0, 1).is_err());
        assert!(GpuCacheConfiguration::new(6, 1, 1, 1, 0).is_err());
        assert!(GpuCacheConfiguration::new(6, 1, 1, 1, 1).is_ok());
    }
}
