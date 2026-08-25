//! Validated resource policy for one production JIT provider.

use std::fmt;

/// Default maximum number of compiled regions retained by one JIT domain.
pub const DEFAULT_MAX_CACHED_REGIONS: usize = 1_024;
/// Default maximum native bytes retained by one JIT domain.
pub const DEFAULT_MAX_CACHE_BYTES: usize = 48 * 1024 * 1024;
/// Default maximum number of simultaneous compilations in one JIT domain.
pub const DEFAULT_MAX_CONCURRENT_COMPILATIONS: usize = 4;

const MIN_CACHE_BYTES: usize = 1024 * 1024;
const MAX_CACHED_REGIONS: usize = DEFAULT_MAX_CACHED_REGIONS;
const MAX_CACHE_BYTES: usize = DEFAULT_MAX_CACHE_BYTES;
const MAX_CONCURRENT_COMPILATIONS: usize = 64;

/// Product-configurable JIT resource bounds.
///
/// These values describe implementation-independent resource quantities. They
/// deliberately expose neither Cranelift settings nor native ABI details.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JitConfiguration {
    max_cached_regions: usize,
    max_cache_bytes: usize,
    max_concurrent_compilations: usize,
}

impl JitConfiguration {
    /// Validates resource bounds before a provider can be registered.
    pub fn new(
        max_cached_regions: usize,
        max_cache_bytes: usize,
        max_concurrent_compilations: usize,
    ) -> Result<Self, JitConfigurationError> {
        if !(1..=MAX_CACHED_REGIONS).contains(&max_cached_regions) {
            return Err(JitConfigurationError::MaxCachedRegions {
                value: max_cached_regions,
                maximum: MAX_CACHED_REGIONS,
            });
        }
        if !(MIN_CACHE_BYTES..=MAX_CACHE_BYTES).contains(&max_cache_bytes) {
            return Err(JitConfigurationError::MaxCacheBytes {
                value: max_cache_bytes,
                minimum: MIN_CACHE_BYTES,
                maximum: MAX_CACHE_BYTES,
            });
        }
        if !(1..=MAX_CONCURRENT_COMPILATIONS).contains(&max_concurrent_compilations) {
            return Err(JitConfigurationError::MaxConcurrentCompilations {
                value: max_concurrent_compilations,
                maximum: MAX_CONCURRENT_COMPILATIONS,
            });
        }
        Ok(Self {
            max_cached_regions,
            max_cache_bytes,
            max_concurrent_compilations,
        })
    }

    #[must_use]
    pub const fn max_cached_regions(self) -> usize {
        self.max_cached_regions
    }

    #[must_use]
    pub const fn max_cache_bytes(self) -> usize {
        self.max_cache_bytes
    }

    #[must_use]
    pub const fn max_concurrent_compilations(self) -> usize {
        self.max_concurrent_compilations
    }
}

impl Default for JitConfiguration {
    fn default() -> Self {
        Self {
            max_cached_regions: DEFAULT_MAX_CACHED_REGIONS,
            max_cache_bytes: DEFAULT_MAX_CACHE_BYTES,
            max_concurrent_compilations: DEFAULT_MAX_CONCURRENT_COMPILATIONS,
        }
    }
}

/// Invalid JIT resource policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JitConfigurationError {
    MaxCachedRegions {
        value: usize,
        maximum: usize,
    },
    MaxCacheBytes {
        value: usize,
        minimum: usize,
        maximum: usize,
    },
    MaxConcurrentCompilations {
        value: usize,
        maximum: usize,
    },
}

impl fmt::Display for JitConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::MaxCachedRegions { value, maximum } => write!(
                formatter,
                "max_cached_regions must be between 1 and {maximum}, got {value}"
            ),
            Self::MaxCacheBytes {
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "max_cache_bytes must be between {minimum} and {maximum}, got {value}"
            ),
            Self::MaxConcurrentCompilations { value, maximum } => write!(
                formatter,
                "max_concurrent_compilations must be between 1 and {maximum}, got {value}"
            ),
        }
    }
}

impl std::error::Error for JitConfigurationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_explicit_bounds_reject_zero_or_excess() {
        let defaults = JitConfiguration::default();
        assert_eq!(
            JitConfiguration::new(
                defaults.max_cached_regions(),
                defaults.max_cache_bytes(),
                defaults.max_concurrent_compilations(),
            ),
            Ok(defaults)
        );
        assert!(matches!(
            JitConfiguration::new(0, DEFAULT_MAX_CACHE_BYTES, 1),
            Err(JitConfigurationError::MaxCachedRegions { .. })
        ));
        assert!(matches!(
            JitConfiguration::new(1, 0, 1),
            Err(JitConfigurationError::MaxCacheBytes { .. })
        ));
        assert!(matches!(
            JitConfiguration::new(1, MIN_CACHE_BYTES, 0),
            Err(JitConfigurationError::MaxConcurrentCompilations { .. })
        ));
    }
}
