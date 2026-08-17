//! Typed `MAXWELL_B` L2 cache-policy state.
//!
//! These registers select guest Maxwell cache policy for independent request
//! producers. They do not directly command cache maintenance and must not be
//! confused with either host CPU caches or a host backend's physical L2.

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// Request class whose ROP L2 eviction priority is being configured.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MaxwellThreeDRopL2CacheRequest {
    PrefetchRead = 0,
    NoninterlockedRead = 1,
    InterlockedRead = 2,
    NoninterlockedWrite = 3,
    InterlockedWrite = 4,
}

impl MaxwellThreeDRopL2CacheRequest {
    const COUNT: usize = 5;

    const fn index(self) -> usize {
        self as usize
    }
}

/// Replacement priority encoded in bits 5:4 of Maxwell L2 controls.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDL2CacheEvictionPolicy {
    EvictFirst = 0,
    EvictNormal = 1,
    EvictLast = 2,
}

impl MaxwellThreeDL2CacheEvictionPolicy {
    pub(super) const fn parse(encoded: u32) -> Option<Self> {
        match encoded {
            0x00 => Some(Self::EvictFirst),
            0x10 => Some(Self::EvictNormal),
            0x20 => Some(Self::EvictLast),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    #[must_use]
    pub const fn encoded(self) -> u32 {
        self.raw() << 4
    }
}

/// Whether VAF requests treat system-memory contents as volatile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDSystemMemoryVolatile {
    Stable = 0,
    Volatile = 1,
}

impl MaxwellThreeDSystemMemoryVolatile {
    const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Stable),
            1 => Some(Self::Volatile),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    #[must_use]
    pub const fn volatile(self) -> bool {
        matches!(self, Self::Volatile)
    }
}

/// Complete `SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS` value.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1399-L1406>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDVafL2CacheControl {
    system_memory: MaxwellThreeDSystemMemoryVolatile,
    policy: MaxwellThreeDL2CacheEvictionPolicy,
}

impl MaxwellThreeDVafL2CacheControl {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        let Some(system_memory) = MaxwellThreeDSystemMemoryVolatile::parse(raw & 1) else {
            return None;
        };
        let Some(policy) = MaxwellThreeDL2CacheEvictionPolicy::parse(raw & 0x30) else {
            return None;
        };
        Some(Self {
            system_memory,
            policy,
        })
    }

    #[must_use]
    pub const fn system_memory(self) -> MaxwellThreeDSystemMemoryVolatile {
        self.system_memory
    }

    #[must_use]
    pub const fn policy(self) -> MaxwellThreeDL2CacheEvictionPolicy {
        self.policy
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.system_memory.raw() | self.policy.encoded()
    }
}

/// One validated L2 cache-policy transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDL2CacheStateWrite {
    VafControl {
        value: MaxwellThreeDVafL2CacheControl,
        source: MaxwellMethodSource,
    },
    RopPolicy {
        request: MaxwellThreeDRopL2CacheRequest,
        value: MaxwellThreeDL2CacheEvictionPolicy,
        source: MaxwellMethodSource,
    },
}

/// Persistent L2 cache-policy configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDL2CacheState {
    vaf_control: MaxwellThreeDRegister<MaxwellThreeDVafL2CacheControl>,
    rop_policies: [MaxwellThreeDRegister<MaxwellThreeDL2CacheEvictionPolicy>;
        MaxwellThreeDRopL2CacheRequest::COUNT],
}

impl MaxwellThreeDL2CacheState {
    #[must_use]
    pub const fn vaf_control(&self) -> &MaxwellThreeDRegister<MaxwellThreeDVafL2CacheControl> {
        &self.vaf_control
    }

    #[must_use]
    pub const fn rop_policy(
        &self,
        request: MaxwellThreeDRopL2CacheRequest,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDL2CacheEvictionPolicy> {
        &self.rop_policies[request.index()]
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDL2CacheStateWrite) {
        match write {
            MaxwellThreeDL2CacheStateWrite::VafControl { value, source } => {
                self.vaf_control = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDL2CacheStateWrite::RopPolicy {
                request,
                value,
                source,
            } => {
                self.rop_policies[request.index()] =
                    MaxwellThreeDRegister::programmed(value.encoded(), value, source);
            }
        }
    }
}
