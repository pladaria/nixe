//! Typed `MAXWELL_B` ROP L2 cache-policy state.
//!
//! These registers select eviction priority for distinct ROP request classes.
//! They do not define guest memory visibility and must not be confused with
//! either host CPU caches or a host backend's physical L2 implementation.
//!
//! NVIDIA publishes the common policy field and all three accepted values for
//! each request class in its pinned public `MAXWELL_B` header:
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L261-L265>
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2127-L2137>

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// ROP request class whose L2 eviction priority is being configured.
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

/// Replacement priority encoded in bits 5:4 of each ROP L2 control method.
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

    /// Unshifted value of the published `POLICY` field.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    /// Complete method argument with the policy placed in bits 5:4.
    #[must_use]
    pub const fn encoded(self) -> u32 {
        self.raw() << 4
    }
}

/// One validated ROP L2 cache-policy transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDRopL2CacheStateWrite {
    Policy {
        request: MaxwellThreeDRopL2CacheRequest,
        value: MaxwellThreeDL2CacheEvictionPolicy,
        source: MaxwellMethodSource,
    },
}

/// Persistent ROP cache-policy configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDRopL2CacheState {
    policies: [MaxwellThreeDRegister<MaxwellThreeDL2CacheEvictionPolicy>;
        MaxwellThreeDRopL2CacheRequest::COUNT],
}

impl MaxwellThreeDRopL2CacheState {
    #[must_use]
    pub const fn policy(
        &self,
        request: MaxwellThreeDRopL2CacheRequest,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDL2CacheEvictionPolicy> {
        &self.policies[request.index()]
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDRopL2CacheStateWrite) {
        match write {
            MaxwellThreeDRopL2CacheStateWrite::Policy {
                request,
                value,
                source,
            } => {
                self.policies[request.index()] =
                    MaxwellThreeDRegister::programmed(value.encoded(), value, source);
            }
        }
    }
}
