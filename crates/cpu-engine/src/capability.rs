//! Host discovery and engine capability declarations.

use core::num::NonZeroU64;

use nixe_cpu::profile::CpuProfileId;

use crate::EngineId;

/// Stable implementation family used for selection and diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EngineKind {
    Interpreter,
    NativeCodeExecution,
    Test,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HostArchitecture {
    Aarch64,
    X86_64,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HostCapabilities {
    pub architecture: HostArchitecture,
    pub logical_parallelism: Option<usize>,
}

impl HostCapabilities {
    /// Discovers only portable host facts. Privileged virtualization features
    /// remain provider-specific probes and are never guessed from the ISA.
    #[must_use]
    pub fn discover() -> Self {
        let architecture = if cfg!(target_arch = "aarch64") {
            HostArchitecture::Aarch64
        } else if cfg!(target_arch = "x86_64") {
            HostArchitecture::X86_64
        } else {
            HostArchitecture::Other
        };
        Self {
            architecture,
            logical_parallelism: std::thread::available_parallelism()
                .ok()
                .map(std::num::NonZero::get),
        }
    }
}

/// Capabilities offered by one provider on the current host.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EngineCapabilities {
    pub a64: bool,
    pub a32: bool,
    pub t32: bool,
    pub precise_instruction_budget: bool,
    pub instruction_trace: bool,
    pub native_execution: bool,
    /// Distinct executors may run concurrently without sharing mutable state.
    pub concurrent_executors: bool,
    /// Maximum guest instructions between control-path polls. `None` means the
    /// provider offers no bounded safepoint guarantee.
    pub max_safepoint_instructions: Option<NonZeroU64>,
    /// Mapping and code invalidation epochs are acknowledged before reuse.
    pub acknowledged_invalidation: bool,
}

impl EngineCapabilities {
    #[must_use]
    pub const fn supports_profile(self, profile: CpuProfileId) -> bool {
        let _ = profile;
        self.a64
    }

    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        (!required.a64 || self.a64)
            && (!required.a32 || self.a32)
            && (!required.t32 || self.t32)
            && (!required.precise_instruction_budget || self.precise_instruction_budget)
            && (!required.instruction_trace || self.instruction_trace)
            && (!required.native_execution || self.native_execution)
            && (!required.concurrent_executors || self.concurrent_executors)
            && match (
                self.max_safepoint_instructions,
                required.max_safepoint_instructions,
            ) {
                (_, None) => true,
                (Some(offered), Some(required)) => offered.get() <= required.get(),
                (None, Some(_)) => false,
            }
            && (!required.acknowledged_invalidation || self.acknowledged_invalidation)
    }

    /// Whether a parallel runtime must retain an out-of-band control path for
    /// every executor created by this provider.
    #[must_use]
    pub const fn requires_control_path(self) -> bool {
        self.concurrent_executors
            || self.max_safepoint_instructions.is_some()
            || self.acknowledged_invalidation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineDescriptor {
    pub id: EngineId,
    pub name: Box<str>,
    pub kind: EngineKind,
    pub capabilities: EngineCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CapabilityRejectionReason {
    HostUnavailable,
    GuestProfileUnsupported,
    MissingCapabilities,
    PrivilegeUnavailable,
    PlatformUnsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityRejection {
    pub engine: EngineId,
    pub reason: CapabilityRejectionReason,
    pub detail: Box<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityReport {
    pub descriptor: EngineDescriptor,
    pub available: bool,
    pub rejections: Box<[CapabilityRejection]>,
}
