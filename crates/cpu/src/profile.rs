//! Immutable guest CPU behavior profiles.

use core::fmt;

use crate::{address::AddressSpaceId, location::ExecutionState};

/// Stable identity of an immutable guest CPU profile.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CpuProfileId(u64);

impl CpuProfileId {
    /// Creates a profile identity from its registry value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the registry value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CpuProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "profile=0x{:016x}", self.0)
    }
}

/// Architecture revision whose behavior is exposed to guest software.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ArchitectureRevision {
    /// Armv8-A; later optional extensions are represented as capabilities.
    Armv8A,
    /// The revision has not yet been established by admissible evidence.
    Unknown,
}

impl fmt::Display for ArchitectureRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Armv8A => "armv8-a",
            Self::Unknown => "unknown",
        })
    }
}

/// Whether evidence establishes the availability of a profile capability.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CapabilityStatus {
    /// The capability is known to be unavailable to the guest.
    Disabled,
    /// The capability is available to the guest.
    Enabled,
    /// Available evidence does not establish either behavior.
    Unknown,
}

impl fmt::Display for CapabilityStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::Unknown => "unknown",
        })
    }
}

/// Named instruction-set capability used by decoder rules.
///
/// Decoders gate encodings on these architectural names. They must never test
/// for a console or SoC name.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum InstructionFeature {
    AdvancedSimd,
    Aes,
    Sha1,
    Sha256,
    Crc32,
    LargeSystemExtensions,
    Fp16,
    Rdm,
    DotProduct,
    Sve,
    Sve2,
}

impl InstructionFeature {
    const COUNT: usize = Self::Sve2 as usize + 1;
    const ALL: [Self; Self::COUNT] = [
        Self::AdvancedSimd,
        Self::Aes,
        Self::Sha1,
        Self::Sha256,
        Self::Crc32,
        Self::LargeSystemExtensions,
        Self::Fp16,
        Self::Rdm,
        Self::DotProduct,
        Self::Sve,
        Self::Sve2,
    ];

    /// Stable decoder-table and diagnostic name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::AdvancedSimd => "advanced-simd",
            Self::Aes => "aes",
            Self::Sha1 => "sha1",
            Self::Sha256 => "sha256",
            Self::Crc32 => "crc32",
            Self::LargeSystemExtensions => "large-system-extensions",
            Self::Fp16 => "fp16",
            Self::Rdm => "rdm",
            Self::DotProduct => "dot-product",
            Self::Sve => "sve",
            Self::Sve2 => "sve2",
        }
    }
}

impl fmt::Display for InstructionFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Typed reason why an instruction capability cannot be used by a profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionFeatureRejection {
    pub feature: InstructionFeature,
    pub status: CapabilityStatus,
}

impl fmt::Display for InstructionFeatureRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "feature={} status={}", self.feature, self.status)
    }
}

impl std::error::Error for InstructionFeatureRejection {}

/// Immutable statuses for all named instruction capabilities.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionFeatures {
    statuses: [CapabilityStatus; InstructionFeature::COUNT],
}

impl InstructionFeatures {
    /// Creates a feature set in which every optional capability is unknown.
    #[must_use]
    pub const fn all_unknown() -> Self {
        Self {
            statuses: [CapabilityStatus::Unknown; InstructionFeature::COUNT],
        }
    }

    /// Returns a new feature set with one named capability changed.
    #[must_use]
    pub const fn with(mut self, feature: InstructionFeature, status: CapabilityStatus) -> Self {
        self.statuses[feature as usize] = status;
        self
    }

    /// Returns the evidence status of a named capability.
    #[must_use]
    pub const fn status(self, feature: InstructionFeature) -> CapabilityStatus {
        self.statuses[feature as usize]
    }

    /// Returns whether a decoder may execute an instruction requiring `feature`.
    #[must_use]
    pub const fn supports(self, feature: InstructionFeature) -> bool {
        matches!(self.status(feature), CapabilityStatus::Enabled)
    }
}

impl Default for InstructionFeatures {
    fn default() -> Self {
        Self::all_unknown()
    }
}

impl fmt::Display for InstructionFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, feature) in InstructionFeature::ALL.into_iter().enumerate() {
            if index != 0 {
                f.write_str(",")?;
            }
            write!(f, "{}={}", feature, self.status(feature))?;
        }
        Ok(())
    }
}

/// Set of execution states legal for one process profile.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ExecutionStateSet(u8);

impl ExecutionStateSet {
    const A64: u8 = 1 << 0;
    const A32: u8 = 1 << 1;
    const T32: u8 = 1 << 2;

    /// No execution state is legal.
    pub const NONE: Self = Self(0);
    /// Only A64 is legal.
    pub const A64_ONLY: Self = Self(Self::A64);
    /// A64, A32, and T32 are legal.
    pub const A64_A32_T32: Self = Self(Self::A64 | Self::A32 | Self::T32);

    /// Creates a set containing one execution state.
    #[must_use]
    pub const fn from_state(state: ExecutionState) -> Self {
        Self(Self::bit(state))
    }

    /// Returns a new set containing `state` in addition to existing states.
    #[must_use]
    pub const fn with(mut self, state: ExecutionState) -> Self {
        self.0 |= Self::bit(state);
        self
    }

    /// Tests whether `state` is legal.
    #[must_use]
    pub const fn contains(self, state: ExecutionState) -> bool {
        self.0 & Self::bit(state) != 0
    }

    const fn bit(state: ExecutionState) -> u8 {
        match state {
            ExecutionState::A64 => Self::A64,
            ExecutionState::A32 => Self::A32,
            ExecutionState::T32 => Self::T32,
        }
    }
}

impl fmt::Display for ExecutionStateSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        for state in [
            ExecutionState::A64,
            ExecutionState::A32,
            ExecutionState::T32,
        ] {
            if self.contains(state) {
                write!(f, "{separator}{state}")?;
                separator = "|";
            }
        }
        if separator.is_empty() {
            f.write_str("none")?;
        }
        Ok(())
    }
}

/// A profile field whose value may still be an open research question.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProfileValue<T> {
    Known(T),
    Unknown,
}

impl<T: fmt::Display> fmt::Display for ProfileValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known(value) => value.fmt(f),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// Guest-visible address-space assumptions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AddressSpaceProfile {
    pub virtual_address_bits: ProfileValue<u8>,
    pub physical_address_bits: ProfileValue<u8>,
    pub translation_granule_bytes: ProfileValue<u32>,
}

impl AddressSpaceProfile {
    pub const UNKNOWN: Self = Self {
        virtual_address_bits: ProfileValue::Unknown,
        physical_address_bits: ProfileValue::Unknown,
        translation_granule_bytes: ProfileValue::Unknown,
    };
}

/// Guest-visible floating-point behavior not represented by instruction bits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FloatingPointProfile {
    pub scalar: CapabilityStatus,
    pub advanced_simd: CapabilityStatus,
    pub fp16: CapabilityStatus,
}

impl FloatingPointProfile {
    pub const UNKNOWN: Self = Self {
        scalar: CapabilityStatus::Unknown,
        advanced_simd: CapabilityStatus::Unknown,
        fp16: CapabilityStatus::Unknown,
    };
}

/// Guest-visible cache-maintenance behavior.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CacheMaintenanceProfile {
    pub user_cache_maintenance: CapabilityStatus,
}

impl CacheMaintenanceProfile {
    pub const UNKNOWN: Self = Self {
        user_cache_maintenance: CapabilityStatus::Unknown,
    };
}

/// Guest-visible exception behavior.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExceptionProfile {
    pub user_mode_exceptions: CapabilityStatus,
}

impl ExceptionProfile {
    pub const UNKNOWN: Self = Self {
        user_mode_exceptions: CapabilityStatus::Unknown,
    };
}

/// Guest-visible timer behavior.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TimerProfile {
    pub virtual_counter: CapabilityStatus,
}

impl TimerProfile {
    pub const UNKNOWN: Self = Self {
        virtual_counter: CapabilityStatus::Unknown,
    };
}

/// Immutable description of CPU behavior exposed to a guest process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestCpuProfile {
    id: CpuProfileId,
    architecture: ArchitectureRevision,
    allowed_execution_states: ExecutionStateSet,
    instruction_features: InstructionFeatures,
    address_space: AddressSpaceProfile,
    floating_point: FloatingPointProfile,
    cache_maintenance: CacheMaintenanceProfile,
    exception_model: ExceptionProfile,
    timer_model: TimerProfile,
}

impl GuestCpuProfile {
    /// Stable registry identity of the Switch 1 behavior profile.
    pub const SWITCH_1_ID: CpuProfileId = CpuProfileId::new(1);
    /// Stable registry identity of the provisional Switch 2 native profile.
    pub const SWITCH_2_NATIVE_ID: CpuProfileId = CpuProfileId::new(2);

    /// Constructs a custom profile, primarily for conformance tests and research.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        id: CpuProfileId,
        architecture: ArchitectureRevision,
        allowed_execution_states: ExecutionStateSet,
        instruction_features: InstructionFeatures,
        address_space: AddressSpaceProfile,
        floating_point: FloatingPointProfile,
        cache_maintenance: CacheMaintenanceProfile,
        exception_model: ExceptionProfile,
        timer_model: TimerProfile,
    ) -> Self {
        Self {
            id,
            architecture,
            allowed_execution_states,
            instruction_features,
            address_space,
            floating_point,
            cache_maintenance,
            exception_model,
            timer_model,
        }
    }

    /// Switch 1 profile. Validated process metadata chooses the initial state.
    #[must_use]
    pub const fn switch_1() -> Self {
        Self::new(
            Self::SWITCH_1_ID,
            ArchitectureRevision::Armv8A,
            ExecutionStateSet::A64_A32_T32,
            InstructionFeatures::all_unknown(),
            AddressSpaceProfile::UNKNOWN,
            FloatingPointProfile::UNKNOWN,
            CacheMaintenanceProfile::UNKNOWN,
            ExceptionProfile::UNKNOWN,
            TimerProfile::UNKNOWN,
        )
    }

    /// Conservative Switch 2 native profile containing only verified behavior.
    ///
    /// Native processes are provisionally A64-only. The exact architecture and
    /// optional instruction capabilities remain explicitly unknown.
    #[must_use]
    pub const fn switch_2_native() -> Self {
        Self::new(
            Self::SWITCH_2_NATIVE_ID,
            ArchitectureRevision::Unknown,
            ExecutionStateSet::A64_ONLY,
            InstructionFeatures::all_unknown(),
            AddressSpaceProfile::UNKNOWN,
            FloatingPointProfile::UNKNOWN,
            CacheMaintenanceProfile::UNKNOWN,
            ExceptionProfile::UNKNOWN,
            TimerProfile::UNKNOWN,
        )
    }

    #[must_use]
    pub const fn id(self) -> CpuProfileId {
        self.id
    }

    #[must_use]
    pub const fn architecture(self) -> ArchitectureRevision {
        self.architecture
    }

    #[must_use]
    pub const fn allowed_execution_states(self) -> ExecutionStateSet {
        self.allowed_execution_states
    }

    #[must_use]
    pub const fn instruction_features(self) -> InstructionFeatures {
        self.instruction_features
    }

    #[must_use]
    pub const fn instruction_feature_status(self, feature: InstructionFeature) -> CapabilityStatus {
        self.instruction_features.status(feature)
    }

    #[must_use]
    pub const fn supports_instruction_feature(self, feature: InstructionFeature) -> bool {
        self.instruction_features.supports(feature)
    }

    /// Applies the decoder's common capability gate.
    ///
    /// Unknown capabilities are conservatively rejected, while their evidence
    /// status remains distinct from a capability known to be disabled.
    pub const fn require_instruction_feature(
        self,
        feature: InstructionFeature,
    ) -> Result<(), InstructionFeatureRejection> {
        let status = self.instruction_feature_status(feature);
        if matches!(status, CapabilityStatus::Enabled) {
            Ok(())
        } else {
            Err(InstructionFeatureRejection { feature, status })
        }
    }

    #[must_use]
    pub const fn address_space(self) -> AddressSpaceProfile {
        self.address_space
    }

    #[must_use]
    pub const fn floating_point(self) -> FloatingPointProfile {
        self.floating_point
    }

    #[must_use]
    pub const fn cache_maintenance(self) -> CacheMaintenanceProfile {
        self.cache_maintenance
    }

    #[must_use]
    pub const fn exception_model(self) -> ExceptionProfile {
        self.exception_model
    }

    #[must_use]
    pub const fn timer_model(self) -> TimerProfile {
        self.timer_model
    }

    /// Creates a derived profile value with a named capability status changed.
    /// Test profiles should use a distinct ID when their behavior differs.
    #[must_use]
    pub const fn with_instruction_feature(
        mut self,
        feature: InstructionFeature,
        status: CapabilityStatus,
    ) -> Self {
        self.instruction_features = self.instruction_features.with(feature, status);
        self
    }
}

impl fmt::Display for GuestCpuProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GuestCpuProfile{{id=0x{:016x},architecture={},states={},features=[{}],\
             address-space=[va-bits={},pa-bits={},granule-bytes={}],\
             floating-point=[scalar={},advanced-simd={},fp16={}],\
             cache-maintenance=[user={}],exceptions=[user-mode={}],timer=[virtual-counter={}]}}",
            self.id.get(),
            self.architecture,
            self.allowed_execution_states,
            self.instruction_features,
            self.address_space.virtual_address_bits,
            self.address_space.physical_address_bits,
            self.address_space.translation_granule_bytes,
            self.floating_point.scalar,
            self.floating_point.advanced_simd,
            self.floating_point.fp16,
            self.cache_maintenance.user_cache_maintenance,
            self.exception_model.user_mode_exceptions,
            self.timer_model.virtual_counter,
        )
    }
}

/// Immutable CPU inputs shared by all threads in a guest process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessCpuContext {
    profile: GuestCpuProfile,
    address_space_id: AddressSpaceId,
}

impl ProcessCpuContext {
    #[must_use]
    pub const fn new(profile: GuestCpuProfile, address_space_id: AddressSpaceId) -> Self {
        Self {
            profile,
            address_space_id,
        }
    }

    #[must_use]
    pub const fn profile(self) -> GuestCpuProfile {
        self.profile
    }

    #[must_use]
    pub const fn address_space_id(self) -> AddressSpaceId {
        self.address_space_id
    }

    /// Validates process metadata and freezes the initial state for a new thread.
    pub const fn thread_configuration(
        self,
        initial_execution_state: ExecutionState,
    ) -> Result<ThreadCpuConfiguration, ProcessConfigurationError> {
        if self
            .profile
            .allowed_execution_states
            .contains(initial_execution_state)
        {
            Ok(ThreadCpuConfiguration {
                profile_id: self.profile.id,
                initial_execution_state,
            })
        } else {
            Err(ProcessConfigurationError {
                profile_id: self.profile.id,
                requested_execution_state: initial_execution_state,
            })
        }
    }
}

/// Immutable inputs used to construct architectural state for one guest thread.
///
/// A later A32/T32 interworking transition changes live architectural state,
/// not this construction record and not the selected profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ThreadCpuConfiguration {
    profile_id: CpuProfileId,
    initial_execution_state: ExecutionState,
}

impl ThreadCpuConfiguration {
    #[must_use]
    pub const fn profile_id(self) -> CpuProfileId {
        self.profile_id
    }

    #[must_use]
    pub const fn initial_execution_state(self) -> ExecutionState {
        self.initial_execution_state
    }
}

/// Invalid execution-state selection from process metadata.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessConfigurationError {
    pub profile_id: CpuProfileId,
    pub requested_execution_state: ExecutionState,
}

impl fmt::Display for ProcessConfigurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "process CPU configuration rejected: {} requested-state={}",
            self.profile_id, self.requested_execution_state
        )
    }
}

impl std::error::Error for ProcessConfigurationError {}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    #[test]
    fn profile_identity_has_an_unambiguous_format() {
        assert_eq!(
            CpuProfileId::new(7).to_string(),
            "profile=0x0000000000000007"
        );
    }

    #[test]
    fn profile_is_hashable_by_all_behavioral_fields() {
        let base = GuestCpuProfile::switch_1();
        let changed =
            base.with_instruction_feature(InstructionFeature::Crc32, CapabilityStatus::Enabled);
        let hash = |profile: GuestCpuProfile| {
            let mut hasher = DefaultHasher::new();
            profile.hash(&mut hasher);
            hasher.finish()
        };

        assert_ne!(base, changed);
        assert_ne!(hash(base), hash(changed));
    }

    #[test]
    fn switch_1_process_metadata_can_select_each_supported_frontend() {
        let process = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(9));

        for state in [
            ExecutionState::A64,
            ExecutionState::A32,
            ExecutionState::T32,
        ] {
            let configuration = process.thread_configuration(state).unwrap();
            assert_eq!(configuration.profile_id(), GuestCpuProfile::SWITCH_1_ID);
            assert_eq!(configuration.initial_execution_state(), state);
        }
    }

    #[test]
    fn provisional_switch_2_profile_is_a64_only_and_keeps_unknowns_explicit() {
        let profile = GuestCpuProfile::switch_2_native();
        let process = ProcessCpuContext::new(profile, AddressSpaceId::new(10));

        assert_eq!(profile.architecture(), ArchitectureRevision::Unknown);
        assert!(process.thread_configuration(ExecutionState::A64).is_ok());
        assert!(process.thread_configuration(ExecutionState::A32).is_err());
        assert!(process.thread_configuration(ExecutionState::T32).is_err());
        for feature in InstructionFeature::ALL {
            assert_eq!(
                profile.instruction_feature_status(feature),
                CapabilityStatus::Unknown
            );
        }
    }

    #[test]
    fn same_capability_gate_accepts_or_rejects_using_only_profile_data() {
        let enabled = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::Crc32, CapabilityStatus::Enabled);
        let disabled = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::Crc32, CapabilityStatus::Disabled);

        fn decode_crc32(profile: GuestCpuProfile) -> Result<(), InstructionFeatureRejection> {
            profile.require_instruction_feature(InstructionFeature::Crc32)
        }

        assert!(decode_crc32(enabled).is_ok());
        assert_eq!(
            decode_crc32(disabled).unwrap_err(),
            InstructionFeatureRejection {
                feature: InstructionFeature::Crc32,
                status: CapabilityStatus::Disabled,
            }
        );
    }

    #[test]
    fn stable_profile_output_identifies_every_assumption() {
        assert_eq!(
            GuestCpuProfile::switch_2_native().to_string(),
            "GuestCpuProfile{id=0x0000000000000002,architecture=unknown,states=A64,\
             features=[advanced-simd=unknown,aes=unknown,sha1=unknown,sha256=unknown,\
             crc32=unknown,large-system-extensions=unknown,fp16=unknown,rdm=unknown,\
             dot-product=unknown,sve=unknown,sve2=unknown],\
             address-space=[va-bits=unknown,pa-bits=unknown,granule-bytes=unknown],\
             floating-point=[scalar=unknown,advanced-simd=unknown,fp16=unknown],\
             cache-maintenance=[user=unknown],exceptions=[user-mode=unknown],\
             timer=[virtual-counter=unknown]}"
        );
    }
}
