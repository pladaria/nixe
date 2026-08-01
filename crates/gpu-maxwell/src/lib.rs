//! Switch 1 Maxwell GPU frontend.
//!
//! This crate owns console GPU profile and command semantics. It depends only
//! on host-independent GPU contracts and never on Horizon or a host backend.

mod address_space;
mod channel;
mod gpfifo;
mod profile;
mod scheduler;

pub use address_space::{
    MAX_MAPPING_DUMP_ENTRIES, MaxwellAddressSpaceError, MaxwellAddressSpaceId,
    MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellGpuAccessError,
    MaxwellGpuAddressSpace, MaxwellGpuMapping, MaxwellMapRequest, MaxwellMappingDiagnostic,
    MaxwellMappingDump, MaxwellMappingId, MaxwellResolvedMapping, MaxwellResolvedRange,
    MaxwellSparseMapping, MaxwellSparseRemapRequest, MaxwellVaRegion, MaxwellVaReservation,
};
pub use channel::{
    MaxwellChannelError, MaxwellChannelFrontendState, MaxwellChannelId, MaxwellChannelOwner,
    MaxwellChannelPriority, MaxwellChannelSchedulingPolicy, MaxwellChannelTimeout,
    MaxwellChannelTimeslice, MaxwellGpuChannel, MaxwellMemoryManagerId, MaxwellObjectContext,
    MaxwellZCullBinding, MaxwellZCullMode,
};
pub use gpfifo::{
    MAXWELL_GPFIFO_CAPTURE_SOURCES, MAXWELL_GPFIFO_ENTRY_SIZE, MaxwellDecodedGpfifoSubmission,
    MaxwellGpfifoCapture, MaxwellGpfifoCaptureSource, MaxwellGpfifoDecodeError, MaxwellGpfifoEntry,
    MaxwellGpfifoFetchMode, MaxwellGpfifoLevel, MaxwellGpfifoSourceError,
    MaxwellGpfifoSourceLocation, MaxwellGpfifoSubmissionMode, MaxwellGpfifoSubmitRequest,
    MaxwellGpfifoSyncMode, MaxwellInvalidGpfifoSubmission, MaxwellRetainedPushbuffer,
    MaxwellUnsupportedGpfifoSubmission, MaxwellValidatedGpfifoSubmission, decode_gpfifo_submission,
    resolve_gpfifo_submission,
};
pub use profile::{
    AddressBitCount, ChipName, GpuArchitecture, GpuBusType, GpuFeatureFlags, GpuImplementation,
    GpuPageSize, GpuPageSizeMask, GpuProfileId, GpuRevision, MaxwellCacheCapabilities,
    MaxwellChipsetIdentity, MaxwellClassCapabilities, MaxwellGpuProfile,
    MaxwellInterconnectCapabilities, MaxwellMemoryCapabilities, MaxwellProfileValidationError,
    MaxwellShaderCapabilities, MaxwellTopology, MaxwellVirtualAddressCapabilities,
    MaxwellZCullCapabilities, SWITCH_1_GM20B_PROFILE, ShaderVersion,
};
pub use scheduler::{
    MaxwellFrontendDispatch, MaxwellFrontendDispatchBoundary, MaxwellScheduleError,
    MaxwellScheduledSubmission, MaxwellScheduler, MaxwellSchedulerSequence,
    MaxwellSubmissionOrderingStage,
};
