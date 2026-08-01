//! Switch 1 Maxwell GPU frontend.
//!
//! This crate owns console GPU profile and command semantics. It depends only
//! on host-independent GPU contracts and never on Horizon or a host backend.

mod address_space;
mod capture;
mod channel;
mod engines;
mod gpfifo;
mod profile;
mod pushbuffer;
mod scheduler;

pub use address_space::{
    MAX_MAPPING_DUMP_ENTRIES, MaxwellAddressSpaceError, MaxwellAddressSpaceId,
    MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellGpuAccessError,
    MaxwellGpuAddressSpace, MaxwellGpuMapping, MaxwellMapRequest, MaxwellMappingDiagnostic,
    MaxwellMappingDump, MaxwellMappingId, MaxwellResolvedMapping, MaxwellResolvedRange,
    MaxwellSparseMapping, MaxwellSparseRemapRequest, MaxwellVaRegion, MaxwellVaReservation,
};
pub use capture::{
    MAXWELL_FRONTEND_CAPTURE_WORDS, MaxwellFrontendCapture, MaxwellFrontendCaptureError,
    MaxwellFrontendFailure, MaxwellFrontendReplay, capture_maxwell_frontend_dispatch,
    replay_maxwell_frontend_capture,
};
pub use channel::{
    MaxwellChannelError, MaxwellChannelFrontendState, MaxwellChannelId, MaxwellChannelOwner,
    MaxwellChannelPriority, MaxwellChannelSchedulingPolicy, MaxwellChannelTimeout,
    MaxwellChannelTimeslice, MaxwellGpuChannel, MaxwellMemoryManagerId, MaxwellObjectContext,
    MaxwellZCullBinding, MaxwellZCullMode,
};
pub use engines::{
    MaxwellEngineCapability, MaxwellEngineDispatchError, MaxwellEngineMethodDispatch,
    MaxwellEngineMethodEffect, MaxwellEngineMethodMetadata, MaxwellEnginePacketDispatch,
    commit_maxwell_engine_packet, dispatch_maxwell_engine_packet,
    dispatch_maxwell_engine_pushbuffer, preflight_maxwell_engine_packet,
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
pub use pushbuffer::dispatch::{
    MAXWELL_SET_OBJECT_METHOD, MaxwellMethodDispatch, MaxwellMethodDispatchError,
    MaxwellMethodDispatchKind, MaxwellMethodSource, MaxwellPacketDispatch,
    MaxwellSetObjectTransition, commit_maxwell_packet, dispatch_maxwell_packet,
    dispatch_maxwell_pushbuffer, preflight_maxwell_packet,
};
pub use pushbuffer::packet::{
    MaxwellDecodedMethod, MaxwellDecodedMethodPacket, MaxwellDecodedPacket,
    MaxwellDecodedPushbuffer, MaxwellMethodPacketMode, MaxwellPushbufferControl,
    MaxwellPushbufferDecodeError, MaxwellPushbufferSubchannel, MaxwellPushbufferWord,
    decode_maxwell_pushbuffer, decode_maxwell_submission,
};
pub use scheduler::{
    MaxwellFrontendDispatch, MaxwellFrontendDispatchBoundary, MaxwellScheduleError,
    MaxwellScheduledSubmission, MaxwellScheduler, MaxwellSchedulerSequence,
    MaxwellSubmissionOrderingStage,
};
