//! Host-independent GPU contracts and diagnostics.
//!
//! Console frontends and host backends meet at this boundary without sharing
//! Horizon ABI, console packet formats, or concrete host graphics objects.

mod address;
mod completion;
mod diagnostics;
mod submission;
mod synchronization;

pub use address::{GpuVirtualAddress, GpuVirtualAddressError};
pub use completion::{
    BackendCompletionError, BackendCompletionSource, CompletionPropagationError,
    CompletionRegistrationError, CompletionSubmission, PublishedSubmission,
    SubmissionCompletionQueue, SubmissionWrite,
};
pub use diagnostics::{
    CpuVirtualAddress, GpfifoEntryIndex, GpuChannelId, GpuClassId, GpuMethodId,
    GraphicsAllocationId, GraphicsGapKind,
};
pub use nixe_memory::MappingGeneration;
pub use submission::{
    BackendSubmissionToken, FrontendSubmissionId, HostCompletion, VisibilityCompletion,
};
pub use synchronization::{
    GuestSyncpointId, GuestSyncpointValue, GuestTimeline, GuestTimelinePoint, OwnerMismatch,
    ReservedTimelinePoint, SyncpointComparisonError, TimelineAdvanceError, TimelineIncrementError,
    TimelineInstanceId, TimelineOwnerId, TimelinePointComparisonError, TimelineReservationError,
};
