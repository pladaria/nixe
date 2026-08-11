//! Console-, runtime-, and engine-independent deterministic scheduling.

mod identity;
mod lifecycle;
mod scheduler;
mod topology;

pub use identity::{GuestThreadId, ProcessId, SchedulerSequence, VirtualCpuId, WakeGeneration};
pub use lifecycle::{
    Continuation, LifecycleTransitionError, ProcessLifecycle, ThreadExitRecord, ThreadLifecycle,
    WaitReason, transition_process, transition_thread,
};
pub use scheduler::{
    Completion, Lease, LeaseGeneration, MigrationEffect, Readiness, ScheduledThreadConfig,
    ScheduledThreadView, SchedulerCommand, SchedulerDecision, SchedulerError, SchedulerState,
    WakeToken,
};
pub use topology::{
    CoreSet, CoreSetError, MachineSchedulerProfile, MachineSchedulerProfileError, PriorityRange,
    VirtualCpuDescriptor,
};
