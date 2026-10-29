//! Kernel run queues, scheduling policy, wait generations and register ownership.

mod clock;
mod identity;
mod lifecycle;
mod scheduler;
mod topology;

pub use clock::{VirtualClock, VirtualClockMode};
pub use identity::{GuestThreadId, ProcessId, VirtualCpuId, WakeGeneration};
pub use lifecycle::{
    LifecycleTransitionError, ProcessLifecycle, ThreadLifecycle, transition_process,
};
pub use scheduler::{
    Completion, Deadline, EventFuture, EventNotifications, EventObject, IdealCore, KernelWait,
    Lease, LeaseGeneration, LockHandoff, LockKey, OrderedWaitKind, ReadableEventObject,
    ScheduledThreadConfig, ScheduledThreadView, SchedulerError, SchedulerState, Selection,
    ThreadPropertyGuard, ThreadPropertyLock, TimerStopped, WaitEffect, WaitOutcome, WaitPhase,
    WaitStart, WakeToken, WritableEventObject, YieldMigration,
};
pub use topology::{
    CoreSet, CoreSetError, MachineSchedulerProfile, MachineSchedulerProfileError, PriorityRange,
    VirtualCpuDescriptor,
};

#[cfg(feature = "performance-counters")]
pub use scheduler::SchedulerStatistics;

#[cfg(feature = "performance-counters")]
pub use scheduler::TimerStatistics;
