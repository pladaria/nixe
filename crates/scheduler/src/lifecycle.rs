use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProcessLifecycle {
    Created,
    Running,
    Terminating,
    Exited,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ThreadLifecycle {
    Created,
    Ready,
    Running,
    Waiting,
    Suspended,
    Exited,
    Faulted,
}

pub fn transition_process(
    state: &mut ProcessLifecycle,
    target: ProcessLifecycle,
) -> Result<(), LifecycleTransitionError<ProcessLifecycle>> {
    if *state == target {
        return Err(LifecycleTransitionError::Duplicate(target));
    }
    let allowed = matches!(
        (*state, target),
        (ProcessLifecycle::Created, ProcessLifecycle::Running)
            | (ProcessLifecycle::Created, ProcessLifecycle::Terminating)
            | (ProcessLifecycle::Created, ProcessLifecycle::Faulted)
            | (ProcessLifecycle::Running, ProcessLifecycle::Terminating)
            | (ProcessLifecycle::Running, ProcessLifecycle::Faulted)
            | (ProcessLifecycle::Terminating, ProcessLifecycle::Exited)
    );
    if !allowed {
        return Err(LifecycleTransitionError::Illegal {
            current: *state,
            target,
        });
    }
    *state = target;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleTransitionError<S> {
    Duplicate(S),
    Illegal { current: S, target: S },
}

impl<S: std::fmt::Debug> Display for LifecycleTransitionError<S> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate(state) => write!(formatter, "lifecycle is already {state:?}"),
            Self::Illegal { current, target } => {
                write!(
                    formatter,
                    "illegal lifecycle transition {current:?} -> {target:?}"
                )
            }
        }
    }
}

impl<S: std::fmt::Debug> Error for LifecycleTransitionError<S> {}
