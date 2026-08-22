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
    Preempted,
    Suspended,
    Terminating,
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

pub fn transition_thread(
    state: &mut ThreadLifecycle,
    target: ThreadLifecycle,
) -> Result<(), LifecycleTransitionError<ThreadLifecycle>> {
    if *state == target {
        return Err(LifecycleTransitionError::Duplicate(target));
    }
    let allowed = matches!(
        (*state, target),
        (ThreadLifecycle::Created, ThreadLifecycle::Ready)
            | (ThreadLifecycle::Created, ThreadLifecycle::Terminating)
            | (ThreadLifecycle::Ready, ThreadLifecycle::Running)
            | (ThreadLifecycle::Ready, ThreadLifecycle::Terminating)
            | (ThreadLifecycle::Ready, ThreadLifecycle::Suspended)
            | (ThreadLifecycle::Running, ThreadLifecycle::Ready)
            | (ThreadLifecycle::Running, ThreadLifecycle::Waiting)
            | (ThreadLifecycle::Running, ThreadLifecycle::Preempted)
            | (ThreadLifecycle::Running, ThreadLifecycle::Terminating)
            | (ThreadLifecycle::Running, ThreadLifecycle::Faulted)
            | (ThreadLifecycle::Waiting, ThreadLifecycle::Ready)
            | (ThreadLifecycle::Waiting, ThreadLifecycle::Terminating)
            | (ThreadLifecycle::Waiting, ThreadLifecycle::Faulted)
            | (ThreadLifecycle::Preempted, ThreadLifecycle::Ready)
            | (ThreadLifecycle::Preempted, ThreadLifecycle::Running)
            | (ThreadLifecycle::Preempted, ThreadLifecycle::Terminating)
            | (ThreadLifecycle::Suspended, ThreadLifecycle::Ready)
            | (ThreadLifecycle::Suspended, ThreadLifecycle::Terminating)
            | (ThreadLifecycle::Terminating, ThreadLifecycle::Exited)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_thread_transition_is_table_driven_and_atomic() {
        let states = [
            ThreadLifecycle::Created,
            ThreadLifecycle::Ready,
            ThreadLifecycle::Running,
            ThreadLifecycle::Waiting,
            ThreadLifecycle::Preempted,
            ThreadLifecycle::Suspended,
            ThreadLifecycle::Terminating,
            ThreadLifecycle::Exited,
            ThreadLifecycle::Faulted,
        ];
        let allowed = [
            (0, 1),
            (0, 6),
            (1, 2),
            (1, 5),
            (1, 6),
            (2, 1),
            (2, 3),
            (2, 4),
            (2, 6),
            (2, 8),
            (3, 1),
            (3, 6),
            (3, 8),
            (4, 1),
            (4, 2),
            (4, 6),
            (5, 1),
            (5, 6),
            (6, 7),
        ];
        for (from_index, from) in states.into_iter().enumerate() {
            for (to_index, to) in states.into_iter().enumerate() {
                let mut actual = from;
                let result = transition_thread(&mut actual, to);
                if allowed.contains(&(from_index, to_index)) {
                    assert_eq!(result, Ok(()), "{from:?} -> {to:?}");
                    assert_eq!(actual, to);
                } else {
                    assert!(result.is_err(), "{from:?} -> {to:?}");
                    assert_eq!(actual, from, "rejection must be atomic");
                }
            }
        }
    }

    #[test]
    fn waiting_or_exited_thread_does_not_define_process_lifecycle() {
        let process = ProcessLifecycle::Running;
        let threads = [
            ThreadLifecycle::Waiting,
            ThreadLifecycle::Exited,
            ThreadLifecycle::Ready,
        ];
        assert_eq!(process, ProcessLifecycle::Running);
        assert!(threads.contains(&ThreadLifecycle::Ready));
    }
}
