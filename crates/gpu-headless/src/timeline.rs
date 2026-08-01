//! Manually controlled and deterministic host completion.

use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

use nixe_gpu::BackendSubmissionToken;

#[derive(Default)]
pub(crate) struct TimelineState {
    pub(crate) accepted: HashSet<BackendSubmissionToken>,
    pub(crate) completed: HashSet<BackendSubmissionToken>,
    pub(crate) device_loss: Option<Box<str>>,
    pub(crate) torn_down: bool,
}

pub(crate) type SharedTimeline = Arc<Mutex<TimelineState>>;

/// External test control for host completion and deterministic device loss.
#[derive(Clone)]
pub struct HeadlessCompletionController {
    pub(crate) timeline: SharedTimeline,
}

impl HeadlessCompletionController {
    /// Marks one accepted submission complete. Completion is monotonic and
    /// does not publish guest visibility or advance a guest timeline.
    pub fn complete(&self, submission: BackendSubmissionToken) -> Result<(), HeadlessControlError> {
        let mut timeline = self.lock()?;
        if timeline.torn_down {
            return Err(HeadlessControlError::TornDown);
        }
        if let Some(reason) = &timeline.device_loss {
            return Err(HeadlessControlError::DeviceLost(reason.clone()));
        }
        if !timeline.accepted.contains(&submission) {
            return Err(HeadlessControlError::UnknownSubmission(submission));
        }
        timeline.completed.insert(submission);
        Ok(())
    }

    /// Causes the next driver observation to report terminal device loss.
    pub fn lose_device(&self, reason: impl Into<Box<str>>) -> Result<(), HeadlessControlError> {
        let mut timeline = self.lock()?;
        if timeline.torn_down {
            return Err(HeadlessControlError::TornDown);
        }
        if timeline.device_loss.is_none() {
            timeline.device_loss = Some(reason.into());
        }
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, TimelineState>, HeadlessControlError> {
        self.timeline
            .lock()
            .map_err(|_| HeadlessControlError::StatePoisoned)
    }
}

/// Invalid use of the manual headless completion controller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeadlessControlError {
    UnknownSubmission(BackendSubmissionToken),
    DeviceLost(Box<str>),
    TornDown,
    StatePoisoned,
}

impl Display for HeadlessControlError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownSubmission(token) => {
                write!(formatter, "unknown headless submission: {token}")
            }
            Self::DeviceLost(reason) => write!(formatter, "headless device lost: {reason}"),
            Self::TornDown => formatter.write_str("headless backend is torn down"),
            Self::StatePoisoned => formatter.write_str("headless completion state is poisoned"),
        }
    }
}

impl std::error::Error for HeadlessControlError {}
