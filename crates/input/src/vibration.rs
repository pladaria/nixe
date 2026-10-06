//! Bounded, latest-state actuator output. Device I/O stays on the input thread.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::ControllerId;

/// Two frequency bands for one linear actuator, in hertz and normalized force.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VibrationValue {
    pub low_amplitude: f32,
    pub low_frequency: f32,
    pub high_amplitude: f32,
    pub high_frequency: f32,
}

impl VibrationValue {
    pub fn is_stopped(self) -> bool {
        self.low_amplitude == 0.0 && self.high_amplitude == 0.0
    }

    fn is_valid(self) -> bool {
        [
            self.low_amplitude,
            self.low_frequency,
            self.high_amplitude,
            self.high_frequency,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum VibrationSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct VibrationState {
    pub controller: Option<ControllerId>,
    pub values: [VibrationValue; 2],
}

impl VibrationState {
    pub(crate) fn is_stopped(self) -> bool {
        self.values.into_iter().all(VibrationValue::is_stopped)
    }
}

#[derive(Debug)]
struct Shared {
    state: Mutex<VibrationState>,
    dirty: AtomicBool,
    closed: AtomicBool,
}

#[derive(Debug)]
pub struct VibrationError(&'static str);

impl fmt::Display for VibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for VibrationError {}

/// Routes both actuators to one controller attachment, never to its replacement.
#[derive(Clone, Debug)]
pub struct VibrationOutput(Arc<Shared>);

impl VibrationOutput {
    pub fn select_controller(
        &self,
        controller: Option<ControllerId>,
    ) -> Result<(), VibrationError> {
        self.update(|state| {
            if state.controller == controller {
                return false;
            }
            *state = VibrationState {
                controller,
                ..VibrationState::default()
            };
            true
        })
    }

    pub fn send(&self, side: VibrationSide, value: VibrationValue) -> Result<(), VibrationError> {
        if !value.is_valid() {
            return Err(VibrationError("invalid actuator frequency or amplitude"));
        }
        self.update(|state| {
            let index = match side {
                VibrationSide::Left => 0,
                VibrationSide::Right => 1,
            };
            if state.values[index] == value {
                return false;
            }
            state.values[index] = value;
            true
        })
    }

    pub fn stop(&self) -> Result<(), VibrationError> {
        self.update(|state| {
            if state.is_stopped() {
                return false;
            }
            state.values = [VibrationValue::default(); 2];
            true
        })
    }

    fn update(
        &self,
        update: impl FnOnce(&mut VibrationState) -> bool,
    ) -> Result<(), VibrationError> {
        if self.0.closed.load(Ordering::Acquire) {
            return Err(VibrationError("actuator worker is unavailable"));
        }
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| VibrationError("actuator mailbox poisoned"))?;
        if update(&mut state) {
            self.0.dirty.store(true, Ordering::Release);
        }
        Ok(())
    }
}

pub(crate) struct VibrationReceiver(Arc<Shared>);

impl VibrationReceiver {
    pub(crate) fn take_latest(&self) -> Result<Option<VibrationState>, VibrationError> {
        if !self.0.dirty.load(Ordering::Acquire) {
            return Ok(None);
        }
        let state = self
            .0
            .state
            .lock()
            .map_err(|_| VibrationError("actuator mailbox poisoned"))?;
        self.0.dirty.store(false, Ordering::Release);
        Ok(Some(*state))
    }
}

impl Drop for VibrationReceiver {
    fn drop(&mut self) {
        self.0.closed.store(true, Ordering::Release);
    }
}

pub(crate) fn vibration_channel() -> (VibrationOutput, VibrationReceiver) {
    let shared = Arc::new(Shared {
        state: Mutex::new(VibrationState::default()),
        dirty: AtomicBool::new(false),
        closed: AtomicBool::new(false),
    });
    (VibrationOutput(shared.clone()), VibrationReceiver(shared))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_actuators_stop_and_do_not_follow_a_reconnected_device() {
        let (output, receiver) = vibration_channel();
        let force = VibrationValue {
            low_amplitude: 0.5,
            low_frequency: 160.0,
            high_frequency: 320.0,
            ..Default::default()
        };
        output
            .select_controller(Some(ControllerId::new(1)))
            .unwrap();
        output.send(VibrationSide::Left, force).unwrap();
        output.send(VibrationSide::Right, force).unwrap();
        output
            .send(VibrationSide::Left, VibrationValue::default())
            .unwrap();
        assert_eq!(
            receiver.take_latest().unwrap().unwrap().values,
            [VibrationValue::default(), force]
        );
        assert!(receiver.take_latest().unwrap().is_none());
        output
            .select_controller(Some(ControllerId::new(2)))
            .unwrap();
        let state = receiver.take_latest().unwrap().unwrap();
        assert_eq!(state.controller, Some(ControllerId::new(2)));
        assert!(state.is_stopped());
        output.send(VibrationSide::Left, force).unwrap();
        output.stop().unwrap();
        assert!(receiver.take_latest().unwrap().unwrap().is_stopped());
        output
            .send(
                VibrationSide::Left,
                VibrationValue {
                    low_amplitude: f32::NAN,
                    ..force
                },
            )
            .unwrap_err();
        drop(receiver);
        output.send(VibrationSide::Right, force).unwrap_err();
    }
}
