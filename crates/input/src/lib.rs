//! Thread-owned host input, controller states and profiles.

mod model;
mod motion;
mod profile;
mod sdl;
mod switch_rumble;
mod touch;
mod vibration;
mod worker;

pub use vibration::{VibrationError, VibrationOutput, VibrationSide, VibrationValue};

pub use worker::{InputReader, InputSample, InputWorker, InputWorkerError};

pub use model::{
    Axis, Button, ButtonLabel, ButtonSet, ControllerId, ControllerKind, ControllerState, DPadState,
    FaceButtonLabels, IdentifierError, MotionSensor, MotionState, MotionVector, StickState,
    TriggerState,
};
pub use motion::{MotionEstimate, MotionInput, MotionSettings};
pub use profile::{
    EmulatedButtonState, EmulatedControllerState, GamepadProfile, GamepadProfiles,
    ProfiledControllerState,
};
pub use touch::{
    EmulatedTouchContact, EmulatedTouchScreenState, MAX_TOUCH_CONTACTS, TOUCH_ATTRIBUTE_END,
    TOUCH_ATTRIBUTE_START, TOUCH_SCREEN_HEIGHT, TOUCH_SCREEN_WIDTH, TouchScreenReader,
    TouchScreenWriter, touch_screen_channel,
};

/// Samples the first attached controller without allocating a device list.
trait HostInputBackend {
    type Error: std::error::Error + Send + Sync + 'static;

    fn poll(&mut self) -> Result<Option<ControllerState>, Self::Error>;
}

/// Selects the first attached controller from a host backend.
///
/// Backends keep controllers in attachment order, so the selected controller
/// remains stable until it disconnects.
struct InputManager<B> {
    backend: B,
    profiles: GamepadProfiles,
    selected: Option<SelectedProfile>,
    motion: motion::MotionInput,
    last_sample: Option<std::time::Instant>,
    motion_settings: MotionSettings,
}

struct SelectedProfile {
    id: ControllerId,
    name: std::sync::Arc<str>,
    kind: ControllerKind,
    mapping: Option<(std::sync::Arc<str>, GamepadProfile)>,
}

impl<B> InputManager<B> {
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self::with_profiles(backend, GamepadProfiles::default())
    }

    #[must_use]
    pub fn with_profiles(backend: B, profiles: GamepadProfiles) -> Self {
        Self {
            backend,
            profiles,
            selected: None,
            motion: motion::MotionInput::default(),
            last_sample: None,
            motion_settings: MotionSettings::default(),
        }
    }
}

impl<B: HostInputBackend> InputManager<B> {
    /// Reads the current state of the first attached controller.
    pub fn read_input(&mut self) -> Result<Option<ControllerState>, B::Error> {
        self.backend.poll()
    }

    /// Reads and maps the first attached controller through an exact profile.
    ///
    /// A missing controller or profile match is exposed as a disconnected
    /// emulated controller.
    pub fn read_profiled_input(&mut self) -> Result<Option<ProfiledControllerState>, B::Error> {
        let Some(controller) = self.backend.poll()? else {
            self.selected = None;
            self.last_sample = None;
            self.motion = motion::MotionInput::default();
            return Ok(None);
        };
        let now = std::time::Instant::now();
        if self.selected.as_ref().is_none_or(|selected| {
            selected.id != controller.id
                || selected.kind != controller.kind
                || !std::sync::Arc::ptr_eq(&selected.name, &controller.name)
        }) {
            self.selected = Some(SelectedProfile {
                id: controller.id,
                name: controller.name.clone(),
                kind: controller.kind,
                mapping: self
                    .profiles
                    .matching_profile(&controller.name, controller.kind)
                    .map(|(name, profile)| (name.into(), profile.clone())),
            });
            self.motion = motion::MotionInput::default();
            self.last_sample = None;
        }
        let delta = self
            .last_sample
            .replace(now)
            .map_or(std::time::Duration::ZERO, |previous| {
                now.duration_since(previous)
            });
        let Some((profile_name, profile)) =
            &self.selected.as_ref().expect("selected above").mapping
        else {
            return Ok(None);
        };
        let mut state = profile.map(&controller);
        state.motion = self.motion.update(
            state.gyroscope,
            state.accelerometer,
            delta,
            self.motion_settings.gyroscope_threshold(),
        );
        Ok(Some(ProfiledControllerState {
            controller_id: controller.id,
            device: controller.name,
            profile_name: profile_name.clone(),
            state,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;

    use super::*;

    struct SnapshotBackend {
        snapshots: VecDeque<Option<ControllerState>>,
    }

    impl HostInputBackend for SnapshotBackend {
        type Error = Infallible;

        fn poll(&mut self) -> Result<Option<ControllerState>, Self::Error> {
            Ok(self.snapshots.pop_front().unwrap_or_default())
        }
    }

    fn controller(id: u64) -> ControllerState {
        ControllerState {
            id: ControllerId::new(id),
            name: format!("Controller {id}").into(),
            kind: ControllerKind::Standard,
            buttons: ButtonSet::default(),
            button_labels: FaceButtonLabels::default(),
            dpad: DPadState::default(),
            left_stick: StickState::default(),
            right_stick: StickState::default(),
            triggers: TriggerState::default(),
            motion: MotionState::default(),
        }
    }

    #[test]
    fn read_input_uses_the_first_controller_until_it_disconnects() {
        let mut input = InputManager::new(SnapshotBackend {
            snapshots: VecDeque::from([Some(controller(1)), Some(controller(2)), None]),
        });

        assert_eq!(
            input.read_input().unwrap().unwrap().id,
            ControllerId::new(1)
        );
        assert_eq!(
            input.read_input().unwrap().unwrap().id,
            ControllerId::new(2)
        );
        assert!(input.read_input().unwrap().is_none());
    }
}
