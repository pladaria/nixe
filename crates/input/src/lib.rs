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

/// Maximum number of independently routed host controller slots.
pub const MAX_PLAYERS: usize = 8;
pub type RawControllerStates = [Option<ControllerState>; MAX_PLAYERS];
pub type ProfiledControllerStates = [Option<ProfiledControllerState>; MAX_PLAYERS];
pub type EmulatedControllerStates = [Option<EmulatedControllerState>; MAX_PLAYERS];

/// A complete poll preserves empty slots when another device disconnects.
trait HostInputBackend {
    type Error: std::error::Error + Send + Sync + 'static;
    fn poll(&mut self) -> Result<RawControllerStates, Self::Error>;
}

struct InputManager<B> {
    backend: B,
    profiles: GamepadProfiles,
    players: [PlayerInput; MAX_PLAYERS],
}

#[derive(Default)]
struct PlayerInput {
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

impl<B: HostInputBackend> InputManager<B> {
    fn with_profiles(
        backend: B,
        profiles: GamepadProfiles,
        settings: [MotionSettings; MAX_PLAYERS],
    ) -> Self {
        Self {
            backend,
            profiles,
            players: settings.map(|motion_settings| PlayerInput {
                motion_settings,
                ..PlayerInput::default()
            }),
        }
    }

    fn read_profiled_input(&mut self) -> Result<ProfiledControllerStates, B::Error> {
        let mut states = self.backend.poll()?;
        let now = std::time::Instant::now();
        Ok(std::array::from_fn(|slot| {
            self.players[slot].map(states[slot].take(), &self.profiles, now)
        }))
    }
}

impl PlayerInput {
    fn map(
        &mut self,
        controller: Option<ControllerState>,
        profiles: &GamepadProfiles,
        now: std::time::Instant,
    ) -> Option<ProfiledControllerState> {
        let Some(controller) = controller else {
            self.selected = None;
            self.last_sample = None;
            self.motion = motion::MotionInput::default();
            return None;
        };
        if self.selected.as_ref().is_none_or(|selected| {
            selected.id != controller.id
                || selected.kind != controller.kind
                || !std::sync::Arc::ptr_eq(&selected.name, &controller.name)
        }) {
            self.selected = Some(SelectedProfile {
                id: controller.id,
                name: controller.name.clone(),
                kind: controller.kind,
                mapping: profiles
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
            return None;
        };
        let mut state = profile.map(&controller);
        state.motion = self.motion.update(
            state.gyroscope,
            state.accelerometer,
            delta,
            self.motion_settings.gyroscope_threshold(),
            self.motion_settings.rest_sensitivity(),
        );
        Some(ProfiledControllerState {
            controller_id: controller.id,
            device: controller.name,
            profile_name: profile_name.clone(),
            state,
        })
    }
}
