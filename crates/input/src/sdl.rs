//! SDL3 host gamepad backend.

use std::fmt;

use sdl3::{
    GamepadSubsystem, Sdl,
    gamepad::{Axis, Button as SdlButton, Gamepad, GamepadType},
    joystick::JoystickId,
};

use crate::{
    Button, ButtonSet, ControllerId, ControllerKind, ControllerState, DPadState, HostInputBackend,
    InputSnapshot, TriggerState,
};

// SDL's position-based button and trigger conventions are defined here:
// https://github.com/libsdl-org/SDL/blob/release-3.4.12/include/SDL3/SDL_gamepad.h
// The safe Rust names used below come from sdl3-rs 0.18.4:
// https://github.com/vhspace/sdl3-rs/tree/v0.18.4

#[derive(Debug)]
pub struct SdlInputError {
    operation: &'static str,
    message: String,
}

impl SdlInputError {
    fn new(operation: &'static str, error: impl fmt::Display) -> Self {
        Self {
            operation,
            message: error.to_string(),
        }
    }
}

impl fmt::Display for SdlInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "SDL input operation {} failed: {}",
            self.operation, self.message
        )
    }
}

impl std::error::Error for SdlInputError {}

struct OpenGamepad {
    joystick_id: JoystickId,
    controller_id: ControllerId,
    name: String,
    gamepad: Gamepad,
}

/// Main-thread SDL3 backend for host gamepads.
///
/// SDL owns its gamepad subsystem on the creating thread, so this backend
/// deliberately does not claim `Send` or `Sync`.
pub struct SdlInputBackend {
    open_gamepads: Vec<OpenGamepad>,
    gamepad_subsystem: GamepadSubsystem,
    next_controller_id: u64,
    _sdl: Sdl,
}

impl SdlInputBackend {
    pub fn new() -> Result<Self, SdlInputError> {
        let sdl = sdl3::init().map_err(|error| SdlInputError::new("initialization", error))?;
        Self::from_sdl(sdl)
    }

    /// Uses a caller-created SDL context so another frontend can initialize
    /// its video and audio subsystems before transferring context ownership.
    pub fn from_sdl(sdl: Sdl) -> Result<Self, SdlInputError> {
        let gamepad_subsystem = sdl
            .gamepad()
            .map_err(|error| SdlInputError::new("gamepad initialization", error))?;
        Ok(Self {
            open_gamepads: Vec::new(),
            gamepad_subsystem,
            next_controller_id: 1,
            _sdl: sdl,
        })
    }

    fn reconcile_gamepads(&mut self) -> Result<(), SdlInputError> {
        self.gamepad_subsystem.update();
        let attached = self
            .gamepad_subsystem
            .gamepads()
            .map_err(|error| SdlInputError::new("controller enumeration", error))?;

        self.open_gamepads
            .retain(|entry| entry.gamepad.connected() && attached.contains(&entry.joystick_id));
        for joystick_id in attached {
            if self
                .open_gamepads
                .iter()
                .any(|entry| entry.joystick_id == joystick_id)
            {
                continue;
            }
            let name = self
                .gamepad_subsystem
                .name_for_id(joystick_id)
                .unwrap_or_else(|_| "Unknown gamepad".to_owned());
            let gamepad = self
                .gamepad_subsystem
                .open(joystick_id)
                .map_err(|error| SdlInputError::new("controller open", error))?;
            let controller_id = ControllerId::new(self.next_controller_id);
            self.next_controller_id = self.next_controller_id.checked_add(1).ok_or_else(|| {
                SdlInputError::new(
                    "controller identity allocation",
                    "identifier space exhausted",
                )
            })?;
            self.open_gamepads.push(OpenGamepad {
                joystick_id,
                controller_id,
                name,
                gamepad,
            });
        }
        Ok(())
    }
}

impl HostInputBackend for SdlInputBackend {
    type Error = SdlInputError;

    fn poll(&mut self) -> Result<InputSnapshot, Self::Error> {
        self.reconcile_gamepads()?;
        let controllers = self.open_gamepads.iter().map(controller_state).collect();
        Ok(InputSnapshot { controllers })
    }
}

fn controller_state(open: &OpenGamepad) -> ControllerState {
    let gamepad = &open.gamepad;
    let (buttons, dpad, triggers) =
        map_controls(|button| gamepad.button(button), |axis| gamepad.axis(axis));
    ControllerState {
        id: open.controller_id,
        name: open.name.clone(),
        kind: controller_kind(gamepad.r#type()),
        buttons,
        dpad,
        triggers,
    }
}

fn map_controls(
    button: impl Fn(SdlButton) -> bool,
    axis: impl Fn(Axis) -> i16,
) -> (ButtonSet, DPadState, TriggerState) {
    let mut buttons = ButtonSet::default();
    for (source, destination) in [
        (SdlButton::South, Button::South),
        (SdlButton::East, Button::East),
        (SdlButton::West, Button::West),
        (SdlButton::North, Button::North),
        (SdlButton::Back, Button::Back),
        (SdlButton::Guide, Button::Guide),
        (SdlButton::Start, Button::Start),
        (SdlButton::LeftStick, Button::LeftStick),
        (SdlButton::RightStick, Button::RightStick),
        (SdlButton::LeftShoulder, Button::LeftShoulder),
        (SdlButton::RightShoulder, Button::RightShoulder),
        (SdlButton::Misc1, Button::Miscellaneous),
    ] {
        buttons.set(destination, button(source));
    }
    (
        buttons,
        DPadState {
            up: button(SdlButton::DPadUp),
            down: button(SdlButton::DPadDown),
            left: button(SdlButton::DPadLeft),
            right: button(SdlButton::DPadRight),
        },
        TriggerState {
            left: normalize_trigger(axis(Axis::TriggerLeft)),
            right: normalize_trigger(axis(Axis::TriggerRight)),
        },
    )
}

fn normalize_trigger(value: i16) -> u16 {
    let value = u32::from(value.max(0) as u16);
    ((value * u32::from(u16::MAX)) / i16::MAX as u32) as u16
}

fn controller_kind(value: GamepadType) -> ControllerKind {
    match value {
        GamepadType::Unknown => ControllerKind::Unknown,
        GamepadType::Standard => ControllerKind::Standard,
        GamepadType::Xbox360 => ControllerKind::Xbox360,
        GamepadType::XboxOne => ControllerKind::XboxOne,
        GamepadType::PS3 => ControllerKind::PlayStation3,
        GamepadType::PS4 => ControllerKind::PlayStation4,
        GamepadType::PS5 => ControllerKind::PlayStation5,
        GamepadType::NintendoSwitchPro => ControllerKind::SwitchPro,
        GamepadType::NintendoSwitchJoyconLeft => ControllerKind::JoyConLeft,
        GamepadType::NintendoSwitchJoyconRight => ControllerKind::JoyConRight,
        GamepadType::NintendoSwitchJoyconPair => ControllerKind::JoyConPair,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_range_is_normalized_and_negative_noise_is_clamped() {
        assert_eq!(normalize_trigger(i16::MIN), 0);
        assert_eq!(normalize_trigger(-1), 0);
        assert_eq!(normalize_trigger(0), 0);
        assert_eq!(normalize_trigger(i16::MAX), u16::MAX);
        assert_eq!(normalize_trigger(16_384), 32_768);
    }

    #[test]
    fn sdl_controls_map_to_the_backend_independent_model() {
        let (buttons, dpad, triggers) = map_controls(
            |button| {
                matches!(
                    button,
                    SdlButton::South
                        | SdlButton::North
                        | SdlButton::LeftShoulder
                        | SdlButton::Start
                        | SdlButton::DPadUp
                        | SdlButton::DPadRight
                )
            },
            |axis| match axis {
                Axis::TriggerLeft => 8_192,
                Axis::TriggerRight => i16::MAX,
                _ => 0,
            },
        );

        assert!(buttons.contains(Button::South));
        assert!(buttons.contains(Button::North));
        assert!(buttons.contains(Button::LeftShoulder));
        assert!(buttons.contains(Button::Start));
        assert!(!buttons.contains(Button::East));
        assert!(!buttons.contains(Button::RightShoulder));
        assert_eq!(
            dpad,
            DPadState {
                up: true,
                down: false,
                left: false,
                right: true,
            }
        );
        assert_eq!(triggers.left, 16_384);
        assert_eq!(triggers.right, u16::MAX);
    }

    #[test]
    fn vendored_headless_backend_initializes_and_polls() {
        let mut backend = SdlInputBackend::new().unwrap();
        backend.poll().unwrap();
    }
}
