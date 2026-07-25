//! SDL3 host gamepad backend.

use std::fmt;

use sdl3::{
    GamepadSubsystem, Sdl,
    gamepad::{Axis, Button as SdlButton, ButtonLabel as SdlButtonLabel, Gamepad, GamepadType},
    joystick::JoystickId,
    sensor::SensorType,
};

use crate::{
    Button, ButtonLabel, ButtonSet, ControllerId, ControllerKind, ControllerState, DPadState,
    FaceButtonLabels, HostInputBackend, InputSnapshot, MotionState, MotionVector, StickState,
    TriggerState,
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
            enable_available_sensors(&gamepad);
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
    let (buttons, dpad, left_stick, right_stick, triggers) =
        map_controls(|button| gamepad.button(button), |axis| gamepad.axis(axis));
    ControllerState {
        id: open.controller_id,
        name: open.name.clone(),
        kind: controller_kind(gamepad.r#type()),
        buttons,
        button_labels: face_button_labels(gamepad),
        dpad,
        left_stick,
        right_stick,
        triggers,
        motion: motion_state(gamepad),
    }
}

fn map_controls(
    button: impl Fn(SdlButton) -> bool,
    axis: impl Fn(Axis) -> i16,
) -> (ButtonSet, DPadState, StickState, StickState, TriggerState) {
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
        (SdlButton::DPadUp, Button::DPadUp),
        (SdlButton::DPadDown, Button::DPadDown),
        (SdlButton::DPadLeft, Button::DPadLeft),
        (SdlButton::DPadRight, Button::DPadRight),
        (SdlButton::Misc1, Button::Miscellaneous),
        (SdlButton::Misc2, Button::Miscellaneous2),
        (SdlButton::Misc3, Button::Miscellaneous3),
        (SdlButton::Misc4, Button::Miscellaneous4),
        (SdlButton::Misc5, Button::Miscellaneous5),
        (SdlButton::Misc6, Button::Miscellaneous6),
        (SdlButton::LeftPaddle1, Button::LeftPaddle1),
        (SdlButton::RightPaddle1, Button::RightPaddle1),
        (SdlButton::LeftPaddle2, Button::LeftPaddle2),
        (SdlButton::RightPaddle2, Button::RightPaddle2),
        (SdlButton::Touchpad, Button::Touchpad),
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
        StickState {
            x: axis(Axis::LeftX),
            y: axis(Axis::LeftY),
        },
        StickState {
            x: axis(Axis::RightX),
            y: axis(Axis::RightY),
        },
        TriggerState {
            left: normalize_trigger(axis(Axis::TriggerLeft)),
            right: normalize_trigger(axis(Axis::TriggerRight)),
        },
    )
}

fn face_button_labels(gamepad: &Gamepad) -> FaceButtonLabels {
    FaceButtonLabels {
        south: button_label(gamepad.button_label_for_gamepad_type(SdlButton::South)),
        east: button_label(gamepad.button_label_for_gamepad_type(SdlButton::East)),
        west: button_label(gamepad.button_label_for_gamepad_type(SdlButton::West)),
        north: button_label(gamepad.button_label_for_gamepad_type(SdlButton::North)),
    }
}

fn button_label(value: SdlButtonLabel) -> ButtonLabel {
    match value {
        SdlButtonLabel::Unknown => ButtonLabel::Unknown,
        SdlButtonLabel::A => ButtonLabel::A,
        SdlButtonLabel::B => ButtonLabel::B,
        SdlButtonLabel::X => ButtonLabel::X,
        SdlButtonLabel::Y => ButtonLabel::Y,
        SdlButtonLabel::Cross => ButtonLabel::Cross,
        SdlButtonLabel::Circle => ButtonLabel::Circle,
        SdlButtonLabel::Square => ButtonLabel::Square,
        SdlButtonLabel::Triangle => ButtonLabel::Triangle,
    }
}

const SENSOR_TYPES: [SensorType; 6] = [
    SensorType::Gyroscope,
    SensorType::Accelerometer,
    SensorType::GyroscopeLeft,
    SensorType::GyroscopeRight,
    SensorType::AccelerometerLeft,
    SensorType::AccelerometerRight,
];

fn enable_available_sensors(gamepad: &Gamepad) {
    for sensor_type in SENSOR_TYPES {
        // The gamepad is open and remains alive for this entire query.
        if unsafe { gamepad.has_sensor(sensor_type) } {
            let _ = gamepad.sensor_set_enabled(sensor_type, true);
        }
    }
}

fn motion_state(gamepad: &Gamepad) -> MotionState {
    MotionState {
        gyroscope: read_sensor(gamepad, SensorType::Gyroscope),
        accelerometer: read_sensor(gamepad, SensorType::Accelerometer),
        left_gyroscope: read_sensor(gamepad, SensorType::GyroscopeLeft),
        right_gyroscope: read_sensor(gamepad, SensorType::GyroscopeRight),
        left_accelerometer: read_sensor(gamepad, SensorType::AccelerometerLeft),
        right_accelerometer: read_sensor(gamepad, SensorType::AccelerometerRight),
    }
}

fn read_sensor(gamepad: &Gamepad, sensor_type: SensorType) -> Option<MotionVector> {
    if !gamepad.sensor_enabled(sensor_type) {
        return None;
    }
    let mut data = [0.0; 3];
    gamepad.sensor_get_data(sensor_type, &mut data).ok()?;
    Some(MotionVector {
        x: data[0],
        y: data[1],
        z: data[2],
    })
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
        let (buttons, dpad, left_stick, right_stick, triggers) = map_controls(
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
                Axis::LeftX => -12_345,
                Axis::LeftY => 23_456,
                Axis::RightX => 10,
                Axis::RightY => -20,
                Axis::TriggerLeft => 8_192,
                Axis::TriggerRight => i16::MAX,
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
        assert_eq!(
            left_stick,
            StickState {
                x: -12_345,
                y: 23_456
            }
        );
        assert_eq!(right_stick, StickState { x: 10, y: -20 });
        assert_eq!(triggers.left, 16_384);
        assert_eq!(triggers.right, u16::MAX);
    }

    #[test]
    fn vendored_headless_backend_initializes_and_polls() {
        let mut backend = SdlInputBackend::new().unwrap();
        backend.poll().unwrap();
    }
}
