use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::Deserialize;

/// Stable Nixe identity for one controller attachment.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ControllerId(u64);

impl ControllerId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Host controller family, kept independent from any backend's type system.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Deserialize)]
#[serde(try_from = "String")]
pub enum ControllerKind {
    #[default]
    Unknown,
    Standard,
    Xbox360,
    XboxOne,
    PlayStation3,
    PlayStation4,
    PlayStation5,
    SwitchPro,
    JoyConLeft,
    JoyConRight,
    JoyConPair,
}

impl ControllerKind {
    #[must_use]
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Standard => "standard",
            Self::Xbox360 => "xbox-360",
            Self::XboxOne => "xbox-one",
            Self::PlayStation3 => "playstation-3",
            Self::PlayStation4 => "playstation-4",
            Self::PlayStation5 => "playstation-5",
            Self::SwitchPro => "switch-pro",
            Self::JoyConLeft => "joycon-left",
            Self::JoyConRight => "joycon-right",
            Self::JoyConPair => "joycon-pair",
        }
    }
}

/// Position-based gamepad buttons.
///
/// Face buttons describe their physical position rather than their printed
/// label. For example, `South` is Xbox A, Nintendo B, and PlayStation Cross.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(try_from = "String")]
#[repr(u8)]
pub enum Button {
    South,
    East,
    West,
    North,
    Back,
    Guide,
    Start,
    LeftStick,
    RightStick,
    LeftShoulder,
    RightShoulder,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
    Miscellaneous,
    Miscellaneous2,
    Miscellaneous3,
    Miscellaneous4,
    Miscellaneous5,
    Miscellaneous6,
    LeftPaddle1,
    RightPaddle1,
    LeftPaddle2,
    RightPaddle2,
    Touchpad,
}

impl Button {
    pub const ALL: [Self; 26] = [
        Self::South,
        Self::East,
        Self::West,
        Self::North,
        Self::Back,
        Self::Guide,
        Self::Start,
        Self::LeftStick,
        Self::RightStick,
        Self::LeftShoulder,
        Self::RightShoulder,
        Self::DPadUp,
        Self::DPadDown,
        Self::DPadLeft,
        Self::DPadRight,
        Self::Miscellaneous,
        Self::Miscellaneous2,
        Self::Miscellaneous3,
        Self::Miscellaneous4,
        Self::Miscellaneous5,
        Self::Miscellaneous6,
        Self::LeftPaddle1,
        Self::RightPaddle1,
        Self::LeftPaddle2,
        Self::RightPaddle2,
        Self::Touchpad,
    ];

    #[must_use]
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::South => "south",
            Self::East => "east",
            Self::West => "west",
            Self::North => "north",
            Self::Back => "back",
            Self::Guide => "guide",
            Self::Start => "start",
            Self::LeftStick => "leftstick",
            Self::RightStick => "rightstick",
            Self::LeftShoulder => "leftshoulder",
            Self::RightShoulder => "rightshoulder",
            Self::DPadUp => "dpup",
            Self::DPadDown => "dpdown",
            Self::DPadLeft => "dpleft",
            Self::DPadRight => "dpright",
            Self::Miscellaneous => "misc1",
            Self::Miscellaneous2 => "misc2",
            Self::Miscellaneous3 => "misc3",
            Self::Miscellaneous4 => "misc4",
            Self::Miscellaneous5 => "misc5",
            Self::Miscellaneous6 => "misc6",
            Self::RightPaddle1 => "paddle1",
            Self::LeftPaddle1 => "paddle2",
            Self::RightPaddle2 => "paddle3",
            Self::LeftPaddle2 => "paddle4",
            Self::Touchpad => "touchpad",
        }
    }
}

/// SDL gamepad axis identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(try_from = "String")]
pub enum Axis {
    LeftX,
    LeftY,
    RightX,
    RightY,
    LeftTrigger,
    RightTrigger,
}

impl Axis {
    pub const ALL: [Self; 6] = [
        Self::LeftX,
        Self::LeftY,
        Self::RightX,
        Self::RightY,
        Self::LeftTrigger,
        Self::RightTrigger,
    ];

    #[must_use]
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::LeftX => "leftx",
            Self::LeftY => "lefty",
            Self::RightX => "rightx",
            Self::RightY => "righty",
            Self::LeftTrigger => "lefttrigger",
            Self::RightTrigger => "righttrigger",
        }
    }
}

/// SDL gamepad motion-sensor identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize)]
#[serde(try_from = "String")]
pub enum MotionSensor {
    Gyroscope,
    Accelerometer,
    LeftGyroscope,
    RightGyroscope,
    LeftAccelerometer,
    RightAccelerometer,
}

impl MotionSensor {
    pub const ALL: [Self; 6] = [
        Self::Gyroscope,
        Self::Accelerometer,
        Self::LeftGyroscope,
        Self::RightGyroscope,
        Self::LeftAccelerometer,
        Self::RightAccelerometer,
    ];

    #[must_use]
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Gyroscope => "gyroscope",
            Self::Accelerometer => "accelerometer",
            Self::LeftGyroscope => "leftgyroscope",
            Self::RightGyroscope => "rightgyroscope",
            Self::LeftAccelerometer => "leftaccelerometer",
            Self::RightAccelerometer => "rightaccelerometer",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentifierError {
    category: &'static str,
    value: String,
}

impl Display for IdentifierError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unknown {} identifier `{}`",
            self.category, self.value
        )
    }
}

impl std::error::Error for IdentifierError {}

macro_rules! identifier_traits {
    ($type:ty, $category:literal, $all:expr) => {
        impl FromStr for $type {
            type Err = IdentifierError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                $all.into_iter()
                    .find(|candidate| candidate.identifier() == value)
                    .ok_or_else(|| IdentifierError {
                        category: $category,
                        value: value.to_owned(),
                    })
            }
        }

        impl TryFrom<String> for $type {
            type Error = IdentifierError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                value.parse()
            }
        }

        impl Display for $type {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.identifier())
            }
        }
    };
}

identifier_traits!(Button, "button", Button::ALL);
identifier_traits!(Axis, "axis", Axis::ALL);
identifier_traits!(MotionSensor, "motion sensor", MotionSensor::ALL);

impl FromStr for ControllerKind {
    type Err = IdentifierError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        [
            Self::Unknown,
            Self::Standard,
            Self::Xbox360,
            Self::XboxOne,
            Self::PlayStation3,
            Self::PlayStation4,
            Self::PlayStation5,
            Self::SwitchPro,
            Self::JoyConLeft,
            Self::JoyConRight,
            Self::JoyConPair,
        ]
        .into_iter()
        .find(|candidate| candidate.identifier() == value)
        .ok_or_else(|| IdentifierError {
            category: "controller type",
            value: value.to_owned(),
        })
    }
}

impl TryFrom<String> for ControllerKind {
    type Error = IdentifierError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl Display for ControllerKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.identifier())
    }
}

/// Compact set of pressed position-based buttons.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ButtonSet(u32);

impl ButtonSet {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & ((1 << Button::ALL.len()) - 1))
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, button: Button) -> bool {
        self.0 & (1 << button as u8) != 0
    }

    pub fn set(&mut self, button: Button, pressed: bool) {
        let mask = 1 << button as u8;
        if pressed {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }
}

/// Label printed on a position-based face button, when SDL knows it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ButtonLabel {
    #[default]
    Unknown,
    A,
    B,
    X,
    Y,
    Cross,
    Circle,
    Square,
    Triangle,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FaceButtonLabels {
    pub south: ButtonLabel,
    pub east: ButtonLabel,
    pub west: ButtonLabel,
    pub north: ButtonLabel,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DPadState {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
}

/// Analog trigger positions over the backend-independent `0..=65535` range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TriggerState {
    pub left: u16,
    pub right: u16,
}

/// Raw signed stick coordinates over SDL's `-32768..=32767` range.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StickState {
    pub x: i16,
    pub y: i16,
}

/// Three-axis motion vector in the units reported by SDL.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionVector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Optional controller motion sensors.
///
/// Gyroscopes use radians per second and accelerometers use metres per
/// second squared.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionState {
    pub gyroscope: Option<MotionVector>,
    pub accelerometer: Option<MotionVector>,
    pub left_gyroscope: Option<MotionVector>,
    pub right_gyroscope: Option<MotionVector>,
    pub left_accelerometer: Option<MotionVector>,
    pub right_accelerometer: Option<MotionVector>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControllerState {
    pub id: ControllerId,
    pub name: String,
    pub kind: ControllerKind,
    pub buttons: ButtonSet,
    pub button_labels: FaceButtonLabels,
    pub dpad: DPadState,
    pub left_stick: StickState,
    pub right_stick: StickState,
    pub triggers: TriggerState,
    pub motion: MotionState,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputSnapshot {
    pub controllers: Vec<ControllerState>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_set_tracks_only_declared_buttons() {
        let mut buttons = ButtonSet::default();
        buttons.set(Button::South, true);
        buttons.set(Button::RightShoulder, true);
        assert!(buttons.contains(Button::South));
        assert!(buttons.contains(Button::RightShoulder));
        assert!(!buttons.contains(Button::North));

        buttons.set(Button::South, false);
        assert!(!buttons.contains(Button::South));
        assert_eq!(
            ButtonSet::from_bits(u32::MAX).bits(),
            (1 << Button::ALL.len()) - 1
        );
    }

    #[test]
    fn canonical_identifiers_round_trip() {
        for button in Button::ALL {
            assert_eq!(button.identifier().parse::<Button>(), Ok(button));
        }
        for axis in Axis::ALL {
            assert_eq!(axis.identifier().parse::<Axis>(), Ok(axis));
        }
        for sensor in MotionSensor::ALL {
            assert_eq!(sensor.identifier().parse::<MotionSensor>(), Ok(sensor));
        }
        assert_eq!(
            ControllerKind::SwitchPro
                .identifier()
                .parse::<ControllerKind>(),
            Ok(ControllerKind::SwitchPro)
        );
    }
}
