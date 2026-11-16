use std::collections::BTreeMap;

use serde::Deserialize;

use crate::{
    Axis, Button, ControllerId, ControllerKind, ControllerState, MotionSensor, MotionVector,
    StickState,
};

/// Mapping from one host gamepad to its assigned emulated controller.
/// Missing bindings and empty strings disable their button, axis or sensor.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GamepadProfile {
    pub device: String,
    #[serde(rename = "type")]
    pub controller_type: ControllerKind,

    #[serde(default, deserialize_with = "optional_mapping")]
    pub a: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub b: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub x: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub y: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub plus: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub minus: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub home: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub capture: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub l: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub r: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub leftstick: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub rightstick: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub dpup: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub dpdown: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub dpleft: Option<Button>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub dpright: Option<Button>,

    #[serde(default, deserialize_with = "optional_mapping")]
    pub zl: Option<Axis>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub zr: Option<Axis>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub leftx: Option<Axis>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub lefty: Option<Axis>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub rightx: Option<Axis>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub righty: Option<Axis>,

    #[serde(default, deserialize_with = "optional_mapping")]
    pub gyroscope: Option<MotionSensor>,
    #[serde(default, deserialize_with = "optional_mapping")]
    pub accelerometer: Option<MotionSensor>,
}

// Optional bindings are resolved once while loading configuration. Empty
// strings and omitted entries both disable a control; other invalid identifiers
// remain configuration errors rather than silently changing the mapping.
fn optional_mapping<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: TryFrom<String>,
    T::Error: std::fmt::Display,
{
    Option::<String>::deserialize(deserializer)?
        .filter(|value| !value.is_empty())
        .map(T::try_from)
        .transpose()
        .map_err(serde::de::Error::custom)
}

/// Switch buttons exposed through the emulated NPad state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EmulatedButtonState {
    pub a: bool,
    pub b: bool,
    pub x: bool,
    pub y: bool,
    pub plus: bool,
    pub minus: bool,
    pub home: bool,
    pub capture: bool,
    pub l: bool,
    pub r: bool,
    pub zl: bool,
    pub zr: bool,
    pub left_stick: bool,
    pub right_stick: bool,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
}

/// Host state translated through a configured profile.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmulatedControllerState {
    pub buttons: EmulatedButtonState,
    pub left_stick: StickState,
    pub right_stick: StickState,
    pub gyroscope: Option<MotionVector>,
    pub accelerometer: Option<MotionVector>,
    pub motion: Option<crate::MotionEstimate>,
}

/// Identifies the source and profile used to produce an emulated state.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfiledControllerState {
    pub controller_id: ControllerId,
    pub device: std::sync::Arc<str>,
    pub profile_name: std::sync::Arc<str>,
    pub state: EmulatedControllerState,
}

/// Named gamepad profiles shared by the independent player slots.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GamepadProfiles {
    profiles: BTreeMap<String, GamepadProfile>,
}

impl GamepadProfiles {
    #[must_use]
    pub const fn new(profiles: BTreeMap<String, GamepadProfile>) -> Self {
        Self { profiles }
    }

    #[must_use]
    pub const fn profiles(&self) -> &BTreeMap<String, GamepadProfile> {
        &self.profiles
    }

    #[must_use]
    pub fn matching_profile(
        &self,
        device: &str,
        controller_type: ControllerKind,
    ) -> Option<(&str, &GamepadProfile)> {
        self.profiles
            .iter()
            .find(|(_, profile)| {
                profile.device == device && profile.controller_type == controller_type
            })
            .map(|(name, profile)| (name.as_str(), profile))
    }
}

impl GamepadProfile {
    #[must_use]
    pub fn map(&self, controller: &ControllerState) -> EmulatedControllerState {
        let button = |source: Option<Button>| {
            source.is_some_and(|source| controller.buttons.contains(source))
        };
        EmulatedControllerState {
            buttons: EmulatedButtonState {
                a: button(self.a),
                b: button(self.b),
                x: button(self.x),
                y: button(self.y),
                plus: button(self.plus),
                minus: button(self.minus),
                home: button(self.home),
                capture: button(self.capture),
                l: button(self.l),
                r: button(self.r),
                zl: axis_pressed(controller, self.zl),
                zr: axis_pressed(controller, self.zr),
                left_stick: button(self.leftstick),
                right_stick: button(self.rightstick),
                dpad_up: button(self.dpup),
                dpad_down: button(self.dpdown),
                dpad_left: button(self.dpleft),
                dpad_right: button(self.dpright),
            },
            left_stick: mapped_stick(controller, self.leftx, self.lefty),
            right_stick: mapped_stick(controller, self.rightx, self.righty),
            gyroscope: self.gyroscope.and_then(|sensor| motion(controller, sensor)),
            accelerometer: self
                .accelerometer
                .and_then(|sensor| motion(controller, sensor)),
            motion: None,
        }
    }
}

const TRIGGER_PRESS_THRESHOLD: u16 = u16::MAX / 2;

fn axis_pressed(controller: &ControllerState, source: Option<Axis>) -> bool {
    source.is_some_and(|source| match source {
        Axis::LeftTrigger => controller.triggers.left >= TRIGGER_PRESS_THRESHOLD,
        Axis::RightTrigger => controller.triggers.right >= TRIGGER_PRESS_THRESHOLD,
        axis => {
            i32::from(stick_axis(controller, axis)).unsigned_abs()
                >= u32::from(TRIGGER_PRESS_THRESHOLD)
        }
    })
}

// Apply the reference input converter's radial deadzone and outer range after
// profile axis selection. SDL inspection retains the original device samples.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/frontend/input_converter.cpp#L366-L436
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/input_common/input_poller.cpp#L926-L955
fn mapped_stick(controller: &ControllerState, x: Option<Axis>, y: Option<Axis>) -> StickState {
    const DEADZONE: f32 = 0.15;
    const RANGE: f32 = 0.95;
    let x = f32::from(mapped_stick_axis(controller, x)) / f32::from(i16::MAX);
    let y = -f32::from(mapped_stick_axis(controller, y)) / f32::from(i16::MAX);
    let radius = (x * x + y * y).sqrt();
    if radius <= DEADZONE {
        return StickState::default();
    }
    let factor = (radius - DEADZONE) / ((1.0 - DEADZONE) * RANGE * radius);
    let factor = factor / (radius * factor).max(1.0);
    StickState {
        x: (x * factor * f32::from(i16::MAX)) as i16,
        y: (y * factor * f32::from(i16::MAX)) as i16,
    }
}

fn mapped_stick_axis(controller: &ControllerState, source: Option<Axis>) -> i16 {
    source.map_or(0, |source| match source {
        Axis::LeftTrigger => (controller.triggers.left / 2) as i16,
        Axis::RightTrigger => (controller.triggers.right / 2) as i16,
        axis => stick_axis(controller, axis),
    })
}

fn stick_axis(controller: &ControllerState, source: Axis) -> i16 {
    match source {
        Axis::LeftX => controller.left_stick.x,
        Axis::LeftY => controller.left_stick.y,
        Axis::RightX => controller.right_stick.x,
        Axis::RightY => controller.right_stick.y,
        Axis::LeftTrigger | Axis::RightTrigger => 0,
    }
}

fn motion(controller: &ControllerState, source: MotionSensor) -> Option<MotionVector> {
    match source {
        MotionSensor::Gyroscope => controller.motion.gyroscope,
        MotionSensor::Accelerometer => controller.motion.accelerometer,
        MotionSensor::LeftGyroscope => controller.motion.left_gyroscope,
        MotionSensor::RightGyroscope => controller.motion.right_gyroscope,
        MotionSensor::LeftAccelerometer => controller.motion.left_accelerometer,
        MotionSensor::RightAccelerometer => controller.motion.right_accelerometer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> GamepadProfile {
        GamepadProfile {
            device: "Nintendo Switch Pro Controller".to_owned(),
            controller_type: ControllerKind::SwitchPro,
            a: Some(Button::East),
            b: Some(Button::South),
            x: None,
            y: None,
            plus: Some(Button::Start),
            minus: None,
            home: Some(Button::Guide),
            capture: Some(Button::Miscellaneous),
            l: None,
            r: None,
            leftstick: None,
            rightstick: None,
            dpup: None,
            dpdown: None,
            dpleft: None,
            dpright: None,
            zl: Some(Axis::LeftTrigger),
            zr: Some(Axis::RightTrigger),
            leftx: Some(Axis::LeftX),
            lefty: Some(Axis::LeftY),
            rightx: Some(Axis::RightX),
            righty: Some(Axis::RightY),
            gyroscope: Some(MotionSensor::Gyroscope),
            accelerometer: Some(MotionSensor::Accelerometer),
        }
    }

    #[test]
    fn requires_an_exact_selector() {
        let profiles = GamepadProfiles::new(BTreeMap::from([("switch-pro".to_owned(), profile())]));
        assert!(
            profiles
                .matching_profile("Another controller", ControllerKind::SwitchPro)
                .is_none()
        );
        assert!(
            profiles
                .matching_profile("Nintendo Switch Pro Controller", ControllerKind::Standard)
                .is_none()
        );
        assert_eq!(
            profiles
                .matching_profile("Nintendo Switch Pro Controller", ControllerKind::SwitchPro)
                .unwrap()
                .0,
            "switch-pro"
        );
    }
}
