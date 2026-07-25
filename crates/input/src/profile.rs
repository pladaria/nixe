use std::collections::BTreeMap;

use serde::Deserialize;

use crate::{
    Axis, Button, ControllerId, ControllerKind, ControllerState, MotionSensor, MotionVector,
    StickState,
};

/// Mapping from one host gamepad to player one's emulated Switch controller.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GamepadProfile {
    pub device: String,
    #[serde(rename = "type")]
    pub controller_type: ControllerKind,

    pub a: Option<Button>,
    pub b: Option<Button>,
    pub x: Option<Button>,
    pub y: Option<Button>,
    pub plus: Option<Button>,
    pub minus: Option<Button>,
    pub home: Option<Button>,
    pub capture: Option<Button>,
    pub l: Option<Button>,
    pub r: Option<Button>,
    pub leftstick: Option<Button>,
    pub rightstick: Option<Button>,
    pub dpup: Option<Button>,
    pub dpdown: Option<Button>,
    pub dpleft: Option<Button>,
    pub dpright: Option<Button>,

    pub zl: Option<Axis>,
    pub zr: Option<Axis>,
    pub leftx: Option<Axis>,
    pub lefty: Option<Axis>,
    pub rightx: Option<Axis>,
    pub righty: Option<Axis>,

    pub gyroscope: Option<MotionSensor>,
    pub accelerometer: Option<MotionSensor>,
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
}

/// Identifies the source and profile used to produce an emulated state.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfiledControllerState {
    pub controller_id: ControllerId,
    pub device: String,
    pub profile_name: String,
    pub state: EmulatedControllerState,
}

/// Selects and applies named gamepad profiles to the first attached controller.
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

    #[must_use]
    pub fn map_first_controller(
        &self,
        controllers: &[ControllerState],
    ) -> Option<ProfiledControllerState> {
        let controller = controllers.first()?;
        let (profile_name, profile) = self.matching_profile(&controller.name, controller.kind)?;
        Some(ProfiledControllerState {
            controller_id: controller.id,
            device: controller.name.clone(),
            profile_name: profile_name.to_owned(),
            state: profile.map(controller),
        })
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
            left_stick: StickState {
                x: mapped_stick_axis(controller, self.leftx),
                y: mapped_stick_axis(controller, self.lefty).saturating_neg(),
            },
            right_stick: StickState {
                x: mapped_stick_axis(controller, self.rightx),
                y: mapped_stick_axis(controller, self.righty).saturating_neg(),
            },
            gyroscope: self.gyroscope.and_then(|sensor| motion(controller, sensor)),
            accelerometer: self
                .accelerometer
                .and_then(|sensor| motion(controller, sensor)),
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
    use crate::{ButtonSet, DPadState, FaceButtonLabels, MotionState, TriggerState};

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

    fn controller(name: &str) -> ControllerState {
        let mut buttons = ButtonSet::default();
        buttons.set(Button::East, true);
        buttons.set(Button::Start, true);
        buttons.set(Button::Guide, true);
        ControllerState {
            id: ControllerId::new(7),
            name: name.to_owned(),
            kind: ControllerKind::SwitchPro,
            buttons,
            button_labels: FaceButtonLabels::default(),
            dpad: DPadState::default(),
            left_stick: StickState { x: 123, y: -456 },
            right_stick: StickState { x: -789, y: 321 },
            triggers: TriggerState {
                left: 40_000,
                right: 10_000,
            },
            motion: MotionState {
                gyroscope: Some(MotionVector {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                }),
                accelerometer: Some(MotionVector {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                }),
                ..MotionState::default()
            },
        }
    }

    #[test]
    fn selects_only_the_first_controller_and_requires_an_exact_selector() {
        let profiles = GamepadProfiles::new(BTreeMap::from([("switch-pro".to_owned(), profile())]));
        assert!(
            profiles
                .map_first_controller(&[
                    controller("Another controller"),
                    controller("Nintendo Switch Pro Controller"),
                ])
                .is_none()
        );

        let mapped = profiles
            .map_first_controller(&[controller("Nintendo Switch Pro Controller")])
            .unwrap();
        assert_eq!(mapped.profile_name, "switch-pro");
        assert_eq!(mapped.controller_id, ControllerId::new(7));
    }

    #[test]
    fn maps_buttons_axes_triggers_and_motion() {
        let mapped = profile().map(&controller("Nintendo Switch Pro Controller"));
        assert!(mapped.buttons.a);
        assert!(!mapped.buttons.b);
        assert!(mapped.buttons.plus);
        assert!(mapped.buttons.home);
        assert!(mapped.buttons.zl);
        assert!(!mapped.buttons.zr);
        assert_eq!(mapped.left_stick, StickState { x: 123, y: 456 });
        assert_eq!(mapped.right_stick, StickState { x: -789, y: -321 });
        assert_eq!(
            mapped.gyroscope,
            Some(MotionVector {
                x: 1.0,
                y: 2.0,
                z: 3.0
            })
        );
    }
}
