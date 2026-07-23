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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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

/// Position-based gamepad buttons.
///
/// Face buttons describe their physical position rather than their printed
/// label. For example, `South` is Xbox A, Nintendo B, and PlayStation Cross.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    Miscellaneous,
}

/// Compact set of pressed position-based buttons.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ButtonSet(u32);

impl ButtonSet {
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & ((1 << 12) - 1))
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControllerState {
    pub id: ControllerId,
    pub name: String,
    pub kind: ControllerKind,
    pub buttons: ButtonSet,
    pub dpad: DPadState,
    pub triggers: TriggerState,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
        assert_eq!(ButtonSet::from_bits(u32::MAX).bits(), (1 << 12) - 1);
    }
}
