//! SDL3 host gamepad backend.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use sdl3::{
    EventSubsystem, GamepadSubsystem, Sdl,
    event::{Event, EventType, EventWatch, EventWatchCallback},
    gamepad::{Axis, Button as SdlButton, ButtonLabel as SdlButtonLabel, Gamepad, GamepadType},
    joystick::JoystickId,
    sensor::SensorType,
};

use crate::{
    Button, ButtonLabel, ButtonSet, ControllerId, ControllerKind, ControllerState, DPadState,
    FaceButtonLabels, HostInputBackend, MotionState, MotionVector, StickState, TriggerState,
};

// SDL's position-based button and trigger conventions are defined here:
// https://github.com/libsdl-org/SDL/blob/release-3.4.12/include/SDL3/SDL_gamepad.h
// The safe Rust names used below come from sdl3-rs 0.18.4:
// https://github.com/vhspace/sdl3-rs/tree/v0.18.4

#[derive(Debug)]
pub(crate) struct SdlInputError {
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
    name: Arc<str>,
    kind: ControllerKind,
    labels: FaceButtonLabels,
    sensors: [bool; 6],
    gamepad: Gamepad,
}

impl OpenGamepad {
    fn send_vibration(&mut self, values: [crate::VibrationValue; 2]) -> Result<(), SdlInputError> {
        // The safe wrapper marks this query unsafe; our Gamepad is live and
        // owned exclusively by this thread, as required by SDL.
        if !unsafe { self.gamepad.has_rumble() } {
            return Err(SdlInputError::new(
                "vibration",
                format!("{} has no rumble actuator", self.name),
            ));
        }
        if self.gamepad.vendor_id() == Some(0x057e) && self.gamepad.product_id() == Some(0x2009) {
            // Native Nintendo Switch Pro: preserve both carriers on each side.
            self.gamepad
                .send_effect(&crate::switch_rumble::packet(values))
                .map_err(|e| SdlInputError::new("Switch Pro HD rumble", e))
        } else {
            // Conventional host motors cannot reproduce arbitrary frequencies.
            // Map each band to SDL's corresponding motor, combining the sides.
            // https://wiki.libsdl.org/SDL3/SDL_RumbleGamepad
            let strength =
                |amplitude: f32| (amplitude.min(1.0) * f32::from(u16::MAX)).round() as u16;
            let low = strength(values[0].low_amplitude.max(values[1].low_amplitude));
            let high = strength(values[0].high_amplitude.max(values[1].high_amplitude));
            self.gamepad
                .set_rumble(low, high, 1000)
                .map_err(|e| SdlInputError::new("gamepad rumble", e))
        }
    }
}

struct DeviceChanges(Arc<AtomicBool>);

impl EventWatchCallback for DeviceChanges {
    fn callback(&mut self, event: Event) {
        if matches!(
            event,
            Event::ControllerDeviceAdded { .. }
                | Event::ControllerDeviceRemoved { .. }
                | Event::ControllerDeviceRemapped { .. }
        ) {
            self.0.store(true, Ordering::Release);
        }
    }
}

/// Main-thread subsystem references. InputWorker retains these until its
/// worker has joined, so no SDL subsystem is finalized by the polling thread.
#[derive(Clone)]
pub(crate) struct InputSubsystems {
    pub(crate) events: EventSubsystem,
    pub(crate) gamepads: GamepadSubsystem,
}

impl InputSubsystems {
    pub(crate) fn new(sdl: &Sdl) -> Result<Self, SdlInputError> {
        Ok(Self {
            events: sdl
                .event()
                .map_err(|e| SdlInputError::new("event initialization", e))?,
            gamepads: sdl
                .gamepad()
                .map_err(|e| SdlInputError::new("gamepad initialization", e))?,
        })
    }
}

/// Created and destroyed on the polling thread; initialization and final
/// subsystem shutdown belong to InputWorker's main-thread owner.
pub(crate) struct SdlInputBackend {
    open_gamepads: Vec<OpenGamepad>,
    failed_gamepads: Vec<JoystickId>,
    devices_changed: Arc<AtomicBool>,
    _device_watch: EventWatch<DeviceChanges>,
    events: EventSubsystem,
    gamepad_subsystem: GamepadSubsystem,
    next_controller_id: u64,
    vibration: crate::vibration::VibrationState,
    vibration_dirty: bool,
    vibration_active: bool,
    last_vibration_send: Option<Instant>,
}

impl SdlInputBackend {
    pub(crate) fn new(subsystems: InputSubsystems) -> Self {
        let InputSubsystems {
            events,
            gamepads: gamepad_subsystem,
        } = subsystems;
        // Callbacks can run on another thread. Only mark topology dirty here;
        // all device operations remain on the input thread.
        // https://wiki.libsdl.org/SDL3/SDL_AddEventWatch
        let devices_changed = Arc::new(AtomicBool::new(true));
        let device_watch = events.add_event_watch(DeviceChanges(Arc::clone(&devices_changed)));
        // Controls are sampled, not handled as events. Keep device events for
        // hotplug, including the joystick events SDL uses to discover gamepads.
        for event in [
            EventType::JoyAxisMotion,
            EventType::JoyHatMotion,
            EventType::JoyButtonDown,
            EventType::JoyButtonUp,
            EventType::ControllerAxisMotion,
            EventType::ControllerButtonDown,
            EventType::ControllerButtonUp,
            EventType::ControllerTouchpadDown,
            EventType::ControllerTouchpadMotion,
            EventType::ControllerTouchpadUp,
            EventType::ControllerSensorUpdated,
        ] {
            EventSubsystem::set_event_enabled(event, false);
        }
        Self {
            open_gamepads: Vec::new(),
            failed_gamepads: Vec::new(),
            devices_changed,
            _device_watch: device_watch,
            events,
            gamepad_subsystem,
            next_controller_id: 1,
            vibration: Default::default(),
            vibration_dirty: false,
            vibration_active: false,
            last_vibration_send: None,
        }
    }

    pub(crate) fn apply_vibration(
        &mut self,
        receiver: &crate::vibration::VibrationReceiver,
    ) -> Result<(), SdlInputError> {
        if let Some(state) = receiver
            .take_latest()
            .map_err(|e| SdlInputError::new("vibration mailbox", e))?
        {
            if state.controller != self.vibration.controller {
                self.stop_vibration()?;
                self.last_vibration_send = None;
            }
            self.vibration_dirty = !state.is_stopped() || self.vibration_active;
            self.vibration = state;
        }
        // SDL's Nintendo driver spaces writes by 30 ms and refreshes sustained
        // rumble every 50 ms. send_effect bypasses that scheduling, so preserve
        // it here without device I/O on the emulation thread.
        // https://github.com/libsdl-org/SDL/blob/release-3.4.12/src/joystick/hidapi/SDL_hidapi_switch.c
        let elapsed = self.last_vibration_send.map(|last| last.elapsed());
        let due = if self.vibration_dirty {
            elapsed.is_none_or(|elapsed| elapsed >= Duration::from_millis(30))
        } else {
            !self.vibration.is_stopped()
                && elapsed.is_some_and(|elapsed| elapsed >= Duration::from_millis(50))
        };
        if due {
            if let Some(open) = self
                .open_gamepads
                .iter_mut()
                .find(|open| Some(open.controller_id) == self.vibration.controller)
                && open.gamepad.connected()
            {
                if self.vibration_dirty {
                    log::debug!(
                        "controller vibration output: device={} values={:?}",
                        open.name,
                        self.vibration.values
                    );
                }
                // A failed transport write may have reached the actuator.
                // Shutdown must still attempt a neutral packet in that case.
                self.vibration_active |= !self.vibration.is_stopped();
                open.send_vibration(self.vibration.values)?;
                self.vibration_active = !self.vibration.is_stopped();
            }
            self.last_vibration_send = Some(Instant::now());
            self.vibration_dirty = false;
        }
        Ok(())
    }

    fn stop_vibration(&mut self) -> Result<(), SdlInputError> {
        if self.vibration_active
            && let Some(open) = self
                .open_gamepads
                .iter_mut()
                .find(|open| Some(open.controller_id) == self.vibration.controller)
            && open.gamepad.connected()
        {
            if let Some(last) = self.last_vibration_send {
                // Closing or retargeting must also respect Nintendo's minimum
                // write interval. This wait belongs to the device thread only.
                std::thread::sleep(Duration::from_millis(30).saturating_sub(last.elapsed()));
            }
            open.send_vibration([crate::VibrationValue::default(); 2])?;
        }
        self.vibration_active = false;
        Ok(())
    }

    fn reconcile_gamepads(&mut self) -> Result<(), SdlInputError> {
        self.gamepad_subsystem.update();
        // Watchers already observed hotplug. Do not accumulate an unused SDL
        // input event queue; flushing does not pump events or poll devices.
        // https://wiki.libsdl.org/SDL3/SDL_FlushEvents
        self.events.flush_events(
            sdl3_sys::events::SDL_EVENT_JOYSTICK_AXIS_MOTION.0,
            sdl3_sys::events::SDL_EVENT_GAMEPAD_STEAM_HANDLE_UPDATED.0,
        );
        if !self.devices_changed.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        let attached = self
            .gamepad_subsystem
            .gamepads()
            .map_err(|error| SdlInputError::new("controller enumeration", error))?;

        self.reconcile_attached_gamepads(&attached)
    }

    fn reconcile_attached_gamepads(
        &mut self,
        attached: &[JoystickId],
    ) -> Result<(), SdlInputError> {
        self.open_gamepads
            .retain(|entry| entry.gamepad.connected() && attached.contains(&entry.joystick_id));
        // An SDL instance ID lasts for one connection. Replugging gets a new ID,
        // even when both hotplug events occur between polls. Ignore a failed
        // instance to avoid repeating blocking handshakes and logs every poll.
        // https://wiki.libsdl.org/SDL3/SDL_JoystickID
        self.failed_gamepads.retain(|id| attached.contains(id));
        // Remapping may change SDL's type and labels without reconnecting.
        // Refresh only on topology/remapping notifications, never per sample.
        for open in &mut self.open_gamepads {
            if let Some(name) = open.gamepad.name()
                && name != *open.name
            {
                open.name = name.into();
            }
            open.kind = controller_kind(open.gamepad.r#type());
            open.labels = face_button_labels(&open.gamepad);
        }
        for &joystick_id in attached {
            if self.failed_gamepads.contains(&joystick_id)
                || self
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
            let gamepad = match self.gamepad_subsystem.open(joystick_id) {
                Ok(gamepad) => gamepad,
                Err(error) => {
                    log::error!(
                        "SDL controller open failed: name={name:?}, instance_id={}: {error}; \
                         ignoring this controller until it is disconnected and reconnected",
                        joystick_id.0
                    );
                    self.failed_gamepads.push(joystick_id);
                    continue;
                }
            };
            let sensors = enable_available_sensors(&gamepad);
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
                name: name.into(),
                kind: controller_kind(gamepad.r#type()),
                labels: face_button_labels(&gamepad),
                sensors,
                gamepad,
            });
        }
        Ok(())
    }
}

impl Drop for SdlInputBackend {
    fn drop(&mut self) {
        if let Err(error) = self.stop_vibration() {
            log::error!("cannot stop controller vibration during shutdown: {error}");
        }
    }
}

impl HostInputBackend for SdlInputBackend {
    type Error = SdlInputError;

    fn poll(&mut self) -> Result<Option<ControllerState>, Self::Error> {
        self.reconcile_gamepads()?;
        Ok(self.open_gamepads.first_mut().map(controller_state))
    }
}

fn controller_state(open: &mut OpenGamepad) -> ControllerState {
    let gamepad = &open.gamepad;
    let (buttons, dpad, left_stick, right_stick, triggers) =
        map_controls(|button| gamepad.button(button), |axis| gamepad.axis(axis));
    ControllerState {
        id: open.controller_id,
        name: open.name.clone(),
        kind: open.kind,
        buttons,
        button_labels: open.labels,
        dpad,
        left_stick,
        right_stick,
        triggers,
        motion: motion_state(gamepad, &mut open.sensors),
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
            up: buttons.contains(Button::DPadUp),
            down: buttons.contains(Button::DPadDown),
            left: buttons.contains(Button::DPadLeft),
            right: buttons.contains(Button::DPadRight),
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

fn enable_available_sensors(gamepad: &Gamepad) -> [bool; 6] {
    SENSOR_TYPES.map(|sensor_type| {
        // The gamepad is open and remains alive for this entire query.
        if unsafe { gamepad.has_sensor(sensor_type) } {
            match gamepad.sensor_set_enabled(sensor_type, true) {
                Ok(()) => return true,
                Err(error) => log::warn!("cannot enable gamepad sensor {sensor_type:?}: {error}"),
            }
        }
        false
    })
}

fn motion_state(gamepad: &Gamepad, enabled: &mut [bool; 6]) -> MotionState {
    let [
        gyroscope,
        accelerometer,
        left_gyroscope,
        right_gyroscope,
        left_accelerometer,
        right_accelerometer,
    ] = std::array::from_fn(|index| read_sensor(gamepad, SENSOR_TYPES[index], &mut enabled[index]));
    MotionState {
        gyroscope,
        accelerometer,
        left_gyroscope,
        right_gyroscope,
        left_accelerometer,
        right_accelerometer,
    }
}

fn read_sensor(
    gamepad: &Gamepad,
    sensor_type: SensorType,
    enabled: &mut bool,
) -> Option<MotionVector> {
    if !*enabled {
        return None;
    }
    let mut data = [0.0; 3];
    if let Err(error) = gamepad.sensor_get_data(sensor_type, &mut data) {
        log::warn!(
            "cannot read gamepad sensor {sensor_type:?}: {error}; disabling it until reconnection"
        );
        *enabled = false;
        return None;
    }
    if !data.iter().all(|value| value.is_finite()) {
        log::warn!(
            "non-finite gamepad sensor data for {sensor_type:?}; disabling it until reconnection"
        );
        *enabled = false;
        return None;
    }
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
    fn backend_ignores_open_failures_and_recognizes_reconnections() {
        use sdl3::joystick::{JoystickType, VirtualJoystickDescription};

        let sdl = sdl3::init().unwrap();
        let subsystems = InputSubsystems::new(&sdl).unwrap();
        let mut backend = SdlInputBackend::new(subsystems.clone());
        assert!(!EventSubsystem::event_enabled(
            EventType::ControllerButtonDown
        ));
        assert!(!EventSubsystem::event_enabled(
            EventType::ControllerButtonUp
        ));
        assert!(!EventSubsystem::event_enabled(
            EventType::ControllerAxisMotion
        ));
        assert!(EventSubsystem::event_enabled(
            EventType::ControllerDeviceAdded
        ));
        assert!(EventSubsystem::event_enabled(
            EventType::ControllerDeviceRemoved
        ));
        backend.poll().unwrap();
        let joystick = sdl.joystick().unwrap();
        let attach = || {
            joystick
                .attach_virtual_joystick(
                    VirtualJoystickDescription::new()
                        .name("Nixe hotplug test")
                        .joystick_type(JoystickType::Gamepad)
                        .with_button(SdlButton::South)
                        .with_axis(Axis::LeftX),
                )
                .unwrap()
        };
        let connected = attach();
        // An invalid SDL instance produces a real open failure. Supply the
        // enumeration explicitly to exercise a failed and a healthy device
        // together, independently of hardware attached to the test host.
        let failed = sdl3_sys::joystick::SDL_JoystickID(0);
        let attached = [failed, connected.id()];
        backend.reconcile_attached_gamepads(&attached).unwrap();
        assert_eq!(backend.failed_gamepads.len(), 1);
        assert_eq!(backend.failed_gamepads[0].0, failed.0);
        assert_eq!(backend.open_gamepads.len(), 1);
        let controller_id = backend.open_gamepads[0].controller_id;

        // A repeated poll must not even attempt to open the failed instance:
        // another SDL open failure would overwrite this thread's error string.
        sdl3::set_error("open was not retried").unwrap();
        backend.reconcile_attached_gamepads(&attached).unwrap();
        assert_eq!(sdl3::get_error().to_string(), "open was not retried");
        assert_eq!(backend.failed_gamepads.len(), 1);
        assert_eq!(backend.open_gamepads[0].controller_id, controller_id);

        // The failed connection disappears and a working connection arrives
        // between polls. Its new SDL instance must be opened immediately.
        let recovered = attach();
        backend
            .reconcile_attached_gamepads(&[connected.id(), recovered.id()])
            .unwrap();
        assert!(backend.failed_gamepads.is_empty());
        assert_eq!(backend.open_gamepads.len(), 2);
        assert_eq!(backend.open_gamepads[0].controller_id, controller_id);
        let recovered_id = backend.open_gamepads[1].controller_id;
        assert_ne!(controller_id, recovered_id);

        // Exercise real SDL enumeration, detachment, reattachment and snapshots.
        joystick
            .open(recovered.id())
            .unwrap()
            .set_virtual_button(0, true)
            .unwrap();
        joystick
            .open(recovered.id())
            .unwrap()
            .set_virtual_axis(0, 12345)
            .unwrap();
        drop(connected);
        let snapshot = backend.poll().unwrap().unwrap();
        assert_ne!(snapshot.id, controller_id);
        assert_eq!(snapshot.id, recovered_id);
        assert!(snapshot.buttons.contains(Button::South));
        assert_eq!(snapshot.left_stick.x, 12345);
        let old_instance = recovered.id().0;
        drop(recovered);
        let reconnected = attach();
        assert_ne!(reconnected.id().0, old_instance);
        let snapshot = backend.poll().unwrap().unwrap();
        assert_ne!(snapshot.id, recovered_id);
        assert_ne!(snapshot.id, controller_id);
        assert_eq!(&*snapshot.name, "Nixe hotplug test");
    }
}
