use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use nixe_cpu::memory::ExecutionMemory;
use nixe_input::{EmulatedControllerState, EmulatedTouchScreenState};
use nixe_runtime::{HandleError, SharedMemoryObject};

const HID_SHARED_MEMORY_SIZE: usize = 0x40000;
const TOUCH_SCREEN_OFFSET: usize = 0x400;
const TOUCH_SCREEN_ENTRY_SIZE: usize = 0x298;
const TOUCH_STATE_SIZE: usize = 0x28;
const NPAD_OFFSET: usize = 0x9a00;
const NPAD_ENTRY_SIZE: usize = 0x5000;
const FULL_KEY_LIFO_OFFSET: usize = 0x28;
const FULL_KEY_SIX_AXIS_LIFO_OFFSET: usize = 0x1758;
const HOME_BUTTON_LIFO_OFFSET: usize = 0x4c00;
const CAPTURE_BUTTON_LIFO_OFFSET: usize = 0x5000;

const LIFO_CAPACITY: u64 = 17;
const COMMON_ENTRY_SIZE: usize = 0x30;
const SIX_AXIS_ENTRY_SIZE: usize = 0x68;
const SYSTEM_BUTTON_ENTRY_SIZE: usize = 0x18;

const NPAD_STYLE_FULL_KEY: u32 = 1;
const NPAD_DEVICE_TYPE_FULL_KEY: u32 = 1;
const NPAD_ATTRIBUTE_CONNECTED: u32 = 1;
const SIX_AXIS_ATTRIBUTE_CONNECTED: u32 = 1;
const APPLET_FOOTER_SWITCH_PRO_CONTROLLER: u8 = 12;
const STANDARD_GRAVITY: f32 = 9.806_65;

/// Position of a modeled LRA actuator: left (1) or right (2).
pub(crate) fn vibration_device_position(handle: u32) -> Option<u32> {
    // The style, Npad ID and actuator index are packed into consecutive bytes.
    // FullKey, Handheld and paired Joy-Con expose both sides; single Joy-Con
    // expose only their own side. The final byte is reserved.
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/hid.c#L1157-L1232
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L486-L497
    let [style, npad_id, device_index, reserved] = handle.to_le_bytes();
    if reserved != 0
        || !matches!(npad_id, 0..=7 | 0x10 | 0x20)
        || !match style {
            3..=5 => device_index <= 1,
            6 => device_index == 0,
            7 => device_index == 1,
            _ => false,
        }
    {
        return None;
    }
    Some(u32::from(device_index) + 1)
}

/// Host-controlled producer for Horizon's HID shared memory.
#[derive(Debug)]
pub struct HidSystem {
    shared_memory: OnceLock<SharedMemoryObject>,
    sampling_number: u64,
    touch_screen_sampling_number: u64,
    touch_screen: Lifo,
    full_key: Lifo,
    six_axis: Lifo,
    home: Lifo,
    capture: Lifo,
    connected: bool,
    configuration: Mutex<HidConfiguration>,
}

#[derive(Debug, Default)]
struct HidConfiguration {
    touch_screen_active: bool,
    npad_active: bool,
    supported_style_set: u32,
    supported_ids: BTreeSet<u32>,
    npad_joy_hold_type: u64,
    active_six_axis_handles: BTreeSet<u32>,
}

#[derive(Debug)]
struct Lifo {
    tail: u64,
    count: u64,
}

impl Default for Lifo {
    fn default() -> Self {
        Self {
            tail: LIFO_CAPACITY - 1,
            count: 0,
        }
    }
}

impl Lifo {
    fn publish(
        &mut self,
        memory: &SharedMemoryObject,
        offset: usize,
        entry: &[u8],
    ) -> Result<(), HandleError> {
        self.tail = (self.tail + 1) % LIFO_CAPACITY;
        self.count = (self.count + 1).min(LIFO_CAPACITY);
        memory.write(offset + 0x20 + self.tail as usize * entry.len(), entry)?;
        let mut header = [0; 24];
        put_u64(&mut header, 0, LIFO_CAPACITY);
        put_u64(&mut header, 8, self.tail);
        put_u64(&mut header, 16, self.count);
        memory.write(offset + 8, &header)
    }
}

impl Default for HidSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl HidSystem {
    #[must_use]
    pub fn new() -> Self {
        Self {
            shared_memory: OnceLock::new(),
            sampling_number: 0,
            touch_screen_sampling_number: 0,
            touch_screen: Lifo::default(),
            full_key: Lifo::default(),
            six_axis: Lifo::default(),
            home: Lifo::default(),
            capture: Lifo::default(),
            connected: false,
            configuration: Mutex::new(HidConfiguration::default()),
        }
    }

    pub fn shared_memory(
        &self,
        memory: &ExecutionMemory,
    ) -> Result<SharedMemoryObject, HandleError> {
        if let Some(shared) = self.shared_memory.get() {
            return Ok(shared.clone());
        }
        let shared = SharedMemoryObject::for_process(
            memory,
            HID_SHARED_MEMORY_SIZE,
            nixe_cpu::memory::MemoryPermissions::READ,
        )?;
        let _ = self.shared_memory.set(shared);
        Ok(self
            .shared_memory
            .get()
            .expect("HID allocation initialized")
            .clone())
    }

    pub(crate) fn activate_npad(&self) {
        let mut configuration = self
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        configuration.npad_active = true;
    }

    pub(crate) fn activate_touch_screen(&self) {
        self.configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .touch_screen_active = true;
    }

    pub(crate) fn set_supported_npad_style_set(&self, style_set: u32) {
        self.configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .supported_style_set = style_set;
    }

    pub(crate) fn set_supported_npad_ids(&self, ids: impl IntoIterator<Item = u32>) -> bool {
        let ids = ids.into_iter().collect::<BTreeSet<_>>();
        if ids.iter().any(|id| !matches!(*id, 0..=7 | 0x10 | 0x20)) {
            return false;
        }
        self.configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .supported_ids = ids;
        true
    }

    pub(crate) fn set_npad_joy_hold_type(&self, hold_type: u64) -> bool {
        // Vertical (0) is the default; horizontal (1) affects single Joy-Con
        // orientation. FullKey publication does not consume this setting.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h
        if !matches!(hold_type, 0 | 1) {
            return false;
        }
        self.configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .npad_joy_hold_type = hold_type;
        true
    }

    pub(crate) fn npad_joy_hold_type(&self) -> u64 {
        self.configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .npad_joy_hold_type
    }

    pub(crate) fn set_six_axis_sensor_active(&self, handle: u32, active: bool) {
        let mut configuration = self
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if active {
            configuration.active_six_axis_handles.insert(handle);
        } else {
            configuration.active_six_axis_handles.remove(&handle);
        }
    }

    /// Publishes one complete touch-screen sample.
    pub fn publish_touch_screen(
        &mut self,
        state: &EmulatedTouchScreenState,
        delta: Duration,
    ) -> Result<(), HandleError> {
        if self.shared_memory.get().is_none()
            || !self
                .configuration
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .touch_screen_active
        {
            return Ok(());
        }
        self.touch_screen_sampling_number = self.touch_screen_sampling_number.saturating_add(1);
        let mut entry = [0_u8; TOUCH_SCREEN_ENTRY_SIZE];
        put_u64(&mut entry, 0, self.touch_screen_sampling_number);
        put_u64(&mut entry, 8, self.touch_screen_sampling_number);
        put_u32(
            &mut entry,
            16,
            u32::try_from(state.contacts().len()).expect("touch count is ABI-bounded"),
        );
        let delta_time = u64::try_from(delta.as_nanos()).unwrap_or(u64::MAX);
        for (index, contact) in state.contacts().iter().enumerate() {
            let offset = 24 + index * TOUCH_STATE_SIZE;
            put_u64(&mut entry, offset, delta_time);
            put_u32(&mut entry, offset + 8, contact.attributes);
            put_u32(&mut entry, offset + 12, contact.finger_id);
            put_u32(&mut entry, offset + 16, contact.x);
            put_u32(&mut entry, offset + 20, contact.y);
            put_u32(&mut entry, offset + 24, contact.diameter_x);
            put_u32(&mut entry, offset + 28, contact.diameter_y);
            put_u32(&mut entry, offset + 32, contact.rotation_angle);
        }
        // Switch HID exposes a 17-entry atomic touchscreen LIFO at 0x400.
        // HidTouchScreenStateAtomicStorage is 0x298 bytes and carries up to 16
        // contacts in the public libnx ABI:
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L666-L713
        self.touch_screen.publish(
            self.shared_memory.get().expect("HID initialized"),
            TOUCH_SCREEN_OFFSET,
            &entry,
        )
    }

    /// Publishes one player-one Pro Controller sample.
    ///
    /// `None` transitions the shared state to a disconnected NPad. Repeated
    /// disconnected updates are ignored.
    pub fn publish(
        &mut self,
        state: Option<&EmulatedControllerState>,
        delta: Duration,
    ) -> Result<(), HandleError> {
        if self.shared_memory.get().is_none() {
            return Ok(());
        }
        let (publish_player_one, publish_six_axis) = {
            let configuration = self
                .configuration
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                configuration.npad_active
                    && configuration.supported_style_set & NPAD_STYLE_FULL_KEY != 0
                    && configuration.supported_ids.contains(&0),
                // FullKey, Player 1, device index 2. The packed handle layout
                // is pinned in the public libnx HidSixAxisSensorHandle ABI:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L1412-L1421
                configuration.active_six_axis_handles.contains(&0x0002_0003),
            )
        };
        let Some(state) = state.filter(|_| publish_player_one) else {
            if self.connected {
                self.sampling_number = self.sampling_number.saturating_add(1);
                self.memory().write(NPAD_OFFSET, &[0; NPAD_ENTRY_SIZE])?;
                self.full_key = Lifo::default();
                self.six_axis = Lifo::default();
                self.publish_system_button(HOME_BUTTON_LIFO_OFFSET, false, true)?;
                self.publish_system_button(CAPTURE_BUTTON_LIFO_OFFSET, false, false)?;
                self.connected = false;
            }
            return Ok(());
        };

        self.sampling_number = self.sampling_number.saturating_add(1);
        if !self.connected {
            self.write_u32(NPAD_OFFSET, NPAD_STYLE_FULL_KEY)?;
            self.write_u32(NPAD_OFFSET + 4, 0)?;
            self.write_u32(NPAD_OFFSET + 8, 0)?;
            self.write_u32(NPAD_OFFSET + 0x4188, NPAD_DEVICE_TYPE_FULL_KEY)?;
            self.write_u64(
                NPAD_OFFSET + 0x4190,
                1 << 3 | 1 << 11 | 1 << 13 | 1 << 14 | 1 << 15,
            )?;
            self.write_u32(NPAD_OFFSET + 0x419c, 4)?;
            self.write_u8(NPAD_OFFSET + 0x41ac, APPLET_FOOTER_SWITCH_PRO_CONTROLLER)?;
            self.connected = true;
        }

        let mut common = [0_u8; COMMON_ENTRY_SIZE];
        put_u64(&mut common, 0, self.sampling_number);
        put_u64(&mut common, 8, self.sampling_number);
        put_u64(&mut common, 16, npad_buttons(state));
        put_i32(&mut common, 24, i32::from(state.left_stick.x));
        put_i32(&mut common, 28, i32::from(state.left_stick.y));
        put_i32(&mut common, 32, i32::from(state.right_stick.x));
        put_i32(&mut common, 36, i32::from(state.right_stick.y));
        put_u32(&mut common, 40, NPAD_ATTRIBUTE_CONNECTED);
        self.full_key.publish(
            self.shared_memory.get().expect("HID initialized"),
            NPAD_OFFSET + FULL_KEY_LIFO_OFFSET,
            &common,
        )?;

        if publish_six_axis {
            let mut sensor = [0_u8; SIX_AXIS_ENTRY_SIZE];
            put_u64(&mut sensor, 0, self.sampling_number);
            put_u64(
                &mut sensor,
                8,
                u64::try_from(delta.as_nanos()).unwrap_or(u64::MAX),
            );
            put_u64(&mut sensor, 16, self.sampling_number);
            if let Some(acceleration) = state.accelerometer {
                // Horizon uses g, turns/s and accumulated turns. Its motion
                // frame is (SDL X, -SDL Z, SDL Y), with acceleration negated.
                // https://github.com/eden-emulator/mirror/blob/master/src/input_common/drivers/sdl_driver.cpp
                // https://github.com/nintendoswitchemulators/ryujinx/blob/master/src/Ryujinx.Input/HLE/NpadController.cs
                put_f32(&mut sensor, 24, -acceleration.x / STANDARD_GRAVITY);
                put_f32(&mut sensor, 28, acceleration.z / STANDARD_GRAVITY);
                put_f32(&mut sensor, 32, -acceleration.y / STANDARD_GRAVITY);
            }
            if let Some(gyroscope) = state.gyroscope {
                put_turns(&mut sensor, 36, gyroscope);
            }
            if let Some(motion) = state.motion {
                put_turns(&mut sensor, 48, motion.angle);
                let axes = [(0, 1.0), (2, -1.0), (1, 1.0)];
                for (row, &(source_row, row_sign)) in axes.iter().enumerate() {
                    for (column, &(source_column, column_sign)) in axes.iter().enumerate() {
                        put_f32(
                            &mut sensor,
                            60 + (row * 3 + column) * 4,
                            motion.orientation[source_row][source_column] * row_sign * column_sign,
                        );
                    }
                }
                put_u32(&mut sensor, 96, SIX_AXIS_ATTRIBUTE_CONNECTED);
            }
            self.six_axis.publish(
                self.shared_memory.get().expect("HID initialized"),
                NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET,
                &sensor,
            )?;
        }

        self.publish_system_button(HOME_BUTTON_LIFO_OFFSET, state.buttons.home, true)?;
        self.publish_system_button(CAPTURE_BUTTON_LIFO_OFFSET, state.buttons.capture, false)
    }

    fn publish_system_button(
        &mut self,
        lifo_offset: usize,
        pressed: bool,
        home: bool,
    ) -> Result<(), HandleError> {
        let lifo = if home {
            &mut self.home
        } else {
            &mut self.capture
        };
        let mut entry = [0_u8; SYSTEM_BUTTON_ENTRY_SIZE];
        put_u64(&mut entry, 0, self.sampling_number);
        put_u64(&mut entry, 8, self.sampling_number);
        put_u64(&mut entry, 16, u64::from(pressed));
        lifo.publish(
            self.shared_memory.get().expect("HID initialized"),
            lifo_offset,
            &entry,
        )
    }

    fn write_u8(&self, offset: usize, value: u8) -> Result<(), HandleError> {
        self.memory().write(offset, &[value])
    }

    fn write_u32(&self, offset: usize, value: u32) -> Result<(), HandleError> {
        self.memory().write(offset, &value.to_le_bytes())
    }

    fn write_u64(&self, offset: usize, value: u64) -> Result<(), HandleError> {
        self.memory().write(offset, &value.to_le_bytes())
    }

    fn memory(&self) -> &SharedMemoryObject {
        self.shared_memory
            .get()
            .expect("HID memory initialized before publication")
    }
}

fn put_turns(bytes: &mut [u8], offset: usize, radians: nixe_input::MotionVector) {
    put_f32(bytes, offset, radians.x / std::f32::consts::TAU);
    put_f32(bytes, offset + 4, -radians.z / std::f32::consts::TAU);
    put_f32(bytes, offset + 8, radians.y / std::f32::consts::TAU);
}

fn npad_buttons(state: &EmulatedControllerState) -> u64 {
    let buttons = state.buttons;
    u64::from(buttons.a)
        | u64::from(buttons.b) << 1
        | u64::from(buttons.x) << 2
        | u64::from(buttons.y) << 3
        | u64::from(buttons.left_stick) << 4
        | u64::from(buttons.right_stick) << 5
        | u64::from(buttons.l) << 6
        | u64::from(buttons.r) << 7
        | u64::from(buttons.zl) << 8
        | u64::from(buttons.zr) << 9
        | u64::from(buttons.plus) << 10
        | u64::from(buttons.minus) << 11
        | u64::from(buttons.dpad_left) << 12
        | u64::from(buttons.dpad_up) << 13
        | u64::from(buttons.dpad_right) << 14
        | u64::from(buttons.dpad_down) << 15
}

fn put_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(output: &mut [u8], offset: usize, value: i32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f32(output: &mut [u8], offset: usize, value: f32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use nixe_input::{
        EmulatedButtonState, EmulatedTouchContact, MotionVector, StickState, TOUCH_ATTRIBUTE_END,
        TOUCH_ATTRIBUTE_START, touch_screen_channel,
    };

    use super::*;

    #[test]
    fn publication_updates_the_mapped_pages_without_synchronization() {
        use nixe_cpu::memory::{MemoryPermissions, ProcessMemory};
        use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
        let memory = ExecutionMemory::new();
        let mut hid = HidSystem::new();
        let shared = hid.shared_memory(&memory).unwrap();
        let space = AddressSpaceId::new(1);
        let address = GuestVirtualAddress::new(0x1000);
        memory
            .map_shared_backing(space, address, shared.backing(), MemoryPermissions::READ)
            .unwrap();
        configure_player_one(&hid, false);
        hid.publish(Some(&EmulatedControllerState::default()), Duration::ZERO)
            .unwrap();
        let mut style = [0; 4];
        memory
            .read_bytes(
                space,
                address.checked_add(NPAD_OFFSET as u64).unwrap(),
                &mut style,
            )
            .unwrap();
        assert_eq!(u32::from_le_bytes(style), NPAD_STYLE_FULL_KEY);
        assert!(memory.write_bytes(space, address, &[1]).is_err());
    }

    fn read_u32(memory: &SharedMemoryObject, offset: usize) -> u32 {
        let mut bytes = [0; 4];
        memory.read(offset, &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    fn read_u64(memory: &SharedMemoryObject, offset: usize) -> u64 {
        let mut bytes = [0; 8];
        memory.read(offset, &mut bytes).unwrap();
        u64::from_le_bytes(bytes)
    }

    fn configure_player_one(hid: &HidSystem, six_axis: bool) {
        hid.activate_npad();
        hid.set_supported_npad_style_set(NPAD_STYLE_FULL_KEY);
        assert!(hid.set_supported_npad_ids([0]));
        if six_axis {
            hid.set_six_axis_sensor_active(0x0002_0003, true);
        }
    }

    #[test]
    fn publishes_atomic_multitouch_samples_after_activation() {
        let mut hid = HidSystem::new();
        let memory = hid.shared_memory(&ExecutionMemory::new()).unwrap();
        let (writer, mut reader) = touch_screen_channel();
        let first = EmulatedTouchContact {
            finger_id: 3,
            x: 120,
            y: 240,
            diameter_x: 4,
            diameter_y: 5,
            rotation_angle: 6,
            ..EmulatedTouchContact::default()
        };
        let second = EmulatedTouchContact {
            finger_id: 8,
            x: 640,
            y: 360,
            diameter_x: 1,
            diameter_y: 1,
            ..EmulatedTouchContact::default()
        };
        assert!(writer.begin(first));
        assert!(writer.begin(second));

        hid.publish_touch_screen(&reader.sample(), Duration::from_millis(5))
            .unwrap();
        assert_eq!(read_u64(&memory, TOUCH_SCREEN_OFFSET + 8), 0);

        hid.activate_touch_screen();
        hid.publish_touch_screen(&reader.sample(), Duration::from_millis(5))
            .unwrap();
        let entry = TOUCH_SCREEN_OFFSET + 0x20;
        assert_eq!(read_u64(&memory, TOUCH_SCREEN_OFFSET + 8), LIFO_CAPACITY);
        assert_eq!(read_u64(&memory, TOUCH_SCREEN_OFFSET + 16), 0);
        assert_eq!(read_u64(&memory, TOUCH_SCREEN_OFFSET + 24), 1);
        assert_eq!(read_u64(&memory, entry), 1);
        assert_eq!(read_u64(&memory, entry + 8), 1);
        assert_eq!(read_u32(&memory, entry + 16), 2);
        assert_eq!(read_u64(&memory, entry + 24), 5_000_000);
        assert_eq!(read_u32(&memory, entry + 32), 0);
        assert_eq!(read_u32(&memory, entry + 36), 3);
        assert_eq!(read_u32(&memory, entry + 40), 120);
        assert_eq!(read_u32(&memory, entry + 44), 240);
        assert_eq!(read_u32(&memory, entry + 48), 4);
        assert_eq!(read_u32(&memory, entry + 52), 5);
        assert_eq!(read_u32(&memory, entry + 56), 6);
        assert_eq!(read_u32(&memory, entry + 24 + TOUCH_STATE_SIZE + 12), 8);

        assert!(writer.end(first));
        let ended = reader.sample();
        assert_eq!(ended.contacts()[0].attributes, TOUCH_ATTRIBUTE_END);
        assert_eq!(ended.contacts()[1].attributes, 0);
        assert_ne!(ended.contacts()[0].attributes, TOUCH_ATTRIBUTE_START);
    }

    #[test]
    fn publishes_player_one_full_key_state_and_disconnects_it() {
        let mut hid = HidSystem::new();
        configure_player_one(&hid, false);
        // Joy-Con orientation must not rotate a FullKey controller's input.
        assert!(hid.set_npad_joy_hold_type(1));
        let memory = hid.shared_memory(&ExecutionMemory::new()).unwrap();
        let state = EmulatedControllerState {
            buttons: EmulatedButtonState {
                a: true,
                zl: true,
                plus: true,
                dpad_up: true,
                ..EmulatedButtonState::default()
            },
            left_stick: StickState { x: 123, y: -456 },
            right_stick: StickState { x: -789, y: 321 },
            ..EmulatedControllerState::default()
        };
        hid.publish(Some(&state), Duration::from_millis(5)).unwrap();

        assert_eq!(read_u32(&memory, NPAD_OFFSET), NPAD_STYLE_FULL_KEY);
        assert_eq!(read_u64(&memory, NPAD_OFFSET + 0x38), 0);
        assert_eq!(read_u64(&memory, NPAD_OFFSET + 0x40), 1);
        assert_eq!(
            read_u64(&memory, NPAD_OFFSET + 0x58),
            1 | 1 << 8 | 1 << 10 | 1 << 13
        );
        assert_eq!(read_u32(&memory, NPAD_OFFSET + 0x60), 123);
        assert_eq!(read_u32(&memory, NPAD_OFFSET + 0x64), (-456_i32) as u32);

        hid.publish(None, Duration::from_millis(5)).unwrap();
        assert_eq!(read_u32(&memory, NPAD_OFFSET), 0);
    }

    #[test]
    fn publishes_motion_and_system_buttons() {
        let mut hid = HidSystem::new();
        configure_player_one(&hid, true);
        let memory = hid.shared_memory(&ExecutionMemory::new()).unwrap();
        let state = EmulatedControllerState {
            buttons: EmulatedButtonState {
                home: true,
                capture: true,
                ..EmulatedButtonState::default()
            },
            gyroscope: Some(MotionVector {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            }),
            accelerometer: Some(MotionVector {
                x: STANDARD_GRAVITY,
                y: 0.0,
                z: 0.0,
            }),
            motion: Some(nixe_input::MotionEstimate {
                angle: MotionVector {
                    x: 0.0,
                    y: std::f32::consts::FRAC_PI_2,
                    z: std::f32::consts::TAU,
                },
                // SDL +90 degrees about Y (yaw).
                orientation: [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            }),
            ..EmulatedControllerState::default()
        };
        hid.publish(Some(&state), Duration::from_millis(5)).unwrap();

        let six_axis_entry = NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET + 0x20;
        assert_eq!(read_u64(&memory, six_axis_entry + 8), 5_000_000);
        assert_eq!(read_u32(&memory, six_axis_entry + 24), (-1.0_f32).to_bits());
        assert_eq!(
            read_u32(&memory, six_axis_entry + 36),
            (1.0_f32 / std::f32::consts::TAU).to_bits()
        );
        assert_eq!(read_u64(&memory, HOME_BUTTON_LIFO_OFFSET + 0x20 + 16), 1);
        assert_eq!(read_u64(&memory, CAPTURE_BUTTON_LIFO_OFFSET + 0x20 + 16), 1);
        assert_eq!(read_u32(&memory, six_axis_entry + 52), (-1.0_f32).to_bits());
        assert_eq!(read_u32(&memory, six_axis_entry + 56), 0.25_f32.to_bits());
        assert_eq!(read_u32(&memory, six_axis_entry + 64), 1.0_f32.to_bits());
        assert_eq!(read_u32(&memory, six_axis_entry + 72), (-1.0_f32).to_bits());
        assert_eq!(
            read_u32(&memory, six_axis_entry + 96),
            SIX_AXIS_ATTRIBUTE_CONNECTED
        );
    }

    #[test]
    fn lifos_wrap_and_restart_cleanly_after_disconnection() {
        let mut hid = HidSystem::new();
        let memory = hid.shared_memory(&ExecutionMemory::new()).unwrap();
        configure_player_one(&hid, true);
        for _ in 0..40 {
            hid.publish(
                Some(&EmulatedControllerState::default()),
                Duration::from_millis(5),
            )
            .unwrap();
        }
        let lifo = NPAD_OFFSET + FULL_KEY_LIFO_OFFSET;
        assert_eq!(read_u64(&memory, lifo + 16), 5);
        assert_eq!(read_u64(&memory, lifo + 24), LIFO_CAPACITY);
        assert_eq!(
            read_u32(
                &memory,
                NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET + 0x20 + 5 * SIX_AXIS_ENTRY_SIZE + 96
            ),
            0
        );
        hid.publish(None, Duration::ZERO).unwrap();
        hid.publish(Some(&EmulatedControllerState::default()), Duration::ZERO)
            .unwrap();
        assert_eq!(read_u64(&memory, lifo + 16), 0);
        assert_eq!(read_u64(&memory, lifo + 24), 1);
    }

    #[test]
    fn configuration_gates_npad_and_six_axis_publication() {
        let mut hid = HidSystem::new();
        let memory = hid.shared_memory(&ExecutionMemory::new()).unwrap();
        let state = EmulatedControllerState::default();

        hid.publish(Some(&state), Duration::from_millis(5)).unwrap();
        assert_eq!(read_u32(&memory, NPAD_OFFSET), 0);

        configure_player_one(&hid, false);
        hid.publish(Some(&state), Duration::from_millis(5)).unwrap();
        assert_eq!(read_u32(&memory, NPAD_OFFSET), NPAD_STYLE_FULL_KEY);
        assert_eq!(
            read_u64(&memory, NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET + 24),
            0
        );

        hid.set_six_axis_sensor_active(0x0002_0003, true);
        hid.publish(Some(&state), Duration::from_millis(5)).unwrap();
        assert_eq!(
            read_u64(&memory, NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET + 24),
            1
        );
    }
}
