use std::collections::BTreeSet;
use std::sync::Mutex;
use std::time::Duration;

use nixe_cpu::{
    address::{AddressSpaceId, GuestVirtualAddress},
    memory::ExecutionMemory,
};
use nixe_input::EmulatedControllerState;
use nixe_runtime::{HandleError, SharedMemoryObject};

const HID_SHARED_MEMORY_SIZE: usize = 0x40000;
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

/// Host-controlled producer for Horizon's HID shared memory.
#[derive(Debug)]
pub struct HidSystem {
    shared_memory: SharedMemoryObject,
    sampling_number: u64,
    full_key_tail: u64,
    six_axis_tail: u64,
    home_tail: u64,
    capture_tail: u64,
    connected: bool,
    guest_mapping: Option<(AddressSpaceId, GuestVirtualAddress)>,
    configuration: Mutex<HidConfiguration>,
}

#[derive(Debug, Default)]
struct HidConfiguration {
    npad_active: bool,
    supported_style_set: u32,
    supported_ids: BTreeSet<u32>,
    active_six_axis_handles: BTreeSet<u32>,
}

impl Default for HidSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl HidSystem {
    #[must_use]
    pub fn new() -> Self {
        let shared_memory = SharedMemoryObject::zeroed_with_remote_permissions(
            HID_SHARED_MEMORY_SIZE,
            nixe_cpu::memory::MemoryPermissions::READ,
        )
        .unwrap_or_else(|error| panic!("cannot allocate fixed HID shared memory: {error}"));
        Self {
            shared_memory,
            sampling_number: 0,
            full_key_tail: LIFO_CAPACITY - 1,
            six_axis_tail: LIFO_CAPACITY - 1,
            home_tail: LIFO_CAPACITY - 1,
            capture_tail: LIFO_CAPACITY - 1,
            connected: false,
            guest_mapping: None,
            configuration: Mutex::new(HidConfiguration::default()),
        }
    }

    #[must_use]
    pub fn shared_memory(&self) -> SharedMemoryObject {
        self.shared_memory.clone()
    }

    pub(crate) fn owns(&self, shared_memory: &SharedMemoryObject) -> bool {
        self.shared_memory.same_backing(shared_memory)
    }

    pub(crate) fn register_mapping(
        &mut self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) {
        self.guest_mapping = Some((address_space, address));
    }

    pub(crate) fn unregister_mapping(
        &mut self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) {
        if self.guest_mapping == Some((address_space, address)) {
            self.guest_mapping = None;
        }
    }

    pub(crate) fn activate_npad(&self) {
        let mut configuration = self
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        configuration.npad_active = true;
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

    pub(crate) fn synchronize(&self, memory: &ExecutionMemory) -> Result<(), HandleError> {
        let Some((address_space, base)) = self.guest_mapping else {
            return Ok(());
        };
        for (offset, size) in [
            (HOME_BUTTON_LIFO_OFFSET, 0x200),
            (CAPTURE_BUTTON_LIFO_OFFSET, 0x200),
            (NPAD_OFFSET, NPAD_ENTRY_SIZE),
        ] {
            let mut bytes = vec![0; size];
            self.shared_memory.read(offset, &mut bytes)?;
            let Some(address) = base.checked_add(offset as u64) else {
                return Err(HandleError::InvalidRange);
            };
            if !memory.overwrite_mapped_ram(address_space, address, &bytes) {
                return Err(HandleError::InvalidRange);
            }
        }
        Ok(())
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
                self.shared_memory
                    .write(NPAD_OFFSET, &vec![0; NPAD_ENTRY_SIZE])?;
                self.publish_system_button(HOME_BUTTON_LIFO_OFFSET, false, true)?;
                self.publish_system_button(CAPTURE_BUTTON_LIFO_OFFSET, false, false)?;
                self.connected = false;
            }
            return Ok(());
        };

        self.connected = true;
        self.sampling_number = self.sampling_number.saturating_add(1);
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

        self.full_key_tail = next_tail(self.full_key_tail);
        let mut common = [0_u8; COMMON_ENTRY_SIZE];
        put_u64(&mut common, 0, self.sampling_number);
        put_u64(&mut common, 8, self.sampling_number);
        put_u64(&mut common, 16, npad_buttons(state));
        put_i32(&mut common, 24, i32::from(state.left_stick.x));
        put_i32(&mut common, 28, i32::from(state.left_stick.y));
        put_i32(&mut common, 32, i32::from(state.right_stick.x));
        put_i32(&mut common, 36, i32::from(state.right_stick.y));
        put_u32(&mut common, 40, NPAD_ATTRIBUTE_CONNECTED);
        self.publish_lifo_entry(
            NPAD_OFFSET + FULL_KEY_LIFO_OFFSET,
            self.full_key_tail,
            COMMON_ENTRY_SIZE,
            &common,
        )?;

        if publish_six_axis {
            self.six_axis_tail = next_tail(self.six_axis_tail);
            let mut sensor = [0_u8; SIX_AXIS_ENTRY_SIZE];
            put_u64(&mut sensor, 0, self.sampling_number);
            put_u64(
                &mut sensor,
                8,
                u64::try_from(delta.as_nanos()).unwrap_or(u64::MAX),
            );
            put_u64(&mut sensor, 16, self.sampling_number);
            if let Some(acceleration) = state.accelerometer {
                put_f32(&mut sensor, 24, acceleration.x / STANDARD_GRAVITY);
                put_f32(&mut sensor, 28, acceleration.y / STANDARD_GRAVITY);
                put_f32(&mut sensor, 32, acceleration.z / STANDARD_GRAVITY);
            }
            if let Some(gyroscope) = state.gyroscope {
                put_f32(&mut sensor, 36, gyroscope.x);
                put_f32(&mut sensor, 40, gyroscope.y);
                put_f32(&mut sensor, 44, gyroscope.z);
            }
            for offset in [60, 76, 92] {
                put_f32(&mut sensor, offset, 1.0);
            }
            if state.gyroscope.is_some() || state.accelerometer.is_some() {
                put_u32(&mut sensor, 96, SIX_AXIS_ATTRIBUTE_CONNECTED);
            }
            self.publish_lifo_entry(
                NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET,
                self.six_axis_tail,
                SIX_AXIS_ENTRY_SIZE,
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
        let tail = if home {
            self.home_tail = next_tail(self.home_tail);
            self.home_tail
        } else {
            self.capture_tail = next_tail(self.capture_tail);
            self.capture_tail
        };
        let mut entry = [0_u8; SYSTEM_BUTTON_ENTRY_SIZE];
        put_u64(&mut entry, 0, self.sampling_number);
        put_u64(&mut entry, 8, self.sampling_number);
        put_u64(&mut entry, 16, u64::from(pressed));
        self.publish_lifo_entry(lifo_offset, tail, SYSTEM_BUTTON_ENTRY_SIZE, &entry)
    }

    fn publish_lifo_entry(
        &self,
        lifo_offset: usize,
        tail: u64,
        entry_size: usize,
        entry: &[u8],
    ) -> Result<(), HandleError> {
        let entry_offset = lifo_offset + 0x20 + tail as usize * entry_size;
        self.shared_memory.write(entry_offset, entry)?;
        self.write_u64(lifo_offset + 8, LIFO_CAPACITY)?;
        self.write_u64(lifo_offset + 16, tail)?;
        let count = self
            .read_u64(lifo_offset + 24)?
            .saturating_add(1)
            .min(LIFO_CAPACITY);
        self.write_u64(lifo_offset + 24, count)
    }

    fn read_u64(&self, offset: usize) -> Result<u64, HandleError> {
        let mut bytes = [0_u8; 8];
        self.shared_memory.read(offset, &mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn write_u8(&self, offset: usize, value: u8) -> Result<(), HandleError> {
        self.shared_memory.write(offset, &[value])
    }

    fn write_u32(&self, offset: usize, value: u32) -> Result<(), HandleError> {
        self.shared_memory.write(offset, &value.to_le_bytes())
    }

    fn write_u64(&self, offset: usize, value: u64) -> Result<(), HandleError> {
        self.shared_memory.write(offset, &value.to_le_bytes())
    }
}

fn next_tail(tail: u64) -> u64 {
    (tail + 1) % LIFO_CAPACITY
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
    use nixe_input::{EmulatedButtonState, MotionVector, StickState};

    use super::*;

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
    fn publishes_player_one_full_key_state_and_disconnects_it() {
        let mut hid = HidSystem::new();
        configure_player_one(&hid, false);
        let memory = hid.shared_memory();
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
        let memory = hid.shared_memory();
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
            ..EmulatedControllerState::default()
        };
        hid.publish(Some(&state), Duration::from_millis(5)).unwrap();

        let six_axis_entry = NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET + 0x20;
        assert_eq!(read_u64(&memory, six_axis_entry + 8), 5_000_000);
        assert_eq!(read_u32(&memory, six_axis_entry + 24), 1.0_f32.to_bits());
        assert_eq!(read_u32(&memory, six_axis_entry + 36), 1.0_f32.to_bits());
        assert_eq!(read_u64(&memory, HOME_BUTTON_LIFO_OFFSET + 0x20 + 16), 1);
        assert_eq!(read_u64(&memory, CAPTURE_BUTTON_LIFO_OFFSET + 0x20 + 16), 1);
    }

    #[test]
    fn configuration_gates_npad_and_six_axis_publication() {
        let mut hid = HidSystem::new();
        let memory = hid.shared_memory();
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
