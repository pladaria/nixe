use std::collections::{BTreeSet, VecDeque};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use nixe_input::{EmulatedControllerState, EmulatedTouchScreenState};
use nixe_memory::{
    Access, AccessResult, MemoryAccount, MemoryError, MemoryPermissions, PendingAccess,
};
use nixe_runtime::{
    EventObject, ExternalEventSource, ReadableEventObject, SharedMemoryObject, WritableEventObject,
};

const HID_SHARED_MEMORY_SIZE: usize = 0x40000;
const TOUCH_SCREEN_OFFSET: usize = 0x400;
const TOUCH_SCREEN_ENTRY_SIZE: usize = 0x298;
const TOUCH_STATE_SIZE: usize = 0x28;
// Public HID shared-memory ABI: mouse is 0x400 bytes, keyboard is 0x400.
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L647-L703
const MOUSE_OFFSET: usize = 0x3400;
const KEYBOARD_OFFSET: usize = 0x3800;
const MOUSE_ENTRY_SIZE: usize = 0x30;
const KEYBOARD_ENTRY_SIZE: usize = 0x38;
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
const NPAD_JOY_ASSIGNMENT_DUAL: u32 = 0;
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

/// System-owned controller state. Clones share configuration and one producer.
#[derive(Clone, Debug)]
pub struct HidSystem {
    state: Arc<HidState>,
    producer: Arc<Mutex<HidProducer>>,
}

#[derive(Debug)]
struct HidState {
    kernel: Arc<nixe_scheduler::SchedulerState>,
    shared_memory: SharedMemoryObject,
    infrared: crate::object::IrsSession,
    style_events: [OnceLock<(WritableEventObject, ReadableEventObject)>; 10],
    connected: AtomicBool,
    configuration: Mutex<HidConfiguration>,
    vibration_output: Mutex<Option<nixe_input::VibrationOutput>>,
    completion_events: Mutex<Option<nixe_runtime::ExternalEventSender>>,
}

impl Drop for HidState {
    fn drop(&mut self) {
        if let Some(output) = self.vibration_output.get_mut().unwrap()
            && let Err(error) = output.stop()
        {
            log::error!("cannot stop Horizon HID vibration: {error}");
        }
    }
}

#[derive(Debug)]
struct HidProducer {
    sampling_number: u64,
    touch_screen_sampling_number: u64,
    touch_screen: Lifo,
    mouse: Lifo,
    keyboard: Lifo,
    peripheral_sampling_number: u64,
    full_key: Lifo,
    six_axis: Lifo,
    home: Lifo,
    capture: Lifo,
    connected: bool,
    disconnected_by_guest: bool,
    publication: Publication,
    latest_controller: Option<(Option<EmulatedControllerState>, Duration)>,
    latest_touch: Option<(EmulatedTouchScreenState, Duration)>,
}

#[derive(Debug, Default)]
struct Publication {
    writes: VecDeque<(usize, Box<[u8]>, bool)>,
    pending: Option<PendingAccess>,
    connected: Option<bool>,
    signal_style: bool,
}
impl Publication {
    fn write(&mut self, offset: usize, bytes: &[u8]) {
        self.writes.push_back((offset, bytes.into(), false));
    }
    fn write_u8(&mut self, offset: usize, value: u8) {
        self.write(offset, &[value]);
    }
    fn write_u32(&mut self, offset: usize, value: u32) {
        self.write(offset, &value.to_le_bytes());
    }
    fn write_u64(&mut self, offset: usize, value: u64) {
        self.write(offset, &value.to_le_bytes());
    }

    // Entry writes precede their headers even when a device owns the bytes.
    // Only one batch is in flight; newer host input is coalesced before any
    // sampling number or LIFO position advances.
    fn poll(&mut self, state: &HidState) -> Result<bool, HidInputError> {
        if let Some(pending) = &mut self.pending {
            match pending.poll() {
                None => return Ok(false),
                Some(Err(error)) => return Err(HidInputError::Memory(error)),
                Some(Ok(AccessResult::Written)) => self.pending = None,
                Some(Ok(_)) => {
                    return Err(HidInputError::Memory(Arc::new(MemoryError::Execution(
                        "invalid HID write completion".into(),
                    ))));
                }
            }
        }
        while let Some((offset, bytes, release)) = self.writes.pop_front() {
            if release {
                std::sync::atomic::fence(Ordering::Release);
            }
            match state
                .shared_memory
                .write(offset, &bytes)
                .map_err(|e| HidInputError::Memory(Arc::new(e)))?
            {
                Access::Ready(()) => {}
                Access::Pending(mut access) => {
                    let events = state.completion_events.lock().unwrap().clone();
                    access.set_waker(Arc::new(move || {
                        if let Some(events) = &events {
                            events.notify_host_service(ExternalEventSource::Input);
                        }
                    }));
                    self.pending = Some(access);
                    return Ok(false);
                }
            }
        }
        if let Some(connected) = self.connected.take() {
            state.connected.store(connected, Ordering::Release);
            log::debug!("hid Npad connection changed: id=0x0 connected={connected}");
        }
        Ok(true)
    }
}

/// Failures from HID shared-memory publication or host actuator output.
#[derive(Debug)]
pub enum HidInputError {
    Memory(Arc<MemoryError>),
    Vibration(nixe_input::VibrationError),
}
impl std::fmt::Display for HidInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory(error) => error.fmt(f),
            Self::Vibration(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for HidInputError {}
impl From<nixe_input::VibrationError> for HidInputError {
    fn from(error: nixe_input::VibrationError) -> Self {
        Self::Vibration(error)
    }
}

#[derive(Debug, Default)]
struct HidConfiguration {
    touch_screen_active: bool,
    mouse_active: bool,
    keyboard_active: bool,
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
    fn publish(&mut self, writes: &mut Publication, offset: usize, entry: &[u8]) {
        self.tail = (self.tail + 1) % LIFO_CAPACITY;
        self.count = (self.count + 1).min(LIFO_CAPACITY);
        writes.write(offset + 0x20 + self.tail as usize * entry.len(), entry);
        let mut header = [0; 24];
        put_u64(&mut header, 0, LIFO_CAPACITY);
        put_u64(&mut header, 8, self.tail);
        put_u64(&mut header, 16, self.count);
        writes.writes.push_back((offset + 8, header.into(), true));
    }
}

impl HidSystem {
    pub fn kernel(&self) -> &Arc<nixe_scheduler::SchedulerState> {
        &self.state.kernel
    }
    pub fn new(
        account: &MemoryAccount,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
    ) -> Result<Self, Arc<MemoryError>> {
        let allocation = account
            .allocate(HID_SHARED_MEMORY_SIZE as u64)
            .map_err(Arc::new)?;
        let shared_memory =
            SharedMemoryObject::new(allocation, HID_SHARED_MEMORY_SIZE, MemoryPermissions::READ)
                .map_err(|e| Arc::new(MemoryError::Execution(e.to_string().into())))?;
        let infrared = crate::object::IrsSession::new(account)?;
        Ok(Self {
            state: Arc::new(HidState {
                kernel: kernel.clone(),
                shared_memory,
                infrared,
                style_events: std::array::from_fn(|_| OnceLock::new()),
                connected: AtomicBool::new(false),
                configuration: Mutex::new(HidConfiguration::default()),
                vibration_output: Mutex::new(None),
                completion_events: Mutex::new(None),
            }),
            producer: Arc::new(Mutex::new(HidProducer {
                sampling_number: 0,
                touch_screen_sampling_number: 0,
                touch_screen: Lifo::default(),
                mouse: Lifo::default(),
                keyboard: Lifo::default(),
                peripheral_sampling_number: 0,
                full_key: Lifo::default(),
                six_axis: Lifo::default(),
                home: Lifo::default(),
                capture: Lifo::default(),
                connected: false,
                disconnected_by_guest: false,
                publication: Publication::default(),
                latest_controller: None,
                latest_touch: None,
            })),
        })
    }
    pub(crate) fn set_completion_events(&self, events: nixe_runtime::ExternalEventSender) {
        *self.state.completion_events.lock().unwrap() = Some(events);
    }
    pub(crate) fn set_vibration_output(&self, output: nixe_input::VibrationOutput) {
        *self.state.vibration_output.lock().unwrap() = Some(output);
    }
    pub(crate) fn send_vibration_values(
        &self,
        values: &[(u32, nixe_input::VibrationValue)],
    ) -> Option<Result<(), nixe_input::VibrationError>> {
        let output = self.state.vibration_output.lock().unwrap().clone()?;
        let mut actuators = [None; 2];
        for &(handle, value) in values {
            let [style, npad_id, device_index, _] = handle.to_le_bytes();
            if self.state.connected.load(Ordering::Acquire) && style == 3 && npad_id == 0 {
                actuators[usize::from(device_index)] = Some(value);
            }
        }
        Some(if actuators.iter().all(Option::is_none) {
            Ok(())
        } else {
            output.send_values(actuators)
        })
    }
    pub(crate) fn connected_full_key_id(&self) -> Option<u32> {
        self.state.connected.load(Ordering::Acquire).then_some(0)
    }
    pub(crate) fn infrared_session(&self) -> crate::object::IrsSession {
        self.state.infrared.clone()
    }
    pub fn shared_memory(&self) -> SharedMemoryObject {
        self.state.shared_memory.clone()
    }

    /// Advances already accepted writes without consuming another host sample.
    pub fn poll_publication(&self) -> Result<(), HidInputError> {
        let signal = {
            let mut producer = self.producer.lock().unwrap();
            let mut signal = false;
            loop {
                if !producer.publication.poll(&self.state)? {
                    break;
                }
                signal |= std::mem::take(&mut producer.publication.signal_style);
                if let Some((sample, delta)) = producer.latest_controller.take() {
                    let mut publication = Publication::default();
                    producer.prepare_controller(
                        &self.state,
                        &mut publication,
                        sample.as_ref(),
                        delta,
                    )?;
                    if let Some((touch, delta)) = producer.latest_touch.take() {
                        producer.prepare_touch(&self.state, &mut publication, &touch, delta);
                    }
                    producer.publication = publication;
                } else if let Some((sample, delta)) = producer.latest_touch.take() {
                    let mut publication = Publication::default();
                    producer.prepare_touch(&self.state, &mut publication, &sample, delta);
                    producer.publication = publication;
                } else {
                    break;
                }
            }
            signal
        };
        if signal {
            self.signal_player_one_style_change();
        }
        Ok(())
    }
    pub fn publish(
        &self,
        state: Option<&EmulatedControllerState>,
        delta: Duration,
    ) -> Result<(), HidInputError> {
        {
            let mut producer = self.producer.lock().unwrap();
            // A software disconnect lasts until a host disconnect/reconnect;
            // ordinary input samples must not immediately reconnect the pad.
            if state.is_none() {
                producer.disconnected_by_guest = false;
            }
            producer.latest_controller = Some((state.copied(), delta));
        }
        self.poll_publication()
    }
    pub fn publish_touch_screen(
        &self,
        state: &EmulatedTouchScreenState,
        delta: Duration,
    ) -> Result<(), HidInputError> {
        self.producer.lock().unwrap().latest_touch = Some((*state, delta));
        self.poll_publication()
    }

    pub(crate) fn acquire_style_event(&self, id: u32) -> Option<ReadableEventObject> {
        let index = npad_index(id)?;
        let (writable, readable) = self.state.style_events[index]
            .get_or_init(|| EventObject::create_pair(&self.state.kernel));
        // Acquiring also signals the event so a client can inspect the initial
        // style. Later signals follow publication of a changed style tag.
        // https://github.com/eden-emulator/mirror/blob/master/src/hid_core/resources/npad/npad_resource.cpp
        writable.signal();
        Some(readable.clone())
    }

    fn signal_player_one_style_change(&self) {
        if let Some((event, _)) = self.state.style_events[0].get() {
            event.signal();
        }
    }

    pub(crate) fn disconnect_npad(&self, id: u32) -> Result<bool, HidInputError> {
        let Some(index) = npad_index(id) else {
            return Ok(false);
        };
        // Disconnect the modeled host device and publish through the existing
        // producer, so an in-flight memory write cannot restore its old style.
        // Other slots have no attached device; their style event still fires.
        // https://github.com/eden-emulator/mirror/blob/master/src/hid_core/resources/npad/npad.cpp
        if index == 0 {
            {
                let mut producer = self.producer.lock().unwrap();
                producer.disconnected_by_guest = true;
                producer.latest_controller = Some((None, Duration::ZERO));
            }
            self.poll_publication()?;
            if self.producer.lock().unwrap().connected
                || self.state.connected.load(Ordering::Acquire)
            {
                // The producer signals after pending shared-memory writes finish.
                return Ok(true);
            }
        }
        if let Some((event, _)) = self.state.style_events[index].get() {
            event.signal();
        }
        Ok(true)
    }

    pub(crate) fn activate_npad(&self) {
        let mut configuration = self
            .state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        configuration.npad_active = true;
    }

    pub(crate) fn activate_touch_screen(&self) {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .touch_screen_active = true;
    }

    pub(crate) fn activate_mouse(&self) {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .mouse_active = true;
    }

    pub(crate) fn activate_keyboard(&self) {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keyboard_active = true;
    }

    pub(crate) fn set_supported_npad_style_set(&self, style_set: u32) {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .supported_style_set = style_set;
    }

    pub(crate) fn supported_npad_style_set(&self) -> u32 {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .supported_style_set
    }

    pub(crate) fn set_supported_npad_ids(&self, ids: impl IntoIterator<Item = u32>) -> bool {
        let ids = ids.into_iter().collect::<BTreeSet<_>>();
        if ids.iter().any(|id| !matches!(*id, 0..=7 | 0x10 | 0x20)) {
            return false;
        }
        self.state
            .configuration
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
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .npad_joy_hold_type = hold_type;
        true
    }

    pub(crate) fn npad_joy_hold_type(&self) -> u64 {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .npad_joy_hold_type
    }

    pub(crate) async fn set_npad_joy_assignment_mode_dual(
        &self,
        id: u32,
    ) -> Result<bool, Arc<MemoryError>> {
        let Some(index) = npad_index(id) else {
            return Ok(false);
        };
        crate::system_memory::write(
            &self.state.shared_memory,
            NPAD_OFFSET + index * NPAD_ENTRY_SIZE + 4,
            &NPAD_JOY_ASSIGNMENT_DUAL.to_le_bytes(),
        )
        .await?;
        Ok(true)
    }

    pub(crate) fn set_six_axis_sensor_active(&self, handle: u32, active: bool) {
        let mut configuration = self
            .state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if active {
            configuration.active_six_axis_handles.insert(handle);
        } else {
            configuration.active_six_axis_handles.remove(&handle);
        }
    }
}

impl HidProducer {
    /// Publishes one complete touch-screen sample.
    fn prepare_touch(
        &mut self,
        system: &HidState,
        writes: &mut Publication,
        state: &EmulatedTouchScreenState,
        delta: Duration,
    ) {
        if !system
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .touch_screen_active
        {
            return;
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
        self.touch_screen
            .publish(writes, TOUCH_SCREEN_OFFSET, &entry)
    }

    /// Publishes one player-one Pro Controller sample.
    ///
    /// `None` transitions the shared state to a disconnected NPad. Repeated
    /// disconnected updates are ignored.
    fn prepare_controller(
        &mut self,
        system: &HidState,
        writes: &mut Publication,
        state: Option<&EmulatedControllerState>,
        delta: Duration,
    ) -> Result<(), HidInputError> {
        let (publish_player_one, publish_six_axis, mouse_active, keyboard_active) = {
            let configuration = system
                .configuration
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                configuration.npad_active
                    && !self.disconnected_by_guest
                    && configuration.supported_style_set & NPAD_STYLE_FULL_KEY != 0
                    && configuration.supported_ids.contains(&0),
                // FullKey, Player 1, device index 2. The packed handle layout
                // is pinned in the public libnx HidSixAxisSensorHandle ABI:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L1412-L1421
                configuration.active_six_axis_handles.contains(&0x0002_0003),
                configuration.mouse_active,
                configuration.keyboard_active,
            )
        };
        // Host keys are controller bindings and the mouse is a touch contact;
        // neither is exposed as a guest USB peripheral. Activated rings still
        // publish idle samples, with the mouse connection attribute cleared.
        if mouse_active || keyboard_active {
            self.peripheral_sampling_number = self.peripheral_sampling_number.wrapping_add(1);
            if mouse_active {
                let mut entry = [0; MOUSE_ENTRY_SIZE];
                put_u64(&mut entry, 0, self.peripheral_sampling_number);
                put_u64(&mut entry, 8, self.peripheral_sampling_number);
                self.mouse.publish(writes, MOUSE_OFFSET, &entry);
            }
            if keyboard_active {
                let mut entry = [0; KEYBOARD_ENTRY_SIZE];
                put_u64(&mut entry, 0, self.peripheral_sampling_number);
                put_u64(&mut entry, 8, self.peripheral_sampling_number);
                self.keyboard.publish(writes, KEYBOARD_OFFSET, &entry);
            }
        }
        let Some(state) = state.filter(|_| publish_player_one) else {
            if self.connected {
                if let Some(output) = system.vibration_output.lock().unwrap().clone() {
                    output.stop()?;
                }
                self.sampling_number = self.sampling_number.saturating_add(1);
                // Assignment mode persists across disconnection.
                writes.write_u32(NPAD_OFFSET, 0);
                writes.write(NPAD_OFFSET + 8, &[0; NPAD_ENTRY_SIZE - 8]);
                self.full_key = Lifo::default();
                self.six_axis = Lifo::default();
                self.publish_system_button(writes, HOME_BUTTON_LIFO_OFFSET, false, true);
                self.publish_system_button(writes, CAPTURE_BUTTON_LIFO_OFFSET, false, false);
                self.connected = false;
                writes.signal_style = true;
                writes.connected = Some(self.connected);
            }
            return Ok(());
        };

        self.sampling_number = self.sampling_number.saturating_add(1);
        let style_changed = !self.connected;
        if style_changed {
            writes.write_u32(NPAD_OFFSET, NPAD_STYLE_FULL_KEY);
            writes.write_u32(NPAD_OFFSET + 4, NPAD_JOY_ASSIGNMENT_DUAL);
            writes.write_u32(NPAD_OFFSET + 8, 0);
            writes.write_u32(NPAD_OFFSET + 0x4188, NPAD_DEVICE_TYPE_FULL_KEY);
            writes.write_u64(
                NPAD_OFFSET + 0x4190,
                1 << 3 | 1 << 11 | 1 << 13 | 1 << 14 | 1 << 15,
            );
            writes.write_u32(NPAD_OFFSET + 0x419c, 4);
            writes.write_u8(NPAD_OFFSET + 0x41ac, APPLET_FOOTER_SWITCH_PRO_CONTROLLER);
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
        self.full_key
            .publish(writes, NPAD_OFFSET + FULL_KEY_LIFO_OFFSET, &common);

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
            self.six_axis
                .publish(writes, NPAD_OFFSET + FULL_KEY_SIX_AXIS_LIFO_OFFSET, &sensor);
        }

        self.publish_system_button(writes, HOME_BUTTON_LIFO_OFFSET, state.buttons.home, true);
        self.publish_system_button(
            writes,
            CAPTURE_BUTTON_LIFO_OFFSET,
            state.buttons.capture,
            false,
        );
        if style_changed {
            writes.signal_style = true;
            writes.connected = Some(self.connected);
        }
        Ok(())
    }

    fn publish_system_button(
        &mut self,
        writes: &mut Publication,
        lifo_offset: usize,
        pressed: bool,
        home: bool,
    ) {
        let lifo = if home {
            &mut self.home
        } else {
            &mut self.capture
        };
        let mut entry = [0_u8; SYSTEM_BUTTON_ENTRY_SIZE];
        put_u64(&mut entry, 0, self.sampling_number);
        put_u64(&mut entry, 8, self.sampling_number);
        put_u64(&mut entry, 16, u64::from(pressed));
        lifo.publish(writes, lifo_offset, &entry)
    }
}

fn put_turns(bytes: &mut [u8], offset: usize, radians: nixe_input::MotionVector) {
    put_f32(bytes, offset, radians.x / std::f32::consts::TAU);
    put_f32(bytes, offset + 4, -radians.z / std::f32::consts::TAU);
    put_f32(bytes, offset + 8, radians.y / std::f32::consts::TAU);
}

fn npad_index(id: u32) -> Option<usize> {
    match id {
        0..=7 => Some(id as usize),
        0x10 => Some(8),
        0x20 => Some(9),
        _ => None,
    }
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

    fn hid() -> HidSystem {
        let machine = nixe_memory::Machine::new(2 * 1024 * 1024).unwrap();
        HidSystem::new(
            &machine.account(machine.arena().capacity()).unwrap(),
            &crate::test_memory::kernel(),
        )
        .unwrap()
    }
    fn read(memory: &SharedMemoryObject, offset: usize, destination: &mut [u8]) {
        let nixe_memory::Access::Ready(bytes) = memory.read(offset, destination.len()).unwrap()
        else {
            panic!("private HID fixture unexpectedly requires device access")
        };
        destination.copy_from_slice(&bytes);
    }
    fn ready<F: std::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let std::task::Poll::Ready(value) = future
            .as_mut()
            .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
        else {
            panic!("private HID fixture unexpectedly parked")
        };
        value
    }

    #[test]
    fn peripheral_activation_publishes_idle_samples_only_after_activation() {
        let hid = hid();
        let shared = hid.shared_memory();
        hid.publish(None, Duration::ZERO).unwrap();
        let mut before = [0; 0x800];
        read(&shared, MOUSE_OFFSET, &mut before);
        assert_eq!(before, [0; 0x800]);
        hid.activate_mouse();
        hid.activate_keyboard();
        for _ in 0..20 {
            hid.publish(None, Duration::ZERO).unwrap();
        }
        for (offset, size) in [
            (MOUSE_OFFSET, MOUSE_ENTRY_SIZE),
            (KEYBOARD_OFFSET, KEYBOARD_ENTRY_SIZE),
        ] {
            let mut header = [0; 32];
            read(&shared, offset, &mut header);
            assert_eq!(u64::from_le_bytes(header[8..16].try_into().unwrap()), 17);
            assert_eq!(u64::from_le_bytes(header[16..24].try_into().unwrap()), 2);
            assert_eq!(u64::from_le_bytes(header[24..32].try_into().unwrap()), 17);
            let mut entry = vec![0; size];
            read(&shared, offset + 32 + 2 * size, &mut entry);
            assert_eq!(u64::from_le_bytes(entry[..8].try_into().unwrap()), 20);
            assert_eq!(u64::from_le_bytes(entry[8..16].try_into().unwrap()), 20);
            assert!(entry[16..].iter().all(|value| *value == 0));
        }
    }

    #[test]
    fn publication_updates_the_mapped_pages_without_synchronization() {
        use nixe_memory::{
            Access, AddressSpace, AddressWidth, GuestVirtualAddress, Machine, MemoryPermissions,
        };
        let machine = Machine::new(2 * 1024 * 1024).unwrap();
        let account = machine.account(machine.arena().capacity()).unwrap();
        let hid = HidSystem::new(&account, &crate::test_memory::kernel()).unwrap();
        let shared = hid.shared_memory();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        let address = GuestVirtualAddress::new(0x1000);
        space
            .map(
                address,
                shared.allocation().clone(),
                0,
                HID_SHARED_MEMORY_SIZE as u64,
                MemoryPermissions::READ,
            )
            .unwrap();
        configure_player_one(&hid, false);
        hid.publish(Some(&EmulatedControllerState::default()), Duration::ZERO)
            .unwrap();
        let mut style = [0; 4];
        assert!(matches!(
            space
                .read_into(address.checked_add(NPAD_OFFSET as u64).unwrap(), &mut style)
                .unwrap(),
            Access::Ready(())
        ));
        assert_eq!(u32::from_le_bytes(style), NPAD_STYLE_FULL_KEY);
        assert!(matches!(
            space.write(address, &[1]),
            Err(nixe_memory::MemoryError::PermissionDenied { .. })
        ));
    }

    fn read_u32(memory: &SharedMemoryObject, offset: usize) -> u32 {
        let mut bytes = [0; 4];
        read(memory, offset, &mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn read_u64(memory: &SharedMemoryObject, offset: usize) -> u64 {
        let mut bytes = [0; 8];
        read(memory, offset, &mut bytes);
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
    fn style_event_copies_track_published_connection_changes() {
        let hid = hid();
        let memory = hid.shared_memory();
        let event = hid.acquire_style_event(0).unwrap();
        assert!(event.is_signalled());
        let copy = hid.acquire_style_event(0).unwrap();
        copy.clear();
        assert!(!event.is_signalled());
        configure_player_one(&hid, false);
        let state = EmulatedControllerState::default();
        hid.publish(Some(&state), Duration::ZERO).unwrap();
        assert!(event.is_signalled());
        assert_eq!(read_u32(&memory, NPAD_OFFSET), NPAD_STYLE_FULL_KEY);
        event.clear();
        hid.publish(Some(&state), Duration::ZERO).unwrap();
        assert!(!event.is_signalled());
        hid.publish(None, Duration::ZERO).unwrap();
        assert!(copy.is_signalled());
        assert_eq!(read_u32(&memory, NPAD_OFFSET), 0);
        copy.clear();
        hid.publish(None, Duration::ZERO).unwrap();
        assert!(!event.is_signalled());
        let other = hid.acquire_style_event(1).unwrap();
        assert!(other.is_signalled());
        other.clear();
        hid.publish(Some(&state), Duration::ZERO).unwrap();
        assert!(!other.is_signalled());
        for id in [8, 9, 0x11, 0x21, u32::MAX] {
            assert!(hid.acquire_style_event(id).is_none());
        }
    }

    #[test]
    fn publishes_atomic_multitouch_samples_after_activation() {
        let hid = hid();
        let memory = hid.shared_memory();
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
        let hid = hid();
        configure_player_one(&hid, false);
        // Joy-Con orientation must not rotate a FullKey controller's input.
        assert!(hid.set_npad_joy_hold_type(1));
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

        assert!(ready(hid.set_npad_joy_assignment_mode_dual(0)).unwrap());
        assert_eq!(read_u32(&memory, NPAD_OFFSET + 4), NPAD_JOY_ASSIGNMENT_DUAL);
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
    fn guest_disconnect_notifies_the_target_and_waits_for_host_reconnection() {
        let hid = hid();
        configure_player_one(&hid, false);
        let memory = hid.shared_memory();
        let state = EmulatedControllerState::default();
        hid.publish(Some(&state), Duration::ZERO).unwrap();
        let event = hid.acquire_style_event(0).unwrap();
        let other = hid.acquire_style_event(1).unwrap();
        event.clear();
        other.clear();
        assert!(hid.disconnect_npad(1).unwrap());
        assert!(other.is_signalled());
        assert!(!event.is_signalled());
        assert_eq!(hid.connected_full_key_id(), Some(0));

        // Assignment is persistent state, independent of device connection.
        memory
            .write(NPAD_OFFSET + 4, &1_u32.to_le_bytes())
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(hid.disconnect_npad(0).unwrap());
        assert!(event.is_signalled());
        assert_eq!(hid.connected_full_key_id(), None);
        assert_eq!(read_u32(&memory, NPAD_OFFSET), 0);
        assert_eq!(read_u32(&memory, NPAD_OFFSET + 4), 1);
        event.clear();
        hid.publish(Some(&state), Duration::ZERO).unwrap();
        assert!(!event.is_signalled());
        assert_eq!(read_u32(&memory, NPAD_OFFSET), 0);
        hid.publish(None, Duration::ZERO).unwrap();
        hid.publish(Some(&state), Duration::ZERO).unwrap();
        assert_eq!(hid.connected_full_key_id(), Some(0));
        assert!(event.is_signalled());
        for id in [8, 9, 0x11, u32::MAX] {
            assert!(!hid.disconnect_npad(id).unwrap());
        }
    }

    #[test]
    fn publishes_motion_and_system_buttons() {
        let hid = hid();
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
        let hid = hid();
        let memory = hid.shared_memory();
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
        let hid = hid();
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
