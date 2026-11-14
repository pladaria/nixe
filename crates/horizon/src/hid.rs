use std::collections::{BTreeSet, VecDeque};
use std::sync::{
    Arc, Mutex, OnceLock, Weak,
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

mod console;
use console::SevenSixAxis;
mod jcms;

const HID_SHARED_MEMORY_SIZE: usize = 0x40000;
const TOUCH_SCREEN_OFFSET: usize = 0x400;
const DEBUG_PAD_ENTRY_SIZE: usize = 0x28;
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
const SYSTEM_EXT_LIFO_OFFSET: usize = 0x1408;
const FULL_KEY_SIX_AXIS_LIFO_OFFSET: usize = 0x1758;
const HOME_BUTTON_LIFO_OFFSET: usize = 0x4c00;
const CAPTURE_BUTTON_LIFO_OFFSET: usize = 0x5000;
// ConsoleSixAxisSensorSharedMemoryFormat follows the gesture region.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/shared_memory_format.h
const CONSOLE_SIX_AXIS_OFFSET: usize = 0x3c200;
// @SPECULATIVE: this global NpadCondition layout is not publicly documented.
// The native SDK reads its initialized/hold/valid fields at +4/+8/+12;
// retain the configured hold mode instead of a disconnected/default response.
// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/hid_core/resources/shared_memory_format.h
const NPAD_CONDITION_OFFSET: usize = 0x3e200;

const LIFO_CAPACITY: u64 = 17;
const COMMON_ENTRY_SIZE: usize = 0x30;
const SIX_AXIS_ENTRY_SIZE: usize = 0x68;
const SYSTEM_BUTTON_ENTRY_SIZE: usize = 0x18;

const NPAD_STYLE_FULL_KEY: u32 = 1;
const NPAD_DEVICE_TYPE_FULL_KEY: u32 = 1;
const NPAD_ATTRIBUTE_CONNECTED: u32 = 1;
const NPAD_ATTRIBUTE_WIRED: u32 = 1 << 1;
const NPAD_JOY_ASSIGNMENT_DUAL: u32 = 0;
const SIX_AXIS_ATTRIBUTE_CONNECTED: u32 = 1;
const APPLET_FOOTER_SWITCH_PRO_CONTROLLER: u8 = 12;

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
    nfc_connected: AtomicBool,
    nfc_readers: Mutex<Vec<Weak<crate::nfp::NfpReader>>>,
    configuration: Mutex<HidConfiguration>,
    vibration_output: Mutex<Option<nixe_input::VibrationOutput>>,
    completion_events: Mutex<Option<nixe_runtime::ExternalEventSender>>,
    motion_settings: OnceLock<nixe_input::MotionSettings>,
    jcms: Mutex<Option<Arc<nixe_runtime::TransferMemoryObject>>>,
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
    six_axis_at_rest: [bool; 10],
    console_six_axis_sampling_number: u64,
    console_motion: nixe_input::MotionInput,
    seven_six_axis: SevenSixAxis,
    touch_screen_sampling_number: u64,
    debug_pad_sampling_number: u64,
    debug_pad: Lifo,
    touch_screen: Lifo,
    mouse: Lifo,
    keyboard: Lifo,
    peripheral_sampling_number: u64,
    full_key: Lifo,
    system_ext: Lifo,
    six_axis: Lifo,
    home: Lifo,
    capture: Lifo,
    connected: bool,
    // The configured virtual controller is independent of its host input source.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/frontend/emulated_controller.cpp
    full_key_connected: bool,
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
                Some(Ok(AccessResult::Written(_))) => self.pending = None,
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
                    state.watch_publication(&mut access);
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

impl HidState {
    fn watch_publication(&self, access: &mut PendingAccess) {
        let events = self.completion_events.lock().unwrap().clone();
        access.set_waker(Arc::new(move || {
            if let Some(events) = &events {
                events.notify_host_service(ExternalEventSource::Input);
            }
        }));
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
    debug_pad_active: bool,
    touch_screen_active: bool,
    touch_screen_output_ranges: Option<[u32; 2]>,
    mouse_active: bool,
    keyboard_active: bool,
    console_six_axis_active: bool,
    npad_active: bool,
    supported_style_set: u32,
    supported_ids: BTreeSet<u32>,
    npad_joy_hold_type: u64,
    npad_handheld_activation_mode: u64,
    npad_lr_assignment_mode: bool,
    active_six_axis_handles: BTreeSet<u32>,
    gyroscope_drift_modes: std::collections::BTreeMap<(u8, u8, u8), u32>,
}

#[derive(Debug, Default)]
struct Lifo {
    tail: u64,
    count: u64,
}

impl Lifo {
    fn publish(&mut self, writes: &mut Publication, offset: usize, entry: &[u8]) {
        // Eden's ring reserves one slot and advances from the zeroed tail.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/ring_lifo.h
        self.tail = (self.tail + 1) % LIFO_CAPACITY;
        self.count = (self.count + 1).min(LIFO_CAPACITY - 1);
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
        let mut condition = [0; 16];
        put_u32(&mut condition, 4, 1);
        put_u32(&mut condition, 12, 1);
        // Fresh shared memory has no guest mappings or device owners yet.
        if !matches!(
            shared_memory
                .write(NPAD_CONDITION_OFFSET, &condition)
                .map_err(Arc::new)?,
            Access::Ready(())
        ) {
            return Err(Arc::new(MemoryError::Execution(
                "fresh HID shared memory unexpectedly has a device owner".into(),
            )));
        }
        let infrared = crate::object::IrsSession::new(account)?;
        Ok(Self {
            state: Arc::new(HidState {
                kernel: kernel.clone(),
                shared_memory,
                infrared,
                style_events: std::array::from_fn(|_| OnceLock::new()),
                connected: AtomicBool::new(false),
                nfc_connected: AtomicBool::new(false),
                nfc_readers: Mutex::new(Vec::new()),
                configuration: Mutex::new(HidConfiguration::default()),
                vibration_output: Mutex::new(None),
                completion_events: Mutex::new(None),
                motion_settings: OnceLock::new(),
                jcms: Mutex::new(None),
            }),
            producer: Arc::new(Mutex::new(HidProducer {
                sampling_number: 0,
                six_axis_at_rest: [true; 10],
                console_six_axis_sampling_number: 0,
                console_motion: nixe_input::MotionInput::default(),
                seven_six_axis: SevenSixAxis::default(),
                touch_screen_sampling_number: 0,
                debug_pad_sampling_number: 0,
                debug_pad: Lifo::default(),
                touch_screen: Lifo::default(),
                mouse: Lifo::default(),
                keyboard: Lifo::default(),
                peripheral_sampling_number: 0,
                full_key: Lifo::default(),
                system_ext: Lifo::default(),
                six_axis: Lifo::default(),
                home: Lifo::default(),
                capture: Lifo::default(),
                connected: false,
                full_key_connected: true,
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

    pub(crate) fn set_motion_settings(&self, settings: nixe_input::MotionSettings) {
        self.state
            .motion_settings
            .set(settings)
            .expect("motion settings are supplied once before execution");
    }

    /// Store per-sensor state and change the emulated controller's real filter,
    /// like Eden SixAxis::SetGyroscopeZeroDriftMode and EmulatedController.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/six_axis/six_axis.cpp
    pub(crate) fn set_gyroscope_drift_mode(
        &self,
        handle: u32,
        mode: u32,
    ) -> Result<(), crate::HorizonIpcResult> {
        let key = six_axis_key(handle)?;
        self.state
            .configuration
            .lock()
            .unwrap()
            .gyroscope_drift_modes
            .insert(key, mode);
        if key.0 == 0
            && let Some(settings) = self.state.motion_settings.get()
        {
            let (gyroscope, rest) = match mode {
                0 => (0.01, 0.02),
                2 => (0.002, 0.005),
                _ => (0.007, 0.01),
            };
            settings.set_thresholds(gyroscope, rest);
        }
        Ok(())
    }

    pub(crate) fn gyroscope_drift_mode(&self, handle: u32) -> Result<u32, crate::HorizonIpcResult> {
        let key = six_axis_key(handle)?;
        Ok(*self
            .state
            .configuration
            .lock()
            .unwrap()
            .gyroscope_drift_modes
            .get(&key)
            .unwrap_or(&1))
    }
    pub(crate) fn six_axis_sensor_at_rest(
        &self,
        handle: u32,
    ) -> Result<bool, crate::HorizonIpcResult> {
        let (npad, _, _) = six_axis_key(handle)?;
        let index = npad_index(u32::from(npad)).expect("validated six-axis Npad ID");
        Ok(self.producer.lock().unwrap().six_axis_at_rest[index])
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
    /// Eden's default controller applet retains valid virtual players and fills
    /// the requested slots independently of attached physical input devices.
    /// Nixe currently models one FullKey player; the caller checks that range.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/frontend/applets/controller.cpp
    pub(crate) fn configure_full_key_controller(
        &self,
        keep_connected: bool,
    ) -> Result<u32, HidInputError> {
        if !keep_connected {
            self.disconnect_npad(0)?;
        }
        {
            let mut producer = self.producer.lock().unwrap();
            if !producer.full_key_connected {
                producer.full_key_connected = true;
                producer
                    .latest_controller
                    .get_or_insert((None, Duration::ZERO));
            }
        }
        self.update_nfc_connection(true);
        self.poll_publication()?;
        Ok(0)
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
                if !producer.publication.poll(&self.state)? || !producer.seven_six_axis.poll()? {
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
        let connected = {
            let mut producer = self.producer.lock().unwrap();
            producer.latest_controller = Some((state.copied(), delta));
            producer.full_key_connected
        };
        self.update_nfc_connection(connected);
        self.poll_publication()
    }

    pub(crate) fn create_nfc_reader(&self) -> Arc<crate::nfp::NfpReader> {
        let mut readers = self.state.nfc_readers.lock().unwrap();
        readers.retain(|reader| reader.strong_count() != 0);
        let reader = Arc::new(crate::nfp::NfpReader::new(
            &self.state.kernel,
            self.state.nfc_connected.load(Ordering::Acquire),
        ));
        readers.push(Arc::downgrade(&reader));
        reader
    }

    fn update_nfc_connection(&self, connected: bool) {
        if self.state.nfc_connected.swap(connected, Ordering::AcqRel) == connected {
            return;
        }
        // As Eden's NfcDevice::NpadUpdate, notify each registered reader
        // directly when the underlying controller connects or disconnects.
        // Npad style filtering does not change the underlying NFC capability.
        let mut readers = self.state.nfc_readers.lock().unwrap();
        readers.retain(|reader| {
            let Some(reader) = reader.upgrade() else {
                return false;
            };
            reader.connection_changed(connected);
            true
        });
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
        // Disconnect the virtual controller and publish through the existing
        // producer, so an in-flight memory write cannot restore its old style.
        // Other slots have no attached device; their style event still fires.
        // https://github.com/eden-emulator/mirror/blob/master/src/hid_core/resources/npad/npad.cpp
        if index == 0 {
            {
                let mut producer = self.producer.lock().unwrap();
                producer.full_key_connected = false;
                producer.latest_controller = Some((None, Duration::ZERO));
            }
            self.update_nfc_connection(false);
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

    pub(crate) fn set_touch_screen_output_ranges(&self, width: u32, height: u32) {
        self.state
            .configuration
            .lock()
            .unwrap()
            .touch_screen_output_ranges = Some([width, height]);
        log::debug!("hid touch-screen output ranges: {width}x{height}");
    }

    pub(crate) fn activate_debug_pad(&self) {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .debug_pad_active = true;
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

    pub(crate) fn activate_console_six_axis_sensor(&self) {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .console_six_axis_active = true;
    }

    pub(crate) fn activate_seven_six_axis_sensor(&self) {
        self.producer.lock().unwrap().seven_six_axis.active = true;
    }

    pub(crate) fn initialize_seven_six_axis_sensor(
        &self,
        memory: Arc<nixe_runtime::TransferMemoryObject>,
    ) {
        self.activate_console_six_axis_sensor();
        let previous = {
            let mut producer = self.producer.lock().unwrap();
            producer.seven_six_axis.active = true;
            producer.seven_six_axis.memory.replace(memory)
        };
        drop(previous);
    }

    pub(crate) fn reset_seven_six_axis_sensor_timestamp(&self) {
        let mut producer = self.producer.lock().unwrap();
        producer.seven_six_axis.saved_timestamp = producer.seven_six_axis.global_timestamp;
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

    pub(crate) async fn set_npad_joy_hold_type(
        &self,
        hold_type: u64,
    ) -> Result<bool, Arc<MemoryError>> {
        // Vertical (0) is the default; horizontal (1) affects single Joy-Con
        // orientation. FullKey publication does not consume this setting.
        // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h
        if !matches!(hold_type, 0 | 1) {
            return Ok(false);
        }
        // The SDK may read the mode directly from shared memory after this
        // IPC completes. Finish its coherent write before returning success.
        if let Access::Pending(pending) = self
            .state
            .shared_memory
            .write(NPAD_CONDITION_OFFSET + 8, &(hold_type as u32).to_le_bytes())
            .map_err(Arc::new)?
            && !matches!(pending.await?, AccessResult::Written(_))
        {
            return Err(Arc::new(MemoryError::Execution(
                "invalid Npad condition write completion".into(),
            )));
        }
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .npad_joy_hold_type = hold_type;
        Ok(true)
    }

    pub(crate) fn npad_joy_hold_type(&self) -> u64 {
        self.state
            .configuration
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .npad_joy_hold_type
    }

    // Dual = 0, Single = 1, None = 2. Like NpadResource, retain the
    // application's setting independently of its connected FullKey controller.
    // https://switchbrew.org/wiki/HID_services#NpadHandheldActivationMode
    pub(crate) fn set_npad_handheld_activation_mode(&self, mode: u64) -> bool {
        if mode > 2 {
            return false;
        }
        self.state
            .configuration
            .lock()
            .unwrap()
            .npad_handheld_activation_mode = mode;
        true
    }

    pub(crate) fn npad_handheld_activation_mode(&self) -> u64 {
        self.state
            .configuration
            .lock()
            .unwrap()
            .npad_handheld_activation_mode
    }

    /// Eden stores the LR assignment mode in the caller's Npad resource;
    /// it does not change an already assigned Pro Controller's input stream.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/npad/npad.cpp
    pub(crate) fn set_npad_lr_assignment_mode(&self, enabled: bool) {
        self.state
            .configuration
            .lock()
            .unwrap()
            .npad_lr_assignment_mode = enabled;
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
        let configuration = system.configuration.lock().unwrap();
        if !configuration.touch_screen_active {
            return;
        }
        let [width, height] = configuration.touch_screen_output_ranges.unwrap_or([
            nixe_input::TOUCH_SCREEN_WIDTH,
            nixe_input::TOUCH_SCREEN_HEIGHT,
        ]);
        drop(configuration);
        self.touch_screen_sampling_number = self.touch_screen_sampling_number.saturating_add(1);
        let mut entry = [0_u8; TOUCH_SCREEN_ENTRY_SIZE];
        put_u64(
            &mut entry,
            0,
            self.touch_screen_sampling_number.wrapping_shl(1),
        );
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
            // Host input is expressed in the canonical touch coordinate space.
            // Convert only when publishing to the application's requested range.
            // https://switchbrew.org/wiki/HID_services#SetTouchScreenOutputRanges
            let scale = |value, range, source| {
                (u64::from(value) * u64::from(range) / u64::from(source)) as u32
            };
            put_u32(
                &mut entry,
                offset + 16,
                scale(
                    contact.x.min(nixe_input::TOUCH_SCREEN_WIDTH - 1),
                    width,
                    nixe_input::TOUCH_SCREEN_WIDTH,
                ),
            );
            put_u32(
                &mut entry,
                offset + 20,
                scale(
                    contact.y.min(nixe_input::TOUCH_SCREEN_HEIGHT - 1),
                    height,
                    nixe_input::TOUCH_SCREEN_HEIGHT,
                ),
            );
            put_u32(
                &mut entry,
                offset + 24,
                scale(contact.diameter_x, width, nixe_input::TOUCH_SCREEN_WIDTH),
            );
            put_u32(
                &mut entry,
                offset + 28,
                scale(contact.diameter_y, height, nixe_input::TOUCH_SCREEN_HEIGHT),
            );
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
    /// A missing host source supplies neutral input to the configured virtual
    /// controller. Only controller configuration or HID requests disconnect it.
    fn prepare_controller(
        &mut self,
        system: &HidState,
        writes: &mut Publication,
        state: Option<&EmulatedControllerState>,
        delta: Duration,
    ) -> Result<(), HidInputError> {
        let (
            publish_player_one,
            publish_six_axis,
            mouse_active,
            keyboard_active,
            console_six_axis_active,
            debug_pad_active,
        ) = {
            let configuration = system
                .configuration
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                configuration.npad_active
                    && self.full_key_connected
                    && configuration.supported_style_set & NPAD_STYLE_FULL_KEY != 0
                    && configuration.supported_ids.contains(&0),
                // FullKey, Player 1, device index 2. The packed handle layout
                // is pinned in the public libnx HidSixAxisSensorHandle ABI:
                // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/hid.h#L1412-L1421
                configuration.active_six_axis_handles.contains(&0x0002_0003),
                configuration.mouse_active,
                configuration.keyboard_active,
                configuration.console_six_axis_active,
                configuration.debug_pad_active,
            )
        };
        if debug_pad_active {
            // A debug pad is a separate device from the ordinary Npad. As in
            // Eden's default desktop configuration it publishes disconnected
            // samples, with no player-one buttons routed into this device.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/debug_pad/debug_pad.cpp
            self.debug_pad_sampling_number = self.debug_pad_sampling_number.wrapping_add(1);
            let mut entry = [0; DEBUG_PAD_ENTRY_SIZE];
            put_u64(
                &mut entry,
                0,
                self.debug_pad_sampling_number.wrapping_shl(1),
            );
            put_u64(&mut entry, 8, self.debug_pad_sampling_number);
            self.debug_pad.publish(writes, 0, &entry);
        }
        if console_six_axis_active || self.seven_six_axis.active {
            self.console_motion.update(
                state.and_then(|state| state.gyroscope),
                state.and_then(|state| state.accelerometer),
                delta,
                0.0,
                0.01,
            );
        }
        if console_six_axis_active {
            // Console motion uses player one's source independently of Npad
            // activation. Verticalization error is zero in Eden as well.
            // https://switchbrew.org/wiki/HID_services#ConsoleSixAxisSensorSharedMemoryFormat
            self.console_six_axis_sampling_number =
                self.console_six_axis_sampling_number.wrapping_add(1);
            let mut sample = [0; 0x20];
            put_u64(&mut sample, 0, self.console_six_axis_sampling_number);
            sample[8] = u8::from(!self.console_motion.moving(0.01));
            for (i, bias) in self.console_motion.bias.into_iter().enumerate() {
                put_f32(&mut sample, 0x10 + 4 * i, bias);
            }
            writes.write(CONSOLE_SIX_AXIS_OFFSET, &sample);
        }
        self.seven_six_axis.publish(system, &self.console_motion)?;
        // Host keys are controller bindings and the mouse is a touch contact;
        // neither is exposed as a guest USB peripheral. Activated rings still
        // publish idle samples, with the mouse connection attribute cleared.
        if mouse_active || keyboard_active {
            self.peripheral_sampling_number = self.peripheral_sampling_number.wrapping_add(1);
            if mouse_active {
                let mut entry = [0; MOUSE_ENTRY_SIZE];
                put_u64(
                    &mut entry,
                    0,
                    self.peripheral_sampling_number.wrapping_shl(1),
                );
                put_u64(&mut entry, 8, self.peripheral_sampling_number);
                self.mouse.publish(writes, MOUSE_OFFSET, &entry);
            }
            if keyboard_active {
                let mut entry = [0; KEYBOARD_ENTRY_SIZE];
                put_u64(
                    &mut entry,
                    0,
                    self.peripheral_sampling_number.wrapping_shl(1),
                );
                put_u64(&mut entry, 8, self.peripheral_sampling_number);
                self.keyboard.publish(writes, KEYBOARD_OFFSET, &entry);
            }
        }
        if !publish_player_one {
            if self.connected {
                if let Some(output) = system.vibration_output.lock().unwrap().clone() {
                    output.stop()?;
                }
                self.sampling_number = self.sampling_number.saturating_add(1);
                // Assignment mode persists across disconnection.
                writes.write_u32(NPAD_OFFSET, 0);
                writes.write(NPAD_OFFSET + 8, &[0; NPAD_ENTRY_SIZE - 8]);
                self.full_key = Lifo::default();
                self.system_ext = Lifo::default();
                self.six_axis = Lifo::default();
                self.publish_system_button(writes, HOME_BUTTON_LIFO_OFFSET, false, true);
                self.publish_system_button(writes, CAPTURE_BUTTON_LIFO_OFFSET, false, false);
                self.connected = false;
                writes.signal_style = true;
                writes.connected = Some(self.connected);
            }
            return Ok(());
        }
        let neutral = EmulatedControllerState::default();
        let state = state.unwrap_or(&neutral);

        self.sampling_number = self.sampling_number.saturating_add(1);
        let style_changed = !self.connected;
        if style_changed {
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
        put_u64(&mut common, 0, self.sampling_number.wrapping_shl(1));
        put_u64(&mut common, 8, self.sampling_number);
        put_u64(&mut common, 16, npad_buttons(state));
        put_i32(&mut common, 24, i32::from(state.left_stick.x));
        put_i32(&mut common, 28, i32::from(state.left_stick.y));
        put_i32(&mut common, 32, i32::from(state.right_stick.x));
        put_i32(&mut common, 36, i32::from(state.right_stick.y));
        put_u32(
            &mut common,
            40,
            NPAD_ATTRIBUTE_CONNECTED | NPAD_ATTRIBUTE_WIRED,
        );
        self.full_key
            .publish(writes, NPAD_OFFSET + FULL_KEY_LIFO_OFFSET, &common);
        // The common SystemExt ring mirrors the active controller's buttons,
        // sticks and connection attributes, as in Npad::OnUpdate.
        // https://switchbrew.org/wiki/HID_services#NpadInternalState
        self.system_ext
            .publish(writes, NPAD_OFFSET + SYSTEM_EXT_LIFO_OFFSET, &common);

        if publish_six_axis {
            self.six_axis_at_rest[0] = state.motion.is_none_or(|motion| motion.is_at_rest);
            let mut sensor = [0_u8; SIX_AXIS_ENTRY_SIZE];
            put_u64(&mut sensor, 0, self.sampling_number.wrapping_shl(1));
            put_u64(
                &mut sensor,
                8,
                u64::try_from(delta.as_nanos()).unwrap_or(u64::MAX),
            );
            put_u64(&mut sensor, 16, self.sampling_number);
            if let Some(motion) = state.motion {
                // The sampling owner integrates every host sample with Eden's
                // MotionInput and publishes the same sensor frame directly.
                for (offset, values) in [
                    (24, motion.acceleration),
                    (36, motion.gyroscope),
                    (48, motion.rotation),
                ] {
                    for (i, value) in values.into_iter().enumerate() {
                        put_f32(&mut sensor, offset + i * 4, value);
                    }
                }
                for row in 0..3 {
                    for column in 0..3 {
                        put_f32(
                            &mut sensor,
                            60 + (row * 3 + column) * 4,
                            motion.orientation[row][column],
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
            // Publish the connected style only after its properties and first
            // samples. A client polling the style may immediately read a LIFO
            // without waiting for the style event.
            writes
                .writes
                .push_back((NPAD_OFFSET, NPAD_STYLE_FULL_KEY.to_le_bytes().into(), true));
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
        put_u64(&mut entry, 0, self.sampling_number.wrapping_shl(1));
        put_u64(&mut entry, 8, self.sampling_number);
        put_u64(&mut entry, 16, u64::from(pressed));
        lifo.publish(writes, lifo_offset, &entry)
    }
}

// Eden selects a sensor state by Npad ID, style and side. FullKey/Pokeball
// share a state; dual Joy-Con select left only for device index zero.
// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/resources/six_axis/six_axis.cpp
fn six_axis_key(handle: u32) -> Result<(u8, u8, u8), crate::HorizonIpcResult> {
    let [style, npad, device, _] = handle.to_le_bytes();
    if npad_index(u32::from(npad)).is_none() {
        return Err(crate::HorizonIpcResult::HID_INVALID_NPAD_ID);
    }
    if device >= 3 {
        return Err(crate::HorizonIpcResult::HID_NPAD_DEVICE_INDEX_OUT_OF_RANGE);
    }
    let (style, device) = match style {
        3 | 9 => (3, 0),
        4 | 6 | 7 => (style, 0),
        5 => (5, u8::from(device != 0)),
        _ => (0, 0),
    };
    Ok((npad, style, device))
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
        // Eden's TransformToStick publishes directional buttons alongside the
        // analog coordinates, with the default per-axis threshold of 0.5.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/hid_core/frontend/input_converter.cpp
        // https://switchbrew.org/wiki/HID_services#NpadButton
        | u64::from(state.left_stick.x < -0x3fff) << 16
        | u64::from(state.left_stick.y > 0x3fff) << 17
        | u64::from(state.left_stick.x > 0x3fff) << 18
        | u64::from(state.left_stick.y < -0x3fff) << 19
        | u64::from(state.right_stick.x < -0x3fff) << 20
        | u64::from(state.right_stick.y > 0x3fff) << 21
        | u64::from(state.right_stick.x > 0x3fff) << 22
        | u64::from(state.right_stick.y < -0x3fff) << 23
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
