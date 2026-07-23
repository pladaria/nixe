//! Horizon-owned objects retained in the generic runtime handle table.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use chrono_tz::Tz;
use nixe_loader_storage::StorageRef;
use nixe_runtime::{HandleObject, ReadOnlyMount, SharedMemoryObject, VirtualClock};

use crate::IpcService;

/// Client session connected to Horizon's global `sm:` named port.
#[derive(Clone, Debug)]
pub struct ServiceManagerSession {
    registered: Arc<AtomicBool>,
    reported_unavailable: Arc<Mutex<BTreeSet<[u8; 8]>>>,
}

impl ServiceManagerSession {
    pub(crate) fn new() -> Self {
        Self {
            registered: Arc::new(AtomicBool::new(false)),
            reported_unavailable: Arc::new(Mutex::new(BTreeSet::new())),
        }
    }

    pub(crate) fn register_client(&self) {
        self.registered.store(true, Ordering::Release);
    }

    pub(crate) fn is_registered(&self) -> bool {
        self.registered.load(Ordering::Acquire)
    }

    pub(crate) fn first_unavailable_request(&self, name: [u8; 8]) -> bool {
        const MAX_REPORTED_SERVICES: usize = 64;
        let mut reported = self
            .reported_unavailable
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if reported.len() >= MAX_REPORTED_SERVICES {
            return false;
        }
        reported.insert(name)
    }
}

const IPC_ROOT_OBJECT_ID: u32 = 1;
const MAX_IPC_DOMAIN_OBJECTS: usize = 0x40;

#[derive(Debug)]
struct IpcDomain {
    converted: bool,
    next_object_id: u32,
    objects: BTreeMap<u32, HandleObject>,
}

/// A connected Horizon service session.
///
/// Domain child objects retain the same type-erased shared identity used by a
/// process handle. This keeps conversion from inventing a second lifetime
/// model while allowing CMIF to address children by object ID.
#[derive(Clone, Debug)]
pub struct IpcSession {
    service: IpcService,
    domain: Arc<Mutex<IpcDomain>>,
}

impl IpcSession {
    pub(crate) fn new(service: IpcService) -> Self {
        Self {
            service,
            domain: Arc::new(Mutex::new(IpcDomain {
                converted: false,
                next_object_id: IPC_ROOT_OBJECT_ID + 1,
                objects: BTreeMap::new(),
            })),
        }
    }

    pub(crate) const fn service(&self) -> IpcService {
        self.service
    }

    pub(crate) fn is_domain(&self) -> bool {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .converted
    }

    pub(crate) fn convert_to_domain(&self) -> u32 {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .converted = true;
        IPC_ROOT_OBJECT_ID
    }

    pub(crate) fn object(&self, object_id: u32) -> Option<HandleObject> {
        if object_id == IPC_ROOT_OBJECT_ID {
            return None;
        }
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .objects
            .get(&object_id)
            .cloned()
    }

    pub(crate) fn insert_object(&self, object: HandleObject) -> Option<u32> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !domain.converted || domain.objects.len() + 1 >= MAX_IPC_DOMAIN_OBJECTS {
            return None;
        }
        let object_id = domain.next_object_id;
        domain.next_object_id = domain.next_object_id.checked_add(1)?;
        domain.objects.insert(object_id, object);
        Some(object_id)
    }

    pub(crate) fn close_object(&self, object_id: u32) -> bool {
        if object_id == IPC_ROOT_OBJECT_ID {
            return false;
        }
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .objects
            .remove(&object_id)
            .is_some()
    }
}

#[cfg(test)]
mod ipc_session_tests {
    use nixe_runtime::ThreadObject;

    use super::*;

    #[test]
    fn generic_domain_retains_child_handle_identity_and_closes_it() {
        let session = IpcSession::new(IpcService::FileSystem);
        assert!(!session.is_domain());
        assert!(
            session
                .insert_object(HandleObject::new(ThreadObject::new(9)))
                .is_none()
        );
        assert_eq!(session.convert_to_domain(), IPC_ROOT_OBJECT_ID);

        let object = HandleObject::new(ThreadObject::new(7));
        let object_id = session.insert_object(object.clone()).unwrap();
        let retained = session.object(object_id).unwrap();
        assert!(retained.same_identity(&object));
        assert!(session.close_object(object_id));
        assert!(session.object(object_id).is_none());
        assert!(!session.close_object(IPC_ROOT_OBJECT_ID));
    }
}

/// Client session connected to Horizon's `set:sys` service.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct SystemSettingsSession;

impl SystemSettingsSession {
    pub(crate) const fn new() -> Self {
        Self
    }
}

/// Client session connected to Horizon's `hid` service.
#[derive(Clone, Debug)]
pub struct HidSession {
    shared_memory: SharedMemoryObject,
}

impl HidSession {
    pub(crate) fn new(shared_memory: SharedMemoryObject) -> Self {
        Self { shared_memory }
    }

    pub(crate) fn create_applet_resource(&self) -> HidAppletResource {
        HidAppletResource {
            shared_memory: self.shared_memory.clone(),
        }
    }
}

/// `IAppletResource` returned by the root HID service.
#[derive(Clone, Debug)]
pub struct HidAppletResource {
    shared_memory: SharedMemoryObject,
}

impl HidAppletResource {
    pub(crate) fn shared_memory(&self) -> SharedMemoryObject {
        self.shared_memory.clone()
    }
}

const TIME_SHARED_MEMORY_SIZE: usize = 0x1000;
const TIME_LOCATION_NAME_SIZE: usize = 0x24;
const TIME_SOURCE_ID: [u8; 16] = *b"NixeTimeSource01";

/// Initial virtual-time environment injected by the application runtime.
#[derive(Clone, Debug)]
pub struct TimeEnvironment {
    clock: VirtualClock,
    location_name: [u8; TIME_LOCATION_NAME_SIZE],
    timezone: Tz,
}

impl TimeEnvironment {
    /// Creates an environment with one validated Horizon location name.
    pub fn new(clock: VirtualClock, timezone: &str) -> Result<Self, &'static str> {
        if timezone.is_empty() || timezone.len() >= TIME_LOCATION_NAME_SIZE || !timezone.is_ascii()
        {
            return Err("timezone is not representable as a Horizon location name");
        }
        // Versioned IANA rule parser used by the calendar-conversion commands:
        // https://docs.rs/chrono-tz/0.10.4/chrono_tz/enum.Tz.html
        let timezone = timezone
            .parse::<Tz>()
            .map_err(|_| "timezone is not present in the IANA database")?;
        let mut location_name = [0; TIME_LOCATION_NAME_SIZE];
        let name = timezone.name();
        location_name[..name.len()].copy_from_slice(name.as_bytes());
        Ok(Self {
            clock,
            location_name,
            timezone,
        })
    }

    pub(crate) fn create_service(&self) -> Result<TimeServiceSession, nixe_runtime::HandleError> {
        TimeServiceSession::new(self.clone())
    }
}

impl Default for TimeEnvironment {
    fn default() -> Self {
        Self::new(
            VirtualClock::new(nixe_runtime::VirtualClockMode::Realtime),
            "UTC",
        )
        .expect("UTC is a representable Horizon location name")
    }
}

#[derive(Debug)]
struct TimeState {
    clock: VirtualClock,
    location_name: [u8; TIME_LOCATION_NAME_SIZE],
    timezone: Tz,
    shared_memory: SharedMemoryObject,
    clock_offsets: Mutex<[i64; 3]>,
}

impl TimeState {
    fn current_time(&self, kind: SystemClockKind) -> i64 {
        self.clock.unix_seconds().saturating_add(
            self.clock_offsets
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)[kind.index()],
        )
    }

    fn set_current_time(
        &self,
        kind: SystemClockKind,
        unix_seconds: i64,
    ) -> Result<(), nixe_runtime::HandleError> {
        self.clock_offsets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[kind.index()] =
            unix_seconds.saturating_sub(self.clock.unix_seconds());
        let steady_seconds = i64::try_from(self.clock.elapsed().as_secs()).unwrap_or(i64::MAX);
        let offset = unix_seconds.saturating_sub(steady_seconds);
        let shared_offset = match kind {
            SystemClockKind::Network => 0x80,
            SystemClockKind::User | SystemClockKind::Local => 0x38,
        };
        write_system_clock_context(
            &self.shared_memory,
            shared_offset,
            offset,
            steady_seconds,
            TIME_SOURCE_ID,
        )
    }
}

/// Client session connected to Horizon's `time:u` static service.
#[derive(Clone, Debug)]
pub struct TimeServiceSession {
    state: Arc<TimeState>,
}

impl TimeServiceSession {
    fn new(environment: TimeEnvironment) -> Result<Self, nixe_runtime::HandleError> {
        let shared_memory = SharedMemoryObject::zeroed_with_remote_permissions(
            TIME_SHARED_MEMORY_SIZE,
            nixe_cpu::memory::MemoryPermissions::READ,
        )?;
        let elapsed_seconds =
            i64::try_from(environment.clock.elapsed().as_secs()).unwrap_or(i64::MAX);
        let wall_anchor = environment
            .clock
            .unix_seconds()
            .saturating_sub(elapsed_seconds);
        initialise_time_shared_memory(&shared_memory, wall_anchor, TIME_SOURCE_ID)?;
        Ok(Self {
            state: Arc::new(TimeState {
                clock: environment.clock,
                location_name: environment.location_name,
                timezone: environment.timezone,
                shared_memory,
                clock_offsets: Mutex::new([0; 3]),
            }),
        })
    }

    pub(crate) fn system_clock(&self, kind: SystemClockKind) -> SystemClockSession {
        SystemClockSession {
            state: Arc::clone(&self.state),
            kind,
        }
    }

    pub(crate) fn steady_clock(&self) -> SteadyClockSession {
        SteadyClockSession {
            state: Arc::clone(&self.state),
        }
    }

    pub(crate) fn timezone_service(&self) -> TimeZoneServiceSession {
        TimeZoneServiceSession {
            state: Arc::clone(&self.state),
        }
    }

    pub(crate) fn shared_memory(&self) -> SharedMemoryObject {
        self.state.shared_memory.clone()
    }
}

/// Kind of one Horizon system clock.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SystemClockKind {
    User,
    Network,
    Local,
}

impl SystemClockKind {
    const fn index(self) -> usize {
        match self {
            Self::User => 0,
            Self::Network => 1,
            Self::Local => 2,
        }
    }
}

/// Horizon `ISystemClock` object.
#[derive(Clone, Debug)]
pub struct SystemClockSession {
    state: Arc<TimeState>,
    kind: SystemClockKind,
}

impl SystemClockSession {
    pub(crate) fn current_time(&self) -> i64 {
        self.state.current_time(self.kind)
    }

    pub(crate) fn set_current_time(
        &self,
        unix_seconds: i64,
    ) -> Result<(), nixe_runtime::HandleError> {
        self.state.set_current_time(self.kind, unix_seconds)
    }
}

/// Horizon `ISteadyClock` object.
#[derive(Clone, Debug)]
pub struct SteadyClockSession {
    state: Arc<TimeState>,
}

impl SteadyClockSession {
    pub(crate) fn time_point(&self) -> (i64, [u8; 16]) {
        let seconds = i64::try_from(self.state.clock.elapsed().as_secs()).unwrap_or(i64::MAX);
        (seconds, TIME_SOURCE_ID)
    }
}

/// Horizon `ITimeZoneService` object.
#[derive(Clone, Debug)]
pub struct TimeZoneServiceSession {
    state: Arc<TimeState>,
}

impl TimeZoneServiceSession {
    pub(crate) fn location_name(&self) -> [u8; TIME_LOCATION_NAME_SIZE] {
        self.state.location_name
    }

    pub(crate) fn timezone(&self) -> Tz {
        self.state.timezone
    }
}

fn initialise_time_shared_memory(
    memory: &SharedMemoryObject,
    unix_seconds: i64,
    source_id: [u8; 16],
) -> Result<(), nixe_runtime::HandleError> {
    // libnx's lock-free reader and the three object offsets define this 4 KiB
    // layout. Both copies begin identical and use an even counter:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/time.c#L96-L158
    let mut steady = [0_u8; 0x18];
    steady[8..].copy_from_slice(&source_id);
    write_shared_object(memory, 0x00, &steady)?;

    write_system_clock_context(memory, 0x38, unix_seconds, 0, source_id)?;
    write_system_clock_context(memory, 0x80, unix_seconds, 0, source_id)
}

fn write_shared_object(
    memory: &SharedMemoryObject,
    offset: usize,
    value: &[u8],
) -> Result<(), nixe_runtime::HandleError> {
    memory.write(offset + 8, value)?;
    memory.write(offset + 8 + value.len(), value)
}

fn write_system_clock_context(
    memory: &SharedMemoryObject,
    offset: usize,
    clock_offset: i64,
    steady_time_point: i64,
    source_id: [u8; 16],
) -> Result<(), nixe_runtime::HandleError> {
    let mut context = [0_u8; 0x20];
    context[..8].copy_from_slice(&clock_offset.to_le_bytes());
    context[8..16].copy_from_slice(&steady_time_point.to_le_bytes());
    context[16..].copy_from_slice(&source_id);
    write_shared_object(memory, offset, &context)
}

const APPLET_ROOT_OBJECT_ID: u32 = 1;
const MAX_APPLET_DOMAIN_OBJECTS: usize = 64;

/// Operation mode reported by Horizon's application applet service.
///
/// The numeric values follow `nn::oe::OperationMode`:
/// https://switchbrew.org/w/index.php?title=Applet_Manager_services&oldid=14546#OperationMode
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum OperationMode {
    /// The console is being used as a handheld.
    #[default]
    Handheld = 0,
    /// The console is connected to its dock and using an external display.
    Console = 1,
}

impl OperationMode {
    pub(crate) const fn as_raw(self) -> u8 {
        self as u8
    }
}

/// Object kinds hosted by the application applet-service domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum AppletObject {
    Root,
    ApplicationProxy,
    ApplicationFunctions,
    LibraryAppletCreator,
    CommonStateGetter,
    SelfController,
    WindowController,
    AudioController,
    DisplayController,
    DebugFunctions,
}

#[derive(Debug)]
struct AppletDomain {
    converted: bool,
    next_object_id: u32,
    objects: BTreeMap<u32, AppletObject>,
    operation_mode: OperationMode,
}

/// Client session connected to Horizon's `appletOE` service.
///
/// Horizon domains multiplex a root object and its child objects over one
/// kernel session. The shared table therefore belongs to the session rather
/// than to the process handle table.
#[derive(Clone, Debug)]
pub struct AppletSession {
    domain: Arc<Mutex<AppletDomain>>,
}

impl AppletSession {
    pub(crate) fn new(operation_mode: OperationMode) -> Self {
        let mut objects = BTreeMap::new();
        objects.insert(APPLET_ROOT_OBJECT_ID, AppletObject::Root);
        Self {
            domain: Arc::new(Mutex::new(AppletDomain {
                converted: false,
                next_object_id: APPLET_ROOT_OBJECT_ID + 1,
                objects,
                operation_mode,
            })),
        }
    }

    pub(crate) fn operation_mode(&self) -> OperationMode {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .operation_mode
    }

    pub(crate) fn is_domain(&self) -> bool {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .converted
    }

    pub(crate) fn convert_to_domain(&self) -> u32 {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        domain.converted = true;
        APPLET_ROOT_OBJECT_ID
    }

    pub(crate) fn object(&self, object_id: u32) -> Option<AppletObject> {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .objects
            .get(&object_id)
            .copied()
    }

    pub(crate) fn insert_object(&self, object: AppletObject) -> Option<u32> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !domain.converted || domain.objects.len() >= MAX_APPLET_DOMAIN_OBJECTS {
            return None;
        }
        let object_id = domain.next_object_id;
        domain.next_object_id = domain.next_object_id.checked_add(1)?;
        domain.objects.insert(object_id, object);
        Some(object_id)
    }

    pub(crate) fn close_object(&self, object_id: u32) -> bool {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if object_id == APPLET_ROOT_OBJECT_ID {
            return false;
        }
        domain.objects.remove(&object_id).is_some()
    }
}

#[cfg(test)]
mod applet_tests {
    use super::*;

    #[test]
    fn applet_domain_owns_bounded_child_object_ids() {
        let session = AppletSession::new(OperationMode::Console);
        assert!(!session.is_domain());
        assert_eq!(session.operation_mode(), OperationMode::Console);
        assert_eq!(session.convert_to_domain(), APPLET_ROOT_OBJECT_ID);
        assert_eq!(
            session.object(APPLET_ROOT_OBJECT_ID),
            Some(AppletObject::Root)
        );

        let child = session
            .insert_object(AppletObject::ApplicationProxy)
            .unwrap();
        assert_eq!(session.object(child), Some(AppletObject::ApplicationProxy));
        assert!(session.close_object(child));
        assert_eq!(session.object(child), None);
        assert!(!session.close_object(APPLET_ROOT_OBJECT_ID));

        for _ in 0..MAX_APPLET_DOMAIN_OBJECTS - 1 {
            assert!(
                session
                    .insert_object(AppletObject::CommonStateGetter)
                    .is_some()
            );
        }
        assert!(
            session
                .insert_object(AppletObject::CommonStateGetter)
                .is_none()
        );
    }
}

/// Client session connected to Horizon's root `apm` service.
#[derive(Clone, Debug)]
pub struct PerformanceManagerSession {
    configurations: Arc<Mutex<[u32; 2]>>,
}

impl PerformanceManagerSession {
    pub(crate) fn new() -> Self {
        Self {
            configurations: Arc::new(Mutex::new([0; 2])),
        }
    }

    pub(crate) fn open_session(&self) -> PerformanceSession {
        PerformanceSession {
            configurations: Arc::clone(&self.configurations),
        }
    }
}

/// Per-client performance-management session returned by `apm`.
#[derive(Clone, Debug)]
pub struct PerformanceSession {
    configurations: Arc<Mutex<[u32; 2]>>,
}

impl PerformanceSession {
    pub(crate) fn configuration(&self, mode: usize) -> Option<u32> {
        self.configurations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(mode)
            .copied()
    }

    pub(crate) fn set_configuration(&self, mode: usize, configuration: u32) -> bool {
        let mut configurations = self
            .configurations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(value) = configurations.get_mut(mode) else {
            return false;
        };
        *value = configuration;
        true
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn manager_sessions_share_bounded_performance_configuration() {
        let manager = PerformanceManagerSession::new();
        let first = manager.open_session();
        let second = manager.open_session();

        assert_eq!(first.configuration(0), Some(0));
        assert!(first.set_configuration(0, 0x1234));
        assert_eq!(second.configuration(0), Some(0x1234));
        assert_eq!(second.configuration(2), None);
        assert!(!second.set_configuration(2, 1));
    }
}

#[cfg(test)]
mod time_tests {
    use super::*;
    use nixe_runtime::{VirtualClock, VirtualClockMode};

    #[test]
    fn fixed_time_is_shared_by_every_clock_and_encoded_for_libnx() {
        let environment = TimeEnvironment::new(
            VirtualClock::new(VirtualClockMode::Fixed {
                unix_seconds: 1_704_067_200,
            }),
            "Europe/Madrid",
        )
        .unwrap();
        let service = environment.create_service().unwrap();

        assert_eq!(
            service.system_clock(SystemClockKind::User).current_time(),
            1_704_067_200
        );
        assert_eq!(
            &service.timezone_service().location_name()[..13],
            b"Europe/Madrid"
        );

        let memory = service.shared_memory();
        let mut user_context = [0_u8; 0x20];
        memory.read(0x40, &mut user_context).unwrap();
        assert_eq!(
            i64::from_le_bytes(user_context[..8].try_into().unwrap()),
            1_704_067_200
        );
        assert_eq!(&user_context[16..], &TIME_SOURCE_ID);
    }
}

/// A mounted, immutable RomFS exposed through a Horizon filesystem object.
#[derive(Clone, Debug)]
pub struct ReadOnlyFileSystem {
    mount: ReadOnlyMount,
}

impl ReadOnlyFileSystem {
    pub(crate) const fn new(mount: ReadOnlyMount) -> Self {
        Self { mount }
    }

    pub(crate) const fn mount(&self) -> &ReadOnlyMount {
        &self.mount
    }
}

/// A bounded immutable Horizon file object.
#[derive(Clone)]
pub struct ReadOnlyFile {
    path: Arc<str>,
    size: u64,
    storage: StorageRef,
}

impl ReadOnlyFile {
    pub(crate) fn new(path: Arc<str>, size: u64, storage: StorageRef) -> Self {
        Self {
            path,
            size,
            storage,
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    pub(crate) fn storage(&self) -> &StorageRef {
        &self.storage
    }
}

impl Debug for ReadOnlyFile {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadOnlyFile")
            .field("path", &self.path)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Kind of one deterministic directory entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DirectoryEntryKind {
    File,
    Directory,
}

/// Guest-visible directory metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DirectoryEntry {
    name: Arc<str>,
    kind: DirectoryEntryKind,
    size: u64,
}

impl DirectoryEntry {
    pub(crate) fn new(name: Arc<str>, kind: DirectoryEntryKind, size: u64) -> Self {
        Self { name, kind, size }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn kind(&self) -> DirectoryEntryKind {
        self.kind
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// A bounded directory snapshot whose cursor is shared by duplicated handles.
#[derive(Clone, Debug)]
pub struct ReadOnlyDirectory {
    path: Arc<str>,
    entries: Arc<[DirectoryEntry]>,
    cursor: Arc<Mutex<usize>>,
}

impl ReadOnlyDirectory {
    pub(crate) fn new(path: Arc<str>, entries: Arc<[DirectoryEntry]>) -> Self {
        Self {
            path,
            entries,
            cursor: Arc::new(Mutex::new(0)),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub(crate) fn entries(&self) -> &[DirectoryEntry] {
        &self.entries
    }

    pub(crate) fn cursor(&self) -> &Mutex<usize> {
        &self.cursor
    }
}
