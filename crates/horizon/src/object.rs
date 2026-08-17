//! Horizon-owned objects retained in the generic runtime handle table.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use chrono_tz::Tz;
use nixe_loader_storage::StorageRef;
use nixe_runtime::{
    HandleObject, ReadOnlyMount, ReadableEventObject, SharedMemoryObject, VirtualClock,
    WritableEventObject,
};

use crate::IpcService;

/// Client session connected to Horizon's global `sm:` named port.
#[derive(Clone, Debug)]
pub struct ServiceManagerSession {
    registered: Arc<AtomicBool>,
}

impl ServiceManagerSession {
    pub(crate) fn new() -> Self {
        Self {
            registered: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn register_client(&self) {
        self.registered.store(true, Ordering::Release);
    }

    pub(crate) fn is_registered(&self) -> bool {
        self.registered.load(Ordering::Acquire)
    }
}

/// Client session connected to Horizon's `acc:u0` account service.
#[derive(Clone, Debug, Default)]
pub struct AccountSession {
    state: Arc<Mutex<AccountState>>,
}

impl AccountSession {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn initialize_application_info(&self, process_id: u64) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .application_process_id = Some(process_id);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AccountState {
    application_process_id: Option<u64>,
}

#[cfg(test)]
mod account_tests {
    use super::*;

    #[test]
    fn cloned_sessions_share_initialized_application_identity() {
        let session = AccountSession::new();
        let clone = session.clone();

        session.initialize_application_info(42);

        assert_eq!(clone.state.lock().unwrap().application_process_id, Some(42));
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

    #[test]
    fn cloned_session_shares_domain_conversion_and_child_objects() {
        let session = IpcSession::new(IpcService::FileSystem);
        assert_eq!(session.convert_to_domain(), IPC_ROOT_OBJECT_ID);
        let cloned = session.clone();
        assert!(cloned.is_domain());

        let object = HandleObject::new(ThreadObject::new(11));
        let object_id = cloned.insert_object(object.clone()).unwrap();
        assert!(session.object(object_id).unwrap().same_identity(&object));
        assert!(session.close_object(object_id));
        assert!(cloned.object(object_id).is_none());
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

/// Language identifiers exposed by Horizon's user settings service.
///
/// Discriminants follow `SetLanguage` in the pinned libnx ABI.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum SystemLanguage {
    Japanese = 0,
    AmericanEnglish = 1,
    French = 2,
    German = 3,
    Italian = 4,
    Spanish = 5,
    Korean = 7,
    Dutch = 8,
    Portuguese = 9,
    Russian = 10,
    BritishEnglish = 12,
    CanadianFrench = 13,
    LatinAmericanSpanish = 14,
    SimplifiedChinese = 15,
    TraditionalChinese = 16,
    BrazilianPortuguese = 17,
}

impl SystemLanguage {
    pub const AVAILABLE: [Self; 16] = [
        Self::Japanese,
        Self::AmericanEnglish,
        Self::French,
        Self::German,
        Self::Italian,
        Self::Spanish,
        Self::Korean,
        Self::Dutch,
        Self::Portuguese,
        Self::Russian,
        Self::BritishEnglish,
        Self::CanadianFrench,
        Self::LatinAmericanSpanish,
        Self::SimplifiedChinese,
        Self::TraditionalChinese,
        Self::BrazilianPortuguese,
    ];

    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Japanese),
            1 => Some(Self::AmericanEnglish),
            2 => Some(Self::French),
            3 => Some(Self::German),
            4 => Some(Self::Italian),
            5 => Some(Self::Spanish),
            7 => Some(Self::Korean),
            8 => Some(Self::Dutch),
            9 => Some(Self::Portuguese),
            10 => Some(Self::Russian),
            12 => Some(Self::BritishEnglish),
            13 => Some(Self::CanadianFrench),
            14 => Some(Self::LatinAmericanSpanish),
            15 => Some(Self::SimplifiedChinese),
            16 => Some(Self::TraditionalChinese),
            17 => Some(Self::BrazilianPortuguese),
            _ => None,
        }
    }

    /// Returns the null-padded, little-endian Horizon language code.
    pub const fn code(self) -> u64 {
        u64::from_le_bytes(match self {
            Self::Japanese => *b"ja\0\0\0\0\0\0",
            Self::AmericanEnglish => *b"en-US\0\0\0",
            Self::French => *b"fr\0\0\0\0\0\0",
            Self::German => *b"de\0\0\0\0\0\0",
            Self::Italian => *b"it\0\0\0\0\0\0",
            Self::Spanish => *b"es\0\0\0\0\0\0",
            Self::Korean => *b"ko\0\0\0\0\0\0",
            Self::Dutch => *b"nl\0\0\0\0\0\0",
            Self::Portuguese => *b"pt\0\0\0\0\0\0",
            Self::Russian => *b"ru\0\0\0\0\0\0",
            Self::BritishEnglish => *b"en-GB\0\0\0",
            Self::CanadianFrench => *b"fr-CA\0\0\0",
            Self::LatinAmericanSpanish => *b"es-419\0\0",
            Self::SimplifiedChinese => *b"zh-Hans\0",
            Self::TraditionalChinese => *b"zh-Hant\0",
            Self::BrazilianPortuguese => *b"pt-BR\0\0\0",
        })
    }
}

/// Region identifiers returned by Horizon's user settings service.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum RegionCode {
    Japan = 0,
    Usa = 1,
    Europe = 2,
    Australia = 3,
    China = 4,
    Korea = 5,
    Taiwan = 6,
}

/// Immutable user locale exposed to guest settings clients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SettingsEnvironment {
    language: SystemLanguage,
    region: RegionCode,
}

impl SettingsEnvironment {
    #[must_use]
    pub const fn new(language: SystemLanguage, region: RegionCode) -> Self {
        Self { language, region }
    }

    #[must_use]
    pub const fn for_language(language: SystemLanguage) -> Self {
        let region = match language {
            SystemLanguage::Japanese => RegionCode::Japan,
            SystemLanguage::Korean => RegionCode::Korea,
            SystemLanguage::SimplifiedChinese => RegionCode::China,
            SystemLanguage::TraditionalChinese => RegionCode::Taiwan,
            SystemLanguage::AmericanEnglish
            | SystemLanguage::LatinAmericanSpanish
            | SystemLanguage::BrazilianPortuguese => RegionCode::Usa,
            SystemLanguage::French
            | SystemLanguage::German
            | SystemLanguage::Italian
            | SystemLanguage::Spanish
            | SystemLanguage::Dutch
            | SystemLanguage::Portuguese
            | SystemLanguage::Russian
            | SystemLanguage::BritishEnglish
            | SystemLanguage::CanadianFrench => RegionCode::Europe,
        };
        Self::new(language, region)
    }

    pub(crate) const fn language(&self) -> SystemLanguage {
        self.language
    }

    pub(crate) const fn region(&self) -> RegionCode {
        self.region
    }
}

impl Default for SettingsEnvironment {
    fn default() -> Self {
        Self::for_language(SystemLanguage::AmericanEnglish)
    }
}

/// Client session connected to Horizon's `set` user settings service.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserSettingsSession {
    environment: SettingsEnvironment,
}

impl UserSettingsSession {
    pub(crate) const fn new(environment: SettingsEnvironment) -> Self {
        Self { environment }
    }

    pub(crate) const fn environment(&self) -> &SettingsEnvironment {
        &self.environment
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

    pub(crate) fn clock(&self) -> VirtualClock {
        self.clock.clone()
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

/// Library applets which the application applet service can create.
///
/// Values follow the pinned libnx `AppletId` ABI:
/// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/applet.h#L90-L114
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub(crate) enum LibraryAppletId {
    Auth = 0x0a,
    Cabinet = 0x0b,
    Controller = 0x0c,
    DataErase = 0x0d,
    Error = 0x0e,
    NetConnect = 0x0f,
    PlayerSelect = 0x10,
    SoftwareKeyboard = 0x11,
    MiiEdit = 0x12,
    Web = 0x13,
    Shop = 0x14,
    PhotoViewer = 0x15,
    Settings = 0x16,
    OfflineWeb = 0x17,
    LoginShare = 0x18,
    WifiWebAuth = 0x19,
    MyPage = 0x1a,
}

impl LibraryAppletId {
    pub(crate) const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0x0a => Self::Auth,
            0x0b => Self::Cabinet,
            0x0c => Self::Controller,
            0x0d => Self::DataErase,
            0x0e => Self::Error,
            0x0f => Self::NetConnect,
            0x10 => Self::PlayerSelect,
            0x11 => Self::SoftwareKeyboard,
            0x12 => Self::MiiEdit,
            0x13 => Self::Web,
            0x14 => Self::Shop,
            0x15 => Self::PhotoViewer,
            0x16 => Self::Settings,
            0x17 => Self::OfflineWeb,
            0x18 => Self::LoginShare,
            0x19 => Self::WifiWebAuth,
            0x1a => Self::MyPage,
            _ => return None,
        })
    }
}

/// Presentation mode requested for a library applet.
///
/// Values follow the pinned libnx `LibAppletMode` ABI:
/// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/applet.h#L116-L123
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub(crate) enum LibraryAppletMode {
    AllForeground = 0,
    Background = 1,
    NoUi = 2,
    BackgroundIndirect = 3,
    AllForegroundInitiallyHidden = 4,
}

impl LibraryAppletMode {
    pub(crate) const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::AllForeground,
            1 => Self::Background,
            2 => Self::NoUi,
            3 => Self::BackgroundIndirect,
            4 => Self::AllForegroundInitiallyHidden,
            _ => return None,
        })
    }

    pub(crate) const fn as_raw(self) -> u32 {
        self as u32
    }
}

/// Object kinds hosted by the application applet-service domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum AppletObject {
    Root,
    Proxy(AppletProxyKind),
    ApplicationFunctions,
    HomeMenuFunctions,
    GlobalStateController,
    ApplicationCreator,
    AppletCommonFunctions,
    LibraryAppletCreator,
    CommonStateGetter,
    SelfController,
    WindowController,
    AudioController,
    DisplayController,
    DebugFunctions,
    LibraryAppletAccessor {
        applet_id: LibraryAppletId,
        mode: LibraryAppletMode,
    },
    Storage {
        storage_id: u64,
    },
    StorageAccessor {
        storage_id: u64,
    },
}

/// Interface family returned by `IApplicationProxyService` for the caller's
/// applet role. Proxy kinds expose overlapping, but not identical, child
/// object graphs and must therefore remain distinct in the domain table.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum AppletProxyKind {
    Application,
    SystemApplet,
}

#[derive(Debug)]
struct ActiveLibraryApplet {
    object_id: u32,
    // Retained for the future accessor lifecycle commands which signal state
    // transitions after Start, exit, or termination.
    _state_changed_event: WritableEventObject,
    state_changed_event_reader: ReadableEventObject,
    input_storage_ids: VecDeque<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CreateLibraryAppletError {
    NotDomain,
    DomainCapacityExhausted,
    Busy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CreateAppletStorageError {
    NotDomain,
    DomainCapacityExhausted,
    SizeOutOfRange,
    AllocationFailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OpenAppletStorageAccessorError {
    StorageNotFound,
    DomainCapacityExhausted,
    ObjectIdExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AppletStorageAccessError {
    NotFound,
    OutOfRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PushLibraryAppletStorageError {
    AppletNotFound,
    StorageNotFound,
    AllocationFailed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LibraryAppletLaunchRequest {
    pub(crate) applet_id: LibraryAppletId,
    pub(crate) mode: LibraryAppletMode,
    pub(crate) input_storages: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrepareLibraryAppletLaunchError {
    AppletNotFound,
    StorageBackingMissing,
    AllocationFailed,
}

#[derive(Debug)]
struct AppletDomain {
    converted: bool,
    next_object_id: u32,
    objects: BTreeMap<u32, AppletObject>,
    operation_mode: OperationMode,
    operation_mode_changed_notification: bool,
    performance_mode_changed_notification: bool,
    focus_handling_mode: [bool; 3],
    foreground_rights_acquired: bool,
    exit_locked: bool,
    library_applet_launchable_event: WritableEventObject,
    library_applet_launchable_event_reader: ReadableEventObject,
    active_library_applet: Option<ActiveLibraryApplet>,
    next_storage_id: u64,
    storages: BTreeMap<u64, Vec<u8>>,
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
        let (launchable_event, launchable_event_reader) = nixe_runtime::EventObject::create_pair();
        let session = Self {
            domain: Arc::new(Mutex::new(AppletDomain {
                converted: false,
                next_object_id: APPLET_ROOT_OBJECT_ID + 1,
                objects,
                operation_mode,
                operation_mode_changed_notification: false,
                performance_mode_changed_notification: false,
                focus_handling_mode: [false; 3],
                foreground_rights_acquired: false,
                exit_locked: false,
                library_applet_launchable_event: launchable_event,
                library_applet_launchable_event_reader: launchable_event_reader,
                active_library_applet: None,
                next_storage_id: 1,
                storages: BTreeMap::new(),
            })),
        };
        // No library applet occupies the process-local launch slot initially.
        session.set_library_applet_launchable(true);
        session
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
        let removed_object = domain.objects.remove(&object_id);
        let removed = removed_object.is_some();
        let released_input_storage_ids = if removed
            && domain
                .active_library_applet
                .as_ref()
                .is_some_and(|active| active.object_id == object_id)
        {
            let active = domain.active_library_applet.take().unwrap();
            domain.library_applet_launchable_event.signal();
            active.input_storage_ids
        } else {
            VecDeque::new()
        };
        if let Some(
            AppletObject::Storage { storage_id } | AppletObject::StorageAccessor { storage_id },
        ) = removed_object
        {
            let still_referenced = domain.objects.values().any(|object| {
                matches!(
                    object,
                    AppletObject::Storage {
                        storage_id: candidate
                    } | AppletObject::StorageAccessor {
                        storage_id: candidate
                    } if *candidate == storage_id
                )
            }) || domain
                .active_library_applet
                .as_ref()
                .is_some_and(|active| active.input_storage_ids.contains(&storage_id));
            if !still_referenced {
                domain.storages.remove(&storage_id);
            }
        }
        for storage_id in released_input_storage_ids {
            let still_referenced = domain.objects.values().any(|object| {
                matches!(
                    object,
                    AppletObject::Storage {
                        storage_id: candidate
                    } | AppletObject::StorageAccessor {
                        storage_id: candidate
                    } if *candidate == storage_id
                )
            });
            if !still_referenced {
                domain.storages.remove(&storage_id);
            }
        }
        removed
    }

    pub(crate) fn create_storage(&self, size: u64) -> Result<u32, CreateAppletStorageError> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !domain.converted {
            return Err(CreateAppletStorageError::NotDomain);
        }
        if domain.objects.len() >= MAX_APPLET_DOMAIN_OBJECTS {
            return Err(CreateAppletStorageError::DomainCapacityExhausted);
        }
        let size = usize::try_from(size).map_err(|_| CreateAppletStorageError::SizeOutOfRange)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| CreateAppletStorageError::AllocationFailed)?;
        bytes.resize(size, 0);

        let object_id = domain.next_object_id;
        let next_object_id = object_id
            .checked_add(1)
            .ok_or(CreateAppletStorageError::DomainCapacityExhausted)?;
        let storage_id = domain.next_storage_id;
        let next_storage_id = storage_id
            .checked_add(1)
            .ok_or(CreateAppletStorageError::AllocationFailed)?;
        domain.storages.insert(storage_id, bytes);
        domain
            .objects
            .insert(object_id, AppletObject::Storage { storage_id });
        domain.next_object_id = next_object_id;
        domain.next_storage_id = next_storage_id;
        Ok(object_id)
    }

    pub(crate) fn open_storage_accessor(
        &self,
        storage_id: u64,
    ) -> Result<u32, OpenAppletStorageAccessorError> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !domain.storages.contains_key(&storage_id) {
            return Err(OpenAppletStorageAccessorError::StorageNotFound);
        }
        if domain.objects.len() >= MAX_APPLET_DOMAIN_OBJECTS {
            return Err(OpenAppletStorageAccessorError::DomainCapacityExhausted);
        }
        let object_id = domain.next_object_id;
        domain.next_object_id = domain
            .next_object_id
            .checked_add(1)
            .ok_or(OpenAppletStorageAccessorError::ObjectIdExhausted)?;
        domain
            .objects
            .insert(object_id, AppletObject::StorageAccessor { storage_id });
        Ok(object_id)
    }

    pub(crate) fn storage_size(&self, storage_id: u64) -> Option<u64> {
        let domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        u64::try_from(domain.storages.get(&storage_id)?.len()).ok()
    }

    pub(crate) fn write_storage(
        &self,
        storage_id: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), AppletStorageAccessError> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let storage = domain
            .storages
            .get_mut(&storage_id)
            .ok_or(AppletStorageAccessError::NotFound)?;
        let start = usize::try_from(offset).map_err(|_| AppletStorageAccessError::OutOfRange)?;
        let end = start
            .checked_add(bytes.len())
            .ok_or(AppletStorageAccessError::OutOfRange)?;
        let destination = storage
            .get_mut(start..end)
            .ok_or(AppletStorageAccessError::OutOfRange)?;
        destination.copy_from_slice(bytes);
        Ok(())
    }

    pub(crate) fn read_storage(
        &self,
        storage_id: u64,
        offset: u64,
        size: usize,
    ) -> Result<Vec<u8>, AppletStorageAccessError> {
        let domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let storage = domain
            .storages
            .get(&storage_id)
            .ok_or(AppletStorageAccessError::NotFound)?;
        let start = usize::try_from(offset).map_err(|_| AppletStorageAccessError::OutOfRange)?;
        let end = start
            .checked_add(size)
            .ok_or(AppletStorageAccessError::OutOfRange)?;
        Ok(storage
            .get(start..end)
            .ok_or(AppletStorageAccessError::OutOfRange)?
            .to_vec())
    }

    pub(crate) fn create_library_applet(
        &self,
        applet_id: LibraryAppletId,
        mode: LibraryAppletMode,
    ) -> Result<u32, CreateLibraryAppletError> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !domain.converted {
            return Err(CreateLibraryAppletError::NotDomain);
        }
        if domain.active_library_applet.is_some() {
            return Err(CreateLibraryAppletError::Busy);
        }
        if domain.objects.len() >= MAX_APPLET_DOMAIN_OBJECTS {
            return Err(CreateLibraryAppletError::DomainCapacityExhausted);
        }
        let object_id = domain.next_object_id;
        let next_object_id = object_id
            .checked_add(1)
            .ok_or(CreateLibraryAppletError::DomainCapacityExhausted)?;
        let (state_changed_event, state_changed_event_reader) =
            nixe_runtime::EventObject::create_pair();
        domain.objects.insert(
            object_id,
            AppletObject::LibraryAppletAccessor { applet_id, mode },
        );
        domain.next_object_id = next_object_id;
        domain.active_library_applet = Some(ActiveLibraryApplet {
            object_id,
            _state_changed_event: state_changed_event,
            state_changed_event_reader,
            input_storage_ids: VecDeque::new(),
        });
        Ok(object_id)
    }

    pub(crate) fn library_applet_state_changed_event(
        &self,
        object_id: u32,
    ) -> Option<ReadableEventObject> {
        let domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let active = domain.active_library_applet.as_ref()?;
        (active.object_id == object_id).then(|| active.state_changed_event_reader.clone())
    }

    pub(crate) fn push_library_applet_input_storage(
        &self,
        applet_object_id: u32,
        storage_object_id: u32,
    ) -> Result<(), PushLibraryAppletStorageError> {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(AppletObject::Storage { storage_id }) =
            domain.objects.get(&storage_object_id).copied()
        else {
            return Err(PushLibraryAppletStorageError::StorageNotFound);
        };
        let Some(active) = domain.active_library_applet.as_mut() else {
            return Err(PushLibraryAppletStorageError::AppletNotFound);
        };
        if active.object_id != applet_object_id {
            return Err(PushLibraryAppletStorageError::AppletNotFound);
        }
        active
            .input_storage_ids
            .try_reserve(1)
            .map_err(|_| PushLibraryAppletStorageError::AllocationFailed)?;
        active.input_storage_ids.push_back(storage_id);
        Ok(())
    }

    pub(crate) fn prepare_library_applet_launch(
        &self,
        applet_object_id: u32,
    ) -> Result<LibraryAppletLaunchRequest, PrepareLibraryAppletLaunchError> {
        let domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(active) = domain.active_library_applet.as_ref() else {
            return Err(PrepareLibraryAppletLaunchError::AppletNotFound);
        };
        if active.object_id != applet_object_id {
            return Err(PrepareLibraryAppletLaunchError::AppletNotFound);
        }
        let Some(AppletObject::LibraryAppletAccessor { applet_id, mode }) =
            domain.objects.get(&applet_object_id).copied()
        else {
            return Err(PrepareLibraryAppletLaunchError::AppletNotFound);
        };
        let mut input_storages = Vec::new();
        input_storages
            .try_reserve_exact(active.input_storage_ids.len())
            .map_err(|_| PrepareLibraryAppletLaunchError::AllocationFailed)?;
        for storage_id in &active.input_storage_ids {
            let Some(source) = domain.storages.get(storage_id) else {
                return Err(PrepareLibraryAppletLaunchError::StorageBackingMissing);
            };
            let mut storage = Vec::new();
            storage
                .try_reserve_exact(source.len())
                .map_err(|_| PrepareLibraryAppletLaunchError::AllocationFailed)?;
            storage.extend_from_slice(source);
            input_storages.push(storage);
        }
        Ok(LibraryAppletLaunchRequest {
            applet_id,
            mode,
            input_storages,
        })
    }

    pub(crate) fn set_operation_mode_changed_notification(&self, enabled: bool) {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if domain.operation_mode_changed_notification != enabled {
            domain.operation_mode_changed_notification = enabled;
        }
    }

    pub(crate) fn set_performance_mode_changed_notification(&self, enabled: bool) {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if domain.performance_mode_changed_notification != enabled {
            domain.performance_mode_changed_notification = enabled;
        }
    }

    pub(crate) fn set_focus_handling_mode(&self, mode: [bool; 3]) {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if domain.focus_handling_mode != mode {
            domain.focus_handling_mode = mode;
        }
    }

    pub(crate) fn acquire_foreground_rights(&self) {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !domain.foreground_rights_acquired {
            domain.foreground_rights_acquired = true;
        }
    }

    pub(crate) fn set_exit_locked(&self, locked: bool) {
        let mut domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        domain.exit_locked = locked;
    }

    pub(crate) fn library_applet_launchable_event(&self) -> ReadableEventObject {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .library_applet_launchable_event_reader
            .clone()
    }

    /// Updates launch-slot availability and its manual-clear notification.
    ///
    /// A future library-applet coordinator can call this at the same state
    /// transition where it reserves or releases the launch slot.
    pub(crate) fn set_library_applet_launchable(&self, launchable: bool) {
        let domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if launchable {
            domain.library_applet_launchable_event.signal();
        } else {
            domain.library_applet_launchable_event.clear();
        }
    }

    #[cfg(test)]
    pub(crate) fn exit_locked(&self) -> bool {
        self.domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .exit_locked
    }

    #[cfg(test)]
    fn requested_runtime_policy(&self) -> (bool, bool, [bool; 3], bool) {
        let domain = self
            .domain
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            domain.operation_mode_changed_notification,
            domain.performance_mode_changed_notification,
            domain.focus_handling_mode,
            domain.foreground_rights_acquired,
        )
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
            .insert_object(AppletObject::Proxy(AppletProxyKind::Application))
            .unwrap();
        assert_eq!(
            session.object(child),
            Some(AppletObject::Proxy(AppletProxyKind::Application))
        );
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

    #[test]
    fn applet_runtime_policy_is_retained_in_the_shared_domain() {
        let session = AppletSession::new(OperationMode::Console);
        let cloned = session.clone();
        cloned.set_operation_mode_changed_notification(true);
        cloned.set_performance_mode_changed_notification(true);
        cloned.set_focus_handling_mode([true, false, true]);
        cloned.acquire_foreground_rights();

        assert_eq!(
            session.requested_runtime_policy(),
            (true, true, [true, false, true], true)
        );
    }

    #[test]
    fn applet_exit_lock_is_shared_idempotent_and_reversible() {
        let session = AppletSession::new(OperationMode::Console);
        let cloned = session.clone();
        assert!(!session.exit_locked());

        cloned.set_exit_locked(true);
        cloned.set_exit_locked(true);
        assert!(session.exit_locked());

        session.set_exit_locked(false);
        session.set_exit_locked(false);
        assert!(!cloned.exit_locked());
    }

    #[test]
    fn library_applet_launchability_event_is_shared_and_tracks_slot_availability() {
        let session = AppletSession::new(OperationMode::Console);
        let first = session.library_applet_launchable_event();
        let second = session.clone().library_applet_launchable_event();
        assert!(first.is_signalled());
        assert!(second.is_signalled());

        session.set_library_applet_launchable(false);
        assert!(!first.is_signalled());
        assert!(!second.is_signalled());

        session.set_library_applet_launchable(true);
        assert!(first.is_signalled());
        assert!(second.is_signalled());
    }

    #[test]
    fn library_applet_creation_keeps_launch_permission_available_until_start() {
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let launchable = session.library_applet_launchable_event();

        let accessor = session
            .create_library_applet(
                LibraryAppletId::Controller,
                LibraryAppletMode::AllForeground,
            )
            .unwrap();
        assert_eq!(
            session.object(accessor),
            Some(AppletObject::LibraryAppletAccessor {
                applet_id: LibraryAppletId::Controller,
                mode: LibraryAppletMode::AllForeground,
            })
        );
        assert!(launchable.is_signalled());
        assert!(
            session
                .library_applet_state_changed_event(accessor)
                .is_some_and(|event| !event.is_signalled())
        );
        assert_eq!(
            session
                .create_library_applet(LibraryAppletId::Error, LibraryAppletMode::AllForeground,),
            Err(CreateLibraryAppletError::Busy)
        );

        assert!(session.close_object(accessor));
        assert!(launchable.is_signalled());
        assert!(
            session
                .library_applet_state_changed_event(accessor)
                .is_none()
        );
    }

    #[test]
    fn applet_storage_backing_lives_until_its_last_domain_object_closes() {
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let storage = session.create_storage(0x20).unwrap();
        let storage_id = match session.object(storage).unwrap() {
            AppletObject::Storage { storage_id } => storage_id,
            object => panic!("unexpected storage object: {object:?}"),
        };
        assert_eq!(session.storage_size(storage_id), Some(0x20));

        let accessor = session.open_storage_accessor(storage_id).unwrap();
        assert_eq!(
            session.object(accessor),
            Some(AppletObject::StorageAccessor { storage_id })
        );
        assert!(session.close_object(storage));
        assert_eq!(session.storage_size(storage_id), Some(0x20));
        assert!(session.close_object(accessor));
        assert_eq!(session.storage_size(storage_id), None);
    }

    #[test]
    fn library_applet_input_channel_retains_storage_after_sender_close() {
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let applet = session
            .create_library_applet(LibraryAppletId::Error, LibraryAppletMode::AllForeground)
            .unwrap();
        let storage = session.create_storage(8).unwrap();
        let AppletObject::Storage { storage_id } = session.object(storage).unwrap() else {
            panic!("CreateStorage returned the wrong object kind");
        };
        session.write_storage(storage_id, 0, b"evidence").unwrap();

        session
            .push_library_applet_input_storage(applet, storage)
            .unwrap();
        assert!(session.close_object(storage));
        assert_eq!(session.storage_size(storage_id), Some(8));
        assert_eq!(
            session.prepare_library_applet_launch(applet).unwrap(),
            LibraryAppletLaunchRequest {
                applet_id: LibraryAppletId::Error,
                mode: LibraryAppletMode::AllForeground,
                input_storages: vec![b"evidence".to_vec()],
            }
        );

        assert!(session.close_object(applet));
        assert_eq!(session.storage_size(storage_id), None);
    }

    #[test]
    fn library_applet_input_channel_rejects_non_storage_objects_atomically() {
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let applet = session
            .create_library_applet(LibraryAppletId::Error, LibraryAppletMode::AllForeground)
            .unwrap();
        let not_storage = session
            .insert_object(AppletObject::CommonStateGetter)
            .unwrap();

        assert_eq!(
            session.push_library_applet_input_storage(applet, not_storage),
            Err(PushLibraryAppletStorageError::StorageNotFound)
        );
        assert!(session.close_object(not_storage));
        assert!(session.close_object(applet));
    }

    #[test]
    fn applet_storage_reads_and_writes_are_bounded_and_atomic() {
        let session = AppletSession::new(OperationMode::Console);
        session.convert_to_domain();
        let storage = session.create_storage(8).unwrap();
        let AppletObject::Storage { storage_id } = session.object(storage).unwrap() else {
            panic!("CreateStorage returned the wrong object kind");
        };

        session.write_storage(storage_id, 2, &[1, 2, 3]).unwrap();
        assert_eq!(
            session.read_storage(storage_id, 0, 8).unwrap(),
            [0, 0, 1, 2, 3, 0, 0, 0]
        );
        assert_eq!(
            session.write_storage(storage_id, 7, &[9, 9]),
            Err(AppletStorageAccessError::OutOfRange)
        );
        assert_eq!(
            session.read_storage(storage_id, 0, 8).unwrap(),
            [0, 0, 1, 2, 3, 0, 0, 0]
        );
        assert_eq!(
            session.read_storage(storage_id, u64::MAX, 1),
            Err(AppletStorageAccessError::OutOfRange)
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

/// Process-local SD-card filesystem with an optional host-directory backing.
///
/// Guest paths are normalized by the semantic IPC layer. Resolution rejects
/// every symbolic-link component so a guest cannot escape the configured root
/// through host filesystem indirection. A filesystem without host backing may
/// still expose immutable launch-time overlay entries.
#[derive(Clone, Debug)]
pub struct HostDirectoryFileSystem {
    root: Option<Arc<PathBuf>>,
}

impl HostDirectoryFileSystem {
    pub(crate) fn new(root: Option<PathBuf>) -> Self {
        Self {
            root: root.map(Arc::new),
        }
    }

    pub(crate) const fn has_host_root(&self) -> bool {
        self.root.is_some()
    }

    pub(crate) fn resolve_existing(&self, guest_path: &str) -> io::Result<PathBuf> {
        let mut resolved = self
            .root
            .as_ref()
            .map(|root| root.as_ref().clone())
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "SD card is not backed"))?;
        for component in guest_path.trim_start_matches('/').split('/') {
            if component.is_empty() {
                continue;
            }
            resolved.push(component);
            let metadata = std::fs::symlink_metadata(&resolved)?;
            if metadata.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "symbolic links are not exposed through sdmc:",
                ));
            }
        }
        Ok(resolved)
    }

    pub(crate) fn resolve_new(&self, guest_path: &str) -> io::Result<PathBuf> {
        let (parent, name) = guest_path.rsplit_once('/').ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "SD-card path has no parent")
        })?;
        if name.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot create the SD-card root",
            ));
        }
        let parent = if parent.is_empty() { "/" } else { parent };
        let mut resolved = self.resolve_existing(parent)?;
        resolved.push(name);
        match std::fs::symlink_metadata(&resolved) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "SD-card entry already exists",
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        Ok(resolved)
    }
}

/// Writable host file opened through the configured SD-card root.
#[derive(Debug)]
pub struct HostFile {
    path: Arc<str>,
    file: Mutex<File>,
    readable: bool,
    writable: bool,
    allow_append: bool,
}

impl HostFile {
    pub(crate) fn new(
        path: Arc<str>,
        file: File,
        readable: bool,
        writable: bool,
        allow_append: bool,
    ) -> Self {
        Self {
            path,
            file: Mutex::new(file),
            readable,
            writable,
            allow_append,
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub(crate) const fn readable(&self) -> bool {
        self.readable
    }

    pub(crate) const fn writable(&self) -> bool {
        self.writable
    }

    pub(crate) const fn allows_append(&self) -> bool {
        self.allow_append
    }

    pub(crate) fn file(&self) -> &Mutex<File> {
        &self.file
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
