use std::fmt::{Debug, Formatter};

use nixe_loader_storage::{FormatLoader, LoadError, StorageRef};

const META_SIZE: usize = 0x80;
const ACI0_HEADER_SIZE: usize = 0x40;
const ACID_HEADER_SIZE: usize = 0x240;
const MAX_NPDM_SIZE: u64 = 16 * 1024 * 1024;
const KNOWN_ACID_FLAGS: u32 = 0xff;

/// Loads Nintendo Process Descriptor Metadata (`main.npdm`).
#[derive(Debug)]
pub struct NpdmLoader;

impl FormatLoader for NpdmLoader {
    type Output = Npdm;

    const FORMAT_NAME: &'static str = "NPDM";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        parse_npdm(storage)
    }
}

/// Address-space layout requested by META.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpaceType {
    AddressSpace32Bit,
    AddressSpace64BitOld,
    AddressSpace32BitNoReserved,
    AddressSpace64Bit,
}

/// Validated process flags from the META header.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ProcessFlags(u8);

impl ProcessFlags {
    pub const fn raw(self) -> u8 {
        self.0
    }
    pub const fn is_64_bit_instruction(self) -> bool {
        self.0 & 1 != 0
    }
    pub const fn address_space(self) -> AddressSpaceType {
        match (self.0 >> 1) & 7 {
            0 => AddressSpaceType::AddressSpace32Bit,
            1 => AddressSpaceType::AddressSpace64BitOld,
            2 => AddressSpaceType::AddressSpace32BitNoReserved,
            _ => AddressSpaceType::AddressSpace64Bit,
        }
    }
    pub const fn optimize_memory_allocation(self) -> bool {
        self.0 & (1 << 4) != 0
    }
    pub const fn disable_device_address_space_merge(self) -> bool {
        self.0 & (1 << 5) != 0
    }
    pub const fn enable_alias_region_extra_size(self) -> bool {
        self.0 & (1 << 6) != 0
    }
    pub const fn prevent_code_reads(self) -> bool {
        self.0 & (1 << 7) != 0
    }
}

impl Debug for ProcessFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessFlags")
            .field("raw", &format_args!("{:#04x}", self.0))
            .field("is_64_bit_instruction", &self.is_64_bit_instruction())
            .field("address_space", &self.address_space())
            .finish_non_exhaustive()
    }
}

/// Validated authorization flags from ACID.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AcidFlags(u32);

/// Resource-memory region selected by ACID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcidMemoryRegion {
    Application,
    Applet,
    SecureSystem,
    NonSecureSystem,
}

impl AcidFlags {
    pub const fn raw(self) -> u32 {
        self.0
    }
    pub const fn production(self) -> bool {
        self.0 & 1 != 0
    }
    pub const fn unqualified_approval(self) -> bool {
        self.0 & 2 != 0
    }
    pub const fn memory_region(self) -> AcidMemoryRegion {
        match (self.0 >> 2) & 0xf {
            0 => AcidMemoryRegion::Application,
            1 => AcidMemoryRegion::Applet,
            2 => AcidMemoryRegion::SecureSystem,
            _ => AcidMemoryRegion::NonSecureSystem,
        }
    }
    pub const fn load_browser_core_dll(self) -> bool {
        self.0 & (1 << 7) != 0
    }
}

impl Debug for AcidFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcidFlags")
            .field("raw", &format_args!("{:#010x}", self.0))
            .field("production", &self.production())
            .field("unqualified_approval", &self.unqualified_approval())
            .field("memory_region", &self.memory_region())
            .finish_non_exhaustive()
    }
}

/// Filesystem access permission bitset.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileSystemPermissions(u64);

impl FileSystemPermissions {
    pub const APPLICATION_INFO: Self = Self(1 << 0);
    pub const BOOT_MODE_CONTROL: Self = Self(1 << 1);
    pub const CALIBRATION: Self = Self(1 << 2);
    pub const SYSTEM_SAVE_DATA: Self = Self(1 << 3);
    pub const GAME_CARD: Self = Self(1 << 4);
    pub const SAVE_DATA_BACKUP: Self = Self(1 << 5);
    pub const SAVE_DATA_MANAGEMENT: Self = Self(1 << 6);
    pub const BIS_ALL_RAW: Self = Self(1 << 7);
    pub const GAME_CARD_RAW: Self = Self(1 << 8);
    pub const GAME_CARD_PRIVATE: Self = Self(1 << 9);
    pub const SET_TIME: Self = Self(1 << 10);
    pub const CONTENT_MANAGER: Self = Self(1 << 11);
    pub const IMAGE_MANAGER: Self = Self(1 << 12);
    pub const CREATE_SAVE_DATA: Self = Self(1 << 13);
    pub const SYSTEM_SAVE_DATA_MANAGEMENT: Self = Self(1 << 14);
    pub const BIS_FILE_SYSTEM: Self = Self(1 << 15);
    pub const SYSTEM_UPDATE: Self = Self(1 << 16);
    pub const SAVE_DATA_META: Self = Self(1 << 17);
    pub const DEVICE_SAVE_DATA: Self = Self(1 << 18);
    pub const SETTINGS_CONTROL: Self = Self(1 << 19);
    pub const SYSTEM_DATA: Self = Self(1 << 20);
    pub const SD_CARD: Self = Self(1 << 21);
    pub const HOST: Self = Self(1 << 22);
    pub const FILL_BIS: Self = Self(1 << 23);
    pub const CORRUPT_SAVE_DATA: Self = Self(1 << 24);
    pub const SAVE_DATA_FOR_DEBUG: Self = Self(1 << 25);
    pub const FORMAT_SD_CARD: Self = Self(1 << 26);
    pub const GET_RIGHTS_ID: Self = Self(1 << 27);
    pub const REGISTER_EXTERNAL_KEY: Self = Self(1 << 28);
    pub const REGISTER_UPDATE_PARTITION: Self = Self(1 << 29);
    pub const SAVE_DATA_TRANSFER: Self = Self(1 << 30);
    pub const DEVICE_DETECTION: Self = Self(1 << 31);
    pub const ACCESS_FAILURE_RESOLUTION: Self = Self(1 << 32);
    pub const SAVE_DATA_TRANSFER_VERSION_2: Self = Self(1 << 33);
    pub const REGISTER_PROGRAM_INDEX_MAP_INFO: Self = Self(1 << 34);
    pub const CREATE_OWN_SAVE_DATA: Self = Self(1 << 35);
    pub const MOVE_CACHE_STORAGE: Self = Self(1 << 36);
    pub const DEVICE_TREE_BLOB: Self = Self(1 << 37);
    pub const NOTIFY_ERROR_CONTEXT_SERVICE_READY: Self = Self(1 << 38);
    pub const CALIBRATION_SYSTEM_DATA: Self = Self(1 << 39);
    pub const CALIBRATION_LOG: Self = Self(1 << 40);
    pub const STORAGE_SECURE: Self = Self(1 << 41);
    pub const STORAGE_CONTROL: Self = Self(1 << 42);
    pub const GAME_CARD_REPORT: Self = Self(1 << 43);
    pub const MARK_BEFORE_ERASE_BIS: Self = Self(1 << 44);
    pub const HTML_VIEWER: Self = Self(1 << 45);
    pub const APPLICATION_SAVE_DATA_BACKUP: Self = Self(1 << 46);
    pub const DEBUG: Self = Self(1 << 62);
    pub const FULL_PERMISSION: Self = Self(1 << 63);

    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
    pub const fn raw(self) -> u64 {
        self.0
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    pub const fn unknown_bits(self) -> u64 {
        self.0 & !(((1_u64 << 47) - 1) | (1_u64 << 62) | (1_u64 << 63))
    }
}

impl Debug for FileSystemPermissions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FileSystemPermissions({:#018x})", self.0)
    }
}

/// Read/write accessibility attached to an ACI0 save-data owner ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveDataOwnerAccess {
    Read,
    Write,
    ReadWrite,
}

/// One requested save-data owner and its access mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveDataOwnerId {
    id: u64,
    access: SaveDataOwnerAccess,
}

impl SaveDataOwnerId {
    pub const fn id(self) -> u64 {
        self.id
    }
    pub const fn access(self) -> SaveDataOwnerAccess {
        self.access
    }
}

/// Parsed filesystem access declaration from ACI0 or ACID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileSystemAccess {
    version: u8,
    permissions: FileSystemPermissions,
    content_owner_ids: Vec<u64>,
    save_data_owner_ids: Vec<SaveDataOwnerId>,
    content_owner_range: Option<(u64, u64)>,
    save_data_owner_range: Option<(u64, u64)>,
}

impl FileSystemAccess {
    pub const fn version(&self) -> u8 {
        self.version
    }
    pub const fn permissions(&self) -> FileSystemPermissions {
        self.permissions
    }
    pub fn content_owner_ids(&self) -> &[u64] {
        &self.content_owner_ids
    }
    pub fn save_data_owner_ids(&self) -> &[SaveDataOwnerId] {
        &self.save_data_owner_ids
    }
    pub const fn content_owner_range(&self) -> Option<(u64, u64)> {
        self.content_owner_range
    }
    pub const fn save_data_owner_range(&self) -> Option<(u64, u64)> {
        self.save_data_owner_range
    }
}

/// Whether a process may connect to or register a named service.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ServiceAccessMode {
    Client,
    Host,
}

/// One service access-control declaration.
#[derive(Clone, PartialEq, Eq)]
pub struct ServiceAccess {
    name: Vec<u8>,
    mode: ServiceAccessMode,
}

impl ServiceAccess {
    pub fn name(&self) -> &[u8] {
        &self.name
    }
    pub fn name_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.name).ok()
    }
    pub const fn mode(&self) -> ServiceAccessMode {
        self.mode
    }
}

impl Debug for ServiceAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceAccess")
            .field("name", &String::from_utf8_lossy(&self.name))
            .field("mode", &self.mode)
            .finish()
    }
}

/// Ordered service access-control list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServiceAccessControl(Vec<ServiceAccess>);

impl ServiceAccessControl {
    pub fn entries(&self) -> &[ServiceAccess] {
        &self.0
    }
    pub fn allows_client(&self, name: &[u8]) -> bool {
        self.allows(name, ServiceAccessMode::Client)
    }
    pub fn allows_host(&self, name: &[u8]) -> bool {
        self.allows(name, ServiceAccessMode::Host)
    }
    fn allows(&self, name: &[u8], mode: ServiceAccessMode) -> bool {
        self.0
            .iter()
            .any(|entry| entry.mode == mode && service_matches(&entry.name, name))
    }
}

/// Process type encoded by the miscellaneous-parameters kernel capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramType {
    System,
    Application,
    Applet,
}

/// Permission attached to a physical memory mapping capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelMemoryPermission {
    ReadWrite,
    ReadOnly,
}

/// Kind of physical memory mapping capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelMemoryRegion {
    Io,
    Static,
}

/// A decoded paired physical memory mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelMemoryMapping {
    address: u64,
    size: u64,
    permission: KernelMemoryPermission,
    region: KernelMemoryRegion,
}

impl KernelMemoryMapping {
    pub const fn address(self) -> u64 {
        self.address
    }
    pub const fn size(self) -> u64 {
        self.size
    }
    pub const fn permission(self) -> KernelMemoryPermission {
        self.permission
    }
    pub const fn region(self) -> KernelMemoryRegion {
        self.region
    }
}

/// Intended Horizon kernel version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelVersion {
    major: u16,
    minor: u8,
}

impl KernelVersion {
    pub const fn major(self) -> u16 {
        self.major
    }
    pub const fn minor(self) -> u8 {
        self.minor
    }
}

/// One decoded kernel capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelCapability {
    ThreadInfo {
        lowest_priority: u8,
        highest_priority: u8,
        min_core: u8,
        max_core: u8,
    },
    SystemCalls {
        index: u8,
        mask: u32,
    },
    MemoryMap(KernelMemoryMapping),
    IoMemoryMap {
        address: u64,
    },
    MemoryRegions {
        regions: [(u8, bool); 3],
    },
    Interrupts {
        numbers: [Option<u16>; 2],
    },
    ProgramType(ProgramType),
    KernelVersion(KernelVersion),
    HandleTableSize(u16),
    DebugFlags {
        allow_debug: bool,
        force_debug_prod: bool,
        force_debug: bool,
    },
}

/// Validated kernel capability stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCapabilities(Vec<KernelCapability>);

impl KernelCapabilities {
    pub fn entries(&self) -> &[KernelCapability] {
        &self.0
    }

    /// Returns the process handle-table capacity requested by this capability
    /// stream, when the descriptor is present.
    pub fn handle_table_size(&self) -> Option<u16> {
        self.0.iter().find_map(|capability| match capability {
            KernelCapability::HandleTableSize(size) => Some(*size),
            _ => None,
        })
    }

    fn authorized_by(&self, ceiling: &Self) -> bool {
        self.0
            .iter()
            .all(|requested| capability_authorized(requested, &ceiling.0))
    }
}

/// Effective, authorization-checked policy requested by ACI0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveNpdmPolicy {
    filesystem: FileSystemAccess,
    services: ServiceAccessControl,
    kernel_capabilities: KernelCapabilities,
}

impl EffectiveNpdmPolicy {
    pub const fn filesystem(&self) -> &FileSystemAccess {
        &self.filesystem
    }
    pub const fn services(&self) -> &ServiceAccessControl {
        &self.services
    }
    pub const fn kernel_capabilities(&self) -> &KernelCapabilities {
        &self.kernel_capabilities
    }
    pub fn handle_table_size(&self) -> Option<u16> {
        self.kernel_capabilities.handle_table_size()
    }
    pub fn allows_client(&self, name: &[u8]) -> bool {
        self.services.allows_client(name)
    }
    pub fn allows_host(&self, name: &[u8]) -> bool {
        self.services.allows_host(name)
    }
}

/// Fully parsed NPDM, retaining requested and authorized security declarations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Npdm {
    signature_key_generation: u32,
    flags: ProcessFlags,
    main_thread_priority: u8,
    default_cpu_core: u8,
    system_resource_size: u32,
    version: u32,
    main_thread_stack_size: u32,
    name: [u8; 16],
    product_code: [u8; 16],
    program_id: u64,
    acid_version: u8,
    acid_unknown_209: u8,
    acid_flags: AcidFlags,
    program_id_range: (u64, u64),
    acid_signature: [u8; 256],
    acid_public_key: [u8; 256],
    requested_filesystem: FileSystemAccess,
    authorized_filesystem: FileSystemAccess,
    requested_services: ServiceAccessControl,
    authorized_services: ServiceAccessControl,
    requested_kernel_capabilities: KernelCapabilities,
    authorized_kernel_capabilities: KernelCapabilities,
    effective: EffectiveNpdmPolicy,
}

impl Npdm {
    pub const fn signature_key_generation(&self) -> u32 {
        self.signature_key_generation
    }
    pub const fn flags(&self) -> ProcessFlags {
        self.flags
    }
    pub const fn main_thread_priority(&self) -> u8 {
        self.main_thread_priority
    }
    pub const fn default_cpu_core(&self) -> u8 {
        self.default_cpu_core
    }
    pub const fn system_resource_size(&self) -> u32 {
        self.system_resource_size
    }
    pub const fn version(&self) -> u32 {
        self.version
    }
    pub const fn main_thread_stack_size(&self) -> u32 {
        self.main_thread_stack_size
    }
    pub fn name(&self) -> &[u8] {
        trim_nul(&self.name)
    }
    pub fn name_str(&self) -> Option<&str> {
        std::str::from_utf8(self.name()).ok()
    }
    pub fn product_code(&self) -> &[u8] {
        trim_nul(&self.product_code)
    }
    pub fn product_code_str(&self) -> Option<&str> {
        std::str::from_utf8(self.product_code()).ok()
    }
    pub const fn program_id(&self) -> u64 {
        self.program_id
    }
    pub const fn acid_version(&self) -> u8 {
        self.acid_version
    }
    pub const fn acid_unknown_209(&self) -> u8 {
        self.acid_unknown_209
    }
    pub const fn acid_flags(&self) -> AcidFlags {
        self.acid_flags
    }
    pub const fn program_id_range(&self) -> (u64, u64) {
        self.program_id_range
    }
    pub const fn acid_signature(&self) -> &[u8; 256] {
        &self.acid_signature
    }
    pub const fn acid_public_key(&self) -> &[u8; 256] {
        &self.acid_public_key
    }
    pub const fn requested_filesystem(&self) -> &FileSystemAccess {
        &self.requested_filesystem
    }
    pub const fn authorized_filesystem(&self) -> &FileSystemAccess {
        &self.authorized_filesystem
    }
    pub const fn requested_services(&self) -> &ServiceAccessControl {
        &self.requested_services
    }
    pub const fn authorized_services(&self) -> &ServiceAccessControl {
        &self.authorized_services
    }
    pub const fn requested_kernel_capabilities(&self) -> &KernelCapabilities {
        &self.requested_kernel_capabilities
    }
    pub const fn authorized_kernel_capabilities(&self) -> &KernelCapabilities {
        &self.authorized_kernel_capabilities
    }
    pub const fn effective_policy(&self) -> &EffectiveNpdmPolicy {
        &self.effective
    }
}

fn parse_npdm(storage: StorageRef) -> Result<Npdm, LoadError> {
    let len = storage.len()?;
    if len < META_SIZE as u64 {
        return Err(invalid_at("META", 0, "header is truncated"));
    }
    if len > MAX_NPDM_SIZE {
        return Err(invalid_at(
            "META",
            0,
            "file exceeds the 16 MiB safety limit",
        ));
    }
    let size = usize::try_from(len)
        .map_err(|_| invalid_at("META", 0, "file size is not representable"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| invalid_at("META", 0, "could not allocate file buffer"))?;
    bytes.resize(size, 0);
    storage.read_at(0, &mut bytes)?;

    if bytes.get(..4) != Some(b"META") {
        return Err(invalid_at("META", 0, "expected META magic"));
    }
    require_zero(&bytes, 0x08..0x0c, "META")?;
    require_zero(&bytes, 0x0d..0x0e, "META")?;
    require_zero(&bytes, 0x10..0x14, "META")?;
    require_zero(&bytes, 0x40..0x70, "META")?;
    let flags = bytes[0x0c];
    if (flags >> 1) & 7 > 3 {
        return Err(invalid_at("META", 0x0c, "unknown process flag bits"));
    }
    let priority = bytes[0x0e];
    if priority > 0x3f {
        return Err(invalid_at(
            "META",
            0x0e,
            "main-thread priority exceeds 0x3f",
        ));
    }
    let stack_size = read_u32(&bytes, 0x1c, "META")?;
    if stack_size & 0xfff != 0 {
        return Err(invalid_at(
            "META",
            0x1c,
            "main-thread stack size is not page aligned",
        ));
    }
    let system_resource_size = read_u32(&bytes, 0x14, "META")?;
    if system_resource_size > 0x1fe0_0000 || system_resource_size & 0xfff != 0 {
        return Err(invalid_at("META", 0x14, "invalid system-resource size"));
    }
    let aci = subregion(
        &bytes,
        0,
        read_u32(&bytes, 0x70, "META")?,
        read_u32(&bytes, 0x74, "META")?,
        ACI0_HEADER_SIZE,
        "ACI0",
    )?;
    let acid = subregion(
        &bytes,
        0,
        read_u32(&bytes, 0x78, "META")?,
        read_u32(&bytes, 0x7c, "META")?,
        ACID_HEADER_SIZE,
        "ACID",
    )?;
    ensure_disjoint(aci.clone(), acid.clone(), "META")?;

    let aci_parsed = parse_aci0(&bytes, aci.clone())?;
    let acid_parsed = parse_acid(&bytes, acid)?;
    if aci_parsed.program_id < acid_parsed.program_id_min
        || aci_parsed.program_id > acid_parsed.program_id_max
    {
        return Err(invalid_at(
            "ACI0",
            aci.start + 0x10,
            "program ID is outside the ACID authorization range",
        ));
    }
    validate_filesystem(&aci_parsed.filesystem, &acid_parsed.filesystem, aci.start)?;
    validate_services(&aci_parsed.services, &acid_parsed.services, aci.start)?;
    if !aci_parsed.kernel.authorized_by(&acid_parsed.kernel) {
        return Err(invalid_at(
            "ACI0 KAC",
            aci.start,
            "requested kernel capabilities exceed ACID authorization",
        ));
    }

    let mut name = [0; 16];
    name.copy_from_slice(&bytes[0x20..0x30]);
    let mut product_code = [0; 16];
    product_code.copy_from_slice(&bytes[0x30..0x40]);
    let effective = EffectiveNpdmPolicy {
        filesystem: aci_parsed.filesystem.clone(),
        services: aci_parsed.services.clone(),
        kernel_capabilities: aci_parsed.kernel.clone(),
    };
    Ok(Npdm {
        signature_key_generation: read_u32(&bytes, 4, "META")?,
        flags: ProcessFlags(flags),
        main_thread_priority: priority,
        default_cpu_core: bytes[0x0f],
        system_resource_size,
        version: read_u32(&bytes, 0x18, "META")?,
        main_thread_stack_size: stack_size,
        name,
        product_code,
        program_id: aci_parsed.program_id,
        acid_version: acid_parsed.version,
        acid_unknown_209: acid_parsed.unknown_209,
        acid_flags: acid_parsed.flags,
        program_id_range: (acid_parsed.program_id_min, acid_parsed.program_id_max),
        acid_signature: acid_parsed.signature,
        acid_public_key: acid_parsed.public_key,
        requested_filesystem: aci_parsed.filesystem,
        authorized_filesystem: acid_parsed.filesystem,
        requested_services: aci_parsed.services,
        authorized_services: acid_parsed.services,
        requested_kernel_capabilities: aci_parsed.kernel,
        authorized_kernel_capabilities: acid_parsed.kernel,
        effective,
    })
}

struct Aci0 {
    program_id: u64,
    filesystem: FileSystemAccess,
    services: ServiceAccessControl,
    kernel: KernelCapabilities,
}
struct Acid {
    signature: [u8; 256],
    public_key: [u8; 256],
    version: u8,
    unknown_209: u8,
    flags: AcidFlags,
    program_id_min: u64,
    program_id_max: u64,
    filesystem: FileSystemAccess,
    services: ServiceAccessControl,
    kernel: KernelCapabilities,
}

fn parse_aci0(bytes: &[u8], range: std::ops::Range<usize>) -> Result<Aci0, LoadError> {
    let data = &bytes[range.clone()];
    if data.get(..4) != Some(b"ACI0") {
        return Err(invalid_at("ACI0", range.start, "expected ACI0 magic"));
    }
    require_zero(data, 4..0x10, "ACI0")?;
    require_zero(data, 0x18..0x20, "ACI0")?;
    require_zero(data, 0x38..0x40, "ACI0")?;
    let fs = local_subregion(data, range.start, 0x20, 0x24, 0x1c, "ACI0 FAH")?;
    let sac = local_subregion(data, range.start, 0x28, 0x2c, 0, "ACI0 SAC")?;
    let kac = local_subregion(data, range.start, 0x30, 0x34, 0, "ACI0 KAC")?;
    ensure_subregions_disjoint(&[fs.clone(), sac.clone(), kac.clone()], "ACI0")?;
    Ok(Aci0 {
        program_id: read_u64(data, 0x10, "ACI0")?,
        filesystem: parse_aci_filesystem(bytes, fs)?,
        services: parse_services(bytes, sac, false, "ACI0 SAC")?,
        kernel: parse_kernel_capabilities(bytes, kac, "ACI0 KAC")?,
    })
}

fn parse_acid(bytes: &[u8], range: std::ops::Range<usize>) -> Result<Acid, LoadError> {
    let data = &bytes[range.clone()];
    if data.get(0x200..0x204) != Some(b"ACID") {
        return Err(invalid_at(
            "ACID",
            range.start + 0x200,
            "expected ACID magic",
        ));
    }
    require_zero(data, 0x20a..0x20c, "ACID")?;
    require_zero(data, 0x238..0x240, "ACID")?;
    let signed_size = usize::try_from(read_u32(data, 0x204, "ACID")?).map_err(|_| {
        invalid_at(
            "ACID",
            range.start + 0x204,
            "signed size is not representable",
        )
    })?;
    let signed_end = 0x100_usize
        .checked_add(signed_size)
        .ok_or_else(|| invalid_at("ACID", range.start + 0x204, "signed size overflows"))?;
    if signed_end > data.len() || signed_end < ACID_HEADER_SIZE {
        return Err(invalid_at(
            "ACID",
            range.start + 0x204,
            "signed region is outside ACID",
        ));
    }
    let raw_flags = read_u32(data, 0x20c, "ACID")?;
    if raw_flags & !KNOWN_ACID_FLAGS != 0 {
        return Err(invalid_at(
            "ACID",
            range.start + 0x20c,
            "unknown ACID flag bits",
        ));
    }
    if (raw_flags >> 2) & 0xf > 3 {
        return Err(invalid_at(
            "ACID",
            range.start + 0x20c,
            "unknown ACID memory region",
        ));
    }
    let min = read_u64(data, 0x210, "ACID")?;
    let max = read_u64(data, 0x218, "ACID")?;
    if min > max {
        return Err(invalid_at(
            "ACID",
            range.start + 0x210,
            "program ID range is reversed",
        ));
    }
    let fs = local_subregion(data, range.start, 0x220, 0x224, 0x2c, "ACID FAC")?;
    let sac = local_subregion(data, range.start, 0x228, 0x22c, 0, "ACID SAC")?;
    let kac = local_subregion(data, range.start, 0x230, 0x234, 0, "ACID KAC")?;
    ensure_subregions_disjoint(&[fs.clone(), sac.clone(), kac.clone()], "ACID")?;
    let mut signature = [0; 256];
    signature.copy_from_slice(&data[..0x100]);
    let mut public_key = [0; 256];
    public_key.copy_from_slice(&data[0x100..0x200]);
    Ok(Acid {
        signature,
        public_key,
        version: data[0x208],
        unknown_209: data[0x209],
        flags: AcidFlags(raw_flags),
        program_id_min: min,
        program_id_max: max,
        filesystem: parse_acid_filesystem(bytes, fs)?,
        services: parse_services(bytes, sac, true, "ACID SAC")?,
        kernel: parse_kernel_capabilities(bytes, kac, "ACID KAC")?,
    })
}

fn parse_aci_filesystem(
    bytes: &[u8],
    range: std::ops::Range<usize>,
) -> Result<FileSystemAccess, LoadError> {
    let data = &bytes[range.clone()];
    if data[0] == 0 {
        return Err(invalid_at(
            "ACI0 FAH",
            range.start,
            "version must be nonzero",
        ));
    }
    require_zero(data, 1..4, "ACI0 FAH")?;
    let permissions = FileSystemPermissions(read_u64(data, 4, "ACI0 FAH")?);
    let content = parse_owner_info(data, range.start, 0x0c, 0x10, false)?;
    let save_raw = parse_owner_info(data, range.start, 0x14, 0x18, true)?;
    let save_data_owner_ids = save_raw
        .into_iter()
        .map(|(id, access)| SaveDataOwnerId {
            id,
            access: access.expect("save entries have access"),
        })
        .collect();
    Ok(FileSystemAccess {
        version: data[0],
        permissions,
        content_owner_ids: content.into_iter().map(|v| v.0).collect(),
        save_data_owner_ids,
        content_owner_range: None,
        save_data_owner_range: None,
    })
}

fn parse_owner_info(
    data: &[u8],
    base: usize,
    offset_at: usize,
    size_at: usize,
    has_access: bool,
) -> Result<Vec<(u64, Option<SaveDataOwnerAccess>)>, LoadError> {
    let offset = usize::try_from(read_u32(data, offset_at, "ACI0 FAH")?).map_err(|_| {
        invalid_at(
            "ACI0 FAH",
            base + offset_at,
            "owner offset is not representable",
        )
    })?;
    let size = usize::try_from(read_u32(data, size_at, "ACI0 FAH")?).map_err(|_| {
        invalid_at(
            "ACI0 FAH",
            base + size_at,
            "owner size is not representable",
        )
    })?;
    if size == 0 {
        return Ok(Vec::new());
    }
    let end = offset
        .checked_add(size)
        .ok_or_else(|| invalid_at("ACI0 FAH", base + offset_at, "owner range overflows"))?;
    let info = data
        .get(offset..end)
        .ok_or_else(|| invalid_at("ACI0 FAH", base + offset_at, "owner range is outside FAH"))?;
    if info.len() < 4 {
        return Err(invalid_at(
            "ACI0 FAH",
            base + offset,
            "owner table is truncated",
        ));
    }
    let count =
        usize::try_from(u32::from_le_bytes(info[..4].try_into().unwrap())).map_err(|_| {
            invalid_at(
                "ACI0 FAH",
                base + offset,
                "owner count is not representable",
            )
        })?;
    if count > 0x10000 {
        return Err(invalid_at(
            "ACI0 FAH",
            base + offset,
            "owner count exceeds safety limit",
        ));
    }
    let (accesses, ids_offset) = if has_access {
        let after_access = 4_usize
            .checked_add(count)
            .ok_or_else(|| invalid_at("ACI0 FAH", base + offset, "owner table overflows"))?;
        let ids_offset = align_up(after_access, 4)
            .ok_or_else(|| invalid_at("ACI0 FAH", base + offset, "owner table overflows"))?;
        (
            Some(info.get(4..after_access).ok_or_else(|| {
                invalid_at(
                    "ACI0 FAH",
                    base + offset,
                    "accessibility table is truncated",
                )
            })?),
            ids_offset,
        )
    } else {
        (None, 4)
    };
    let ids_size = count
        .checked_mul(8)
        .ok_or_else(|| invalid_at("ACI0 FAH", base + offset, "owner IDs overflow"))?;
    let ids = info
        .get(ids_offset..ids_offset + ids_size)
        .ok_or_else(|| invalid_at("ACI0 FAH", base + offset, "owner ID table is truncated"))?;
    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let id = u64::from_le_bytes(ids[index * 8..index * 8 + 8].try_into().unwrap());
        let access = accesses
            .map(|values| match values[index] {
                1 => Ok(SaveDataOwnerAccess::Read),
                2 => Ok(SaveDataOwnerAccess::Write),
                3 => Ok(SaveDataOwnerAccess::ReadWrite),
                _ => Err(invalid_at(
                    "ACI0 FAH",
                    base + offset + 4 + index,
                    "invalid save-data accessibility",
                )),
            })
            .transpose()?;
        result.push((id, access));
    }
    Ok(result)
}

fn parse_acid_filesystem(
    bytes: &[u8],
    range: std::ops::Range<usize>,
) -> Result<FileSystemAccess, LoadError> {
    let data = &bytes[range.clone()];
    if data[0] == 0 {
        return Err(invalid_at(
            "ACID FAC",
            range.start,
            "version must be nonzero",
        ));
    }
    if data[3] != 0 {
        return Err(invalid_at(
            "ACID FAC",
            range.start + 3,
            "padding is nonzero",
        ));
    }
    let content_count = usize::from(data[1]);
    let save_count = usize::from(data[2]);
    let required = 0x2c_usize
        .checked_add(
            (content_count + save_count)
                .checked_mul(8)
                .ok_or_else(|| invalid_at("ACID FAC", range.start, "owner table size overflows"))?,
        )
        .ok_or_else(|| invalid_at("ACID FAC", range.start, "owner table size overflows"))?;
    if required > data.len() {
        return Err(invalid_at(
            "ACID FAC",
            range.start,
            "owner tables are truncated",
        ));
    }
    let content_range = (
        read_u64(data, 0x0c, "ACID FAC")?,
        read_u64(data, 0x14, "ACID FAC")?,
    );
    let save_range = (
        read_u64(data, 0x1c, "ACID FAC")?,
        read_u64(data, 0x24, "ACID FAC")?,
    );
    if content_range.0 > content_range.1 || save_range.0 > save_range.1 {
        return Err(invalid_at(
            "ACID FAC",
            range.start + 0x0c,
            "owner authorization range is reversed",
        ));
    }
    let mut content_owner_ids = Vec::with_capacity(content_count);
    let mut cursor = 0x2c;
    for _ in 0..content_count {
        content_owner_ids.push(read_u64(data, cursor, "ACID FAC")?);
        cursor += 8;
    }
    let mut save_data_owner_ids = Vec::with_capacity(save_count);
    for _ in 0..save_count {
        save_data_owner_ids.push(SaveDataOwnerId {
            id: read_u64(data, cursor, "ACID FAC")?,
            access: SaveDataOwnerAccess::ReadWrite,
        });
        cursor += 8;
    }
    Ok(FileSystemAccess {
        version: data[0],
        permissions: FileSystemPermissions(read_u64(data, 4, "ACID FAC")?),
        content_owner_ids,
        save_data_owner_ids,
        content_owner_range: Some(content_range),
        save_data_owner_range: Some(save_range),
    })
}

fn parse_services(
    bytes: &[u8],
    range: std::ops::Range<usize>,
    allow_wildcards: bool,
    component: &'static str,
) -> Result<ServiceAccessControl, LoadError> {
    let data = &bytes[range.clone()];
    let mut entries = Vec::new();
    let mut cursor = 0;
    while cursor < data.len() {
        let control = data[cursor];
        let length = usize::from(control & 7) + 1;
        if control & 0x78 != 0 {
            return Err(invalid_at(
                component,
                range.start + cursor,
                "reserved service control bits are set",
            ));
        }
        let end = cursor
            .checked_add(1 + length)
            .ok_or_else(|| invalid_at(component, range.start + cursor, "entry length overflows"))?;
        let name = data.get(cursor + 1..end).ok_or_else(|| {
            invalid_at(
                component,
                range.start + cursor,
                "service entry is truncated",
            )
        })?;
        if name.contains(&0) {
            return Err(invalid_at(
                component,
                range.start + cursor + 1,
                "service name contains NUL",
            ));
        }
        if name.contains(&b'*')
            && (!allow_wildcards
                || name.last() != Some(&b'*')
                || name[..name.len() - 1].contains(&b'*'))
        {
            return Err(invalid_at(
                component,
                range.start + cursor + 1,
                "invalid service wildcard",
            ));
        }
        entries.push(ServiceAccess {
            name: name.to_vec(),
            mode: if control & 0x80 != 0 {
                ServiceAccessMode::Host
            } else {
                ServiceAccessMode::Client
            },
        });
        cursor = end;
    }
    Ok(ServiceAccessControl(entries))
}

fn parse_kernel_capabilities(
    bytes: &[u8],
    range: std::ops::Range<usize>,
    component: &'static str,
) -> Result<KernelCapabilities, LoadError> {
    let data = &bytes[range.clone()];
    if !data.len().is_multiple_of(4) {
        return Err(invalid_at(
            component,
            range.start,
            "capability stream is not word aligned",
        ));
    }
    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let mut entries = Vec::new();
    let mut singleton = [false; 17];
    let mut syscall_indices = [false; 8];
    let mut cursor = 0;
    while cursor < words.len() {
        let word = words[cursor];
        if word == u32::MAX {
            cursor += 1;
            continue;
        }
        let tag = word.trailing_ones() as usize;
        let offset = range.start + cursor * 4;
        let capability = match tag {
            3 => {
                check_singleton(&mut singleton, tag, component, offset)?;
                let low = ((word >> 4) & 0x3f) as u8;
                let high = ((word >> 10) & 0x3f) as u8;
                let min_core = ((word >> 16) & 0xff) as u8;
                let max_core = (word >> 24) as u8;
                // Horizon priority zero is the highest priority, so the field named
                // "lowest priority" normally has the larger numeric value.
                if high > low || min_core > max_core {
                    return Err(invalid_at(
                        component,
                        offset,
                        "thread priority/core range is reversed",
                    ));
                }
                KernelCapability::ThreadInfo {
                    lowest_priority: low,
                    highest_priority: high,
                    min_core,
                    max_core,
                }
            }
            4 => {
                let index = (word >> 29) as u8;
                if syscall_indices[usize::from(index)] {
                    return Err(invalid_at(
                        component,
                        offset,
                        "duplicate syscall-mask index",
                    ));
                }
                syscall_indices[usize::from(index)] = true;
                KernelCapability::SystemCalls {
                    index,
                    mask: (word >> 5) & 0x00ff_ffff,
                }
            }
            6 => {
                let second = *words.get(cursor + 1).ok_or_else(|| {
                    invalid_at(component, offset, "memory-map continuation is missing")
                })?;
                if second == u32::MAX || second.trailing_ones() != 6 {
                    return Err(invalid_at(
                        component,
                        offset + 4,
                        "invalid memory-map continuation",
                    ));
                }
                if second & (0xf << 27) != 0 {
                    return Err(invalid_at(
                        component,
                        offset + 4,
                        "reserved memory-map bits are set",
                    ));
                }
                let address = u64::from((word >> 7) & 0x00ff_ffff) << 12;
                let size = u64::from((second >> 7) & 0x000f_ffff) << 12;
                if size == 0 || address.checked_add(size).is_none() {
                    return Err(invalid_at(component, offset, "invalid memory-map range"));
                }
                cursor += 1;
                KernelCapability::MemoryMap(KernelMemoryMapping {
                    address,
                    size,
                    permission: if word >> 31 != 0 {
                        KernelMemoryPermission::ReadOnly
                    } else {
                        KernelMemoryPermission::ReadWrite
                    },
                    region: if second >> 31 != 0 {
                        KernelMemoryRegion::Static
                    } else {
                        KernelMemoryRegion::Io
                    },
                })
            }
            7 => KernelCapability::IoMemoryMap {
                address: u64::from(word >> 8) << 12,
            },
            10 => {
                check_singleton(&mut singleton, tag, component, offset)?;
                let regions = [
                    ((word >> 11) & 0x3f, word & (1 << 17) != 0),
                    ((word >> 18) & 0x3f, word & (1 << 24) != 0),
                    ((word >> 25) & 0x3f, word >> 31 != 0),
                ];
                if regions.iter().any(|(kind, _)| *kind > 3) {
                    return Err(invalid_at(
                        component,
                        offset,
                        "unknown kernel memory-region type",
                    ));
                }
                KernelCapability::MemoryRegions {
                    regions: regions.map(|(kind, read_only)| (kind as u8, read_only)),
                }
            }
            11 => {
                let values = [((word >> 12) & 0x3ff) as u16, ((word >> 22) & 0x3ff) as u16];
                KernelCapability::Interrupts {
                    numbers: values.map(|value| (value != 0x3ff).then_some(value)),
                }
            }
            13 => {
                check_singleton(&mut singleton, tag, component, offset)?;
                if word & !0x0007_ffff != 0 {
                    return Err(invalid_at(
                        component,
                        offset,
                        "reserved miscellaneous-parameter bits are set",
                    ));
                }
                let kind = ((word >> 14) & 7) as u8;
                KernelCapability::ProgramType(match kind {
                    0 => ProgramType::System,
                    1 => ProgramType::Application,
                    2 => ProgramType::Applet,
                    _ => return Err(invalid_at(component, offset, "unknown program type")),
                })
            }
            14 => {
                check_singleton(&mut singleton, tag, component, offset)?;
                KernelCapability::KernelVersion(KernelVersion {
                    minor: ((word >> 15) & 0xf) as u8,
                    major: ((word >> 19) & 0x1fff) as u16,
                })
            }
            15 => {
                check_singleton(&mut singleton, tag, component, offset)?;
                if word & 0xfc00_0000 != 0 {
                    return Err(invalid_at(
                        component,
                        offset,
                        "reserved handle-table bits are set",
                    ));
                }
                KernelCapability::HandleTableSize(((word >> 16) & 0x3ff) as u16)
            }
            16 => {
                check_singleton(&mut singleton, tag, component, offset)?;
                if word & !0x000f_ffff != 0 {
                    return Err(invalid_at(
                        component,
                        offset,
                        "reserved debug-flag bits are set",
                    ));
                }
                KernelCapability::DebugFlags {
                    allow_debug: word & (1 << 17) != 0,
                    force_debug_prod: word & (1 << 18) != 0,
                    force_debug: word & (1 << 19) != 0,
                }
            }
            _ => {
                return Err(invalid_at(
                    component,
                    offset,
                    format!("unsupported capability descriptor tag {tag}"),
                ));
            }
        };
        entries.push(capability);
        cursor += 1;
    }
    Ok(KernelCapabilities(entries))
}

fn capability_authorized(requested: &KernelCapability, allowed: &[KernelCapability]) -> bool {
    allowed
        .iter()
        .any(|candidate| match (requested, candidate) {
            (
                KernelCapability::ThreadInfo {
                    lowest_priority: rl,
                    highest_priority: rh,
                    min_core: rmin,
                    max_core: rmax,
                },
                KernelCapability::ThreadInfo {
                    lowest_priority: al,
                    highest_priority: ah,
                    min_core: amin,
                    max_core: amax,
                },
            ) => rl <= al && rh >= ah && rmin >= amin && rmax <= amax,
            (
                KernelCapability::SystemCalls {
                    index: ri,
                    mask: rm,
                },
                KernelCapability::SystemCalls {
                    index: ai,
                    mask: am,
                },
            ) => ri == ai && rm & !am == 0,
            (KernelCapability::MemoryMap(r), KernelCapability::MemoryMap(a)) => {
                r.permission == a.permission
                    && r.region == a.region
                    && r.address >= a.address
                    && r.address
                        .checked_add(r.size)
                        .is_some_and(|end| end <= a.address + a.size)
            }
            (
                KernelCapability::IoMemoryMap { address: r },
                KernelCapability::IoMemoryMap { address: a },
            ) => r == a,
            (
                KernelCapability::MemoryRegions { regions: r },
                KernelCapability::MemoryRegions { regions: a },
            ) => r.iter().zip(a).all(|(r, a)| r.0 == 0 || r == a),
            (
                KernelCapability::Interrupts { numbers: r },
                KernelCapability::Interrupts { numbers: a },
            ) => r.iter().flatten().all(|number| a.contains(&Some(*number))),
            (KernelCapability::ProgramType(r), KernelCapability::ProgramType(a)) => r == a,
            (KernelCapability::KernelVersion(r), KernelCapability::KernelVersion(a)) => r <= a,
            (KernelCapability::HandleTableSize(r), KernelCapability::HandleTableSize(a)) => r <= a,
            (
                KernelCapability::DebugFlags {
                    allow_debug: rd,
                    force_debug_prod: rp,
                    force_debug: rf,
                },
                KernelCapability::DebugFlags {
                    allow_debug: ad,
                    force_debug_prod: ap,
                    force_debug: af,
                },
            ) => (!rd || *ad) && (!rp || *ap) && (!rf || *af),
            _ => false,
        })
}

fn validate_filesystem(
    requested: &FileSystemAccess,
    allowed: &FileSystemAccess,
    offset: usize,
) -> Result<(), LoadError> {
    if !allowed.permissions.contains(requested.permissions) {
        return Err(invalid_at(
            "ACI0 FAH",
            offset,
            "requested filesystem permissions exceed ACID authorization",
        ));
    }
    if requested
        .content_owner_ids
        .iter()
        .any(|id| !owner_authorized(*id, &allowed.content_owner_ids, allowed.content_owner_range))
    {
        return Err(invalid_at(
            "ACI0 FAH",
            offset,
            "content-owner ID exceeds ACID authorization",
        ));
    }
    if requested.save_data_owner_ids.iter().any(|owner| {
        !owner_authorized(
            owner.id,
            &allowed
                .save_data_owner_ids
                .iter()
                .map(|v| v.id)
                .collect::<Vec<_>>(),
            allowed.save_data_owner_range,
        )
    }) {
        return Err(invalid_at(
            "ACI0 FAH",
            offset,
            "save-data-owner ID exceeds ACID authorization",
        ));
    }
    Ok(())
}

fn owner_authorized(id: u64, explicit: &[u64], range: Option<(u64, u64)>) -> bool {
    explicit.contains(&id) || range.is_some_and(|(min, max)| id >= min && id <= max)
}

fn validate_services(
    requested: &ServiceAccessControl,
    allowed: &ServiceAccessControl,
    offset: usize,
) -> Result<(), LoadError> {
    if requested
        .0
        .iter()
        .any(|entry| !allowed.allows(&entry.name, entry.mode))
    {
        return Err(invalid_at(
            "ACI0 SAC",
            offset,
            "requested service access exceeds ACID authorization",
        ));
    }
    Ok(())
}

fn service_matches(pattern: &[u8], name: &[u8]) -> bool {
    pattern
        .strip_suffix(b"*")
        .map_or(pattern == name, |prefix| name.starts_with(prefix))
}

fn check_singleton(
    seen: &mut [bool; 17],
    tag: usize,
    component: &'static str,
    offset: usize,
) -> Result<(), LoadError> {
    if seen[tag] {
        return Err(invalid_at(
            component,
            offset,
            "duplicate singleton capability",
        ));
    }
    seen[tag] = true;
    Ok(())
}

fn trim_nul(bytes: &[u8]) -> &[u8] {
    bytes.split(|byte| *byte == 0).next().unwrap_or(bytes)
}

fn local_subregion(
    data: &[u8],
    base: usize,
    offset_at: usize,
    size_at: usize,
    minimum: usize,
    component: &'static str,
) -> Result<std::ops::Range<usize>, LoadError> {
    let offset = read_u32(data, offset_at, component)?;
    let size = read_u32(data, size_at, component)?;
    let local = subregion(data, 0, offset, size, minimum, component)?;
    Ok(base + local.start..base + local.end)
}

fn subregion(
    bytes: &[u8],
    base: usize,
    offset: u32,
    size: u32,
    minimum: usize,
    component: &'static str,
) -> Result<std::ops::Range<usize>, LoadError> {
    let start = base
        .checked_add(offset as usize)
        .ok_or_else(|| invalid_at(component, base, "offset overflows"))?;
    let end = start
        .checked_add(size as usize)
        .ok_or_else(|| invalid_at(component, start, "size overflows"))?;
    if (size as usize) < minimum {
        return Err(invalid_at(
            component,
            start,
            "region is smaller than its header",
        ));
    }
    if end > bytes.len() {
        return Err(invalid_at(component, start, "region is outside its parent"));
    }
    Ok(start..end)
}

fn ensure_disjoint(
    a: std::ops::Range<usize>,
    b: std::ops::Range<usize>,
    component: &'static str,
) -> Result<(), LoadError> {
    if a.start < b.end && b.start < a.end {
        return Err(invalid_at(
            component,
            a.start.max(b.start),
            "declared regions overlap",
        ));
    }
    Ok(())
}

fn ensure_subregions_disjoint(
    ranges: &[std::ops::Range<usize>],
    component: &'static str,
) -> Result<(), LoadError> {
    for (index, a) in ranges.iter().enumerate() {
        for b in &ranges[index + 1..] {
            if !a.is_empty() && !b.is_empty() {
                ensure_disjoint(a.clone(), b.clone(), component)?;
            }
        }
    }
    Ok(())
}

fn require_zero(
    bytes: &[u8],
    range: std::ops::Range<usize>,
    component: &'static str,
) -> Result<(), LoadError> {
    if bytes
        .get(range.clone())
        .is_none_or(|values| values.iter().any(|value| *value != 0))
    {
        return Err(invalid_at(
            component,
            range.start,
            "reserved bytes are nonzero",
        ));
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize, component: &'static str) -> Result<u32, LoadError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| invalid_at(component, offset, "u32 field is truncated"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize, component: &'static str) -> Result<u64, LoadError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| invalid_at(component, offset, "u64 field is truncated"))?;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

fn invalid_at(component: &'static str, offset: usize, reason: impl std::fmt::Display) -> LoadError {
    LoadError::invalid(
        NpdmLoader::FORMAT_NAME,
        format!("{component} at {offset:#x}: {reason}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(tag: u32, fields: u32) -> u32 {
        fields | ((1 << tag) - 1)
    }

    fn encoded(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn decodes_every_supported_kernel_capability_class() {
        let words = [
            descriptor(3, (0x20 << 4) | (0x10 << 10) | (1 << 16) | (3 << 24)),
            descriptor(4, (0x55 << 5) | (2 << 29)),
            descriptor(6, (0x100 << 7) | (1 << 31)),
            descriptor(6, (0x20 << 7) | (1 << 31)),
            descriptor(7, 0x200 << 8),
            descriptor(10, (1 << 11) | (1 << 17) | (2 << 18)),
            descriptor(11, (5 << 12) | (0x3ff << 22)),
            descriptor(13, 1 << 14),
            descriptor(14, (4 << 15) | (14 << 19)),
            descriptor(15, 0x80 << 16),
            descriptor(16, 1 << 17),
            u32::MAX,
        ];
        let bytes = encoded(&words);
        let capabilities = parse_kernel_capabilities(&bytes, 0..bytes.len(), "test KAC").unwrap();

        assert_eq!(capabilities.entries().len(), 10);
        assert_eq!(capabilities.handle_table_size(), Some(0x80));
        assert!(matches!(
            capabilities.entries()[0],
            KernelCapability::ThreadInfo {
                lowest_priority: 0x20,
                highest_priority: 0x10,
                min_core: 1,
                max_core: 3
            }
        ));
        assert!(matches!(
            capabilities.entries()[2],
            KernelCapability::MemoryMap(mapping)
                if mapping.address() == 0x10_0000 && mapping.size() == 0x20_000
        ));
        assert!(matches!(
            capabilities.entries().last(),
            Some(KernelCapability::DebugFlags {
                allow_debug: true,
                ..
            })
        ));
    }

    #[test]
    fn rejects_malformed_kernel_capability_streams() {
        let missing = encoded(&[descriptor(6, 0x100 << 7)]);
        assert!(parse_kernel_capabilities(&missing, 0..missing.len(), "test").is_err());

        let duplicate = encoded(&[descriptor(15, 4 << 16), descriptor(15, 8 << 16)]);
        assert!(parse_kernel_capabilities(&duplicate, 0..duplicate.len(), "test").is_err());

        let unknown = encoded(&[descriptor(5, 0)]);
        assert!(parse_kernel_capabilities(&unknown, 0..unknown.len(), "test").is_err());
    }

    #[test]
    fn parses_content_and_save_owner_tables() {
        let mut bytes = vec![0_u8; 0x50];
        bytes[0] = 1;
        bytes[4..12].copy_from_slice(&3_u64.to_le_bytes());
        bytes[0x0c..0x10].copy_from_slice(&0x1c_u32.to_le_bytes());
        bytes[0x10..0x14].copy_from_slice(&12_u32.to_le_bytes());
        bytes[0x14..0x18].copy_from_slice(&0x28_u32.to_le_bytes());
        bytes[0x18..0x1c].copy_from_slice(&16_u32.to_le_bytes());
        bytes[0x1c..0x20].copy_from_slice(&1_u32.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&0x1234_u64.to_le_bytes());
        bytes[0x28..0x2c].copy_from_slice(&1_u32.to_le_bytes());
        bytes[0x2c] = 3;
        bytes[0x30..0x38].copy_from_slice(&0x5678_u64.to_le_bytes());

        let access = parse_aci_filesystem(&bytes, 0..bytes.len()).unwrap();
        assert_eq!(access.content_owner_ids(), &[0x1234]);
        assert_eq!(access.save_data_owner_ids()[0].id(), 0x5678);
        assert_eq!(
            access.save_data_owner_ids()[0].access(),
            SaveDataOwnerAccess::ReadWrite
        );
    }

    #[test]
    fn rejects_excessive_owner_counts_before_allocation() {
        let mut bytes = vec![0_u8; 0x20];
        bytes[0x0c..0x10].copy_from_slice(&0x1c_u32.to_le_bytes());
        bytes[0x10..0x14].copy_from_slice(&4_u32.to_le_bytes());
        bytes[0x1c..0x20].copy_from_slice(&0x1_0001_u32.to_le_bytes());
        let error = parse_owner_info(&bytes, 0, 0x0c, 0x10, false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("owner count exceeds safety limit")
        );
    }

    #[test]
    fn service_wildcards_are_suffix_only_and_mode_sensitive() {
        let bytes = [1, b'f', b'*', 0x81, b'f', b's'];
        let access = parse_services(&bytes, 0..bytes.len(), true, "test SAC").unwrap();
        assert!(access.allows_client(b"fsp-srv"));
        assert!(access.allows_host(b"fs"));
        assert!(!access.allows_host(b"fsp-srv"));

        let invalid = [1, b'*', b'f'];
        assert!(parse_services(&invalid, 0..invalid.len(), true, "test SAC").is_err());
    }
}
