//! Format-independent installation of prepared executable mappings.

use std::error::Error;
use std::fmt::{Display, Formatter};

use swiitx_cpu::address::{AddressSpaceId, GuestVirtualAddress};
use swiitx_cpu::memory::{
    MemoryPermissions as CpuPermissions, SyntheticInstallStage, SyntheticMemory, SyntheticRamPage,
};
use swiitx_loader_executable::{MemoryPermissions as LoaderPermissions, PreparedModule};

/// Stage of a module-memory installation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InstallStage {
    /// Runtime validation and exact type conversion.
    Preflight,
    /// Private physical-page allocation.
    Allocation,
    /// Private physical-page initialization.
    Initialization,
    /// Atomic virtual-mapping publication.
    Publication,
}

/// Backend failure reported through the transactional installation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendInstallError {
    /// Stage which failed.
    pub stage: InstallStage,
    /// Guest page associated with the failure, when available.
    pub address: Option<GuestVirtualAddress>,
    /// Backend-specific diagnostic.
    pub cause: Box<str>,
}

impl BackendInstallError {
    /// Creates a structured backend failure.
    #[must_use]
    pub fn new(
        stage: InstallStage,
        address: Option<GuestVirtualAddress>,
        cause: impl Into<Box<str>>,
    ) -> Self {
        Self {
            stage,
            address,
            cause: cause.into(),
        }
    }
}

/// One ephemeral initialized page passed to a transactional memory backend.
#[derive(Clone, Copy, Debug)]
pub struct PageRequest<'a> {
    address: GuestVirtualAddress,
    bytes: &'a [u8],
    permissions: CpuPermissions,
}

impl<'a> PageRequest<'a> {
    /// Returns the page-aligned guest virtual address.
    #[must_use]
    pub const fn address(self) -> GuestVirtualAddress {
        self.address
    }

    /// Returns the exact initialized page contents.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the exact final guest permissions.
    #[must_use]
    pub const fn permissions(self) -> CpuPermissions {
        self.permissions
    }
}

/// Atomic process-memory publication boundary used by executable installation.
///
/// Implementations must either publish every requested page or leave both the
/// address space and backend-owned physical resources unchanged.
pub trait ModuleMemoryBackend {
    /// Returns the backend page size in bytes.
    fn page_size(&self) -> usize;

    /// Allocates, initializes, and atomically publishes all requested pages.
    ///
    /// Before publication, implementations must validate address ranges,
    /// collisions, resource limits, byte lengths, and permission support for
    /// the complete request. Failure must release private allocations and must
    /// not replace existing mappings.
    fn install_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        pages: &[PageRequest<'_>],
    ) -> Result<(), BackendInstallError>;
}

/// Fail-closed error while adapting one prepared module to process memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleInstallError {
    module_id: [u8; 32],
    address: Option<GuestVirtualAddress>,
    stage: InstallStage,
    cause: Box<str>,
}

impl ModuleInstallError {
    /// Returns the prepared module/build identifier.
    #[must_use]
    pub const fn module_id(&self) -> &[u8; 32] {
        &self.module_id
    }

    /// Returns the guest page associated with the failure, when available.
    #[must_use]
    pub const fn address(&self) -> Option<GuestVirtualAddress> {
        self.address
    }

    /// Returns the failed installation stage.
    #[must_use]
    pub const fn stage(&self) -> InstallStage {
        self.stage
    }

    /// Returns the precise validation or backend failure.
    #[must_use]
    pub const fn cause(&self) -> &str {
        &self.cause
    }
}

impl Display for ModuleInstallError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cannot install prepared module")?;
        if let Some(address) = self.address {
            write!(formatter, " at {address}")?;
        }
        write!(formatter, " during {:?}: {}", self.stage, self.cause)
    }
}

impl Error for ModuleInstallError {}

/// Installs a prepared module directly into a caller-selected address space.
///
/// This adapter only performs checked page splitting and domain conversion.
/// The original [`PreparedModule`] remains the authority for entry and mapping
/// metadata, and the backend owns successful physical-page allocations.
pub fn install_prepared_module(
    backend: &mut impl ModuleMemoryBackend,
    address_space: AddressSpaceId,
    module: &PreparedModule,
) -> Result<(), ModuleInstallError> {
    let page_size = backend.page_size();
    let page_size_u64 = u64::try_from(page_size).map_err(|_| {
        module_error(
            module,
            None,
            InstallStage::Preflight,
            "backend page size is not representable",
        )
    })?;
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(module_error(
            module,
            None,
            InstallStage::Preflight,
            "backend page size must be a nonzero power of two",
        ));
    }

    let mut pages = Vec::new();
    for mapping in module.mappings() {
        let mapping_address = GuestVirtualAddress::new(mapping.guest_address());
        if !mapping_address.is_aligned_to(page_size_u64) || mapping.bytes().len() % page_size != 0 {
            return Err(module_error(
                module,
                Some(mapping_address),
                InstallStage::Preflight,
                "prepared mapping cannot be represented by the backend page geometry",
            ));
        }
        let permissions = translate_permissions(mapping.permissions()).ok_or_else(|| {
            module_error(
                module,
                Some(mapping_address),
                InstallStage::Preflight,
                "prepared mapping permissions are unsupported",
            )
        })?;
        for (page_index, bytes) in mapping.bytes().chunks_exact(page_size).enumerate() {
            let byte_offset = page_index.checked_mul(page_size).ok_or_else(|| {
                module_error(
                    module,
                    Some(mapping_address),
                    InstallStage::Preflight,
                    "mapping page offset overflows host indexing",
                )
            })?;
            let byte_offset_u64 = u64::try_from(byte_offset).map_err(|_| {
                module_error(
                    module,
                    Some(mapping_address),
                    InstallStage::Preflight,
                    "mapping page offset is not representable",
                )
            })?;
            let address = mapping_address
                .checked_add(byte_offset_u64)
                .ok_or_else(|| {
                    module_error(
                        module,
                        Some(mapping_address),
                        InstallStage::Preflight,
                        "mapping guest address overflows",
                    )
                })?;
            address.checked_add(page_size_u64 - 1).ok_or_else(|| {
                module_error(
                    module,
                    Some(address),
                    InstallStage::Preflight,
                    "mapping guest page range overflows",
                )
            })?;
            pages.push(PageRequest {
                address,
                bytes,
                permissions,
            });
        }
    }

    backend
        .install_pages_atomic(address_space, &pages)
        .map_err(|error| module_error(module, error.address, error.stage, error.cause))
}

fn translate_permissions(permissions: LoaderPermissions) -> Option<CpuPermissions> {
    match (
        permissions.is_readable(),
        permissions.is_writable(),
        permissions.is_executable(),
    ) {
        (true, false, false) => Some(CpuPermissions::READ),
        (true, true, false) => Some(CpuPermissions::READ_WRITE),
        (true, false, true) => Some(CpuPermissions::READ_EXECUTE),
        (false, false, true) => Some(CpuPermissions::EXECUTE),
        _ => None,
    }
}

fn module_error(
    module: &PreparedModule,
    address: Option<GuestVirtualAddress>,
    stage: InstallStage,
    cause: impl Into<Box<str>>,
) -> ModuleInstallError {
    ModuleInstallError {
        module_id: *module.module_id(),
        address,
        stage,
        cause: cause.into(),
    }
}

impl ModuleMemoryBackend for SyntheticMemory {
    fn page_size(&self) -> usize {
        swiitx_cpu::memory::SYNTHETIC_PAGE_SIZE
    }

    fn install_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        pages: &[PageRequest<'_>],
    ) -> Result<(), BackendInstallError> {
        let requests = pages
            .iter()
            .map(|page| SyntheticRamPage {
                virtual_address: page.address,
                bytes: page.bytes,
                permissions: page.permissions,
            })
            .collect::<Vec<_>>();
        self.install_ram_pages_atomic(address_space, &requests)
            .map_err(|error| {
                let stage = match error.stage {
                    SyntheticInstallStage::Preflight => InstallStage::Preflight,
                    SyntheticInstallStage::Allocation => InstallStage::Allocation,
                    SyntheticInstallStage::Initialization => InstallStage::Initialization,
                    SyntheticInstallStage::Publication => InstallStage::Publication,
                };
                BackendInstallError::new(stage, error.address, error.reason)
            })
    }
}
