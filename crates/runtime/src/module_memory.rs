//! Installation of executable segments into the machine's physical RAM.
use crate::memory::{Mapping, MemoryKind, ProcessMemory};
use nixe_loader_executable::PreparedModule;
use nixe_memory::{GuestVirtualAddress, MemoryPermissions};
use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InstallStage {
    Validation,
    Installation,
}

/// Fail-closed error while loading one prepared module to process memory.
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

/// Loads segments directly, with their final guest permissions. The caller
/// owns an unpublished process and discards it if construction fails. Earlier
/// segments are not copied, staged in page objects or transactionally replayed.
pub fn install_prepared_module(
    memory: &ProcessMemory,
    module: &PreparedModule,
    homebrew: bool,
) -> Result<(), ModuleInstallError> {
    for segment in module.mappings() {
        let address = GuestVirtualAddress::new(segment.guest_address());
        let permissions = segment.permissions();
        let bits = u8::from(permissions.is_readable())
            | (u8::from(permissions.is_writable()) << 1)
            | (u8::from(permissions.is_executable()) << 2);
        let permissions = MemoryPermissions::from_bits(bits)
            .filter(|permissions| {
                !permissions.contains(MemoryPermissions::WRITE)
                    || !permissions.contains(MemoryPermissions::EXECUTE)
            })
            .ok_or_else(|| {
                module_error(
                    module,
                    Some(address),
                    InstallStage::Validation,
                    "executable segment requests unsupported permissions",
                )
            })?;
        let mutable = permissions.contains(MemoryPermissions::WRITE);
        let kind = match (homebrew, mutable) {
            (false, false) => MemoryKind::Code,
            (false, true) => MemoryKind::MutableCode,
            (true, false) => MemoryKind::ModuleCode,
            (true, true) => MemoryKind::MutableModuleCode,
        };
        memory
            .map_initialized(
                address,
                segment.bytes(),
                Mapping {
                    kind,
                    permissions,
                    uncached: false,
                    locked: false,
                    permission_locked: false,
                },
            )
            .map_err(|error| {
                module_error(
                    module,
                    Some(address),
                    InstallStage::Installation,
                    error.to_string(),
                )
            })?;
    }
    Ok(())
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
