//! Loaders for executable images that can be mapped into emulated memory.

mod npdm;
mod nro;
mod nso;
mod preparation;

use std::fmt::{Debug, Formatter};

use nixe_loader_storage::StorageRef;

pub use npdm::{
    AcidFlags, AcidMemoryRegion, AddressSpaceType, EffectiveNpdmPolicy, FileSystemAccess,
    FileSystemPermissions, KernelCapabilities, KernelCapability, KernelMemoryMapping,
    KernelMemoryPermission, KernelMemoryRegion, KernelVersion, Npdm, NpdmLoader, ProcessFlags,
    ProgramType, SaveDataOwnerAccess, SaveDataOwnerId, ServiceAccess, ServiceAccessControl,
    ServiceAccessMode,
};
pub use nro::{NroAssets, NroImage, NroLoader, NroMetadata, NroRange};
pub use nso::{Mod0Metadata, NsoImage, NsoLoader, NsoMetadata, NsoRange, NsoSegmentCompression};
pub use preparation::{
    ExternalSymbol, MappingRegion, NsoBatchModule, PreparationConfig, PrepareError, PreparedModule,
    RelocationState, RuntimeExport, SymbolResolution, SymbolResolver, prepare_nso_batch,
};

/// Identifies the source executable format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutableFormat {
    /// Nintendo Relocatable Object, normally used by homebrew.
    Nro,
    /// Nintendo Shared Object, normally used by official software.
    Nso,
}

/// Classifies a loadable executable segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutableSegmentKind {
    /// Executable code.
    Text,
    /// Read-only data.
    ReadOnly,
    /// Writable initialized data followed by optional zero-filled BSS.
    Data,
}

/// Guest-memory access permissions for an executable segment.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryPermissions(u8);

impl MemoryPermissions {
    /// Permission to read memory.
    pub const READ: Self = Self(1 << 0);
    /// Permission to write memory.
    pub const WRITE: Self = Self(1 << 1);
    /// Permission to execute memory.
    pub const EXECUTE: Self = Self(1 << 2);

    const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns whether all permissions in `other` are present.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns whether the region is readable.
    pub const fn is_readable(self) -> bool {
        self.contains(Self::READ)
    }

    /// Returns whether the region is writable.
    pub const fn is_writable(self) -> bool {
        self.contains(Self::WRITE)
    }

    /// Returns whether the region is executable.
    pub const fn is_executable(self) -> bool {
        self.contains(Self::EXECUTE)
    }
}

impl Debug for MemoryPermissions {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let mut permissions = String::with_capacity(3);
        permissions.push(if self.is_readable() { 'r' } else { '-' });
        permissions.push(if self.is_writable() { 'w' } else { '-' });
        permissions.push(if self.is_executable() { 'x' } else { '-' });
        formatter.write_str(&permissions)
    }
}

/// One validated, file-backed executable segment.
pub struct ExecutableSegment {
    kind: ExecutableSegmentKind,
    memory_offset: u64,
    file_size: u64,
    memory_size: u64,
    mapping_size: u64,
    permissions: MemoryPermissions,
    storage: StorageRef,
}

impl ExecutableSegment {
    pub(crate) fn new(
        kind: ExecutableSegmentKind,
        memory_offset: u64,
        file_size: u64,
        memory_size: u64,
        mapping_size: u64,
        permissions: MemoryPermissions,
        storage: StorageRef,
    ) -> Self {
        Self {
            kind,
            memory_offset,
            file_size,
            memory_size,
            mapping_size,
            permissions,
            storage,
        }
    }

    /// Returns the semantic role of the segment.
    pub const fn kind(&self) -> ExecutableSegmentKind {
        self.kind
    }

    /// Returns the segment offset relative to the image base.
    pub const fn memory_offset(&self) -> u64 {
        self.memory_offset
    }

    /// Returns the number of bytes backed by the source file.
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns the mapped size, including zero-filled BSS where applicable.
    pub const fn memory_size(&self) -> u64 {
        self.memory_size
    }

    /// Returns the page-aligned region size needed when mapping the segment.
    ///
    /// This can exceed [`Self::memory_size`] for a data segment whose BSS ends
    /// partway through a page. Bytes in the alignment tail are not file-backed.
    pub const fn mapping_size(&self) -> u64 {
        self.mapping_size
    }

    /// Returns the access permissions for the mapped segment.
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    /// Returns a bounded, lazy view of the file-backed bytes.
    pub fn storage(&self) -> &StorageRef {
        &self.storage
    }
}

impl Debug for ExecutableSegment {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutableSegment")
            .field("kind", &self.kind)
            .field("memory_offset", &format_args!("{:#x}", self.memory_offset))
            .field("file_size", &format_args!("{:#x}", self.file_size))
            .field("memory_size", &format_args!("{:#x}", self.memory_size))
            .field("mapping_size", &format_args!("{:#x}", self.mapping_size))
            .field("permissions", &self.permissions)
            .finish_non_exhaustive()
    }
}

/// Format-independent description of a validated executable image.
pub struct ExecutableImage {
    format: ExecutableFormat,
    entry_offset: u64,
    module_id: [u8; 32],
    segments: Vec<ExecutableSegment>,
}

impl ExecutableImage {
    pub(crate) fn new(
        format: ExecutableFormat,
        entry_offset: u64,
        module_id: [u8; 32],
        segments: Vec<ExecutableSegment>,
    ) -> Self {
        Self {
            format,
            entry_offset,
            module_id,
            segments,
        }
    }

    /// Returns the executable's source format.
    pub const fn format(&self) -> ExecutableFormat {
        self.format
    }

    /// Returns the entry-point offset relative to the image base.
    pub const fn entry_offset(&self) -> u64 {
        self.entry_offset
    }

    /// Returns the module/build identifier embedded in the executable.
    pub const fn module_id(&self) -> &[u8; 32] {
        &self.module_id
    }

    /// Returns the validated loadable segments in memory order.
    pub fn segments(&self) -> &[ExecutableSegment] {
        &self.segments
    }
}

impl Debug for ExecutableImage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutableImage")
            .field("format", &self.format)
            .field("entry_offset", &format_args!("{:#x}", self.entry_offset))
            .field("module_id", &self.module_id)
            .field("segments", &self.segments)
            .finish()
    }
}
