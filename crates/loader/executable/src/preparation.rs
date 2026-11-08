//! Immutable guest mappings for executable segments.
//!
//! The guest runtime owns dynamic linking and relocation. The host maps text,
//! read-only data, initialized data and BSS, then starts the ABI entry point.
//! https://switchbrew.org/wiki/Rtld
//! https://switchbrew.org/wiki/Homebrew_ABI#Entrypoint

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    ExecutableFormat, ExecutableImage, ExecutableSegment, MemoryPermissions, NroImage, NsoImage,
};

const PAGE_SIZE: u64 = 0x1000;
const MAX_IMAGE_SIZE: u64 = 512 * 1024 * 1024;
const MAX_PAGES: usize = (MAX_IMAGE_SIZE / PAGE_SIZE) as usize;

/// Caller-selected placement and guest-address boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparationConfig {
    /// Page-aligned guest address at which image offset zero is placed.
    pub image_base: u64,
    /// Exclusive upper bound of the guest address range available to the module.
    pub address_limit: u64,
}

/// A page-aligned, immutable mapping with its exact final permissions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingRegion {
    guest_address: u64,
    image_offset: u64,
    logical_size: u64,
    permissions: MemoryPermissions,
    bytes: Box<[u8]>,
}

impl MappingRegion {
    /// Returns the first guest address in this mapping.
    pub const fn guest_address(&self) -> u64 {
        self.guest_address
    }

    /// Returns the image-relative start of this mapping.
    pub const fn image_offset(&self) -> u64 {
        self.image_offset
    }

    /// Returns the logical segment bytes before the deterministic alignment tail.
    pub const fn logical_size(&self) -> u64 {
        self.logical_size
    }

    /// Returns the exact final guest permissions.
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    /// Returns initialized mapping bytes, including BSS and zero alignment tails.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the exclusive guest end address.
    pub fn guest_end(&self) -> u64 {
        self.guest_address + self.bytes.len() as u64
    }
}

/// Materialized segments ready for the runtime mapper and guest startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedModule {
    format: ExecutableFormat,
    module_id: [u8; 32],
    image_base: u64,
    image_extent: u64,
    entry_address: u64,
    mappings: Box<[MappingRegion]>,
}

impl PreparedModule {
    /// Returns the source executable format.
    pub const fn format(&self) -> ExecutableFormat {
        self.format
    }

    /// Returns the module/build identifier.
    pub const fn module_id(&self) -> &[u8; 32] {
        &self.module_id
    }

    /// Returns the selected guest image base.
    pub const fn image_base(&self) -> u64 {
        self.image_base
    }

    /// Returns the image-relative extent through the last mapped byte.
    pub const fn image_extent(&self) -> u64 {
        self.image_extent
    }

    /// Returns the relocated guest entry address.
    pub const fn entry_address(&self) -> u64 {
        self.entry_address
    }

    /// Returns sorted, non-overlapping immutable mappings.
    pub fn mappings(&self) -> &[MappingRegion] {
        &self.mappings
    }
}

/// A fail-closed executable preparation error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareError {
    reason: String,
}

impl PrepareError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Returns the precise mapping preparation failure reason.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl Display for PrepareError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cannot prepare executable: {}", self.reason)
    }
}

impl Error for PrepareError {}

impl NroImage {
    /// Materializes segments while leaving relocation to the NRO's startup code.
    pub fn prepare(&self, config: PreparationConfig) -> Result<PreparedModule, PrepareError> {
        prepare(self.executable(), config)
    }
}

impl NsoImage {
    /// Materializes segments for the guest runtime linker.
    pub fn prepare(&self, config: PreparationConfig) -> Result<PreparedModule, PrepareError> {
        prepare(self.executable(), config)
    }
}

struct WorkingRegion {
    offset: u64,
    logical_size: u64,
    permissions: MemoryPermissions,
    bytes: Vec<u8>,
}

fn prepare(
    executable: &ExecutableImage,
    config: PreparationConfig,
) -> Result<PreparedModule, PrepareError> {
    if !config.image_base.is_multiple_of(PAGE_SIZE) {
        return Err(PrepareError::new("image base is not page aligned"));
    }
    if config.image_base >= config.address_limit {
        return Err(PrepareError::new(
            "image base is outside the guest address range",
        ));
    }
    let regions = materialize(executable)?;
    let image_extent = regions
        .last()
        .and_then(|region| region.offset.checked_add(region.bytes.len() as u64))
        .ok_or_else(|| PrepareError::new("executable has no mapped extent"))?;
    if image_extent > MAX_IMAGE_SIZE {
        return Err(PrepareError::new(
            "materialized image exceeds the size limit",
        ));
    }
    let guest_end = checked_add(config.image_base, image_extent, "guest image extent")?;
    if guest_end > config.address_limit {
        return Err(PrepareError::new("module exceeds the guest address range"));
    }
    let entry_address = checked_add(
        config.image_base,
        executable.entry_offset(),
        "entry address",
    )?;
    let entry_region = find_region(&regions, executable.entry_offset(), 1)
        .ok_or_else(|| PrepareError::new("entry point is outside logical module memory"))?;
    if !entry_region.permissions.is_executable() {
        return Err(PrepareError::new("entry point is not executable"));
    }

    let mappings = regions
        .into_iter()
        .map(|region| {
            Ok(MappingRegion {
                guest_address: checked_add(config.image_base, region.offset, "mapping address")?,
                image_offset: region.offset,
                logical_size: region.logical_size,
                permissions: region.permissions,
                bytes: region.bytes.into_boxed_slice(),
            })
        })
        .collect::<Result<Box<[_]>, PrepareError>>()?;
    Ok(PreparedModule {
        format: executable.format(),
        module_id: *executable.module_id(),
        image_base: config.image_base,
        image_extent,
        entry_address,
        mappings,
    })
}

fn materialize(executable: &ExecutableImage) -> Result<Vec<WorkingRegion>, PrepareError> {
    let segments = executable.segments();
    if segments.is_empty() || segments.len() > MAX_PAGES {
        return Err(PrepareError::new("invalid executable segment count"));
    }
    let mut regions = Vec::with_capacity(segments.len());
    let mut previous_end = 0;
    let mut pages = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        validate_segment(segment, index)?;
        let end = checked_add(
            segment.memory_offset(),
            segment.mapping_size(),
            "segment mapping extent",
        )?;
        if index != 0 && segment.memory_offset() < previous_end {
            return Err(PrepareError::new(format!(
                "segment {index} overlaps a previous mapping"
            )));
        }
        previous_end = end;
        if end > MAX_IMAGE_SIZE {
            return Err(PrepareError::new(
                "materialized image exceeds the size limit",
            ));
        }
        let segment_pages = usize::try_from(segment.mapping_size() / PAGE_SIZE)
            .map_err(|_| PrepareError::new("segment page count is not representable"))?;
        pages = pages
            .checked_add(segment_pages)
            .ok_or_else(|| PrepareError::new("module page count overflows"))?;
        if pages > MAX_PAGES {
            return Err(PrepareError::new("module exceeds the page-count limit"));
        }
        let length = usize::try_from(segment.mapping_size())
            .map_err(|_| PrepareError::new("segment mapping is too large for the host"))?;
        let file_size = usize::try_from(segment.file_size())
            .map_err(|_| PrepareError::new("segment file size is too large for the host"))?;
        let mut bytes = vec![0; length];
        segment
            .storage()
            .read_at(0, &mut bytes[..file_size])
            .map_err(|error| {
                PrepareError::new(format!("cannot materialize segment {index}: {error}"))
            })?;
        regions.push(WorkingRegion {
            offset: segment.memory_offset(),
            logical_size: segment.memory_size(),
            permissions: segment.permissions(),
            bytes,
        });
    }
    Ok(regions)
}

fn validate_segment(segment: &ExecutableSegment, index: usize) -> Result<(), PrepareError> {
    if !segment.memory_offset().is_multiple_of(PAGE_SIZE)
        || !segment.mapping_size().is_multiple_of(PAGE_SIZE)
    {
        return Err(PrepareError::new(format!(
            "segment {index} is not page aligned"
        )));
    }
    if segment.file_size() > segment.memory_size() || segment.memory_size() > segment.mapping_size()
    {
        return Err(PrepareError::new(format!(
            "segment {index} has inconsistent sizes"
        )));
    }
    let source_len = segment
        .storage()
        .len()
        .map_err(|error| PrepareError::new(format!("cannot query segment {index}: {error}")))?;
    if source_len != segment.file_size() {
        return Err(PrepareError::new(format!(
            "segment {index} storage length does not match file size"
        )));
    }
    let permissions = segment.permissions();
    if permissions.is_writable() && permissions.is_executable() {
        return Err(PrepareError::new(format!("segment {index} violates W^X")));
    }
    Ok(())
}

fn find_region(regions: &[WorkingRegion], offset: u64, size: u64) -> Option<&WorkingRegion> {
    let end = offset.checked_add(size)?;
    regions.iter().find(|region| {
        let region_end = region.offset.checked_add(region.logical_size);
        offset >= region.offset && region_end.is_some_and(|value| end <= value)
    })
}

fn checked_add(left: u64, right: u64, name: &str) -> Result<u64, PrepareError> {
    left.checked_add(right)
        .ok_or_else(|| PrepareError::new(format!("{name} overflows")))
}
