//! Relocation and immutable guest-mapping preparation for loaded executables.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{
    ExecutableFormat, ExecutableImage, ExecutableSegment, MemoryPermissions, Mod0Metadata,
    NroImage, NsoImage,
};

const PAGE_SIZE: u64 = 0x1000;
const MAX_IMAGE_SIZE: u64 = 512 * 1024 * 1024;
const MAX_PAGES: usize = (MAX_IMAGE_SIZE / PAGE_SIZE) as usize;
const MAX_DYNAMIC_ENTRIES: usize = 16_384;
const MAX_RELOCATIONS: usize = 1_000_000;
const ELF64_DYN_SIZE: u64 = 16;
const ELF64_SYM_SIZE: u64 = 24;
const ELF64_RELA_SIZE: u64 = 24;

const DT_NULL: i64 = 0;
const DT_HASH: i64 = 4;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_REL: i64 = 17;
const DT_RELSZ: i64 = 18;
const DT_RELENT: i64 = 19;
const DT_PLTREL: i64 = 20;
const DT_JMPREL: i64 = 23;
const DT_PLTRELSZ: i64 = 2;
const DT_RELACOUNT: i64 = 0x6fff_fff9;
const DT_GNU_HASH: i64 = 0x6fff_fef5;
const DT_RELR: i64 = 36;
const DT_RELRSZ: i64 = 35;
const DT_RELRENT: i64 = 37;

const R_AARCH64_ABS64: u32 = 257;
const R_AARCH64_GLOB_DAT: u32 = 1025;
const R_AARCH64_JUMP_SLOT: u32 = 1026;
const R_AARCH64_RELATIVE: u32 = 1027;

/// Caller-selected placement and guest-address boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparationConfig {
    /// Page-aligned guest address at which image offset zero is placed.
    pub image_base: u64,
    /// Exclusive upper bound of the guest address range available to the module.
    pub address_limit: u64,
}

/// A symbol presented to an external resolver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalSymbol<'a> {
    name: &'a [u8],
    binding: u8,
    symbol_type: u8,
    visibility: u8,
    size: u64,
}

impl<'a> ExternalSymbol<'a> {
    /// Returns the bounded symbol name exactly as stored in the ELF string table.
    pub const fn name(&self) -> &'a [u8] {
        self.name
    }

    /// Returns the symbol name as UTF-8 when it is valid UTF-8.
    pub fn name_str(&self) -> Option<&'a str> {
        std::str::from_utf8(self.name).ok()
    }

    /// Returns the ELF binding value.
    pub const fn binding(&self) -> u8 {
        self.binding
    }

    /// Returns the ELF symbol type.
    pub const fn symbol_type(&self) -> u8 {
        self.symbol_type
    }

    /// Returns the ELF visibility value.
    pub const fn visibility(&self) -> u8 {
        self.visibility
    }

    /// Returns the declared symbol size.
    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// Result supplied by an external symbol resolver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolResolution {
    /// The symbol is defined at this guest address.
    Address(u64),
    /// The resolver has no definition in its current lookup scope.
    Unresolved,
}

/// Resolves undefined global and weak symbols against caller-controlled scope.
pub trait SymbolResolver {
    /// Resolves one bounded ELF symbol name.
    fn resolve(&self, symbol: ExternalSymbol<'_>) -> SymbolResolution;
}

/// One NSO and its final caller-selected placement in an atomic link batch.
#[derive(Clone, Copy, Debug)]
pub struct NsoBatchModule<'a> {
    pub image: &'a NsoImage,
    pub config: PreparationConfig,
}

/// One explicit runtime-provided symbol available after module definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeExport<'a> {
    pub name: &'a [u8],
    pub address: u64,
}

/// Collects a complete process scope and atomically prepares every NSO.
///
/// `scope` is a permutation of module indices in lookup precedence. Visible
/// strong definitions are unique process-wide; a strong definition replaces
/// weak definitions, while the first weak definition in scope order wins.
/// Runtime exports are consulted only when no module defines a name.
pub fn prepare_nso_batch(
    modules: &[NsoBatchModule<'_>],
    scope: &[usize],
    runtime_exports: &[RuntimeExport<'_>],
) -> Result<Vec<PreparedModule>, PrepareError> {
    if modules.is_empty() || modules.len() > 64 {
        return Err(PrepareError::new("invalid NSO batch module count"));
    }
    if scope.len() != modules.len() {
        return Err(PrepareError::new(
            "symbol scope must contain every batch module exactly once",
        ));
    }
    let mut seen = BTreeSet::new();
    for index in scope {
        if *index >= modules.len() || !seen.insert(*index) {
            return Err(PrepareError::new(
                "symbol scope is not a permutation of batch modules",
            ));
        }
    }

    let mut definitions: BTreeMap<Vec<u8>, CollectedDefinition> = BTreeMap::new();
    for index in scope {
        let module = modules[*index];
        for definition in collect_nso_definitions(module.image, module.config.image_base)? {
            match definitions.get(&definition.name) {
                Some(existing) if existing.strong && definition.strong => {
                    return Err(PrepareError::new(format!(
                        "duplicate strong symbol {}",
                        String::from_utf8_lossy(&definition.name)
                    )));
                }
                Some(existing) if existing.strong || !definition.strong => {}
                _ => {
                    definitions.insert(definition.name.clone(), definition);
                }
            }
        }
    }
    let mut runtime = BTreeMap::new();
    for export in runtime_exports {
        if export.name.is_empty() || export.name.contains(&0) {
            return Err(PrepareError::new("runtime export has an invalid name"));
        }
        if runtime
            .insert(export.name.to_vec(), export.address)
            .is_some()
        {
            return Err(PrepareError::new(format!(
                "duplicate runtime export {}",
                String::from_utf8_lossy(export.name)
            )));
        }
    }
    let resolver = |symbol: ExternalSymbol<'_>| {
        definitions
            .get(symbol.name())
            .map(|definition| SymbolResolution::Address(definition.address))
            .or_else(|| {
                runtime
                    .get(symbol.name())
                    .copied()
                    .map(SymbolResolution::Address)
            })
            .unwrap_or(SymbolResolution::Unresolved)
    };
    modules
        .iter()
        .map(|module| module.image.prepare(module.config, &resolver))
        .collect()
}

impl<F> SymbolResolver for F
where
    F: for<'a> Fn(ExternalSymbol<'a>) -> SymbolResolution,
{
    fn resolve(&self, symbol: ExternalSymbol<'_>) -> SymbolResolution {
        self(symbol)
    }
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

/// A completely validated and relocated module ready for a runtime mapper.
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

    /// Finds the mapping containing a guest address.
    pub fn mapping_at(&self, guest_address: u64) -> Option<&MappingRegion> {
        self.mappings.iter().find(|mapping| {
            guest_address >= mapping.guest_address && guest_address < mapping.guest_end()
        })
    }

    /// Reads initialized bytes by guest address, without crossing a mapping boundary.
    pub fn read_guest(&self, guest_address: u64, length: usize) -> Option<&[u8]> {
        let mapping = self.mapping_at(guest_address)?;
        let start = usize::try_from(guest_address.checked_sub(mapping.guest_address)?).ok()?;
        let end = start.checked_add(length)?;
        mapping.bytes.get(start..end)
    }

    /// Reads initialized bytes by image-relative address.
    pub fn read_image(&self, image_offset: u64, length: usize) -> Option<&[u8]> {
        self.read_guest(self.image_base.checked_add(image_offset)?, length)
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

    /// Returns the precise validation or linking failure reason.
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
    /// Materializes, relocates, and seals this NRO into immutable mappings.
    pub fn prepare(
        &self,
        config: PreparationConfig,
        resolver: &impl SymbolResolver,
    ) -> Result<PreparedModule, PrepareError> {
        let module_offset = self.metadata().module_header_offset();
        let hints = TableHints {
            strings: Some((
                self.metadata().dynamic_string_table().offset(),
                self.metadata().dynamic_string_table().size(),
            )),
            symbols: Some((
                self.metadata().dynamic_symbol_table().offset(),
                self.metadata().dynamic_symbol_table().size(),
            )),
        };
        prepare(
            self.executable(),
            module_offset.then_some_nonzero(),
            hints,
            config,
            resolver,
        )
    }
}

impl NsoImage {
    /// Materializes, relocates, and seals this NSO into immutable mappings.
    pub fn prepare(
        &self,
        config: PreparationConfig,
        resolver: &impl SymbolResolver,
    ) -> Result<PreparedModule, PrepareError> {
        let read_only = self
            .executable()
            .segments()
            .iter()
            .find(|segment| segment.kind() == crate::ExecutableSegmentKind::ReadOnly)
            .ok_or_else(|| PrepareError::new("NSO has no read-only segment"))?;
        let hints = TableHints {
            strings: Some((
                checked_add(
                    read_only.memory_offset(),
                    self.metadata().dynamic_string_table().offset(),
                    "NSO string-table offset",
                )?,
                self.metadata().dynamic_string_table().size(),
            )),
            symbols: Some((
                checked_add(
                    read_only.memory_offset(),
                    self.metadata().dynamic_symbol_table().offset(),
                    "NSO symbol-table offset",
                )?,
                self.metadata().dynamic_symbol_table().size(),
            )),
        };
        prepare(
            self.executable(),
            self.mod0().map(Mod0Metadata::header_offset),
            hints,
            config,
            resolver,
        )
    }
}

trait NonZeroOption {
    fn then_some_nonzero(self) -> Option<Self>
    where
        Self: Sized;
}

impl NonZeroOption for u64 {
    fn then_some_nonzero(self) -> Option<Self> {
        (self != 0).then_some(self)
    }
}

#[derive(Clone, Copy)]
struct TableHints {
    strings: Option<(u64, u64)>,
    symbols: Option<(u64, u64)>,
}

struct WorkingRegion {
    offset: u64,
    logical_size: u64,
    permissions: MemoryPermissions,
    bytes: Vec<u8>,
}

fn prepare(
    executable: &ExecutableImage,
    mod0_offset: Option<u64>,
    hints: TableHints,
    config: PreparationConfig,
    resolver: &impl SymbolResolver,
) -> Result<PreparedModule, PrepareError> {
    if !config.image_base.is_multiple_of(PAGE_SIZE) {
        return Err(PrepareError::new("image base is not page aligned"));
    }
    if config.image_base >= config.address_limit {
        return Err(PrepareError::new(
            "image base is outside the guest address range",
        ));
    }
    let mut regions = materialize(executable)?;
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
    let entry_region = find_region(&regions, executable.entry_offset(), 1, true)
        .ok_or_else(|| PrepareError::new("entry point is outside logical module memory"))?;
    if !entry_region.permissions.is_executable() {
        return Err(PrepareError::new("entry point is not executable"));
    }

    if let Some(offset) = mod0_offset {
        let mod0 = parse_mod0(&regions, offset)?;
        let dynamic = parse_dynamic(&regions, mod0.dynamic_offset)?;
        apply_dynamic(&mut regions, &dynamic, hints, config.image_base, resolver)?;
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

#[derive(Clone, Copy)]
struct ParsedMod0 {
    dynamic_offset: u64,
}

fn parse_mod0(regions: &[WorkingRegion], offset: u64) -> Result<ParsedMod0, PrepareError> {
    let bytes = read(regions, offset, 0x1c, true)
        .ok_or_else(|| PrepareError::new("MOD0 header is outside logical module memory"))?;
    if &bytes[..4] != b"MOD0" {
        return Err(PrepareError::new("expected MOD0 magic"));
    }
    let resolve = |field: usize, name: &str| {
        let relative = i32::from_le_bytes(
            bytes[field..field + 4]
                .try_into()
                .expect("fixed MOD0 field"),
        );
        checked_signed_add(offset, i64::from(relative), name)
    };
    let dynamic_offset = resolve(4, "MOD0 dynamic offset")?;
    let bss_start = resolve(8, "MOD0 BSS start")?;
    let bss_end = resolve(12, "MOD0 BSS end")?;
    let exception_start = resolve(16, "MOD0 exception-frame start")?;
    let exception_end = resolve(20, "MOD0 exception-frame end")?;
    let module_object = resolve(24, "MOD0 module-object offset")?;
    if bss_end < bss_start || exception_end < exception_start {
        return Err(PrepareError::new("MOD0 contains a reversed range"));
    }
    validate_logical_range(
        regions,
        dynamic_offset,
        ELF64_DYN_SIZE,
        "MOD0 dynamic table",
    )?;
    if bss_start != bss_end {
        validate_logical_range(regions, bss_start, bss_end - bss_start, "MOD0 BSS")?;
    }
    if exception_start != exception_end {
        validate_logical_range(
            regions,
            exception_start,
            exception_end - exception_start,
            "MOD0 exception frames",
        )?;
    }
    validate_logical_range(regions, module_object, 1, "MOD0 module object")?;
    Ok(ParsedMod0 { dynamic_offset })
}

#[derive(Default)]
struct DynamicInfo {
    values: BTreeMap<i64, u64>,
}

fn parse_dynamic(regions: &[WorkingRegion], offset: u64) -> Result<DynamicInfo, PrepareError> {
    let mut info = DynamicInfo::default();
    for index in 0..MAX_DYNAMIC_ENTRIES {
        let entry_offset = checked_add(
            offset,
            (index as u64) * ELF64_DYN_SIZE,
            "dynamic entry offset",
        )?;
        let entry = read(regions, entry_offset, ELF64_DYN_SIZE, true).ok_or_else(|| {
            PrepareError::new(format!(
                "dynamic table entry {index} is outside logical memory"
            ))
        })?;
        let tag = i64::from_le_bytes(entry[..8].try_into().expect("fixed dynamic tag"));
        let value = u64::from_le_bytes(entry[8..].try_into().expect("fixed dynamic value"));
        if tag == DT_NULL {
            reject_unsupported_dynamic(&info)?;
            return Ok(info);
        }
        if is_unique_tag(tag) && info.values.insert(tag, value).is_some() {
            return Err(PrepareError::new(format!(
                "dynamic tag {tag:#x} is duplicated"
            )));
        }
    }
    Err(PrepareError::new(
        "dynamic table has no bounded DT_NULL terminator",
    ))
}

fn is_unique_tag(tag: i64) -> bool {
    matches!(
        tag,
        DT_HASH
            | DT_STRTAB
            | DT_SYMTAB
            | DT_RELA
            | DT_RELASZ
            | DT_RELAENT
            | DT_STRSZ
            | DT_SYMENT
            | DT_REL
            | DT_RELSZ
            | DT_RELENT
            | DT_PLTREL
            | DT_JMPREL
            | DT_PLTRELSZ
            | DT_RELACOUNT
            | DT_GNU_HASH
            | DT_RELR
            | DT_RELRSZ
            | DT_RELRENT
    )
}

fn reject_unsupported_dynamic(info: &DynamicInfo) -> Result<(), PrepareError> {
    for (tag, name) in [
        (DT_REL, "DT_REL"),
        (DT_RELSZ, "DT_RELSZ"),
        (DT_RELENT, "DT_RELENT"),
        (DT_RELR, "DT_RELR"),
        (DT_RELRSZ, "DT_RELRSZ"),
        (DT_RELRENT, "DT_RELRENT"),
    ] {
        if info.values.contains_key(&tag) {
            return Err(PrepareError::new(format!(
                "unsupported relocation encoding {name}"
            )));
        }
    }
    Ok(())
}

fn apply_dynamic(
    regions: &mut [WorkingRegion],
    dynamic: &DynamicInfo,
    hints: TableHints,
    image_base: u64,
    resolver: &impl SymbolResolver,
) -> Result<(), PrepareError> {
    let rela = relocation_range(dynamic, DT_RELA, DT_RELASZ, "RELA")?;
    let plt = relocation_range(dynamic, DT_JMPREL, DT_PLTRELSZ, "PLT RELA")?;
    if plt.is_some() && dynamic.values.get(&DT_PLTREL).copied() != Some(DT_RELA as u64) {
        return Err(PrepareError::new("DT_PLTREL is not DT_RELA"));
    }
    if let Some(entry_size) = dynamic.values.get(&DT_RELAENT)
        && *entry_size != ELF64_RELA_SIZE
    {
        return Err(PrepareError::new("DT_RELAENT is not 24 bytes"));
    }
    let mut records = Vec::new();
    let mut seen_entries = BTreeSet::new();
    let mut targets = BTreeSet::new();
    for (name, range) in [("RELA", rela), ("PLT RELA", plt)] {
        if let Some((offset, size)) = range {
            validate_logical_range(regions, offset, size, name)?;
            let count = size / ELF64_RELA_SIZE;
            if count > MAX_RELOCATIONS as u64 {
                return Err(PrepareError::new(format!(
                    "{name} exceeds the relocation limit"
                )));
            }
            for index in 0..count {
                let entry_offset =
                    checked_add(offset, index * ELF64_RELA_SIZE, "relocation entry offset")?;
                if !seen_entries.insert(entry_offset) {
                    continue;
                }
                let bytes =
                    read(regions, entry_offset, ELF64_RELA_SIZE, true).ok_or_else(|| {
                        PrepareError::new(format!("{name} relocation {index} is outside memory"))
                    })?;
                let target = read_u64(bytes, 0);
                let info = read_u64(bytes, 8);
                let addend = read_i64(bytes, 16);
                if !target.is_multiple_of(8) {
                    return Err(PrepareError::new(format!(
                        "{name} relocation {index} target is unaligned"
                    )));
                }
                let target_region = find_region(regions, target, 8, true).ok_or_else(|| {
                    PrepareError::new(format!(
                        "{name} relocation {index} target is outside logical memory"
                    ))
                })?;
                if target_region.permissions.is_executable() {
                    return Err(PrepareError::new(format!(
                        "{name} relocation {index} targets executable memory"
                    )));
                }
                let page_offset = target % PAGE_SIZE;
                if page_offset + 8 > PAGE_SIZE {
                    return Err(PrepareError::new(format!(
                        "{name} relocation {index} crosses a page"
                    )));
                }
                if !targets.insert(target) {
                    return Err(PrepareError::new(format!(
                        "duplicate relocation target {target:#x}"
                    )));
                }
                records.push(Relocation {
                    target,
                    symbol: (info >> 32) as u32,
                    kind: info as u32,
                    addend,
                    description: format!("{name} relocation {index}"),
                });
            }
        }
    }
    if let Some(relative_count) = dynamic.values.get(&DT_RELACOUNT) {
        let regular_count = rela.map_or(0, |(_, size)| size / ELF64_RELA_SIZE);
        if *relative_count > regular_count {
            return Err(PrepareError::new(
                "DT_RELACOUNT exceeds the regular RELA table",
            ));
        }
    }
    if records.is_empty() {
        return Ok(());
    }
    let tables = SymbolTables::parse(regions, dynamic, hints, &records)?;
    let mut writes = Vec::with_capacity(records.len());
    for relocation in &records {
        let value = match relocation.kind {
            R_AARCH64_RELATIVE => {
                if relocation.symbol != 0 {
                    return Err(PrepareError::new(format!(
                        "{} has a nonzero symbol index",
                        relocation.description
                    )));
                }
                checked_signed_add(image_base, relocation.addend, &relocation.description)?
            }
            R_AARCH64_ABS64 => {
                let symbol = tables.resolve(
                    relocation.symbol,
                    image_base,
                    resolver,
                    &relocation.description,
                )?;
                checked_signed_add(symbol, relocation.addend, &relocation.description)?
            }
            R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT => {
                if relocation.addend != 0 {
                    return Err(PrepareError::new(format!(
                        "{} has a nonzero addend",
                        relocation.description
                    )));
                }
                tables.resolve(
                    relocation.symbol,
                    image_base,
                    resolver,
                    &relocation.description,
                )?
            }
            kind => {
                return Err(PrepareError::new(format!(
                    "{} has unsupported AArch64 type {kind}",
                    relocation.description
                )));
            }
        };
        writes.push((relocation.target, value));
    }
    for (target, value) in writes {
        write_u64(regions, target, value)?;
    }
    Ok(())
}

fn relocation_range(
    dynamic: &DynamicInfo,
    address_tag: i64,
    size_tag: i64,
    name: &str,
) -> Result<Option<(u64, u64)>, PrepareError> {
    match (
        dynamic.values.get(&address_tag),
        dynamic.values.get(&size_tag),
    ) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(PrepareError::new(format!(
            "{name} address and size tags must appear together"
        ))),
        (Some(address), Some(size)) => {
            if size % ELF64_RELA_SIZE != 0 {
                return Err(PrepareError::new(format!(
                    "{name} size is not a multiple of 24"
                )));
            }
            Ok(Some((*address, *size)))
        }
    }
}

struct Relocation {
    target: u64,
    symbol: u32,
    kind: u32,
    addend: i64,
    description: String,
}

struct SymbolTables {
    strings: Vec<u8>,
    symbols: Vec<ElfSymbol>,
}

#[derive(Clone, Copy)]
struct ElfSymbol {
    name: u32,
    info: u8,
    other: u8,
    section: u16,
    value: u64,
    size: u64,
}

struct CollectedDefinition {
    name: Vec<u8>,
    address: u64,
    strong: bool,
}

fn collect_nso_definitions(
    image: &NsoImage,
    image_base: u64,
) -> Result<Vec<CollectedDefinition>, PrepareError> {
    let Some(mod0_offset) = image.mod0().map(Mod0Metadata::header_offset) else {
        return Ok(Vec::new());
    };
    let regions = materialize(image.executable())?;
    let dynamic = parse_dynamic(&regions, parse_mod0(&regions, mod0_offset)?.dynamic_offset)?;
    if !dynamic.values.contains_key(&DT_SYMTAB) {
        return Ok(Vec::new());
    }
    let read_only = image
        .executable()
        .segments()
        .iter()
        .find(|segment| segment.kind() == crate::ExecutableSegmentKind::ReadOnly)
        .ok_or_else(|| PrepareError::new("NSO has no read-only segment"))?;
    let hints = TableHints {
        strings: Some((
            checked_add(
                read_only.memory_offset(),
                image.metadata().dynamic_string_table().offset(),
                "NSO string-table offset",
            )?,
            image.metadata().dynamic_string_table().size(),
        )),
        symbols: Some((
            checked_add(
                read_only.memory_offset(),
                image.metadata().dynamic_symbol_table().offset(),
                "NSO symbol-table offset",
            )?,
            image.metadata().dynamic_symbol_table().size(),
        )),
    };
    let tables = SymbolTables::parse(&regions, &dynamic, hints, &[])?;
    let mut definitions = Vec::new();
    for (index, symbol) in tables.symbols.iter().enumerate().skip(1) {
        if symbol.section == 0 {
            continue;
        }
        let binding = symbol.info >> 4;
        let visibility = symbol.other & 3;
        if visibility == 1 || visibility == 2 || binding == 0 {
            continue;
        }
        if !matches!(binding, 1 | 2) {
            return Err(PrepareError::new(format!(
                "defined symbol {index} has unsupported binding {binding}"
            )));
        }
        validate_logical_range(&regions, symbol.value, 1, "defined symbol")?;
        let name = tables.name(index as u32)?.to_vec();
        if name.is_empty() {
            continue;
        }
        definitions.push(CollectedDefinition {
            name,
            address: checked_add(image_base, symbol.value, "defined symbol address")?,
            strong: binding == 1,
        });
    }
    Ok(definitions)
}

impl SymbolTables {
    fn name(&self, index: u32) -> Result<&[u8], PrepareError> {
        let symbol = self
            .symbols
            .get(index as usize)
            .ok_or_else(|| PrepareError::new(format!("invalid symbol {index}")))?;
        let tail = self.strings.get(symbol.name as usize..).ok_or_else(|| {
            PrepareError::new(format!("symbol {index} name offset is outside DT_STRTAB"))
        })?;
        let end = tail.iter().position(|byte| *byte == 0).ok_or_else(|| {
            PrepareError::new(format!("symbol {index} name has no bounded terminator"))
        })?;
        Ok(&tail[..end])
    }

    fn parse(
        regions: &[WorkingRegion],
        dynamic: &DynamicInfo,
        hints: TableHints,
        relocations: &[Relocation],
    ) -> Result<Self, PrepareError> {
        let strtab = required(dynamic, DT_STRTAB, "DT_STRTAB")?;
        let strsz = required(dynamic, DT_STRSZ, "DT_STRSZ")?;
        let symtab = required(dynamic, DT_SYMTAB, "DT_SYMTAB")?;
        let syment = required(dynamic, DT_SYMENT, "DT_SYMENT")?;
        if syment != ELF64_SYM_SIZE {
            return Err(PrepareError::new("DT_SYMENT is not 24 bytes"));
        }
        validate_hint(hints.strings, strtab, strsz, "string table")?;
        validate_logical_range(regions, strtab, strsz, "dynamic string table")?;
        let strings = read(regions, strtab, strsz, true)
            .ok_or_else(|| PrepareError::new("dynamic string table crosses a mapping"))?
            .to_vec();
        let minimum = relocations
            .iter()
            .map(|record| u64::from(record.symbol) + 1)
            .max()
            .unwrap_or(1);
        let hash_count = symbol_count_from_hash(regions, dynamic)?;
        let hint_count = match hints.symbols {
            Some((offset, size)) if size != 0 => {
                if offset != symtab || size % ELF64_SYM_SIZE != 0 {
                    return Err(PrepareError::new(
                        "loader symbol-table metadata conflicts with DT_SYMTAB",
                    ));
                }
                Some(size / ELF64_SYM_SIZE)
            }
            _ => None,
        };
        let count = hash_count.or(hint_count).unwrap_or(minimum);
        if count < minimum || count > MAX_RELOCATIONS as u64 {
            return Err(PrepareError::new(
                "symbol-table bound does not cover relocation references",
            ));
        }
        let bytes_len = count
            .checked_mul(ELF64_SYM_SIZE)
            .ok_or_else(|| PrepareError::new("symbol table size overflows"))?;
        validate_logical_range(regions, symtab, bytes_len, "dynamic symbol table")?;
        let bytes = read(regions, symtab, bytes_len, true)
            .ok_or_else(|| PrepareError::new("dynamic symbol table crosses a mapping"))?;
        let mut symbols = Vec::with_capacity(count as usize);
        for chunk in bytes.chunks_exact(ELF64_SYM_SIZE as usize) {
            symbols.push(ElfSymbol {
                name: read_u32(chunk, 0),
                info: chunk[4],
                other: chunk[5],
                section: u16::from_le_bytes(chunk[6..8].try_into().expect("fixed symbol field")),
                value: read_u64(chunk, 8),
                size: read_u64(chunk, 16),
            });
        }
        Ok(Self { strings, symbols })
    }

    fn resolve(
        &self,
        index: u32,
        image_base: u64,
        resolver: &impl SymbolResolver,
        context: &str,
    ) -> Result<u64, PrepareError> {
        let symbol = self.symbols.get(index as usize).ok_or_else(|| {
            PrepareError::new(format!("{context} references invalid symbol {index}"))
        })?;
        let binding = symbol.info >> 4;
        let symbol_type = symbol.info & 0xf;
        let visibility = symbol.other & 3;
        if symbol.section != 0 {
            return checked_add(image_base, symbol.value, "defined symbol address");
        }
        if binding == 0 {
            return Err(PrepareError::new(format!(
                "{context} references an undefined local symbol"
            )));
        }
        if !matches!(binding, 1 | 2) {
            return Err(PrepareError::new(format!(
                "{context} references a symbol with unsupported binding {binding}"
            )));
        }
        if visibility != 0 {
            return Err(PrepareError::new(format!(
                "{context} references an undefined non-default-visibility symbol"
            )));
        }
        let name = self.name(index)?;
        let external = ExternalSymbol {
            name,
            binding,
            symbol_type,
            visibility,
            size: symbol.size,
        };
        match resolver.resolve(external) {
            SymbolResolution::Address(address) => Ok(address),
            SymbolResolution::Unresolved if binding == 2 => Ok(0),
            SymbolResolution::Unresolved => Err(PrepareError::new(format!(
                "{context} has unresolved symbol {}",
                String::from_utf8_lossy(name)
            ))),
        }
    }
}

fn symbol_count_from_hash(
    regions: &[WorkingRegion],
    dynamic: &DynamicInfo,
) -> Result<Option<u64>, PrepareError> {
    if let Some(offset) = dynamic.values.get(&DT_HASH) {
        let header = read(regions, *offset, 8, true)
            .ok_or_else(|| PrepareError::new("DT_HASH header is outside logical memory"))?;
        let buckets = u64::from(read_u32(header, 0));
        let chains = u64::from(read_u32(header, 4));
        let size = 8_u64
            .checked_add(
                buckets
                    .checked_add(chains)
                    .and_then(|v| v.checked_mul(4))
                    .ok_or_else(|| PrepareError::new("DT_HASH size overflows"))?,
            )
            .ok_or_else(|| PrepareError::new("DT_HASH size overflows"))?;
        validate_logical_range(regions, *offset, size, "DT_HASH")?;
        return Ok(Some(chains));
    }
    if let Some(offset) = dynamic.values.get(&DT_GNU_HASH) {
        let header = read(regions, *offset, 16, true)
            .ok_or_else(|| PrepareError::new("DT_GNU_HASH header is outside logical memory"))?;
        let buckets = u64::from(read_u32(header, 0));
        let symbol_base = u64::from(read_u32(header, 4));
        let bloom_words = u64::from(read_u32(header, 8));
        let buckets_offset = checked_add(
            *offset,
            checked_add(
                16,
                bloom_words
                    .checked_mul(8)
                    .ok_or_else(|| PrepareError::new("GNU hash bloom size overflows"))?,
                "GNU hash bloom",
            )?,
            "GNU hash buckets",
        )?;
        let bucket_bytes = buckets
            .checked_mul(4)
            .ok_or_else(|| PrepareError::new("GNU hash bucket size overflows"))?;
        let bucket_data = read(regions, buckets_offset, bucket_bytes, true)
            .ok_or_else(|| PrepareError::new("GNU hash buckets are outside logical memory"))?;
        let mut maximum_bucket = 0;
        for index in 0..buckets as usize {
            maximum_bucket = maximum_bucket.max(u64::from(read_u32(bucket_data, index * 4)));
        }
        if maximum_bucket == 0 {
            return Ok(Some(symbol_base));
        }
        if maximum_bucket < symbol_base {
            return Err(PrepareError::new(
                "GNU hash bucket precedes its symbol base",
            ));
        }
        let chains_offset = checked_add(buckets_offset, bucket_bytes, "GNU hash chains")?;
        let mut current = maximum_bucket;
        for _ in 0..MAX_RELOCATIONS {
            let chain_index = current
                .checked_sub(symbol_base)
                .ok_or_else(|| PrepareError::new("GNU hash chain underflows"))?;
            let address = checked_add(
                chains_offset,
                chain_index
                    .checked_mul(4)
                    .ok_or_else(|| PrepareError::new("GNU hash chain offset overflows"))?,
                "GNU hash chain address",
            )?;
            let value = read(regions, address, 4, true).ok_or_else(|| {
                PrepareError::new("GNU hash chain is unterminated in logical memory")
            })?;
            current += 1;
            if read_u32(value, 0) & 1 != 0 {
                return Ok(Some(current));
            }
        }
        return Err(PrepareError::new("GNU hash chain exceeds the symbol limit"));
    }
    Ok(None)
}

fn required(dynamic: &DynamicInfo, tag: i64, name: &str) -> Result<u64, PrepareError> {
    dynamic
        .values
        .get(&tag)
        .copied()
        .ok_or_else(|| PrepareError::new(format!("{name} is required by symbol relocations")))
}

fn validate_hint(
    hint: Option<(u64, u64)>,
    offset: u64,
    size: u64,
    name: &str,
) -> Result<(), PrepareError> {
    if let Some((hint_offset, hint_size)) = hint
        && hint_size != 0
        && (hint_offset != offset || hint_size != size)
    {
        return Err(PrepareError::new(format!(
            "loader {name} metadata conflicts with dynamic tags"
        )));
    }
    Ok(())
}

fn validate_logical_range(
    regions: &[WorkingRegion],
    offset: u64,
    size: u64,
    name: &str,
) -> Result<(), PrepareError> {
    if size == 0 {
        return Ok(());
    }
    find_region(regions, offset, size, true)
        .ok_or_else(|| PrepareError::new(format!("{name} is outside one logical mapping")))?;
    Ok(())
}

fn find_region(
    regions: &[WorkingRegion],
    offset: u64,
    size: u64,
    logical: bool,
) -> Option<&WorkingRegion> {
    let end = offset.checked_add(size)?;
    regions.iter().find(|region| {
        let length = if logical {
            region.logical_size
        } else {
            region.bytes.len() as u64
        };
        let region_end = region.offset.checked_add(length);
        offset >= region.offset && region_end.is_some_and(|value| end <= value)
    })
}

fn read(regions: &[WorkingRegion], offset: u64, size: u64, logical: bool) -> Option<&[u8]> {
    let region = find_region(regions, offset, size, logical)?;
    let start = usize::try_from(offset.checked_sub(region.offset)?).ok()?;
    let end = start.checked_add(usize::try_from(size).ok()?)?;
    region.bytes.get(start..end)
}

fn write_u64(regions: &mut [WorkingRegion], offset: u64, value: u64) -> Result<(), PrepareError> {
    let region = regions
        .iter_mut()
        .find(|region| {
            offset >= region.offset
                && offset
                    .checked_add(8)
                    .is_some_and(|end| end <= region.offset + region.logical_size)
        })
        .ok_or_else(|| PrepareError::new("relocation target disappeared during commit"))?;
    let start = usize::try_from(offset - region.offset)
        .map_err(|_| PrepareError::new("relocation host index is not representable"))?;
    region.bytes[start..start + 8].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn checked_add(left: u64, right: u64, name: &str) -> Result<u64, PrepareError> {
    left.checked_add(right)
        .ok_or_else(|| PrepareError::new(format!("{name} overflows")))
}

fn checked_signed_add(base: u64, addend: i64, name: &str) -> Result<u64, PrepareError> {
    if addend >= 0 {
        base.checked_add(addend as u64)
    } else {
        base.checked_sub(addend.unsigned_abs())
    }
    .ok_or_else(|| PrepareError::new(format!("{name} address overflows")))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated u32 field"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated u64 field"),
    )
}

fn read_i64(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated i64 field"),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_storage::{FormatLoader, Storage, StorageError, StorageRef};

    use super::*;
    use crate::{ExecutableSegmentKind, MemoryPermissions, NroLoader, NsoLoader};

    #[derive(Debug)]
    struct Bytes(Vec<u8>);

    impl Storage for Bytes {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.0.len() as u64)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(self.0.get(start..end).ok_or(StorageError::OutOfBounds)?);
            Ok(())
        }
    }

    fn storage(bytes: Vec<u8>) -> StorageRef {
        Arc::new(Bytes(bytes))
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i64(bytes: &mut [u8], offset: usize, value: i64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn dynamic(bytes: &mut [u8], index: usize, tag: i64, value: u64) {
        let offset = 0x100 + index * 16;
        put_i64(bytes, offset, tag);
        put_u64(bytes, offset + 8, value);
    }

    fn symbol(bytes: &mut [u8], index: usize, name: u32, info: u8, section: u16, value: u64) {
        let offset = 0x500 + index * 24;
        put_u32(bytes, offset, name);
        bytes[offset + 4] = info;
        bytes[offset + 6..offset + 8].copy_from_slice(&section.to_le_bytes());
        put_u64(bytes, offset + 8, value);
    }

    fn relocation(
        bytes: &mut [u8],
        offset: usize,
        target: u64,
        symbol: u32,
        kind: u32,
        addend: i64,
    ) {
        put_u64(bytes, offset, target);
        put_u64(
            bytes,
            offset + 8,
            (u64::from(symbol) << 32) | u64::from(kind),
        );
        put_i64(bytes, offset + 16, addend);
    }

    fn fixture(execute_only: bool) -> (ExecutableImage, TableHints) {
        let mut text = vec![0; 0x1000];
        text[0x100..0x104].copy_from_slice(b"MOD0");
        put_i32(&mut text, 0x104, 0x1000);
        put_i32(&mut text, 0x108, 0x2f00);
        put_i32(&mut text, 0x10c, 0x3700);
        put_i32(&mut text, 0x110, 0x0f00);
        put_i32(&mut text, 0x114, 0x0f00);
        put_i32(&mut text, 0x118, 0x1f00);

        let strings = b"\0external\0weak\0local\0";
        let mut read_only = vec![0; 0x1000];
        for (index, (tag, value)) in [
            (DT_STRTAB, 0x1400),
            (DT_STRSZ, strings.len() as u64),
            (DT_SYMTAB, 0x1500),
            (DT_SYMENT, ELF64_SYM_SIZE),
            (DT_HASH, 0x1600),
            (DT_RELA, 0x1700),
            (DT_RELASZ, 3 * ELF64_RELA_SIZE),
            (DT_RELAENT, ELF64_RELA_SIZE),
            (DT_RELACOUNT, 1),
            (DT_JMPREL, 0x1800),
            (DT_PLTRELSZ, ELF64_RELA_SIZE),
            (DT_PLTREL, DT_RELA as u64),
            (DT_NULL, 0),
        ]
        .into_iter()
        .enumerate()
        {
            dynamic(&mut read_only, index, tag, value);
        }
        read_only[0x400..0x400 + strings.len()].copy_from_slice(strings);
        symbol(&mut read_only, 0, 0, 0, 0, 0);
        symbol(&mut read_only, 1, 1, 0x10, 0, 0);
        symbol(&mut read_only, 2, 10, 0x20, 0, 0);
        symbol(&mut read_only, 3, 15, 0x00, 1, 0x1200);
        put_u32(&mut read_only, 0x600, 1);
        put_u32(&mut read_only, 0x604, 4);
        relocation(&mut read_only, 0x700, 0x2000, 0, R_AARCH64_RELATIVE, -0x10);
        relocation(&mut read_only, 0x718, 0x2008, 3, R_AARCH64_ABS64, -8);
        relocation(&mut read_only, 0x730, 0x2010, 2, R_AARCH64_GLOB_DAT, 0);
        relocation(&mut read_only, 0x800, 0x2018, 1, R_AARCH64_JUMP_SLOT, 0);

        let text_permissions = if execute_only {
            MemoryPermissions::EXECUTE
        } else {
            MemoryPermissions::from_bits(MemoryPermissions::READ.0 | MemoryPermissions::EXECUTE.0)
        };
        let data = vec![0xaa; 0x800];
        let segments = vec![
            ExecutableSegment::new(
                ExecutableSegmentKind::Text,
                0,
                0x1000,
                0x1000,
                0x1000,
                text_permissions,
                storage(text),
            ),
            ExecutableSegment::new(
                ExecutableSegmentKind::ReadOnly,
                0x1000,
                0x1000,
                0x1000,
                0x1000,
                MemoryPermissions::READ,
                storage(read_only),
            ),
            ExecutableSegment::new(
                ExecutableSegmentKind::Data,
                0x2000,
                0x800,
                0x1800,
                0x2000,
                MemoryPermissions::from_bits(
                    MemoryPermissions::READ.0 | MemoryPermissions::WRITE.0,
                ),
                storage(data),
            ),
        ];
        (
            ExecutableImage::new(ExecutableFormat::Nso, 0, [0x42; 32], segments),
            TableHints {
                strings: Some((0x1400, strings.len() as u64)),
                symbols: Some((0x1500, 4 * ELF64_SYM_SIZE)),
            },
        )
    }

    fn resolver(symbol: ExternalSymbol<'_>) -> SymbolResolution {
        match symbol.name() {
            b"external" => SymbolResolution::Address(0x7777_0000),
            _ => SymbolResolution::Unresolved,
        }
    }

    fn segment_bytes(segment: &ExecutableSegment) -> Vec<u8> {
        let mut bytes = vec![0; segment.file_size() as usize];
        segment.storage().read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn synthetic_nro() -> Vec<u8> {
        let (image, _) = fixture(false);
        let mut bytes = vec![0; 0x2800];
        bytes[..0x1000].copy_from_slice(&segment_bytes(&image.segments()[0]));
        bytes[0x1000..0x2000].copy_from_slice(&segment_bytes(&image.segments()[1]));
        bytes[0x2000..0x2800].copy_from_slice(&segment_bytes(&image.segments()[2]));
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        put_u32(&mut bytes, 0x04, 0x100);
        put_u32(&mut bytes, 0x18, 0x2800);
        put_u32(&mut bytes, 0x20, 0);
        put_u32(&mut bytes, 0x24, 0x1000);
        put_u32(&mut bytes, 0x28, 0x1000);
        put_u32(&mut bytes, 0x2c, 0x1000);
        put_u32(&mut bytes, 0x30, 0x2000);
        put_u32(&mut bytes, 0x34, 0x800);
        put_u32(&mut bytes, 0x38, 0x1000);
        bytes[0x40..0x60].fill(0x42);
        put_u32(&mut bytes, 0x70, 0x1400);
        put_u32(&mut bytes, 0x74, 21);
        put_u32(&mut bytes, 0x78, 0x1500);
        put_u32(&mut bytes, 0x7c, 4 * ELF64_SYM_SIZE as u32);
        bytes
    }

    fn synthetic_nso() -> Vec<u8> {
        let (image, _) = fixture(true);
        let mut text = segment_bytes(&image.segments()[0]);
        put_u32(&mut text, 4, 0x100);
        let read_only = segment_bytes(&image.segments()[1]);
        let data = segment_bytes(&image.segments()[2]);
        let mut bytes = vec![0; 0x2900];
        bytes[..4].copy_from_slice(b"NSO0");
        put_u32(&mut bytes, 0x0c, 1 << 6);
        for (descriptor, file, memory, size) in [
            (0x10, 0x100, 0, 0x1000),
            (0x20, 0x1100, 0x1000, 0x1000),
            (0x30, 0x2100, 0x2000, 0x800),
        ] {
            put_u32(&mut bytes, descriptor, file);
            put_u32(&mut bytes, descriptor + 4, memory);
            put_u32(&mut bytes, descriptor + 8, size);
        }
        put_u32(&mut bytes, 0x1c, 0x100);
        put_u32(&mut bytes, 0x2c, 0);
        put_u32(&mut bytes, 0x3c, 0x1000);
        bytes[0x40..0x60].fill(0x42);
        put_u32(&mut bytes, 0x60, 0x1000);
        put_u32(&mut bytes, 0x64, 0x1000);
        put_u32(&mut bytes, 0x68, 0x800);
        put_u32(&mut bytes, 0x90, 0x400);
        put_u32(&mut bytes, 0x94, 21);
        put_u32(&mut bytes, 0x98, 0x500);
        put_u32(&mut bytes, 0x9c, 4 * ELF64_SYM_SIZE as u32);
        bytes[0x100..0x1100].copy_from_slice(&text);
        bytes[0x1100..0x2100].copy_from_slice(&read_only);
        bytes[0x2100..0x2900].copy_from_slice(&data);
        bytes
    }

    fn nso_defining_external_with(binding: u8, visibility: u8) -> Vec<u8> {
        let mut bytes = synthetic_nso();
        // Dynamic symbol 1 lives at read-only offset 0x500. Make it a visible
        // strong definition in the text segment.
        let symbol = 0x1100 + 0x500 + ELF64_SYM_SIZE as usize;
        bytes[symbol + 4] = binding << 4;
        bytes[symbol + 5] = visibility;
        bytes[symbol + 6..symbol + 8].copy_from_slice(&1_u16.to_le_bytes());
        put_u64(&mut bytes, symbol + 8, 0x100);
        bytes
    }

    fn nso_defining_external() -> Vec<u8> {
        nso_defining_external_with(1, 0)
    }

    #[test]
    fn prepares_synthetic_nro_and_nso_through_public_entry_points() {
        let config = PreparationConfig {
            image_base: 0x7100_0000,
            address_limit: 0x8000_0000,
        };
        let nro = NroLoader::load(storage(synthetic_nro())).unwrap();
        let prepared_nro = nro.prepare(config, &resolver).unwrap();
        assert_eq!(prepared_nro.format(), ExecutableFormat::Nro);
        assert_eq!(
            prepared_nro.read_image(0x2018, 8).unwrap(),
            &0x7777_0000_u64.to_le_bytes()
        );

        let nso = NsoLoader::load(storage(synthetic_nso())).unwrap();
        let prepared_nso = nso.prepare(config, &resolver).unwrap();
        assert_eq!(prepared_nso.format(), ExecutableFormat::Nso);
        assert_eq!(
            prepared_nso.mappings()[0].permissions(),
            MemoryPermissions::EXECUTE
        );
        assert_eq!(
            prepared_nro.read_image(0x2000, 0x1800),
            prepared_nso.read_image(0x2000, 0x1800)
        );
    }

    #[test]
    fn batch_linking_collects_forward_definitions_before_relocation() {
        let consumer = NsoLoader::load(storage(synthetic_nso())).unwrap();
        let provider = NsoLoader::load(storage(nso_defining_external())).unwrap();
        let modules = [
            NsoBatchModule {
                image: &consumer,
                config: PreparationConfig {
                    image_base: 0x7100_0000,
                    address_limit: 0x7200_0000,
                },
            },
            NsoBatchModule {
                image: &provider,
                config: PreparationConfig {
                    image_base: 0x7110_0000,
                    address_limit: 0x7200_0000,
                },
            },
        ];
        let prepared = prepare_nso_batch(&modules, &[0, 1], &[]).unwrap();
        assert_eq!(prepared.len(), 2);
        assert_eq!(
            prepared[0].read_image(0x2018, 8).unwrap(),
            &(0x7110_0100_u64).to_le_bytes()
        );
    }

    #[test]
    fn batch_linking_rejects_duplicate_strong_definitions_atomically() {
        let first = NsoLoader::load(storage(nso_defining_external())).unwrap();
        let second = NsoLoader::load(storage(nso_defining_external())).unwrap();
        let modules = [
            NsoBatchModule {
                image: &first,
                config: PreparationConfig {
                    image_base: 0x7100_0000,
                    address_limit: 0x7200_0000,
                },
            },
            NsoBatchModule {
                image: &second,
                config: PreparationConfig {
                    image_base: 0x7110_0000,
                    address_limit: 0x7200_0000,
                },
            },
        ];
        let error = prepare_nso_batch(&modules, &[0, 1], &[]).unwrap_err();
        assert!(error.reason().contains("duplicate strong symbol external"));
    }

    #[test]
    fn batch_scope_applies_binding_visibility_runtime_and_unresolved_rules() {
        let consumer = NsoLoader::load(storage(synthetic_nso())).unwrap();
        let weak = NsoLoader::load(storage(nso_defining_external_with(2, 0))).unwrap();
        let strong = NsoLoader::load(storage(nso_defining_external())).unwrap();
        let configs = [0x7100_0000, 0x7110_0000, 0x7120_0000];
        let modules = [
            NsoBatchModule {
                image: &consumer,
                config: PreparationConfig {
                    image_base: configs[0],
                    address_limit: 0x7200_0000,
                },
            },
            NsoBatchModule {
                image: &weak,
                config: PreparationConfig {
                    image_base: configs[1],
                    address_limit: 0x7200_0000,
                },
            },
            NsoBatchModule {
                image: &strong,
                config: PreparationConfig {
                    image_base: configs[2],
                    address_limit: 0x7200_0000,
                },
            },
        ];
        let prepared = prepare_nso_batch(
            &modules,
            &[1, 2, 0],
            &[RuntimeExport {
                name: b"external",
                address: 0x6000_0000,
            }],
        )
        .unwrap();
        assert_eq!(
            prepared[0].read_image(0x2018, 8).unwrap(),
            &(configs[2] + 0x100).to_le_bytes()
        );

        let hidden = NsoLoader::load(storage(nso_defining_external_with(1, 2))).unwrap();
        let hidden_modules = [
            NsoBatchModule {
                image: &consumer,
                config: modules[0].config,
            },
            NsoBatchModule {
                image: &hidden,
                config: modules[1].config,
            },
        ];
        let prepared = prepare_nso_batch(
            &hidden_modules,
            &[1, 0],
            &[RuntimeExport {
                name: b"external",
                address: 0x6000_0000,
            }],
        )
        .unwrap();
        assert_eq!(
            prepared[0].read_image(0x2018, 8).unwrap(),
            &0x6000_0000_u64.to_le_bytes()
        );
        let error = prepare_nso_batch(&hidden_modules[..1], &[0], &[]).unwrap_err();
        assert!(error.reason().contains("unresolved symbol external"));
    }

    fn prepare_nro_bytes(bytes: Vec<u8>) -> Result<PreparedModule, PrepareError> {
        NroLoader::load(storage(bytes)).unwrap().prepare(
            PreparationConfig {
                image_base: 0x7100_0000,
                address_limit: 0x8000_0000,
            },
            &resolver,
        )
    }

    #[test]
    fn supports_gnu_hash_symbol_bounds() {
        let mut bytes = synthetic_nro();
        put_i64(&mut bytes, 0x1140, DT_GNU_HASH);
        bytes[0x1600..0x1628].fill(0);
        put_u32(&mut bytes, 0x1600, 1);
        put_u32(&mut bytes, 0x1604, 1);
        put_u32(&mut bytes, 0x1608, 1);
        put_u32(&mut bytes, 0x1618, 1);
        put_u32(&mut bytes, 0x161c, 0x100);
        put_u32(&mut bytes, 0x1620, 0x200);
        put_u32(&mut bytes, 0x1624, 0x301);
        let prepared = prepare_nro_bytes(bytes).unwrap();
        assert_eq!(
            prepared.read_image(0x2018, 8).unwrap(),
            &0x7777_0000_u64.to_le_bytes()
        );
    }

    #[test]
    fn rejects_adversarial_dynamic_and_relocation_records() {
        let cases: &[(usize, &[u8], &str)] = &[
            (0x1110, &DT_STRTAB.to_le_bytes(), "duplicated"),
            (
                0x11c0,
                &DT_REL.to_le_bytes(),
                "unsupported relocation encoding",
            ),
            (0x1178, &8_u64.to_le_bytes(), "DT_RELAENT"),
            (0x1700, &0_u64.to_le_bytes(), "targets executable memory"),
            (
                0x1708,
                &u64::from(999_u32).to_le_bytes(),
                "unsupported AArch64 type",
            ),
        ];
        for (offset, replacement, expected) in cases {
            let mut bytes = synthetic_nro();
            bytes[*offset..*offset + replacement.len()].copy_from_slice(replacement);
            let error = prepare_nro_bytes(bytes).unwrap_err();
            assert!(
                error.reason().contains(expected),
                "expected {expected:?} in {:?}",
                error.reason()
            );
        }
    }

    #[test]
    fn applies_all_supported_relocations_and_seals_exact_mappings() {
        for base in [0x7100_0000, 0x7200_0000] {
            let (image, hints) = fixture(true);
            let prepared = prepare(
                &image,
                Some(0x100),
                hints,
                PreparationConfig {
                    image_base: base,
                    address_limit: 0x8000_0000,
                },
                &resolver,
            )
            .unwrap();
            assert_eq!(prepared.entry_address(), base);
            assert_eq!(prepared.image_extent(), 0x4000);
            assert_eq!(prepared.mappings().len(), 3);
            assert_eq!(
                prepared.mappings()[0].permissions(),
                MemoryPermissions::EXECUTE
            );
            assert_eq!(
                prepared.read_image(0x2000, 8).unwrap(),
                &(base - 0x10).to_le_bytes()
            );
            assert_eq!(
                prepared.read_image(0x2008, 8).unwrap(),
                &(base + 0x1200 - 8).to_le_bytes()
            );
            assert_eq!(
                prepared.read_image(0x2010, 8).unwrap(),
                &0_u64.to_le_bytes()
            );
            assert_eq!(
                prepared.read_image(0x2018, 8).unwrap(),
                &0x7777_0000_u64.to_le_bytes()
            );
            assert!(
                prepared
                    .read_image(0x2800, 0x1000)
                    .unwrap()
                    .iter()
                    .all(|byte| *byte == 0)
            );
            assert!(
                prepared
                    .mappings()
                    .iter()
                    .all(|mapping| !(mapping.permissions().is_writable()
                        && mapping.permissions().is_executable()))
            );
        }
    }

    #[test]
    fn failure_is_atomic_and_does_not_mutate_loader_storage() {
        let (image, hints) = fixture(false);
        let before = {
            let mut bytes = vec![0; 0x800];
            image.segments()[2]
                .storage()
                .read_at(0, &mut bytes)
                .unwrap();
            bytes
        };
        let error = prepare(
            &image,
            Some(0x100),
            hints,
            PreparationConfig {
                image_base: 0x7100_0000,
                address_limit: 0x8000_0000,
            },
            &|_: ExternalSymbol<'_>| SymbolResolution::Unresolved,
        )
        .unwrap_err();
        assert!(error.reason().contains("external"));
        let mut after = vec![0; 0x800];
        image.segments()[2]
            .storage()
            .read_at(0, &mut after)
            .unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn rejects_bad_placement_and_corrupt_dynamic_metadata() {
        let (image, hints) = fixture(false);
        let error = prepare(
            &image,
            Some(0x100),
            hints,
            PreparationConfig {
                image_base: 1,
                address_limit: u64::MAX,
            },
            &resolver,
        )
        .unwrap_err();
        assert!(error.reason().contains("page aligned"));

        let error = prepare(
            &image,
            Some(0x100),
            hints,
            PreparationConfig {
                image_base: !(PAGE_SIZE - 1),
                address_limit: u64::MAX,
            },
            &resolver,
        )
        .unwrap_err();
        assert!(error.reason().contains("guest image extent"));

        let error = prepare(
            &image,
            Some(0x100),
            TableHints {
                strings: Some((0x1401, 4)),
                symbols: hints.symbols,
            },
            PreparationConfig {
                image_base: 0x7100_0000,
                address_limit: 0x8000_0000,
            },
            &resolver,
        )
        .unwrap_err();
        assert!(error.reason().contains("string table metadata conflicts"));
    }

    #[test]
    fn read_helpers_never_cross_mapping_boundaries() {
        let (image, hints) = fixture(false);
        let prepared = prepare(
            &image,
            Some(0x100),
            hints,
            PreparationConfig {
                image_base: 0x7100_0000,
                address_limit: 0x8000_0000,
            },
            &resolver,
        )
        .unwrap();
        assert!(prepared.mapping_at(0x7100_0000).is_some());
        assert!(prepared.mapping_at(0x7100_3000).is_some());
        assert!(prepared.read_image(0x0fff, 2).is_none());
        assert!(prepared.read_guest(0x8000_0000, 1).is_none());
    }
}
