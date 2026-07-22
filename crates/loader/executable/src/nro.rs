use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use nixe_loader_storage::{FormatLoader, LoadError, StorageRef, SubStorage};

use crate::{
    ExecutableFormat, ExecutableImage, ExecutableSegment, ExecutableSegmentKind, MemoryPermissions,
};

const HEADER_SIZE: u64 = 0x80;
const ASSET_HEADER_SIZE: u64 = 0x38;
const PAGE_SIZE: u64 = 0x1000;
const MOD0_HEADER_SIZE: u64 = 0x1c;

/// Loads Nintendo Relocatable Object (NRO) files.
///
/// NRO is the executable format normally used by Nintendo Switch homebrew. It
/// carries its code and data segments in one directly loadable file, unlike an
/// NSO, which is normally retrieved from an official title's ExeFS partition.
#[derive(Debug)]
pub struct NroLoader;

impl FormatLoader for NroLoader {
    type Output = NroImage;

    const FORMAT_NAME: &'static str = "NRO";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        parse_nro(storage)
    }
}

/// A validated image-relative byte range recorded in NRO metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NroRange {
    offset: u64,
    size: u64,
}

impl NroRange {
    const fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Returns the range offset relative to the executable image base.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the range size in bytes.
    pub const fn size(self) -> u64 {
        self.size
    }

    /// Returns whether the range contains no bytes.
    pub const fn is_empty(self) -> bool {
        self.size == 0
    }
}

/// NRO-specific header fields retained for relocation and runtime setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NroMetadata {
    version: u32,
    flags: u32,
    executable_size: u64,
    module_header_offset: u64,
    dso_handle_offset: u64,
    embedded_api_info: NroRange,
    dynamic_string_table: NroRange,
    dynamic_symbol_table: NroRange,
}

impl NroMetadata {
    /// Returns the preserved NRO format version/unknown field.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the preserved NRO flags/unknown field.
    pub const fn flags(&self) -> u32 {
        self.flags
    }

    /// Returns the declared size of the executable portion.
    pub const fn executable_size(&self) -> u64 {
        self.executable_size
    }

    /// Returns the image-relative standard `MOD0` header offset.
    ///
    /// Zero indicates that no standard module header was declared.
    pub const fn module_header_offset(&self) -> u64 {
        self.module_header_offset
    }

    /// Returns the image-relative DSO handle offset.
    pub const fn dso_handle_offset(&self) -> u64 {
        self.dso_handle_offset
    }

    /// Returns the embedded API-info range.
    pub const fn embedded_api_info(&self) -> NroRange {
        self.embedded_api_info
    }

    /// Returns the dynamic string-table range.
    pub const fn dynamic_string_table(&self) -> NroRange {
        self.dynamic_string_table
    }

    /// Returns the dynamic symbol-table range.
    pub const fn dynamic_symbol_table(&self) -> NroRange {
        self.dynamic_symbol_table
    }
}

struct AssetView {
    relative_offset: u64,
    size: u64,
    storage: StorageRef,
}

/// Optional homebrew assets appended to an NRO executable.
pub struct NroAssets {
    version: u32,
    icon: Option<AssetView>,
    nacp: Option<AssetView>,
    romfs: Option<AssetView>,
}

impl NroAssets {
    /// Returns the supported ASET version (currently zero).
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns a bounded, lazy view of the icon bytes when present.
    pub fn icon(&self) -> Option<&StorageRef> {
        self.icon.as_ref().map(|asset| &asset.storage)
    }

    /// Returns a bounded, lazy view of the NACP bytes when present.
    pub fn nacp(&self) -> Option<&StorageRef> {
        self.nacp.as_ref().map(|asset| &asset.storage)
    }

    /// Returns a bounded, lazy view of the embedded RomFS when present.
    pub fn romfs(&self) -> Option<&StorageRef> {
        self.romfs.as_ref().map(|asset| &asset.storage)
    }
}

impl Debug for NroAssets {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NroAssets")
            .field("version", &self.version)
            .field("icon", &asset_debug_range(&self.icon))
            .field("nacp", &asset_debug_range(&self.nacp))
            .field("romfs", &asset_debug_range(&self.romfs))
            .finish()
    }
}

fn asset_debug_range(asset: &Option<AssetView>) -> Option<(u64, u64)> {
    asset
        .as_ref()
        .map(|asset| (asset.relative_offset, asset.size))
}

/// Parsed NRO executable plus its format-specific metadata and optional assets.
#[derive(Debug)]
pub struct NroImage {
    executable: ExecutableImage,
    metadata: NroMetadata,
    assets: Option<NroAssets>,
}

impl NroImage {
    /// Returns the format-independent executable description.
    pub const fn executable(&self) -> &ExecutableImage {
        &self.executable
    }

    /// Returns NRO-specific metadata needed by later loading stages.
    pub const fn metadata(&self) -> &NroMetadata {
        &self.metadata
    }

    /// Returns the appended homebrew asset section when present.
    pub const fn assets(&self) -> Option<&NroAssets> {
        self.assets.as_ref()
    }

    /// Splits this result into its generic executable and NRO-specific parts.
    pub fn into_parts(self) -> (ExecutableImage, NroMetadata, Option<NroAssets>) {
        (self.executable, self.metadata, self.assets)
    }
}

#[derive(Clone, Copy)]
struct RawSegment {
    name: &'static str,
    kind: ExecutableSegmentKind,
    offset: u64,
    size: u64,
    memory_size: u64,
    mapping_size: u64,
    permissions: MemoryPermissions,
}

fn parse_nro(storage: StorageRef) -> Result<NroImage, LoadError> {
    let source_len = storage.len()?;
    if source_len < HEADER_SIZE {
        return Err(invalid("header is truncated"));
    }

    let mut header = [0_u8; HEADER_SIZE as usize];
    storage.read_at(0, &mut header)?;
    if &header[0x10..0x14] != b"NRO0" {
        return Err(invalid("expected NRO0 magic"));
    }

    let executable_size = u64::from(read_u32(&header, 0x18));
    if executable_size < HEADER_SIZE {
        return Err(invalid(
            "declared executable size is smaller than the header",
        ));
    }
    if executable_size > source_len {
        return Err(invalid("declared executable size exceeds the source"));
    }

    let text_offset = u64::from(read_u32(&header, 0x20));
    let text_size = u64::from(read_u32(&header, 0x24));
    let read_only_offset = u64::from(read_u32(&header, 0x28));
    let read_only_size = u64::from(read_u32(&header, 0x2c));
    let data_offset = u64::from(read_u32(&header, 0x30));
    let data_size = u64::from(read_u32(&header, 0x34));
    let bss_size = u64::from(read_u32(&header, 0x38));
    let data_memory_size = data_size
        .checked_add(bss_size)
        .ok_or_else(|| invalid("data plus BSS size overflows"))?;
    let data_mapping_size = checked_align_up(data_memory_size, PAGE_SIZE)
        .ok_or_else(|| invalid("aligned data plus BSS size overflows"))?;
    data_offset
        .checked_add(data_mapping_size)
        .ok_or_else(|| invalid("image memory extent overflows"))?;

    let segments = [
        RawSegment {
            name: "text",
            kind: ExecutableSegmentKind::Text,
            offset: text_offset,
            size: text_size,
            memory_size: text_size,
            mapping_size: text_size,
            permissions: MemoryPermissions::from_bits(
                MemoryPermissions::READ.0 | MemoryPermissions::EXECUTE.0,
            ),
        },
        RawSegment {
            name: "read-only",
            kind: ExecutableSegmentKind::ReadOnly,
            offset: read_only_offset,
            size: read_only_size,
            memory_size: read_only_size,
            mapping_size: read_only_size,
            permissions: MemoryPermissions::READ,
        },
        RawSegment {
            name: "data",
            kind: ExecutableSegmentKind::Data,
            offset: data_offset,
            size: data_size,
            memory_size: data_memory_size,
            mapping_size: data_mapping_size,
            permissions: MemoryPermissions::from_bits(
                MemoryPermissions::READ.0 | MemoryPermissions::WRITE.0,
            ),
        },
    ];

    validate_segments(&segments, executable_size)?;

    let module_header_offset = u64::from(read_u32(&header, 0x04));
    if module_header_offset != 0 {
        validate_range(
            module_header_offset,
            MOD0_HEADER_SIZE,
            executable_size,
            "module header",
        )?;
    }

    let embedded_api_info = NroRange::new(
        u64::from(read_u32(&header, 0x68)),
        u64::from(read_u32(&header, 0x6c)),
    );
    let dynamic_string_table = NroRange::new(
        u64::from(read_u32(&header, 0x70)),
        u64::from(read_u32(&header, 0x74)),
    );
    let dynamic_symbol_table = NroRange::new(
        u64::from(read_u32(&header, 0x78)),
        u64::from(read_u32(&header, 0x7c)),
    );
    validate_metadata_range(embedded_api_info, executable_size, "embedded API-info")?;
    validate_metadata_range(
        dynamic_string_table,
        executable_size,
        "dynamic string table",
    )?;
    validate_metadata_range(
        dynamic_symbol_table,
        executable_size,
        "dynamic symbol table",
    )?;

    let mut module_id = [0_u8; 32];
    module_id.copy_from_slice(&header[0x40..0x60]);

    let loaded_segments = segments
        .into_iter()
        .map(|segment| {
            let view: StorageRef = Arc::new(SubStorage::new(
                storage.clone(),
                segment.offset,
                segment.size,
            )?);
            Ok(ExecutableSegment::new(
                segment.kind,
                segment.offset,
                segment.size,
                segment.memory_size,
                segment.mapping_size,
                segment.permissions,
                view,
            ))
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let executable = ExecutableImage::new(ExecutableFormat::Nro, 0, module_id, loaded_segments);
    let metadata = NroMetadata {
        version: read_u32(&header, 0x14),
        flags: read_u32(&header, 0x1c),
        executable_size,
        module_header_offset,
        dso_handle_offset: u64::from(read_u32(&header, 0x60)),
        embedded_api_info,
        dynamic_string_table,
        dynamic_symbol_table,
    };
    let assets = if source_len == executable_size {
        None
    } else {
        Some(parse_assets(storage, executable_size, source_len)?)
    };

    Ok(NroImage {
        executable,
        metadata,
        assets,
    })
}

fn validate_segments(segments: &[RawSegment; 3], executable_size: u64) -> Result<(), LoadError> {
    let [text, read_only, data] = segments;

    if text.offset != 0 {
        return Err(invalid("text segment does not start at image offset zero"));
    }
    if text.size < HEADER_SIZE {
        return Err(invalid("text segment does not contain the fixed header"));
    }

    for segment in segments {
        if segment.offset % PAGE_SIZE != 0 {
            return Err(invalid(format!(
                "{} segment offset is not page-aligned",
                segment.name
            )));
        }
        validate_range(
            segment.offset,
            segment.size,
            executable_size,
            &format!("{} segment", segment.name),
        )?;
    }

    if text.memory_size % PAGE_SIZE != 0 {
        return Err(invalid("text segment size is not page-aligned"));
    }
    if read_only.memory_size % PAGE_SIZE != 0 {
        return Err(invalid("read-only segment size is not page-aligned"));
    }
    let text_end = checked_end(text.offset, text.size, "text segment")?;
    if read_only.offset < text_end {
        return Err(invalid("read-only segment overlaps the text segment"));
    }
    let read_only_end = checked_end(read_only.offset, read_only.size, "read-only segment")?;
    if data.offset < read_only_end {
        return Err(invalid("data segment overlaps the read-only segment"));
    }

    Ok(())
}

fn validate_metadata_range(
    range: NroRange,
    executable_size: u64,
    name: &str,
) -> Result<(), LoadError> {
    if !range.is_empty() {
        validate_range(range.offset, range.size, executable_size, name)?;
    }
    Ok(())
}

fn parse_assets(
    storage: StorageRef,
    asset_base: u64,
    source_len: u64,
) -> Result<NroAssets, LoadError> {
    let header_end = asset_base
        .checked_add(ASSET_HEADER_SIZE)
        .ok_or_else(|| invalid("ASET header range overflows"))?;
    if header_end > source_len {
        return Err(invalid("ASET header is truncated"));
    }

    let mut header = [0_u8; ASSET_HEADER_SIZE as usize];
    storage.read_at(asset_base, &mut header)?;
    if &header[..4] != b"ASET" {
        return Err(invalid("expected ASET magic"));
    }
    let version = read_u32(&header, 0x04);
    if version != 0 {
        return Err(invalid("unsupported ASET version"));
    }

    let raw_assets = [
        ("icon", read_u64(&header, 0x08), read_u64(&header, 0x10)),
        ("NACP", read_u64(&header, 0x18), read_u64(&header, 0x20)),
        ("RomFS", read_u64(&header, 0x28), read_u64(&header, 0x30)),
    ];
    let mut ranges = Vec::with_capacity(3);
    for (name, relative_offset, size) in raw_assets {
        if size == 0 {
            continue;
        }
        if relative_offset < ASSET_HEADER_SIZE {
            return Err(invalid(format!(
                "{name} asset starts inside the ASET header"
            )));
        }
        let absolute_offset = asset_base
            .checked_add(relative_offset)
            .ok_or_else(|| invalid(format!("{name} asset offset overflows")))?;
        let end = absolute_offset
            .checked_add(size)
            .ok_or_else(|| invalid(format!("{name} asset range overflows")))?;
        if end > source_len {
            return Err(invalid(format!("{name} asset is outside the source")));
        }
        ranges.push((name, absolute_offset, end));
    }

    ranges.sort_unstable_by_key(|(_, start, _)| *start);
    for pair in ranges.windows(2) {
        if pair[1].1 < pair[0].2 {
            return Err(invalid(format!(
                "{} and {} assets overlap",
                pair[0].0, pair[1].0
            )));
        }
    }

    Ok(NroAssets {
        version,
        icon: open_asset(storage.clone(), asset_base, raw_assets[0])?,
        nacp: open_asset(storage.clone(), asset_base, raw_assets[1])?,
        romfs: open_asset(storage, asset_base, raw_assets[2])?,
    })
}

fn open_asset(
    storage: StorageRef,
    asset_base: u64,
    asset: (&'static str, u64, u64),
) -> Result<Option<AssetView>, LoadError> {
    let (name, relative_offset, size) = asset;
    if size == 0 {
        return Ok(None);
    }
    let absolute_offset = asset_base
        .checked_add(relative_offset)
        .ok_or_else(|| invalid(format!("{name} asset offset overflows")))?;
    let view: StorageRef = Arc::new(SubStorage::new(storage, absolute_offset, size)?);
    Ok(Some(AssetView {
        relative_offset,
        size,
        storage: view,
    }))
}

fn validate_range(offset: u64, size: u64, limit: u64, name: &str) -> Result<(), LoadError> {
    let end = checked_end(offset, size, name)?;
    if end > limit {
        return Err(invalid(format!("{name} is outside the executable portion")));
    }
    Ok(())
}

fn checked_end(offset: u64, size: u64, name: &str) -> Result<u64, LoadError> {
    offset
        .checked_add(size)
        .ok_or_else(|| invalid(format!("{name} range overflows")))
}

fn checked_align_up(value: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed header field"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed header field"),
    )
}

fn invalid(reason: impl Into<String>) -> LoadError {
    LoadError::invalid(NroLoader::FORMAT_NAME, reason)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use nixe_loader_storage::{Storage, StorageError};

    use super::*;

    const EXECUTABLE_SIZE: usize = 0x2800;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            u64::try_from(self.0.len()).map_err(|_| StorageError::OutOfBounds)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            read_bytes(&self.0, offset, buffer)
        }
    }

    #[derive(Debug)]
    struct CountingStorage {
        bytes: Vec<u8>,
        bytes_read: Arc<AtomicU64>,
    }

    impl Storage for CountingStorage {
        fn len(&self) -> Result<u64, StorageError> {
            u64::try_from(self.bytes.len()).map_err(|_| StorageError::OutOfBounds)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            read_bytes(&self.bytes, offset, buffer)?;
            self.bytes_read.fetch_add(
                u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?,
                Ordering::Relaxed,
            );
            Ok(())
        }
    }

    fn read_bytes(bytes: &[u8], offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
        let end = start
            .checked_add(buffer.len())
            .ok_or(StorageError::OutOfBounds)?;
        let source = bytes.get(start..end).ok_or(StorageError::OutOfBounds)?;
        buffer.copy_from_slice(source);
        Ok(())
    }

    fn storage(bytes: Vec<u8>) -> StorageRef {
        Arc::new(VecStorage(bytes))
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn canonical_nro() -> Vec<u8> {
        let mut bytes = vec![0_u8; EXECUTABLE_SIZE];
        bytes[..0x1000].fill(0x11);
        bytes[0x1000..0x2000].fill(0x22);
        bytes[0x2000..0x2800].fill(0x33);

        put_u32(&mut bytes, 0x04, 0x100);
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        put_u32(&mut bytes, 0x14, 7);
        put_u32(&mut bytes, 0x18, EXECUTABLE_SIZE as u32);
        put_u32(&mut bytes, 0x1c, 0xa5a5_5a5a);
        put_u32(&mut bytes, 0x20, 0);
        put_u32(&mut bytes, 0x24, 0x1000);
        put_u32(&mut bytes, 0x28, 0x1000);
        put_u32(&mut bytes, 0x2c, 0x1000);
        put_u32(&mut bytes, 0x30, 0x2000);
        put_u32(&mut bytes, 0x34, 0x800);
        put_u32(&mut bytes, 0x38, 0x800);
        for (index, byte) in bytes[0x40..0x60].iter_mut().enumerate() {
            *byte = u8::try_from(index).unwrap();
        }
        put_u32(&mut bytes, 0x60, 0x2040);
        put_u32(&mut bytes, 0x68, 0x1100);
        put_u32(&mut bytes, 0x6c, 0x20);
        put_u32(&mut bytes, 0x70, 0x1200);
        put_u32(&mut bytes, 0x74, 0x30);
        put_u32(&mut bytes, 0x78, 0x1300);
        put_u32(&mut bytes, 0x7c, 0x48);
        bytes
    }

    fn append_assets(
        bytes: &mut Vec<u8>,
        icon: (u64, &[u8]),
        nacp: (u64, &[u8]),
        romfs: (u64, &[u8]),
        trailing_padding: usize,
    ) {
        let asset_base = bytes.len();
        let max_end = [icon, nacp, romfs]
            .into_iter()
            .filter(|(_, contents)| !contents.is_empty())
            .map(|(offset, contents)| usize::try_from(offset).unwrap() + contents.len())
            .max()
            .unwrap_or(0)
            .max(ASSET_HEADER_SIZE as usize);
        bytes.resize(asset_base + max_end + trailing_padding, 0xcc);
        bytes[asset_base..asset_base + 4].copy_from_slice(b"ASET");
        put_u32(bytes, asset_base + 4, 0);

        for (entry_offset, (relative_offset, contents)) in
            [0x08, 0x18, 0x28].into_iter().zip([icon, nacp, romfs])
        {
            put_u64(bytes, asset_base + entry_offset, relative_offset);
            put_u64(
                bytes,
                asset_base + entry_offset + 8,
                u64::try_from(contents.len()).unwrap(),
            );
            if !contents.is_empty() {
                let start = asset_base + usize::try_from(relative_offset).unwrap();
                bytes[start..start + contents.len()].copy_from_slice(contents);
            }
        }
    }

    fn load(bytes: Vec<u8>) -> Result<NroImage, LoadError> {
        NroLoader::load(storage(bytes))
    }

    fn read_all(source: &StorageRef) -> Vec<u8> {
        let len = usize::try_from(source.len().unwrap()).unwrap();
        let mut bytes = vec![0_u8; len];
        source.read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn assert_invalid(bytes: Vec<u8>, expected_reason: &str) {
        match load(bytes) {
            Err(LoadError::InvalidFormat { format, reason }) => {
                assert_eq!(format, "NRO");
                assert!(
                    reason.contains(expected_reason),
                    "expected {expected_reason:?} in {reason:?}"
                );
            }
            other => panic!("expected invalid NRO, got {other:?}"),
        }
    }

    #[test]
    fn loads_canonical_nro_and_exposes_segments() {
        let image = load(canonical_nro()).unwrap();
        let executable = image.executable();

        assert_eq!(executable.format(), ExecutableFormat::Nro);
        assert_eq!(executable.entry_offset(), 0);
        assert_eq!(
            executable.module_id(),
            &std::array::from_fn::<_, 32, _>(|index| u8::try_from(index).unwrap())
        );
        assert_eq!(executable.segments().len(), 3);

        let text = &executable.segments()[0];
        assert_eq!(text.kind(), ExecutableSegmentKind::Text);
        assert_eq!((text.memory_offset(), text.file_size()), (0, 0x1000));
        assert_eq!(text.memory_size(), 0x1000);
        assert_eq!(text.mapping_size(), 0x1000);
        assert!(text.permissions().is_readable());
        assert!(text.permissions().is_executable());
        assert!(!text.permissions().is_writable());
        assert_eq!(read_all(text.storage()).len(), 0x1000);

        let read_only = &executable.segments()[1];
        assert_eq!(read_only.kind(), ExecutableSegmentKind::ReadOnly);
        assert_eq!(
            (read_only.memory_offset(), read_only.file_size()),
            (0x1000, 0x1000)
        );
        assert_eq!(read_all(read_only.storage()), vec![0x22; 0x1000]);
        assert_eq!(read_only.permissions(), MemoryPermissions::READ);

        let data = &executable.segments()[2];
        assert_eq!(data.kind(), ExecutableSegmentKind::Data);
        assert_eq!((data.memory_offset(), data.file_size()), (0x2000, 0x800));
        assert_eq!(data.memory_size(), 0x1000);
        assert_eq!(data.mapping_size(), 0x1000);
        assert_eq!(read_all(data.storage()), vec![0x33; 0x800]);
        assert!(data.permissions().is_readable());
        assert!(data.permissions().is_writable());
        assert!(!data.permissions().is_executable());
        assert!(image.assets().is_none());
    }

    #[test]
    fn preserves_nro_metadata() {
        let image = load(canonical_nro()).unwrap();
        let metadata = image.metadata();

        assert_eq!(metadata.version(), 7);
        assert_eq!(metadata.flags(), 0xa5a5_5a5a);
        assert_eq!(metadata.executable_size(), EXECUTABLE_SIZE as u64);
        assert_eq!(metadata.module_header_offset(), 0x100);
        assert_eq!(metadata.dso_handle_offset(), 0x2040);
        assert_eq!(metadata.embedded_api_info(), NroRange::new(0x1100, 0x20));
        assert_eq!(metadata.dynamic_string_table(), NroRange::new(0x1200, 0x30));
        assert_eq!(metadata.dynamic_symbol_table(), NroRange::new(0x1300, 0x48));
    }

    #[test]
    fn accepts_segment_gaps_zero_module_id_and_legacy_module_header() {
        let mut bytes = canonical_nro();
        bytes.resize(0x4000, 0);
        put_u32(&mut bytes, 0x18, 0x4000);
        put_u32(&mut bytes, 0x28, 0x2000);
        put_u32(&mut bytes, 0x30, 0x3000);
        bytes[0x40..0x60].fill(0);
        put_u32(&mut bytes, 0x04, 0);

        let image = load(bytes).unwrap();
        assert_eq!(image.executable().module_id(), &[0; 32]);
        assert_eq!(image.metadata().module_header_offset(), 0);
        assert_eq!(image.executable().segments()[1].memory_offset(), 0x2000);
        assert_eq!(image.executable().segments()[2].memory_offset(), 0x3000);
    }

    #[test]
    fn rounds_unaligned_data_and_bss_for_mapping_without_changing_memory_size() {
        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x38, 0x7ff);

        let image = load(bytes).unwrap();
        let data = &image.executable().segments()[2];
        assert_eq!(data.file_size(), 0x800);
        assert_eq!(data.memory_size(), 0xfff);
        assert_eq!(data.mapping_size(), 0x1000);
    }

    #[test]
    fn loads_assets_in_any_physical_order_and_ignores_padding() {
        let mut bytes = canonical_nro();
        append_assets(
            &mut bytes,
            (0x90, b"icon"),
            (0x60, b"control"),
            (0x78, b"romfs"),
            0x20,
        );

        let image = load(bytes).unwrap();
        let assets = image.assets().unwrap();
        assert_eq!(assets.version(), 0);
        assert_eq!(read_all(assets.icon().unwrap()), b"icon");
        assert_eq!(read_all(assets.nacp().unwrap()), b"control");
        assert_eq!(read_all(assets.romfs().unwrap()), b"romfs");
    }

    #[test]
    fn zero_sized_assets_are_absent_and_ignore_their_offsets() {
        let mut bytes = canonical_nro();
        append_assets(&mut bytes, (u64::MAX, b""), (0x40, b"nacp"), (1, b""), 0);

        let image = load(bytes).unwrap();
        let assets = image.assets().unwrap();
        assert!(assets.icon().is_none());
        assert_eq!(read_all(assets.nacp().unwrap()), b"nacp");
        assert!(assets.romfs().is_none());
    }

    #[test]
    fn loading_is_lazy() {
        let bytes_read = Arc::new(AtomicU64::new(0));
        let source: StorageRef = Arc::new(CountingStorage {
            bytes: canonical_nro(),
            bytes_read: bytes_read.clone(),
        });

        let image = NroLoader::load(source).unwrap();
        assert_eq!(bytes_read.load(Ordering::Relaxed), HEADER_SIZE);

        let mut byte = [0_u8; 1];
        image.executable().segments()[2]
            .storage()
            .read_at(0, &mut byte)
            .unwrap();
        assert_eq!(byte, [0x33]);
        assert_eq!(bytes_read.load(Ordering::Relaxed), HEADER_SIZE + 1);
    }

    #[test]
    fn loading_assets_does_not_eagerly_read_the_romfs() {
        let romfs = vec![0x5a; 1024 * 1024];
        let mut bytes = canonical_nro();
        append_assets(&mut bytes, (0, b""), (0, b""), (0x40, &romfs), 0);
        let bytes_read = Arc::new(AtomicU64::new(0));
        let source: StorageRef = Arc::new(CountingStorage {
            bytes,
            bytes_read: bytes_read.clone(),
        });

        let image = NroLoader::load(source).unwrap();
        assert_eq!(
            bytes_read.load(Ordering::Relaxed),
            HEADER_SIZE + ASSET_HEADER_SIZE
        );

        let mut byte = [0_u8; 1];
        image
            .assets()
            .unwrap()
            .romfs()
            .unwrap()
            .read_at(romfs.len() as u64 - 1, &mut byte)
            .unwrap();
        assert_eq!(byte, [0x5a]);
        assert_eq!(
            bytes_read.load(Ordering::Relaxed),
            HEADER_SIZE + ASSET_HEADER_SIZE + 1
        );
    }

    #[test]
    fn rejects_truncated_or_invalid_fixed_header() {
        assert_invalid(vec![0; 0x7f], "header is truncated");

        let mut bytes = canonical_nro();
        bytes[0x10..0x14].copy_from_slice(b"nro0");
        assert_invalid(bytes, "expected NRO0 magic");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x18, 0x7f);
        assert_invalid(bytes, "smaller than the header");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x18, EXECUTABLE_SIZE as u32 + 1);
        assert_invalid(bytes, "exceeds the source");
    }

    #[test]
    fn rejects_invalid_segment_ranges_and_order() {
        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x20, 0x1000);
        assert_invalid(bytes, "does not start at image offset zero");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x24, 0x40);
        assert_invalid(bytes, "does not contain the fixed header");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x28, 0x1001);
        assert_invalid(bytes, "read-only segment offset is not page-aligned");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x24, 0xfff);
        assert_invalid(bytes, "text segment size is not page-aligned");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x2c, 0xfff);
        assert_invalid(bytes, "read-only segment size is not page-aligned");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x28, 0);
        assert_invalid(bytes, "read-only segment overlaps the text segment");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x30, 0x1000);
        assert_invalid(bytes, "data segment overlaps the read-only segment");

        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x30, 0x3000);
        assert_invalid(bytes, "data segment is outside the executable portion");
    }

    #[test]
    fn rejects_invalid_module_and_metadata_ranges() {
        let mut bytes = canonical_nro();
        put_u32(&mut bytes, 0x04, EXECUTABLE_SIZE as u32 - 0x10);
        assert_invalid(bytes, "module header is outside the executable portion");

        for (offset_field, size_field, expected) in [
            (0x68, 0x6c, "embedded API-info"),
            (0x70, 0x74, "dynamic string table"),
            (0x78, 0x7c, "dynamic symbol table"),
        ] {
            let mut bytes = canonical_nro();
            put_u32(&mut bytes, offset_field, EXECUTABLE_SIZE as u32 - 4);
            put_u32(&mut bytes, size_field, 8);
            assert_invalid(bytes, expected);
        }
    }

    #[test]
    fn rejects_invalid_asset_header() {
        let mut bytes = canonical_nro();
        bytes.push(0);
        assert_invalid(bytes, "ASET header is truncated");

        let mut bytes = canonical_nro();
        bytes.extend_from_slice(&[0; ASSET_HEADER_SIZE as usize]);
        assert_invalid(bytes, "expected ASET magic");

        let mut bytes = canonical_nro();
        bytes.extend_from_slice(&[0; ASSET_HEADER_SIZE as usize]);
        let base = EXECUTABLE_SIZE;
        bytes[base..base + 4].copy_from_slice(b"ASET");
        put_u32(&mut bytes, base + 4, 1);
        assert_invalid(bytes, "unsupported ASET version");
    }

    #[test]
    fn rejects_invalid_asset_ranges() {
        let mut bytes = canonical_nro();
        append_assets(&mut bytes, (0x20, b"icon"), (0, b""), (0, b""), 0);
        assert_invalid(bytes, "icon asset starts inside the ASET header");

        let mut bytes = canonical_nro();
        append_assets(&mut bytes, (0x40, b"icon"), (0, b""), (0, b""), 0);
        let base = EXECUTABLE_SIZE;
        put_u64(&mut bytes, base + 0x08, u64::MAX);
        assert_invalid(bytes, "icon asset offset overflows");

        let mut bytes = canonical_nro();
        append_assets(&mut bytes, (0x40, b"icon"), (0, b""), (0, b""), 0);
        let base = EXECUTABLE_SIZE;
        put_u64(&mut bytes, base + 0x10, 0x1000);
        assert_invalid(bytes, "icon asset is outside the source");

        let mut bytes = canonical_nro();
        append_assets(
            &mut bytes,
            (0x40, b"12345678"),
            (0x48, b"abcdefgh"),
            (0, b""),
            0,
        );
        let base = EXECUTABLE_SIZE;
        put_u64(&mut bytes, base + 0x18, 0x44);
        assert_invalid(bytes, "assets overlap");
    }

    #[test]
    fn debug_output_does_not_read_or_dump_storage() {
        let image = load(canonical_nro()).unwrap();
        let output = format!("{image:?}");

        assert!(output.contains("ExecutableSegment"));
        assert!(output.contains("0x2000"));
        assert!(!output.contains(&"33".repeat(100)));
    }
}
