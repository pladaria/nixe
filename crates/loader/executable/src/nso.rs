use std::fmt::Debug;
use std::sync::Arc;

use swiitx_loader_storage::{
    FormatLoader, LoadError, Storage, StorageError, StorageRef, SubStorage,
};

use crate::{
    ExecutableFormat, ExecutableImage, ExecutableSegment, ExecutableSegmentKind, MemoryPermissions,
};

const HEADER_SIZE: u64 = 0x100;
const PAGE_SIZE: u64 = 0x1000;
const MOD0_SIZE: u64 = 0x1c;
const KNOWN_FLAGS: u32 = 0xff;
const FLAG_EXECUTE_ONLY: u32 = 1 << 6;
const FLAG_ZBIC: u32 = 1 << 7;

/// Describes how an NSO segment is stored in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsoSegmentCompression {
    /// The segment is stored verbatim and remains a lazy view of the source.
    None,
    /// The segment is a classic raw LZ4 block.
    Lz4,
    /// The segment uses Nintendo's Zstandard binary-interpolative-coding variant.
    Zbic,
}

/// Loads classic Nintendo Shared Object (NSO) files.
#[derive(Debug)]
pub struct NsoLoader;

impl FormatLoader for NsoLoader {
    type Output = NsoImage;

    const FORMAT_NAME: &'static str = "NSO";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        parse_nso(storage)
    }
}

/// A validated byte range relative to the decompressed read-only segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NsoRange {
    offset: u64,
    size: u64,
}

impl NsoRange {
    const fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Returns the offset from the start of the read-only segment.
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the range size.
    pub const fn size(self) -> u64 {
        self.size
    }

    /// Returns whether the range is empty.
    pub const fn is_empty(self) -> bool {
        self.size == 0
    }
}

/// NSO header metadata retained for later runtime and linker stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsoMetadata {
    version: u32,
    flags: u32,
    module_name: Vec<u8>,
    compressed: [bool; 3],
    compression: [NsoSegmentCompression; 3],
    hash_present: [bool; 3],
    digests: [[u8; 32]; 3],
    embedded_api_info: NsoRange,
    dynamic_string_table: NsoRange,
    dynamic_symbol_table: NsoRange,
}

impl NsoMetadata {
    /// Returns the preserved NSO version field.
    pub const fn version(&self) -> u32 {
        self.version
    }
    /// Returns the validated NSO flags.
    pub const fn flags(&self) -> u32 {
        self.flags
    }
    /// Returns the module-name bytes exactly as stored in the NSO.
    pub fn module_name(&self) -> &[u8] {
        &self.module_name
    }

    /// Interprets the module name as UTF-8 after removing at most one trailing NUL.
    pub fn module_name_str(&self) -> Option<&str> {
        let bytes = self
            .module_name
            .strip_suffix(&[0])
            .unwrap_or(&self.module_name);
        std::str::from_utf8(bytes).ok()
    }

    /// Returns compression flags in text, read-only, data order.
    pub const fn compressed(&self) -> &[bool; 3] {
        &self.compressed
    }
    /// Returns segment storage codecs in text, read-only, data order.
    pub const fn compression(&self) -> &[NsoSegmentCompression; 3] {
        &self.compression
    }
    /// Returns hash-presence flags in text, read-only, data order.
    pub const fn hash_present(&self) -> &[bool; 3] {
        &self.hash_present
    }
    /// Returns declared SHA-256 values in text, read-only, data order.
    pub const fn digests(&self) -> &[[u8; 32]; 3] {
        &self.digests
    }
    /// Returns the embedded SDK/API-info range relative to read-only data.
    pub const fn embedded_api_info(&self) -> NsoRange {
        self.embedded_api_info
    }
    /// Returns the dynamic string-table range relative to read-only data.
    pub const fn dynamic_string_table(&self) -> NsoRange {
        self.dynamic_string_table
    }
    /// Returns the dynamic symbol-table range relative to read-only data.
    pub const fn dynamic_symbol_table(&self) -> NsoRange {
        self.dynamic_symbol_table
    }
}

/// Resolved classic `MOD0` metadata. All addresses are image-relative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mod0Metadata {
    header_offset: u64,
    dynamic_offset: u64,
    bss_start: u64,
    bss_end: u64,
    exception_frame_header_start: u64,
    exception_frame_header_end: u64,
    module_object_offset: u64,
}

impl Mod0Metadata {
    /// Returns the image-relative `MOD0` header address.
    pub const fn header_offset(&self) -> u64 {
        self.header_offset
    }
    /// Returns the image-relative dynamic-section address.
    pub const fn dynamic_offset(&self) -> u64 {
        self.dynamic_offset
    }
    /// Returns the image-relative BSS range described by `MOD0`.
    pub const fn bss_range(&self) -> std::ops::Range<u64> {
        self.bss_start..self.bss_end
    }
    /// Returns the image-relative exception-frame-header range.
    pub const fn exception_frame_header_range(&self) -> std::ops::Range<u64> {
        self.exception_frame_header_start..self.exception_frame_header_end
    }
    /// Returns the image-relative module-object address.
    pub const fn module_object_offset(&self) -> u64 {
        self.module_object_offset
    }
}

/// A parsed NSO image and its format-specific metadata.
#[derive(Debug)]
pub struct NsoImage {
    executable: ExecutableImage,
    metadata: NsoMetadata,
    mod0: Option<Mod0Metadata>,
}

impl NsoImage {
    /// Returns the common executable description.
    pub const fn executable(&self) -> &ExecutableImage {
        &self.executable
    }
    /// Returns NSO-specific header metadata.
    pub const fn metadata(&self) -> &NsoMetadata {
        &self.metadata
    }
    /// Returns resolved classic `MOD0` metadata when declared.
    pub const fn mod0(&self) -> Option<&Mod0Metadata> {
        self.mod0.as_ref()
    }
    /// Splits the result into the common image, NSO metadata, and optional `MOD0` metadata.
    pub fn into_parts(self) -> (ExecutableImage, NsoMetadata, Option<Mod0Metadata>) {
        (self.executable, self.metadata, self.mod0)
    }
}

#[derive(Clone, Copy)]
struct RawSegment {
    name: &'static str,
    kind: ExecutableSegmentKind,
    file_offset: u64,
    memory_offset: u64,
    decompressed_size: u64,
    stored_size: u64,
    compression: NsoSegmentCompression,
    permissions: MemoryPermissions,
}

#[derive(Debug)]
struct ByteStorage(Vec<u8>);

impl Storage for ByteStorage {
    fn len(&self) -> Result<u64, StorageError> {
        u64::try_from(self.0.len()).map_err(|_| StorageError::OutOfBounds)
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

fn parse_nso(storage: StorageRef) -> Result<NsoImage, LoadError> {
    let source_len = storage.len()?;
    if source_len < HEADER_SIZE {
        return Err(invalid("header is truncated"));
    }
    let mut header = [0_u8; HEADER_SIZE as usize];
    storage.read_at(0, &mut header)?;
    if &header[..4] != b"NSO0" {
        return Err(invalid("expected NSO0 magic"));
    }

    let flags = read_u32(&header, 0x0c);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(invalid("unsupported NSO flag bits are set"));
    }

    let segment_compression = |index: u32| {
        if flags & (1 << index) == 0 {
            NsoSegmentCompression::None
        } else if flags & FLAG_ZBIC != 0 {
            NsoSegmentCompression::Zbic
        } else {
            NsoSegmentCompression::Lz4
        }
    };
    let text_permissions = if flags & FLAG_EXECUTE_ONLY != 0 {
        MemoryPermissions::EXECUTE
    } else {
        MemoryPermissions::from_bits(MemoryPermissions::READ.0 | MemoryPermissions::EXECUTE.0)
    };

    let raw = [
        raw_segment(
            &header,
            "text",
            ExecutableSegmentKind::Text,
            0x10,
            0x60,
            segment_compression(0),
            text_permissions,
        ),
        raw_segment(
            &header,
            "read-only",
            ExecutableSegmentKind::ReadOnly,
            0x20,
            0x64,
            segment_compression(1),
            MemoryPermissions::READ,
        ),
        raw_segment(
            &header,
            "data",
            ExecutableSegmentKind::Data,
            0x30,
            0x68,
            segment_compression(2),
            MemoryPermissions::from_bits(MemoryPermissions::READ.0 | MemoryPermissions::WRITE.0),
        ),
    ];
    validate_segments(&raw, source_len)?;

    let name_offset = u64::from(read_u32(&header, 0x1c));
    let name_size = u64::from(read_u32(&header, 0x2c));
    validate_file_range(name_offset, name_size, source_len, "module name")?;
    if name_size != 0 {
        if name_offset < HEADER_SIZE {
            return Err(invalid("module name overlaps the fixed header"));
        }
        for segment in &raw {
            if overlaps(
                name_offset,
                name_size,
                segment.file_offset,
                segment.stored_size,
            )? {
                return Err(invalid(format!(
                    "module name overlaps the {} segment",
                    segment.name
                )));
            }
        }
    } else if name_offset < HEADER_SIZE || name_offset > source_len {
        return Err(invalid("empty module-name offset is invalid"));
    }
    let module_name = read_vec(&storage, name_offset, name_size, "module name")?;

    let ranges = [
        read_range(&header, 0x88),
        read_range(&header, 0x90),
        read_range(&header, 0x98),
    ];
    for (range, name) in ranges.iter().zip([
        "embedded API-info",
        "dynamic string table",
        "dynamic symbol table",
    ]) {
        validate_metadata_range(*range, raw[1].decompressed_size, name)?;
    }

    let loaded = raw
        .iter()
        .map(|segment| load_segment(&storage, *segment))
        .collect::<Result<Vec<_>, _>>()?;
    let bss_size = u64::from(read_u32(&header, 0x3c));
    let data_memory_size = raw[2]
        .decompressed_size
        .checked_add(bss_size)
        .ok_or_else(|| invalid("data plus BSS size overflows"))?;
    let image_extent = raw[2]
        .memory_offset
        .checked_add(data_memory_size)
        .ok_or_else(|| invalid("image memory extent overflows"))?;
    validate_memory(&raw, data_memory_size)?;

    let mod0 = parse_mod0(&loaded, &raw, raw[2], bss_size, image_extent)?;
    let segments = raw
        .into_iter()
        .zip(loaded)
        .map(|(segment, view)| {
            let memory_size = if segment.kind == ExecutableSegmentKind::Data {
                data_memory_size
            } else {
                segment.decompressed_size
            };
            let mapping_size = checked_align_up(memory_size, PAGE_SIZE)
                .ok_or_else(|| invalid(format!("{} mapping size overflows", segment.name)))?;
            Ok(ExecutableSegment::new(
                segment.kind,
                segment.memory_offset,
                segment.decompressed_size,
                memory_size,
                mapping_size,
                segment.permissions,
                view,
            ))
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let mut module_id = [0; 32];
    module_id.copy_from_slice(&header[0x40..0x60]);
    let mut digests = [[0; 32]; 3];
    for (digest, offset) in digests.iter_mut().zip([0xa0, 0xc0, 0xe0]) {
        digest.copy_from_slice(&header[offset..offset + 32]);
    }
    Ok(NsoImage {
        executable: ExecutableImage::new(ExecutableFormat::Nso, 0, module_id, segments),
        metadata: NsoMetadata {
            version: read_u32(&header, 4),
            flags,
            module_name,
            compressed: [flags & 1 != 0, flags & 2 != 0, flags & 4 != 0],
            compression: [
                segment_compression(0),
                segment_compression(1),
                segment_compression(2),
            ],
            hash_present: [flags & 8 != 0, flags & 16 != 0, flags & 32 != 0],
            digests,
            embedded_api_info: ranges[0],
            dynamic_string_table: ranges[1],
            dynamic_symbol_table: ranges[2],
        },
        mod0,
    })
}

fn raw_segment(
    header: &[u8],
    name: &'static str,
    kind: ExecutableSegmentKind,
    descriptor: usize,
    stored: usize,
    compression: NsoSegmentCompression,
    permissions: MemoryPermissions,
) -> RawSegment {
    RawSegment {
        name,
        kind,
        file_offset: u64::from(read_u32(header, descriptor)),
        memory_offset: u64::from(read_u32(header, descriptor + 4)),
        decompressed_size: u64::from(read_u32(header, descriptor + 8)),
        stored_size: u64::from(read_u32(header, stored)),
        compression,
        permissions,
    }
}

fn validate_segments(segments: &[RawSegment; 3], source_len: u64) -> Result<(), LoadError> {
    if segments[0].memory_offset != 0 {
        return Err(invalid("text segment does not start at image offset zero"));
    }
    for segment in segments {
        if segment.memory_offset % PAGE_SIZE != 0 {
            return Err(invalid(format!(
                "{} memory offset is not page-aligned",
                segment.name
            )));
        }
        validate_file_range(
            segment.file_offset,
            segment.stored_size,
            source_len,
            &format!("{} segment", segment.name),
        )?;
        if segment.stored_size != 0 && segment.file_offset < HEADER_SIZE {
            return Err(invalid(format!(
                "{} segment overlaps the fixed header",
                segment.name
            )));
        }
        if segment.compression == NsoSegmentCompression::None
            && segment.stored_size != segment.decompressed_size
        {
            return Err(invalid(format!(
                "{} uncompressed stored size differs from decompressed size",
                segment.name
            )));
        }
    }
    for pair in [(0, 1), (0, 2), (1, 2)] {
        if overlaps(
            segments[pair.0].file_offset,
            segments[pair.0].stored_size,
            segments[pair.1].file_offset,
            segments[pair.1].stored_size,
        )? {
            return Err(invalid(format!(
                "{} and {} stored segments overlap",
                segments[pair.0].name, segments[pair.1].name
            )));
        }
    }
    Ok(())
}

fn validate_memory(segments: &[RawSegment; 3], data_memory_size: u64) -> Result<(), LoadError> {
    let sizes = [
        segments[0].decompressed_size,
        segments[1].decompressed_size,
        data_memory_size,
    ];
    for index in 0..2 {
        let mapping = checked_align_up(sizes[index], PAGE_SIZE)
            .ok_or_else(|| invalid(format!("{} mapping size overflows", segments[index].name)))?;
        let end = segments[index]
            .memory_offset
            .checked_add(mapping)
            .ok_or_else(|| invalid(format!("{} memory range overflows", segments[index].name)))?;
        if segments[index + 1].memory_offset < end {
            return Err(invalid(format!(
                "{} memory segment overlaps {}",
                segments[index].name,
                segments[index + 1].name
            )));
        }
    }
    checked_align_up(data_memory_size, PAGE_SIZE)
        .and_then(|size| segments[2].memory_offset.checked_add(size))
        .ok_or_else(|| invalid("data mapped memory range overflows"))?;
    Ok(())
}

fn load_segment(storage: &StorageRef, segment: RawSegment) -> Result<StorageRef, LoadError> {
    if segment.compression == NsoSegmentCompression::None {
        return Ok(Arc::new(SubStorage::new(
            storage.clone(),
            segment.file_offset,
            segment.stored_size,
        )?));
    }
    let encoded = read_vec(
        storage,
        segment.file_offset,
        segment.stored_size,
        segment.name,
    )?;
    let expected = usize::try_from(segment.decompressed_size)
        .map_err(|_| invalid(format!("{} decompressed size is too large", segment.name)))?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(expected)
        .map_err(|_| invalid(format!("{} decompressed allocation failed", segment.name)))?;
    decoded.resize(expected, 0);
    let decoded_size = match segment.compression {
        NsoSegmentCompression::None => unreachable!("uncompressed segments returned above"),
        NsoSegmentCompression::Lz4 => lz4_flex::block::decompress_into(&encoded, &mut decoded)
            .map_err(|error| invalid(format!("{} LZ4 data is invalid: {error}", segment.name)))?,
        NsoSegmentCompression::Zbic => {
            let mut decoder = ruzstd_zbic::decoding::FrameDecoder::new();
            decoder
                .decode_all(&encoded, &mut decoded)
                .map_err(|error| {
                    invalid(format!("{} ZBIC data is invalid: {error}", segment.name))
                })?
        }
    };
    if decoded_size != expected {
        return Err(invalid(format!(
            "{} {} output has the wrong size",
            segment.name,
            match segment.compression {
                NsoSegmentCompression::Lz4 => "LZ4",
                NsoSegmentCompression::Zbic => "ZBIC",
                NsoSegmentCompression::None => unreachable!(),
            }
        )));
    }
    Ok(Arc::new(ByteStorage(decoded)))
}

fn parse_mod0(
    loaded: &[StorageRef],
    raw: &[RawSegment; 3],
    data: RawSegment,
    bss_size: u64,
    image_extent: u64,
) -> Result<Option<Mod0Metadata>, LoadError> {
    if raw[0].decompressed_size < 8 {
        return Err(invalid("text segment is too small for the MOD0 locator"));
    }
    let mut locator = [0; 4];
    loaded[0].read_at(4, &mut locator)?;
    let header_offset = u64::from(u32::from_le_bytes(locator));
    if header_offset == 0 {
        return Ok(None);
    }
    let end = header_offset
        .checked_add(MOD0_SIZE)
        .ok_or_else(|| invalid("MOD0 header range overflows"))?;
    let (segment, storage) = raw
        .iter()
        .zip(loaded)
        .find(|(segment, _)| {
            header_offset >= segment.memory_offset
                && segment
                    .memory_offset
                    .checked_add(segment.decompressed_size)
                    .is_some_and(|segment_end| end <= segment_end)
        })
        .ok_or_else(|| invalid("MOD0 header is outside the initialized image segments"))?;
    let mut bytes = [0; MOD0_SIZE as usize];
    storage.read_at(header_offset - segment.memory_offset, &mut bytes)?;
    if &bytes[..4] != b"MOD0" {
        return Err(invalid("expected MOD0 magic"));
    }
    let base = header_offset;
    let resolve = |offset| resolve_signed(base, read_i32(&bytes, offset), image_extent);
    let dynamic_offset = resolve(4)?;
    let bss_start = resolve(8)?;
    let bss_end = resolve(12)?;
    let exception_start = resolve(16)?;
    let exception_end = resolve(20)?;
    let module_object_offset = resolve(24)?;
    if bss_end < bss_start {
        return Err(invalid("MOD0 BSS range is reversed"));
    }
    if exception_end < exception_start {
        return Err(invalid("MOD0 exception-frame range is reversed"));
    }
    let expected_bss_start = data
        .memory_offset
        .checked_add(data.decompressed_size)
        .ok_or_else(|| invalid("data end overflows"))?;
    let expected_bss_end = expected_bss_start
        .checked_add(bss_size)
        .ok_or_else(|| invalid("BSS end overflows"))?;
    if bss_start < expected_bss_start || bss_end > expected_bss_end {
        return Err(invalid(format!(
            "MOD0 BSS range {bss_start:#x}..{bss_end:#x} is outside the NSO header BSS range {expected_bss_start:#x}..{expected_bss_end:#x}"
        )));
    }
    Ok(Some(Mod0Metadata {
        header_offset: base,
        dynamic_offset,
        bss_start,
        bss_end,
        exception_frame_header_start: exception_start,
        exception_frame_header_end: exception_end,
        module_object_offset,
    }))
}

fn resolve_signed(base: u64, relative: i32, limit: u64) -> Result<u64, LoadError> {
    let value = if relative >= 0 {
        base.checked_add(relative as u64)
    } else {
        base.checked_sub(u64::from(relative.unsigned_abs()))
    }
    .ok_or_else(|| invalid("MOD0 relative offset overflows"))?;
    if value > limit {
        return Err(invalid("MOD0 relative offset is outside image memory"));
    }
    Ok(value)
}

fn validate_file_range(offset: u64, size: u64, limit: u64, name: &str) -> Result<(), LoadError> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| invalid(format!("{name} range overflows")))?;
    if end > limit {
        return Err(invalid(format!("{name} is outside the source")));
    }
    Ok(())
}

fn validate_metadata_range(range: NsoRange, limit: u64, name: &str) -> Result<(), LoadError> {
    let end = range
        .offset
        .checked_add(range.size)
        .ok_or_else(|| invalid(format!("{name} range overflows")))?;
    if end > limit {
        return Err(invalid(format!("{name} is outside the read-only segment")));
    }
    Ok(())
}

fn overlaps(a_offset: u64, a_size: u64, b_offset: u64, b_size: u64) -> Result<bool, LoadError> {
    if a_size == 0 || b_size == 0 {
        return Ok(false);
    }
    let a_end = a_offset
        .checked_add(a_size)
        .ok_or_else(|| invalid("file range overflows"))?;
    let b_end = b_offset
        .checked_add(b_size)
        .ok_or_else(|| invalid("file range overflows"))?;
    Ok(a_offset < b_end && b_offset < a_end)
}

fn read_vec(
    storage: &StorageRef,
    offset: u64,
    size: u64,
    name: &str,
) -> Result<Vec<u8>, LoadError> {
    let size = usize::try_from(size).map_err(|_| invalid(format!("{name} size is too large")))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| invalid(format!("{name} allocation failed")))?;
    bytes.resize(size, 0);
    storage.read_at(offset, &mut bytes)?;
    Ok(bytes)
}

fn read_range(bytes: &[u8], offset: usize) -> NsoRange {
    NsoRange::new(
        u64::from(read_u32(bytes, offset)),
        u64::from(read_u32(bytes, offset + 4)),
    )
}
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed field"))
}
fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed field"))
}
fn checked_align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}
fn invalid(reason: impl Into<String>) -> LoadError {
    LoadError::invalid(NsoLoader::FORMAT_NAME, reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestStorage(Vec<u8>);

    impl Storage for TestStorage {
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

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn fixture(compressed: [bool; 3]) -> (Vec<u8>, [Vec<u8>; 3]) {
        fixture_with_compression(compressed.map(|compressed| {
            if compressed {
                NsoSegmentCompression::Lz4
            } else {
                NsoSegmentCompression::None
            }
        }))
    }

    fn fixture_with_compression(
        compression: [NsoSegmentCompression; 3],
    ) -> (Vec<u8>, [Vec<u8>; 3]) {
        let mut payloads = [vec![0x11; 0x100], vec![0x22; 0x80], vec![0x33; 0x40]];
        payloads[0][4..8].fill(0);
        let encoded = std::array::from_fn::<_, 3, _>(|index| match compression[index] {
            NsoSegmentCompression::None => payloads[index].clone(),
            NsoSegmentCompression::Lz4 => lz4_flex::block::compress(&payloads[index]),
            NsoSegmentCompression::Zbic => {
                let (decoded, encoded) = zbic_fixture();
                payloads[index] = decoded;
                encoded
            }
        });
        let mut bytes = vec![0; 0x106];
        bytes[..4].copy_from_slice(b"NSO0");
        put_u32(&mut bytes, 4, 3);
        put_u32(
            &mut bytes,
            0x0c,
            compression
                .iter()
                .enumerate()
                .fold(0, |flags, (index, codec)| {
                    flags | (u32::from(*codec != NsoSegmentCompression::None) << index)
                })
                | (u32::from(compression.contains(&NsoSegmentCompression::Zbic)) * FLAG_ZBIC)
                | 0x38,
        );
        put_u32(&mut bytes, 0x1c, 0x100);
        put_u32(&mut bytes, 0x2c, 6);
        bytes[0x100..0x106].copy_from_slice(b"main\0\0");
        let descriptors = [0x10, 0x20, 0x30];
        let stored_offsets = [0x60, 0x64, 0x68];
        let memory_offsets = [0, 0x1000, 0x2000];
        for index in 0..3 {
            let file_offset = bytes.len();
            put_u32(&mut bytes, descriptors[index], file_offset as u32);
            put_u32(&mut bytes, descriptors[index] + 4, memory_offsets[index]);
            put_u32(
                &mut bytes,
                descriptors[index] + 8,
                payloads[index].len() as u32,
            );
            put_u32(
                &mut bytes,
                stored_offsets[index],
                encoded[index].len() as u32,
            );
            bytes.extend_from_slice(&encoded[index]);
        }
        put_u32(&mut bytes, 0x3c, 0x41);
        for index in 0..32 {
            bytes[0x40 + index] = index as u8;
        }
        put_u32(&mut bytes, 0x88, 4);
        put_u32(&mut bytes, 0x8c, 8);
        put_u32(&mut bytes, 0x90, 0x20);
        put_u32(&mut bytes, 0x94, 0x10);
        put_u32(&mut bytes, 0x98, 0x80);
        put_u32(&mut bytes, 0x9c, 0);
        for index in 0..3 {
            bytes[0xa0 + index * 32..0xc0 + index * 32].fill((index + 1) as u8);
        }
        (bytes, payloads)
    }

    fn zbic_fixture() -> (Vec<u8>, Vec<u8>) {
        // Produced from synthetic bytes by Atmosphere's ZBIC-enabled Zstandard compressor.
        // This frame contains compressed entropy tables, so it exercises BIC rather than merely
        // the alternate frame magic.
        const ENCODED: &str = "5a424943600001450b0094120000899aabbccddef0061728394a5b6d7e8fa0b1c2d3e5f60c1d2e3f5062738495a6b7c8daeb01122334455768798a9bacbdcfe0f10718293a4c5d6e7f90a1b2c4d5e6f70d1e2f415263748596a7b9cadbec021324364758697a8b9caebfd0e1f208192b3c4d5e6f8091a3b4c5d6e7f80e2031425364758698a9bacbdced0315263748596a7b8d9eafc0d1e2f30a1b2c3d4e5f708293a4b5c6d7e8fa102132435465778899aabbccddef05162738495a6c7d8e9fb0c1d2e4f50b1c2d3e4f61728394a5b6c7d9ea0011223344566778cedf4b5cc3d44051b8c93546adbe2a3ba2b31f3097a814258c9d091a8192f90f7687ee046b7ce3f46071d8e95566cdde4a5bc2d33f50b7c83445acbd293aa1b21e2f96a713248b9c08198091f80e7586ed036a7b8c9daebfd0e22b200466ae2b6b820f372cf870c3820f372cf870c3820f372cf870c3821d6e58f0e186051f6e58f0e186051f6e58f0e186051f6e986070b809b5725858";
        let encoded = ENCODED
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let mut decoded: Vec<u8> = (0..512)
            .map(|index| ((index * 17 + index / 7) % 251) as u8)
            .collect();
        decoded[..8].fill(0);
        (decoded, encoded)
    }

    fn load(bytes: Vec<u8>) -> Result<NsoImage, LoadError> {
        NsoLoader::load(Arc::new(TestStorage(bytes)))
    }

    fn read_all(storage: &StorageRef) -> Vec<u8> {
        let mut bytes = vec![0; storage.len().unwrap() as usize];
        storage.read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn assert_invalid(bytes: Vec<u8>, needle: &str) {
        match load(bytes) {
            Err(LoadError::InvalidFormat { format, reason }) => {
                assert_eq!(format, "NSO");
                assert!(reason.contains(needle), "expected {needle:?} in {reason:?}");
            }
            result => panic!("expected invalid NSO, got {result:?}"),
        }
    }

    #[test]
    fn loads_uncompressed_segments_and_metadata() {
        let (bytes, expected) = fixture([false; 3]);
        let image = load(bytes).unwrap();
        let executable = image.executable();
        assert_eq!(executable.format(), ExecutableFormat::Nso);
        assert_eq!(executable.entry_offset(), 0);
        assert_eq!(
            executable.module_id(),
            &std::array::from_fn(|index| index as u8)
        );
        assert_eq!(executable.segments().len(), 3);
        for (segment, bytes) in executable.segments().iter().zip(expected) {
            assert_eq!(read_all(segment.storage()), bytes);
            assert_eq!(segment.file_size(), bytes.len() as u64);
        }
        assert_eq!(executable.segments()[2].memory_size(), 0x81);
        assert_eq!(executable.segments()[2].mapping_size(), 0x1000);
        assert!(executable.segments()[0].permissions().is_executable());
        assert!(executable.segments()[2].permissions().is_writable());
        assert_eq!(image.metadata().version(), 3);
        assert_eq!(image.metadata().module_name(), b"main\0\0");
        assert_eq!(image.metadata().module_name_str(), Some("main\0"));
        assert_eq!(image.metadata().embedded_api_info(), NsoRange::new(4, 8));
        assert_eq!(
            image.metadata().dynamic_symbol_table(),
            NsoRange::new(0x80, 0)
        );
        assert_eq!(image.metadata().digests()[2], [3; 32]);
        assert_eq!(
            image.metadata().compression(),
            &[NsoSegmentCompression::None; 3]
        );
        assert!(image.mod0().is_none());
    }

    #[test]
    fn reconstructs_independent_lz4_blocks_and_mixed_segments() {
        for flags in [[true; 3], [true, false, true]] {
            let (bytes, expected) = fixture(flags);
            let image = load(bytes).unwrap();
            assert_eq!(image.metadata().compressed(), &flags);
            for (segment, expected) in image.executable().segments().iter().zip(expected) {
                assert_eq!(read_all(segment.storage()), expected);
            }
        }
    }

    #[test]
    fn supports_execute_only_with_uncompressed_and_lz4_text() {
        for compressed in [false, true] {
            let (mut bytes, _) = fixture([compressed, false, false]);
            let flags = read_u32(&bytes, 0x0c) | FLAG_EXECUTE_ONLY;
            put_u32(&mut bytes, 0x0c, flags);
            let image = load(bytes).unwrap();
            let permissions = image.executable().segments()[0].permissions();
            assert!(permissions.is_executable());
            assert!(!permissions.is_readable());
            assert!(!permissions.is_writable());
        }
    }

    #[test]
    fn reconstructs_independent_and_mixed_zbic_segments() {
        for compression in [
            [NsoSegmentCompression::Zbic; 3],
            [
                NsoSegmentCompression::Zbic,
                NsoSegmentCompression::None,
                NsoSegmentCompression::Zbic,
            ],
            [
                NsoSegmentCompression::None,
                NsoSegmentCompression::Zbic,
                NsoSegmentCompression::None,
            ],
        ] {
            let (mut bytes, expected) = fixture_with_compression(compression);
            let flags = read_u32(&bytes, 0x0c) | FLAG_EXECUTE_ONLY;
            put_u32(&mut bytes, 0x0c, flags);
            let image = load(bytes).unwrap();
            assert_eq!(image.metadata().compression(), &compression);
            assert!(!image.executable().segments()[0].permissions().is_readable());
            for (segment, expected) in image.executable().segments().iter().zip(expected) {
                assert_eq!(read_all(segment.storage()), expected);
            }
        }
    }

    #[test]
    fn parses_and_resolves_mod0() {
        let (mut bytes, _) = fixture([false; 3]);
        let text_file = read_u32(&bytes, 0x10) as usize;
        put_u32(&mut bytes, text_file + 4, 0x20);
        let base = 0x20_i32;
        bytes[text_file + 0x20..text_file + 0x24].copy_from_slice(b"MOD0");
        put_i32(&mut bytes, text_file + 0x24, 0x30 - base);
        put_i32(&mut bytes, text_file + 0x28, 0x2040 - base);
        put_i32(&mut bytes, text_file + 0x2c, 0x2081 - base);
        put_i32(&mut bytes, text_file + 0x30, 0x40 - base);
        put_i32(&mut bytes, text_file + 0x34, 0x50 - base);
        put_i32(&mut bytes, text_file + 0x38, 0x60 - base);
        let image = load(bytes).unwrap();
        let mod0 = image.mod0().unwrap();
        assert_eq!(mod0.header_offset(), 0x20);
        assert_eq!(mod0.dynamic_offset(), 0x30);
        assert_eq!(mod0.bss_range(), 0x2040..0x2081);
        assert_eq!(mod0.exception_frame_header_range(), 0x40..0x50);
    }

    #[test]
    fn accepts_mod0_in_an_initialized_non_text_segment() {
        let (mut bytes, _) = fixture([false; 3]);
        let text_file = read_u32(&bytes, 0x10) as usize;
        let read_only_file = read_u32(&bytes, 0x20) as usize;
        put_u32(&mut bytes, text_file + 4, 0x1020);
        let base = 0x1020_i32;
        bytes[read_only_file + 0x20..read_only_file + 0x24].copy_from_slice(b"MOD0");
        put_i32(&mut bytes, read_only_file + 0x24, 0x1040 - base);
        put_i32(&mut bytes, read_only_file + 0x28, 0x2040 - base);
        put_i32(&mut bytes, read_only_file + 0x2c, 0x2081 - base);
        put_i32(&mut bytes, read_only_file + 0x30, 0x1040 - base);
        put_i32(&mut bytes, read_only_file + 0x34, 0x1050 - base);
        put_i32(&mut bytes, read_only_file + 0x38, 0x1060 - base);
        let image = load(bytes).unwrap();
        let mod0 = image.mod0().unwrap();
        assert_eq!(mod0.header_offset(), 0x1020);
        assert_eq!(mod0.dynamic_offset(), 0x1040);
    }

    #[test]
    fn rejects_bad_layout_flags_metadata_and_compression() {
        let (mut bytes, _) = fixture([false; 3]);
        bytes[..4].copy_from_slice(b"BAD!");
        assert_invalid(bytes, "NSO0 magic");

        let (mut bytes, _) = fixture([false; 3]);
        put_u32(&mut bytes, 0x0c, 1 << 8);
        assert_invalid(bytes, "unsupported NSO flag bits");

        let (mut bytes, _) = fixture([false; 3]);
        put_u32(&mut bytes, 0x24, 1);
        assert_invalid(bytes, "page-aligned");

        let (mut bytes, _) = fixture([false; 3]);
        put_u32(&mut bytes, 0x8c, 0x1000);
        assert_invalid(bytes, "read-only segment");

        let (mut bytes, _) = fixture([false; 3]);
        let text_offset = read_u32(&bytes, 0x10);
        put_u32(&mut bytes, 0x1c, text_offset);
        assert_invalid(bytes, "module name overlaps");

        let (mut bytes, _) = fixture([true, false, false]);
        let text_offset = read_u32(&bytes, 0x10) as usize;
        bytes[text_offset] = 0xff;
        assert_invalid(bytes, "LZ4");

        let (mut bytes, _) = fixture_with_compression([
            NsoSegmentCompression::Zbic,
            NsoSegmentCompression::None,
            NsoSegmentCompression::None,
        ]);
        let text_offset = read_u32(&bytes, 0x10) as usize;
        bytes[text_offset] = 0xff;
        assert_invalid(bytes, "ZBIC");

        let (mut bytes, _) = fixture_with_compression([
            NsoSegmentCompression::Zbic,
            NsoSegmentCompression::None,
            NsoSegmentCompression::None,
        ]);
        put_u32(&mut bytes, 0x18, 511);
        assert_invalid(bytes, "ZBIC");

        let (mut bytes, _) = fixture_with_compression([
            NsoSegmentCompression::None,
            NsoSegmentCompression::None,
            NsoSegmentCompression::Zbic,
        ]);
        let stored = read_u32(&bytes, 0x68);
        put_u32(&mut bytes, 0x68, stored + 1);
        bytes.push(0xaa);
        assert_invalid(bytes, "ZBIC");

        let (mut bytes, _) = fixture_with_compression([
            NsoSegmentCompression::Zbic,
            NsoSegmentCompression::None,
            NsoSegmentCompression::None,
        ]);
        let stored = read_u32(&bytes, 0x60);
        put_u32(&mut bytes, 0x60, stored - 1);
        assert_invalid(bytes, "ZBIC");
    }

    #[test]
    fn rejects_invalid_mod0_and_bss_disagreement() {
        let (mut bytes, _) = fixture([false; 3]);
        let text_file = read_u32(&bytes, 0x10) as usize;
        put_u32(&mut bytes, text_file + 4, 0x20);
        assert_invalid(bytes, "MOD0 magic");

        let (mut bytes, _) = fixture([false; 3]);
        let text_file = read_u32(&bytes, 0x10) as usize;
        put_u32(&mut bytes, text_file + 4, 0x20);
        bytes[text_file + 0x20..text_file + 0x3c].fill(0);
        bytes[text_file + 0x20..text_file + 0x24].copy_from_slice(b"MOD0");
        put_i32(&mut bytes, text_file + 0x28, 0x2000 - 0x20);
        put_i32(&mut bytes, text_file + 0x2c, 0x2081 - 0x20);
        assert_invalid(bytes, "outside the NSO header BSS range");
    }
}
