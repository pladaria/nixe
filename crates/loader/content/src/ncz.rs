use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use nixe_loader_storage::{FormatLoader, LoadError, Storage, StorageError, StorageRef};

use crate::crypto::apply_ctr_at;

const PREFIX_SIZE: u64 = 0x4000;
const SECTION_HEADER_SIZE: u64 = 0x10;
const SECTION_SIZE: u64 = 0x40;
const BLOCK_HEADER_SIZE: u64 = 0x18;
const MAX_SECTION_COUNT: u64 = 65_536;
const MAX_BLOCK_COUNT: u64 = 4 * 1024 * 1024;
const MAX_METADATA_SIZE: u64 = 64 * 1024 * 1024;
const SOLID_BUFFER_SIZE: usize = 1024 * 1024;
const MAX_ZSTD_WINDOW_LOG: u32 = 27;

type CacheFileFactory = fn() -> std::io::Result<File>;

/// Compression layout used by the NCZ payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NczCompressionKind {
    /// Independent Zstandard frames (or raw data) indexed by `NCZBLOCK`.
    Block,
    /// One sequential Zstandard stream.
    Solid,
}

/// Reconstruction metadata for one contiguous logical NCA section.
#[derive(Clone, PartialEq, Eq)]
pub struct NczSection {
    offset: u64,
    size: u64,
    crypto_type: u64,
    crypto_key: [u8; 16],
    crypto_counter: [u8; 16],
}

impl NczSection {
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub const fn crypto_type(&self) -> u64 {
        self.crypto_type
    }

    pub const fn is_aes_ctr(&self) -> bool {
        matches!(self.crypto_type, 3 | 4)
    }

    pub const fn counter(&self) -> &[u8; 16] {
        &self.crypto_counter
    }
}

impl std::fmt::Debug for NczSection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NczSection")
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("crypto_type", &self.crypto_type)
            .field("crypto_key", &"<redacted>")
            .field("crypto_counter", &"<redacted>")
            .finish()
    }
}

/// Parsed `NCZBLOCK` index information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NczBlockInfo {
    block_size: u64,
    block_count: u32,
    decompressed_size: u64,
}

impl NczBlockInfo {
    pub const fn block_size(&self) -> u64 {
        self.block_size
    }

    pub const fn block_count(&self) -> u32 {
        self.block_count
    }

    pub const fn decompressed_size(&self) -> u64 {
        self.decompressed_size
    }
}

/// Parsed NCZ metadata and its lazily reconstructed NCA view.
pub struct NczArchive {
    sections: Arc<[NczSection]>,
    compression: NczCompressionKind,
    block_info: Option<NczBlockInfo>,
    logical_size: u64,
    nca_storage: StorageRef,
}

impl NczArchive {
    pub fn sections(&self) -> &[NczSection] {
        &self.sections
    }

    pub const fn compression_kind(&self) -> NczCompressionKind {
        self.compression
    }

    pub const fn block_info(&self) -> Option<&NczBlockInfo> {
        self.block_info.as_ref()
    }

    pub const fn logical_size(&self) -> u64 {
        self.logical_size
    }

    /// Returns the same shared, lazy reconstructed NCA view on every call.
    pub fn nca_storage(&self) -> StorageRef {
        self.nca_storage.clone()
    }
}

impl std::fmt::Debug for NczArchive {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NczArchive")
            .field("sections", &self.sections)
            .field("compression", &self.compression)
            .field("block_info", &self.block_info)
            .field("logical_size", &self.logical_size)
            .finish_non_exhaustive()
    }
}

/// Loads an NCZ and exposes its original logical NCA byte stream.
#[derive(Debug)]
pub struct NczLoader;

impl FormatLoader for NczLoader {
    type Output = NczArchive;

    const FORMAT_NAME: &'static str = "NCZ";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        parse_ncz(storage)
    }
}

#[derive(Clone)]
struct BlockLayout {
    block_size: u64,
    decompressed_size: u64,
    compressed_sizes: Arc<[u32]>,
    compressed_offsets: Arc<[u64]>,
}

enum CacheMode {
    Block {
        layout: BlockLayout,
        valid: Vec<bool>,
    },
    Solid {
        compressed_offset: u64,
        compressed_end: u64,
        decompressed_end: u64,
        decoder: Option<SolidDecoder>,
    },
}

type SolidDecoder = zstd::stream::read::Decoder<'static, BufReader<StorageReader>>;

struct CacheState {
    file: Option<File>,
    terminal_error: Option<String>,
    mode: CacheMode,
}

struct NczCachedStorage {
    source: StorageRef,
    logical_size: u64,
    sections: Arc<[NczSection]>,
    create_cache_file: CacheFileFactory,
    state: Mutex<CacheState>,
}

impl std::fmt::Debug for NczCachedStorage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NczCachedStorage")
            .field("logical_size", &self.logical_size)
            .field("sections", &self.sections)
            .finish_non_exhaustive()
    }
}

impl Storage for NczCachedStorage {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.logical_size)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        let length = u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?;
        let end = offset
            .checked_add(length)
            .ok_or(StorageError::OutOfBounds)?;
        if end > self.logical_size {
            return Err(StorageError::OutOfBounds);
        }
        if buffer.is_empty() {
            return Ok(());
        }

        let mut output_offset = 0_usize;
        if offset < PREFIX_SIZE {
            let prefix_end = end.min(PREFIX_SIZE);
            let prefix_length =
                usize::try_from(prefix_end - offset).map_err(|_| StorageError::OutOfBounds)?;
            self.source.read_at(offset, &mut buffer[..prefix_length])?;
            output_offset = prefix_length;
        }
        if output_offset == buffer.len() {
            return Ok(());
        }

        let tail_start = offset.max(PREFIX_SIZE);
        let tail_end = end;
        let mut state = self
            .state
            .lock()
            .map_err(|_| StorageError::InvalidData("NCZ cache lock is poisoned".to_owned()))?;
        if let Some(reason) = &state.terminal_error {
            return Err(StorageError::InvalidData(reason.clone()));
        }

        let result = self.ensure_cached(&mut state, tail_start, tail_end);
        if let Err(error) = result {
            let reason = error.to_string();
            state.terminal_error = Some(reason.clone());
            return Err(StorageError::InvalidData(reason));
        }
        let file = state
            .file
            .as_mut()
            .ok_or_else(|| StorageError::InvalidData("NCZ cache was not initialized".to_owned()))?;
        file.seek(SeekFrom::Start(tail_start))?;
        file.read_exact(&mut buffer[output_offset..])?;
        Ok(())
    }
}

impl NczCachedStorage {
    fn ensure_cached(
        &self,
        state: &mut CacheState,
        start: u64,
        end: u64,
    ) -> Result<(), StorageError> {
        if state.file.is_none() {
            state.file = Some((self.create_cache_file)().map_err(|error| {
                StorageError::Io(std::io::Error::new(
                    error.kind(),
                    format!("cannot create NCZ temporary cache: {error}"),
                ))
            })?);
        }

        match &mut state.mode {
            CacheMode::Block { layout, valid } => self.ensure_blocks(
                state.file.as_mut().expect("cache initialized"),
                layout,
                valid,
                start,
                end,
            ),
            CacheMode::Solid {
                compressed_offset,
                compressed_end,
                decompressed_end,
                decoder,
            } => {
                let cached_until = PREFIX_SIZE
                    .checked_add(*decompressed_end)
                    .ok_or(StorageError::OutOfBounds)?;
                if end > cached_until {
                    log::debug!(
                        "NCZ solid decompression requested: offset={start} bytes, length={} bytes, cached_until={cached_until} bytes, decoding={} bytes",
                        end - start,
                        end - cached_until
                    );
                }
                self.ensure_solid(
                    state.file.as_mut().expect("cache initialized"),
                    *compressed_offset,
                    *compressed_end,
                    decompressed_end,
                    decoder,
                    end,
                )
            }
        }
    }

    fn ensure_blocks(
        &self,
        file: &mut File,
        layout: &BlockLayout,
        valid: &mut [bool],
        start: u64,
        end: u64,
    ) -> Result<(), StorageError> {
        let tail_start = start
            .checked_sub(PREFIX_SIZE)
            .ok_or(StorageError::OutOfBounds)?;
        let tail_end = end
            .checked_sub(PREFIX_SIZE)
            .ok_or(StorageError::OutOfBounds)?;
        let first = tail_start / layout.block_size;
        let last = tail_end.saturating_sub(1) / layout.block_size;
        let compressed_blocks = (first..=last)
            .filter(|block_index| {
                let index = usize::try_from(*block_index)
                    .expect("validated NCZ block index fits in memory");
                !valid[index]
                    && u64::from(layout.compressed_sizes[index])
                        != layout
                            .block_size
                            .min(layout.decompressed_size - *block_index * layout.block_size)
            })
            .count();
        if compressed_blocks != 0 {
            log::debug!(
                "NCZ block decompression requested: offset={start} bytes, length={} bytes, blocks={first}..={last}, compressed_blocks={compressed_blocks}",
                end - start
            );
        }
        for block_index in first..=last {
            let index = usize::try_from(block_index).map_err(|_| StorageError::OutOfBounds)?;
            if valid[index] {
                continue;
            }
            let block_tail_offset = block_index
                .checked_mul(layout.block_size)
                .ok_or(StorageError::OutOfBounds)?;
            let expected_size = layout
                .block_size
                .min(layout.decompressed_size - block_tail_offset);
            let compressed_size = u64::from(layout.compressed_sizes[index]);
            let source_offset = layout.compressed_offsets[index];
            let logical_offset = PREFIX_SIZE
                .checked_add(block_tail_offset)
                .ok_or(StorageError::OutOfBounds)?;
            if compressed_size == expected_size {
                self.copy_raw_block(file, source_offset, logical_offset, expected_size)?;
            } else {
                self.decode_block(
                    file,
                    source_offset,
                    compressed_size,
                    logical_offset,
                    expected_size,
                    index,
                )?;
            }
            valid[index] = true;
        }
        Ok(())
    }

    fn copy_raw_block(
        &self,
        file: &mut File,
        source_offset: u64,
        logical_offset: u64,
        size: u64,
    ) -> Result<(), StorageError> {
        let mut buffer = vec![0_u8; SOLID_BUFFER_SIZE];
        let mut produced = 0_u64;
        while produced < size {
            let count = usize::try_from((size - produced).min(SOLID_BUFFER_SIZE as u64))
                .expect("bounded raw block chunk fits usize");
            self.source
                .read_at(source_offset + produced, &mut buffer[..count])?;
            let output = &mut buffer[..count];
            reconstruct_sections(&self.sections, logical_offset + produced, output)?;
            file.seek(SeekFrom::Start(logical_offset + produced))?;
            file.write_all(output)?;
            produced += count as u64;
        }
        Ok(())
    }

    fn decode_block(
        &self,
        file: &mut File,
        source_offset: u64,
        compressed_size: u64,
        logical_offset: u64,
        expected_size: u64,
        index: usize,
    ) -> Result<(), StorageError> {
        let source_end = source_offset
            .checked_add(compressed_size)
            .ok_or(StorageError::OutOfBounds)?;
        let reader = StorageReader {
            source: self.source.clone(),
            position: source_offset,
            end: source_end,
        };
        let mut decoder = zstd::stream::read::Decoder::new(reader)
            .map_err(|error| invalid_data(format!("cannot start NCZ block {index}: {error}")))?
            .single_frame();
        decoder
            .window_log_max(MAX_ZSTD_WINDOW_LOG)
            .map_err(|error| {
                invalid_data(format!("cannot limit NCZ block {index} window: {error}"))
            })?;

        let mut buffer = vec![0_u8; SOLID_BUFFER_SIZE];
        let mut produced = 0_u64;
        while produced < expected_size {
            let count = usize::try_from((expected_size - produced).min(SOLID_BUFFER_SIZE as u64))
                .expect("bounded decoded block chunk fits usize");
            let output = &mut buffer[..count];
            decoder.read_exact(output).map_err(|error| {
                invalid_data(format!("cannot decode NCZ block {index}: {error}"))
            })?;
            reconstruct_sections(&self.sections, logical_offset + produced, output)?;
            file.seek(SeekFrom::Start(logical_offset + produced))?;
            file.write_all(output)?;
            produced += count as u64;
        }
        let mut extra = [0_u8; 1];
        if decoder
            .read(&mut extra)
            .map_err(|error| invalid_data(format!("cannot finish NCZ block {index}: {error}")))?
            != 0
        {
            return Err(invalid_data(format!(
                "NCZ block {index} produces more data than declared"
            )));
        }
        let buffered = decoder.finish();
        let consumed = buffered
            .get_ref()
            .position
            .checked_sub(buffered.buffer().len() as u64)
            .ok_or_else(|| {
                invalid_data(format!("NCZ block {index} decoder position is invalid"))
            })?;
        if consumed != source_end {
            return Err(invalid_data(format!(
                "NCZ block {index} contains trailing data"
            )));
        }
        Ok(())
    }

    fn ensure_solid(
        &self,
        file: &mut File,
        compressed_offset: u64,
        compressed_end: u64,
        decompressed_end: &mut u64,
        decoder: &mut Option<SolidDecoder>,
        requested_end: u64,
    ) -> Result<(), StorageError> {
        let requested_tail_end = requested_end
            .checked_sub(PREFIX_SIZE)
            .ok_or(StorageError::OutOfBounds)?;
        if *decompressed_end >= requested_tail_end {
            return Ok(());
        }
        if decoder.is_none() {
            let reader = StorageReader {
                source: self.source.clone(),
                position: compressed_offset,
                end: compressed_end,
            };
            let mut stream = zstd::stream::read::Decoder::new(reader)
                .map_err(|error| invalid_data(format!("cannot start NCZ stream: {error}")))?
                .single_frame();
            stream
                .window_log_max(MAX_ZSTD_WINDOW_LOG)
                .map_err(|error| {
                    invalid_data(format!("cannot limit NCZ Zstandard window: {error}"))
                })?;
            *decoder = Some(stream);
        }

        let logical_tail_size = self.logical_size - PREFIX_SIZE;
        let mut chunk = vec![0_u8; SOLID_BUFFER_SIZE];
        while *decompressed_end < requested_tail_end {
            let remaining = logical_tail_size - *decompressed_end;
            let wanted = usize::try_from(
                remaining
                    .min(requested_tail_end - *decompressed_end)
                    .min(SOLID_BUFFER_SIZE as u64),
            )
            .expect("bounded solid chunk fits usize");
            let output = &mut chunk[..wanted];
            decoder
                .as_mut()
                .expect("solid decoder initialized")
                .read_exact(output)
                .map_err(|error| invalid_data(format!("cannot decode NCZ stream: {error}")))?;
            let logical_offset = PREFIX_SIZE
                .checked_add(*decompressed_end)
                .ok_or(StorageError::OutOfBounds)?;
            reconstruct_sections(&self.sections, logical_offset, output)?;
            file.seek(SeekFrom::Start(logical_offset))?;
            file.write_all(output)?;
            *decompressed_end += u64::try_from(wanted).expect("solid chunk length fits u64");
        }

        if *decompressed_end == logical_tail_size {
            let mut extra = [0_u8; 1];
            let count = decoder
                .as_mut()
                .expect("solid decoder initialized")
                .read(&mut extra)
                .map_err(|error| invalid_data(format!("cannot finish NCZ stream: {error}")))?;
            if count != 0 {
                return Err(invalid_data("NCZ stream produces more data than declared"));
            }
            let buffered = decoder.take().expect("solid decoder initialized").finish();
            let consumed = buffered
                .get_ref()
                .position
                .checked_sub(buffered.buffer().len() as u64)
                .ok_or_else(|| invalid_data("NCZ stream decoder position is invalid"))?;
            if consumed != compressed_end {
                return Err(invalid_data("trailing data follows the NCZ stream"));
            }
        }
        Ok(())
    }
}

struct StorageReader {
    source: StorageRef,
    position: u64,
    end: u64,
}

impl Read for StorageReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.position == self.end || buffer.is_empty() {
            return Ok(0);
        }
        let count = usize::try_from((self.end - self.position).min(buffer.len() as u64))
            .expect("bounded read length fits usize");
        self.source
            .read_at(self.position, &mut buffer[..count])
            .map_err(std::io::Error::other)?;
        self.position += u64::try_from(count).expect("read length fits u64");
        Ok(count)
    }
}

fn parse_ncz(storage: StorageRef) -> Result<NczArchive, LoadError> {
    parse_ncz_with_cache_factory(storage, tempfile::tempfile)
}

fn parse_ncz_with_cache_factory(
    storage: StorageRef,
    create_cache_file: CacheFileFactory,
) -> Result<NczArchive, LoadError> {
    let source_len = storage.len()?;
    if source_len < PREFIX_SIZE + SECTION_HEADER_SIZE {
        return Err(LoadError::invalid("NCZ", "header is truncated"));
    }
    let mut section_header = [0_u8; SECTION_HEADER_SIZE as usize];
    read_metadata(&storage, PREFIX_SIZE, &mut section_header, "section header")?;
    if &section_header[..8] != b"NCZSECTN" {
        return Err(LoadError::invalid("NCZ", "expected NCZSECTN magic"));
    }
    let section_count = read_u64(&section_header, 8);
    if section_count == 0 || section_count > MAX_SECTION_COUNT {
        return Err(LoadError::invalid(
            "NCZ",
            "section count is outside the safety limit",
        ));
    }
    let sections_size = section_count
        .checked_mul(SECTION_SIZE)
        .ok_or_else(|| LoadError::invalid("NCZ", "section table size overflows"))?;
    let metadata_end = PREFIX_SIZE
        .checked_add(SECTION_HEADER_SIZE)
        .and_then(|value| value.checked_add(sections_size))
        .ok_or_else(|| LoadError::invalid("NCZ", "section metadata size overflows"))?;
    if metadata_end - PREFIX_SIZE > MAX_METADATA_SIZE || metadata_end > source_len {
        return Err(LoadError::invalid(
            "NCZ",
            "section metadata is truncated or excessive",
        ));
    }
    let table_len = usize::try_from(sections_size)
        .map_err(|_| LoadError::invalid("NCZ", "section metadata does not fit in memory"))?;
    let mut table = vec![0_u8; table_len];
    read_metadata(
        &storage,
        PREFIX_SIZE + SECTION_HEADER_SIZE,
        &mut table,
        "section table",
    )?;

    let mut sections = Vec::with_capacity(section_count as usize);
    let mut expected_offset = None;
    for index in 0..section_count {
        let start = usize::try_from(index * SECTION_SIZE).expect("bounded section offset fits");
        let bytes = &table[start..start + SECTION_SIZE as usize];
        let offset = read_u64(bytes, 0);
        let size = read_u64(bytes, 8);
        let crypto_type = read_u64(bytes, 16);
        let padding = read_u64(bytes, 24);
        if padding != 0 {
            return Err(LoadError::invalid(
                "NCZ",
                format!("section {index} has non-zero padding"),
            ));
        }
        let invalid_offset = match expected_offset {
            None => offset < PREFIX_SIZE,
            Some(expected) => offset != expected,
        };
        if size == 0 || invalid_offset {
            return Err(LoadError::invalid(
                "NCZ",
                format!("section {index} is empty, overlapping, or leaves a gap"),
            ));
        }
        if !matches!(crypto_type, 1 | 3 | 4) {
            return Err(LoadError::invalid(
                "NCZ",
                format!("section {index} uses unsupported crypto type {crypto_type}"),
            ));
        }
        let crypto_key = bytes[32..48].try_into().expect("validated section record");
        let crypto_counter = bytes[48..64].try_into().expect("validated section record");
        expected_offset = Some(
            offset
                .checked_add(size)
                .ok_or_else(|| LoadError::invalid("NCZ", "logical NCA size overflows"))?,
        );
        sections.push(NczSection {
            offset,
            size,
            crypto_type,
            crypto_key,
            crypto_counter,
        });
    }
    let logical_size = expected_offset.expect("non-empty section table");
    let logical_tail_size = logical_size - PREFIX_SIZE;
    let sections: Arc<[NczSection]> = sections.into();
    let reconstruction_sections: Arc<[NczSection]> = if sections[0].offset > PREFIX_SIZE {
        let mut ranges = Vec::with_capacity(sections.len() + 1);
        ranges.push(NczSection {
            offset: PREFIX_SIZE,
            size: sections[0].offset - PREFIX_SIZE,
            crypto_type: 1,
            crypto_key: [0; 16],
            crypto_counter: [0; 16],
        });
        ranges.extend(sections.iter().cloned());
        ranges.into()
    } else {
        sections.clone()
    };

    let has_block_header = if source_len - metadata_end >= 8 {
        let mut magic = [0_u8; 8];
        read_metadata(&storage, metadata_end, &mut magic, "compression header")?;
        &magic == b"NCZBLOCK"
    } else {
        false
    };

    let (compression, block_info, mode) = if has_block_header {
        let mut header = [0_u8; BLOCK_HEADER_SIZE as usize];
        read_metadata(&storage, metadata_end, &mut header, "block header")?;
        if header[8] != 2 || header[9] != 1 || header[10] != 0 {
            return Err(LoadError::invalid(
                "NCZ",
                "unsupported NCZBLOCK version, compression type, or reserved field",
            ));
        }
        let exponent = header[11];
        if !(14..=32).contains(&exponent) {
            return Err(LoadError::invalid(
                "NCZ",
                "NCZBLOCK exponent is outside 14..=32",
            ));
        }
        let block_size = 1_u64
            .checked_shl(u32::from(exponent))
            .ok_or_else(|| LoadError::invalid("NCZ", "NCZBLOCK size overflows"))?;
        let block_count = u64::from(read_u32(&header, 12));
        let decompressed_size = read_u64(&header, 16);
        if block_count == 0 || block_count > MAX_BLOCK_COUNT {
            return Err(LoadError::invalid(
                "NCZ",
                "NCZBLOCK count is outside the safety limit",
            ));
        }
        if decompressed_size != logical_tail_size {
            return Err(LoadError::invalid(
                "NCZ",
                "NCZBLOCK decompressed size does not match sections",
            ));
        }
        let expected_count = decompressed_size.div_ceil(block_size);
        if block_count != expected_count {
            return Err(LoadError::invalid(
                "NCZ",
                "NCZBLOCK count does not match decompressed size",
            ));
        }
        let table_size = block_count
            .checked_mul(4)
            .ok_or_else(|| LoadError::invalid("NCZ", "NCZBLOCK table size overflows"))?;
        let data_offset = metadata_end
            .checked_add(BLOCK_HEADER_SIZE)
            .and_then(|value| value.checked_add(table_size))
            .ok_or_else(|| LoadError::invalid("NCZ", "NCZBLOCK metadata size overflows"))?;
        if data_offset - PREFIX_SIZE > MAX_METADATA_SIZE || data_offset > source_len {
            return Err(LoadError::invalid(
                "NCZ",
                "NCZBLOCK table is truncated or excessive",
            ));
        }
        let mut table = vec![0_u8; table_size as usize];
        read_metadata(
            &storage,
            metadata_end + BLOCK_HEADER_SIZE,
            &mut table,
            "block table",
        )?;
        let mut compressed_sizes = Vec::with_capacity(block_count as usize);
        let mut compressed_offsets = Vec::with_capacity(block_count as usize);
        let mut source_offset = data_offset;
        for index in 0..block_count {
            let expected = block_size.min(decompressed_size - index * block_size);
            let size = read_u32(&table, index as usize * 4);
            if size == 0 || u64::from(size) > expected {
                return Err(LoadError::invalid(
                    "NCZ",
                    format!("compressed size for block {index} is invalid"),
                ));
            }
            compressed_offsets.push(source_offset);
            source_offset = source_offset
                .checked_add(u64::from(size))
                .ok_or_else(|| LoadError::invalid("NCZ", "compressed block offsets overflow"))?;
            if source_offset > source_len {
                return Err(LoadError::invalid("NCZ", "compressed block is truncated"));
            }
            compressed_sizes.push(size);
        }
        if source_offset != source_len {
            return Err(LoadError::invalid(
                "NCZ",
                "trailing data follows NCZ blocks",
            ));
        }
        let info = NczBlockInfo {
            block_size,
            block_count: block_count as u32,
            decompressed_size,
        };
        let layout = BlockLayout {
            block_size,
            decompressed_size,
            compressed_sizes: compressed_sizes.into(),
            compressed_offsets: compressed_offsets.into(),
        };
        (
            NczCompressionKind::Block,
            Some(info),
            CacheMode::Block {
                layout,
                valid: vec![false; block_count as usize],
            },
        )
    } else {
        if metadata_end == source_len {
            return Err(LoadError::invalid(
                "NCZ",
                "solid Zstandard stream is missing",
            ));
        }
        (
            NczCompressionKind::Solid,
            None,
            CacheMode::Solid {
                compressed_offset: metadata_end,
                compressed_end: source_len,
                decompressed_end: 0,
                decoder: None,
            },
        )
    };

    let nca_storage: StorageRef = Arc::new(NczCachedStorage {
        source: storage,
        logical_size,
        sections: reconstruction_sections,
        create_cache_file,
        state: Mutex::new(CacheState {
            file: None,
            terminal_error: None,
            mode,
        }),
    });
    Ok(NczArchive {
        sections,
        compression,
        block_info,
        logical_size,
        nca_storage,
    })
}

fn reconstruct_sections(
    sections: &[NczSection],
    offset: u64,
    data: &mut [u8],
) -> Result<(), StorageError> {
    let end = offset
        .checked_add(u64::try_from(data.len()).map_err(|_| StorageError::OutOfBounds)?)
        .ok_or(StorageError::OutOfBounds)?;
    let mut position = offset;
    while position < end {
        let section = sections
            .iter()
            .find(|section| position >= section.offset && position < section.offset + section.size)
            .ok_or_else(|| invalid_data("decompressed bytes are outside NCZ sections"))?;
        let part_end = end.min(section.offset + section.size);
        if section.is_aes_ctr() {
            let start =
                usize::try_from(position - offset).map_err(|_| StorageError::OutOfBounds)?;
            let part_len =
                usize::try_from(part_end - position).map_err(|_| StorageError::OutOfBounds)?;
            let counter_prefix = section.crypto_counter[..8]
                .try_into()
                .expect("fixed counter prefix");
            apply_ctr_at(
                &section.crypto_key,
                counter_prefix,
                position,
                &mut data[start..start + part_len],
            );
        }
        position = part_end;
    }
    Ok(())
}

fn read_metadata(
    storage: &StorageRef,
    offset: u64,
    buffer: &mut [u8],
    what: &str,
) -> Result<(), LoadError> {
    match storage.read_at(offset, buffer) {
        Ok(()) => Ok(()),
        Err(StorageError::OutOfBounds) => {
            Err(LoadError::invalid("NCZ", format!("{what} is truncated")))
        }
        Err(error) => Err(LoadError::Storage(error)),
    }
}

fn invalid_data(reason: impl Into<String>) -> StorageError {
    StorageError::InvalidData(reason.into())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated NCZ range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated NCZ range"),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use sha2::{Digest, Sha256};

    use super::*;

    static CACHE_CREATIONS: AtomicUsize = AtomicUsize::new(0);

    fn counted_cache_file() -> std::io::Result<File> {
        CACHE_CREATIONS.fetch_add(1, Ordering::Relaxed);
        tempfile::tempfile()
    }

    fn failing_cache_file() -> std::io::Result<File> {
        Err(std::io::Error::other("synthetic cache failure"))
    }

    #[derive(Debug)]
    struct Bytes {
        data: Vec<u8>,
        reads: AtomicUsize,
        offsets: Mutex<Vec<u64>>,
    }

    impl Bytes {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                reads: AtomicUsize::new(0),
                offsets: Mutex::new(Vec::new()),
            }
        }
    }

    impl Storage for Bytes {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.data.len() as u64)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(self.data.get(start..end).ok_or(StorageError::OutOfBounds)?);
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.offsets.lock().unwrap().push(offset);
            Ok(())
        }
    }

    fn prefix() -> Vec<u8> {
        (0..PREFIX_SIZE).map(|index| index as u8).collect()
    }

    fn synthetic_nca() -> Vec<u8> {
        const SECTION_OFFSET: usize = 0x4000;
        const SECTION_SIZE: usize = 0x400;
        const DATA_OFFSET: usize = 0x200;
        const DATA_SIZE: usize = 0x100;

        let mut nca = vec![0_u8; SECTION_OFFSET + SECTION_SIZE];
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x205] = 1;
        nca[0x206] = 1;
        let nca_size = nca.len() as u64;
        nca[0x208..0x210].copy_from_slice(&nca_size.to_le_bytes());
        nca[0x210..0x218].copy_from_slice(&0x0100_0000_0000_1000_u64.to_le_bytes());
        nca[0x240..0x244].copy_from_slice(&((SECTION_OFFSET / 0x200) as u32).to_le_bytes());
        nca[0x244..0x248]
            .copy_from_slice(&(((SECTION_OFFSET + SECTION_SIZE) / 0x200) as u32).to_le_bytes());

        let data_start = SECTION_OFFSET + DATA_OFFSET;
        for (index, byte) in nca[data_start..data_start + DATA_SIZE]
            .iter_mut()
            .enumerate()
        {
            *byte = index as u8;
        }
        let data_hash: [u8; 32] = Sha256::digest(&nca[data_start..data_start + DATA_SIZE]).into();
        nca[SECTION_OFFSET..SECTION_OFFSET + 0x20].copy_from_slice(&data_hash);
        let master_hash: [u8; 32] =
            Sha256::digest(&nca[SECTION_OFFSET..SECTION_OFFSET + 0x20]).into();
        let fs = &mut nca[0x400..0x600];
        fs[2] = 1;
        fs[3] = 2;
        fs[4] = 1;
        fs[0x08..0x28].copy_from_slice(&master_hash);
        fs[0x28..0x2C].copy_from_slice(&(DATA_SIZE as u32).to_le_bytes());
        fs[0x2C..0x30].copy_from_slice(&2_u32.to_le_bytes());
        fs[0x38..0x40].copy_from_slice(&0x20_u64.to_le_bytes());
        fs[0x40..0x48].copy_from_slice(&(DATA_OFFSET as u64).to_le_bytes());
        fs[0x48..0x50].copy_from_slice(&(DATA_SIZE as u64).to_le_bytes());
        let fs_hash: [u8; 32] = Sha256::digest(&nca[0x400..0x600]).into();
        nca[0x280..0x2A0].copy_from_slice(&fs_hash);
        nca
    }

    fn section_record(
        offset: u64,
        size: u64,
        crypto_type: u64,
        key: [u8; 16],
        counter: [u8; 16],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&crypto_type.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&key);
        bytes.extend_from_slice(&counter);
        bytes
    }

    fn solid_ncz(tail: &[u8], sections: &[Vec<u8>]) -> Vec<u8> {
        let mut result = prefix();
        result.extend_from_slice(b"NCZSECTN");
        result.extend_from_slice(&(sections.len() as u64).to_le_bytes());
        for section in sections {
            result.extend_from_slice(section);
        }
        result.extend_from_slice(&zstd::stream::encode_all(tail, 3).unwrap());
        result
    }

    fn block_ncz(tail: &[u8], exponent: u8, sections: &[Vec<u8>]) -> Vec<u8> {
        let block_size = 1_usize << exponent;
        let blocks = tail.chunks(block_size).collect::<Vec<_>>();
        let mut stored = Vec::new();
        let mut sizes = Vec::new();
        for block in &blocks {
            let compressed = zstd::stream::encode_all(*block, 3).unwrap();
            let bytes = if compressed.len() < block.len() {
                compressed
            } else {
                block.to_vec()
            };
            sizes.push(bytes.len() as u32);
            stored.push(bytes);
        }

        let mut result = prefix();
        result.extend_from_slice(b"NCZSECTN");
        result.extend_from_slice(&(sections.len() as u64).to_le_bytes());
        for section in sections {
            result.extend_from_slice(section);
        }
        result.extend_from_slice(b"NCZBLOCK");
        result.extend_from_slice(&[2, 1, 0, exponent]);
        result.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
        result.extend_from_slice(&(tail.len() as u64).to_le_bytes());
        for size in sizes {
            result.extend_from_slice(&size.to_le_bytes());
        }
        for bytes in stored {
            result.extend_from_slice(&bytes);
        }
        result
    }

    fn load_counted(data: Vec<u8>) -> (NczArchive, Arc<Bytes>) {
        let source = Arc::new(Bytes::new(data));
        let archive = NczLoader::load(source.clone()).unwrap();
        (archive, source)
    }

    #[test]
    fn solid_reconstructs_prefix_and_plaintext_with_progressive_cache() {
        let tail = (0..100_000)
            .map(|index| (index * 13) as u8)
            .collect::<Vec<_>>();
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let (archive, source) = load_counted(solid_ncz(&tail, &[section]));
        assert_eq!(archive.compression_kind(), NczCompressionKind::Solid);
        assert_eq!(archive.logical_size(), PREFIX_SIZE + tail.len() as u64);

        let reads_after_parse = source.reads.load(Ordering::Relaxed);
        let storage = archive.nca_storage();
        let mut header = [0_u8; 32];
        storage.read_at(0, &mut header).unwrap();
        assert_eq!(source.reads.load(Ordering::Relaxed), reads_after_parse + 1);

        let mut later = [0_u8; 37];
        storage.read_at(PREFIX_SIZE + 70_000, &mut later).unwrap();
        assert_eq!(&later, &tail[70_000..70_037]);
        let reads_after_forward = source.reads.load(Ordering::Relaxed);
        let mut earlier = [0_u8; 29];
        storage.read_at(PREFIX_SIZE + 100, &mut earlier).unwrap();
        assert_eq!(&earlier, &tail[100..129]);
        assert_eq!(source.reads.load(Ordering::Relaxed), reads_after_forward);
    }

    #[test]
    fn reconstructed_bytes_are_accepted_by_the_existing_nca_loader() {
        let original = synthetic_nca();
        let tail = &original[PREFIX_SIZE as usize..];
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let mut compressed = solid_ncz(tail, &[section]);
        compressed[..PREFIX_SIZE as usize].copy_from_slice(&original[..PREFIX_SIZE as usize]);
        let archive = NczLoader::load(Arc::new(Bytes::new(compressed))).unwrap();
        let storage = archive.nca_storage();
        let nca = crate::NcaLoader::load(storage.clone()).unwrap();
        assert_eq!(nca.header().size(), original.len() as u64);
        let mut reconstructed = vec![0_u8; original.len()];
        storage.read_at(0, &mut reconstructed).unwrap();
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn cache_file_is_created_lazily_and_creation_failure_is_stable() {
        CACHE_CREATIONS.store(0, Ordering::Relaxed);
        let tail = vec![9; 1024];
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let source: StorageRef =
            Arc::new(Bytes::new(solid_ncz(&tail, std::slice::from_ref(&section))));
        let archive = parse_ncz_with_cache_factory(source, counted_cache_file).unwrap();
        assert_eq!(CACHE_CREATIONS.load(Ordering::Relaxed), 0);
        archive.nca_storage().read_at(0, &mut [0; 4]).unwrap();
        assert_eq!(CACHE_CREATIONS.load(Ordering::Relaxed), 0);
        archive
            .nca_storage()
            .read_at(PREFIX_SIZE, &mut [0; 1])
            .unwrap();
        assert_eq!(CACHE_CREATIONS.load(Ordering::Relaxed), 1);

        let source: StorageRef = Arc::new(Bytes::new(solid_ncz(&tail, &[section])));
        let failed = parse_ncz_with_cache_factory(source, failing_cache_file).unwrap();
        let first = failed
            .nca_storage()
            .read_at(PREFIX_SIZE, &mut [0; 1])
            .unwrap_err()
            .to_string();
        let second = failed
            .nca_storage()
            .read_at(PREFIX_SIZE, &mut [0; 1])
            .unwrap_err()
            .to_string();
        assert_eq!(first, second);
    }

    #[test]
    fn preserves_plaintext_gap_before_first_recorded_section() {
        let gap = vec![0x11; 0x200];
        let section_data = vec![0x22; 0x300];
        let mut tail = gap.clone();
        tail.extend_from_slice(&section_data);
        let section = section_record(
            PREFIX_SIZE + gap.len() as u64,
            section_data.len() as u64,
            1,
            [0; 16],
            [0; 16],
        );
        let archive = NczLoader::load(Arc::new(Bytes::new(solid_ncz(&tail, &[section])))).unwrap();
        assert_eq!(archive.sections().len(), 1);
        let mut actual = vec![0_u8; tail.len()];
        archive
            .nca_storage()
            .read_at(PREFIX_SIZE, &mut actual)
            .unwrap();
        assert_eq!(actual, tail);
    }

    #[test]
    fn block_mode_supports_reverse_and_cross_boundary_reads() {
        let tail = (0..45_000)
            .map(|index| (index * 7) as u8)
            .collect::<Vec<_>>();
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let archive =
            NczLoader::load(Arc::new(Bytes::new(block_ncz(&tail, 14, &[section])))).unwrap();
        let info = archive.block_info().unwrap();
        assert_eq!(info.block_size(), 16_384);
        assert_eq!(info.block_count(), 3);

        let storage = archive.nca_storage();
        let mut end = [0_u8; 41];
        storage.read_at(PREFIX_SIZE + 40_000, &mut end).unwrap();
        assert_eq!(&end, &tail[40_000..40_041]);

        let mut crossing = [0_u8; 64];
        storage
            .read_at(PREFIX_SIZE + 16_360, &mut crossing)
            .unwrap();
        assert_eq!(&crossing, &tail[16_360..16_424]);
    }

    #[test]
    fn block_mode_mixes_raw_and_zstandard_frames() {
        let mut state = 0x1234_5678_u32;
        let mut tail = (0..16_384)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect::<Vec<_>>();
        tail.extend_from_slice(&vec![0_u8; 16_384]);
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let bytes = block_ncz(&tail, 14, &[section]);
        let table = PREFIX_SIZE as usize
            + SECTION_HEADER_SIZE as usize
            + SECTION_SIZE as usize
            + BLOCK_HEADER_SIZE as usize;
        assert_eq!(read_u32(&bytes, table), 16_384);
        assert!(read_u32(&bytes, table + 4) < 16_384);

        let archive = NczLoader::load(Arc::new(Bytes::new(bytes))).unwrap();
        let mut actual = vec![0_u8; tail.len()];
        archive
            .nca_storage()
            .read_at(PREFIX_SIZE, &mut actual)
            .unwrap();
        assert_eq!(actual, tail);
    }

    #[test]
    fn late_block_read_does_not_decode_preceding_blocks_or_repeat_work() {
        let tail = (0..45_000)
            .map(|index| (index * 17) as u8)
            .collect::<Vec<_>>();
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let bytes = block_ncz(&tail, 14, &[section]);
        let block_header =
            PREFIX_SIZE as usize + SECTION_HEADER_SIZE as usize + SECTION_SIZE as usize;
        let table = block_header + BLOCK_HEADER_SIZE as usize;
        let first_size = read_u32(&bytes, table) as u64;
        let second_size = read_u32(&bytes, table + 4) as u64;
        let third_offset = (table + 12) as u64 + first_size + second_size;
        let (archive, source) = load_counted(bytes);
        source.offsets.lock().unwrap().clear();

        let mut output = [0_u8; 32];
        archive
            .nca_storage()
            .read_at(PREFIX_SIZE + 40_000, &mut output)
            .unwrap();
        assert_eq!(&output, &tail[40_000..40_032]);
        assert_eq!(&*source.offsets.lock().unwrap(), &[third_offset]);

        archive
            .nca_storage()
            .read_at(PREFIX_SIZE + 40_000, &mut output)
            .unwrap();
        assert_eq!(&*source.offsets.lock().unwrap(), &[third_offset]);
    }

    #[test]
    fn accepts_documented_maximum_block_exponent_with_bounded_io() {
        let section = section_record(PREFIX_SIZE, 1, 1, [0; 16], [0; 16]);
        let mut bytes = block_ncz(&[0x7f], 14, &[section]);
        let exponent_offset =
            PREFIX_SIZE as usize + SECTION_HEADER_SIZE as usize + SECTION_SIZE as usize + 11;
        bytes[exponent_offset] = 32;
        let archive = NczLoader::load(Arc::new(Bytes::new(bytes))).unwrap();
        assert_eq!(archive.block_info().unwrap().block_size(), 1_u64 << 32);
        let mut output = [0_u8; 1];
        archive
            .nca_storage()
            .read_at(PREFIX_SIZE, &mut output)
            .unwrap();
        assert_eq!(output, [0x7f]);
    }

    #[test]
    fn reconstructs_aes_ctr_sections_and_unaligned_reads() {
        let plaintext = (0..50_000)
            .map(|index| (index * 29) as u8)
            .collect::<Vec<_>>();
        let key = [0x42; 16];
        let mut counter = [0_u8; 16];
        counter[..8].copy_from_slice(&0x1020_3040_5060_7080_u64.to_be_bytes());
        let section = section_record(PREFIX_SIZE, plaintext.len() as u64, 3, key, counter);
        let archive =
            NczLoader::load(Arc::new(Bytes::new(block_ncz(&plaintext, 14, &[section])))).unwrap();

        let mut expected = plaintext.clone();
        apply_ctr_at(
            &key,
            counter[..8].try_into().unwrap(),
            PREFIX_SIZE,
            &mut expected,
        );
        let mut actual = vec![0_u8; 20_003];
        archive
            .nca_storage()
            .read_at(PREFIX_SIZE + 7, &mut actual)
            .unwrap();
        assert_eq!(actual, expected[7..20_010]);
    }

    #[test]
    fn one_shared_cache_is_safe_for_concurrent_reads() {
        let tail = vec![0x5a; 70_000];
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let archive =
            NczLoader::load(Arc::new(Bytes::new(block_ncz(&tail, 14, &[section])))).unwrap();
        let storage = archive.nca_storage();
        let handles = (0..8)
            .map(|index| {
                let storage = storage.clone();
                thread::spawn(move || {
                    let mut bytes = [0_u8; 1024];
                    storage
                        .read_at(PREFIX_SIZE + index * 4096, &mut bytes)
                        .unwrap();
                    assert_eq!(bytes, [0x5a; 1024]);
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn rejects_malformed_metadata_and_block_tables() {
        assert!(NczLoader::load(Arc::new(Bytes::new(vec![0; 8]))).is_err());

        let tail = vec![1; 20_000];
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let mut bad_magic = solid_ncz(&tail, std::slice::from_ref(&section));
        bad_magic[PREFIX_SIZE as usize] ^= 1;
        assert!(NczLoader::load(Arc::new(Bytes::new(bad_magic))).is_err());

        let mut bad_count = solid_ncz(&tail, std::slice::from_ref(&section));
        bad_count[PREFIX_SIZE as usize + 8..PREFIX_SIZE as usize + 16]
            .copy_from_slice(&0_u64.to_le_bytes());
        assert!(NczLoader::load(Arc::new(Bytes::new(bad_count))).is_err());

        let mut trailing = block_ncz(&tail, 14, &[section]);
        trailing.push(0);
        assert!(NczLoader::load(Arc::new(Bytes::new(trailing))).is_err());

        let first = section_record(PREFIX_SIZE, 100, 1, [0; 16], [0; 16]);
        let second = section_record(PREFIX_SIZE + 101, 100, 1, [0; 16], [0; 16]);
        assert!(
            NczLoader::load(Arc::new(Bytes::new(solid_ncz(
                &vec![0; 201],
                &[first, second]
            ))))
            .is_err()
        );

        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let mut bad_padding = solid_ncz(&tail, std::slice::from_ref(&section));
        bad_padding[PREFIX_SIZE as usize + SECTION_HEADER_SIZE as usize + 24] = 1;
        assert!(NczLoader::load(Arc::new(Bytes::new(bad_padding))).is_err());

        let mut bad_crypto = solid_ncz(&tail, std::slice::from_ref(&section));
        bad_crypto[PREFIX_SIZE as usize + SECTION_HEADER_SIZE as usize + 16] = 9;
        assert!(NczLoader::load(Arc::new(Bytes::new(bad_crypto))).is_err());

        let mut bad_exponent = block_ncz(&tail, 14, &[section]);
        let exponent_offset =
            PREFIX_SIZE as usize + SECTION_HEADER_SIZE as usize + SECTION_SIZE as usize + 11;
        bad_exponent[exponent_offset] = 13;
        assert!(NczLoader::load(Arc::new(Bytes::new(bad_exponent))).is_err());
    }

    #[test]
    fn corrupt_solid_stream_becomes_a_stable_terminal_error() {
        let tail = vec![7; 20_000];
        let section = section_record(PREFIX_SIZE, tail.len() as u64, 1, [0; 16], [0; 16]);
        let mut bytes = solid_ncz(&tail, &[section]);
        bytes.truncate(bytes.len() - 5);
        let archive = NczLoader::load(Arc::new(Bytes::new(bytes))).unwrap();
        let storage = archive.nca_storage();
        let mut output = vec![0_u8; tail.len()];
        let first = storage
            .read_at(PREFIX_SIZE, &mut output)
            .unwrap_err()
            .to_string();
        let second = storage
            .read_at(PREFIX_SIZE, &mut [0; 1])
            .unwrap_err()
            .to_string();
        assert_eq!(first, second);
    }
}
