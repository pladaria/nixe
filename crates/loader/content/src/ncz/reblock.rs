//! Streaming solid-to-block conversion, without NCA reconstruction or keys.
//!
//! https://github.com/nicoboss/nsz/blob/master/docs/formats.md

use sha2::{Digest, Sha256};

use super::*;

const BLOCK_EXPONENT: u8 = 20;
const BLOCK_SIZE: usize = 1 << BLOCK_EXPONENT;

fn header_end(archive: &NczArchive) -> u64 {
    PREFIX_SIZE + SECTION_HEADER_SIZE + archive.sections.len() as u64 * SECTION_SIZE
}

/// Returns a digest of the unchanged header and the decompressed NCZ payload.
/// Hashing before reconstruction avoids decrypting and re-encrypting any NCA.
pub(crate) fn reblock_solid(
    source: StorageRef,
    archive: &NczArchive,
    output: &mut File,
    progress: &mut impl FnMut(u64),
) -> Result<[u8; 32], LoadError> {
    let compressed_offset = header_end(archive);
    let mut header = vec![0; compressed_offset as usize];
    source.read_at(0, &mut header)?;
    output.write_all(&header).map_err(StorageError::from)?;
    let mut digest = Sha256::new();
    digest.update(&header);

    let tail_size = archive.logical_size - PREFIX_SIZE;
    let block_count = tail_size.div_ceil(BLOCK_SIZE as u64);
    if block_count > MAX_BLOCK_COUNT {
        return Err(LoadError::invalid(
            "NCZ",
            "converted block count exceeds the safety limit",
        ));
    }
    output.write_all(b"NCZBLOCK").map_err(StorageError::from)?;
    output
        .write_all(&[2, 1, 0, BLOCK_EXPONENT])
        .map_err(StorageError::from)?;
    output
        .write_all(&(block_count as u32).to_le_bytes())
        .map_err(StorageError::from)?;
    output
        .write_all(&tail_size.to_le_bytes())
        .map_err(StorageError::from)?;
    let table_offset = output.stream_position().map_err(StorageError::from)?;
    let mut sizes = vec![0_u8; block_count as usize * 4];
    output.write_all(&sizes).map_err(StorageError::from)?;

    let compressed_end = source.len()?;
    let reader = StorageReader {
        source,
        position: compressed_offset,
        end: compressed_end,
    };
    let mut decoder = zstd::stream::read::Decoder::new(reader)
        .map_err(StorageError::from)?
        .single_frame();
    decoder
        .set_parameter(zstd::zstd_safe::DParameter::WindowLogMax(
            MAX_ZSTD_WINDOW_LOG,
        ))
        .map_err(StorageError::from)?;
    let mut compressor = zstd::bulk::Compressor::new(3).map_err(StorageError::from)?;
    let mut plain = vec![0; BLOCK_SIZE];
    let mut compressed = vec![0; zstd::zstd_safe::compress_bound(BLOCK_SIZE)];
    let mut produced = 0;
    for size_slot in sizes.chunks_exact_mut(4) {
        let count = (tail_size - produced).min(BLOCK_SIZE as u64) as usize;
        let block = &mut plain[..count];
        decoder
            .read_exact(block)
            .map_err(|error| invalid_data(format!("cannot convert NCZ stream: {error}")))?;
        digest.update(&*block);
        let compressed_size = compressor
            .compress_to_buffer(block, &mut compressed[..])
            .map_err(StorageError::from)?;
        let stored = if compressed_size < count {
            &compressed[..compressed_size]
        } else {
            &*block
        };
        output.write_all(stored).map_err(StorageError::from)?;
        size_slot.copy_from_slice(&(stored.len() as u32).to_le_bytes());
        produced += count as u64;
        progress(produced);
    }
    let mut extra = [0; 1];
    if decoder.read(&mut extra).map_err(StorageError::from)? != 0 {
        return Err(LoadError::invalid(
            "NCZ",
            "solid stream produces more data than declared",
        ));
    }
    let reader = decoder.finish();
    if reader.get_ref().position - reader.buffer().len() as u64 != compressed_end {
        return Err(LoadError::invalid(
            "NCZ",
            "trailing data follows the solid stream",
        ));
    }
    let end = output.stream_position().map_err(StorageError::from)?;
    output
        .seek(SeekFrom::Start(table_offset))
        .map_err(StorageError::from)?;
    output.write_all(&sizes).map_err(StorageError::from)?;
    output
        .seek(SeekFrom::Start(end))
        .map_err(StorageError::from)?;
    Ok(digest.finalize().into())
}

/// Checks the actual output on disk using bounded memory and no temporary NCA.
pub(crate) fn verify_reblocked(
    source: StorageRef,
    archive: &NczArchive,
    expected: [u8; 32],
    progress: &mut impl FnMut(u64),
) -> Result<(), LoadError> {
    let info = archive
        .block_info()
        .ok_or_else(|| LoadError::invalid("NCZ", "converted entry is not block compressed"))?;
    if info.block_size != BLOCK_SIZE as u64 {
        return Err(LoadError::invalid(
            "NCZ",
            "converted entry has an unexpected block size",
        ));
    }
    let header_end = header_end(archive);
    let mut header = vec![0; header_end as usize];
    source.read_at(0, &mut header)?;
    let mut digest = Sha256::new();
    digest.update(&header);
    let mut sizes = vec![0; info.block_count as usize * 4];
    source.read_at(header_end + BLOCK_HEADER_SIZE, &mut sizes)?;
    let mut offset = header_end + BLOCK_HEADER_SIZE + sizes.len() as u64;
    let mut compressed = vec![0; BLOCK_SIZE];
    let mut plain = vec![0; BLOCK_SIZE];
    let mut decoder = zstd::bulk::Decompressor::new().map_err(StorageError::from)?;
    decoder
        .window_log_max(MAX_ZSTD_WINDOW_LOG)
        .map_err(StorageError::from)?;
    let mut produced = 0;
    for size_slot in sizes.chunks_exact(4) {
        let size = read_u32(size_slot, 0) as usize;
        let expected_size = (info.decompressed_size - produced).min(BLOCK_SIZE as u64) as usize;
        source.read_at(offset, &mut compressed[..size])?;
        if size == expected_size {
            digest.update(&compressed[..size]);
        } else {
            let decoded = decoder
                .decompress_to_buffer(&compressed[..size], &mut plain[..expected_size])
                .map_err(StorageError::from)?;
            if decoded != expected_size {
                return Err(LoadError::invalid(
                    "NCZ",
                    "converted block has an unexpected decoded size",
                ));
            }
            digest.update(&plain[..decoded]);
        }
        offset += size as u64;
        produced += expected_size as u64;
        progress(produced);
    }
    if <[u8; 32]>::from(digest.finalize()) != expected {
        return Err(LoadError::invalid(
            "NCZ",
            "converted content does not match the original solid stream",
        ));
    }
    Ok(())
}
