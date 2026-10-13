//! Flat RomFS image for generated system resources, including real hash chains.
//! https://switchbrew.org/wiki/RomFS
use std::collections::BTreeMap;

pub(super) fn build(files: &BTreeMap<String, Vec<u8>>) -> Result<Vec<u8>, String> {
    const NONE: u32 = u32::MAX;
    let mut metadata = Vec::new();
    let mut data = Vec::new();
    let mut offsets = Vec::new();
    // One bucket is valid; every entry participates in the collision chain.
    for (name, bytes) in files {
        if name.is_empty() || name.contains('/') || name.contains('\0') {
            return Err("invalid generated RomFS file name".into());
        }
        offsets.push(
            u32::try_from(metadata.len())
                .map_err(|_| "generated RomFS metadata exceeds 32 bits")?,
        );
        data.resize(data.len().next_multiple_of(16), 0);
        metadata.extend(0_u32.to_le_bytes()); // parent directory
        metadata.extend(NONE.to_le_bytes()); // sibling, linked below
        metadata.extend((data.len() as u64).to_le_bytes());
        metadata.extend((bytes.len() as u64).to_le_bytes());
        metadata.extend(
            offsets
                .iter()
                .rev()
                .nth(1)
                .copied()
                .unwrap_or(NONE)
                .to_le_bytes(),
        );
        metadata.extend(
            u32::try_from(name.len())
                .map_err(|_| "RomFS name exceeds 32 bits")?
                .to_le_bytes(),
        );
        metadata.extend(name.as_bytes());
        metadata.resize(metadata.len().next_multiple_of(4), 0);
        data.extend(bytes);
    }
    for pair in offsets.windows(2) {
        metadata[pair[0] as usize + 4..pair[0] as usize + 8]
            .copy_from_slice(&pair[1].to_le_bytes());
    }
    let directory_hash_offset = 0x50_u64;
    let directory_offset = directory_hash_offset + 4;
    let file_hash_offset = directory_offset + 24;
    let file_offset = file_hash_offset + 4;
    let data_offset = (file_offset + metadata.len() as u64).next_multiple_of(0x1000);
    let mut image = Vec::new();
    for value in [
        0x50,
        directory_hash_offset,
        4,
        directory_offset,
        24,
        file_hash_offset,
        4,
        file_offset,
        metadata.len() as u64,
        data_offset,
    ] {
        image.extend(value.to_le_bytes());
    }
    image.extend(0_u32.to_le_bytes()); // root hash bucket
    for value in [
        0,
        NONE,
        NONE,
        offsets.first().copied().unwrap_or(NONE),
        NONE,
        0,
    ] {
        image.extend(value.to_le_bytes());
    }
    image.extend(offsets.last().copied().unwrap_or(NONE).to_le_bytes());
    image.extend(metadata);
    image.resize(
        usize::try_from(data_offset).map_err(|_| "generated RomFS exceeds host address space")?,
        0,
    );
    image.extend(data);
    Ok(image)
}
