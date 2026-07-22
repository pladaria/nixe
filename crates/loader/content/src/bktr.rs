use std::sync::Arc;

use nixe_loader_storage::{FormatLoader, LoadError, Storage, StorageError, StorageRef, SubStorage};

use crate::crypto::apply_ctr;
use crate::{
    BktrPatchInfo, NcaEncryptionType, NcaSection, NcaSectionType, RomFsArchive, RomFsLoader,
};

const FORMAT: &str = "BKTR";
const NODE_SIZE: u64 = 0x4000;
const NODE_HEADER_SIZE: u64 = 0x10;
const INDIRECT_ENTRY_SIZE: u64 = 0x14;
const SUBSECTION_ENTRY_SIZE: u64 = 0x10;
const AES_BLOCK_SIZE: u64 = 0x10;
const MAX_ENTRY_COUNT: u32 = 1_000_000;

/// The header stored in an NCA FS header for one BKTR bucket tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketTreeHeader {
    entry_count: u32,
}

impl BucketTreeHeader {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, LoadError> {
        if bytes.len() != 0x10 || &bytes[..4] != b"BKTR" {
            return Err(LoadError::invalid(FORMAT, "bucket-tree magic is invalid"));
        }
        let version = read_u32(bytes, 4);
        if version != 1 {
            return Err(LoadError::invalid(
                FORMAT,
                format!("unsupported bucket-tree version {version}"),
            ));
        }
        let entry_count = read_u32(bytes, 8);
        if entry_count == 0 || entry_count > MAX_ENTRY_COUNT {
            return Err(LoadError::invalid(
                FORMAT,
                format!("invalid bucket-tree entry count {entry_count}"),
            ));
        }
        if read_u32(bytes, 0xC) != 0 {
            return Err(LoadError::invalid(
                FORMAT,
                "bucket-tree header reserved field is nonzero",
            ));
        }
        Ok(Self { entry_count })
    }

    pub const fn entry_count(self) -> u32 {
        self.entry_count
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelocationSource {
    Base,
    Patch,
}

#[derive(Clone, Copy, Debug)]
struct RelocationEntry {
    virtual_offset: u64,
    source_offset: u64,
    source: RelocationSource,
}

#[derive(Clone, Copy, Debug)]
struct SubsectionEntry {
    offset: u64,
    encrypted: bool,
    generation: u32,
}

/// A lazy, read-only effective image composed from base and update NCA sections.
pub struct BktrPatch {
    virtual_storage: StorageRef,
    romfs_storage: StorageRef,
}

impl BktrPatch {
    /// Opens caller-selected base RomFS and update BKTR sections.
    ///
    /// The returned virtual storage uses effective-image offsets and forwards
    /// each read to the retained base or physical update storage as described
    /// by the relocation table. No complete image or file is materialized.
    pub fn open(base: &NcaSection, update: &NcaSection) -> Result<Self, LoadError> {
        if base.section_type() != NcaSectionType::RomFs {
            return Err(LoadError::invalid(
                "Patch RomFS",
                "base section is not an ordinary RomFS section",
            ));
        }
        if update.section_type() != NcaSectionType::Bktr
            || update.encryption_type() != NcaEncryptionType::AesCtrEx
        {
            return Err(LoadError::invalid(
                "Patch RomFS",
                "update section is not BKTR/AES-CTR-Ex",
            ));
        }

        let info = update
            .bktr_patch_info()
            .ok_or_else(|| LoadError::invalid("Patch RomFS", "update section has no PatchInfo"))?;
        let crypto = update.bktr_crypto().ok_or_else(|| {
            LoadError::invalid("Patch RomFS", "update section has no crypto context")
        })?;
        if !info.indirect_offset().is_multiple_of(AES_BLOCK_SIZE) {
            return Err(LoadError::invalid(
                FORMAT,
                "physical patch-data size is not AES-block aligned",
            ));
        }

        let subsection_entries = parse_subsection_table(update.storage(), info)?;
        let subsection_end = bucket_end(update.storage(), info.aes_ctr_ex_offset())?;
        if subsection_end != info.aes_ctr_ex_offset() {
            return Err(LoadError::invalid(
                FORMAT,
                "subsection tree does not cover its advertised physical domain",
            ));
        }
        validate_subsections(&subsection_entries, info.aes_ctr_ex_offset())?;
        let patch_data: StorageRef = Arc::new(AesCtrExStorage {
            parent: crypto.raw_storage.clone(),
            len: info.indirect_offset(),
            entries: subsection_entries,
            key: crypto.key,
            secure_value: crypto.secure_value,
            counter_offset: crypto.counter_offset,
            source_is_decrypted: crypto.source_is_decrypted,
        });

        let relocations = parse_relocation_table(update.storage(), info)?;
        let base_virtual_size = ivfc_image_size(base.romfs_ivfc().ok_or_else(|| {
            LoadError::invalid("Patch RomFS", "base section has no IVFC layout")
        })?)?;
        let virtual_size = validate_relocations(
            &relocations,
            base_virtual_size,
            patch_data.len()?,
            bucket_end(update.storage(), info.indirect_offset())?,
        )?;
        let virtual_storage: StorageRef = Arc::new(RelocationStorage {
            base: base.storage(),
            patch: patch_data,
            entries: relocations,
            len: virtual_size,
        });

        let data_level = update
            .bktr_ivfc()
            .and_then(|layout| layout.levels.last())
            .ok_or_else(|| LoadError::invalid("Patch RomFS", "virtual IVFC has no data level"))?;
        validate_range(
            data_level.offset,
            data_level.size,
            virtual_size,
            "virtual IVFC data level",
        )?;
        let romfs_storage: StorageRef = Arc::new(SubStorage::new(
            virtual_storage.clone(),
            data_level.offset,
            data_level.size,
        )?);

        Ok(Self {
            virtual_storage,
            romfs_storage,
        })
    }

    /// Returns the complete composed virtual IVFC image.
    pub fn virtual_storage(&self) -> StorageRef {
        self.virtual_storage.clone()
    }

    /// Returns the bounded final IVFC data level beginning at the RomFS header.
    pub fn romfs_storage(&self) -> StorageRef {
        self.romfs_storage.clone()
    }

    /// Parses the effective filesystem using the ordinary RomFS loader.
    pub fn load_romfs(&self) -> Result<RomFsArchive, LoadError> {
        RomFsLoader::load(self.romfs_storage())
    }
}

fn ivfc_image_size(layout: &crate::integrity::IvfcLayout) -> Result<u64, LoadError> {
    layout.levels.iter().try_fold(0_u64, |end, level| {
        let level_end = level
            .offset
            .checked_add(level.size)
            .ok_or_else(|| LoadError::invalid("Patch RomFS", "base IVFC range overflows"))?;
        Ok(end.max(level_end))
    })
}

fn parse_relocation_table(
    storage: StorageRef,
    info: &BktrPatchInfo,
) -> Result<Vec<RelocationEntry>, LoadError> {
    let raw = parse_bucket_entries(
        storage,
        info.indirect_offset(),
        info.indirect_size(),
        info.indirect_header(),
        INDIRECT_ENTRY_SIZE,
    )?;
    raw.into_iter()
        .map(|bytes| {
            let selector = read_u32(&bytes, 0x10);
            let source = match selector {
                0 => RelocationSource::Base,
                1 => RelocationSource::Patch,
                _ => {
                    return Err(LoadError::invalid(
                        FORMAT,
                        format!("invalid relocation source selector {selector}"),
                    ));
                }
            };
            Ok(RelocationEntry {
                virtual_offset: read_u64(&bytes, 0),
                source_offset: read_u64(&bytes, 8),
                source,
            })
        })
        .collect()
}

fn parse_subsection_table(
    storage: StorageRef,
    info: &BktrPatchInfo,
) -> Result<Vec<SubsectionEntry>, LoadError> {
    let raw = parse_bucket_entries(
        storage,
        info.aes_ctr_ex_offset(),
        info.aes_ctr_ex_size(),
        info.aes_ctr_ex_header(),
        SUBSECTION_ENTRY_SIZE,
    )?;
    raw.into_iter()
        .map(|bytes| {
            let encrypted = match bytes[8] {
                0 => true,
                1 => false,
                value => {
                    return Err(LoadError::invalid(
                        FORMAT,
                        format!("invalid subsection encryption selector {value}"),
                    ));
                }
            };
            Ok(SubsectionEntry {
                offset: read_u64(&bytes, 0),
                encrypted,
                generation: read_u32(&bytes, 0xC),
            })
        })
        .collect()
}

fn parse_bucket_entries(
    storage: StorageRef,
    table_offset: u64,
    table_size: u64,
    header: BucketTreeHeader,
    entry_size: u64,
) -> Result<Vec<Vec<u8>>, LoadError> {
    let entries_per_set = (NODE_SIZE - NODE_HEADER_SIZE) / entry_size;
    let entry_count = u64::from(header.entry_count());
    let set_count = entry_count.div_ceil(entries_per_set);
    let root_capacity = (NODE_SIZE - NODE_HEADER_SIZE) / 8;
    if set_count > root_capacity {
        return Err(LoadError::invalid(
            FORMAT,
            "bucket tree requires an unsupported second node level",
        ));
    }
    let entry_storage_size = set_count
        .checked_mul(NODE_SIZE)
        .ok_or_else(|| LoadError::invalid(FORMAT, "bucket-tree storage size overflows"))?;
    let required_size = NODE_SIZE
        .checked_add(entry_storage_size)
        .ok_or_else(|| LoadError::invalid(FORMAT, "bucket-tree storage size overflows"))?;
    if required_size > table_size {
        return Err(LoadError::invalid(
            FORMAT,
            "bucket-tree storage is truncated",
        ));
    }
    validate_range(
        table_offset,
        required_size,
        storage.len()?,
        "bucket-tree table",
    )?;

    let root = read_node_header(&storage, table_offset)?;
    if root.index != 0 || root.count != set_count || root.end_offset == 0 {
        return Err(LoadError::invalid(FORMAT, "invalid root bucket node"));
    }
    let root_offsets = read_offsets(&storage, table_offset + NODE_HEADER_SIZE, set_count)?;
    ensure_strictly_increasing(&root_offsets, "root bucket offsets")?;

    let mut result = Vec::with_capacity(
        usize::try_from(entry_count)
            .map_err(|_| LoadError::invalid(FORMAT, "entry count is too large"))?,
    );
    let entry_base = table_offset + NODE_SIZE;
    let mut previous_end = None;
    for set_index in 0..set_count {
        let set_offset = entry_base
            .checked_add(set_index * NODE_SIZE)
            .ok_or_else(|| LoadError::invalid(FORMAT, "entry-set offset overflows"))?;
        let node = read_node_header(&storage, set_offset)?;
        let remaining = entry_count - u64::try_from(result.len()).unwrap();
        let expected_count = remaining.min(entries_per_set);
        if node.index != set_index || node.count != expected_count || node.end_offset == 0 {
            return Err(LoadError::invalid(
                FORMAT,
                "invalid bucket entry-set header",
            ));
        }

        let mut set_entries = Vec::with_capacity(usize::try_from(expected_count).unwrap());
        for index in 0..expected_count {
            let offset = set_offset
                .checked_add(NODE_HEADER_SIZE + index * entry_size)
                .ok_or_else(|| LoadError::invalid(FORMAT, "bucket entry offset overflows"))?;
            let mut bytes = vec![0_u8; usize::try_from(entry_size).unwrap()];
            storage.read_at(offset, &mut bytes)?;
            set_entries.push(bytes);
        }
        let starts: Vec<u64> = set_entries.iter().map(|entry| read_u64(entry, 0)).collect();
        ensure_strictly_increasing(&starts, "bucket entries")?;
        if starts.first().copied()
            != root_offsets
                .get(usize::try_from(set_index).unwrap())
                .copied()
        {
            return Err(LoadError::invalid(
                FORMAT,
                "root and entry-set offsets disagree",
            ));
        }
        if let Some(end) = previous_end
            && starts.first().copied() != Some(end)
        {
            return Err(LoadError::invalid(FORMAT, "bucket entry sets have a gap"));
        }
        let set_start = starts[0];
        if set_start >= node.end_offset {
            return Err(LoadError::invalid(FORMAT, "bucket entry set is reversed"));
        }
        previous_end = Some(node.end_offset);
        result.extend(set_entries);
    }
    if previous_end != Some(root.end_offset)
        || result.len() != usize::try_from(entry_count).unwrap()
    {
        return Err(LoadError::invalid(
            FORMAT,
            "bucket-tree entry count is inconsistent",
        ));
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct NodeHeader {
    index: u64,
    count: u64,
    end_offset: u64,
}

fn read_node_header(storage: &StorageRef, offset: u64) -> Result<NodeHeader, LoadError> {
    let mut bytes = [0_u8; 0x10];
    storage.read_at(offset, &mut bytes)?;
    let index = read_i32(&bytes, 0);
    let count = read_i32(&bytes, 4);
    let end = read_i64(&bytes, 8);
    if index < 0 || count <= 0 || end <= 0 {
        return Err(LoadError::invalid(FORMAT, "invalid bucket node header"));
    }
    Ok(NodeHeader {
        index: index as u64,
        count: count as u64,
        end_offset: end as u64,
    })
}

fn read_offsets(storage: &StorageRef, offset: u64, count: u64) -> Result<Vec<u64>, LoadError> {
    let mut result = Vec::with_capacity(usize::try_from(count).unwrap());
    for index in 0..count {
        let mut bytes = [0_u8; 8];
        storage.read_at(offset + index * 8, &mut bytes)?;
        let value = i64::from_le_bytes(bytes);
        if value < 0 {
            return Err(LoadError::invalid(FORMAT, "bucket offset is negative"));
        }
        result.push(value as u64);
    }
    Ok(result)
}

fn ensure_strictly_increasing(values: &[u64], description: &str) -> Result<(), LoadError> {
    if values.is_empty() || values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(LoadError::invalid(
            FORMAT,
            format!("{description} are not strictly increasing"),
        ));
    }
    Ok(())
}

fn bucket_end(storage: StorageRef, table_offset: u64) -> Result<u64, LoadError> {
    let root = read_node_header(&storage, table_offset)?;
    Ok(root.end_offset)
}

fn validate_relocations(
    entries: &[RelocationEntry],
    base_len: u64,
    patch_len: u64,
    virtual_size: u64,
) -> Result<u64, LoadError> {
    if entries.first().map(|entry| entry.virtual_offset) != Some(0) {
        return Err(LoadError::invalid(
            FORMAT,
            "relocation map does not begin at zero",
        ));
    }
    let starts: Vec<u64> = entries.iter().map(|entry| entry.virtual_offset).collect();
    ensure_strictly_increasing(&starts, "relocation offsets")?;
    for (index, entry) in entries.iter().enumerate() {
        let end = entries
            .get(index + 1)
            .map_or(virtual_size, |next| next.virtual_offset);
        let size = end
            .checked_sub(entry.virtual_offset)
            .ok_or_else(|| LoadError::invalid(FORMAT, "relocation range is reversed"))?;
        let source_len = match entry.source {
            RelocationSource::Base => base_len,
            RelocationSource::Patch => patch_len,
        };
        validate_range(
            entry.source_offset,
            size,
            source_len,
            "relocation source range",
        )?;
    }
    Ok(virtual_size)
}

fn validate_subsections(entries: &[SubsectionEntry], physical_end: u64) -> Result<(), LoadError> {
    if entries.first().map(|entry| entry.offset) != Some(0) {
        return Err(LoadError::invalid(
            FORMAT,
            "subsection map does not begin at zero",
        ));
    }
    let starts: Vec<u64> = entries.iter().map(|entry| entry.offset).collect();
    ensure_strictly_increasing(&starts, "subsection offsets")?;
    if starts
        .iter()
        .any(|offset| !offset.is_multiple_of(AES_BLOCK_SIZE))
        || physical_end == 0
        || !physical_end.is_multiple_of(AES_BLOCK_SIZE)
        || starts.last().copied().unwrap() >= physical_end
    {
        return Err(LoadError::invalid(FORMAT, "subsection coverage is invalid"));
    }
    Ok(())
}

struct RelocationStorage {
    base: StorageRef,
    patch: StorageRef,
    entries: Vec<RelocationEntry>,
    len: u64,
}

impl Storage for RelocationStorage {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.len)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        storage_range(offset, buffer.len(), self.len)?;
        if buffer.is_empty() {
            return Ok(());
        }
        let end = offset + u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?;
        let mut current = offset;
        while current < end {
            let index = self
                .entries
                .partition_point(|entry| entry.virtual_offset <= current)
                - 1;
            let entry = self.entries[index];
            let next = self
                .entries
                .get(index + 1)
                .map_or(self.len, |entry| entry.virtual_offset);
            let chunk_end = end.min(next);
            let chunk_len =
                usize::try_from(chunk_end - current).map_err(|_| StorageError::OutOfBounds)?;
            let output =
                usize::try_from(current - offset).map_err(|_| StorageError::OutOfBounds)?;
            let source_offset = entry
                .source_offset
                .checked_add(current - entry.virtual_offset)
                .ok_or(StorageError::OutOfBounds)?;
            let source = match entry.source {
                RelocationSource::Base => &self.base,
                RelocationSource::Patch => &self.patch,
            };
            source.read_at(source_offset, &mut buffer[output..output + chunk_len])?;
            current = chunk_end;
        }
        Ok(())
    }
}

struct AesCtrExStorage {
    parent: StorageRef,
    len: u64,
    entries: Vec<SubsectionEntry>,
    key: Option<[u8; 16]>,
    secure_value: u32,
    counter_offset: u64,
    source_is_decrypted: bool,
}

impl Storage for AesCtrExStorage {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.len)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
        storage_range(offset, buffer.len(), self.len)?;
        if buffer.is_empty() {
            return Ok(());
        }
        let end = offset + u64::try_from(buffer.len()).map_err(|_| StorageError::OutOfBounds)?;
        let mut current = offset;
        while current < end {
            let index = self
                .entries
                .partition_point(|entry| entry.offset <= current)
                - 1;
            let entry = self.entries[index];
            let next = self
                .entries
                .get(index + 1)
                .map_or(self.len, |entry| entry.offset);
            let chunk_end = end.min(next);
            let chunk_len =
                usize::try_from(chunk_end - current).map_err(|_| StorageError::OutOfBounds)?;
            let output =
                usize::try_from(current - offset).map_err(|_| StorageError::OutOfBounds)?;

            if self.source_is_decrypted || !entry.encrypted {
                self.parent
                    .read_at(current, &mut buffer[output..output + chunk_len])?;
            } else {
                let key = self.key.ok_or(StorageError::OutOfBounds)?;
                let aligned_start = current & !(AES_BLOCK_SIZE - 1);
                let aligned_end = align_up(chunk_end, AES_BLOCK_SIZE)?;
                let mut encrypted = vec![
                    0_u8;
                    usize::try_from(aligned_end - aligned_start)
                        .map_err(|_| StorageError::OutOfBounds)?
                ];
                self.parent.read_at(aligned_start, &mut encrypted)?;
                let mut prefix = [0_u8; 8];
                prefix[..4].copy_from_slice(&self.secure_value.to_be_bytes());
                prefix[4..].copy_from_slice(&entry.generation.to_be_bytes());
                let absolute = self
                    .counter_offset
                    .checked_add(aligned_start)
                    .ok_or(StorageError::OutOfBounds)?;
                apply_ctr(&key, prefix, absolute / AES_BLOCK_SIZE, &mut encrypted);
                let within = usize::try_from(current - aligned_start)
                    .map_err(|_| StorageError::OutOfBounds)?;
                buffer[output..output + chunk_len]
                    .copy_from_slice(&encrypted[within..within + chunk_len]);
            }
            current = chunk_end;
        }
        Ok(())
    }
}

fn storage_range(offset: u64, len: usize, total: u64) -> Result<(), StorageError> {
    let end = offset
        .checked_add(u64::try_from(len).map_err(|_| StorageError::OutOfBounds)?)
        .ok_or(StorageError::OutOfBounds)?;
    if end > total {
        return Err(StorageError::OutOfBounds);
    }
    Ok(())
}

fn validate_range(offset: u64, size: u64, total: u64, name: &str) -> Result<(), LoadError> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| LoadError::invalid(FORMAT, format!("{name} overflows")))?;
    if end > total {
        return Err(LoadError::invalid(
            FORMAT,
            format!("{name} is outside its source storage"),
        ));
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, StorageError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(StorageError::OutOfBounds)
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn read_i64(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::NcaLoader;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            Ok(self.0.len() as u64)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            storage_range(offset, buffer.len(), self.0.len() as u64)?;
            let start = offset as usize;
            buffer.copy_from_slice(&self.0[start..start + buffer.len()]);
            Ok(())
        }
    }

    fn storage(bytes: &[u8]) -> StorageRef {
        Arc::new(VecStorage(bytes.to_vec()))
    }

    fn header(entry_count: u32) -> BucketTreeHeader {
        BucketTreeHeader { entry_count }
    }

    fn build_bucket_table(entries: &[Vec<u8>], entry_size: usize, end_offset: u64) -> Vec<u8> {
        let entries_per_set = (NODE_SIZE as usize - NODE_HEADER_SIZE as usize) / entry_size;
        let set_count = entries.len().div_ceil(entries_per_set);
        let mut table = vec![0_u8; NODE_SIZE as usize * (1 + set_count)];
        put_i32(&mut table, 0, 0);
        put_i32(&mut table, 4, set_count as i32);
        put_u64(&mut table, 8, end_offset);

        for (set_index, chunk) in entries.chunks(entries_per_set).enumerate() {
            let first = read_u64(&chunk[0], 0);
            put_u64(&mut table, 0x10 + set_index * 8, first);
            let set = NODE_SIZE as usize * (set_index + 1);
            put_i32(&mut table, set, set_index as i32);
            put_i32(&mut table, set + 4, chunk.len() as i32);
            let set_end = entries
                .get((set_index + 1) * entries_per_set)
                .map_or(end_offset, |entry| read_u64(entry, 0));
            put_u64(&mut table, set + 8, set_end);
            for (index, entry) in chunk.iter().enumerate() {
                let offset = set + 0x10 + index * entry_size;
                table[offset..offset + entry_size].copy_from_slice(entry);
            }
        }
        table
    }

    fn relocation_entry(virtual_offset: u64, source_offset: u64, selector: u32) -> Vec<u8> {
        let mut entry = vec![0_u8; INDIRECT_ENTRY_SIZE as usize];
        put_u64(&mut entry, 0, virtual_offset);
        put_u64(&mut entry, 8, source_offset);
        entry[0x10..0x14].copy_from_slice(&selector.to_le_bytes());
        entry
    }

    fn subsection_entry(offset: u64, encrypted: bool, generation: u32) -> Vec<u8> {
        let mut entry = vec![0_u8; SUBSECTION_ENTRY_SIZE as usize];
        put_u64(&mut entry, 0, offset);
        entry[8] = u8::from(!encrypted);
        entry[0xC..0x10].copy_from_slice(&generation.to_le_bytes());
        entry
    }

    fn build_romfs(files: &[(&str, &[u8])]) -> Vec<u8> {
        const EMPTY: u32 = u32::MAX;
        let mut file_meta = Vec::new();
        let mut data_offset = 0_u64;
        for (index, (name, data)) in files.iter().enumerate() {
            let next = if index + 1 == files.len() {
                EMPTY
            } else {
                (file_meta.len() + 0x20 + name.len().next_multiple_of(4)) as u32
            };
            file_meta.extend_from_slice(&0_u32.to_le_bytes());
            file_meta.extend_from_slice(&next.to_le_bytes());
            file_meta.extend_from_slice(&data_offset.to_le_bytes());
            file_meta.extend_from_slice(&(data.len() as u64).to_le_bytes());
            file_meta.extend_from_slice(&EMPTY.to_le_bytes());
            file_meta.extend_from_slice(&(name.len() as u32).to_le_bytes());
            file_meta.extend_from_slice(name.as_bytes());
            while file_meta.len() % 4 != 0 {
                file_meta.push(0);
            }
            data_offset += data.len() as u64;
        }

        let file_data_offset = (0x70_u64 + file_meta.len() as u64).next_multiple_of(0x10);
        let mut bytes = vec![0_u8; file_data_offset as usize];
        for (offset, value) in [
            (0, 0x50),
            (0x08, 0x50),
            (0x10, 4),
            (0x18, 0x54),
            (0x20, 0x18),
            (0x28, 0x6C),
            (0x30, 4),
            (0x38, 0x70),
            (0x40, file_meta.len() as u64),
            (0x48, file_data_offset),
        ] {
            put_u64(&mut bytes, offset, value);
        }
        let root = 0x54;
        put_u32(&mut bytes, root, 0);
        put_u32(&mut bytes, root + 4, EMPTY);
        put_u32(&mut bytes, root + 8, EMPTY);
        put_u32(
            &mut bytes,
            root + 0xC,
            if files.is_empty() { EMPTY } else { 0 },
        );
        put_u32(&mut bytes, root + 0x10, EMPTY);
        put_u32(&mut bytes, root + 0x14, 0);
        bytes[0x70..0x70 + file_meta.len()].copy_from_slice(&file_meta);
        for (_, data) in files {
            bytes.extend_from_slice(data);
        }
        bytes
    }

    fn ivfc_header(fs: &mut [u8], data_size: u64) {
        fs[0x08..0x0C].copy_from_slice(b"IVFC");
        put_u32(fs, 0x10, 0x20);
        put_u32(fs, 0x14, 2);
        put_u64(fs, 0x18, 0);
        put_u64(fs, 0x20, data_size);
        put_u32(fs, 0x28, 20);
    }

    fn build_nca(section: Vec<u8>, fs: [u8; 0x200]) -> Vec<u8> {
        const SECTION_OFFSET: usize = 0xC00;
        let section_size = section.len().next_multiple_of(0x200);
        let mut nca = vec![0_u8; SECTION_OFFSET + section_size];
        nca[SECTION_OFFSET..SECTION_OFFSET + section.len()].copy_from_slice(&section);
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x205] = 0;
        nca[0x206] = 1;
        put_u64(&mut nca, 0x208, (SECTION_OFFSET + section_size) as u64);
        put_u32(&mut nca, 0x240, (SECTION_OFFSET / 0x200) as u32);
        put_u32(
            &mut nca,
            0x244,
            ((SECTION_OFFSET + section_size) / 0x200) as u32,
        );
        nca[0x400..0x600].copy_from_slice(&fs);
        let hash: [u8; 32] = Sha256::digest(fs).into();
        nca[0x280..0x2A0].copy_from_slice(&hash);
        nca
    }

    fn load_nca(bytes: Vec<u8>) -> crate::NcaArchive {
        NcaLoader::load(Arc::new(VecStorage(bytes))).unwrap()
    }

    #[test]
    fn relocation_reads_across_base_and_patch_entries() {
        let view = RelocationStorage {
            base: storage(b"abcdefgh"),
            patch: storage(b"12345678"),
            entries: vec![
                RelocationEntry {
                    virtual_offset: 0,
                    source_offset: 1,
                    source: RelocationSource::Base,
                },
                RelocationEntry {
                    virtual_offset: 3,
                    source_offset: 2,
                    source: RelocationSource::Patch,
                },
                RelocationEntry {
                    virtual_offset: 6,
                    source_offset: 0,
                    source: RelocationSource::Base,
                },
            ],
            len: 8,
        };
        let mut output = [0_u8; 8];
        view.read_at(0, &mut output).unwrap();
        assert_eq!(&output, b"bcd345ab");
    }

    #[test]
    fn decrypted_ctr_ex_view_splits_subsections_without_transforming_bytes() {
        let view = AesCtrExStorage {
            parent: storage(b"0123456789abcdefABCDEFGHIJKLMNOP"),
            len: 32,
            entries: vec![
                SubsectionEntry {
                    offset: 0,
                    encrypted: true,
                    generation: 1,
                },
                SubsectionEntry {
                    offset: 16,
                    encrypted: true,
                    generation: 9,
                },
            ],
            key: None,
            secure_value: 7,
            counter_offset: 0xC00,
            source_is_decrypted: true,
        };
        let mut output = [0_u8; 20];
        view.read_at(7, &mut output).unwrap();
        assert_eq!(&output, b"789abcdefABCDEFGHIJK");
    }

    #[test]
    fn encrypted_ctr_ex_view_honors_generation_boundaries_and_unaligned_reads() {
        let key = [0x27; 16];
        let secure_value = 0x1234_5678_u32;
        let counter_offset = 0xC00;
        let generations = [3_u32, 11_u32];
        let plaintext = *b"0123456789abcdefABCDEFGHIJKLMNOP";
        let mut ciphertext = plaintext;
        for (index, chunk) in ciphertext.chunks_exact_mut(16).enumerate() {
            let mut prefix = [0_u8; 8];
            prefix[..4].copy_from_slice(&secure_value.to_be_bytes());
            prefix[4..].copy_from_slice(&generations[index].to_be_bytes());
            apply_ctr(
                &key,
                prefix,
                (counter_offset + index as u64 * 16) / 16,
                chunk,
            );
        }
        let view = AesCtrExStorage {
            parent: storage(&ciphertext),
            len: 32,
            entries: vec![
                SubsectionEntry {
                    offset: 0,
                    encrypted: true,
                    generation: generations[0],
                },
                SubsectionEntry {
                    offset: 16,
                    encrypted: true,
                    generation: generations[1],
                },
            ],
            key: Some(key),
            secure_value,
            counter_offset,
            source_is_decrypted: false,
        };
        let mut output = [0_u8; 20];
        view.read_at(7, &mut output).unwrap();
        assert_eq!(&output, &plaintext[7..27]);
    }

    #[test]
    fn parses_multiple_bucket_entry_sets() {
        let per_set = (NODE_SIZE as usize - 0x10) / INDIRECT_ENTRY_SIZE as usize;
        let entries: Vec<_> = (0..=per_set)
            .map(|index| relocation_entry(index as u64 * 2, index as u64, 0))
            .collect();
        let end = entries.len() as u64 * 2;
        let table = build_bucket_table(&entries, INDIRECT_ENTRY_SIZE as usize, end);
        let parsed = parse_bucket_entries(
            storage(&table),
            0,
            table.len() as u64,
            header(entries.len() as u32),
            INDIRECT_ENTRY_SIZE,
        )
        .unwrap();
        assert_eq!(parsed.len(), entries.len());
        assert_eq!(
            read_u64(parsed.last().unwrap(), 0),
            (entries.len() as u64 - 1) * 2
        );
    }

    #[test]
    fn composes_synthetic_ncas_and_reuses_romfs_loader() {
        let base_romfs = build_romfs(&[
            ("keep", b"same"),
            ("replace", b"old!"),
            ("removed", b"gone"),
        ]);
        let final_romfs = build_romfs(&[
            ("keep", b"same"),
            ("replace", b"new!"),
            ("addedxx", b"plus"),
        ]);
        assert_eq!(base_romfs.len(), final_romfs.len());

        let mut base_fs = [0_u8; 0x200];
        base_fs[3] = 3;
        base_fs[4] = 1;
        ivfc_header(&mut base_fs, base_romfs.len() as u64);
        let base = load_nca(build_nca(base_romfs.clone(), base_fs));
        let base_archive =
            RomFsLoader::load(base.sections()[0].payload_storage().unwrap()).unwrap();
        assert!(base_archive.open("/removed").unwrap().is_some());

        let mut relocations = Vec::new();
        let mut start = 0_usize;
        let mut from_base = base_romfs[0] == final_romfs[0];
        for index in 1..final_romfs.len() {
            let next_from_base = base_romfs[index] == final_romfs[index];
            if next_from_base != from_base {
                relocations.push(relocation_entry(
                    start as u64,
                    start as u64,
                    u32::from(!from_base),
                ));
                start = index;
                from_base = next_from_base;
            }
        }
        relocations.push(relocation_entry(
            start as u64,
            start as u64,
            u32::from(!from_base),
        ));

        let indirect_offset = final_romfs.len().next_multiple_of(0x200);
        let indirect_table = build_bucket_table(
            &relocations,
            INDIRECT_ENTRY_SIZE as usize,
            final_romfs.len() as u64,
        );
        let aes_offset = indirect_offset + indirect_table.len();
        let subsection_table = build_bucket_table(
            &[subsection_entry(0, false, 0)],
            SUBSECTION_ENTRY_SIZE as usize,
            aes_offset as u64,
        );
        let mut patch_section = vec![0_u8; aes_offset + subsection_table.len()];
        patch_section[..final_romfs.len()].copy_from_slice(&final_romfs);
        patch_section[indirect_offset..indirect_offset + indirect_table.len()]
            .copy_from_slice(&indirect_table);
        patch_section[aes_offset..].copy_from_slice(&subsection_table);

        let mut patch_fs = [0_u8; 0x200];
        patch_fs[3] = 3;
        patch_fs[4] = 4;
        ivfc_header(&mut patch_fs, final_romfs.len() as u64);
        put_u64(&mut patch_fs, 0x100, indirect_offset as u64);
        put_u64(&mut patch_fs, 0x108, indirect_table.len() as u64);
        write_bucket_header(&mut patch_fs[0x110..0x120], relocations.len() as u32);
        put_u64(&mut patch_fs, 0x120, aes_offset as u64);
        put_u64(&mut patch_fs, 0x128, subsection_table.len() as u64);
        write_bucket_header(&mut patch_fs[0x130..0x140], 1);
        let patch = load_nca(build_nca(patch_section, patch_fs));

        let composed = BktrPatch::open(&base.sections()[0], &patch.sections()[0]).unwrap();
        let archive = composed.load_romfs().unwrap();
        assert!(archive.open("/keep").unwrap().is_some());
        assert!(archive.open("/replace").unwrap().is_some());
        assert!(archive.open("/addedxx").unwrap().is_some());
        assert!(archive.open("/removed").unwrap().is_none());
        let mut replacement = [0_u8; 4];
        archive
            .open("/replace")
            .unwrap()
            .unwrap()
            .read_at(0, &mut replacement)
            .unwrap();
        assert_eq!(&replacement, b"new!");
    }

    #[test]
    fn rejects_gapped_bucket_entry_sets() {
        let per_set = (NODE_SIZE as usize - 0x10) / INDIRECT_ENTRY_SIZE as usize;
        let entries: Vec<_> = (0..=per_set)
            .map(|index| relocation_entry(index as u64 * 2, index as u64, 0))
            .collect();
        let end = entries.len() as u64 * 2;
        let mut table = build_bucket_table(&entries, INDIRECT_ENTRY_SIZE as usize, end);
        let first_set_end = NODE_SIZE as usize + 8;
        put_u64(&mut table, first_set_end, end - 1);
        assert!(
            parse_bucket_entries(
                storage(&table),
                0,
                table.len() as u64,
                header(entries.len() as u32),
                INDIRECT_ENTRY_SIZE,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_invalid_relocation_source_ranges() {
        let entries = [RelocationEntry {
            virtual_offset: 0,
            source_offset: 7,
            source: RelocationSource::Base,
        }];
        assert!(validate_relocations(&entries, 8, 8, 2).is_err());
    }

    #[test]
    fn header_rejects_bad_magic_version_and_reserved_data() {
        let mut header = [0_u8; 0x10];
        header[..4].copy_from_slice(b"BKTR");
        header[4..8].copy_from_slice(&1_u32.to_le_bytes());
        header[8..12].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(BucketTreeHeader::parse(&header).unwrap().entry_count(), 1);
        header[4] = 2;
        assert!(BucketTreeHeader::parse(&header).is_err());
        header[4] = 1;
        header[12] = 1;
        assert!(BucketTreeHeader::parse(&header).is_err());
        header[..4].copy_from_slice(b"NOPE");
        assert!(BucketTreeHeader::parse(&header).is_err());
    }

    fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_bucket_header(bytes: &mut [u8], entry_count: u32) {
        bytes[..4].copy_from_slice(b"BKTR");
        bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&entry_count.to_le_bytes());
    }
}
