use std::sync::Arc;

use sha2::{Digest, Sha256};
use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef, SubStorage};

use crate::crypto::{ctr_storage, decrypt_ecb_blocks, decrypt_xts, xts_storage};
use crate::integrity::{
    self, IntegrityLayout, IntegrityReport, IvfcLayout, IvfcLevel, Sha256Layout,
};
use crate::keys::{KeyAreaKeyIndex, NcaKeyProvider};

const HEADER_SIZE: usize = 0xC00;
const MAIN_HEADER_SIZE: usize = 0x400;
const FS_HEADER_SIZE: usize = 0x200;
const MEDIA_UNIT_SIZE: u64 = 0x200;
const MAX_SECTION_COUNT: usize = 4;
const MAX_INTEGRITY_BLOCK_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
pub struct NcaLoader;

impl NcaLoader {
    /// Loads an NCA using key material owned by the caller.
    pub fn load_with_key_provider(
        storage: StorageRef,
        keys: &dyn NcaKeyProvider,
    ) -> Result<NcaArchive, LoadError> {
        parse_nca(storage, keys)
    }
}

impl FormatLoader for NcaLoader {
    type Output = NcaArchive;

    const FORMAT_NAME: &'static str = "NCA";

    /// Loads a fully decrypted NCA. Encrypted archives should use
    /// load_with_key_provider.
    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        parse_nca(storage, &EmptyKeyProvider)
    }
}

struct EmptyKeyProvider;

impl NcaKeyProvider for EmptyKeyProvider {
    fn header_key(&self) -> Option<[u8; 32]> {
        None
    }

    fn key_area_key(&self, _generation: u8, _index: KeyAreaKeyIndex) -> Option<[u8; 16]> {
        None
    }

    fn title_key(&self, _rights_id: &[u8; 16], _generation: u8) -> Option<[u8; 16]> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaFormatVersion {
    Nca2,
    Nca3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaDistributionType {
    Download,
    GameCard,
    Unknown(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaContentType {
    Program,
    Meta,
    Control,
    Manual,
    Data,
    PublicData,
    Unknown(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaSectionType {
    Pfs0,
    RomFs,
    Bktr,
    Unknown {
        partition_type: u8,
        file_system_type: u8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaEncryptionType {
    None,
    AesXts,
    AesCtr,
    AesCtrEx,
}

/// Parsed fields from the NCA main header.
#[derive(Clone, Debug)]
pub struct NcaHeader {
    version: NcaFormatVersion,
    distribution_type: NcaDistributionType,
    content_type: NcaContentType,
    size: u64,
    title_id: u64,
    sdk_version: u32,
    key_generation: u8,
    key_area_key_index: u8,
    rights_id: Option<[u8; 16]>,
    source_is_decrypted: bool,
}

impl NcaHeader {
    pub const fn version(&self) -> NcaFormatVersion {
        self.version
    }

    pub const fn distribution_type(&self) -> NcaDistributionType {
        self.distribution_type
    }

    pub const fn content_type(&self) -> NcaContentType {
        self.content_type
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub const fn title_id(&self) -> u64 {
        self.title_id
    }

    pub const fn sdk_version(&self) -> u32 {
        self.sdk_version
    }

    pub const fn key_generation(&self) -> u8 {
        self.key_generation
    }

    pub const fn key_area_key_index(&self) -> u8 {
        self.key_area_key_index
    }

    pub const fn rights_id(&self) -> Option<&[u8; 16]> {
        self.rights_id.as_ref()
    }

    pub const fn source_is_decrypted(&self) -> bool {
        self.source_is_decrypted
    }
}

/// Parsed NCA retaining bounded, random-access views of all present sections.
pub struct NcaArchive {
    header: NcaHeader,
    sections: Vec<NcaSection>,
}

impl NcaArchive {
    pub const fn header(&self) -> &NcaHeader {
        &self.header
    }

    pub fn sections(&self) -> &[NcaSection] {
        &self.sections
    }

    pub fn section(&self, index: u8) -> Option<&NcaSection> {
        self.sections.iter().find(|section| section.index == index)
    }

    /// Streams every advertised integrity hierarchy without loading a complete
    /// section into memory.
    pub fn validate_integrity(&self) -> Result<Vec<IntegrityReport>, LoadError> {
        self.sections
            .iter()
            .map(NcaSection::validate_integrity)
            .collect()
    }
}

impl std::fmt::Debug for NcaArchive {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NcaArchive")
            .field("header", &self.header)
            .field("sections", &self.sections)
            .finish()
    }
}

/// One physical section discovered from the NCA media-unit table.
pub struct NcaSection {
    index: u8,
    offset: u64,
    size: u64,
    section_type: NcaSectionType,
    encryption_type: NcaEncryptionType,
    fs_header_hash_valid: bool,
    storage: StorageRef,
    integrity: IntegrityLayout,
}

impl NcaSection {
    pub const fn index(&self) -> u8 {
        self.index
    }

    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub const fn section_type(&self) -> NcaSectionType {
        self.section_type
    }

    pub const fn encryption_type(&self) -> NcaEncryptionType {
        self.encryption_type
    }

    pub const fn fs_header_hash_valid(&self) -> bool {
        self.fs_header_hash_valid
    }

    /// Returns a bounded plaintext view. For CTR-Ex/BKTR this is the physical
    /// initial-counter view; composing its virtual RomFS remains a separate
    /// operation requiring the base NCA.
    pub fn storage(&self) -> StorageRef {
        self.storage.clone()
    }

    /// Returns the bounded data payload, excluding advertised integrity/hash
    /// layers. For PFS0 sections this begins at the PFS0 magic rather than at
    /// physical section offset zero.
    pub fn payload_storage(&self) -> Result<StorageRef, LoadError> {
        let (offset, size) = match &self.integrity {
            IntegrityLayout::HierarchicalSha256(layout) => (layout.data_offset, layout.data_size),
            IntegrityLayout::Ivfc(layout) => layout
                .levels
                .last()
                .map(|level| (level.offset, level.size))
                .ok_or_else(|| LoadError::invalid("NCA", "IVFC section has no data level"))?,
            IntegrityLayout::None | IntegrityLayout::Bktr => (0, self.size),
        };
        Ok(Arc::new(SubStorage::new(
            self.storage.clone(),
            offset,
            size,
        )?))
    }

    pub fn validate_integrity(&self) -> Result<IntegrityReport, LoadError> {
        integrity::validate(
            self.index,
            self.fs_header_hash_valid,
            &self.storage,
            &self.integrity,
        )
    }
}

impl std::fmt::Debug for NcaSection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NcaSection")
            .field("index", &self.index)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("section_type", &self.section_type)
            .field("encryption_type", &self.encryption_type)
            .field("fs_header_hash_valid", &self.fs_header_hash_valid)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct SectionEntry {
    index: usize,
    offset: u64,
    size: u64,
}

enum ContentKeys {
    Rights([u8; 16]),
    KeyArea([[u8; 16]; 4]),
}

fn parse_nca(storage: StorageRef, keys: &dyn NcaKeyProvider) -> Result<NcaArchive, LoadError> {
    let storage_len = storage.len()?;
    if storage_len < HEADER_SIZE as u64 {
        return Err(LoadError::invalid("NCA", "header is truncated"));
    }

    let mut raw_header = [0_u8; HEADER_SIZE];
    storage.read_at(0, &mut raw_header)?;
    let (header_bytes, version, source_is_decrypted) = decrypt_header(raw_header, keys)?;

    let declared_size = read_u64(&header_bytes, 0x208);
    if declared_size != storage_len {
        return Err(LoadError::invalid(
            "NCA",
            format!(
                "declared archive size {declared_size:#x} does not match source size {storage_len:#x}"
            ),
        ));
    }

    let crypto_type = header_bytes[0x206];
    let crypto_type_2 = header_bytes[0x220];
    let key_generation = crypto_type.max(crypto_type_2).saturating_sub(1);
    let rights_id_raw: [u8; 16] = header_bytes[0x230..0x240]
        .try_into()
        .expect("fixed NCA header range");
    let rights_id = rights_id_raw
        .iter()
        .any(|byte| *byte != 0)
        .then_some(rights_id_raw);

    let header = NcaHeader {
        version,
        distribution_type: match header_bytes[0x204] {
            0 => NcaDistributionType::Download,
            1 => NcaDistributionType::GameCard,
            value => NcaDistributionType::Unknown(value),
        },
        content_type: match header_bytes[0x205] {
            0 => NcaContentType::Program,
            1 => NcaContentType::Meta,
            2 => NcaContentType::Control,
            3 => NcaContentType::Manual,
            4 => NcaContentType::Data,
            5 => NcaContentType::PublicData,
            value => NcaContentType::Unknown(value),
        },
        size: declared_size,
        title_id: read_u64(&header_bytes, 0x210),
        sdk_version: read_u32(&header_bytes, 0x21C),
        key_generation,
        key_area_key_index: header_bytes[0x207],
        rights_id,
        source_is_decrypted,
    };

    let entries = parse_section_entries(&header_bytes, storage_len)?;
    let needs_content_key = !source_is_decrypted
        && entries
            .iter()
            .any(|entry| header_bytes[0x400 + entry.index * FS_HEADER_SIZE + 4] != 1);
    let content_keys = needs_content_key
        .then(|| resolve_content_keys(&header_bytes, &header, keys))
        .transpose()?;

    let mut sections = Vec::with_capacity(entries.len());
    for entry in entries {
        let fs_start = 0x400 + entry.index * FS_HEADER_SIZE;
        let fs_header: &[u8; FS_HEADER_SIZE] = (&header_bytes[fs_start..fs_start + FS_HEADER_SIZE])
            .try_into()
            .expect("fixed FS header range");
        let expected_hash: [u8; 32] = header_bytes
            [0x280 + entry.index * 0x20..0x2A0 + entry.index * 0x20]
            .try_into()
            .expect("fixed FS header hash range");
        let actual_hash: [u8; 32] = Sha256::digest(fs_header).into();
        let fs_header_hash_valid = actual_hash == expected_hash;

        let partition_type = fs_header[2];
        let file_system_type = fs_header[3];
        let encryption_type = parse_encryption_type(fs_header[4])?;
        let section_type = if partition_type == 1 && file_system_type == 2 {
            NcaSectionType::Pfs0
        } else if partition_type == 0 && file_system_type == 3 {
            if encryption_type == NcaEncryptionType::AesCtrEx {
                NcaSectionType::Bktr
            } else {
                NcaSectionType::RomFs
            }
        } else {
            NcaSectionType::Unknown {
                partition_type,
                file_system_type,
            }
        };

        let raw_storage: StorageRef =
            Arc::new(SubStorage::new(storage.clone(), entry.offset, entry.size)?);
        let section_counter: [u8; 8] = fs_header[0x140..0x148]
            .try_into()
            .expect("fixed section counter range");
        let plaintext_storage = open_plaintext_section(
            raw_storage,
            entry.offset,
            encryption_type,
            section_counter,
            source_is_decrypted,
            content_keys.as_ref(),
        )?;
        let integrity = parse_integrity_layout(section_type, fs_header, entry.size)?;

        sections.push(NcaSection {
            index: u8::try_from(entry.index).expect("NCA has four section slots"),
            offset: entry.offset,
            size: entry.size,
            section_type,
            encryption_type,
            fs_header_hash_valid,
            storage: plaintext_storage,
            integrity,
        });
    }

    Ok(NcaArchive { header, sections })
}

fn decrypt_header(
    raw_header: [u8; HEADER_SIZE],
    keys: &dyn NcaKeyProvider,
) -> Result<([u8; HEADER_SIZE], NcaFormatVersion, bool), LoadError> {
    if let Some(version) = parse_magic(&raw_header[0x200..0x204]) {
        return Ok((raw_header, version, true));
    }

    let header_key = keys.header_key().ok_or_else(|| LoadError::MissingKey {
        key: "header_key".to_owned(),
    })?;
    let mut first = raw_header[..MAIN_HEADER_SIZE].to_vec();
    decrypt_xts(&header_key, &mut first, 0, MEDIA_UNIT_SIZE as usize);
    let version = parse_magic(&first[0x200..0x204]).ok_or_else(|| {
        LoadError::invalid("NCA", "header magic is invalid after AES-XTS decryption")
    })?;

    let mut decrypted = raw_header;
    match version {
        NcaFormatVersion::Nca3 => {
            decrypt_xts(&header_key, &mut decrypted, 0, MEDIA_UNIT_SIZE as usize);
        }
        NcaFormatVersion::Nca2 => {
            decrypted[..MAIN_HEADER_SIZE].copy_from_slice(&first);
            for index in 0..MAX_SECTION_COUNT {
                let start = 0x400 + index * FS_HEADER_SIZE;
                decrypt_xts(
                    &header_key,
                    &mut decrypted[start..start + FS_HEADER_SIZE],
                    0,
                    MEDIA_UNIT_SIZE as usize,
                );
            }
        }
    }
    Ok((decrypted, version, false))
}

fn parse_magic(magic: &[u8]) -> Option<NcaFormatVersion> {
    match magic {
        b"NCA2" => Some(NcaFormatVersion::Nca2),
        b"NCA3" => Some(NcaFormatVersion::Nca3),
        _ => None,
    }
}

fn parse_section_entries(
    header: &[u8; HEADER_SIZE],
    storage_len: u64,
) -> Result<Vec<SectionEntry>, LoadError> {
    let mut entries = Vec::new();
    for index in 0..MAX_SECTION_COUNT {
        let offset = 0x240 + index * 0x10;
        let media_start = u64::from(read_u32(header, offset));
        let media_end = u64::from(read_u32(header, offset + 4));
        if media_start == 0 && media_end == 0 {
            continue;
        }
        if media_start == 0 || media_end <= media_start {
            return Err(LoadError::invalid(
                "NCA",
                format!("section {index} has an invalid media-unit range"),
            ));
        }

        let start = media_start
            .checked_mul(MEDIA_UNIT_SIZE)
            .ok_or_else(|| LoadError::invalid("NCA", "section offset overflows"))?;
        let end = media_end
            .checked_mul(MEDIA_UNIT_SIZE)
            .ok_or_else(|| LoadError::invalid("NCA", "section end overflows"))?;
        if start < HEADER_SIZE as u64 || end > storage_len {
            return Err(LoadError::invalid(
                "NCA",
                format!("section {index} points outside the archive"),
            ));
        }
        entries.push(SectionEntry {
            index,
            offset: start,
            size: end - start,
        });
    }

    let mut by_offset = entries.clone();
    by_offset.sort_by_key(|entry| entry.offset);
    for pair in by_offset.windows(2) {
        if pair[0].offset + pair[0].size > pair[1].offset {
            return Err(LoadError::invalid("NCA", "section ranges overlap"));
        }
    }
    Ok(entries)
}

fn parse_encryption_type(value: u8) -> Result<NcaEncryptionType, LoadError> {
    match value {
        1 => Ok(NcaEncryptionType::None),
        2 => Ok(NcaEncryptionType::AesXts),
        3 => Ok(NcaEncryptionType::AesCtr),
        4 => Ok(NcaEncryptionType::AesCtrEx),
        _ => Err(LoadError::invalid(
            "NCA",
            format!("unsupported section encryption type {value}"),
        )),
    }
}

fn resolve_content_keys(
    header_bytes: &[u8; HEADER_SIZE],
    header: &NcaHeader,
    keys: &dyn NcaKeyProvider,
) -> Result<ContentKeys, LoadError> {
    if let Some(rights_id) = header.rights_id {
        let title_key = keys
            .title_key(&rights_id, header.key_generation)
            .ok_or_else(|| LoadError::MissingKey {
                key: format!("title key for rights ID {}", hex(&rights_id)),
            })?;
        return Ok(ContentKeys::Rights(title_key));
    }

    let index = KeyAreaKeyIndex::from_raw(header.key_area_key_index).ok_or_else(|| {
        LoadError::invalid(
            "NCA",
            format!(
                "unsupported key-area encryption-key index {}",
                header.key_area_key_index
            ),
        )
    })?;
    let key_area_key = keys
        .key_area_key(header.key_generation, index)
        .ok_or_else(|| LoadError::MissingKey {
            key: format!("{}_{:02x}", index.key_name(), header.key_generation),
        })?;
    let mut decrypted = header_bytes[0x300..0x340].to_vec();
    decrypt_ecb_blocks(&key_area_key, &mut decrypted);
    let mut slots = [[0_u8; 16]; 4];
    for (slot, bytes) in slots.iter_mut().zip(decrypted.chunks_exact(16)) {
        slot.copy_from_slice(bytes);
    }
    Ok(ContentKeys::KeyArea(slots))
}

fn open_plaintext_section(
    raw_storage: StorageRef,
    absolute_offset: u64,
    encryption_type: NcaEncryptionType,
    section_counter: [u8; 8],
    source_is_decrypted: bool,
    content_keys: Option<&ContentKeys>,
) -> Result<StorageRef, LoadError> {
    if source_is_decrypted || encryption_type == NcaEncryptionType::None {
        return Ok(raw_storage);
    }

    let content_keys = content_keys.expect("encrypted sections resolved content keys");
    match encryption_type {
        NcaEncryptionType::None => Ok(raw_storage),
        NcaEncryptionType::AesXts => {
            let ContentKeys::KeyArea(slots) = content_keys else {
                return Err(LoadError::invalid(
                    "NCA",
                    "rights-ID sections cannot use AES-XTS",
                ));
            };
            let mut key = [0_u8; 32];
            key[..16].copy_from_slice(&slots[0]);
            key[16..].copy_from_slice(&slots[1]);
            Ok(xts_storage(raw_storage, key))
        }
        NcaEncryptionType::AesCtr | NcaEncryptionType::AesCtrEx => {
            let key = match content_keys {
                ContentKeys::Rights(key) => *key,
                ContentKeys::KeyArea(slots) => slots[2],
            };
            Ok(ctr_storage(
                raw_storage,
                key,
                section_counter,
                absolute_offset,
            ))
        }
    }
}

fn parse_integrity_layout(
    section_type: NcaSectionType,
    fs_header: &[u8; FS_HEADER_SIZE],
    section_size: u64,
) -> Result<IntegrityLayout, LoadError> {
    match section_type {
        NcaSectionType::Pfs0 => {
            parse_sha256_layout(fs_header, section_size).map(IntegrityLayout::HierarchicalSha256)
        }
        NcaSectionType::RomFs => {
            parse_ivfc_layout(fs_header, section_size).map(IntegrityLayout::Ivfc)
        }
        NcaSectionType::Bktr => Ok(IntegrityLayout::Bktr),
        NcaSectionType::Unknown { .. } => Ok(IntegrityLayout::None),
    }
}

fn parse_sha256_layout(
    fs_header: &[u8; FS_HEADER_SIZE],
    section_size: u64,
) -> Result<Sha256Layout, LoadError> {
    let master_hash = fs_header[0x08..0x28]
        .try_into()
        .expect("fixed SHA-256 superblock range");
    let block_size = u64::from(read_u32(fs_header, 0x28));
    let hash_table_offset = read_u64(fs_header, 0x30);
    let hash_table_size = read_u64(fs_header, 0x38);
    let data_offset = read_u64(fs_header, 0x40);
    let data_size = read_u64(fs_header, 0x48);

    validate_block_size(block_size)?;
    validate_subrange(
        hash_table_offset,
        hash_table_size,
        section_size,
        "hash table",
    )?;
    validate_subrange(data_offset, data_size, section_size, "PFS0 data")?;
    let required_hashes = data_size
        .div_ceil(block_size)
        .checked_mul(0x20)
        .ok_or_else(|| LoadError::invalid("NCA", "PFS0 hash table size overflows"))?;
    if required_hashes > hash_table_size {
        return Err(LoadError::invalid(
            "NCA",
            "PFS0 hash table is too small for its data",
        ));
    }

    Ok(Sha256Layout {
        master_hash,
        hash_table_offset,
        hash_table_size,
        data_offset,
        data_size,
        block_size,
    })
}

fn parse_ivfc_layout(
    fs_header: &[u8; FS_HEADER_SIZE],
    section_size: u64,
) -> Result<IvfcLayout, LoadError> {
    if &fs_header[0x08..0x0C] != b"IVFC" {
        return Err(LoadError::invalid("NCA", "RomFS IVFC magic is invalid"));
    }
    let master_hash_size = usize::try_from(read_u32(fs_header, 0x10))
        .map_err(|_| LoadError::invalid("NCA", "IVFC master hash size is invalid"))?;
    let level_count_with_master = usize::try_from(read_u32(fs_header, 0x14))
        .map_err(|_| LoadError::invalid("NCA", "IVFC level count is invalid"))?;
    if !(2..=7).contains(&level_count_with_master) {
        return Err(LoadError::invalid(
            "NCA",
            format!("unsupported IVFC level count {level_count_with_master}"),
        ));
    }
    let level_count = level_count_with_master - 1;
    if master_hash_size == 0 || master_hash_size > 0x20 || master_hash_size % 0x20 != 0 {
        return Err(LoadError::invalid(
            "NCA",
            "IVFC master hash table has an invalid size",
        ));
    }

    let mut levels = Vec::with_capacity(level_count);
    for index in 0..level_count {
        let start = 0x18 + index * 0x18;
        let offset = read_u64(fs_header, start);
        let size = read_u64(fs_header, start + 8);
        let exponent = read_u32(fs_header, start + 0x10);
        let block_size = 1_u64
            .checked_shl(exponent)
            .ok_or_else(|| LoadError::invalid("NCA", "IVFC block size exponent is invalid"))?;
        validate_block_size(block_size)?;
        validate_subrange(offset, size, section_size, "IVFC level")?;
        levels.push(IvfcLevel {
            offset,
            size,
            block_size,
        });
    }

    let master_hash_count = master_hash_size / 0x20;
    let level_zero_hash_count = usize::try_from(levels[0].size.div_ceil(levels[0].block_size))
        .map_err(|_| LoadError::invalid("NCA", "IVFC master hash count is too large"))?;
    if level_zero_hash_count > master_hash_count {
        return Err(LoadError::invalid(
            "NCA",
            "IVFC master hash table is too small for level zero",
        ));
    }
    for index in 1..levels.len() {
        let required = levels[index]
            .size
            .div_ceil(levels[index].block_size)
            .checked_mul(0x20)
            .ok_or_else(|| LoadError::invalid("NCA", "IVFC hash count overflows"))?;
        if required > levels[index - 1].size {
            return Err(LoadError::invalid(
                "NCA",
                format!("IVFC level {} hash table is truncated", index - 1),
            ));
        }
    }

    let mut master_hashes = Vec::with_capacity(master_hash_count);
    for bytes in fs_header[0xC8..0xC8 + master_hash_size].chunks_exact(0x20) {
        master_hashes.push(bytes.try_into().expect("32-byte master hash"));
    }

    Ok(IvfcLayout {
        master_hashes,
        levels,
    })
}

fn validate_block_size(block_size: u64) -> Result<(), LoadError> {
    if !(0x20..=MAX_INTEGRITY_BLOCK_SIZE).contains(&block_size) || !block_size.is_power_of_two() {
        return Err(LoadError::invalid(
            "NCA",
            format!("invalid integrity block size {block_size:#x}"),
        ));
    }
    Ok(())
}

fn validate_subrange(
    offset: u64,
    size: u64,
    container_size: u64,
    name: &str,
) -> Result<(), LoadError> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| LoadError::invalid("NCA", format!("{name} range overflows")))?;
    if end > container_size {
        return Err(LoadError::invalid(
            "NCA",
            format!("{name} range points outside its section"),
        ));
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated NCA metadata range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated NCA metadata range"),
    )
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 0x0F)]));
    }
    result
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use swiitx_loader_storage::{Storage, StorageError};

    use super::*;

    #[derive(Debug)]
    struct VecStorage(Vec<u8>);

    impl Storage for VecStorage {
        fn len(&self) -> Result<u64, StorageError> {
            u64::try_from(self.0.len()).map_err(|_| StorageError::OutOfBounds)
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), StorageError> {
            let start = usize::try_from(offset).map_err(|_| StorageError::OutOfBounds)?;
            let end = start
                .checked_add(buffer.len())
                .ok_or(StorageError::OutOfBounds)?;
            let source = self.0.get(start..end).ok_or(StorageError::OutOfBounds)?;
            buffer.copy_from_slice(source);
            Ok(())
        }
    }

    fn synthetic_pfs0_nca() -> Vec<u8> {
        const SECTION_OFFSET: usize = 0xC00;
        const SECTION_SIZE: usize = 0x400;
        const DATA_OFFSET: usize = 0x200;
        const DATA_SIZE: usize = 0x100;
        const BLOCK_SIZE: usize = 0x100;

        let mut nca = vec![0_u8; SECTION_OFFSET + SECTION_SIZE];
        nca[0x200..0x204].copy_from_slice(b"NCA3");
        nca[0x204] = 0;
        nca[0x205] = 1;
        nca[0x206] = 1;
        nca[0x207] = 0;
        let nca_size = nca.len() as u64;
        put_u64(&mut nca, 0x208, nca_size);
        put_u64(&mut nca, 0x210, 0x0100_0000_0000_1000);
        put_u32(&mut nca, 0x21C, 0x0012_0304);
        put_u32(
            &mut nca,
            0x240,
            (SECTION_OFFSET as u64 / MEDIA_UNIT_SIZE) as u32,
        );
        put_u32(
            &mut nca,
            0x244,
            ((SECTION_OFFSET + SECTION_SIZE) as u64 / MEDIA_UNIT_SIZE) as u32,
        );

        let data_start = SECTION_OFFSET + DATA_OFFSET;
        for (index, byte) in nca[data_start..data_start + DATA_SIZE]
            .iter_mut()
            .enumerate()
        {
            *byte = u8::try_from(index).unwrap();
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
        put_u32(fs, 0x28, BLOCK_SIZE as u32);
        put_u32(fs, 0x2C, 2);
        put_u64(fs, 0x30, 0);
        put_u64(fs, 0x38, 0x20);
        put_u64(fs, 0x40, DATA_OFFSET as u64);
        put_u64(fs, 0x48, DATA_SIZE as u64);

        let fs_hash: [u8; 32] = Sha256::digest(&nca[0x400..0x600]).into();
        nca[0x280..0x2A0].copy_from_slice(&fs_hash);
        nca
    }

    fn load_bytes(bytes: Vec<u8>) -> Result<NcaArchive, LoadError> {
        NcaLoader::load(Arc::new(VecStorage(bytes)))
    }

    #[test]
    fn parses_sections_and_validates_hierarchical_sha256() {
        let archive = load_bytes(synthetic_pfs0_nca()).unwrap();

        assert_eq!(archive.header().version(), NcaFormatVersion::Nca3);
        assert_eq!(archive.header().content_type(), NcaContentType::Meta);
        assert!(archive.header().source_is_decrypted());
        assert_eq!(archive.sections().len(), 1);

        let section = &archive.sections()[0];
        assert_eq!(section.index(), 0);
        assert_eq!(section.section_type(), NcaSectionType::Pfs0);
        assert_eq!(section.encryption_type(), NcaEncryptionType::None);
        assert!(section.fs_header_hash_valid());

        let report = section.validate_integrity().unwrap();
        assert!(report.is_valid());
        assert_eq!(report.checks().len(), 3);

        let payload = section.payload_storage().unwrap();
        assert_eq!(payload.len().unwrap(), 0x100);
        let mut first = [0_u8; 2];
        payload.read_at(0, &mut first).unwrap();
        assert_eq!(first, [0, 1]);
    }

    #[test]
    fn reports_corrupted_section_data() {
        let mut bytes = synthetic_pfs0_nca();
        bytes[0xE40] ^= 0x80;
        let archive = load_bytes(bytes).unwrap();

        let report = archive.sections()[0].validate_integrity().unwrap();
        assert!(!report.is_valid());
        assert_eq!(
            report.checks().last().unwrap().status,
            crate::IntegrityStatus::Invalid
        );
    }

    #[test]
    fn reports_corrupted_fs_header_hash_without_trusting_padding() {
        let mut bytes = synthetic_pfs0_nca();
        bytes[0x5FF] ^= 1;
        let archive = load_bytes(bytes).unwrap();

        assert!(!archive.sections()[0].fs_header_hash_valid());
        assert!(
            !archive.sections()[0]
                .validate_integrity()
                .unwrap()
                .is_valid()
        );
    }

    #[test]
    fn rejects_section_outside_archive() {
        let mut bytes = synthetic_pfs0_nca();
        put_u32(&mut bytes, 0x244, u32::MAX);

        assert!(matches!(
            load_bytes(bytes),
            Err(LoadError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn encrypted_header_requires_a_caller_key() {
        let bytes = vec![0xA5; HEADER_SIZE];
        assert!(matches!(
            load_bytes(bytes),
            Err(LoadError::MissingKey { .. })
        ));
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}
