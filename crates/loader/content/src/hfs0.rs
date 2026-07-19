use std::collections::HashSet;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef, SubStorage};

const HEADER_SIZE: u64 = 0x10;
const ENTRY_SIZE: u64 = 0x40;
const MAX_FILE_COUNT: u64 = 65_536;
const MAX_METADATA_SIZE: u64 = 64 * 1024 * 1024;
const HASH_BUFFER_SIZE: usize = 1024 * 1024;

/// Metadata describing one file stored in an HFS0 archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hfs0Entry {
    name: String,
    offset: u64,
    size: u64,
    hashed_region_size: u64,
    expected_hash: [u8; 32],
    reserved: [u8; 8],
}

impl Hfs0Entry {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn size(&self) -> u64 {
        self.size
    }

    pub const fn hashed_region_size(&self) -> u64 {
        self.hashed_region_size
    }

    pub const fn expected_hash(&self) -> &[u8; 32] {
        &self.expected_hash
    }

    /// Whether this entry advertises a usable integrity digest.
    ///
    /// XCZ writers may leave this field all-zero when rebuilding HFS0 tables.
    pub fn has_advertised_hash(&self) -> bool {
        self.expected_hash != [0; 32]
    }

    pub const fn reserved(&self) -> &[u8; 8] {
        &self.reserved
    }
}

/// Result of validating one HFS0 entry's advertised hashed region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hfs0HashResult {
    pub name: String,
    pub hashed_region_size: u64,
    pub expected: [u8; 32],
    pub actual: [u8; 32],
}

impl Hfs0HashResult {
    pub fn is_valid(&self) -> bool {
        self.expected == self.actual
    }
}

/// Parsed Hash File System 0 archive.
pub struct Hfs0Archive {
    storage: StorageRef,
    entries: Vec<Hfs0Entry>,
    data_offset: u64,
}

impl Hfs0Archive {
    pub(crate) fn parse(storage: StorageRef, format: &'static str) -> Result<Self, LoadError> {
        Self::parse_internal(storage, format, false)
    }

    pub(crate) fn parse_xcz(storage: StorageRef, format: &'static str) -> Result<Self, LoadError> {
        Self::parse_internal(storage, format, true)
    }

    fn parse_internal(
        storage: StorageRef,
        format: &'static str,
        allow_implicit_final_terminator: bool,
    ) -> Result<Self, LoadError> {
        let storage_len = storage.len()?;
        if storage_len < HEADER_SIZE {
            return Err(LoadError::invalid(format, "header is truncated"));
        }

        let mut header = [0_u8; HEADER_SIZE as usize];
        storage.read_at(0, &mut header)?;
        if &header[..4] != b"HFS0" {
            return Err(LoadError::invalid(format, "expected HFS0 magic"));
        }
        if header[12..16] != [0; 4] {
            return Err(LoadError::invalid(
                format,
                "header reserved field is non-zero",
            ));
        }

        let file_count = u64::from(read_u32(&header, 4));
        let string_table_size = u64::from(read_u32(&header, 8));
        if file_count > MAX_FILE_COUNT {
            return Err(LoadError::invalid(
                format,
                "file count exceeds the 65536-entry safety limit",
            ));
        }
        let entry_table_size = file_count
            .checked_mul(ENTRY_SIZE)
            .ok_or_else(|| LoadError::invalid(format, "entry table size overflows"))?;
        let metadata_size = HEADER_SIZE
            .checked_add(entry_table_size)
            .and_then(|size| size.checked_add(string_table_size))
            .ok_or_else(|| LoadError::invalid(format, "metadata size overflows"))?;
        if metadata_size > MAX_METADATA_SIZE {
            return Err(LoadError::invalid(
                format,
                "metadata exceeds the 64 MiB safety limit",
            ));
        }
        if metadata_size > storage_len {
            return Err(LoadError::invalid(format, "metadata is truncated"));
        }

        let metadata_len = usize::try_from(metadata_size)
            .map_err(|_| LoadError::invalid(format, "metadata does not fit in memory"))?;
        let mut metadata = vec![0_u8; metadata_len];
        storage.read_at(0, &mut metadata)?;
        let string_table_start = usize::try_from(HEADER_SIZE + entry_table_size)
            .map_err(|_| LoadError::invalid(format, "string table offset is invalid"))?;
        let string_table = &metadata[string_table_start..metadata_len];

        let capacity = usize::try_from(file_count)
            .map_err(|_| LoadError::invalid(format, "file count does not fit in memory"))?;
        let mut entries = Vec::with_capacity(capacity);
        let mut names = HashSet::with_capacity(capacity);
        for index in 0..file_count {
            let entry_start = HEADER_SIZE
                .checked_add(
                    index
                        .checked_mul(ENTRY_SIZE)
                        .ok_or_else(|| LoadError::invalid(format, "entry offset overflows"))?,
                )
                .ok_or_else(|| LoadError::invalid(format, "entry offset overflows"))?;
            let entry_start = usize::try_from(entry_start)
                .map_err(|_| LoadError::invalid(format, "entry offset is invalid"))?;
            let relative_offset = read_u64(&metadata, entry_start);
            let size = read_u64(&metadata, entry_start + 8);
            let name_offset = usize::try_from(read_u32(&metadata, entry_start + 16))
                .map_err(|_| LoadError::invalid(format, "file name offset is invalid"))?;
            let hashed_region_size = u64::from(read_u32(&metadata, entry_start + 20));
            if hashed_region_size > size {
                return Err(LoadError::invalid(
                    format,
                    "hashed region is larger than its file entry",
                ));
            }
            let reserved = metadata[entry_start + 24..entry_start + 32]
                .try_into()
                .expect("validated HFS0 entry range");
            if reserved != [0; 8] {
                return Err(LoadError::invalid(
                    format,
                    "entry reserved field is non-zero",
                ));
            }
            let expected_hash = metadata[entry_start + 32..entry_start + 64]
                .try_into()
                .expect("validated HFS0 entry range");

            let name_bytes = string_table
                .get(name_offset..)
                .ok_or_else(|| LoadError::invalid(format, "file name is outside string table"))?;
            if name_offset != 0 && string_table.get(name_offset - 1) != Some(&0) {
                return Err(LoadError::invalid(
                    format,
                    "file name offset does not point to the start of a name",
                ));
            }
            let name_end = match name_bytes.iter().position(|byte| *byte == 0) {
                Some(end) => end,
                None if allow_implicit_final_terminator => {
                    let mut terminator = [0_u8; 1];
                    storage.read_at(metadata_size, &mut terminator)?;
                    if terminator[0] != 0 {
                        return Err(LoadError::invalid(
                            format,
                            "file name is not null-terminated",
                        ));
                    }
                    name_bytes.len()
                }
                None => {
                    return Err(LoadError::invalid(
                        format,
                        "file name is not null-terminated",
                    ));
                }
            };
            if name_end == 0 {
                return Err(LoadError::invalid(format, "file name is empty"));
            }
            let name = std::str::from_utf8(&name_bytes[..name_end])
                .map_err(|_| LoadError::invalid(format, "file name is not valid UTF-8"))?
                .to_owned();
            if !names.insert(name.clone()) {
                return Err(LoadError::invalid(format, "file names are duplicated"));
            }

            let offset = metadata_size
                .checked_add(relative_offset)
                .ok_or_else(|| LoadError::invalid(format, "file offset overflows"))?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| LoadError::invalid(format, "file range overflows"))?;
            if end > storage_len {
                return Err(LoadError::invalid(
                    format,
                    "file entry points outside the source",
                ));
            }
            entries.push(Hfs0Entry {
                name,
                offset,
                size,
                hashed_region_size,
                expected_hash,
                reserved,
            });
        }

        Ok(Self {
            storage,
            entries,
            data_offset: metadata_size,
        })
    }

    pub fn entries(&self) -> &[Hfs0Entry] {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&Hfs0Entry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn open_entry(&self, entry: &Hfs0Entry) -> Result<StorageRef, LoadError> {
        Ok(Arc::new(SubStorage::new(
            self.storage.clone(),
            entry.offset,
            entry.size,
        )?))
    }

    pub fn open(&self, name: &str) -> Result<Option<StorageRef>, LoadError> {
        self.entry(name)
            .map(|entry| self.open_entry(entry))
            .transpose()
    }

    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    pub fn validate_entry(&self, entry: &Hfs0Entry) -> Result<Hfs0HashResult, LoadError> {
        let storage = self.open_entry(entry)?;
        let actual = hash_prefix(&storage, entry.hashed_region_size)?;
        Ok(Hfs0HashResult {
            name: entry.name.clone(),
            hashed_region_size: entry.hashed_region_size,
            expected: entry.expected_hash,
            actual,
        })
    }

    pub fn validate_all(&self) -> Result<Vec<Hfs0HashResult>, LoadError> {
        self.entries
            .iter()
            .map(|entry| self.validate_entry(entry))
            .collect()
    }
}

/// Loads a generic Hash File System 0 archive.
#[derive(Debug)]
pub struct Hfs0Loader;

impl FormatLoader for Hfs0Loader {
    type Output = Hfs0Archive;

    const FORMAT_NAME: &'static str = "HFS0";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        Hfs0Archive::parse(storage, Self::FORMAT_NAME)
    }
}

impl std::fmt::Debug for Hfs0Archive {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Hfs0Archive")
            .field("entries", &self.entries)
            .field("data_offset", &self.data_offset)
            .finish_non_exhaustive()
    }
}

fn hash_prefix(storage: &StorageRef, size: u64) -> Result<[u8; 32], LoadError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_SIZE];
    let mut offset = 0_u64;
    while offset < size {
        let remaining = size - offset;
        let read_size = usize::try_from(remaining.min(HASH_BUFFER_SIZE as u64))
            .expect("hash buffer size fits usize");
        storage.read_at(offset, &mut buffer[..read_size])?;
        hasher.update(&buffer[..read_size]);
        offset += read_size as u64;
    }
    Ok(hasher.finalize().into())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated range"),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sha2::{Digest, Sha256};
    use swiitx_loader_storage::{Storage, StorageError};

    use super::*;

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

    fn build(files: &[(&str, &[u8], usize)]) -> Vec<u8> {
        let mut strings = Vec::new();
        let mut name_offsets = Vec::new();
        for (name, _, _) in files {
            name_offsets.push(strings.len() as u32);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut result = Vec::new();
        result.extend_from_slice(b"HFS0");
        result.extend_from_slice(&(files.len() as u32).to_le_bytes());
        result.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        result.extend_from_slice(&0_u32.to_le_bytes());
        let mut offset = 0_u64;
        for ((_, data, hashed_size), name_offset) in files.iter().zip(name_offsets) {
            result.extend_from_slice(&offset.to_le_bytes());
            result.extend_from_slice(&(data.len() as u64).to_le_bytes());
            result.extend_from_slice(&name_offset.to_le_bytes());
            result.extend_from_slice(&(*hashed_size as u32).to_le_bytes());
            result.extend_from_slice(&[0_u8; 8]);
            result.extend_from_slice(&Sha256::digest(&data[..*hashed_size]));
            offset += data.len() as u64;
        }
        result.extend_from_slice(&strings);
        for (_, data, _) in files {
            result.extend_from_slice(data);
        }
        result
    }

    fn load(bytes: Vec<u8>) -> Result<Hfs0Archive, LoadError> {
        Hfs0Loader::load(Arc::new(Bytes(bytes)))
    }

    #[test]
    fn lists_opens_and_validates_entries() {
        let archive = load(build(&[("a", b"abc", 3), ("empty", b"", 0)])).unwrap();
        assert_eq!(archive.entries()[0].name(), "a");
        assert_eq!(archive.entries()[0].size(), 3);
        assert!(
            archive
                .validate_all()
                .unwrap()
                .iter()
                .all(Hfs0HashResult::is_valid)
        );
        let storage = archive.open("a").unwrap().unwrap();
        let mut bytes = [0; 3];
        storage.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"abc");
        assert!(matches!(
            storage.read_at(2, &mut [0; 2]),
            Err(StorageError::OutOfBounds)
        ));
    }

    #[test]
    fn reports_hash_mismatch() {
        let mut bytes = build(&[("a", b"abc", 3)]);
        *bytes.last_mut().unwrap() ^= 1;
        assert!(!load(bytes).unwrap().validate_all().unwrap()[0].is_valid());
    }

    #[test]
    fn rejects_malformed_metadata() {
        assert!(load(b"HFS0".to_vec()).is_err());
        let mut bad_magic = build(&[]);
        bad_magic[..4].copy_from_slice(b"NOPE");
        assert!(load(bad_magic).is_err());
        let mut excessive = build(&[]);
        excessive[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(load(excessive).is_err());
        let mut reserved = build(&[]);
        reserved[12] = 1;
        assert!(load(reserved).is_err());
        let mut duplicate = build(&[("a", b"", 0), ("a", b"", 0)]);
        assert!(load(std::mem::take(&mut duplicate)).is_err());
    }

    #[test]
    fn rejects_invalid_names_and_ranges() {
        let mut unterminated = build(&[("a", b"", 0)]);
        let string_offset = 0x10 + 0x40;
        unterminated[string_offset + 1] = b'x';
        assert!(load(unterminated).is_err());

        let mut invalid_utf8 = build(&[("a", b"", 0)]);
        invalid_utf8[string_offset] = 0xff;
        assert!(load(invalid_utf8).is_err());

        let mut outside = build(&[("a", b"", 0)]);
        outside[0x10..0x18].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(load(outside).is_err());

        let mut oversized_hash = build(&[("a", b"a", 1)]);
        oversized_hash[0x24..0x28].copy_from_slice(&2_u32.to_le_bytes());
        assert!(load(oversized_hash).is_err());
    }
}
