use std::collections::HashSet;
use std::sync::Arc;

use swiitx_loader_storage::FormatLoader;
use swiitx_loader_storage::{LoadError, StorageRef, SubStorage};

const HEADER_SIZE: u64 = 0x10;
const ENTRY_SIZE: u64 = 0x18;
const MAX_FILE_COUNT: u64 = 65_536;
const MAX_METADATA_SIZE: u64 = 64 * 1024 * 1024;

/// Metadata describing one file stored in a PFS0 archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pfs0Entry {
    name: String,
    offset: u64,
    size: u64,
}

impl Pfs0Entry {
    /// Returns the file name stored in the PFS0 string table.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the absolute byte offset of the file in the source storage.
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the file size in bytes.
    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// Parsed Partition File System 0 archive.
pub struct Pfs0Archive {
    storage: StorageRef,
    entries: Vec<Pfs0Entry>,
    data_offset: u64,
}

impl Pfs0Archive {
    pub(crate) fn parse(storage: StorageRef, format: &'static str) -> Result<Self, LoadError> {
        Self::parse_internal(storage, format, false)
    }

    pub(crate) fn parse_nsz(storage: StorageRef, format: &'static str) -> Result<Self, LoadError> {
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
        if &header[..4] != b"PFS0" {
            return Err(LoadError::invalid(format, "expected PFS0 magic"));
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
        let string_table_end = usize::try_from(metadata_size)
            .map_err(|_| LoadError::invalid(format, "string table size is invalid"))?;
        let string_table = &metadata[string_table_start..string_table_end];

        let entry_capacity = usize::try_from(file_count)
            .map_err(|_| LoadError::invalid(format, "file count does not fit in memory"))?;
        let mut entries = Vec::with_capacity(entry_capacity);
        let mut names = HashSet::with_capacity(entry_capacity);

        for index in 0..file_count {
            let entry_start_u64 =
                HEADER_SIZE
                    .checked_add(index.checked_mul(ENTRY_SIZE).ok_or_else(|| {
                        LoadError::invalid(format, "entry table offset overflows")
                    })?)
                    .ok_or_else(|| LoadError::invalid(format, "entry table offset overflows"))?;
            let entry_start = usize::try_from(entry_start_u64)
                .map_err(|_| LoadError::invalid(format, "entry offset is invalid"))?;

            let relative_offset = read_u64(&metadata, entry_start);
            let size = read_u64(&metadata, entry_start + 8);
            let name_offset = usize::try_from(read_u32(&metadata, entry_start + 16))
                .map_err(|_| LoadError::invalid(format, "file name offset is invalid"))?;

            let name_bytes = string_table
                .get(name_offset..)
                .ok_or_else(|| LoadError::invalid(format, "file name is outside string table"))?;
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

            entries.push(Pfs0Entry { name, offset, size });
        }

        Ok(Self {
            storage,
            entries,
            data_offset: metadata_size,
        })
    }

    pub fn entries(&self) -> &[Pfs0Entry] {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&Pfs0Entry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn open_entry(&self, entry: &Pfs0Entry) -> Result<StorageRef, LoadError> {
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
}

/// Loads a generic Partition File System 0 archive.
#[derive(Debug)]
pub struct Pfs0Loader;

impl FormatLoader for Pfs0Loader {
    type Output = Pfs0Archive;

    const FORMAT_NAME: &'static str = "PFS0";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        Pfs0Archive::parse(storage, Self::FORMAT_NAME)
    }
}

impl std::fmt::Debug for Pfs0Archive {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Pfs0Archive")
            .field("entries", &self.entries)
            .field("data_offset", &self.data_offset)
            .finish_non_exhaustive()
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated PFS0 metadata range"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated PFS0 metadata range"),
    )
}
