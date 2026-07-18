use std::collections::BTreeSet;
use std::sync::Arc;

use swiitx_loader_storage::{FormatLoader, LoadError, StorageRef, SubStorage};

const HEADER_SIZE: u64 = 0x50;
const DIRECTORY_ENTRY_SIZE: u64 = 0x18;
const FILE_ENTRY_SIZE: u64 = 0x20;
const EMPTY_ENTRY: u32 = u32::MAX;
const MAX_NAME_SIZE: u64 = 0x1000;

/// Loads read-only Nintendo Switch RomFS images.
#[derive(Debug)]
pub struct RomFsLoader;

impl FormatLoader for RomFsLoader {
    type Output = RomFsArchive;

    const FORMAT_NAME: &'static str = "RomFS";

    fn load(storage: StorageRef) -> Result<Self::Output, LoadError> {
        RomFsArchive::parse(storage)
    }
}

/// Parsed, bounded view of a RomFS image.
pub struct RomFsArchive {
    storage: StorageRef,
    file_data_offset: u64,
    files: Vec<RomFsFile>,
}

impl RomFsArchive {
    fn parse(storage: StorageRef) -> Result<Self, LoadError> {
        let storage_len = storage.len()?;
        if storage_len < HEADER_SIZE {
            return Err(LoadError::invalid("RomFS", "header is truncated"));
        }

        let mut header = [0_u8; HEADER_SIZE as usize];
        storage.read_at(0, &mut header)?;
        if read_u64(&header, 0) != HEADER_SIZE {
            return Err(LoadError::invalid("RomFS", "header size is not 0x50"));
        }

        let directory_hash = Region::new(read_u64(&header, 0x08), read_u64(&header, 0x10));
        let directory_meta = Region::new(read_u64(&header, 0x18), read_u64(&header, 0x20));
        let file_hash = Region::new(read_u64(&header, 0x28), read_u64(&header, 0x30));
        let file_meta = Region::new(read_u64(&header, 0x38), read_u64(&header, 0x40));
        let file_data_offset = read_u64(&header, 0x48);

        for (name, region) in [
            ("directory hash table", directory_hash),
            ("directory metadata table", directory_meta),
            ("file hash table", file_hash),
            ("file metadata table", file_meta),
        ] {
            region.validate(storage_len, name)?;
        }
        if directory_meta.size < DIRECTORY_ENTRY_SIZE {
            return Err(LoadError::invalid(
                "RomFS",
                "directory metadata table does not contain a root entry",
            ));
        }
        if file_data_offset > storage_len {
            return Err(LoadError::invalid(
                "RomFS",
                "file data offset points outside the image",
            ));
        }

        let mut parser = TreeParser {
            storage: storage.clone(),
            storage_len,
            directory_meta,
            file_meta,
            file_data_offset,
            visited_directories: BTreeSet::new(),
            visited_files: BTreeSet::new(),
            paths: BTreeSet::new(),
            files: Vec::new(),
        };
        parser.parse_directory(0, None, "")?;
        parser
            .files
            .sort_by(|left, right| left.path.cmp(&right.path));

        Ok(Self {
            storage,
            file_data_offset,
            files: parser.files,
        })
    }

    /// Returns files in deterministic absolute-path order.
    pub fn files(&self) -> &[RomFsFile] {
        &self.files
    }

    /// Finds a file by an exact absolute path.
    pub fn file(&self, path: &str) -> Option<&RomFsFile> {
        self.files
            .binary_search_by(|entry| entry.path.as_str().cmp(path))
            .ok()
            .map(|index| &self.files[index])
    }

    /// Opens one file as an independent bounded storage view.
    pub fn open_file(&self, file: &RomFsFile) -> Result<StorageRef, LoadError> {
        let absolute_offset = self
            .file_data_offset
            .checked_add(file.offset)
            .ok_or_else(|| LoadError::invalid("RomFS", "file offset overflows"))?;
        Ok(Arc::new(SubStorage::new(
            self.storage.clone(),
            absolute_offset,
            file.size,
        )?))
    }

    /// Finds and opens a file by an exact absolute path.
    pub fn open(&self, path: &str) -> Result<Option<StorageRef>, LoadError> {
        self.file(path).map(|file| self.open_file(file)).transpose()
    }
}

impl std::fmt::Debug for RomFsArchive {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RomFsArchive")
            .field("file_data_offset", &self.file_data_offset)
            .field("files", &self.files)
            .finish_non_exhaustive()
    }
}

/// One regular file recorded in a RomFS image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RomFsFile {
    path: String,
    offset: u64,
    size: u64,
}

impl RomFsFile {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn size(&self) -> u64 {
        self.size
    }
}

#[derive(Clone, Copy)]
struct Region {
    offset: u64,
    size: u64,
}

impl Region {
    const fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    fn validate(self, container_size: u64, name: &str) -> Result<(), LoadError> {
        let end = self
            .offset
            .checked_add(self.size)
            .ok_or_else(|| LoadError::invalid("RomFS", format!("{name} range overflows")))?;
        if self.offset < HEADER_SIZE || end > container_size {
            return Err(LoadError::invalid(
                "RomFS",
                format!("{name} points outside the image"),
            ));
        }
        Ok(())
    }
}

struct TreeParser {
    storage: StorageRef,
    storage_len: u64,
    directory_meta: Region,
    file_meta: Region,
    file_data_offset: u64,
    visited_directories: BTreeSet<u32>,
    visited_files: BTreeSet<u32>,
    paths: BTreeSet<String>,
    files: Vec<RomFsFile>,
}

impl TreeParser {
    fn parse_directory(
        &mut self,
        offset: u32,
        expected_parent: Option<u32>,
        parent_path: &str,
    ) -> Result<(), LoadError> {
        if !self.visited_directories.insert(offset) {
            return Err(LoadError::invalid(
                "RomFS",
                "directory metadata contains a cycle or duplicate reference",
            ));
        }
        let entry = self.read_directory(offset)?;
        if let Some(expected_parent) = expected_parent
            && entry.parent != expected_parent
        {
            return Err(LoadError::invalid(
                "RomFS",
                "directory has an inconsistent parent reference",
            ));
        }

        let path = if offset == 0 {
            if !entry.name.is_empty() {
                return Err(LoadError::invalid(
                    "RomFS",
                    "root directory has a non-empty name",
                ));
            }
            String::new()
        } else {
            join_path(parent_path, &entry.name)
        };

        self.parse_file_chain(entry.child_file, offset, &path)?;
        self.parse_directory_chain(entry.child_directory, offset, &path)
    }

    fn parse_directory_chain(
        &mut self,
        mut offset: u32,
        parent: u32,
        parent_path: &str,
    ) -> Result<(), LoadError> {
        let mut chain = BTreeSet::new();
        while offset != EMPTY_ENTRY {
            if !chain.insert(offset) {
                return Err(LoadError::invalid(
                    "RomFS",
                    "directory sibling chain contains a cycle",
                ));
            }
            let sibling = self.read_directory(offset)?.sibling;
            self.parse_directory(offset, Some(parent), parent_path)?;
            offset = sibling;
        }
        Ok(())
    }

    fn parse_file_chain(
        &mut self,
        mut offset: u32,
        parent: u32,
        parent_path: &str,
    ) -> Result<(), LoadError> {
        let mut chain = BTreeSet::new();
        while offset != EMPTY_ENTRY {
            if !chain.insert(offset) || !self.visited_files.insert(offset) {
                return Err(LoadError::invalid(
                    "RomFS",
                    "file metadata contains a cycle or duplicate reference",
                ));
            }
            let entry = self.read_file(offset)?;
            if entry.parent != parent {
                return Err(LoadError::invalid(
                    "RomFS",
                    "file has an inconsistent parent reference",
                ));
            }
            let end = self
                .file_data_offset
                .checked_add(entry.offset)
                .and_then(|start| start.checked_add(entry.size))
                .ok_or_else(|| LoadError::invalid("RomFS", "file data range overflows"))?;
            if end > self.storage_len {
                return Err(LoadError::invalid(
                    "RomFS",
                    "file data points outside the image",
                ));
            }
            let path = join_path(parent_path, &entry.name);
            if !self.paths.insert(path.clone()) {
                return Err(LoadError::invalid("RomFS", "duplicate file path"));
            }
            self.files.push(RomFsFile {
                path,
                offset: entry.offset,
                size: entry.size,
            });
            offset = entry.sibling;
        }
        Ok(())
    }

    fn read_directory(&self, relative_offset: u32) -> Result<DirectoryEntry, LoadError> {
        let offset = self.metadata_offset(
            self.directory_meta,
            relative_offset,
            DIRECTORY_ENTRY_SIZE,
            "directory",
        )?;
        let mut fixed = [0_u8; DIRECTORY_ENTRY_SIZE as usize];
        self.storage.read_at(offset, &mut fixed)?;
        let name = self.read_name(
            offset + DIRECTORY_ENTRY_SIZE,
            u64::from(read_u32(&fixed, 0x14)),
            self.directory_meta,
            "directory",
        )?;
        if relative_offset != 0 && name.is_empty() {
            return Err(LoadError::invalid("RomFS", "directory name is empty"));
        }
        Ok(DirectoryEntry {
            parent: read_u32(&fixed, 0),
            sibling: read_u32(&fixed, 4),
            child_directory: read_u32(&fixed, 8),
            child_file: read_u32(&fixed, 0x0C),
            name,
        })
    }

    fn read_file(&self, relative_offset: u32) -> Result<FileEntry, LoadError> {
        let offset =
            self.metadata_offset(self.file_meta, relative_offset, FILE_ENTRY_SIZE, "file")?;
        let mut fixed = [0_u8; FILE_ENTRY_SIZE as usize];
        self.storage.read_at(offset, &mut fixed)?;
        let name = self.read_name(
            offset + FILE_ENTRY_SIZE,
            u64::from(read_u32(&fixed, 0x1C)),
            self.file_meta,
            "file",
        )?;
        if name.is_empty() {
            return Err(LoadError::invalid("RomFS", "file name is empty"));
        }
        Ok(FileEntry {
            parent: read_u32(&fixed, 0),
            sibling: read_u32(&fixed, 4),
            offset: read_u64(&fixed, 8),
            size: read_u64(&fixed, 0x10),
            name,
        })
    }

    fn metadata_offset(
        &self,
        region: Region,
        relative_offset: u32,
        fixed_size: u64,
        kind: &str,
    ) -> Result<u64, LoadError> {
        if !relative_offset.is_multiple_of(4) {
            return Err(LoadError::invalid(
                "RomFS",
                format!("{kind} metadata offset is not aligned"),
            ));
        }
        let relative_offset = u64::from(relative_offset);
        let end = relative_offset
            .checked_add(fixed_size)
            .ok_or_else(|| LoadError::invalid("RomFS", format!("{kind} entry overflows")))?;
        if end > region.size {
            return Err(LoadError::invalid(
                "RomFS",
                format!("{kind} entry points outside its metadata table"),
            ));
        }
        region
            .offset
            .checked_add(relative_offset)
            .ok_or_else(|| LoadError::invalid("RomFS", format!("{kind} offset overflows")))
    }

    fn read_name(
        &self,
        offset: u64,
        name_size: u64,
        region: Region,
        kind: &str,
    ) -> Result<String, LoadError> {
        if name_size > MAX_NAME_SIZE {
            return Err(LoadError::invalid(
                "RomFS",
                format!("{kind} name is unreasonably large"),
            ));
        }
        let end = offset
            .checked_add(name_size)
            .ok_or_else(|| LoadError::invalid("RomFS", format!("{kind} name overflows")))?;
        if end > region.offset + region.size {
            return Err(LoadError::invalid(
                "RomFS",
                format!("{kind} name points outside its metadata table"),
            ));
        }
        let mut bytes = vec![
            0_u8;
            usize::try_from(name_size).map_err(|_| {
                LoadError::invalid("RomFS", format!("{kind} name size is invalid"))
            })?
        ];
        self.storage.read_at(offset, &mut bytes)?;
        let name = std::str::from_utf8(&bytes)
            .map_err(|_| LoadError::invalid("RomFS", format!("{kind} name is not UTF-8")))?;
        if name.contains('/') || name.contains('\0') {
            return Err(LoadError::invalid(
                "RomFS",
                format!("{kind} name is invalid"),
            ));
        }
        Ok(name.to_owned())
    }
}

struct DirectoryEntry {
    parent: u32,
    sibling: u32,
    child_directory: u32,
    child_file: u32,
    name: String,
}

struct FileEntry {
    parent: u32,
    sibling: u32,
    offset: u64,
    size: u64,
    name: String,
}

fn join_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("fixed range"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("fixed range"))
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

    fn build_romfs(files: &[(&str, &[u8])]) -> Vec<u8> {
        let directory_hash_offset = 0x50_u64;
        let directory_hash_size = 4_u64;
        let directory_meta_offset = 0x54_u64;
        let directory_meta_size = 0x18_u64;
        let file_hash_offset = 0x6C_u64;
        let file_hash_size = 4_u64;
        let file_meta_offset = 0x70_u64;

        let mut file_meta = Vec::new();
        let mut data_offset = 0_u64;
        for (index, (name, data)) in files.iter().enumerate() {
            let entry_offset = u32::try_from(file_meta.len()).unwrap();
            let next_offset = if index + 1 == files.len() {
                EMPTY_ENTRY
            } else {
                let next = file_meta.len() + 0x20 + name.len().next_multiple_of(4);
                u32::try_from(next).unwrap()
            };
            file_meta.extend_from_slice(&0_u32.to_le_bytes());
            file_meta.extend_from_slice(&next_offset.to_le_bytes());
            file_meta.extend_from_slice(&data_offset.to_le_bytes());
            file_meta.extend_from_slice(&(data.len() as u64).to_le_bytes());
            file_meta.extend_from_slice(&EMPTY_ENTRY.to_le_bytes());
            file_meta.extend_from_slice(&(name.len() as u32).to_le_bytes());
            file_meta.extend_from_slice(name.as_bytes());
            while file_meta.len() % 4 != 0 {
                file_meta.push(0);
            }
            data_offset += data.len() as u64;
            assert_eq!(
                entry_offset as usize,
                file_meta.len() - (0x20 + name.len().next_multiple_of(4))
            );
        }

        let file_data_offset = (file_meta_offset + file_meta.len() as u64).next_multiple_of(0x10);
        let mut bytes = vec![0_u8; file_data_offset as usize];
        put_u64(&mut bytes, 0, HEADER_SIZE);
        put_u64(&mut bytes, 0x08, directory_hash_offset);
        put_u64(&mut bytes, 0x10, directory_hash_size);
        put_u64(&mut bytes, 0x18, directory_meta_offset);
        put_u64(&mut bytes, 0x20, directory_meta_size);
        put_u64(&mut bytes, 0x28, file_hash_offset);
        put_u64(&mut bytes, 0x30, file_hash_size);
        put_u64(&mut bytes, 0x38, file_meta_offset);
        put_u64(&mut bytes, 0x40, file_meta.len() as u64);
        put_u64(&mut bytes, 0x48, file_data_offset);

        let root = directory_meta_offset as usize;
        put_u32(&mut bytes, root, 0);
        put_u32(&mut bytes, root + 4, EMPTY_ENTRY);
        put_u32(&mut bytes, root + 8, EMPTY_ENTRY);
        put_u32(
            &mut bytes,
            root + 0x0C,
            if files.is_empty() { EMPTY_ENTRY } else { 0 },
        );
        put_u32(&mut bytes, root + 0x10, EMPTY_ENTRY);
        put_u32(&mut bytes, root + 0x14, 0);
        bytes[file_meta_offset as usize..file_meta_offset as usize + file_meta.len()]
            .copy_from_slice(&file_meta);
        for (_, data) in files {
            bytes.extend_from_slice(data);
        }
        bytes
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn load(bytes: Vec<u8>) -> Result<RomFsArchive, LoadError> {
        RomFsLoader::load(Arc::new(VecStorage(bytes)))
    }

    #[test]
    fn lists_and_opens_root_files() {
        let archive = load(build_romfs(&[
            ("control.nacp", b"nacp"),
            ("icon_AmericanEnglish.dat", b"jpeg"),
        ]))
        .unwrap();
        assert_eq!(archive.files().len(), 2);
        assert_eq!(archive.files()[0].path(), "/control.nacp");

        let storage = archive.open("/control.nacp").unwrap().unwrap();
        let mut bytes = [0_u8; 4];
        storage.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"nacp");
        assert!(archive.open("/missing").unwrap().is_none());
    }

    #[test]
    fn rejects_file_outside_data_region() {
        let mut bytes = build_romfs(&[("bad", b"x")]);
        let file_meta_offset = read_u64(&bytes, 0x38) as usize;
        put_u64(&mut bytes, file_meta_offset + 8, u64::MAX);
        assert!(matches!(load(bytes), Err(LoadError::InvalidFormat { .. })));
    }

    #[test]
    fn rejects_file_sibling_cycle() {
        let mut bytes = build_romfs(&[("bad", b"x")]);
        let file_meta_offset = read_u64(&bytes, 0x38) as usize;
        put_u32(&mut bytes, file_meta_offset + 4, 0);
        assert!(matches!(load(bytes), Err(LoadError::InvalidFormat { .. })));
    }
}
