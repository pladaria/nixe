//! RomFS metadata over lazy file extents. No payload is materialized while building.
//! https://switchbrew.org/wiki/RomFS

use super::*;
use crate::tree::{ReadOnlyTree, TreeEntry};
use nixe_loader_storage::{Storage, StorageError};
use std::collections::BTreeMap;

struct Directory {
    parent: u32,
    name: String,
    offset: u32,
    directories: Vec<usize>,
    files: Vec<usize>,
}
struct File {
    parent: u32,
    name: String,
    path: String,
    offset: u32,
    data: u64,
    size: u64,
    source: StorageRef,
}
struct Extent {
    start: u64,
    size: u64,
    source: StorageRef,
}
struct Image {
    metadata: Vec<u8>,
    extents: Vec<Extent>,
    size: u64,
}

impl Storage for Image {
    fn len(&self) -> Result<u64, StorageError> {
        Ok(self.size)
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), StorageError> {
        let end = offset
            .checked_add(bytes.len() as u64)
            .ok_or(StorageError::OutOfBounds)?;
        if end > self.size {
            return Err(StorageError::OutOfBounds);
        }
        let mut position = offset;
        let mut written = 0;
        if position < self.metadata.len() as u64 {
            let count = bytes.len().min(self.metadata.len() - position as usize);
            bytes[..count]
                .copy_from_slice(&self.metadata[position as usize..position as usize + count]);
            position += count as u64;
            written += count;
        }
        let mut index = self
            .extents
            .partition_point(|extent| extent.start + extent.size <= position);
        while written < bytes.len() {
            let extent = self.extents.get(index);
            if let Some(extent) = extent.filter(|extent| extent.start <= position) {
                let count = (end - position).min(extent.start + extent.size - position) as usize;
                extent.source.read_at(
                    position - extent.start,
                    &mut bytes[written..written + count],
                )?;
                written += count;
                position += count as u64;
                index += 1;
            } else {
                let next = extent.map_or(end, |extent| extent.start.min(end));
                let count = (next - position) as usize;
                bytes[written..written + count].fill(0);
                written += count;
                position = next;
            }
        }
        Ok(())
    }
}

fn invalid(reason: impl Into<String>) -> LoadError {
    LoadError::invalid("RomFS", reason)
}
fn add(a: u64, b: u64) -> Result<u64, LoadError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("image size overflows"))
}
fn align(value: u64, alignment: u64) -> Result<u64, LoadError> {
    Ok(add(value, alignment - 1)? & !(alignment - 1))
}
fn offset(value: u64) -> Result<u32, LoadError> {
    u32::try_from(value)
        .ok()
        .filter(|value| *value != EMPTY_ENTRY)
        .ok_or_else(|| invalid("metadata exceeds 32-bit offsets"))
}
fn word(out: &mut [u8], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn wide(out: &mut [u8], at: usize, value: u64) {
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
fn hash(parent: u32, name: &str, buckets: usize) -> usize {
    let value = name.bytes().fold(parent ^ 123456789, |value, byte| {
        value.rotate_right(5) ^ u32::from(byte)
    });
    value as usize % buckets
}
fn buckets(count: usize) -> usize {
    if count < 3 {
        return 3;
    }
    if count < 19 {
        return count | 1;
    }
    let mut result = count;
    while [2, 3, 5, 7, 11, 13, 17]
        .iter()
        .any(|divisor| result.is_multiple_of(*divisor))
    {
        result += 1;
    }
    result
}

pub(super) fn build(tree: ReadOnlyTree) -> Result<RomFsArchive, LoadError> {
    let mut directories = vec![Directory {
        parent: 0,
        name: String::new(),
        offset: 0,
        directories: Vec::new(),
        files: Vec::new(),
    }];
    let mut directory_paths = BTreeMap::from([("/".to_owned(), 0usize)]);
    let mut files = Vec::new();
    let mut directory_size = DIRECTORY_ENTRY_SIZE;
    let mut file_size = 0;
    let mut data_size = 0;
    for (path, entry) in tree.entries {
        if path == "/" {
            continue;
        }
        let (parent_path, name) = path.rsplit_once('/').expect("normalized path");
        let parent_path = if parent_path.is_empty() {
            "/"
        } else {
            parent_path
        };
        let parent = *directory_paths
            .get(parent_path)
            .ok_or_else(|| invalid("missing parent directory"))?;
        let parent_offset = directories[parent].offset;
        match entry {
            TreeEntry::Directory => {
                let index = directories.len();
                let entry_offset = offset(directory_size)?;
                directory_size = align(
                    add(directory_size, DIRECTORY_ENTRY_SIZE + name.len() as u64)?,
                    4,
                )?;
                directories[parent].directories.push(index);
                directories.push(Directory {
                    parent: parent_offset,
                    name: name.into(),
                    offset: entry_offset,
                    directories: Vec::new(),
                    files: Vec::new(),
                });
                directory_paths.insert(path, index);
            }
            TreeEntry::File(source) => {
                let size = source.len()?;
                let data = align(data_size, 16)?;
                data_size = add(data, size)?;
                let entry_offset = offset(file_size)?;
                file_size = align(add(file_size, FILE_ENTRY_SIZE + name.len() as u64)?, 4)?;
                directories[parent].files.push(files.len());
                files.push(File {
                    parent: parent_offset,
                    name: name.into(),
                    path,
                    offset: entry_offset,
                    data,
                    size,
                    source,
                });
            }
        }
    }
    offset(directory_size)?;
    offset(file_size)?;
    let mut directory_hash = vec![EMPTY_ENTRY; buckets(directories.len())];
    let mut file_hash = vec![EMPTY_ENTRY; buckets(files.len())];
    let mut directory_meta = vec![0; directory_size as usize];
    let mut file_meta = vec![0; file_size as usize];
    for directory in &directories {
        let at = directory.offset as usize;
        let bucket = hash(directory.parent, &directory.name, directory_hash.len());
        word(&mut directory_meta, at, directory.parent);
        word(&mut directory_meta, at + 4, EMPTY_ENTRY);
        word(
            &mut directory_meta,
            at + 8,
            directory
                .directories
                .first()
                .map_or(EMPTY_ENTRY, |index| directories[*index].offset),
        );
        word(
            &mut directory_meta,
            at + 12,
            directory
                .files
                .first()
                .map_or(EMPTY_ENTRY, |index| files[*index].offset),
        );
        word(&mut directory_meta, at + 16, directory_hash[bucket]);
        word(&mut directory_meta, at + 20, directory.name.len() as u32);
        directory_meta[at + 24..at + 24 + directory.name.len()]
            .copy_from_slice(directory.name.as_bytes());
        directory_hash[bucket] = directory.offset;
    }
    for file in &files {
        let at = file.offset as usize;
        let bucket = hash(file.parent, &file.name, file_hash.len());
        word(&mut file_meta, at, file.parent);
        word(&mut file_meta, at + 4, EMPTY_ENTRY);
        wide(&mut file_meta, at + 8, file.data);
        wide(&mut file_meta, at + 16, file.size);
        word(&mut file_meta, at + 24, file_hash[bucket]);
        word(&mut file_meta, at + 28, file.name.len() as u32);
        file_meta[at + 32..at + 32 + file.name.len()].copy_from_slice(file.name.as_bytes());
        file_hash[bucket] = file.offset;
    }
    for directory in &directories {
        for pair in directory.directories.windows(2) {
            word(
                &mut directory_meta,
                directories[pair[0]].offset as usize + 4,
                directories[pair[1]].offset,
            );
        }
        for pair in directory.files.windows(2) {
            word(
                &mut file_meta,
                files[pair[0]].offset as usize + 4,
                files[pair[1]].offset,
            );
        }
    }
    let directory_hash_at = HEADER_SIZE;
    let directory_meta_at = add(directory_hash_at, directory_hash.len() as u64 * 4)?;
    let file_hash_at = add(directory_meta_at, directory_size)?;
    let file_meta_at = add(file_hash_at, file_hash.len() as u64 * 4)?;
    let data_at = align(add(file_meta_at, file_size)?, 0x1000)?;
    let size = add(data_at, data_size)?;
    let capacity =
        usize::try_from(data_at).map_err(|_| invalid("metadata exceeds host address space"))?;
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(capacity)
        .map_err(|error| invalid(error.to_string()))?;
    for field in [
        HEADER_SIZE,
        directory_hash_at,
        directory_hash.len() as u64 * 4,
        directory_meta_at,
        directory_size,
        file_hash_at,
        file_hash.len() as u64 * 4,
        file_meta_at,
        file_size,
        data_at,
    ] {
        metadata.extend(field.to_le_bytes());
    }
    for value in directory_hash {
        metadata.extend(value.to_le_bytes());
    }
    metadata.extend(directory_meta);
    for value in file_hash {
        metadata.extend(value.to_le_bytes());
    }
    metadata.extend(file_meta);
    metadata.resize(capacity, 0);
    let mut extents = Vec::with_capacity(files.len());
    let mut index = Vec::with_capacity(files.len());
    for file in files {
        index.push(RomFsFile {
            path: file.path,
            offset: file.data,
            size: file.size,
        });
        if file.size != 0 {
            extents.push(Extent {
                start: data_at + file.data,
                size: file.size,
                source: file.source,
            });
        }
    }
    Ok(RomFsArchive {
        storage: Arc::new(Image {
            metadata,
            extents,
            size,
        }),
        file_data_offset: data_at,
        files: index,
        directories: directory_paths.into_keys().collect(),
    })
}
