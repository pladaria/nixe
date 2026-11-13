use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use nixe_loader_storage::{Storage, StorageError, StorageRef};
use sha2::{Digest, Sha256};

use crate::zstd_stream::ProgressiveZstd;

const BUDGET: u64 = 512 * 1024 * 1024;
const WINDOW: u64 = 128 * 1024 * 1024;

/// One container owns the cache; opening the same physical extent shares its decoding state.
#[derive(Default)]
pub(super) struct Cache {
    entries: HashMap<u64, Entry>,
    ages: BTreeMap<u64, u64>,
    clock: u64,
    bytes: u64,
}

struct Entry {
    age: u64,
    charge: u64,
    member: Arc<Mutex<Member>>,
}

impl Cache {
    fn get(&mut self, view: &CompressedFile) -> Result<Arc<Mutex<Member>>, StorageError> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| invalid("NUP cache age overflow"))?;
        if let Some(entry) = self.entries.get_mut(&view.key) {
            self.ages.remove(&entry.age);
            entry.age = self.clock;
            self.ages.insert(entry.age, view.key);
            return Ok(entry.member.clone());
        }
        // Validate on first consumption, so indexing a large tree never touches
        // every compressed member or allocates a decoder for unopened files.
        let mut header = [0; 18];
        let count = view.source.len()?.min(header.len() as u64) as usize;
        view.source.read_at(0, &mut header[..count])?;
        if count < 4
            || header[..4] != [0x28, 0xb5, 0x2f, 0xfd]
            || zstd::zstd_safe::get_frame_content_size(&header[..count])
                .ok()
                .flatten()
                != Some(view.size)
        {
            return Err(invalid("invalid NUP Zstandard frame or content size"));
        }
        let memory = view.size <= BUDGET / 2;
        // Account for output, decoder window and scratch space. In-flight readers retain
        // their member if evicted; it cannot be recycled underneath a read.
        let charge = u64::from(memory) * view.size + view.size.min(WINDOW) + 1024 * 1024;
        while self.bytes + charge > BUDGET {
            let Some((_, key)) = self.ages.pop_first() else {
                break;
            };
            let entry = self.entries.remove(&key).expect("cache age has an entry");
            self.bytes -= entry.charge;
        }
        let output = if memory {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(view.size as usize)
                .map_err(|error| invalid(error.to_string()))?;
            bytes.resize(view.size as usize, 0);
            Output::Memory(bytes)
        } else {
            Output::File(tempfile::tempfile()?)
        };
        let member = Arc::new(Mutex::new(Member {
            stream: ProgressiveZstd::new(view.source.clone(), view.size),
            output,
            hash: Sha256::new(),
        }));
        self.bytes += charge;
        self.ages.insert(self.clock, view.key);
        self.entries.insert(
            view.key,
            Entry {
                age: self.clock,
                charge,
                member: member.clone(),
            },
        );
        Ok(member)
    }
}

enum Output {
    Memory(Vec<u8>),
    File(File),
}

struct Member {
    stream: ProgressiveZstd,
    output: Output,
    hash: Sha256,
}

pub(super) struct CompressedFile {
    pub(super) source: StorageRef,
    pub(super) key: u64,
    pub(super) size: u64,
    pub(super) hash: [u8; 32],
    pub(super) cache: Arc<Mutex<Cache>>,
}

impl Storage for CompressedFile {
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
        if bytes.is_empty() {
            return Ok(());
        }
        let member = self
            .cache
            .lock()
            .map_err(|_| invalid("NUP cache lock poisoned"))?
            .get(self)?;
        let mut member = member
            .lock()
            .map_err(|_| invalid("NUP member lock poisoned"))?;
        let Member {
            stream,
            output,
            hash,
        } = &mut *member;
        stream.decode_until(end, |position, data| {
            hash.update(&*data);
            if position + data.len() as u64 == self.size
                && hash.clone().finalize().as_slice() != self.hash
            {
                return Err(invalid("NUP member SHA-256 mismatch"));
            }
            match output {
                Output::Memory(memory) => {
                    memory[position as usize..position as usize + data.len()].copy_from_slice(data)
                }
                Output::File(file) => {
                    file.seek(SeekFrom::Start(position))?;
                    file.write_all(data)?;
                }
            }
            Ok(())
        })?;
        match output {
            Output::Memory(memory) => bytes.copy_from_slice(&memory[offset as usize..end as usize]),
            Output::File(file) => {
                file.seek(SeekFrom::Start(offset))?;
                file.read_exact(bytes)?;
            }
        }
        Ok(())
    }
}

fn invalid(reason: impl Into<String>) -> StorageError {
    StorageError::InvalidData(reason.into())
}
