//! Optional bounded derived-data storage. Corruption is a miss, never an input
//! to Vulkan. This wire format contains only explicit scalars/bytes, not IR layout.
use nixe_gpu::ssa::artifact::ContentId;
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
const MAGIC: &[u8; 8] = b"NXVCACH1";
const HEADER: usize = 8 + 32 + 8 + 32;

pub(super) struct Store {
    directory: PathBuf,
    budget: u64,
    lock: Mutex<()>,
}
impl Store {
    pub fn new(directory: PathBuf, budget: u64) -> Self {
        Self {
            directory,
            budget,
            lock: Mutex::new(()),
        }
    }
    pub fn path(&self, key: ContentId) -> PathBuf {
        self.directory.join(format!("{}.cache", key.hex()))
    }
    pub fn read(&self, key: ContentId) -> Option<Vec<u8>> {
        let mut f = std::fs::File::open(self.path(key)).ok()?;
        let size = f.metadata().ok()?.len();
        if size < HEADER as u64 || size > self.budget {
            return None;
        }
        let mut header = [0; HEADER];
        f.read_exact(&mut header).ok()?;
        if &header[..8] != MAGIC || header[8..40] != key.0 {
            return None;
        }
        let length = u64::from_le_bytes(header[40..48].try_into().ok()?);
        if length != size - HEADER as u64 {
            return None;
        }
        let mut data = vec![0; usize::try_from(length).ok()?];
        f.read_exact(&mut data).ok()?;
        (ContentId::of(b"nixe-cache-payload", &[&data]).0 == header[48..80]).then_some(data)
    }
    pub fn write(&self, key: ContentId, data: &[u8]) -> std::io::Result<()> {
        if data.len() as u64 + HEADER as u64 > self.budget {
            return Ok(());
        }
        let _lock = self.lock.lock().unwrap();
        std::fs::create_dir_all(&self.directory)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let temporary = self.directory.join(format!(
            ".{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            f.write_all(MAGIC)?;
            f.write_all(&key.0)?;
            f.write_all(&(data.len() as u64).to_le_bytes())?;
            f.write_all(&ContentId::of(b"nixe-cache-payload", &[data]).0)?;
            f.write_all(data)?;
            f.sync_all()?;
            std::fs::rename(&temporary, self.path(key))?;
            self.prune()
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }
    fn prune(&self) -> std::io::Result<()> {
        let mut files: Vec<_> = std::fs::read_dir(&self.directory)?
            .filter_map(|e| {
                let e = e.ok()?;
                let path = e.path();
                if path.extension() != Some(std::ffi::OsStr::new("cache")) {
                    return None;
                }
                let m = e.metadata().ok()?;
                m.is_file().then(|| (m.modified().ok(), m.len(), path))
            })
            .collect();
        let mut bytes: u64 = files.iter().map(|(_, size, _)| size).sum();
        files.sort_by_key(|(time, _, _)| *time);
        for (_, size, path) in files {
            if bytes <= self.budget {
                break;
            }
            std::fs::remove_file(path)?;
            bytes -= size;
        }
        Ok(())
    }
}
pub(super) fn store(directory: Option<&Path>, child: &str, budget: u64) -> Option<Store> {
    directory.map(|dir| Store::new(dir.join(child), budget))
}

pub(super) struct Reader<'a>(pub &'a [u8]);
impl<'a> Reader<'a> {
    pub fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (a, b) = self.0.split_at_checked(n)?;
        self.0 = b;
        Some(a)
    }
    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}
pub(super) fn word(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
