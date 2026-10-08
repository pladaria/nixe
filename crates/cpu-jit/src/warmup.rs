//! Versioned, bounded warmup hints. Only module offsets and exact guest words
//! survive a process. Native code, relocations, faults and lifetime identities
//! are always rebuilt by normal capture/compilation/publication.
use crate::{
    abi::{BlockKey, FpSpecialization, HostAbi},
    engine::JitProcess,
    jit_error::Error,
    lcq::{
        Compilation, Fragment,
        compiler::{Compiler, PublishError},
    },
    lifetime::{Lifetime, compile::Request},
};
use nixe_cpu::{memory::ExecutionMemory, profile::ProcessCpuContext};
use nixe_memory::GuestVirtualAddress;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

const MAGIC: &[u8; 8] = b"NXWARM01";
// Bump whenever compiler policy, native ABI, profile encoding or code options change.
const COMPILER_ABI: &str = "nixe-lcq-warmup-5";
const MAX_RECORDS: usize = 65_536;
const MAX_BYTES: usize = 32 * 1024 * 1024;

/// Caller-provided content identity and current placement, independent of ASLR.
#[derive(Clone, Debug)]
pub struct WarmupModule {
    pub content: [u8; 32],
    pub base: u64,
    pub extent: u64,
}
#[derive(Clone, Debug)]
pub struct WarmupConfig {
    pub directory: PathBuf,
    pub modules: Vec<WarmupModule>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    module: u16,
    offset: u64,
    words: Vec<u32>,
}
struct Records {
    ordered: Vec<Record>,
    index: HashMap<(u16, u64), usize>,
    bytes: usize,
}
pub(crate) struct Warmup {
    path: PathBuf,
    identity: [u8; 32],
    modules: Vec<WarmupModule>,
    records: Mutex<Records>,
    saved: AtomicBool,
    cancelled: AtomicBool,
    start: (Mutex<Option<bool>>, Condvar),
}

pub(crate) struct Task {
    pub profile: Arc<Warmup>,
    pub memory: Arc<ExecutionMemory>,
    pub cpu: ProcessCpuContext,
    pub arena_size: usize,
}

pub(crate) fn host_abi() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}
impl Warmup {
    pub fn load(cpu: ProcessCpuContext, config: WarmupConfig) -> Result<Arc<Self>, Error> {
        if config.modules.len() > usize::from(u16::MAX)
            || config
                .modules
                .iter()
                .any(|m| m.base & 3 != 0 || m.extent == 0 || m.base.checked_add(m.extent).is_none())
        {
            return Err(Error::invalid("invalid JIT warmup module placement"));
        }
        let mut ranges: Vec<_> = config
            .modules
            .iter()
            .map(|m| (m.base, m.base + m.extent))
            .collect();
        ranges.sort_unstable();
        if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::invalid("overlapping JIT warmup modules"));
        }
        let isa = crate::frontend::target::build(host_abi(), crate::frontend::target::Policy::Lcq)?;
        let mut hash = Sha256::new();
        hash.update(COMPILER_ABI);
        hash.update(format!("{:?}:{:?}:{}", cpu, host_abi(), isa.flags()));
        for flag in isa.isa_flags() {
            hash.update(flag.to_string());
            hash.update([0]);
        }
        for module in &config.modules {
            hash.update(module.content);
            hash.update(module.extent.to_le_bytes());
        }
        let identity: [u8; 32] = hash.finalize().into();
        let name: String = identity.iter().map(|byte| format!("{byte:02x}")).collect();
        let path = config.directory.join(format!("{name}.warmup"));
        let ordered = match read(&path, &identity, &config.modules) {
            Ok(records) => records,
            Err(error) => {
                log::debug!("JIT warmup profile miss: {}: {error}", path.display());
                Vec::new()
            }
        };
        let bytes = ordered.iter().map(|r| 12 + r.words.len() * 4).sum();
        let index = ordered
            .iter()
            .enumerate()
            .map(|(i, r)| ((r.module, r.offset), i))
            .collect();
        log::info!(
            "JIT warmup profile: entries={} path={}",
            ordered.len(),
            path.display()
        );
        Ok(Arc::new(Self {
            path,
            identity,
            modules: config.modules,
            records: Mutex::new(Records {
                ordered,
                index,
                bytes,
            }),
            saved: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            start: (Mutex::new(None), Condvar::new()),
        }))
    }
    pub(crate) fn record(&self, fragment: &Fragment) -> Option<Record> {
        if fragment.key.fp != FpSpecialization::Dynamic {
            return None;
        }
        let pc = fragment.key.pc.get();
        let bytes = fragment.image.words().len() as u64 * 4;
        let (module, image) = self.modules.iter().enumerate().find(|(_, m)| {
            pc >= m.base
                && pc
                    .checked_add(bytes)
                    .is_some_and(|end| end <= m.base + m.extent)
        })?;
        Some(Record {
            module: module as u16,
            offset: pc - image.base,
            words: fragment
                .image
                .words()
                .iter()
                .map(|word| word.bits)
                .collect(),
        })
    }
    pub fn observe(&self, record: Record) {
        let mut records = self.records.lock().expect("JIT warmup records poisoned");
        let bytes = 12 + record.words.len() * 4;
        if let Some(&index) = records.index.get(&(record.module, record.offset)) {
            let old = 12 + records.ordered[index].words.len() * 4;
            if records.bytes - old + bytes <= MAX_BYTES {
                records.bytes = records.bytes - old + bytes;
                records.ordered[index] = record;
            }
        } else if records.ordered.len() < MAX_RECORDS && records.bytes + bytes <= MAX_BYTES {
            let index = records.ordered.len();
            records.index.insert((record.module, record.offset), index);
            records.ordered.push(record);
            records.bytes += bytes;
        }
    }
    pub fn start(&self, enabled: bool) {
        let mut state = self.start.0.lock().expect("JIT warmup start poisoned");
        if state.is_none() {
            *state = Some(enabled);
            self.start.1.notify_all();
        }
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.start(false);
    }
    pub fn save(&self) {
        if self.saved.swap(true, Ordering::AcqRel) {
            return;
        }
        let records = self.records.lock().expect("JIT warmup records poisoned");
        let result = write(&self.path, &self.identity, &records.ordered);
        match result {
            Ok(()) => log::info!(
                "saved JIT warmup profile: entries={} path={}",
                records.ordered.len(),
                self.path.display()
            ),
            Err(error) => log::warn!(
                "cannot save optional JIT warmup profile {}: {error}",
                self.path.display()
            ),
        }
    }
}

impl Task {
    /// Runs on one existing compiler worker before its HCQ loop. No extra pool,
    /// process ownership cycle or unbounded waiting for a demanded compilation.
    pub fn run(
        self,
        lifetime: &Arc<Lifetime>,
        mut service_hcq: impl FnMut() -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut ready = self
            .profile
            .start
            .0
            .lock()
            .expect("JIT warmup start poisoned");
        while ready.is_none() {
            ready = self
                .profile
                .start
                .1
                .wait(ready)
                .expect("JIT warmup start poisoned");
        }
        if *ready == Some(false) {
            return Ok(());
        }
        drop(ready);
        let records = self
            .profile
            .records
            .lock()
            .expect("JIT warmup records poisoned")
            .ordered
            .clone();
        if records.is_empty() {
            return Ok(());
        }
        let mut reader = match lifetime.register() {
            Ok(reader) => reader,
            Err(crate::lifetime::Error::Shutdown) => return Ok(()),
            Err(error) => return Err(lifetime.diagnostic(error)),
        };
        let mut compiler = Compiler::for_arena(host_abi(), self.arena_size)?;
        let mut capture = crate::lcq::Capture::default();
        let mut published = 0;
        let mut skipped = 0;
        for (ordinal, record) in records.into_iter().enumerate() {
            if ordinal.is_multiple_of(16) {
                service_hcq()?;
            }
            if self.profile.cancelled.load(Ordering::Acquire) {
                break;
            }
            let module = &self.profile.modules[usize::from(record.module)];
            let key = BlockKey::new(
                self.cpu,
                GuestVirtualAddress::new(module.base + record.offset),
                FpSpecialization::Dynamic,
            )
            .unwrap();
            let result = (|| -> Result<(), PublishError> {
                // A warmup hint is not execution demand. Never wait on a vCPU's
                // claim and never reuse stored permissions/content stamps.
                let Request::Owner(claim) = reader.claim(key)? else {
                    skipped += 1;
                    return Ok(());
                };
                let compilation = Compilation::capture_with(claim, &*self.memory, &mut capture)?;
                if !matches_words(&compilation.fragment, &record.words) {
                    skipped += 1;
                    return Ok(());
                }
                compiler.publish(
                    compilation,
                    lifetime,
                    lifetime.executable_cache(),
                    &*self.memory,
                )?;
                published += 1;
                Ok(())
            })();
            match result {
                Ok(())
                | Err(
                    PublishError::StaleCapture
                    | PublishError::Lifetime(crate::lifetime::Error::StalePublication),
                ) => {}
                Err(PublishError::Lifetime(crate::lifetime::Error::Closed)) => {
                    if !lifetime
                        .wait_for_warmup(&self.profile.cancelled)
                        .map_err(|e| lifetime.diagnostic(e))?
                    {
                        break;
                    }
                }
                Err(PublishError::Lifetime(crate::lifetime::Error::Shutdown)) => break,
                Err(error) if error.capacity().is_some() => break,
                Err(error) => {
                    return Err(Error::internal(format!("JIT warmup compilation: {error}")));
                }
            }
        }
        log::info!("JIT warmup completed: published={published} skipped={skipped}");
        Ok(())
    }
}
fn matches_words(fragment: &Fragment, words: &[u32]) -> bool {
    !words.is_empty()
        && fragment.image.fault().is_none()
        && fragment.image.words().len() == words.len()
        && fragment
            .image
            .words()
            .iter()
            .zip(words)
            .all(|(word, bits)| word.bits == *bits)
}
fn read(
    path: &Path,
    identity: &[u8; 32],
    modules: &[WarmupModule],
) -> std::io::Result<Vec<Record>> {
    use std::io::{Error as IoError, ErrorKind};
    let invalid = || {
        IoError::new(
            ErrorKind::InvalidData,
            "invalid or stale JIT warmup profile",
        )
    };
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > MAX_BYTES as u64 + 44 {
        return Err(invalid());
    }
    let mut data = Vec::new();
    file.take((MAX_BYTES + 45) as u64).read_to_end(&mut data)?;
    if data.len() < 44 || &data[..8] != MAGIC || &data[8..40] != identity {
        return Err(invalid());
    }
    let count = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
    if count > MAX_RECORDS {
        return Err(invalid());
    }
    let mut offset = 44;
    let mut seen = std::collections::HashSet::new();
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let header = data.get(offset..offset + 12).ok_or_else(invalid)?;
        let module = u16::from_le_bytes(header[..2].try_into().unwrap());
        let words = usize::from(u16::from_le_bytes(header[2..4].try_into().unwrap()));
        let pc = u64::from_le_bytes(header[4..12].try_into().unwrap());
        let image = modules.get(usize::from(module)).ok_or_else(invalid)?;
        if !(1..=512).contains(&words)
            || pc & 3 != 0
            || !pc
                .checked_add(words as u64 * 4)
                .is_some_and(|end| end <= image.extent)
            || !seen.insert((module, pc))
        {
            return Err(invalid());
        }
        offset += 12;
        let bytes = data.get(offset..offset + words * 4).ok_or_else(invalid)?;
        offset += bytes.len();
        records.push(Record {
            module,
            offset: pc,
            words: bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        });
    }
    if offset != data.len() {
        return Err(invalid());
    }
    Ok(records)
}
fn write(path: &Path, identity: &[u8; 32], records: &[Record]) -> std::io::Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut created = false;
    let result = (|| {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        created = true;
        let mut file = BufWriter::new(file);
        file.write_all(MAGIC)?;
        file.write_all(identity)?;
        file.write_all(&(records.len() as u32).to_le_bytes())?;
        for record in records {
            file.write_all(&record.module.to_le_bytes())?;
            file.write_all(&(record.words.len() as u16).to_le_bytes())?;
            file.write_all(&record.offset.to_le_bytes())?;
            for word in &record.words {
                file.write_all(&word.to_le_bytes())?;
            }
        }
        file.flush()?;
        std::fs::rename(&temporary, path)?;
        evict_profiles(path, 8)
    })();
    if result.is_err() && created {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

// A dedicated cache directory retains at most eight profiles (256 MiB plus a
// bounded temporary file). Only our exact content-keyed regular files qualify.
fn evict_profiles(current: &Path, limit: usize) -> std::io::Result<()> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(current.parent().unwrap())? {
        let entry = entry?;
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if path == current
            || path.extension().and_then(|s| s.to_str()) != Some("warmup")
            || stem.len() != 64
            || !stem.bytes().all(|b| b.is_ascii_hexdigit())
            || !entry.file_type()?.is_file()
        {
            continue;
        }
        files.push((entry.metadata()?.modified()?, path));
    }
    files.sort_unstable();
    let remove = files.len().saturating_sub(limit.saturating_sub(1));
    for (_, path) in files.into_iter().take(remove) {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl JitProcess {
    pub(crate) fn save_warmup(&self) {
        if let Some(warmup) = &self.warmup {
            warmup.save();
        }
    }
}
#[cfg(test)]
mod tests;
