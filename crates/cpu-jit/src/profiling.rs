//! Opt-in, bounded publication diagnostics. File I/O belongs to one writer,
//! never the execution loop, a fault handler, or the lifetime state lock.
//!
//! Linux's format has LOAD/MOVE but no UNLOAD record. Reclamation is recorded
//! separately with monotonic timestamps; a reused address gets a fresh index.
//! https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/jitdump-specification.txt
use crate::executable::{Installed, Tier};
use crate::lifetime::unit::Input;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};

const QUEUED_BYTES: usize = 128 * 1024 * 1024;
const QUEUED_EVENTS: usize = 262_144;
const MAX_DUMP_BYTES: u64 = 4 * 1024 * 1024 * 1024;
static PROFILE: OnceLock<Option<Profiler>> = OnceLock::new();

#[derive(Clone, Debug)]
pub(crate) struct NativeRegion {
    pub start: u32,
    pub end: u32,
    /// One-based ordinal in the captured instruction image, not a guest PC.
    pub instruction: u32,
}

struct Status {
    queued: AtomicUsize,
    dropped: AtomicU64,
    failed: AtomicBool,
}
struct Permit {
    status: Arc<Status>,
    bytes: usize,
}
impl Drop for Permit {
    fn drop(&mut self) {
        self.status.queued.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}
struct Profiler {
    sender: SyncSender<Event>,
    status: Arc<Status>,
    next: AtomicU64,
    order: Mutex<()>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Region {
    start: u32,
    end: u32,
    pc: Option<u64>,
    category: &'static str,
}

pub(crate) struct Load {
    index: u64,
    address: usize,
    name: String,
    bytes: Box<[u8]>,
    regions: Vec<Region>,
    _permit: Permit,
}
enum Event {
    Load {
        timestamp: u64,
        tid: u32,
        load: Load,
    },
    Retire {
        timestamp: u64,
        address: usize,
        size: usize,
    },
    Flush(SyncSender<()>),
}

fn profile() -> Option<&'static Profiler> {
    PROFILE
        .get_or_init(|| {
            let directory = std::env::var_os("NIXE_JITDUMP")?;
            match Profiler::new(Path::new(&directory)) {
                Ok(profile) => Some(profile),
                Err(error) => {
                    log::error!("JIT profile initialization failed: {error}");
                    None
                }
            }
        })
        .as_ref()
}
pub(crate) fn enabled() -> bool {
    profile().is_some()
}

fn now() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // CLOCK_MONOTONIC is the clock used by perf record --clockid mono.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    assert_eq!(result, 0, "Linux CLOCK_MONOTONIC is available");
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}
fn tid() -> u32 {
    unsafe { libc::syscall(libc::SYS_gettid) as u32 }
}

impl Profiler {
    fn new(directory: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        let writer = Writer::new(directory)?;
        let (sender, receiver) = mpsc::sync_channel(QUEUED_EVENTS);
        let status = Arc::new(Status {
            queued: AtomicUsize::new(0),
            dropped: AtomicU64::new(0),
            failed: AtomicBool::new(false),
        });
        let shared = Arc::clone(&status);
        std::thread::Builder::new()
            .name("nixe-jit-profile".into())
            .spawn(move || {
                let mut writer = writer;
                while let Ok(event) = receiver.recv() {
                    if shared.failed.load(Ordering::Acquire) && !matches!(event, Event::Flush(_)) {
                        continue;
                    }
                    let mut acknowledgement = None;
                    let result = match event {
                        Event::Load {
                            timestamp,
                            tid,
                            load,
                        } => writer.load(timestamp, tid, &load),
                        Event::Retire {
                            timestamp,
                            address,
                            size,
                        } => writeln!(
                            writer.lifetime,
                            "retire,{timestamp},0,{address:#x},{size},,0,0,0,reclaimed"
                        ),
                        Event::Flush(reply) => {
                            acknowledgement = Some(reply);
                            writer.flush()
                        }
                    };
                    if let Err(error) = result
                        && !shared.failed.swap(true, Ordering::AcqRel)
                    {
                        log::error!("JIT profile export failed; capture is incomplete: {error}");
                        // Still drain acknowledgements and owned payloads after I/O
                        // failure so diagnostics cannot prevent emulator shutdown.
                    }
                    // Publish failure status before waking shutdown's status reader.
                    if let Some(reply) = acknowledgement {
                        let _ = reply.send(());
                    }
                }
            })?;
        log::info!(
            "JIT profile export enabled: directory={}",
            directory.display()
        );
        Ok(Self {
            sender,
            status,
            next: AtomicU64::new(1),
            order: Mutex::new(()),
        })
    }

    fn reserve(&self, bytes: usize) -> Option<Permit> {
        if self.status.failed.load(Ordering::Acquire)
            || self
                .status
                .queued
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current
                        .checked_add(bytes)
                        .filter(|total| *total <= QUEUED_BYTES)
                })
                .is_err()
        {
            self.status.dropped.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(Permit {
            status: Arc::clone(&self.status),
            bytes,
        })
    }
    fn send(&self, mut event: Event) {
        // Serialize only timestamp/enqueue, never file I/O. This preserves the
        // temporal order when multiple compiler/link publishers are preempted.
        let _order = self
            .order
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut event {
            Event::Load { timestamp, .. } | Event::Retire { timestamp, .. } => *timestamp = now(),
            Event::Flush(_) => {}
        }
        if self.sender.try_send(event).is_err() {
            self.status.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn capture(&self, code: &Installed, name: String, regions: Vec<Region>) -> Option<Load> {
        let size = code.allocation.len();
        let address = code.allocation.address();
        // The caller owns this unpublished allocation. Relocations and native
        // adapters are final; no live patcher can concurrently mutate it.
        let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, size) };
        self.capture_bytes(address, bytes, name, regions)
    }
    fn capture_bytes(
        &self,
        address: usize,
        bytes: &[u8],
        name: String,
        regions: Vec<Region>,
    ) -> Option<Load> {
        let permit = self.reserve(
            bytes.len() + regions.capacity() * std::mem::size_of::<Region>() + name.capacity(),
        )?;
        Some(Load {
            index: self.next.fetch_add(1, Ordering::Relaxed),
            address,
            name,
            bytes: bytes.into(),
            regions,
            _permit: permit,
        })
    }
}

pub(crate) fn prepare(input: &Input) -> Option<Load> {
    let profile = profile()?;
    let first = input.instructions.first()?.key.block_key();
    let tier = match input.tier {
        Tier::Lcq => "LCQ",
        Tier::Hcq => "HCQ",
    };
    let name = format!(
        "nixe_{tier}_p{}_as{}_u{}_v{}_pc{:x}",
        input.identity.process(),
        first.address_space.get(),
        input.identity.id().get(),
        input.identity.version().get(),
        first.pc.get()
    );
    let proofs = input.code.proofs.as_ref()?;
    let sources = proofs
        .regions
        .iter()
        .filter_map(|region| {
            let word = input
                .instructions
                .get(region.instruction.checked_sub(1)? as usize)?;
            (region.start < region.end && region.end as usize <= input.code.allocation.len())
                .then_some(Region {
                    start: region.start,
                    end: region.end,
                    pc: Some(word.key.block_key().pc.get()),
                    category: "guest_lowering",
                })
        })
        .collect::<Vec<_>>();
    let transfers = input
        .states
        .iter()
        .filter_map(|state| {
            let transfer = state.transfer.as_ref()?;
            let end = state
                .native_offset
                .checked_add(u32::from(transfer.patch_bytes))?;
            (state.native_offset < end && end as usize <= input.code.allocation.len()).then_some(
                Region {
                    start: state.native_offset,
                    end,
                    pc: state.exit.map(|exit| exit.pc.get()),
                    category: "dispatch_link",
                },
            )
        })
        .collect::<Vec<_>>();
    let regions = partition_regions(
        input.code.allocation.len() as u32,
        proofs.profile_body_length,
        sources,
        transfers,
    );
    profile.capture(&input.code, name, regions)
}

fn partition_regions(
    size: u32,
    body_end: u32,
    mut sources: Vec<Region>,
    mut transfers: Vec<Region>,
) -> Vec<Region> {
    sources.sort_unstable_by_key(|region| region.start);
    transfers.sort_unstable_by_key(|region| region.start);
    let mut boundaries = vec![0, size, body_end.min(size)];
    for region in sources.iter().chain(&transfers) {
        boundaries.extend([region.start, region.end]);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
        .windows(2)
        .filter_map(|window| {
            let [start, end] = [window[0], window[1]];
            if start == end || end > size {
                return None;
            }
            let containing = |regions: &[Region]| {
                regions
                    .partition_point(|region| region.start <= start)
                    .checked_sub(1)
                    .filter(|index| regions[*index].end >= end)
            };
            if let Some(index) = containing(&transfers) {
                return Some(Region {
                    start,
                    end,
                    ..transfers[index].clone()
                });
            }
            if start >= body_end {
                return Some(Region {
                    start,
                    end,
                    pc: None,
                    category: "entry_exit",
                });
            }
            if let Some(index) = containing(&sources) {
                return Some(Region {
                    start,
                    end,
                    ..sources[index].clone()
                });
            }
            Some(Region {
                start,
                end,
                pc: None,
                category: "generated_scaffolding",
            })
        })
        .collect()
}

pub(crate) fn publish(load: Option<Load>) {
    if let (Some(profile), Some(load)) = (profile(), load) {
        profile.send(Event::Load {
            timestamp: now(),
            tid: tid(),
            load,
        });
    }
}

pub(crate) fn bridge(
    code: &Installed,
    source: &crate::lifetime::unit::CodeUnit,
    target: &crate::lifetime::unit::CodeUnit,
    process: u64,
    dynamic: bool,
) {
    let Some(profile) = profile() else {
        return;
    };
    let tier = match source.tier {
        Tier::Lcq => "LCQ",
        Tier::Hcq => "HCQ",
    };
    let kind = if dynamic {
        "dynamic_link"
    } else {
        "static_link"
    };
    let space = source
        .instructions
        .get(0)
        .unwrap()
        .key
        .block_key()
        .address_space
        .get();
    let name = format!(
        "nixe_{kind}_{tier}_p{process}_as{space}_u{}_v{}_to_u{}_v{}_{:x}",
        source.id.get(),
        source.version.get(),
        target.id.get(),
        target.version.get(),
        code.allocation.address()
    );
    publish(profile.capture(code, name.clone(), Vec::new()));
    if let Some(address) = code.allocation.island_address(0) {
        // Static bridges own one reserved tail island; unused slots have no
        // samples. Initialized slots are final before source exposure.
        let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, 16) };
        publish(profile.capture_bytes(
            address,
            bytes,
            format!("{name}_island"),
            vec![Region {
                start: 0,
                end: 16,
                pc: None,
                category: "dispatch_link",
            }],
        ));
    }
}

/// Link patches change native bytes while preserving semantic maps. Export
/// exact changed intervals before execution reopens, including far islands.
pub(crate) fn patch(
    code: &crate::lifetime::unit::CodeUnit,
    process: u64,
    writes: &[crate::executable::Write<'_>],
) {
    use crate::executable::Write;
    let Some(profile) = profile() else {
        return;
    };
    for write in writes {
        let (address, bytes, kind) = match write {
            Write::Code { offset, bytes } => (
                code.code.allocation.address() + offset,
                *bytes,
                "link_patch",
            ),
            Write::Island { index, bytes } => (
                code.code.allocation.island_address(*index).unwrap(),
                bytes.as_slice(),
                "island",
            ),
        };
        let tier = match code.tier {
            Tier::Lcq => "LCQ",
            Tier::Hcq => "HCQ",
        };
        let space = code
            .instructions
            .get(0)
            .unwrap()
            .key
            .block_key()
            .address_space
            .get();
        let name = format!(
            "nixe_{kind}_{tier}_p{process}_as{space}_u{}_v{}_{address:x}",
            code.id.get(),
            code.version.get()
        );
        let regions = vec![Region {
            start: 0,
            end: bytes.len() as u32,
            pc: None,
            category: "dispatch_link",
        }];
        publish(profile.capture_bytes(address, bytes, name, regions));
    }
}

pub(crate) fn retire(address: usize, size: usize) {
    if let Some(profile) = profile() {
        profile.send(Event::Retire {
            timestamp: now(),
            address,
            size,
        });
    }
}

/// Called only after execution/compiler owners have joined, outside all JIT
/// locks. EOF is a valid end marker; other guest processes may still publish.
pub(crate) fn flush() {
    let Some(profile) = profile() else {
        return;
    };
    let (sender, receiver) = mpsc::sync_channel(0);
    if profile.sender.send(Event::Flush(sender)).is_ok() {
        let _ = receiver.recv();
    }
    log::info!(
        "JIT profile status: dropped_events={} io_failed={}",
        profile.status.dropped.load(Ordering::Relaxed),
        profile.status.failed.load(Ordering::Acquire)
    );
}

struct Writer {
    dump: BufWriter<File>,
    lifetime: BufWriter<File>,
    marker: usize,
    marker_size: usize,
    written: u64,
}
impl Writer {
    fn new(directory: &Path) -> io::Result<Self> {
        let pid = std::process::id();
        // create_new avoids mixing runs or following an existing destination.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(directory.join(format!("jit-{pid}.dump")))?;
        let machine = if cfg!(target_arch = "x86_64") {
            62
        } else {
            183
        };
        for value in [0x4a695444, 1, 40, machine, 0, pid] {
            file.write_all(&u32::to_ne_bytes(value))?;
        }
        file.write_all(&now().to_ne_bytes())?;
        file.write_all(&0_u64.to_ne_bytes())?;
        file.flush()?;
        let lifetime = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(format!("jit-{pid}.regions.csv")))?;
        let marker_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        // Perf recognizes this executable file mapping as a jitdump marker.
        // It is never executed or dereferenced by this runtime.
        let marker = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                marker_size,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if marker == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let mut writer = Self {
            dump: BufWriter::new(file),
            lifetime: BufWriter::new(lifetime),
            marker: marker as usize,
            marker_size,
            written: 40,
        };
        writeln!(
            writer.lifetime,
            "event,monotonic_ns,index,address,size,name,guest_pc,native_start,native_end,category"
        )?;
        Ok(writer)
    }
    fn record(&mut self, id: u32, timestamp: u64, body: &[u8]) -> io::Result<()> {
        let size = 16 + body.len();
        self.written = self
            .written
            .checked_add(size as u64)
            .filter(|size| *size <= MAX_DUMP_BYTES)
            .ok_or_else(|| io::Error::other("jitdump disk budget exhausted"))?;
        let size =
            u32::try_from(size).map_err(|_| io::Error::other("jitdump record exceeds u32"))?;
        self.dump.write_all(&id.to_ne_bytes())?;
        self.dump.write_all(&size.to_ne_bytes())?;
        self.dump.write_all(&timestamp.to_ne_bytes())?;
        self.dump.write_all(body)
    }
    fn load(&mut self, timestamp: u64, tid: u32, load: &Load) -> io::Result<()> {
        // Ends delimit exact source intervals; uncovered native scaffolding is
        // explicitly named, rather than attributed to the preceding guest PC.
        let mut locations = BTreeMap::new();
        for region in &load.regions {
            locations.insert(region.end, (None, "generated_scaffolding"));
        }
        for region in &load.regions {
            locations.insert(region.start, (region.pc, region.category));
        }
        if !locations.is_empty() {
            locations
                .entry(0)
                .or_insert((None, "generated_scaffolding"));
            let mut debug = Vec::new();
            debug.extend_from_slice(&(load.address as u64).to_ne_bytes());
            debug.extend_from_slice(&(locations.len() as u64).to_ne_bytes());
            for (offset, (pc, category)) in locations {
                debug.extend_from_slice(&(load.address as u64 + u64::from(offset)).to_ne_bytes());
                debug.extend_from_slice(&1_u32.to_ne_bytes());
                debug.extend_from_slice(&0_u32.to_ne_bytes());
                let source = pc.map_or_else(
                    || format!("jit-{category}"),
                    |pc| format!("guest-pc-{pc:x}.a64"),
                );
                debug.extend_from_slice(source.as_bytes());
                debug.push(0);
            }
            self.record(2, timestamp, &debug)?;
        }
        let mut body = Vec::with_capacity(40 + load.name.len() + 1 + load.bytes.len());
        body.extend_from_slice(&std::process::id().to_ne_bytes());
        body.extend_from_slice(&tid.to_ne_bytes());
        for value in [
            load.address as u64,
            load.address as u64,
            load.bytes.len() as u64,
            load.index,
        ] {
            body.extend_from_slice(&value.to_ne_bytes());
        }
        body.extend_from_slice(load.name.as_bytes());
        body.push(0);
        body.extend_from_slice(&load.bytes);
        self.record(0, timestamp, &body)?;
        writeln!(
            self.lifetime,
            "load,{timestamp},{},{:#x},{},{},0,0,0,unit",
            load.index,
            load.address,
            load.bytes.len(),
            load.name
        )?;
        for region in &load.regions {
            writeln!(
                self.lifetime,
                "region,{timestamp},{},0,0,,{:#x},{},{},{}",
                load.index,
                region.pc.unwrap_or(0),
                region.start,
                region.end,
                region.category
            )?;
        }
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.dump.flush()?;
        self.lifetime.flush()
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.flush();
        unsafe {
            libc::munmap(self.marker as *mut libc::c_void, self.marker_size);
        }
    }
}

#[cfg(test)]
mod tests;
