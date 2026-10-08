use std::{
    cell::RefCell,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

#[cfg(not(test))]
const MAX_EVENTS_PER_THREAD: usize = 1_048_576;
#[cfg(test)]
const MAX_EVENTS_PER_THREAD: usize = 64;
static OWNED: AtomicBool = AtomicBool::new(false);
static EPOCH: AtomicU64 = AtomicU64::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static ORIGIN: OnceLock<Instant> = OnceLock::new();
static THREADS: Mutex<Vec<Arc<Mutex<Buffer>>>> = Mutex::new(Vec::new());
struct Buffer {
    name: String,
    events: Vec<Event>,
    dropped: u64,
}
struct Event {
    name: &'static str,
    start: u64,
    duration: Option<u64>,
    id: u64,
    value: u64,
}
thread_local! {
    static BUFFER: RefCell<Option<Arc<Mutex<Buffer>>>> = const { RefCell::new(None) };
}
#[inline]
pub fn enabled() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}
pub fn clock_ns() -> u64 {
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
fn record(event: Event, epoch: u64) {
    BUFFER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let buffer = slot.get_or_insert_with(|| {
            let buffer = Arc::new(Mutex::new(Buffer {
                name: std::thread::current()
                    .name()
                    .unwrap_or("unnamed")
                    .to_owned(),
                events: Vec::new(),
                dropped: 0,
            }));
            THREADS.lock().unwrap().push(buffer.clone());
            buffer
        });
        let mut buffer = buffer.lock().unwrap();
        if !enabled() || EPOCH.load(Ordering::Relaxed) != epoch {
            return;
        }
        if buffer.events.len() < MAX_EVENTS_PER_THREAD {
            buffer.events.push(event);
        } else {
            buffer.dropped += 1;
        }
    });
}
#[inline]
pub fn event(name: &'static str, id: u64, value: u64) {
    let epoch = epoch();
    if enabled() {
        record(
            Event {
                name,
                start: clock_ns(),
                duration: None,
                id,
                value,
            },
            epoch,
        );
    }
}
pub struct Span {
    name: &'static str,
    start: Option<u64>,
    epoch: u64,
    id: u64,
    value: u64,
}
impl Span {
    #[inline]
    pub fn new(name: &'static str, id: u64, value: u64) -> Self {
        let epoch = epoch();
        Self {
            name,
            start: enabled().then(clock_ns),
            epoch,
            id,
            value,
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start
            && enabled()
        {
            record(
                Event {
                    name: self.name,
                    start,
                    duration: Some(clock_ns() - start),
                    id: self.id,
                    value: self.value,
                },
                self.epoch,
            );
        }
    }
}
/// Linux thread CPU time separates actual execution from off-CPU elapsed time.
/// Unsupported hosts omit this diagnostic instead of inventing a CPU duration.
fn thread_cpu_ns() -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: clock_gettime initializes the supplied timespec on success.
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
            return None;
        }
        let time = unsafe { time.assume_init() };
        Some((time.tv_sec as u64).saturating_mul(1_000_000_000) + time.tv_nsec as u64)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}
pub struct CpuSpan {
    name: &'static str,
    id: u64,
    epoch: u64,
    start: Option<u64>,
}
impl CpuSpan {
    pub fn new(name: &'static str, id: u64) -> Self {
        Self {
            name,
            id,
            epoch: epoch(),
            start: enabled().then(thread_cpu_ns).flatten(),
        }
    }
}
impl Drop for CpuSpan {
    fn drop(&mut self) {
        if let Some(start) = self.start
            && enabled()
            && let Some(end) = thread_cpu_ns()
        {
            record(
                Event {
                    name: self.name,
                    start: clock_ns(),
                    duration: None,
                    id: self.id,
                    value: end.saturating_sub(start),
                },
                self.epoch,
            );
        }
    }
}
/// A process-wide capture, exported only after recording has stopped.
pub struct Capture;
impl Capture {
    pub fn start() -> io::Result<Self> {
        ORIGIN.get_or_init(Instant::now);
        OWNED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .map_err(|_| io::Error::other("a trace capture is already active"))?;
        for buffer in THREADS.lock().unwrap().iter() {
            let mut buffer = buffer.lock().unwrap();
            buffer.events.clear();
            buffer.dropped = 0;
        }
        EPOCH.fetch_add(1, Ordering::Relaxed);
        ACTIVE.store(true, Ordering::Release);
        Ok(Self)
    }
    pub fn finish(self, path: &Path) -> io::Result<()> {
        ACTIVE.store(false, Ordering::Release);
        let mut out = BufWriter::new(File::create(path)?);
        write!(out, "{{\"displayTimeUnit\":\"ms\",\"traceEvents\":[")?;
        let mut first = true;
        for (tid, buffer) in THREADS.lock().unwrap().iter().enumerate() {
            let buffer = buffer.lock().unwrap();
            if !first {
                write!(out, ",")?;
            }
            first = false;
            write!(
                out,
                "{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":{tid},\"args\":{{\"name\":"
            )?;
            write_string(&mut out, &buffer.name)?;
            write!(out, "}}}}")?;
            for e in &buffer.events {
                write!(out, ",{{\"name\":")?;
                write_string(&mut out, e.name)?;
                write!(
                    out,
                    ",\"pid\":1,\"tid\":{tid},\"ts\":{},\"ph\":\"{}\"",
                    e.start as f64 / 1000.0,
                    if e.duration.is_some() { "X" } else { "i" }
                )?;
                if let Some(d) = e.duration {
                    write!(out, ",\"dur\":{}", d as f64 / 1000.0)?;
                } else {
                    write!(out, ",\"s\":\"t\"")?;
                }
                write!(out, ",\"args\":{{\"id\":{},\"value\":{}}}}}", e.id, e.value)?;
            }
            write!(
                out,
                ",{{\"name\":\"dropped_events\",\"ph\":\"C\",\"pid\":1,\"tid\":{tid},\"ts\":0,\"args\":{{\"count\":{}}}}}",
                buffer.dropped
            )?;
        }
        write!(out, "]}}")?;
        out.flush()
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
        OWNED.store(false, Ordering::Release);
    }
}
fn write_string(out: &mut impl Write, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for c in value.chars() {
        match c {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            c if c.is_control() => write!(out, "\\u{:04x}", c as u32)?,
            c => write!(out, "{c}")?,
        }
    }
    write!(out, "\"")
}

pub fn epoch() -> u64 {
    EPOCH.load(Ordering::Acquire)
}
pub fn device_interval(epoch: u64, id: u64, segment: u64, start: Option<u64>, duration: u64) {
    if !enabled() {
        return;
    }
    match start {
        Some(start) => record(
            Event {
                name: "gpu.device_interval",
                start,
                duration: Some(duration),
                id,
                value: segment,
            },
            epoch,
        ),
        None => record(
            Event {
                name: "gpu.device_duration_ns",
                start: clock_ns(),
                duration: None,
                id,
                value: duration,
            },
            epoch,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn captures_are_exclusive_bounded_and_do_not_accept_old_spans() {
        let capture = Capture::start().unwrap();
        assert!(Capture::start().is_err());
        let stale = Span::new("old span", 0, 0);
        let stale_cpu = CpuSpan::new("old cpu", 0);
        for id in 0..MAX_EVENTS_PER_THREAD as u64 + 9 {
            event("quoted \" event", id, 1);
        }
        let path =
            std::env::temp_dir().join(format!("nixe-trace-test-{}.json", std::process::id()));
        capture.finish(&path).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let events = json["traceEvents"].as_array().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| e["name"] == "quoted \" event")
                .count(),
            MAX_EVENTS_PER_THREAD
        );
        assert!(
            events
                .iter()
                .any(|e| e["name"] == "dropped_events" && e["args"]["count"] == 9)
        );
        let next = Capture::start().unwrap();
        drop(stale);
        drop(stale_cpu);
        {
            let _cpu = CpuSpan::new("thread cpu", 7);
            std::hint::black_box((0..10_000_u64).fold(0, u64::wrapping_add));
        }
        event("next capture", 1, 2);
        next.finish(&path).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(
            !json["traceEvents"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["name"] == "old span" || e["name"] == "old cpu")
        );
        #[cfg(any(target_os = "linux", target_os = "android"))]
        assert!(
            json["traceEvents"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["name"] == "thread cpu"
                    && e["args"]["id"] == 7
                    && e["args"]["value"].as_u64().is_some_and(|ns| ns > 0))
        );
        std::fs::remove_file(path).unwrap();
    }
}
