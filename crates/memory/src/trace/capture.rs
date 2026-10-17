use std::{
    cell::RefCell,
    fs::File,
    io::{self, BufWriter, Write},
    panic::Location,
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

const MAX_THREADS: usize = 64;
const EVENTS_PER_THREAD: usize = 1_048_576;
static CAPTURE: OnceLock<Capture> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);
static FORCE_INTERVAL_ONE: AtomicBool = AtomicBool::new(false);
thread_local! {
    static BUFFER: RefCell<Option<Arc<Mutex<Buffer>>>> = const { RefCell::new(None) };
}
struct Capture {
    path: PathBuf,
    started: Instant,
    detailed: bool,
    windows: Vec<(u64, u64)>,
    buffers: Mutex<Vec<Arc<Mutex<Buffer>>>>,
}
struct Buffer {
    name: String,
    events: Vec<Event>,
    dropped: u64,
}
struct Event {
    begin: u64,
    end: u64,
    name: &'static str,
    values: [u64; 3],
    caller: Option<&'static Location<'static>>,
}

/// `NIXE_EXECUTION_TRACE` selects the output. Optional comma-separated
/// `NIXE_EXECUTION_TRACE_WINDOWS` contains start:end seconds. Detailed spans
/// require `NIXE_EXECUTION_TRACE_DETAIL=1`; frame events are always recorded.
pub fn start() -> io::Result<()> {
    FORCE_INTERVAL_ONE.store(
        std::env::var_os("NIXE_TEST_SWAP_INTERVAL").is_some_and(|v| v == "1"),
        Ordering::Relaxed,
    );
    let Some(path) = std::env::var_os("NIXE_EXECUTION_TRACE") else {
        return Ok(());
    };
    let windows = std::env::var("NIXE_EXECUTION_TRACE_WINDOWS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|window| {
            let (a, b) = window
                .split_once(':')
                .ok_or_else(|| io::Error::other("invalid trace window"))?;
            let parse = |s: &str| {
                s.parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .map(|v| (v * 1e9) as u64)
                    .ok_or_else(|| io::Error::other("invalid trace time"))
            };
            let (a, b) = (parse(a)?, parse(b)?);
            if a >= b {
                return Err(io::Error::other("empty trace window"));
            }
            Ok((a, b))
        })
        .collect::<io::Result<Vec<_>>>()?;
    CAPTURE
        .set(Capture {
            path: path.into(),
            started: Instant::now(),
            windows,
            detailed: std::env::var_os("NIXE_EXECUTION_TRACE_DETAIL").is_some_and(|v| v == "1"),
            buffers: Mutex::new(Vec::new()),
        })
        .map_err(|_| io::Error::other("execution capture already started"))?;
    ACTIVE.store(true, Ordering::Release);
    Ok(())
}

/// Explicit benchmark override, unavailable in ordinary builds. Keep the
/// requested interval in the trace; this is not a guest compatibility policy.
#[inline]
pub fn presentation_interval(requested: u32) -> u32 {
    if requested == 2 && FORCE_INTERVAL_ONE.load(Ordering::Relaxed) {
        1
    } else {
        requested
    }
}

fn now(detailed: bool) -> Option<u64> {
    if !ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    let capture = CAPTURE.get()?;
    if detailed && !capture.detailed {
        return None;
    }
    let time = capture.started.elapsed().as_nanos() as u64;
    (!detailed
        || capture.windows.is_empty()
        || capture.windows.iter().any(|&(a, b)| a <= time && time < b))
    .then_some(time)
}

fn record(event: Event) {
    let capture = CAPTURE.get().unwrap();
    BUFFER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            let mut buffers = capture.buffers.lock().unwrap();
            if buffers.len() == MAX_THREADS {
                panic!("execution trace thread limit exceeded");
            }
            let buffer = Arc::new(Mutex::new(Buffer {
                name: format!(
                    "{}:{:?}",
                    std::thread::current().name().unwrap_or("unnamed"),
                    std::thread::current().id()
                ),
                events: Vec::with_capacity(EVENTS_PER_THREAD),
                dropped: 0,
            }));
            buffers.push(Arc::clone(&buffer));
            *slot = Some(buffer);
        }
        let mut buffer = slot.as_ref().unwrap().lock().unwrap();
        if buffer.events.len() == EVENTS_PER_THREAD {
            buffer.dropped += 1;
        } else {
            buffer.events.push(event);
        }
    });
}

#[inline]
pub fn event(name: &'static str, a: u64, b: u64, c: u64) {
    if let Some(time) = now(false) {
        record(Event {
            begin: time,
            end: time,
            name,
            values: [a, b, c],
            caller: None,
        });
    }
}
#[inline]
pub fn detail_event(name: &'static str, a: u64, b: u64, c: u64) {
    if let Some(time) = now(true) {
        record(Event {
            begin: time,
            end: time,
            name,
            values: [a, b, c],
            caller: None,
        });
    }
}
pub struct Span {
    event: Option<Event>,
}
#[track_caller]
#[inline]
pub fn span(name: &'static str, a: u64, b: u64, c: u64) -> Span {
    let caller = Location::caller();
    Span {
        event: now(true).map(|time| Event {
            begin: time,
            end: time,
            name,
            values: [a, b, c],
            caller: Some(caller),
        }),
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(mut event) = self.event.take() {
            event.end = CAPTURE.get().unwrap().started.elapsed().as_nanos() as u64;
            record(event);
        }
    }
}

/// Call after the runtime has joined its workers. Exporting is outside capture.
pub fn finish() -> io::Result<()> {
    ACTIVE.store(false, Ordering::Release);
    let Some(capture) = CAPTURE.get() else {
        return Ok(());
    };
    let mut output = BufWriter::new(File::create(&capture.path)?);
    writeln!(output, "thread,begin_ns,end_ns,event,a,b,c,caller")?;
    let mut dropped = 0;
    for buffer in capture.buffers.lock().unwrap().iter() {
        let buffer = buffer.lock().unwrap();
        for e in &buffer.events {
            write!(
                output,
                "{},{},{},{},{},{},{},",
                buffer.name, e.begin, e.end, e.name, e.values[0], e.values[1], e.values[2]
            )?;
            if let Some(caller) = e.caller {
                write!(output, "{}:{}", caller.file(), caller.line())?;
            }
            writeln!(output)?;
        }
        dropped += buffer.dropped;
    }
    output.flush()?;
    if dropped != 0 {
        return Err(io::Error::other(format!(
            "execution trace dropped {dropped} events"
        )));
    }
    Ok(())
}
