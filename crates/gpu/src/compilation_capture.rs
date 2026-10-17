//! Optional cold-cache diagnostics. No state, timers or IO in ordinary builds.
use rustc_hash::FxHasher;
use std::{
    fs::{File, OpenOptions},
    hash::{Hash, Hasher},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::Instant,
};

struct Capture {
    started: Instant,
    output: Mutex<BufWriter<File>>,
    modules: PathBuf,
}
static CAPTURE: OnceLock<Option<Capture>> = OnceLock::new();
fn capture() -> Option<&'static Capture> {
    CAPTURE
        .get_or_init(|| {
            let path = PathBuf::from(std::env::var_os("NIXE_COMPILATION_CAPTURE")?);
            let modules = path.with_extension("modules");
            std::fs::create_dir_all(&modules).expect("create compilation module capture directory");
            let mut output =
                BufWriter::new(File::create(path).expect("create compilation capture"));
            writeln!(output, "seconds,thread,cache,key,event,worker,value,detail").unwrap();
            let unix_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            writeln!(output, "0,init,Capture,0,origin,0,{unix_ns},\"\"").unwrap();
            Some(Capture {
                started: Instant::now(),
                output: Mutex::new(output),
                modules,
            })
        })
        .as_ref()
}

#[derive(Clone, Copy)]
pub struct Key {
    kind: &'static str,
    hash: u64,
}
impl Key {
    pub fn new<T: Hash>(value: &T) -> Self {
        let mut h = FxHasher::default();
        value.hash(&mut h);
        Self {
            kind: std::any::type_name::<T>(),
            hash: h.finish(),
        }
    }
    pub fn event(self, event: &str, worker: usize, value: u64) {
        self.detail(event, worker, value, "");
    }
    pub fn detail(self, event: &str, worker: usize, value: u64, detail: &str) {
        let Some(capture) = capture() else {
            return;
        };
        let seconds = capture.started.elapsed().as_secs_f64();
        let thread = std::thread::current();
        let mut output = capture.output.lock().unwrap();
        writeln!(
            output,
            "{seconds:.9},{},{},{:016x},{event},{worker},{value},\"{}\"",
            thread.name().unwrap_or("unnamed"),
            self.kind.rsplit("::").next().unwrap(),
            self.hash,
            detail.replace('"', "\"\"")
        )
        .expect("write compilation capture");
    }
}
pub fn enabled() -> bool {
    capture().is_some()
}
pub fn save_module(identity: &str, words: &[u32]) {
    let Some(capture) = capture() else {
        return;
    };
    let path = capture.modules.join(format!("{identity}.spv"));
    let mut file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => BufWriter::new(file),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return,
        Err(e) => panic!("create compilation module capture: {e}"),
    };
    for word in words {
        file.write_all(&word.to_le_bytes())
            .expect("write SPIR-V capture");
    }
    file.flush().expect("flush SPIR-V capture");
}
pub fn finish() -> std::io::Result<()> {
    if let Some(capture) = CAPTURE.get().and_then(Option::as_ref) {
        capture.output.lock().unwrap().flush()?;
    }
    Ok(())
}
