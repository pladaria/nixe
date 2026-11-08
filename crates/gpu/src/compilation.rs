//! Pipeline construction workers, as the renderer's ThreadWorker. The owner
//! admits each new pipeline once; this queue has no cache or admission budget.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/common/thread_worker.h
use std::{
    sync::{Arc, Condvar, Mutex, mpsc},
    thread,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    Stopped,
    Failed(Box<str>),
    Source {
        stage: Option<crate::ShaderStage>,
        source: Option<nixe_shader_recompiler::types::ProgramOffset>,
        detail: Box<str>,
    },
}
impl From<nixe_shader_recompiler::CompileError> for CompileError {
    fn from(error: nixe_shader_recompiler::CompileError) -> Self {
        Self::Source {
            stage: error.stage.map(crate::shader_compilation::neutral_stage),
            source: error.pc,
            detail: error.to_string().into(),
        }
    }
}
impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => f.write_str("compilation workers stopped"),
            Self::Failed(e) => f.write_str(e),
            Self::Source {
                stage,
                source,
                detail,
            } => write!(f, "shader {stage:?} at PC {source:?}: {detail}"),
        }
    }
}
impl std::error::Error for CompileError {}

type Job = Box<dyn FnOnce(usize) + Send + 'static>;
#[derive(Default)]
struct Requests {
    count: Mutex<usize>,
    idle: Condvar,
}
pub struct CompilePool {
    sender: Option<mpsc::Sender<Job>>,
    threads: Vec<thread::JoinHandle<()>>,
    requests: Arc<Requests>,
}
impl CompilePool {
    pub fn new(workers: usize) -> std::io::Result<Self> {
        Self::with_name(workers, "gpu-pipeline")
    }
    pub fn with_name(workers: usize, name: &str) -> std::io::Result<Self> {
        if workers == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "pipeline worker count must be nonzero",
            ));
        }
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let requests = Arc::new(Requests::default());
        let mut pool = Self {
            sender: Some(sender),
            threads: Vec::new(),
            requests,
        };
        for index in 0..workers {
            let receiver = Arc::clone(&receiver);
            let requests = Arc::clone(&pool.requests);
            pool.threads.push(
                thread::Builder::new()
                    .name(format!("{name}-{index}"))
                    .spawn(move || {
                        loop {
                            let Ok(job) = receiver.lock().unwrap().recv() else {
                                break;
                            };
                            job(index);
                            let mut pending = requests.count.lock().unwrap();
                            *pending -= 1;
                            if *pending == 0 {
                                requests.idle.notify_all();
                            }
                        }
                    })?,
            );
        }
        Ok(pool)
    }
    pub fn submit(&self, job: impl FnOnce(usize) + Send + 'static) -> Result<(), CompileError> {
        let sender = self.sender.as_ref().ok_or(CompileError::Stopped)?;
        let mut pending = self.requests.count.lock().unwrap();
        *pending += 1;
        if sender.send(Box::new(job)).is_err() {
            *pending -= 1;
            return Err(CompileError::Stopped);
        }
        Ok(())
    }
    pub fn wait_for_requests(&self) {
        let pending = self.requests.count.lock().unwrap();
        drop(
            self.requests
                .idle
                .wait_while(pending, |count| *count != 0)
                .unwrap(),
        );
    }
}
impl Drop for CompilePool {
    fn drop(&mut self) {
        self.sender.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
