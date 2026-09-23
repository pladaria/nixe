//! Fixed background compiler owners. Production supplies the real seed/reshape
//! consumer; each thread owns its reusable backend scratch.

use super::{Queue, work::Work};
use crate::jit_error::Error;
use crate::lifetime::{self, Lifetime};
use cranelift_codegen::Context;
use cranelift_frontend::FunctionBuilderContext;
use nixe_cpu::decode::DecodeResult;
use std::sync::Arc;
use std::thread::JoinHandle;

pub(crate) const MAX_INSTRUCTIONS: usize = 2048;

/// Expected abandonment is neither implementation failure nor permanent seed
/// rejection. The consumer can propagate lifetime checks with `?`.
#[derive(Debug)]
pub(crate) enum CompileError {
    Cancelled,
    Deferred,
    Failed(Error),
}

impl From<lifetime::Error> for CompileError {
    fn from(error: lifetime::Error) -> Self {
        match error {
            lifetime::Error::StalePublication
            | lifetime::Error::Closed
            | lifetime::Error::Shutdown => Self::Cancelled,
            lifetime::Error::Capacity(_) => Self::Deferred,
            _ => Self::Failed(fail(error)),
        }
    }
}

impl From<Error> for CompileError {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

pub(crate) fn count(logical_cpus: usize) -> usize {
    if logical_cpus <= 2 {
        0
    } else {
        ((logical_cpus - 2) / 2).clamp(1, 4)
    }
}

/// Moved into one OS thread, never shared with another compiler or a vCPU.
/// Cranelift Context itself owns the reusable regalloc2 context.
pub(crate) struct Resources {
    pub context: Context,
    pub frontend: FunctionBuilderContext,
    pub decoded: Vec<DecodeResult>,
}

impl Resources {
    fn new() -> Self {
        Self {
            context: Context::new(),
            frontend: FunctionBuilderContext::new(),
            decoded: Vec::with_capacity(MAX_INSTRUCTIONS),
        }
    }

    fn clear(&mut self) {
        self.context.clear();
        self.decoded.clear();
        // FunctionBuilder::finalize clears frontend state while retaining its
        // allocations. Each compilation owns and finalizes its own builder.
    }
}

pub(crate) struct Workers {
    queue: Arc<Queue>,
    process: Arc<Lifetime>,
    threads: Vec<JoinHandle<Result<(), Error>>>,
}

impl Workers {
    /// All storage exists before any thread starts and the queue is exposed
    /// only after successful startup. The callback is the HCQ entry point, not
    /// an arbitrary task executor: inputs are validated background Work only.
    /// Capture shared memory/Lifetime, never the owning JitProcess or pool.
    pub(crate) fn start(
        selected: usize,
        process: Arc<Lifetime>,
        compile: impl Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync + 'static,
    ) -> Result<Option<Self>, Error> {
        Self::start_inner(
            selected,
            process,
            compile,
            #[cfg(test)]
            None,
        )
    }

    fn start_inner(
        selected: usize,
        process: Arc<Lifetime>,
        compile: impl Fn(&mut Resources, Work<'_>) -> Result<(), CompileError> + Send + Sync + 'static,
        #[cfg(test)] fail_spawn: Option<usize>,
    ) -> Result<Option<Self>, Error> {
        let Some(queue) = Queue::new(selected, &process).map_err(fail)? else {
            return Ok(None);
        };
        let resources: Vec<_> = (0..selected).map(|_| Resources::new()).collect();
        let compile = Arc::new(compile);
        let mut pool = Self {
            queue: Arc::new(queue),
            process,
            threads: Vec::with_capacity(selected),
        };
        {
            let mut state = pool.process.lock();
            state.open().map_err(fail)?;
            if state.background_queue.upgrade().is_some() {
                return Err(Error::internal("background worker pool already registered"));
            }
            state.background_queue = Arc::downgrade(&pool.queue);
        }
        for (index, mut resources) in resources.into_iter().enumerate() {
            let queue = Arc::clone(&pool.queue);
            let process = Arc::clone(&pool.process);
            let compile = Arc::clone(&compile);
            let run = move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    while let Some(job) = queue.wait().map_err(fail)? {
                        let Some(work) = process.accept_background(job).map_err(fail)? else {
                            continue;
                        };
                        match compile(&mut resources, work) {
                            Ok(()) => {}
                            Err(CompileError::Cancelled | CompileError::Deferred) => {
                                // Abandonment may leave an unfinished builder.
                                resources.frontend = FunctionBuilderContext::new();
                            }
                            Err(CompileError::Failed(error)) => return Err(error),
                        }
                        resources.clear();
                        // The completed Work has released its predecessor pins.
                        // Collect already unlinked units without initiating a
                        // new stop, including when execution has become idle.
                        process.reclaim_retired().map_err(fail)?;
                    }
                    Ok(())
                }))
                .unwrap_or_else(|payload| {
                    let detail = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("non-string panic payload");
                    Err(Error::internal(format!(
                        "HCQ worker {index} panicked: {detail}"
                    )))
                });
                if let Err(error) = &result {
                    process.background_failed(error.clone());
                }
                result
            };
            #[cfg(test)]
            let thread = if fail_spawn == Some(index) {
                Err(std::io::Error::other(
                    "injected HCQ thread creation failure",
                ))
            } else {
                std::thread::Builder::new()
                    .name(format!("nixe-hcq-{index}"))
                    .spawn(run)
            };
            #[cfg(not(test))]
            let thread = std::thread::Builder::new()
                .name(format!("nixe-hcq-{index}"))
                .spawn(run);
            match thread {
                Ok(thread) => pool.threads.push(thread),
                Err(error) => {
                    // Drop closes/wakes/joins the already-created owners. No
                    // queue, registry or memory lock spans join.
                    return Err(Error::internal(format!(
                        "start HCQ worker {index}: {error}"
                    )));
                }
            }
        }
        // A terminal request during startup has already closed the registered
        // queue. Do not hand an unusable pool back as successful startup.
        pool.process.lock().open().map_err(fail)?;
        Ok(Some(pool))
    }

    pub(crate) fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Close admission before draining and joining every worker, even if one
    /// failed. Call outside all JIT/memory locks, after dependent GPU teardown.
    pub(crate) fn shutdown(&mut self) -> Result<(), Error> {
        let mut failure = self.queue.close().map(drop).map_err(fail).err();
        for thread in self.threads.drain(..) {
            let result = thread.join().unwrap_or_else(|_| {
                Err(Error::internal("HCQ worker panicked outside compile loop"))
            });
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        }
        // Thread join order is not failure order. A peer may only have seen
        // terminal admission, so retain the originating worker's diagnostic.
        if let Some(error) = &failure {
            self.process.background_failed(error.clone());
        }
        failure = self.process.background_failure().or(failure);
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        // shutdown records failures in Lifetime before returning them. Drop
        // cannot return an error, but neither detaches threads nor loses it.
        let _ = self.shutdown();
    }
}

impl Lifetime {
    pub(crate) fn background_failed(&self, error: Error) {
        {
            let mut state = self.lock();
            // Do not replace a prior lifecycle failure or a different worker's
            // root cause with a secondary cancellation/poison diagnostic.
            if state.failure.is_none() {
                state.background_failure = Some(error);
                self.fail(&mut state, lifetime::Error::BackgroundWorker);
            }
        }
        // Publication above precedes the existing native pending notification;
        // queue draining/wakeup acquires no JIT-state lock at the same time.
        let _ = self.close_background();
    }

    pub(crate) fn background_failure(&self) -> Option<Error> {
        self.lock().background_failure.clone()
    }

    pub(crate) fn diagnostic(&self, error: lifetime::Error) -> Error {
        if error == lifetime::Error::BackgroundWorker {
            self.background_failure()
                .expect("background failure has an owned diagnostic")
        } else {
            Error::internal(error.to_string())
        }
    }
}

fn fail(error: lifetime::Error) -> Error {
    Error::internal(format!("HCQ worker: {error}"))
}

#[cfg(test)]
mod tests;
