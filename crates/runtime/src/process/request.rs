//! Accepted server requests retain their actual client until their work retires.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/kernel/k_server_session.cpp
use super::*;
use std::sync::atomic::Ordering;

pub struct KernelRequest {
    client: Arc<crate::ThreadObject>,
    generation: u64,
    source: LocationDescriptor,
}
impl KernelRequest {
    pub(crate) fn new(context: &crate::ExceptionDispatchContext<'_>) -> Self {
        context
            .process()
            .active_requests
            .fetch_add(1, Ordering::AcqRel);
        Self {
            client: context.thread().object().clone(),
            generation: context.generation(),
            source: context.source(),
        }
    }
    pub fn process(&self) -> &Arc<RunnableProcess> {
        self.client
            .owner()
            .expect("accepted request retains its client process")
    }
    pub fn thread(&self) -> &Arc<crate::ThreadObject> {
        &self.client
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn is_retired(&self) -> bool {
        self.process().generation() != self.generation
    }
    pub fn report_failure(&self, detail: impl Into<Box<str>>) {
        let process = self.process();
        let detail = format!("{}; source=[{}]", detail.into(), self.source).into_boxed_str();
        if let Some(runtime) = process.runtime.get().and_then(std::sync::Weak::upgrade) {
            runtime.fail(crate::RuntimeError::Execution {
                process: nixe_scheduler::ProcessId::new(process.process_id()),
                thread: self.client.id(),
                diagnostic: detail,
            });
        } else {
            log::error!("accepted server request failed: {detail}");
        }
    }
}
impl Drop for KernelRequest {
    fn drop(&mut self) {
        let process = self.process();
        if process.active_requests.fetch_sub(1, Ordering::AcqRel) == 1
            && self.is_retired()
            && let Some(events) = process.events.get()
        {
            events.notify_host_service(crate::ExternalEventSource::Worker);
        }
    }
}

impl RunnableProcess {
    pub(crate) fn has_active_requests(&self) -> bool {
        self.active_requests.load(Ordering::Acquire) != 0
    }
}
