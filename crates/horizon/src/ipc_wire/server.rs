//! Owned sessions and ready-session servers. A deferred request releases its worker.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/server_manager.cpp#L350-L410
use super::{IpcWireError, SyncRequestResult, operation::CallLifetime};
use crate::HorizonIpcObject;
use std::{
    collections::{HashMap, VecDeque},
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex, OnceLock, Weak},
    task::{Context, Poll, Wake, Waker},
    thread::JoinHandle,
};

type Reply = Result<SyncRequestResult, IpcWireError>;
type Operation = Pin<Box<dyn Future<Output = Reply> + Send>>;

#[derive(Clone, Copy)]
pub(crate) enum Domain {
    Graphics,
    Audio,
    Files,
    Network,
    System,
}
impl Domain {
    fn for_handler(handler: &HorizonIpcObject) -> Self {
        use HorizonIpcObject::*;
        match handler {
            Vi(_) | NvDrv(_) | NvDrv2(_) | Gsv(_) => Self::Graphics,
            AudioOutManager(_)
            | AudioOut(_)
            | AudioRendererManager(_)
            | AudioRenderer(_)
            | AudioDevice(_)
            | OpusManager(_)
            | OpusDecoder(_) => Self::Audio,
            SemanticService(_) | SemanticObject(_) | SharedFont(_) | PlayReport(_) => Self::Files,
            Bsd(_)
            | Ssl(_)
            | NetworkInterface(_)
            | NetworkGeneralService(_)
            | NetworkRequest(_) => Self::Network,
            _ => Self::System,
        }
    }
}

struct CallState {
    operation: Option<Operation>,
    result: Option<Reply>,
    wake: Option<Waker>,
    queued: Option<nixe_memory::trace::Span>,
    pending: Option<nixe_memory::trace::Span>,
}
struct Call {
    id: u64,
    lifetime: Arc<CallLifetime>,
    state: Mutex<CallState>,
}
impl Call {
    fn cleanup(mut result: Reply, operation: Option<Operation>) -> Reply {
        // Destruction is part of completing owned work, not an unobserved
        // worker epilogue. No future or user resource is dropped under a lock.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(operation))).is_err() {
            let retained_failure = result.as_ref().err().is_some_and(|error| {
                super::HorizonIpcFault::from_wire(error.clone())
                    .guest_result()
                    .is_none()
            });
            if !retained_failure {
                result = Err(IpcWireError::Internal("service operation cleanup panicked"));
            }
        }
        result
    }
    fn finish(&self, result: Reply) -> Option<Waker> {
        if let Err(error) = &result {
            self.lifetime.report_failure(error);
        }
        let mut state = self.state.lock().unwrap();
        assert!(state.result.is_none(), "one terminal service result");
        state.result = Some(result);
        state.wake.take()
    }
}
struct Node {
    call: Arc<Call>,
    previous: Option<u64>,
    next: Option<u64>,
}
#[derive(Default)]
struct SessionState {
    next: u64,
    calls: HashMap<u64, Node>,
    head: Option<u64>,
    tail: Option<u64>,
    active: Option<Arc<Call>>,
    running: bool,
    queued: bool,
    notified: bool,
    closed: Option<IpcWireError>,
}
impl SessionState {
    fn close(&mut self, error: IpcWireError) -> HashMap<u64, Node> {
        self.closed.get_or_insert(error);
        self.head = None;
        self.tail = None;
        std::mem::take(&mut self.calls)
    }
    fn remove(&mut self, id: u64) -> Option<Arc<Call>> {
        let node = self.calls.remove(&id)?;
        if let Some(previous) = node.previous {
            self.calls.get_mut(&previous).unwrap().next = node.next;
        } else {
            self.head = node.next;
        }
        if let Some(next) = node.next {
            self.calls.get_mut(&next).unwrap().previous = node.previous;
        } else {
            self.tail = node.previous;
        }
        Some(node.call)
    }
}

/// A guest service connection owns its handler and all queued/active requests.
/// Copy handles retain this object; CMIF cloning creates another connection.
pub struct ServiceSession {
    command: Mutex<Option<Vec<u8>>>,
    budget: OnceLock<(u64, Arc<super::payload::PayloadBudget>)>,
    handler: HorizonIpcObject,
    queue: OnceLock<Weak<ReadyQueue>>,
    state: Mutex<SessionState>,
}
impl std::fmt::Debug for ServiceSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceSession")
            .field("handler", &self.handler)
            .finish_non_exhaustive()
    }
}
impl ServiceSession {
    pub fn new(handler: HorizonIpcObject) -> Self {
        Self {
            command: Mutex::default(),
            budget: OnceLock::new(),
            handler,
            queue: OnceLock::new(),
            state: Mutex::default(),
        }
    }
    pub(crate) fn payload_budget(
        &self,
        budgets: &super::payload::PayloadBudgets,
        process: u64,
    ) -> Arc<super::payload::PayloadBudget> {
        let (owner, budget) = self.budget.get_or_init(|| (process, budgets.get(process)));
        if *owner == process {
            budget.clone()
        } else {
            budgets.get(process)
        }
    }
    pub(crate) fn take_command(&self) -> Vec<u8> {
        self.command.lock().unwrap().take().unwrap_or_default()
    }
    pub(crate) fn recycle_command(&self, mut command: Vec<u8>) {
        command.clear();
        *self.command.lock().unwrap() = Some(command);
    }
    pub fn object(&self) -> &HorizonIpcObject {
        &self.handler
    }
    fn schedule(self: &Arc<Self>, state: &mut SessionState) {
        if state.closed.is_some() && state.active.is_none() {
            return;
        }
        state.notified = true;
        if !state.running
            && !std::mem::replace(&mut state.queued, true)
            && let Some(queue) = self.queue.get().and_then(Weak::upgrade)
        {
            queue.push(self.clone());
        }
    }
    pub(crate) fn reserve(
        self: &Arc<Self>,
        queue: &Arc<ReadyQueue>,
        lifetime: Arc<CallLifetime>,
    ) -> Result<Response, IpcWireError> {
        let binding = self.queue.get_or_init(|| Arc::downgrade(queue));
        if binding.as_ptr() != Arc::as_ptr(queue) {
            return Err(IpcWireError::Kernel(
                crate::HorizonKernelResult::SESSION_CLOSED,
            ));
        }
        let mut state = self.state.lock().unwrap();
        if state.closed.is_some() {
            return Err(IpcWireError::Kernel(
                crate::HorizonKernelResult::SESSION_CLOSED,
            ));
        }
        state.next = state
            .next
            .checked_add(1)
            .ok_or(IpcWireError::Internal("service request identity exhausted"))?;
        lifetime.delegate()?;
        let call = Arc::new(Call {
            id: state.next,
            lifetime,
            state: Mutex::new(CallState {
                operation: None,
                result: None,
                wake: None,
                queued: Some(nixe_memory::trace::span(
                    "ipc_service_queue",
                    Arc::as_ptr(self) as u64,
                    state.next,
                    0,
                )),
                pending: None,
            }),
        });
        let previous = state.tail;
        if let Some(previous) = previous {
            state.calls.get_mut(&previous).unwrap().next = Some(call.id);
        } else {
            state.head = Some(call.id);
        }
        state.tail = Some(call.id);
        state.calls.insert(
            call.id,
            Node {
                call: call.clone(),
                previous,
                next: None,
            },
        );
        Ok(Response {
            session: self.clone(),
            call,
            completed: false,
        })
    }
    fn cancel(self: &Arc<Self>, call: &Arc<Call>) {
        call.lifetime.cancel();
        let retired = {
            let mut state = self.state.lock().unwrap();
            if let Some(retired) = state.remove(call.id) {
                if state.active.is_none() && state.head.is_some() {
                    self.schedule(&mut state);
                }
                Some(retired)
            } else {
                if state
                    .active
                    .as_ref()
                    .is_some_and(|active| active.id == call.id)
                {
                    self.schedule(&mut state);
                }
                None
            }
        };
        if let Some(retired) = retired {
            let operation = retired.state.lock().unwrap().operation.take();
            nixe_memory::trace::event("ipc_reply_ready", Arc::as_ptr(self) as u64, retired.id, 1);
            let result = Call::cleanup(Err(IpcWireError::Cancelled), operation);
            if let Err(error) = &result
                && super::HorizonIpcFault::from_wire(error.clone())
                    .guest_result()
                    .is_none()
            {
                self.close(error.clone());
            }
            if let Some(wake) = retired.finish(result) {
                wake.wake();
            }
        }
    }
    /// Retire queued work without stealing an executing effect from its worker.
    /// A closed session still schedules its active cancellation/drain to completion.
    fn close(self: &Arc<Self>, error: IpcWireError) {
        let (calls, error) = {
            let mut state = self.state.lock().unwrap();
            let calls = state.close(error);
            let error = state.closed.as_ref().unwrap().clone();
            if let Some(active) = &state.active {
                active.lifetime.cancel();
                self.schedule(&mut state);
            }
            (calls, error)
        };
        Self::finish_queued(calls, error);
    }
    fn finish_queued(calls: HashMap<u64, Node>, error: IpcWireError) {
        for (_, node) in calls {
            let operation = node.call.state.lock().unwrap().operation.take();
            let result = Call::cleanup(Err(error.clone()), operation);
            if let Some(wake) = node.call.finish(result) {
                wake.wake();
            }
        }
    }
    fn poll(self: &Arc<Self>) {
        let mut current = None;
        // A failed server turn must complete its callers, including failures
        // outside Future::poll. Eden treats failed server processing as fatal.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/server_manager.cpp#L317-L324
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.poll_request(&mut current);
        }))
        .is_err()
        {
            self.fail(current);
        }
    }
    fn fail(self: &Arc<Self>, current: Option<Arc<Call>>) {
        let mut calls = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = Some(IpcWireError::Internal("service server processing panicked"));
            state.running = false;
            state.notified = false;
            state.head = None;
            state.tail = None;
            let mut calls: Vec<_> = state.calls.drain().map(|(_, node)| node.call).collect();
            if let Some(active) = state.active.take() {
                calls.push(active);
            }
            calls
        };
        if let Some(current) = current
            && !calls.iter().any(|call| Arc::ptr_eq(call, &current))
        {
            calls.push(current);
        }
        for call in calls {
            let (operation, wake) = {
                let mut state = call
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state.result.is_none() {
                    state.result = Some(Err(IpcWireError::Internal(
                        "service server processing panicked",
                    )));
                }
                (state.operation.take(), state.wake.take())
            };
            // Cleanup failures cannot hide the original failure or prevent
            // the other accepted callers from receiving a terminal result.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(operation)));
            if let Some(wake) = wake {
                wake.wake();
            }
        }
    }
    fn poll_request(self: &Arc<Self>, current: &mut Option<Arc<Call>>) {
        let call = {
            let mut state = self.state.lock().unwrap();
            state.queued = false;
            if state.running {
                state.notified = true;
                return;
            }
            if state.active.is_none()
                && let Some(head) = state.head
            {
                state.active = state.remove(head);
            }
            let Some(call) = state.active.clone() else {
                return;
            };
            state.running = true;
            state.notified = false;
            call
        };
        let call = current.insert(call);
        let mut operation = {
            let mut state = call.state.lock().unwrap();
            state.pending.take();
            if state.operation.is_some() {
                state.queued.take();
            }
            state.operation.take()
        };
        let waker = Waker::from(self.clone());
        let _poll =
            nixe_memory::trace::span("ipc_service_poll", Arc::as_ptr(self) as u64, call.id, 0);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match operation.as_mut() {
                Some(operation) => operation.as_mut().poll(&mut Context::from_waker(&waker)),
                None => Poll::Pending,
            }));
        let result = match result {
            Ok(Poll::Pending)
                if call.lifetime.check().is_ok() || call.lifetime.must_drain_effect() =>
            {
                None
            }
            Ok(Poll::Pending) => Some(Err(IpcWireError::Cancelled)),
            Ok(Poll::Ready(result)) => Some(result),
            Err(_) => Some(Err(IpcWireError::Internal("service handler panicked"))),
        };
        let Some(result) = result else {
            let mut state = self.state.lock().unwrap();
            state.running = false;
            if operation.is_some() {
                let mut call_state = call.state.lock().unwrap();
                call_state.operation = operation;
                call_state.pending = Some(nixe_memory::trace::span(
                    "ipc_service_wait",
                    Arc::as_ptr(self) as u64,
                    call.id,
                    0,
                ));
            }
            if state.notified {
                self.schedule(&mut state);
            }
            return;
        };
        // Keep the session active/running through destruction. Another file
        // worker must not consume the next call while this call still owns work.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/server_manager.cpp#L378-L408
        let mut result = Call::cleanup(result, operation);
        let (wake, retired, closed) = {
            let mut state = self.state.lock().unwrap();
            let retired = match &result {
                Ok(SyncRequestResult::SessionClosed) => {
                    let retired = state.close(IpcWireError::Kernel(
                        crate::HorizonKernelResult::SESSION_CLOSED,
                    ));
                    result = Ok(SyncRequestResult::Success);
                    retired
                }
                Err(error)
                    if super::HorizonIpcFault::from_wire(error.clone())
                        .guest_result()
                        .is_none() =>
                {
                    state.close(error.clone())
                }
                _ => HashMap::new(),
            };
            // A failure from cancelled queued cleanup also belongs to an active
            // call which must drain. Preserve its own backend error if it has one.
            if let Some(error) = &state.closed
                && super::HorizonIpcFault::from_wire(error.clone())
                    .guest_result()
                    .is_none()
                && !result.as_ref().err().is_some_and(|error| {
                    super::HorizonIpcFault::from_wire(error.clone())
                        .guest_result()
                        .is_none()
                })
            {
                result = Err(error.clone());
            }
            nixe_memory::trace::event(
                "ipc_reply_ready",
                Arc::as_ptr(self) as u64,
                call.id,
                u64::from(result.is_err()),
            );
            // Publish before re-admission while holding the session's one
            // state lock; wake the caller only after releasing both locks.
            let wake = call.finish(result);
            state.active = None;
            state.running = false;
            if state.head.is_some() {
                self.schedule(&mut state);
            }
            (wake, retired, state.closed.clone())
        };
        current.take();
        if let Some(wake) = wake {
            wake.wake();
        }
        if let Some(error) = closed {
            Self::finish_queued(retired, error);
        }
    }
}
impl Wake for ServiceSession {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule(&mut self.state.lock().unwrap());
    }
}

pub(crate) struct Response {
    session: Arc<ServiceSession>,
    call: Arc<Call>,
    completed: bool,
}
impl Response {
    pub(crate) fn start(
        &self,
        operation: impl Future<Output = Reply> + Send + 'static,
    ) -> Result<(), IpcWireError> {
        self.call.lifetime.check()?;
        let mut state = self.session.state.lock().unwrap();
        let mut call = self.call.state.lock().unwrap();
        if call.result.is_some() {
            return Ok(());
        }
        assert!(call.operation.is_none(), "one accepted service operation");
        call.operation = Some(Box::pin(operation));
        drop(call);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.id == self.call.id)
            || (state.active.is_none() && state.head == Some(self.call.id))
        {
            self.session.schedule(&mut state);
        }
        Ok(())
    }
}
impl Future for Response {
    type Output = Reply;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Reply> {
        let this = self.get_mut();
        if this.call.lifetime.check().is_err() {
            this.session.cancel(&this.call);
        }
        let mut state = this.call.state.lock().unwrap();
        if let Some(result) = state.result.take() {
            this.completed = true;
            nixe_memory::trace::event(
                "ipc_caller_reply",
                Arc::as_ptr(&this.session) as u64,
                this.call.id,
                0,
            );
            this.call.lifetime.consume_reply();
            return Poll::Ready(result);
        }
        if state
            .wake
            .as_ref()
            .is_none_or(|old| !old.will_wake(context.waker()))
        {
            state.wake = Some(context.waker().clone());
        }
        Poll::Pending
    }
}
impl Drop for Response {
    fn drop(&mut self) {
        if !self.completed {
            self.session.cancel(&self.call);
            let error = self
                .call
                .state
                .lock()
                .unwrap()
                .result
                .as_ref()
                .and_then(|result| result.as_ref().err())
                .cloned();
            if let Some(error) = error {
                self.call.lifetime.report_failure(&error);
            }
        }
    }
}

#[derive(Default)]
struct Ready {
    sessions: VecDeque<Arc<ServiceSession>>,
    stopped: bool,
}
#[derive(Default)]
pub(crate) struct ReadyQueue {
    state: Mutex<Ready>,
    changed: Condvar,
}
impl ReadyQueue {
    fn push(&self, session: Arc<ServiceSession>) {
        self.state.lock().unwrap().sessions.push_back(session);
        self.changed.notify_one();
    }
    fn run(&self) {
        loop {
            let session = {
                let mut ready = self
                    .changed
                    .wait_while(self.state.lock().unwrap(), |ready| {
                        ready.sessions.is_empty() && !ready.stopped
                    })
                    .unwrap();
                let Some(session) = ready.sessions.pop_front() else {
                    return;
                };
                session
            };
            session.poll();
        }
    }
}
pub(crate) struct Servers {
    queues: [Arc<ReadyQueue>; 5],
    workers: Vec<JoinHandle<()>>,
}
impl std::fmt::Debug for Servers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceServers")
            .field("workers", &self.workers.len())
            .finish()
    }
}
impl Servers {
    pub fn queue(&self, session: &ServiceSession) -> &Arc<ReadyQueue> {
        &self.queues[Domain::for_handler(&session.handler) as usize]
    }
    pub fn start() -> std::io::Result<Self> {
        let mut servers = Self {
            queues: std::array::from_fn(|_| Arc::default()),
            workers: Vec::new(),
        };
        for (index, (name, count)) in [
            ("graphics", 1),
            ("audio", 1),
            ("files", 2),
            ("network", 2),
            ("system", 1),
        ]
        .into_iter()
        .enumerate()
        {
            for number in 0..count {
                let queue = servers.queues[index].clone();
                let worker = std::thread::Builder::new()
                    .name(format!("nixe-service-{name}-{number}"))
                    .spawn(move || queue.run())?;
                servers.workers.push(worker);
            }
        }
        Ok(servers)
    }
}
impl Drop for Servers {
    fn drop(&mut self) {
        for queue in &self.queues {
            queue.state.lock().unwrap().stopped = true;
            queue.changed.notify_all();
        }
        for worker in self.workers.drain(..) {
            assert_ne!(
                worker.thread().id(),
                std::thread::current().id(),
                "service owner joins from outside its workers"
            );
            if worker.join().is_err() {
                log::error!("service worker panicked during shutdown");
            }
        }
    }
}
