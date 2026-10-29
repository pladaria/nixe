//! Accepted kernel waits own their resources and cancel with the caller.
use super::*;
use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

#[derive(Debug, Default)]
pub(super) struct SvcCompletions(Mutex<VecDeque<(GuestThreadId, u64)>>);
pub(super) struct SvcWait {
    thread: GuestThreadId,
    generation: u64,
    queued: AtomicBool,
    inbox: Arc<SvcCompletions>,
    owner: Arc<nixe_runtime::RunnableProcess>,
    caller: Arc<ThreadObject>,
    process_generation: u64,
    state: Mutex<WaitState>,
}
#[derive(Default)]
struct WaitState {
    operation: Option<MemoryOperation>,
    wait: Option<Arc<nixe_scheduler::KernelWait>>,
}
impl std::fmt::Debug for SvcWait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SvcWait")
            .field("thread", &self.thread)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}
impl nixe_runtime::ThreadWait for SvcWait {
    fn cancel(&self) {
        let cancelled = {
            let mut state = self.state.lock().unwrap();
            state.wait = None;
            match &state.operation {
                Some(operation) if operation.cancel() => None,
                _ => state.operation.take(),
            }
        };
        // Release request turns, pins and unpublished handles outside the wait lock.
        drop(cancelled);
        self.enqueue();
    }
}
impl SvcWait {
    fn enqueue(&self) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            self.inbox
                .0
                .lock()
                .unwrap()
                .push_back((self.thread, self.generation));
            self.owner.notify_kernel_completion();
        }
    }
}

impl Wake for SvcWait {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.enqueue();
    }
}
type AccessContinuation = Box<
    dyn FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
            nixe_memory::AccessResultValue,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send,
>;

type KernelContinuation = Box<
    dyn FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send,
>;
enum KernelResume {
    Access {
        result: nixe_memory::AccessResultValue,
        continuation: AccessContinuation,
    },
    Operation(KernelContinuation),
}
pub(super) struct KernelCompletion {
    source: nixe_cpu::location::LocationDescriptor,
    vcpu: VirtualCpuId,
    operation: KernelResume,
}
impl KernelCompletion {
    fn resume(
        self,
        dispatcher: &mut HorizonSvcDispatcher,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match self.operation {
            KernelResume::Access {
                result,
                continuation,
            } => continuation(dispatcher, context, result),
            KernelResume::Operation(continuation) => continuation(dispatcher, context),
        }
    }
}

pub(super) enum SvcCompletion {
    Kernel(KernelCompletion),
    Return {
        code: HorizonKernelResult,
        output1: Option<u64>,
    },
    ExitProcess {
        source: nixe_cpu::location::LocationDescriptor,
        vcpu: VirtualCpuId,
    },
}
impl SvcCompletion {
    pub(super) fn kernel(
        source: nixe_cpu::location::LocationDescriptor,
        vcpu: VirtualCpuId,
        continuation: impl FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> Self {
        Self::Kernel(KernelCompletion {
            source,
            vcpu,
            operation: KernelResume::Operation(Box::new(continuation)),
        })
    }
    pub(super) fn success(output1: Option<u64>) -> Self {
        Self::Return {
            code: HorizonKernelResult::SUCCESS,
            output1,
        }
    }
}

pub(super) enum MemoryOperation {
    Access {
        access: nixe_memory::PendingAccess,
        source: nixe_cpu::location::LocationDescriptor,
        vcpu: VirtualCpuId,
        continuation: Option<AccessContinuation>,
    },
    Owned {
        future: Pin<Box<dyn Future<Output = Result<SvcCompletion, HorizonSvcFault>> + Send>>,
        lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
        wait: Option<Arc<nixe_scheduler::KernelWait>>,
    },
    Heap(nixe_runtime::memory::HeapResize),
    Write {
        access: nixe_memory::PendingAccess,
        immediate: u32,
        output1: Option<u64>,
    },
}
impl MemoryOperation {
    fn cancel(&self) -> bool {
        if let Self::Owned { lifetime, .. } = self {
            lifetime.cancel()
        } else {
            false
        }
    }
    fn poll(
        &mut self,
        wake: Option<&Arc<SvcWait>>,
    ) -> Poll<Result<SvcCompletion, HorizonSvcFault>> {
        match self {
            Self::Access {
                access,
                source,
                vcpu,
                continuation,
            } => match access.poll() {
                None => Poll::Pending,
                Some(result) => Poll::Ready(Ok(SvcCompletion::Kernel(KernelCompletion {
                    source: *source,
                    vcpu: *vcpu,
                    operation: KernelResume::Access {
                        result,
                        continuation: continuation.take().expect("kernel access resumes once"),
                    },
                }))),
            },
            Self::Owned { future, .. } => {
                let waker = wake.map(|wake| Waker::from(Arc::clone(wake)));
                future.as_mut().poll(&mut Context::from_waker(
                    waker.as_ref().unwrap_or(Waker::noop()),
                ))
            }
            Self::Heap(heap) => {
                let waker = wake.map(|wake| Waker::from(Arc::clone(wake)));
                Pin::new(heap)
                    .poll(&mut Context::from_waker(
                        waker.as_ref().unwrap_or(Waker::noop()),
                    ))
                    .map(|result| {
                        result
                            .map(|base| SvcCompletion::success(Some(base.get())))
                            .map_err(|error| HorizonSvcFault::MemoryMapping {
                                fault: Arc::new(error),
                            })
                    })
            }
            Self::Write {
                access,
                immediate,
                output1,
            } => match access.poll() {
                None => Poll::Pending,
                Some(Err(fault)) => Poll::Ready(Err(HorizonSvcFault::MemoryAccess {
                    immediate: *immediate,
                    fault,
                })),
                Some(Ok(nixe_memory::AccessResult::Written)) => {
                    Poll::Ready(Ok(SvcCompletion::success(*output1)))
                }
                Some(Ok(_)) => Poll::Ready(Err(HorizonSvcFault::MemoryAccess {
                    immediate: *immediate,
                    fault: Arc::new(nixe_memory::MemoryError::Execution(
                        "invalid owned write completion".into(),
                    )),
                })),
            },
        }
    }
}
impl HorizonSvcDispatcher {
    pub(super) fn read_svc_input(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        address: GuestVirtualAddress,
        bytes: usize,
        continuation: impl FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
            Box<[u8]>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let mut output = vec![0; bytes];
        match context
            .process()
            .memory()
            .space()
            .read_into(address, &mut output)
        {
            Ok(nixe_memory::Access::Ready(())) => {
                continuation(self, context, output.into_boxed_slice())
            }
            Err(fault) => reject(
                context,
                HorizonSvcFault::MemoryAccess {
                    immediate,
                    fault: Arc::new(fault),
                },
            ),
            Ok(nixe_memory::Access::Pending(access)) => {
                self.defer_svc_read(context, immediate, access, continuation)
            }
        }
    }

    pub(super) fn defer_svc_read(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        access: nixe_memory::PendingAccess,
        continuation: impl FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
            Box<[u8]>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        self.kernel_access(
            context,
            Ok(nixe_memory::Access::Pending(access)),
            move |dispatcher, context, result| match result {
                Ok(nixe_memory::AccessResult::Read(bytes)) => {
                    continuation(dispatcher, context, bytes)
                }
                Ok(_) => {
                    ExceptionDispatchOutcome::Fault(runtime_fault("kernel read completion kind"))
                }
                Err(fault) => reject(context, HorizonSvcFault::MemoryAccess { immediate, fault }),
            },
        )
    }

    pub(super) fn kernel_access(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        access: Result<nixe_memory::Access<nixe_memory::AccessResult>, nixe_memory::MemoryError>,
        continuation: impl FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
            nixe_memory::AccessResultValue,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let access = match access {
            Ok(nixe_memory::Access::Ready(result)) => {
                return continuation(self, context, Ok(result));
            }
            Err(error) => return continuation(self, context, Err(Arc::new(error))),
            Ok(nixe_memory::Access::Pending(access)) => access,
        };
        let source = context.source();
        let vcpu = context.thread().vcpu();
        self.start_memory_operation(
            context,
            MemoryOperation::Access {
                access,
                source,
                vcpu,
                continuation: Some(Box::new(continuation)),
            },
        )
    }

    pub(super) fn finish_wait_start(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        start: Result<nixe_scheduler::WaitStart, nixe_scheduler::SchedulerError>,
        continuation: impl FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
            nixe_scheduler::WaitOutcome,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let wait = match start {
            Ok(nixe_scheduler::WaitStart::Ready(outcome)) => {
                return continuation(self, context, outcome);
            }
            Ok(nixe_scheduler::WaitStart::Pending(wait)) => wait,
            Err(fault) => {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                    operation: "kernel object wait",
                    fault,
                });
            }
        };
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let owned = wait.clone();
        self.start_owned_svc(context, Arc::default(), Some(wait), async move {
            let outcome = owned.wait().await;
            Ok(SvcCompletion::kernel(
                source,
                vcpu,
                move |dispatcher, context| continuation(dispatcher, context, outcome),
            ))
        })
    }

    pub(super) fn start_owned_svc(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
        wait: Option<Arc<nixe_scheduler::KernelWait>>,
        operation: impl Future<Output = Result<SvcCompletion, HorizonSvcFault>> + Send + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        self.start_memory_operation(
            context,
            MemoryOperation::Owned {
                future: Box::pin(operation),
                lifetime,
                wait,
            },
        )
    }
    #[must_use]
    pub fn with_input_completions(self, events: nixe_runtime::ExternalEventSender) -> Self {
        self.hid_system.set_completion_events(events);
        self
    }
    pub(super) fn set_heap_size(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let bytes = read_register(context.thread().state(), 1);
        if !bytes.is_multiple_of(HORIZON_HEAP_ALIGNMENT)
            || bytes > HORIZON_MAX_HEAP_SIZE
            || bytes > context.process().memory_layout().heap().size()
        {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        let operation = MemoryOperation::Heap(context.process().memory().resize_heap(bytes));
        self.start_memory_operation(context, operation)
    }
    fn new_svc_wait(
        &mut self,
        owner: Arc<nixe_runtime::RunnableProcess>,
        caller: Arc<ThreadObject>,
        process_generation: u64,
    ) -> Result<Arc<SvcWait>, HorizonSvcFault> {
        let thread = caller.id();
        if self.svc_waits.contains_key(&thread) {
            return Err(runtime_fault("duplicate memory continuation"));
        }
        self.next_memory_generation = self
            .next_memory_generation
            .checked_add(1)
            .ok_or_else(|| runtime_fault("memory continuation identity exhausted"))?;
        Ok(Arc::new(SvcWait {
            thread,
            generation: self.next_memory_generation,
            queued: AtomicBool::new(false),
            inbox: Arc::clone(&self.memory_completions),
            process_generation,
            owner,
            caller,
            state: Mutex::default(),
        }))
    }
    fn start_memory_operation(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        mut operation: MemoryOperation,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Preserve the admission generation even if the first poll races exit.
        let process_generation = context.process().generation();
        // Ordinary operations finish on the caller before acquiring wait state.
        // Device accesses already returned Pending and install their observer below.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_physical_memory.cpp#L14-L30
        if matches!(
            operation,
            MemoryOperation::Owned { .. } | MemoryOperation::Heap(_)
        ) && let Poll::Ready(completed) = operation.poll(None)
        {
            if let MemoryOperation::Owned {
                wait: Some(wait), ..
            } = &operation
                && let Err(fault) = context.process().scheduler().continue_wait(wait.token())
            {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                    operation: "inline wait completion",
                    fault,
                });
            }
            return self.finish_memory_operation(context, completed);
        }
        let wake = match self.new_svc_wait(
            context.process().clone(),
            context.thread().object().clone(),
            process_generation,
        ) {
            Ok(wake) => wake,
            Err(fault) => return ExceptionDispatchOutcome::Fault(fault),
        };
        if let MemoryOperation::Access { access, .. } | MemoryOperation::Write { access, .. } =
            &mut operation
        {
            let notified = wake.clone();
            access.set_waker(Arc::new(move || notified.wake_by_ref()));
        }
        match operation.poll(Some(&wake)) {
            Poll::Ready(completed) => {
                if let MemoryOperation::Owned {
                    wait: Some(wait), ..
                } = &operation
                    && let Err(fault) = context.process().scheduler().continue_wait(wait.token())
                {
                    return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                        operation: "inline wait completion",
                        fault,
                    });
                }
                self.finish_memory_operation(context, completed)
            }
            Poll::Pending => {
                wake.state.lock().unwrap().operation = Some(operation);
                self.install_svc_wait(wake);
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Next)
            }
        }
    }
    fn finish_memory_operation(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        completed: Result<SvcCompletion, HorizonSvcFault>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match completed {
            Ok(SvcCompletion::Kernel(read)) => read.resume(self, context),
            Ok(SvcCompletion::ExitProcess { .. }) => terminate(ExceptionTerminationScope::Process),
            Ok(SvcCompletion::Return { code, output1 }) => {
                result(context, code);
                if let Some(value) = output1 {
                    write_register(context.thread_mut().state_mut(), 1, value);
                }
                resume()
            }
            Err(fault) => reject(context, fault),
        }
    }
    pub(super) fn write_svc_output(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        address: GuestVirtualAddress,
        bytes: &[u8],
        output1: Option<u64>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match context.process().memory().space().write(address, bytes) {
            Ok(access) => self.finish_svc_write(context, immediate, access, output1),
            Err(fault) => reject(
                context,
                HorizonSvcFault::MemoryAccess {
                    immediate,
                    fault: Arc::new(fault),
                },
            ),
        }
    }
    pub(super) fn finish_svc_write(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        access: nixe_memory::Access<()>,
        output1: Option<u64>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match access {
            nixe_memory::Access::Ready(()) => {
                result(context, HorizonKernelResult::SUCCESS);
                if let Some(value) = output1 {
                    write_register(context.thread_mut().state_mut(), 1, value);
                }
                resume()
            }
            nixe_memory::Access::Pending(access) => self.start_memory_operation(
                context,
                MemoryOperation::Write {
                    access,
                    immediate,
                    output1,
                },
            ),
        }
    }
    pub(super) fn query_memory(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        use nixe_runtime::memory::{MemoryInfo, MemoryKind};
        let output = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 2));
        let limit = context.process().address_space_limit();
        let info = if address.get() >= limit {
            // The terminal inaccessible region wraps base + size to zero.
            MemoryInfo {
                base: GuestVirtualAddress::new(limit),
                bytes: 0_u64.wrapping_sub(limit),
                mapping: None,
            }
        } else {
            match context.process().memory().query(address) {
                Ok(info) => info,
                Err(fault) => {
                    return reject(
                        context,
                        HorizonSvcFault::MemoryMapping {
                            fault: Arc::new(fault),
                        },
                    );
                }
            }
        };
        let kind = info.mapping.map_or(0u32, |mapping| match mapping.kind {
            MemoryKind::Data => 2,
            MemoryKind::Code => 3,
            MemoryKind::MutableCode => 4,
            MemoryKind::Heap => 5,
            MemoryKind::Shared => 6,
            MemoryKind::ModuleCode => 8,
            MemoryKind::MutableModuleCode => 9,
            MemoryKind::Stack => 0x0b,
            MemoryKind::ThreadLocal => 0x0c,
        });
        let attributes = info.mapping.map_or(0u32, |mapping| {
            u32::from(mapping.locked)
                | (u32::from(mapping.uncached) << 3)
                | (u32::from(mapping.permission_locked) << 4)
        });
        let permissions = info
            .mapping
            .map_or(0u32, |mapping| u32::from(mapping.permissions.bits()));
        let mut bytes = [0; 0x28];
        bytes[0..8].copy_from_slice(&info.base.get().to_le_bytes());
        bytes[8..16].copy_from_slice(&info.bytes.to_le_bytes());
        bytes[16..20].copy_from_slice(&kind.to_le_bytes());
        bytes[20..24].copy_from_slice(&attributes.to_le_bytes());
        bytes[24..28].copy_from_slice(&permissions.to_le_bytes());
        self.write_svc_output(context, immediate, output, &bytes, Some(0))
    }
    fn install_svc_wait(&mut self, wait: Arc<SvcWait>) {
        let kernel: Arc<dyn nixe_runtime::ThreadWait> = wait.clone();
        wait.caller.set_wait(Arc::downgrade(&kernel));
        assert!(self.svc_waits.insert(wait.thread, wait).is_none());
    }

    pub(super) fn register_memory_wait(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        thread: GuestThreadId,
    ) -> Result<bool, nixe_runtime::CoordinatorError> {
        let Some(wait) = self.svc_waits.get(&thread) else {
            return Ok(false);
        };
        let mut state = wait.state.lock().unwrap();
        let Some(operation) = &state.operation else {
            return Ok(false);
        };
        let wait = if let MemoryOperation::Owned {
            wait: Some(wait), ..
        } = operation
        {
            wait.clone()
        } else {
            coordinator.scheduler().register_wait(thread)?
        };
        wait.commit();
        state.wait = Some(wait);
        Ok(true)
    }

    /// Called by the execution owner after ingress and before dispatching guest
    /// work. Wakers enqueue identities only; they cannot touch handles/registers.
    pub fn poll_memory_completions(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
    ) -> Result<(), HorizonSvcFault> {
        let ready = std::mem::take(&mut *self.memory_completions.0.lock().unwrap());
        for (thread, generation) in ready {
            let Some(pending) = self.svc_waits.remove(&thread) else {
                continue;
            };
            if pending.generation != generation {
                self.svc_waits.insert(thread, pending);
                continue;
            }
            pending.queued.store(false, Ordering::Release);
            let mut state = pending.state.lock().unwrap();
            let current = pending.process_generation == pending.owner.generation()
                && state.wait.as_ref().is_some_and(|wait| {
                    let token = wait.token();
                    coordinator.scheduler().thread(thread).is_some_and(|view| {
                        view.process.get() == pending.owner.process_id()
                            && view.active_wait == Some(token)
                            && view.lifecycle == nixe_scheduler::ThreadLifecycle::Waiting
                    })
                });
            let completed = match state.operation.as_mut() {
                None => {
                    drop(state);
                    continue;
                }
                Some(operation) => {
                    if !current && !operation.cancel() {
                        let cancelled = state.operation.take();
                        drop(state);
                        pending.caller.clear_wait();
                        drop(cancelled);
                        continue;
                    }
                    operation.poll(Some(&pending))
                }
            };
            let completed = match completed {
                Poll::Pending => {
                    drop(state);
                    self.svc_waits.insert(thread, pending);
                    continue;
                }
                Poll::Ready(result) => result,
            };
            let operation = state.operation.take();
            let token = state.wait.as_ref().map(|wait| wait.token());
            drop(state);
            pending.caller.clear_wait();
            drop(operation);
            if !current {
                if let Err(fault) = completed
                    && fault.guest_result().is_none()
                {
                    return Err(fault);
                }
                continue;
            }
            let token = token.expect("current SVC has its wait token");
            let process = ProcessId::new(pending.owner.process_id());
            if let Ok(SvcCompletion::Kernel(read)) = completed {
                let handling = coordinator
                    .resume_supervisor_call(token, read.vcpu, read.source, |context| {
                        read.resume(self, context)
                    })
                    .map_err(|_| runtime_fault("kernel access continuation"))?
                    .ok_or_else(|| runtime_fault("kernel access continuation generation"))?;
                let handling = self
                    .finish_scheduled_call(coordinator, process, thread, handling)
                    .map_err(|_| runtime_fault("kernel access continuation scheduling"))?;
                if let nixe_runtime::ExceptionHandlingResult::Fault(fault) = handling {
                    return Err(fault);
                }
                continue;
            }
            let (code, output1) = match completed {
                Ok(SvcCompletion::Kernel(_)) => {
                    unreachable!("kernel access continuation was handled above")
                }
                Ok(SvcCompletion::Return { code, output1 }) => (code, output1),
                Ok(SvcCompletion::ExitProcess { source, vcpu }) => {
                    let handling = coordinator
                        .resume_supervisor_call(token, vcpu, source, |_| {
                            terminate(ExceptionTerminationScope::Process)
                        })
                        .map_err(|_| runtime_fault("IPC process exit"))?
                        .ok_or_else(|| runtime_fault("IPC process exit generation"))?;
                    self.finish_scheduled_call(coordinator, process, thread, handling)
                        .map_err(|_| runtime_fault("IPC process exit cleanup"))?;
                    continue;
                }
                Err(fault) => {
                    let Some(code) = fault.guest_result() else {
                        return Err(fault);
                    };
                    (code, None)
                }
            };
            {
                let mut state = pending.caller.registers();
                write_register(&mut state, 0, u64::from(code.raw()));
                if let Some(value) = output1 {
                    write_register(&mut state, 1, value);
                }
            }
            if !coordinator
                .complete_wait(token)
                .map_err(|_| runtime_fault("memory continuation wake"))?
            {
                return Err(runtime_fault(
                    "memory continuation lost its kernel generation",
                ));
            }
        }
        Ok(())
    }
}
