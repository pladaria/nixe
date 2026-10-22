//! Memory-dependent SVCs leave an owned operation and resume after the SVC.
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
pub(super) struct MemoryCompletions(Mutex<VecDeque<(GuestThreadId, u64)>>);
struct MemoryWake {
    thread: GuestThreadId,
    generation: u64,
    queued: AtomicBool,
    inbox: Arc<MemoryCompletions>,
    events: Option<nixe_runtime::ExternalEventSender>,
}
impl Wake for MemoryWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            self.inbox
                .0
                .lock()
                .unwrap()
                .push_back((self.thread, self.generation));
            if let Some(events) = &self.events {
                events.notify_host_service(nixe_runtime::ExternalEventSource::Device);
            }
        }
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
    mutex_updates: Option<(ProcessId, Vec<u64>)>,
}
impl KernelCompletion {
    fn resume(
        self,
        dispatcher: &mut HorizonSvcDispatcher,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        if let Some((_, addresses)) = self.mutex_updates
            && let Err(fault) = dispatcher.synchronize_mutex_waits(context, addresses)
        {
            return ExceptionDispatchOutcome::Fault(fault);
        }
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
            mutex_updates: None,
        })
    }
    pub(super) fn mutex_effects(mut self, process: ProcessId, addresses: Vec<u64>) -> Self {
        let Self::Kernel(completion) = &mut self else {
            unreachable!("kernel metadata accompanies its continuation");
        };
        completion.mutex_updates = Some((process, addresses));
        self
    }
    pub(super) fn success(output1: Option<u64>) -> Self {
        Self::Return {
            code: HorizonKernelResult::SUCCESS,
            output1,
        }
    }
}

enum MemoryOperation {
    Access {
        access: nixe_memory::PendingAccess,
        source: nixe_cpu::location::LocationDescriptor,
        vcpu: VirtualCpuId,
        continuation: Option<AccessContinuation>,
    },
    Owned {
        future: Pin<Box<dyn Future<Output = Result<SvcCompletion, HorizonSvcFault>> + Send>>,
        lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
        deadline: Option<u64>,
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
            lifetime.cancel();
            lifetime.started()
        } else {
            false
        }
    }
    fn poll(&mut self, wake: &Arc<MemoryWake>) -> Poll<Result<SvcCompletion, HorizonSvcFault>> {
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
                    mutex_updates: None,
                }))),
            },
            Self::Owned { future, .. } => {
                let waker = Waker::from(Arc::clone(wake));
                future.as_mut().poll(&mut Context::from_waker(&waker))
            }
            Self::Heap(heap) => {
                let waker = Waker::from(Arc::clone(wake));
                Pin::new(heap)
                    .poll(&mut Context::from_waker(&waker))
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
pub(super) struct PendingMemory {
    process: ProcessId,
    generation: u64,
    wake: Arc<MemoryWake>,
    token: Option<nixe_scheduler::WakeToken>,
    operation: MemoryOperation,
}
impl std::fmt::Debug for PendingMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingMemory")
            .field("process", &self.process)
            .field("generation", &self.generation)
            .field("token", &self.token)
            .finish_non_exhaustive()
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

    pub(super) fn read_svc_capture(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        ranges: Vec<nixe_memory::AllocationRange>,
        continuation: impl FnOnce(
            &mut HorizonSvcDispatcher,
            &mut ExceptionDispatchContext<'_>,
            Box<[u8]>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let count = ranges.iter().map(|range| range.bytes() as usize).sum();
        let mut bytes = vec![0; count];
        let machine = context.process().memory().account().machine();
        match nixe_memory::CapturedMemory::new(ranges)
            .and_then(|ranges| machine.read_captured_into(&ranges, 0, &mut bytes))
        {
            Ok(nixe_memory::Access::Ready(())) => {
                continuation(self, context, bytes.into_boxed_slice())
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
        let mut access = match access {
            Ok(nixe_memory::Access::Ready(result)) => {
                return continuation(self, context, Ok(result));
            }
            Err(error) => return continuation(self, context, Err(Arc::new(error))),
            Ok(nixe_memory::Access::Pending(access)) => access,
        };
        let wake = match self.memory_wake(context.thread().id()) {
            Ok(wake) => wake,
            Err(fault) => return ExceptionDispatchOutcome::Fault(fault),
        };
        let notified = wake.clone();
        access.set_waker(Arc::new(move || notified.wake_by_ref()));
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
            wake,
        )
    }

    pub(super) fn start_owned_svc(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        lifetime: Arc<crate::ipc_wire::operation::CallLifetime>,
        deadline: Option<u64>,
        operation: impl Future<Output = Result<SvcCompletion, HorizonSvcFault>> + Send + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let wake = match self.memory_wake(context.thread().id()) {
            Ok(wake) => wake,
            Err(fault) => return ExceptionDispatchOutcome::Fault(fault),
        };
        self.start_memory_operation(
            context,
            MemoryOperation::Owned {
                future: Box::pin(operation),
                lifetime,
                deadline,
            },
            wake,
        )
    }
    #[must_use]
    pub fn with_memory_completions(mut self, events: nixe_runtime::ExternalEventSender) -> Self {
        self.hid_system.set_completion_events(events.clone());
        self.memory_events = Some(events);
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
        let wake = match self.memory_wake(context.thread().id()) {
            Ok(wake) => wake,
            Err(fault) => return ExceptionDispatchOutcome::Fault(fault),
        };
        let operation = MemoryOperation::Heap(context.process().memory().resize_heap(bytes));
        self.start_memory_operation(context, operation, wake)
    }
    fn memory_wake(&mut self, thread: GuestThreadId) -> Result<Arc<MemoryWake>, HorizonSvcFault> {
        if self.memory_waits.contains_key(&thread) {
            return Err(runtime_fault("duplicate memory continuation"));
        }
        self.next_memory_generation = self
            .next_memory_generation
            .checked_add(1)
            .ok_or_else(|| runtime_fault("memory continuation identity exhausted"))?;
        Ok(Arc::new(MemoryWake {
            thread,
            generation: self.next_memory_generation,
            queued: AtomicBool::new(false),
            inbox: Arc::clone(&self.memory_completions),
            events: self.memory_events.clone(),
        }))
    }
    fn start_memory_operation(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        mut operation: MemoryOperation,
        wake: Arc<MemoryWake>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match operation.poll(&wake) {
            Poll::Ready(Ok(SvcCompletion::Kernel(read))) => read.resume(self, context),
            Poll::Ready(Ok(SvcCompletion::ExitProcess { .. })) => {
                terminate(ExceptionTerminationScope::Process)
            }
            Poll::Ready(Ok(SvcCompletion::Return { code, output1 })) => {
                result(context, code);
                if let Some(value) = output1 {
                    write_register(context.thread_mut().state_mut(), 1, value);
                }
                resume()
            }
            Poll::Ready(Err(fault)) => reject(context, fault),
            Poll::Pending => {
                if self.memory_events.is_none() {
                    return ExceptionDispatchOutcome::Fault(runtime_fault(
                        "memory completion ingress is not installed",
                    ));
                }
                self.memory_waits.insert(
                    context.thread().id(),
                    PendingMemory {
                        process: ProcessId::new(context.process().process_id()),
                        generation: wake.generation,
                        wake,
                        token: None,
                        operation,
                    },
                );
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Next)
            }
        }
    }
    pub(super) fn write_runtime_output(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        process: ProcessId,
        thread: GuestThreadId,
        immediate: u32,
        ranges: Vec<nixe_memory::AllocationRange>,
        bytes: &[u8],
    ) -> Result<(), HorizonSvcFault> {
        let machine = coordinator
            .process(process)
            .ok_or_else(|| runtime_fault("kernel output process"))?
            .memory()
            .account()
            .machine()
            .clone();
        match nixe_memory::CapturedMemory::new(ranges)
            .and_then(|ranges| machine.write_captured(&ranges, 0, bytes))
        {
            Ok(nixe_memory::Access::Ready(())) => finish_pending_caller(
                coordinator,
                process,
                thread,
                "kernel memory output",
                HorizonKernelResult::SUCCESS,
                |_| {},
            ),
            Err(fault) => {
                let fault = HorizonSvcFault::MemoryAccess {
                    immediate,
                    fault: Arc::new(fault),
                };
                if let Some(code) = fault.guest_result() {
                    finish_pending_caller(
                        coordinator,
                        process,
                        thread,
                        "kernel memory output",
                        code,
                        |_| {},
                    )
                } else {
                    Err(fault)
                }
            }
            Ok(nixe_memory::Access::Pending(mut access)) => {
                let wake = self.memory_wake(thread)?;
                let notified = wake.clone();
                access.set_waker(Arc::new(move || notified.wake_by_ref()));
                let token = coordinator
                    .register_wait(thread, nixe_scheduler::Readiness::Pending)
                    .map_err(|_| runtime_fault("kernel output wait registration"))?;
                self.memory_waits.insert(
                    thread,
                    PendingMemory {
                        process,
                        generation: wake.generation,
                        wake,
                        token,
                        operation: MemoryOperation::Write {
                            access,
                            immediate,
                            output1: None,
                        },
                    },
                );
                Ok(())
            }
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
            Ok(nixe_memory::Access::Ready(())) => {
                result(context, HorizonKernelResult::SUCCESS);
                if let Some(value) = output1 {
                    write_register(context.thread_mut().state_mut(), 1, value);
                }
                resume()
            }
            Err(fault) => reject(
                context,
                HorizonSvcFault::MemoryAccess {
                    immediate,
                    fault: Arc::new(fault),
                },
            ),
            Ok(nixe_memory::Access::Pending(mut access)) => {
                let wake = match self.memory_wake(context.thread().id()) {
                    Ok(wake) => wake,
                    Err(fault) => return ExceptionDispatchOutcome::Fault(fault),
                };
                let notified = Arc::clone(&wake);
                access.set_waker(Arc::new(move || notified.wake_by_ref()));
                self.start_memory_operation(
                    context,
                    MemoryOperation::Write {
                        access,
                        immediate,
                        output1,
                    },
                    wake,
                )
            }
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
    pub(super) fn cancel_memory_wait(&mut self, thread: GuestThreadId) {
        if let Some(mut pending) = self.memory_waits.remove(&thread)
            && pending.operation.cancel()
        {
            pending.token = None;
            pending.wake.wake_by_ref();
            self.memory_waits.insert(thread, pending);
        }
    }
    pub(super) fn register_memory_wait(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        thread: GuestThreadId,
    ) -> Result<bool, nixe_runtime::CoordinatorError> {
        let Some(pending) = self.memory_waits.get_mut(&thread) else {
            return Ok(false);
        };
        pending.token = if let MemoryOperation::Owned {
            deadline: Some(deadline),
            ..
        } = &pending.operation
        {
            Some(coordinator.register_timed_wait(
                thread,
                Some(deadline.saturating_sub(coordinator.virtual_time_ns())),
            )?)
        } else {
            coordinator.register_wait(thread, nixe_scheduler::Readiness::Pending)?
        };
        Ok(true)
    }
    pub fn next_service_deadline(&self) -> Option<Duration> {
        self.service_scheduling.next_deadline()
    }

    /// Called by the execution owner after ingress and before dispatching guest
    /// work. Wakers enqueue identities only; they cannot touch handles/registers.
    pub fn poll_memory_completions(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
    ) -> Result<(), HorizonSvcFault> {
        self.service_scheduling.poll_deadlines();
        let ready = std::mem::take(&mut *self.memory_completions.0.lock().unwrap());
        for (thread, generation) in ready {
            let Some(mut pending) = self.memory_waits.remove(&thread) else {
                continue;
            };
            if pending.generation != generation {
                self.memory_waits.insert(thread, pending);
                continue;
            }
            let current = pending.token.is_some_and(|token| {
                coordinator
                    .thread_scheduling_info(thread.get())
                    .is_ok_and(|view| {
                        view.process == pending.process
                            && view.active_wait == Some(token)
                            && view.lifecycle == nixe_scheduler::ThreadLifecycle::Waiting
                    })
            });
            if !current && !pending.operation.cancel() {
                continue;
            }
            pending.wake.queued.store(false, Ordering::Release);
            let mut completed = match pending.operation.poll(&pending.wake) {
                Poll::Pending => {
                    self.memory_waits.insert(thread, pending);
                    continue;
                }
                Poll::Ready(result) => result,
            };
            // Started kernel effects belong to the retained process/wait owner,
            // including when their initiating caller no longer needs a reply.
            if let Ok(SvcCompletion::Kernel(completion)) = &mut completed
                && let Some((process, addresses)) = completion.mutex_updates.take()
                && coordinator.process(process).is_some()
            {
                coordinator
                    .synchronize_mutex_waits(process, &addresses)
                    .map_err(|_| runtime_fault("completed kernel ownership change"))?;
            }
            if !current {
                if let Err(fault) = completed
                    && fault.guest_result().is_none()
                {
                    return Err(fault);
                }
                continue;
            }
            let token = pending
                .token
                .expect("current memory continuation has its wait token");
            if let Ok(SvcCompletion::Kernel(read)) = completed {
                self.synchronize_virtual_time(coordinator.virtual_time_ns());
                let handling = coordinator
                    .resume_supervisor_call(token, read.vcpu, read.source, |context| {
                        read.resume(self, context)
                    })
                    .map_err(|_| runtime_fault("kernel access continuation"))?
                    .ok_or_else(|| runtime_fault("kernel access continuation generation"))?;
                let handling = self
                    .finish_scheduled_call(coordinator, pending.process, thread, handling)
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
                    self.finish_scheduled_call(coordinator, pending.process, thread, handling)
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
            let state = coordinator
                .process_mut(pending.process)
                .and_then(|process| process.thread_mut(thread))
                .map(nixe_runtime::GuestThread::state_mut)
                .ok_or_else(|| runtime_fault("memory continuation caller"))?;
            write_register(state, 0, u64::from(code.raw()));
            if let Some(value) = output1 {
                write_register(state, 1, value);
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
