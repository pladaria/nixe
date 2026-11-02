//! Accepted kernel waits own their resources and cancel with the caller.
#[cfg(test)]
mod tests;
use super::*;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

pub(super) struct SvcWait {
    queued: AtomicBool,
    queue: Weak<nixe_runtime::KernelQueue>,
    dispatcher: Arc<HorizonSvcDispatcher>,
    owner: Arc<nixe_runtime::RunnableProcess>,
    caller: Arc<ThreadObject>,
    process_generation: u64,
    source: nixe_cpu::location::LocationDescriptor,
    state: Mutex<WaitState>,
}
#[derive(Default)]
struct WaitState {
    phase: SvcPhase,
    wait: Option<Arc<nixe_scheduler::KernelWait>>,
}
#[derive(Default)]
enum SvcPhase {
    Pending(MemoryOperation),
    Polling(Option<Arc<crate::ipc_wire::operation::CallLifetime>>),
    Ready(Result<SvcCompletion, HorizonSvcFault>),
    Failed(nixe_runtime::DispatchFault),
    #[default]
    Closed,
}
impl Wake for SvcWait {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel)
            && let Some(queue) = self.queue.upgrade()
        {
            queue.enqueue(self.clone());
        }
    }
}
impl nixe_runtime::ThreadWait for SvcWait {
    fn caller(&self) -> &Arc<ThreadObject> {
        &self.caller
    }
    fn source(&self) -> nixe_cpu::location::LocationDescriptor {
        self.source
    }
    fn token(&self) -> Option<nixe_scheduler::WakeToken> {
        self.state
            .lock()
            .unwrap()
            .wait
            .as_ref()
            .map(|wait| wait.token())
    }
    fn cancel(self: Arc<Self>) {
        let cancelled = {
            let mut state = self.state.lock().unwrap();
            match &state.phase {
                SvcPhase::Pending(operation) if operation.cancel() => None,
                SvcPhase::Polling(Some(lifetime)) if lifetime.cancel() => None,
                SvcPhase::Failed(_) => None,
                _ => {
                    let fault = match &mut state.phase {
                        SvcPhase::Pending(operation) => operation.cancelled_access_fault(),
                        _ => None,
                    };
                    Some(std::mem::replace(
                        &mut state.phase,
                        fault.map_or(SvcPhase::Closed, SvcPhase::Failed),
                    ))
                }
            }
        };
        drop(cancelled);
        self.wake_by_ref();
    }
    fn poll(
        self: Arc<Self>,
    ) -> Poll<Result<nixe_runtime::WaitCompletion, nixe_runtime::DispatchFault>> {
        use nixe_runtime::WaitCompletion;
        self.queued.store(false, Ordering::Release);
        // Runtime owns canonical request/token validation before publishing a wake.
        // This request owns its operation phase, cancellation and process admission.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread_queue.cpp#L21-L35
        let mut operation = {
            let mut state = self.state.lock().unwrap();
            if matches!(state.phase, SvcPhase::Failed(_)) {
                let SvcPhase::Failed(fault) = std::mem::take(&mut state.phase) else {
                    unreachable!()
                };
                return Poll::Ready(Err(fault));
            }
            // A duplicate consumes its notification, never the ready continuation.
            if matches!(state.phase, SvcPhase::Ready(_)) {
                return Poll::Pending;
            }
            let SvcPhase::Pending(operation) = &state.phase else {
                return Poll::Ready(Ok(WaitCompletion::Cancelled));
            };
            if self.process_generation != self.owner.generation() && !operation.cancel() {
                let fault = match &mut state.phase {
                    SvcPhase::Pending(operation) => operation.cancelled_access_fault(),
                    _ => unreachable!(),
                };
                let cancelled = std::mem::take(&mut state.phase);
                drop(state);
                drop(cancelled);
                return Poll::Ready(fault.map_or(Ok(WaitCompletion::Cancelled), Err));
            }
            let lifetime = match operation {
                MemoryOperation::Owned { lifetime, .. } => Some(lifetime.clone()),
                _ => None,
            };
            let SvcPhase::Pending(operation) =
                std::mem::replace(&mut state.phase, SvcPhase::Polling(lifetime))
            else {
                unreachable!("accepted pending operation");
            };
            operation
        };
        // A service executes outside the wait lock, as the kernel request owner.
        // Cancellation targets the actual call lifetime while the core owns its future.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_server_session.cpp
        let polled =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation.poll(Some(&self))));
        let mut state = self.state.lock().unwrap();
        let completed = match polled {
            Ok(Poll::Pending) if matches!(state.phase, SvcPhase::Polling(_)) => {
                state.phase = SvcPhase::Pending(operation);
                return Poll::Pending;
            }
            Ok(Poll::Pending) => {
                drop(state);
                drop(operation);
                return Poll::Ready(Ok(WaitCompletion::Cancelled));
            }
            Ok(Poll::Ready(completed)) => completed,
            Err(panic) => {
                operation.cancel();
                state.phase = SvcPhase::Closed;
                drop(state);
                drop(operation);
                std::panic::resume_unwind(panic);
            }
        };
        let active = matches!(state.phase, SvcPhase::Polling(_))
            && self.process_generation == self.owner.generation();
        state.phase = SvcPhase::Closed;
        // A backend failure is terminal host work, not a guest continuation.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp#L363-L380
        let completed = match completed {
            Err(fault) if fault.guest_result().is_none() => {
                drop(state);
                drop(operation);
                return Poll::Ready(Err(Box::new(fault)));
            }
            Ok(SvcCompletion::Access {
                result: Err(fault),
                continuation,
            }) if !fault.is_access_fault() => {
                drop(state);
                drop(operation);
                drop(continuation);
                return Poll::Ready(Err(Box::new(fault)));
            }
            completed => completed,
        };
        if !active {
            drop(state);
            drop(operation);
            return Poll::Ready(Ok(WaitCompletion::Cancelled));
        }
        let result = match completed {
            Ok(SvcCompletion::Return { code, output1 }) => Ok(WaitCompletion::Return {
                x0: u64::from(code.raw()),
                x1: output1,
            }),
            Err(fault) => {
                state.phase = SvcPhase::Ready(Err(fault));
                Ok(WaitCompletion::Resume)
            }
            Ok(completed) => {
                state.phase = SvcPhase::Ready(Ok(completed));
                Ok(WaitCompletion::Resume)
            }
        };
        drop(state);
        drop(operation);
        Poll::Ready(result)
    }
    fn resume(
        self: Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<nixe_runtime::DispatchFault> {
        let completed = match std::mem::take(&mut self.state.lock().unwrap().phase) {
            SvcPhase::Ready(completed) => completed,
            SvcPhase::Failed(fault) => return ExceptionDispatchOutcome::Fault(fault),
            SvcPhase::Closed if context.generation() != self.owner.generation() => {
                return ExceptionDispatchOutcome::Resume(ExceptionResume::Next);
            }
            _ => {
                return ExceptionDispatchOutcome::Fault(Box::new(runtime_fault(
                    "kernel continuation already consumed",
                )));
            }
        };
        self.dispatcher
            .finish_memory_operation(context, completed)
            .map_fault(|fault| Box::new(fault) as nixe_runtime::DispatchFault)
    }
}

type AccessContinuation = Box<
    dyn FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
            &mut ExceptionDispatchContext<'_>,
            nixe_memory::AccessResultValue,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send,
>;

type KernelContinuation = Box<
    dyn FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
            &mut ExceptionDispatchContext<'_>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send,
>;
pub(super) enum SvcCompletion {
    Access {
        result: nixe_memory::AccessResultValue,
        continuation: AccessContinuation,
    },
    Kernel(KernelContinuation),
    Return {
        code: HorizonKernelResult,
        output1: Option<u64>,
    },
    ExitProcess,
}
impl SvcCompletion {
    pub(super) fn kernel(
        continuation: impl FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
            &mut ExceptionDispatchContext<'_>,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> Self {
        Self::Kernel(Box::new(continuation))
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
    // Cancellation may discard guest results, but not an already completed backend failure.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp#L363-L380
    fn cancelled_access_fault(&mut self) -> Option<nixe_runtime::DispatchFault> {
        let access = match self {
            Self::Access { access, .. } | Self::Write { access, .. } => access,
            _ => return None,
        };
        match access.poll() {
            Some(Err(fault)) if !fault.is_access_fault() => Some(Box::new(fault)),
            _ => None,
        }
    }
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
                continuation,
            } => match access.poll() {
                None => Poll::Pending,
                Some(result) => Poll::Ready(Ok(SvcCompletion::Access {
                    result,
                    continuation: continuation.take().expect("kernel access resumes once"),
                })),
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
                Some(Ok(nixe_memory::AccessResult::Written(_))) => {
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
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        address: GuestVirtualAddress,
        bytes: usize,
        continuation: impl FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
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
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        access: nixe_memory::PendingAccess,
        continuation: impl FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
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
                    continuation(dispatcher, context, bytes.into_boxed_slice())
                }
                Ok(_) => {
                    ExceptionDispatchOutcome::Fault(runtime_fault("kernel read completion kind"))
                }
                Err(fault) => reject(context, HorizonSvcFault::MemoryAccess { immediate, fault }),
            },
        )
    }

    pub(super) fn kernel_access(
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        access: Result<nixe_memory::Access<nixe_memory::AccessResult>, nixe_memory::MemoryError>,
        continuation: impl FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
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
        self.start_memory_operation(
            context,
            MemoryOperation::Access {
                access,
                continuation: Some(Box::new(continuation)),
            },
        )
    }

    pub(super) fn finish_wait_start(
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        start: Result<nixe_scheduler::WaitStart, nixe_scheduler::SchedulerError>,
        continuation: impl FnOnce(
            &std::sync::Arc<HorizonSvcDispatcher>,
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
        let owned = wait.clone();
        self.start_owned_svc(context, Arc::default(), Some(wait), async move {
            let outcome = owned.wait().await;
            Ok(SvcCompletion::kernel(move |dispatcher, context| {
                continuation(dispatcher, context, outcome)
            }))
        })
    }

    pub(super) fn start_owned_svc(
        self: &std::sync::Arc<Self>,
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
    pub(super) fn set_heap_size(
        self: &std::sync::Arc<Self>,
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
    fn start_memory_operation(
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        mut operation: MemoryOperation,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Preserve the admission generation even if the first poll races exit.
        let process_generation = context.generation();
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
                return reject(
                    context,
                    HorizonSvcFault::Scheduling {
                        operation: "inline wait completion",
                        fault,
                    },
                );
            }
            return self.finish_memory_operation(context, completed);
        }
        let wake = Arc::new(SvcWait {
            queued: AtomicBool::new(false),
            queue: Arc::downgrade(context.completion_queue()),
            dispatcher: self.clone(),
            owner: context.process().clone(),
            caller: context.thread().object().clone(),
            process_generation,
            source: context.source(),
            state: Mutex::default(),
        });
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
                    return reject(
                        context,
                        HorizonSvcFault::Scheduling {
                            operation: "inline wait completion",
                            fault,
                        },
                    );
                }
                self.finish_memory_operation(context, completed)
            }
            Poll::Pending => {
                wake.state.lock().unwrap().phase = SvcPhase::Pending(operation);
                let registration = {
                    let mut state = wake.state.lock().unwrap();
                    let registration = match &state.phase {
                        SvcPhase::Pending(MemoryOperation::Owned {
                            wait: Some(wait), ..
                        }) => Ok(wait.clone()),
                        _ => context
                            .process()
                            .scheduler()
                            .register_wait(wake.caller.id()),
                    };
                    match registration {
                        Ok(wait) => {
                            wait.commit();
                            state.wait = Some(wait);
                            Ok(())
                        }
                        Err(fault) => Err(fault),
                    }
                };
                if let Err(fault) = registration {
                    wake.caller.set_wait(wake.clone());
                    nixe_runtime::ThreadWait::cancel(wake);
                    // Termination can win admission; it remains a kernel result.
                    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1328-L1356
                    return reject(
                        context,
                        HorizonSvcFault::Scheduling {
                            operation: "kernel continuation registration",
                            fault,
                        },
                    );
                }
                wake.caller.set_wait(wake.clone());
                if wake.owner.generation() != process_generation {
                    nixe_runtime::ThreadWait::cancel(wake);
                }
                ExceptionDispatchOutcome::Suspend(ExceptionResume::Next)
            }
        }
    }
    fn finish_memory_operation(
        self: &std::sync::Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        completed: Result<SvcCompletion, HorizonSvcFault>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match completed {
            Ok(SvcCompletion::Access {
                result,
                continuation,
            }) => continuation(self, context, result),
            Ok(SvcCompletion::Kernel(continuation)) => continuation(self, context),
            Ok(SvcCompletion::ExitProcess) => terminate(ExceptionTerminationScope::Process),
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
        self: &std::sync::Arc<Self>,
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
        self: &std::sync::Arc<Self>,
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
        self: &std::sync::Arc<Self>,
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
}
