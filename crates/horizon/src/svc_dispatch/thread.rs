use super::*;
use memory_wait::SvcCompletion;
use nixe_scheduler::{CoreSet, IdealCore, ThreadPropertyGuard};
use std::sync::Arc;

impl HorizonSvcDispatcher {
    pub(super) fn cancel_synchronization(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Pending cancellation and cancellable-wait semantics:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_thread.cpp#L832-L846
        let handle = read_register(context.thread().state(), 0) as u32;
        self.stage_thread_object_request(context, handle, "CancelSynchronization", |thread| {
            PendingRuntimeRequest::CancelSynchronization { thread }
        })
    }
    pub(super) fn create_thread(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // ABI layout and processor-id rules:
        // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#CreateThread
        let request = PendingRuntimeRequest::CreateThread {
            entry: GuestVirtualAddress::new(read_register(context.thread().state(), 1)),
            argument: read_register(context.thread().state(), 2),
            stack_top: GuestVirtualAddress::new(read_register(context.thread().state(), 3)),
            priority: read_register(context.thread().state(), 4) as u32 as i32,
            core_id: read_register(context.thread().state(), 5) as u32 as i32,
        };
        self.suspend_for_runtime_request(context.thread().id(), request, "CreateThread")
    }

    pub(super) fn start_thread(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#StartThread
        let handle = read_register(context.thread().state(), 0) as u32;
        let Some(thread) = context
            .process()
            .handles()
            .lookup_as::<ThreadObject>(handle)
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        self.suspend_for_runtime_request(
            context.thread().id(),
            PendingRuntimeRequest::StartThread { thread },
            "StartThread",
        )
    }

    pub(super) fn stage_thread_object_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        handle: u32,
        operation: &'static str,
        build: impl FnOnce(std::sync::Arc<ThreadObject>) -> PendingRuntimeRequest,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let thread = thread_object(context, handle);
        let Some(thread) = thread else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        self.suspend_for_runtime_request(context.thread().id(), build(thread), operation)
    }

    pub(super) fn get_thread_priority(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 1) as u32;
        self.stage_thread_object_request(context, handle, "GetThreadPriority", |thread| {
            PendingRuntimeRequest::GetThreadPriority { thread }
        })
    }

    pub(super) fn set_thread_priority(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let priority = read_register(context.thread().state(), 1) as u32 as i32;
        // Permissions belong to the caller, including operations on foreign handles.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_thread.cpp#L182-L196
        if !(0..64).contains(&priority)
            || context
                .process()
                .allowed_priority_mask()
                .is_some_and(|mask| mask & (1_u64 << priority) == 0)
        {
            result(context, HorizonKernelResult::INVALID_PRIORITY);
            return resume();
        }
        self.stage_thread_object_request(context, handle, "SetThreadPriority", |thread| {
            PendingRuntimeRequest::SetThreadPriority { thread, priority }
        })
    }

    pub(super) fn get_thread_core_mask(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 2) as u32;
        self.stage_thread_object_request(context, handle, "GetThreadCoreMask", |thread| {
            PendingRuntimeRequest::GetThreadCoreMask { thread }
        })
    }

    pub(super) fn set_thread_core_mask(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let mut ideal_core = read_register(context.thread().state(), 1) as u32 as i32;
        let mut affinity_mask = read_register(context.thread().state(), 2);
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_thread.cpp#L252-L280
        let code = if ideal_core == -2 {
            ideal_core = context.process().initial_ideal_vcpu().get() as i32;
            affinity_mask = 1u64 << ideal_core;
            None
        } else if context
            .process()
            .allowed_core_mask()
            .is_some_and(|mask| affinity_mask & !mask != 0)
            || (!matches!(ideal_core, -1 | -3) && !(0..64).contains(&ideal_core))
        {
            Some(HorizonKernelResult::INVALID_CORE_ID)
        } else if affinity_mask == 0
            || (ideal_core >= 0 && affinity_mask & (1_u64 << ideal_core) == 0)
        {
            Some(HorizonKernelResult::INVALID_COMBINATION)
        } else {
            None
        };
        if let Some(code) = code {
            result(context, code);
            return resume();
        }
        let mut mask = affinity_mask;
        let affinity = context
            .process()
            .scheduler()
            .profile()
            .core_set(std::iter::from_fn(move || {
                if mask == 0 {
                    return None;
                }
                let core = mask.trailing_zeros();
                mask &= mask - 1;
                Some(VirtualCpuId::new(core))
            }));
        let Ok(affinity) = affinity else {
            result(context, HorizonKernelResult::INVALID_CORE_ID);
            return resume();
        };
        let ideal = match ideal_core {
            -3 => IdealCore::Preserve,
            -1 => IdealCore::Any,
            _ => IdealCore::Core(VirtualCpuId::new(ideal_core as u32)),
        };
        self.with_thread_property(
            context,
            handle,
            false,
            "SetThreadCoreMask",
            move |dispatcher, context, thread, guard| {
                let scheduler = context.process().scheduler();
                match scheduler.migrate(GuestThreadId::new(thread.thread_id()), ideal, affinity) {
                    Ok(()) => dispatcher.finish_property_change(
                        context,
                        thread,
                        guard,
                        Some(affinity),
                        "SetThreadCoreMask",
                    ),
                    Err(nixe_scheduler::SchedulerError::IdealOutsideAffinity(_)) => {
                        result(context, HorizonKernelResult::INVALID_COMBINATION);
                        resume()
                    }
                    Err(fault) => ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                        operation: "SetThreadCoreMask",
                        fault,
                    }),
                }
            },
        )
    }

    pub(super) fn sleep_thread(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#SleepThread
        let nanoseconds = read_register(context.thread().state(), 0) as i64;
        self.suspend_for_runtime_request(
            context.thread().id(),
            PendingRuntimeRequest::SleepThread { nanoseconds },
            "SleepThread",
        )
    }

    pub(super) fn set_thread_activity(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let activity = read_register(context.thread().state(), 1) as u32;
        if activity > 1 {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        self.with_thread_property(
            context,
            handle,
            true,
            "SetThreadActivity",
            move |dispatcher, context, thread, guard| match context
                .process()
                .scheduler()
                .set_activity(GuestThreadId::new(thread.thread_id()), activity == 1)
            {
                Ok(()) if activity == 1 => dispatcher.finish_property_change(
                    context,
                    thread,
                    guard,
                    None,
                    "SetThreadActivity",
                ),
                Ok(()) => {
                    drop(guard);
                    result(context, HorizonKernelResult::SUCCESS);
                    resume()
                }
                Err(nixe_scheduler::SchedulerError::InvalidThreadState { .. }) => {
                    result(context, HorizonKernelResult::INVALID_STATE);
                    resume()
                }
                Err(fault) => ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                    operation: "SetThreadActivity",
                    fault,
                }),
            },
        )
    }

    pub(super) fn get_thread_context(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Horizon A64 ThreadContext layout and size:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libvapours/include/vapours/svc/svc_types_common.hpp#L260-L292
        let context_size = 0x320;
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
        let handle = read_register(context.thread().state(), 1) as u32;
        let ranges = match context.process().memory().space().capture(
            address,
            context_size,
            nixe_memory::MemoryPermissions::WRITE,
        ) {
            Ok(ranges) => ranges,
            Err(error) => {
                return reject(
                    context,
                    HorizonSvcFault::MemoryAccess {
                        immediate: 0x33,
                        fault: std::sync::Arc::new(error),
                    },
                );
            }
        };
        self.with_thread_property(
            context,
            handle,
            true,
            "GetThreadContext3",
            move |dispatcher, context, thread, guard| {
                dispatcher.suspend_for_runtime_request(
                    context.thread().id(),
                    PendingRuntimeRequest::GetThreadContext {
                        thread,
                        ranges,
                        guard,
                    },
                    "GetThreadContext3",
                )
            },
        )
    }
    fn with_thread_property(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        handle: u32,
        local: bool,
        operation: &'static str,
        continuation: impl FnOnce(
            &mut Self,
            &mut ExceptionDispatchContext<'_>,
            Arc<ThreadObject>,
            ThreadPropertyGuard,
        ) -> ExceptionDispatchOutcome<HorizonSvcFault>
        + Send
        + 'static,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let Some(thread) = thread_object(context, handle) else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        let target = GuestThreadId::new(thread.thread_id());
        let caller = context.thread().id();
        let scheduler = context.process().scheduler().clone();
        if local {
            let Some(view) = scheduler.thread(target) else {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            };
            if view.process.get() != context.process().process_id() {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            }
            if caller == target {
                result(context, HorizonKernelResult::INVALID_STATE);
                return resume();
            }
        }
        match scheduler.try_thread_property(caller, target) {
            Ok(Some(guard)) => continuation(self, context, thread, guard),
            Err(fault) => {
                ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling { operation, fault })
            }
            Ok(None) => {
                let source = context.source();
                let vcpu = context.thread().vcpu();
                self.start_owned_svc(context, Arc::default(), None, async move {
                    let guard = scheduler
                        .lock_thread_property(caller, target)
                        .await
                        .map_err(|fault| HorizonSvcFault::Scheduling { operation, fault })?;
                    Ok(SvcCompletion::kernel(
                        source,
                        vcpu,
                        move |dispatcher, context| continuation(dispatcher, context, thread, guard),
                    ))
                })
            }
        }
    }
    fn finish_property_change(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        thread: Arc<ThreadObject>,
        guard: ThreadPropertyGuard,
        allowed: Option<CoreSet>,
        operation: &'static str,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        match guard.is_quiescent(allowed) {
            Ok(true) => {
                drop(guard);
                result(context, HorizonKernelResult::SUCCESS);
                resume()
            }
            Err(fault) => {
                ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling { operation, fault })
            }
            Ok(false) => self.start_owned_svc(context, Arc::default(), None, async move {
                guard
                    .wait_quiescent(allowed)
                    .await
                    .map_err(|fault| HorizonSvcFault::Scheduling { operation, fault })?;
                drop(guard);
                drop(thread);
                Ok(SvcCompletion::success(None))
            }),
        }
    }
}

fn thread_object(context: &ExceptionDispatchContext<'_>, handle: u32) -> Option<Arc<ThreadObject>> {
    if handle == CURRENT_THREAD_HANDLE {
        Some(context.thread().object())
    } else {
        context
            .process()
            .handles()
            .lookup_as::<ThreadObject>(handle)
    }
}
