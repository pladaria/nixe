use super::*;
use memory_wait::SvcCompletion;
use nixe_scheduler::{CoreSet, IdealCore, ThreadPropertyGuard};
use std::sync::Arc;

impl HorizonSvcDispatcher {
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
        if nanoseconds <= 0 {
            let migration = match nanoseconds {
                0 => Some(nixe_scheduler::YieldMigration::None),
                -1 => Some(nixe_scheduler::YieldMigration::Suggested),
                -2 => Some(nixe_scheduler::YieldMigration::Any),
                _ => None,
            };
            if let Some(migration) = migration
                && let Err(fault) = context
                    .process()
                    .scheduler()
                    .yield_thread(context.thread().id(), migration)
            {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                    operation: "SleepThread",
                    fault,
                });
            }
            resume()
        } else {
            let start = context.process().scheduler().wait_on_objects(
                context.thread().id(),
                &[],
                Some(nanoseconds as u64),
                false,
            );
            self.finish_wait_start(context, start, |_, _, _| resume())
        }
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
        let address = GuestVirtualAddress::new(read_register(context.thread().state(), 0));
        let handle = read_register(context.thread().state(), 1) as u32;
        self.with_thread_property(
            context,
            handle,
            true,
            "GetThreadContext3",
            move |dispatcher, context, thread, guard| {
                if !context
                    .process()
                    .scheduler()
                    .thread(thread.id())
                    .expect("retained thread")
                    .paused
                {
                    result(context, HorizonKernelResult::INVALID_STATE);
                    return resume();
                }
                match guard.is_quiescent(None) {
                    Ok(true) => {
                        return dispatcher.write_thread_context(context, thread, guard, address);
                    }
                    Err(fault) => {
                        return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                            operation: "GetThreadContext3",
                            fault,
                        });
                    }
                    Ok(false) => {}
                }
                let source = context.source();
                let vcpu = context.thread().vcpu();
                dispatcher.start_owned_svc(context, Arc::default(), None, async move {
                    guard.wait_quiescent(None).await.map_err(|fault| {
                        HorizonSvcFault::Scheduling {
                            operation: "GetThreadContext3",
                            fault,
                        }
                    })?;
                    Ok(SvcCompletion::kernel(
                        source,
                        vcpu,
                        move |dispatcher, context| {
                            dispatcher.write_thread_context(context, thread, guard, address)
                        },
                    ))
                })
            },
        )
    }
    fn write_thread_context(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        thread: Arc<ThreadObject>,
        guard: ThreadPropertyGuard,
        address: GuestVirtualAddress,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // The property owner excludes competing activity changes; capture the
        // saved context only after the native owner has released it.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L918-L949
        let info = context
            .process()
            .scheduler()
            .thread(thread.id())
            .expect("retained thread");
        let bytes = if matches!(
            info.lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
        ) {
            [0; 0x320]
        } else if let Some(state) = thread.snapshot() {
            encode_thread_context(&state)
        } else {
            return ExceptionDispatchOutcome::Fault(runtime_fault("GetThreadContext3 registers"));
        };
        drop(guard);
        let ranges = match context.process().memory().space().capture(
            address,
            0x320,
            nixe_memory::MemoryPermissions::WRITE,
        ) {
            Ok(ranges) => ranges,
            Err(fault) => {
                return reject(
                    context,
                    HorizonSvcFault::MemoryAccess {
                        immediate: 0x33,
                        fault: Arc::new(fault),
                    },
                );
            }
        };
        let machine = context.process().memory().account().machine();
        match nixe_memory::CapturedMemory::new(ranges)
            .and_then(|ranges| machine.write_captured(&ranges, 0, &bytes))
        {
            Ok(access) => self.finish_svc_write(context, 0x33, access, None),
            Err(fault) => reject(
                context,
                HorizonSvcFault::MemoryAccess {
                    immediate: 0x33,
                    fault: Arc::new(fault),
                },
            ),
        }
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
            if thread
                .owner()
                .is_none_or(|owner| !Arc::ptr_eq(owner, context.process()))
            {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            }
            if caller == target {
                result(context, HorizonKernelResult::BUSY);
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
        Some(context.thread().object().clone())
    } else {
        context
            .process()
            .handles()
            .lookup_as::<ThreadObject>(handle)
    }
}

pub(super) fn cancel_synchronization(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let Some(thread) = thread_object(context, handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    match context
        .process()
        .scheduler()
        .cancel_synchronization(thread.id())
    {
        Ok(_) => {}
        Err(fault) => {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                operation: "CancelSynchronization",
                fault,
            });
        }
    }
    result(context, HorizonKernelResult::SUCCESS);
    resume()
}

impl HorizonSvcDispatcher {
    pub(super) fn create_thread(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let core = read_register(context.thread().state(), 5) as u32 as i32;
        let ideal = if core == -2 {
            context.process().initial_ideal_vcpu()
        } else if let Ok(core) = u32::try_from(core) {
            VirtualCpuId::new(core)
        } else {
            result(context, HorizonKernelResult::INVALID_CORE_ID);
            return resume();
        };
        let Ok(affinity) = context.process().scheduler().profile().core_set([ideal]) else {
            result(context, HorizonKernelResult::INVALID_CORE_ID);
            return resume();
        };
        let request = ThreadCreateRequest {
            entry: GuestVirtualAddress::new(read_register(context.thread().state(), 1)),
            argument: read_register(context.thread().state(), 2),
            fpcr: context.thread().state().fpcr(),
            fpsr: context.thread().state().fpsr(),
            stack_top: GuestVirtualAddress::new(read_register(context.thread().state(), 3)),
            priority: read_register(context.thread().state(), 4) as u32 as i32,
            ideal_vcpu: Some(ideal),
            affinity,
        };
        let process = context.process().clone();
        self.start_owned_svc(context, Arc::default(), None, async move {
            let code = match process.create_thread(request).await {
                Ok(created) => return Ok(SvcCompletion::success(Some(u64::from(created.handle)))),
                Err(ThreadCreateError::InvalidPriority(_)) => HorizonKernelResult::INVALID_PRIORITY,
                Err(ThreadCreateError::InvalidVirtualCpu(_)) => {
                    HorizonKernelResult::INVALID_CORE_ID
                }
                Err(ThreadCreateError::OutOfHandles) => HorizonKernelResult::OUT_OF_HANDLES,
                Err(ThreadCreateError::Memory(fault)) => {
                    return Err(HorizonSvcFault::MemoryAccess {
                        immediate: 0x08,
                        fault,
                    });
                }
                Err(ThreadCreateError::OutOfMemory) => HorizonKernelResult::OUT_OF_MEMORY,
                Err(ThreadCreateError::Terminating) => HorizonKernelResult::THREAD_TERMINATING,
                Err(ThreadCreateError::IdentityExhausted | ThreadCreateError::Internal) => {
                    return Err(runtime_fault("CreateThread"));
                }
            };
            Ok(SvcCompletion::Return {
                code,
                output1: None,
            })
        })
    }
}

pub(super) fn start_thread(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let Some(thread) = thread_object(context, handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    let code = match thread.start() {
        Ok(()) => HorizonKernelResult::SUCCESS,
        Err(nixe_runtime::ThreadStartError::InvalidState) => HorizonKernelResult::INVALID_STATE,
        Err(nixe_runtime::ThreadStartError::Internal) => {
            return ExceptionDispatchOutcome::Fault(runtime_fault("StartThread"));
        }
    };
    result(context, code);
    resume()
}

pub(super) fn get_thread_priority(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 1) as u32;
    let Some(thread) = thread_object(context, handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    let Some(info) = context.process().scheduler().thread(thread.id()) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    result(context, HorizonKernelResult::SUCCESS);
    write_register(
        context.thread_mut().state_mut(),
        1,
        info.effective_priority as u32 as u64,
    );
    resume()
}

pub(super) fn set_thread_priority(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 0) as u32;
    let priority = read_register(context.thread().state(), 1) as u32 as i32;
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_thread.cpp#L182-L196
    if !(0..64).contains(&priority)
        || context
            .process()
            .allowed_priority_mask()
            .is_some_and(|mask| mask & (1u64 << priority) == 0)
    {
        result(context, HorizonKernelResult::INVALID_PRIORITY);
        return resume();
    }
    let Some(thread) = thread_object(context, handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    match context.process().scheduler().set_priority(
        thread.id(),
        priority,
        Some(context.thread().id()),
    ) {
        Ok(()) => {
            result(context, HorizonKernelResult::SUCCESS);
            resume()
        }
        Err(fault) => ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
            operation: "SetThreadPriority",
            fault,
        }),
    }
}

pub(super) fn get_thread_core_mask(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 2) as u32;
    let Some(thread) = thread_object(context, handle) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    let Some(info) = context.process().scheduler().thread(thread.id()) else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
        return resume();
    };
    let Some(mask) = horizon_affinity_mask(&info.affinity) else {
        return ExceptionDispatchOutcome::Fault(runtime_fault("GetThreadCoreMask topology"));
    };
    result(context, HorizonKernelResult::SUCCESS);
    write_register(
        context.thread_mut().state_mut(),
        1,
        info.ideal_vcpu.map_or(-1, |core| core.get() as i32) as u32 as u64,
    );
    write_register(context.thread_mut().state_mut(), 2, mask);
    resume()
}
