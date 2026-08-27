use super::*;

impl HorizonSvcDispatcher {
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
        let Some(object_id) = context
            .process()
            .handles()
            .get_as::<ThreadObject>(handle)
            .map(|thread| thread.thread_id())
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        self.suspend_for_runtime_request(
            context.thread().id(),
            PendingRuntimeRequest::StartThread { object_id },
            "StartThread",
        )
    }

    pub(super) fn stage_thread_object_request(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        handle: u32,
        operation: &'static str,
        build: impl FnOnce(u64) -> PendingRuntimeRequest,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let object_id = if handle == CURRENT_THREAD_HANDLE || handle == context.thread().handle() {
            Some(context.thread().object().thread_id())
        } else {
            context
                .process()
                .handles()
                .get_as::<ThreadObject>(handle)
                .map(|thread| thread.thread_id())
        };
        let Some(object_id) = object_id else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        self.suspend_for_runtime_request(context.thread().id(), build(object_id), operation)
    }

    pub(super) fn get_thread_priority(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 1) as u32;
        self.stage_thread_object_request(context, handle, "GetThreadPriority", |object_id| {
            PendingRuntimeRequest::GetThreadPriority { object_id }
        })
    }

    pub(super) fn set_thread_priority(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let priority = read_register(context.thread().state(), 1) as u32 as i32;
        self.stage_thread_object_request(context, handle, "SetThreadPriority", |object_id| {
            PendingRuntimeRequest::SetThreadPriority {
                object_id,
                priority,
            }
        })
    }

    pub(super) fn get_thread_core_mask(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 2) as u32;
        self.stage_thread_object_request(context, handle, "GetThreadCoreMask", |object_id| {
            PendingRuntimeRequest::GetThreadCoreMask { object_id }
        })
    }

    pub(super) fn set_thread_core_mask(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let handle = read_register(context.thread().state(), 0) as u32;
        let ideal_core = read_register(context.thread().state(), 1) as u32 as i32;
        let affinity_mask = read_register(context.thread().state(), 2);
        self.stage_thread_object_request(context, handle, "SetThreadCoreMask", |object_id| {
            PendingRuntimeRequest::SetThreadCoreMask {
                object_id,
                ideal_core,
                affinity_mask,
            }
        })
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
        self.stage_thread_object_request(context, handle, "SetThreadActivity", |object_id| {
            PendingRuntimeRequest::SetThreadActivity {
                object_id,
                paused: activity == 1,
            }
        })
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
        let end = address.get().checked_add(context_size);
        let mapping = context.process().memory().query_memory(
            context.process().cpu().address_space_id(),
            address,
            GuestVirtualAddress::new(context.process().address_space_limit()),
        );
        if end.is_none_or(|end| {
            mapping.as_ref().is_none_or(|mapping| {
                mapping.base.get() > address.get()
                    || mapping.base.get().saturating_add(mapping.size) < end
                    || !mapping.permissions.contains(MemoryPermissions::WRITE)
            })
        }) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        self.stage_thread_object_request(context, handle, "GetThreadContext3", |object_id| {
            PendingRuntimeRequest::GetThreadContext { object_id, address }
        })
    }
}
