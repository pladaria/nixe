//! Direct kernel memory and waits on the selected guest fiber's stack.
use super::*;
use std::{future::Future, sync::Arc};

pub(super) fn wait_future<F: Future + Send>(
    context: &mut ExceptionDispatchContext<'_>,
    future: F,
    wait: Option<Arc<nixe_scheduler::KernelWait>>,
) -> Result<F::Output, HorizonSvcFault> {
    context
        .wait_future(future, wait)
        .map_err(|fault| HorizonSvcFault::Scheduling {
            operation: "kernel stack wait",
            fault,
        })
}
pub(super) fn wait_io<F: Future>(
    context: &mut ExceptionDispatchContext<'_>,
    future: F,
) -> F::Output {
    context.wait_io(future)
}
pub(super) fn wait_access(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    access: nixe_memory::PendingAccess,
) -> Result<nixe_memory::AccessResult, HorizonSvcFault> {
    context
        .wait_access(access)
        .map_err(|fault| HorizonSvcFault::MemoryAccess { immediate, fault })
}
pub(super) fn wait_read(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    access: nixe_memory::PendingAccess,
) -> Result<Box<[u8]>, HorizonSvcFault> {
    match wait_access(context, immediate, access)? {
        nixe_memory::AccessResult::Read(bytes) => Ok(bytes.into_boxed_slice()),
        _ => Err(runtime_fault("kernel read completion kind")),
    }
}
pub(super) fn wait_kernel(
    context: &mut ExceptionDispatchContext<'_>,
    start: Result<nixe_scheduler::WaitStart, nixe_scheduler::SchedulerError>,
) -> Result<nixe_scheduler::WaitOutcome, HorizonSvcFault> {
    match start.map_err(|fault| HorizonSvcFault::Scheduling {
        operation: "kernel object wait",
        fault,
    })? {
        nixe_scheduler::WaitStart::Ready(outcome) => Ok(outcome),
        nixe_scheduler::WaitStart::Pending(wait) => {
            let owned = wait.clone();
            wait_future(context, async move { owned.wait().await }, Some(wait))
        }
    }
}
pub(super) fn finish(
    context: &mut ExceptionDispatchContext<'_>,
    completed: Result<(HorizonKernelResult, Option<u64>), HorizonSvcFault>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    match completed {
        Ok((code, output1)) => {
            result(context, code);
            if let Some(value) = output1 {
                write_register(context.thread_mut().state_mut(), 1, value);
            }
            resume()
        }
        Err(fault) => reject(context, fault),
    }
}
impl HorizonSvcDispatcher {
    pub(super) fn read_svc_input(
        &self,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        address: GuestVirtualAddress,
        bytes: usize,
    ) -> Result<Box<[u8]>, HorizonSvcFault> {
        let mut output = vec![0; bytes];
        match context
            .process()
            .memory()
            .space()
            .read_into(address, &mut output)
            .map_err(|fault| HorizonSvcFault::MemoryAccess {
                immediate,
                fault: Arc::new(fault),
            })? {
            nixe_memory::Access::Ready(()) => Ok(output.into_boxed_slice()),
            nixe_memory::Access::Pending(access) => wait_read(context, immediate, access),
        }
    }
    pub(super) fn set_heap_size(
        self: &Arc<Self>,
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
        let process = context.process().clone();
        let completed = wait_io(context, process.memory().resize_heap(bytes))
            .map_err(|fault| HorizonSvcFault::MemoryMapping {
                fault: Arc::new(fault),
            })
            .map(|base| (HorizonKernelResult::SUCCESS, Some(base.get())));
        finish(context, completed)
    }
    pub(super) fn write_svc_output(
        self: &Arc<Self>,
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
        self: &Arc<Self>,
        context: &mut ExceptionDispatchContext<'_>,
        immediate: u32,
        access: nixe_memory::Access<()>,
        output1: Option<u64>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let completed = match access {
            nixe_memory::Access::Ready(()) => Ok(()),
            nixe_memory::Access::Pending(access) => wait_access(context, immediate, access)
                .and_then(|result| {
                    if matches!(result, nixe_memory::AccessResult::Written(_)) {
                        Ok(())
                    } else {
                        Err(runtime_fault("kernel write completion kind"))
                    }
                }),
        };
        finish(
            context,
            completed.map(|()| (HorizonKernelResult::SUCCESS, output1)),
        )
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
            // @SPECULATIVE: expose remote backing as Normal memory. Its exact
            // kernel MemoryState flags and allowed transitions remain unknown.
            MemoryKind::Remote => 5,
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
