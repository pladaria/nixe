use super::address_word::{AddressWord, address_word_fault};
use super::*;

#[derive(Clone, Debug)]
enum ReplyWaitTarget {
    Port(std::sync::Arc<PortObject>),
    Session(std::sync::Arc<SessionObject>),
}

#[derive(Debug)]
pub(super) struct ReplyReceiveWait {
    targets: Vec<ReplyWaitTarget>,
    ranges: Vec<nixe_memory::AllocationRange>,
    size: usize,
    immediate: u32,
    timeout: i64,
    deadline: Option<u64>,
}

// The pending receive owns the server slot until it completes or is cancelled.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_light_server_session.cpp#L44-L74
#[derive(Debug)]
pub(super) struct LightReplyWait {
    session: std::sync::Arc<SessionObject>,
    server: SessionRequestOwner,
    pending: bool,
}
impl Drop for LightReplyWait {
    fn drop(&mut self) {
        if self.pending {
            self.session.cancel_light_receive(self.server);
        }
    }
}

impl HorizonSvcDispatcher {
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_synchronization.cpp#L117-L132
    pub(super) fn synchronize_preemption_state(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let scheduler = context.process().scheduler();
        let thread = context.thread().id();
        let core = context.thread().vcpu();
        let process = ProcessId::new(context.process().process_id());
        match scheduler.pinned_thread(process, core) {
            Ok(pin) if pin != Some(thread) => return resume(),
            Ok(_) => {}
            Err(fault) => {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                    operation: "SynchronizePreemptionState",
                    fault,
                });
            }
        }
        let address = context
            .thread()
            .tls_base()
            .checked_add(0x102)
            .expect("bounded kernel TLS");
        let unpin = move |scheduler: &nixe_scheduler::SchedulerState| match scheduler
            .unpin_thread(thread)
        {
            Ok(()) => resume(),
            Err(fault) => ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                operation: "SynchronizePreemptionState",
                fault,
            }),
        };
        let pending = match context
            .process()
            .memory()
            .space()
            .write_value(address, 2, 0)
        {
            Ok(nixe_memory::Access::Ready(())) => return unpin(scheduler),
            Err(fault) => {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::MemoryAccess {
                    immediate: 0x36,
                    fault: std::sync::Arc::new(fault),
                });
            }
            Ok(nixe_memory::Access::Pending(pending)) => pending,
        };
        let scheduler = scheduler.clone();
        self.kernel_access(
            context,
            Ok(nixe_memory::Access::Pending(pending)),
            move |_, _, result| match result {
                Err(fault) => ExceptionDispatchOutcome::Fault(HorizonSvcFault::MemoryAccess {
                    immediate: 0x36,
                    fault,
                }),
                Ok(nixe_memory::AccessResult::Written) => unpin(&scheduler),
                Ok(_) => ExceptionDispatchOutcome::Fault(runtime_fault("TLS flag completion kind")),
            },
        )
    }

    pub(super) fn reply_and_receive(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        user_buffer: bool,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // ABI and ordering reference:
        // https://switchbrew.org/w/index.php?title=SVC&oldid=14679#ReplyAndReceive
        // Kernel control-flow reference:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_ipc.cpp
        let (immediate, address, size, handles_address, count, reply_target, timeout) =
            if user_buffer {
                let address = read_register(context.thread().state(), 1);
                let size = read_register(context.thread().state(), 2);
                let timeout = read_reply_timeout(context.thread().state(), true);
                (
                    0x44,
                    GuestVirtualAddress::new(address),
                    size,
                    read_register(context.thread().state(), 3),
                    read_register(context.thread().state(), 4) as u32,
                    read_register(context.thread().state(), 5) as u32,
                    timeout,
                )
            } else {
                let timeout = read_reply_timeout(context.thread().state(), false);
                (
                    0x43,
                    thread_tls(context.thread().state()),
                    TLS_COMMAND_BUFFER_SIZE as u64,
                    read_register(context.thread().state(), 1),
                    read_register(context.thread().state(), 2) as u32,
                    read_register(context.thread().state(), 3) as u32,
                    timeout,
                )
            };
        if user_buffer && !address.get().is_multiple_of(USER_BUFFER_ALIGNMENT) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        if user_buffer && !size.is_multiple_of(USER_BUFFER_ALIGNMENT) {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        if user_buffer && size == 0 {
            result(context, HorizonKernelResult::INVALID_SIZE);
            return resume();
        }
        if user_buffer
            && address
                .get()
                .checked_add(size)
                .is_none_or(|end| address.get() >= end)
        {
            result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
            return resume();
        }
        let Ok(size) = usize::try_from(size) else {
            result(context, HorizonKernelResult::OUT_OF_RESOURCE);
            return resume();
        };
        if count > MAX_WAIT_HANDLES {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        let permission = if reply_target == 0 {
            nixe_memory::MemoryPermissions::WRITE
        } else {
            nixe_memory::MemoryPermissions::READ_WRITE
        };
        let ranges =
            match context
                .process()
                .memory()
                .space()
                .capture(address, size as u64, permission)
            {
                Ok(ranges) => ranges,
                Err(error) => {
                    return reject(
                        context,
                        HorizonSvcFault::MemoryAccess {
                            immediate,
                            fault: std::sync::Arc::new(error),
                        },
                    );
                }
            };
        self.read_svc_input(
            context,
            immediate,
            GuestVirtualAddress::new(handles_address),
            count as usize * 4,
            move |dispatcher, context, bytes| {
                let mut handles = [0; MAX_WAIT_HANDLES as usize];
                for (handle, encoded) in handles.iter_mut().zip(bytes.chunks_exact(4)) {
                    *handle = u32::from_le_bytes(encoded.try_into().unwrap());
                }
                let targets =
                    context
                        .process()
                        .handles()
                        .lookup_many(&handles[..count as usize], |object| {
                            if let Some(port) = object.downcast::<PortObject>()
                                && port.endpoint() == PortEndpoint::Server
                            {
                                Some(ReplyWaitTarget::Port(port))
                            } else if let Some(session) = object.downcast::<SessionObject>()
                                && session.endpoint() == SessionEndpoint::Server
                                && !session.is_light()
                            {
                                Some(ReplyWaitTarget::Session(session))
                            } else {
                                None
                            }
                        });
                let Ok(targets) = targets else {
                    result(context, HorizonKernelResult::INVALID_HANDLE);
                    return resume();
                };
                let wait = ReplyReceiveWait {
                    targets,
                    ranges,
                    size,
                    immediate,
                    timeout,
                    deadline: None,
                };
                if reply_target == 0 {
                    return dispatcher.poll_reply_wait(context, wait);
                }
                let Some(session) = context
                    .process()
                    .handles()
                    .lookup_as::<SessionObject>(reply_target)
                else {
                    result(context, HorizonKernelResult::INVALID_HANDLE);
                    return resume();
                };
                if session.endpoint() != SessionEndpoint::Server || session.is_light() {
                    result(context, HorizonKernelResult::INVALID_HANDLE);
                    return resume();
                }
                let request = match session.reply() {
                    Ok(request) => request,
                    Err(error) => {
                        write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
                        return session_error(context, error);
                    }
                };
                dispatcher.read_svc_message(
                    context,
                    immediate,
                    wait.ranges.clone(),
                    move |dispatcher, context, bytes| {
                        let reply = match bytes
                            .and_then(|bytes| capture_message_handles(context, bytes, true))
                        {
                            Ok(reply) => reply,
                            Err(code) => {
                                if let Err(error) =
                                    request.fail(SessionError::TransferFailed(code.raw()))
                                {
                                    write_register(
                                        context.thread_mut().state_mut(),
                                        1,
                                        u32::MAX as u64,
                                    );
                                    return session_error(context, error);
                                }
                                return dispatcher.poll_reply_wait(context, wait);
                            }
                        };
                        if let Err(error) = request.reply(reply) {
                            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
                            return session_error(context, error);
                        }
                        dispatcher.poll_reply_wait(context, wait)
                    },
                )
            },
        )
    }

    fn poll_reply_wait(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        wait: ReplyReceiveWait,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let immediate = wait.immediate;
        'receive: loop {
            for (index, target) in wait.targets.iter().enumerate() {
                match target {
                    ReplyWaitTarget::Port(port) if port.is_signalled() => {
                        result(context, HorizonKernelResult::SUCCESS);
                        write_register(context.thread_mut().state_mut(), 1, index as u64);
                        return resume();
                    }
                    ReplyWaitTarget::Session(session) if session.is_signalled() => {
                        match session.receive() {
                            Ok(mut receive) => {
                                let incoming = receive.take_message();
                                let mut message = match materialize_message_handles(
                                    context.process().handles(),
                                    incoming,
                                    wait.size,
                                ) {
                                    Ok(message) => message,
                                    Err(code) => {
                                        // ReceiveRequest completes the client with
                                        // the transfer error, then returns NotFound
                                        // so ReplyAndReceive selects the next work.
                                        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_server_session.cpp#L1158-L1218
                                        receive.fail(SessionError::TransferFailed(code.raw()));
                                        continue 'receive;
                                    }
                                };
                                message.receive = Some(receive);
                                return self.write_received_message(
                                    context,
                                    immediate,
                                    wait.ranges,
                                    message,
                                    Some(index as u64),
                                );
                            }
                            Err(SessionError::NoRequest) => {}
                            Err(error) => {
                                write_register(context.thread_mut().state_mut(), 1, index as u64);
                                return session_error(context, error);
                            }
                        }
                    }
                    _ => {}
                }
            }
            break;
        }
        let mut wait = wait;
        let timeout = if wait.timeout < 0 {
            None
        } else {
            let deadline = *wait
                .deadline
                .get_or_insert_with(|| self.virtual_time_ns().saturating_add(wait.timeout as u64));
            Some(deadline.saturating_sub(self.virtual_time_ns()))
        };
        let events: Vec<_> = wait
            .targets
            .iter()
            .map(|target| match target {
                ReplyWaitTarget::Port(port) => port.readable_event(),
                ReplyWaitTarget::Session(session) => session.readable_event(),
            })
            .collect();
        let start = context.process().scheduler().wait_on_objects(
            context.thread().id(),
            &events,
            timeout,
            true,
        );
        self.finish_wait_start(
            context,
            start,
            move |dispatcher, context, outcome| match outcome {
                nixe_scheduler::WaitOutcome::Signalled(_) => {
                    dispatcher.poll_reply_wait(context, wait)
                }
                _ => {
                    let code = if outcome == nixe_scheduler::WaitOutcome::TimedOut {
                        HorizonKernelResult::TIMED_OUT
                    } else {
                        HorizonKernelResult::CANCELLED
                    };
                    result(context, code);
                    write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
                    resume()
                }
            },
        )
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn reply_and_receive_light(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Retain the accepted endpoint for the entire wait, as the scoped object
        // in ReplyAndReceiveLight does; closing the source handle cannot revoke it.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/svc/svc_light_ipc.cpp
        let handle = read_register(context.thread().state(), 0) as u32;
        let Some(session) = context
            .process()
            .handles()
            .lookup_as::<SessionObject>(handle)
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Server || !session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let server = session_request_owner(context);
        let mut words = [0_u32; 7];
        for (index, word) in words.iter_mut().enumerate() {
            *word = read_register(context.thread().state(), index as u8 + 1) as u32;
        }
        if words[0] & (1 << 31) != 0
            && let Err(error) = session.reply_light(server, words)
        {
            return session_error(context, error);
        }
        let wait = LightReplyWait {
            session,
            server,
            pending: false,
        };
        self.poll_light_reply_wait(context, wait)
    }

    fn poll_light_reply_wait(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        mut wait: LightReplyWait,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let event = wait.session.readable_event();
        match wait.session.receive_light(wait.server) {
            Ok(SessionMessage::Light(request)) => {
                wait.pending = false;
                for (index, word) in request.into_iter().enumerate() {
                    write_register(
                        context.thread_mut().state_mut(),
                        index as u8 + 1,
                        u64::from(word),
                    );
                }
                result(context, HorizonKernelResult::SUCCESS);
                resume()
            }
            Ok(SessionMessage::Buffer(_) | SessionMessage::TransportedBuffer { .. })
            | Err(SessionError::MessageKindMismatch) => invalid_state(context),
            Err(SessionError::NoRequest) => {
                wait.pending = true;
                let start = context.process().scheduler().wait_on_objects(
                    context.thread().id(),
                    &[event],
                    None,
                    true,
                );
                self.finish_wait_start(context, start, move |dispatcher, context, outcome| {
                    if matches!(outcome, nixe_scheduler::WaitOutcome::Signalled(_)) {
                        dispatcher.poll_light_reply_wait(context, wait)
                    } else {
                        result(context, HorizonKernelResult::CANCELLED);
                        resume()
                    }
                })
            }
            Err(error) => session_error(context, error),
        }
    }
}

pub(super) fn insert_pair<A, B>(
    handles: &HandleTable,
    first: A,
    second: B,
) -> Result<(u32, u32), ()>
where
    A: nixe_runtime::HandleValue,
    B: nixe_runtime::HandleValue,
{
    let first_handle = handles.insert(first).map_err(|_| ())?;
    match handles.insert(second) {
        Ok(second_handle) => Ok((first_handle, second_handle)),
        Err(_) => {
            let _ = handles.close(first_handle);
            Err(())
        }
    }
}

pub(super) fn get_process_id(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/svc/svc_process.cpp#L25-L61
    let handle = read_register(context.thread().state(), 1) as u32;
    let process_id = if matches!(handle, CURRENT_PROCESS_HANDLE | CURRENT_THREAD_HANDLE) {
        Some(context.process().process_id())
    } else {
        context
            .process()
            .handles()
            .lookup(handle)
            .and_then(|object| {
                if let Some(process) = object.downcast_ref::<nixe_runtime::RunnableProcess>() {
                    Some(process.process_id())
                } else {
                    object
                        .downcast_ref::<ThreadObject>()?
                        .owner()
                        .map(|process| process.process_id())
                }
            })
    };
    if let Some(process_id) = process_id {
        result(context, HorizonKernelResult::SUCCESS);
        write_u64(context.thread_mut().state_mut(), 1, process_id);
    } else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
    }
    resume()
}

pub(super) fn get_thread_id(
    context: &mut ExceptionDispatchContext<'_>,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    let handle = read_register(context.thread().state(), 1) as u32;
    let thread_id = if handle == CURRENT_THREAD_HANDLE {
        Some(context.thread().id().get())
    } else {
        context
            .process()
            .handles()
            .lookup_as::<ThreadObject>(handle)
            .map(|thread| thread.thread_id())
    };
    if let Some(thread_id) = thread_id {
        result(context, HorizonKernelResult::SUCCESS);
        write_u64(context.thread_mut().state_mut(), 1, thread_id);
    } else {
        result(context, HorizonKernelResult::INVALID_HANDLE);
    }
    resume()
}

impl HorizonSvcDispatcher {
    pub(super) fn wait_synchronization(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let pointer = GuestVirtualAddress::new(read_register(context.thread().state(), 1));
        let count = read_register(context.thread().state(), 2) as u32;
        let timeout = read_wait_timeout(context.thread().state());
        if count > MAX_WAIT_HANDLES {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        self.read_svc_input(
            context,
            0x18,
            pointer,
            count as usize * 4,
            move |dispatcher, context, bytes| {
                let mut handles = [0; MAX_WAIT_HANDLES as usize];
                for (handle, encoded) in handles.iter_mut().zip(bytes.chunks_exact(4)) {
                    *handle = u32::from_le_bytes(encoded.try_into().unwrap());
                }
                let objects =
                    context
                        .process()
                        .handles()
                        .lookup_many(&handles[..count as usize], |object| {
                            let event =
                                if let Some(event) = object.downcast_ref::<ReadableEventObject>() {
                                    event.clone()
                                } else {
                                    object.downcast_ref::<ThreadObject>()?.readable_event()
                                };
                            Some((object.clone(), event))
                        });
                let Ok(objects) = objects else {
                    result(context, HorizonKernelResult::INVALID_HANDLE);
                    return resume();
                };
                let start = context.process().scheduler().wait_on_objects(
                    context.thread().id(),
                    objects.iter().map(|(_, event)| event),
                    (timeout >= 0).then_some(timeout as u64),
                    true,
                );
                dispatcher.finish_wait_start(context, start, move |_, context, outcome| {
                    // Keep every accepted object alive through terminal result publication.
                    let _objects = objects;
                    let (code, index) = match outcome {
                        nixe_scheduler::WaitOutcome::Signalled(index) => {
                            (HorizonKernelResult::SUCCESS, index)
                        }
                        nixe_scheduler::WaitOutcome::TimedOut => {
                            (HorizonKernelResult::TIMED_OUT, u32::MAX)
                        }
                        nixe_scheduler::WaitOutcome::Cancelled
                        | nixe_scheduler::WaitOutcome::Terminated => {
                            (HorizonKernelResult::CANCELLED, u32::MAX)
                        }
                        nixe_scheduler::WaitOutcome::Ready => {
                            unreachable!("object wait requires signal or cancellation")
                        }
                    };
                    result(context, code);
                    write_register(context.thread_mut().state_mut(), 1, index as u64);
                    resume()
                })
            },
        )
    }
}

// Public ABI declarations and kernel behavior:
// https://github.com/switchbrew/libnx/blob/master/nx/include/switch/kernel/svc.h
// https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_condition_variable.cpp
impl HorizonSvcDispatcher {
    // Kernel mutex ownership must be published before waking a waiter, and the
    // wait bit remains set until the final queued waiter has acquired the lock.
    // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_condition_variable.cpp
    pub(super) fn wait_process_wide_key_atomic(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let mutex = read_register(context.thread().state(), 0);
        let key = read_register(context.thread().state(), 1) & !3;
        let tag = read_register(context.thread().state(), 2) as u32;
        let timeout = read_register(context.thread().state(), 3) as i64;
        if !mutex.is_multiple_of(4) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        let key_word = match AddressWord::capture(context, key, MemoryPermissions::WRITE) {
            Ok(word) => Some(word),
            Err(error) if error.is_access_fault() => None,
            Err(error) => return address_word_fault(context, 0x1c, std::sync::Arc::new(error)),
        };
        let word = AddressWord::capture(context, mutex, MemoryPermissions::READ_WRITE)
            .map_err(std::sync::Arc::new);
        let waits = context.process().address_waits().clone();
        let handles = context.process().handles().clone();
        let kernel = context.process().scheduler().clone();
        let thread = context.thread().id();
        let deadline = (timeout > 0).then(|| self.virtual_time_ns().saturating_add(timeout as u64));
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime.begin().expect("new condition release");
        self.start_owned_svc(context, lifetime, None, async move {
            let operation = waits.address_operation(mutex, thread).await;
            let mut registration = if timeout != 0 {
                word.as_ref()
                    .ok()
                    .map(|word| {
                        waits
                            .lock()
                            .enqueue_condition(key, thread, tag, mutex, word.range())
                    })
                    .transpose()
                    .map_err(|fault| HorizonSvcFault::Scheduling {
                        operation: "WaitProcessWideKeyAtomic",
                        fault,
                    })?
            } else {
                None
            };
            if let Some(registration) = &registration
                && let Some(deadline) = deadline
            {
                kernel.arm_deadline(registration.registration(), deadline);
            }
            let handoff = waits.lock().claim_mutex(mutex, thread).map_err(|fault| {
                HorizonSvcFault::Scheduling {
                    operation: "condition mutex release",
                    fault,
                }
            })?;
            let released: Result<(), std::sync::Arc<nixe_memory::MemoryError>> = async {
                if let Some(key_word) = key_word
                    && let Err(error) = key_word.write(1).await
                    && !error.is_access_fault()
                {
                    return Err(error);
                }
                complete_mutex_handoff(&waits, &handoff, &word?).await
            }
            .await;
            if let Err(fault) = released {
                waits.lock().finish_mutex_handoff(
                    &handoff,
                    nixe_runtime::AddressWaitCompletion::InvalidMemory,
                );
                {
                    let mut state = waits.lock();
                    state.remove(key, thread);
                    state.remove(mutex, thread);
                }
                drop(registration);
                return if fault.is_access_fault() {
                    Ok(memory_wait::SvcCompletion::Return {
                        code: HorizonKernelResult::INVALID_CURRENT_MEMORY,
                        output1: None,
                    })
                } else {
                    Err(HorizonSvcFault::MemoryAccess {
                        immediate: 0x1c,
                        fault,
                    })
                };
            }
            drop(handoff);
            let claim = registration
                .as_mut()
                .and_then(|registration| waits.lock().finish_condition_release(registration));
            drop(operation);
            if let Some(claim) = claim {
                reacquire_condition_mutex(&waits, &handles, claim, 0x1c, thread).await?;
            }
            let wait = registration
                .as_ref()
                .map(|registration| registration.registration().clone());
            drop(registration);
            let code = if let Some(wait) = wait {
                let outcome = wait.wait().await;
                let mut state = waits.lock();
                let completion = state
                    .completion(key, thread)
                    .or_else(|| state.completion(mutex, thread));
                state.remove(key, thread);
                state.remove(mutex, thread);
                if matches!(
                    outcome,
                    nixe_scheduler::WaitOutcome::Cancelled
                        | nixe_scheduler::WaitOutcome::Terminated
                ) {
                    HorizonKernelResult::CANCELLED
                } else {
                    address_wait_result(completion.ok_or_else(|| {
                        runtime_fault("condition wait completed without its retained result")
                    })?)
                }
            } else {
                HorizonKernelResult::TIMED_OUT
            };
            Ok(memory_wait::SvcCompletion::Return {
                code,
                output1: None,
            })
        })
    }

    pub(super) fn signal_process_wide_key(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let thread = context.thread().id();
        let key = read_register(context.thread().state(), 0) & !3;
        let count = read_register(context.thread().state(), 1) as u32 as i32;
        let limit = if count <= 0 {
            usize::MAX
        } else {
            count as usize
        };
        let waits = context.process().address_waits().clone();
        let (claims, clear) = {
            let mut state = waits.lock();
            let mut claims = Vec::new();
            for _ in 0..limit {
                match state.claim_condition(key) {
                    None => break,
                    Some(nixe_runtime::ConditionSignal::PendingRelease) => {}
                    Some(nixe_runtime::ConditionSignal::Reacquire(claim)) => claims.push(claim),
                }
            }
            let clear = !state.has_condition_waiters(key);
            (claims, clear)
        };
        let key_word = if clear {
            match AddressWord::capture(context, key, MemoryPermissions::WRITE) {
                Ok(word) => Some(word),
                Err(error) if error.is_access_fault() => None,
                Err(error) => return address_word_fault(context, 0x1d, std::sync::Arc::new(error)),
            }
        } else {
            None
        };
        let handles = context.process().handles().clone();
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime
            .begin()
            .expect("new condition signal is not cancelled");
        self.start_owned_svc(context, lifetime, None, async move {
            // Publish the accepted queue's flag before yielding for any mutex.
            // A later registering wait must publish its own flag after this one.
            if let Some(key_word) = key_word
                && let Err(fault) = key_word.write(0).await
                && !fault.is_access_fault()
            {
                for claim in &claims {
                    waits
                        .lock()
                        .finish_condition_acquire(
                            claim,
                            nixe_runtime::ConditionAcquire::InvalidMemory,
                            thread,
                        )
                        .map_err(|fault| HorizonSvcFault::Scheduling {
                            operation: "condition cancellation",
                            fault,
                        })?;
                }
                return Err(HorizonSvcFault::MemoryAccess {
                    immediate: 0x1d,
                    fault,
                });
            }
            for claim in claims {
                reacquire_condition_mutex(&waits, &handles, claim, 0x1d, thread).await?;
            }
            Ok(memory_wait::SvcCompletion::kernel(source, vcpu, |_, _| {
                resume()
            }))
        })
    }

    pub(super) fn arbitrate_lock(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let owner_handle = read_register(context.thread().state(), 0) as u32;
        let mutex = read_register(context.thread().state(), 1);
        let tag = read_register(context.thread().state(), 2) as u32;
        if !mutex.is_multiple_of(4) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        let word = match AddressWord::capture(context, mutex, MemoryPermissions::READ) {
            Ok(word) => word,
            Err(error) => return address_word_fault(context, 0x1a, std::sync::Arc::new(error)),
        };
        let owner = context
            .process()
            .handles()
            .lookup_as::<ThreadObject>(owner_handle);
        let thread = context.thread().id();
        let waits = context.process().address_waits().clone();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime.begin().expect("new mutex acquisition");
        self.start_owned_svc(context, lifetime, None, async move {
            let operation = waits.address_operation(mutex, thread).await;
            let wait = match word.read_owned().await {
                Err(error) if error.is_access_fault() => {
                    return Ok(memory_wait::SvcCompletion::Return {
                        code: HorizonKernelResult::INVALID_CURRENT_MEMORY,
                        output1: None,
                    });
                }
                Err(fault) => {
                    return Err(HorizonSvcFault::MemoryAccess {
                        immediate: 0x1a,
                        fault,
                    });
                }
                Ok(value) if value != owner_handle | MUTEX_WAITERS => {
                    return Ok(memory_wait::SvcCompletion::success(None));
                }
                Ok(_) => match owner {
                    None => {
                        return Ok(memory_wait::SvcCompletion::Return {
                            code: HorizonKernelResult::INVALID_HANDLE,
                            output1: None,
                        });
                    }
                    Some(owner) => waits
                        .lock()
                        .enqueue(mutex, owner.id(), thread, tag)
                        .map_err(|fault| HorizonSvcFault::Scheduling {
                            operation: "ArbitrateLock",
                            fault,
                        })?,
                },
            };
            drop(operation);
            let outcome = wait.wait().await;
            let code = {
                let mut state = waits.lock();
                let completion = state.completion(mutex, thread);
                state.remove(mutex, thread);
                if matches!(
                    outcome,
                    nixe_scheduler::WaitOutcome::Cancelled
                        | nixe_scheduler::WaitOutcome::Terminated
                ) {
                    HorizonKernelResult::CANCELLED
                } else {
                    address_wait_result(completion.ok_or_else(|| {
                        runtime_fault("mutex wait completed without its retained result")
                    })?)
                }
            };
            Ok(memory_wait::SvcCompletion::Return {
                code,
                output1: None,
            })
        })
    }

    pub(super) fn arbitrate_unlock(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let mutex = read_register(context.thread().state(), 0);
        if !mutex.is_multiple_of(4) {
            result(context, HorizonKernelResult::INVALID_ADDRESS);
            return resume();
        }
        let word = AddressWord::capture(context, mutex, MemoryPermissions::WRITE)
            .map_err(std::sync::Arc::new);
        let waits = context.process().address_waits().clone();
        let thread = context.thread().id();
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime
            .begin()
            .expect("new kernel handoff is not cancelled");
        self.start_owned_svc(context, lifetime, None, async move {
            let written = {
                let _operation = waits.address_operation(mutex, thread).await;
                let handoff = waits.lock().claim_mutex(mutex, thread).map_err(|fault| {
                    HorizonSvcFault::Scheduling {
                        operation: "mutex handoff",
                        fault,
                    }
                })?;
                match word {
                    Ok(word) => complete_mutex_handoff(&waits, &handoff, &word).await,
                    Err(error) => {
                        waits.lock().finish_mutex_handoff(
                            &handoff,
                            nixe_runtime::AddressWaitCompletion::InvalidMemory,
                        );
                        Err(error)
                    }
                }
            };
            let code = match written {
                Ok(()) => HorizonKernelResult::SUCCESS,
                Err(error) if error.is_access_fault() => {
                    HorizonKernelResult::INVALID_CURRENT_MEMORY
                }
                Err(fault) => {
                    return Err(HorizonSvcFault::MemoryAccess {
                        immediate: 0x1b,
                        fault,
                    });
                }
            };
            Ok(memory_wait::SvcCompletion::kernel(
                source,
                vcpu,
                move |_, context| {
                    result(context, code);
                    resume()
                },
            ))
        })
    }
}

const MUTEX_WAITERS: u32 = 1 << 30;

fn address_wait_result(result: nixe_runtime::AddressWaitCompletion) -> HorizonKernelResult {
    match result {
        nixe_runtime::AddressWaitCompletion::Success => HorizonKernelResult::SUCCESS,
        nixe_runtime::AddressWaitCompletion::InvalidOwner => HorizonKernelResult::INVALID_STATE,
        nixe_runtime::AddressWaitCompletion::TimedOut => HorizonKernelResult::TIMED_OUT,
        nixe_runtime::AddressWaitCompletion::InvalidMemory => {
            HorizonKernelResult::INVALID_CURRENT_MEMORY
        }
    }
}

async fn complete_mutex_handoff(
    waits: &nixe_runtime::AddressWaitRegistry,
    handoff: &nixe_runtime::MutexHandoff,
    word: &AddressWord,
) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
    let value = handoff.next_tag.map_or(0, |tag| {
        tag | if handoff.more_waiters {
            MUTEX_WAITERS
        } else {
            0
        }
    });
    let written = word.write(value).await;
    let completion = if written.is_err() {
        nixe_runtime::AddressWaitCompletion::InvalidMemory
    } else {
        nixe_runtime::AddressWaitCompletion::Success
    };
    waits.lock().finish_mutex_handoff(handoff, completion);
    written
}

async fn reacquire_condition_mutex(
    waits: &std::sync::Arc<nixe_runtime::AddressWaitRegistry>,
    handles: &HandleTable,
    claim: nixe_runtime::ConditionClaim,
    immediate: u32,
    current: GuestThreadId,
) -> Result<(), HorizonSvcFault> {
    use nixe_runtime::ConditionAcquire;
    let _operation = waits.address_operation(claim.mutex, claim.thread).await;
    let word = AddressWord::from_range(claim.memory.clone());
    let outcome: Result<ConditionAcquire, std::sync::Arc<nixe_memory::MemoryError>> = async {
        let mut previous = word.read_owned().await?;
        loop {
            let replacement = if previous == 0 {
                claim.tag
            } else {
                previous | MUTEX_WAITERS
            };
            let (observed, written) = word.compare_exchange_owned(previous, replacement).await?;
            if written {
                break;
            }
            previous = observed;
        }
        Ok(if previous == 0 {
            ConditionAcquire::Acquired
        } else if let Some(owner) = handles
            .lookup_as::<ThreadObject>(previous & !MUTEX_WAITERS)
            .filter(|thread| !thread.is_signalled())
        {
            ConditionAcquire::Contended(GuestThreadId::new(owner.thread_id()))
        } else {
            ConditionAcquire::InvalidOwner
        })
    }
    .await;
    match outcome {
        Ok(outcome) => {
            waits
                .lock()
                .finish_condition_acquire(&claim, outcome, current)
                .map_err(|fault| HorizonSvcFault::Scheduling {
                    operation: "condition mutex acquisition",
                    fault,
                })?;
            Ok(())
        }
        Err(error) => {
            waits
                .lock()
                .finish_condition_acquire(&claim, ConditionAcquire::InvalidMemory, current)
                .map_err(|fault| HorizonSvcFault::Scheduling {
                    operation: "condition cancellation",
                    fault,
                })?;
            if error.is_access_fault() {
                Ok(())
            } else {
                Err(HorizonSvcFault::MemoryAccess {
                    immediate,
                    fault: error,
                })
            }
        }
    }
}
