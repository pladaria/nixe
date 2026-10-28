use super::address_word::{AddressWord, address_word_fault};
use super::*;

#[derive(Clone, Debug)]
enum ReplyWaitTarget {
    Port(PortObject),
    Session(SessionObject),
}

#[derive(Debug)]
pub(super) struct ReplyReceiveWait {
    targets: Vec<ReplyWaitTarget>,
    ranges: Vec<nixe_memory::AllocationRange>,
    size: usize,
    immediate: u32,
    timeout: i64,
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
        if let Some(wait) = self.reply_waits.remove(&context.thread().id().get()) {
            return self.poll_reply_wait(context, wait);
        }
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
                let mut targets = Vec::with_capacity(count as usize);
                for encoded in bytes.chunks_exact(4) {
                    let handle = u32::from_le_bytes(encoded.try_into().unwrap());
                    if let Some(port) = context.process().handles().lookup_as::<PortObject>(handle)
                        && port.endpoint() == PortEndpoint::Server
                    {
                        targets.push(ReplyWaitTarget::Port(std::sync::Arc::unwrap_or_clone(port)));
                    } else if let Some(session) = context
                        .process()
                        .handles()
                        .lookup_as::<SessionObject>(handle)
                        && session.endpoint() == SessionEndpoint::Server
                        && !session.is_light()
                    {
                        targets.push(ReplyWaitTarget::Session(std::sync::Arc::unwrap_or_clone(
                            session,
                        )));
                    } else {
                        result(context, HorizonKernelResult::INVALID_HANDLE);
                        return resume();
                    }
                }
                let wait = ReplyReceiveWait {
                    targets,
                    ranges,
                    size,
                    immediate,
                    timeout,
                };
                let thread = context.thread().id().get();
                if dispatcher.cancelled_waits.remove(&thread) {
                    dispatcher.finish_reply_wait(thread, immediate);
                    result(context, HorizonKernelResult::CANCELLED);
                    write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
                    return resume();
                }
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
                dispatcher.read_svc_capture(
                    context,
                    immediate,
                    wait.ranges.clone(),
                    move |dispatcher, context, bytes| {
                        let reply = match capture_message_handles(context, bytes.into_vec(), true) {
                            Ok(reply) => reply,
                            Err(code) => {
                                result(context, code);
                                write_register(
                                    context.thread_mut().state_mut(),
                                    1,
                                    u32::MAX as u64,
                                );
                                return resume();
                            }
                        };
                        if let Err(error) = session.reply(reply) {
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
        let thread = context.thread().id().get();
        let immediate = wait.immediate;
        if self.cancelled_waits.remove(&thread) {
            self.finish_reply_wait(thread, immediate);
            result(context, HorizonKernelResult::CANCELLED);
            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
            return resume();
        }
        let events: Vec<_> = wait
            .targets
            .iter()
            .map(|target| match target {
                ReplyWaitTarget::Port(port) => port.readable_event(),
                ReplyWaitTarget::Session(session) => session.readable_event(),
            })
            .collect();
        // Clear hints before querying object state, so an arrival between the
        // query and observer registration remains signalled.
        for event in &events {
            event.clear();
        }
        for (index, target) in wait.targets.iter().enumerate() {
            match target {
                ReplyWaitTarget::Port(port) if port.is_signalled() => {
                    self.finish_reply_wait(thread, immediate);
                    result(context, HorizonKernelResult::SUCCESS);
                    write_register(context.thread_mut().state_mut(), 1, index as u64);
                    return resume();
                }
                ReplyWaitTarget::Session(session) if session.is_signalled() => {
                    match session.receive() {
                        Ok(
                            message @ (SessionMessage::Buffer(_)
                            | SessionMessage::TransportedBuffer { .. }),
                        ) => {
                            let message = match materialize_message_handles(context, message) {
                                Ok(Some(message)) => message,
                                Ok(None) => unreachable!("normal request has buffer transport"),
                                Err(code) => {
                                    self.finish_reply_wait(thread, immediate);
                                    result(context, code);
                                    return resume();
                                }
                            };
                            self.finish_reply_wait(thread, immediate);
                            if message.bytes.len() > wait.size {
                                result(context, HorizonKernelResult::INVALID_SIZE);
                                return resume();
                            }
                            return self.write_received_message(
                                context,
                                immediate,
                                wait.ranges,
                                message,
                                Some(index as u64),
                            );
                        }
                        Ok(SessionMessage::Light(_)) => {
                            self.finish_reply_wait(thread, immediate);
                            return invalid_state(context);
                        }
                        Err(SessionError::NoRequest) => {}
                        Err(error) => {
                            self.finish_reply_wait(thread, immediate);
                            write_register(context.thread_mut().state_mut(), 1, index as u64);
                            return session_error(context, error);
                        }
                    }
                }
                _ => {}
            }
        }
        if self.wait_expired(thread, immediate, wait.timeout) {
            self.finish_reply_wait(thread, immediate);
            result(context, HorizonKernelResult::TIMED_OUT);
            return resume();
        }
        if self.pending_wait_cancellations.remove(&thread) {
            self.finish_reply_wait(thread, immediate);
            result(context, HorizonKernelResult::CANCELLED);
            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
            return resume();
        }
        self.cancellable_waits.insert(thread);
        let deadline = self.wait_deadlines.get(&(thread, immediate)).copied();
        self.pending_wakes
            .insert(thread, PendingThreadWake { events, deadline });
        self.reply_waits.insert(thread, wait);
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
    }

    pub(super) fn wait_expired(&mut self, thread_id: u64, immediate: u32, timeout: i64) -> bool {
        if timeout == 0 {
            return true;
        }
        if timeout < 0 {
            return false;
        }
        let now = self.virtual_time_ns();
        let deadline = self
            .wait_deadlines
            .entry((thread_id, immediate))
            .or_insert_with(|| now.saturating_add(timeout as u64));
        now >= *deadline
    }

    pub(super) fn finish_reply_wait(&mut self, thread_id: u64, immediate: u32) {
        self.reply_waits.remove(&thread_id);
        self.reply_sent.remove(&thread_id);
        self.finish_wait(thread_id, immediate);
    }

    pub(super) fn finish_wait(&mut self, thread_id: u64, immediate: u32) {
        if immediate == 0x18 {
            self.synchronization_waits.remove(&thread_id);
        }
        if matches!(immediate, 0x18 | 0x43 | 0x44) {
            self.cancellable_waits.remove(&thread_id);
        }
        self.wait_deadlines.remove(&(thread_id, immediate));
        self.pending_wakes.remove(&thread_id);
    }
}

impl HorizonSvcDispatcher {
    pub(super) fn reply_and_receive_light(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        // Light IPC carries seven u32 words in registers and uses bit 31 of the
        // first word as the reply flag:
        // https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/kern_k_light_server_session.cpp
        let handle = read_register(context.thread().state(), 0) as u32;
        let Some(session) = context
            .process()
            .handles()
            .lookup_as::<SessionObject>(handle)
            .map(std::sync::Arc::unwrap_or_clone)
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        if session.endpoint() != SessionEndpoint::Server || !session.is_light() {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }
        let mut words = [0_u32; 7];
        for (index, word) in words.iter_mut().enumerate() {
            *word = read_register(context.thread().state(), index as u8 + 1) as u32;
        }
        let thread_id = context.thread().id().get();
        if words[0] & (1 << 31) != 0
            && !self.reply_sent.contains(&thread_id)
            && let Err(error) = session.reply(SessionMessage::Light(words))
        {
            return session_error(context, error);
        }
        if words[0] & (1 << 31) != 0 {
            self.reply_sent.insert(thread_id);
        } else {
            self.reply_sent.remove(&thread_id);
        }
        match session.receive() {
            Ok(SessionMessage::Light(request)) => {
                self.reply_sent.remove(&thread_id);
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
            | Err(SessionError::MessageKindMismatch) => {
                self.reply_sent.remove(&thread_id);
                result(context, HorizonKernelResult::INVALID_STATE);
                resume()
            }
            Err(SessionError::NoRequest) => self.suspend_on_session(context, &session),
            Err(error) => {
                self.reply_sent.remove(&thread_id);
                session_error(context, error)
            }
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
    let handle = read_register(context.thread().state(), 1) as u32;
    let process_id = if handle == CURRENT_PROCESS_HANDLE {
        Some(context.process().process_id())
    } else {
        context
            .process()
            .handles()
            .lookup_as::<ProcessObject>(handle)
            .map(|process| process.process_id())
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

/// Accepted synchronization objects remain alive until the wait ends, even
/// when another guest thread closes or reuses their numeric handles.
#[derive(Debug)]
pub(super) struct SynchronizationWait {
    _objects: Vec<HandleObject>,
    events: Vec<ReadableEventObject>,
    timeout: i64,
}
impl HorizonSvcDispatcher {
    pub(super) fn wait_synchronization(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let thread = context.thread().id().get();
        if let Some(wait) = self.synchronization_waits.remove(&thread) {
            return self.poll_synchronization(context, wait);
        }
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
                let mut objects = Vec::with_capacity(count as usize);
                let mut events = Vec::with_capacity(count as usize);
                for encoded in bytes.chunks_exact(4) {
                    let handle = u32::from_le_bytes(encoded.try_into().unwrap());
                    let Some(object) = context.process().handles().lookup(handle) else {
                        result(context, HorizonKernelResult::INVALID_HANDLE);
                        return resume();
                    };
                    let event = if let Some(event) = object.downcast_ref::<ReadableEventObject>() {
                        event.clone()
                    } else if let Some(thread) = object.downcast_ref::<ThreadObject>() {
                        thread.readable_event()
                    } else {
                        result(context, HorizonKernelResult::INVALID_HANDLE);
                        return resume();
                    };
                    objects.push(object);
                    events.push(event);
                }
                dispatcher.poll_synchronization(
                    context,
                    SynchronizationWait {
                        _objects: objects,
                        events,
                        timeout,
                    },
                )
            },
        )
    }
    fn poll_synchronization(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        wait: SynchronizationWait,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let thread = context.thread().id().get();
        if self.cancelled_waits.remove(&thread) {
            self.finish_wait(thread, 0x18);
            result(context, HorizonKernelResult::CANCELLED);
            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
            return resume();
        }
        // Validate/capture the whole object list before checking readiness,
        // matching KSynchronizationObject::Wait's retained-object contract.
        // https://git.eden-emu.dev/eden-emu/eden/src/branch/master/src/core/hle/kernel/k_synchronization_object.cpp
        if let Some(index) = wait
            .events
            .iter()
            .position(ReadableEventObject::is_signalled)
        {
            self.finish_wait(thread, 0x18);
            result(context, HorizonKernelResult::SUCCESS);
            write_register(context.thread_mut().state_mut(), 1, index as u64);
            return resume();
        }
        if self.wait_expired(thread, 0x18, wait.timeout) {
            self.finish_wait(thread, 0x18);
            result(context, HorizonKernelResult::TIMED_OUT);
            return resume();
        }
        if self.pending_wait_cancellations.remove(&thread) {
            self.finish_wait(thread, 0x18);
            result(context, HorizonKernelResult::CANCELLED);
            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
            return resume();
        }
        self.cancellable_waits.insert(thread);
        let deadline = self.wait_deadlines.get(&(thread, 0x18)).copied();
        self.pending_wakes.insert(
            thread,
            PendingThreadWake {
                events: wait.events.clone(),
                deadline,
            },
        );
        self.synchronization_waits.insert(thread, wait);
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
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
        let thread = context.thread().id();
        for address in [key, mutex] {
            if context
                .process()
                .address_waits()
                .lock()
                .is_signalled(address, thread)
            {
                let completion = context
                    .process()
                    .address_waits()
                    .lock()
                    .completion(address, thread)
                    .unwrap();
                context
                    .process()
                    .address_waits()
                    .lock()
                    .remove(address, thread);
                self.finish_wait(thread.get(), 0x1c);
                result(context, address_wait_result(completion));
                return resume();
            }
        }
        let waiting = {
            let waits = context.process().address_waits().lock();
            waits.contains(key, thread) || waits.contains(mutex, thread)
        };
        if waiting {
            if self.wait_expired(thread.get(), 0x1c, timeout) {
                context.process().address_waits().lock().remove(key, thread);
                context
                    .process()
                    .address_waits()
                    .lock()
                    .remove(mutex, thread);
                self.finish_wait(thread.get(), 0x1c);
                result(context, HorizonKernelResult::TIMED_OUT);
                return resume();
            }
            return ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        }
        let key_word = match AddressWord::capture(context, key, MemoryPermissions::WRITE) {
            Ok(word) => Some(word),
            Err(error) if error.is_access_fault() => None,
            Err(error) => return address_word_fault(context, 0x1c, std::sync::Arc::new(error)),
        };
        let word = AddressWord::capture(context, mutex, MemoryPermissions::READ_WRITE)
            .map_err(std::sync::Arc::new);
        let waits = context.process().address_waits().clone();
        self.wait_expired(thread.get(), 0x1c, timeout);
        let deadline = self.wait_deadlines.get(&(thread.get(), 0x1c)).copied();
        let registration = if timeout != 0 {
            word.as_ref().ok().map(|word| {
                waits
                    .lock()
                    .enqueue_condition(key, thread, tag, mutex, word.range())
            })
        } else {
            None
        };
        let registration = match registration.transpose() {
            Ok(registration) => registration,
            Err(fault) => {
                return ExceptionDispatchOutcome::Fault(HorizonSvcFault::Scheduling {
                    operation: "WaitProcessWideKeyAtomic",
                    fault,
                });
            }
        };
        if let Some(registration) = &registration {
            self.pending_wakes.insert(
                thread.get(),
                PendingThreadWake {
                    events: vec![registration.event()],
                    deadline,
                },
            );
        }
        let handles = context.process().handles().clone();
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let clock = self.virtual_clock.clone();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime
            .begin()
            .expect("new condition release is not cancelled");
        let memory_deadline = registration.as_ref().and(deadline);
        self.start_owned_svc(context, lifetime, memory_deadline, async move {
            let executed: Result<(), HorizonSvcFault> = async {
                {
                    // Eden holds its scheduler lock across waiter selection
                    // and the userspace write. Keep that same exclusion for
                    // this word while a physical operation is pending.
                    let _operation = waits.address_operation(mutex, thread, &clock).await;
                    let handoff = waits.lock().claim_mutex(mutex, thread).map_err(|fault| {
                        HorizonSvcFault::Scheduling {
                            operation: "mutex handoff",
                            fault,
                        }
                    })?;
                    let released = async {
                        if let Some(key_word) = key_word
                            && let Err(error) = key_word.write(1).await
                            && !error.is_access_fault()
                        {
                            return Err(error);
                        }
                        let word = word?;
                        complete_mutex_handoff(&waits, &handoff, &word).await
                    }
                    .await;
                    if released.is_err() {
                        waits.lock().finish_mutex_handoff(
                            &handoff,
                            nixe_runtime::AddressWaitCompletion::InvalidMemory,
                        );
                    }
                    released.map_err(|fault| HorizonSvcFault::MemoryAccess {
                        immediate: 0x1c,
                        fault,
                    })?;
                }
                if let Some(registration) = &registration {
                    let claim = waits.lock().finish_condition_release(registration);
                    if let Some(claim) = claim {
                        reacquire_condition_mutex(&waits, &handles, claim, &clock, 0x1c, thread)
                            .await?;
                    }
                }
                Ok(())
            }
            .await;
            match executed {
                Err(HorizonSvcFault::MemoryAccess { fault, .. }) if fault.is_access_fault() => Ok(
                    memory_wait::SvcCompletion::kernel(source, vcpu, move |dispatcher, context| {
                        let mut state = context.process().address_waits().lock();
                        state.remove(key, thread);
                        state.remove(mutex, thread);
                        drop(state);
                        dispatcher.finish_wait(thread.get(), 0x1c);
                        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
                        resume()
                    }),
                ),
                Err(fault) => Err(fault),
                Ok(()) => Ok(memory_wait::SvcCompletion::kernel(
                    source,
                    vcpu,
                    move |dispatcher, context| {
                        if timeout == 0 {
                            let waits = context.process().address_waits().clone();
                            {
                                let mut state = waits.lock();
                                state.remove(key, thread);
                                state.remove(mutex, thread);
                            }
                            dispatcher.finish_wait(thread.get(), 0x1c);
                            result(context, HorizonKernelResult::TIMED_OUT);
                            resume()
                        } else {
                            dispatcher.finish_condition_call(context, key, mutex)
                        }
                    },
                )),
            }
        })
    }

    fn finish_condition_call(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
        key: u64,
        mutex: u64,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        let thread = context.thread().id();
        let completion = {
            let mut state = context.process().address_waits().lock();
            [key, mutex].into_iter().find_map(|address| {
                let completion = state.completion(address, thread)?;
                state.remove(address, thread);
                Some(completion)
            })
        };
        if let Some(completion) = completion {
            self.finish_wait(thread.get(), 0x1c);
            result(context, address_wait_result(completion));
            resume()
        } else {
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
        }
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
        let clock = self.virtual_clock.clone();
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
                reacquire_condition_mutex(&waits, &handles, claim, &clock, 0x1d, thread).await?;
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
        let thread = context.thread().id();
        if context
            .process()
            .address_waits()
            .lock()
            .is_signalled(mutex, thread)
        {
            let completion = context
                .process()
                .address_waits()
                .lock()
                .completion(mutex, thread)
                .unwrap();
            context
                .process()
                .address_waits()
                .lock()
                .remove(mutex, thread);
            self.pending_wakes.remove(&thread.get());
            result(context, address_wait_result(completion));
            return resume();
        }
        if context
            .process()
            .address_waits()
            .lock()
            .contains(mutex, thread)
        {
            return ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        }
        let word = match AddressWord::capture(context, mutex, MemoryPermissions::READ) {
            Ok(word) => word,
            Err(error) => return address_word_fault(context, 0x1a, std::sync::Arc::new(error)),
        };
        let owner = context
            .process()
            .handles()
            .lookup_as::<ThreadObject>(owner_handle);
        let waits = context.process().address_waits().clone();
        let clock = self.virtual_clock.clone();
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime
            .begin()
            .expect("new mutex acquisition is not cancelled");
        self.start_owned_svc(context, lifetime, None, async move {
            let (code, event) = {
                let _operation = waits.address_operation(mutex, thread, &clock).await;
                // Read and register under the same kernel exclusion as unlock.
                // A pending release must finish before this read can select
                // the old owner. Eden WaitForAddress uses this same order.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_condition_variable.cpp
                match word.read_owned().await {
                    Err(error) if error.is_access_fault() => {
                        (Some(HorizonKernelResult::INVALID_CURRENT_MEMORY), None)
                    }
                    Err(fault) => {
                        return Err(HorizonSvcFault::MemoryAccess {
                            immediate: 0x1a,
                            fault,
                        });
                    }
                    Ok(observed) if observed != (owner_handle | MUTEX_WAITERS) => {
                        (Some(HorizonKernelResult::SUCCESS), None)
                    }
                    Ok(_) => match owner {
                        None => (Some(HorizonKernelResult::INVALID_HANDLE), None),
                        Some(owner) => {
                            let mut state = waits.lock();
                            (
                                None,
                                Some(
                                    state
                                        .enqueue(
                                            mutex,
                                            GuestThreadId::new(owner.thread_id()),
                                            thread,
                                            tag,
                                        )
                                        .map_err(|fault| HorizonSvcFault::Scheduling {
                                            operation: "ArbitrateLock",
                                            fault,
                                        })?,
                                ),
                            )
                        }
                    },
                }
            };
            Ok(memory_wait::SvcCompletion::kernel(
                source,
                vcpu,
                move |dispatcher, context| {
                    if let Some(code) = code {
                        result(context, code);
                        resume()
                    } else {
                        dispatcher.pending_wakes.insert(
                            thread.get(),
                            PendingThreadWake {
                                events: vec![event.expect("a contended mutex retains its wake")],
                                deadline: None,
                            },
                        );
                        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
                    }
                },
            ))
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
        let clock = self.virtual_clock.clone();
        let source = context.source();
        let vcpu = context.thread().vcpu();
        let lifetime = std::sync::Arc::new(crate::ipc_wire::operation::CallLifetime::default());
        lifetime
            .begin()
            .expect("new kernel handoff is not cancelled");
        self.start_owned_svc(context, lifetime, None, async move {
            let written = {
                let _operation = waits.address_operation(mutex, thread, &clock).await;
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
    clock: &nixe_runtime::VirtualClock,
    immediate: u32,
    current: GuestThreadId,
) -> Result<(), HorizonSvcFault> {
    use nixe_runtime::ConditionAcquire;
    let _operation = waits
        .address_operation(claim.mutex, claim.thread, clock)
        .await;
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
