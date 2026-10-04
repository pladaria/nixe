use super::*;

#[derive(Clone, Debug)]
enum ReplyWaitTarget {
    Port(PortObject),
    Session(SessionObject),
}

impl HorizonSvcDispatcher {
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
        let handles = match read_handle_array(context, handles_address, count, immediate) {
            Ok(handles) => handles,
            Err(outcome) => return outcome,
        };
        let mut targets = Vec::with_capacity(handles.len());
        for handle in handles {
            if let Some(port) = context
                .process()
                .handles()
                .get_as::<PortObject>(handle)
                .cloned()
                && port.endpoint() == PortEndpoint::Server
            {
                targets.push(ReplyWaitTarget::Port(port));
                continue;
            }
            if let Some(session) = context
                .process()
                .handles()
                .get_as::<SessionObject>(handle)
                .cloned()
                && session.endpoint() == SessionEndpoint::Server
                && !session.is_light()
            {
                targets.push(ReplyWaitTarget::Session(session));
                continue;
            }
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        }

        let thread_id = context.thread().object().thread_id();
        if self.cancelled_waits.remove(&thread_id) {
            self.finish_reply_wait(thread_id, immediate);
            result(context, HorizonKernelResult::CANCELLED);
            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
            return resume();
        }
        if reply_target != 0 && !self.reply_sent.contains(&thread_id) {
            let Some(session) = context
                .process()
                .handles()
                .get_as::<SessionObject>(reply_target)
                .cloned()
            else {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            };
            if session.endpoint() != SessionEndpoint::Server || session.is_light() {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            }
            let reply = match read_guest_message(context, address, size, immediate) {
                Ok(reply) => reply,
                Err(outcome) => {
                    write_register(context.thread_mut().state_mut(), 1, u64::from(u32::MAX));
                    return outcome;
                }
            };
            let reply = match capture_message_handles(context, reply, true) {
                Ok(reply) => reply,
                Err(code) => {
                    result(context, code);
                    write_register(context.thread_mut().state_mut(), 1, u64::from(u32::MAX));
                    return resume();
                }
            };
            if let Err(error) = session.reply(reply) {
                write_register(context.thread_mut().state_mut(), 1, u64::from(u32::MAX));
                return session_error(context, error);
            }
            self.reply_sent.insert(thread_id);
        } else if reply_target == 0 {
            self.reply_sent.remove(&thread_id);
        }

        for (index, target) in targets.iter().enumerate() {
            match target {
                ReplyWaitTarget::Port(port) if port.is_signalled() => {
                    self.finish_reply_wait(thread_id, immediate);
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
                            let request = match materialize_message_handles(context, message) {
                                Ok(Some(request)) => request,
                                Ok(None) => unreachable!("buffer message materializes as bytes"),
                                Err(code) => {
                                    self.finish_reply_wait(thread_id, immediate);
                                    result(context, code);
                                    return resume();
                                }
                            };
                            if request.len() > size {
                                close_encoded_handles(
                                    context.process_mut().handles_mut(),
                                    &request,
                                );
                                self.finish_reply_wait(thread_id, immediate);
                                result(context, HorizonKernelResult::INVALID_SIZE);
                                return resume();
                            }
                            if let Err(error) =
                                crate::ipc_wire::write_bytes(context.process(), address, &request)
                            {
                                close_encoded_handles(
                                    context.process_mut().handles_mut(),
                                    &request,
                                );
                                self.finish_reply_wait(thread_id, immediate);
                                return reject_ipc(context, immediate, error);
                            }
                            self.finish_reply_wait(thread_id, immediate);
                            result(context, HorizonKernelResult::SUCCESS);
                            write_register(context.thread_mut().state_mut(), 1, index as u64);
                            return resume();
                        }
                        Ok(SessionMessage::Light(_)) | Err(SessionError::MessageKindMismatch) => {
                            self.finish_reply_wait(thread_id, immediate);
                            result(context, HorizonKernelResult::INVALID_STATE);
                            return resume();
                        }
                        Err(SessionError::PeerClosed) => {
                            self.finish_reply_wait(thread_id, immediate);
                            result(context, HorizonKernelResult::SESSION_CLOSED);
                            write_register(context.thread_mut().state_mut(), 1, index as u64);
                            return resume();
                        }
                        Err(SessionError::NoRequest) => {}
                        Err(error) => {
                            self.finish_reply_wait(thread_id, immediate);
                            return session_error(context, error);
                        }
                    }
                }
                ReplyWaitTarget::Port(_) | ReplyWaitTarget::Session(_) => {}
            }
        }

        if self.wait_expired(thread_id, immediate, timeout) {
            self.finish_reply_wait(thread_id, immediate);
            result(context, HorizonKernelResult::TIMED_OUT);
            resume()
        } else {
            if self.pending_wait_cancellations.remove(&thread_id) {
                self.finish_reply_wait(thread_id, immediate);
                result(context, HorizonKernelResult::CANCELLED);
                write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
                return resume();
            }
            self.cancellable_waits.insert(thread_id);
            let mut events: Vec<_> = targets
                .iter()
                .map(|target| match target {
                    ReplyWaitTarget::Port(port) => port.readable_event(),
                    ReplyWaitTarget::Session(session) => session.readable_event(),
                })
                .collect();
            let handles_changed = context.process().handles().changed_event();
            handles_changed.clear();
            events.push(handles_changed);
            for event in &events {
                event.clear();
            }
            let became_ready = targets.iter().any(|target| match target {
                ReplyWaitTarget::Port(port) => port.is_signalled(),
                ReplyWaitTarget::Session(session) => session.is_signalled(),
            });
            if !became_ready && !events.is_empty() {
                let deadline = self.wait_deadlines.get(&(thread_id, immediate)).copied();
                self.pending_wakes
                    .insert(thread_id, PendingThreadWake { events, deadline });
            }
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
        }
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
        self.reply_sent.remove(&thread_id);
        self.finish_wait(thread_id, immediate);
    }

    pub(super) fn finish_wait(&mut self, thread_id: u64, immediate: u32) {
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
            .get_as::<SessionObject>(handle)
            .cloned()
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
        let thread_id = context.thread().object().thread_id();
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
    handles: &mut HandleTable,
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
            .get_as::<ProcessObject>(handle)
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
    let thread_id = if handle == CURRENT_THREAD_HANDLE || handle == context.thread().handle() {
        Some(context.thread().object().thread_id())
    } else {
        context
            .process()
            .handles()
            .get_as::<ThreadObject>(handle)
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
        let thread_id = context.thread().object().thread_id();
        if self.cancelled_waits.remove(&thread_id) {
            self.finish_wait(thread_id, 0x18);
            result(context, HorizonKernelResult::CANCELLED);
            write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
            return resume();
        }
        let pointer = read_register(context.thread().state(), 1);
        let count = read_register(context.thread().state(), 2) as u32;
        let timeout = read_wait_timeout(context.thread().state());
        if count > MAX_WAIT_HANDLES {
            result(context, HorizonKernelResult::OUT_OF_RANGE);
            return resume();
        }
        let mut handles = Vec::with_capacity(count as usize);
        for index in 0..count {
            let Some(address) = pointer.checked_add(u64::from(index) * 4) else {
                result(context, HorizonKernelResult::INVALID_ADDRESS);
                return resume();
            };
            let value = match context.process().memory().read(
                context.process().cpu().address_space_id(),
                GuestVirtualAddress::new(address),
                MemoryAccess::normal(MemoryAccessSize::Word),
            ) {
                Ok(read) => read.value,
                Err(_) => {
                    result(context, HorizonKernelResult::INVALID_ADDRESS);
                    return resume();
                }
            };
            let MemoryValue::U32(handle) = value else {
                unreachable!("word access returns a word value")
            };
            handles.push(handle);
        }
        let mut events = Vec::with_capacity(handles.len() + 1);
        for (index, handle) in handles.iter().copied().enumerate() {
            let event = context
                .process()
                .handles()
                .get_as::<ReadableEventObject>(handle)
                .cloned()
                .or_else(|| {
                    context
                        .process()
                        .handles()
                        .get_as::<ThreadObject>(handle)
                        .map(ThreadObject::readable_event)
                });
            let Some(event) = event else {
                result(context, HorizonKernelResult::INVALID_HANDLE);
                return resume();
            };
            if event.is_signalled() {
                self.finish_wait(context.thread().object().thread_id(), 0x18);
                result(context, HorizonKernelResult::SUCCESS);
                write_register(context.thread_mut().state_mut(), 1, index as u64);
                return resume();
            }
            events.push(event);
        }
        let handles_changed = context.process().handles().changed_event();
        handles_changed.clear();
        events.push(handles_changed);
        let thread_id = context.thread().object().thread_id();
        if self.wait_expired(thread_id, 0x18, timeout) {
            self.finish_wait(thread_id, 0x18);
            result(context, HorizonKernelResult::TIMED_OUT);
            resume()
        } else {
            // Readiness and zero-timeout polling precede a latched cancellation.
            // https://github.com/Atmosphere-NX/Atmosphere/blob/master/libraries/libmesosphere/source/kern_k_synchronization_object.cpp#L98-L117
            if self.pending_wait_cancellations.remove(&thread_id) {
                self.finish_wait(thread_id, 0x18);
                result(context, HorizonKernelResult::CANCELLED);
                write_register(context.thread_mut().state_mut(), 1, u32::MAX as u64);
                return resume();
            }
            self.cancellable_waits.insert(thread_id);
            if !events.is_empty() {
                let deadline = self.wait_deadlines.get(&(thread_id, 0x18)).copied();
                self.pending_wakes
                    .insert(thread_id, PendingThreadWake { events, deadline });
            }
            ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
        }
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
                .is_signalled(address, thread)
            {
                let completion = context
                    .process()
                    .address_waits()
                    .completion(address, thread)
                    .unwrap();
                context
                    .process_mut()
                    .address_waits_mut()
                    .remove(address, thread);
                self.finish_wait(thread.get(), 0x1c);
                result(context, address_wait_result(completion));
                return resume();
            }
        }
        let waiting = context.process().address_waits().contains(key, thread)
            || context.process().address_waits().contains(mutex, thread);
        if waiting {
            if self.wait_expired(thread.get(), 0x1c, timeout) {
                context
                    .process_mut()
                    .address_waits_mut()
                    .remove(key, thread);
                context
                    .process_mut()
                    .address_waits_mut()
                    .remove(mutex, thread);
                self.finish_wait(thread.get(), 0x1c);
                if let Err(fault) = self.synchronize_mutex_waits(context, vec![mutex]) {
                    return ExceptionDispatchOutcome::Fault(fault);
                }
                result(context, HorizonKernelResult::TIMED_OUT);
                return resume();
            }
            return ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        }
        if let Err(fault) = write_process_wide_key_word(context, key, 1)
            && process_wide_key_fault_is_internal(&fault)
        {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::GuestMemory {
                immediate: 0x1c,
                fault,
            });
        }
        let released = handoff_mutex(context, mutex, false);
        if let Err(fault) = self.synchronize_mutex_waits(context, vec![mutex]) {
            return ExceptionDispatchOutcome::Fault(fault);
        }
        if let Err(fault) = released {
            return mutex_memory_fault(context, 0x1c, fault);
        }
        if timeout == 0 {
            result(context, HorizonKernelResult::TIMED_OUT);
            return resume();
        }
        let event = context
            .process_mut()
            .address_waits_mut()
            .enqueue_condition(key, thread, tag, mutex);
        self.wait_expired(thread.get(), 0x1c, timeout);
        let deadline = self.wait_deadlines.get(&(thread.get(), 0x1c)).copied();
        self.pending_wakes.insert(
            thread.get(),
            PendingThreadWake {
                events: vec![event],
                deadline,
            },
        );
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
    }

    pub(super) fn signal_process_wide_key(
        &mut self,
        context: &mut ExceptionDispatchContext<'_>,
    ) -> ExceptionDispatchOutcome<HorizonSvcFault> {
        use nixe_runtime::AddressWaitCompletion;
        let key = read_register(context.thread().state(), 0) & !3;
        let count = read_register(context.thread().state(), 1) as u32 as i32;
        let limit = if count <= 0 {
            usize::MAX
        } else {
            count as usize
        };
        let mut changed = Vec::new();
        for _ in 0..limit {
            let Some((thread, tag, mutex)) = context.process().address_waits().next_condition(key)
            else {
                break;
            };
            // Signal reacquires each mutex now, rather than letting resumed
            // waiters overwrite a mutex that another thread may still own.
            match update_condition_mutex(context, mutex, tag) {
                Ok(0) => {
                    context
                        .process_mut()
                        .address_waits_mut()
                        .set_owner(mutex, thread);
                    context.process_mut().address_waits_mut().signal_result(
                        key,
                        thread,
                        AddressWaitCompletion::Success,
                    );
                    changed.push(mutex);
                }
                Ok(previous) => {
                    let owner = context
                        .process()
                        .handles()
                        .get_as::<ThreadObject>(previous & !MUTEX_WAITERS)
                        .map(ThreadObject::thread_id);
                    if let Some(owner) = owner {
                        let waits = context.process_mut().address_waits_mut();
                        waits.set_owner(mutex, GuestThreadId::new(owner));
                        waits.move_waiter(key, mutex, thread);
                        changed.push(mutex);
                    } else {
                        context.process_mut().address_waits_mut().signal_result(
                            key,
                            thread,
                            AddressWaitCompletion::InvalidOwner,
                        );
                    }
                }
                Err(fault) => {
                    if process_wide_key_fault_is_internal(&fault) {
                        return ExceptionDispatchOutcome::Fault(HorizonSvcFault::GuestMemory {
                            immediate: 0x1d,
                            fault,
                        });
                    }
                    context.process_mut().address_waits_mut().signal_result(
                        key,
                        thread,
                        AddressWaitCompletion::InvalidMemory,
                    );
                }
            }
        }
        if context
            .process()
            .address_waits()
            .next_condition(key)
            .is_none()
            && let Err(fault) = write_process_wide_key_word(context, key, 0)
            && process_wide_key_fault_is_internal(&fault)
        {
            return ExceptionDispatchOutcome::Fault(HorizonSvcFault::GuestMemory {
                immediate: 0x1d,
                fault,
            });
        }
        changed.sort_unstable();
        changed.dedup();
        if !changed.is_empty()
            && let Err(fault) = self.synchronize_mutex_waits(context, changed)
        {
            return ExceptionDispatchOutcome::Fault(fault);
        }
        resume()
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
            .is_signalled(mutex, thread)
        {
            let completion = context
                .process()
                .address_waits()
                .completion(mutex, thread)
                .unwrap();
            context
                .process_mut()
                .address_waits_mut()
                .remove(mutex, thread);
            self.pending_wakes.remove(&thread.get());
            result(context, address_wait_result(completion));
            return resume();
        }
        if context.process().address_waits().contains(mutex, thread) {
            return ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry);
        }
        let observed = match read_process_wide_key_word(context, mutex) {
            Ok(value) => value,
            Err(fault) => return mutex_memory_fault(context, 0x1a, fault),
        };
        // Userspace retries acquisition when the observed owner has changed.
        // The kernel must neither enqueue against the old owner nor steal it.
        if observed != (owner_handle | MUTEX_WAITERS) {
            result(context, HorizonKernelResult::SUCCESS);
            return resume();
        }
        let Some(owner) = context
            .process()
            .handles()
            .get_as::<ThreadObject>(owner_handle)
            .map(ThreadObject::thread_id)
        else {
            result(context, HorizonKernelResult::INVALID_HANDLE);
            return resume();
        };
        let waits = context.process_mut().address_waits_mut();
        waits.set_owner(mutex, GuestThreadId::new(owner));
        let event = waits.enqueue(mutex, thread, tag);
        self.pending_wakes.insert(
            thread.get(),
            PendingThreadWake {
                events: vec![event],
                deadline: None,
            },
        );
        if let Err(fault) = self.synchronize_mutex_waits(context, vec![mutex]) {
            return ExceptionDispatchOutcome::Fault(fault);
        }
        ExceptionDispatchOutcome::Suspend(ExceptionResume::Retry)
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
        let released = handoff_mutex(context, mutex, true);
        if let Err(fault) = self.synchronize_mutex_waits(context, vec![mutex]) {
            return ExceptionDispatchOutcome::Fault(fault);
        }
        if let Err(fault) = released {
            return mutex_memory_fault(context, 0x1b, fault);
        }
        result(context, HorizonKernelResult::SUCCESS);
        resume()
    }

    fn synchronize_mutex_waits(
        &mut self,
        context: &ExceptionDispatchContext<'_>,
        addresses: Vec<u64>,
    ) -> Result<(), HorizonSvcFault> {
        self.queue_runtime_request(
            context.thread().id(),
            PendingRuntimeRequest::SynchronizeMutexWaits { addresses },
            "mutex priority handoff",
        )
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

fn handoff_mutex(
    context: &mut ExceptionDispatchContext<'_>,
    mutex: u64,
    report_write_failure_to_waiter: bool,
) -> Result<(), DataAccessFault> {
    let owns_waiters =
        context.process().address_waits().owner(mutex) == Some(context.thread().id());
    let next = owns_waiters
        .then(|| context.process().address_waits().next_waiter(mutex))
        .flatten();
    let value = next.map_or(0, |(_, tag)| {
        tag | if context.process().address_waits().unsignalled_count(mutex) > 1 {
            MUTEX_WAITERS
        } else {
            0
        }
    });
    let written = write_process_wide_key_word(context, mutex, value);
    let waits = context.process_mut().address_waits_mut();
    if owns_waiters {
        waits.remove_owner(mutex);
    }
    if let Some((thread, _)) = next {
        waits.set_owner(mutex, thread);
        let completion = if report_write_failure_to_waiter && written.is_err() {
            nixe_runtime::AddressWaitCompletion::InvalidMemory
        } else {
            nixe_runtime::AddressWaitCompletion::Success
        };
        waits.signal_result(mutex, thread, completion);
    }
    written
}

fn update_condition_mutex(
    context: &ExceptionDispatchContext<'_>,
    mutex: u64,
    tag: u32,
) -> Result<u32, DataAccessFault> {
    let mut previous = read_process_wide_key_word(context, mutex)?;
    loop {
        let value = if previous == 0 {
            tag
        } else {
            previous | MUTEX_WAITERS
        };
        let observed =
            super::address_arbitration::compare_exchange(context, mutex, previous, value)?;
        if observed == previous {
            return Ok(previous);
        }
        previous = observed;
    }
}

fn mutex_memory_fault(
    context: &mut ExceptionDispatchContext<'_>,
    immediate: u32,
    fault: DataAccessFault,
) -> ExceptionDispatchOutcome<HorizonSvcFault> {
    if process_wide_key_fault_is_internal(&fault) {
        ExceptionDispatchOutcome::Fault(HorizonSvcFault::GuestMemory { immediate, fault })
    } else {
        result(context, HorizonKernelResult::INVALID_CURRENT_MEMORY);
        resume()
    }
}

pub(super) fn read_process_wide_key_word(
    context: &ExceptionDispatchContext<'_>,
    address: u64,
) -> Result<u32, DataAccessFault> {
    context
        .process()
        .memory()
        .read(
            context.process().cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            MemoryAccess::normal(MemoryAccessSize::Word),
        )
        .map(|read| match read.value {
            MemoryValue::U32(value) => value,
            _ => unreachable!("word access returns a word value"),
        })
}

pub(super) fn write_process_wide_key_word(
    context: &ExceptionDispatchContext<'_>,
    address: u64,
    value: u32,
) -> Result<(), DataAccessFault> {
    context
        .process()
        .memory()
        .write(
            context.process().cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(value),
        )
        .map(|_| ())
}

pub(super) fn process_wide_key_fault_is_internal(fault: &DataAccessFault) -> bool {
    matches!(
        fault.reason,
        DataAccessFaultReason::ContentGenerationExhausted | DataAccessFaultReason::HostBacking(_)
    )
}
