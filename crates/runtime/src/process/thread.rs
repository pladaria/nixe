use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use nixe_cpu::state::ThreadCpuState;
use nixe_memory::GuestVirtualAddress;
use nixe_scheduler::{CoreSet, GuestThreadId, VirtualCpuId};

use crate::{ThreadExit, ThreadObject};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadCreateRequest {
    pub entry: GuestVirtualAddress,
    pub argument: u64,
    pub stack_top: GuestVirtualAddress,
    pub priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadCreation {
    pub id: GuestThreadId,
    pub handle: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadCreateError {
    InvalidEntry,
    InvalidStack,
    InvalidPriority(i32),
    InvalidVirtualCpu(VirtualCpuId),
    PolicyDenied,
    ResourceLimit,
    IdentityExhausted,
    Internal,
}

impl Display for ThreadCreateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "guest thread creation failed: {self:?}")
    }
}

impl Error for ThreadCreateError {}

/// Runtime-owned guest thread state. Scheduler and handle identities remain
/// stable even when the table's storage moves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuestThread {
    pub(crate) id: GuestThreadId,
    pub(crate) object: ThreadObject,
    pub(crate) exit: Option<ThreadExit>,
    pub(crate) state: Option<ThreadCpuState>,
    pub handle: u32,
    pub stack_bottom: GuestVirtualAddress,
    pub stack_top: GuestVirtualAddress,
    pub tls_base: GuestVirtualAddress,
    pub abi_context: Option<GuestVirtualAddress>,
    /// Runtime-owned guest address installed as the original NRO link register.
    pub loader_return: Option<GuestVirtualAddress>,
}

impl GuestThread {
    #[must_use]
    pub const fn id(&self) -> GuestThreadId {
        self.id
    }

    #[must_use]
    pub fn object(&self) -> ThreadObject {
        self.object.clone()
    }

    pub(crate) fn object_identity_reference_count(&self) -> usize {
        self.object.identity_reference_count()
    }

    #[must_use]
    pub const fn exit(&self) -> Option<ThreadExit> {
        self.exit
    }

    #[must_use]
    pub fn state(&self) -> &ThreadCpuState {
        self.state
            .as_ref()
            .expect("thread state is unavailable only while its scheduler lease runs")
    }

    pub fn state_mut(&mut self) -> &mut ThreadCpuState {
        self.state
            .as_mut()
            .expect("thread state is unavailable only while its scheduler lease runs")
    }

    pub(crate) fn take_state(&mut self) -> Option<ThreadCpuState> {
        self.state.take()
    }

    pub(crate) fn restore_state(&mut self, state: ThreadCpuState) {
        assert!(
            self.state.replace(state).is_none(),
            "a scheduler lease restores thread state exactly once"
        );
    }
}

#[derive(Debug, Default)]
pub struct ThreadTable {
    entries: BTreeMap<GuestThreadId, GuestThread>,
}

impl ThreadTable {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, thread: GuestThread) -> Result<GuestThreadId, ThreadTableError> {
        let id = thread.id();
        if self.entries.contains_key(&id) {
            return Err(ThreadTableError::Duplicate(id));
        }
        self.entries.insert(id, thread);
        Ok(id)
    }

    #[must_use]
    pub fn get(&self, id: GuestThreadId) -> Option<&GuestThread> {
        self.entries.get(&id)
    }

    pub fn get_mut(&mut self, id: GuestThreadId) -> Option<&mut GuestThread> {
        self.entries.get_mut(&id)
    }

    pub fn remove(&mut self, id: GuestThreadId) -> Result<GuestThread, ThreadTableError> {
        self.entries
            .remove(&id)
            .ok_or(ThreadTableError::Missing(id))
    }

    /// Reassigns a runtime identity without changing the process-local kernel
    /// object identity exposed through guest handles.
    pub(crate) fn rekey(
        &mut self,
        old: GuestThreadId,
        new: GuestThreadId,
    ) -> Result<(), ThreadTableError> {
        if self.entries.contains_key(&new) {
            return Err(ThreadTableError::Duplicate(new));
        }
        let mut thread = self
            .entries
            .remove(&old)
            .ok_or(ThreadTableError::Missing(old))?;
        thread.id = new;
        self.entries.insert(new, thread);
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&GuestThreadId, &GuestThread)> {
        self.entries.iter()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadTableError {
    Duplicate(GuestThreadId),
    Missing(GuestThreadId),
}

impl Display for ThreadTableError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate(id) => write!(formatter, "guest thread {id} already exists"),
            Self::Missing(id) => write!(formatter, "guest thread {id} does not exist"),
        }
    }
}

impl Error for ThreadTableError {}

#[cfg(test)]
mod tests {
    use super::*;
    fn thread(id: u64) -> GuestThread {
        GuestThread {
            id: GuestThreadId::new(id),
            object: ThreadObject::new(id),
            exit: None,
            state: Some(ThreadCpuState::A64(Box::default())),
            handle: 0,
            stack_bottom: GuestVirtualAddress::new(0),
            stack_top: GuestVirtualAddress::new(0),
            tls_base: GuestVirtualAddress::new(0),
            abi_context: None,
            loader_return: None,
        }
    }

    #[test]
    fn insertion_and_removal_are_failure_atomic() {
        let mut table = ThreadTable::new();
        table.insert(thread(7)).unwrap();
        assert_eq!(
            table.insert(thread(7)),
            Err(ThreadTableError::Duplicate(GuestThreadId::new(7)))
        );
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.remove(GuestThreadId::new(8)),
            Err(ThreadTableError::Missing(GuestThreadId::new(8)))
        );
        assert_eq!(table.len(), 1);
        assert_eq!(
            table.remove(GuestThreadId::new(7)).unwrap().id(),
            GuestThreadId::new(7)
        );
    }
}
