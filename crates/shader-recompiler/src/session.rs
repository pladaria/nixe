//! Compact session identities. Node IDs carry an index and a unique session tag;
//! completed artifacts contain neither. Tag allocation happens once per session,
//! never per IR allocation/use, and exhaustion fails instead of reusing identities.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::{CompileError, Result, diagnostics::FailureCategory};

static NEXT_SESSION: AtomicU32 = AtomicU32::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionId(u32);

impl SessionId {
    pub(crate) fn pack(self, index: u32) -> u64 {
        (u64::from(self.0) << 32) | u64::from(index)
    }

    pub(crate) fn index(self, raw: u64) -> Result<usize> {
        if (raw >> 32) as u32 != self.0 {
            return Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Node handle belongs to another compilation session",
            ));
        }
        Ok(raw as u32 as usize)
    }

    pub fn fresh() -> Result<Self> {
        NEXT_SESSION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(Self)
            .map_err(|_| {
                CompileError::new(
                    FailureCategory::ResourceLimit,
                    "Compiler session identity space exhausted",
                )
            })
    }
}

/// A handle can be retained by the caller but is rejected outside its original
/// session, including when another worker has allocated the same program index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgramId(u64);

impl ProgramId {
    pub(crate) fn new(session: SessionId, index: u32) -> Self {
        Self((u64::from(session.0) << 32) | u64::from(index))
    }

    pub(crate) fn index(self, session: SessionId) -> Result<usize> {
        if (self.0 >> 32) as u32 != session.0 {
            return Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Program handle belongs to another compilation session",
            ));
        }
        Ok(self.0 as u32 as usize)
    }
}

/// Owns session-scoped programs. Reset retains allocation capacity. Each worker
/// has its own store; no program storage is shared across mutable compilers.
pub struct ProgramStore<T> {
    session: SessionId,
    programs: Vec<T>,
}

impl<T> ProgramStore<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            session: SessionId::fresh()?,
            programs: Vec::new(),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.session = SessionId::fresh()?;
        self.programs.clear();
        Ok(())
    }

    pub fn insert(&mut self, program: T) -> Result<ProgramId> {
        let index = u32::try_from(self.programs.len()).map_err(|_| {
            CompileError::new(
                FailureCategory::ResourceLimit,
                "Too many programs in one compilation session",
            )
        })?;
        self.programs.push(program);
        Ok(ProgramId::new(self.session, index))
    }

    pub fn get(&self, id: ProgramId) -> Result<&T> {
        self.programs.get(id.index(self.session)?).ok_or_else(|| {
            CompileError::new(
                FailureCategory::Internal,
                "Program handle refers to an unallocated program",
            )
        })
    }

    pub fn get_mut(&mut self, id: ProgramId) -> Result<&mut T> {
        self.programs
            .get_mut(id.index(self.session)?)
            .ok_or_else(|| {
                CompileError::new(
                    FailureCategory::Internal,
                    "Program handle refers to an unallocated program",
                )
            })
    }

    pub fn session(&self) -> SessionId {
        self.session
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_and_other_workers_cannot_resurrect_handles() {
        let mut programs = ProgramStore::new().unwrap();
        let first = programs.insert(17).unwrap();
        let capacity = programs.programs.capacity();
        let mut other = ProgramStore::new().unwrap();
        let other_id = other.insert(99).unwrap();
        assert!(other.get(first).is_err());
        assert!(programs.get(other_id).is_err());
        programs.reset().unwrap();
        assert_eq!(programs.programs.capacity(), capacity);
        let second = programs.insert(18).unwrap();
        assert_ne!(first, second);
        assert!(programs.get(first).is_err());
        assert_eq!(*programs.get(second).unwrap(), 18);
    }
}
