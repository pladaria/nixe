//! Append-only allocation gives stable identities; IR lists own execution order.
//! Reset reuses capacity and changes the session tag. No allocation has an address
//! dependency, and no reference count or lock is needed to refer to a node.

use crate::{CompileError, Result, diagnostics::FailureCategory, session::SessionId};
use std::{marker::PhantomData, num::NonZeroU64};

pub trait ArenaId: Copy {
    fn from_raw(raw: NonZeroU64) -> Self;
    fn raw(self) -> NonZeroU64;
}

macro_rules! ids {
    ($($name:ident),*) => {$ (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(NonZeroU64);
        impl ArenaId for $name {
            fn from_raw(raw: NonZeroU64) -> Self { Self(raw) }
            fn raw(self) -> NonZeroU64 { self.0 }
        }
    )*};
}

ids!(InstId, BlockId, StatementId, AssociatedId, FlowBlockId);

pub struct Arena<T, I> {
    session: SessionId,
    entries: Vec<T>,
    marker: PhantomData<I>,
}

impl<T, I: ArenaId> Arena<T, I> {
    pub(crate) fn new(session: SessionId) -> Self {
        Self {
            session,
            entries: Vec::new(),
            marker: PhantomData,
        }
    }

    pub(crate) fn reset(&mut self, session: SessionId) {
        self.entries.clear();
        self.session = session;
    }

    pub fn allocate(&mut self, value: T) -> Result<I> {
        let index = u32::try_from(self.entries.len()).map_err(|_| {
            CompileError::new(
                FailureCategory::ResourceLimit,
                "Compiler arena exceeds 32-bit node indices",
            )
        })?;
        self.entries.push(value);
        Ok(I::from_raw(
            NonZeroU64::new(self.session.pack(index)).unwrap(),
        ))
    }

    pub fn get(&self, id: I) -> Result<&T> {
        self.entries
            .get(self.session.index(id.raw().get())?)
            .ok_or_else(|| {
                CompileError::new(FailureCategory::Internal, "Unallocated compiler node")
            })
    }

    pub fn get_mut(&mut self, id: I) -> Result<&mut T> {
        self.entries
            .get_mut(self.session.index(id.raw().get())?)
            .ok_or_else(|| {
                CompileError::new(FailureCategory::Internal, "Unallocated compiler node")
            })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
        self.entries.iter().enumerate().map(|(index, value)| {
            (
                I::from_raw(NonZeroU64::new(self.session.pack(index as u32)).unwrap()),
                value,
            )
        })
    }
}
