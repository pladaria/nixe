//! Physical allocation operations share Eden's map-physical-memory lock.
//! Allocation waits release the host worker while retaining the guest operation.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/kernel/k_page_table_base.cpp
use super::*;
use std::{
    future::{Future, poll_fn},
    ops::Range,
    pin::Pin,
    task::{Context, Poll, Waker},
};

#[derive(Default)]
pub(super) struct PhysicalRequests {
    next: u64,
    active: Option<u64>,
    waiters: BTreeMap<u64, Arc<Waker>>,
}
impl PhysicalRequests {
    pub(super) fn wakers(&self) -> Vec<Arc<Waker>> {
        self.waiters.values().cloned().collect()
    }
}
pub(super) struct PhysicalMemoryLock {
    memory: Arc<ProcessMemory>,
    ticket: Option<u64>,
}
impl PhysicalMemoryLock {
    fn check_open(&self) -> Result<(), MappingError> {
        if self.memory.closed.load(Ordering::Acquire) {
            Err(MemoryError::Invalid("process memory is closed").into())
        } else {
            Ok(())
        }
    }
    fn claim(&mut self, context: &Context<'_>) -> Poll<Result<(), MappingError>> {
        self.check_open()?;
        let incoming = context.waker().clone();
        let mut requests = self.memory.physical_requests.lock().unwrap();
        let ticket = match self.ticket {
            Some(ticket) => ticket,
            None => {
                requests.next = requests
                    .next
                    .checked_add(1)
                    .ok_or(MemoryError::Invalid("physical request identity exhausted"))?;
                self.ticket = Some(requests.next);
                requests.next
            }
        };
        if requests.active.is_none()
            && requests
                .waiters
                .first_key_value()
                .is_none_or(|(&first, _)| first == ticket)
        {
            requests.active = Some(ticket);
        }
        let acquired = requests.active == Some(ticket);
        let previous = if acquired {
            None
        } else {
            requests.waiters.insert(ticket, Arc::new(incoming))
        };
        drop(requests);
        drop(previous);
        self.check_open()?;
        if acquired {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
    pub(super) async fn allocate(&self, bytes: u64) -> Result<Allocation, MappingError> {
        let mut allocation = self.memory.account.allocate_async(bytes);
        poll_fn(|context| {
            self.check_open()?;
            match Pin::new(&mut allocation).poll(context) {
                Poll::Ready(result) => Poll::Ready(result.map_err(MappingError::Memory)),
                Poll::Pending => {
                    let incoming = Arc::new(context.waker().clone());
                    let previous = self
                        .memory
                        .physical_requests
                        .lock()
                        .unwrap()
                        .waiters
                        .insert(self.ticket.expect("active physical memory owner"), incoming);
                    drop(previous);
                    // close() may have signalled the old executor before this
                    // registration. Observe it after installing the new waker.
                    self.check_open()?;
                    Poll::Pending
                }
            }
        })
        .await
    }
}
impl Drop for PhysicalMemoryLock {
    fn drop(&mut self) {
        let Some(ticket) = self.ticket.take() else {
            return;
        };
        let mut requests = self.memory.physical_requests.lock().unwrap();
        let removed = requests.waiters.remove(&ticket);
        if requests.active == Some(ticket) {
            requests.active = None;
        }
        let next = if requests.active.is_none() {
            requests
                .waiters
                .first_key_value()
                .map(|(_, wake)| Arc::clone(wake))
        } else {
            None
        };
        drop(requests);
        drop(removed);
        if let Some(wake) = next {
            wake.wake_by_ref();
        }
    }
}
impl ProcessMemory {
    pub(super) async fn lock_physical(
        self: &Arc<Self>,
    ) -> Result<PhysicalMemoryLock, MappingError> {
        let mut lock = PhysicalMemoryLock {
            memory: Arc::clone(self),
            ticket: None,
        };
        poll_fn(|context| lock.claim(context)).await?;
        Ok(lock)
    }
    fn physical_holes(regions: &BTreeMap<u64, Region>, start: u64, end: u64) -> Vec<Range<u64>> {
        let mut cursor = start;
        let mut holes = Vec::new();
        let first = regions
            .range(..=start)
            .next_back()
            .map_or(start, |(&base, _)| base);
        for (&base, region) in regions.range(first..end) {
            if region.end <= cursor {
                continue;
            }
            if base > cursor {
                holes.push(cursor..base);
            }
            cursor = region.end;
        }
        if cursor < end {
            holes.push(cursor..end);
        }
        holes
    }
    /// Fill only free pages; already mapped intervals keep their bytes, state,
    /// physical identity and permissions, as KPageTableBase::MapPhysicalMemory.
    pub async fn map_physical(
        self: &Arc<Self>,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        let physical = self.lock_physical().await?;
        loop {
            let needed: u64 = {
                let regions = self.regions.lock().unwrap();
                Self::physical_holes(&regions, start.get(), end)
                    .iter()
                    .map(|range| range.end - range.start)
                    .sum()
            };
            if needed == 0 {
                return Ok(());
            }
            let allocation = physical.allocate(needed).await?;
            let mut regions = self.regions.lock().unwrap();
            let holes = Self::physical_holes(&regions, start.get(), end);
            if holes
                .iter()
                .map(|range| range.end - range.start)
                .sum::<u64>()
                != needed
            {
                // Other page-table operations may have changed the intervals
                // during physical allocation. Eden retries under its table lock.
                continue;
            }
            let mapping = Mapping {
                kind: MemoryKind::Heap,
                permissions: MemoryPermissions::READ_WRITE,
                uncached: false,
                locked: false,
                permission_locked: false,
            };
            let mut offset = 0;
            for (index, hole) in holes.iter().enumerate() {
                let size = hole.end - hole.start;
                if let Err(error) = self.space.map(
                    GuestVirtualAddress::new(hole.start),
                    allocation.clone(),
                    offset,
                    size,
                    mapping.permissions,
                ) {
                    // Undo just the pages created by this operation; existing
                    // mappings and their contents are never rolled back.
                    for mapped in &holes[..index] {
                        self.space.unmap(
                            GuestVirtualAddress::new(mapped.start),
                            mapped.end - mapped.start,
                        )?;
                        Self::remove(&mut regions, mapped.start, mapped.end);
                    }
                    return Err(error.into());
                }
                Self::insert(
                    &mut regions,
                    hole.start,
                    Region {
                        end: hole.end,
                        mapping,
                    },
                );
                offset += size;
            }
            return Ok(());
        }
    }
    /// Release normal pages and allow holes, but reject locked or non-normal
    /// mappings before changing the page table, as Eden UnmapPhysicalMemory.
    pub async fn unmap_physical(
        self: &Arc<Self>,
        start: GuestVirtualAddress,
        bytes: u64,
    ) -> Result<(), MappingError> {
        let end = self.end(start, bytes)?;
        let _physical = self.lock_physical().await?;
        let mut regions = self.regions.lock().unwrap();
        let mut mapped = Vec::new();
        let first = regions
            .range(..=start.get())
            .next_back()
            .map_or(start.get(), |(&base, _)| base);
        for (&base, region) in regions.range(first..end) {
            if region.end <= start.get() {
                continue;
            }
            let state = region.mapping;
            if state.kind != MemoryKind::Heap
                || state.locked
                || state.uncached
                || state.permission_locked
            {
                return Err(MappingError::StateMismatch);
            }
            mapped.push(base.max(start.get())..region.end.min(end));
        }
        for range in mapped {
            self.space.unmap(
                GuestVirtualAddress::new(range.start),
                range.end - range.start,
            )?;
            Self::remove(&mut regions, range.start, range.end);
        }
        Ok(())
    }
}
