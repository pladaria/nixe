//! Short physical access guards also cover inputs retained after VA removal.
use crate::Extent;
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicUsize, Ordering},
};

const REGION_SHIFT: u32 = 21;
const CLOSED: usize = 1 << (usize::BITS - 1);

pub(crate) struct PhysicalAccess {
    regions: Box<[AtomicUsize]>,
    notification: Mutex<()>,
    changed: Condvar,
}
pub(crate) struct PhysicalGuard<'a> {
    access: &'a PhysicalAccess,
    first: usize,
    end: usize,
}
pub(crate) struct ClosedPhysical<'a> {
    access: &'a PhysicalAccess,
    regions: Vec<usize>,
}
impl PhysicalAccess {
    pub fn new(bytes: u64) -> Self {
        Self {
            regions: (0..bytes.div_ceil(1 << REGION_SHIFT))
                .map(|_| AtomicUsize::new(0))
                .collect(),
            notification: Mutex::new(()),
            changed: Condvar::new(),
        }
    }
    pub fn try_enter(&self, extent: Extent) -> Option<PhysicalGuard<'_>> {
        let first = (extent.start.get() >> REGION_SHIFT) as usize;
        let end = (extent.start.get() + extent.bytes).div_ceil(1 << REGION_SHIFT) as usize;
        let mut guard = PhysicalGuard {
            access: self,
            first,
            end: first,
        };
        for region in first..end {
            if self.regions[region]
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                    (count & CLOSED == 0).then_some(count + 1)
                })
                .is_err()
            {
                return None;
            }
            guard.end = region + 1;
        }
        Some(guard)
    }
    pub fn enter(&self, extent: Extent) -> PhysicalGuard<'_> {
        let first = (extent.start.get() >> REGION_SHIFT) as usize;
        let end = (extent.start.get() + extent.bytes).div_ceil(1 << REGION_SHIFT) as usize;
        loop {
            if let Some(guard) = self.try_enter(extent) {
                return guard;
            }
            let mut notification = self.notification.lock().unwrap();
            while self.regions[first..end]
                .iter()
                .any(|region| region.load(Ordering::Acquire) & CLOSED != 0)
            {
                notification = self.changed.wait(notification).unwrap();
            }
        }
    }
    /// Retain all selected spans before inspecting tracking or modifying RAM.
    /// Never park with partial admission: a protection writer may have closed
    /// both the acquired and the contested regions before draining them.
    pub fn enter_many(&self, extents: &[Extent]) -> smallvec::SmallVec<[PhysicalGuard<'_>; 4]> {
        let mut guards: smallvec::SmallVec<[PhysicalGuard<'_>; 4]> = smallvec::SmallVec::new();
        'retry: loop {
            for &extent in extents {
                if guards.last().is_some_and(|guard| {
                    guard.first <= (extent.start.get() >> REGION_SHIFT) as usize
                        && guard.end
                            >= (extent.start.get() + extent.bytes).div_ceil(1 << REGION_SHIFT)
                                as usize
                }) {
                    continue;
                }
                if let Some(guard) = self.try_enter(extent) {
                    guards.push(guard);
                } else {
                    guards.clear();
                    drop(self.enter(extent));
                    continue 'retry;
                }
            }
            return guards;
        }
    }
    /// Caller holds the machine mapping/protection writer. Closing all regions
    /// precedes waiting; admission never holds a partial set while parked.
    pub fn close(&self, extents: &[Extent]) -> ClosedPhysical<'_> {
        let mut regions = Vec::new();
        for extent in extents {
            let first = extent.start.get() >> REGION_SHIFT;
            let end = (extent.start.get() + extent.bytes).div_ceil(1 << REGION_SHIFT);
            regions.extend((first..end).map(|index| index as usize));
        }
        regions.sort_unstable();
        regions.dedup();
        for &region in &regions {
            let before = self.regions[region].fetch_or(CLOSED, Ordering::AcqRel);
            debug_assert_eq!(before & CLOSED, 0);
        }
        ClosedPhysical {
            access: self,
            regions,
        }
    }
    fn notify(&self) {
        let _notification = self.notification.lock().unwrap();
        self.changed.notify_all();
    }
}
impl PhysicalGuard<'_> {
    pub(crate) fn covers(&self, extent: Extent) -> bool {
        self.first <= (extent.start.get() >> REGION_SHIFT) as usize
            && self.end >= (extent.start.get() + extent.bytes).div_ceil(1 << REGION_SHIFT) as usize
    }

    fn release(&mut self) {
        for region in self.first..self.end {
            let before = self.access.regions[region].fetch_sub(1, Ordering::Release);
            if before & CLOSED != 0 {
                self.access.notify();
            }
        }
        self.end = self.first;
    }
}
impl Drop for PhysicalGuard<'_> {
    fn drop(&mut self) {
        self.release();
    }
}
impl ClosedPhysical<'_> {
    pub fn wait(&self) {
        let mut notification = self.access.notification.lock().unwrap();
        while self
            .regions
            .iter()
            .any(|&index| self.access.regions[index].load(Ordering::Acquire) != CLOSED)
        {
            notification = self.access.changed.wait(notification).unwrap();
        }
    }
}
impl Drop for ClosedPhysical<'_> {
    fn drop(&mut self) {
        for &region in &self.regions {
            self.access.regions[region].fetch_and(!CLOSED, Ordering::Release);
        }
        self.access.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_admission_keeps_holes_open_and_releases_every_region() {
        let bytes = 1 << REGION_SHIFT;
        let access = PhysicalAccess::new(3 * bytes);
        let extent = |region| Extent {
            start: crate::ArenaOffset::new(region * bytes),
            bytes,
        };
        {
            let _guards = access.enter_many(&[extent(0), extent(0), extent(2)]);
            assert_eq!(access.regions[0].load(Ordering::Acquire), 1);
            assert_eq!(access.regions[1].load(Ordering::Acquire), 0);
            assert_eq!(access.regions[2].load(Ordering::Acquire), 1);
            let hole = access.close(&[extent(1)]);
            hole.wait();
        }
        let closed = access.close(&[extent(0), extent(2)]);
        closed.wait();
    }

    #[test]
    fn fragmented_admission_does_not_hold_a_prefix_while_waiting() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;
        let bytes = 1 << REGION_SHIFT;
        let access = Arc::new(PhysicalAccess::new(2 * bytes));
        let first = Extent {
            start: crate::ArenaOffset::new(0),
            bytes,
        };
        let second = Extent {
            start: crate::ArenaOffset::new(bytes),
            bytes,
        };
        let blocked = access.close(&[second]);
        let (started, start) = mpsc::channel();
        let (done, completion) = mpsc::channel();
        let reader_access = access.clone();
        let reader = std::thread::spawn(move || {
            started.send(()).unwrap();
            let _guards = reader_access.enter_many(&[first, second]);
            done.send(()).unwrap();
        });
        start.recv().unwrap();
        assert!(matches!(
            completion.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let (drained, drain) = mpsc::channel();
        let writer_access = access.clone();
        let writer = std::thread::spawn(move || {
            let closed = writer_access.close(&[first]);
            closed.wait();
            drained.send(()).unwrap();
        });
        let released_prefix = drain.recv_timeout(Duration::from_secs(2));
        drop(blocked);
        reader.join().unwrap();
        writer.join().unwrap();
        released_prefix.expect("blocked fragment must release earlier admission");
        completion.recv().unwrap();
    }

    #[test]
    fn mapped_ram_reader_releases_all_prior_regions_before_waiting() {
        use std::sync::mpsc;
        use std::time::Duration;
        let bytes = 1 << REGION_SHIFT;
        let machine = crate::Machine::new(3 * bytes).unwrap();
        let allocation = machine
            .account(3 * bytes)
            .unwrap()
            .allocate(3 * bytes)
            .unwrap();
        let first = allocation.physical_extents(0, 16).unwrap().next().unwrap();
        let second = allocation
            .physical_extents(bytes, 16)
            .unwrap()
            .next()
            .unwrap();
        let third = allocation
            .physical_extents(2 * bytes, 16)
            .unwrap()
            .next()
            .unwrap();
        let blocked = machine.physical_access().close(&[third]);
        let (started, start) = mpsc::channel();
        let reader_machine = machine.clone();
        let reader = std::thread::spawn(move || {
            let _allocation = allocation;
            let mut reader = reader_machine.mapped_ram_reader();
            let mut output = [0; 16];
            // SAFETY: the allocation retains all three disjoint physical regions.
            unsafe {
                assert!(reader.try_read(first.start, &mut output).unwrap());
                assert!(reader.try_read(second.start, &mut output).unwrap());
                started.send(()).unwrap();
                assert!(reader.try_read(third.start, &mut output).unwrap());
            }
        });
        start.recv().unwrap();
        let (drained, drain) = mpsc::channel();
        let writer_machine = machine.clone();
        let writer = std::thread::spawn(move || {
            let closed = writer_machine.physical_access().close(&[first, second]);
            closed.wait();
            drained.send(()).unwrap();
        });
        let released_previous = drain.recv_timeout(Duration::from_secs(2));
        // Always unblock and join both threads, including a failed regression.
        drop(blocked);
        reader.join().unwrap();
        writer.join().unwrap();
        released_previous.expect("a reader waiting on another region must release all old guards");
    }

    #[test]
    fn mapped_ram_reader_refreshes_bytes_across_retained_regions() {
        let bytes = 1 << REGION_SHIFT;
        let machine = crate::Machine::new(5 * bytes).unwrap();
        let allocation = machine
            .account(5 * bytes)
            .unwrap()
            .allocate(5 * bytes)
            .unwrap();
        let starts: Vec<_> = (0..5)
            .map(|region| {
                allocation
                    .physical_extents(region * bytes, 16)
                    .unwrap()
                    .next()
                    .unwrap()
                    .start
            })
            .collect();
        let mut reader = machine.mapped_ram_reader();
        let mut output = [0; 16];
        for revision in [0x31, 0x42] {
            for index in [0, 1, 2, 0, 3, 4, 0, 2] {
                let start = starts[index];
                let range = crate::AllocationRange::new(allocation.pin(), index as u64 * bytes, 16)
                    .unwrap();
                machine.write_resolved(&[range], &[revision; 16]).unwrap();
                // SAFETY: the allocation retains every physical region.
                assert!(unsafe { reader.try_read(start, &mut output) }.unwrap());
                assert_eq!(output, [revision; 16]);
            }
        }
        reader.release();
        let extents = starts
            .into_iter()
            .map(|start| Extent { start, bytes: 16 })
            .collect::<Vec<_>>();
        let closed = machine.physical_access().close(&extents);
        closed.wait();
    }

    #[test]
    fn nonblocking_cross_region_access_releases_its_partial_admission() {
        let bytes = 1 << REGION_SHIFT;
        let access = PhysicalAccess::new(bytes * 2);
        let first = Extent {
            start: crate::ArenaOffset::new(0),
            bytes,
        };
        let second = Extent {
            start: crate::ArenaOffset::new(bytes),
            bytes,
        };
        let closed = access.close(&[second]);
        assert!(
            access
                .try_enter(Extent {
                    start: first.start,
                    bytes: bytes * 2
                })
                .is_none()
        );
        assert_eq!(access.regions[0].load(Ordering::Acquire), 0);
        let first_closed = access.close(&[first]);
        first_closed.wait();
        closed.wait();
        drop(first_closed);
        drop(closed);
        assert!(
            access
                .try_enter(Extent {
                    start: first.start,
                    bytes: bytes * 2
                })
                .is_some()
        );
    }
}
