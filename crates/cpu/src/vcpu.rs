//! Transient resources owned by a host-side virtual CPU executor.
//!
//! These values are intentionally separate from [`crate::state::ThreadCpuState`].
//! They follow an executing vCPU rather than forming the architectural register
//! file of a guest thread, and they are never part of a save-state format.

use core::{
    cell::{Ref, RefCell, RefMut},
    sync::atomic::{AtomicU32, Ordering},
};

use crate::address::{CodeGeneration, GuestPhysicalPageId};

/// Physical reservation recorded by a local exclusive monitor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExclusiveReservation {
    pub page: GuestPhysicalPageId,
    pub byte_offset: u16,
    pub access_size: u8,
    pub generation: CodeGeneration,
}

/// Local exclusive monitor attached to an executing vCPU.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ExclusiveMonitorState {
    reservation: Option<ExclusiveReservation>,
}

impl ExclusiveMonitorState {
    #[must_use]
    pub const fn reservation(self) -> Option<ExclusiveReservation> {
        self.reservation
    }

    pub const fn reserve(&mut self, reservation: ExclusiveReservation) {
        self.reservation = Some(reservation);
    }

    pub const fn clear(&mut self) {
        self.reservation = None;
    }
}

/// Executor-local scheduling and dispatch progress.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DispatchState {
    budget: u64,
    safepoint_requested: bool,
}

impl DispatchState {
    #[must_use]
    pub const fn new(budget: u64) -> Self {
        Self {
            budget,
            safepoint_requested: false,
        }
    }

    #[must_use]
    pub const fn budget(self) -> u64 {
        self.budget
    }

    pub const fn set_budget(&mut self, budget: u64) {
        self.budget = budget;
    }

    #[must_use]
    pub const fn safepoint_requested(self) -> bool {
        self.safepoint_requested
    }

    pub const fn request_safepoint(&mut self) {
        self.safepoint_requested = true;
    }

    pub const fn clear_safepoint(&mut self) {
        self.safepoint_requested = false;
    }
}

/// Non-architectural resources attached to a currently executing vCPU.
///
/// `Tlb` is supplied by the memory subsystem so architectural state does not
/// depend on a particular software-TLB implementation or geometry.
pub struct VcpuExecutionState<Tlb> {
    software_tlb: Tlb,
    exclusive_monitor: RefCell<ExclusiveMonitorState>,
    pending_interrupts: AtomicU32,
    dispatch: DispatchState,
}

impl<Tlb> VcpuExecutionState<Tlb> {
    #[must_use]
    pub const fn new(software_tlb: Tlb, dispatch_budget: u64) -> Self {
        Self {
            software_tlb,
            exclusive_monitor: RefCell::new(ExclusiveMonitorState { reservation: None }),
            pending_interrupts: AtomicU32::new(0),
            dispatch: DispatchState::new(dispatch_budget),
        }
    }

    #[must_use]
    pub const fn software_tlb(&self) -> &Tlb {
        &self.software_tlb
    }

    #[must_use]
    pub const fn software_tlb_mut(&mut self) -> &mut Tlb {
        &mut self.software_tlb
    }

    #[must_use]
    pub fn exclusive_monitor(&self) -> Ref<'_, ExclusiveMonitorState> {
        self.exclusive_monitor.borrow()
    }

    #[must_use]
    pub fn exclusive_monitor_mut(&self) -> RefMut<'_, ExclusiveMonitorState> {
        self.exclusive_monitor.borrow_mut()
    }

    #[must_use]
    pub const fn exclusive_monitor_cell(&self) -> &RefCell<ExclusiveMonitorState> {
        &self.exclusive_monitor
    }

    /// Atomically publishes interrupt/event bits to the executor.
    pub fn post_interrupts(&self, mask: u32) {
        self.pending_interrupts.fetch_or(mask, Ordering::Release);
    }

    /// Takes all pending bits at an executor safepoint.
    pub fn take_pending_interrupts(&self) -> u32 {
        self.pending_interrupts.swap(0, Ordering::AcqRel)
    }

    #[must_use]
    pub const fn dispatch(&self) -> &DispatchState {
        &self.dispatch
    }

    #[must_use]
    pub const fn dispatch_mut(&mut self) -> &mut DispatchState {
        &mut self.dispatch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, Eq, PartialEq)]
    struct TestTlb {
        epoch: u64,
    }

    #[test]
    fn executor_resources_are_owned_by_vcpu_state() {
        let mut vcpu = VcpuExecutionState::new(TestTlb::default(), 1_000);
        vcpu.software_tlb_mut().epoch = 7;
        vcpu.exclusive_monitor_mut().reserve(ExclusiveReservation {
            page: GuestPhysicalPageId::new(3),
            byte_offset: 64,
            access_size: 8,
            generation: CodeGeneration::new(9),
        });
        vcpu.post_interrupts(0b0001);
        vcpu.post_interrupts(0b0100);
        vcpu.dispatch_mut().request_safepoint();

        assert_eq!(vcpu.software_tlb().epoch, 7);
        assert!(vcpu.exclusive_monitor().reservation().is_some());
        assert_eq!(vcpu.take_pending_interrupts(), 0b0101);
        assert_eq!(vcpu.take_pending_interrupts(), 0);
        assert!(vcpu.dispatch().safepoint_requested());
        assert_eq!(vcpu.dispatch().budget(), 1_000);
    }
}
