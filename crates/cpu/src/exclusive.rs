//! Engine-neutral exclusive-reservation value types.
//!
//! Dispatch budgets, pending events, safepoints, memory acceleration, and mutable
//! local-monitor storage belong to concrete engine executors. This module keeps
//! only values referenced by the common CPU memory contract.

/// Portable local-monitor state used by CPU execution engines.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExclusiveMonitorState<R = nixe_memory::Reservation> {
    reservation: Option<R>,
}
impl<R> Default for ExclusiveMonitorState<R> {
    fn default() -> Self {
        Self { reservation: None }
    }
}

impl<R: Copy> ExclusiveMonitorState<R> {
    #[must_use]
    pub const fn reservation(self) -> Option<R> {
        self.reservation
    }

    pub const fn reserve(&mut self, reservation: R) {
        self.reservation = Some(reservation);
    }

    pub const fn clear(&mut self) {
        self.reservation = None;
    }
}
