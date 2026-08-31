//! Engine-neutral exclusive-reservation value types.
//!
//! Dispatch budgets, pending events, safepoints, memory acceleration, and mutable
//! local-monitor storage belong to concrete engine executors. This module keeps
//! only values referenced by the common CPU memory contract.

use nixe_memory::GuestPhysicalPageId;

use crate::memory::MemoryValue;

/// Physical reservation recorded by a local exclusive monitor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExclusiveReservation {
    pub page: GuestPhysicalPageId,
    pub byte_offset: u16,
    pub access_size: u8,
    pub expected: MemoryValue,
}

/// Portable local-monitor state used by CPU execution engines.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_monitor_value_retains_a_typed_reservation() {
        let reservation = ExclusiveReservation {
            page: GuestPhysicalPageId::new(3),
            byte_offset: 64,
            access_size: 8,
            expected: MemoryValue::U64(9),
        };
        let mut monitor = ExclusiveMonitorState::default();
        monitor.reserve(reservation);
        assert_eq!(monitor.reservation(), Some(reservation));
        monitor.clear();
        assert_eq!(monitor.reservation(), None);
    }
}
