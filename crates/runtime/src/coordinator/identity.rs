use nixe_scheduler::{GuestThreadId, ProcessId};

#[derive(Debug)]
pub(super) struct ProcessIdAllocator {
    next: u64,
}

impl ProcessIdAllocator {
    pub(super) const fn new() -> Self {
        Self { next: 1 }
    }

    pub(super) fn candidate(&self) -> Option<(ProcessId, u64)> {
        self.next
            .checked_add(1)
            .map(|next| (ProcessId::new(self.next), next))
    }

    pub(super) const fn commit(&mut self, next: u64) {
        self.next = next;
    }
}

/// Monotonic runtime-global guest-thread identity allocation. A candidate is
/// committed only after process and scheduler publication both succeed.
#[derive(Debug)]
pub(super) struct GuestThreadIdAllocator {
    next: u64,
}

impl GuestThreadIdAllocator {
    pub(super) const fn new() -> Self {
        Self { next: 1 }
    }

    pub(super) fn candidate(&self) -> Option<(GuestThreadId, u64)> {
        self.next
            .checked_add(1)
            .map(|next| (GuestThreadId::new(self.next), next))
    }

    pub(super) const fn commit(&mut self, next: u64) {
        self.next = next;
    }
}
