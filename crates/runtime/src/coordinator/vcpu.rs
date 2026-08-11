use nixe_scheduler::Lease;

/// Serialized-mode runtime state for one emulated vCPU. The executor itself is
/// domain-owned by the process, while this slot owns dispatch and lease state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct RuntimeVcpuSlot {
    current_lease: Option<Lease>,
    dispatch_epoch: u64,
}

impl RuntimeVcpuSlot {
    pub(super) fn begin(&mut self, lease: Lease) {
        debug_assert!(self.current_lease.is_none());
        self.current_lease = Some(lease);
        self.dispatch_epoch = self.dispatch_epoch.saturating_add(1);
    }

    pub(super) fn finish(&mut self, lease: Lease) {
        debug_assert_eq!(self.current_lease, Some(lease));
        self.current_lease = None;
    }
}
