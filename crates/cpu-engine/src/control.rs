//! Lock-free cross-vCPU control publication and acknowledgement.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

const CONTROL_PREEMPT: u32 = 1 << 0;
const CONTROL_CODE_INVALIDATION: u32 = 1 << 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CrossVcpuRequest {
    Preempt,
    CodeInvalidation,
}

impl CrossVcpuRequest {
    const fn bit(self) -> u32 {
        match self {
            Self::Preempt => CONTROL_PREEMPT,
            Self::CodeInvalidation => CONTROL_CODE_INVALIDATION,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ControlSnapshot {
    pub requests: u32,
    pub invalidation_epoch: u64,
}

impl ControlSnapshot {
    #[must_use]
    pub const fn contains(self, request: CrossVcpuRequest) -> bool {
        self.requests & request.bit() != 0
    }
}

#[derive(Default)]
struct EngineControlState {
    pending: AtomicBool,
    requests: AtomicU32,
    invalidation_epoch: AtomicU64,
    acknowledged_invalidation_epoch: AtomicU64,
    active_executions: AtomicU32,
}

/// Cloneable, lock-free control path retained outside a worker-owned executor.
#[derive(Clone, Default)]
pub struct EngineControl {
    state: Arc<EngineControlState>,
}

/// RAII publication that one executor is currently inside `run_slice`.
pub struct EngineExecutionGuard {
    state: Arc<EngineControlState>,
}

impl Drop for EngineExecutionGuard {
    fn drop(&mut self) {
        self.state.active_executions.fetch_sub(1, Ordering::AcqRel);
    }
}

impl EngineControl {
    #[must_use]
    pub fn enter_execution(&self) -> EngineExecutionGuard {
        self.state.active_executions.fetch_add(1, Ordering::AcqRel);
        EngineExecutionGuard {
            state: Arc::clone(&self.state),
        }
    }

    #[must_use]
    pub fn execution_active(&self) -> bool {
        self.state.active_executions.load(Ordering::Acquire) != 0
    }

    pub fn request(&self, request: CrossVcpuRequest) {
        self.state
            .requests
            .fetch_or(request.bit(), Ordering::Release);
        self.state.pending.store(true, Ordering::Release);
    }

    pub fn request_invalidation(&self, epoch: u64) {
        self.state
            .invalidation_epoch
            .fetch_max(epoch, Ordering::AcqRel);
        self.request(CrossVcpuRequest::CodeInvalidation);
    }

    #[must_use]
    pub fn take_pending(&self) -> Option<ControlSnapshot> {
        if !self.state.pending.load(Ordering::Acquire)
            || !self.state.pending.swap(false, Ordering::AcqRel)
        {
            return None;
        }
        let requests = self.state.requests.swap(0, Ordering::AcqRel);
        if requests == 0 {
            return None;
        }
        let invalidation_epoch = self.state.invalidation_epoch.load(Ordering::Acquire);
        Some(ControlSnapshot {
            requests,
            invalidation_epoch,
        })
    }

    /// Confirms that invalidated resources represented by `snapshot` cannot be
    /// re-entered.
    pub fn acknowledge(&self, snapshot: ControlSnapshot) {
        if snapshot.requests & CONTROL_CODE_INVALIDATION != 0 {
            self.state
                .acknowledged_invalidation_epoch
                .fetch_max(snapshot.invalidation_epoch, Ordering::AcqRel);
        }
    }

    /// Acknowledges a mapping epoch after an executor has synchronously made
    /// stale translations unreachable, without consuming other control bits.
    pub fn acknowledge_invalidation(&self, epoch: u64) {
        self.state
            .acknowledged_invalidation_epoch
            .fetch_max(epoch, Ordering::AcqRel);
    }

    #[must_use]
    pub fn acknowledged_invalidation(&self, epoch: u64) -> bool {
        self.state
            .acknowledged_invalidation_epoch
            .load(Ordering::Acquire)
            >= epoch
    }
}
