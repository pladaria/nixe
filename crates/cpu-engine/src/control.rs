//! Lock-free cross-vCPU control publication and acknowledgement.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::ControlEpoch;

const CONTROL_PREEMPT: u32 = 1 << 0;
const CONTROL_PROCESS_STOP: u32 = 1 << 1;
const CONTROL_DEBUGGER_STOP: u32 = 1 << 2;
const CONTROL_TLB_SHOOTDOWN: u32 = 1 << 3;
const CONTROL_CODE_INVALIDATION: u32 = 1 << 4;
const CONTROL_HANDOFF: u32 = 1 << 5;
const CONTROL_REQUEST_MASK: u64 = (1 << 6) - 1;
const CONTROL_EPOCH_SHIFT: u32 = 6;
const MAX_CONTROL_EPOCH: u64 = u64::MAX >> CONTROL_EPOCH_SHIFT;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CrossVcpuRequest {
    Preempt,
    ProcessStop,
    DebuggerStop,
    TlbShootdown,
    CodeInvalidation,
    EngineHandoff,
}

impl CrossVcpuRequest {
    const fn bit(self) -> u32 {
        match self {
            Self::Preempt => CONTROL_PREEMPT,
            Self::ProcessStop => CONTROL_PROCESS_STOP,
            Self::DebuggerStop => CONTROL_DEBUGGER_STOP,
            Self::TlbShootdown => CONTROL_TLB_SHOOTDOWN,
            Self::CodeInvalidation => CONTROL_CODE_INVALIDATION,
            Self::EngineHandoff => CONTROL_HANDOFF,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ControlSnapshot {
    pub epoch: ControlEpoch,
    pub requests: u32,
    pub event_mask: u32,
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
    /// The epoch and coalesced request bits are one atomic publication unit.
    /// This prevents a consumer from acknowledging an epoch whose request bit
    /// has not become visible yet.
    published: AtomicU64,
    acknowledged_epoch: AtomicU64,
    events: AtomicU32,
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

    #[must_use]
    pub fn request(&self, request: CrossVcpuRequest) -> ControlEpoch {
        let bit = u64::from(request.bit());
        let mut published = self.state.published.load(Ordering::Acquire);
        loop {
            let epoch = published >> CONTROL_EPOCH_SHIFT;
            let next_epoch = epoch
                .checked_add(1)
                .filter(|epoch| *epoch <= MAX_CONTROL_EPOCH)
                .expect("control epoch exhausted");
            let pending = (published & CONTROL_REQUEST_MASK) | bit;
            let next = (next_epoch << CONTROL_EPOCH_SHIFT) | pending;
            match self.state.published.compare_exchange_weak(
                published,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return ControlEpoch::new(next_epoch),
                Err(observed) => published = observed,
            }
        }
    }

    pub fn post_event(&self, mask: u32) {
        self.state.events.fetch_or(mask, Ordering::Release);
    }

    #[must_use]
    pub fn request_invalidation(&self, epoch: u64) -> ControlEpoch {
        self.state
            .invalidation_epoch
            .fetch_max(epoch, Ordering::AcqRel);
        self.request(CrossVcpuRequest::CodeInvalidation)
    }

    #[must_use]
    pub fn take_pending(&self) -> Option<ControlSnapshot> {
        let mut published = self.state.published.load(Ordering::Acquire);
        let (epoch, requests) = loop {
            let requests = (published & CONTROL_REQUEST_MASK) as u32;
            if requests == 0 {
                break (published >> CONTROL_EPOCH_SHIFT, 0);
            }
            let cleared = published & !CONTROL_REQUEST_MASK;
            match self.state.published.compare_exchange_weak(
                published,
                cleared,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break (published >> CONTROL_EPOCH_SHIFT, requests),
                Err(observed) => published = observed,
            }
        };
        let event_mask = self.state.events.swap(0, Ordering::AcqRel);
        if requests == 0 && event_mask == 0 {
            return None;
        }
        let invalidation_epoch = self.state.invalidation_epoch.load(Ordering::Acquire);
        Some(ControlSnapshot {
            epoch: ControlEpoch::new(epoch),
            requests,
            event_mask,
            invalidation_epoch,
        })
    }

    /// Confirms that the executor has applied every control effect represented
    /// by `snapshot`. Cache/TLB-owning engines must call this only after stale
    /// resources can no longer be re-entered.
    pub fn acknowledge(&self, snapshot: ControlSnapshot) {
        self.state
            .acknowledged_epoch
            .fetch_max(snapshot.epoch.get(), Ordering::AcqRel);
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
    pub fn acknowledged(&self, epoch: ControlEpoch) -> bool {
        self.state.acknowledged_epoch.load(Ordering::Acquire) >= epoch.get()
    }

    #[must_use]
    pub fn acknowledged_invalidation(&self, epoch: u64) -> bool {
        self.state
            .acknowledged_invalidation_epoch
            .load(Ordering::Acquire)
            >= epoch
    }
}
