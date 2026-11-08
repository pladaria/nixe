//! Per-process admission of captured service payloads. Waiting consumes no worker.
use super::{IpcWireError, operation::CallLifetime};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, Weak},
    task::{Context, Poll, Waker},
};
const CAPACITY: usize = 64 * 1024 * 1024;

#[derive(Debug, Default)]
pub(crate) struct PayloadBudgets(Mutex<HashMap<u64, Weak<PayloadBudget>>>);
impl PayloadBudgets {
    pub fn get(&self, process: u64) -> Arc<PayloadBudget> {
        let mut budgets = self.0.lock().unwrap();
        if let Some(budget) = budgets.get(&process).and_then(Weak::upgrade) {
            return budget;
        }
        budgets.retain(|_, budget| budget.strong_count() != 0);
        let budget = Arc::default();
        budgets.insert(process, Arc::downgrade(&budget));
        budget
    }
}
#[derive(Debug, Default)]
struct State {
    used: usize,
    next: u64,
    waiting: BTreeMap<u64, Option<Waker>>,
}
#[derive(Debug, Default)]
pub(crate) struct PayloadBudget(Mutex<State>);
impl PayloadBudget {
    pub fn reserve<'a>(self: &Arc<Self>, size: usize, lifetime: &'a CallLifetime) -> Admission<'a> {
        Admission {
            budget: self.clone(),
            size,
            id: None,
            lifetime,
            pressure: None,
        }
    }
}
pub(crate) struct Admission<'a> {
    budget: Arc<PayloadBudget>,
    size: usize,
    id: Option<u64>,
    lifetime: &'a CallLifetime,
    pressure: Option<nixe_memory::trace::Span>,
}
impl Future for Admission<'_> {
    type Output = Result<PayloadPermit, IpcWireError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.lifetime.check()?;
        if this.size > CAPACITY {
            return Poll::Ready(Err(IpcWireError::HostResourceExhausted(
                "service transfer exceeds the captured-payload chunk limit",
            )));
        }
        let mut state = this.budget.0.lock().unwrap();
        let first = state.waiting.first_key_value().map(|(id, _)| *id);
        if (first.is_none() || first == this.id) && this.size <= CAPACITY - state.used {
            if let Some(id) = this.id.take() {
                state.waiting.remove(&id);
            }
            state.used += this.size;
            let wake = state
                .waiting
                .first_entry()
                .and_then(|mut entry| entry.get_mut().take());
            drop(state);
            if let Some(wake) = wake {
                wake.wake();
            }
            return Poll::Ready(Ok(PayloadPermit {
                budget: this.budget.clone(),
                size: this.size,
            }));
        }
        let id = if let Some(id) = this.id {
            id
        } else {
            state.next = state
                .next
                .checked_add(1)
                .expect("payload admission identity exhausted");
            this.id = Some(state.next);
            this.pressure = Some(nixe_memory::trace::span(
                "ipc_payload_pressure",
                Arc::as_ptr(&this.budget) as u64,
                this.size as u64,
                state.used as u64,
            ));
            state.next
        };
        state.waiting.insert(id, Some(context.waker().clone()));
        Poll::Pending
    }
}
impl Drop for Admission<'_> {
    fn drop(&mut self) {
        let Some(id) = self.id else {
            return;
        };
        let wake = {
            let mut state = self.budget.0.lock().unwrap();
            let head = state
                .waiting
                .first_key_value()
                .is_some_and(|(first, _)| *first == id);
            state.waiting.remove(&id);
            if head {
                state
                    .waiting
                    .first_entry()
                    .and_then(|mut entry| entry.get_mut().take())
            } else {
                None
            }
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
pub(crate) struct PayloadPermit {
    budget: Arc<PayloadBudget>,
    size: usize,
}
impl Drop for PayloadPermit {
    fn drop(&mut self) {
        if self.size == 0 {
            return;
        }
        let wake = {
            let mut state = self.budget.0.lock().unwrap();
            state.used -= self.size;
            state
                .waiting
                .first_entry()
                .and_then(|mut entry| entry.get_mut().take())
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
