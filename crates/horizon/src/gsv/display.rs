//! Native display events share the kernel timer, independently of GPU completions.
use nixe_runtime::EventObject;
use std::{
    sync::{Arc, Mutex},
    task::{Wake, Waker},
};

#[derive(Debug, Default)]
pub(crate) struct DisplayContext {
    pub initialized: bool,
    pub events: Arc<Mutex<[Option<DisplayEvent>; 8]>>,
    pub timer: Option<nixe_scheduler::LoopingEvent>,
    pub connection_event: Option<EventObject>,
    pub opened: bool,
    pub layers: std::collections::BTreeMap<u64, DisplayLayer>,
    pub active_layer: Option<u64>,
    pub timestamp_recorders: Option<u32>,
}
#[derive(Debug)]
pub(crate) struct DisplayLayer {
    pub buffer_count: u32,
    pub attached: bool,
}
#[derive(Debug)]
pub(crate) struct DisplayEvent {
    pub event: EventObject,
    pub binding: Option<u32>,
}
struct Tick(Arc<Mutex<[Option<DisplayEvent>; 8]>>);
impl Wake for Tick {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let events = self.0.lock().unwrap();
        for slot in events
            .iter()
            .flatten()
            .filter(|slot| slot.binding == Some(261))
        {
            slot.event.signal();
        }
    }
}
impl DisplayContext {
    pub fn start_timer(
        &mut self,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
    ) -> Result<(), nixe_scheduler::TimerStopped> {
        if self.timer.is_none() {
            let interval = std::num::NonZeroU64::new(1_000_000_000 / 60).unwrap();
            self.timer = Some(
                kernel.schedule_looping_event(
                    kernel
                        .clock()
                        .scheduler_time_ns()
                        .saturating_add(interval.get()),
                    interval,
                    Waker::from(Arc::new(Tick(self.events.clone()))),
                )?,
            );
        }
        Ok(())
    }
}
impl Drop for DisplayContext {
    fn drop(&mut self) {
        // Cancellation waits out an executing tick before releasing its events.
        if let Some(timer) = self.timer.take() {
            timer.cancel();
        }
    }
}
