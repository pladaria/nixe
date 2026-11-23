use super::*;
use nixe_nvn::event::{EventSignals, EventWait, EventWord, Signal};

impl NativeGpuExecution {
    pub fn use_nvn_events(&mut self, events: Arc<EventSignals>) {
        self.state.nvn.events = events;
    }
}

impl NativeExecution<'_> {
    pub(super) fn wait_nvn_event(
        &mut self,
        address: u64,
        mode: u32,
        expected: u32,
    ) -> Result<Option<EventWait>, VulkanError> {
        self.residency.flush_read_span(address, 4)?;
        let word = EventWord::capture(self.residency.device_memory(), address).map_err(error)?;
        let wait = self
            .state
            .nvn
            .events
            .wait(word, mode, expected)
            .map_err(error)?;
        if wait.is_none() {
            self.residency.order_nvn(0x100)?;
        }
        Ok(wait)
    }

    pub(super) fn signal_nvn_event(
        &mut self,
        address: u64,
        mode: u32,
        value: u32,
        location: u32,
        flags: u32,
    ) -> Result<(), VulkanError> {
        if location > 2 || flags & !1 != 0 {
            return Err(error(format!(
                "unsupported NVN event signal location {location} / flags {flags:#x}"
            )));
        }
        let signal = Signal::decode(mode).map_err(error)?;
        self.residency.flush_read_span(address, 4)?;
        let word = EventWord::capture(self.residency.device_memory(), address).map_err(error)?;
        let events = self.state.nvn.events.clone();
        let serial = events.record(&word, signal, value);
        // @SIMPLIFICATION Publish all locations at the end of the native
        // prefix, retaining execution ordering and atomic CPU visibility.
        let device = self.residency.device();
        let work = if self.residency.has_recorded_work() {
            device.current_work()
        } else {
            let submission = &device.submission;
            submission.ticket(submission.scheduled)
        };
        self.state.fences.queue(
            Some(work),
            false,
            Some(Box::new(move |result| {
                result?;
                events.publish(word, signal, value, Some(serial));
                Ok(())
            })),
        )
    }
}
