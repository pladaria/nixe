//! Detailed macro timing is deliberately absent from normal builds.
use super::*;
use std::time::{Duration, Instant};
#[derive(Default, Debug)]
pub(super) struct Statistics {
    pub owner: &'static str,
    pub analyses: u64,
    pub eligible_entries: u64,
    pub scheduled_calls: u64,
    pub parameter_bytes: u64,
    pub borrowed_parameter_bytes: u64,
    pub materialized_parameter_bytes: u64,
    pub code_copies: u64,
    pub calls: u64,
    pub execution: Duration,
    pub callbacks: Duration,
    pub state_callbacks: u64,
}
impl Drop for Statistics {
    fn drop(&mut self) {
        log::info!(
            "Ampere MME {:?}; pure_execution={:?}",
            self,
            self.execution.saturating_sub(self.callbacks)
        );
    }
}
pub(super) struct Meter<'a, H> {
    pub host: &'a mut H,
    pub callbacks: Duration,
    pub state_callbacks: u64,
}
impl<H: Host> Host for Meter<'_, H> {
    const TRACK_OWNERSHIP: bool = H::TRACK_OWNERSHIP;
    fn state(&mut self, method: u32) -> Result<u32> {
        let start = Instant::now();
        self.state_callbacks += 1;
        let result = self.host.state(method);
        self.callbacks += start.elapsed();
        result
    }
    fn emit(&mut self, method: u32, value: u32) -> Result<Emission> {
        let start = Instant::now();
        let result = self.host.emit(method, value);
        self.callbacks += start.elapsed();
        result
    }
    fn dma(&mut self, method: u32, dram: &mut [u32], fifo: &mut Fifo<'_>) -> Result<()> {
        let start = Instant::now();
        let result = self.host.dma(method, dram, fifo);
        self.callbacks += start.elapsed();
        result
    }
    fn stopped(&self) -> bool {
        self.host.stopped()
    }
    #[cfg(target_arch = "x86_64")]
    fn bank(&self) -> (*const u32, *const u32, *const u64) {
        self.host.bank()
    }
}
