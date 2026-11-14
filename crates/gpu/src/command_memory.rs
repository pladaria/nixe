//! Shared physical command-page publication for split GPU frontends.
use std::sync::Arc;

pub trait CommandMemory: Send + Sync + std::fmt::Debug {
    fn must_flush_region(&self, address: u64, bytes: u64) -> bool;
    fn command_tick_done(&self, tick: u64) -> Result<bool, Arc<nixe_memory::MemoryError>>;
    fn read_command(
        &self,
        address: u64,
        output: &mut [u8],
        past_sync: bool,
    ) -> Result<CommandRead, Arc<nixe_memory::MemoryError>>;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandRead {
    Unknown,
    NeedsSync,
    Synced,
    Copied,
    Pending(u64),
}
