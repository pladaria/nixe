//! Ampere command processors. This crate owns the native method and MME ISAs;
//! it neither interprets Horizon requests nor translates methods into Maxwell.
pub mod blit;
mod channel;
pub mod compute;
pub mod dma;
mod engines;
mod mme;
pub mod ordered;
mod packet;
mod registers;
pub mod targets;
mod writes;

pub use channel::{Channel, FrontendHost, Progress, SemaphoreAcquire};
pub use engines::BackState;
pub use engines::{Draw, GraphicsState, Preparation, PreparationHost};
pub use mme::MacroProgram;
pub use packet::{GpfifoEntry, decode_entries};
pub use registers::{Class, DirtyFlags, Registers};
pub use writes::MemoryWrites;

pub type Result<T> = std::result::Result<T, Box<str>>;

/// The source is carried through macro expansion and the ordered-method ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Source {
    pub channel: u64,
    pub submission: u64,
    pub entry: u32,
    pub pushbuffer: u64,
    pub offset: u32,
    pub subchannel: u8,
    pub class: u32,
    pub method: u32,
    pub argument: u32,
}
impl Source {
    pub fn error(self, detail: impl std::fmt::Display) -> Box<str> {
        format!("Ampere execution: channel={} submission={} entry={} pushbuffer={:#x} word={} subchannel={} class={:#x} method={:#x} argument={:#010x}: {detail}",
            self.channel, self.submission, self.entry, self.pushbuffer, self.offset,
            self.subchannel, self.class, self.method, self.argument).into_boxed_str()
    }
}

/// Native preparation consumes generation-specific state with static dispatch.
/// Canonical memory, cache ownership and queue/fence execution stay shared.
pub trait Rasterizer: nixe_gpu::GpuExecution {
    fn command_memory(&self) -> std::sync::Arc<dyn nixe_gpu::command_memory::CommandMemory>;
    fn write_constants(
        &mut self,
        source: Source,
        registers: &Registers,
        offset: u32,
        bytes: &[u8],
        memory: &mut impl RenderMemory,
    ) -> Result<()>;
    fn flush_constants(&mut self) -> Result<()>;
    fn read_memory_into(
        &mut self,
        capture: &nixe_memory::CapturedMemory,
        destination: &mut [u8],
    ) -> Result<()>;
    fn write_memory(&mut self, capture: &nixe_memory::CapturedMemory, bytes: &[u8]) -> Result<()>;
    fn reduce_report(
        &mut self,
        source: Source,
        target: &[nixe_memory::AllocationRange],
        payload: u32,
        operation: u32,
        signed: bool,
    ) -> Result<()>;
    fn dispatch(
        &mut self,
        source: Source,
        launch: &compute::Launch,
        registers: &Registers,
        memory: &mut impl RenderMemory,
    ) -> Result<()>;
    fn reset_samples_passed(&mut self, source: Source) -> Result<()>;
    fn report_samples_passed(
        &mut self,
        source: Source,
        target: &[nixe_memory::AllocationRange],
        short: bool,
        timestamp: u64,
    ) -> Result<()>;
    fn fragment_barrier(&mut self, source: Source, before_depth: bool) -> Result<()>;
    fn create_channel(&mut self, channel: u64);
    fn bind_channel(&mut self, channel: u64);
    fn close_channel(&mut self, channel: u64);
    fn draw(
        &mut self,
        draw: Draw,
        registers: &Registers,
        state: &mut GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<()>;
    /// Return false only for an exact byte copy that can use coherent RAM.
    fn blit(
        &mut self,
        source: Source,
        operation: &blit::Blit,
        memory: &mut impl RenderMemory,
    ) -> Result<bool>;
    fn invalidate_space(&mut self, space: u64, address: u64, bytes: u64);
    fn composite(
        &mut self,
        image: nixe_gpu::native::NativePresentationImage,
        swap_interval: u32,
    ) -> Result<nixe_gpu::ResidentImage>;
    fn clear(
        &mut self,
        source: Source,
        registers: &Registers,
        state: &mut GraphicsState,
        memory: &mut impl RenderMemory,
    ) -> Result<()>;
}
/// Commit lazy native GPU backing only when a resource is consumed.
pub trait RenderMemory: nixe_gpu::native::ImageAddressSpace {
    /// A numeric address, not a Rust borrow. Valid while the address-space owner
    /// retains its mappings. Retained selections must check mapping generations
    /// before reuse and resolve device coherence before dereferencing it.
    fn read_span(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<Option<(u64, nixe_memory::HostAddress)>>;
    /// Same-layout virtual extent, including reserved native backing. Physical
    /// fragmentation is resolved separately when the resource is consumed.
    fn buffer_layout_size(&self, address: u64) -> Result<u64>;
    fn capture_memory(&mut self, address: u64, bytes: u64) -> Result<nixe_memory::CapturedMemory>;
    fn mapping_generation(&self) -> u64;
    fn prepare_image(&mut self, address: u64, bytes: u64) -> Result<()>;
}
