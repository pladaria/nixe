//! Ampere command processors. This crate owns the native method and MME ISAs;
//! it neither interprets Horizon requests nor translates methods into Maxwell.
pub mod blit;
mod channel;
pub mod compute;
mod engines;
mod mme;
pub mod ordered;
mod packet;
mod registers;
pub mod targets;

pub use channel::{Channel, FrontendHost, Progress, SemaphoreAcquire};
pub use engines::{GraphicsState, Preparation, PreparationHost};
pub use packet::{GpfifoEntry, decode_entries};
pub use registers::{Class, Registers};

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

/// Effects that can modify input memory must finish before the command owner
/// captures a subsequent pushbuffer. Unsupported effects fail on consumption.
pub fn writes_memory(source: Source) -> bool {
    if source.method < 0x100 {
        return source.method == 0x1c;
    }
    let Ok(class) = Class::from_id(source.class) else {
        return false;
    };
    match class {
        Class::ThreeD => {
            matches!(source.method, 0x1b4 | 0x26c | 0x274 | 0x304 | 0xd78 | 0x1214 | 0x1218 | 0x123c | 0x17e0..=0x17f8 | 0x19d0 | 0x1b0c | 0x2390..=0x23cc)
        }
        Class::Compute => matches!(source.method, 0x1b4 | 0x2b8..=0x2c0 | 0x41c | 0x1b0c),
        Class::Inline => source.method == 0x1b4,
        Class::Copy => source.method == 0x300,
        Class::TwoD => matches!(source.method, 0x860 | 0x8dc | 0x918),
    }
}

/// Native preparation consumes generation-specific state with static dispatch.
/// Canonical memory, cache ownership and queue/fence execution stay shared.
pub trait Rasterizer: nixe_gpu::GpuExecution {
    fn create_channel(&mut self, channel: u64);
    fn bind_channel(&mut self, channel: u64);
    fn close_channel(&mut self, channel: u64);
    fn blit(
        &mut self,
        source: Source,
        operation: &blit::Blit,
        memory: &mut impl RenderMemory,
    ) -> Result<()>;
    fn invalidate_space(&mut self, space: u64, address: u64, bytes: u64);
    fn composite(
        &mut self,
        image: nixe_gpu::native::NativePresentationImage,
        swap_interval: u32,
        pacing: nixe_gpu::frame_metrics::FramePacing,
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
    fn prepare_image(&mut self, address: u64, bytes: u64) -> Result<()>;
}
