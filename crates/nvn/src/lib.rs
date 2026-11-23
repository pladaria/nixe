//! Guest NVN objects and command recording. The renderer receives queue operations,
//! while object storage and procedure resolution stay in the guest ABI.
mod abi;
use abi::*;
pub use abi::{
    ComputeLayout, CopyStrides, IMAGE_HANDLE_BASE_SLOT, IMAGE_HANDLE_SLOTS, Range, Sampler, Shader,
    Texture, VertexStream,
};
pub use guest_image::{GUEST_CODE, GUEST_ERROR_PC, QUERY_VERSIONS_OFFSET, guest_offset};
mod bindings;
mod command;
mod device;
pub mod display;
pub mod event;
mod guest_image;

mod layout;
mod view;
type MipList = smallvec::SmallVec<[Mip; 1]>;
mod memory;
mod pool;
pub use pool::PoolBounds;
mod program;
mod queue;
mod sampler;
mod shader_usage;
pub use shader_usage::{ShaderUsage, ShaderValidity, StageSource, UsageSlot};
mod state;
mod texels;
mod texture;

pub use bindings::{
    BUFFER_STAGES, Bindings, BindlessPools, Sampled, SampledBindings, TextureHandles, TextureShape,
};
pub use command::{
    Attachments, CommandState, Dispatch, Draw, DrawPipeline, Operation, Submission,
    SubmissionCursor,
};
pub use layout::{Layout, Mip};
use nixe_cpu::platform::TargetPlatform;
use nixe_memory::{AddressSpace, DeviceHeapMemory};

pub use queue::{Fence, PendingWait};

pub use state::RasterState;
use std::sync::{Arc, Mutex, OnceLock};

pub type Result<T> = std::result::Result<T, Box<str>>;
pub trait Backend: Send + Sync {
    fn submit(&self, queue: u64, commands: Submission) -> Result<()>;
    fn drain(&self) -> Result<()>;
    fn finalize_queue(&self, queue: u64) -> Result<()>;
    fn acquire_fence(&self, fence: Arc<Fence>) -> Result<()>;
    fn fence(&self, queue: u64, fence: Arc<Fence>, flush_for_cpu: bool) -> Result<()>;
    fn flush(&self, queue: u64) -> Result<()>;
    fn present(
        &self,
        queue: u64,
        texture: Texture,
        interval: u32,
        flip_vertical: bool,
        crop: [u32; 4],
    ) -> Result<()>;
}

/// An NvMap handle exported to other guest drivers (e.g. video decoding).
pub trait ExportedPool: Send + Sync {
    fn handle(&self) -> u32;
}
pub trait SurfaceExporter: Send + Sync {
    fn export(
        &self,
        memory: &AddressSpace,
        address: u64,
        bytes: u64,
    ) -> Result<Box<dyn ExportedPool>>;
}
pub struct Graphics {
    pub surfaces: Option<Arc<dyn SurfaceExporter>>,
    pub heap: DeviceHeapMemory,
    pub backend: Arc<dyn Backend>,
    pub display: Option<Arc<display::DisplayHistory>>,
}

// Pool slots outlive the source object and may be copied by the guest.
#[derive(Debug, Default)]
struct Descriptors<T> {
    ids: std::collections::HashMap<T, u32>,
    values: Vec<T>,
}
impl<T: Eq + std::hash::Hash + Copy> Descriptors<T> {
    fn intern(&mut self, v: T) -> u32 {
        *self.ids.entry(v).or_insert_with(|| {
            self.values.push(v);
            self.values.len() as u32
        })
    }
    fn get(&self, id: u32) -> Option<T> {
        self.values.get(id.wrapping_sub(1) as usize).copied()
    }
}
#[derive(Debug, Default)]
struct DescriptorRegistry {
    samplers: Mutex<Descriptors<Sampler>>,
    textures: Mutex<Descriptors<Texture>>,
}
pub struct Session {
    graphics: OnceLock<Graphics>,
    sdk_api_major: OnceLock<u32>,
    pools: Mutex<pool::Mappings>,
    queues: Mutex<queue::Queues>,
    descriptors: Arc<DescriptorRegistry>,
    events: Arc<event::EventSignals>,
    platform: TargetPlatform,
}
impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
impl Session {
    pub fn new() -> Self {
        Self::for_platform(TargetPlatform::Switch1)
    }
    pub fn for_platform(platform: TargetPlatform) -> Self {
        Self {
            graphics: OnceLock::new(),
            sdk_api_major: OnceLock::new(),
            pools: Default::default(),
            queues: Default::default(),
            descriptors: Default::default(),
            events: Default::default(),
            platform,
        }
    }
    pub fn event_signals(&self) -> Arc<event::EventSignals> {
        self.events.clone()
    }
    pub fn sdk_api_major(&self) -> Option<u32> {
        self.sdk_api_major.get().copied()
    }
    pub fn record_sdk_api_major(&self, version: u32) {
        self.sdk_api_major.get_or_init(|| version);
    }
    pub fn attach(&self, graphics: Graphics) -> Result<()> {
        self.graphics
            .set(graphics)
            .map_err(|_| "graphics backend already attached".into())
    }
    fn graphics(&self) -> Result<&Graphics> {
        self.graphics
            .get()
            .ok_or_else(|| "no NVN graphics backend for this platform".into())
    }
}
pub struct Call<'a> {
    pub session: &'a Session,
    pub memory: &'a AddressSpace,
    pub args: [u64; 8],
    pub pending: std::cell::Cell<Option<PendingWait>>,
}
pub type Handler = fn(&Call<'_>) -> Result<u64>;
macro_rules! procedures {
    ($($family:ident { $($name:ident => $handler:ident),* $(,)? }),* $(,)?) => {
        pub fn resolve(platform: TargetPlatform, name: &str) -> Option<Handler> {
            if platform == TargetPlatform::Switch2 {
                match name {
                    "nvnProgramSetShaders" => return Some(program::shaders_ampere),
                    "nvnProgramSetShadersExt" => return Some(program::shaders_ext_ampere),
                    "nvnProgramInitializeFromBuilder" => return Some(program::from_builder),
                    "nvnMemoryPoolInitialize" => return Some(pool::initialize_ampere),
                    _ => {}
                }
            }
            Some(match name {
                $($(concat!("nvn", stringify!($name)) => $family::$handler, )*)*
                _ => return None,
            })
        }
    };
}
procedures! {
    event {
        EventSignal => signal,
    },
    sampler {
        SamplerPoolRegisterSampler => register,
    },
    device {
        DeviceInitialize => initialize,
    },
    pool {
        MemoryPoolMapVirtual => map_virtual,
        MemoryPoolFinalize => finalize,
        MemoryPoolInitialize => initialize,
    },
    texture {
        TextureGetNvRmSurfaceNVX => nv_surface,
        TexturePoolRegisterTexture => register_texture,
        TexturePoolRegisterImage => register_texture,
    },
    program {
        ProgramSetShadersExt => shaders_ext,
        ProgramSetShaders => shaders,
    },
    queue {
        WindowAcquireTexture => window_acquire,
        WindowInitialize => window_initialize,
        WindowFinalize => window_finalize,
        QueueInitialize => initialize,
        QueueFinalize => finalize,
        SyncInitialize => sync_initialize,
        SyncFinalize => sync_finalize,
        QueueFenceSync => fence,
        QueueFlush => flush,
        QueueAcquireTexture => acquire,
        SyncWait => wait,
        QueueSubmitCommands => submit,
        QueuePresentTexture => present,
    },
}
