//! Guest NVN objects and command recording. The renderer receives queue operations,
//! while object storage and procedure resolution stay in the guest ABI.
mod abi;
use abi::*;
pub use abi::{
    IMAGE_HANDLE_BASE_SLOT, IMAGE_HANDLE_SLOTS, Range, Sampler, Shader, Texture, VertexStream,
};
pub use guest_image::{GUEST_CODE, GUEST_ERROR_PC, QUERY_VERSIONS_OFFSET, guest_offset};
mod bindings;
mod command;
mod device;
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
    BUFFER_STAGES, Bindings, Sampled, SampledBindings, TextureHandles, TextureShape,
};
pub use command::{Attachments, CommandState, Draw, DrawPipeline, Operation, Submission};
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
    fn fence(&self, fence: Arc<Fence>, flush_for_cpu: bool) -> Result<()>;
    fn flush(&self) -> Result<()>;
    fn present(
        &self,
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
    pub surfaces: Arc<dyn SurfaceExporter>,
    pub heap: DeviceHeapMemory,
    pub backend: Arc<dyn Backend>,
}

// Pool slots outlive the source object and may be copied by the guest.
#[derive(Default)]
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
pub struct Session {
    graphics: OnceLock<Graphics>,
    pools: Mutex<pool::Mappings>,
    queues: Mutex<queue::Queues>,
    samplers: Mutex<Descriptors<Sampler>>,
    textures: Mutex<Descriptors<Texture>>,
}
impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
impl Session {
    pub fn new() -> Self {
        Self {
            graphics: OnceLock::new(),
            pools: Default::default(),
            queues: Default::default(),
            samplers: Default::default(),
            textures: Default::default(),
        }
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
                return None;
            }
            Some(match name {
                $($(concat!("nvn", stringify!($name)) => $family::$handler, )*)*
                _ => return None,
            })
        }
    };
}
procedures! {
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
        QueueInitialize => initialize,
        QueueFinalize => finalize,
        SyncInitialize => sync_initialize,
        QueueFenceSync => fence,
        QueueFlush => flush,
        QueueAcquireTexture => acquire,
        SyncWait => wait,
        QueueSubmitCommands => submit,
        QueuePresentTexture => present,
    },
}
