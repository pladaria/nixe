//! Guest NVN objects and command recording. The renderer receives queue operations,
//! while object storage and procedure resolution stay in the guest ABI.
mod buffer;
mod command;
mod device;
mod memory;
mod pool;
mod program;
mod queue;
mod state;
mod texels;
mod texture;
mod vertex;

pub use command::{Draw, Operation, Range, Sampled, VertexStream};
use nixe_cpu::platform::TargetPlatform;
use nixe_memory::{AddressSpace, DeviceHeapMemory};
pub use program::Shader;
pub use queue::{Fence, PendingWait};
pub use state::RasterState;
use std::sync::{Arc, Mutex, OnceLock};
pub use texture::{Layout, Sampler, Texture};

pub type Result<T> = std::result::Result<T, Box<str>>;
pub trait Backend: Send + Sync {
    fn submit(&self, commands: Vec<Operation>) -> Result<()>;
    fn acquire_fence(&self, fence: Arc<Fence>) -> Result<()>;
    fn fence(&self, fence: Arc<Fence>) -> Result<()>;
    fn flush(&self) -> Result<()>;
    fn present(&self, texture: Texture, interval: u32) -> Result<()>;
}

pub struct Graphics {
    pub heap: DeviceHeapMemory,
    pub backend: Arc<dyn Backend>,
}

pub struct Session {
    platform: TargetPlatform,
    pub api_version: OnceLock<[u32; 2]>,
    graphics: OnceLock<Graphics>,
    pools: Mutex<pool::Mappings>,
    queues: Mutex<queue::Queues>,
}
impl Session {
    pub fn new(platform: TargetPlatform) -> Self {
        Self {
            platform,
            api_version: Default::default(),
            graphics: OnceLock::new(),
            pools: Default::default(),
            queues: Default::default(),
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
    pub args: [u64; 6],
    pub float: &'a dyn Fn(u8) -> f32,
    pub pending: std::cell::Cell<Option<PendingWait>>,
}
pub type Handler = fn(&Call<'_>) -> Result<u64>;
macro_rules! procedures {
    ($($family:ident { $($name:ident => $handler:ident),* $(,)? }),* $(,)?) => {
        pub fn resolve(platform: TargetPlatform, name: &str) -> Option<Handler> {
            if platform == TargetPlatform::Switch2 {
                return (name == "nvnDeviceGetTextureHandle").then_some(device::texture_handle);
            }
            Some(match name {
                $($(concat!("nvn", stringify!($name)) => $family::$handler, )*)*
                _ => return None,
            })
        }
    };
}
procedures! {
    texels {
        TextureViewSetDefaults => defaults,
        TextureViewSetFormat => format,
        TextureViewSetLevels => levels,
        TextureViewSetTarget => target,
        TextureReadTexels => read,
        TextureWriteTexels => write,
        TextureFlushTexels => cache,
        TextureInvalidateTexels => cache,
    },
    state {
        PolygonStateSetDefaults => poly,
        PolygonStateSetFrontFace => front,
        PolygonStateSetCullFace => cull,
        BlendStateSetDefaults => blend,
        BlendStateSetBlendFunc => factors,
        BlendStateSetBlendEquation => equations,
        BlendStateSetBlendTarget => target,
        ColorStateSetDefaults => color,
        ColorStateSetBlendEnable => enable,
        DepthStencilStateSetDefaults => depth,
        DepthStencilStateSetDepthTestEnable => test,
        DepthStencilStateSetDepthWriteEnable => write,
        DepthStencilStateSetDepthFunc => compare,
        ChannelMaskStateSetDefaults => mask,
        ChannelMaskStateSetChannelMask => channels,
    },
    buffer {
        BufferBuilderSetDevice => set_device,
        BufferBuilderSetStorage => storage,
        BufferBuilderSetDefaults => defaults,
        BufferInitialize => initialize,
        BufferFinalize => finalize,
        BufferMap => map,
        BufferGetAddress => address,
    },
    device {
        DeviceBuilderSetFlags => flags,
        DeviceBuilderSetDefaults => defaults,
        DeviceInitialize => initialize,
        DeviceGetInteger => integer,
        DeviceGetTextureHandle => texture_handle,
    },
    pool {
        MemoryPoolFinalize => finalize,
        MemoryPoolBuilderSetDefaults => defaults,
        MemoryPoolBuilderSetDevice => set_device,
        MemoryPoolBuilderSetStorage => storage,
        MemoryPoolBuilderSetFlags => flags,
        MemoryPoolInitialize => initialize,
        MemoryPoolMap => map,
        MemoryPoolGetBufferAddress => address,
        MemoryPoolFlushMappedRange => flush,
        TexturePoolInitialize => descriptor_pool,
        SamplerPoolInitialize => descriptor_pool,
    },
    texture {
        TextureGetWidth => width,
        TextureGetHeight => height,
        TextureGetFormat => get_format,
        TextureGetTarget => get_target,
        TextureBuilderSetDepth => depth,
        TextureFinalize => finalize,
        TextureBuilderSetDevice => set_device,
        TextureBuilderSetDefaults => defaults,
        TextureBuilderSetFlags => flags,
        TextureBuilderSetTarget => target,
        TextureBuilderSetSize2D => size,
        TextureBuilderSetLevels => levels,
        TextureBuilderSetFormat => format,
        TextureBuilderGetStorageAlignment => alignment,
        TextureBuilderGetStorageSize => storage_size,
        TextureBuilderSetStorage => storage,
        TextureBuilderSetSamples => samples,
        TextureBuilderSetStride => stride,
        TextureInitialize => initialize,
        TexturePoolRegisterTexture => register_texture,
        SamplerBuilderSetDevice => set_device,
        SamplerBuilderSetDefaults => sampler_defaults,
        SamplerBuilderSetMinMagFilter => filter,
        SamplerBuilderSetWrapMode => wrap,
        SamplerInitialize => sampler_initialize,
        SamplerPoolRegisterSampler => register_sampler,
    },
    program {
        ProgramInitialize => initialize,
        ProgramSetShaders => shaders,
    },
    vertex {
        VertexAttribStateSetDefaults => attrib_defaults,
        VertexAttribStateSetStreamIndex => stream_index,
        VertexAttribStateSetFormat => format,
        VertexStreamStateSetDefaults => stream_defaults,
        VertexStreamStateSetStride => stride,
        VertexStreamStateSetDivisor => divisor,
    },
    command {
        CommandBufferDrawElementsBaseVertex => indexed,
        CommandBufferSetDepthRange => depth_range,
        CommandBufferSetViewportSwizzles => swizzles,
        CommandBufferClearDepthStencil => clear_depth,
        CommandBufferSetBlendColor => blend_color,
        CommandBufferBindPolygonState => poly,
        CommandBufferBindBlendState => blend,
        CommandBufferBindColorState => color,
        CommandBufferBindChannelMaskState => mask,
        CommandBufferBindDepthStencilState => depth,
        CommandBufferInitialize => initialize,
        CommandBufferAddCommandMemory => add_commands,
        CommandBufferAddControlMemory => add_control,
        CommandBufferBeginRecording => begin,
        CommandBufferEndRecording => end,
        CommandBufferSetSamplerPool => sampler_pool,
        CommandBufferSetTexturePool => texture_pool,
        CommandBufferSetRenderTargets => targets,
        CommandBufferSetScissor => scissor,
        CommandBufferSetViewport => viewport,
        CommandBufferBarrier => barrier,
        CommandBufferClearColor => clear,
        CommandBufferBindProgram => program,
        CommandBufferBindVertexAttribState => attributes,
        CommandBufferBindVertexStreamState => streams,
        CommandBufferBindVertexBuffer => vertex,
        CommandBufferBindUniformBuffer => uniform,
        CommandBufferBindTexture => texture,
        CommandBufferDrawArrays => draw,
    },
    queue {
        WindowAcquireTexture => window_acquire,
        QueueBuilderSetCommandMemorySize => command_bytes,
        QueueBuilderSetComputeMemorySize => compute_bytes,
        QueueBuilderGetQueueMemorySize => queue_bytes,
        QueueBuilderSetQueueMemory => queue_memory,
        QueueBuilderSetCommandFlushThreshold => threshold,
        WindowBuilderSetDefaults => window_defaults,
        WindowBuilderSetDevice => set_device,
        WindowBuilderSetNativeWindow => native_window,
        WindowBuilderSetTextures => textures,
        WindowInitialize => window_initialize,
        QueueBuilderSetDevice => set_device,
        QueueBuilderSetDefaults => defaults,
        QueueInitialize => initialize,
        SyncInitialize => sync_initialize,
        QueueFenceSync => fence,
        QueueFlush => flush,
        QueueAcquireTexture => acquire,
        SyncWait => wait,
        QueueSubmitCommands => submit,
        QueuePresentTexture => present,
    },
}
