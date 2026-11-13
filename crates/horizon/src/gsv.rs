//! Switch 2 graphics-service ownership, separate from the Maxwell nvdrv stack.
//! https://switchbrew.org/wiki/Switch_2%3A_GSV_services
use std::sync::{Arc, Mutex};
pub(crate) mod channel;
pub(crate) mod characteristics;
pub(crate) mod display;
pub(crate) mod gpu;
mod syncpt;
pub(crate) use syncpt::SyncptMirror;

// @SPECULATIVE: counter capacity follows the observed reference driver profile.
// CPU mirror, GPU mapping and channel allocation share this identity bank.
pub(crate) const SYNCPOINT_COUNT: u32 = 192;

#[derive(Clone, Debug)]
pub struct GsvSession {
    pub(crate) process_id: u64,
    pub(crate) operation_mode: crate::OperationMode,
    pub(crate) interface: GsvInterface,
    pub(crate) state: Arc<Mutex<GsvState>>,
}

#[derive(Debug, Default)]
pub(crate) struct GsvState {
    pub boot: Option<GsvBoot>,
    pub process: Option<Arc<nixe_runtime::RunnableProcess>>,
    pub context_created: bool,
    pub native_memory_features: Option<u64>,
    pub syncpt: Option<SyncptContext>,
    pub gpu: Option<gpu::GpuContext>,
    pub nvdrv: std::sync::Weak<Mutex<NvDrv2State>>,
    pub inter_module_images: Option<InterModuleImages>,
    pub frame: Option<GraphicsFrame>,
    pub display: Option<display::DisplayContext>,
}

/// Lifetime of the image-exchange context. Native pools are registered only
/// after initialization; opening the context does not allocate guest images.
#[derive(Debug, Default)]
pub(crate) struct InterModuleImages {
    pub initialized: bool,
}

pub(crate) enum GsvBoot {
    Remote {
        account: nixe_memory::MemoryAccount,
        version: u64,
    },
    Transfer(Arc<nixe_runtime::TransferMemoryObject>),
}

impl GsvSession {
    pub(crate) fn new(process_id: u64, operation_mode: crate::OperationMode) -> Self {
        Self {
            process_id,
            operation_mode,
            interface: GsvInterface::User,
            state: Arc::new(Mutex::new(GsvState::default())),
        }
    }
}

impl std::fmt::Debug for GsvBoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Remote { account, version } => f
                .debug_struct("Remote")
                .field("version", version)
                .field("limit", &account.limit())
                .field("used", &account.used_bytes())
                .finish(),
            Self::Transfer(memory) => memory.fmt(f),
        }
    }
}

/// NvDrv2 descriptors and configuration belong to the client, shared by CMIF clones.
#[derive(Clone, Debug)]
pub struct NvDrv2Session {
    pub(crate) owner: GsvSession,
    pub(crate) state: Arc<Mutex<NvDrv2State>>,
}

#[derive(Debug, Default)]
pub(crate) struct NvDrv2State {
    pub configuration: u64,
    pub next_fd: u32,
    pub files: std::collections::BTreeMap<u32, NvDrv2Device>,
    pub events: std::collections::BTreeMap<(u32, u32), nixe_runtime::EventObject>,
}

#[derive(Clone, Debug)]
pub(crate) enum NvDrv2Device {
    ControlGpu,
    AddressSpaceGpu(Arc<Mutex<Option<u32>>>),
    ChannelGpu(Arc<Mutex<channel::Channel>>),
    TsgGpu(Arc<Mutex<Tsg>>),
    Control,
}
#[derive(Debug, Default)]
pub(crate) struct Tsg {
    pub enabled: bool,
    pub timeslice_us: u32,
    pub interleave: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GsvInterface {
    User,
    Syncpt,
    Gpu,
    Worker,
    Display,
}

#[derive(Default)]
pub(crate) struct SyncptContext {
    pub initialized: bool,
    pub slots: Vec<Option<nixe_runtime::EventObject>>,
    pub mirror: Option<SyncptMirror>,
    pub gpu_values: Option<nixe_memory::Allocation>,
    pub timelines: std::collections::BTreeMap<u32, Arc<Mutex<nixe_gpu::GuestTimeline>>>,
    pub next_timeline: u64,
}
impl std::fmt::Debug for SyncptContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncptContext")
            .field("initialized", &self.initialized)
            .field("slots", &self.slots)
            .field("mirror", &self.mirror)
            .field("gpu_values", &self.gpu_values.is_some())
            .finish()
    }
}

impl GsvSession {
    pub(crate) fn child(&self, interface: GsvInterface) -> Self {
        Self {
            interface,
            process_id: self.process_id,
            operation_mode: self.operation_mode,
            state: self.state.clone(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct GraphicsFrame {
    pub initialized: bool,
    pub length_mode: u32,
}
