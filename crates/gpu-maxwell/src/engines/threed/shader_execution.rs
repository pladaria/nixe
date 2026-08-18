//! Typed `MAXWELL_B` shader-execution configuration.
//!
//! Register storage is deliberately separate from shader translation and host
//! watchdog policy. A verified field encoding does not establish its time
//! unit or execution effect.

use crate::MaxwellMethodSource;

use super::{MaxwellThreeDRegister, MaxwellThreeDUnresolvedAddress};

/// Whether API semantics require depth/stencil testing before pixel shading.
///
/// NVIDIA publishes a one-bit enable field. The ordering becomes observable
/// for pixel shaders with discard or side effects, so an active value must not
/// be treated as a performance hint:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L249-L252>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDApiMandatedEarlyZ {
    Disabled = 0,
    Enabled = 1,
}

/// Conflict-detection granularity for Maxwell pixel-shader interlocks.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPixelShaderInterlockMode {
    NoConflictDetect = 0,
    ConflictDetectSample = 1,
    ConflictDetectPixel = 2,
}

/// Tile size used by the pixel-shader interlock coalescer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPixelShaderInterlockTileSize {
    Tile16x16 = 0,
    Tile8x8 = 1,
}

/// Fragment-order policy used by the pixel-shader interlock coalescer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDPixelShaderInterlockFragmentOrder {
    Ordered = 0,
    Unordered = 1,
}

/// Validated `SET_PIXEL_SHADER_INTERLOCK_CONTROL` fields.
///
/// NVIDIA publishes the two-bit mode and the two one-bit coalescer fields:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2030-L2040>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDPixelShaderInterlockControl {
    mode: MaxwellThreeDPixelShaderInterlockMode,
    tile_size: MaxwellThreeDPixelShaderInterlockTileSize,
    fragment_order: MaxwellThreeDPixelShaderInterlockFragmentOrder,
}

impl MaxwellThreeDPixelShaderInterlockControl {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x0f != 0 {
            return None;
        }
        let mode = match raw & 3 {
            0 => MaxwellThreeDPixelShaderInterlockMode::NoConflictDetect,
            1 => MaxwellThreeDPixelShaderInterlockMode::ConflictDetectSample,
            2 => MaxwellThreeDPixelShaderInterlockMode::ConflictDetectPixel,
            _ => return None,
        };
        Some(Self {
            mode,
            tile_size: if raw & 4 == 0 {
                MaxwellThreeDPixelShaderInterlockTileSize::Tile16x16
            } else {
                MaxwellThreeDPixelShaderInterlockTileSize::Tile8x8
            },
            fragment_order: if raw & 8 == 0 {
                MaxwellThreeDPixelShaderInterlockFragmentOrder::Ordered
            } else {
                MaxwellThreeDPixelShaderInterlockFragmentOrder::Unordered
            },
        })
    }

    #[must_use]
    pub const fn mode(self) -> MaxwellThreeDPixelShaderInterlockMode {
        self.mode
    }

    #[must_use]
    pub const fn tile_size(self) -> MaxwellThreeDPixelShaderInterlockTileSize {
        self.tile_size
    }

    #[must_use]
    pub const fn fragment_order(self) -> MaxwellThreeDPixelShaderInterlockFragmentOrder {
        self.fragment_order
    }

    #[must_use]
    pub const fn conflict_detection_enabled(self) -> bool {
        !matches!(
            self.mode,
            MaxwellThreeDPixelShaderInterlockMode::NoConflictDetect
        )
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.mode as u32 | (self.tile_size as u32) << 2 | (self.fragment_order as u32) << 3
    }
}

impl MaxwellThreeDApiMandatedEarlyZ {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// Whether Maxwell shader-exception reporting is enabled.
///
/// This is guest execution-diagnostic policy rather than a host shader or
/// pipeline capability. NVIDIA's public class header defines the boolean
/// field but does not specify an exception payload or reporting transport.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2717-L2720>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDShaderExceptionsEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDShaderExceptionsEnable {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Maxwell SPM resource fractions requested for each hardware subtile.
///
/// These fields are performance policy for Maxwell's internal work
/// distribution. They are retained for diagnostics and replay, but are not a
/// semantic host-pipeline dependency.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L495-L499>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDSubtilingPerfKnobA {
    register_file: u8,
    pixel_output_buffer: u8,
    triangle_ram: u8,
    max_quads: u8,
}

impl MaxwellThreeDSubtilingPerfKnobA {
    #[must_use]
    pub const fn parse(raw: u32) -> Self {
        Self {
            register_file: raw as u8,
            pixel_output_buffer: (raw >> 8) as u8,
            triangle_ram: (raw >> 16) as u8,
            max_quads: (raw >> 24) as u8,
        }
    }

    #[must_use]
    pub const fn register_file(self) -> u8 {
        self.register_file
    }

    #[must_use]
    pub const fn pixel_output_buffer(self) -> u8 {
        self.pixel_output_buffer
    }

    #[must_use]
    pub const fn triangle_ram(self) -> u8 {
        self.triangle_ram
    }

    #[must_use]
    pub const fn max_quads(self) -> u8 {
        self.max_quads
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.register_file as u32
            | (self.pixel_output_buffer as u32) << 8
            | (self.triangle_ram as u32) << 16
            | (self.max_quads as u32) << 24
    }
}

/// Maxwell maximum-primitive fraction requested for each hardware subtile.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L501-L502>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDSubtilingPerfKnobB(u8);

impl MaxwellThreeDSubtilingPerfKnobB {
    #[must_use]
    pub const fn new(raw: u32) -> Option<Self> {
        if raw <= u8::MAX as u32 {
            Some(Self(raw as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn max_primitives(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// Low and high hardware-occupancy watermarks for one shader resource.
///
/// NVIDIA exposes both fields as independent 16-bit values. Their scheduling
/// effect is internal to the guest GPU, so neutral lowering retains them for
/// diagnostics and replay without making them host-pipeline dependencies.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cl9297.h>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDShaderWatermarkRange {
    low: u16,
    high: u16,
}

impl MaxwellThreeDShaderWatermarkRange {
    #[must_use]
    pub const fn parse(raw: u32) -> Self {
        Self {
            low: raw as u16,
            high: (raw >> 16) as u16,
        }
    }

    #[must_use]
    pub const fn low(self) -> u16 {
        self.low
    }

    #[must_use]
    pub const fn high(self) -> u16 {
        self.high
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.low as u32 | (self.high as u32) << 16
    }
}

/// Shader resource whose internal occupancy watermarks are being programmed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellThreeDShaderWatermarkTarget {
    VertexTessellationGeometryWarps,
    PixelWarps,
    PixelRegisters,
}

/// Largest byte count representable by
/// `SET_SHADER_LOCAL_MEMORY_E.DEFAULT_SIZE_PER_WARP`.
pub const MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX: u32 = 0x03ff_ffff;

/// Default shader-local-memory allocation requested for one Maxwell warp.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L603-L604>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDShaderLocalMemoryPerWarpSize(u32);

impl MaxwellThreeDShaderLocalMemoryPerWarpSize {
    #[must_use]
    pub const fn new(raw: u32) -> Option<Self> {
        if raw <= MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX {
            Some(Self(raw))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Backing-region and allocation registers for Maxwell shader local memory.
///
/// NVIDIA exposes a 40-bit address, a 38-bit size, a per-warp default size,
/// and an independent 32-bit window base. Each register remains independently
/// sourced because the public ABI does not define hardware reset values.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L588-L604>
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDShaderLocalMemoryState {
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    size_upper: MaxwellThreeDRegister<u8>,
    size_lower: MaxwellThreeDRegister<u32>,
    default_size_per_warp: MaxwellThreeDRegister<MaxwellThreeDShaderLocalMemoryPerWarpSize>,
    window_base_address: MaxwellThreeDRegister<u32>,
}

impl MaxwellThreeDShaderLocalMemoryState {
    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub fn address(&self) -> Option<MaxwellThreeDUnresolvedAddress> {
        Some(MaxwellThreeDUnresolvedAddress::new(
            *self.address_upper.value()?,
            *self.address_lower.value()?,
        ))
    }

    #[must_use]
    pub const fn size_upper(&self) -> &MaxwellThreeDRegister<u8> {
        &self.size_upper
    }

    #[must_use]
    pub const fn size_lower(&self) -> &MaxwellThreeDRegister<u32> {
        &self.size_lower
    }

    #[must_use]
    pub fn size(&self) -> Option<u64> {
        Some((u64::from(*self.size_upper.value()?) << 32) | u64::from(*self.size_lower.value()?))
    }

    #[must_use]
    pub const fn default_size_per_warp(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDShaderLocalMemoryPerWarpSize> {
        &self.default_size_per_warp
    }

    #[must_use]
    pub const fn window_base_address(&self) -> &MaxwellThreeDRegister<u32> {
        &self.window_base_address
    }

    pub(super) fn region_is_partially_programmed(&self) -> bool {
        let programmed = self.address_upper.raw().is_some()
            || self.address_lower.raw().is_some()
            || self.size_upper.raw().is_some()
            || self.size_lower.raw().is_some();
        programmed && (self.address().is_none() || self.size().is_none())
    }

    fn append_pipeline_dependencies(&self, dependencies: &mut Vec<Option<u32>>) {
        dependencies.push(self.address_upper.raw());
        dependencies.push(self.address_lower.raw());
        dependencies.push(self.size_upper.raw());
        dependencies.push(self.size_lower.raw());
        dependencies.push(self.default_size_per_warp.raw());
        dependencies.push(self.window_base_address.raw());
    }
}

/// Directly addressable memory partition selected by `SET_L1_CONFIGURATION`.
///
/// This is guest Maxwell shader memory, not a description of a host CPU/GPU
/// cache. NVIDIA's public class header defines only these two selector values;
/// it does not establish a reset value.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L388-L391>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDDirectlyAddressableMemory {
    Size16KiB = 1,
    Size48KiB = 3,
}

impl MaxwellThreeDDirectlyAddressableMemory {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            1 => Some(Self::Size16KiB),
            3 => Some(Self::Size48KiB),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Size16KiB => 16 * 1024,
            Self::Size48KiB => 48 * 1024,
        }
    }
}

/// API-visible draw-call limit selected by `SET_API_VISIBLE_CALL_LIMIT`.
///
/// The numeric method encodings are selectors, not literal call counts: for
/// example, encoding eight selects a limit of 128 visible calls. NVIDIA's
/// public class header also defines `NoCheck` explicitly. Finite selectors are
/// checked later against conservative call-use evidence emitted by T10 rather
/// than being treated as host scheduling hints.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L871-L885>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDVisibleCallLimit {
    Calls0 = 0,
    Calls1 = 1,
    Calls2 = 2,
    Calls4 = 3,
    Calls8 = 4,
    Calls16 = 5,
    Calls32 = 6,
    Calls64 = 7,
    Calls128 = 8,
    NoCheck = 15,
}

impl MaxwellThreeDVisibleCallLimit {
    #[must_use]
    pub const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Calls0),
            1 => Some(Self::Calls1),
            2 => Some(Self::Calls2),
            3 => Some(Self::Calls4),
            4 => Some(Self::Calls8),
            5 => Some(Self::Calls16),
            6 => Some(Self::Calls32),
            7 => Some(Self::Calls64),
            8 => Some(Self::Calls128),
            15 => Some(Self::NoCheck),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }

    /// Returns the verified limit selected by this encoding without claiming
    /// what constitutes an API-visible call or where hardware accounts it.
    #[must_use]
    pub const fn limit(self) -> Option<u16> {
        match self {
            Self::Calls0 => Some(0),
            Self::Calls1 => Some(1),
            Self::Calls2 => Some(2),
            Self::Calls4 => Some(4),
            Self::Calls8 => Some(8),
            Self::Calls16 => Some(16),
            Self::Calls32 => Some(32),
            Self::Calls64 => Some(64),
            Self::Calls128 => Some(128),
            Self::NoCheck => None,
        }
    }
}

/// Largest value representable by `SET_SM_TIMEOUT_INTERVAL.COUNTER_BIT`.
pub const MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX: u32 = 0x3f;

/// Source-preserving six-bit `COUNTER_BIT` field.
///
/// NVIDIA's public class header defines the field width but does not document
/// a time unit, duration formula, or watchdog behavior. Neutral lowering keeps
/// this guest execution-policy value for diagnostics but does not derive a host
/// deadline from it; host watchdog and device-loss handling remain independent.
///
/// ABI source:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1079-L1090>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDSmTimeoutCounterBit(u8);

impl MaxwellThreeDSmTimeoutCounterBit {
    #[must_use]
    pub const fn new(raw: u32) -> Option<Self> {
        if raw <= MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX {
            Some(Self(raw as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0 as u32
    }
}

/// One validated shader-execution register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDShaderExecutionStateWrite {
    PixelShaderInterlockControl {
        value: MaxwellThreeDPixelShaderInterlockControl,
        source: MaxwellMethodSource,
    },
    ApiMandatedEarlyZ {
        value: MaxwellThreeDApiMandatedEarlyZ,
        source: MaxwellMethodSource,
    },
    ShaderExceptionsEnable {
        value: MaxwellThreeDShaderExceptionsEnable,
        source: MaxwellMethodSource,
    },
    SubtilingPerfKnobA {
        value: MaxwellThreeDSubtilingPerfKnobA,
        source: MaxwellMethodSource,
    },
    SubtilingPerfKnobB {
        value: MaxwellThreeDSubtilingPerfKnobB,
        source: MaxwellMethodSource,
    },
    ShaderWatermarks {
        target: MaxwellThreeDShaderWatermarkTarget,
        value: MaxwellThreeDShaderWatermarkRange,
        source: MaxwellMethodSource,
    },
    L1Configuration {
        value: MaxwellThreeDDirectlyAddressableMemory,
        source: MaxwellMethodSource,
    },
    VisibleCallLimit {
        value: MaxwellThreeDVisibleCallLimit,
        source: MaxwellMethodSource,
    },
    SmTimeoutCounterBit {
        value: MaxwellThreeDSmTimeoutCounterBit,
        source: MaxwellMethodSource,
    },
    ShaderLocalMemoryWindowBaseAddress {
        value: u32,
        source: MaxwellMethodSource,
    },
    ShaderLocalMemoryAddressUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    ShaderLocalMemoryAddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    ShaderLocalMemorySizeUpper {
        value: u8,
        source: MaxwellMethodSource,
    },
    ShaderLocalMemorySizeLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    ShaderLocalMemoryDefaultSizePerWarp {
        value: MaxwellThreeDShaderLocalMemoryPerWarpSize,
        source: MaxwellMethodSource,
    },
}

/// Persistent shader-execution configuration on one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDShaderExecutionState {
    pixel_shader_interlock_control: MaxwellThreeDRegister<MaxwellThreeDPixelShaderInterlockControl>,
    api_mandated_early_z: MaxwellThreeDRegister<MaxwellThreeDApiMandatedEarlyZ>,
    shader_exceptions_enable: MaxwellThreeDRegister<MaxwellThreeDShaderExceptionsEnable>,
    subtiling_perf_knob_a: MaxwellThreeDRegister<MaxwellThreeDSubtilingPerfKnobA>,
    subtiling_perf_knob_b: MaxwellThreeDRegister<MaxwellThreeDSubtilingPerfKnobB>,
    vtg_warp_watermarks: MaxwellThreeDRegister<MaxwellThreeDShaderWatermarkRange>,
    ps_warp_watermarks: MaxwellThreeDRegister<MaxwellThreeDShaderWatermarkRange>,
    ps_register_watermarks: MaxwellThreeDRegister<MaxwellThreeDShaderWatermarkRange>,
    l1_configuration: MaxwellThreeDRegister<MaxwellThreeDDirectlyAddressableMemory>,
    visible_call_limit: MaxwellThreeDRegister<MaxwellThreeDVisibleCallLimit>,
    sm_timeout_counter_bit: MaxwellThreeDRegister<MaxwellThreeDSmTimeoutCounterBit>,
    shader_local_memory: MaxwellThreeDShaderLocalMemoryState,
}

impl MaxwellThreeDShaderExecutionState {
    #[must_use]
    pub const fn pixel_shader_interlock_control(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDPixelShaderInterlockControl> {
        &self.pixel_shader_interlock_control
    }

    #[must_use]
    pub const fn api_mandated_early_z(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDApiMandatedEarlyZ> {
        &self.api_mandated_early_z
    }

    #[must_use]
    pub const fn shader_exceptions_enable(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDShaderExceptionsEnable> {
        &self.shader_exceptions_enable
    }

    #[must_use]
    pub const fn subtiling_perf_knob_a(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSubtilingPerfKnobA> {
        &self.subtiling_perf_knob_a
    }

    #[must_use]
    pub const fn subtiling_perf_knob_b(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSubtilingPerfKnobB> {
        &self.subtiling_perf_knob_b
    }

    #[must_use]
    pub const fn vtg_warp_watermarks(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDShaderWatermarkRange> {
        &self.vtg_warp_watermarks
    }

    #[must_use]
    pub const fn ps_warp_watermarks(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDShaderWatermarkRange> {
        &self.ps_warp_watermarks
    }

    #[must_use]
    pub const fn ps_register_watermarks(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDShaderWatermarkRange> {
        &self.ps_register_watermarks
    }

    #[must_use]
    pub const fn l1_configuration(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDDirectlyAddressableMemory> {
        &self.l1_configuration
    }

    #[must_use]
    pub const fn visible_call_limit(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDVisibleCallLimit> {
        &self.visible_call_limit
    }

    #[must_use]
    pub const fn sm_timeout_counter_bit(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDSmTimeoutCounterBit> {
        &self.sm_timeout_counter_bit
    }

    #[must_use]
    pub const fn shader_local_memory(&self) -> &MaxwellThreeDShaderLocalMemoryState {
        &self.shader_local_memory
    }

    pub(super) fn append_shader_pipeline_dependencies(&self, dependencies: &mut Vec<Option<u32>>) {
        if self.api_mandated_early_z.value() == Some(&MaxwellThreeDApiMandatedEarlyZ::Enabled) {
            dependencies.push(self.api_mandated_early_z.raw());
        }
        if self
            .pixel_shader_interlock_control
            .value()
            .is_some_and(|value| value.conflict_detection_enabled())
        {
            dependencies.push(self.pixel_shader_interlock_control.raw());
        }
        self.shader_local_memory
            .append_pipeline_dependencies(dependencies);
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDShaderExecutionStateWrite) {
        match write {
            MaxwellThreeDShaderExecutionStateWrite::PixelShaderInterlockControl {
                value,
                source,
            } => {
                self.pixel_shader_interlock_control =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ApiMandatedEarlyZ { value, source } => {
                self.api_mandated_early_z =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderExceptionsEnable { value, source } => {
                self.shader_exceptions_enable =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::SubtilingPerfKnobA { value, source } => {
                self.subtiling_perf_knob_a =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::SubtilingPerfKnobB { value, source } => {
                self.subtiling_perf_knob_b =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderWatermarks {
                target,
                value,
                source,
            } => {
                let register = match target {
                    MaxwellThreeDShaderWatermarkTarget::VertexTessellationGeometryWarps => {
                        &mut self.vtg_warp_watermarks
                    }
                    MaxwellThreeDShaderWatermarkTarget::PixelWarps => &mut self.ps_warp_watermarks,
                    MaxwellThreeDShaderWatermarkTarget::PixelRegisters => {
                        &mut self.ps_register_watermarks
                    }
                };
                *register = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::L1Configuration { value, source } => {
                self.l1_configuration =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::VisibleCallLimit { value, source } => {
                self.visible_call_limit =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::SmTimeoutCounterBit { value, source } => {
                self.sm_timeout_counter_bit =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryWindowBaseAddress {
                value,
                source,
            } => {
                self.shader_local_memory.window_base_address =
                    MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryAddressUpper {
                value,
                source,
            } => {
                self.shader_local_memory.address_upper =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryAddressLower {
                value,
                source,
            } => {
                self.shader_local_memory.address_lower =
                    MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemorySizeUpper {
                value,
                source,
            } => {
                self.shader_local_memory.size_upper =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemorySizeLower {
                value,
                source,
            } => {
                self.shader_local_memory.size_lower =
                    MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryDefaultSizePerWarp {
                value,
                source,
            } => {
                self.shader_local_memory.default_size_per_warp =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
