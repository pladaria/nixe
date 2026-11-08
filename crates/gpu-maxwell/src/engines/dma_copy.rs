//! GM20B `MAXWELL_DMA_COPY_A` state and virtual-memory copy semantics.

use nixe_gpu::{GpuClassId, GpuMethodId};

use super::memory_copy::{
    MaxwellMemoryCopyComponentSource, MaxwellMemoryCopyLayout, MaxwellMemoryCopyOperation,
    MaxwellMemoryCopyRemap, required_range_size,
};

use super::{MaxwellEngineDispatchError, MaxwellEngineMethodMetadata};
use crate::MaxwellMethodSource;

pub(super) const CLASS: GpuClassId = GpuClassId(0xb0b5);
const CLASS_NAME: &str = "MAXWELL_DMA_COPY_A";
const GPU_ADDRESS_UPPER_MASK: u32 = 0xff;

/// One persistent DMA register selected independently of its method encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MaxwellDmaCopyRegisterName {
    SemaphoreAddressUpper,
    SemaphoreAddressLower,
    SemaphorePayload,
    RenderEnableAddressUpper,
    RenderEnableAddressLower,
    RenderEnableControl,
    SourcePhysicalTarget,
    DestinationPhysicalTarget,
    SourceAddressUpper,
    SourceAddressLower,
    DestinationAddressUpper,
    DestinationAddressLower,
    SourcePitch,
    DestinationPitch,
    LineLength,
    LineCount,
    RemapConstantA,
    RemapConstantB,
    RemapComponents,
    DestinationBlockDimensions,
    DestinationSizeX,
    DestinationSizeY,
    DestinationSizeZ,
    DestinationPositionZ,
    DestinationPositionXy,
    SourceBlockDimensions,
    SourceSizeX,
    SourceSizeY,
    SourceSizeZ,
    SourcePositionZ,
    SourcePositionXy,
    Launch,
}

const DMA_REGISTER_COUNT: usize = MaxwellDmaCopyRegisterName::Launch as usize + 1;

/// One source-preserving register in the DMA copy engine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaxwellDmaCopyRegister {
    raw: u32,
    source: Option<MaxwellMethodSource>,
}

impl MaxwellDmaCopyRegister {
    const fn programmed(raw: u32, source: MaxwellMethodSource) -> Self {
        Self {
            raw,
            source: Some(source),
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.raw
    }

    #[must_use]
    pub const fn source(self) -> Option<MaxwellMethodSource> {
        self.source
    }
}

/// Persistent state owned by one channel's `MAXWELL_DMA_COPY_A` object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellDmaCopyState {
    registers: [MaxwellDmaCopyRegister; DMA_REGISTER_COUNT],
}

impl Default for MaxwellDmaCopyState {
    fn default() -> Self {
        // DMA register reset state matches Eden's zero-initialized Regs bank.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/engines/maxwell_dma.h
        Self {
            registers: [MaxwellDmaCopyRegister::default(); DMA_REGISTER_COUNT],
        }
    }
}

impl MaxwellDmaCopyState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn register(&self, name: MaxwellDmaCopyRegisterName) -> MaxwellDmaCopyRegister {
        self.registers[name as usize]
    }

    fn apply(&mut self, write: MaxwellDmaCopyStateWrite) {
        self.registers[write.register as usize] =
            MaxwellDmaCopyRegister::programmed(write.value, write.source);
    }
}

/// One atomic persistent-state transition produced by a DMA method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellDmaCopyStateWrite {
    register: MaxwellDmaCopyRegisterName,
    value: u32,
    source: MaxwellMethodSource,
}

#[derive(Clone, Copy)]
struct MethodDeclaration {
    metadata: &'static MaxwellEngineMethodMetadata,
    defined_mask: u32,
    register: MaxwellDmaCopyRegisterName,
}

macro_rules! methods {
    ($($identifier:ident => ($method:literal, $name:literal, $mask:expr, $register:ident)),+ $(,)?) => {
        $(const $identifier: MaxwellEngineMethodMetadata = MaxwellEngineMethodMetadata::new(
            CLASS,
            CLASS_NAME,
            GpuMethodId($method),
            $name,
        );)+
        const METHODS: &[MethodDeclaration] = &[
            $(MethodDeclaration {
                metadata: &$identifier,
                defined_mask: $mask,
                register: MaxwellDmaCopyRegisterName::$register,
            }),+
        ];
    };
}

// The method offsets and bitfields are pinned to NVIDIA's public A0B5 copy
// class, inherited by GM20B's B0B5 class. The Maxwell block-dimension methods
// are additionally recorded by the pinned envytools register database.
// https://github.com/torvalds/linux/blob/v6.16/drivers/gpu/drm/nouveau/include/nvhw/class/cla0b5.h
// https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/rnndb/fifo/gk104_copy.xml
methods!(
    // Configuration alone does not release a semaphore. LAUNCH_DMA consumes
    // its semaphore flags and still rejects unimplemented completion modes.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/dma-copy/clb0b5.h
    SET_SEMAPHORE_A => (0x0240, "SET_SEMAPHORE_A", 0xff, SemaphoreAddressUpper),
    SET_SEMAPHORE_B => (0x0244, "SET_SEMAPHORE_B", u32::MAX, SemaphoreAddressLower),
    SET_SEMAPHORE_PAYLOAD => (0x0248, "SET_SEMAPHORE_PAYLOAD", u32::MAX, SemaphorePayload),
    SET_RENDER_ENABLE_A => (0x0254, "SET_RENDER_ENABLE_A", 0xff, RenderEnableAddressUpper),
    SET_RENDER_ENABLE_B => (0x0258, "SET_RENDER_ENABLE_B", u32::MAX, RenderEnableAddressLower),
    SET_RENDER_ENABLE_C => (0x025c, "SET_RENDER_ENABLE_C", 7, RenderEnableControl),
    SET_SRC_PHYS_MODE => (0x0260, "SET_SRC_PHYS_MODE", 3, SourcePhysicalTarget),
    SET_DST_PHYS_MODE => (0x0264, "SET_DST_PHYS_MODE", 3, DestinationPhysicalTarget),
    LAUNCH_DMA => (0x0300, "LAUNCH_DMA", 0x000f_ffff, Launch),
    OFFSET_IN_UPPER => (0x0400, "OFFSET_IN_UPPER", GPU_ADDRESS_UPPER_MASK, SourceAddressUpper),
    OFFSET_IN_LOWER => (0x0404, "OFFSET_IN_LOWER", u32::MAX, SourceAddressLower),
    OFFSET_OUT_UPPER => (0x0408, "OFFSET_OUT_UPPER", GPU_ADDRESS_UPPER_MASK, DestinationAddressUpper),
    OFFSET_OUT_LOWER => (0x040c, "OFFSET_OUT_LOWER", u32::MAX, DestinationAddressLower),
    PITCH_IN => (0x0410, "PITCH_IN", u32::MAX, SourcePitch),
    PITCH_OUT => (0x0414, "PITCH_OUT", u32::MAX, DestinationPitch),
    LINE_LENGTH_IN => (0x0418, "LINE_LENGTH_IN", u32::MAX, LineLength),
    LINE_COUNT => (0x041c, "LINE_COUNT", u32::MAX, LineCount),
    SET_REMAP_CONST_A => (0x0700, "SET_REMAP_CONST_A", u32::MAX, RemapConstantA),
    SET_REMAP_CONST_B => (0x0704, "SET_REMAP_CONST_B", u32::MAX, RemapConstantB),
    SET_REMAP_COMPONENTS => (0x0708, "SET_REMAP_COMPONENTS", 0x0333_7777, RemapComponents),
    SET_DST_BLOCK_SIZE => (0x070c, "SET_DST_BLOCK_SIZE", 0x0000_ffff, DestinationBlockDimensions),
    SET_DST_WIDTH => (0x0710, "SET_DST_WIDTH", u32::MAX, DestinationSizeX),
    SET_DST_HEIGHT => (0x0714, "SET_DST_HEIGHT", u32::MAX, DestinationSizeY),
    SET_DST_DEPTH => (0x0718, "SET_DST_DEPTH", u32::MAX, DestinationSizeZ),
    SET_DST_LAYER => (0x071c, "SET_DST_LAYER", u32::MAX, DestinationPositionZ),
    SET_DST_ORIGIN => (0x0720, "SET_DST_ORIGIN", u32::MAX, DestinationPositionXy),
    SET_SRC_BLOCK_SIZE => (0x0728, "SET_SRC_BLOCK_SIZE", 0x0000_ffff, SourceBlockDimensions),
    SET_SRC_WIDTH => (0x072c, "SET_SRC_WIDTH", u32::MAX, SourceSizeX),
    SET_SRC_HEIGHT => (0x0730, "SET_SRC_HEIGHT", u32::MAX, SourceSizeY),
    SET_SRC_DEPTH => (0x0734, "SET_SRC_DEPTH", u32::MAX, SourceSizeZ),
    SET_SRC_LAYER => (0x0738, "SET_SRC_LAYER", u32::MAX, SourcePositionZ),
    SET_SRC_ORIGIN => (0x073c, "SET_SRC_ORIGIN", u32::MAX, SourcePositionXy),
);

pub(super) fn call_method<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    source: MaxwellMethodSource,
    candidate: &mut MaxwellDmaCopyState,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<bool, super::MaxwellEngineStreamError> {
    let declaration = METHODS
        .iter()
        .find(|declaration| declaration.metadata.method() == source.method())
        .ok_or(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: CLASS_NAME,
        })?;
    if source.argument() & !declaration.defined_mask != 0 {
        return Err((MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            metadata: declaration.metadata,
            defined_mask: declaration.defined_mask,
        })
        .into());
    }

    if (declaration.register == MaxwellDmaCopyRegisterName::RenderEnableControl
        && source.argument() > 4)
        || (matches!(
            declaration.register,
            MaxwellDmaCopyRegisterName::SourcePhysicalTarget
                | MaxwellDmaCopyRegisterName::DestinationPhysicalTarget
        ) && source.argument() > 2)
    {
        return Err((MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            metadata: declaration.metadata,
            defined_mask: declaration.defined_mask,
        })
        .into());
    }
    let write = MaxwellDmaCopyStateWrite {
        register: declaration.register,
        value: source.argument(),
        source,
    };
    let operation = if declaration.register == MaxwellDmaCopyRegisterName::Launch {
        Some(build_operation(candidate, source)?)
    } else {
        None
    };
    candidate.apply(write);
    if let Some(operation) = operation {
        execution.copy_memory(operation)?;
    }
    Ok(true)
}

fn build_operation(
    state: &MaxwellDmaCopyState,
    source: MaxwellMethodSource,
) -> Result<MaxwellMemoryCopyOperation, MaxwellEngineDispatchError> {
    let raw = source.argument();
    let copy_mode = raw & 0x3;
    if !matches!(copy_mode, 1 | 2) {
        return Err(invalid_encoding(
            source,
            "LAUNCH_DMA requires a copy transfer mode",
        ));
    }
    if raw & 0x000f_f860 != 0 {
        return Err(invalid_encoding(
            source,
            "interrupts, physical addressing, L2 bypass, and reductions are not implemented",
        ));
    }
    let semaphore_release = match (raw >> 3) & 3 {
        0 => None,
        1 => {
            let address = address(
                state,
                MaxwellDmaCopyRegisterName::SemaphoreAddressUpper,
                MaxwellDmaCopyRegisterName::SemaphoreAddressLower,
            );
            if address & 3 != 0 {
                return Err(invalid_encoding(
                    source,
                    "one-word DMA semaphore address must be four-byte aligned",
                ));
            }
            Some((
                address,
                state
                    .register(MaxwellDmaCopyRegisterName::SemaphorePayload)
                    .raw(),
            ))
        }
        _ => {
            return Err(invalid_encoding(
                source,
                "four-word or reserved DMA semaphore releases are not implemented",
            ));
        }
    };
    let multi_line = raw & (1 << 9) != 0;
    let remap_enabled = raw & (1 << 10) != 0;
    let width = state.register(MaxwellDmaCopyRegisterName::LineLength).raw();
    let height = if multi_line {
        state.register(MaxwellDmaCopyRegisterName::LineCount).raw()
    } else {
        1
    };
    if width == 0 || height == 0 {
        return Err(invalid_encoding(source, "DMA dimensions must be nonzero"));
    }

    let source_address = address(
        state,
        MaxwellDmaCopyRegisterName::SourceAddressUpper,
        MaxwellDmaCopyRegisterName::SourceAddressLower,
    );
    let destination_address = address(
        state,
        MaxwellDmaCopyRegisterName::DestinationAddressUpper,
        MaxwellDmaCopyRegisterName::DestinationAddressLower,
    );
    let remap = remap_enabled
        .then(|| parse_remap(state, source))
        .transpose()?;
    let (source_element_bytes, destination_element_bytes) = remap.map_or((1, 1), |remap| {
        (
            u32::from(remap.component_bytes) * u32::from(remap.source_components),
            u32::from(remap.component_bytes) * u32::from(remap.destination_components),
        )
    });
    // Like Eden MaxwellDMA::Launch, single-line transfers are contiguous;
    // line count, pitch and surface-geometry registers belong to multi-line mode.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/engines/maxwell_dma.cpp
    let (source_layout, destination_layout) = if multi_line {
        (
            layout(
                state,
                raw & (1 << 7) != 0,
                true,
                width,
                height,
                source_element_bytes,
                source,
            )?,
            layout(
                state,
                raw & (1 << 8) != 0,
                false,
                width,
                height,
                destination_element_bytes,
                source,
            )?,
        )
    } else {
        let contiguous = |element_bytes| {
            width
                .checked_mul(element_bytes)
                .map(|pitch| MaxwellMemoryCopyLayout::Pitch { pitch })
                .ok_or_else(|| invalid_encoding(source, "DMA row size overflows"))
        };
        (
            contiguous(source_element_bytes)?,
            contiguous(destination_element_bytes)?,
        )
    };
    let source_range_size =
        required_range_size(source_layout, width, height, source_element_bytes)?;
    let destination_range_size =
        required_range_size(destination_layout, width, height, destination_element_bytes)?;
    if source_address
        .checked_add(source_range_size)
        .is_none_or(|end| end > 1_u64 << 40)
        || destination_address
            .checked_add(destination_range_size)
            .is_none_or(|end| end > 1_u64 << 40)
    {
        return Err(invalid_encoding(source, "DMA GPU range overflows"));
    }

    Ok(MaxwellMemoryCopyOperation {
        semaphore_release,
        source_address,
        destination_address,
        source_layout,
        destination_layout,
        width,
        height,
        remap,
        source_range_size,
        destination_range_size,
        source,
    })
}

fn address(
    state: &MaxwellDmaCopyState,
    upper: MaxwellDmaCopyRegisterName,
    lower: MaxwellDmaCopyRegisterName,
) -> u64 {
    (u64::from(state.register(upper).raw()) << 32) | u64::from(state.register(lower).raw())
}

fn parse_remap(
    state: &MaxwellDmaCopyState,
    source: MaxwellMethodSource,
) -> Result<MaxwellMemoryCopyRemap, MaxwellEngineDispatchError> {
    let raw = state
        .register(MaxwellDmaCopyRegisterName::RemapComponents)
        .raw();
    let mut components = [MaxwellMemoryCopyComponentSource::NoWrite; 4];
    for (index, component) in components.iter_mut().enumerate() {
        *component = match (raw >> (index * 4)) & 0x7 {
            value @ 0..=3 => MaxwellMemoryCopyComponentSource::Source(value as u8),
            4 => MaxwellMemoryCopyComponentSource::ConstantA,
            5 => MaxwellMemoryCopyComponentSource::ConstantB,
            6 => MaxwellMemoryCopyComponentSource::NoWrite,
            _ => return Err(invalid_encoding(source, "invalid component remap selector")),
        };
    }
    let remap = MaxwellMemoryCopyRemap {
        components,
        component_bytes: ((raw >> 16) & 0x3) as u8 + 1,
        source_components: ((raw >> 20) & 0x3) as u8 + 1,
        destination_components: ((raw >> 24) & 0x3) as u8 + 1,
        constant_a: state
            .register(MaxwellDmaCopyRegisterName::RemapConstantA)
            .raw(),
        constant_b: state
            .register(MaxwellDmaCopyRegisterName::RemapConstantB)
            .raw(),
    };
    if remap.components[..remap.destination_components as usize]
        .iter()
        .any(|component| {
            matches!(component, MaxwellMemoryCopyComponentSource::Source(index) if *index >= remap.source_components)
        })
    {
        return Err(invalid_encoding(
            source,
            "component remap reads beyond the configured source element",
        ));
    }
    Ok(remap)
}

fn layout(
    state: &MaxwellDmaCopyState,
    pitch: bool,
    source_side: bool,
    width: u32,
    height: u32,
    element_bytes: u32,
    source: MaxwellMethodSource,
) -> Result<MaxwellMemoryCopyLayout, MaxwellEngineDispatchError> {
    let (pitch_register, block, size_x, size_y, size_z, position_z, position_xy) = if source_side {
        (
            MaxwellDmaCopyRegisterName::SourcePitch,
            MaxwellDmaCopyRegisterName::SourceBlockDimensions,
            MaxwellDmaCopyRegisterName::SourceSizeX,
            MaxwellDmaCopyRegisterName::SourceSizeY,
            MaxwellDmaCopyRegisterName::SourceSizeZ,
            MaxwellDmaCopyRegisterName::SourcePositionZ,
            MaxwellDmaCopyRegisterName::SourcePositionXy,
        )
    } else {
        (
            MaxwellDmaCopyRegisterName::DestinationPitch,
            MaxwellDmaCopyRegisterName::DestinationBlockDimensions,
            MaxwellDmaCopyRegisterName::DestinationSizeX,
            MaxwellDmaCopyRegisterName::DestinationSizeY,
            MaxwellDmaCopyRegisterName::DestinationSizeZ,
            MaxwellDmaCopyRegisterName::DestinationPositionZ,
            MaxwellDmaCopyRegisterName::DestinationPositionXy,
        )
    };
    if pitch {
        let pitch = state.register(pitch_register).raw();
        let row_bytes = width
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid_encoding(source, "DMA row size overflows"))?;
        if pitch < row_bytes {
            return Err(invalid_encoding(
                source,
                "DMA pitch is shorter than one copied row",
            ));
        }
        return Ok(MaxwellMemoryCopyLayout::Pitch { pitch });
    }

    let dimensions = state.register(block).raw();
    let gob_height = (dimensions >> 12) & 0xf;
    let block_depth_log2 = (dimensions >> 8) & 0xf;
    let block_height_log2 = ((dimensions >> 4) & 0xf) as u8;
    let block_width_log2 = dimensions & 0xf;
    if gob_height != 1 || block_width_log2 != 0 || block_depth_log2 != 0 || block_height_log2 > 5 {
        return Err(invalid_encoding(
            source,
            "only 16Bx2 GOBs with unit block width/depth are implemented",
        ));
    }
    let surface_width = state.register(size_x).raw();
    let surface_height = state.register(size_y).raw();
    if state.register(size_z).raw() != 1 || state.register(position_z).raw() != 0 {
        return Err(invalid_encoding(
            source,
            "three-dimensional block-linear DMA is not implemented",
        ));
    }
    let position = state.register(position_xy).raw();
    let x = position & 0xffff;
    let y = position >> 16;
    if x.checked_add(width).is_none_or(|end| end > surface_width)
        || y.checked_add(height).is_none_or(|end| end > surface_height)
    {
        return Err(invalid_encoding(
            source,
            "DMA rectangle exceeds its block-linear surface",
        ));
    }
    Ok(MaxwellMemoryCopyLayout::BlockLinear {
        surface_width,
        surface_height,
        x,
        y,
        block_height_log2,
    })
}

fn invalid_encoding(
    source: MaxwellMethodSource,
    reason: &'static str,
) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidDmaCopyMethodEncoding {
        source,
        method_name: "LAUNCH_DMA",
        reason,
    }
}
