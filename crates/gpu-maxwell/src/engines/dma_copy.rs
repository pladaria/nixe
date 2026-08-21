//! GM20B `MAXWELL_DMA_COPY_A` state and virtual-memory copy semantics.

use std::fmt::{Display, Formatter};

use nixe_gpu::{GpuClassId, GpuMethodId};

use super::{
    AppliedMethod, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata, MaxwellEngineOperation,
};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};

pub(super) const CLASS: GpuClassId = GpuClassId(0xb0b5);
const CLASS_NAME: &str = "MAXWELL_DMA_COPY_A";
const GPU_ADDRESS_UPPER_MASK: u32 = 0xff;

/// One persistent DMA register selected independently of its method encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MaxwellDmaCopyRegisterName {
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
    raw: Option<u32>,
    source: Option<MaxwellMethodSource>,
}

impl MaxwellDmaCopyRegister {
    const fn programmed(raw: u32, source: MaxwellMethodSource) -> Self {
        Self {
            raw: Some(raw),
            source: Some(source),
        }
    }

    #[must_use]
    pub const fn raw(self) -> Option<u32> {
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

/// Memory organization selected for one side of a DMA copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellDmaCopyMemoryLayout {
    Pitch {
        pitch: u32,
    },
    BlockLinear {
        surface_width: u32,
        surface_height: u32,
        x: u32,
        y: u32,
        block_height_log2: u8,
    },
}

/// Source selected for one remapped destination component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellDmaCopyComponentSource {
    Source(u8),
    ConstantA,
    ConstantB,
    NoWrite,
}

/// Component mapping applied independently to every copied element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellDmaCopyRemap {
    components: [MaxwellDmaCopyComponentSource; 4],
    component_bytes: u8,
    source_components: u8,
    destination_components: u8,
    constant_a: u32,
    constant_b: u32,
}

impl MaxwellDmaCopyRemap {
    #[must_use]
    pub const fn components(self) -> [MaxwellDmaCopyComponentSource; 4] {
        self.components
    }

    #[must_use]
    pub const fn component_bytes(self) -> u8 {
        self.component_bytes
    }

    #[must_use]
    pub const fn source_components(self) -> u8 {
        self.source_components
    }

    #[must_use]
    pub const fn destination_components(self) -> u8 {
        self.destination_components
    }
}

/// Fully validated virtual-memory copy emitted at `LAUNCH_DMA`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellDmaCopyOperation {
    source_address: u64,
    destination_address: u64,
    source_layout: MaxwellDmaCopyMemoryLayout,
    destination_layout: MaxwellDmaCopyMemoryLayout,
    width: u32,
    height: u32,
    remap: Option<MaxwellDmaCopyRemap>,
    source_range_size: u64,
    destination_range_size: u64,
    source: MaxwellMethodSource,
}

impl MaxwellDmaCopyOperation {
    #[must_use]
    pub const fn source_address(self) -> u64 {
        self.source_address
    }

    #[must_use]
    pub const fn destination_address(self) -> u64 {
        self.destination_address
    }

    #[must_use]
    pub const fn source_range_size(self) -> u64 {
        self.source_range_size
    }

    #[must_use]
    pub const fn destination_range_size(self) -> u64 {
        self.destination_range_size
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }

    pub(crate) fn copy_bytes(
        self,
        source: &[u8],
        destination: &mut [u8],
    ) -> Result<(), MaxwellDmaCopyError> {
        if source.len() as u64 != self.source_range_size
            || destination.len() as u64 != self.destination_range_size
        {
            return Err(MaxwellDmaCopyError::RangeSizeMismatch);
        }
        let (source_element_bytes, destination_element_bytes) = self.element_sizes();
        for y in 0..self.height {
            for x in 0..self.width {
                let source_offset = layout_offset(self.source_layout, x, y, source_element_bytes)?;
                let destination_offset =
                    layout_offset(self.destination_layout, x, y, destination_element_bytes)?;
                let source_end = source_offset
                    .checked_add(source_element_bytes as usize)
                    .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?;
                let destination_end = destination_offset
                    .checked_add(destination_element_bytes as usize)
                    .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?;
                let source_element = source
                    .get(source_offset..source_end)
                    .ok_or(MaxwellDmaCopyError::RangeSizeMismatch)?;
                let destination_element = destination
                    .get_mut(destination_offset..destination_end)
                    .ok_or(MaxwellDmaCopyError::RangeSizeMismatch)?;
                if let Some(remap) = self.remap {
                    remap_element(remap, source_element, destination_element);
                } else {
                    destination_element.copy_from_slice(source_element);
                }
            }
        }
        Ok(())
    }

    const fn element_sizes(self) -> (u32, u32) {
        match self.remap {
            Some(remap) => (
                remap.component_bytes as u32 * remap.source_components as u32,
                remap.component_bytes as u32 * remap.destination_components as u32,
            ),
            None => (1, 1),
        }
    }
}

/// Failure while applying a previously validated DMA operation to bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellDmaCopyError {
    ArithmeticOverflow,
    RangeSizeMismatch,
    ResourceExhausted,
}

impl Display for MaxwellDmaCopyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ArithmeticOverflow => "DMA copy address arithmetic overflowed",
            Self::RangeSizeMismatch => "DMA copy byte ranges do not match the validated layout",
            Self::ResourceExhausted => "DMA copy exhausted host resources",
        })
    }
}

impl std::error::Error for MaxwellDmaCopyError {}

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

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellDmaCopyState,
) -> Result<AppliedMethod, MaxwellEngineDispatchError> {
    let source = method.source();
    let declaration = METHODS
        .iter()
        .find(|declaration| declaration.metadata.method() == source.method())
        .ok_or(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: CLASS_NAME,
        })?;
    if source.argument() & !declaration.defined_mask != 0 {
        return Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            metadata: declaration.metadata,
            defined_mask: declaration.defined_mask,
        });
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
    Ok(AppliedMethod::new(
        method,
        *declaration.metadata,
        operation.map(MaxwellEngineOperation::DmaCopy),
    ))
}

fn build_operation(
    state: &MaxwellDmaCopyState,
    source: MaxwellMethodSource,
) -> Result<MaxwellDmaCopyOperation, MaxwellEngineDispatchError> {
    let raw = source.argument();
    let copy_mode = raw & 0x3;
    if !matches!(copy_mode, 1 | 2) {
        return Err(invalid_encoding(
            source,
            "LAUNCH_DMA requires a copy transfer mode",
        ));
    }
    if raw & 0x000f_f878 != 0 {
        return Err(invalid_encoding(
            source,
            "semaphores, interrupts, physical addressing, L2 bypass, and reductions are not implemented",
        ));
    }
    let multi_line = raw & (1 << 9) != 0;
    let remap_enabled = raw & (1 << 10) != 0;
    let width = required(state, MaxwellDmaCopyRegisterName::LineLength, source)?;
    let height = required(state, MaxwellDmaCopyRegisterName::LineCount, source)?;
    if width == 0 || height == 0 {
        return Err(invalid_encoding(source, "DMA dimensions must be nonzero"));
    }
    if !multi_line && height != 1 {
        return Err(invalid_encoding(
            source,
            "LINE_COUNT must be one when multi-line mode is disabled",
        ));
    }

    let source_address = address(
        state,
        MaxwellDmaCopyRegisterName::SourceAddressUpper,
        MaxwellDmaCopyRegisterName::SourceAddressLower,
        source,
    )?;
    let destination_address = address(
        state,
        MaxwellDmaCopyRegisterName::DestinationAddressUpper,
        MaxwellDmaCopyRegisterName::DestinationAddressLower,
        source,
    )?;
    let remap = remap_enabled
        .then(|| parse_remap(state, source))
        .transpose()?;
    let (source_element_bytes, destination_element_bytes) = remap.map_or((1, 1), |remap| {
        (
            u32::from(remap.component_bytes) * u32::from(remap.source_components),
            u32::from(remap.component_bytes) * u32::from(remap.destination_components),
        )
    });
    let source_layout = layout(
        state,
        raw & (1 << 7) != 0,
        true,
        width,
        height,
        source_element_bytes,
        source,
    )?;
    let destination_layout = layout(
        state,
        raw & (1 << 8) != 0,
        false,
        width,
        height,
        destination_element_bytes,
        source,
    )?;
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

    Ok(MaxwellDmaCopyOperation {
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

fn required(
    state: &MaxwellDmaCopyState,
    register: MaxwellDmaCopyRegisterName,
    source: MaxwellMethodSource,
) -> Result<u32, MaxwellEngineDispatchError> {
    state
        .register(register)
        .raw()
        .ok_or_else(|| invalid_encoding(source, "LAUNCH_DMA requires complete copy state"))
}

fn address(
    state: &MaxwellDmaCopyState,
    upper: MaxwellDmaCopyRegisterName,
    lower: MaxwellDmaCopyRegisterName,
    source: MaxwellMethodSource,
) -> Result<u64, MaxwellEngineDispatchError> {
    Ok((u64::from(required(state, upper, source)?) << 32)
        | u64::from(required(state, lower, source)?))
}

fn parse_remap(
    state: &MaxwellDmaCopyState,
    source: MaxwellMethodSource,
) -> Result<MaxwellDmaCopyRemap, MaxwellEngineDispatchError> {
    let raw = required(state, MaxwellDmaCopyRegisterName::RemapComponents, source)?;
    let mut components = [MaxwellDmaCopyComponentSource::NoWrite; 4];
    for (index, component) in components.iter_mut().enumerate() {
        *component = match (raw >> (index * 4)) & 0x7 {
            value @ 0..=3 => MaxwellDmaCopyComponentSource::Source(value as u8),
            4 => MaxwellDmaCopyComponentSource::ConstantA,
            5 => MaxwellDmaCopyComponentSource::ConstantB,
            6 => MaxwellDmaCopyComponentSource::NoWrite,
            _ => return Err(invalid_encoding(source, "invalid component remap selector")),
        };
    }
    let remap = MaxwellDmaCopyRemap {
        components,
        component_bytes: ((raw >> 16) & 0x3) as u8 + 1,
        source_components: ((raw >> 20) & 0x3) as u8 + 1,
        destination_components: ((raw >> 24) & 0x3) as u8 + 1,
        constant_a: state
            .register(MaxwellDmaCopyRegisterName::RemapConstantA)
            .raw()
            .unwrap_or(0),
        constant_b: state
            .register(MaxwellDmaCopyRegisterName::RemapConstantB)
            .raw()
            .unwrap_or(0),
    };
    if remap.components[..remap.destination_components as usize]
        .iter()
        .any(|component| {
            matches!(component, MaxwellDmaCopyComponentSource::Source(index) if *index >= remap.source_components)
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
) -> Result<MaxwellDmaCopyMemoryLayout, MaxwellEngineDispatchError> {
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
        let pitch = required(state, pitch_register, source)?;
        let row_bytes = width
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid_encoding(source, "DMA row size overflows"))?;
        if pitch < row_bytes {
            return Err(invalid_encoding(
                source,
                "DMA pitch is shorter than one copied row",
            ));
        }
        return Ok(MaxwellDmaCopyMemoryLayout::Pitch { pitch });
    }

    let dimensions = required(state, block, source)?;
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
    let surface_width = required(state, size_x, source)?;
    let surface_height = required(state, size_y, source)?;
    if required(state, size_z, source)? != 1 || required(state, position_z, source)? != 0 {
        return Err(invalid_encoding(
            source,
            "three-dimensional block-linear DMA is not implemented",
        ));
    }
    let position = required(state, position_xy, source)?;
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
    Ok(MaxwellDmaCopyMemoryLayout::BlockLinear {
        surface_width,
        surface_height,
        x,
        y,
        block_height_log2,
    })
}

fn required_range_size(
    layout: MaxwellDmaCopyMemoryLayout,
    width: u32,
    height: u32,
    element_bytes: u32,
) -> Result<u64, MaxwellEngineDispatchError> {
    let offset = layout_offset(layout, width - 1, height - 1, element_bytes)
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
    u64::try_from(offset)
        .ok()
        .and_then(|offset| offset.checked_add(u64::from(element_bytes)))
        .ok_or(MaxwellEngineDispatchError::ResourceExhausted)
}

fn layout_offset(
    layout: MaxwellDmaCopyMemoryLayout,
    x: u32,
    y: u32,
    element_bytes: u32,
) -> Result<usize, MaxwellDmaCopyError> {
    let (x, y) = match layout {
        MaxwellDmaCopyMemoryLayout::Pitch { pitch } => {
            let offset = u64::from(y)
                .checked_mul(u64::from(pitch))
                .and_then(|offset| {
                    u64::from(x)
                        .checked_mul(u64::from(element_bytes))
                        .and_then(|x| offset.checked_add(x))
                })
                .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?;
            return usize::try_from(offset).map_err(|_| MaxwellDmaCopyError::ArithmeticOverflow);
        }
        MaxwellDmaCopyMemoryLayout::BlockLinear {
            x: origin_x,
            y: origin_y,
            ..
        } => (
            origin_x
                .checked_add(x)
                .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?,
            origin_y
                .checked_add(y)
                .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?,
        ),
    };
    let MaxwellDmaCopyMemoryLayout::BlockLinear {
        surface_width,
        block_height_log2,
        ..
    } = layout
    else {
        unreachable!()
    };
    let byte_x = u64::from(x)
        .checked_mul(u64::from(element_bytes))
        .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?;
    let row_bytes = u64::from(surface_width)
        .checked_mul(u64::from(element_bytes))
        .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?;
    let row_pitch = row_bytes
        .checked_add(63)
        .map(|value| value / 64 * 64)
        .ok_or(MaxwellDmaCopyError::ArithmeticOverflow)?;
    let width_in_gobs = row_pitch / 64;
    let block_height_gobs = 1_u64 << block_height_log2;
    let y = u64::from(y);
    let offset = (y / (8 * block_height_gobs)) * 512 * block_height_gobs * width_in_gobs
        + (byte_x / 64) * 512 * block_height_gobs
        + ((y % (8 * block_height_gobs)) / 8) * 512
        + ((byte_x % 64) / 32) * 256
        + ((y % 8) / 2) * 64
        + ((byte_x % 32) / 16) * 32
        + (y % 2) * 16
        + byte_x % 16;
    usize::try_from(offset).map_err(|_| MaxwellDmaCopyError::ArithmeticOverflow)
}

fn remap_element(remap: MaxwellDmaCopyRemap, source: &[u8], destination: &mut [u8]) {
    let component_bytes = remap.component_bytes as usize;
    for destination_component in 0..remap.destination_components as usize {
        let start = destination_component * component_bytes;
        let output = &mut destination[start..start + component_bytes];
        match remap.components[destination_component] {
            MaxwellDmaCopyComponentSource::Source(source_component) => {
                let source_start = source_component as usize * component_bytes;
                output.copy_from_slice(&source[source_start..source_start + component_bytes]);
            }
            MaxwellDmaCopyComponentSource::ConstantA => {
                output.copy_from_slice(&remap.constant_a.to_le_bytes()[..component_bytes]);
            }
            MaxwellDmaCopyComponentSource::ConstantB => {
                output.copy_from_slice(&remap.constant_b.to_le_bytes()[..component_bytes]);
            }
            MaxwellDmaCopyComponentSource::NoWrite => {}
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_rgba_remap_is_identity() {
        let remap = MaxwellDmaCopyRemap {
            components: [
                MaxwellDmaCopyComponentSource::Source(0),
                MaxwellDmaCopyComponentSource::Source(1),
                MaxwellDmaCopyComponentSource::Source(2),
                MaxwellDmaCopyComponentSource::Source(3),
            ],
            component_bytes: 1,
            source_components: 4,
            destination_components: 4,
            constant_a: 0,
            constant_b: 0,
        };
        let mut destination = [0_u8; 4];
        remap_element(remap, &[0x10, 0x20, 0x30, 0x40], &mut destination);
        assert_eq!(destination, [0x10, 0x20, 0x30, 0x40]);
    }

    #[test]
    fn block_linear_offsets_follow_the_tegra_gob_layout() {
        let layout = MaxwellDmaCopyMemoryLayout::BlockLinear {
            surface_width: 64,
            surface_height: 16,
            x: 0,
            y: 0,
            block_height_log2: 1,
        };
        assert_eq!(layout_offset(layout, 0, 0, 4), Ok(0));
        assert_eq!(layout_offset(layout, 4, 0, 4), Ok(32));
        assert_eq!(layout_offset(layout, 0, 1, 4), Ok(16));
        assert_eq!(layout_offset(layout, 0, 2, 4), Ok(64));
        assert_eq!(layout_offset(layout, 16, 0, 4), Ok(1024));
    }
}
