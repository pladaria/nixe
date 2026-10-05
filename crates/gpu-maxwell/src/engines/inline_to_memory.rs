//! GM20B `MAXWELL_INLINE_TO_MEMORY_A` state and pitch/block-linear upload semantics.

use nixe_gpu::{GpuClassId, GpuMethodId};

use super::{
    AppliedMethod, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata, PendingEngineOperation,
};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};

pub(super) const CLASS: GpuClassId = GpuClassId(0xa140);
const CLASS_NAME: &str = "MAXWELL_INLINE_TO_MEMORY_A";

/// One source-preserving register in the inline-to-memory engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryRegister<T> {
    raw: Option<u32>,
    value: Option<T>,
    source: Option<MaxwellMethodSource>,
}

impl<T> Default for MaxwellInlineToMemoryRegister<T> {
    fn default() -> Self {
        Self {
            raw: None,
            value: None,
            source: None,
        }
    }
}

impl<T> MaxwellInlineToMemoryRegister<T> {
    const fn programmed(raw: u32, value: T, source: MaxwellMethodSource) -> Self {
        Self {
            raw: Some(raw),
            value: Some(value),
            source: Some(source),
        }
    }

    #[must_use]
    pub const fn raw(&self) -> Option<u32> {
        self.raw
    }

    #[must_use]
    pub const fn value(&self) -> Option<&T> {
        self.value.as_ref()
    }

    #[must_use]
    pub const fn source(&self) -> Option<MaxwellMethodSource> {
        self.source
    }
}

/// Complete GPU address accepted by the Switch 1 frontend profile.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellInlineToMemoryAddress(u64);

impl MaxwellInlineToMemoryAddress {
    pub(super) const fn new(upper: u32, lower: u32) -> Option<Self> {
        if upper <= 0xff {
            Some(Self((upper as u64) << 32 | lower as u64))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Semaphore payload shape retained by `LAUNCH_DMA`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellInlineToMemorySemaphoreStructureSize {
    FourWords,
    OneWord,
}

/// Validated inline upload launch configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryLaunch {
    system_memory_barrier_disabled: bool,
    semaphore_structure_size: MaxwellInlineToMemorySemaphoreStructureSize,
}

impl MaxwellInlineToMemoryLaunch {
    const fn new(
        system_memory_barrier_disabled: bool,
        semaphore_structure_size: MaxwellInlineToMemorySemaphoreStructureSize,
    ) -> Self {
        Self {
            system_memory_barrier_disabled,
            semaphore_structure_size,
        }
    }

    #[must_use]
    pub const fn system_memory_barrier_disabled(self) -> bool {
        self.system_memory_barrier_disabled
    }

    #[must_use]
    pub const fn semaphore_structure_size(self) -> MaxwellInlineToMemorySemaphoreStructureSize {
        self.semaphore_structure_size
    }
}

/// Cursor for an armed inline-to-memory transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryPendingTransfer {
    address: MaxwellInlineToMemoryAddress,
    byte_length: u32,
    next_offset: u32,
    line_length: u32,
    layout: DestinationLayout,
}

impl MaxwellInlineToMemoryPendingTransfer {
    const fn new(
        address: MaxwellInlineToMemoryAddress,
        byte_length: u32,
        line_length: u32,
        layout: DestinationLayout,
    ) -> Self {
        Self {
            address,
            byte_length,
            next_offset: 0,
            line_length,
            layout,
        }
    }

    #[must_use]
    pub const fn address(self) -> MaxwellInlineToMemoryAddress {
        self.address
    }

    #[must_use]
    pub const fn byte_length(self) -> u32 {
        self.byte_length
    }

    #[must_use]
    pub const fn next_offset(self) -> u32 {
        self.next_offset
    }

    const fn advance(self, next_offset: u32) -> Option<Self> {
        if next_offset == self.byte_length {
            None
        } else {
            Some(Self {
                next_offset,
                ..self
            })
        }
    }
}

// Pitch uploads use OFFSET_OUT + row * PITCH_OUT. Origins and block geometry
// are consumed only by block-linear uploads, whose X coordinate is in bytes.
// https://github.com/eden-emulator/mirror/blob/master/src/video_core/engines/engine_upload.cpp
// Tegra 16Bx2 GOB address mapping (also used by the DMA engine):
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DestinationLayout {
    Pitch {
        pitch: u32,
    },
    BlockLinear {
        width_in_gobs: u64,
        block_height_gobs: u64,
        x: u32,
        y: u32,
    },
}

impl DestinationLayout {
    fn offset(self, x: u32, y: u32) -> u64 {
        match self {
            Self::Pitch { pitch } => u64::from(y) * u64::from(pitch) + u64::from(x),
            Self::BlockLinear {
                width_in_gobs,
                block_height_gobs,
                x: origin_x,
                y: origin_y,
            } => {
                let x = u64::from(x) + u64::from(origin_x);
                let y = u64::from(y) + u64::from(origin_y);
                let block_rows = 8 * block_height_gobs;
                (y / block_rows) * 512 * block_height_gobs * width_in_gobs
                    + (x / 64) * 512 * block_height_gobs
                    + ((y % block_rows) / 8) * 512
                    + ((x % 64) / 32) * 256
                    + ((y % 8) / 2) * 64
                    + ((x % 32) / 16) * 32
                    + (y % 2) * 16
                    + x % 16
            }
        }
    }
}

/// Persistent setup and upload cursor for `MAXWELL_INLINE_TO_MEMORY_A`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryState {
    line_length: MaxwellInlineToMemoryRegister<u32>,
    line_count: MaxwellInlineToMemoryRegister<u32>,
    address_upper: MaxwellInlineToMemoryRegister<u32>,
    address_lower: MaxwellInlineToMemoryRegister<u32>,
    pitch: MaxwellInlineToMemoryRegister<u32>,
    block_size: MaxwellInlineToMemoryRegister<u32>,
    width: MaxwellInlineToMemoryRegister<u32>,
    height: MaxwellInlineToMemoryRegister<u32>,
    depth: MaxwellInlineToMemoryRegister<u32>,
    layer: MaxwellInlineToMemoryRegister<u32>,
    origin_x: MaxwellInlineToMemoryRegister<u32>,
    origin_y: MaxwellInlineToMemoryRegister<u32>,

    launch: MaxwellInlineToMemoryRegister<MaxwellInlineToMemoryLaunch>,
    last_data: MaxwellInlineToMemoryRegister<u32>,
    pending: Option<MaxwellInlineToMemoryPendingTransfer>,
}

impl MaxwellInlineToMemoryState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn line_length(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.line_length
    }

    #[must_use]
    pub const fn line_count(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.line_count
    }

    #[must_use]
    pub const fn address_upper(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.address_upper
    }

    #[must_use]
    pub const fn address_lower(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.address_lower
    }

    #[must_use]
    pub const fn pitch(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.pitch
    }

    #[must_use]
    pub const fn block_size(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.block_size
    }

    #[must_use]
    pub const fn width(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.width
    }

    #[must_use]
    pub const fn height(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.height
    }

    #[must_use]
    pub const fn depth(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.depth
    }

    #[must_use]
    pub const fn layer(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.layer
    }

    #[must_use]
    pub const fn origin_x(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.origin_x
    }

    #[must_use]
    pub const fn origin_y(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.origin_y
    }

    #[must_use]
    pub const fn launch(&self) -> &MaxwellInlineToMemoryRegister<MaxwellInlineToMemoryLaunch> {
        &self.launch
    }

    #[must_use]
    pub const fn last_data(&self) -> &MaxwellInlineToMemoryRegister<u32> {
        &self.last_data
    }

    #[must_use]
    pub const fn pending(&self) -> Option<MaxwellInlineToMemoryPendingTransfer> {
        self.pending
    }

    fn apply(&mut self, write: MaxwellInlineToMemoryStateWrite) {
        match write {
            MaxwellInlineToMemoryStateWrite::LineLength { value, source } => {
                self.line_length = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::LineCount { value, source } => {
                self.line_count = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::AddressUpper { value, source } => {
                self.address_upper =
                    MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::AddressLower { value, source } => {
                self.address_lower =
                    MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::Pitch { value, source } => {
                self.pitch = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::BlockSize { value, source } => {
                self.block_size = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::Width { value, source } => {
                self.width = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::Height { value, source } => {
                self.height = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::Depth { value, source } => {
                self.depth = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::Layer { value, source } => {
                self.layer = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::OriginX { value, source } => {
                self.origin_x = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::OriginY { value, source } => {
                self.origin_y = MaxwellInlineToMemoryRegister::programmed(value, value, source);
            }
            MaxwellInlineToMemoryStateWrite::Launch {
                value,
                pending,
                source,
            } => {
                self.launch =
                    MaxwellInlineToMemoryRegister::programmed(source.argument(), value, source);
                self.pending = Some(pending);
            }
            MaxwellInlineToMemoryStateWrite::Data {
                value,
                next_offset,
                source,
            } => {
                self.last_data = MaxwellInlineToMemoryRegister::programmed(value, value, source);
                self.pending = self
                    .pending
                    .and_then(|pending| pending.advance(next_offset));
            }
        }
    }
}

/// One atomic state transition produced by an inline-to-memory method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellInlineToMemoryStateWrite {
    LineLength {
        value: u32,
        source: MaxwellMethodSource,
    },
    LineCount {
        value: u32,
        source: MaxwellMethodSource,
    },
    AddressUpper {
        value: u32,
        source: MaxwellMethodSource,
    },
    AddressLower {
        value: u32,
        source: MaxwellMethodSource,
    },
    Pitch {
        value: u32,
        source: MaxwellMethodSource,
    },
    BlockSize {
        value: u32,
        source: MaxwellMethodSource,
    },
    Width {
        value: u32,
        source: MaxwellMethodSource,
    },
    Height {
        value: u32,
        source: MaxwellMethodSource,
    },
    Depth {
        value: u32,
        source: MaxwellMethodSource,
    },
    Layer {
        value: u32,
        source: MaxwellMethodSource,
    },
    OriginX {
        value: u32,
        source: MaxwellMethodSource,
    },
    OriginY {
        value: u32,
        source: MaxwellMethodSource,
    },
    Launch {
        value: MaxwellInlineToMemoryLaunch,
        pending: MaxwellInlineToMemoryPendingTransfer,
        source: MaxwellMethodSource,
    },
    Data {
        value: u32,
        next_offset: u32,
        source: MaxwellMethodSource,
    },
}

/// One validated inline word awaiting an ordered GPU-memory write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryUpload {
    address: MaxwellInlineToMemoryAddress,
    offset: u32,
    value: u32,
    source: MaxwellMethodSource,
}

impl MaxwellInlineToMemoryUpload {
    pub(super) const fn new(
        address: MaxwellInlineToMemoryAddress,
        offset: u32,
        value: u32,
        source: MaxwellMethodSource,
    ) -> Self {
        Self {
            address,
            offset,
            value,
            source,
        }
    }

    #[must_use]
    pub const fn address(self) -> MaxwellInlineToMemoryAddress {
        self.address
    }

    #[must_use]
    pub const fn offset(self) -> u32 {
        self.offset
    }

    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }
}

#[derive(Clone, Copy)]
enum MethodAction {
    LineLength,
    LineCount,
    AddressUpper,
    AddressLower,
    Pitch,
    BlockSize,
    Width,
    Height,
    Depth,
    Layer,
    OriginX,
    OriginY,
    Launch,
    Data,
}

#[derive(Clone, Copy)]
struct MethodDeclaration {
    metadata: &'static MaxwellEngineMethodMetadata,
    defined_mask: u32,
    action: MethodAction,
}

macro_rules! methods {
    ($($identifier:ident => ($method:literal, $name:literal, $mask:expr, $action:expr)),+ $(,)?) => {
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
                action: $action,
            }),+
        ];
    };
}

// Method fields are pinned to NVIDIA's public MAXWELL_INLINE_TO_MEMORY_A
// header. The standalone class exposes a 25-bit address-upper field, while
// the Switch 1 address-space profile accepted below remains 40-bit.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/inline-to-memory/cla140.h#L86-L100
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/inline-to-memory/cla140.h#L137-L171
methods!(
    LINE_LENGTH_IN => (0x0180, "LINE_LENGTH_IN", u32::MAX, MethodAction::LineLength),
    LINE_COUNT => (0x0184, "LINE_COUNT", u32::MAX, MethodAction::LineCount),
    OFFSET_OUT_UPPER => (0x0188, "OFFSET_OUT_UPPER", 0x01ff_ffff, MethodAction::AddressUpper),
    OFFSET_OUT => (0x018c, "OFFSET_OUT", u32::MAX, MethodAction::AddressLower),
    PITCH_OUT => (0x0190, "PITCH_OUT", u32::MAX, MethodAction::Pitch),
    SET_DST_BLOCK_SIZE => (0x0194, "SET_DST_BLOCK_SIZE", 0xfff, MethodAction::BlockSize),
    SET_DST_WIDTH => (0x0198, "SET_DST_WIDTH", u32::MAX, MethodAction::Width),
    SET_DST_HEIGHT => (0x019c, "SET_DST_HEIGHT", u32::MAX, MethodAction::Height),
    SET_DST_DEPTH => (0x01a0, "SET_DST_DEPTH", u32::MAX, MethodAction::Depth),
    SET_DST_LAYER => (0x01a4, "SET_DST_LAYER", u32::MAX, MethodAction::Layer),
    SET_DST_ORIGIN_BYTES_X => (0x01a8, "SET_DST_ORIGIN_BYTES_X", 0x001f_ffff, MethodAction::OriginX),
    SET_DST_ORIGIN_SAMPLES_Y => (0x01ac, "SET_DST_ORIGIN_SAMPLES_Y", 0x0001_ffff, MethodAction::OriginY),
    LAUNCH_DMA => (0x01b0, "LAUNCH_DMA", 0x0000_f37f, MethodAction::Launch),
    LOAD_INLINE_DATA => (0x01b4, "LOAD_INLINE_DATA", u32::MAX, MethodAction::Data),
);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellInlineToMemoryState,
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

    let raw = source.argument();
    if matches!(declaration.action, MethodAction::Data) {
        let pending = candidate.pending().ok_or_else(|| {
            invalid_encoding(
                source,
                declaration.metadata.method_name(),
                "inline data requires an armed LAUNCH_DMA transfer",
            )
        })?;
        let next_offset = pending.next_offset().checked_add(4).ok_or_else(|| {
            invalid_encoding(
                source,
                declaration.metadata.method_name(),
                "inline upload cursor overflows",
            )
        })?;
        if next_offset > pending.byte_length() {
            return Err(invalid_encoding(
                source,
                declaration.metadata.method_name(),
                "inline data exceeds the armed transfer length",
            ));
        }
        let write = MaxwellInlineToMemoryStateWrite::Data {
            value: raw,
            next_offset,
            source,
        };
        let upload = MaxwellInlineToMemoryUpload {
            address: pending.address(),
            offset: pending.layout.offset(
                pending.next_offset() % pending.line_length,
                pending.next_offset() / pending.line_length,
            ) as u32,
            value: raw,
            source,
        };
        candidate.apply(write);
        return Ok(AppliedMethod::new(
            method,
            *declaration.metadata,
            Some(PendingEngineOperation::InlineToMemory(upload)),
        ));
    }

    let write = match declaration.action {
        MethodAction::LineLength => {
            MaxwellInlineToMemoryStateWrite::LineLength { value: raw, source }
        }
        MethodAction::LineCount => {
            MaxwellInlineToMemoryStateWrite::LineCount { value: raw, source }
        }
        MethodAction::AddressUpper => {
            MaxwellInlineToMemoryStateWrite::AddressUpper { value: raw, source }
        }
        MethodAction::AddressLower => {
            MaxwellInlineToMemoryStateWrite::AddressLower { value: raw, source }
        }
        MethodAction::Pitch => MaxwellInlineToMemoryStateWrite::Pitch { value: raw, source },
        MethodAction::BlockSize => {
            MaxwellInlineToMemoryStateWrite::BlockSize { value: raw, source }
        }
        MethodAction::Width => MaxwellInlineToMemoryStateWrite::Width { value: raw, source },
        MethodAction::Height => MaxwellInlineToMemoryStateWrite::Height { value: raw, source },
        MethodAction::Depth => MaxwellInlineToMemoryStateWrite::Depth { value: raw, source },
        MethodAction::Layer => MaxwellInlineToMemoryStateWrite::Layer { value: raw, source },
        MethodAction::OriginX => MaxwellInlineToMemoryStateWrite::OriginX { value: raw, source },
        MethodAction::OriginY => MaxwellInlineToMemoryStateWrite::OriginY { value: raw, source },
        MethodAction::Launch => {
            if raw & !0x0000_1051 != 0 {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "interrupt, reduction, and semaphore release inline uploads are not implemented",
                ));
            }
            if candidate.pending().is_some() {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "cannot replace an incomplete inline upload",
                ));
            }
            let upper = *candidate.address_upper().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires OFFSET_OUT_UPPER",
                )
            })?;
            let lower = *candidate.address_lower().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires OFFSET_OUT",
                )
            })?;
            let address = MaxwellInlineToMemoryAddress::new(upper, lower).ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "destination address exceeds the Switch 1 40-bit GPU address space",
                )
            })?;
            let line_length = *candidate.line_length().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires LINE_LENGTH_IN",
                )
            })?;
            let line_count = *candidate.line_count().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires LINE_COUNT",
                )
            })?;
            if line_length == 0 || !line_length.is_multiple_of(4) {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "inline upload length must be nonzero and word-aligned",
                ));
            }
            let byte_length = line_length
                .checked_mul(line_count)
                .filter(|&length| length != 0)
                .ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "LAUNCH_DMA",
                        "inline transfer size is zero or overflows",
                    )
                })?;
            let layout = if raw & 1 != 0 {
                let pitch = if line_count == 1 {
                    0
                } else {
                    *candidate.pitch.value().ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "LAUNCH_DMA",
                            "multi-line launch requires PITCH_OUT",
                        )
                    })?
                };
                if line_count > 1 && pitch < line_length {
                    return Err(invalid_encoding(
                        source,
                        "LAUNCH_DMA",
                        "destination pitch is smaller than the line length",
                    ));
                }
                DestinationLayout::Pitch { pitch }
            } else {
                let required = |register: &MaxwellInlineToMemoryRegister<u32>| {
                    register.value().copied().ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "LAUNCH_DMA",
                            "block-linear launch requires destination geometry and origins",
                        )
                    })
                };
                let block_size = required(&candidate.block_size)?;
                let width = required(&candidate.width)?;
                let height = required(&candidate.height)?;
                let depth = required(&candidate.depth)?;
                let layer = required(&candidate.layer)?;
                let x = required(&candidate.origin_x)?;
                let y = required(&candidate.origin_y)?;
                let height_log2 = (block_size >> 4) & 0xf;
                if block_size & 0xf0f != 0 || height_log2 > 5 || depth != 1 || layer != 0 {
                    return Err(invalid_encoding(
                        source,
                        "LAUNCH_DMA",
                        "only two-dimensional, one-GOB-wide block-linear uploads are implemented",
                    ));
                }
                if !x.is_multiple_of(4)
                    || x.checked_add(line_length).is_none_or(|end| end > width)
                    || y.checked_add(line_count).is_none_or(|end| end > height)
                {
                    return Err(invalid_encoding(
                        source,
                        "LAUNCH_DMA",
                        "block-linear upload is unaligned or exceeds destination dimensions",
                    ));
                }
                DestinationLayout::BlockLinear {
                    width_in_gobs: u64::from(width).div_ceil(64),
                    block_height_gobs: 1 << height_log2,
                    x,
                    y,
                }
            };
            let last_byte = layout.offset(line_length - 1, line_count - 1);
            if last_byte > u64::from(u32::MAX)
                || address
                    .get()
                    .checked_add(last_byte + 1)
                    .is_none_or(|end| end > 1 << 40)
            {
                return Err(invalid_encoding(
                    source,
                    "LAUNCH_DMA",
                    "inline upload GPU range overflows",
                ));
            }
            MaxwellInlineToMemoryStateWrite::Launch {
                value: MaxwellInlineToMemoryLaunch::new(
                    raw & 0x40 != 0,
                    if raw & 0x1000 == 0 {
                        MaxwellInlineToMemorySemaphoreStructureSize::FourWords
                    } else {
                        MaxwellInlineToMemorySemaphoreStructureSize::OneWord
                    },
                ),
                pending: MaxwellInlineToMemoryPendingTransfer::new(
                    address,
                    byte_length,
                    line_length,
                    layout,
                ),
                source,
            }
        }
        MethodAction::Data => unreachable!("LOAD_INLINE_DATA returns before state decoding"),
    };
    candidate.apply(write);
    Ok(AppliedMethod::new(method, *declaration.metadata, None))
}

fn invalid_encoding(
    source: MaxwellMethodSource,
    method_name: &'static str,
    reason: &'static str,
) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidInlineToMemoryMethodEncoding {
        source,
        method_name,
        reason,
    }
}
