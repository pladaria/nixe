//! Embedded `MAXWELL_B` inline-to-memory register family.
//!
//! Method offsets and fields are pinned to NVIDIA's public B197 class header.
//! In particular, `LAUNCH_DMA` bit zero selects block-linear at zero and pitch
//! at one; decoders using the standalone A140 labels for this aperture invert
//! the captured B197 meaning.
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L109-L186>

use nixe_gpu::GpuMethodId;

use super::{MaxwellThreeDRegister, MaxwellThreeDStateWrite};
use crate::MaxwellMethodSource;
use crate::engines::{
    MaxwellEngineDispatchError, MaxwellInlineToMemoryAddress, MaxwellInlineToMemoryUpload,
};

/// Destination layout selected by `LAUNCH_DMA`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDInlineToMemoryLayout {
    BlockLinear,
    Pitch,
}

/// Completion behavior selected for an embedded upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDInlineToMemoryCompletion {
    FlushDisabled,
    FlushOnly,
}

/// Validated launch configuration retained by the embedded engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDInlineToMemoryLaunch {
    layout: MaxwellThreeDInlineToMemoryLayout,
    completion: MaxwellThreeDInlineToMemoryCompletion,
    system_memory_barrier_disabled: bool,
}

impl MaxwellThreeDInlineToMemoryLaunch {
    #[must_use]
    pub const fn layout(self) -> MaxwellThreeDInlineToMemoryLayout {
        self.layout
    }

    #[must_use]
    pub const fn completion(self) -> MaxwellThreeDInlineToMemoryCompletion {
        self.completion
    }

    #[must_use]
    pub const fn system_memory_barrier_disabled(self) -> bool {
        self.system_memory_barrier_disabled
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingTransfer {
    address: MaxwellInlineToMemoryAddress,
    byte_length: u32,
    next_offset: u32,
}

/// Persistent embedded upload setup and stream cursor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDInlineToMemoryState {
    line_length: MaxwellThreeDRegister<u32>,
    line_count: MaxwellThreeDRegister<u32>,
    address_upper: MaxwellThreeDRegister<u8>,
    address_lower: MaxwellThreeDRegister<u32>,
    pitch: MaxwellThreeDRegister<u32>,
    block_size: MaxwellThreeDRegister<u32>,
    width: MaxwellThreeDRegister<u32>,
    height: MaxwellThreeDRegister<u32>,
    depth: MaxwellThreeDRegister<u32>,
    layer: MaxwellThreeDRegister<u32>,
    origin_x: MaxwellThreeDRegister<u32>,
    origin_y: MaxwellThreeDRegister<u32>,
    launch: MaxwellThreeDRegister<MaxwellThreeDInlineToMemoryLaunch>,
    last_data: MaxwellThreeDRegister<u32>,
    pending: Option<PendingTransfer>,
}

impl Default for MaxwellThreeDInlineToMemoryState {
    fn default() -> Self {
        // Public Maxwell implementations independently initialize the class
        // register file to zero. These layout registers therefore have real
        // zero reset values, unlike size/address fields which remain required
        // here before a transfer can be consumed.
        // <https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/engines/maxwell_3d.cpp#L37-L42>
        // <https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Device/DeviceState.cs#L16-L30>
        Self {
            line_length: Default::default(),
            line_count: Default::default(),
            address_upper: Default::default(),
            address_lower: Default::default(),
            pitch: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            block_size: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            width: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            height: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            depth: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            layer: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            origin_x: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            origin_y: MaxwellThreeDRegister::verified_reset(0, Some(0)),
            launch: Default::default(),
            last_data: Default::default(),
            pending: None,
        }
    }
}

impl MaxwellThreeDInlineToMemoryState {
    #[must_use]
    pub const fn line_length(&self) -> &MaxwellThreeDRegister<u32> {
        &self.line_length
    }

    #[must_use]
    pub const fn line_count(&self) -> &MaxwellThreeDRegister<u32> {
        &self.line_count
    }

    #[must_use]
    pub const fn launch(&self) -> &MaxwellThreeDRegister<MaxwellThreeDInlineToMemoryLaunch> {
        &self.launch
    }

    #[must_use]
    pub const fn last_data(&self) -> &MaxwellThreeDRegister<u32> {
        &self.last_data
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDInlineToMemoryStateWrite) {
        match write {
            MaxwellThreeDInlineToMemoryStateWrite::LineLength { value, source } => {
                self.line_length = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::LineCount { value, source } => {
                self.line_count = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::AddressUpper { value, source } => {
                self.address_upper =
                    MaxwellThreeDRegister::programmed(u32::from(value), value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::AddressLower { value, source } => {
                self.address_lower = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::Pitch { value, source } => {
                self.pitch = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::BlockSize { value, source } => {
                self.block_size = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::Width { value, source } => {
                self.width = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::Height { value, source } => {
                self.height = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::Depth { value, source } => {
                self.depth = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::Layer { value, source } => {
                self.layer = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::OriginX { value, source } => {
                self.origin_x = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::OriginY { value, source } => {
                self.origin_y = MaxwellThreeDRegister::programmed(value, value, source);
            }
            MaxwellThreeDInlineToMemoryStateWrite::Launch {
                value,
                address,
                byte_length,
                source,
            } => {
                self.launch = MaxwellThreeDRegister::programmed(source.argument(), value, source);
                self.pending = Some(PendingTransfer {
                    address,
                    byte_length,
                    next_offset: 0,
                });
            }
            MaxwellThreeDInlineToMemoryStateWrite::Data {
                value,
                next_offset,
                source,
            } => {
                self.last_data = MaxwellThreeDRegister::programmed(value, value, source);
                self.pending = self.pending.and_then(|pending| {
                    (next_offset != pending.byte_length).then_some(PendingTransfer {
                        next_offset,
                        ..pending
                    })
                });
            }
        }
    }
}

/// One checked transition in the embedded upload register family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDInlineToMemoryStateWrite {
    LineLength {
        value: u32,
        source: MaxwellMethodSource,
    },
    LineCount {
        value: u32,
        source: MaxwellMethodSource,
    },
    AddressUpper {
        value: u8,
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
        value: MaxwellThreeDInlineToMemoryLaunch,
        address: MaxwellInlineToMemoryAddress,
        byte_length: u32,
        source: MaxwellMethodSource,
    },
    Data {
        value: u32,
        next_offset: u32,
        source: MaxwellMethodSource,
    },
}

const LINE_LENGTH_IN: u32 = 0x0180;
const LINE_COUNT: u32 = 0x0184;
const OFFSET_OUT_UPPER: u32 = 0x0188;
const OFFSET_OUT: u32 = 0x018c;
const PITCH_OUT: u32 = 0x0190;
const SET_DST_BLOCK_SIZE: u32 = 0x0194;
const SET_DST_WIDTH: u32 = 0x0198;
const SET_DST_HEIGHT: u32 = 0x019c;
const SET_DST_DEPTH: u32 = 0x01a0;
const SET_DST_LAYER: u32 = 0x01a4;
const SET_DST_ORIGIN_BYTES_X: u32 = 0x01a8;
const SET_DST_ORIGIN_SAMPLES_Y: u32 = 0x01ac;
const LAUNCH_DMA: u32 = 0x01b0;
const LOAD_INLINE_DATA: u32 = 0x01b4;

pub(super) fn preflight(
    source: MaxwellMethodSource,
    state: &MaxwellThreeDInlineToMemoryState,
) -> Result<
    Option<(
        MaxwellThreeDStateWrite,
        &'static str,
        Option<MaxwellInlineToMemoryUpload>,
    )>,
    MaxwellEngineDispatchError,
> {
    let raw = source.argument();
    let write = match source.method().0 {
        LINE_LENGTH_IN => MaxwellThreeDInlineToMemoryStateWrite::LineLength { value: raw, source },
        LINE_COUNT => MaxwellThreeDInlineToMemoryStateWrite::LineCount { value: raw, source },
        OFFSET_OUT_UPPER if raw <= 0xff => MaxwellThreeDInlineToMemoryStateWrite::AddressUpper {
            value: raw as u8,
            source,
        },
        OFFSET_OUT_UPPER => {
            return Err(invalid(
                source,
                "OFFSET_OUT_UPPER",
                "address upper exceeds its eight-bit field",
            ));
        }
        OFFSET_OUT => MaxwellThreeDInlineToMemoryStateWrite::AddressLower { value: raw, source },
        PITCH_OUT => MaxwellThreeDInlineToMemoryStateWrite::Pitch { value: raw, source },
        SET_DST_BLOCK_SIZE if raw & !0x0000_0fff == 0 => {
            MaxwellThreeDInlineToMemoryStateWrite::BlockSize { value: raw, source }
        }
        SET_DST_BLOCK_SIZE => {
            return Err(invalid(
                source,
                "SET_DST_BLOCK_SIZE",
                "block-size fields set undefined bits",
            ));
        }
        SET_DST_WIDTH => MaxwellThreeDInlineToMemoryStateWrite::Width { value: raw, source },
        SET_DST_HEIGHT => MaxwellThreeDInlineToMemoryStateWrite::Height { value: raw, source },
        SET_DST_DEPTH => MaxwellThreeDInlineToMemoryStateWrite::Depth { value: raw, source },
        SET_DST_LAYER => MaxwellThreeDInlineToMemoryStateWrite::Layer { value: raw, source },
        SET_DST_ORIGIN_BYTES_X if raw <= 0x000f_ffff => {
            MaxwellThreeDInlineToMemoryStateWrite::OriginX { value: raw, source }
        }
        SET_DST_ORIGIN_BYTES_X => {
            return Err(invalid(
                source,
                "SET_DST_ORIGIN_BYTES_X",
                "origin exceeds its twenty-bit field",
            ));
        }
        SET_DST_ORIGIN_SAMPLES_Y if raw <= 0xffff => {
            MaxwellThreeDInlineToMemoryStateWrite::OriginY { value: raw, source }
        }
        SET_DST_ORIGIN_SAMPLES_Y => {
            return Err(invalid(
                source,
                "SET_DST_ORIGIN_SAMPLES_Y",
                "origin exceeds its sixteen-bit field",
            ));
        }
        LAUNCH_DMA => return preflight_launch(source, state),
        LOAD_INLINE_DATA => return preflight_data(source, state),
        _ => return Ok(None),
    };
    let name = method_name(source.method()).expect("recognized embedded inline method has a name");
    Ok(Some((
        MaxwellThreeDStateWrite::InlineToMemory(write),
        name,
        None,
    )))
}

fn preflight_launch(
    source: MaxwellMethodSource,
    state: &MaxwellThreeDInlineToMemoryState,
) -> Result<
    Option<(
        MaxwellThreeDStateWrite,
        &'static str,
        Option<MaxwellInlineToMemoryUpload>,
    )>,
    MaxwellEngineDispatchError,
> {
    let raw = source.argument();
    if raw & !0x51 != 0 {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "completion, interrupt, reduction, and semaphore modes are not implemented",
        ));
    }
    if state.pending.is_some() {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "cannot replace an incomplete inline upload",
        ));
    }
    let line_length = *state
        .line_length
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires LINE_LENGTH_IN"))?;
    let line_count = *state
        .line_count
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires LINE_COUNT"))?;
    if line_length == 0 || !line_length.is_multiple_of(4) {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "line length must be nonzero and word-aligned",
        ));
    }
    if line_count != 1 {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "multi-line embedded uploads are not implemented",
        ));
    }
    let upper = *state
        .address_upper
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires OFFSET_OUT_UPPER"))?;
    let lower = *state
        .address_lower
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires OFFSET_OUT"))?;
    let address = MaxwellInlineToMemoryAddress::new(u32::from(upper), lower).ok_or_else(|| {
        invalid(
            source,
            "LAUNCH_DMA",
            "destination exceeds the Switch 1 GPU address space",
        )
    })?;
    if address
        .get()
        .checked_add(u64::from(line_length))
        .is_none_or(|end| end > (1_u64 << 40))
    {
        return Err(invalid(source, "LAUNCH_DMA", "destination range overflows"));
    }
    let layout = if raw & 1 == 0 {
        if line_length > 64
            || state.block_size.value() != Some(&0)
            || state.layer.value() != Some(&0)
            || state.origin_x.value() != Some(&0)
            || state.origin_y.value() != Some(&0)
        {
            return Err(invalid(
                source,
                "LAUNCH_DMA",
                "only one contiguous row of the first block-linear GOB is implemented",
            ));
        }
        MaxwellThreeDInlineToMemoryLayout::BlockLinear
    } else {
        MaxwellThreeDInlineToMemoryLayout::Pitch
    };
    let write = MaxwellThreeDInlineToMemoryStateWrite::Launch {
        value: MaxwellThreeDInlineToMemoryLaunch {
            layout,
            completion: if raw & 0x10 == 0 {
                MaxwellThreeDInlineToMemoryCompletion::FlushDisabled
            } else {
                MaxwellThreeDInlineToMemoryCompletion::FlushOnly
            },
            system_memory_barrier_disabled: raw & 0x40 != 0,
        },
        address,
        byte_length: line_length,
        source,
    };
    Ok(Some((
        MaxwellThreeDStateWrite::InlineToMemory(write),
        "LAUNCH_DMA",
        None,
    )))
}

fn preflight_data(
    source: MaxwellMethodSource,
    state: &MaxwellThreeDInlineToMemoryState,
) -> Result<
    Option<(
        MaxwellThreeDStateWrite,
        &'static str,
        Option<MaxwellInlineToMemoryUpload>,
    )>,
    MaxwellEngineDispatchError,
> {
    let pending = state.pending.ok_or_else(|| {
        invalid(
            source,
            "LOAD_INLINE_DATA",
            "inline data requires an armed LAUNCH_DMA transfer",
        )
    })?;
    let next_offset = pending
        .next_offset
        .checked_add(4)
        .ok_or_else(|| invalid(source, "LOAD_INLINE_DATA", "inline upload cursor overflows"))?;
    if next_offset > pending.byte_length {
        return Err(invalid(
            source,
            "LOAD_INLINE_DATA",
            "inline data exceeds the armed transfer length",
        ));
    }
    let write = MaxwellThreeDInlineToMemoryStateWrite::Data {
        value: source.argument(),
        next_offset,
        source,
    };
    let upload = MaxwellInlineToMemoryUpload::new(
        pending.address,
        pending.next_offset,
        source.argument(),
        source,
    );
    Ok(Some((
        MaxwellThreeDStateWrite::InlineToMemory(write),
        "LOAD_INLINE_DATA",
        Some(upload),
    )))
}

fn method_name(method: GpuMethodId) -> Option<&'static str> {
    Some(match method.0 {
        LINE_LENGTH_IN => "LINE_LENGTH_IN",
        LINE_COUNT => "LINE_COUNT",
        OFFSET_OUT_UPPER => "OFFSET_OUT_UPPER",
        OFFSET_OUT => "OFFSET_OUT",
        PITCH_OUT => "PITCH_OUT",
        SET_DST_BLOCK_SIZE => "SET_DST_BLOCK_SIZE",
        SET_DST_WIDTH => "SET_DST_WIDTH",
        SET_DST_HEIGHT => "SET_DST_HEIGHT",
        SET_DST_DEPTH => "SET_DST_DEPTH",
        SET_DST_LAYER => "SET_DST_LAYER",
        SET_DST_ORIGIN_BYTES_X => "SET_DST_ORIGIN_BYTES_X",
        SET_DST_ORIGIN_SAMPLES_Y => "SET_DST_ORIGIN_SAMPLES_Y",
        LAUNCH_DMA => "LAUNCH_DMA",
        LOAD_INLINE_DATA => "LOAD_INLINE_DATA",
        _ => return None,
    })
}

fn invalid(
    source: MaxwellMethodSource,
    method_name: &'static str,
    reason: &'static str,
) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidMethodEncoding {
        source,
        method_name,
        reason,
    }
}
