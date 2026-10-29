//! Embedded `MAXWELL_B` inline-to-memory register family.
//!
//! Method offsets and fields are pinned to NVIDIA's public B197 class header.
//! In particular, `LAUNCH_DMA` bit zero selects block-linear at zero and pitch
//! at one; decoders using the standalone A140 labels for this aperture invert
//! the captured B197 meaning.
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L109-L186>

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
pub(super) struct PendingTransfer {
    pub(super) address: MaxwellInlineToMemoryAddress,
    pub(super) byte_length: u32,
    pub(super) next_offset: u32,
}

/// Persistent embedded upload setup and stream cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDInlineToMemoryState<'a> {
    pub(super) registers: &'a super::state::MaxwellThreeDRawRegisterFile,
    pub(super) pending: &'a Option<PendingTransfer>,
}
impl MaxwellThreeDInlineToMemoryState<'_> {
    #[must_use]
    pub fn line_length(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(LINE_LENGTH_IN, Some)
    }
    #[must_use]
    pub fn line_count(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(LINE_COUNT, Some)
    }
    #[must_use]
    pub fn address_upper(&self) -> MaxwellThreeDRegister<u8> {
        self.registers
            .read(OFFSET_OUT_UPPER, |raw| u8::try_from(raw).ok())
    }
    #[must_use]
    pub fn address_lower(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(OFFSET_OUT, Some)
    }
    #[must_use]
    pub fn pitch(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(PITCH_OUT, Some)
    }
    #[must_use]
    pub fn block_size(&self) -> MaxwellThreeDRegister<u32> {
        self.registers
            .read(SET_DST_BLOCK_SIZE, |raw| (raw & !0xfff == 0).then_some(raw))
    }
    #[must_use]
    pub fn width(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(SET_DST_WIDTH, Some)
    }
    #[must_use]
    pub fn height(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(SET_DST_HEIGHT, Some)
    }
    #[must_use]
    pub fn depth(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(SET_DST_DEPTH, Some)
    }
    #[must_use]
    pub fn layer(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(SET_DST_LAYER, Some)
    }
    #[must_use]
    pub fn origin_x(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(SET_DST_ORIGIN_BYTES_X, |raw| {
            (raw <= 0xfffff).then_some(raw)
        })
    }
    #[must_use]
    pub fn origin_y(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(SET_DST_ORIGIN_SAMPLES_Y, |raw| {
            (raw <= 0xffff).then_some(raw)
        })
    }
    #[must_use]
    pub fn last_data(&self) -> MaxwellThreeDRegister<u32> {
        self.registers.read(LOAD_INLINE_DATA, Some)
    }
    #[must_use]
    pub fn launch(&self) -> MaxwellThreeDRegister<MaxwellThreeDInlineToMemoryLaunch> {
        self.registers.read(LAUNCH_DMA, |raw| {
            (raw & !0x1051 == 0).then_some(MaxwellThreeDInlineToMemoryLaunch {
                layout: if raw & 1 == 0 {
                    MaxwellThreeDInlineToMemoryLayout::BlockLinear
                } else {
                    MaxwellThreeDInlineToMemoryLayout::Pitch
                },
                completion: if raw & 0x10 == 0 {
                    MaxwellThreeDInlineToMemoryCompletion::FlushDisabled
                } else {
                    MaxwellThreeDInlineToMemoryCompletion::FlushOnly
                },
                system_memory_barrier_disabled: raw & 0x40 != 0,
            })
        })
    }
    pub(super) fn pending_words(&self) -> usize {
        self.pending
            .map_or(0, |p| (p.byte_length - p.next_offset).div_ceil(4) as usize)
    }
}

/// One checked transition in the embedded upload register family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDInlineToMemoryStateWrite {
    Launch {
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

type PreflightResult = Result<
    Option<(
        MaxwellThreeDStateWrite,
        &'static str,
        Option<MaxwellInlineToMemoryUpload>,
    )>,
    MaxwellEngineDispatchError,
>;

pub(super) fn preflight(
    source: MaxwellMethodSource,
    state: &MaxwellThreeDInlineToMemoryState,
) -> PreflightResult {
    match source.method().0 {
        LAUNCH_DMA => preflight_launch(source, state),
        LOAD_INLINE_DATA => preflight_data(source, state),
        _ => Ok(None),
    }
}

fn preflight_launch(
    source: MaxwellMethodSource,
    state: &MaxwellThreeDInlineToMemoryState,
) -> PreflightResult {
    let raw = source.argument();
    // SEMAPHORE_STRUCT_SIZE (bit 12) only selects the payload format for
    // RELEASE_SEMAPHORE. Both sizes are inert for FLUSH_DISABLE/FLUSH_ONLY;
    // accepting the selector must not enable semaphore release (bit 5).
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
    if raw & !0x1051 != 0 {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "interrupt, reduction, and semaphore release modes are not implemented, or undefined launch bits are set",
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
        .line_length()
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires LINE_LENGTH_IN"))?;
    let line_count = *state
        .line_count()
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires LINE_COUNT"))?;
    if line_length != 0 && line_count > 1 {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "multi-line embedded uploads are not implemented",
        ));
    }
    let byte_length = line_length * line_count;
    let upper = *state
        .address_upper()
        .value()
        .ok_or_else(|| invalid(source, "LAUNCH_DMA", "launch requires OFFSET_OUT_UPPER"))?;
    let lower = *state
        .address_lower()
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
        .checked_add(u64::from(byte_length))
        .is_none_or(|end| end > (1_u64 << 40))
    {
        return Err(invalid(source, "LAUNCH_DMA", "destination range overflows"));
    }
    if raw & 1 == 0
        && byte_length != 0
        && (line_length > 64
            || state.block_size().value() != Some(&0)
            || state.layer().value() != Some(&0)
            || state.origin_x().value() != Some(&0)
            || state.origin_y().value() != Some(&0))
    {
        return Err(invalid(
            source,
            "LAUNCH_DMA",
            "only one contiguous row of the first block-linear GOB is implemented",
        ));
    }
    let write = MaxwellThreeDInlineToMemoryStateWrite::Launch {
        address,
        byte_length,
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
) -> PreflightResult {
    let pending = (*state.pending).ok_or_else(|| {
        invalid(
            source,
            "LOAD_INLINE_DATA",
            "inline data requires an armed LAUNCH_DMA transfer",
        )
    })?;
    let next_offset = pending.next_offset + (pending.byte_length - pending.next_offset).min(4);
    let write = MaxwellThreeDInlineToMemoryStateWrite::Data {
        value: source.argument(),
        next_offset,
        source,
    };
    let upload = MaxwellInlineToMemoryUpload::new(
        pending.address,
        pending.next_offset,
        pending.byte_length - pending.next_offset,
        source.argument(),
        source,
    );
    Ok(Some((
        MaxwellThreeDStateWrite::InlineToMemory(write),
        "LOAD_INLINE_DATA",
        Some(upload),
    )))
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
