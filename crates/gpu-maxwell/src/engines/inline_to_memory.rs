//! GM20B `MAXWELL_INLINE_TO_MEMORY_A` state and pitch-upload semantics.

use nixe_gpu::{GpuClassId, GpuMethodId};

use super::{
    MaxwellEngineDispatchError, MaxwellEngineMethodDispatch, MaxwellEngineMethodEffect,
    MaxwellEngineMethodMetadata,
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

/// Validated single-line pitch launch configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryLaunch {
    system_memory_barrier_disabled: bool,
    semaphore_structure_size: MaxwellInlineToMemorySemaphoreStructureSize,
}

impl MaxwellInlineToMemoryLaunch {
    const fn pitch(
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
}

impl MaxwellInlineToMemoryPendingTransfer {
    const fn new(address: MaxwellInlineToMemoryAddress, byte_length: u32) -> Self {
        Self {
            address,
            byte_length,
            next_offset: 0,
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

/// Persistent setup and upload cursor for `MAXWELL_INLINE_TO_MEMORY_A`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellInlineToMemoryState {
    line_length: MaxwellInlineToMemoryRegister<u32>,
    line_count: MaxwellInlineToMemoryRegister<u32>,
    address_upper: MaxwellInlineToMemoryRegister<u32>,
    address_lower: MaxwellInlineToMemoryRegister<u32>,
    pitch: MaxwellInlineToMemoryRegister<u32>,
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
    LAUNCH_DMA => (0x01b0, "LAUNCH_DMA", 0x0000_f37f, MethodAction::Launch),
    LOAD_INLINE_DATA => (0x01b4, "LOAD_INLINE_DATA", u32::MAX, MethodAction::Data),
);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellInlineToMemoryState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
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
            offset: pending.next_offset(),
            value: raw,
            source,
        };
        candidate.apply(write);
        return Ok(MaxwellEngineMethodDispatch::new(
            method,
            *declaration.metadata,
            MaxwellEngineMethodEffect::InlineToMemoryStateAndUpload {
                state: write,
                upload,
            },
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
        MethodAction::Launch => {
            if raw & !0x0000_1040 != 1 {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "only pitch, no-reduction, no-completion inline uploads are implemented",
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
            if line_count != 1 {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "multi-line pitch uploads are not implemented",
                ));
            }
            if address
                .get()
                .checked_add(u64::from(line_length))
                .is_none_or(|end| end > (1_u64 << 40))
            {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "inline upload GPU range overflows",
                ));
            }
            MaxwellInlineToMemoryStateWrite::Launch {
                value: MaxwellInlineToMemoryLaunch::pitch(
                    raw & 0x40 != 0,
                    if raw & 0x1000 == 0 {
                        MaxwellInlineToMemorySemaphoreStructureSize::FourWords
                    } else {
                        MaxwellInlineToMemorySemaphoreStructureSize::OneWord
                    },
                ),
                pending: MaxwellInlineToMemoryPendingTransfer::new(address, line_length),
                source,
            }
        }
        MethodAction::Data => unreachable!("LOAD_INLINE_DATA returns before state decoding"),
    };
    candidate.apply(write);
    Ok(MaxwellEngineMethodDispatch::new(
        method,
        *declaration.metadata,
        MaxwellEngineMethodEffect::InlineToMemoryState(write),
    ))
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
