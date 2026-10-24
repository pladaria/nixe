//! Switch 1 Maxwell GPFIFO descriptor and submission semantics.
//!
//! Horizon owns the ioctl structure and buffer transport. This module accepts
//! already decoded scalar fields plus the complete descriptor byte array and
//! validates it without retaining memory or mutating a channel.

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress, MappingGeneration};

use crate::{
    MaxwellAddressSpaceId, MaxwellChannelId, MaxwellGpuAccessError, MaxwellGpuMemory,
    MaxwellGpuProfile, MaxwellMappingId,
};

/// Size of one hardware-format GPFIFO descriptor.
pub const MAXWELL_GPFIFO_ENTRY_SIZE: usize = 8;

const ENTRY0_FETCH_CONDITIONAL: u32 = 1 << 0;
const ENTRY0_RESERVED: u32 = 1 << 1;
const ENTRY1_ADDRESS_HIGH_MASK: u32 = 0xff;
const ENTRY1_ALLOW_FLUSH: u32 = 1 << 8;
const ENTRY1_LEVEL_SUBROUTINE: u32 = 1 << 9;
const ENTRY1_WORD_COUNT_SHIFT: u32 = 10;
const ENTRY1_WORD_COUNT_MASK: u32 = 0x1f_ffff;
const ENTRY1_SYNC_WAIT: u32 = 1 << 31;

const SUBMIT_FENCE_WAIT: u32 = 1 << 0;
const SUBMIT_FENCE_GET: u32 = 1 << 1;
const SUBMIT_HARDWARE_FORMAT: u32 = 1 << 2;
const SUBMIT_SYNC_FENCE: u32 = 1 << 3;
const SUBMIT_SUPPRESS_WFI: u32 = 1 << 4;
const SUBMIT_SKIP_BUFFER_REFCOUNTING: u32 = 1 << 5;
// libnx uses this Switch-specific flag when the submitted command stream
// already contains `fence_value` increments of the channel syncpoint. The
// driver must retain that many completion increments, not append another one:
// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/gpu_channel.c#L73-L105
// The pinned Nouveau adapter shows the increment commands being appended
// before `nvGpuChannelKickoff` updates that accumulated value:
// https://github.com/devkitPro/libdrm_nouveau/blob/v1.0.1/source/pushbuf.c#L224-L242
const SUBMIT_FENCE_INCREMENT_VALUE: u32 = 1 << 8;
const KNOWN_SUBMIT_FLAGS: u32 = SUBMIT_FENCE_WAIT
    | SUBMIT_FENCE_GET
    | SUBMIT_HARDWARE_FORMAT
    | SUBMIT_SYNC_FENCE
    | SUBMIT_SUPPRESS_WFI
    | SUBMIT_SKIP_BUFFER_REFCOUNTING
    | SUBMIT_FENCE_INCREMENT_VALUE;

/// Scalar fields decoded from `NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO2`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellGpfifoSubmitRequest {
    pub entry_count: u32,
    pub flags: u32,
    pub fence_id: u32,
    pub fence_value: u32,
}

/// Whether a descriptor is fetched regardless of subdevice state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellGpfifoFetchMode {
    Unconditional,
    Conditional,
}

/// Whether a pushbuffer segment participates in top-level progress reporting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellGpfifoLevel {
    Main,
    Subroutine,
}

/// Whether fetching waits for Host processing of the preceding segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellGpfifoSyncMode {
    Proceed,
    Wait,
}

/// One fully decoded hardware-format GPFIFO descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellGpfifoEntry {
    address: GpuVirtualAddress,
    word_count: u32,
    allow_flush: bool,
    level: MaxwellGpfifoLevel,
    sync: MaxwellGpfifoSyncMode,
    fetch: MaxwellGpfifoFetchMode,
}

impl MaxwellGpfifoEntry {
    #[must_use]
    pub const fn address(self) -> GpuVirtualAddress {
        self.address
    }

    #[must_use]
    pub const fn word_count(self) -> u32 {
        self.word_count
    }

    #[must_use]
    pub const fn byte_count(self) -> u64 {
        self.word_count as u64 * 4
    }

    #[must_use]
    pub const fn allow_flush(self) -> bool {
        self.allow_flush
    }

    #[must_use]
    pub const fn level(self) -> MaxwellGpfifoLevel {
        self.level
    }

    #[must_use]
    pub const fn sync(self) -> MaxwellGpfifoSyncMode {
        self.sync
    }

    #[must_use]
    pub const fn fetch(self) -> MaxwellGpfifoFetchMode {
        self.fetch
    }
}

/// Typed submission behavior selected by the Switch channel flags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellGpfifoSubmissionMode {
    fence_wait: bool,
    fence_get: bool,
    suppress_wfi: bool,
    skip_buffer_refcounting: bool,
    fence_increment_value: bool,
}

impl MaxwellGpfifoSubmissionMode {
    #[must_use]
    pub const fn fence_wait(self) -> bool {
        self.fence_wait
    }

    #[must_use]
    pub const fn fence_get(self) -> bool {
        self.fence_get
    }

    #[must_use]
    pub const fn suppress_wfi(self) -> bool {
        self.suppress_wfi
    }

    #[must_use]
    pub const fn skip_buffer_refcounting(self) -> bool {
        self.skip_buffer_refcounting
    }

    #[must_use]
    pub const fn fence_increment_value(self) -> bool {
        self.fence_increment_value
    }
}

/// A complete decoded submission which has not resolved or retained memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellDecodedGpfifoSubmission {
    mode: MaxwellGpfifoSubmissionMode,
    fence_id: u32,
    fence_value: u32,
    entries: Box<[MaxwellGpfifoEntry]>,
}

impl MaxwellDecodedGpfifoSubmission {
    #[must_use]
    pub const fn mode(&self) -> MaxwellGpfifoSubmissionMode {
        self.mode
    }

    #[must_use]
    pub const fn fence_id(&self) -> u32 {
        self.fence_id
    }

    #[must_use]
    pub const fn fence_value(&self) -> u32 {
        self.fence_value
    }

    #[must_use]
    pub fn entries(&self) -> &[MaxwellGpfifoEntry] {
        &self.entries
    }
}

/// Validated submission metadata. Command memory is fetched when its entry runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellValidatedGpfifoSubmission {
    channel: MaxwellChannelId,
    frontend: FrontendSubmissionId,
    address_space: MaxwellAddressSpaceId,
    decoded: MaxwellDecodedGpfifoSubmission,
}

impl MaxwellValidatedGpfifoSubmission {
    #[must_use]
    pub const fn channel(&self) -> MaxwellChannelId {
        self.channel
    }

    #[must_use]
    pub const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }

    #[must_use]
    pub const fn address_space(&self) -> MaxwellAddressSpaceId {
        self.address_space
    }

    #[must_use]
    pub const fn decoded(&self) -> &MaxwellDecodedGpfifoSubmission {
        &self.decoded
    }
}

/// Exact pointer-free source location used by packet and retention errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellGpfifoSourceLocation {
    pub channel: MaxwellChannelId,
    pub frontend: FrontendSubmissionId,
    pub entry_index: u32,
    pub pushbuffer: GpuVirtualAddress,
    pub word_offset: u64,
    pub mapping: MaxwellMappingId,
    pub generation: MappingGeneration,
}

impl Display for MaxwellGpfifoSourceLocation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {} gpfifo-entry={} pushbuffer={} word-offset={} {} {}",
            self.channel,
            self.frontend,
            self.entry_index,
            self.pushbuffer,
            self.word_offset,
            self.mapping,
            self.generation
        )
    }
}

/// Failure while resolving or reading the current command source.
#[derive(Clone, Debug)]
pub enum MaxwellGpfifoSourceError {
    ChannelAddressSpaceUnbound {
        channel: MaxwellChannelId,
    },
    WrongAddressSpace {
        expected: MaxwellAddressSpaceId,
        actual: MaxwellAddressSpaceId,
    },
    UnknownEntry {
        entry_index: u32,
    },
    UnknownWord {
        entry_index: u32,
        word_offset: u64,
        word_count: u32,
    },
    Resolution {
        channel: MaxwellChannelId,
        frontend: FrontendSubmissionId,
        entry_index: u32,
        pushbuffer: GpuVirtualAddress,
        address_space_generation: MappingGeneration,
        error: MaxwellGpuAccessError,
    },
    Access {
        location: MaxwellGpfifoSourceLocation,
        error: MaxwellGpuAccessError,
    },
    ResourceExhausted,
    ArithmeticOverflow,
}

impl Display for MaxwellGpfifoSourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChannelAddressSpaceUnbound { channel } => write!(
                formatter,
                "GPFIFO channel has no bound GPU address space: {channel}"
            ),
            Self::WrongAddressSpace { expected, actual } => write!(
                formatter,
                "GPFIFO source belongs to {expected}, not {actual}"
            ),
            Self::UnknownEntry { entry_index } => write!(
                formatter,
                "unknown retained GPFIFO entry: entry={entry_index}"
            ),
            Self::UnknownWord {
                entry_index,
                word_offset,
                word_count,
            } => write!(
                formatter,
                "unknown retained GPFIFO word: entry={entry_index} word-offset={word_offset} word-count={word_count}"
            ),
            Self::Resolution {
                channel,
                frontend,
                entry_index,
                pushbuffer,
                address_space_generation,
                error,
            } => write!(
                formatter,
                "GPFIFO source resolution failed: {channel} {frontend} gpfifo-entry={entry_index} pushbuffer={pushbuffer} word-offset=0 {address_space_generation} detail=[{error}]"
            ),
            Self::Access { location, error } => write!(
                formatter,
                "GPFIFO source access failed: {location} detail=[{error}]"
            ),
            Self::ResourceExhausted => {
                formatter.write_str("GPFIFO source retention exhausted host resources")
            }
            Self::ArithmeticOverflow => {
                formatter.write_str("GPFIFO source location arithmetic overflow")
            }
        }
    }
}

impl std::error::Error for MaxwellGpfifoSourceError {}

/// Verify the channel binding and retain decoded entry addresses for ordered fetch.
pub fn validate_gpfifo_submission(
    channel: &crate::MaxwellChannelConfiguration,
    frontend: FrontendSubmissionId,
    decoded: MaxwellDecodedGpfifoSubmission,
    address_space: &MaxwellGpuMemory,
) -> Result<MaxwellValidatedGpfifoSubmission, MaxwellGpfifoSourceError> {
    let channel_id = channel.id();
    let expected_address_space =
        channel
            .address_space()
            .ok_or(MaxwellGpfifoSourceError::ChannelAddressSpaceUnbound {
                channel: channel_id,
            })?;
    if expected_address_space != address_space.id() {
        return Err(MaxwellGpfifoSourceError::WrongAddressSpace {
            expected: expected_address_space,
            actual: address_space.id(),
        });
    }
    Ok(MaxwellValidatedGpfifoSubmission {
        channel: channel_id,
        frontend,
        address_space: address_space.id(),
        decoded,
    })
}

/// Verified invalid guest input which the Switch driver rejects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellInvalidGpfifoSubmission {
    EntryCountExceedsAllocation {
        requested: u32,
        allocated: u32,
    },
    EntryByteCountOverflow {
        entries: u32,
    },
    EntryByteCountMismatch {
        expected: usize,
        actual: usize,
    },
    ReservedEntryBit {
        entry: u32,
        word: u8,
        mask: u32,
    },
    AddressOutOfRange {
        entry: u32,
        address: u64,
        bits: u8,
    },
    PushbufferRangeOutOfRange {
        entry: u32,
        address: u64,
        words: u32,
        bits: u8,
    },
}

impl Display for MaxwellInvalidGpfifoSubmission {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntryCountExceedsAllocation {
                requested,
                allocated,
            } => write!(
                formatter,
                "GPFIFO entry count exceeds channel allocation: requested={requested} allocated={allocated}"
            ),
            Self::EntryByteCountOverflow { entries } => {
                write!(
                    formatter,
                    "GPFIFO descriptor byte count overflows: entries={entries}"
                )
            }
            Self::EntryByteCountMismatch { expected, actual } => write!(
                formatter,
                "GPFIFO descriptor byte count does not match entry count: expected={expected} actual={actual}"
            ),
            Self::ReservedEntryBit { entry, word, mask } => write!(
                formatter,
                "GPFIFO descriptor sets a reserved bit: entry={entry} word={word} mask={mask:#010x}"
            ),
            Self::AddressOutOfRange {
                entry,
                address,
                bits,
            } => write!(
                formatter,
                "GPFIFO address exceeds profile width: entry={entry} address=0x{address:016x} bits={bits}"
            ),
            Self::PushbufferRangeOutOfRange {
                entry,
                address,
                words,
                bits,
            } => write!(
                formatter,
                "GPFIFO pushbuffer range exceeds profile width: entry={entry} address=0x{address:016x} words={words} bits={bits}"
            ),
        }
    }
}

/// Known submission semantics which this frontend stage cannot yet execute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellUnsupportedGpfifoSubmission {
    UnknownSubmitFlags { flags: u32 },
    NonHardwareDescriptorFormat,
    SyncFenceFileDescriptor,
    ConflictingFenceCompletionModes,
    ZeroFenceIncrementValue,
    ConditionalFetch { entry: u32 },
    ControlEntry { entry: u32, opcode: u8 },
}

impl Display for MaxwellUnsupportedGpfifoSubmission {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownSubmitFlags { flags } => {
                write!(
                    formatter,
                    "unknown GPFIFO submission flags: flags={flags:#010x}"
                )
            }
            Self::NonHardwareDescriptorFormat => {
                formatter.write_str("non-hardware GPFIFO descriptor format is not implemented")
            }
            Self::SyncFenceFileDescriptor => formatter
                .write_str("GPFIFO sync-fence file-descriptor semantics are not implemented"),
            Self::ConflictingFenceCompletionModes => formatter.write_str(
                "fence-get combined with a command-stream increment count is not implemented",
            ),
            Self::ZeroFenceIncrementValue => formatter
                .write_str("command-stream fence increment mode has a zero increment count"),
            Self::ConditionalFetch { entry } => write!(
                formatter,
                "conditional GPFIFO fetch is not implemented: entry={entry}"
            ),
            Self::ControlEntry { entry, opcode } => write!(
                formatter,
                "GPFIFO control entry is not implemented: entry={entry} opcode={opcode:#04x}"
            ),
        }
    }
}

/// Result of complete, side-effect-free GPFIFO request validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellGpfifoDecodeError {
    Invalid(MaxwellInvalidGpfifoSubmission),
    Unsupported(MaxwellUnsupportedGpfifoSubmission),
}

impl Display for MaxwellGpfifoDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(error) => write!(formatter, "invalid GPFIFO submission: {error}"),
            Self::Unsupported(error) => write!(formatter, "unsupported GPFIFO submission: {error}"),
        }
    }
}

impl std::error::Error for MaxwellGpfifoDecodeError {}

/// Decodes and validates one complete hardware-format GPFIFO request.
///
/// The field layout is pinned to Switchbrew's Switch ABI table and libnx's
/// exact `KickoffPb` path:
/// https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO
/// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/gpu_channel.c
///
/// Address, length, level, and synchronization meanings are independently
/// documented by NVIDIA's public GP-entry specification:
/// https://github.com/NVIDIA/open-gpu-doc/blob/ab27fc22db5de0d02a4cabe08e555663b62db4d4/manuals/volta/gv100/dev_pbdma.ref.txt#L62-L250
pub fn decode_gpfifo_submission(
    profile: MaxwellGpuProfile,
    allocated_entries: u32,
    request: MaxwellGpfifoSubmitRequest,
    descriptor_bytes: &[u8],
) -> Result<MaxwellDecodedGpfifoSubmission, MaxwellGpfifoDecodeError> {
    if request.entry_count > allocated_entries {
        return Err(MaxwellGpfifoDecodeError::Invalid(
            MaxwellInvalidGpfifoSubmission::EntryCountExceedsAllocation {
                requested: request.entry_count,
                allocated: allocated_entries,
            },
        ));
    }
    let expected_bytes = usize::try_from(request.entry_count)
        .ok()
        .and_then(|entries| entries.checked_mul(MAXWELL_GPFIFO_ENTRY_SIZE))
        .ok_or(MaxwellGpfifoDecodeError::Invalid(
            MaxwellInvalidGpfifoSubmission::EntryByteCountOverflow {
                entries: request.entry_count,
            },
        ))?;
    if descriptor_bytes.len() != expected_bytes {
        return Err(MaxwellGpfifoDecodeError::Invalid(
            MaxwellInvalidGpfifoSubmission::EntryByteCountMismatch {
                expected: expected_bytes,
                actual: descriptor_bytes.len(),
            },
        ));
    }

    let unknown_flags = request.flags & !KNOWN_SUBMIT_FLAGS;
    if unknown_flags != 0 {
        return Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::UnknownSubmitFlags {
                flags: unknown_flags,
            },
        ));
    }
    if request.flags & SUBMIT_HARDWARE_FORMAT == 0 {
        return Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::NonHardwareDescriptorFormat,
        ));
    }
    if request.flags & SUBMIT_SYNC_FENCE != 0 {
        return Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::SyncFenceFileDescriptor,
        ));
    }
    if request.flags & SUBMIT_FENCE_INCREMENT_VALUE != 0 {
        if request.flags & SUBMIT_FENCE_GET != 0 {
            return Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ConflictingFenceCompletionModes,
            ));
        }
        if request.fence_value == 0 {
            return Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ZeroFenceIncrementValue,
            ));
        }
    }

    let address_bits = profile.virtual_address().address_bits().bits();
    let mut entries = Vec::with_capacity(expected_bytes / MAXWELL_GPFIFO_ENTRY_SIZE);
    for (index, bytes) in descriptor_bytes
        .chunks_exact(MAXWELL_GPFIFO_ENTRY_SIZE)
        .enumerate()
    {
        let entry = u32::try_from(index).expect("u32 entry count bounds the decoded index");
        let word0 = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let word1 = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if word0 & ENTRY0_RESERVED != 0 {
            return Err(MaxwellGpfifoDecodeError::Invalid(
                MaxwellInvalidGpfifoSubmission::ReservedEntryBit {
                    entry,
                    word: 0,
                    mask: word0 & ENTRY0_RESERVED,
                },
            ));
        }
        if word0 & ENTRY0_FETCH_CONDITIONAL != 0 {
            return Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ConditionalFetch { entry },
            ));
        }

        let word_count = (word1 >> ENTRY1_WORD_COUNT_SHIFT) & ENTRY1_WORD_COUNT_MASK;
        if word_count == 0 {
            return Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ControlEntry {
                    entry,
                    opcode: (word1 & ENTRY1_ADDRESS_HIGH_MASK) as u8,
                },
            ));
        }
        let address_value =
            u64::from(word0 & !0b11) | (u64::from(word1 & ENTRY1_ADDRESS_HIGH_MASK) << 32);
        let address = GpuVirtualAddress::try_new(address_value, address_bits).map_err(|_| {
            MaxwellGpfifoDecodeError::Invalid(MaxwellInvalidGpfifoSubmission::AddressOutOfRange {
                entry,
                address: address_value,
                bits: address_bits,
            })
        })?;
        if address
            .checked_add(u64::from(word_count) * 4, address_bits)
            .is_err()
        {
            return Err(MaxwellGpfifoDecodeError::Invalid(
                MaxwellInvalidGpfifoSubmission::PushbufferRangeOutOfRange {
                    entry,
                    address: address_value,
                    words: word_count,
                    bits: address_bits,
                },
            ));
        }
        entries.push(MaxwellGpfifoEntry {
            address,
            word_count,
            allow_flush: word1 & ENTRY1_ALLOW_FLUSH != 0,
            level: if word1 & ENTRY1_LEVEL_SUBROUTINE != 0 {
                MaxwellGpfifoLevel::Subroutine
            } else {
                MaxwellGpfifoLevel::Main
            },
            sync: if word1 & ENTRY1_SYNC_WAIT != 0 {
                MaxwellGpfifoSyncMode::Wait
            } else {
                MaxwellGpfifoSyncMode::Proceed
            },
            fetch: MaxwellGpfifoFetchMode::Unconditional,
        });
    }

    Ok(MaxwellDecodedGpfifoSubmission {
        mode: MaxwellGpfifoSubmissionMode {
            fence_wait: request.flags & SUBMIT_FENCE_WAIT != 0,
            fence_get: request.flags & SUBMIT_FENCE_GET != 0,
            suppress_wfi: request.flags & SUBMIT_SUPPRESS_WFI != 0,
            skip_buffer_refcounting: request.flags & SUBMIT_SKIP_BUFFER_REFCOUNTING != 0,
            fence_increment_value: request.flags & SUBMIT_FENCE_INCREMENT_VALUE != 0,
        },
        fence_id: request.fence_id,
        fence_value: request.fence_value,
        entries: entries.into_boxed_slice(),
    })
}
