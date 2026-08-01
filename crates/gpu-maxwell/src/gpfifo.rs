//! Switch 1 Maxwell GPFIFO descriptor and submission semantics.
//!
//! Horizon owns the ioctl structure and buffer transport. This module accepts
//! already decoded scalar fields plus the complete descriptor byte array and
//! validates it without retaining memory or mutating a channel.

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress, MappingGeneration};
use nixe_memory::MemoryPermissions;

use crate::{
    MaxwellAddressSpaceId, MaxwellAllocationId, MaxwellChannelId, MaxwellGpuAccessError,
    MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellGpuMapping, MaxwellGpuProfile,
    MaxwellMappingId, MaxwellResolvedRange,
};

/// Size of one hardware-format GPFIFO descriptor.
pub const MAXWELL_GPFIFO_ENTRY_SIZE: usize = 8;

/// Hard upper bound for mapping records copied into one host diagnostic.
pub const MAXWELL_GPFIFO_CAPTURE_SOURCES: usize = 64;

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

/// Exact retained source of one decoded pushbuffer entry.
///
/// The resolved range owns canonical backing pages and mapping generations;
/// it contains neither a guest CPU address nor a host pointer. Consumers must
/// validate the complete submission against its live address space before
/// reading the first command word.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellRetainedPushbuffer {
    entry_index: u32,
    entry: MaxwellGpfifoEntry,
    source: MaxwellResolvedRange,
}

impl MaxwellRetainedPushbuffer {
    #[must_use]
    pub const fn entry_index(&self) -> u32 {
        self.entry_index
    }

    #[must_use]
    pub const fn entry(&self) -> MaxwellGpfifoEntry {
        self.entry
    }

    #[must_use]
    pub const fn source(&self) -> &MaxwellResolvedRange {
        &self.source
    }
}

/// Immutable GPFIFO submission whose complete command source is retained.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellValidatedGpfifoSubmission {
    channel: MaxwellChannelId,
    frontend: FrontendSubmissionId,
    address_space: MaxwellAddressSpaceId,
    decoded: MaxwellDecodedGpfifoSubmission,
    pushbuffers: Box<[MaxwellRetainedPushbuffer]>,
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

    #[must_use]
    pub fn pushbuffers(&self) -> &[MaxwellRetainedPushbuffer] {
        &self.pushbuffers
    }

    /// Returns the exact source of the first packet header, when work exists.
    ///
    /// This identifies the frontend-consumer boundary without interpreting a
    /// Maxwell packet. Packet decoding and method dispatch remain T7 work.
    pub fn first_packet_location(
        &self,
    ) -> Result<Option<MaxwellGpfifoSourceLocation>, MaxwellGpfifoSourceError> {
        self.pushbuffers
            .first()
            .map(|pushbuffer| first_source_location(self, pushbuffer))
            .transpose()
    }

    /// Checks every retained mapping before any command backing is accessed.
    pub fn validate_sources(
        &self,
        address_space: &MaxwellGpuAddressSpace,
    ) -> Result<(), MaxwellGpfifoSourceError> {
        if address_space.id() != self.address_space {
            return Err(MaxwellGpfifoSourceError::WrongAddressSpace {
                expected: self.address_space,
                actual: address_space.id(),
            });
        }
        for pushbuffer in &self.pushbuffers {
            for segment in pushbuffer.source.segments() {
                if !address_space.retained_mapping_is_current(segment.mapping()) {
                    return Err(MaxwellGpfifoSourceError::StaleMapping {
                        location: source_location(
                            self.channel,
                            self.frontend,
                            pushbuffer.entry_index,
                            pushbuffer.entry.address(),
                            segment.mapping(),
                            segment.gpu_offset(),
                        )?,
                    });
                }
            }
        }
        Ok(())
    }

    /// Reads one complete entry after atomically revalidating every source.
    ///
    /// Revalidating the whole submission first prevents a later entry's hole
    /// or remap from being discovered after a prefix was already consumed.
    pub fn read_pushbuffer(
        &self,
        address_space: &MaxwellGpuAddressSpace,
        entry_index: u32,
        output: &mut [u8],
    ) -> Result<(), MaxwellGpfifoSourceError> {
        self.validate_sources(address_space)?;
        let pushbuffer = self
            .pushbuffers
            .get(entry_index as usize)
            .ok_or(MaxwellGpfifoSourceError::UnknownEntry { entry_index })?;
        address_space
            .read_resolved(&pushbuffer.source, output)
            .map_err(|error| MaxwellGpfifoSourceError::Access {
                location: first_source_location(self, pushbuffer)
                    .expect("resolved non-empty range has a first mapping"),
                error,
            })
    }

    /// Builds a pointer-free, hard-bounded summary for host diagnostics.
    #[must_use]
    pub fn capture(&self) -> MaxwellGpfifoCapture {
        let mut sources = Vec::new();
        let total_sources = self
            .pushbuffers
            .iter()
            .map(|pushbuffer| pushbuffer.source.segments().len())
            .sum();
        for pushbuffer in &self.pushbuffers {
            for segment in pushbuffer.source.segments() {
                if sources.len() == MAXWELL_GPFIFO_CAPTURE_SOURCES {
                    break;
                }
                let mapping = segment.mapping();
                sources.push(MaxwellGpfifoCaptureSource {
                    entry_index: pushbuffer.entry_index,
                    pushbuffer: pushbuffer.entry.address(),
                    word_offset: (segment.gpu_offset().get() - pushbuffer.entry.address().get())
                        / 4,
                    size: segment.size(),
                    mapping: mapping.id(),
                    generation: mapping.generation(),
                    allocation: mapping.allocation(),
                    backing_offset: segment.backing_offset(),
                });
            }
            if sources.len() == MAXWELL_GPFIFO_CAPTURE_SOURCES {
                break;
            }
        }
        MaxwellGpfifoCapture {
            channel: self.channel,
            frontend: self.frontend,
            address_space: self.address_space,
            total_entries: self.pushbuffers.len(),
            total_sources,
            sources: sources.into_boxed_slice(),
        }
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

/// One mapping fragment in a bounded GPFIFO capture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellGpfifoCaptureSource {
    pub entry_index: u32,
    pub pushbuffer: GpuVirtualAddress,
    pub word_offset: u64,
    pub size: u64,
    pub mapping: MaxwellMappingId,
    pub generation: MappingGeneration,
    pub allocation: MaxwellAllocationId,
    pub backing_offset: u64,
}

/// Bounded metadata capture of a completely retained submission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellGpfifoCapture {
    channel: MaxwellChannelId,
    frontend: FrontendSubmissionId,
    address_space: MaxwellAddressSpaceId,
    total_entries: usize,
    total_sources: usize,
    sources: Box<[MaxwellGpfifoCaptureSource]>,
}

impl MaxwellGpfifoCapture {
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
    pub const fn total_entries(&self) -> usize {
        self.total_entries
    }
    #[must_use]
    pub const fn total_sources(&self) -> usize {
        self.total_sources
    }
    #[must_use]
    pub fn sources(&self) -> &[MaxwellGpfifoCaptureSource] {
        &self.sources
    }
    #[must_use]
    pub const fn omitted_sources(&self) -> usize {
        self.total_sources.saturating_sub(self.sources.len())
    }
}

impl Display for MaxwellGpfifoCapture {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {} address-space={} entries={} sources={} shown={}",
            self.channel,
            self.frontend,
            self.address_space.get(),
            self.total_entries,
            self.total_sources,
            self.sources.len()
        )?;
        if let Some(first) = self.sources.first() {
            write!(
                formatter,
                " first=[gpfifo-entry={} pushbuffer={} word-offset={} {} {} {} backing-offset=0x{:x} size=0x{:x}]",
                first.entry_index,
                first.pushbuffer,
                first.word_offset,
                first.mapping,
                first.generation,
                first.allocation,
                first.backing_offset,
                first.size
            )?;
        }
        if self.omitted_sources() != 0 {
            write!(formatter, " sources-truncated={}", self.omitted_sources())?;
        }
        Ok(())
    }
}

/// Failure while resolving or revalidating a complete command source.
#[derive(Clone, Debug, Eq, PartialEq)]
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
    Resolution {
        channel: MaxwellChannelId,
        frontend: FrontendSubmissionId,
        entry_index: u32,
        pushbuffer: GpuVirtualAddress,
        address_space_generation: MappingGeneration,
        error: MaxwellGpuAccessError,
    },
    StaleMapping {
        location: MaxwellGpfifoSourceLocation,
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
            Self::StaleMapping { location } => {
                write!(formatter, "GPFIFO source mapping is stale: {location}")
            }
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

/// Resolves and retains every command byte before returning any usable work.
pub fn resolve_gpfifo_submission(
    channel: &MaxwellGpuChannel,
    frontend: FrontendSubmissionId,
    decoded: MaxwellDecodedGpfifoSubmission,
    address_space: &MaxwellGpuAddressSpace,
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
    let mut pushbuffers = Vec::new();
    pushbuffers
        .try_reserve_exact(decoded.entries.len())
        .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
    for (index, entry) in decoded.entries.iter().copied().enumerate() {
        let entry_index =
            u32::try_from(index).map_err(|_| MaxwellGpfifoSourceError::ArithmeticOverflow)?;
        let source = address_space
            .resolve_range(entry.address(), entry.byte_count(), MemoryPermissions::READ)
            .map_err(|error| MaxwellGpfifoSourceError::Resolution {
                channel: channel_id,
                frontend,
                entry_index,
                pushbuffer: entry.address(),
                address_space_generation: address_space.mapping_generation(),
                error,
            })?;
        pushbuffers.push(MaxwellRetainedPushbuffer {
            entry_index,
            entry,
            source,
        });
    }
    let submission = MaxwellValidatedGpfifoSubmission {
        channel: channel_id,
        frontend,
        address_space: address_space.id(),
        decoded,
        pushbuffers: pushbuffers.into_boxed_slice(),
    };
    submission.validate_sources(address_space)?;
    Ok(submission)
}

fn first_source_location(
    submission: &MaxwellValidatedGpfifoSubmission,
    pushbuffer: &MaxwellRetainedPushbuffer,
) -> Result<MaxwellGpfifoSourceLocation, MaxwellGpfifoSourceError> {
    let segment = pushbuffer
        .source
        .segments()
        .first()
        .ok_or(MaxwellGpfifoSourceError::ArithmeticOverflow)?;
    source_location(
        submission.channel,
        submission.frontend,
        pushbuffer.entry_index,
        pushbuffer.entry.address(),
        segment.mapping(),
        segment.gpu_offset(),
    )
}

fn source_location(
    channel: MaxwellChannelId,
    frontend: FrontendSubmissionId,
    entry_index: u32,
    pushbuffer: GpuVirtualAddress,
    mapping: &MaxwellGpuMapping,
    segment_offset: GpuVirtualAddress,
) -> Result<MaxwellGpfifoSourceLocation, MaxwellGpfifoSourceError> {
    let byte_offset = segment_offset
        .get()
        .checked_sub(pushbuffer.get())
        .ok_or(MaxwellGpfifoSourceError::ArithmeticOverflow)?;
    Ok(MaxwellGpfifoSourceLocation {
        channel,
        frontend,
        entry_index,
        pushbuffer,
        word_offset: byte_offset / 4,
        mapping: mapping.id(),
        generation: mapping.generation(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellChannelOwner,
        MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
    };
    use nixe_memory::CanonicalAllocation;

    fn descriptor(address: u64, words: u32, modes: u32) -> [u8; 8] {
        let word0 = address as u32;
        let word1 = ((address >> 32) as u32 & ENTRY1_ADDRESS_HIGH_MASK)
            | (words << ENTRY1_WORD_COUNT_SHIFT)
            | modes;
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&word0.to_le_bytes());
        bytes[4..].copy_from_slice(&word1.to_le_bytes());
        bytes
    }

    fn request(entries: u32, flags: u32) -> MaxwellGpfifoSubmitRequest {
        MaxwellGpfifoSubmitRequest {
            entry_count: entries,
            flags,
            fence_id: 7,
            fence_value: 11,
        }
    }

    fn bound_channel(id: u64, address_space: MaxwellAddressSpaceId) -> MaxwellGpuChannel {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(id),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        channel.bind_address_space(address_space).unwrap();
        channel
    }

    #[test]
    fn decodes_every_descriptor_and_switch_mode_without_retaining_guest_memory() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&descriptor(0x12_3456_7000, 3, 0));
        bytes.extend_from_slice(&descriptor(
            0x23_4567_8000,
            5,
            ENTRY1_ALLOW_FLUSH | ENTRY1_LEVEL_SUBROUTINE | ENTRY1_SYNC_WAIT,
        ));
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            0x800,
            request(
                2,
                SUBMIT_HARDWARE_FORMAT
                    | SUBMIT_FENCE_WAIT
                    | SUBMIT_SUPPRESS_WFI
                    | SUBMIT_FENCE_INCREMENT_VALUE,
            ),
            &bytes,
        )
        .unwrap();

        assert_eq!(decoded.fence_id(), 7);
        assert_eq!(decoded.fence_value(), 11);
        assert!(decoded.mode().fence_wait());
        assert!(decoded.mode().suppress_wfi());
        assert!(decoded.mode().fence_increment_value());
        assert_eq!(decoded.entries()[0].address().get(), 0x12_3456_7000);
        assert_eq!(decoded.entries()[0].word_count(), 3);
        assert_eq!(decoded.entries()[0].byte_count(), 12);
        assert_eq!(decoded.entries()[1].level(), MaxwellGpfifoLevel::Subroutine);
        assert_eq!(decoded.entries()[1].sync(), MaxwellGpfifoSyncMode::Wait);
        assert!(decoded.entries()[1].allow_flush());
    }

    #[test]
    fn rejects_count_size_and_profile_range_failures() {
        assert_eq!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                1,
                request(2, SUBMIT_HARDWARE_FORMAT),
                &[0; 16],
            ),
            Err(MaxwellGpfifoDecodeError::Invalid(
                MaxwellInvalidGpfifoSubmission::EntryCountExceedsAllocation {
                    requested: 2,
                    allocated: 1,
                }
            ))
        );
        assert!(matches!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                2,
                request(2, SUBMIT_HARDWARE_FORMAT),
                &[0; 8],
            ),
            Err(MaxwellGpfifoDecodeError::Invalid(
                MaxwellInvalidGpfifoSubmission::EntryByteCountMismatch { .. }
            ))
        ));
        let at_end = descriptor((1_u64 << 40) - 4, 1, 0);
        assert!(matches!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                1,
                request(1, SUBMIT_HARDWARE_FORMAT),
                &at_end,
            ),
            Err(MaxwellGpfifoDecodeError::Invalid(
                MaxwellInvalidGpfifoSubmission::PushbufferRangeOutOfRange { .. }
            ))
        ));
    }

    #[test]
    fn rejects_unsupported_modes_before_returning_any_prefix() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&descriptor(0x1000, 4, 0));
        bytes.extend_from_slice(&descriptor(0x2001, 4, 0));
        assert_eq!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                2,
                request(2, SUBMIT_HARDWARE_FORMAT),
                &bytes,
            ),
            Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ConditionalFetch { entry: 1 }
            ))
        );
        assert!(matches!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                0,
                request(0, SUBMIT_HARDWARE_FORMAT | SUBMIT_SYNC_FENCE),
                &[],
            ),
            Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::SyncFenceFileDescriptor
            ))
        ));
        assert_eq!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                0,
                MaxwellGpfifoSubmitRequest {
                    entry_count: 0,
                    flags: SUBMIT_HARDWARE_FORMAT | SUBMIT_FENCE_GET | SUBMIT_FENCE_INCREMENT_VALUE,
                    fence_id: 0,
                    fence_value: 1,
                },
                &[],
            ),
            Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ConflictingFenceCompletionModes
            ))
        );
        assert_eq!(
            decode_gpfifo_submission(
                SWITCH_1_GM20B_PROFILE,
                0,
                MaxwellGpfifoSubmitRequest {
                    entry_count: 0,
                    flags: SUBMIT_HARDWARE_FORMAT | SUBMIT_FENCE_INCREMENT_VALUE,
                    fence_id: 0,
                    fence_value: 0,
                },
                &[],
            ),
            Err(MaxwellGpfifoDecodeError::Unsupported(
                MaxwellUnsupportedGpfifoSubmission::ZeroFenceIncrementValue
            ))
        );
    }

    #[test]
    fn accepts_a_completely_empty_hardware_format_submission() {
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            0x800,
            request(0, SUBMIT_HARDWARE_FORMAT),
            &[],
        )
        .unwrap();
        assert!(decoded.entries().is_empty());
    }

    #[test]
    fn resolves_complete_ranges_and_retains_exact_mapping_lifetimes() {
        let first_allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let second_allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        first_allocation.write(0xffc, &[1, 2, 3, 4]).unwrap();
        second_allocation
            .write(0, &[5, 6, 7, 8, 9, 10, 11, 12])
            .unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(9), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let first = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: first_allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let second = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(2),
                backing: second_allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        assert_eq!(second.offset().get(), first.offset().get() + 0x1000);

        let pushbuffer = first.offset().get() + 0xffc;
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            4,
            request(1, SUBMIT_HARDWARE_FORMAT),
            &descriptor(pushbuffer, 3, 0),
        )
        .unwrap();
        let channel = bound_channel(4, address_space.id());
        let validated = resolve_gpfifo_submission(
            &channel,
            FrontendSubmissionId::new(7),
            decoded,
            &address_space,
        )
        .unwrap();

        assert_eq!(validated.pushbuffers()[0].source().segments().len(), 2);
        let capture = validated.capture();
        assert_eq!(capture.channel(), MaxwellChannelId::new(4));
        assert_eq!(capture.frontend(), FrontendSubmissionId::new(7));
        assert_eq!(capture.total_entries(), 1);
        assert_eq!(capture.total_sources(), 2);
        assert_eq!(capture.sources()[0].word_offset, 0);
        assert_eq!(capture.sources()[1].word_offset, 1);
        let mut bytes = [0; 12];
        validated
            .read_pushbuffer(&address_space, 0, &mut bytes)
            .unwrap();
        assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        drop(first_allocation);
        address_space.unmap(first.offset()).unwrap();
        assert!(matches!(
            validated.validate_sources(&address_space),
            Err(MaxwellGpfifoSourceError::StaleMapping { location })
                if location.entry_index == 0
                    && location.word_offset == 0
                    && location.mapping == first.id()
                    && location.generation == first.generation()
        ));
        let retained_mapping = validated.pushbuffers()[0].source().segments()[0].mapping();
        let mut retained_bytes = [0; 4];
        retained_mapping
            .backing()
            .read(
                retained_mapping.backing_offset() + 0xffc,
                &mut retained_bytes,
            )
            .unwrap();
        assert_eq!(retained_bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn rejects_a_later_hole_before_returning_a_retained_prefix() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(3), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(1),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let mut entries = Vec::new();
        entries.extend_from_slice(&descriptor(mapping.offset().get(), 1, 0));
        entries.extend_from_slice(&descriptor(mapping.offset().get() + 0x2000, 1, 0));
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            4,
            request(2, SUBMIT_HARDWARE_FORMAT),
            &entries,
        )
        .unwrap();

        assert!(matches!(
            resolve_gpfifo_submission(
                &bound_channel(1, address_space.id()),
                FrontendSubmissionId::new(1),
                decoded,
                &address_space,
            ),
            Err(MaxwellGpfifoSourceError::Resolution {
                entry_index: 1,
                error: MaxwellGpuAccessError::UnmappedAddress { .. },
                ..
            })
        ));
    }

    #[test]
    fn multiple_entry_subroutine_capture_replays_the_same_first_packet() {
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        allocation
            .write(0, &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
            .unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(5), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(3),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x2000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let mut entries = Vec::new();
        entries.extend_from_slice(&descriptor(mapping.offset().get(), 1, 0));
        entries.extend_from_slice(&descriptor(
            mapping.offset().get() + 4,
            1,
            ENTRY1_LEVEL_SUBROUTINE | ENTRY1_SYNC_WAIT,
        ));
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            4,
            request(2, SUBMIT_HARDWARE_FORMAT),
            &entries,
        )
        .unwrap();
        let channel = bound_channel(6, address_space.id());
        let validated = resolve_gpfifo_submission(
            &channel,
            FrontendSubmissionId::new(12),
            decoded,
            &address_space,
        )
        .unwrap();

        assert_eq!(validated.pushbuffers().len(), 2);
        assert_eq!(
            validated.pushbuffers()[1].entry().level(),
            MaxwellGpfifoLevel::Subroutine
        );
        assert_eq!(
            validated.pushbuffers()[1].entry().sync(),
            MaxwellGpfifoSyncMode::Wait
        );
        let first_location = validated.first_packet_location().unwrap().unwrap();
        let replay = validated.clone();
        assert_eq!(replay.capture(), validated.capture());
        assert_eq!(
            replay.first_packet_location().unwrap(),
            Some(first_location)
        );
        assert_eq!(first_location.entry_index, 0);
        assert_eq!(first_location.pushbuffer, mapping.offset());
        assert_eq!(first_location.word_offset, 0);
    }

    #[test]
    fn capture_is_hard_bounded_without_changing_submission_identity() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(6), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(20),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let entry_count = u32::try_from(MAXWELL_GPFIFO_CAPTURE_SOURCES + 1).unwrap();
        let mut entries = Vec::new();
        for _ in 0..entry_count {
            entries.extend_from_slice(&descriptor(mapping.offset().get(), 1, 0));
        }
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            entry_count,
            request(entry_count, SUBMIT_HARDWARE_FORMAT),
            &entries,
        )
        .unwrap();
        let validated = resolve_gpfifo_submission(
            &bound_channel(7, address_space.id()),
            FrontendSubmissionId::new(14),
            decoded,
            &address_space,
        )
        .unwrap();

        let capture = validated.capture();
        assert_eq!(capture.channel(), validated.channel());
        assert_eq!(capture.frontend(), validated.frontend());
        assert_eq!(capture.total_entries(), entry_count as usize);
        assert_eq!(capture.total_sources(), entry_count as usize);
        assert_eq!(capture.sources().len(), MAXWELL_GPFIFO_CAPTURE_SOURCES);
        assert_eq!(capture.omitted_sources(), 1);
        assert_eq!(
            validated.first_packet_location().unwrap(),
            validated.clone().first_packet_location().unwrap()
        );
    }

    #[test]
    fn remapping_the_same_gpu_va_cannot_retarget_a_retained_submission() {
        let original = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let replacement = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        original.write(0, &[1, 2, 3, 4]).unwrap();
        replacement.write(0, &[9, 8, 7, 6]).unwrap();
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(7), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let first = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(4),
                backing: original
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let channel = bound_channel(8, address_space.id());
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            1,
            request(1, SUBMIT_HARDWARE_FORMAT),
            &descriptor(first.offset().get(), 1, 0),
        )
        .unwrap();
        let retained = resolve_gpfifo_submission(
            &channel,
            FrontendSubmissionId::new(13),
            decoded,
            &address_space,
        )
        .unwrap();

        address_space.unmap(first.offset()).unwrap();
        let remapped = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(5),
                backing: replacement
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        assert_eq!(remapped.offset(), first.offset());
        assert_ne!(remapped.id(), first.id());
        assert_ne!(remapped.generation(), first.generation());
        assert!(matches!(
            retained.validate_sources(&address_space),
            Err(MaxwellGpfifoSourceError::StaleMapping { location })
                if location.mapping == first.id()
                    && location.generation == first.generation()
        ));

        let retained_mapping = retained.pushbuffers()[0].source().segments()[0].mapping();
        let mut retained_bytes = [0; 4];
        retained_mapping
            .backing()
            .read(retained_mapping.backing_offset(), &mut retained_bytes)
            .unwrap();
        assert_eq!(retained_bytes, [1, 2, 3, 4]);
        let mut replacement_bytes = [0; 4];
        let replacement_range = address_space
            .resolve_range(remapped.offset(), 4, MemoryPermissions::READ)
            .unwrap();
        address_space
            .read_resolved(&replacement_range, &mut replacement_bytes)
            .unwrap();
        assert_eq!(replacement_bytes, [9, 8, 7, 6]);
    }
}
