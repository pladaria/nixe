//! Opt-in deterministic reports for one frontend translation block.

use crate::{
    address::{AddressSpaceId, CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
    error::{
        FrontendError, FrontendInternalError, InstructionFetchFault, InstructionFetchFaultReason,
    },
    ir::{
        block::{BlockEndReason, IrBlock},
        print::{IrDumpStage, IrPrintOptions, print_ir_dump},
    },
    location::{ExecutionState, LocationDescriptor},
    memory::{
        CodeDependencies, CodePageDependency, CodePageSpan, FetchedCode, InstructionMemory,
        SYNTHETIC_PAGE_SIZE,
    },
    profile::GuestCpuProfile,
};

use super::{BlockTranslationConfig, block::translate_block_with_disassembly};

/// Stable address-space identity used only by raw-byte diagnostic fixtures.
const RAW_DIAGNOSTIC_ADDRESS_SPACE: AddressSpaceId = AddressSpaceId::new(0x5357_4954_5844_4247);

/// Reason translation failed before a valid IR block existed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlockTranslationFailureReason {
    /// Instruction memory could not provide the requested bytes.
    FetchFault,
    /// Configuration, address validation, IR verification, or an internal
    /// frontend invariant prevented translation.
    TranslationFailure,
}

impl core::fmt::Display for BlockTranslationFailureReason {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::FetchFault => "fetch-fault",
            Self::TranslationFailure => "translation-failure",
        })
    }
}

/// Complete reason represented by a successful or failed block report.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlockReportEndReason {
    /// A verified block ended for the contained semantic or policy reason.
    Block(BlockEndReason),
    /// Translation failed before a valid block existed.
    Failure(BlockTranslationFailureReason),
}

impl core::fmt::Display for BlockReportEndReason {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Block(reason) => reason.fmt(formatter),
            Self::Failure(reason) => reason.fmt(formatter),
        }
    }
}

/// Result of an opt-in single-block diagnostic translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockTranslationReport {
    /// A verified pre-optimization IR block was produced.
    Translated(IrBlock),
    /// Translation stopped before a valid block could be produced.
    Failed {
        start: LocationDescriptor,
        end_reason: BlockTranslationFailureReason,
        error: FrontendError,
    },
}

impl BlockTranslationReport {
    /// Returns the exact semantic, policy, or failure reason which ended work.
    #[must_use]
    pub const fn end_reason(&self) -> BlockReportEndReason {
        match self {
            Self::Translated(block) => BlockReportEndReason::Block(block.metadata.end_reason),
            Self::Failed { end_reason, .. } => BlockReportEndReason::Failure(*end_reason),
        }
    }

    /// Returns the verified block when translation succeeded.
    #[must_use]
    pub const fn block(&self) -> Option<&IrBlock> {
        match self {
            Self::Translated(block) => Some(block),
            Self::Failed { .. } => None,
        }
    }

    /// Returns the structured frontend failure when translation failed.
    #[must_use]
    pub const fn error(&self) -> Option<&FrontendError> {
        match self {
            Self::Translated(_) => None,
            Self::Failed { error, .. } => Some(error),
        }
    }

    /// Converts the report back into the ordinary translation result surface.
    pub fn into_result(self) -> Result<IrBlock, FrontendError> {
        match self {
            Self::Translated(block) => Ok(block),
            Self::Failed { error, .. } => Err(error),
        }
    }

    /// Prints a compact deterministic report with pre-optimization IR.
    #[must_use]
    pub fn print(&self) -> String {
        use core::fmt::Write;

        let mut output = String::from("nixe-frontend-block-report-v1\n");
        match self {
            Self::Translated(block) => {
                writeln!(
                    output,
                    "start={} state={} {}",
                    block.metadata.start.pc,
                    block.metadata.start.execution_state,
                    block.metadata.start.profile_id
                )
                .expect("writing to a String cannot fail");
                writeln!(
                    output,
                    "outcome=translated end={}",
                    block.metadata.end_reason
                )
                .expect("writing to a String cannot fail");
                output.push_str(&print_ir_dump(
                    block,
                    IrDumpStage::PreOptimization,
                    IrPrintOptions::default(),
                ));
            }
            Self::Failed {
                start,
                end_reason,
                error,
            } => {
                writeln!(
                    output,
                    "start={} state={} {}",
                    start.pc, start.execution_state, start.profile_id
                )
                .expect("writing to a String cannot fail");
                writeln!(output, "outcome=failed end={end_reason}")
                    .expect("writing to a String cannot fail");
                writeln!(output, "error={error}").expect("writing to a String cannot fail");
            }
        }
        output.push_str("end-report\n");
        output
    }
}

/// Translates one process-memory block while collecting optional source text.
///
/// Unlike [`super::translate_block`], this path constructs a disassembly string
/// for each recognized source instruction. Callers should invoke it only when a
/// diagnostic report or IR dump has been requested.
#[must_use]
pub fn translate_block_report(
    config: BlockTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &impl InstructionMemory,
) -> BlockTranslationReport {
    match translate_block_with_disassembly(config, profile, address_space, start, memory) {
        Ok(block) => BlockTranslationReport::Translated(block),
        Err(error) => failure(start, error),
    }
}

/// Translates raw little-endian guest bytes through the normal frontend.
///
/// The supplied byte extent acts as a bounded synthetic executable mapping.
/// Its physical page IDs and generations are deterministic diagnostic
/// identities, not host addresses. This helper is intended for commands,
/// regression tests, and bug reproduction, never as process memory.
#[must_use]
pub fn translate_raw_block_report(
    config: BlockTranslationConfig,
    profile: &GuestCpuProfile,
    base_pc: GuestVirtualAddress,
    execution_state: ExecutionState,
    bytes: &[u8],
) -> BlockTranslationReport {
    let start = LocationDescriptor::new(base_pc, execution_state, profile.id());
    let memory = match RawInstructionMemory::new(base_pc, bytes) {
        Ok(memory) => memory,
        Err(error) => return failure(start, error),
    };
    translate_block_report(
        config,
        profile,
        RAW_DIAGNOSTIC_ADDRESS_SPACE,
        start,
        &memory,
    )
}

fn failure(start: LocationDescriptor, error: FrontendError) -> BlockTranslationReport {
    let end_reason = if matches!(error, FrontendError::InstructionFetch(_)) {
        BlockTranslationFailureReason::FetchFault
    } else {
        BlockTranslationFailureReason::TranslationFailure
    };
    BlockTranslationReport::Failed {
        start,
        end_reason,
        error,
    }
}

struct RawInstructionMemory<'a> {
    base: GuestVirtualAddress,
    end: GuestVirtualAddress,
    bytes: &'a [u8],
}

impl<'a> RawInstructionMemory<'a> {
    fn new(base: GuestVirtualAddress, bytes: &'a [u8]) -> Result<Self, FrontendError> {
        let byte_count = u64::try_from(bytes.len()).map_err(|_| {
            FrontendInternalError::new(None, "raw diagnostic input length exceeds the guest domain")
        })?;
        let end = base.checked_add(byte_count).ok_or_else(|| {
            FrontendInternalError::new(
                None,
                "raw diagnostic input range overflows the guest domain",
            )
        })?;
        Ok(Self { base, end, bytes })
    }

    fn fetch<const N: usize>(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        alignment: u8,
    ) -> Result<([u8; N], CodeDependencies), InstructionFetchFault> {
        if !address.is_aligned_to(u64::from(alignment)) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Misaligned {
                    required_alignment: alignment,
                },
            ));
        }
        let Some(offset) = address.get().checked_sub(self.base.get()) else {
            return Err(unmapped(address_space, address));
        };
        let offset = usize::try_from(offset).map_err(|_| unmapped(address_space, address))?;
        let end = offset
            .checked_add(N)
            .ok_or_else(|| unmapped(address_space, address))?;
        let source = self
            .bytes
            .get(offset..end)
            .ok_or_else(|| unmapped(address_space, address))?;
        let bytes: [u8; N] = source
            .try_into()
            .expect("the checked raw diagnostic slice has the requested width");
        let last_address = address
            .checked_add((N - 1) as u64)
            .ok_or_else(|| unmapped(address_space, address))?;
        Ok((bytes, dependencies(address, last_address)))
    }
}

impl InstructionMemory for RawInstructionMemory<'_> {
    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        if address.get() < self.base.get() || address.get() >= self.end.get() {
            return Err(unmapped(address_space, address));
        }
        let page_size = SYNTHETIC_PAGE_SIZE as u64;
        let page_start = address.get() & !(page_size - 1);
        let available_start = page_start.max(self.base.get());
        let page_end = page_start.checked_add(page_size);
        let available_end = match page_end {
            Some(page_end) => self.end.get().min(page_end),
            None => self.end.get(),
        };
        CodePageSpan::containing(
            GuestVirtualAddress::new(available_start),
            Some(GuestVirtualAddress::new(available_end)),
            address,
        )
        .ok_or_else(|| unmapped(address_space, address))
    }

    fn fetch16(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u16>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<2>(address_space, address, 2)?;
        Ok(FetchedCode {
            bits: u16::from_le_bytes(bytes),
            dependencies,
        })
    }

    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<4>(address_space, address, 4)?;
        Ok(FetchedCode {
            bits: u32::from_le_bytes(bytes),
            dependencies,
        })
    }
}

fn dependencies(first: GuestVirtualAddress, last: GuestVirtualAddress) -> CodeDependencies {
    let first = dependency(first);
    let last = dependency(last);
    if first == last {
        CodeDependencies::one(first)
    } else {
        CodeDependencies::two(first, last)
    }
}

fn dependency(address: GuestVirtualAddress) -> CodePageDependency {
    CodePageDependency {
        page: GuestPhysicalPageId::new(address.get() / SYNTHETIC_PAGE_SIZE as u64),
        generation: CodeGeneration::new(1),
    }
}

fn unmapped(address_space: AddressSpaceId, address: GuestVirtualAddress) -> InstructionFetchFault {
    InstructionFetchFault::new(
        address_space,
        address,
        InstructionFetchFaultReason::Unmapped,
    )
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::{
        ir::{block::BlockEndReason, print::IrPrintOptions},
        location::InstructionEncoding,
    };

    #[test]
    fn raw_helper_reports_every_source_and_pre_optimization_ir_deterministically() {
        let profile = GuestCpuProfile::switch_1();
        let bytes = [
            0x1f, 0x20, 0x03, 0xd5, // nop
            0x01, 0x00, 0x00, 0xd4, // svc #0
        ];
        let report = translate_raw_block_report(
            BlockTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            &bytes,
        );

        let block = report.block().expect("raw block should translate");
        assert_eq!(block.metadata.end_reason, BlockEndReason::Exception);
        assert_eq!(block.metadata.sources.len(), 2);
        assert_eq!(
            block.metadata.sources[0].encoding,
            InstructionEncoding::from_u32(0xd503_201f)
        );
        assert_eq!(
            block.metadata.sources[0].disassembly.as_deref(),
            Some("nop")
        );
        assert_eq!(
            block.metadata.sources[1].disassembly.as_deref(),
            Some("svc")
        );

        let output = report.print();
        assert_eq!(output, report.print());
        assert!(output.contains("outcome=translated end=exception"));
        assert!(output.contains("ir-dump stage=pre-optimization"));
        assert!(
            output.contains(
                "source pc=0x0000000000001000 state=A64 ; raw=0xd503201f ; guest=\"nop\""
            )
        );
        assert!(
            output.contains(
                "source pc=0x0000000000001004 state=A64 ; raw=0xd4000001 ; guest=\"svc\""
            )
        );
        assert!(
            output.contains("dependency page=0x0000000000000001 generation=0x0000000000000001")
        );
        assert!(!output.contains("0x7f"));

        let post = print_ir_dump(
            block,
            IrDumpStage::PostOptimization,
            IrPrintOptions::default(),
        );
        assert!(post.starts_with("ir-dump stage=post-optimization\n"));
    }

    #[test]
    fn raw_helper_covers_a32_t32_limits_and_page_dependencies() {
        let profile = GuestCpuProfile::switch_1();
        let a32 = translate_raw_block_report(
            BlockTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x2000),
            ExecutionState::A32,
            &0xeaff_ffff_u32.to_le_bytes(),
        );
        assert_eq!(
            a32.end_reason(),
            BlockReportEndReason::Block(BlockEndReason::DirectBranch)
        );
        assert!(a32.print().contains("state=A32"));

        let t32 = translate_raw_block_report(
            BlockTranslationConfig {
                max_guest_instructions: NonZeroU32::new(1).unwrap(),
            },
            &profile,
            GuestVirtualAddress::new(0x2ffe),
            ExecutionState::T32,
            &[0xaf, 0xf3, 0x00, 0x80],
        );
        let block = t32
            .block()
            .expect("cross-page T32 instruction should translate");
        assert_eq!(
            block.metadata.end_reason,
            BlockEndReason::InstructionLimitAtPageBoundary
        );
        assert_eq!(block.metadata.code_dependencies.len(), 2);
        assert_eq!(block.metadata.sources[0].dependencies.iter().count(), 2);
    }

    #[test]
    fn fetch_faults_and_other_failures_have_distinct_compact_reports() {
        let profile = GuestCpuProfile::switch_1();
        let fetch = translate_raw_block_report(
            BlockTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            &[],
        );
        assert_eq!(
            fetch.end_reason(),
            BlockReportEndReason::Failure(BlockTranslationFailureReason::FetchFault)
        );
        assert!(matches!(
            fetch.error(),
            Some(FrontendError::InstructionFetch(_))
        ));
        assert!(fetch.print().contains("outcome=failed end=fetch-fault"));

        let invalid = translate_raw_block_report(
            BlockTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x1002),
            ExecutionState::A64,
            &0xd503_201f_u32.to_le_bytes(),
        );
        assert_eq!(
            invalid.end_reason(),
            BlockReportEndReason::Failure(BlockTranslationFailureReason::TranslationFailure)
        );
        assert!(
            invalid
                .print()
                .contains("outcome=failed end=translation-failure")
        );
    }

    #[test]
    fn architectural_decode_rejections_keep_source_context_in_reports() {
        let profile = GuestCpuProfile::switch_1();
        let report = translate_raw_block_report(
            BlockTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x4000),
            ExecutionState::A64,
            &0_u32.to_le_bytes(),
        );
        let block = report
            .block()
            .expect("undefined instruction forms an exception block");
        assert_eq!(block.metadata.end_reason, BlockEndReason::Exception);
        assert!(
            block.metadata.sources[0]
                .disassembly
                .as_deref()
                .is_some_and(|text| text.starts_with("<unallocated:"))
        );
        assert!(report.print().contains("guest=\"<unallocated:"));
    }
}
