//! Opt-in deterministic reports for one frontend translation region.

use nixe_memory::{
    AddressSpaceId, ContentGeneration, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration,
};

use crate::{
    error::{
        FrontendError, FrontendInternalError, InstructionFetchFault, InstructionFetchFaultReason,
    },
    ir::{
        print::{IrDumpStage, IrPrintOptions, print_ir_dump},
        region::IrRegion,
    },
    location::{ExecutionState, LocationDescriptor},
    memory::{
        CodeDependencies, CodePageDependency, CodePageSpan, FetchedCode, InstructionMemory,
        SYNTHETIC_PAGE_SIZE,
    },
    profile::GuestCpuProfile,
};

use super::region::{RegionTranslationConfig, translate_region_with_disassembly};

/// Stable address-space identity used only by raw-byte diagnostic fixtures.
const RAW_DIAGNOSTIC_ADDRESS_SPACE: AddressSpaceId = AddressSpaceId::new(0x5357_4954_5844_4247);

/// Reason translation failed before a valid IR region existed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegionTranslationFailureReason {
    /// Instruction memory could not provide the requested bytes.
    FetchFault,
    /// Configuration, address validation, IR verification, or an internal
    /// frontend invariant prevented translation.
    TranslationFailure,
}

impl core::fmt::Display for RegionTranslationFailureReason {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::FetchFault => "fetch-fault",
            Self::TranslationFailure => "translation-failure",
        })
    }
}

/// Result of an opt-in bounded-region diagnostic translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegionTranslationReport {
    /// A verified pre-optimization IR region was produced.
    Translated(IrRegion),
    /// Translation stopped before a valid region could be produced.
    Failed {
        start: LocationDescriptor,
        reason: RegionTranslationFailureReason,
        error: FrontendError,
    },
}

impl RegionTranslationReport {
    /// Returns the verified region when translation succeeded.
    #[must_use]
    pub const fn region(&self) -> Option<&IrRegion> {
        match self {
            Self::Translated(region) => Some(region),
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
    pub fn into_result(self) -> Result<IrRegion, FrontendError> {
        match self {
            Self::Translated(region) => Ok(region),
            Self::Failed { error, .. } => Err(error),
        }
    }

    /// Prints a compact deterministic report with pre-optimization IR.
    #[must_use]
    pub fn print(&self) -> String {
        use core::fmt::Write;

        let mut output = String::from("nixe-frontend-region-report-v1\n");
        match self {
            Self::Translated(region) => {
                writeln!(
                    output,
                    "start={} state={} {}",
                    region.metadata.start.pc,
                    region.metadata.start.execution_state,
                    region.metadata.start.profile_id
                )
                .expect("writing to a String cannot fail");
                writeln!(
                    output,
                    "outcome=translated blocks={} entries={} exits={}",
                    region.blocks.len(),
                    region.metadata.entries.len(),
                    region.metadata.exits.len()
                )
                .expect("writing to a String cannot fail");
                output.push_str(&print_ir_dump(
                    region,
                    IrDumpStage::PreOptimization,
                    IrPrintOptions::default(),
                ));
            }
            Self::Failed {
                start,
                reason,
                error,
            } => {
                writeln!(
                    output,
                    "start={} state={} {}",
                    start.pc, start.execution_state, start.profile_id
                )
                .expect("writing to a String cannot fail");
                writeln!(output, "outcome=failed reason={reason}")
                    .expect("writing to a String cannot fail");
                writeln!(output, "error={error}").expect("writing to a String cannot fail");
            }
        }
        output.push_str("end-report\n");
        output
    }
}

/// Translates one process-memory region while collecting optional source text.
///
/// Unlike [`super::translate_region`], this path constructs a disassembly string
/// for each recognized source instruction. Callers should invoke it only when a
/// diagnostic report or IR dump has been requested.
#[must_use]
pub fn translate_region_report(
    config: RegionTranslationConfig,
    profile: &GuestCpuProfile,
    address_space: AddressSpaceId,
    start: LocationDescriptor,
    memory: &(impl InstructionMemory + ?Sized),
) -> RegionTranslationReport {
    match translate_region_with_disassembly(
        config,
        crate::platform::PlatformDecoder::new(crate::platform::TargetPlatform::from_profile(
            *profile,
        )),
        profile,
        address_space,
        start,
        memory,
    ) {
        Ok(region) => RegionTranslationReport::Translated(region),
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
pub fn translate_raw_region_report(
    config: RegionTranslationConfig,
    profile: &GuestCpuProfile,
    base_pc: GuestVirtualAddress,
    execution_state: ExecutionState,
    bytes: &[u8],
) -> RegionTranslationReport {
    let start = LocationDescriptor::new(base_pc, execution_state, profile.id());
    let memory = match RawInstructionMemory::new(base_pc, bytes) {
        Ok(memory) => memory,
        Err(error) => return failure(start, error),
    };
    translate_region_report(
        config,
        profile,
        RAW_DIAGNOSTIC_ADDRESS_SPACE,
        start,
        &memory,
    )
}

fn failure(start: LocationDescriptor, error: FrontendError) -> RegionTranslationReport {
    let reason = if matches!(error, FrontendError::InstructionFetch(_)) {
        RegionTranslationFailureReason::FetchFault
    } else {
        RegionTranslationFailureReason::TranslationFailure
    };
    RegionTranslationReport::Failed {
        start,
        reason,
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
    fn content_mutation_epoch(&self) -> nixe_memory::ContentMutationEpoch {
        nixe_memory::ContentMutationEpoch::INITIAL
    }

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
        generation: ContentGeneration::new(1),
        mapping_generation: MappingGeneration::new(1),
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
        let report = translate_raw_region_report(
            RegionTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            &bytes,
        );

        let region = report.region().expect("raw region should translate");
        let block = region.entry_block();
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
        assert!(output.contains("outcome=translated blocks=1 entries=1 exits=1"));
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
        assert!(output.contains(
            "dependency page=0x0000000000000001 generation=0x0000000000000001 \
                 mapping-generation=1"
        ));
        assert!(!output.contains("0x7f"));

        let post = print_ir_dump(
            region,
            IrDumpStage::PostOptimization,
            IrPrintOptions::default(),
        );
        assert!(post.starts_with("ir-dump stage=post-optimization\n"));
    }

    #[test]
    fn raw_helper_covers_a32_t32_limits_and_page_dependencies() {
        let profile = GuestCpuProfile::switch_1();
        let a32 = translate_raw_region_report(
            RegionTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x2000),
            ExecutionState::A32,
            &0xeaff_ffff_u32.to_le_bytes(),
        );
        assert_eq!(
            a32.region().unwrap().entry_block().metadata.end_reason,
            BlockEndReason::DirectBranch
        );
        assert!(a32.print().contains("state=A32"));

        let t32 = translate_raw_region_report(
            RegionTranslationConfig {
                max_blocks: NonZeroU32::new(1).unwrap(),
                max_guest_instructions: NonZeroU32::new(1).unwrap(),
                max_guest_instructions_per_block: NonZeroU32::new(1).unwrap(),
                ..RegionTranslationConfig::default()
            },
            &profile,
            GuestVirtualAddress::new(0x2ffe),
            ExecutionState::T32,
            &[0xaf, 0xf3, 0x00, 0x80],
        );
        let region = t32
            .region()
            .expect("cross-page T32 instruction should translate");
        let block = region.entry_block();
        assert_eq!(
            block.metadata.end_reason,
            BlockEndReason::InstructionLimitAtPageBoundary
        );
        assert_eq!(region.metadata.code_dependencies.len(), 2);
        assert_eq!(block.metadata.sources[0].dependencies.iter().count(), 2);
    }

    #[test]
    fn fetch_faults_and_other_failures_have_distinct_compact_reports() {
        let profile = GuestCpuProfile::switch_1();
        let fetch = translate_raw_region_report(
            RegionTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            &[],
        );
        assert!(matches!(
            &fetch,
            RegionTranslationReport::Failed {
                reason: RegionTranslationFailureReason::FetchFault,
                ..
            }
        ));
        assert!(matches!(
            fetch.error(),
            Some(FrontendError::InstructionFetch(_))
        ));
        assert!(fetch.print().contains("outcome=failed reason=fetch-fault"));

        let invalid = translate_raw_region_report(
            RegionTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x1002),
            ExecutionState::A64,
            &0xd503_201f_u32.to_le_bytes(),
        );
        assert!(matches!(
            &invalid,
            RegionTranslationReport::Failed {
                reason: RegionTranslationFailureReason::TranslationFailure,
                ..
            }
        ));
        assert!(
            invalid
                .print()
                .contains("outcome=failed reason=translation-failure")
        );
    }

    #[test]
    fn architectural_decode_rejections_keep_source_context_in_reports() {
        let profile = GuestCpuProfile::switch_1();
        let report = translate_raw_region_report(
            RegionTranslationConfig::default(),
            &profile,
            GuestVirtualAddress::new(0x4000),
            ExecutionState::A64,
            &0_u32.to_le_bytes(),
        );
        let block = report
            .region()
            .expect("undefined instruction forms an exception region")
            .entry_block();
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
