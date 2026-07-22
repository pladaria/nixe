//! Generated frontend coverage and policy-controlled missing-instruction reports.

use core::fmt;
use std::collections::BTreeMap;

use crate::{
    decode::{self, DecodeSupport, InstructionPattern, table::EngineAvailability},
    location::{ExecutionState, InstructionEncoding, InstructionSize},
    profile::{CapabilityStatus, CpuProfileId, GuestCpuProfile, InstructionFeature},
};

/// Maximum local instruction context retained for one missing instruction.
pub const MAX_SURROUNDING_INSTRUCTION_BYTES: usize = 32;

/// Maximum number of unique missing-instruction records retained per tracker.
///
/// Counts for already-known records continue to saturate after this limit is
/// reached, while new records are dropped. This bounds process-local memory and
/// the size of deterministic diagnostic exports.
pub const MAX_MISSING_INSTRUCTION_RECORDS: usize = 4_096;

/// Conservative upper bound for either text export of one full tracker.
pub const MAX_MISSING_INSTRUCTION_EXPORT_BYTES: usize = 2 * 1024 * 1024;

/// Stable, explicitly assigned identity for one architectural instruction.
///
/// Values are grouped by execution state and must not be renumbered when table
/// entries move. They are suitable for counters, profiles, and test reports.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CoverageId(u32);

impl CoverageId {
    /// Creates an ID assigned by an instruction table.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the stable numeric value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for CoverageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "insn-{:08x}", self.0)
    }
}

/// Decoder availability of one declarative entry under a selected profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderCoverage {
    Available,
    RecognizedUnimplemented,
    ProfileDisabled {
        feature: InstructionFeature,
        status: CapabilityStatus,
    },
    ExecutionStateDisabled,
}

/// Independently tracked availability of an execution engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineCoverage {
    Implemented,
    EncodingDependent,
    Missing,
}

/// Evidence required before the generated table calls an entry lifted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionEvidence {
    pub decoder_classified: bool,
    pub interpreter_semantics: bool,
    pub explicit_exception: bool,
    pub ir_lowering: bool,
    pub printer_output: bool,
    pub regression_fixture: bool,
}

impl CompletionEvidence {
    #[must_use]
    pub const fn qualifies_as_lifted(self) -> bool {
        self.decoder_classified
            && (self.interpreter_semantics || self.explicit_exception)
            && self.ir_lowering
            && self.printer_output
            && self.regression_fixture
    }
}

/// Aggregate completion state used by coverage reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionCoverage {
    Lifted,
    InterpreterOnly,
    Incomplete,
    Unavailable,
}

/// One row generated from a declarative decoder table and engine registries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoverageEntry {
    pub profile_id: CpuProfileId,
    pub execution_state: ExecutionState,
    pub coverage_id: CoverageId,
    pub instruction_name: &'static str,
    pub decoder: DecoderCoverage,
    pub interpreter: EngineCoverage,
    pub lifter: EngineCoverage,
    pub evidence: CompletionEvidence,
    pub completion: CompletionCoverage,
}

/// Builds a deterministic coverage table for every decoder entry and state.
///
/// The table is generated from the implementation registration attached to
/// every declarative pattern, so adding a decoder rule cannot silently
/// disappear from coverage output or disagree with a parallel ID list.
#[must_use]
pub fn coverage_table(profile: &GuestCpuProfile) -> Vec<CoverageEntry> {
    let mut result = Vec::new();
    for patterns in all_pattern_tables() {
        for pattern in patterns {
            result.push(entry_for_pattern(profile, pattern));
        }
    }
    result.sort_by_key(|entry| entry.coverage_id);
    result
}

fn all_pattern_tables() -> [&'static [InstructionPattern]; 4] {
    [
        decode::a64::patterns(),
        decode::a32::patterns(),
        decode::t32::patterns_16(),
        decode::t32::patterns_32(),
    ]
}

fn entry_for_pattern(profile: &GuestCpuProfile, pattern: &InstructionPattern) -> CoverageEntry {
    let decoder = decoder_coverage(profile, pattern);
    let interpreter = engine_coverage(pattern.registration.interpreter);
    let lifter = engine_coverage(pattern.registration.lifter);
    let enabled = matches!(decoder, DecoderCoverage::Available);
    let evidence = CompletionEvidence {
        decoder_classified: enabled,
        interpreter_semantics: interpreter == EngineCoverage::Implemented,
        explicit_exception: false,
        ir_lowering: lifter == EngineCoverage::Implemented,
        printer_output: enabled,
        regression_fixture: enabled && pattern.registration.regression_fixture.is_some(),
    };
    let completion = if !enabled {
        CompletionCoverage::Unavailable
    } else if evidence.qualifies_as_lifted() {
        CompletionCoverage::Lifted
    } else if interpreter == EngineCoverage::Implemented && lifter == EngineCoverage::Missing {
        CompletionCoverage::InterpreterOnly
    } else {
        CompletionCoverage::Incomplete
    };
    CoverageEntry {
        profile_id: profile.id(),
        execution_state: pattern.execution_state,
        coverage_id: pattern.coverage_id,
        instruction_name: pattern.name,
        decoder,
        interpreter,
        lifter,
        evidence,
        completion,
    }
}

fn decoder_coverage(profile: &GuestCpuProfile, pattern: &InstructionPattern) -> DecoderCoverage {
    if !profile
        .allowed_execution_states()
        .contains(pattern.execution_state)
    {
        return DecoderCoverage::ExecutionStateDisabled;
    }
    for feature in pattern.required_features {
        let status = profile.instruction_features().status(*feature);
        if status != CapabilityStatus::Enabled {
            return DecoderCoverage::ProfileDisabled {
                feature: *feature,
                status,
            };
        }
    }
    match pattern.registration.decoder {
        DecodeSupport::Ready => DecoderCoverage::Available,
        DecodeSupport::RecognizedUnimplemented => DecoderCoverage::RecognizedUnimplemented,
    }
}

const fn engine_coverage(availability: EngineAvailability) -> EngineCoverage {
    match availability {
        EngineAvailability::Implemented => EngineCoverage::Implemented,
        EngineAvailability::EncodingDependent => EngineCoverage::EncodingDependent,
        EngineAvailability::Missing => EngineCoverage::Missing,
    }
}

pub(crate) fn interpreter_coverage(
    state: ExecutionState,
    coverage_id: CoverageId,
) -> EngineCoverage {
    registered_pattern(state, coverage_id).map_or(EngineCoverage::Missing, |pattern| {
        engine_coverage(pattern.registration.interpreter)
    })
}

pub(crate) fn lifter_coverage(state: ExecutionState, coverage_id: CoverageId) -> EngineCoverage {
    registered_pattern(state, coverage_id).map_or(EngineCoverage::Missing, |pattern| {
        engine_coverage(pattern.registration.lifter)
    })
}

fn registered_pattern(
    state: ExecutionState,
    coverage_id: CoverageId,
) -> Option<&'static InstructionPattern> {
    all_pattern_tables()
        .into_iter()
        .flatten()
        .find(|pattern| pattern.execution_state == state && pattern.coverage_id == coverage_id)
}

/// Runtime-owned opaque identity of the module containing an instruction.
///
/// The CPU frontend intentionally accepts no module path or title name. The
/// runtime assigns a numeric identity that is safe to place in debug reports.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ModuleIdentity(u64);

impl ModuleIdentity {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ModuleIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "module-{:016x}", self.0)
    }
}

/// Invalid local context supplied to the missing-instruction collector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionContextTooLarge {
    pub supplied: usize,
    pub maximum: usize,
}

impl fmt::Display for InstructionContextTooLarge {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "instruction context is {} bytes; maximum is {}",
            self.supplied, self.maximum
        )
    }
}

impl std::error::Error for InstructionContextTooLarge {}

/// One observed unsupported instruction before process-local deduplication.
pub struct MissingInstructionObservation {
    pub coverage_id: CoverageId,
    pub encoding: InstructionEncoding,
    pub pc: crate::address::GuestVirtualAddress,
    pub module: ModuleIdentity,
    pub execution_state: ExecutionState,
    surrounding_bytes: Box<[u8]>,
}

impl MissingInstructionObservation {
    pub fn new(
        coverage_id: CoverageId,
        encoding: InstructionEncoding,
        pc: crate::address::GuestVirtualAddress,
        module: ModuleIdentity,
        execution_state: ExecutionState,
        surrounding_bytes: impl Into<Box<[u8]>>,
    ) -> Result<Self, InstructionContextTooLarge> {
        let surrounding_bytes = surrounding_bytes.into();
        if surrounding_bytes.len() > MAX_SURROUNDING_INSTRUCTION_BYTES {
            return Err(InstructionContextTooLarge {
                supplied: surrounding_bytes.len(),
                maximum: MAX_SURROUNDING_INSTRUCTION_BYTES,
            });
        }
        Ok(Self {
            coverage_id,
            encoding,
            pc,
            module,
            execution_state,
            surrounding_bytes,
        })
    }

    /// Returns local-only diagnostic bytes. Sanitized exports never include them.
    #[must_use]
    pub fn surrounding_bytes(&self) -> &[u8] {
        &self.surrounding_bytes
    }
}

impl fmt::Debug for MissingInstructionObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MissingInstructionObservation")
            .field("coverage_id", &self.coverage_id)
            .field("encoding", &self.encoding)
            .field("pc", &self.pc)
            .field("module", &self.module)
            .field("execution_state", &self.execution_state)
            .field("surrounding_bytes", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MissingInstructionKey {
    coverage_id: CoverageId,
    encoding_bits: u32,
    encoding_size: InstructionSize,
}

impl MissingInstructionKey {
    const fn new(coverage_id: CoverageId, encoding: InstructionEncoding) -> Self {
        Self {
            coverage_id,
            encoding_bits: encoding.bits(),
            encoding_size: encoding.size(),
        }
    }
}

/// Deduplicated report entry retaining the first observed execution context.
pub struct MissingInstructionRecord {
    first: MissingInstructionObservation,
    occurrences: u64,
}

impl MissingInstructionRecord {
    #[must_use]
    pub const fn first_observation(&self) -> &MissingInstructionObservation {
        &self.first
    }

    #[must_use]
    pub const fn occurrences(&self) -> u64 {
        self.occurrences
    }

    /// Produces the minimal redistributable input expected in a regression test.
    #[must_use]
    pub const fn fixture(&self) -> MissingInstructionFixture {
        MissingInstructionFixture {
            coverage_id: self.first.coverage_id,
            encoding: self.first.encoding,
            execution_state: self.first.execution_state,
        }
    }
}

impl fmt::Debug for MissingInstructionRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MissingInstructionRecord")
            .field("first", &self.first)
            .field("occurrences", &self.occurrences)
            .finish()
    }
}

/// Minimal fixture to commit when implementing an instruction from a report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MissingInstructionFixture {
    pub coverage_id: CoverageId,
    pub encoding: InstructionEncoding,
    pub execution_state: ExecutionState,
}

/// Amount of missing-instruction context retained and exported by the CPU.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MissingInstructionReportDetail {
    /// Keep and export the bounded surrounding instruction window.
    #[default]
    Detailed,
    /// Discard the surrounding window and export only minimal identifiers.
    Sanitized,
}

/// Narrow diagnostic policy consumed by CPU frontend resources.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CpuDiagnosticsConfig {
    pub missing_instruction_reports: bool,
    pub report_detail: MissingInstructionReportDetail,
}

impl CpuDiagnosticsConfig {
    /// Returns whether the runtime should fetch surrounding bytes for reports.
    #[must_use]
    pub const fn captures_surrounding_instruction_bytes(self) -> bool {
        self.missing_instruction_reports
            && matches!(self.report_detail, MissingInstructionReportDetail::Detailed)
    }
}

impl Default for CpuDiagnosticsConfig {
    fn default() -> Self {
        Self {
            missing_instruction_reports: true,
            report_detail: MissingInstructionReportDetail::Detailed,
        }
    }
}

/// Process- or title-local missing-instruction counts.
///
/// Callers create one tracker per isolation scope. Keys are stable coverage IDs
/// plus exact raw encodings; repeated observations increment frequency without
/// replacing the first actionable context.
pub struct MissingInstructionTracker {
    config: CpuDiagnosticsConfig,
    records: BTreeMap<MissingInstructionKey, MissingInstructionRecord>,
    total_observations: u64,
}

impl MissingInstructionTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a tracker using the runtime-derived CPU diagnostics policy.
    #[must_use]
    pub fn with_config(config: CpuDiagnosticsConfig) -> Self {
        Self {
            config,
            records: BTreeMap::new(),
            total_observations: 0,
        }
    }

    #[must_use]
    pub const fn config(&self) -> CpuDiagnosticsConfig {
        self.config
    }

    /// Records an observation and returns whether it was unique in this scope.
    pub fn record(&mut self, mut observation: MissingInstructionObservation) -> bool {
        if !self.config.missing_instruction_reports {
            return false;
        }
        if self.config.report_detail == MissingInstructionReportDetail::Sanitized {
            observation.surrounding_bytes = Box::new([]);
        }
        self.total_observations = self.total_observations.saturating_add(1);
        let key = MissingInstructionKey::new(observation.coverage_id, observation.encoding);
        if !self.records.contains_key(&key) && self.records.len() >= MAX_MISSING_INSTRUCTION_RECORDS
        {
            return false;
        }
        match self.records.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(MissingInstructionRecord {
                    first: observation,
                    occurrences: 1,
                });
                true
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let occurrences = entry.get().occurrences.saturating_add(1);
                entry.get_mut().occurrences = occurrences;
                false
            }
        }
    }

    /// Records a frontend unsupported-instruction exit using bounded local
    /// context supplied by the runtime. Other terminator kinds are ignored.
    pub fn record_terminator(
        &mut self,
        terminator: &crate::ir::terminator::Terminator,
        module: ModuleIdentity,
        surrounding_bytes: impl Into<Box<[u8]>>,
    ) -> Result<Option<bool>, InstructionContextTooLarge> {
        if !self.config.missing_instruction_reports {
            return Ok(None);
        }
        let crate::ir::terminator::Terminator::UnsupportedInstruction {
            source,
            encoding,
            coverage_id,
            ..
        } = terminator
        else {
            return Ok(None);
        };
        let observation = MissingInstructionObservation::new(
            CoverageId::new(*coverage_id),
            *encoding,
            source.pc,
            module,
            source.execution_state,
            surrounding_bytes,
        )?;
        Ok(Some(self.record(observation)))
    }

    #[must_use]
    pub fn unique_instructions(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub const fn total_observations(&self) -> u64 {
        self.total_observations
    }

    pub fn records(&self) -> impl ExactSizeIterator<Item = &MissingInstructionRecord> {
        self.records.values()
    }

    /// Exports deterministic debug text with no code window, title data, paths,
    /// host pointers, or caller-provided strings.
    #[must_use]
    pub fn export_sanitized(&self) -> String {
        use fmt::Write;

        let mut output = String::from("swiitx-missing-instructions-v1\n");
        writeln!(
            output,
            "unique={} observations={}",
            self.unique_instructions(),
            self.total_observations
        )
        .expect("writing to a String cannot fail");
        for record in self.records.values() {
            let first = &record.first;
            writeln!(
                output,
                "coverage={} encoding={} state={} pc={} module={} occurrences={}",
                first.coverage_id,
                first.encoding,
                first.execution_state,
                first.pc,
                first.module,
                record.occurrences
            )
            .expect("writing to a String cannot fail");
        }
        output
    }

    /// Exports local diagnostics including the bounded surrounding byte window.
    #[must_use]
    pub fn export_detailed(&self) -> String {
        use fmt::Write;

        let mut output = String::from("swiitx-missing-instructions-detailed-v1\n");
        writeln!(
            output,
            "unique={} observations={}",
            self.unique_instructions(),
            self.total_observations
        )
        .expect("writing to a String cannot fail");
        for record in self.records.values() {
            let first = &record.first;
            write!(
                output,
                "coverage={} encoding={} state={} pc={} module={} occurrences={} context=",
                first.coverage_id,
                first.encoding,
                first.execution_state,
                first.pc,
                first.module,
                record.occurrences
            )
            .expect("writing to a String cannot fail");
            for byte in first.surrounding_bytes() {
                write!(output, "{byte:02x}").expect("writing to a String cannot fail");
            }
            output.push('\n');
        }
        output
    }

    /// Exports using the detail level selected by the runtime policy.
    #[must_use]
    pub fn export(&self) -> String {
        match self.config.report_detail {
            MissingInstructionReportDetail::Detailed => self.export_detailed(),
            MissingInstructionReportDetail::Sanitized => self.export_sanitized(),
        }
    }
}

impl Default for MissingInstructionTracker {
    fn default() -> Self {
        Self::with_config(CpuDiagnosticsConfig::default())
    }
}

impl fmt::Debug for MissingInstructionTracker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MissingInstructionTracker")
            .field("config", &self.config)
            .field("unique_instructions", &self.unique_instructions())
            .field("total_observations", &self.total_observations)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress},
        ir::{print::print_block, terminator::Terminator},
        memory::{MemoryPermissions, SYNTHETIC_PAGE_SIZE, SyntheticMemory},
        translate::{BlockTranslationConfig, translate_block},
    };

    #[test]
    fn generated_table_tracks_profile_and_independent_engines() {
        let switch_1 = coverage_table(&GuestCpuProfile::switch_1());
        let branch = entry(&switch_1, CoverageId::new(0x0000_0002));
        assert_eq!(branch.decoder, DecoderCoverage::Available);
        assert_eq!(branch.interpreter, EngineCoverage::Implemented);
        assert_eq!(branch.lifter, EngineCoverage::Implemented);
        assert_eq!(branch.completion, CompletionCoverage::Lifted);

        let integer = entry(&switch_1, CoverageId::new(0x0000_0003));
        assert_eq!(integer.interpreter, EngineCoverage::Implemented);
        assert_eq!(integer.lifter, EngineCoverage::Implemented);
        assert_eq!(integer.completion, CompletionCoverage::Incomplete);

        let simd = entry(&switch_1, CoverageId::new(0x0000_0030));
        assert_eq!(simd.decoder, DecoderCoverage::Available);

        let switch_2 = coverage_table(&GuestCpuProfile::switch_2_native());
        let simd = entry(&switch_2, CoverageId::new(0x0000_0030));
        assert!(matches!(
            simd.decoder,
            DecoderCoverage::ProfileDisabled {
                feature: InstructionFeature::AdvancedSimd,
                status: CapabilityStatus::Unknown
            }
        ));

        assert_eq!(
            entry(&switch_2, CoverageId::new(0x0001_0001)).decoder,
            DecoderCoverage::ExecutionStateDisabled
        );
    }

    #[test]
    fn every_registry_entry_routes_through_decode_normalization_and_disassembly() {
        let profile = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::AdvancedSimd, CapabilityStatus::Enabled);
        let table = coverage_table(&profile);
        let expected_entries: usize = all_pattern_tables()
            .iter()
            .map(|patterns| patterns.len())
            .sum();
        assert_eq!(table.len(), expected_entries);

        for pattern in all_pattern_tables().into_iter().flatten() {
            assert_eq!(
                pattern.registration,
                crate::decode::registry::registration(
                    pattern.execution_state,
                    pattern.coverage_id.get()
                )
            );
            let decoded = find_registered_encoding(&profile, pattern).unwrap_or_else(|| {
                panic!(
                    "registry entry {} {} has no accepted encoding",
                    pattern.execution_state, pattern.coverage_id
                )
            });
            let text = decode::disassemble(&decoded.instruction).to_string();
            assert!(text.starts_with(pattern.name));
            match pattern.execution_state {
                ExecutionState::A64 => {
                    let _ = decode::a64::normalize(&decoded.instruction, decoded.encoding);
                }
                ExecutionState::A32 => {
                    let _ = decode::a32::normalize(&decoded.instruction, decoded.encoding);
                }
                ExecutionState::T32 => {
                    let _ = decode::t32::normalize(&decoded.instruction, decoded.encoding);
                }
            }

            let coverage = entry(&table, pattern.coverage_id);
            assert_eq!(
                coverage.interpreter,
                engine_coverage(pattern.registration.interpreter)
            );
            assert_eq!(
                coverage.lifter,
                engine_coverage(pattern.registration.lifter)
            );
            assert_eq!(
                coverage.evidence.regression_fixture,
                pattern.registration.regression_fixture.is_some()
                    && matches!(coverage.decoder, DecoderCoverage::Available)
            );

            let block = translate_registered_encoding(&profile, &decoded);
            match pattern.registration.lifter {
                EngineAvailability::Implemented => assert!(
                    !matches!(
                        block.terminator,
                        Terminator::InterpretOne { .. } | Terminator::UnsupportedInstruction { .. }
                    ),
                    "{} {} declares IR lowering but routed to {:?}",
                    pattern.execution_state,
                    pattern.coverage_id,
                    block.terminator
                ),
                EngineAvailability::Missing => assert!(
                    matches!(
                        block.terminator,
                        Terminator::InterpretOne { .. } | Terminator::UnsupportedInstruction { .. }
                    ),
                    "{} {} declares missing IR lowering but lowered to {:?}",
                    pattern.execution_state,
                    pattern.coverage_id,
                    block.terminator
                ),
                EngineAvailability::EncodingDependent => {}
            }
        }
    }

    fn translate_registered_encoding(
        profile: &GuestCpuProfile,
        decoded: &crate::location::DecodedInstruction<crate::decode::DecodedOpcode>,
    ) -> crate::ir::block::IrBlock {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(1)));
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x1000),
            GuestPhysicalPageId::new(1),
            MemoryPermissions::READ_EXECUTE,
        ));
        let bytes = match (decoded.location.execution_state, decoded.encoding.size()) {
            (ExecutionState::T32, InstructionSize::Bits32) => {
                let bits = decoded.encoding.bits();
                [(bits >> 16) as u16, bits as u16]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>()
            }
            (_, InstructionSize::Bits16) => decoded.encoding.bits().to_le_bytes()[..2].to_vec(),
            (_, InstructionSize::Bits32) => decoded.encoding.bits().to_le_bytes().to_vec(),
        };
        assert!(memory.initialize_ram(GuestPhysicalPageId::new(1), 0, &bytes));
        translate_block(
            BlockTranslationConfig {
                max_guest_instructions: core::num::NonZeroU32::new(1).unwrap(),
            },
            profile,
            AddressSpaceId::new(1),
            decoded.location,
            &memory,
        )
        .unwrap()
    }

    fn find_registered_encoding(
        profile: &GuestCpuProfile,
        pattern: &'static InstructionPattern,
    ) -> Option<crate::location::DecodedInstruction<crate::decode::DecodedOpcode>> {
        let width_mask = match pattern.size {
            InstructionSize::Bits16 => 0xffff,
            InstructionSize::Bits32 => u32::MAX,
        };
        let variable_mask = !pattern.mask & width_mask;
        let mut sample = 0_u32;
        for attempt in 0..65_536_u32 {
            if attempt != 0 {
                sample = sample.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            }
            let bits = pattern.value | (sample & variable_mask);
            let encoding = match pattern.size {
                InstructionSize::Bits16 => InstructionEncoding::from_u16(bits as u16),
                InstructionSize::Bits32 => InstructionEncoding::from_u32(bits),
            };
            let location = crate::location::LocationDescriptor::new(
                GuestVirtualAddress::new(0x1000),
                pattern.execution_state,
                profile.id(),
            );
            let decoded = match decode::decode(profile, location, encoding) {
                decode::DecodeResult::Decoded(decoded)
                | decode::DecodeResult::RecognizedUnimplemented(decoded) => decoded,
                _ => continue,
            };
            if decoded.instruction.coverage_id() == pattern.coverage_id {
                return Some(decoded);
            }
        }
        None
    }

    #[test]
    fn every_lifted_completion_fixture_decodes_lowers_and_prints() {
        let profile = GuestCpuProfile::switch_1();
        let table = coverage_table(&profile);
        for pattern in all_pattern_tables()
            .into_iter()
            .flatten()
            .filter(|pattern| pattern.registration.regression_fixture.is_some())
        {
            let fixture = pattern.registration.regression_fixture.unwrap();
            let decoded = match decode::decode(
                &profile,
                crate::location::LocationDescriptor::new(
                    GuestVirtualAddress::new(0x1000),
                    pattern.execution_state,
                    profile.id(),
                ),
                fixture.encoding,
            ) {
                decode::DecodeResult::Decoded(decoded) => decoded,
                other => panic!("completion fixture did not decode: {other:?}"),
            };
            assert_eq!(decoded.instruction.coverage_id(), pattern.coverage_id);

            let mut memory = SyntheticMemory::new();
            assert!(memory.add_ram_page(GuestPhysicalPageId::new(1)));
            assert!(memory.map_page(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x1000),
                GuestPhysicalPageId::new(1),
                MemoryPermissions::READ_EXECUTE,
            ));
            let bytes = match (pattern.execution_state, fixture.encoding.size()) {
                (ExecutionState::T32, InstructionSize::Bits32) => {
                    let bits = fixture.encoding.bits();
                    let first = (bits >> 16) as u16;
                    let second = bits as u16;
                    [first.to_le_bytes(), second.to_le_bytes()].concat()
                }
                (_, InstructionSize::Bits16) => fixture.encoding.bits().to_le_bytes()[..2].to_vec(),
                (_, InstructionSize::Bits32) => fixture.encoding.bits().to_le_bytes().to_vec(),
            };
            assert!(memory.initialize_ram(GuestPhysicalPageId::new(1), 0, &bytes));
            assert_eq!(SYNTHETIC_PAGE_SIZE, 4096);
            let block = translate_block(
                BlockTranslationConfig {
                    max_guest_instructions: core::num::NonZeroU32::new(1).unwrap(),
                },
                &profile,
                AddressSpaceId::new(1),
                decoded.location,
                &memory,
            )
            .unwrap();
            assert!(
                !matches!(
                    block.terminator,
                    Terminator::InterpretOne { .. } | Terminator::UnsupportedInstruction { .. }
                ),
                "completion fixture {:?} {} lowered to {:?}",
                pattern.execution_state,
                pattern.coverage_id,
                block.terminator
            );
            let printed = print_block(&block, Default::default());
            assert!(printed.contains("source pc=0x0000000000001000 state="));
            assert!(printed.contains(" ; raw="));
            assert!(printed.contains("terminator "));
        }

        for entry in table
            .iter()
            .filter(|entry| entry.completion == CompletionCoverage::Lifted)
        {
            assert!(
                registered_pattern(entry.execution_state, entry.coverage_id)
                    .unwrap()
                    .registration
                    .regression_fixture
                    .is_some()
            );
        }
    }

    #[test]
    fn tracker_deduplicates_frequency_and_preserves_first_context() {
        let mut tracker = MissingInstructionTracker::new();
        let first = observation(0x1000, &[0xaa, 0xbb, 0xcc]);
        let fixture = MissingInstructionFixture {
            coverage_id: first.coverage_id,
            encoding: first.encoding,
            execution_state: first.execution_state,
        };
        assert!(tracker.record(first));
        assert!(!tracker.record(observation(0x2000, &[0x11, 0x22])));
        assert_eq!(tracker.unique_instructions(), 1);
        assert_eq!(tracker.total_observations(), 2);
        let record = tracker.records().next().unwrap();
        assert_eq!(record.occurrences(), 2);
        assert_eq!(
            record.first_observation().pc,
            GuestVirtualAddress::new(0x1000)
        );
        assert_eq!(
            record.first_observation().surrounding_bytes(),
            &[0xaa, 0xbb, 0xcc]
        );
        assert_eq!(record.fixture(), fixture);
    }

    #[test]
    fn sanitized_export_and_debug_never_include_context_bytes() {
        let mut tracker = MissingInstructionTracker::with_config(CpuDiagnosticsConfig {
            missing_instruction_reports: true,
            report_detail: MissingInstructionReportDetail::Sanitized,
        });
        tracker.record(observation(0x1000, &[0xde, 0xad, 0xbe, 0xef]));
        let export = tracker.export();
        assert!(export.contains("coverage=insn-00000038"));
        assert!(export.contains("occurrences=1"));
        assert!(!export.contains("deadbeef"));
        assert!(
            tracker
                .records()
                .next()
                .unwrap()
                .first_observation()
                .surrounding_bytes()
                .is_empty()
        );
        assert!(!format!("{tracker:?}").contains("deadbeef"));
    }

    #[test]
    fn detailed_reports_are_default_and_include_bounded_context() {
        let mut tracker = MissingInstructionTracker::new();
        assert_eq!(
            tracker.config().report_detail,
            MissingInstructionReportDetail::Detailed
        );
        assert!(tracker.config().captures_surrounding_instruction_bytes());
        tracker.record(observation(0x1000, &[0xde, 0xad, 0xbe, 0xef]));
        let export = tracker.export();
        assert!(export.starts_with("swiitx-missing-instructions-detailed-v1"));
        assert!(export.contains("context=deadbeef"));
    }

    #[test]
    fn disabled_reports_do_not_retain_observations() {
        let mut tracker = MissingInstructionTracker::with_config(CpuDiagnosticsConfig {
            missing_instruction_reports: false,
            report_detail: MissingInstructionReportDetail::Detailed,
        });
        assert!(!tracker.record(observation(0x1000, &[1, 2, 3, 4])));
        assert_eq!(tracker.unique_instructions(), 0);
        assert_eq!(tracker.total_observations(), 0);
        assert!(!tracker.config().captures_surrounding_instruction_bytes());
    }

    #[test]
    fn unsupported_terminator_flows_directly_into_the_tracker() {
        let source = crate::location::LocationDescriptor::new(
            GuestVirtualAddress::new(0x4000),
            ExecutionState::A64,
            GuestCpuProfile::switch_1().id(),
        );
        let terminator = Terminator::UnsupportedInstruction {
            source,
            encoding: InstructionEncoding::from_u32(0x0e20_1c00),
            coverage_id: 0x0000_0038,
            disassembly: "advanced-simd-fallback".into(),
            reason: "missing semantics".into(),
        };
        let mut tracker = MissingInstructionTracker::new();
        assert_eq!(
            tracker
                .record_terminator(&terminator, ModuleIdentity::new(9), &[1, 2, 3, 4][..])
                .unwrap(),
            Some(true)
        );
        let record = tracker.records().next().unwrap();
        assert_eq!(record.first_observation().pc, source.pc);
        assert_eq!(record.first_observation().module, ModuleIdentity::new(9));
        assert_eq!(
            record.first_observation().surrounding_bytes(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn local_context_is_strictly_bounded() {
        let result = MissingInstructionObservation::new(
            CoverageId::new(1),
            InstructionEncoding::from_u32(0),
            GuestVirtualAddress::new(0),
            ModuleIdentity::new(0),
            ExecutionState::A64,
            vec![0; MAX_SURROUNDING_INSTRUCTION_BYTES + 1],
        );
        assert_eq!(
            result.unwrap_err(),
            InstructionContextTooLarge {
                supplied: MAX_SURROUNDING_INSTRUCTION_BYTES + 1,
                maximum: MAX_SURROUNDING_INSTRUCTION_BYTES,
            }
        );
    }

    #[test]
    fn tracker_and_exports_have_hard_resource_bounds() {
        let mut tracker = MissingInstructionTracker::new();
        for index in 0..=MAX_MISSING_INSTRUCTION_RECORDS {
            let recorded = tracker.record(
                MissingInstructionObservation::new(
                    CoverageId::new(index as u32),
                    InstructionEncoding::from_u32(index as u32),
                    GuestVirtualAddress::new(index as u64 * 4),
                    ModuleIdentity::new(index as u64),
                    ExecutionState::A64,
                    [0xaa; MAX_SURROUNDING_INSTRUCTION_BYTES],
                )
                .unwrap(),
            );
            assert_eq!(recorded, index < MAX_MISSING_INSTRUCTION_RECORDS);
        }

        assert_eq!(
            tracker.unique_instructions(),
            MAX_MISSING_INSTRUCTION_RECORDS
        );
        assert!(tracker.export_sanitized().len() <= MAX_MISSING_INSTRUCTION_EXPORT_BYTES);
        assert!(tracker.export_detailed().len() <= MAX_MISSING_INSTRUCTION_EXPORT_BYTES);
    }

    fn entry(table: &[CoverageEntry], id: CoverageId) -> &CoverageEntry {
        table
            .iter()
            .find(|entry| entry.coverage_id == id)
            .expect("coverage entry")
    }

    fn observation(pc: u64, context: &[u8]) -> MissingInstructionObservation {
        MissingInstructionObservation::new(
            CoverageId::new(0x0000_0038),
            InstructionEncoding::from_u32(0x0e20_1c00),
            GuestVirtualAddress::new(pc),
            ModuleIdentity::new(7),
            ExecutionState::A64,
            context,
        )
        .unwrap()
    }
}
