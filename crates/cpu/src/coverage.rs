//! Generated frontend coverage and policy-controlled missing-instruction reports.

use core::fmt;
use std::collections::BTreeMap;

use crate::{
    decode::{self, DecodeSupport, InstructionPattern, table::LoweringAvailability},
    location::{ExecutionState, InstructionEncoding, InstructionSize},
    profile::{CpuProfileId, GuestCpuProfile, InstructionFeature},
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
    UnavailableOnPlatform { feature: InstructionFeature },
    ExecutionStateDisabled,
}

/// Independently tracked availability of the shared IR lowerer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoweringCoverage {
    Implemented,
    Missing,
}

/// Evidence required before the generated table calls an entry lifted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompletionEvidence {
    pub decoder_classified: bool,
    pub explicit_exception: bool,
    pub ir_lowering: bool,
    pub printer_output: bool,
    pub regression_fixture: bool,
}

impl CompletionEvidence {
    #[must_use]
    pub const fn qualifies_as_lifted(self) -> bool {
        self.decoder_classified
            && (self.ir_lowering || self.explicit_exception)
            && self.printer_output
            && self.regression_fixture
    }
}

/// Aggregate completion state used by coverage reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionCoverage {
    Lifted,
    Incomplete,
    Unavailable,
}

/// One row generated from the declarative instruction catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoverageEntry {
    pub profile_id: CpuProfileId,
    pub execution_state: ExecutionState,
    pub coverage_id: CoverageId,
    pub instruction_name: &'static str,
    pub decoder: DecoderCoverage,
    pub lifter: LoweringCoverage,
    pub evidence: CompletionEvidence,
    pub completion: CompletionCoverage,
}

/// Builds a deterministic coverage table for every decoder entry and state.
///
/// Decoder, lowering, and fixture status come directly from each pattern.
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
    let lifter = lowering_coverage(pattern.lowering);
    let enabled = matches!(decoder, DecoderCoverage::Available);
    let evidence = CompletionEvidence {
        decoder_classified: enabled,
        explicit_exception: false,
        ir_lowering: lifter == LoweringCoverage::Implemented,
        printer_output: enabled,
        regression_fixture: enabled && pattern.regression_fixture.is_some(),
    };
    let completion = if !enabled {
        CompletionCoverage::Unavailable
    } else if evidence.qualifies_as_lifted() {
        CompletionCoverage::Lifted
    } else {
        CompletionCoverage::Incomplete
    };
    CoverageEntry {
        profile_id: profile.id(),
        execution_state: pattern.execution_state,
        coverage_id: pattern.coverage_id,
        instruction_name: pattern.name,
        decoder,
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
    let platform = crate::platform::TargetPlatform::from_profile(*profile);
    for feature in pattern.required_features {
        if !platform.supports(*feature) {
            return DecoderCoverage::UnavailableOnPlatform { feature: *feature };
        }
    }
    match pattern.decoder {
        DecodeSupport::Ready => DecoderCoverage::Available,
        DecodeSupport::RecognizedUnimplemented => DecoderCoverage::RecognizedUnimplemented,
    }
}

const fn lowering_coverage(availability: LoweringAvailability) -> LoweringCoverage {
    match availability {
        LoweringAvailability::Implemented => LoweringCoverage::Implemented,
        LoweringAvailability::Missing => LoweringCoverage::Missing,
    }
}

#[cfg(test)]
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
    pub pc: nixe_memory::GuestVirtualAddress,
    pub module: ModuleIdentity,
    pub execution_state: ExecutionState,
    surrounding_bytes: Box<[u8]>,
}

impl MissingInstructionObservation {
    pub fn new(
        coverage_id: CoverageId,
        encoding: InstructionEncoding,
        pc: nixe_memory::GuestVirtualAddress,
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

/// Process- or title-local missing-instruction counts.
///
/// Callers create one tracker per isolation scope. Keys are stable coverage IDs
/// plus exact raw encodings; repeated observations increment frequency without
/// replacing the first actionable context.
#[derive(Default)]
pub struct MissingInstructionTracker {
    records: BTreeMap<MissingInstructionKey, MissingInstructionRecord>,
    total_observations: u64,
}

impl MissingInstructionTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an observation and returns whether it was unique in this scope.
    pub fn record(&mut self, observation: MissingInstructionObservation) -> bool {
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
    ) -> Result<bool, InstructionContextTooLarge> {
        let crate::ir::terminator::Terminator::UnsupportedInstruction {
            source,
            encoding,
            coverage_id,
            ..
        } = terminator
        else {
            return Ok(false);
        };
        let observation = MissingInstructionObservation::new(
            CoverageId::new(*coverage_id),
            *encoding,
            source.pc,
            module,
            source.execution_state,
            surrounding_bytes,
        )?;
        Ok(self.record(observation))
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

    /// Exports diagnostics including the bounded surrounding byte window.
    #[must_use]
    pub fn export(&self) -> String {
        use fmt::Write;

        let mut output = String::from("nixe-missing-instructions-v2\n");
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
}

impl fmt::Debug for MissingInstructionTracker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MissingInstructionTracker")
            .field("unique_instructions", &self.unique_instructions())
            .field("total_observations", &self.total_observations)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress};

    use crate::{
        ir::{print::print_region, terminator::Terminator},
        memory::{MemoryPermissions, SYNTHETIC_PAGE_SIZE, SyntheticMemory},
        translate::{RegionTranslationConfig, translate_region},
    };

    #[test]
    fn generated_table_tracks_profile_and_frontend_lowering() {
        let switch_1 = coverage_table(&GuestCpuProfile::switch_1());
        let branch = entry(&switch_1, CoverageId::new(0x0000_0002));
        assert_eq!(branch.decoder, DecoderCoverage::Available);
        assert_eq!(branch.lifter, LoweringCoverage::Implemented);
        assert_eq!(branch.completion, CompletionCoverage::Lifted);

        let integer = entry(&switch_1, CoverageId::new(0x0000_0003));
        assert_eq!(integer.lifter, LoweringCoverage::Implemented);
        assert_eq!(integer.completion, CompletionCoverage::Lifted);

        let simd = entry(&switch_1, CoverageId::new(0x0000_0030));
        assert_eq!(simd.decoder, DecoderCoverage::Available);

        let switch_2 = coverage_table(&GuestCpuProfile::switch_2_native());
        let simd = entry(&switch_2, CoverageId::new(0x0000_0030));
        assert!(matches!(
            simd.decoder,
            DecoderCoverage::UnavailableOnPlatform {
                feature: InstructionFeature::AdvancedSimd,
            }
        ));

        assert_eq!(
            entry(&switch_2, CoverageId::new(0x0001_0001)).decoder,
            DecoderCoverage::ExecutionStateDisabled
        );

        for entry in switch_1
            .iter()
            .filter(|entry| entry.decoder == DecoderCoverage::Available)
        {
            assert_eq!(
                entry.completion,
                CompletionCoverage::Lifted,
                "{} {} is available to the Switch 1 interpreter but lacks complete frontend evidence",
                entry.execution_state,
                entry.coverage_id
            );
        }
    }

    #[test]
    fn every_switch_1_entry_routes_through_decode_normalization_and_disassembly() {
        let profile = GuestCpuProfile::switch_1();
        let platform = crate::platform::TargetPlatform::Switch1;
        let table = coverage_table(&profile);
        let expected_entries: usize = all_pattern_tables()
            .iter()
            .map(|patterns| patterns.len())
            .sum();
        assert_eq!(table.len(), expected_entries);

        for pattern in all_pattern_tables()
            .into_iter()
            .flatten()
            .filter(|pattern| {
                pattern
                    .required_features
                    .iter()
                    .all(|feature| platform.supports(*feature))
            })
        {
            let decoded = find_registered_encoding(&profile, pattern).unwrap_or_else(|| {
                panic!(
                    "catalog entry {} {} has no accepted encoding",
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
            assert_eq!(coverage.lifter, lowering_coverage(pattern.lowering));
            assert_eq!(
                coverage.evidence.regression_fixture,
                pattern.regression_fixture.is_some()
                    && matches!(coverage.decoder, DecoderCoverage::Available)
            );
        }
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
            .filter(|pattern| {
                pattern.regression_fixture.is_some()
                    && pattern.decoder == DecodeSupport::Ready
                    && matches!(
                        decoder_coverage(&profile, pattern),
                        DecoderCoverage::Available
                    )
            })
        {
            let fixture = pattern.regression_fixture.unwrap();
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
                other => panic!(
                    "completion fixture for {} {} did not decode: {other:?}",
                    pattern.execution_state, pattern.coverage_id
                ),
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
            let region = translate_region(
                RegionTranslationConfig {
                    max_blocks: core::num::NonZeroU32::new(1).unwrap(),
                    max_guest_instructions: core::num::NonZeroU32::new(1).unwrap(),
                    max_guest_instructions_per_block: core::num::NonZeroU32::new(1).unwrap(),
                    ..RegionTranslationConfig::default()
                },
                &profile,
                AddressSpaceId::new(1),
                decoded.location,
                &memory,
            )
            .unwrap();
            assert!(
                !matches!(
                    region.entry_block().terminator,
                    Terminator::UnsupportedInstruction { .. }
                ),
                "completion fixture {:?} {} lowered to {:?}",
                pattern.execution_state,
                pattern.coverage_id,
                region.entry_block().terminator
            );
            let printed = print_region(&region, Default::default());
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
    fn reports_include_bounded_context() {
        let mut tracker = MissingInstructionTracker::new();
        tracker.record(observation(0x1000, &[0xde, 0xad, 0xbe, 0xef]));
        let export = tracker.export();
        assert!(export.starts_with("nixe-missing-instructions-v2"));
        assert!(export.contains("context=deadbeef"));
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
            disassembly: "advanced-simd-unsupported".into(),
            reason: "missing semantics".into(),
        };
        let mut tracker = MissingInstructionTracker::new();
        assert!(
            tracker
                .record_terminator(&terminator, ModuleIdentity::new(9), &[1, 2, 3, 4][..])
                .unwrap()
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
        assert!(tracker.export().len() <= MAX_MISSING_INSTRUCTION_EXPORT_BYTES);
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
