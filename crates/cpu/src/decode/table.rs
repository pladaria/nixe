//! Declarative instruction patterns and compiled indexed decode tables.

use core::fmt;

use crate::{
    coverage::CoverageId,
    error::InstructionDiagnostic,
    location::{DecodedInstruction, InstructionEncoding, LocationDescriptor},
};

use super::DecodeResult;

const BUCKET_COUNT: usize = 256;
const MAX_OPERANDS: usize = 8;

/// Whether downstream semantics currently exist for a recognized entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecodeSupport {
    Ready,
    RecognizedUnimplemented,
}

/// One redistributable encoding which exercises a registered decoder entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RegressionFixture {
    pub encoding: InstructionEncoding,
}

/// Result of applying instruction-specific architectural allocation rules.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AllocationStatus {
    Allocated,
    Reserved(&'static str),
    Unallocated(&'static str),
}

/// Architectural register namespace used by an extracted operand.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegisterClass {
    A64General,
}

/// Stable role of an operand within an instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum OperandId {
    Destination,
    FirstSource,
    SecondSource,
    Immediate,
    Condition,
}

impl OperandId {
    const fn name(self) -> &'static str {
        match self {
            Self::Destination => "dst",
            Self::FirstSource => "src1",
            Self::SecondSource => "src2",
            Self::Immediate => "imm",
            Self::Condition => "cond",
        }
    }
}

/// Typed interpretation applied to a contiguous encoded field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OperandKind {
    Unsigned,
    SignedScaled { scale: u8 },
    Register(RegisterClass),
}

/// Declarative operand field.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OperandField {
    pub id: OperandId,
    pub lsb: u8,
    pub width: u8,
    pub kind: OperandKind,
}

/// One declarative instruction entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionPattern {
    pub name: &'static str,
    pub mask: u32,
    pub value: u32,
    pub operands: &'static [OperandField],
    pub coverage_id: CoverageId,
    /// Higher values win intentional overlaps; equal values remain an error.
    pub priority: u16,
    pub decoder: DecodeSupport,
    pub regression_fixture: Option<RegressionFixture>,
}

impl InstructionPattern {
    #[must_use]
    pub const fn fixture32(mut self, bits: u32) -> Self {
        self.regression_fixture = Some(RegressionFixture {
            encoding: InstructionEncoding::from_u32(bits),
        });
        self
    }

    #[must_use]
    pub const fn recognized_unimplemented(mut self) -> Self {
        self.decoder = DecodeSupport::RecognizedUnimplemented;
        self
    }
}

/// A typed value extracted from an instruction encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OperandValue {
    Unsigned(u64),
    Signed(i64),
    Register { class: RegisterClass, index: u8 },
}

/// Fixed-capacity operand collection; decoding performs no heap allocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DecodedOperands {
    values: [Option<(OperandId, OperandValue)>; MAX_OPERANDS],
    len: u8,
}

impl DecodedOperands {
    const EMPTY: Self = Self {
        values: [None; MAX_OPERANDS],
        len: 0,
    };

    #[must_use]
    pub const fn len(self) -> usize {
        self.len as usize
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn get(self, id: OperandId) -> Option<OperandValue> {
        self.values[..self.len()]
            .iter()
            .find_map(|entry| match entry {
                Some((present, value)) if *present == id => Some(*value),
                _ => None,
            })
    }

    pub fn iter(self) -> impl Iterator<Item = (OperandId, OperandValue)> {
        self.values.into_iter().take(self.len()).flatten()
    }
}

/// Pattern identity and raw bits used to extract typed operands on demand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedOpcode {
    pattern: &'static InstructionPattern,
    bits: u32,
}

impl DecodedOpcode {
    #[must_use]
    pub const fn pattern(&self) -> &'static InstructionPattern {
        self.pattern
    }

    #[must_use]
    pub const fn operands(&self) -> DecodedOperands {
        extract_operands(self.pattern, self.bits)
    }

    #[must_use]
    pub const fn coverage_id(&self) -> CoverageId {
        self.pattern.coverage_id
    }

    pub(crate) fn fmt_disassembly(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.pattern.name)?;
        for (index, (id, value)) in self.operands().iter().enumerate() {
            f.write_str(if index == 0 { " " } else { ", " })?;
            match value {
                OperandValue::Unsigned(value) => write!(f, "{}=#{value}", id.name())?,
                OperandValue::Signed(value) => write!(f, "{}=#{value}", id.name())?,
                OperandValue::Register { class, index } => {
                    let prefix = match class {
                        RegisterClass::A64General => 'x',
                    };
                    write!(f, "{}={prefix}{index}", id.name())?;
                }
            }
        }
        Ok(())
    }
}

/// Deterministic table-definition error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TableError {
    Empty,
    TooManyPatterns,
    InvalidPattern {
        name: &'static str,
        reason: &'static str,
    },
    Overlap {
        first: &'static str,
        second: &'static str,
    },
    DuplicateCoverageId(CoverageId),
}

/// Precompiled byte-indexed decoder. Compilation occurs once per static ISA table.
#[derive(Debug)]
pub struct DecoderTable {
    patterns: &'static [InstructionPattern],
    index_shift: u8,
    buckets: Box<[Box<[u16]>; BUCKET_COUNT]>,
}

impl DecoderTable {
    /// Validates patterns and builds an index using the cheapest 8-bit window.
    pub fn compile(patterns: &'static [InstructionPattern]) -> Result<Self, TableError> {
        validate_patterns(patterns)?;
        let width = 32;
        let mut best_shift = 0;
        let mut best_cost = usize::MAX;
        for shift in (0..width).step_by(8) {
            let cost = index_cost(patterns, shift);
            if cost < best_cost {
                best_cost = cost;
                best_shift = shift;
            }
        }

        let mut buckets: [Vec<u16>; BUCKET_COUNT] = std::array::from_fn(|_| Vec::new());
        for (pattern_index, pattern) in patterns.iter().enumerate() {
            for byte in u8::MIN..=u8::MAX {
                if bucket_can_match(pattern, best_shift, byte) {
                    buckets[usize::from(byte)].push(pattern_index as u16);
                }
            }
        }
        for bucket in &mut buckets {
            bucket.sort_by_key(|index| core::cmp::Reverse(patterns[usize::from(*index)].priority));
        }
        let buckets = Box::new(buckets.map(Vec::into_boxed_slice));
        Ok(Self {
            patterns,
            index_shift: best_shift,
            buckets,
        })
    }

    /// Builds the immutable table selected once for a target platform.
    pub fn compile_for_platform(
        patterns: &'static [InstructionPattern],
        _platform: crate::platform::TargetPlatform,
    ) -> Result<Self, TableError> {
        Self::compile(patterns)
    }

    /// Classifies one encoding without allocation or a full-table scan.
    #[must_use]
    pub fn decode(
        &self,
        location: LocationDescriptor,
        encoding: InstructionEncoding,
    ) -> DecodeResult {
        let diagnostic = InstructionDiagnostic::new(location, encoding);
        let bits = encoding.bits();
        let bucket = ((bits >> self.index_shift) & 0xff) as usize;
        let mut allocation_rejection = None;
        for index in self.buckets[bucket].iter().copied() {
            let pattern = &self.patterns[usize::from(index)];
            if bits & pattern.mask != pattern.value {
                continue;
            }
            match super::allocation::validate(pattern.coverage_id, bits) {
                AllocationStatus::Allocated => {}
                AllocationStatus::Reserved(reason) => {
                    allocation_rejection.get_or_insert((true, pattern.name, reason));
                    continue;
                }
                AllocationStatus::Unallocated(reason) => {
                    allocation_rejection.get_or_insert((false, pattern.name, reason));
                    continue;
                }
            }
            let decoded =
                DecodedInstruction::new(location, encoding, DecodedOpcode { pattern, bits });
            return match pattern.decoder {
                DecodeSupport::Ready => DecodeResult::Decoded(decoded),
                DecodeSupport::RecognizedUnimplemented => {
                    DecodeResult::RecognizedUnimplemented(decoded)
                }
            };
        }
        match allocation_rejection {
            Some((true, name, reason)) => DecodeResult::Reserved {
                instruction: diagnostic,
                name,
                reason,
            },
            Some((false, _, reason)) => DecodeResult::Unallocated {
                instruction: diagnostic,
                reason,
            },
            None => DecodeResult::Unallocated {
                instruction: diagnostic,
                reason: "no allocated instruction pattern matched",
            },
        }
    }

    #[must_use]
    pub const fn index_shift(&self) -> u8 {
        self.index_shift
    }

    #[must_use]
    pub fn candidate_count(&self, encoding: u32) -> usize {
        self.buckets[((encoding >> self.index_shift) & 0xff) as usize].len()
    }
}

const fn extract_operands(pattern: &InstructionPattern, bits: u32) -> DecodedOperands {
    let mut result = DecodedOperands::EMPTY;
    let mut field_index = 0;
    while field_index < pattern.operands.len() {
        let field = pattern.operands[field_index];
        let mask = low_mask(field.width);
        let raw = ((bits >> field.lsb) & mask) as u64;
        let value = match field.kind {
            OperandKind::Unsigned => OperandValue::Unsigned(raw),
            OperandKind::Register(class) => OperandValue::Register {
                class,
                index: raw as u8,
            },
            OperandKind::SignedScaled { scale } => {
                let shift = 64 - field.width;
                let signed = ((raw << shift) as i64) >> shift;
                OperandValue::Signed(signed << scale)
            }
        };
        result.values[result.len()] = Some((field.id, value));
        result.len += 1;
        field_index += 1;
    }
    result
}

fn validate_patterns(patterns: &[InstructionPattern]) -> Result<(), TableError> {
    if patterns.is_empty() {
        return Err(TableError::Empty);
    }
    if patterns.len() > usize::from(u16::MAX) + 1 {
        return Err(TableError::TooManyPatterns);
    }
    for pattern in patterns {
        let width_mask = u32::MAX;
        if pattern.value & !pattern.mask != 0 || pattern.mask & !width_mask != 0 {
            return Err(TableError::InvalidPattern {
                name: pattern.name,
                reason: "mask/value exceeds width or value has unmasked bits",
            });
        }
        if pattern.operands.len() > MAX_OPERANDS {
            return Err(TableError::InvalidPattern {
                name: pattern.name,
                reason: "too many operands",
            });
        }
        let mut seen = [false; 5];
        let mut operand_bits = 0_u32;
        for field in pattern.operands {
            let end = field.lsb.checked_add(field.width);
            if field.width == 0 || end.is_none_or(|end| end > 32) {
                return Err(TableError::InvalidPattern {
                    name: pattern.name,
                    reason: "operand field lies outside encoding",
                });
            }
            match field.kind {
                OperandKind::Register(RegisterClass::A64General) if field.width > 5 => {
                    return Err(TableError::InvalidPattern {
                        name: pattern.name,
                        reason: "A64 register field exceeds five bits",
                    });
                }
                OperandKind::SignedScaled { scale }
                    if field
                        .width
                        .checked_add(scale)
                        .is_none_or(|width| width > 63) =>
                {
                    return Err(TableError::InvalidPattern {
                        name: pattern.name,
                        reason: "signed scaled operand does not fit i64",
                    });
                }
                _ => {}
            }
            let field_mask = low_mask(field.width) << field.lsb;
            if operand_bits & field_mask != 0 {
                return Err(TableError::InvalidPattern {
                    name: pattern.name,
                    reason: "operand fields overlap",
                });
            }
            operand_bits |= field_mask;
            let id = field.id as usize;
            if seen[id] {
                return Err(TableError::InvalidPattern {
                    name: pattern.name,
                    reason: "duplicate operand role",
                });
            }
            seen[id] = true;
        }
        if let Some(fixture) = pattern.regression_fixture
            && fixture.encoding.bits() & pattern.mask != pattern.value
        {
            return Err(TableError::InvalidPattern {
                name: pattern.name,
                reason: "regression fixture does not match its decoder entry",
            });
        }
    }
    for (index, left) in patterns.iter().enumerate() {
        for right in &patterns[index + 1..] {
            if left.coverage_id == right.coverage_id {
                return Err(TableError::DuplicateCoverageId(left.coverage_id));
            }
            let overlap = ((left.value ^ right.value) & (left.mask & right.mask)) == 0;
            if overlap && left.priority == right.priority {
                return Err(TableError::Overlap {
                    first: left.name,
                    second: right.name,
                });
            }
        }
    }
    Ok(())
}

fn index_cost(patterns: &[InstructionPattern], shift: u8) -> usize {
    let mut maximum = 0;
    let mut total = 0;
    for byte in u8::MIN..=u8::MAX {
        let count = patterns
            .iter()
            .filter(|pattern| bucket_can_match(pattern, shift, byte))
            .count();
        maximum = maximum.max(count);
        total += count;
    }
    maximum * BUCKET_COUNT + total
}

fn bucket_can_match(pattern: &InstructionPattern, shift: u8, byte: u8) -> bool {
    let mask = ((pattern.mask >> shift) & 0xff) as u8;
    let value = ((pattern.value >> shift) & 0xff) as u8;
    byte & mask == value
}

const fn low_mask(width: u8) -> u32 {
    if width == 32 {
        u32::MAX
    } else {
        (1_u32 << width) - 1
    }
}
