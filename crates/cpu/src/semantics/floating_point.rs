//! Explicit boundary for architectural floating-point semantics.
//!
//! Arithmetic is expressed in raw IEEE bit patterns and routed to an
//! architectural provider. This deliberately prevents Rust/host FP defaults
//! from silently deciding Arm NaN propagation, rounding, denormal, exception,
//! or fused-operation behavior.

use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FpFormat {
    Binary16 = 16,
    Binary32 = 32,
    Binary64 = 64,
}

impl FpFormat {
    #[must_use]
    pub const fn bits(self) -> u8 {
        self as u8
    }

    const fn value_mask(self) -> u64 {
        match self {
            Self::Binary16 => u16::MAX as u64,
            Self::Binary32 => u32::MAX as u64,
            Self::Binary64 => u64::MAX,
        }
    }

    const fn sign_mask(self) -> u64 {
        1_u64 << (self.bits() - 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FpRoundingMode {
    TiesToEven,
    TowardPositive,
    TowardNegative,
    TowardZero,
    TiesAway,
    ToOdd,
}

/// Guest floating-point controls relevant to a semantic operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FpControl {
    pub rounding_mode: FpRoundingMode,
    pub default_nan: bool,
    pub flush_to_zero: bool,
    pub flush_inputs_to_zero: bool,
    /// Architectural exception trap-enable bits, retained for exact handling.
    pub exception_traps: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FpOperation {
    Abs,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    SquareRoot,
    FusedMultiplyAdd,
    Maximum,
    Minimum,
    Compare { signaling: bool },
    Convert { destination: FpFormat },
    RoundToIntegral { exact: bool },
}

impl FpOperation {
    /// Only sign-bit transforms are intrinsically exact without host FP state.
    #[must_use]
    pub const fn execution_path(self) -> FpExecutionPath {
        match self {
            Self::Abs | Self::Negate => FpExecutionPath::BitExact,
            _ => FpExecutionPath::ArchitecturalProvider,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FpExecutionPath {
    BitExact,
    ArchitecturalProvider,
}

/// A semantic request carrying raw operands, including NaN payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FpRequest {
    pub operation: FpOperation,
    pub format: FpFormat,
    pub operands: [u64; 3],
    pub operand_count: u8,
    pub control: FpControl,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FpStatus {
    pub invalid_operation: bool,
    pub divide_by_zero: bool,
    pub overflow: bool,
    pub underflow: bool,
    pub inexact: bool,
    pub input_denormal: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FpValue {
    Bits(u64),
    /// Packed N, Z, C, V bits in positions 31..28.
    CompareFlags(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FpOutcome {
    pub value: FpValue,
    pub status: FpStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FpError {
    InvalidOperandCount { operation: FpOperation, count: u8 },
    OperandOutsideFormat { operand: u8, format: FpFormat },
    ArchitecturalTrap,
    Unsupported,
}

impl fmt::Display for FpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOperandCount { operation, count } => {
                write!(formatter, "{operation:?} received {count} operands")
            }
            Self::OperandOutsideFormat { operand, format } => {
                write!(formatter, "operand {operand} has bits outside {format:?}")
            }
            Self::ArchitecturalTrap => {
                formatter.write_str("floating-point operation raised an enabled architectural trap")
            }
            Self::Unsupported => formatter.write_str(
                "floating-point operation is not implemented by the architectural provider",
            ),
        }
    }
}

impl std::error::Error for FpError {}

/// Provider implemented by a software FP engine or another verified exact
/// implementation. A JIT may lower selected requests inline only after proving
/// equivalence for the request's controls.
pub trait ArchitecturalFpProvider {
    fn execute(&mut self, request: FpRequest) -> Result<FpOutcome, FpError>;
}

fn expected_operands(operation: FpOperation) -> u8 {
    match operation {
        FpOperation::Abs
        | FpOperation::Negate
        | FpOperation::SquareRoot
        | FpOperation::Convert { .. }
        | FpOperation::RoundToIntegral { .. } => 1,
        FpOperation::Add
        | FpOperation::Subtract
        | FpOperation::Multiply
        | FpOperation::Divide
        | FpOperation::Maximum
        | FpOperation::Minimum
        | FpOperation::Compare { .. } => 2,
        FpOperation::FusedMultiplyAdd => 3,
    }
}

/// Validates and executes an FP request through its required semantic path.
pub fn execute(
    request: FpRequest,
    provider: &mut dyn ArchitecturalFpProvider,
) -> Result<FpOutcome, FpError> {
    let expected = expected_operands(request.operation);
    if request.operand_count != expected {
        return Err(FpError::InvalidOperandCount {
            operation: request.operation,
            count: request.operand_count,
        });
    }
    for index in 0..request.operand_count {
        if request.operands[usize::from(index)] & !request.format.value_mask() != 0 {
            return Err(FpError::OperandOutsideFormat {
                operand: index,
                format: request.format,
            });
        }
    }

    match request.operation {
        FpOperation::Abs => Ok(FpOutcome {
            value: FpValue::Bits(request.operands[0] & !request.format.sign_mask()),
            status: FpStatus::default(),
        }),
        FpOperation::Negate => Ok(FpOutcome {
            value: FpValue::Bits(request.operands[0] ^ request.format.sign_mask()),
            status: FpStatus::default(),
        }),
        _ => provider.execute(request),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTROL: FpControl = FpControl {
        rounding_mode: FpRoundingMode::TowardNegative,
        default_nan: true,
        flush_to_zero: true,
        flush_inputs_to_zero: false,
        exception_traps: 0x12,
    };

    #[derive(Default)]
    struct RecordingProvider {
        request: Option<FpRequest>,
    }

    impl ArchitecturalFpProvider for RecordingProvider {
        fn execute(&mut self, request: FpRequest) -> Result<FpOutcome, FpError> {
            self.request = Some(request);
            Ok(FpOutcome {
                value: FpValue::Bits(0x7fc0_1234),
                status: FpStatus {
                    invalid_operation: true,
                    ..FpStatus::default()
                },
            })
        }
    }

    #[test]
    fn arithmetic_routes_raw_bits_and_controls_to_exact_provider() {
        let request = FpRequest {
            operation: FpOperation::Add,
            format: FpFormat::Binary32,
            operands: [0x7fa0_1234, 0x3f80_0000, 0],
            operand_count: 2,
            control: CONTROL,
        };
        let mut provider = RecordingProvider::default();
        let outcome = execute(request, &mut provider).unwrap();
        assert_eq!(provider.request, Some(request));
        assert_eq!(outcome.value, FpValue::Bits(0x7fc0_1234));
        assert!(outcome.status.invalid_operation);
    }

    #[test]
    fn bit_exact_operations_do_not_invoke_provider() {
        let request = FpRequest {
            operation: FpOperation::Negate,
            format: FpFormat::Binary16,
            operands: [0x7e01, 0, 0],
            operand_count: 1,
            control: CONTROL,
        };
        let mut provider = RecordingProvider::default();
        assert_eq!(
            execute(request, &mut provider).unwrap().value,
            FpValue::Bits(0xfe01)
        );
        assert_eq!(provider.request, None);
    }

    #[test]
    fn malformed_requests_are_rejected_before_execution() {
        let mut provider = RecordingProvider::default();
        let request = FpRequest {
            operation: FpOperation::SquareRoot,
            format: FpFormat::Binary16,
            operands: [0x1_0000, 0, 0],
            operand_count: 1,
            control: CONTROL,
        };
        assert_eq!(
            execute(request, &mut provider),
            Err(FpError::OperandOutsideFormat {
                operand: 0,
                format: FpFormat::Binary16
            })
        );
        assert_eq!(provider.request, None);
    }
}
