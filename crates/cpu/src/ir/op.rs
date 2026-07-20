//! Typed IR operations and explicit architectural effects.

use crate::{
    location::LocationDescriptor,
    memory::{MemoryAccess, MemoryAccessSize},
    state::{a32::A32GeneralRegister, a64::A64GeneralRegister},
};

use super::{
    types::IrType,
    value::{Operand, Value},
};

/// Validated index in a 32-register architectural bank.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RegisterIndex(u8);

impl RegisterIndex {
    /// Creates an index for registers 0 through 31.
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if index < 32 { Some(Self(index)) } else { None }
    }

    /// Returns the architectural index.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Semantic architectural state field, independent of its host layout.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateRegister {
    /// A64 X0-X30.
    A64X(A64GeneralRegister),
    /// A64 stack pointer.
    A64Sp,
    /// A64 current instruction address.
    A64Pc,
    /// A64 packed NZCV.
    A64Nzcv,
    /// A64 V0-V31.
    A64V(RegisterIndex),
    /// A64 floating-point control register.
    A64Fpcr,
    /// A64 floating-point status register.
    A64Fpsr,
    /// A64 writable userspace thread pointer.
    A64TpidrEl0,
    /// A64 read-only userspace thread pointer.
    A64TpidrroEl0,
    /// A32 R0-R14.
    A32R(A32GeneralRegister),
    /// A32 architectural PC, whose operand read behavior is resolved by the frontend.
    A32Pc,
    /// A32 CPSR.
    A32Cpsr,
    /// A32 D0-D31 backing register.
    A32D(RegisterIndex),
    /// A32 floating-point status and control register.
    A32Fpscr,
    /// A32 writable userspace thread pointer.
    A32Tpidrurw,
    /// A32 read-only userspace thread pointer.
    A32Tpidruro,
}

impl StateRegister {
    /// Returns the canonical value type of the state field.
    #[must_use]
    pub const fn ty(self) -> IrType {
        match self {
            Self::A64X(_)
            | Self::A64Sp
            | Self::A64Pc
            | Self::A64TpidrEl0
            | Self::A64TpidrroEl0
            | Self::A32D(_) => IrType::I64,
            Self::A64V(_) => IrType::V128,
            Self::A64Nzcv
            | Self::A64Fpcr
            | Self::A64Fpsr
            | Self::A32R(_)
            | Self::A32Pc
            | Self::A32Cpsr
            | Self::A32Fpscr
            | Self::A32Tpidrurw
            | Self::A32Tpidruro => IrType::I32,
        }
    }
}

/// Integer binary arithmetic and bitwise operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IntegerBinaryKind {
    /// Wrapping addition.
    Add,
    /// Wrapping subtraction.
    Subtract,
    /// Low-half multiplication.
    Multiply,
    /// Unsigned division.
    UnsignedDivide,
    /// Signed division.
    SignedDivide,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise exclusive OR.
    Xor,
}

/// Shift or rotate operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShiftKind {
    /// Logical left shift.
    LogicalLeft,
    /// Logical right shift.
    LogicalRight,
    /// Arithmetic right shift.
    ArithmeticRight,
    /// Rotate left.
    RotateLeft,
    /// Rotate right.
    RotateRight,
}

/// Integer comparison predicate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IntegerPredicate {
    /// Equal.
    Equal,
    /// Not equal.
    NotEqual,
    /// Unsigned less than.
    UnsignedLessThan,
    /// Unsigned less than or equal.
    UnsignedLessThanOrEqual,
    /// Signed less than.
    SignedLessThan,
    /// Signed less than or equal.
    SignedLessThanOrEqual,
}

/// Explicit scalar integer operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScalarOperation {
    /// Arithmetic or bitwise binary operation.
    Binary {
        kind: IntegerBinaryKind,
        lhs: Operand,
        rhs: Operand,
    },
    /// Add with an incoming carry, producing result, carry, and signed overflow.
    AddWithCarry {
        lhs: Operand,
        rhs: Operand,
        carry_in: Operand,
    },
    /// Derives unsigned overflow from two operands and a result.
    UnsignedOverflow {
        operation: IntegerBinaryKind,
        lhs: Operand,
        rhs: Operand,
        result: Operand,
    },
    /// Derives signed overflow from two operands and a result.
    SignedOverflow {
        operation: IntegerBinaryKind,
        lhs: Operand,
        rhs: Operand,
        result: Operand,
    },
    /// Typed comparison producing I1.
    Compare {
        predicate: IntegerPredicate,
        lhs: Operand,
        rhs: Operand,
    },
    /// Chooses between same-typed values using an I1 condition.
    Select {
        condition: Operand,
        when_true: Operand,
        when_false: Operand,
    },
    /// Shift or rotate by an explicitly typed amount.
    Shift {
        kind: ShiftKind,
        value: Operand,
        amount: Operand,
    },
    /// Counts leading zero bits.
    CountLeadingZeros { value: Operand },
    /// Reverses all bits.
    ReverseBits { value: Operand },
    /// Zero extension into a wider integer type.
    ZeroExtend { value: Operand, to: IrType },
    /// Sign extension into a wider integer type.
    SignExtend { value: Operand, to: IrType },
    /// Truncation into a narrower integer type.
    Truncate { value: Operand, to: IrType },
    /// Width-preserving reinterpretation between integer, FP, vector, or address domains.
    Bitcast { value: Operand, to: IrType },
}

/// All Arm condition encodings, shared by A64 and AArch32 consumers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Condition {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

/// Lazy flag creation, consumption, and architectural materialization.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FlagOperation {
    /// Forms lazy NZCV from an arithmetic result and explicit carry/overflow bits.
    FromArithmetic {
        result: Operand,
        carry: Operand,
        overflow: Operand,
    },
    /// Forms lazy NZCV for a logical result with an explicit carry bit.
    FromLogical { result: Operand, carry: Operand },
    /// Converts packed architectural NZCV/CPSR bits into lazy flags.
    FromPacked { value: Operand },
    /// Evaluates one architectural condition into I1.
    Evaluate {
        flags: Operand,
        condition: Condition,
    },
    /// Materializes lazy flags as packed I32 architectural bits.
    Materialize { flags: Operand },
}

/// Byte order of one guest data access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ByteOrder {
    Little,
    Big,
}

/// Whether an access may be eliminated when its value is otherwise unused.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Volatility {
    NonVolatile,
    Volatile,
}

/// Common semantic descriptor for typed memory operations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryDescriptor {
    /// Access size, alignment, ordering, and class.
    pub access: MemoryAccess,
    /// Guest byte order.
    pub byte_order: ByteOrder,
    /// Explicit observability independent of the access class.
    pub volatility: Volatility,
}

impl MemoryDescriptor {
    /// Returns the integer/vector type transferred by this descriptor.
    #[must_use]
    pub const fn value_type(self) -> IrType {
        match self.access.size {
            MemoryAccessSize::Byte => IrType::I8,
            MemoryAccessSize::Halfword => IrType::I16,
            MemoryAccessSize::Word => IrType::I32,
            MemoryAccessSize::Doubleword => IrType::I64,
            MemoryAccessSize::Quadword => IrType::I128,
        }
    }
}

/// Typed ordinary memory operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryOperation {
    /// Read from a guest address.
    Load {
        address: Operand,
        descriptor: MemoryDescriptor,
    },
    /// Write to a guest address.
    Store {
        address: Operand,
        value: Operand,
        descriptor: MemoryDescriptor,
    },
}

/// Barrier domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierDomain {
    NonShareable,
    InnerShareable,
    OuterShareable,
    FullSystem,
}

/// Accesses ordered by a memory barrier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierAccess {
    Reads,
    Writes,
    ReadsAndWrites,
}

/// Architectural barrier operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierOperation {
    DataMemory {
        domain: BarrierDomain,
        access: BarrierAccess,
    },
    DataSynchronization {
        domain: BarrierDomain,
        access: BarrierAccess,
    },
    InstructionSynchronization,
}

/// Semantic cache-maintenance action.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheMaintenanceKind {
    InstructionInvalidate,
    DataInvalidate,
    DataClean,
    DataCleanAndInvalidate,
    InstructionPrefetch,
}

/// Cache-maintenance operation, optionally applied to a guest address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CacheMaintenanceOperation {
    pub kind: CacheMaintenanceKind,
    pub address: Option<Operand>,
}

/// Exclusive-monitor operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExclusiveOperation {
    Load {
        address: Operand,
        descriptor: MemoryDescriptor,
    },
    Store {
        address: Operand,
        value: Operand,
        descriptor: MemoryDescriptor,
    },
    Clear,
}

/// Atomic read-modify-write function.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AtomicRmwKind {
    Add,
    Clear,
    Xor,
    Set,
    SignedMaximum,
    SignedMinimum,
    UnsignedMaximum,
    UnsignedMinimum,
    Swap,
}

/// Atomic memory operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AtomicOperation {
    ReadModifyWrite {
        kind: AtomicRmwKind,
        address: Operand,
        value: Operand,
        descriptor: MemoryDescriptor,
    },
    CompareExchange {
        address: Operand,
        expected: Operand,
        replacement: Operand,
        descriptor: MemoryDescriptor,
    },
}

/// Vector lane element interpretation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LaneType {
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

/// Lane arrangement retained as semantic lowering information.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VectorArrangement {
    pub lane_type: LaneType,
    pub lane_count: u8,
}

/// Semantic vector family; operations are not expanded into scalar lanes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VectorOperation {
    Arithmetic {
        kind: IntegerBinaryKind,
        arrangement: VectorArrangement,
        lhs: Operand,
        rhs: Operand,
    },
    Compare {
        predicate: IntegerPredicate,
        arrangement: VectorArrangement,
        lhs: Operand,
        rhs: Operand,
    },
    Shift {
        kind: ShiftKind,
        arrangement: VectorArrangement,
        value: Operand,
        amount: Operand,
    },
    Widen {
        signed: bool,
        from: VectorArrangement,
        to: VectorArrangement,
        value: Operand,
    },
    Narrow {
        saturating: bool,
        signed: bool,
        saturation_status: SaturationStatus,
        from: VectorArrangement,
        to: VectorArrangement,
        value: Operand,
    },
    SaturatingArithmetic {
        subtract: bool,
        signed: bool,
        saturation_status: SaturationStatus,
        arrangement: VectorArrangement,
        lhs: Operand,
        rhs: Operand,
    },
    Permute {
        arrangement: VectorArrangement,
        first: Operand,
        second: Option<Operand>,
        indices: Operand,
    },
    ExtractLane {
        arrangement: VectorArrangement,
        vector: Operand,
        lane: u8,
    },
    InsertLane {
        arrangement: VectorArrangement,
        vector: Operand,
        lane: u8,
        value: Operand,
    },
}

/// Architectural sticky status updated by a saturating vector operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SaturationStatus {
    None,
    A64FpsrQc,
    A32CpsrQ,
}

/// Floating-point rounding rule.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpRoundingMode {
    FromFpcr,
    ToNearestTiesEven,
    TowardPositiveInfinity,
    TowardNegativeInfinity,
    TowardZero,
    ToNearestTiesAway,
}

/// NaN processing rule required by Arm semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NaNMode {
    /// Honor FPCR default-NaN control and otherwise apply Arm propagation rules.
    FpcrControlled,
    /// Force the architectural default NaN.
    DefaultNaN,
    /// Apply Arm operand propagation rules even if a host would canonicalize.
    Propagate,
}

/// How architectural FP exceptions are exposed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpExceptionMode {
    /// Accumulate applicable FPSR status bits.
    RecordStatus,
    /// Produce a precise architectural exception when enabled.
    Trap,
}

/// Complete control/status behavior required by an FP operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FpBehavior {
    pub rounding: FpRoundingMode,
    pub nan: NaNMode,
    /// Read flush-to-zero and input-denormal controls from FPCR.
    pub fpcr_controls_denormals: bool,
    /// Whether the operation may update FPSR cumulative exception bits.
    pub updates_fpsr: bool,
    pub exception_mode: FpExceptionMode,
}

/// Floating-point binary arithmetic operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FpBinaryKind {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    MaximumNumber,
    MinimumNumber,
}

/// Floating-point operation family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FloatingPointOperation {
    Binary {
        kind: FpBinaryKind,
        lhs: Operand,
        rhs: Operand,
        behavior: FpBehavior,
    },
    FusedMultiplyAdd {
        multiplicand: Operand,
        multiplier: Operand,
        addend: Operand,
        behavior: FpBehavior,
    },
    Compare {
        lhs: Operand,
        rhs: Operand,
        signaling: bool,
        behavior: FpBehavior,
    },
    Convert {
        value: Operand,
        to: IrType,
        signed: bool,
        behavior: FpBehavior,
    },
    RoundToIntegral {
        value: Operand,
        behavior: FpBehavior,
    },
}

/// Optional slow helper with an explicit effect declaration.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct HelperOperation {
    pub helper: Box<str>,
    pub arguments: Box<[Operand]>,
    pub effects: OperationEffects,
}

/// Complete operation payload.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum OperationKind {
    Constant(super::value::Immediate),
    Scalar(ScalarOperation),
    ReadState(StateRegister),
    WriteState {
        register: StateRegister,
        value: Operand,
    },
    Flags(FlagOperation),
    Memory(MemoryOperation),
    Barrier(BarrierOperation),
    CacheMaintenance(CacheMaintenanceOperation),
    Exclusive(ExclusiveOperation),
    Atomic(AtomicOperation),
    Vector(VectorOperation),
    FloatingPoint(FloatingPointOperation),
    Helper(HelperOperation),
}

/// Bit set of side effects visible to verification and optimization.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EffectSet(u16);

impl EffectSet {
    pub const NONE: Self = Self(0);
    pub const READ_STATE: Self = Self(1 << 0);
    pub const WRITE_STATE: Self = Self(1 << 1);
    pub const READ_MEMORY: Self = Self(1 << 2);
    pub const WRITE_MEMORY: Self = Self(1 << 3);
    pub const BARRIER: Self = Self(1 << 4);
    pub const CACHE: Self = Self(1 << 5);
    pub const EXCLUSIVE: Self = Self(1 << 6);
    pub const ATOMIC: Self = Self(1 << 7);
    pub const READ_FPCR: Self = Self(1 << 8);
    pub const WRITE_FPSR: Self = Self(1 << 9);
    pub const VOLATILE: Self = Self(1 << 10);
    pub const HELPER: Self = Self(1 << 11);

    /// Returns the union of two effect sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Checks whether every requested effect is present.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// Effect and precise-fault metadata attached to every operation.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct OperationEffects {
    pub side_effects: EffectSet,
    pub may_fault: bool,
}

impl OperationEffects {
    #[must_use]
    pub const fn new(side_effects: EffectSet, may_fault: bool) -> Self {
        Self {
            side_effects,
            may_fault,
        }
    }
}

/// Up to three typed results without allocating per ordinary operation.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct OperationResults([Option<Value>; 3]);

impl OperationResults {
    pub const NONE: Self = Self([None, None, None]);

    #[must_use]
    pub const fn one(first: Value) -> Self {
        Self([Some(first), None, None])
    }

    #[must_use]
    pub const fn two(first: Value, second: Value) -> Self {
        Self([Some(first), Some(second), None])
    }

    #[must_use]
    pub const fn three(first: Value, second: Value, third: Value) -> Self {
        Self([Some(first), Some(second), Some(third)])
    }

    /// Iterates defined results in declaration order.
    pub fn iter(self) -> impl Iterator<Item = Value> {
        self.0.into_iter().flatten()
    }
}

/// One sourced typed operation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IrOperation {
    /// Instruction whose semantics produced the operation.
    pub source: LocationDescriptor,
    /// Immutable results defined by this operation.
    pub results: OperationResults,
    /// Semantic operation payload.
    pub kind: OperationKind,
    /// Explicit optimizer/backend contract.
    pub effects: OperationEffects,
}

impl IrOperation {
    /// Creates an operation and attaches effects derived from its semantics.
    #[must_use]
    pub fn new(source: LocationDescriptor, results: OperationResults, kind: OperationKind) -> Self {
        let effects = kind.derived_effects();
        Self {
            source,
            results,
            kind,
            effects,
        }
    }
}

impl OperationKind {
    /// Derives the non-optional effect contract for this operation.
    #[must_use]
    pub fn derived_effects(&self) -> OperationEffects {
        match self {
            Self::Constant(_) | Self::Scalar(_) | Self::Flags(_) => OperationEffects::default(),
            Self::Vector(operation) => {
                let status = match operation {
                    VectorOperation::Narrow {
                        saturation_status, ..
                    }
                    | VectorOperation::SaturatingArithmetic {
                        saturation_status, ..
                    } => *saturation_status,
                    VectorOperation::Arithmetic { .. }
                    | VectorOperation::Compare { .. }
                    | VectorOperation::Shift { .. }
                    | VectorOperation::Widen { .. }
                    | VectorOperation::Permute { .. }
                    | VectorOperation::ExtractLane { .. }
                    | VectorOperation::InsertLane { .. } => SaturationStatus::None,
                };
                let effects = if status == SaturationStatus::None {
                    EffectSet::NONE
                } else {
                    EffectSet::WRITE_STATE
                };
                OperationEffects::new(effects, false)
            }
            Self::ReadState(_) => OperationEffects::new(EffectSet::READ_STATE, false),
            Self::WriteState { .. } => OperationEffects::new(EffectSet::WRITE_STATE, false),
            Self::Memory(MemoryOperation::Load { descriptor, .. }) => {
                memory_effects(EffectSet::READ_MEMORY, *descriptor)
            }
            Self::Memory(MemoryOperation::Store { descriptor, .. }) => {
                memory_effects(EffectSet::WRITE_MEMORY, *descriptor)
            }
            Self::Barrier(_) => OperationEffects::new(EffectSet::BARRIER, false),
            Self::CacheMaintenance(operation) => {
                OperationEffects::new(EffectSet::CACHE, operation.address.is_some())
            }
            Self::Exclusive(ExclusiveOperation::Clear) => {
                OperationEffects::new(EffectSet::EXCLUSIVE, false)
            }
            Self::Exclusive(ExclusiveOperation::Load { descriptor, .. }) => memory_effects(
                EffectSet::READ_MEMORY.union(EffectSet::EXCLUSIVE),
                *descriptor,
            ),
            Self::Exclusive(ExclusiveOperation::Store { descriptor, .. }) => memory_effects(
                EffectSet::WRITE_MEMORY.union(EffectSet::EXCLUSIVE),
                *descriptor,
            ),
            Self::Atomic(operation) => {
                let descriptor = match operation {
                    AtomicOperation::ReadModifyWrite { descriptor, .. }
                    | AtomicOperation::CompareExchange { descriptor, .. } => *descriptor,
                };
                memory_effects(
                    EffectSet::READ_MEMORY
                        .union(EffectSet::WRITE_MEMORY)
                        .union(EffectSet::ATOMIC),
                    descriptor,
                )
            }
            Self::FloatingPoint(operation) => {
                let behavior = match operation {
                    FloatingPointOperation::Binary { behavior, .. }
                    | FloatingPointOperation::FusedMultiplyAdd { behavior, .. }
                    | FloatingPointOperation::Compare { behavior, .. }
                    | FloatingPointOperation::Convert { behavior, .. }
                    | FloatingPointOperation::RoundToIntegral { behavior, .. } => behavior,
                };
                let effects = if behavior.updates_fpsr {
                    EffectSet::READ_STATE
                        .union(EffectSet::READ_FPCR)
                        .union(EffectSet::WRITE_STATE)
                        .union(EffectSet::WRITE_FPSR)
                } else {
                    EffectSet::READ_STATE.union(EffectSet::READ_FPCR)
                };
                OperationEffects::new(effects, behavior.exception_mode == FpExceptionMode::Trap)
            }
            Self::Helper(helper) => OperationEffects::new(
                helper.effects.side_effects.union(EffectSet::HELPER),
                helper.effects.may_fault,
            ),
        }
    }
}

fn memory_effects(base: EffectSet, descriptor: MemoryDescriptor) -> OperationEffects {
    let side_effects = if descriptor.volatility == Volatility::Volatile {
        base.union(EffectSet::VOLATILE)
    } else {
        base
    };
    OperationEffects::new(side_effects, true)
}

/// Generic envelope retained for frontend components which need to attach a
/// precise source before converting a payload into [`IrOperation`].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FaultingOperation<T> {
    pub location: LocationDescriptor,
    pub operation: T,
}

impl<T> FaultingOperation<T> {
    #[must_use]
    pub const fn new(location: LocationDescriptor, operation: T) -> Self {
        Self {
            location,
            operation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::value::{Immediate, ValueId};
    use super::*;
    use crate::{
        address::GuestVirtualAddress,
        location::ExecutionState,
        memory::{MemoryAccessClass, MemoryAlignment, MemoryOrdering},
        profile::CpuProfileId,
    };

    fn location() -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(0x8000),
            ExecutionState::A64,
            CpuProfileId::new(3),
        )
    }

    #[test]
    fn state_accesses_are_semantic_and_typed() {
        let register = StateRegister::A64V(RegisterIndex::new(31).unwrap());
        let operation = IrOperation::new(
            location(),
            OperationResults::one(Value::new(ValueId::new(0), register.ty())),
            OperationKind::ReadState(register),
        );

        assert_eq!(register.ty(), IrType::V128);
        assert!(
            operation
                .effects
                .side_effects
                .contains(EffectSet::READ_STATE)
        );
        assert!(!operation.effects.may_fault);
    }

    #[test]
    fn memory_effects_preserve_fault_and_volatility_metadata() {
        let descriptor = MemoryDescriptor {
            access: MemoryAccess::new(
                MemoryAccessSize::Word,
                MemoryAlignment::Natural,
                MemoryOrdering::Acquire,
                MemoryAccessClass::Normal,
            ),
            byte_order: ByteOrder::Little,
            volatility: Volatility::Volatile,
        };
        let operation = IrOperation::new(
            location(),
            OperationResults::one(Value::new(ValueId::new(1), IrType::I32)),
            OperationKind::Memory(MemoryOperation::Load {
                address: Immediate::Address(GuestVirtualAddress::new(0x9000)).into(),
                descriptor,
            }),
        );

        assert_eq!(descriptor.value_type(), IrType::I32);
        assert!(operation.effects.may_fault);
        assert!(
            operation
                .effects
                .side_effects
                .contains(EffectSet::READ_MEMORY)
        );
        assert!(operation.effects.side_effects.contains(EffectSet::VOLATILE));
        assert_eq!(operation.source, location());
    }

    #[test]
    fn add_with_carry_can_define_result_carry_and_overflow() {
        let results = OperationResults::three(
            Value::new(ValueId::new(0), IrType::I64),
            Value::new(ValueId::new(1), IrType::I1),
            Value::new(ValueId::new(2), IrType::I1),
        );
        let operation = IrOperation::new(
            location(),
            results,
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs: Immediate::I64(1).into(),
                rhs: Immediate::I64(2).into(),
                carry_in: Immediate::I1(false).into(),
            }),
        );

        assert_eq!(operation.results.iter().count(), 3);
        assert_eq!(operation.effects, OperationEffects::default());
    }

    #[test]
    fn fp_behavior_is_explicit_and_updates_status() {
        let behavior = FpBehavior {
            rounding: FpRoundingMode::FromFpcr,
            nan: NaNMode::FpcrControlled,
            fpcr_controls_denormals: true,
            updates_fpsr: true,
            exception_mode: FpExceptionMode::RecordStatus,
        };
        let operation = IrOperation::new(
            location(),
            OperationResults::one(Value::new(ValueId::new(0), IrType::F32)),
            OperationKind::FloatingPoint(FloatingPointOperation::Binary {
                kind: FpBinaryKind::Add,
                lhs: Immediate::F32(0x3f80_0000).into(),
                rhs: Immediate::F32(0x4000_0000).into(),
                behavior,
            }),
        );

        assert!(
            operation
                .effects
                .side_effects
                .contains(EffectSet::READ_STATE)
        );
        assert!(
            operation
                .effects
                .side_effects
                .contains(EffectSet::READ_FPCR)
        );
        assert!(
            operation
                .effects
                .side_effects
                .contains(EffectSet::WRITE_FPSR)
        );
        assert!(!operation.effects.may_fault);
    }

    #[test]
    fn system_memory_families_have_non_optional_effects() {
        let descriptor = MemoryDescriptor {
            access: MemoryAccess::new(
                MemoryAccessSize::Doubleword,
                MemoryAlignment::Natural,
                MemoryOrdering::AcquireRelease,
                MemoryAccessClass::Atomic,
            ),
            byte_order: ByteOrder::Little,
            volatility: Volatility::NonVolatile,
        };
        let address = Immediate::Address(GuestVirtualAddress::new(0xa000)).into();
        let barrier = OperationKind::Barrier(BarrierOperation::DataMemory {
            domain: BarrierDomain::InnerShareable,
            access: BarrierAccess::ReadsAndWrites,
        });
        let cache = OperationKind::CacheMaintenance(CacheMaintenanceOperation {
            kind: CacheMaintenanceKind::InstructionInvalidate,
            address: Some(address),
        });
        let exclusive = OperationKind::Exclusive(ExclusiveOperation::Load {
            address,
            descriptor,
        });
        let atomic = OperationKind::Atomic(AtomicOperation::CompareExchange {
            address,
            expected: Immediate::I64(1).into(),
            replacement: Immediate::I64(2).into(),
            descriptor,
        });

        assert!(
            barrier
                .derived_effects()
                .side_effects
                .contains(EffectSet::BARRIER)
        );
        assert!(
            cache
                .derived_effects()
                .side_effects
                .contains(EffectSet::CACHE)
        );
        assert!(cache.derived_effects().may_fault);
        assert!(
            exclusive
                .derived_effects()
                .side_effects
                .contains(EffectSet::EXCLUSIVE)
        );
        assert!(
            atomic
                .derived_effects()
                .side_effects
                .contains(EffectSet::ATOMIC)
        );
        assert!(atomic.derived_effects().may_fault);
    }

    #[test]
    fn semantic_vector_saturation_records_the_architectural_sticky_flag() {
        let operation = OperationKind::Vector(VectorOperation::SaturatingArithmetic {
            subtract: false,
            signed: true,
            saturation_status: SaturationStatus::A64FpsrQc,
            arrangement: VectorArrangement {
                lane_type: LaneType::I16,
                lane_count: 8,
            },
            lhs: Immediate::V128(1).into(),
            rhs: Immediate::V128(2).into(),
        });

        assert!(
            operation
                .derived_effects()
                .side_effects
                .contains(EffectSet::WRITE_STATE)
        );
    }
}
