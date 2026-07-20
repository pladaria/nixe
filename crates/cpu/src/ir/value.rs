//! Immutable SSA-like IR values and typed immediates.

use crate::address::GuestVirtualAddress;

use super::types::IrType;

/// Block-local identity assigned exactly once by an IR constructor.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ValueId(u32);

impl ValueId {
    /// Creates an ID in a block-local namespace.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the block-local numeric index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Typed reference to an immutable SSA-like result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Value {
    /// Block-local identity.
    pub id: ValueId,
    /// Type fixed at definition time.
    pub ty: IrType,
}

impl Value {
    /// Creates a typed result reference.
    #[must_use]
    pub const fn new(id: ValueId, ty: IrType) -> Self {
        Self { id, ty }
    }
}

/// Immediate with a type intrinsic to its variant.
///
/// Floating-point variants contain raw IEEE bits so creating IR never invokes
/// host floating-point canonicalization. Vector variants likewise preserve all
/// source bits.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Immediate {
    /// Boolean integer.
    I1(bool),
    /// 8-bit integer bits.
    I8(u8),
    /// 16-bit integer bits.
    I16(u16),
    /// 32-bit integer bits.
    I32(u32),
    /// 64-bit integer bits.
    I64(u64),
    /// 128-bit integer bits.
    I128(u128),
    /// Raw IEEE binary16 bits.
    F16(u16),
    /// Raw IEEE binary32 bits.
    F32(u32),
    /// Raw IEEE binary64 bits.
    F64(u64),
    /// Raw 64-bit vector bits.
    V64(u64),
    /// Raw 128-bit vector bits.
    V128(u128),
    /// Guest virtual address.
    Address(GuestVirtualAddress),
}

impl Immediate {
    /// Returns the IR type carried by this immediate.
    #[must_use]
    pub const fn ty(self) -> IrType {
        match self {
            Self::I1(_) => IrType::I1,
            Self::I8(_) => IrType::I8,
            Self::I16(_) => IrType::I16,
            Self::I32(_) => IrType::I32,
            Self::I64(_) => IrType::I64,
            Self::I128(_) => IrType::I128,
            Self::F16(_) => IrType::F16,
            Self::F32(_) => IrType::F32,
            Self::F64(_) => IrType::F64,
            Self::V64(_) => IrType::V64,
            Self::V128(_) => IrType::V128,
            Self::Address(_) => IrType::Address,
        }
    }
}

/// Operand accepted by an IR operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Operand {
    /// Previously defined SSA-like result.
    Value(Value),
    /// Typed constant.
    Immediate(Immediate),
}

impl Operand {
    /// Returns the operand type without consulting external tables.
    #[must_use]
    pub const fn ty(self) -> IrType {
        match self {
            Self::Value(value) => value.ty,
            Self::Immediate(immediate) => immediate.ty(),
        }
    }
}

impl From<Value> for Operand {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<Immediate> for Operand {
    fn from(value: Immediate) -> Self {
        Self::Immediate(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_immediates_are_intrinsically_typed() {
        let value = Value::new(ValueId::new(4), IrType::V128);
        let address = Immediate::Address(GuestVirtualAddress::new(0x0071_0000_0000));

        assert_eq!(Operand::from(value).ty(), IrType::V128);
        assert_eq!(Operand::from(address).ty(), IrType::Address);
        assert_ne!(Immediate::I64(0).ty(), address.ty());
    }
}
