//! Host-independent IR value types.

/// Type of an immutable IR value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum IrType {
    /// One-bit boolean.
    I1,
    /// 8-bit integer bits.
    I8,
    /// 16-bit integer bits.
    I16,
    /// 32-bit integer bits.
    I32,
    /// 64-bit integer bits.
    I64,
    /// 128-bit integer bits.
    I128,
    /// IEEE binary16 value interpreted using explicit Arm FP behavior.
    F16,
    /// IEEE binary32 value interpreted using explicit Arm FP behavior.
    F32,
    /// IEEE binary64 value interpreted using explicit Arm FP behavior.
    F64,
    /// Opaque 64-bit vector value.
    V64,
    /// Opaque 128-bit vector value.
    V128,
    /// Guest virtual address, distinct from integer bits.
    Address,
    /// Lazy architectural integer flags.
    Flags,
}

impl IrType {
    /// Returns the storage width when the type has a fixed bit representation.
    #[must_use]
    pub const fn bit_width(self) -> Option<u16> {
        Some(match self {
            Self::I1 => 1,
            Self::I8 => 8,
            Self::I16 | Self::F16 => 16,
            Self::I32 | Self::F32 => 32,
            Self::I64 | Self::F64 | Self::V64 | Self::Address => 64,
            Self::I128 | Self::V128 => 128,
            Self::Flags => return None,
        })
    }

    /// Returns whether this is a scalar integer type.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I1 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128
        )
    }

    /// Returns whether this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Returns whether this is a semantic vector type.
    #[must_use]
    pub const fn is_vector(self) -> bool {
        matches!(self, Self::V64 | Self::V128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_types_remain_distinct() {
        let required = [
            IrType::I1,
            IrType::I8,
            IrType::I16,
            IrType::I32,
            IrType::I64,
            IrType::I128,
            IrType::F16,
            IrType::F32,
            IrType::F64,
            IrType::V64,
            IrType::V128,
            IrType::Address,
        ];
        for (index, ty) in required.iter().enumerate() {
            assert!(!required[..index].contains(ty));
            assert!(ty.bit_width().is_some());
        }
        assert_eq!(IrType::Flags.bit_width(), None);
    }
}
