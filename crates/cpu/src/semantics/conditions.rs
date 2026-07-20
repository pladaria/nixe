//! Architectural condition flags and condition-code evaluation.

use crate::ir::op::Condition;

/// The four integer condition flags in architectural N, Z, C, V order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Nzcv {
    pub negative: bool,
    pub zero: bool,
    pub carry: bool,
    pub overflow: bool,
}

impl Nzcv {
    /// Extracts A64 PSTATE.NZCV from its architectural packed representation.
    #[must_use]
    pub const fn from_a64_nzcv(value: u32) -> Self {
        Self::from_packed(value)
    }

    /// Extracts the AArch32 CPSR condition flags without interpreting other bits.
    #[must_use]
    pub const fn from_a32_cpsr(value: u32) -> Self {
        Self::from_packed(value)
    }

    const fn from_packed(value: u32) -> Self {
        Self {
            negative: value & (1 << 31) != 0,
            zero: value & (1 << 30) != 0,
            carry: value & (1 << 29) != 0,
            overflow: value & (1 << 28) != 0,
        }
    }

    #[must_use]
    pub const fn packed(self) -> u32 {
        (self.negative as u32) << 31
            | (self.zero as u32) << 30
            | (self.carry as u32) << 29
            | (self.overflow as u32) << 28
    }
}

const fn base_condition(flags: Nzcv, condition: Condition) -> bool {
    match condition {
        Condition::Eq | Condition::Ne => flags.zero,
        Condition::Cs | Condition::Cc => flags.carry,
        Condition::Mi | Condition::Pl => flags.negative,
        Condition::Vs | Condition::Vc => flags.overflow,
        Condition::Hi | Condition::Ls => flags.carry && !flags.zero,
        Condition::Ge | Condition::Lt => flags.negative == flags.overflow,
        Condition::Gt | Condition::Le => !flags.zero && flags.negative == flags.overflow,
        Condition::Al | Condition::Nv => true,
    }
}

const fn inverted(condition: Condition) -> bool {
    matches!(
        condition,
        Condition::Ne
            | Condition::Cc
            | Condition::Pl
            | Condition::Vc
            | Condition::Ls
            | Condition::Lt
            | Condition::Le
    )
}

/// Evaluates an A64 condition. Both `AL` and the reserved `NV` encoding behave
/// as unconditional in the architectural `ConditionHolds` primitive.
#[must_use]
pub const fn evaluate_a64(condition: Condition, nzcv: u32) -> bool {
    let flags = Nzcv::from_a64_nzcv(nzcv);
    let result = base_condition(flags, condition);
    if inverted(condition) { !result } else { result }
}

/// Evaluates the ordinary AArch32 condition space.
///
/// Encoding `0b1111` (`NV`) belongs to AArch32's unconditional instruction
/// space and must be classified by the decoder, not passed as a normal
/// condition. Returning false here prevents accidental conditional execution.
#[must_use]
pub const fn evaluate_a32(condition: Condition, cpsr: u32) -> bool {
    if matches!(condition, Condition::Nv) {
        return false;
    }
    let flags = Nzcv::from_a32_cpsr(cpsr);
    let result = base_condition(flags, condition);
    if inverted(condition) { !result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONDITIONS: [Condition; 16] = [
        Condition::Eq,
        Condition::Ne,
        Condition::Cs,
        Condition::Cc,
        Condition::Mi,
        Condition::Pl,
        Condition::Vs,
        Condition::Vc,
        Condition::Hi,
        Condition::Ls,
        Condition::Ge,
        Condition::Lt,
        Condition::Gt,
        Condition::Le,
        Condition::Al,
        Condition::Nv,
    ];

    #[test]
    fn all_conditions_are_checked_for_every_flag_combination() {
        for packed_nibble in 0_u32..16 {
            let packed = packed_nibble << 28;
            for pair in CONDITIONS[..14].chunks_exact(2) {
                assert_ne!(evaluate_a64(pair[0], packed), evaluate_a64(pair[1], packed));
                assert_ne!(evaluate_a32(pair[0], packed), evaluate_a32(pair[1], packed));
            }
            assert!(evaluate_a64(Condition::Al, packed));
            assert!(evaluate_a64(Condition::Nv, packed));
            assert!(evaluate_a32(Condition::Al, packed));
            assert!(!evaluate_a32(Condition::Nv, packed));
        }
    }

    #[test]
    fn nzcv_round_trips_without_other_status_bits() {
        for nibble in 0_u32..16 {
            assert_eq!(
                Nzcv::from_a32_cpsr((nibble << 28) | 0xffff).packed(),
                nibble << 28
            );
        }
    }
}
