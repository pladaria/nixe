//! Architectural condition flags and condition-code evaluation.

/// All A64 condition encodings.
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

impl Condition {
    /// Decodes the complete four-bit Arm condition space.
    #[must_use]
    pub const fn from_encoding(encoding: u8) -> Self {
        match encoding & 0xf {
            0 => Self::Eq,
            1 => Self::Ne,
            2 => Self::Cs,
            3 => Self::Cc,
            4 => Self::Mi,
            5 => Self::Pl,
            6 => Self::Vs,
            7 => Self::Vc,
            8 => Self::Hi,
            9 => Self::Ls,
            10 => Self::Ge,
            11 => Self::Lt,
            12 => Self::Gt,
            13 => Self::Le,
            14 => Self::Al,
            15 => Self::Nv,
            _ => unreachable!(),
        }
    }

    #[must_use]
    pub const fn encoding(self) -> u8 {
        self as u8
    }
}

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
            }
            assert!(evaluate_a64(Condition::Al, packed));
            assert!(evaluate_a64(Condition::Nv, packed));
        }
    }

    #[test]
    fn nzcv_round_trips_without_other_status_bits() {
        for nibble in 0_u32..16 {
            assert_eq!(
                Nzcv::from_a64_nzcv((nibble << 28) | 0xffff).packed(),
                nibble << 28
            );
        }
    }
}
