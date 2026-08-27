//! Pure architectural decisions shared by A64 execution engines.

use crate::{
    memory::{
        BarrierAccess, BarrierDomain, BarrierOperation, CacheMaintenanceKind, MemoryAccessSize,
    },
    platform::TargetPlatform,
    semantics::shifts::ShiftKind,
};

/// Architecturally meaningful operations allocated in the A64 HINT space.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HintOperation {
    NoOperation,
    Yield,
    WaitForEvent,
    WaitForInterrupt,
    SendEvent,
    SendEventLocal,
}

/// Normalizes the baseline A64 scheduling and event hints.
///
/// Arm A-profile A64 instruction definitions (2025-12):
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions
#[must_use]
pub const fn hint_operation(immediate: u8) -> Option<HintOperation> {
    match immediate {
        0 => Some(HintOperation::NoOperation),
        1 => Some(HintOperation::Yield),
        2 => Some(HintOperation::WaitForEvent),
        3 => Some(HintOperation::WaitForInterrupt),
        4 => Some(HintOperation::SendEvent),
        5 => Some(HintOperation::SendEventLocal),
        _ => None,
    }
}

/// Runtime-owned component needed to read one user-visible A64 system register.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RuntimeRegisterRead {
    Constant(u64),
    TimerFrequency,
    TimerCounter,
}

/// Resolves the profile-dependent part of an A64 userspace system-register read.
#[must_use]
pub const fn runtime_register_read(
    platform: TargetPlatform,
    system_key: u32,
) -> Option<RuntimeRegisterRead> {
    match system_key {
        // CTR_EL0 reports log2(cache-line words) in DminLine and IminLine.
        // Switch 1's Cortex-A57 profile exposes 64-byte lines.
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/CTR-EL0--Cache-Type-Register
        0xd53b_0020 => Some(RuntimeRegisterRead::Constant((4_u64 << 16) | 4)),
        // DCZID_EL0 derives its block size and prohibition bit from the guest
        // profile rather than the host cache implementation.
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/DCZID-EL0--Data-Cache-Zero-ID-Register
        0xd53b_00e0 => {
            let bytes = platform.data_zero_block_bytes();
            let prohibited = if platform.user_cache_maintenance_prohibited() {
                1 << 4
            } else {
                0
            };
            Some(RuntimeRegisterRead::Constant(
                prohibited | (bytes.trailing_zeros() - 2) as u64,
            ))
        }
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/CNTFRQ-EL0--Counter-timer-Frequency-register
        0xd53b_e000 => Some(RuntimeRegisterRead::TimerFrequency),
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/CNTVCT-EL0--Counter-timer-Virtual-Count-register
        0xd53b_e020 => Some(RuntimeRegisterRead::TimerCounter),
        _ => None,
    }
}

/// Normalizes an A64 DSB, DMB, or ISB encoding into the engine-neutral memory
/// contract.
#[must_use]
pub const fn barrier_operation(opcode: u8, option: u8) -> Option<BarrierOperation> {
    if opcode == 6 {
        return if option == 15 {
            Some(BarrierOperation::InstructionSynchronization)
        } else {
            None
        };
    }
    if opcode != 4 && opcode != 5 {
        return None;
    }
    let access = match option & 3 {
        1 => BarrierAccess::Reads,
        2 => BarrierAccess::Writes,
        3 => BarrierAccess::ReadsAndWrites,
        _ => return None,
    };
    let domain = match option >> 2 {
        0 => BarrierDomain::OuterShareable,
        1 => BarrierDomain::NonShareable,
        2 => BarrierDomain::InnerShareable,
        3 => BarrierDomain::FullSystem,
        _ => return None,
    };
    Some(if opcode == 4 {
        BarrierOperation::DataSynchronization { domain, access }
    } else {
        BarrierOperation::DataMemory { domain, access }
    })
}

/// One userspace cache-maintenance operation implemented by the memory owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheMaintenanceOperation {
    pub kind: CacheMaintenanceKind,
    pub uses_address: bool,
}

#[must_use]
pub const fn cache_maintenance_operation(system_key: u32) -> Option<CacheMaintenanceOperation> {
    let (kind, uses_address) = match system_key {
        0xd508_7500 => (CacheMaintenanceKind::InstructionInvalidate, false),
        0xd50b_7520 => (CacheMaintenanceKind::InstructionInvalidate, true),
        0xd508_7620 => (CacheMaintenanceKind::DataInvalidate, true),
        0xd50b_7b20 => (CacheMaintenanceKind::DataClean, true),
        0xd50b_7e20 => (CacheMaintenanceKind::DataCleanAndInvalidate, true),
        _ => return None,
    };
    Some(CacheMaintenanceOperation { kind, uses_address })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadSpec {
    pub signed: bool,
    pub destination_bits: u8,
}

impl LoadSpec {
    #[must_use]
    pub const fn unsigned(size: MemoryAccessSize) -> Self {
        Self {
            signed: false,
            destination_bits: if matches!(size, MemoryAccessSize::Doubleword) {
                64
            } else {
                32
            },
        }
    }

    const fn signed(destination_bits: u8) -> Self {
        Self {
            signed: true,
            destination_bits,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarTransfer {
    Store,
    Load(LoadSpec),
}

#[must_use]
pub const fn memory_size(size_bits: u8) -> MemoryAccessSize {
    match size_bits {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        _ => panic!("A64 memory size is a two-bit field"),
    }
}

#[must_use]
pub const fn scalar_transfer(opc: u8, size: MemoryAccessSize) -> Option<ScalarTransfer> {
    match (opc, size) {
        (0, _) => Some(ScalarTransfer::Store),
        (1, _) => Some(ScalarTransfer::Load(LoadSpec::unsigned(size))),
        (2, MemoryAccessSize::Byte | MemoryAccessSize::Halfword | MemoryAccessSize::Word) => {
            Some(ScalarTransfer::Load(LoadSpec::signed(64)))
        }
        (3, MemoryAccessSize::Byte | MemoryAccessSize::Halfword) => {
            Some(ScalarTransfer::Load(LoadSpec::signed(32)))
        }
        _ => None,
    }
}

#[must_use]
pub const fn literal_load(opc: u8) -> Option<(MemoryAccessSize, LoadSpec)> {
    match opc {
        0 => Some((
            MemoryAccessSize::Word,
            LoadSpec::unsigned(MemoryAccessSize::Word),
        )),
        1 => Some((
            MemoryAccessSize::Doubleword,
            LoadSpec::unsigned(MemoryAccessSize::Doubleword),
        )),
        2 => Some((MemoryAccessSize::Word, LoadSpec::signed(64))),
        _ => None,
    }
}

#[must_use]
pub const fn pair_transfer(size_bits: u8, load: bool) -> Option<(MemoryAccessSize, LoadSpec)> {
    match (size_bits, load) {
        (0, _) => Some((
            MemoryAccessSize::Word,
            LoadSpec::unsigned(MemoryAccessSize::Word),
        )),
        (1, true) => Some((MemoryAccessSize::Word, LoadSpec::signed(64))),
        (2, _) => Some((
            MemoryAccessSize::Doubleword,
            LoadSpec::unsigned(MemoryAccessSize::Doubleword),
        )),
        _ => None,
    }
}

#[must_use]
pub const fn shift_kind(encoded: u8, allow_rotate: bool) -> Option<ShiftKind> {
    match encoded {
        0 => Some(ShiftKind::LogicalLeft),
        1 => Some(ShiftKind::LogicalRight),
        2 => Some(ShiftKind::ArithmeticRight),
        3 if allow_rotate => Some(ShiftKind::RotateRight),
        _ => None,
    }
}

#[must_use]
pub const fn signed_immediate(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_transfer_table_covers_legal_boundaries() {
        use MemoryAccessSize::{Doubleword, Halfword, Word};
        use ScalarTransfer::{Load, Store};
        let unsigned32 = Load(LoadSpec::unsigned(Word));
        let cases = [
            (0, Doubleword, Some(Store)),
            (1, Word, Some(unsigned32)),
            (2, Word, Some(Load(LoadSpec::signed(64)))),
            (2, Doubleword, None),
            (3, Halfword, Some(Load(LoadSpec::signed(32)))),
            (3, Word, None),
        ];
        for (opc, size, expected) in cases {
            assert_eq!(scalar_transfer(opc, size), expected);
        }
    }

    #[test]
    fn signed_immediates_and_shift_kinds_are_width_aware() {
        assert_eq!(signed_immediate(0x1ff, 9), -1);
        assert_eq!(signed_immediate(0x100, 9), -256);
        assert_eq!(shift_kind(3, false), None);
        assert_eq!(shift_kind(3, true), Some(ShiftKind::RotateRight));
    }
}
