//! Pure architectural decisions shared by A64 execution engines.

use crate::{
    decode::a64::fp_simd,
    memory::{
        AtomicRmwKind, BarrierAccess, BarrierDomain, BarrierOperation, CacheMaintenanceKind,
        MemoryAccessSize, MemoryOrdering,
    },
    platform::TargetPlatform,
    semantics::shifts::ShiftKind,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdMemoryMode {
    Multiple,
    Lane(u8),
    Replicate,
}

/// Compile-time architectural shape of one Advanced SIMD structure transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimdMemoryShape {
    pub element_size: MemoryAccessSize,
    pub vector_bytes: u8,
    pub repetitions: u8,
    pub elements_per_register: u8,
    pub structure_registers: u8,
    pub mode: SimdMemoryMode,
    pub transfer_bytes: u8,
    pub immediate_post_index: u8,
}

impl SimdMemoryShape {
    #[must_use]
    pub const fn register_count(self) -> u8 {
        self.repetitions * self.structure_registers
    }
}

#[must_use]
pub const fn simd_multiple_structure_shape(fields: fp_simd::Operands) -> Option<SimdMemoryShape> {
    let (structure_registers, repetitions) = match fields.structure_opcode {
        0 => (4, 1),
        2 => (1, 4),
        4 => (3, 1),
        6 => (1, 3),
        7 => (1, 1),
        8 => (2, 1),
        10 => (1, 2),
        _ => return None,
    };
    let element_size = match fields.element_size {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        _ => return None,
    };
    let vector_bytes = if fields.vector_128 { 16 } else { 8 };
    let elements_per_register = vector_bytes / element_size.bytes() as u8;
    let transfer_bytes = vector_bytes * structure_registers * repetitions;
    Some(SimdMemoryShape {
        element_size,
        vector_bytes,
        repetitions,
        elements_per_register,
        structure_registers,
        mode: SimdMemoryMode::Multiple,
        transfer_bytes,
        immediate_post_index: transfer_bytes,
    })
}

#[must_use]
pub const fn simd_single_structure_shape(fields: fp_simd::Operands) -> Option<SimdMemoryShape> {
    let opcode = fields.structure_opcode >> 1;
    let structure_registers = 1 + fields.structure_r as u8 + 2 * (opcode & 1);
    let vector_bytes = if fields.vector_128 { 16 } else { 8 };
    let (element_size, mode) = match opcode {
        0 | 1 => (
            MemoryAccessSize::Byte,
            SimdMemoryMode::Lane(
                ((fields.vector_128 as u8) << 3)
                    | ((fields.structure_opcode & 1) << 2)
                    | fields.element_size,
            ),
        ),
        2 | 3 if fields.element_size & 1 == 0 => (
            MemoryAccessSize::Halfword,
            SimdMemoryMode::Lane(
                ((fields.vector_128 as u8) << 2)
                    | ((fields.structure_opcode & 1) << 1)
                    | (fields.element_size >> 1),
            ),
        ),
        4 | 5 if fields.element_size == 0 => (
            MemoryAccessSize::Word,
            SimdMemoryMode::Lane(((fields.vector_128 as u8) << 1) | (fields.structure_opcode & 1)),
        ),
        4 | 5 if fields.element_size == 1 && fields.structure_opcode & 1 == 0 => (
            MemoryAccessSize::Doubleword,
            SimdMemoryMode::Lane(fields.vector_128 as u8),
        ),
        6 | 7 if fields.load => {
            let element_size = match fields.element_size {
                0 => MemoryAccessSize::Byte,
                1 => MemoryAccessSize::Halfword,
                2 => MemoryAccessSize::Word,
                3 => MemoryAccessSize::Doubleword,
                _ => return None,
            };
            (element_size, SimdMemoryMode::Replicate)
        }
        _ => return None,
    };
    let elements_per_register = vector_bytes / element_size.bytes() as u8;
    let transfer_bytes = structure_registers * element_size.bytes() as u8;
    Some(SimdMemoryShape {
        element_size,
        vector_bytes,
        repetitions: 1,
        elements_per_register,
        structure_registers,
        mode,
        transfer_bytes,
        immediate_post_index: transfer_bytes,
    })
}

#[must_use]
pub const fn simd_pair_access_size(size_bits: u8) -> Option<MemoryAccessSize> {
    match size_bits {
        0 => Some(MemoryAccessSize::Word),
        1 => Some(MemoryAccessSize::Doubleword),
        2 => Some(MemoryAccessSize::Quadword),
        _ => None,
    }
}

#[must_use]
pub const fn simd_memory_access_size(size_bits: u8, opcode: u8) -> Option<MemoryAccessSize> {
    match size_bits + ((opcode & 2) << 1) {
        0 => Some(MemoryAccessSize::Byte),
        1 => Some(MemoryAccessSize::Halfword),
        2 => Some(MemoryAccessSize::Word),
        3 => Some(MemoryAccessSize::Doubleword),
        4 => Some(MemoryAccessSize::Quadword),
        _ => None,
    }
}

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

/// Normalizes baseline A64 hints and profile-owned compatibility aliases.
///
/// Arm A-profile A64 instruction definitions (2025-12):
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions
#[must_use]
pub const fn hint_operation(platform: TargetPlatform, immediate: u8) -> Option<HintOperation> {
    match immediate {
        0 => Some(HintOperation::NoOperation),
        1 => Some(HintOperation::Yield),
        2 => Some(HintOperation::WaitForEvent),
        3 => Some(HintOperation::WaitForInterrupt),
        4 => Some(HintOperation::SendEvent),
        5 => Some(HintOperation::SendEventLocal),
        // These backwards-compatible Pointer Authentication aliases occupy
        // HINT space and therefore execute as NOP on Switch 1's Cortex-A57,
        // which predates FEAT_PAuth. Keep Switch 2 explicit until its CPU
        // feature profile and PAC state are modeled instead of silently
        // discarding an operation that may alter X30 there.
        7 | 8 | 10 | 12 | 14 | 24..=31 if matches!(platform, TargetPlatform::Switch1) => {
            Some(HintOperation::NoOperation)
        }
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
pub const fn atomic_ordering(acquire: bool, release: bool) -> MemoryOrdering {
    match (acquire, release) {
        (false, false) => MemoryOrdering::Relaxed,
        (true, false) => MemoryOrdering::Acquire,
        (false, true) => MemoryOrdering::Release,
        (true, true) => MemoryOrdering::AcquireRelease,
    }
}

#[must_use]
pub const fn atomic_rmw_kind(opcode: u8) -> Option<AtomicRmwKind> {
    match opcode {
        0 => Some(AtomicRmwKind::Add),
        1 => Some(AtomicRmwKind::Clear),
        2 => Some(AtomicRmwKind::Xor),
        3 => Some(AtomicRmwKind::Set),
        4 => Some(AtomicRmwKind::SignedMaximum),
        5 => Some(AtomicRmwKind::SignedMinimum),
        6 => Some(AtomicRmwKind::UnsignedMaximum),
        7 => Some(AtomicRmwKind::UnsignedMinimum),
        8 => Some(AtomicRmwKind::Swap),
        _ => None,
    }
}

#[must_use]
pub const fn compare_exchange_pair_sizes(
    size_bits: u8,
) -> Option<(MemoryAccessSize, MemoryAccessSize)> {
    match size_bits {
        0 => Some((MemoryAccessSize::Word, MemoryAccessSize::Doubleword)),
        1 => Some((MemoryAccessSize::Doubleword, MemoryAccessSize::Quadword)),
        _ => None,
    }
}

#[must_use]
pub const fn exclusive_transfer_sizes(
    size_bits: u8,
    pair: bool,
) -> Option<(MemoryAccessSize, MemoryAccessSize)> {
    if !pair {
        let size = memory_size(size_bits);
        return Some((size, size));
    }
    match size_bits {
        2 => Some((MemoryAccessSize::Word, MemoryAccessSize::Doubleword)),
        3 => Some((MemoryAccessSize::Doubleword, MemoryAccessSize::Quadword)),
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
    fn switch_1_pointer_authentication_hint_family_is_compatibility_nop_space() {
        for immediate in [7, 8, 10, 12, 14, 24, 25, 26, 27, 28, 29, 30, 31] {
            assert_eq!(
                hint_operation(TargetPlatform::Switch1, immediate),
                Some(HintOperation::NoOperation)
            );
            assert_eq!(hint_operation(TargetPlatform::Switch2, immediate), None);
        }
    }

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
    fn simd_access_size_tables_cover_the_complete_encoding_space() {
        use MemoryAccessSize::{Byte, Doubleword, Halfword, Quadword, Word};

        assert_eq!(simd_pair_access_size(0), Some(Word));
        assert_eq!(simd_pair_access_size(1), Some(Doubleword));
        assert_eq!(simd_pair_access_size(2), Some(Quadword));
        assert_eq!(simd_pair_access_size(3), None);

        for size_bits in 0..=3 {
            for opcode in 0..=3 {
                let encoded = size_bits + ((opcode & 2) << 1);
                let expected = match encoded {
                    0 => Some(Byte),
                    1 => Some(Halfword),
                    2 => Some(Word),
                    3 => Some(Doubleword),
                    4 => Some(Quadword),
                    _ => None,
                };
                assert_eq!(simd_memory_access_size(size_bits, opcode), expected);
            }
        }
    }

    #[test]
    fn simd_structure_shapes_cover_every_allocated_normalized_form() {
        for vector_128 in [false, true] {
            for element_size in 0..=3 {
                for structure_opcode in 0..=15 {
                    let mut fields = fp_simd::Operands::empty();
                    fields.vector_128 = vector_128;
                    fields.element_size = element_size;
                    fields.structure_opcode = structure_opcode;
                    let shape = simd_multiple_structure_shape(fields);
                    assert_eq!(
                        shape.is_some(),
                        matches!(structure_opcode, 0 | 2 | 4 | 6 | 7 | 8 | 10),
                        "multiple structure opcode={structure_opcode} size={element_size} q={vector_128}"
                    );
                    if let Some(shape) = shape {
                        assert_eq!(shape.mode, SimdMemoryMode::Multiple);
                        assert_eq!(
                            shape.transfer_bytes,
                            shape.vector_bytes * shape.register_count()
                        );
                        assert_eq!(shape.immediate_post_index, shape.transfer_bytes);
                        assert_eq!(
                            shape.elements_per_register,
                            shape.vector_bytes / shape.element_size.bytes() as u8
                        );
                    }
                }
            }
        }

        for load in [false, true] {
            for vector_128 in [false, true] {
                for structure_r in [false, true] {
                    for opcode in 0..=7 {
                        for s in 0..=1 {
                            for element_size in 0..=3 {
                                let mut fields = fp_simd::Operands::empty();
                                fields.load = load;
                                fields.vector_128 = vector_128;
                                fields.structure_r = structure_r;
                                fields.structure_opcode = (opcode << 1) | s;
                                fields.element_size = element_size;
                                let shape = simd_single_structure_shape(fields);
                                let allocated = match opcode {
                                    0 | 1 => true,
                                    2 | 3 => element_size & 1 == 0,
                                    4 | 5 => element_size == 0 || (element_size == 1 && s == 0),
                                    6 | 7 => load,
                                    _ => unreachable!(),
                                };
                                assert_eq!(
                                    shape.is_some(),
                                    allocated,
                                    "single structure load={load} opcode={opcode} s={s} size={element_size} q={vector_128} r={structure_r}"
                                );
                                if let Some(shape) = shape {
                                    assert!((1..=4).contains(&shape.register_count()));
                                    assert_eq!(
                                        shape.transfer_bytes,
                                        shape.register_count() * shape.element_size.bytes() as u8
                                    );
                                    assert_eq!(shape.immediate_post_index, shape.transfer_bytes);
                                    match shape.mode {
                                        SimdMemoryMode::Lane(lane) => {
                                            assert!(lane < shape.elements_per_register);
                                        }
                                        SimdMemoryMode::Replicate => assert!(load),
                                        SimdMemoryMode::Multiple => unreachable!(),
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
