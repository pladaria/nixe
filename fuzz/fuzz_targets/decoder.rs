#![no_main]

use std::collections::BTreeSet;

use libfuzzer_sys::fuzz_target;
use swiitx_cpu::{
    address::GuestVirtualAddress,
    decode::{self, DecodeResult, OperandValue},
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    profile::GuestCpuProfile,
    semantics::{
        bits::{self, BitWidth},
        immediate,
        shifts::{self, A32ShiftKind, ShiftKind},
    },
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 4_096 {
        return;
    }

    let mut input = [0_u8; 20];
    let copied = data.len().min(input.len());
    input[..copied].copy_from_slice(&data[..copied]);

    let bits = u32::from_le_bytes(input[0..4].try_into().unwrap());
    let state = match input[4] % 3 {
        0 => ExecutionState::A64,
        1 => ExecutionState::A32,
        _ => ExecutionState::T32,
    };
    let encoding = match state {
        ExecutionState::T32 if input[5] & 1 == 0 => InstructionEncoding::from_u16(bits as u16),
        _ => InstructionEncoding::from_u32(bits),
    };
    let profile = if state == ExecutionState::A64 && input[5] & 2 != 0 {
        GuestCpuProfile::switch_2_native()
    } else {
        GuestCpuProfile::switch_1()
    };
    let location = LocationDescriptor::new(GuestVirtualAddress::new(0x1000), state, profile.id());

    match decode::decode(&profile, location, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            let operands = decoded.instruction.operands();
            assert!(operands.len() <= 8);
            let mut identities = BTreeSet::new();
            for (identity, value) in operands.iter() {
                assert!(identities.insert(identity));
                if let OperandValue::Register { class, index } = value {
                    let maximum = match class {
                        decode::RegisterClass::A64General => 32,
                        decode::RegisterClass::A32General => 16,
                    };
                    assert!(index < maximum);
                }
            }
            assert!(decode::disassemble(&decoded.instruction).to_string().len() <= 512);
            match state {
                ExecutionState::A64 => {
                    let _ = decode::a64::normalize(&decoded.instruction, encoding);
                }
                ExecutionState::A32 => {
                    let _ = decode::a32::normalize(&decoded.instruction, encoding);
                }
                ExecutionState::T32 => {
                    let _ = decode::t32::normalize(&decoded.instruction, encoding);
                }
            }
        }
        DecodeResult::Unallocated { .. }
        | DecodeResult::Reserved { .. }
        | DecodeResult::ProfileDisabled { .. } => {}
    }

    let source_width = BitWidth::new(input[6]);
    let destination_width = BitWidth::new(input[7]);
    if let (Ok(source_width), Ok(destination_width)) = (source_width, destination_width) {
        let value = u128::from_le_bytes(input[4..20].try_into().unwrap());
        let _ = bits::extract(value, source_width, input[8], destination_width);
        let _ = bits::insert(value, source_width, !value, input[9], destination_width);
        let _ = bits::sign_extend(value, source_width, destination_width);
        let _ = bits::replicate(value, source_width, destination_width);
        let _ = bits::rotate_left(value, source_width, u32::from(input[10]));
        let _ = bits::rotate_right(value, source_width, u32::from(input[11]));
    }

    let immediate_bits = u16::from_le_bytes([input[12], input[13]]);
    let carry = input[14] & 1 != 0;
    let _ = immediate::decode_a64_bit_masks(
        input[15] & 1 != 0,
        input[16],
        input[17],
        input[18],
        input[19] & 1 != 0,
    );
    let _ = immediate::expand_a32_modified_immediate(immediate_bits, carry);
    let _ = immediate::expand_t32_modified_immediate(immediate_bits, carry);
    if let Ok(width) = BitWidth::new(input[18]) {
        let kind = match input[19] % 4 {
            0 => ShiftKind::LogicalLeft,
            1 => ShiftKind::LogicalRight,
            2 => ShiftKind::ArithmeticRight,
            _ => ShiftKind::RotateRight,
        };
        let _ = shifts::a64_shift_with_carry(
            u128::from(bits),
            width,
            kind,
            u32::from(input[17]),
            carry,
        );
    }
    let a32_kind = match input[18] % 5 {
        0 => A32ShiftKind::LogicalLeft,
        1 => A32ShiftKind::LogicalRight,
        2 => A32ShiftKind::ArithmeticRight,
        3 => A32ShiftKind::RotateRight,
        _ => A32ShiftKind::RotateRightExtended,
    };
    let _ = shifts::a32_shift_with_carry(bits, a32_kind, u32::from(input[19]), carry);
    let _ = shifts::decode_a32_immediate_shift(input[18], input[19]);
});
