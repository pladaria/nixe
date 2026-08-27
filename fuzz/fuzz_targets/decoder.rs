#![no_main]

use std::collections::BTreeSet;

use libfuzzer_sys::fuzz_target;
use nixe_cpu::{
    decode::{self, DecodeResult, OperandValue},
    location::{InstructionEncoding, LocationDescriptor},
    platform::TargetPlatform,
    semantics::{
        bits::{self, BitWidth},
        immediate,
        shifts::{self, ShiftKind},
    },
};
use nixe_memory::GuestVirtualAddress;

fuzz_target!(|data: &[u8]| {
    if data.len() > 4_096 {
        return;
    }

    let mut input = [0_u8; 20];
    let copied = data.len().min(input.len());
    input[..copied].copy_from_slice(&data[..copied]);

    let bits = u32::from_le_bytes(input[0..4].try_into().unwrap());
    let encoding = InstructionEncoding::from_u32(bits);
    let platform = if input[5] & 1 == 0 {
        TargetPlatform::Switch1
    } else {
        TargetPlatform::Switch2
    };
    let location =
        LocationDescriptor::new(GuestVirtualAddress::new(0x1000), platform.profile_id());

    match decode::decode(platform, location, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            let operands = decoded.instruction.operands();
            assert!(operands.len() <= 8);
            let mut identities = BTreeSet::new();
            for (identity, value) in operands.iter() {
                assert!(identities.insert(identity));
                if let OperandValue::Register { class, index } = value {
                    assert_eq!(class, decode::RegisterClass::A64General);
                    assert!(index < 32);
                }
            }
            assert!(decode::disassemble(&decoded.instruction).to_string().len() <= 512);
            let _ = decode::a64::normalize(&decoded.instruction, encoding);
        }
        DecodeResult::Unallocated { .. } | DecodeResult::Reserved { .. } => {}
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

    let carry = input[14] & 1 != 0;
    let _ = immediate::decode_a64_bit_masks(
        input[15] & 1 != 0,
        input[16],
        input[17],
        input[18],
        input[19] & 1 != 0,
    );
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
});
