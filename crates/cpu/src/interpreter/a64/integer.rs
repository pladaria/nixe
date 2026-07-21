use crate::{
    decode::{
        DecodedOpcode,
        a64::integer::{Instruction, Operands},
    },
    ir::op::Condition,
    location::{DecodedInstruction, LocationDescriptor},
    semantics::{
        arithmetic::{add_with_carry, subtract_with_carry},
        bits::{BitWidth, rotate_right},
        conditions::evaluate_a64,
        immediate::{decode_a64_bit_masks, decode_a64_logical_immediate},
        shifts::{ShiftKind, a64_shift_with_carry},
    },
    state::a64::{A64State, Nzcv},
};

use super::{advance, read, resume, sign_extend, write};
use crate::interpreter::{InterpreterError, InterpreterOutcome};

pub(super) fn execute(
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    let fields = instruction.operands();
    let supported = match instruction {
        Instruction::MoveWide(_) => move_wide(state, fields),
        Instruction::AddSubImmediate(_) => add_sub_immediate(state, fields),
        Instruction::AddSubShifted(_) => add_sub_shifted(state, fields),
        Instruction::AddSubExtended(_) => add_sub_extended(state, fields),
        Instruction::AddSubCarry(_) => add_sub_carry(state, fields),
        Instruction::LogicalImmediate(_) => logical_immediate(state, fields),
        Instruction::LogicalShifted(_) => logical_shifted(state, fields),
        Instruction::Bitfield(_) => bitfield(state, fields),
        Instruction::Extract(_) => extract(state, fields),
        Instruction::TwoSource(_) => two_source(state, fields),
        Instruction::ConditionalCompareRegister(_)
        | Instruction::ConditionalCompareImmediate(_) => conditional_compare(state, fields),
        Instruction::ConditionalSelect(_) => conditional_select(state, fields),
        Instruction::ThreeSource(_) => three_source(state, fields),
        Instruction::OneSource(_) => one_source(state, fields),
        Instruction::Adr(_) => adr(state, decoded.location, fields, false),
        Instruction::Adrp(_) => adr(state, decoded.location, fields, true),
    };
    if !supported {
        return Err(super::super::unsupported(decoded));
    }
    advance(state);
    Ok(resume(state, decoded))
}

fn width(fields: Operands) -> u8 {
    if fields.width_64 { 64 } else { 32 }
}

fn bit_width(fields: Operands) -> BitWidth {
    BitWidth::new(width(fields)).expect("A64 integer width is 32 or 64")
}

fn mask(fields: Operands) -> u64 {
    if fields.width_64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    }
}

fn set_arithmetic_flags(state: &mut A64State, result: u64, carry: bool, overflow: bool, bits: u8) {
    let sign = 1_u64 << (bits - 1);
    let packed = if result & sign != 0 { Nzcv::N } else { 0 }
        | if result == 0 { Nzcv::Z } else { 0 }
        | if carry { Nzcv::C } else { 0 }
        | if overflow { Nzcv::V } else { 0 };
    state.set_nzcv(Nzcv::from_bits(packed));
}

fn set_logical_flags(state: &mut A64State, result: u64, bits: u8) {
    let sign = 1_u64 << (bits - 1);
    state.set_nzcv(Nzcv::from_bits(
        if result & sign != 0 { Nzcv::N } else { 0 } | if result == 0 { Nzcv::Z } else { 0 },
    ));
}

fn move_wide(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let halfword = u32::from(fields.opcode_2);
    if bits == 32 && halfword >= 2 {
        return false;
    }
    let shift = halfword * 16;
    let immediate = u64::from(fields.immediate_16) << shift;
    let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
    let value = match opcode {
        0 => !immediate & mask(fields),
        2 => immediate,
        3 => {
            let old = read(state, fields.rd, bits, false);
            (old & !(0xffff_u64 << shift)) | immediate
        }
        _ => return false,
    };
    write(state, fields.rd, bits, false, value);
    true
}

fn apply_add_sub(state: &mut A64State, fields: Operands, lhs: u64, rhs: u64, carry: bool) {
    let bit_width = bit_width(fields);
    let result = if fields.subtract {
        subtract_with_carry(u128::from(lhs), u128::from(rhs), carry, bit_width)
    } else {
        add_with_carry(u128::from(lhs), u128::from(rhs), carry, bit_width)
    };
    let value = result.result as u64;
    write(
        state,
        fields.rd,
        width(fields),
        !fields.set_flags && matches!(fields.rd, 31),
        value,
    );
    if fields.set_flags {
        set_arithmetic_flags(
            state,
            value,
            result.carry_out,
            result.overflow,
            width(fields),
        );
    }
}

fn add_sub_immediate(state: &mut A64State, fields: Operands) -> bool {
    let lhs = read(state, fields.rn, width(fields), true);
    let rhs = u64::from(fields.immediate_12) << if fields.n { 12 } else { 0 };
    apply_add_sub(state, fields, lhs, rhs, fields.subtract);
    true
}

fn shifted(value: u64, fields: Operands) -> Option<u64> {
    if (width(fields) == 32 && fields.shift_amount >= 32) || fields.shift_kind == 3 {
        return None;
    }
    let kind = match fields.shift_kind {
        0 => ShiftKind::LogicalLeft,
        1 => ShiftKind::LogicalRight,
        2 => ShiftKind::ArithmeticRight,
        _ => return None,
    };
    Some(
        a64_shift_with_carry(
            u128::from(value),
            bit_width(fields),
            kind,
            u32::from(fields.shift_amount),
            false,
        )
        .expect("validated A64 width")
        .result as u64,
    )
}

fn add_sub_shifted(state: &mut A64State, fields: Operands) -> bool {
    let Some(rhs) = shifted(read(state, fields.rm, width(fields), false), fields) else {
        return false;
    };
    let lhs = read(state, fields.rn, width(fields), false);
    apply_add_sub(state, fields, lhs, rhs, fields.subtract);
    true
}

fn extend_register(value: u64, option: u8, shift: u8, destination_width: u8) -> Option<u64> {
    if shift > 4 || option > 7 {
        return None;
    }
    let source_width = match option & 3 {
        0 => 8,
        1 => 16,
        2 => 32,
        3 => 64,
        _ => unreachable!(),
    };
    if source_width == 64 && destination_width == 32 {
        return None;
    }
    let source_mask = if source_width == 64 {
        u64::MAX
    } else {
        (1_u64 << source_width) - 1
    };
    let truncated = value & source_mask;
    let extended = if option & 4 == 0 || source_width == 64 {
        truncated
    } else {
        sign_extend(truncated, source_width) as u64
    };
    Some(
        (extended << shift)
            & if destination_width == 64 {
                u64::MAX
            } else {
                u64::from(u32::MAX)
            },
    )
}

fn add_sub_extended(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let rm = read(state, fields.rm, bits, false);
    let Some(rhs) = extend_register(rm, fields.extension, fields.small_shift, bits) else {
        return false;
    };
    let lhs = read(state, fields.rn, bits, true);
    apply_add_sub(state, fields, lhs, rhs, fields.subtract);
    true
}

fn add_sub_carry(state: &mut A64State, fields: Operands) -> bool {
    let lhs = read(state, fields.rn, width(fields), false);
    let rhs = read(state, fields.rm, width(fields), false);
    apply_add_sub(state, fields, lhs, rhs, state.nzcv().carry());
    true
}

fn logical(opcode: u8, lhs: u64, rhs: u64) -> u64 {
    match opcode {
        0 | 3 => lhs & rhs,
        1 => lhs | rhs,
        2 => lhs ^ rhs,
        _ => unreachable!(),
    }
}

fn logical_immediate(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let Ok(immediate) =
        decode_a64_logical_immediate(fields.n, fields.immediate_6_high, fields.shift_amount, bits)
    else {
        return false;
    };
    let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
    let result = logical(opcode, read(state, fields.rn, bits, false), immediate) & mask(fields);
    write(state, fields.rd, bits, false, result);
    if opcode == 3 {
        set_logical_flags(state, result, bits);
    }
    true
}

fn logical_shifted(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let Some(mut rhs) = shifted(read(state, fields.rm, bits, false), fields) else {
        return false;
    };
    if fields.invert {
        rhs = !rhs & mask(fields);
    }
    let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
    let result = logical(opcode, read(state, fields.rn, bits, false), rhs) & mask(fields);
    write(state, fields.rd, bits, false, result);
    if opcode == 3 {
        set_logical_flags(state, result, bits);
    }
    true
}

fn bitfield(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    if fields.n != fields.width_64 || (!fields.width_64 && fields.subtract_product) {
        return false;
    }
    let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
    if opcode == 3 {
        return false;
    }
    let Ok(masks) = decode_a64_bit_masks(
        fields.n,
        fields.immediate_6_high,
        fields.shift_amount,
        bits,
        false,
    ) else {
        return false;
    };
    let source = read(state, fields.rn, bits, false);
    let rotated = rotate_right(
        u128::from(source),
        bit_width(fields),
        u32::from(fields.immediate_6_high),
    ) as u64;
    let bottom = match opcode {
        1 => {
            (read(state, fields.rd, bits, false) & !masks.write_mask) | (rotated & masks.write_mask)
        }
        _ => rotated & masks.write_mask,
    };
    let top = if opcode == 0 && source & (1_u64 << fields.shift_amount) != 0 {
        mask(fields)
    } else {
        0
    };
    let result = ((top & !masks.test_mask) | (bottom & masks.test_mask)) & mask(fields);
    write(state, fields.rd, bits, false, result);
    true
}

fn extract(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let lsb = u32::from(fields.shift_amount);
    if fields.n != fields.width_64 || (bits == 32 && lsb >= 32) {
        return false;
    }
    let low = read(state, fields.rm, bits, false);
    let high = read(state, fields.rn, bits, false);
    let result = if lsb == 0 {
        low
    } else {
        (low >> lsb) | (high << (u32::from(bits) - lsb))
    } & mask(fields);
    write(state, fields.rd, bits, false, result);
    true
}

fn two_source(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let lhs = read(state, fields.rn, bits, false);
    let rhs = read(state, fields.rm, bits, false);
    let shift = u32::from((rhs & u64::from(bits - 1)) as u8);
    let result = match fields.shift_amount {
        2 => lhs.checked_div(rhs).unwrap_or(0),
        3 => signed_divide(lhs, rhs, bits),
        8 => lhs.wrapping_shl(shift),
        9 => lhs.wrapping_shr(shift),
        10 => arithmetic_shift_right(lhs, shift, bits),
        11 => rotate_right(u128::from(lhs), bit_width(fields), shift) as u64,
        _ => return false,
    } & mask(fields);
    write(state, fields.rd, bits, false, result);
    true
}

fn signed_divide(lhs: u64, rhs: u64, bits: u8) -> u64 {
    if bits == 32 {
        let lhs = lhs as u32 as i32;
        let rhs = rhs as u32 as i32;
        if rhs == 0 {
            0
        } else {
            lhs.wrapping_div(rhs) as u32 as u64
        }
    } else {
        let lhs = lhs as i64;
        let rhs = rhs as i64;
        if rhs == 0 {
            0
        } else {
            lhs.wrapping_div(rhs) as u64
        }
    }
}

fn arithmetic_shift_right(value: u64, amount: u32, bits: u8) -> u64 {
    if bits == 32 {
        ((value as u32 as i32) >> amount) as u32 as u64
    } else {
        ((value as i64) >> amount) as u64
    }
}

fn conditional_compare(state: &mut A64State, fields: Operands) -> bool {
    if !evaluate_a64(
        Condition::from_encoding(fields.condition),
        state.nzcv().bits(),
    ) {
        state.set_nzcv(Nzcv::from_bits(u32::from(fields.nzcv) << 28));
        return true;
    }
    let bits = width(fields);
    let lhs = read(state, fields.rn, bits, false);
    let rhs = if fields.immediate_form {
        u64::from(fields.rm)
    } else {
        read(state, fields.rm, bits, false)
    };
    let result = if fields.subtract {
        subtract_with_carry(u128::from(lhs), u128::from(rhs), true, bit_width(fields))
    } else {
        add_with_carry(u128::from(lhs), u128::from(rhs), false, bit_width(fields))
    };
    set_arithmetic_flags(
        state,
        result.result as u64,
        result.carry_out,
        result.overflow,
        bits,
    );
    true
}

fn conditional_select(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let value = if evaluate_a64(
        Condition::from_encoding(fields.condition),
        state.nzcv().bits(),
    ) {
        read(state, fields.rn, bits, false)
    } else {
        let mut value = read(state, fields.rm, bits, false);
        if fields.subtract {
            value = !value;
        }
        if fields.bit10 {
            value = value.wrapping_add(1);
        }
        value & mask(fields)
    };
    write(state, fields.rd, bits, false, value);
    true
}

fn three_source(state: &mut A64State, fields: Operands) -> bool {
    let opcode = fields.opcode_3;
    let result = match opcode {
        0 => {
            let bits = width(fields);
            let product = read(state, fields.rn, bits, false)
                .wrapping_mul(read(state, fields.rm, bits, false));
            let addend = read(state, fields.ra, bits, false);
            let value = if fields.subtract_product {
                addend.wrapping_sub(product)
            } else {
                addend.wrapping_add(product)
            } & mask(fields);
            write(state, fields.rd, bits, false, value);
            return true;
        }
        1 => {
            let product = i64::from(read(state, fields.rn, 32, false) as u32 as i32)
                .wrapping_mul(i64::from(read(state, fields.rm, 32, false) as u32 as i32));
            let addend = read(state, fields.ra, 64, false) as i64;
            if fields.subtract_product {
                addend.wrapping_sub(product) as u64
            } else {
                addend.wrapping_add(product) as u64
            }
        }
        2 if fields.ra == 31 && !fields.subtract_product => {
            let product = i128::from(read(state, fields.rn, 64, false) as i64)
                * i128::from(read(state, fields.rm, 64, false) as i64);
            (product >> 64) as u64
        }
        5 => {
            let product = u64::from(read(state, fields.rn, 32, false) as u32)
                .wrapping_mul(u64::from(read(state, fields.rm, 32, false) as u32));
            let addend = read(state, fields.ra, 64, false);
            if fields.subtract_product {
                addend.wrapping_sub(product)
            } else {
                addend.wrapping_add(product)
            }
        }
        6 if fields.ra == 31 && !fields.subtract_product => {
            let product = u128::from(read(state, fields.rn, 64, false))
                * u128::from(read(state, fields.rm, 64, false));
            (product >> 64) as u64
        }
        _ => return false,
    };
    write(state, fields.rd, 64, false, result);
    true
}

fn one_source(state: &mut A64State, fields: Operands) -> bool {
    let bits = width(fields);
    let input = read(state, fields.rn, bits, false);
    let result = match fields.shift_amount {
        0 => {
            if bits == 64 {
                input.reverse_bits()
            } else {
                u64::from((input as u32).reverse_bits())
            }
        }
        1 => reverse_chunks(input, bits, 2),
        2 => reverse_chunks(input, bits, 4),
        3 => {
            if bits == 64 {
                input.swap_bytes()
            } else {
                u64::from((input as u32).swap_bytes())
            }
        }
        4 => {
            if bits == 64 {
                u64::from(input.leading_zeros())
            } else {
                u64::from((input as u32).leading_zeros())
            }
        }
        5 => {
            let count = if bits == 64 {
                if input >> 63 == 0 {
                    input.leading_zeros()
                } else {
                    (!input).leading_zeros()
                }
            } else {
                let value = input as u32;
                if value >> 31 == 0 {
                    value.leading_zeros()
                } else {
                    (!value).leading_zeros()
                }
            };
            u64::from(count.saturating_sub(1))
        }
        _ => return false,
    };
    write(state, fields.rd, bits, false, result);
    true
}

fn reverse_chunks(value: u64, bits: u8, chunk_bytes: usize) -> u64 {
    let mut bytes = value.to_le_bytes();
    for chunk in bytes[..usize::from(bits / 8)].chunks_mut(chunk_bytes) {
        chunk.reverse();
    }
    u64::from_le_bytes(bytes)
}

fn adr(state: &mut A64State, source: LocationDescriptor, fields: Operands, page: bool) -> bool {
    let displacement = sign_extend(u64::from(fields.adr_immediate), 21);
    let base = if page {
        source.pc.get() & !0xfff
    } else {
        source.pc.get()
    };
    let value = base.wrapping_add_signed(if page {
        displacement << 12
    } else {
        displacement
    });
    write(state, fields.rd, 64, false, value);
    true
}
