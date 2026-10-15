//! Maxwell condition tests lowered to boolean-valued neutral registers.
use super::*;
use crate::shader::decoded::Operation as ShaderOperation;

/// Absent flags have no translated reaching definition. Consumers must not
/// invent reset values; these registers are shared by all ISA flag producers.
#[derive(Default)]
pub(in crate::shader) struct IntegerConditionCodes {
    pub(in crate::shader) zero: Option<ShaderRegister>,
    pub(in crate::shader) sign: Option<ShaderRegister>,
    pub(in crate::shader) carry: Option<ShaderRegister>,
    pub(in crate::shader) overflow: Option<ShaderRegister>,
}

pub(in crate::shader) const fn is_condition_set_predicate(word: u64) -> bool {
    word & 0xfff8_0000_0000_0000 == 0x50a0_0000_0000_0000
}

// Flow-test values and Z/S/C/O truth functions follow the public Maxwell IR:
// https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/ir/ir_emitter.cpp (GetFlowTest)
#[derive(Clone, Copy)]
enum Test {
    Constant(bool),
    Flag(u8),
    Not(&'static Test),
    Binary(ShaderBitwiseOperation, &'static Test, &'static Test),
}
use Test::{Constant, Flag, Not};
const Z: Test = Flag(0);
const S: Test = Flag(1);
const C: Test = Flag(2);
const O: Test = Flag(3);
macro_rules! and {
    ($a:expr, $b:expr) => {
        Test::Binary(ShaderBitwiseOperation::And, &$a, &$b)
    };
}
macro_rules! or {
    ($a:expr, $b:expr) => {
        Test::Binary(ShaderBitwiseOperation::Or, &$a, &$b)
    };
}
macro_rules! xor {
    ($a:expr, $b:expr) => {
        Test::Binary(ShaderBitwiseOperation::Xor, &$a, &$b)
    };
}

fn flow_test(test: u8) -> Option<Test> {
    Some(match test {
        0 => Constant(false),
        1 => xor!(and!(S, Not(&Z)), O),
        2 => and!(Not(&S), Z),
        3 => xor!(S, or!(Z, O)),
        4 => and!(xor!(Not(&S), O), Not(&Z)),
        5 => Not(&Z),
        6 => Not(&xor!(S, O)),
        7 => or!(Not(&S), Not(&Z)),
        8 => and!(S, Z),
        9 => xor!(S, O),
        10 => Z,
        11 => or!(xor!(S, O), Z),
        12 => xor!(Not(&S), or!(Z, O)),
        13 => or!(S, Not(&Z)),
        14 => xor!(or!(Not(&S), Z), O),
        15 => Constant(true),
        16 => Not(&O),
        17 => Not(&C),
        18 => Not(&S),
        19 => or!(Z, Not(&C)),
        20 => and!(C, Not(&Z)),
        21 => S,
        22 => C,
        23 => O,
        30 => or!(S, Z),
        31 => and!(Not(&S), Not(&Z)),
        // CSM/FCSM tests refer to other hardware state, not Z/S/C/O.
        _ => return None,
    })
}

pub(in crate::shader) fn decode_condition_set_predicate(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    next: &mut u32,
    flags: &IntegerConditionCodes,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/condition_code_set.cpp
    let allowed =
        0xfff8_0000_0000_0000 | 0x3f | (0x1f << 8) | (0xf << 16) | (0xf << 39) | (3 << 45);
    if encoding & !allowed != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "CSETP reserved field is nonzero",
        ));
    }
    let destinations =
        [((encoding >> 3) & 7) as u8, (encoding & 7) as u8].map(|p| (p != 7).then_some(p));
    if destinations[0].is_some() && destinations[0] == destinations[1] {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "CSETP aliased predicate destinations",
        ));
    }
    let combine = match (encoding >> 45) & 3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "CSETP reserved boolean operation",
            ));
        }
    };
    let unsupported = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
        stage,
        instruction_offset: offset,
        encoding,
        detail,
    };
    let test = flow_test(((encoding >> 8) & 31) as u8)
        .ok_or_else(|| unsupported("CSETP CSM/FCSM flow test"))?;
    let mut temp =
        || allocate_shader_temporary(stage, offset, encoding, "CSETP temporary overflow", next);
    let mut ops = Vec::new();
    let zero = temp()?;
    let one = temp()?;
    for (destination, bits) in [(zero, 0), (one, 1)] {
        ops.push(ShaderOperation::MoveImmediate32 {
            destination,
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    }
    let result =
        emit_test(test, flags, zero, one, &mut ops, &mut temp).map_err(|error| match error {
            TestError::Undefined(detail) => unsupported(detail),
            TestError::Temporary(error) => error,
        })?;
    ops.push(ShaderOperation::SetPredicateInteger32 {
        destinations,
        left: result,
        right: zero,
        signed: false,
        comparison: ShaderIntegerComparison::NotEqual,
        accumulator: decode_predicate_fields(encoding, 39, 42),
        set_operation: combine,
    });
    Ok(DecodedIntegerOperation {
        operations: ops,
        constant_buffer_binding: None,
    })
}

enum TestError {
    Undefined(&'static str),
    Temporary(MaxwellShaderTranslationError),
}

fn emit_test(
    test: Test,
    flags: &IntegerConditionCodes,
    zero: ShaderRegister,
    one: ShaderRegister,
    ops: &mut Vec<ShaderOperation>,
    temp: &mut impl FnMut() -> Result<ShaderRegister, MaxwellShaderTranslationError>,
) -> Result<ShaderRegister, TestError> {
    let (left, right, operation) = match test {
        Constant(value) => return Ok(if value { one } else { zero }),
        Flag(index) => {
            let (value, detail) = match index {
                0 => (flags.zero, "CSETP has no translated Z flag definition"),
                1 => (flags.sign, "CSETP has no translated S flag definition"),
                2 => (flags.carry, "CSETP has no translated C flag definition"),
                _ => (flags.overflow, "CSETP has no translated O flag definition"),
            };
            return value.ok_or(TestError::Undefined(detail));
        }
        Not(value) => (
            emit_test(*value, flags, zero, one, ops, temp)?,
            one,
            ShaderBitwiseOperation::Xor,
        ),
        Test::Binary(operation, left, right) => (
            emit_test(*left, flags, zero, one, ops, temp)?,
            emit_test(*right, flags, zero, one, ops, temp)?,
            operation,
        ),
    };
    let destination = temp().map_err(TestError::Temporary)?;
    ops.push(ShaderOperation::Bitwise32 {
        destination,
        left,
        right,
        operation,
    });
    Ok(destination)
}
