use crate::{
    Result,
    ir::{Emitter, Value},
};
pub(super) fn apply(
    ir: &mut Emitter<'_>,
    a: Value,
    b: Value,
    c: Value,
    table: u8,
) -> Result<Value> {
    match table {
        0 => Ok(Value::U32(0)),
        1 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        2 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_and(c, t1)?;
            Ok(t2)
        }
        3 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            Ok(t1)
        }
        4 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_and(b, t1)?;
            Ok(t2)
        }
        5 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            Ok(t1)
        }
        6 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        7 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        8 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        9 => {
            let t0 = ir.bitwise_xor(b, c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        10 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(c, t0)?;
            Ok(t1)
        }
        11 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        12 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(b, t0)?;
            Ok(t1)
        }
        13 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_or(b, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        14 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        15 => {
            let t0 = ir.bitwise_not(a)?;
            Ok(t0)
        }
        16 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_and(a, t1)?;
            Ok(t2)
        }
        17 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_not(t0)?;
            Ok(t1)
        }
        18 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        19 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        20 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        21 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        22 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(a, b)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_xor(t0, t2)?;
            Ok(t3)
        }
        23 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(a)?;
            let t4 = ir.bitwise_xor(t2, t3)?;
            Ok(t4)
        }
        24 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        25 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        26 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_xor(a, c)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        27 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_or(b, c)?;
            let t3 = ir.bitwise_xor(t1, t2)?;
            Ok(t3)
        }
        28 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        29 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_or(b, c)?;
            let t3 = ir.bitwise_xor(t1, t2)?;
            Ok(t3)
        }
        30 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            Ok(t1)
        }
        31 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        32 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        33 => {
            let t0 = ir.bitwise_xor(a, c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        34 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            Ok(t1)
        }
        35 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        36 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        37 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        38 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        39 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_or(b, t1)?;
            let t3 = ir.bitwise_xor(t0, t2)?;
            Ok(t3)
        }
        40 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            Ok(t1)
        }
        41 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(a, b)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            let t4 = ir.bitwise_xor(t0, t3)?;
            Ok(t4)
        }
        42 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_and(c, t1)?;
            Ok(t2)
        }
        43 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_xor(a, c)?;
            let t3 = ir.bitwise_or(b, t2)?;
            let t4 = ir.bitwise_xor(t1, t3)?;
            Ok(t4)
        }
        44 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        45 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            Ok(t2)
        }
        46 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        47 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_not(a)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        48 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(a, t0)?;
            Ok(t1)
        }
        49 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_or(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        50 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        51 => {
            let t0 = ir.bitwise_not(b)?;
            Ok(t0)
        }
        52 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        53 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(b, t1)?;
            let t3 = ir.bitwise_xor(t0, t2)?;
            Ok(t3)
        }
        54 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_xor(b, t0)?;
            Ok(t1)
        }
        55 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        56 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        57 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            Ok(t2)
        }
        58 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        59 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_not(b)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        60 => {
            let t0 = ir.bitwise_xor(a, b)?;
            Ok(t0)
        }
        61 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        62 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        63 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            Ok(t1)
        }
        64 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        65 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        66 => {
            let t0 = ir.bitwise_xor(a, c)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        67 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        68 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            Ok(t1)
        }
        69 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(b, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        70 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        71 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_xor(t0, t2)?;
            Ok(t3)
        }
        72 => {
            let t0 = ir.bitwise_xor(a, c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            Ok(t1)
        }
        73 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_and(a, c)?;
            let t2 = ir.bitwise_not(b)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            let t4 = ir.bitwise_xor(t0, t3)?;
            Ok(t4)
        }
        74 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        75 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            Ok(t2)
        }
        76 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_and(b, t1)?;
            Ok(t2)
        }
        77 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_or(c, t2)?;
            let t4 = ir.bitwise_xor(t1, t3)?;
            Ok(t4)
        }
        78 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        79 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_not(a)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        80 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            Ok(t1)
        }
        81 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_or(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        82 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_xor(a, c)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        83 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_xor(t0, t2)?;
            Ok(t3)
        }
        84 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        85 => {
            let t0 = ir.bitwise_not(c)?;
            Ok(t0)
        }
        86 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_xor(c, t0)?;
            Ok(t1)
        }
        87 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        88 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        89 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_xor(c, t1)?;
            Ok(t2)
        }
        90 => {
            let t0 = ir.bitwise_xor(a, c)?;
            Ok(t0)
        }
        91 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_xor(a, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        92 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        93 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        94 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_xor(a, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        95 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            Ok(t1)
        }
        96 => {
            let t0 = ir.bitwise_xor(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            Ok(t1)
        }
        97 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_and(b, c)?;
            let t2 = ir.bitwise_not(a)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            let t4 = ir.bitwise_xor(t0, t3)?;
            Ok(t4)
        }
        98 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        99 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            Ok(t2)
        }
        100 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            Ok(t2)
        }
        101 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_xor(c, t1)?;
            Ok(t2)
        }
        102 => {
            let t0 = ir.bitwise_xor(b, c)?;
            Ok(t0)
        }
        103 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        104 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(a, b)?;
            let t2 = ir.bitwise_xor(c, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        105 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        106 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_xor(c, t0)?;
            Ok(t1)
        }
        107 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_not(b)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_xor(t1, t3)?;
            Ok(t4)
        }
        108 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_xor(b, t0)?;
            Ok(t1)
        }
        109 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_xor(t1, t3)?;
            Ok(t4)
        }
        110 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        111 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        112 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_and(a, t1)?;
            Ok(t2)
        }
        113 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_or(c, t2)?;
            let t4 = ir.bitwise_xor(t1, t3)?;
            Ok(t4)
        }
        114 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        115 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_not(b)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        116 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        117 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        118 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        119 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(t0)?;
            Ok(t1)
        }
        120 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            Ok(t1)
        }
        121 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_xor(b, t2)?;
            let t4 = ir.bitwise_xor(t1, t3)?;
            Ok(t4)
        }
        122 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_xor(a, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        123 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        124 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_xor(a, b)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        125 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        126 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        127 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        128 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            Ok(t1)
        }
        129 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        130 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            let t2 = ir.bitwise_and(c, t1)?;
            Ok(t2)
        }
        131 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_not(b)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_and(t1, t3)?;
            Ok(t4)
        }
        132 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            let t2 = ir.bitwise_and(b, t1)?;
            Ok(t2)
        }
        133 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_and(t1, t3)?;
            Ok(t4)
        }
        134 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        135 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        136 => {
            let t0 = ir.bitwise_and(b, c)?;
            Ok(t0)
        }
        137 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_xor(b, t2)?;
            let t4 = ir.bitwise_and(t1, t3)?;
            Ok(t4)
        }
        138 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_and(c, t1)?;
            Ok(t2)
        }
        139 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_not(t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        140 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_and(b, t1)?;
            Ok(t2)
        }
        141 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_not(t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        142 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            Ok(t3)
        }
        143 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        144 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(b, t0)?;
            let t2 = ir.bitwise_and(a, t1)?;
            Ok(t2)
        }
        145 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_xor(b, t2)?;
            let t4 = ir.bitwise_and(t1, t3)?;
            Ok(t4)
        }
        146 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        147 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        148 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        149 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        150 => {
            let t0 = ir.bitwise_xor(b, c)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            Ok(t1)
        }
        151 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_or(t1, t3)?;
            Ok(t4)
        }
        152 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        153 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(b, t0)?;
            Ok(t1)
        }
        154 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_xor(c, t1)?;
            Ok(t2)
        }
        155 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        156 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            Ok(t2)
        }
        157 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        158 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        159 => {
            let t0 = ir.bitwise_xor(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        160 => {
            let t0 = ir.bitwise_and(a, c)?;
            Ok(t0)
        }
        161 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_not(c)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_and(t1, t3)?;
            Ok(t4)
        }
        162 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_and(c, t1)?;
            Ok(t2)
        }
        163 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_not(t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        164 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        165 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            Ok(t1)
        }
        166 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_xor(c, t1)?;
            Ok(t2)
        }
        167 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        168 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            Ok(t1)
        }
        169 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        170 => Ok(c),
        171 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_or(c, t1)?;
            Ok(t2)
        }
        172 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        173 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        174 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_or(c, t1)?;
            Ok(t2)
        }
        175 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(c, t0)?;
            Ok(t1)
        }
        176 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            let t2 = ir.bitwise_and(a, t1)?;
            Ok(t2)
        }
        177 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_not(t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        178 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_xor(b, t2)?;
            Ok(t3)
        }
        179 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        180 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            Ok(t2)
        }
        181 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        182 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        183 => {
            let t0 = ir.bitwise_xor(a, c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        184 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        185 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        186 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_or(c, t1)?;
            Ok(t2)
        }
        187 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            Ok(t1)
        }
        188 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        189 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        190 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            Ok(t1)
        }
        191 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_or(c, t1)?;
            Ok(t2)
        }
        192 => {
            let t0 = ir.bitwise_and(a, b)?;
            Ok(t0)
        }
        193 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_not(b)?;
            let t3 = ir.bitwise_xor(a, t2)?;
            let t4 = ir.bitwise_and(t1, t3)?;
            Ok(t4)
        }
        194 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        195 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            Ok(t1)
        }
        196 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_and(b, t1)?;
            Ok(t2)
        }
        197 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_not(t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        198 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            Ok(t2)
        }
        199 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        200 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            Ok(t1)
        }
        201 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        202 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(a)?;
            let t2 = ir.bitwise_or(b, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        203 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        204 => Ok(b),
        205 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_or(b, t1)?;
            Ok(t2)
        }
        206 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_or(b, t1)?;
            Ok(t2)
        }
        207 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, t0)?;
            Ok(t1)
        }
        208 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            let t2 = ir.bitwise_and(a, t1)?;
            Ok(t2)
        }
        209 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_not(t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        210 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            Ok(t2)
        }
        211 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_xor(a, b)?;
            let t2 = ir.bitwise_and(t0, t1)?;
            let t3 = ir.bitwise_not(t2)?;
            Ok(t3)
        }
        212 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            let t3 = ir.bitwise_xor(c, t2)?;
            Ok(t3)
        }
        213 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        214 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_xor(c, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        215 => {
            let t0 = ir.bitwise_xor(a, b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_not(t1)?;
            Ok(t2)
        }
        216 => {
            let t0 = ir.bitwise_or(a, c)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_or(b, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        217 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(b, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        218 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_xor(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        219 => {
            let t0 = ir.bitwise_xor(a, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        220 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            let t2 = ir.bitwise_or(b, t1)?;
            Ok(t2)
        }
        221 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            Ok(t1)
        }
        222 => {
            let t0 = ir.bitwise_xor(a, c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            Ok(t1)
        }
        223 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_or(b, t1)?;
            Ok(t2)
        }
        224 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_and(a, t0)?;
            Ok(t1)
        }
        225 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_xor(t0, t1)?;
            Ok(t2)
        }
        226 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_or(b, c)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        227 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_not(b)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        228 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            let t2 = ir.bitwise_or(b, c)?;
            let t3 = ir.bitwise_and(t1, t2)?;
            Ok(t3)
        }
        229 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_xor(a, t1)?;
            let t3 = ir.bitwise_or(t0, t2)?;
            Ok(t3)
        }
        230 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_xor(b, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        231 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            let t2 = ir.bitwise_xor(b, c)?;
            let t3 = ir.bitwise_or(t1, t2)?;
            Ok(t3)
        }
        232 => {
            let t0 = ir.bitwise_or(a, b)?;
            let t1 = ir.bitwise_and(a, b)?;
            let t2 = ir.bitwise_or(c, t1)?;
            let t3 = ir.bitwise_and(t0, t2)?;
            Ok(t3)
        }
        233 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_not(c)?;
            let t2 = ir.bitwise_or(a, b)?;
            let t3 = ir.bitwise_xor(t1, t2)?;
            let t4 = ir.bitwise_or(t0, t3)?;
            Ok(t4)
        }
        234 => {
            let t0 = ir.bitwise_and(a, b)?;
            let t1 = ir.bitwise_or(c, t0)?;
            Ok(t1)
        }
        235 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            let t2 = ir.bitwise_or(c, t1)?;
            Ok(t2)
        }
        236 => {
            let t0 = ir.bitwise_and(a, c)?;
            let t1 = ir.bitwise_or(b, t0)?;
            Ok(t1)
        }
        237 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(a, t0)?;
            let t2 = ir.bitwise_or(b, t1)?;
            Ok(t2)
        }
        238 => {
            let t0 = ir.bitwise_or(b, c)?;
            Ok(t0)
        }
        239 => {
            let t0 = ir.bitwise_not(a)?;
            let t1 = ir.bitwise_or(b, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        240 => Ok(a),
        241 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_or(a, t1)?;
            Ok(t2)
        }
        242 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_and(c, t0)?;
            let t2 = ir.bitwise_or(a, t1)?;
            Ok(t2)
        }
        243 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, t0)?;
            Ok(t1)
        }
        244 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_and(b, t0)?;
            let t2 = ir.bitwise_or(a, t1)?;
            Ok(t2)
        }
        245 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            Ok(t1)
        }
        246 => {
            let t0 = ir.bitwise_xor(b, c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            Ok(t1)
        }
        247 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_not(t0)?;
            let t2 = ir.bitwise_or(a, t1)?;
            Ok(t2)
        }
        248 => {
            let t0 = ir.bitwise_and(b, c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            Ok(t1)
        }
        249 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_xor(b, t0)?;
            let t2 = ir.bitwise_or(a, t1)?;
            Ok(t2)
        }
        250 => {
            let t0 = ir.bitwise_or(a, c)?;
            Ok(t0)
        }
        251 => {
            let t0 = ir.bitwise_not(b)?;
            let t1 = ir.bitwise_or(a, c)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        252 => {
            let t0 = ir.bitwise_or(a, b)?;
            Ok(t0)
        }
        253 => {
            let t0 = ir.bitwise_not(c)?;
            let t1 = ir.bitwise_or(a, b)?;
            let t2 = ir.bitwise_or(t0, t1)?;
            Ok(t2)
        }
        254 => {
            let t0 = ir.bitwise_or(b, c)?;
            let t1 = ir.bitwise_or(a, t0)?;
            Ok(t1)
        }
        255 => Ok(Value::U32(0xFFFFFFFF)),
    }
}
