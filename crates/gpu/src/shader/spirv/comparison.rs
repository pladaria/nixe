//! Bit comparisons preserve subnormals without requiring host FP denormal modes.
//! The ordered key reverses negative magnitudes; +0/-0 compare equal, and NaNs
//! are classified independently of their sign or payload.
use super::*;
impl Math {
    pub(crate) fn compare_float_bits(
        &mut self,
        a: u32,
        b: u32,
        comparison: ShaderFloatComparison,
        flush: bool,
    ) -> Result<u32> {
        let a = if flush { self.flush_denormal(a)? } else { a };
        let b = if flush { self.flush_denormal(b)? } else { b };
        let mask = self.constant(0x7fffffff);
        let sign = self.constant(0x80000000);
        let zero = self.constant(0);
        let infinity = self.constant(0x7f800000);
        let aa = self.b.bitwise_and(self.uint, None, a, mask)?;
        let ba = self.b.bitwise_and(self.uint, None, b, mask)?;
        let an = self.b.u_greater_than(self.boolean, None, aa, infinity)?;
        let bn = self.b.u_greater_than(self.boolean, None, ba, infinity)?;
        let nan = self.b.logical_or(self.boolean, None, an, bn)?;
        let az = self.b.i_equal(self.boolean, None, aa, zero)?;
        let bz = self.b.i_equal(self.boolean, None, ba, zero)?;
        let bothzero = self.b.logical_and(self.boolean, None, az, bz)?;
        let same = self.b.i_equal(self.boolean, None, a, b)?;
        let equal = self.b.logical_or(self.boolean, None, same, bothzero)?;
        let mut keys = [0; 2];
        for (index, bits) in [a, b].into_iter().enumerate() {
            let negative = self
                .b
                .u_greater_than_equal(self.boolean, None, bits, sign)?;
            let flipped = self.b.not(self.uint, None, bits)?;
            let positive = self.b.bitwise_xor(self.uint, None, bits, sign)?;
            keys[index] = self
                .b
                .select(self.uint, None, negative, flipped, positive)?;
        }
        use ShaderFloatComparison::*;
        if comparison == IsNan {
            return Ok(nan);
        }
        let number = self.b.logical_not(self.boolean, None, nan)?;
        if comparison == IsNumber {
            return Ok(number);
        }
        let ordered = match comparison {
            OrderedEqual | UnorderedEqual => equal,
            OrderedNotEqual | UnorderedNotEqual => self.b.logical_not(self.boolean, None, equal)?,
            c => {
                let strict = match c {
                    OrderedLess | UnorderedLess | OrderedLessOrEqual | UnorderedLessOrEqual => {
                        self.b.u_less_than(self.boolean, None, keys[0], keys[1])?
                    }
                    _ => self
                        .b
                        .u_greater_than(self.boolean, None, keys[0], keys[1])?,
                };
                let notzero = self.b.logical_not(self.boolean, None, bothzero)?;
                let strict = self.b.logical_and(self.boolean, None, strict, notzero)?;
                if matches!(
                    c,
                    OrderedLessOrEqual
                        | UnorderedLessOrEqual
                        | OrderedGreaterOrEqual
                        | UnorderedGreaterOrEqual
                ) {
                    self.b.logical_or(self.boolean, None, strict, equal)?
                } else {
                    strict
                }
            }
        };
        let unordered = matches!(
            comparison,
            UnorderedLess
                | UnorderedEqual
                | UnorderedLessOrEqual
                | UnorderedGreater
                | UnorderedNotEqual
                | UnorderedGreaterOrEqual
        );
        Ok(if unordered {
            self.b.logical_or(self.boolean, None, nan, ordered)?
        } else {
            self.b.logical_and(self.boolean, None, number, ordered)?
        })
    }
}
