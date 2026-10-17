use super::*;
impl Emitter<'_> {
    pub(super) fn unary(&mut self, op: Unary, scalar: Type, input: u32) -> Result<u32, Error> {
        if scalar == Type::F16 && matches!(op, Unary::Negate | Unary::Absolute) {
            let bits = self.half_pack(input)?;
            let mask = self.math.constant(if op == Unary::Absolute {
                0x7fff
            } else {
                0x8000
            });
            let bits = if op == Unary::Absolute {
                self.math.b.bitwise_and(self.math.uint, None, bits, mask)
            } else {
                self.math.b.bitwise_xor(self.math.uint, None, bits, mask)
            }
            .map_err(|e| self.error(e.to_string()))?;
            return self.half_from_bits(bits);
        }
        let ty = self.ty(scalar)?;
        let result = match op {
            Unary::Not if scalar == Type::Bool => self.math.b.logical_not(ty, None, input),
            Unary::Not => self.math.b.not(ty, None, input),
            Unary::BitReverse => self.math.b.bit_reverse(ty, None, input),
            Unary::Negate | Unary::Absolute if matches!(scalar, Type::F32 | Type::F64) => {
                // Sign operations are bit operations, including subnormal/NaN
                // operands. No host FP flush mode or SAbs overflow enters them.
                let integer = if scalar == Type::F32 {
                    self.math.uint
                } else {
                    self.ty(Type::U64)?
                };
                let bits = self
                    .math
                    .b
                    .bitcast(integer, None, input)
                    .map_err(|e| self.error(e.to_string()))?;
                let mask = if scalar == Type::F32 {
                    self.math.constant(if op == Unary::Absolute {
                        0x7fffffff
                    } else {
                        0x80000000
                    })
                } else {
                    self.math.b.constant_bit64(
                        integer,
                        if op == Unary::Absolute {
                            0x7fffffffffffffff
                        } else {
                            0x8000000000000000
                        },
                    )
                };
                let bits = if op == Unary::Absolute {
                    self.math.b.bitwise_and(integer, None, bits, mask)
                } else {
                    self.math.b.bitwise_xor(integer, None, bits, mask)
                }
                .map_err(|e| self.error(e.to_string()))?;
                self.math.b.bitcast(ty, None, bits)
            }
            Unary::Negate => self.math.b.s_negate(ty, None, input),
            Unary::Absolute if matches!(scalar, Type::U32 | Type::U64) => return Ok(input),
            Unary::Absolute => {
                let zero = if scalar == Type::I32 {
                    self.math.b.constant_bit32(ty, 0)
                } else {
                    self.math.b.constant_bit64(ty, 0)
                };
                let negative = self
                    .math
                    .b
                    .s_less_than(self.math.boolean, None, input, zero)
                    .map_err(|e| self.error(e.to_string()))?;
                let negated = self
                    .math
                    .b
                    .s_negate(ty, None, input)
                    .map_err(|e| self.error(e.to_string()))?;
                self.math.b.select(ty, None, negative, negated, input)
            }
        };
        result.map_err(|e| self.error(e.to_string()))
    }
}
