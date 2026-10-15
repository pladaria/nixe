//! Maxwell min/max order -0 below +0 and select the numeric operand when only
//! one operand is NaN. Integer keys avoid optional host denormal modes.
//! https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min
use super::*;
impl Math {
    pub(crate) fn minmax_bits(
        &mut self,
        a: u32,
        b: u32,
        minimum: u32,
        control: ShaderFloatControl,
    ) -> Result<u32> {
        let a = if control.denormals_are_zero {
            self.flush_denormal(a)?
        } else {
            a
        };
        let b = if control.denormals_are_zero {
            self.flush_denormal(b)?
        } else {
            b
        };
        let sign = self.constant(0x80000000);
        let abs = self.constant(0x7fffffff);
        let infinity = self.constant(0x7f800000);
        let mut keys = [0; 2];
        for (i, bits) in [a, b].into_iter().enumerate() {
            let negative = self
                .b
                .u_greater_than_equal(self.boolean, None, bits, sign)?;
            let reverse = self.b.not(self.uint, None, bits)?;
            let positive = self.b.bitwise_xor(self.uint, None, bits, sign)?;
            keys[i] = self
                .b
                .select(self.uint, None, negative, reverse, positive)?;
        }
        let less = self.b.u_less_than(self.boolean, None, keys[0], keys[1])?;
        let greater = self
            .b
            .u_greater_than(self.boolean, None, keys[0], keys[1])?;
        let pick = self.b.select(self.boolean, None, minimum, less, greater)?;
        let value = self.b.select(self.uint, None, pick, a, b)?;
        let aa = self.b.bitwise_and(self.uint, None, a, abs)?;
        let ba = self.b.bitwise_and(self.uint, None, b, abs)?;
        let an = self.b.u_greater_than(self.boolean, None, aa, infinity)?;
        let bn = self.b.u_greater_than(self.boolean, None, ba, infinity)?;
        let value = self.b.select(self.uint, None, an, b, value)?;
        let value = self.b.select(self.uint, None, bn, a, value)?;
        let value = self.b.bitcast(self.float, None, value)?;
        self.float_result(value, control)
    }
}
