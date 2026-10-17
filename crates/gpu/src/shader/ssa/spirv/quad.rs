//! Vulkan's quad/subgroup mapping defines i = SubgroupLocalInvocationId & 3.
//! This avoids the oracle's FragCoord broadcasts used to recover spatial lanes.
//! https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-scope-quad
use super::*;
impl Emitter<'_> {
    pub(super) fn declare_quad_lane(&mut self) {
        if self.quad_lane.is_some() {
            return;
        }
        let ptr = self
            .math
            .b
            .type_pointer(None, spv::StorageClass::Input, self.math.uint);
        let variable = self
            .math
            .b
            .variable(ptr, None, spv::StorageClass::Input, None);
        self.math.b.decorate(
            variable,
            spv::Decoration::BuiltIn,
            [SpvOperand::BuiltIn(spv::BuiltIn::SubgroupLocalInvocationId)],
        );
        self.math.b.decorate(variable, spv::Decoration::Flat, []);
        self.io.variables.push(variable);
        self.quad_lane = Some(variable);
    }
    pub(super) fn quad(&mut self, op: &Op, args: &[u32]) -> Result<u32, Error> {
        if self.function.stage != ShaderStage::Fragment || !self.options.subgroup_quad {
            return Err(
                self.error("fragment quad operations require host fragment subgroup QUAD support")
            );
        }
        self.math.b.capability(spv::Capability::GroupNonUniform);
        self.math.b.capability(spv::Capability::GroupNonUniformQuad);
        let scope = self.math.constant(spv::Scope::Subgroup as u32);
        if let Op::Quad(operation) = *op {
            let value = match operation {
                ShaderQuadOperation::Broadcast(lane) if lane < 4 => {
                    let lane = self.math.constant(u32::from(lane));
                    self.math.b.group_non_uniform_quad_broadcast(
                        self.math.uint,
                        None,
                        scope,
                        args[0],
                        lane,
                    )
                }
                ShaderQuadOperation::Broadcast(_) => {
                    return Err(self.error("quad broadcast lane must be 0..3"));
                }
                operation => {
                    let direction = self.math.constant(match operation {
                        ShaderQuadOperation::SwapHorizontal => 0,
                        ShaderQuadOperation::SwapVertical => 1,
                        ShaderQuadOperation::SwapDiagonal => 2,
                        _ => unreachable!(),
                    });
                    self.math.b.group_non_uniform_quad_swap(
                        self.math.uint,
                        None,
                        scope,
                        args[0],
                        direction,
                    )
                }
            };
            return value.map_err(|e| self.error(e.to_string()));
        }
        let Op::QuadSwizzleAdd {
            swizzle,
            float_control,
        } = *op
        else {
            unreachable!()
        };
        if swizzle > 255 {
            return Err(self.error("quad swizzle must fit four two-bit lane selectors"));
        }
        let variable = self
            .quad_lane
            .ok_or_else(|| self.error("quad lane builtin was not reflected"))?;
        let lane = self
            .math
            .b
            .load(self.math.uint, None, variable, None, [])
            .map_err(|e| self.error(e.to_string()))?;
        let three = self.math.constant(3);
        let two = self.math.constant(2);
        let lane = self
            .math
            .b
            .bitwise_and(self.math.uint, None, lane, three)
            .map_err(|e| self.error(e.to_string()))?;
        let shift = self
            .math
            .b
            .i_mul(self.math.uint, None, lane, two)
            .map_err(|e| self.error(e.to_string()))?;
        let swizzle = self.math.constant(swizzle);
        let selector = self
            .math
            .b
            .shift_right_logical(self.math.uint, None, swizzle, shift)
            .map_err(|e| self.error(e.to_string()))?;
        let selector = self
            .math
            .b
            .bitwise_and(self.math.uint, None, selector, three)
            .map_err(|e| self.error(e.to_string()))?;
        // Coefficients are A=[-1,+1,-1,0], B=[-1,-1,+1,-1].
        // Sign flips and identity are exact bit operations, including subnormals.
        // The zero coefficient retains signed zero and 0*Inf/NaN classification.
        // No intermediate float multiply or DenormPreserve32 is needed.
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp
        let products = (|| -> std::result::Result<[u32; 2], crate::SpirvShaderError> {
            let a = self.math.b.bitcast(self.math.uint, None, args[0])?;
            let b = self.math.b.bitcast(self.math.uint, None, args[1])?;
            let zero = self.math.constant(0);
            let one = self.math.constant(1);
            let sign = self.math.constant(0x8000_0000);
            let parity = self
                .math
                .b
                .bitwise_and(self.math.uint, None, selector, one)?;
            let negative_a = self.math.b.i_equal(self.math.boolean, None, parity, zero)?;
            let negative_b = self
                .math
                .b
                .i_not_equal(self.math.boolean, None, selector, two)?;
            let sign_a = self
                .math
                .b
                .select(self.math.uint, None, negative_a, sign, zero)?;
            let sign_b = self
                .math
                .b
                .select(self.math.uint, None, negative_b, sign, zero)?;
            let signed_a = self.math.b.bitwise_xor(self.math.uint, None, a, sign_a)?;
            let signed_b = self.math.b.bitwise_xor(self.math.uint, None, b, sign_b)?;
            let magnitude_mask = self.math.constant(0x7fff_ffff);
            let magnitude = self
                .math
                .b
                .bitwise_and(self.math.uint, None, a, magnitude_mask)?;
            let infinity = self.math.constant(0x7f80_0000);
            let special =
                self.math
                    .b
                    .u_greater_than_equal(self.math.boolean, None, magnitude, infinity)?;
            let quiet = self.math.constant(0x0040_0000);
            let nan = self.math.b.bitwise_or(self.math.uint, None, a, quiet)?;
            let signed_zero = self.math.b.bitwise_and(self.math.uint, None, a, sign)?;
            let zero_product =
                self.math
                    .b
                    .select(self.math.uint, None, special, nan, signed_zero)?;
            let zero_coefficient = self
                .math
                .b
                .i_equal(self.math.boolean, None, selector, three)?;
            let a = self.math.b.select(
                self.math.uint,
                None,
                zero_coefficient,
                zero_product,
                signed_a,
            )?;
            Ok([a, signed_b])
        })()
        .map_err(|e| self.error(e.to_string()))?;
        let bits = self
            .math
            .float_binary(products[0], products[1], float_control, false)
            .map_err(|e| self.error(e.to_string()))?;
        self.math
            .b
            .bitcast(self.math.float, None, bits)
            .map_err(|e| self.error(e.to_string()))
    }
}
