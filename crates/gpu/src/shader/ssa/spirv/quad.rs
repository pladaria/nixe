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
        self.math
            .require_float(float_control)
            .map_err(|e| self.error(e.to_string()))?;
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
        let vector = self.math.b.type_vector(self.math.float, 4);
        let negative = self
            .math
            .b
            .constant_bit32(self.math.float, (-1.0f32).to_bits());
        let positive = self
            .math
            .b
            .constant_bit32(self.math.float, 1.0f32.to_bits());
        let zero = self.math.b.constant_bit32(self.math.float, 0);
        let coefficients = [
            self.math
                .b
                .constant_composite(vector, [negative, positive, negative, zero]),
            self.math
                .b
                .constant_composite(vector, [negative, negative, positive, negative]),
        ];
        let mut products = [0; 2];
        for i in 0..2 {
            let bits = self
                .math
                .b
                .bitcast(self.math.uint, None, args[i])
                .map_err(|e| self.error(e.to_string()))?;
            let input = self
                .math
                .float_operand(bits, float_control)
                .map_err(|e| self.error(e.to_string()))?;
            let coefficient = self
                .math
                .b
                .vector_extract_dynamic(self.math.float, None, coefficients[i], selector)
                .map_err(|e| self.error(e.to_string()))?;
            let product = self
                .math
                .b
                .f_mul(self.math.float, None, input, coefficient)
                .map_err(|e| self.error(e.to_string()))?;
            self.math
                .b
                .decorate(product, spv::Decoration::NoContraction, []);
            products[i] = self
                .math
                .b
                .bitcast(self.math.uint, None, product)
                .map_err(|e| self.error(e.to_string()))?;
        }
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
