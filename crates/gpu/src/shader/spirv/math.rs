//! Shared numerical SPIR-V primitives. No guest register state or old IR.
use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Repair {
    PreservingAdd,
    PreservingMultiply,
    PreservingFma,
    FtzMultiply,
    FtzFma,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Denorm32 {
    Preserve,
    Unspecified,
    Flush,
}

pub(crate) struct Math {
    pub(crate) b: Builder,
    pub(crate) options: SpirvShaderOptions,
    pub(crate) uint: u32,
    pub(crate) int: u32,
    pub(crate) float: u32,
    pub(crate) boolean: u32,
    pub(crate) undefined_uint: u32,
    pub(crate) undefined_bool: u32,
    pub(crate) true_value: u32,
    pub(crate) false_value: u32,
    pub(crate) constants: HashMap<u32, u32>,
    pub(crate) entry: u32,
    pub(crate) block: u32,
    pub(crate) source: ShaderSourceLocation,
    pub(crate) float_modes: bool,
    pub(crate) denorm32: Denorm32,
    pub(crate) rounding32: ShaderRoundingMode,
    pub(crate) repair_float: Option<u32>,
    pub(crate) fma_extension: bool,
    pub(crate) glsl_import: Option<u32>,
    repairs: HashMap<Repair, u32>,
    pub(super) native_half: Option<u32>,
}

impl Math {
    pub(crate) fn new(options: SpirvShaderOptions) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(spv::Capability::Shader);
        b.memory_model(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
        let uint = b.type_int(32, 0);
        let int = b.type_int(32, 1);
        let float = b.type_float(32, None);
        let boolean = b.type_bool();
        // Global undef IDs dominate both selection predecessors. Creating undef
        // in the merge would violate SSA dominance and the leading-phi rule.
        let undefined_uint = b.undef(uint, None);
        let undefined_bool = b.undef(boolean, None);
        let true_value = b.constant_true(boolean);
        let false_value = b.constant_false(boolean);
        Self {
            b,
            options,
            uint,
            int,
            float,
            boolean,
            undefined_uint,
            undefined_bool,
            true_value,
            false_value,
            constants: HashMap::new(),
            entry: 0,
            block: 0,
            source: ShaderSourceLocation::new(0),
            float_modes: false,
            denorm32: if options.float32.denorm_preserve {
                Denorm32::Preserve
            } else {
                Denorm32::Unspecified
            },
            rounding32: ShaderRoundingMode::NearestEven,
            repair_float: None,
            fma_extension: false,
            glsl_import: None,
            repairs: HashMap::new(),
            native_half: None,
        }
    }
    /// Share only the exceptional repair body. Native arithmetic and the guard
    /// stay inline, so ordinary values never pay a GPU function-call cost.
    /// DontInline is a hint; correctness does not depend on driver inlining.
    pub(super) fn repair_call<const N: usize>(
        &mut self,
        kind: Repair,
        arguments: [u32; N],
        emit: impl FnOnce(&mut Self, [u32; N]) -> Result<u32>,
    ) -> Result<u32> {
        let function = if let Some(&function) = self.repairs.get(&kind) {
            function
        } else {
            let caller = self.b.selected_function();
            let caller_block = self.b.selected_block();
            let label = self.block;
            self.b.select_function(None)?;
            let ty = self.b.type_function(self.uint, [self.float; N]);
            let function =
                self.b
                    .begin_function(self.uint, None, spv::FunctionControl::DONT_INLINE, ty)?;
            let mut parameters = [0; N];
            for parameter in &mut parameters {
                *parameter = self.b.function_parameter(self.float)?;
            }
            self.block = self.b.begin_block(None)?;
            let result = emit(self, parameters)?;
            self.b.ret_value(result)?;
            self.b.end_function()?;
            self.b.select_function(caller)?;
            self.b.select_block(caller_block)?;
            self.block = label;
            self.repairs.insert(kind, function);
            function
        };
        Ok(self.b.function_call(self.uint, None, function, arguments)?)
    }
    pub(crate) fn unsupported(&self, reason: &'static str) -> SpirvShaderError {
        SpirvShaderError::Instruction {
            source: self.source,
            reason,
        }
    }
    pub(crate) fn constant(&mut self, bits: u32) -> u32 {
        // rspirv interns types, but not constants. Reusing constant IDs also
        // allows equal-length array types to share their declaration.
        *self
            .constants
            .entry(bits)
            .or_insert_with(|| self.b.constant_bit32(self.uint, bits))
    }
    pub(crate) fn preserves_float32(&self) -> bool {
        self.denorm32 == Denorm32::Preserve
    }
    pub(crate) fn require_float(&mut self, control: ShaderFloatControl) -> Result<()> {
        if !self.preserves_float32()
            && !(control.denormals_are_zero && control.flush_denormals_to_zero)
        {
            return Err(self.unsupported(
                "this operation requires native DenormPreserve32 or an exact preserving lowering",
            ));
        }
        self.require_repaired_float(control)
    }
    pub(crate) fn require_repaired_float(&mut self, control: ShaderFloatControl) -> Result<()> {
        if control.rounding != self.rounding32 {
            return Err(self
                .unsupported("arithmetic rounding differs from the selected shader environment"));
        }
        let caps = self.options.float32;
        let rounding = match self.rounding32 {
            ShaderRoundingMode::NearestEven if caps.rounding_mode_rte => (
                spv::Capability::RoundingModeRTE,
                spv::ExecutionMode::RoundingModeRTE,
            ),
            ShaderRoundingMode::TowardZero if caps.rounding_mode_rtz => (
                spv::Capability::RoundingModeRTZ,
                spv::ExecutionMode::RoundingModeRTZ,
            ),
            _ => {
                return Err(
                    self.unsupported("selected float32 rounding mode is not enabled on the host")
                );
            }
        };
        if !caps.signed_zero_inf_nan_preserve {
            return Err(self.unsupported("float32 requires SignedZeroInfNanPreserve"));
        }
        if !self.float_modes {
            self.b.extension("SPV_KHR_float_controls");
            for (capability, mode) in [
                rounding,
                (
                    spv::Capability::SignedZeroInfNanPreserve,
                    spv::ExecutionMode::SignedZeroInfNanPreserve,
                ),
            ] {
                self.b.capability(capability);
                self.b.execution_mode(self.entry, mode, [32]);
            }
            if self.preserves_float32() {
                self.b.capability(spv::Capability::DenormPreserve);
                self.b
                    .execution_mode(self.entry, spv::ExecutionMode::DenormPreserve, [32]);
            }
            if self.denorm32 == Denorm32::Flush {
                self.b.capability(spv::Capability::DenormFlushToZero);
                self.b
                    .execution_mode(self.entry, spv::ExecutionMode::DenormFlushToZero, [32]);
            }
            self.float_modes = true;
        }
        Ok(())
    }

    pub(crate) fn require_scalar32(&self, ty: ShaderScalarType) -> Result<()> {
        if matches!(
            ty,
            ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 | ShaderScalarType::Float32
        ) {
            Ok(())
        } else {
            Err(self.unsupported("64-bit scalar moves are not implemented"))
        }
    }

    pub(crate) fn float_binary(
        &mut self,
        left: u32,
        right: u32,
        control: ShaderFloatControl,
        multiply: bool,
    ) -> Result<u32> {
        self.float_binary_known_normal(left, right, control, multiply, false)
    }
    pub(crate) fn float_binary_known_normal(
        &mut self,
        left: u32,
        right: u32,
        mut control: ShaderFloatControl,
        multiply: bool,
        known_normal: bool,
    ) -> Result<u32> {
        if known_normal {
            control.denormals_are_zero = false;
            control.flush_denormals_to_zero = false;
        }
        if multiply
            && control.rounding == ShaderRoundingMode::TowardZero
            && (known_normal || control.denormals_are_zero && control.flush_denormals_to_zero)
        {
            if self.rounding32 == ShaderRoundingMode::TowardZero {
                self.require_repaired_float(control)?;
                let a = self.float_operand(left, control)?;
                let b = self.float_operand(right, control)?;
                let value = self.b.f_mul(self.float, None, a, b)?;
                self.b.decorate(value, spv::Decoration::NoContraction, []);
                return self.float_result(value, control);
            }
            let bits = self.multiply_rz_ftz(left, right)?;
            let value = self.b.bitcast(self.float, None, bits)?;
            return self.float_result(value, control);
        }
        self.require_repaired_float(control)?;
        let left = self.float_operand(left, control)?;
        let right = self.float_operand(right, control)?;
        let value = if multiply {
            self.b.f_mul(self.float, None, left, right)?
        } else {
            self.b.f_add(self.float, None, left, right)?
        };
        self.b.decorate(value, spv::Decoration::NoContraction, []);
        let value = if known_normal
            || self.preserves_float32()
            || (!multiply
                && self.denorm32 == Denorm32::Flush
                && control.denormals_are_zero
                && control.flush_denormals_to_zero)
        {
            value
        } else {
            if control.denormals_are_zero && control.flush_denormals_to_zero {
                self.repair_binary_underflow(value, left, right, multiply)?
            } else {
                self.repair_preserving(value, left, right, None, multiply)?
            }
        };
        self.float_result(value, control)
    }

    pub(crate) fn flush_denormal(&mut self, bits: u32) -> Result<u32> {
        // Integer classification preserves the sign of zero and leaves NaNs and
        // infinities untouched. Per-operation DAZ/FTZ can coexist in one shader.
        let exponent_mask = self.constant(0x7f80_0000);
        let exponent = self.b.bitwise_and(self.uint, None, bits, exponent_mask)?;
        let zero = self.constant(0);
        let denormal_or_zero = self.b.i_equal(self.boolean, None, exponent, zero)?;
        let sign_mask = self.constant(0x8000_0000);
        let sign = self.b.bitwise_and(self.uint, None, bits, sign_mask)?;
        Ok(self
            .b
            .select(self.uint, None, denormal_or_zero, sign, bits)?)
    }

    pub(crate) fn float_operand(
        &mut self,
        mut bits: u32,
        control: ShaderFloatControl,
    ) -> Result<u32> {
        if control.denormals_are_zero {
            bits = self.flush_denormal(bits)?;
        }
        Ok(self.b.bitcast(self.float, None, bits)?)
    }

    pub(crate) fn float_result(&mut self, value: u32, control: ShaderFloatControl) -> Result<u32> {
        let mut bits = self.b.bitcast(self.uint, None, value)?;
        if control.nan_mode == ShaderNanMode::Canonicalize {
            let is_nan = self.b.is_nan(self.boolean, None, value)?;
            let nan = self.constant(f32::NAN.to_bits());
            bits = self.b.select(self.uint, None, is_nan, nan, bits)?;
        }
        if control.flush_denormals_to_zero && self.denorm32 != Denorm32::Flush {
            bits = self.flush_denormal(bits)?;
        }
        if control.saturate {
            let sign = self.constant(0x8000_0000);
            let negative = self
                .b
                .u_greater_than_equal(self.boolean, None, bits, sign)?;
            let infinity = self.constant(0x7f80_0000);
            let nan = self.b.u_greater_than(self.boolean, None, bits, infinity)?;
            let zero_result = self.b.logical_or(self.boolean, None, negative, nan)?;
            let one = self.constant(0x3f80_0000);
            let above_one = self.b.u_greater_than(self.boolean, None, bits, one)?;
            bits = self.b.select(self.uint, None, above_one, one, bits)?;
            let zero = self.constant(0);
            bits = self.b.select(self.uint, None, zero_result, zero, bits)?;
        }
        Ok(bits)
    }
    pub(crate) fn enable_fma(&mut self) {
        // https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_fma.html
        if !self.fma_extension {
            self.b.extension("SPV_KHR_fma");
            self.b.capability(spv::Capability::FMAKHR);
            self.fma_extension = true;
        }
    }
    pub(crate) fn fma(
        &mut self,
        left: u32,
        right: u32,
        addend: u32,
        control: ShaderFloatControl,
    ) -> Result<u32> {
        self.fma_known_normal(left, right, addend, control, false)
    }
    pub(crate) fn fma_known_normal(
        &mut self,
        left: u32,
        right: u32,
        addend: u32,
        mut control: ShaderFloatControl,
        known_normal: bool,
    ) -> Result<u32> {
        if known_normal {
            control.denormals_are_zero = false;
            control.flush_denormals_to_zero = false;
        }
        self.require_repaired_float(control)?;
        if !self.options.float32.fused_multiply_add {
            return Err(self.unsupported(
                "correctly rounded float32 FMA requires VK_KHR_shader_fma / shaderFmaFloat32",
            ));
        }
        self.enable_fma();
        let left = self.float_operand(left, control)?;
        let right = self.float_operand(right, control)?;
        let addend = self.float_operand(addend, control)?;
        let value = self.b.fma_khr(self.float, None, left, right, addend)?;
        self.b.decorate(value, spv::Decoration::NoContraction, []);
        let value = if known_normal || self.preserves_float32() {
            value
        } else {
            if control.denormals_are_zero && control.flush_denormals_to_zero {
                self.repair_product_underflow(value, left, right, Some(addend))?
            } else {
                self.repair_preserving(value, left, right, Some(addend), true)?
            }
        };
        self.float_result(value, control)
    }
}
