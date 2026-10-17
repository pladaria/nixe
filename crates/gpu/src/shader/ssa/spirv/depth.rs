//! Recover guest window depth after interpolation, without changing clip space
//! or interpolants. Viewport endpoint clamping would change every depth value.
//! https://docs.vulkan.org/spec/latest/chapters/fragops.html#fragops-depth
use super::*;
use std::borrow::Cow;

type Interfaces<'a> = (
    Cow<'a, [crate::ShaderInterfaceElement]>,
    Cow<'a, [crate::ShaderInterfaceElement]>,
);

impl<'a> Emitter<'a> {
    pub(super) fn depth_interfaces(&self) -> Interfaces<'a> {
        let mut inputs = Cow::Borrowed(self.function.inputs.as_slice());
        let mut outputs = Cow::Borrowed(self.function.outputs.as_slice());
        if self.options.fragment_depth_transform.is_some() {
            for (elements, location, component) in [
                (&mut inputs, ShaderIoLocation::Position, 2),
                (&mut outputs, ShaderIoLocation::FragmentDepth, 0),
            ] {
                if !elements
                    .iter()
                    .any(|e| e.location() == location && e.component() == component)
                {
                    elements.to_mut().push(
                        crate::ShaderInterfaceElement::new(
                            location,
                            component,
                            crate::ShaderScalarType::Float32,
                            None,
                        )
                        .expect("window depth is a scalar float interface"),
                    );
                }
            }
        }
        (inputs, outputs)
    }

    pub(super) fn window_depth(&mut self, bits: u32) -> Result<u32, Error> {
        let Some(transform) = self.options.fragment_depth_transform else {
            return Ok(bits);
        };
        let [scale, offset, _, _] = transform.coefficients();
        let result = (|| {
            let scale = self.math.constant(scale.to_bits());
            let offset = self.math.constant(offset.to_bits());
            let value = self
                .math
                .float_binary(bits, scale, ShaderFloatControl::PRECISE, true)?;
            self.math
                .float_binary(value, offset, ShaderFloatControl::PRECISE, false)
        })();
        result.map_err(|e| self.error(e.to_string()))
    }

    pub(super) fn clamp_depth(&mut self, bits: u32) -> Result<u32, Error> {
        let Some(transform) = self.options.fragment_depth_transform else {
            return Ok(bits);
        };
        let [_, _, minimum, maximum] = transform.coefficients();
        let result = (|| -> Result<u32, rspirv::dr::Error> {
            let value = self.math.b.bitcast(self.math.float, None, bits)?;
            let minimum = self
                .math
                .b
                .constant_bit32(self.math.float, minimum.to_bits());
            let maximum = self
                .math
                .b
                .constant_bit32(self.math.float, maximum.to_bits());
            let below = self
                .math
                .b
                .f_ord_less_than(self.math.boolean, None, value, minimum)?;
            let value = self
                .math
                .b
                .select(self.math.float, None, below, minimum, value)?;
            let above = self
                .math
                .b
                .f_ord_greater_than(self.math.boolean, None, value, maximum)?;
            let value = self
                .math
                .b
                .select(self.math.float, None, above, maximum, value)?;
            self.math.b.bitcast(self.math.uint, None, value)
        })();
        result.map_err(|e| self.error(e.to_string()))
    }

    pub(super) fn initialize_fragment_depth(&mut self) -> Result<(), Error> {
        if self.options.fragment_depth_transform.is_none() {
            return Ok(());
        }
        // A later guest depth export replaces this value. Clamp exports only;
        // guest Position.z reads must retain the unclamped affine window value.
        let bits = self
            .io
            .load_interface(&mut self.math, true, ShaderIoLocation::Position, 2, None)
            .map_err(|e| self.error(e.to_string()))?;
        let bits = self.window_depth(bits)?;
        let bits = self.clamp_depth(bits)?;
        self.io
            .store_interface(
                &mut self.math,
                ShaderIoLocation::FragmentDepth,
                0,
                None,
                bits,
            )
            .map_err(|e| self.error(e.to_string()))
    }
}
