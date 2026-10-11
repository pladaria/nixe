//! Reconstruct window depth after rasterization; endpoint clamping would change
//! interpolation and the depth test throughout a triangle.
use super::*;

impl VerifiedShaderIr {
    /// The host viewport must map homogeneous depth to [0,1]. Guest FragCoord.z
    /// keeps the affine window value; only the depth-test export is clamped.
    /// Explicit shader depth replaces the interpolated value without applying
    /// the viewport a second time.
    /// https://registry.khronos.org/OpenGL/extensions/NV/NV_depth_buffer_float.txt
    pub fn with_fragment_depth_transform(
        &self,
        transform: crate::FragmentDepthTransform,
    ) -> Result<Self, ShaderVerificationError> {
        let mut ir = self.0.clone();
        if ir.stage != ShaderStage::Fragment {
            return Err(ShaderVerificationError::FragmentDepthTransform(
                "requires a fragment shader",
            ));
        }
        let count = ir.register_count();
        if count > usize::from(u16::MAX) - 6 {
            return Err(ShaderVerificationError::FragmentDepthTransform(
                "virtual register space exhausted",
            ));
        }
        let [z, scale, offset, minimum, maximum, scratch] =
            std::array::from_fn(|i| ShaderRegister::new((count + i) as u16));
        let source = ShaderSourceLocation::new(
            ir.instructions
                .iter()
                .map(|i| i.source.byte_offset())
                .max()
                .unwrap_or(0)
                .checked_add(8)
                .ok_or(ShaderVerificationError::FragmentDepthTransform(
                    "source address space exhausted",
                ))?,
        );
        let [factor, bias, lower, upper] = transform.coefficients();
        if !ir
            .inputs
            .iter()
            .any(|e| e.location == ShaderIoLocation::Position && e.component == 2)
        {
            let mut inputs = ir.inputs.to_vec();
            inputs.push(
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Position,
                    2,
                    ShaderScalarType::Float32,
                    None,
                )
                .unwrap(),
            );
            ir.inputs = inputs.into_boxed_slice();
        }
        if !ir
            .outputs
            .iter()
            .any(|e| e.location == ShaderIoLocation::FragmentDepth)
        {
            let mut outputs = ir.outputs.to_vec();
            outputs.push(
                ShaderInterfaceElement::new(
                    ShaderIoLocation::FragmentDepth,
                    0,
                    ShaderScalarType::Float32,
                    None,
                )
                .unwrap(),
            );
            ir.outputs = outputs.into_boxed_slice();
        }
        let make = |operation| ShaderInstruction::new(source, ShaderPredicate::Always, operation);
        let mut instructions = Vec::with_capacity(ir.instructions.len() + 12);
        for (destination, value) in [
            (scale, factor),
            (offset, bias),
            (minimum, lower),
            (maximum, upper),
        ] {
            instructions.push(make(ShaderOperation::MoveImmediate32 {
                destination,
                bits: value.to_bits(),
                scalar_type: ShaderScalarType::Float32,
            }));
        }
        instructions.push(make(ShaderOperation::LoadInput {
            destinations: Box::new([z]),
            location: ShaderIoLocation::Position,
            first_component: 2,
            scalar_type: ShaderScalarType::Float32,
        }));
        let affine = |destination, predicate, source| {
            [
                ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::Multiply32 {
                        destination,
                        left: destination,
                        right: scale,
                        scalar_type: ShaderScalarType::Float32,
                        float_control: ShaderFloatControl::PRECISE,
                    },
                ),
                ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::Add32 {
                        destination,
                        left: destination,
                        right: offset,
                        scalar_type: ShaderScalarType::Float32,
                        float_control: ShaderFloatControl::PRECISE,
                    },
                ),
            ]
        };
        let clamp = |value, predicate, source| {
            [
                ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::FloatMinMax32 {
                        destination: scratch,
                        left: value,
                        right: minimum,
                        minimum: ShaderPredicate::Never,
                        float_control: ShaderFloatControl::PRECISE,
                    },
                ),
                ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::FloatMinMax32 {
                        destination: scratch,
                        left: scratch,
                        right: maximum,
                        minimum: ShaderPredicate::Always,
                        float_control: ShaderFloatControl::PRECISE,
                    },
                ),
            ]
        };
        instructions.extend(affine(z, ShaderPredicate::Always, source));
        instructions.extend(clamp(z, ShaderPredicate::Always, source));
        instructions.push(make(ShaderOperation::StoreOutput {
            sources: Box::new([scratch]),
            location: ShaderIoLocation::FragmentDepth,
            first_component: 0,
            scalar_type: ShaderScalarType::Float32,
        }));
        for instruction in ir.instructions.iter() {
            match &instruction.operation {
                ShaderOperation::StoreOutput {
                    sources,
                    location: ShaderIoLocation::FragmentDepth,
                    ..
                } => {
                    instructions.extend(clamp(
                        sources[0],
                        instruction.predicate,
                        instruction.source,
                    ));
                    let mut store = instruction.clone();
                    if let ShaderOperation::StoreOutput { sources, .. } = &mut store.operation {
                        *sources = Box::new([scratch]);
                    }
                    instructions.push(store);
                }
                ShaderOperation::LoadInput {
                    destinations,
                    location: ShaderIoLocation::Position,
                    first_component,
                    ..
                } => {
                    instructions.push(instruction.clone());
                    if *first_component <= 2
                        && let Some(destination) =
                            destinations.get(usize::from(2 - first_component))
                    {
                        instructions.extend(affine(
                            *destination,
                            instruction.predicate,
                            instruction.source,
                        ));
                    }
                }
                _ => instructions.push(instruction.clone()),
            }
        }
        ir.instructions = instructions.into_boxed_slice();
        Self::verify(ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_depth_clamps_after_interpolation_without_changing_guest_position_reads() {
        let source = ShaderSourceLocation::new(0);
        let make = |operation| ShaderInstruction::new(source, ShaderPredicate::Always, operation);
        let input = ShaderInterfaceElement::new(
            ShaderIoLocation::Position,
            2,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let output = ShaderInterfaceElement::new(
            ShaderIoLocation::Color(0),
            0,
            ShaderScalarType::Float32,
            None,
        )
        .unwrap();
        let original = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![input],
            vec![output],
            vec![],
            vec![
                make(ShaderOperation::LoadInput {
                    destinations: Box::new([ShaderRegister(0)]),
                    location: ShaderIoLocation::Position,
                    first_component: 2,
                    scalar_type: ShaderScalarType::Float32,
                }),
                make(ShaderOperation::StoreOutput {
                    sources: Box::new([ShaderRegister(0)]),
                    location: ShaderIoLocation::Color(0),
                    first_component: 0,
                    scalar_type: ShaderScalarType::Float32,
                }),
                make(ShaderOperation::Exit),
            ],
        ))
        .unwrap();
        for (factor, bias) in [(2.0, -1.0), (-2.0, 1.0)] {
            let shader = original
                .with_fragment_depth_transform(
                    crate::FragmentDepthTransform::new(factor, bias, 0.0, 1.0).unwrap(),
                )
                .unwrap();
            for host in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                let result = evaluate_shader_ir(
                    &shader,
                    &ShaderEvaluationInputs::default().with_interface_bits(
                        ShaderIoLocation::Position,
                        2,
                        host.to_bits(),
                    ),
                    1000,
                )
                .unwrap();
                let guest = host * factor + bias;
                assert_eq!(
                    f32::from_bits(result.output_bits(ShaderIoLocation::Color(0), 0).unwrap()),
                    guest
                );
                assert_eq!(
                    f32::from_bits(
                        result
                            .output_bits(ShaderIoLocation::FragmentDepth, 0)
                            .unwrap()
                    ),
                    guest.clamp(0.0, 1.0)
                );
            }
            let wgsl = lower_shader_ir_to_wgsl(&shader).unwrap();
            let parsed = naga::front::wgsl::parse_str(wgsl.source()).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&parsed)
            .unwrap();
        }
    }

    #[test]
    fn explicit_depth_is_clamped_without_applying_viewport_twice() {
        let source = ShaderSourceLocation::new(0);
        let make = |operation| ShaderInstruction::new(source, ShaderPredicate::Always, operation);
        let original = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::FragmentDepth,
                    0,
                    ShaderScalarType::Float32,
                    None,
                )
                .unwrap(),
            ],
            vec![],
            vec![
                make(ShaderOperation::MoveImmediate32 {
                    destination: ShaderRegister(0),
                    bits: 0.75_f32.to_bits(),
                    scalar_type: ShaderScalarType::Float32,
                }),
                make(ShaderOperation::StoreOutput {
                    sources: Box::new([ShaderRegister(0)]),
                    location: ShaderIoLocation::FragmentDepth,
                    first_component: 0,
                    scalar_type: ShaderScalarType::Float32,
                }),
                make(ShaderOperation::Exit),
            ],
        ))
        .unwrap();
        let transformed = original
            .with_fragment_depth_transform(
                crate::FragmentDepthTransform::new(2.0, -1.0, 0.0, 1.0).unwrap(),
            )
            .unwrap();
        let result = evaluate_shader_ir(
            &transformed,
            &ShaderEvaluationInputs::default().with_interface_bits(
                ShaderIoLocation::Position,
                2,
                0.1_f32.to_bits(),
            ),
            1000,
        )
        .unwrap();
        assert_eq!(
            result.output_bits(ShaderIoLocation::FragmentDepth, 0),
            Some(0.75_f32.to_bits())
        );
    }
}
