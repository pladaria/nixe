//! Patch semantics shared by shader frontends and native shader emitters.
use super::*;

pub(super) fn verify_metadata(ir: &ShaderIr) -> Result<(), ShaderVerificationError> {
    if match (ir.stage, ir.tessellation_control_points) {
        (ShaderStage::TessellationControl, Some(points)) => points != 0,
        (ShaderStage::TessellationControl, None) | (_, Some(_)) => false,
        (_, None) => true,
    } {
        Ok(())
    } else {
        Err(ShaderVerificationError::InvalidTessellationMetadata)
    }
}

pub(super) fn per_vertex(location: ShaderIoLocation) -> bool {
    matches!(
        location,
        ShaderIoLocation::Position
            | ShaderIoLocation::PointSize
            | ShaderIoLocation::Generic(_)
            | ShaderIoLocation::Color(_)
    )
}

fn per_patch(location: ShaderIoLocation) -> bool {
    matches!(
        location,
        ShaderIoLocation::Patch(_)
            | ShaderIoLocation::TessLevelOuter
            | ShaderIoLocation::TessLevelInner
    )
}

pub(super) fn verify_interface(
    stage: ShaderStage,
    input: bool,
    element: &ShaderInterfaceElement,
) -> Result<(), ShaderVerificationError> {
    use ShaderIoLocation as L;
    use ShaderStage as S;
    let valid = match element.location {
        L::InvocationId => {
            input
                && stage == S::TessellationControl
                && element.scalar_type == ShaderScalarType::Unsigned32
        }
        L::PatchVertices => {
            input
                && matches!(stage, S::TessellationControl | S::TessellationEvaluation)
                && element.scalar_type == ShaderScalarType::Unsigned32
        }
        L::PrimitiveId => {
            input
                && matches!(
                    stage,
                    S::TessellationControl | S::TessellationEvaluation | S::Geometry | S::Fragment
                )
                && element.scalar_type == ShaderScalarType::Unsigned32
        }
        L::TessCoord => {
            input
                && stage == S::TessellationEvaluation
                && element.scalar_type == ShaderScalarType::Float32
        }
        L::TessLevelOuter | L::TessLevelInner => {
            ((stage == S::TessellationControl && !input)
                || (stage == S::TessellationEvaluation && input))
                && element.scalar_type == ShaderScalarType::Float32
        }
        L::Patch(_) => {
            (stage == S::TessellationControl && !input)
                || (stage == S::TessellationEvaluation && input)
        }
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(ShaderVerificationError::InvalidTessellationInterface {
            stage,
            input,
            location: element.location,
        })
    }
}

pub(super) fn verify_scalar_access(
    ir: &ShaderIr,
    source: ShaderSourceLocation,
    input: bool,
    location: ShaderIoLocation,
) -> Result<(), ShaderVerificationError> {
    if per_vertex(location)
        && (ir.stage == ShaderStage::TessellationControl
            || (input && ir.stage == ShaderStage::TessellationEvaluation))
    {
        return Err(ShaderVerificationError::InvalidPatchOperation {
            source,
            reason: "per-vertex patch I/O requires a control-point index",
        });
    }
    Ok(())
}

pub(super) fn verify_operation(
    ir: &ShaderIr,
    instruction: &ShaderInstruction,
    index: usize,
) -> Result<(), ShaderVerificationError> {
    let invalid = |reason| ShaderVerificationError::InvalidPatchOperation {
        source: instruction.source,
        reason,
    };
    let (input, location, component) = match instruction.operation {
        ShaderOperation::LoadControlPoint {
            output,
            location,
            component,
            ..
        } => {
            if !per_vertex(location)
                || !(ir.stage == ShaderStage::TessellationControl
                    || (!output && ir.stage == ShaderStage::TessellationEvaluation))
            {
                return Err(invalid(
                    "indexed control-point read requires TCS/TES per-vertex input or TCS output",
                ));
            }
            (!output, location, component)
        }
        ShaderOperation::StoreControlPoint {
            location,
            component,
            ..
        } => {
            if ir.stage != ShaderStage::TessellationControl || !per_vertex(location) {
                return Err(invalid(
                    "indexed control-point write requires TCS per-vertex output",
                ));
            }
            (false, location, component)
        }
        ShaderOperation::LoadPatchOutput {
            location,
            component,
            ..
        } => {
            if ir.stage != ShaderStage::TessellationControl || !per_patch(location) {
                return Err(invalid("patch output read requires TCS per-patch output"));
            }
            (false, location, component)
        }
        ShaderOperation::PatchBarrier => {
            if ir.stage != ShaderStage::TessellationControl {
                return Err(invalid("patch barrier requires a control shader"));
            }
            // Until uniform-control-flow analysis is available, accept only
            // straight-line rendezvous reached by every invocation. Do not
            // silently weaken an execution barrier into a memory fence.
            if instruction.predicate != ShaderPredicate::Always
                || ir
                    .instructions
                    .iter()
                    .any(|i| matches!(i.operation, ShaderOperation::Branch { .. }))
                || ir.instructions[..index]
                    .iter()
                    .any(|i| matches!(i.operation, ShaderOperation::Exit))
            {
                return Err(invalid(
                    "patch barrier uniformity is not established for conditional control flow",
                ));
            }
            return Ok(());
        }
        _ => unreachable!("patch verifier called for non-patch operation"),
    };
    verify_interface_range(ir, instruction.source, input, location, component, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(
        location: ShaderIoLocation,
        component: u8,
        scalar: ShaderScalarType,
    ) -> ShaderInterfaceElement {
        ShaderInterfaceElement::new(location, component, scalar, None).unwrap()
    }

    fn control_shader(operations: Vec<ShaderOperation>) -> ShaderIr {
        ShaderIr::new(
            ShaderStage::TessellationControl,
            vec![
                element(
                    ShaderIoLocation::InvocationId,
                    0,
                    ShaderScalarType::Unsigned32,
                ),
                element(ShaderIoLocation::Generic(0), 2, ShaderScalarType::Float32),
            ],
            vec![
                element(ShaderIoLocation::Generic(0), 2, ShaderScalarType::Float32),
                element(ShaderIoLocation::Patch(0), 1, ShaderScalarType::Float32),
            ],
            vec![],
            operations
                .into_iter()
                .enumerate()
                .map(|(index, operation)| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(index as u32 * 8),
                        ShaderPredicate::Always,
                        operation,
                    )
                })
                .collect(),
        )
        .with_tessellation_control_points(Some(5))
    }

    fn invocation() -> ShaderOperation {
        ShaderOperation::LoadInput {
            destinations: vec![ShaderRegister::new(0)].into_boxed_slice(),
            location: ShaderIoLocation::InvocationId,
            first_component: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        }
    }

    #[test]
    fn arrayed_io_output_reads_and_patch_barrier_have_verified_neutral_storage() {
        let ir = VerifiedShaderIr::verify(control_shader(vec![
            invocation(),
            ShaderOperation::LoadControlPoint {
                destination: ShaderRegister::new(1),
                vertex: ShaderRegister::new(0),
                output: false,
                location: ShaderIoLocation::Generic(0),
                component: 2,
            },
            ShaderOperation::StoreControlPoint {
                source: ShaderRegister::new(1),
                vertex: ShaderRegister::new(0),
                location: ShaderIoLocation::Generic(0),
                component: 2,
            },
            ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(1)].into_boxed_slice(),
                location: ShaderIoLocation::Patch(0),
                first_component: 1,
                scalar_type: ShaderScalarType::Float32,
            },
            ShaderOperation::PatchBarrier,
            ShaderOperation::LoadControlPoint {
                destination: ShaderRegister::new(2),
                vertex: ShaderRegister::new(0),
                output: true,
                location: ShaderIoLocation::Generic(0),
                component: 2,
            },
            ShaderOperation::LoadPatchOutput {
                destination: ShaderRegister::new(3),
                location: ShaderIoLocation::Patch(0),
                component: 1,
            },
            ShaderOperation::Exit,
        ]))
        .unwrap();
        assert_eq!(ir.ir().tessellation_control_points(), Some(5));
        let module = ShaderBackendModule::new(ir);
        assert!(module.retained_bytes() > std::mem::size_of::<ShaderIr>());
        assert_eq!(
            lower_shader_ir_to_wgsl(module.ir()),
            Err(ShaderBackendLoweringError::UnsupportedStage(
                ShaderStage::TessellationControl
            ))
        );
    }

    #[test]
    fn patch_metadata_and_builtin_shapes_are_validated() {
        let ir = control_shader(vec![ShaderOperation::Exit]);
        for points in [None, Some(0)] {
            assert_eq!(
                VerifiedShaderIr::verify(ir.clone().with_tessellation_control_points(points)),
                Err(ShaderVerificationError::InvalidTessellationMetadata)
            );
        }
        let vertex = ShaderIr::new(
            ShaderStage::Vertex,
            vec![],
            vec![],
            vec![],
            vec![ShaderInstruction::new(
                ShaderSourceLocation::new(0),
                ShaderPredicate::Always,
                ShaderOperation::Exit,
            )],
        )
        .with_tessellation_control_points(Some(3));
        assert_eq!(
            VerifiedShaderIr::verify(vertex),
            Err(ShaderVerificationError::InvalidTessellationMetadata)
        );
        for (location, count) in [
            (ShaderIoLocation::TessCoord, 3),
            (ShaderIoLocation::TessLevelInner, 2),
            (ShaderIoLocation::TessLevelOuter, 4),
        ] {
            assert!(
                ShaderInterfaceElement::new(location, count, ShaderScalarType::Float32, None)
                    .is_err()
            );
            assert!(
                ShaderInterfaceElement::new(location, count - 1, ShaderScalarType::Float32, None)
                    .is_ok()
            );
        }
    }

    #[test]
    fn indexed_patch_access_requires_defined_indices_declared_components_and_correct_stage() {
        let load = ShaderOperation::LoadControlPoint {
            destination: ShaderRegister::new(1),
            vertex: ShaderRegister::new(0),
            output: false,
            location: ShaderIoLocation::Generic(0),
            component: 2,
        };
        assert!(
            matches!(VerifiedShaderIr::verify(control_shader(vec![load.clone(), ShaderOperation::Exit])), Err(ShaderVerificationError::UndefinedRegister { register, .. }) if register == ShaderRegister::new(0))
        );
        let scalar = ShaderOperation::LoadInput {
            destinations: vec![ShaderRegister::new(1)].into_boxed_slice(),
            location: ShaderIoLocation::Generic(0),
            first_component: 2,
            scalar_type: ShaderScalarType::Float32,
        };
        assert!(matches!(
            VerifiedShaderIr::verify(control_shader(vec![scalar, ShaderOperation::Exit])),
            Err(ShaderVerificationError::InvalidPatchOperation { .. })
        ));
        let mut wrong = control_shader(vec![invocation(), load, ShaderOperation::Exit]);
        wrong.stage = ShaderStage::Vertex;
        wrong.tessellation_control_points = None;
        assert!(matches!(
            VerifiedShaderIr::verify(wrong),
            Err(ShaderVerificationError::InvalidTessellationInterface { .. })
        ));
        let load = ShaderOperation::LoadPatchOutput {
            destination: ShaderRegister::new(1),
            location: ShaderIoLocation::Patch(0),
            component: 0,
        };
        assert!(matches!(
            VerifiedShaderIr::verify(control_shader(vec![load, ShaderOperation::Exit])),
            Err(ShaderVerificationError::UndeclaredInterfaceAccess { .. })
        ));
    }

    #[test]
    fn barrier_does_not_claim_unproven_control_flow_uniformity() {
        let ir = control_shader(vec![
            ShaderOperation::Branch {
                target: ShaderSourceLocation::new(8),
            },
            ShaderOperation::PatchBarrier,
            ShaderOperation::Exit,
        ]);
        assert!(matches!(
            VerifiedShaderIr::verify(ir),
            Err(ShaderVerificationError::InvalidPatchOperation {
                reason: "patch barrier uniformity is not established for conditional control flow",
                ..
            })
        ));
    }
}
