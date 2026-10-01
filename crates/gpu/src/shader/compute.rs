//! Compute execution metadata and storage operations, independent of guest ISA.
use super::*;

/// Unsigned compute invocation coordinates.
/// https://www.w3.org/TR/WGSL/#builtin-values
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderComputeBuiltin {
    GlobalInvocationId,
    LocalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    LocalInvocationIndex,
}

impl ShaderComputeBuiltin {
    pub(super) const fn wgsl_name(self) -> &'static str {
        match self {
            Self::GlobalInvocationId => "global_invocation_id",
            Self::LocalInvocationId => "local_invocation_id",
            Self::WorkgroupId => "workgroup_id",
            Self::NumWorkgroups => "num_workgroups",
            Self::LocalInvocationIndex => "local_invocation_index",
        }
    }
}

pub(super) fn verify(ir: &ShaderIr) -> Result<(), ShaderVerificationError> {
    if ir.stage == ShaderStage::Compute {
        if ir.workgroup_size.is_none_or(|size| size.contains(&0))
            || !ir.inputs.is_empty()
            || !ir.outputs.is_empty()
        {
            return Err(ShaderVerificationError::InvalidComputeMetadata);
        }
    } else if ir.workgroup_size.is_some() {
        return Err(ShaderVerificationError::InvalidComputeMetadata);
    }
    for instruction in &ir.instructions {
        let invalid = |reason| ShaderVerificationError::InvalidComputeOperation {
            source: instruction.source,
            reason,
        };
        let access = match instruction.operation {
            ShaderOperation::LoadComputeBuiltin32 {
                builtin, component, ..
            } => {
                let components = if builtin == ShaderComputeBuiltin::LocalInvocationIndex {
                    1
                } else {
                    3
                };
                if ir.stage != ShaderStage::Compute || component >= components {
                    return Err(invalid("invalid builtin stage or component"));
                }
                None
            }
            ShaderOperation::LoadStorageBuffer32 { binding, .. } => Some((binding, false)),
            ShaderOperation::StoreStorageBuffer32 { binding, .. } => Some((binding, true)),
            _ => None,
        };
        if let Some((binding, write)) = access {
            if ir.stage != ShaderStage::Compute {
                return Err(invalid("storage operations require the compute stage"));
            }
            if !ir.resources.iter().any(|resource| {
                resource.binding == binding
                    && resource.kind == ShaderResourceKind::StorageBuffer
                    && if write {
                        resource.writable
                    } else {
                        resource.readable
                    }
            }) {
                return Err(ShaderVerificationError::UndeclaredResourceAccess {
                    source: instruction.source,
                    binding,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn emit_entry(source: &mut String, size: [u32; 3]) {
    source.push_str(&format!(
        "@compute @workgroup_size({}, {}, {})\nfn main(\n",
        size[0], size[1], size[2]
    ));
    for builtin in [
        ShaderComputeBuiltin::GlobalInvocationId,
        ShaderComputeBuiltin::LocalInvocationId,
        ShaderComputeBuiltin::WorkgroupId,
        ShaderComputeBuiltin::NumWorkgroups,
        ShaderComputeBuiltin::LocalInvocationIndex,
    ] {
        let name = builtin.wgsl_name();
        let ty = if builtin == ShaderComputeBuiltin::LocalInvocationIndex {
            "u32"
        } else {
            "vec3<u32>"
        };
        source.push_str(&format!("  @builtin({name}) {name}: {ty},\n"));
    }
    source.push_str(") {\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program() -> ShaderIr {
        ShaderIr::new(
            ShaderStage::Compute,
            vec![],
            vec![],
            vec![
                ShaderResourceAccess::new(0, ShaderResourceKind::StorageBuffer, true, true)
                    .unwrap(),
            ],
            [
                ShaderOperation::LoadComputeBuiltin32 {
                    destination: ShaderRegister::new(0),
                    builtin: ShaderComputeBuiltin::GlobalInvocationId,
                    component: 0,
                },
                ShaderOperation::LoadStorageBuffer32 {
                    destination: ShaderRegister::new(1),
                    binding: 0,
                    word_index: ShaderRegister::new(0),
                },
                ShaderOperation::StoreStorageBuffer32 {
                    source: ShaderRegister::new(1),
                    binding: 0,
                    word_index: ShaderRegister::new(0),
                },
                ShaderOperation::Exit,
            ]
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
        .with_workgroup_size([32, 1, 1])
    }

    fn validate_wgsl(ir: ShaderIr) {
        let verified = VerifiedShaderIr::verify(ir).unwrap();
        let wgsl = lower_shader_ir_to_wgsl(&verified).unwrap();
        let module = naga::front::wgsl::parse_str(wgsl.source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [32, 1, 1]);
    }

    #[test]
    fn compute_storage_and_builtins_lower_to_valid_wgsl() {
        for builtin in [
            ShaderComputeBuiltin::GlobalInvocationId,
            ShaderComputeBuiltin::LocalInvocationId,
            ShaderComputeBuiltin::WorkgroupId,
            ShaderComputeBuiltin::NumWorkgroups,
            ShaderComputeBuiltin::LocalInvocationIndex,
        ] {
            for component in 0..if builtin == ShaderComputeBuiltin::LocalInvocationIndex {
                1
            } else {
                3
            } {
                let mut ir = program();
                ir.instructions[0].operation = ShaderOperation::LoadComputeBuiltin32 {
                    destination: ShaderRegister::new(0),
                    builtin,
                    component,
                };
                validate_wgsl(ir);
            }
        }
    }

    #[test]
    fn compute_cfg_returns_void_and_storage_stores_are_live() {
        let mut ir = program();
        assert_eq!(liveness::live_instructions(&ir).unwrap(), vec![true; 4]);
        let mut instructions = ir.instructions.into_vec();
        instructions.insert(
            2,
            ShaderInstruction::new(
                ShaderSourceLocation::new(12),
                ShaderPredicate::Always,
                ShaderOperation::Branch {
                    target: ShaderSourceLocation::new(16),
                },
            ),
        );
        ir.instructions = instructions.into_boxed_slice();
        validate_wgsl(ir);
    }

    #[test]
    fn compute_metadata_and_resource_permissions_are_verified() {
        for size in [None, Some([0, 1, 1]), Some([1, 0, 1]), Some([1, 1, 0])] {
            let mut ir = program();
            ir.workgroup_size = size;
            assert_eq!(
                VerifiedShaderIr::verify(ir),
                Err(ShaderVerificationError::InvalidComputeMetadata)
            );
        }
        let mut ir = program();
        ir.stage = ShaderStage::Vertex;
        assert_eq!(
            VerifiedShaderIr::verify(ir),
            Err(ShaderVerificationError::InvalidComputeMetadata)
        );
        for (read, write) in [(true, false), (false, true)] {
            let mut ir = program();
            ir.resources = vec![
                ShaderResourceAccess::new(0, ShaderResourceKind::StorageBuffer, read, write)
                    .unwrap(),
            ]
            .into_boxed_slice();
            assert!(matches!(
                VerifiedShaderIr::verify(ir),
                Err(ShaderVerificationError::UndeclaredResourceAccess { .. })
            ));
        }
        let mut ir = program();
        ir.instructions[0].operation = ShaderOperation::LoadComputeBuiltin32 {
            destination: ShaderRegister::new(0),
            builtin: ShaderComputeBuiltin::GlobalInvocationId,
            component: 3,
        };
        assert!(matches!(
            VerifiedShaderIr::verify(ir),
            Err(ShaderVerificationError::InvalidComputeOperation { .. })
        ));
        let mut ir = program();
        ir.instructions[1].operation = ShaderOperation::LoadStorageBuffer32 {
            destination: ShaderRegister::new(1),
            binding: 0,
            word_index: ShaderRegister::new(2),
        };
        assert!(matches!(
            VerifiedShaderIr::verify(ir),
            Err(ShaderVerificationError::UndefinedRegister { .. })
        ));
    }
}
