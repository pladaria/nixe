//! Host topology stages consume the same owned ABI as guest compilation.
use nixe_shader_recompiler::{
    Result, Stage, StageArtifact,
    artifact::{ExecutionMetadata, ShaderInfo, StageInterface, StageProvenance},
    ir::Attribute,
    spirv::Builder,
};
use spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, MemoryModel,
    Op, StorageClass,
};

/// Vulkan requires a control stage even when the guest supplies default levels.
/// Level values stay in push constants; only topology and interfaces specialize code.
pub fn default_control(
    previous: &StageInterface,
    evaluation: &StageInterface,
    points: u32,
    domain: crate::TessellationDomain,
    version: u32,
) -> Result<StageArtifact> {
    use nixe_shader_recompiler::{CompileError, diagnostics::FailureCategory};
    if points == 0 || points > 32 || evaluation.patches.iter().any(|&used| used) {
        return Err(CompileError::new(
            FailureCategory::Unsupported,
            "Default tessellation control requires 1..32 points and no per-patch user inputs",
        ));
    }
    if evaluation.inputs.clip_distances()
        || [
            Attribute::Layer,
            Attribute::ViewportIndex,
            Attribute::ViewportMask,
            Attribute::PointSize,
        ]
        .into_iter()
        .any(|attribute| evaluation.inputs.get(attribute))
    {
        return Err(CompileError::new(
            FailureCategory::Unsupported,
            "Default tessellation control forwarding of clip distances, layer, viewport or point size is not implemented",
        ));
    }
    let required_levels = match domain {
        crate::TessellationDomain::Isolines => 0b00_0011,
        crate::TessellationDomain::Triangles => 0b01_0111,
        crate::TessellationDomain::Quads => 0b11_1111,
    };
    let mut b = Builder::new(version);
    b.capability(Capability::Shader)?;
    b.capability(Capability::Tessellation)?;
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450)?;
    let void = b.type_void()?;
    let uint = b.type_int(32, false)?;
    let boolean = b.type_bool()?;
    let float = b.type_float(32)?;
    let vector = b.type_vector(float, 4)?;
    let constants = (0..=32)
        .map(|i| b.constant(uint, &[i]))
        .collect::<Result<Vec<_>>>()?;
    let input_array = b.type_array(vector, constants[32])?;
    let output_array = b.type_array(vector, constants[points as usize])?;
    let input_ptr = b.type_pointer(StorageClass::Input, input_array)?;
    let output_ptr = b.type_pointer(StorageClass::Output, output_array)?;
    let input_element = b.type_pointer(StorageClass::Input, vector)?;
    let output_element = b.type_pointer(StorageClass::Output, vector)?;
    let invocation_ptr = b.type_pointer(StorageClass::Input, uint)?;
    let invocation = b.variable(invocation_ptr, StorageClass::Input, None)?;
    b.decorate(
        invocation,
        Decoration::BuiltIn,
        &[BuiltIn::InvocationId as u32],
    )?;
    let mut interfaces = vec![invocation];
    let mut fields = Vec::new();
    let mut interface = StageInterface::default();
    for location in std::iter::once(None)
        .filter(|_| evaluation.inputs.any_component(Attribute::PositionX))
        .chain(
            (0..32)
                .filter(|&slot| previous.outputs.generic(slot) && evaluation.inputs.generic(slot))
                .map(Some),
        )
    {
        let input = b.variable(input_ptr, StorageClass::Input, None)?;
        let output = b.variable(output_ptr, StorageClass::Output, None)?;
        for id in [input, output] {
            if let Some(slot) = location {
                b.decorate(id, Decoration::Location, &[slot as u32])?;
            } else {
                b.decorate(id, Decoration::BuiltIn, &[BuiltIn::Position as u32])?;
            }
        }
        for component in 0..4 {
            let attribute = location.map_or_else(
                || Attribute::from_index(28 + component as u16).unwrap(),
                |slot| Attribute::generic(slot, component).unwrap(),
            );
            interface.inputs.set(attribute, true);
            interface.outputs.set(attribute, true);
        }
        interfaces.extend([input, output]);
        fields.push((input, output));
    }
    let mut levels = Vec::new();
    for (count, builtin) in [(4, BuiltIn::TessLevelOuter), (2, BuiltIn::TessLevelInner)] {
        let array = b.type_array(float, constants[count])?;
        let pointer = b.type_pointer(StorageClass::Output, array)?;
        let variable = b.variable(pointer, StorageClass::Output, None)?;
        b.decorate(variable, Decoration::BuiltIn, &[builtin as u32])?;
        b.decorate(variable, Decoration::Patch, &[])?;
        interfaces.push(variable);
        levels.push(variable);
    }
    let parameters = b.type_array(float, constants[6])?;
    b.decorate(parameters, Decoration::ArrayStride, &[4])?;
    let structure = b.type_struct(&[parameters])?;
    b.decorate(structure, Decoration::Block, &[])?;
    b.member_decorate(structure, 0, Decoration::Offset, &[32])?;
    let parameter_ptr = b.type_pointer(StorageClass::PushConstant, structure)?;
    let parameters = b.variable(parameter_ptr, StorageClass::PushConstant, None)?;
    if version >= 0x10400 {
        interfaces.push(parameters);
    }
    let parameter_element = b.type_pointer(StorageClass::PushConstant, float)?;
    let level_element = b.type_pointer(StorageClass::Output, float)?;
    let signature = b.type_function(void, &[])?;
    let function = b.begin_function(void, signature)?;
    b.entry_point(
        ExecutionModel::TessellationControl,
        function,
        "main",
        &interfaces,
    )?;
    b.execution_mode(function, ExecutionMode::OutputVertices, &[points])?;
    let entry = b.reserve_id()?;
    let write = b.reserve_id()?;
    let end = b.reserve_id()?;
    b.label(entry)?;
    let index = b.result(Op::Load, uint, &[invocation.0])?;
    for (input, output) in fields {
        let input = b.result(Op::AccessChain, input_element, &[input.0, index.0])?;
        let value = b.result(Op::Load, vector, &[input.0])?;
        let output = b.result(Op::AccessChain, output_element, &[output.0, index.0])?;
        b.instruction(Op::Store, &[output.0, value.0])?;
    }
    let first = b.result(Op::IEqual, boolean, &[index.0, constants[0].0])?;
    b.instruction(Op::SelectionMerge, &[end.0, 0])?;
    b.instruction(Op::BranchConditional, &[first.0, write.0, end.0])?;
    b.label(write)?;
    for lane in 0..6 {
        if required_levels & (1 << lane) == 0 {
            continue;
        }
        let pointer = b.result(
            Op::AccessChain,
            parameter_element,
            &[parameters.0, constants[0].0, constants[lane].0],
        )?;
        let value = b.result(Op::Load, float, &[pointer.0])?;
        let pointer = b.result(
            Op::AccessChain,
            level_element,
            &[
                levels[usize::from(lane >= 4)].0,
                constants[if lane >= 4 { lane - 4 } else { lane }].0,
            ],
        )?;
        b.instruction(Op::Store, &[pointer.0, value.0])?;
    }
    b.instruction(Op::Branch, &[end.0])?;
    b.label(end)?;
    b.instruction(Op::Return, &[])?;
    b.end_function()?;
    interface.tess_level_outer = true;
    interface.tess_level_inner = required_levels & 0b11_0000 != 0;
    Ok(StageArtifact {
        stage: Stage::TessellationControl,
        words: b.assemble()?.into(),
        info: ShaderInfo {
            loads: interface.inputs.clone(),
            stores: interface.outputs.clone(),
            stores_tess_level_outer: true,
            stores_tess_level_inner: interface.tess_level_inner,
            ..Default::default()
        },
        bindings: Box::new([]),
        push_constant_bytes: nixe_shader_recompiler::host_abi::PUSH_CONSTANT_BYTES,
        interface,
        execution: ExecutionMetadata {
            output_vertices: points,
            invocations: points,
            provenance: StageProvenance::HostTessellationControl,
            ..Default::default()
        },
    })
}
