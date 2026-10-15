//! Exact quad assembly: VS executes on the four original corners, and a host
//! geometry shader emits the 0--2 diagonal. Flat outputs come from the selected guest provoking corner.
//! No vertex re-evaluation, index rotation or duplicated guest storage effects.
//! https://registry.khronos.org/OpenGL/extensions/NV/NV_geometry_program4.txt
use super::*;
use rspirv::dr::Builder;
/// Input assembly is LINE_LIST_WITH_ADJACENCY (four vertices per primitive).
/// This host lowering does not add support for guest geometry instructions.
pub fn emit_quad_geometry(vertex: &Function, provoking_first: bool) -> Result<Module, Error> {
    if vertex.stage != ShaderStage::Vertex {
        return Err(Error::at(
            ShaderSourceLocation::new(0),
            "quad assembly requires linked vertex outputs",
        ));
    }
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spv::Capability::Shader);
    b.capability(spv::Capability::Geometry);
    b.memory_model(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    let uint = b.type_int(32, 0);
    let int = b.type_int(32, 1);
    let float = b.type_float(32, None);
    let void = b.type_void();
    let four = b.constant_bit32(uint, 4);
    let indices: Vec<_> = (0..4).map(|i| b.constant_bit32(uint, i)).collect();
    let mut groups = BTreeMap::new();
    for output in &vertex.outputs {
        if output.location() == ShaderIoLocation::PointSize {
            continue;
        }
        let group = groups.entry(output.location()).or_insert((
            output.scalar_type(),
            output.interpolation(),
            0u32,
        ));
        if group.0 != output.scalar_type() || group.1 != output.interpolation() {
            return Err(Error::at(
                ShaderSourceLocation::new(0),
                "quad output group has incompatible types/interpolation",
            ));
        }
        group.2 = group.2.max(u32::from(output.component()) + 1);
    }
    let mut interface = Vec::new();
    let mut fields = Vec::new();
    for (location, (scalar, interpolation, components)) in groups {
        let ty = match scalar {
            crate::ShaderScalarType::Float32 => float,
            crate::ShaderScalarType::Unsigned32 => uint,
            crate::ShaderScalarType::Signed32 => int,
            _ => {
                return Err(Error::at(
                    ShaderSourceLocation::new(0),
                    "quad assembly supports only 32-bit stage interfaces",
                ));
            }
        };
        let count = if location == ShaderIoLocation::Position {
            4
        } else {
            components
        };
        let ty = if count == 1 {
            ty
        } else {
            b.type_vector(ty, count)
        };
        let array = b.type_array(ty, four);
        let input_ptr = b.type_pointer(None, spv::StorageClass::Input, array);
        let output_ptr = b.type_pointer(None, spv::StorageClass::Output, ty);
        let element_ptr = b.type_pointer(None, spv::StorageClass::Input, ty);
        let input = b.variable(input_ptr, None, spv::StorageClass::Input, None);
        let output = b.variable(output_ptr, None, spv::StorageClass::Output, None);
        for variable in [input, output] {
            match location {
                ShaderIoLocation::Position => b.decorate(
                    variable,
                    spv::Decoration::BuiltIn,
                    [SpvOperand::BuiltIn(spv::BuiltIn::Position)],
                ),
                ShaderIoLocation::PointSize => b.decorate(
                    variable,
                    spv::Decoration::BuiltIn,
                    [SpvOperand::BuiltIn(spv::BuiltIn::PointSize)],
                ),
                ShaderIoLocation::Generic(slot) => b.decorate(
                    variable,
                    spv::Decoration::Location,
                    [SpvOperand::LiteralBit32(u32::from(slot))],
                ),
                _ => {
                    return Err(Error::at(
                        ShaderSourceLocation::new(0),
                        format!("quad assembly cannot forward {location:?}"),
                    ));
                }
            }
            if interpolation == Some(crate::ShaderInterpolation::Constant) {
                b.decorate(variable, spv::Decoration::Flat, []);
            } else if interpolation == Some(crate::ShaderInterpolation::ScreenLinear) {
                b.decorate(variable, spv::Decoration::NoPerspective, []);
            }
        }
        interface.extend([input, output]);
        fields.push((
            input,
            output,
            ty,
            element_ptr,
            interpolation == Some(crate::ShaderInterpolation::Constant),
        ));
    }
    let signature = b.type_function(void, []);
    let function = b
        .begin_function(void, None, spv::FunctionControl::NONE, signature)
        .map_err(|e| Error::at(ShaderSourceLocation::new(0), e.to_string()))?;
    b.entry_point(spv::ExecutionModel::Geometry, function, "main", interface);
    b.execution_mode(function, spv::ExecutionMode::InputLinesAdjacency, []);
    b.execution_mode(function, spv::ExecutionMode::OutputTriangleStrip, []);
    b.execution_mode(function, spv::ExecutionMode::OutputVertices, [6]);
    b.execution_mode(function, spv::ExecutionMode::Invocations, [1]);
    b.begin_block(None).unwrap();
    for triangle in [[0usize, 1, 2], [0, 2, 3]] {
        for corner in triangle {
            for &(input, output, ty, pointer, flat) in &fields {
                let ptr = b
                    .access_chain(
                        pointer,
                        None,
                        input,
                        [indices[if flat {
                            if provoking_first { 0 } else { 3 }
                        } else {
                            corner
                        }]],
                    )
                    .unwrap();
                let value = b.load(ty, None, ptr, None, []).unwrap();
                b.store(output, value, None, []).unwrap();
            }
            b.emit_vertex().unwrap();
        }
        b.end_primitive().unwrap();
    }
    b.ret().unwrap();
    b.end_function().unwrap();
    Ok(Module {
        words: b.module().assemble().into_boxed_slice(),
        bindings: Box::new([]),
        push_constant_bytes: 0,
    })
}
