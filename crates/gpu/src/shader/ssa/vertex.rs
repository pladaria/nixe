//! Exact legalization for 32-bit scaled attributes: Vulkan exposes the integer
//! fetch format, and SSA converts it to the guest's floating-point attribute.
use super::*;
/// `signed` identifies the physical integer input, not a reinterpretation of
/// the converted result. Invoke on a shader variant miss before optimization.
pub fn scale_vertex_inputs(function: &mut Function, inputs: &[(u32, bool)]) -> Result<(), Error> {
    if function.stage != ShaderStage::Vertex {
        return Err(Error::at(
            ShaderSourceLocation::new(0),
            "scaled vertex legalization requires a vertex shader",
        ));
    }
    for &(location, signed) in inputs {
        let location = ShaderIoLocation::Generic(u8::try_from(location).map_err(|_| {
            Error::at(
                ShaderSourceLocation::new(0),
                "generic vertex location exceeds the shader interface",
            )
        })?);
        let ty = if signed { Type::I32 } else { Type::U32 };
        for (block_id, block) in function.blocks.iter_mut().enumerate() {
            let mut rewritten = Vec::with_capacity(block.instructions.len());
            for mut instruction in std::mem::take(&mut block.instructions) {
                if let Op::Input {
                    location: l,
                    component,
                    ty: Type::F32,
                } = instruction.op
                    && l == location
                {
                    let result = instruction.results[0];
                    let physical = ValueId(u32::try_from(function.values.len()).map_err(|_| {
                        Error::at(
                            instruction.source,
                            "scaled vertex values exceed dense SSA IDs",
                        )
                    })?);
                    function.values.push(Value {
                        ty,
                        definition: Definition::Instruction {
                            block: BlockId(block_id as u32),
                            index: rewritten.len() as u32,
                            result: 0,
                        },
                    });
                    instruction.op = Op::Input {
                        location,
                        component,
                        ty,
                    };
                    instruction.results = Box::new([physical]);
                    let source = instruction.source;
                    rewritten.push(instruction);
                    rewritten.push(Instruction {
                        op: Op::Convert {
                            from: ty,
                            to: Type::F32,
                            rounding: ShaderRoundingMode::NearestEven,
                            flush_denormals: false,
                        },
                        arguments: Box::new([physical]),
                        results: Box::new([result]),
                        source,
                    });
                } else {
                    rewritten.push(instruction);
                }
            }
            block.instructions = rewritten;
        }
        for input in function
            .inputs
            .iter_mut()
            .filter(|i| i.location() == location)
        {
            *input = ShaderInterfaceElement::new(
                location,
                input.component(),
                if signed {
                    crate::ShaderScalarType::Signed32
                } else {
                    crate::ShaderScalarType::Unsigned32
                },
                input.interpolation(),
            )
            .map_err(|e| Error::at(ShaderSourceLocation::new(0), format!("{e:?}")))?;
        }
    }
    function.recompact();
    function.optimize()
}
