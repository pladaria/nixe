//! Direct Maxwell CFG frontend. No ShaderIr/VerifiedShaderIr importer is used.
//! Checked ISA field decoders are shared with the oracle, not its register IR,
//! verifier, evaluator, WGSL emitter or host pipeline conventions.

use super::binary::{
    MaxwellShaderBinary, MaxwellShaderMemoryView, MaxwellShaderMetadata, read_shader_binary,
    read_shader_code, validate_program_header,
};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::{MaxwellGpuAddressSpace, MaxwellShaderStage};
use nixe_gpu::ssa::{
    Binary, BlockId, Constant, Function, FunctionBuilder, InitialValue, MemorySpace, Op, Operand,
    ResourceId, Terminator, Type, Unary, VariableId,
};
use nixe_gpu::{
    ShaderFloatControl, ShaderPredicate, ShaderPredicateSetOperation, ShaderScalarType,
    ShaderSourceLocation,
};
use nixe_memory::CanonicalWriteBatch;
use std::collections::BTreeMap;

mod cache;
mod demand;
mod link;
pub use cache::{
    MAXWELL_SHADER_NAMESPACE, MaxwellShaderCode, MaxwellShaderCompiler, MaxwellShaderSnapshot,
};
mod program;
mod vertex;
pub use program::{
    MaxwellSsaBinding, MaxwellSsaBindingSource, MaxwellSsaPipeline, MaxwellSsaProgram,
    MaxwellSsaSource, compile_maxwell_pipeline_ssa,
};

/// Cold compilation inputs, selected by the concrete Maxwell frontend. A
/// compute program is headerless and takes its geometry from the consumed QMD.
pub struct MaxwellSsaRequest {
    pub stage: MaxwellShaderStage,
    pub address: u64,
    pub register_count: u8,
    pub workgroup_size: Option<[u32; 3]>,
}

/// Compile an actual bounded canonical program snapshot. Shader cache ownership
/// is added by task 09; pipeline linkage is available through the pipeline
/// entry point. This API introduces no backend dispatch and cannot silently fall back to the oracle.
pub fn compile_maxwell_shader_ssa(
    address_space: &MaxwellGpuAddressSpace,
    writes: &CanonicalWriteBatch,
    request: MaxwellSsaRequest,
) -> Result<MaxwellSsaProgram, MaxwellShaderTranslationError> {
    let binary = read_requested_binary(address_space, writes, &request)?;
    lower_binary(&binary, request.register_count)
}
fn read_requested_binary(
    address_space: &MaxwellGpuAddressSpace,
    writes: &CanonicalWriteBatch,
    request: &MaxwellSsaRequest,
) -> Result<MaxwellShaderBinary, MaxwellShaderTranslationError> {
    let memory = MaxwellShaderMemoryView::new(address_space, writes);
    let binary = if request.stage == MaxwellShaderStage::Compute {
        let workgroup_size = request
            .workgroup_size
            .filter(|s| !s.contains(&0))
            .ok_or_else(|| {
                malformed(
                    request.stage,
                    0,
                    0,
                    "SSA compute compilation requires nonzero QMD workgroup dimensions",
                )
            })?;
        read_shader_code(
            &memory,
            request.stage,
            request.address,
            MaxwellShaderMetadata::Compute {
                register_count: request.register_count,
                workgroup_size,
            },
            None,
            0,
        )?
    } else {
        if request.workgroup_size.is_some() {
            return Err(malformed(
                request.stage,
                0,
                0,
                "graphics shader cannot declare a compute workgroup",
            ));
        }
        let binary = read_shader_binary(&memory, request.stage, request.address)?;
        validate_program_header(request.stage, binary.header())?;
        binary
    };
    Ok(binary)
}

struct Emitter {
    builder: FunctionBuilder,
    registers: Vec<VariableId>,
    predicates: [VariableId; 7],
    snapshots: [Option<Operand>; 7],
    resource_kinds: BTreeMap<u32, nixe_gpu::ShaderResourceKind>,
}

fn lower_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
) -> Result<MaxwellSsaProgram, MaxwellShaderTranslationError> {
    let decoded =
        super::translate::decode_shader_binary(binary, register_count, &BTreeMap::new(), false)?;
    MaxwellSsaProgram::new(binary, decoded)
}

fn lower_decoded(
    binary: &MaxwellShaderBinary,
    decoded: super::translate::DecodedProgram,
) -> Result<Function, MaxwellShaderTranslationError> {
    use super::decoded::Operation as D;
    let mut builder = FunctionBuilder::new(decoded.stage);
    let mut count = decoded.temporary_count as usize;
    for i in &decoded.instructions {
        i.operation()
            .visit_destination_registers(|r| count = count.max(r.index() as usize + 1));
        for r in i.operation().source_registers() {
            count = count.max(r.index() as usize + 1);
        }
    }
    let registers = (0..count)
        .map(|_| builder.variable(Type::U32, InitialValue::Undefined))
        .collect();
    let predicates = std::array::from_fn(|_| builder.variable(Type::Bool, InitialValue::Undefined));
    let mut groups: Vec<(ShaderSourceLocation, Vec<super::decoded::Instruction>)> = Vec::new();
    for i in decoded.instructions {
        if groups.last().is_none_or(|(pc, _)| *pc != i.source()) {
            groups.push((i.source(), Vec::new()));
        }
        groups.last_mut().unwrap().1.push(i);
    }
    if groups.is_empty() {
        return Err(malformed(binary.stage(), 0, 0, "empty decoded shader"));
    }
    let mut blocks = BTreeMap::new();
    for (pc, _) in &groups {
        if blocks
            .insert(pc.byte_offset(), builder.block(*pc))
            .is_some()
        {
            return Err(malformed(
                binary.stage(),
                pc.byte_offset(),
                0,
                "noncontiguous decoded guest instruction",
            ));
        }
    }
    let root = builder.entry();
    if binary.stage() == MaxwellShaderStage::VertexCullBeforeFetch {
        // Maxwell's output prologue, also used by Eden/yuzu, initializes
        // homogeneous position to (0,0,0,1). Keep this guest rule in Maxwell.
        // https://github.com/eden-emulator/mirror/blob/67bada77f8a43a90da2e94e89b8e7da73c256989/src/shader_recompiler/backend/glsl/emit_glsl_special.cpp#L16-L22
        for component in 0..4 {
            let value = builder.emit(
                root,
                ShaderSourceLocation::new(0),
                Op::Constant(Constant::u32(if component == 3 {
                    1.0f32.to_bits()
                } else {
                    0
                })),
                vec![],
            )[0];
            builder.emit(
                root,
                ShaderSourceLocation::new(0),
                Op::Output {
                    location: nixe_gpu::ShaderIoLocation::Position,
                    component,
                    ty: Type::U32,
                },
                vec![value.into()],
            );
        }
    }

    builder.terminate(
        root,
        ShaderSourceLocation::new(0),
        Terminator::Jump(blocks[&groups[0].0.byte_offset()]),
    );
    let mut e = Emitter {
        builder,
        registers,
        predicates,
        snapshots: [None; 7],
        resource_kinds: decoded
            .resources
            .iter()
            .map(|r| (r.binding(), r.kind()))
            .collect(),
    };
    for (index, (pc, instructions)) in groups.iter().enumerate() {
        let mut current = blocks[&pc.byte_offset()];
        e.snapshot_predicates(current, *pc, instructions);
        let next = groups
            .get(index + 1)
            .map(|(pc, _)| blocks[&pc.byte_offset()]);
        let mut terminated = false;
        let mut segment = 0;
        while segment < instructions.len() {
            let predicate = instructions[segment].predicate();
            let end = (segment + 1..instructions.len())
                .find(|&i| instructions[i].predicate() != predicate)
                .unwrap_or(instructions.len());
            if terminated {
                return Err(malformed(
                    binary.stage(),
                    pc.byte_offset(),
                    0,
                    "decoded operations follow a terminator",
                ));
            }
            if predicate == ShaderPredicate::Never {
                segment = end;
                continue;
            }
            let conditional = predicate != ShaderPredicate::Always;
            let after = if conditional {
                Some(e.builder.block(*pc))
            } else {
                None
            };
            let execute = if let Some(after) = after {
                let condition = e.predicate(current, *pc, predicate);
                let body = e.builder.block(*pc);
                e.builder.terminate(
                    current,
                    *pc,
                    Terminator::Branch {
                        condition,
                        when_true: body,
                        when_false: after,
                    },
                );
                body
            } else {
                current
            };
            let mut body_terminated = false;
            for instruction in &instructions[segment..end] {
                if body_terminated {
                    return Err(malformed(
                        binary.stage(),
                        pc.byte_offset(),
                        0,
                        "decoded operation after control transfer",
                    ));
                }
                let term = match instruction.operation() {
                    D::Branch { target } => Some(Terminator::Jump(
                        *blocks.get(&target.byte_offset()).ok_or_else(|| {
                            malformed(
                                binary.stage(),
                                pc.byte_offset(),
                                0,
                                "decoded branch target has no block",
                            )
                        })?,
                    )),
                    D::BranchTable { value, targets } => {
                        let default = e.builder.block(*pc);
                        e.builder.terminate(default, *pc, Terminator::Unreachable);
                        Some(Terminator::Switch {
                            selector: e.read(*value),
                            cases: targets
                                .iter()
                                .map(|&(bits, pc)| {
                                    blocks
                                        .get(&pc.byte_offset())
                                        .copied()
                                        .map(|b| (bits, b))
                                        .ok_or_else(|| {
                                            malformed(
                                                binary.stage(),
                                                pc.byte_offset(),
                                                0,
                                                "decoded table target has no block",
                                            )
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .into_boxed_slice(),
                            default,
                        })
                    }
                    D::Exit => Some(Terminator::Return),
                    D::Discard => Some(Terminator::Discard),
                    _ => None,
                };
                if let Some(term) = term {
                    e.builder.terminate(execute, *pc, term);
                    body_terminated = true;
                } else {
                    e.operation(execute, *pc, instruction.operation())?;
                }
            }
            if let Some(after) = after {
                if !body_terminated {
                    e.builder.terminate(execute, *pc, Terminator::Jump(after));
                }
                current = after;
            } else {
                terminated = body_terminated;
            }
            segment = end;
        }
        if !terminated {
            e.builder.terminate(
                current,
                *pc,
                Terminator::Jump(next.ok_or_else(|| {
                    malformed(
                        binary.stage(),
                        pc.byte_offset(),
                        0,
                        "decoded path falls out of shader",
                    )
                })?),
            );
        }
    }
    let mut function = e.builder.finish()?;
    function.inputs = decoded.inputs;
    function.outputs = decoded.outputs;
    if function.stage == nixe_gpu::ShaderStage::Vertex {
        for component in 0..4 {
            if !function.outputs.iter().any(|e| {
                e.location() == nixe_gpu::ShaderIoLocation::Position && e.component() == component
            }) {
                function.outputs.push(
                    nixe_gpu::ShaderInterfaceElement::new(
                        nixe_gpu::ShaderIoLocation::Position,
                        component,
                        nixe_gpu::ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap(),
                );
            }
        }
    }
    function.resources = decoded
        .resources
        .iter()
        .map(|r| nixe_gpu::ssa::Resource {
            id: ResourceId::new(r.binding()),
            kind: r.kind(),
        })
        .collect();
    function.workgroup_size = decoded.workgroup_size;
    function.local_memory_bytes = decoded.local_memory_bytes;
    function.tessellation_control_points = decoded.tessellation_control_points;
    function.optimize()?;
    demand::resolve_range_bits(binary, &mut function)?;
    Ok(function)
}

mod emit;

#[cfg(test)]
mod tests;
