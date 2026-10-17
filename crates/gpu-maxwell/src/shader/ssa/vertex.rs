//! A/B execute as one vertex shader. Return from A enters B; their GPR SSA
//! values stay independent. A's user attributes feed B's matching attribute
//! reads as compiler values, rather than a host buffer or renderer transition.
//! https://git.axenov.dev/Museum/ryujinx/src/commit/8c720783f50051b28188a330f87d3b74bbe26cc7/Ryujinx.Graphics.Shader/Translation/TranslatorContext.cs
//! Eden removes A's epilogue and B's prologue when combining both programs:
//! https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/ir_opt/dual_vertex_pass.cpp
use super::*;
use nixe_gpu::{ShaderInterfaceElement, ShaderIoLocation};

pub(super) fn merge(a: &Function, z: &Function) -> Result<Function, MaxwellShaderTranslationError> {
    let mut b = FunctionBuilder::new(nixe_gpu::ShaderStage::Vertex);
    let source = ShaderSourceLocation::new(0);
    let attributes: BTreeMap<_, _> = a
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i.op {
            Op::Output {
                location: location @ ShaderIoLocation::Generic(_),
                component,
                ..
            } => Some((
                (location, component),
                b.variable(Type::U32, InitialValue::Undefined),
            )),
            _ => None,
        })
        .collect();
    let variables: Vec<Vec<_>> = [a, z]
        .iter()
        .map(|f| {
            f.values
                .iter()
                .map(|v| {
                    if v.ty == Type::Effect {
                        None
                    } else {
                        Some(b.variable(v.ty, InitialValue::Undefined))
                    }
                })
                .collect()
        })
        .collect();
    let blocks: Vec<Vec<_>> = [a, z]
        .iter()
        .map(|f| f.blocks.iter().map(|block| b.block(block.source)).collect())
        .collect();
    let root = b.entry();
    b.terminate(root, source, Terminator::Jump(blocks[0][a.entry().index()]));
    for (stage, f) in [a, z].into_iter().enumerate() {
        let names = &variables[stage];
        let labels = &blocks[stage];
        for (index, block) in f.blocks.iter().enumerate() {
            let label = labels[index];
            for inst in &block.instructions {
                if inst.op == Op::EffectStart {
                    continue;
                }
                // Output initialization occurs once, before A. Guest stores
                // have nonzero PCs, so this removes only B's frontend prologue.
                if stage == 1
                    && inst.source.byte_offset() == 0
                    && matches!(
                        inst.op,
                        Op::Output {
                            location: ShaderIoLocation::Position,
                            ..
                        }
                    )
                {
                    continue;
                }
                let args: Vec<_> = inst
                    .arguments
                    .iter()
                    .filter_map(|v| names[v.index()].map(Operand::from))
                    .collect();
                let values = match inst.op {
                    Op::Output {
                        location,
                        component,
                        ty,
                    } if stage == 0 && attributes.contains_key(&(location, component)) => {
                        let bits = if ty == Type::U32 {
                            args[0]
                        } else {
                            b.emit(
                                label,
                                inst.source,
                                Op::Bitcast {
                                    from: ty,
                                    to: Type::U32,
                                },
                                args.clone(),
                            )[0]
                            .into()
                        };
                        b.assign(label, inst.source, attributes[&(location, component)], bits);
                        // Preserve the attribute as a final output too: B can
                        // consume or overwrite it, and unwritten B components
                        // retain A's export rather than fabricated zero stores.
                        b.emit(label, inst.source, inst.op.clone(), args)
                    }
                    Op::Input {
                        location,
                        component,
                        ty,
                    } if stage == 1 && attributes.contains_key(&(location, component)) => {
                        let input = attributes[&(location, component)].into();
                        b.emit(
                            label,
                            inst.source,
                            if ty == Type::U32 {
                                Op::Copy(Type::U32)
                            } else {
                                Op::Bitcast {
                                    from: Type::U32,
                                    to: ty,
                                }
                            },
                            vec![input],
                        )
                    }
                    _ => b.emit(label, inst.source, inst.op.clone(), args),
                };
                for (&old, &new) in inst
                    .results
                    .iter()
                    .filter(|v| names[v.index()].is_some())
                    .zip(values.iter())
                {
                    b.assign(label, inst.source, names[old.index()].unwrap(), new.into());
                }
            }
            let mut edges = BTreeMap::new();
            for target in block.terminator.successors() {
                let edge = b.block(block.terminator_source);
                let mut assigned = Vec::new();
                for phi in &f.blocks[target.index()].phis {
                    let Some(variable) = names[phi.result.index()] else {
                        continue;
                    };
                    let input = phi
                        .incoming
                        .iter()
                        .find(|(p, _)| p.index() == index)
                        .expect("verified vertex phi")
                        .1;
                    let ty = f.values[input.index()].ty;
                    let value = b.emit(
                        edge,
                        phi.source,
                        Op::Copy(ty),
                        vec![names[input.index()].unwrap().into()],
                    )[0];
                    assigned.push((variable, value));
                }
                for (variable, value) in assigned {
                    b.assign(edge, block.terminator_source, variable, value.into());
                }
                b.terminate(
                    edge,
                    block.terminator_source,
                    Terminator::Jump(labels[target.index()]),
                );
                edges.insert(target, edge);
            }
            let term = match &block.terminator {
                Terminator::Jump(target) => Terminator::Jump(edges[target]),
                Terminator::Branch {
                    condition,
                    when_true,
                    when_false,
                } => Terminator::Branch {
                    condition: names[condition.index()].unwrap().into(),
                    when_true: edges[when_true],
                    when_false: edges[when_false],
                },
                Terminator::Switch {
                    selector,
                    cases,
                    default,
                } => Terminator::Switch {
                    selector: names[selector.index()].unwrap().into(),
                    cases: cases
                        .iter()
                        .map(|&(literal, target)| (literal, edges[&target]))
                        .collect(),
                    default: edges[default],
                },
                Terminator::Return if stage == 0 => Terminator::Jump(blocks[1][z.entry().index()]),
                Terminator::Return => Terminator::Return,
                Terminator::Discard => Terminator::Discard,
                Terminator::Unreachable => Terminator::Unreachable,
            };
            b.terminate(label, block.terminator_source, term);
        }
    }
    let mut result = b.finish()?;
    result.local_memory_bytes = a.local_memory_bytes.max(z.local_memory_bytes);
    let mut inputs = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    let mut resources = BTreeMap::new();
    for (stage, f) in [a, z].into_iter().enumerate() {
        for e in &f.inputs {
            if stage != 1 || !attributes.contains_key(&(e.location(), u32::from(e.component()))) {
                inputs.insert((e.location(), e.component()), *e);
            }
        }
        for e in &f.outputs {
            outputs.insert((e.location(), e.component()), *e);
        }
        for r in &f.resources {
            resources.insert(r.id, *r);
        }
    }
    result.inputs = inputs
        .values()
        .copied()
        .collect::<Vec<ShaderInterfaceElement>>();
    result.outputs = outputs.values().copied().collect();
    result.resources = resources.values().copied().collect();
    result.optimize()?;
    Ok(result)
}
