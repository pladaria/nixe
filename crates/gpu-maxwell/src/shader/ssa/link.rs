//! Maxwell interface specialization before native host shader generation.
use super::*;
use nixe_gpu::{ShaderInterfaceElement, ShaderIoLocation, ShaderScalarType, ShaderStage};
use std::collections::BTreeSet;

pub(super) fn link(
    programs: &mut Vec<MaxwellSsaProgram>,
) -> Result<(), MaxwellShaderTranslationError> {
    let vertex_a = programs
        .iter()
        .position(|p| p.stage == MaxwellShaderStage::VertexCullBeforeFetch);
    let vertex_b = programs
        .iter()
        .position(|p| p.stage == MaxwellShaderStage::Vertex);
    if let (Some(a), Some(z)) = (vertex_a, vertex_b) {
        let first = programs.remove(a);
        let z = if z > a { z - 1 } else { z };
        // Canonical Maxwell reads are bounded to 64 KiB; give B a disjoint PC
        // namespace so host feature/CFG diagnostics can identify A or B.
        let base = super::super::binary::MAXWELL_SHADER_READ_LIMIT as u32;
        let shift = |s: ShaderSourceLocation| ShaderSourceLocation::new(s.byte_offset() + base);
        for block in &mut programs[z].function.blocks {
            for phi in &mut block.phis {
                phi.source = shift(phi.source);
            }
            for inst in &mut block.instructions {
                if inst.source.byte_offset() == 0
                    && matches!(
                        inst.op,
                        Op::Output {
                            location: ShaderIoLocation::Position,
                            ..
                        }
                    )
                {
                    // The merger identifies B's frontend prologue at zero.
                    continue;
                }
                inst.source = shift(inst.source);
            }
            block.source = shift(block.source);
            block.terminator_source = shift(block.terminator_source);
        }
        programs[z].function = super::vertex::merge(&first.function, &programs[z].function)?;
        programs[z].sources = first
            .sources
            .iter()
            .copied()
            .chain(programs[z].sources.iter().map(|s| MaxwellSsaSource {
                pc_base: s.pc_base + base,
                ..*s
            }))
            .collect();
        programs[z].bindings = first
            .bindings
            .iter()
            .chain(programs[z].bindings.iter())
            .copied()
            .collect();
        programs[z].append_buffers(&first);
        programs[z].retain_live_bindings();
    }
    let fragment = programs
        .iter()
        .position(|p| p.function.stage == ShaderStage::Fragment);
    let producer = [
        ShaderStage::Geometry,
        ShaderStage::TessellationEvaluation,
        ShaderStage::Vertex,
    ]
    .into_iter()
    .find_map(|stage| programs.iter().position(|p| p.function.stage == stage));
    if let (Some(fragment), Some(producer)) = (fragment, producer) {
        let outputs: BTreeSet<_> = programs[producer]
            .function
            .outputs
            .iter()
            .map(|e| (e.location(), e.component()))
            .collect();
        let missing: BTreeSet<_> = programs[fragment]
            .function
            .inputs
            .iter()
            .filter(|e| {
                matches!(e.location(), ShaderIoLocation::Generic(_))
                    && e.scalar_type() == ShaderScalarType::Float32
                    && !outputs.contains(&(e.location(), e.component()))
            })
            .map(|e| (e.location(), e.component()))
            .collect();
        // Disabled generic attributes are Maxwell (0,0,0,1), per component.
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp
        for block in &mut programs[fragment].function.blocks {
            for inst in &mut block.instructions {
                let (location, component, ty) = match inst.op {
                    Op::Input {
                        location,
                        component,
                        ty,
                    } => (location, component, ty),
                    Op::Interpolate {
                        location,
                        component,
                        ..
                    } => (location, component, Type::F32),
                    _ => continue,
                };
                if missing.contains(&(location, component as u8)) {
                    inst.op = Op::Constant(Constant {
                        ty,
                        bits: if component == 3 {
                            1.0f32.to_bits() as u64
                        } else {
                            0
                        },
                    });
                    inst.arguments = Box::new([]);
                }
            }
        }
        programs[fragment]
            .function
            .inputs
            .retain(|e| !missing.contains(&(e.location(), e.component())));
        let interpolation: BTreeMap<_, _> = programs[fragment]
            .function
            .inputs
            .iter()
            .map(|e| ((e.location(), e.component()), e.interpolation()))
            .collect();
        let function = &mut programs[producer].function;
        for output in &mut function.outputs {
            if let Some(&i) = interpolation.get(&(output.location(), output.component())) {
                *output = ShaderInterfaceElement::new(
                    output.location(),
                    output.component(),
                    output.scalar_type(),
                    i,
                )
                .unwrap();
            }
        }
        // Output reads are TCS operations. Only prune the final producer's
        // unused generic stores; all raster builtins remain observable.
        let mut aliases = BTreeMap::new();
        for block in &mut function.blocks {
            block.instructions.retain(|inst| {
                if let Op::Output {
                    location: location @ ShaderIoLocation::Generic(_),
                    component,
                    ..
                } = inst.op
                    && !interpolation.contains_key(&(location, component as u8))
                {
                    aliases.insert(
                        *inst.results.last().unwrap(),
                        *inst.arguments.last().unwrap(),
                    );
                    false
                } else {
                    true
                }
            });
        }
        let resolve = |mut value: nixe_gpu::ssa::ValueId| {
            while let Some(&next) = aliases.get(&value) {
                value = next;
            }
            value
        };
        for block in &mut function.blocks {
            for phi in &mut block.phis {
                for (_, value) in &mut phi.incoming {
                    *value = resolve(*value);
                }
            }
            for inst in &mut block.instructions {
                for value in &mut inst.arguments {
                    *value = resolve(*value);
                }
            }
            block.effect = resolve(block.effect);
        }
        function.outputs.retain(|e| {
            !matches!(e.location(), ShaderIoLocation::Generic(_))
                || interpolation.contains_key(&(e.location(), e.component()))
        });
        // Removed effects leave orphan definition metadata until compaction.
        // Use the normal SSA optimizer only after refreshing dense definitions.
        // The compiler-facing compactor is shared with CFG/legalization passes.
        function.recompact();
    }
    for program in programs.iter_mut() {
        program.function.optimize()?;
        program.retain_live_bindings();
    }
    let ordered: Vec<_> = [
        ShaderStage::Vertex,
        ShaderStage::TessellationControl,
        ShaderStage::TessellationEvaluation,
        ShaderStage::Geometry,
        ShaderStage::Fragment,
    ]
    .into_iter()
    .filter_map(|stage| programs.iter().find(|p| p.function.stage == stage))
    .collect();
    for pair in ordered.windows(2) {
        let previous = &pair[0].function;
        let next = &pair[1].function;
        for input in &next.inputs {
            if !matches!(
                input.location(),
                ShaderIoLocation::Generic(_)
                    | ShaderIoLocation::Patch(_)
                    | ShaderIoLocation::Position
                    | ShaderIoLocation::PointSize
            ) {
                continue;
            }
            if input.location() == ShaderIoLocation::Position && next.stage == ShaderStage::Fragment
            {
                continue;
            }
            let output = previous
                .outputs
                .iter()
                .find(|e| e.location() == input.location() && e.component() == input.component());
            if !output.is_some_and(|e| e.scalar_type() == input.scalar_type()) {
                return Err(MaxwellShaderTranslationError::StageInterfaceMismatch {
                    producer: previous.stage,
                    consumer: next.stage,
                    location: input.location(),
                    component: input.component(),
                    reason: "native stage input has no matching producer scalar",
                });
            }
        }
    }
    Ok(())
}
