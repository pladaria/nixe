use super::*;

pub(super) fn verify(function: &Function) -> Result<(), Error> {
    let dom = Dominance::compute(function)?;
    if dom.reverse_postorder.len() != function.blocks.len() {
        return Err(Error::at(
            ShaderSourceLocation::new(0),
            "unreachable block retained in SSA",
        ));
    }
    let mut definitions = vec![false; function.values.len()];
    let value = |id: ValueId, source| {
        function
            .values
            .get(id.index())
            .ok_or_else(|| Error::at(source, format!("missing v{}", id.0)))
    };
    let mut effect_start = None;
    for (index, block) in function.blocks.iter().enumerate() {
        let id = BlockId(index as u32);
        let mut define = |result: ValueId, expected: Definition, source| -> Result<(), Error> {
            if value(result, source)?.definition != expected {
                return Err(Error::at(
                    source,
                    "value definition metadata disagrees with instruction/phi",
                ));
            }
            if std::mem::replace(&mut definitions[result.index()], true) {
                return Err(Error::at(source, "value defined twice"));
            }
            Ok(())
        };
        let mut effect_phi = None;
        for phi in &block.phis {
            define(phi.result, Definition::Phi { block: id }, phi.source)?;
            let ty = value(phi.result, phi.source)?.ty;
            if ty == Type::Effect && effect_phi.replace(phi.result).is_some() {
                return Err(Error::at(phi.source, "multiple effect phis in one block"));
            }
            let mut incoming: Vec<_> = phi.incoming.iter().map(|(b, _)| *b).collect();
            incoming.sort_unstable();
            if incoming != dom.predecessors[index] {
                return Err(Error::at(
                    phi.source,
                    "phi inputs do not match CFG predecessors exactly",
                ));
            }
            for &(predecessor, input) in &phi.incoming {
                if value(input, phi.source)?.ty != ty {
                    return Err(Error::at(phi.source, "phi input type mismatch"));
                }
                check_use(function, &dom, input, predecessor, u32::MAX, phi.source)?;
                if ty == Type::Effect && input != function.blocks[predecessor.index()].effect {
                    return Err(Error::at(
                        phi.source,
                        "effect phi bypasses predecessor effects",
                    ));
                }
            }
        }
        let mut current_effect = effect_phi;
        if id != function.entry() && current_effect.is_none() {
            for &predecessor in &dom.predecessors[index] {
                let effect = function.blocks[predecessor.index()].effect;
                if current_effect.is_some_and(|old| old != effect) {
                    return Err(Error::at(
                        block.source,
                        "distinct incoming effects need an effect phi",
                    ));
                }
                current_effect = Some(effect);
            }
        }
        for (instruction_index, instruction) in block.instructions.iter().enumerate() {
            valid_op(&instruction.op).map_err(|detail| Error::at(instruction.source, detail))?;
            let (args, results) = instruction.op.signature();
            if args.len() != instruction.arguments.len()
                || results.len() != instruction.results.len()
            {
                return Err(Error::at(instruction.source, "opcode arity mismatch"));
            }
            for (&argument, ty) in instruction.arguments.iter().zip(args) {
                if value(argument, instruction.source)?.ty != ty {
                    return Err(Error::at(
                        instruction.source,
                        "opcode operand type mismatch",
                    ));
                }
                check_use(
                    function,
                    &dom,
                    argument,
                    id,
                    instruction_index as u32,
                    instruction.source,
                )?;
            }
            for (result_index, (&result, ty)) in instruction.results.iter().zip(results).enumerate()
            {
                define(
                    result,
                    Definition::Instruction {
                        block: id,
                        index: instruction_index as u32,
                        result: result_index as u32,
                    },
                    instruction.source,
                )?;
                if value(result, instruction.source)?.ty != ty {
                    return Err(Error::at(instruction.source, "opcode result type mismatch"));
                }
            }
            if instruction.op == Op::EffectStart {
                if id != function.entry() || effect_start.replace(instruction.results[0]).is_some()
                {
                    return Err(Error::at(
                        instruction.source,
                        "effect start must be unique and in the entry",
                    ));
                }
                current_effect = Some(instruction.results[0]);
            }
            if instruction.op.ordered() {
                if instruction.arguments.last().copied() != current_effect {
                    return Err(Error::at(
                        instruction.source,
                        "ordered operation bypasses or forks the effect chain",
                    ));
                }
                current_effect = instruction.results.last().copied();
            }
        }
        if current_effect != Some(block.effect)
            || value(block.effect, block.terminator_source)?.ty != Type::Effect
        {
            return Err(Error::at(
                block.terminator_source,
                "terminator does not retain the last effect",
            ));
        }
        check_use(
            function,
            &dom,
            block.effect,
            id,
            u32::MAX,
            block.terminator_source,
        )?;
        match &block.terminator {
            Terminator::Branch { condition, .. } => {
                if value(*condition, block.terminator_source)?.ty != Type::Bool {
                    return Err(Error::at(
                        block.terminator_source,
                        "branch condition must be boolean",
                    ));
                }
                check_use(
                    function,
                    &dom,
                    *condition,
                    id,
                    u32::MAX,
                    block.terminator_source,
                )?;
            }
            Terminator::Switch {
                selector, cases, ..
            } => {
                if value(*selector, block.terminator_source)?.ty != Type::U32 {
                    return Err(Error::at(
                        block.terminator_source,
                        "switch selector must be u32",
                    ));
                }
                check_use(
                    function,
                    &dom,
                    *selector,
                    id,
                    u32::MAX,
                    block.terminator_source,
                )?;
                let mut labels: Vec<_> = cases.iter().map(|(label, _)| *label).collect();
                labels.sort_unstable();
                if labels.windows(2).any(|p| p[0] == p[1]) {
                    return Err(Error::at(block.terminator_source, "duplicate switch case"));
                }
            }
            Terminator::Discard if function.stage != ShaderStage::Fragment => {
                return Err(Error::at(
                    block.terminator_source,
                    "discard outside fragment stage",
                ));
            }
            _ => {}
        }
    }
    if effect_start.is_none() || definitions.contains(&false) {
        return Err(Error::at(
            ShaderSourceLocation::new(0),
            "missing effect start or orphan value",
        ));
    }
    Ok(())
}

fn check_use(
    function: &Function,
    dom: &Dominance,
    id: ValueId,
    block: BlockId,
    index: u32,
    source: ShaderSourceLocation,
) -> Result<(), Error> {
    let definition = function
        .values
        .get(id.index())
        .ok_or_else(|| Error::at(source, "missing operand value"))?
        .definition;
    let (def_block, def_index) = match definition {
        Definition::Phi { block } => (block, None),
        Definition::Instruction { block, index, .. } => (block, Some(index)),
    };
    if def_block.index() >= function.blocks.len()
        || !dom.dominates(def_block, block)
        || (def_block == block && def_index.is_some_and(|i| i >= index))
    {
        return Err(Error::at(
            source,
            format!("v{} does not dominate its use in b{}", id.0, block.0),
        ));
    }
    Ok(())
}

fn valid_op(op: &Op) -> Result<(), &'static str> {
    use Op::*;
    let scalar = |ty| ty != Type::Effect;
    let valid_component = |location: ShaderIoLocation, component: u32| {
        component
            < match location {
                ShaderIoLocation::Position
                | ShaderIoLocation::Generic(_)
                | ShaderIoLocation::Color(_)
                | ShaderIoLocation::Patch(_)
                | ShaderIoLocation::TessLevelOuter => 4,
                ShaderIoLocation::TessCoord => 3,
                ShaderIoLocation::TessLevelInner => 2,
                _ => 1,
            }
    };
    match *op {
        Input {
            location,
            component,
            ..
        }
        | Output {
            location,
            component,
            ..
        }
        | Interpolate {
            location,
            component,
            ..
        }
        | ControlPointLoad {
            location,
            component,
            ..
        }
        | ControlPointStore {
            location,
            component,
        }
        | PatchOutputLoad {
            location,
            component,
        } if !valid_component(location, component) => {
            Err("interface component exceeds its semantic location width")
        }
        Unary {
            op: super::Unary::BitReverse,
            ty,
        } if !ty.integer() => Err("bit reversal requires integer"),
        FloatToInteger { to, bits, .. }
            if !matches!(to, Type::U32 | Type::I32) || !matches!(bits, 8 | 16 | 32) =>
        {
            Err("integer conversion requires 8/16/32-bit precision in a 32-bit scalar")
        }
        Quad(ShaderQuadOperation::Broadcast(lane)) if lane >= 4 => {
            Err("quad broadcast index must be 0..3")
        }
        QuadSwizzleAdd { swizzle, .. } if swizzle > 255 => {
            Err("quad swizzle must contain four two-bit selectors")
        }
        TextureSample {
            kind,
            mode,
            sampler,
            ..
        } if !matches!(
            kind,
            ShaderResourceKind::SampledImage
                | ShaderResourceKind::SampledImage2DArray
                | ShaderResourceKind::SampledDepthImage
                | ShaderResourceKind::SampledDepthImage2DArray
                | ShaderResourceKind::SampledImageCube
        ) || matches!(mode, SampleMode::Fetch(_)) != sampler.is_none() =>
        {
            Err("sampled image operation has invalid image/sampler classes")
        }
        TextureSample {
            kind,
            mode: SampleMode::Compare | SampleMode::CompareLevelZero,
            ..
        } if !matches!(
            kind,
            ShaderResourceKind::SampledDepthImage | ShaderResourceKind::SampledDepthImage2DArray
        ) =>
        {
            Err("comparison sampling requires a depth image")
        }
        Constant(c)
            if c.ty == Type::Effect
                || c.ty
                    .bits()
                    .is_none_or(|width| width < 64 && c.bits >> width != 0) =>
        {
            Err("invalid constant bits/type")
        }
        Undefined(ty) | Parameter { ty, .. } | Copy(ty) | Select(ty) if !scalar(ty) => {
            Err("effect tokens cannot be scalar data")
        }
        Bitcast { from, to } if from.bits() != to.bits() || !scalar(from) || !scalar(to) => {
            Err("bitcast width/type mismatch")
        }
        Convert { from, to, .. }
            if !scalar(from) || !scalar(to) || from == Type::Bool || to == Type::Bool =>
        {
            Err("invalid numeric conversion")
        }
        Unary {
            op: super::Unary::Not,
            ty,
        } if !ty.integer() && ty != Type::Bool => Err("not requires integer/boolean"),
        Unary {
            op: super::Unary::Negate | super::Unary::Absolute,
            ty,
        } if !scalar(ty) || ty == Type::Bool => Err("invalid unary numeric type"),
        Binary {
            op: super::Binary::And | super::Binary::Or | super::Binary::Xor,
            ty,
            ..
        } if !ty.integer() && ty != Type::Bool => Err("bitwise operation requires integer/boolean"),
        Binary {
            op: super::Binary::ShiftLeft | super::Binary::ShiftRight,
            ty,
            ..
        } if !ty.integer() => Err("shift requires integer"),
        Binary {
            op: super::Binary::Add | super::Binary::Multiply,
            ty,
            ..
        } if !scalar(ty) || ty == Type::Bool => Err("invalid binary numeric type"),
        IntegerCompare { ty, .. } if !ty.integer() => Err("integer comparison requires integer"),
        FloatCompare { ty, .. } | Fma { ty, .. } if !matches!(ty, Type::F32 | Type::F64) => {
            Err("floating operation requires float")
        }
        Input { ty, .. } | Output { ty, .. } | Load { ty, .. } | Store { ty, .. }
            if !scalar(ty) =>
        {
            Err("effect cannot be an interface/memory element")
        }
        Load {
            space: MemorySpace::Constant | MemorySpace::Storage,
            resource: None,
            ..
        }
        | Store {
            space: MemorySpace::Storage,
            resource: None,
            ..
        } => Err("resource-backed memory operation has no resource"),
        Load {
            space: MemorySpace::Local | MemorySpace::Shared,
            resource: Some(_),
            ..
        }
        | Store {
            space: MemorySpace::Local | MemorySpace::Shared,
            resource: Some(_),
            ..
        } => Err("invocation/workgroup memory cannot name a descriptor"),
        Load { address_type, .. } | Store { address_type, .. }
            if !matches!(address_type, Type::U32 | Type::U64) =>
        {
            Err("memory address must be u32/u64")
        }
        Store {
            space: MemorySpace::Constant,
            ..
        } => Err("constant memory cannot be written"),
        _ => Ok(()),
    }
}
