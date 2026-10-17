//! Native SPIR-V emission from dense semantic SSA. Effect values constrain
//! emission order and disappear from the executable; no guest register array,
//! program-counter dispatcher, old-IR bridge or shader-language round trip.
use super::*;
use crate::shader::spirv::{InterfaceLayout, Math, SpirvShaderOptions};
use rspirv::{binary::Assemble, dr::Operand as SpvOperand, spirv as spv};
use std::collections::{BTreeMap, HashMap, HashSet};

mod canonicalize;
mod control;
mod depth;
mod float_control;
mod float_facts;
mod half;
mod legalize;
mod numeric;
mod operations;
mod quad;
mod quad_geometry;
pub use quad_geometry::emit_quad_geometry;
mod texture;
mod unary;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Options {
    pub shader: SpirvShaderOptions,
    /// Reconstruct window Z after rasterization when the viewport was normalized
    /// to [0,1]. Explicit guest depth replaces it and is only clamped.
    pub fragment_depth_transform: Option<crate::FragmentDepthTransform>,
    pub integer64: bool,
    pub subgroup_quad: bool,
    pub shared_memory_bytes: u32,
    /// Uniform ABI extent selected from the enabled host descriptor range.
    pub uniform_buffer_bytes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Binding {
    pub resource: ResourceId,
    pub binding: u32,
    pub kind: crate::ShaderResourceKind,
    pub readable: bool,
    pub writable: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Module {
    pub words: Box<[u32]>,
    pub bindings: Box<[Binding]>,
    pub push_constant_bytes: u32,
}

/// Invoke only on a shader-cache miss. Host guarantees are explicit enabled
/// features; vendor names and guest register counts never select an emission.
pub fn emit(function: &Function, options: Options) -> Result<Module, Error> {
    emit_with_bindings(function, options, None).map_err(|e| e.in_stage(function.stage))
}
/// Link a shared descriptor ABI on a pipeline cache miss. Logical resource
/// identities must already be namespaced by the selected guest frontend.
/// Every stage uses the same dense binding for a shared logical resource.
pub fn emit_pipeline(stages: &[(&Function, Options)]) -> Result<Box<[Module]>, Error> {
    let functions: Vec<_> = stages.iter().map(|&(function, _)| function).collect();
    let bindings = pipeline_bindings(&functions);
    let mut modules = Vec::with_capacity(stages.len());
    let mut kinds = BTreeMap::new();
    for &(function, options) in stages {
        let module = emit_with_bindings(function, options, Some(&bindings))
            .map_err(|e| e.in_stage(function.stage))?;
        for binding in &module.bindings {
            if let Some(old) = kinds.insert(binding.resource, binding.kind)
                && old != binding.kind
            {
                return Err(Error::at(
                    ShaderSourceLocation::new(0),
                    "pipeline resource has inconsistent stage descriptor classes",
                ));
            }
        }
        modules.push(module);
    }
    Ok(modules.into_boxed_slice())
}
/// Construct the dense shared descriptor ABI before emitting any stage.
pub fn pipeline_bindings(stages: &[&Function]) -> BTreeMap<ResourceId, u32> {
    let mut bindings = BTreeMap::new();
    for &function in stages {
        for block in &function.blocks {
            for inst in &block.instructions {
                match inst.op {
                    Op::Load {
                        resource: Some(id), ..
                    }
                    | Op::Store {
                        resource: Some(id), ..
                    }
                    | Op::TextureLevels { image: id } => {
                        bindings.insert(id, 0);
                    }
                    Op::TextureSample { image, sampler, .. } => {
                        bindings.insert(image, 0);
                        if let Some(sampler) = sampler {
                            bindings.insert(sampler, 0);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    for (index, binding) in bindings.values_mut().enumerate() {
        *binding = index as u32;
    }
    bindings
}
/// Emit one stage using its already selected linked descriptor ABI. This can
/// run as a leaf of a stage cache; it never queues work on another compiler.
pub fn emit_linked(
    function: &Function,
    options: Options,
    bindings: &BTreeMap<ResourceId, u32>,
) -> Result<Module, Error> {
    emit_with_bindings(function, options, Some(bindings)).map_err(|e| e.in_stage(function.stage))
}

fn emit_with_bindings(
    function: &Function,
    options: Options,
    bindings: Option<&BTreeMap<ResourceId, u32>>,
) -> Result<Module, Error> {
    #[cfg(debug_assertions)]
    function.verify()?;
    let legalized = legalize::legalize(function)?;
    let mut e = Emitter::new(legalized.as_ref().unwrap_or(function), options, bindings);
    e.emit()
}

struct Buffer {
    variable: u32,
    pointer: u32,
    space: MemorySpace,
}
struct Emitter<'a> {
    function: &'a Function,
    options: Options,
    math: Math,
    io: InterfaceLayout,
    values: Vec<u32>,
    float_facts: float_facts::FloatFacts,
    labels: Vec<u32>,
    ends: Vec<u32>,
    buffers: BTreeMap<ResourceId, Buffer>,
    descriptors: BTreeMap<ResourceId, texture::Descriptor>,
    builtins: HashMap<ShaderComputeBuiltin, u32>,
    bindings: Vec<Binding>,
    binding_map: Option<&'a BTreeMap<ResourceId, u32>>,
    aliases: HashMap<u32, u32>,
    shared: Option<u32>,
    local: Option<u32>,
    quad_lane: Option<u32>,
    parameters: Option<u32>,
    push_constant_bytes: u32,
    backedges: HashMap<(BlockId, BlockId), u32>,
    current: BlockId,
}

impl<'a> Emitter<'a> {
    fn new(
        function: &'a Function,
        options: Options,
        binding_map: Option<&'a BTreeMap<ResourceId, u32>>,
    ) -> Self {
        let mut math = Math::new(options.shader);
        float_control::configure(&mut math, function);
        if options.fragment_depth_transform.is_some() {
            // Viewport reconstruction has RTE semantics even if all guest
            // arithmetic happened to permit an RTZ-only execution environment.
            math.rounding32 = ShaderRoundingMode::NearestEven;
        }
        // The backend baseline is Vulkan 1.2+KHR/1.3; SPIR-V 1.5 also permits
        // precise entry-point declaration of descriptor/workgroup variables.
        math.b.set_version(1, 5);
        let values = function
            .values
            .iter()
            .map(|v| if v.ty == Type::Effect { 0 } else { math.b.id() })
            .collect();
        let labels: Vec<_> = function.blocks.iter().map(|_| math.b.id()).collect();
        Self {
            function,
            options,
            math,
            io: InterfaceLayout::default(),
            values,
            float_facts: float_facts::FloatFacts::analyze(function),
            ends: labels.clone(),
            labels,
            buffers: BTreeMap::new(),
            descriptors: BTreeMap::new(),
            builtins: HashMap::new(),
            bindings: Vec::new(),
            binding_map,
            aliases: HashMap::new(),
            shared: None,
            local: None,
            quad_lane: None,
            parameters: None,
            push_constant_bytes: 0,
            backedges: HashMap::new(),
            current: BlockId(0),
        }
    }
    fn binding(&self, resource: ResourceId) -> u32 {
        self.binding_map
            .map_or(self.bindings.len() as u32, |m| m[&resource])
    }
    fn error(&self, detail: impl Into<String>) -> Error {
        Error::at(self.math.source, detail)
    }
    fn ty(&mut self, ty: Type) -> Result<u32, Error> {
        Ok(match ty {
            Type::U32 => self.math.uint,
            Type::I32 => self.math.int,
            Type::F32 => self.math.float,
            Type::F16 => self.half_type()?,
            Type::Bool => self.math.boolean,
            Type::U64 | Type::I64 => {
                if !self.options.integer64 {
                    return Err(
                        self.error("64-bit integer arithmetic requires enabled shaderInt64")
                    );
                }
                self.math.b.capability(spv::Capability::Int64);
                self.math.b.type_int(64, u32::from(ty == Type::I64))
            }
            Type::F64 => {
                if !self.options.shader.float64.enabled {
                    return Err(self.error("float64 requires enabled shaderFloat64"));
                }
                self.math.b.capability(spv::Capability::Float64);
                self.math.b.type_float(64, None)
            }
            Type::Effect => {
                return Err(self.error("effect token cannot become executable scalar data"));
            }
        })
    }
    fn id(&self, value: ValueId) -> u32 {
        self.values[value.index()]
    }
    fn result(&mut self, value: ValueId, native: u32) {
        let reserved = self.id(value);
        if reserved != native {
            self.aliases.insert(reserved, native);
        }
    }
    fn emit(&mut self) -> Result<Module, Error> {
        if self.options.fragment_depth_transform.is_some()
            && self.function.stage != ShaderStage::Fragment
        {
            return Err(self.error("window depth reconstruction requires a fragment shader"));
        }
        if (self.options.shader.vertex_position_transform.is_some()
            || self.options.shader.depth_clip_negative_one_to_one)
            && !matches!(
                self.function.stage,
                ShaderStage::Vertex | ShaderStage::TessellationEvaluation
            )
        {
            return Err(self.error("final position conversion requires the final pre-raster stage"));
        }
        let (inputs, outputs) = self.depth_interfaces();
        self.io
            .emit_interfaces(
                &mut self.math,
                self.function.stage,
                &inputs,
                &outputs,
                self.function.tessellation_control_points,
            )
            .map_err(|e| Error::at(self.math.source, e.to_string()))?;
        self.resources()?;
        let void = self.math.b.type_void();
        let function_type = self.math.b.type_function(void, []);
        self.math.entry = self
            .math
            .b
            .begin_function(void, None, spv::FunctionControl::NONE, function_type)
            .map_err(|e| self.error(e.to_string()))?;
        let structure = control::Structure::analyze(self.function)?;
        let continuations: BTreeMap<_, _> = structure
            .loops
            .keys()
            .map(|&h| (h, self.math.b.id()))
            .collect();
        for (&header, loop_info) in &structure.loops {
            for &tail in &loop_info.tails {
                self.backedges
                    .insert((tail, header), continuations[&header]);
            }
        }
        let mut continuation_phis = BTreeMap::<BlockId, Vec<(u32, u32, Vec<(u32, u32)>)>>::new();
        let mut ghost_merges = Vec::new();
        let mut patched_phis = HashSet::new();
        for &block_id in &structure.order {
            self.current = block_id;
            let block = &self.function.blocks[block_id.index()];
            self.math.block = self
                .math
                .b
                .begin_block(Some(self.labels[block_id.index()]))
                .map_err(|e| self.error(e.to_string()))?;
            for phi in &block.phis {
                if self.function.values[phi.result.index()].ty == Type::Effect {
                    continue;
                }
                let ty = self.ty(self.function.values[phi.result.index()].ty)?;
                let mut incoming = Vec::new();
                let mut back = Vec::new();
                for &(from, value) in &phi.incoming {
                    let pair = (self.id(value), self.labels[from.index()]);
                    if self.backedges.contains_key(&(from, block_id)) {
                        back.push(pair);
                    } else {
                        incoming.push(pair);
                    }
                }
                if !back.is_empty() {
                    let result = self.math.b.id();
                    incoming.push((result, continuations[&block_id]));
                    continuation_phis
                        .entry(block_id)
                        .or_default()
                        .push((ty, result, back));
                    patched_phis.insert(result);
                }
                self.math
                    .b
                    .phi(ty, Some(self.id(phi.result)), incoming)
                    .map_err(|e| self.error(e.to_string()))?;
                patched_phis.insert(self.id(phi.result));
            }
            if let Some(loop_info) = structure.loops.get(&block_id) {
                let merge = if let Some(merge) = loop_info.merge {
                    self.labels[merge.index()]
                } else {
                    let label = self.math.b.id();
                    ghost_merges.push(label);
                    label
                };
                self.math
                    .b
                    .loop_merge(merge, continuations[&block_id], spv::LoopControl::NONE, [])
                    .map_err(|e| self.error(e.to_string()))?;
            }
            if structure.loops.contains_key(&block_id) {
                // Helpers may introduce selections. Keep the loop's header and
                // backedge destination ahead of them; evaluate its guest body
                // and condition in a separate ordinary construct.
                let body = self.math.b.id();
                self.math
                    .b
                    .branch(body)
                    .map_err(|e| self.error(e.to_string()))?;
                self.math.block = self
                    .math
                    .b
                    .begin_block(Some(body))
                    .map_err(|e| self.error(e.to_string()))?;
            }
            if block_id == self.function.entry() {
                self.initialize_fragment_depth()?;
            }
            for instruction in &block.instructions {
                self.math.source = instruction.source;
                self.operation(instruction)?;
            }
            self.math.source = block.terminator_source;
            if structure.loops.contains_key(&block_id)
                && matches!(
                    block.terminator,
                    Terminator::Branch { .. } | Terminator::Switch { .. }
                )
            {
                let merge = self.math.b.id();
                ghost_merges.push(merge);
                self.math
                    .b
                    .selection_merge(merge, spv::SelectionControl::NONE)
                    .map_err(|e| self.error(e.to_string()))?;
            } else if let Some(merge) = structure.selections.get(&block_id) {
                let label = if let Some(merge) = merge {
                    self.labels[merge.index()]
                } else {
                    let label = self.math.b.id();
                    ghost_merges.push(label);
                    label
                };
                self.math
                    .b
                    .selection_merge(label, spv::SelectionControl::NONE)
                    .map_err(|e| self.error(e.to_string()))?;
            }
            self.terminator(&block.terminator)?;
            self.ends[block_id.index()] = self.math.block;
        }
        for (&header, &label) in &continuations {
            self.math
                .b
                .begin_block(Some(label))
                .map_err(|e| self.error(e.to_string()))?;
            for (ty, result, pairs) in continuation_phis.remove(&header).unwrap_or_default() {
                self.math
                    .b
                    .phi(ty, Some(result), pairs)
                    .map_err(|e| self.error(e.to_string()))?;
            }
            self.math
                .b
                .branch(self.labels[header.index()])
                .map_err(|e| self.error(e.to_string()))?;
        }
        for label in ghost_merges {
            self.math
                .b
                .begin_block(Some(label))
                .map_err(|e| self.error(e.to_string()))?;
            self.math
                .b
                .unreachable()
                .map_err(|e| self.error(e.to_string()))?;
        }
        self.stage()?;
        self.math
            .b
            .end_function()
            .map_err(|e| self.error(e.to_string()))?;
        let mut module = std::mem::take(&mut self.math.b).module();
        let resolve = |mut id: u32| {
            while let Some(&next) = self.aliases.get(&id) {
                id = next;
            }
            id
        };
        // Math helpers may introduce internal structured blocks. Phi edges use
        // their final block, while branch targets retain the SSA block's entry.
        let ends: HashMap<_, _> = self
            .labels
            .iter()
            .copied()
            .zip(self.ends.iter().copied())
            .collect();
        for f in &mut module.functions {
            for b in &mut f.blocks {
                for instruction in &mut b.instructions {
                    if instruction.class.opcode == spv::Op::Phi
                        && instruction
                            .result_id
                            .is_some_and(|id| patched_phis.contains(&id))
                    {
                        for pair in instruction.operands.chunks_exact_mut(2) {
                            if let SpvOperand::IdRef(label) = &mut pair[1]
                                && let Some(&end) = ends.get(label)
                            {
                                *label = end;
                            }
                        }
                    }
                }
            }
        }
        for instruction in module.all_inst_iter_mut() {
            for operand in &mut instruction.operands {
                if let SpvOperand::IdRef(id) = operand {
                    *id = resolve(*id);
                }
            }
        }
        canonicalize::run(&mut module);
        Ok(Module {
            words: module.assemble().into_boxed_slice(),
            bindings: std::mem::take(&mut self.bindings).into_boxed_slice(),
            push_constant_bytes: self.push_constant_bytes,
        })
    }
    fn stage(&mut self) -> Result<(), Error> {
        use spv::ExecutionMode as M;
        let model = match self.function.stage {
            ShaderStage::Compute => {
                let size = self
                    .function
                    .workgroup_size
                    .filter(|s| !s.contains(&0))
                    .ok_or_else(|| self.error("compute workgroup size is missing/zero"))?;
                self.math
                    .b
                    .execution_mode(self.math.entry, M::LocalSize, size);
                spv::ExecutionModel::GLCompute
            }
            ShaderStage::Vertex => spv::ExecutionModel::Vertex,
            ShaderStage::Fragment => {
                self.math
                    .b
                    .execution_mode(self.math.entry, M::OriginUpperLeft, []);
                if self.options.fragment_depth_transform.is_some()
                    || self
                        .function
                        .outputs
                        .iter()
                        .any(|o| o.location() == ShaderIoLocation::FragmentDepth)
                {
                    self.math
                        .b
                        .execution_mode(self.math.entry, M::DepthReplacing, []);
                }
                spv::ExecutionModel::Fragment
            }
            ShaderStage::TessellationControl => {
                self.math.b.capability(spv::Capability::Tessellation);
                let count = self
                    .function
                    .tessellation_control_points
                    .filter(|c| *c != 0)
                    .ok_or_else(|| self.error("control output cardinality is missing/zero"))?;
                self.math
                    .b
                    .execution_mode(self.math.entry, M::OutputVertices, [count]);
                spv::ExecutionModel::TessellationControl
            }
            ShaderStage::TessellationEvaluation => {
                self.math.b.capability(spv::Capability::Tessellation);
                let mode = self
                    .options
                    .shader
                    .tessellation_mode
                    .ok_or_else(|| self.error("evaluation tessellation mode is missing"))?;
                use crate::{
                    TessellationDomain as D, TessellationOutput as O, TessellationSpacing as S,
                    TessellationWinding as W,
                };
                self.math.b.execution_mode(
                    self.math.entry,
                    match mode.domain {
                        D::Triangles => M::Triangles,
                        D::Quads => M::Quads,
                        D::Isolines => M::Isolines,
                    },
                    [],
                );
                self.math.b.execution_mode(
                    self.math.entry,
                    match mode.spacing {
                        S::Equal => M::SpacingEqual,
                        S::FractionalEven => M::SpacingFractionalEven,
                        S::FractionalOdd => M::SpacingFractionalOdd,
                    },
                    [],
                );
                match (mode.domain, mode.output) {
                    (_, O::Points) => self
                        .math
                        .b
                        .execution_mode(self.math.entry, M::PointMode, []),
                    (D::Isolines, O::Lines) => {}
                    (D::Triangles | D::Quads, O::Triangles(w)) => self.math.b.execution_mode(
                        self.math.entry,
                        match w {
                            W::Clockwise => M::VertexOrderCw,
                            W::CounterClockwise => M::VertexOrderCcw,
                        },
                        [],
                    ),
                    _ => return Err(self.error("incompatible tessellation domain/output")),
                }
                spv::ExecutionModel::TessellationEvaluation
            }
            ShaderStage::Geometry => {
                return Err(self.error("geometry shader semantics are not implemented"));
            }
        };
        self.math
            .b
            .entry_point(model, self.math.entry, "main", &self.io.variables);
        Ok(())
    }
    fn target(&self, to: BlockId) -> u32 {
        self.backedges
            .get(&(self.current, to))
            .copied()
            .unwrap_or(self.labels[to.index()])
    }
    fn terminator(&mut self, term: &Terminator) -> Result<(), Error> {
        let result = match term {
            Terminator::Jump(b) => self.math.b.branch(self.target(*b)),
            Terminator::Branch {
                condition,
                when_true,
                when_false,
            } => self.math.b.branch_conditional(
                self.id(*condition),
                self.target(*when_true),
                self.target(*when_false),
                [],
            ),
            Terminator::Switch {
                selector,
                cases,
                default,
            } => {
                let cases = cases
                    .iter()
                    .map(|(literal, b)| (SpvOperand::LiteralBit32(*literal), self.target(*b)))
                    .collect::<Vec<_>>();
                self.math
                    .b
                    .switch(self.id(*selector), self.target(*default), cases)
            }
            Terminator::Return => {
                self.io
                    .final_position(&mut self.math)
                    .map_err(|e| Error::at(self.math.source, e.to_string()))?;
                self.math.b.ret()
            }
            Terminator::Discard => self.math.b.kill(),
            Terminator::Unreachable => self.math.b.unreachable(),
        };
        result.map_err(|e| self.error(e.to_string()))
    }
    fn resources(&mut self) -> Result<(), Error> {
        let mut words = 0;
        for block in &self.function.blocks {
            for inst in &block.instructions {
                if let Op::Parameter { index, ty } = inst.op {
                    if !matches!(ty, Type::U32 | Type::I32 | Type::F32 | Type::Bool) {
                        return Err(Error::at(
                            inst.source,
                            "parameter ABI requires a 32-bit scalar word",
                        ));
                    }
                    words =
                        words.max(index.checked_add(1).ok_or_else(|| {
                            Error::at(inst.source, "parameter index overflows ABI")
                        })?);
                }
            }
        }
        self.push_constant_bytes = words
            .checked_mul(4)
            .ok_or_else(|| self.error("push constant byte range overflows"))?;
        if words != 0 {
            let count = self.math.constant(words);
            let array = self.math.b.type_array(self.math.uint, count);
            self.math.b.decorate(
                array,
                spv::Decoration::ArrayStride,
                [SpvOperand::LiteralBit32(4)],
            );
            let block = self.math.b.type_struct([array]);
            self.math.b.decorate(block, spv::Decoration::Block, []);
            self.math.b.member_decorate(
                block,
                0,
                spv::Decoration::Offset,
                [SpvOperand::LiteralBit32(0)],
            );
            let ptr = self
                .math
                .b
                .type_pointer(None, spv::StorageClass::PushConstant, block);
            let variable = self
                .math
                .b
                .variable(ptr, None, spv::StorageClass::PushConstant, None);
            self.io.variables.push(variable);
            self.parameters = Some(variable);
        }

        // rspirv interns structurally identical types. Buffer variables share
        // their ABI type, whose layout decorations must be emitted only once.
        let mut storage_block = None;
        let mut uniform_block = None;
        for block in &self.function.blocks {
            for inst in &block.instructions {
                self.reflect_texture(inst)?;
                if matches!(inst.op, Op::QuadSwizzleAdd { .. }) {
                    self.declare_quad_lane();
                }
                if let Op::Builtin { builtin, .. } = inst.op {
                    self.declare_builtin(builtin)?;
                }
                let (space, resource, read, write) = match inst.op {
                    Op::Load {
                        space, resource, ..
                    } => (space, resource, true, false),
                    Op::Store {
                        space, resource, ..
                    } => (space, resource, false, true),
                    _ => continue,
                };
                let Some(resource) = resource else {
                    self.declare_memory(space)?;
                    continue;
                };
                if self.descriptors.contains_key(&resource) {
                    return Err(Error::at(
                        inst.source,
                        "buffer identity aliases an image/sampler descriptor",
                    ));
                }
                if let Some(buffer) = self.buffers.get(&resource) {
                    if buffer.space != space {
                        return Err(Error::at(
                            inst.source,
                            "resource identity has conflicting memory classes",
                        ));
                    }
                    let binding = self
                        .bindings
                        .iter_mut()
                        .find(|b| b.resource == resource)
                        .unwrap();
                    binding.readable |= read;
                    binding.writable |= write;
                    continue;
                }
                // std140 uvec4 words retain the guest's byte-addressed ABI
                // without scalar-layout features or padding between words.
                // https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources-layout
                let uniform = space == MemorySpace::Constant;
                if uniform
                    && (self.options.uniform_buffer_bytes < 16
                        || !self.options.uniform_buffer_bytes.is_multiple_of(16))
                {
                    return Err(
                        self.error("uniform ABI extent must be a nonzero multiple of 16 bytes")
                    );
                }
                let storage = if uniform {
                    spv::StorageClass::Uniform
                } else {
                    spv::StorageClass::StorageBuffer
                };
                let block = if uniform {
                    *uniform_block.get_or_insert_with(|| {
                        let vector = self.math.b.type_vector(self.math.uint, 4);
                        let count = self.math.constant(self.options.uniform_buffer_bytes / 16);
                        let array = self.math.b.type_array(vector, count);
                        self.math.b.decorate(
                            array,
                            spv::Decoration::ArrayStride,
                            [SpvOperand::LiteralBit32(16)],
                        );
                        let block = self.math.b.type_struct([array]);
                        self.math.b.decorate(block, spv::Decoration::Block, []);
                        self.math.b.member_decorate(
                            block,
                            0,
                            spv::Decoration::Offset,
                            [SpvOperand::LiteralBit32(0)],
                        );
                        block
                    })
                } else {
                    *storage_block.get_or_insert_with(|| {
                        let array = self.math.b.type_runtime_array(self.math.uint);
                        self.math.b.decorate(
                            array,
                            spv::Decoration::ArrayStride,
                            [SpvOperand::LiteralBit32(4)],
                        );
                        let block = self.math.b.type_struct([array]);
                        self.math.b.decorate(block, spv::Decoration::Block, []);
                        self.math.b.member_decorate(
                            block,
                            0,
                            spv::Decoration::Offset,
                            [SpvOperand::LiteralBit32(0)],
                        );
                        block
                    })
                };
                let ptr = self.math.b.type_pointer(None, storage, block);
                let variable = self.math.b.variable(ptr, None, storage, None);
                let binding = self.binding(resource);
                self.math.b.decorate(
                    variable,
                    spv::Decoration::DescriptorSet,
                    [SpvOperand::LiteralBit32(0)],
                );
                self.math.b.decorate(
                    variable,
                    spv::Decoration::Binding,
                    [SpvOperand::LiteralBit32(binding)],
                );
                self.io.variables.push(variable);
                let pointer = self.math.b.type_pointer(None, storage, self.math.uint);
                self.buffers.insert(
                    resource,
                    Buffer {
                        variable,
                        pointer,
                        space,
                    },
                );
                self.bindings.push(Binding {
                    resource,
                    binding,
                    kind: if space == MemorySpace::Constant {
                        crate::ShaderResourceKind::ConstantBuffer
                    } else {
                        crate::ShaderResourceKind::StorageBuffer
                    },
                    readable: read,
                    writable: write,
                });
            }
        }
        for binding in &self.bindings {
            if let Some(buffer) = self.buffers.get(&binding.resource) {
                if !binding.writable {
                    self.math
                        .b
                        .decorate(buffer.variable, spv::Decoration::NonWritable, []);
                }
                if !binding.readable {
                    self.math
                        .b
                        .decorate(buffer.variable, spv::Decoration::NonReadable, []);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
