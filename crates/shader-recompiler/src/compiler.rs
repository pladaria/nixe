use crate::{
    CompileError, CompilerSettings, Environment, HostTranslateInfo, Profile, ProgramId, Result,
    RuntimeInfo, Stage, StageArtifact,
    arena::InstId,
    artifact::Bindings,
    diagnostics::FailureCategory,
    environment::EnvironmentCaches,
    ir::{Ir, Program, SyntaxNode, Value},
    maxwell::{
        asl::build_asl, flow::FlowGraph, location::Location, structure::StatementTree,
        translate::MaxwellTranslator,
    },
    passes::{self, GraphicsRuntimeState},
    session::ProgramStore,
    types::OutputTopology,
};

/// Reusable, worker-local compilation scratch. Handles are invalid after reset,
/// merging or emission; artifacts have no references into this storage.
pub struct Compiler {
    ir: Ir,
    programs: ProgramStore<Option<Program>>,
    caches: EnvironmentCaches<InstId, Value>,
}
/// A linked stage ready for emission; valid only in its creating session.
pub struct PreparedStage {
    id: ProgramId,
    pub stage: Stage,
    pub runtime: RuntimeInfo,
}
impl Compiler {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ir: Ir::new()?,
            programs: ProgramStore::new()?,
            caches: EnvironmentCaches::default(),
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        self.ir.reset()?;
        self.programs.reset()?;
        self.caches.clear();
        Ok(())
    }
    pub fn program(&self, id: ProgramId) -> Result<&Program> {
        self.programs.get(id)?.as_ref().ok_or_else(|| {
            CompileError::new(
                FailureCategory::InvalidArgument,
                "Program has already been consumed",
            )
        })
    }
    fn take(&mut self, id: ProgramId) -> Result<Program> {
        self.programs.get_mut(id)?.take().ok_or_else(|| {
            CompileError::new(
                FailureCategory::InvalidArgument,
                "Program has already been consumed",
            )
        })
    }
    pub fn dump_ir(&self, id: ProgramId) -> Result<String> {
        self.ir.dump_program(self.program(id)?)
    }
    /// Inspects cold compiler state without transferring ownership of its arenas.
    pub fn inspect_program<T>(
        &self,
        id: ProgramId,
        inspect: impl FnOnce(&Ir, &Program) -> Result<T>,
    ) -> Result<T> {
        inspect(&self.ir, self.program(id)?)
    }
    pub fn translate<E: Environment>(
        &mut self,
        env: &mut E,
        host: &HostTranslateInfo,
        settings: &CompilerSettings,
    ) -> Result<ProgramId> {
        let mut host = host.clone();
        host.apply_descriptor_limit_policy();
        let stage = env.metadata().stage;
        let entry = Location::new(env.metadata().entry_address()?)?;
        let graph = FlowGraph::analyze(env, entry, stage == Stage::VertexA)?;
        let mut tree = StatementTree::from_flow(&graph)?;
        let syntax = build_asl(
            env,
            &mut self.ir,
            &graph,
            &mut tree,
            &host,
            &mut MaxwellTranslator,
        )?;
        let mut p = Program::new(stage);
        p.syntax = syntax;
        p.blocks = p
            .syntax
            .iter()
            .filter_map(|node| {
                if let SyntaxNode::Block(id) = node {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        let first = p.syntax.first().ok_or_else(|| {
            CompileError::new(
                FailureCategory::Internal,
                "Shader has no structured entry block",
            )
        })?;
        p.post_order = self.ir.post_order(first)?;
        passes::configure_stage(&mut self.ir, &mut p, env, &host)?;
        if p.blocks.len() != p.post_order.len() {
            let mut retained = Vec::with_capacity(p.blocks.len());
            for (i, &block) in p.blocks.iter().enumerate() {
                if i == 0 || !self.ir.block(block)?.predecessors().is_empty() {
                    retained.push(block);
                }
            }
            p.blocks = retained;
        }
        passes::early_passes(&mut self.ir, &p, &host)?;
        passes::constant_propagation(env, &mut self.ir, &p)?;
        passes::position(env, &mut self.ir, &mut p)?;
        passes::global_memory_to_storage(&mut self.ir, &mut p, &host)?;
        passes::geometry_compaction(&mut self.ir, &p, &host)?;
        passes::texture_pass(&mut self.ir, &mut p, env, &host, &mut self.caches)?;
        if settings.resolution_info.active {
            passes::rescaling(&mut self.ir, &p, &settings.resolution_info)?;
        }
        passes::dead_code_elimination(&mut self.ir, &p)?;
        passes::tighten_output_vertices(&self.ir, &mut p)?;
        if settings.renderer_debug {
            passes::verify_program(&self.ir, &p)?;
        }
        passes::collect_shader_info(env, &self.ir, &mut p)?;
        passes::layer(&mut self.ir, &mut p, &host)?;
        passes::vendor_workaround(&mut self.ir, &p)?;
        passes::collect_interpolation(env, &mut p)?;
        passes::add_nvn_storage_buffers(&mut p)?;
        self.programs.insert(Some(p))
    }
    pub fn merge_dual_vertex<E: Environment>(
        &mut self,
        a: ProgramId,
        b: ProgramId,
        env_b: &E,
        settings: &CompilerSettings,
    ) -> Result<ProgramId> {
        if self.program(a)?.stage != Stage::VertexA || self.program(b)?.stage != Stage::VertexB {
            return Err(CompileError::new(
                FailureCategory::InvalidArgument,
                "Dual vertex merge requires Vertex A followed by Vertex B",
            ));
        }
        let a = self.take(a)?;
        let b = self.take(b)?;
        let p = passes::merge_dual_vertex(&mut self.ir, a, b, env_b, settings.renderer_debug)?;
        self.programs.insert(Some(p))
    }
    pub fn has_broken_shuffle_pattern(&self, id: ProgramId) -> Result<bool> {
        passes::has_broken_shuffle_pattern(&self.ir, self.program(id)?)
    }
    pub fn emit(
        &mut self,
        id: ProgramId,
        profile: &Profile,
        runtime: &RuntimeInfo,
        settings: &CompilerSettings,
        bindings: &mut Bindings,
    ) -> Result<StageArtifact> {
        let mut p = self.take(id)?;
        passes::convert_legacy_to_generic(&mut self.ir, &mut p, runtime)?;
        crate::spirv::emit_spirv(&mut self.ir, p, profile, runtime, settings, bindings)
    }
    /// Stages must be ordered Vertex B, tessellation control/evaluation, geometry,
    /// fragment. Dual vertex merging is performed while the B environment exists.
    pub fn emit_graphics(
        &mut self,
        ids: &[ProgramId],
        profile: &Profile,
        state: &GraphicsRuntimeState,
        settings: &CompilerSettings,
        bindings: &mut Bindings,
    ) -> Result<Vec<StageArtifact>> {
        self.prepare_graphics(ids, state)?
            .into_iter()
            .map(|stage| self.emit_prepared(stage, profile, settings, bindings))
            .collect()
    }
    pub fn prepare_stage(&mut self, id: ProgramId, runtime: RuntimeInfo) -> Result<PreparedStage> {
        let mut program = self.take(id)?;
        passes::convert_legacy_to_generic(&mut self.ir, &mut program, &runtime)?;
        let stage = program.stage;
        let id = self.programs.insert(Some(program))?;
        Ok(PreparedStage { id, stage, runtime })
    }
    pub fn emit_prepared(
        &mut self,
        stage: PreparedStage,
        profile: &Profile,
        settings: &CompilerSettings,
        bindings: &mut Bindings,
    ) -> Result<StageArtifact> {
        let program = self.take(stage.id)?;
        crate::spirv::emit_spirv(
            &mut self.ir,
            program,
            profile,
            &stage.runtime,
            settings,
            bindings,
        )
    }
    /// Apply a renderer-owned legalization to cold session scratch. The callback
    /// must keep the program's resource/interface metadata consistent with its IR.
    pub fn transform_prepared(
        &mut self,
        stage: &PreparedStage,
        transform: impl FnOnce(&mut Ir, &mut Program) -> Result<()>,
    ) -> Result<()> {
        let program = self.programs.get_mut(stage.id)?.as_mut().ok_or_else(|| {
            CompileError::new(
                FailureCategory::InvalidArgument,
                "Prepared stage was consumed",
            )
        })?;
        transform(&mut self.ir, program)
    }
    /// Resolve linkage before consulting an emitted-stage cache. The resulting
    /// runtime includes preceding-stage varyings and legacy remapping.
    pub fn prepare_graphics(
        &mut self,
        ids: &[ProgramId],
        state: &GraphicsRuntimeState,
    ) -> Result<Vec<PreparedStage>> {
        let mut programs = Vec::with_capacity(ids.len() + 1);
        let mut previous_stage = None;
        for &id in ids {
            let stage = self.program(id)?.stage;
            if matches!(stage, Stage::VertexA | Stage::Compute)
                || previous_stage.is_some_and(|prev| prev >= stage as u32)
            {
                return Err(CompileError::new(
                    FailureCategory::InvalidArgument,
                    "Graphics stages must be unique, merged and ordered",
                ));
            }
            previous_stage = Some(stage as u32);
            programs.push(self.take(id)?);
        }
        let has_guest_geometry = programs.iter().any(|p| p.stage == Stage::Geometry);
        if !has_guest_geometry
            && let Some(source) = programs
                .iter()
                .rev()
                .find(|p| p.info.requires_layer_emulation)
        {
            let topology = match state.input_topology {
                crate::types::InputTopology::Points => OutputTopology::PointList,
                crate::types::InputTopology::Lines
                | crate::types::InputTopology::LinesAdjacency => OutputTopology::LineStrip,
                _ => OutputTopology::TriangleStrip,
            };
            let generated = passes::generate_geometry_passthrough(&mut self.ir, source, topology)?;
            let position = programs
                .iter()
                .position(|p| p.stage == Stage::Fragment)
                .unwrap_or(programs.len());
            programs.insert(position, generated);
        }
        let mut runtimes = Vec::with_capacity(programs.len());
        for i in 0..programs.len() {
            let runtime = passes::make_runtime_info(
                &programs[i],
                i.checked_sub(1).map(|j| &programs[j]),
                has_guest_geometry,
                state,
            );
            passes::convert_legacy_to_generic(&mut self.ir, &mut programs[i], &runtime)?;
            runtimes.push(runtime);
        }
        programs
            .into_iter()
            .zip(runtimes)
            .map(|(program, runtime)| {
                let stage = program.stage;
                let id = self.programs.insert(Some(program))?;
                Ok(PreparedStage { id, stage, runtime })
            })
            .collect()
    }
    pub fn compile<E: Environment>(
        &mut self,
        env: &mut E,
        host: &HostTranslateInfo,
        profile: &Profile,
        runtime: &RuntimeInfo,
        settings: &CompilerSettings,
        bindings: &mut Bindings,
    ) -> Result<StageArtifact> {
        self.reset()?;
        let id = self.translate(env, host, settings)?;
        self.emit(id, profile, runtime, settings, bindings)
    }
}
