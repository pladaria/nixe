//! Small shaders for native resource and execution tests.
use crate::cache::{CompiledStage, ShaderArtifact};
use nixe_shader_recompiler::{
    Profile, RuntimeInfo, Stage,
    arena::BlockId,
    artifact::{Bindings, StorageBufferDescriptor, TextureDescriptor},
    ir::{Attribute, Ir, Opcode as O, Program, SyntaxNode, TextureInstInfo, Type, Value as V},
    types::{AttributeType, ByteOffset, CbufIndex, Interpolation, TextureType},
};
use std::sync::Arc;

pub(crate) struct Shader {
    pub ir: Ir,
    pub program: Program,
    pub block: BlockId,
    pub runtime: RuntimeInfo,
}
impl Shader {
    pub fn new(stage: Stage) -> Self {
        let mut ir = Ir::new().unwrap();
        let block = ir.new_block().unwrap();
        let mut program = Program::new(stage);
        program.blocks = vec![block];
        program.post_order = vec![block];
        program.syntax = vec![SyntaxNode::Block(block), SyntaxNode::Return];
        program.execution.workgroup_size = [1; 3];
        let mut runtime = RuntimeInfo::default();
        runtime.previous_stage_stores.mask = [u64::MAX; 8];
        Self {
            ir,
            program,
            block,
            runtime,
        }
    }
    pub fn op(&mut self, opcode: O, args: &[V]) -> V {
        V::Inst(self.ir.append(self.block, opcode, args, 0).unwrap())
    }
    pub fn storage(&mut self) {
        self.program.info.used_storage_buffer_types = Type::U32;
        self.program
            .info
            .storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: CbufIndex(0),
                cbuf_offset: ByteOffset(0),
                count: 1,
                is_written: true,
            })
            .unwrap();
    }
    pub fn input(&mut self, attr: Attribute, vertex: u32) -> V {
        self.program.info.loads.set(attr, true);
        self.op(O::GetAttribute, &[V::Attribute(attr), V::U32(vertex)])
    }
    pub fn output(&mut self, attr: Attribute, value: V) {
        self.program.info.stores.set(attr, true);
        self.op(O::SetAttribute, &[V::Attribute(attr), value, V::U32(0)]);
    }
    pub fn color(&mut self, component: u32, value: V) {
        self.program.info.stores_frag_color[0] = true;
        self.op(O::SetFragColor, &[V::U32(0), V::U32(component), value]);
    }
    pub fn compile(mut self, profile: &Profile, bindings: &mut Bindings) -> CompiledStage {
        let artifact = nixe_shader_recompiler::spirv::emit_spirv(
            &mut self.ir,
            self.program,
            profile,
            &self.runtime,
            &Default::default(),
            bindings,
        )
        .unwrap();
        CompiledStage::from_guest(artifact).unwrap()
    }
}
pub(crate) fn compile(
    profile: &Profile,
    shaders: impl IntoIterator<Item = Shader>,
) -> Arc<ShaderArtifact> {
    let mut bindings = Bindings::default();
    Arc::new(ShaderArtifact::new(
        shaders
            .into_iter()
            .map(|s| s.compile(profile, &mut bindings))
            .collect(),
    ))
}
pub(crate) fn increment(amount: u32, destination: u32) -> Shader {
    let mut s = Shader::new(Stage::Compute);
    s.storage();
    let value = s.op(O::LoadStorage32, &[V::U32(0), V::U32(0)]);
    let value = s.op(O::IAdd32, &[value, V::U32(amount)]);
    s.op(O::WriteStorage32, &[V::U32(0), V::U32(destination), value]);
    s
}
pub(crate) fn vertex_fragment(counter: bool, scaled: bool) -> [Shader; 2] {
    let mut vertex = Shader::new(Stage::VertexB);
    if scaled {
        vertex.runtime.generic_input_types[1] = AttributeType::UnsignedScaled;
    }
    for c in 0..4 {
        let v = if c == 3 {
            V::f32(1.)
        } else {
            vertex.input(Attribute::generic(0, c).unwrap(), 0)
        };
        vertex.output(
            Attribute::from_index((Attribute::PositionX.index() + c) as u16).unwrap(),
            v,
        );
    }
    for c in 0..3 {
        let v = vertex.input(Attribute::generic(1, c).unwrap(), 0);
        vertex.output(Attribute::generic(0, c).unwrap(), v);
    }
    if counter {
        vertex.storage();
        vertex.program.info.loads.set(Attribute::VertexId, true);
        let index = vertex.op(
            O::GetAttributeU32,
            &[V::Attribute(Attribute::VertexId), V::U32(0)],
        );
        let address = vertex.op(O::IMul32, &[index, V::U32(4)]);
        let value = vertex.op(O::LoadStorage32, &[V::U32(0), address]);
        let value = vertex.op(O::IAdd32, &[value, V::U32(1)]);
        vertex.op(O::WriteStorage32, &[V::U32(0), address, value]);
    }
    let mut fragment = Shader::new(Stage::Fragment);
    fragment.program.info.interpolation[0] = Interpolation::Flat;
    for c in 0..4 {
        let v = if c == 3 {
            V::f32(1.)
        } else {
            fragment.input(Attribute::generic(0, c as usize).unwrap(), 0)
        };
        fragment.color(c, v);
    }
    [vertex, fragment]
}
pub(crate) fn sampled_fragment(coords: [f32; 2]) -> Shader {
    let mut s = Shader::new(Stage::Fragment);
    s.program.info.texture_descriptors.push(TextureDescriptor {
        texture_type: TextureType::Color2D,
        count: 1,
        ..Default::default()
    });
    let coords = s.op(O::CompositeConstructF32x2, &coords.map(V::f32));
    let mut flags = TextureInstInfo::default();
    flags.set_texture_type(TextureType::Color2D);
    let sample = V::Inst(
        s.ir.append(
            s.block,
            O::ImageSampleExplicitLod,
            &[V::U32(0), coords, V::f32(0.), V::None],
            flags.0,
        )
        .unwrap(),
    );
    for c in 0..4 {
        let v = s.op(O::CompositeExtractF32x4, &[sample, V::U32(c)]);
        s.color(c, v);
    }
    s
}

#[test]
fn native_execution_fixtures_validate_without_a_gpu() {
    let profile = Profile {
        supported_spirv: 0x10500,
        unified_descriptor_binding: true,
        ..Default::default()
    };
    let shaders = vertex_fragment(false, false)
        .into_iter()
        .chain(vertex_fragment(true, true))
        .chain([increment(7, 4), sampled_fragment([0.5; 2])]);
    let root = tempfile::tempdir().unwrap();
    for (index, shader) in shaders.enumerate() {
        let stage = shader.compile(&profile, &mut Bindings::default());
        let path = root.path().join(format!("{index}.spv"));
        std::fs::write(
            &path,
            stage
                .module
                .words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let output = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(path)
            .output()
            .expect("spirv-val must be installed for shader validation");
        assert!(
            output.status.success(),
            "stage {index}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
