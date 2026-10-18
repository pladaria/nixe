use super::{Builder, CommonConstants, CommonTypes, Id};
use crate::{
    Profile, Result, RuntimeInfo,
    artifact::{Bindings, ResourceBinding},
    ir::Program,
    types::Stage,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputLoad {
    #[default]
    None,
    Bitcast,
    SignedToFloat,
    UnsignedToFloat,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct InputGeneric {
    pub id: Id,
    pub pointer: Id,
    pub component: Id,
    pub load: InputLoad,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputGeneric {
    pub id: Id,
    pub first: u32,
    pub components: u32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageDefinition {
    pub id: Id,
    pub sampled_type: Id,
    pub pointer_type: Id,
    pub image_type: Id,
    pub count: u32,
    pub is_multisample: bool,
    pub is_integer: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum BufferKind {
    U8,
    S8,
    U16,
    S16,
    U32,
    F32,
    U64,
    U32x2,
    U32x4,
}
impl BufferKind {
    pub const COUNT: usize = 9;
}
#[derive(Clone, Copy, Debug, Default)]
pub struct StoragePointers {
    pub array: Id,
    pub element: Id,
}

pub struct EmitContext<'a> {
    pub builder: Builder,
    pub profile: &'a Profile,
    pub runtime: &'a RuntimeInfo,
    pub stage: Stage,
    pub types: CommonTypes,
    pub constants: CommonConstants,
    pub objects: Objects,
    pub interfaces: Vec<Id>,
    pub resource_bindings: Vec<ResourceBinding>,
    pub host_uniforms: Vec<crate::artifact::HostUniformLayout>,
    pub uniform_types: [Id; BufferKind::COUNT],
    pub storage_types: [StoragePointers; BufferKind::COUNT],
    pub cbufs: [[Id; BufferKind::COUNT]; 18],
    pub ssbos: [[Id; BufferKind::COUNT]; 32],
    pub texture_buffers: Vec<ImageDefinition>,
    pub image_buffers: Vec<ImageDefinition>,
    pub textures: Vec<ImageDefinition>,
    pub images: Vec<ImageDefinition>,
    pub input_generics: [InputGeneric; 32],
    pub output_generics: [[OutputGeneric; 4]; 32],
    pub patches: [Id; 30],
    pub frag_colors: [Id; 8],
    pub need_input_position_indirect: bool,
    pub uses_explicit_workgroup_layout: bool,
    pub texture_scaling_index: u32,
    pub image_scaling_index: u32,
    pub clip_distance_written: u8,
    pub fix_bloom_effects: bool,
    pub(super) deferred_phis: Vec<(crate::arena::InstId, super::DeferredPhi)>,
    pub opcode_rewrites: Vec<(Id, spirv::Op)>,
    pub uses_nonuniform: [bool; 4],
    pub non_uniform_ids: std::collections::HashSet<Id>,
}
impl<'a> EmitContext<'a> {
    pub fn new(
        profile: &'a Profile,
        runtime: &'a RuntimeInfo,
        program: &Program,
        bindings: &mut Bindings,
    ) -> Result<Self> {
        let mut builder = Builder::new(profile.supported_spirv);
        builder.capability(spirv::Capability::Shader)?;
        let types = CommonTypes::define(&mut builder, profile, &program.info)?;
        let constants = CommonConstants::define(&mut builder, &types)?;
        let mut ctx = Self {
            builder,
            profile,
            runtime,
            stage: program.stage,
            types,
            constants,
            objects: Objects::default(),
            interfaces: Vec::new(),
            resource_bindings: Vec::new(),
            host_uniforms: Vec::new(),
            uniform_types: [Id::NONE; BufferKind::COUNT],
            storage_types: [StoragePointers::default(); BufferKind::COUNT],
            cbufs: [[Id::NONE; BufferKind::COUNT]; 18],
            ssbos: [[Id::NONE; BufferKind::COUNT]; 32],
            texture_buffers: Vec::new(),
            image_buffers: Vec::new(),
            textures: Vec::new(),
            images: Vec::new(),
            input_generics: [InputGeneric::default(); 32],
            output_generics: [[OutputGeneric::default(); 4]; 32],
            patches: [Id::NONE; 30],
            frag_colors: [Id::NONE; 8],
            need_input_position_indirect: false,
            uses_explicit_workgroup_layout: false,
            texture_scaling_index: bindings.texture_scaling_index,
            image_scaling_index: bindings.image_scaling_index,
            clip_distance_written: 0,
            fix_bloom_effects: false,
            deferred_phis: Vec::new(),
            opcode_rewrites: Vec::new(),
            uses_nonuniform: [false; 4],
            non_uniform_ids: std::collections::HashSet::new(),
        };
        ctx.define_inputs(program)?;
        ctx.define_outputs(program)?;
        ctx.define_local_memory(program)?;
        ctx.define_shared_memory(program)?;
        ctx.define_shared_atomics(program)?;
        ctx.define_constant_buffers(
            &program.info,
            if profile.unified_descriptor_binding {
                &mut bindings.unified
            } else {
                &mut bindings.uniform_buffer
            },
        )?;
        ctx.define_cbuf_accessors(&program.info)?;
        ctx.define_storage_buffers(
            &program.info,
            if profile.unified_descriptor_binding {
                &mut bindings.unified
            } else {
                &mut bindings.storage_buffer
            },
        )?;
        ctx.define_storage_atomics(&program.info)?;
        ctx.define_texture_buffers(
            &program.info,
            if profile.unified_descriptor_binding {
                &mut bindings.unified
            } else {
                &mut bindings.texture
            },
        )?;
        ctx.define_image_buffers(
            &program.info,
            if profile.unified_descriptor_binding {
                &mut bindings.unified
            } else {
                &mut bindings.image
            },
        )?;
        ctx.define_textures(
            &program.info,
            if profile.unified_descriptor_binding {
                &mut bindings.unified
            } else {
                &mut bindings.texture
            },
            &mut bindings.texture_scaling_index,
        )?;
        ctx.define_images(
            &program.info,
            if profile.unified_descriptor_binding {
                &mut bindings.unified
            } else {
                &mut bindings.image
            },
            &mut bindings.image_scaling_index,
        )?;
        ctx.define_attribute_accessors(&program.info)?;
        ctx.define_narrow_storage_write(&program.info)?;
        ctx.define_global_accessors(&program.info)?;
        ctx.define_host_uniforms(&program.info)?;
        Ok(ctx)
    }
    pub fn const_u32(&mut self, value: u32) -> Result<Id> {
        self.builder.constant(self.types.u32[0], &[value])
    }
    pub fn const_f32(&mut self, value: f32) -> Result<Id> {
        self.builder.constant(self.types.f32[0], &[value.to_bits()])
    }
}

#[derive(Clone, Debug, Default)]
pub struct Objects {
    pub workgroup_id: Id,
    pub local_invocation_id: Id,
    pub invocation_id: Id,
    pub patch_vertices_in: Id,
    pub sample_id: Id,
    pub is_helper_invocation: Id,
    pub subgroup_local_invocation_id: Id,
    pub subgroup_mask_eq: Id,
    pub subgroup_mask_lt: Id,
    pub subgroup_mask_le: Id,
    pub subgroup_mask_gt: Id,
    pub subgroup_mask_ge: Id,
    pub instance_id: Id,
    pub instance_index: Id,
    pub base_instance: Id,
    pub vertex_id: Id,
    pub vertex_index: Id,
    pub draw_index: Id,
    pub base_vertex: Id,
    pub front_face: Id,
    pub point_coord: Id,
    pub tess_coord: Id,
    pub clip_distances: Id,
    pub layer: Id,
    pub viewport_index: Id,
    pub viewport_mask: Id,
    pub primitive_id: Id,
    pub fswzadd_lut_a: Id,
    pub fswzadd_lut_b: Id,
    pub indexed_load_func: Id,
    pub indexed_store_func: Id,
    pub rescaling_uniform_constant: Id,
    pub rescaling_push_constants: Id,
    pub rescaling_textures_type: Id,
    pub rescaling_images_type: Id,
    pub render_area_push_constant: Id,
    pub local_memory: Id,
    pub shared_memory_u8: Id,
    pub shared_memory_u16: Id,
    pub shared_memory_u32: Id,
    pub shared_memory_u64: Id,
    pub shared_memory_u32x2: Id,
    pub shared_memory_u32x4: Id,
    pub shared_memory_u32_type: Id,
    pub shared_u8: Id,
    pub shared_u16: Id,
    pub shared_u32: Id,
    pub shared_u64: Id,
    pub shared_u32x2: Id,
    pub shared_u32x4: Id,
    pub shared_store_u8_func: Id,
    pub shared_store_u16_func: Id,
    pub increment_cas_shared: Id,
    pub increment_cas_ssbo: Id,
    pub decrement_cas_shared: Id,
    pub decrement_cas_ssbo: Id,
    pub f32_add_cas: Id,
    pub f16x2_add_cas: Id,
    pub f16x2_min_cas: Id,
    pub f16x2_max_cas: Id,
    pub f32x2_add_cas: Id,
    pub f32x2_min_cas: Id,
    pub f32x2_max_cas: Id,
    pub write_storage_cas_loop_func: Id,
    pub load_global_func_u32: Id,
    pub load_global_func_u32x2: Id,
    pub load_global_func_u32x4: Id,
    pub write_global_func_u32: Id,
    pub write_global_func_u32x2: Id,
    pub write_global_func_u32x4: Id,
    pub input_position: Id,
    pub output_point_size: Id,
    pub output_position: Id,
    pub output_tess_level_outer: Id,
    pub output_tess_level_inner: Id,
    pub sample_mask: Id,
    pub frag_depth: Id,
    pub load_const_func_u8: Id,
    pub load_const_func_u16: Id,
    pub load_const_func_u32: Id,
    pub load_const_func_f32: Id,
    pub load_const_func_u32x2: Id,
    pub load_const_func_u32x4: Id,
    pub image_buffer_type: Id,
    pub image_u32: Id,
}

impl EmitContext<'_> {
    pub fn def(&mut self, ir: &crate::ir::Ir, value: crate::ir::Value) -> Result<Id> {
        use crate::ir::Value;
        Ok(match ir.resolve(value)? {
            Value::None => Id::NONE,
            Value::U1(value) => {
                if value {
                    self.constants.true_value
                } else {
                    self.constants.false_value
                }
            }
            Value::U32(value) => self.const_u32(value)?,
            Value::F32(bits) => self.builder.constant(self.types.f32[0], &[bits])?,
            Value::U64(value) => self
                .builder
                .constant(self.types.u64, &[value as u32, (value >> 32) as u32])?,
            Value::F64(bits) => self
                .builder
                .constant(self.types.f64[0], &[bits as u32, (bits >> 32) as u32])?,
            Value::Inst(id) => {
                let id = Id(ir.inst(id)?.definition);
                if id == Id::NONE {
                    return Err(crate::CompileError::new(
                        crate::diagnostics::FailureCategory::Internal,
                        "SPIR-V operand has no definition",
                    ));
                }
                id
            }
            value => {
                return Err(crate::CompileError::new(
                    crate::diagnostics::FailureCategory::Unsupported,
                    format!("Unsupported SPIR-V immediate {value:?}"),
                ));
            }
        })
    }
    pub fn bit_offset(
        &mut self,
        ir: &crate::ir::Ir,
        offset: crate::ir::Value,
        width: u32,
    ) -> Result<Id> {
        let mask = match width {
            8 => 24,
            16 => 16,
            _ => {
                return Err(crate::CompileError::new(
                    crate::diagnostics::FailureCategory::InvalidArgument,
                    "Bit offset width must be 8 or 16",
                ));
            }
        };
        if let crate::ir::Value::U32(offset) = ir.resolve(offset)? {
            return self.const_u32((offset << 3) & mask);
        }
        let offset = self.def(ir, offset)?;
        let shift = self.const_u32(3)?;
        let mask = self.const_u32(mask)?;
        let shift = self.builder.result(
            spirv::Op::ShiftLeftLogical,
            self.types.u32[0],
            &[offset.0, shift.0],
        )?;
        self.builder
            .result(spirv::Op::BitwiseAnd, self.types.u32[0], &[shift.0, mask.0])
    }
}
