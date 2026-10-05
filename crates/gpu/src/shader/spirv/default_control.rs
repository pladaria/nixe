//! Native semantic adapter for a patch pipeline with no guest control shader.
//! The patch cardinality and per-vertex data pass through unchanged; programmed
//! levels are dynamic, not shader constants. This is not a substitute for a TCS.
//!
//! OpenGL 4.6 sections 11.2 and 11.2.3 describe this patch behavior:
//! https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf
//! Nouveau programs the six native default-level registers in the same order:
//! https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-24.3.0/src/gallium/drivers/nouveau/nvc0/nvc0_state_validate.c
//! Its no-TCS path disables SP_SELECT(2), rather than changing patch cardinality:
//! https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-24.3.0/src/gallium/drivers/nouveau/nvc0/nvc0_shader_state.c#L180-L204
//! Vulkan requires an explicit TCS, whose outputs carry those levels:
//! https://docs.vulkan.org/spec/latest/chapters/tessellation.html

use super::*;
use crate::TessellationControl;
use rspirv::dr::Operand;

/// Structural specialization only. Level values and their defined mask MUST NOT
/// enter shader/pipeline cache keys. The caller reserves 24 push-constant bytes
/// visible to the control stage; no upload buffer or descriptor is needed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpirvDefaultControlOptions {
    pub input_control_points: u8,
    pub domain: TessellationDomain,
    pub push_constant_offset: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpirvDefaultControlShader {
    module: SpirvShaderModule,
    required_levels: u8,
}

impl SpirvDefaultControlShader {
    pub const PARAMETER_BYTES: u32 = 24;

    pub fn module(&self) -> &SpirvShaderModule {
        &self.module
    }

    /// O(1) draw-time validation and raw-bit parameter extraction. Neither a
    /// shader interface scan nor recompilation is needed when levels change.
    pub fn parameters(&self, control: TessellationControl) -> Result<[u32; 6]> {
        parameters(self.required_levels, control)
    }

    pub const fn required_levels(&self) -> u8 {
        self.required_levels
    }

    pub(super) fn into_parts(self) -> (SpirvShaderModule, u8) {
        (self.module, self.required_levels)
    }
}

pub(super) fn parameters(required_levels: u8, control: TessellationControl) -> Result<[u32; 6]> {
    let TessellationControl::DefaultLevels {
        outer,
        inner,
        defined,
    } = control
    else {
        return Err(SpirvShaderError::Options(
            "default control adapter cannot replace an active guest control shader",
        ));
    };
    if defined & required_levels != required_levels {
        return Err(SpirvShaderError::DefaultTessellationLevels {
            required: required_levels,
            defined,
        });
    }
    Ok([outer[0], outer[1], outer[2], outer[3], inner[0], inner[1]])
}

/// Compile on a native shader-cache miss. Interfaces come from the linked VS/TES;
/// no guest binary identity, level values or guessed missing attributes are used.
pub fn lower_default_tessellation_control_to_spirv(
    vertex: &VerifiedShaderIr,
    evaluation: &VerifiedShaderIr,
    options: SpirvDefaultControlOptions,
) -> Result<SpirvDefaultControlShader> {
    if vertex.ir().stage() != ShaderStage::Vertex
        || evaluation.ir().stage() != ShaderStage::TessellationEvaluation
    {
        return Err(SpirvShaderError::Options(
            "default control adapter requires vertex and evaluation programs",
        ));
    }
    if options.input_control_points == 0
        || !options.push_constant_offset.is_multiple_of(4)
        || options
            .push_constant_offset
            .checked_add(SpirvDefaultControlShader::PARAMETER_BYTES)
            .is_none()
    {
        return Err(SpirvShaderError::Options(
            "default control adapter requires nonzero patch size and an aligned, non-overflowing push-constant range",
        ));
    }
    let mut required_levels = match options.domain {
        TessellationDomain::Isolines => 0b00_0011,
        TessellationDomain::Triangles => 0b01_0111,
        TessellationDomain::Quads => 0b11_1111,
    };
    let producer: BTreeMap<_, _> = vertex
        .ir()
        .outputs()
        .iter()
        .map(|e| ((e.location(), e.component()), e.scalar_type()))
        .collect();
    let mut forwarded = BTreeMap::new();
    for element in evaluation.ir().inputs() {
        let location = element.location();
        if tessellation::per_vertex(location) {
            let key = (location, element.component());
            if producer.get(&key) != Some(&element.scalar_type()) {
                return Err(SpirvShaderError::Interface(location));
            }
            forwarded.insert(
                key,
                ShaderInterfaceElement {
                    location,
                    component: element.component(),
                    scalar_type: element.scalar_type(),
                    interpolation: None,
                },
            );
        } else {
            match location {
                ShaderIoLocation::TessLevelOuter => required_levels |= 1 << element.component(),
                ShaderIoLocation::TessLevelInner => {
                    required_levels |= 1 << (4 + element.component())
                }
                ShaderIoLocation::TessCoord
                | ShaderIoLocation::PrimitiveId
                | ShaderIoLocation::PatchVertices => {}
                // No vertex output can supply user per-patch data without a TCS.
                _ => return Err(SpirvShaderError::Interface(location)),
            }
        }
    }
    let mut inputs: Vec<_> = forwarded.values().copied().collect();
    let mut outputs = inputs.clone();
    inputs.push(ShaderInterfaceElement {
        location: ShaderIoLocation::InvocationId,
        component: 0,
        scalar_type: ShaderScalarType::Unsigned32,
        interpolation: None,
    });
    for lane in 0..6 {
        if required_levels & (1 << lane) != 0 {
            let (location, component) = level_lane(lane);
            outputs.push(ShaderInterfaceElement {
                location,
                component,
                scalar_type: ShaderScalarType::Float32,
                interpolation: None,
            });
        }
    }
    let ir = ShaderIr::new(
        ShaderStage::TessellationControl,
        inputs,
        outputs,
        vec![],
        vec![],
    )
    .with_tessellation_control_points(Some(u32::from(options.input_control_points)));
    let mut e = Emitter::new(SpirvShaderOptions {
        depth_clip_negative_one_to_one: false,
        input_control_points: u32::from(options.input_control_points),
        tessellation_mode: None,
        float32: Default::default(),
        float64: Default::default(),
    });
    e.emit_interfaces(&ir)?;
    let block = e.b.type_struct([e.uint; 6]);
    e.b.decorate(block, spv::Decoration::Block, []);
    for lane in 0..6 {
        e.b.member_decorate(
            block,
            lane,
            spv::Decoration::Offset,
            [Operand::LiteralBit32(
                options.push_constant_offset + 4 * lane,
            )],
        );
    }
    let ptr =
        e.b.type_pointer(None, spv::StorageClass::PushConstant, block);
    let parameters =
        e.b.variable(ptr, None, spv::StorageClass::PushConstant, None);
    let word_ptr =
        e.b.type_pointer(None, spv::StorageClass::PushConstant, e.uint);
    let void = e.b.type_void();
    let function = e.b.type_function(void, []);
    e.entry =
        e.b.begin_function(void, None, spv::FunctionControl::NONE, function)?;
    e.configure_stage(&ir)?;
    e.block = e.b.begin_block(None)?;
    let invocation = e.load_interface(true, ShaderIoLocation::InvocationId, 0, None)?;
    for &(location, component) in forwarded.keys() {
        let bits = e.load_interface(true, location, component, Some(invocation))?;
        e.store_interface(location, component, Some(invocation), bits)?;
    }
    let zero = e.constant(0);
    let first = e.b.i_equal(e.boolean, None, invocation, zero)?;
    let write_levels = e.b.id();
    let merge = e.b.id();
    e.b.selection_merge(merge, spv::SelectionControl::NONE)?;
    e.b.branch_conditional(first, write_levels, merge, [])?;
    e.b.begin_block(Some(write_levels))?;
    for lane in 0..6 {
        if required_levels & (1 << lane) == 0 {
            continue;
        }
        let index = e.constant(u32::from(lane));
        let pointer = e.b.access_chain(word_ptr, None, parameters, [index])?;
        let bits = e.b.load(e.uint, None, pointer, None, [])?;
        let (location, component) = level_lane(lane);
        e.store_interface(location, component, None, bits)?;
    }
    // There are no cross-invocation output reads. Stage completion suffices;
    // an extra patch-wide rendezvous would add work without providing semantics.
    e.b.branch(merge)?;
    e.b.begin_block(Some(merge))?;
    e.b.ret()?;
    e.b.end_function()?;
    Ok(SpirvDefaultControlShader {
        module: SpirvShaderModule {
            words: e.b.module().assemble().into_boxed_slice(),
            bindings: Box::new([]),
        },
        required_levels,
    })
}

fn level_lane(lane: u8) -> (ShaderIoLocation, u8) {
    if lane < 4 {
        (ShaderIoLocation::TessLevelOuter, lane)
    } else {
        (ShaderIoLocation::TessLevelInner, lane - 4)
    }
}
