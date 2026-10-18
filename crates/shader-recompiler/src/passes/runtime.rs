use crate::{RuntimeInfo, ir::Program, profile::TransformFeedbackVarying, types::*};

/// Decoded pipeline state supplied by the renderer. Transform feedback entries
/// and attribute types have already been converted from device register formats.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphicsRuntimeState {
    pub base_instance: Option<u32>,
    pub vertex_input_types: [AttributeType; 32],
    pub input_topology: InputTopology,
    pub point_size: f32,
    pub ndc_minus_one_to_one: bool,
    pub force_early_z: bool,
    pub y_negate: bool,
    pub tess_clockwise: bool,
    pub tess_primitive: TessPrimitive,
    pub tess_spacing: TessSpacing,
    pub alpha_test_func: CompareFunction,
    pub alpha_test_reference: f32,
    pub dual_source_blend: bool,
    pub use_integer_fragment_outputs: bool,
    pub color_output_types: [AttributeType; 8],
    pub xfb_enabled: bool,
    pub support_transform_feedback: bool,
    pub xfb_varyings: [TransformFeedbackVarying; 256],
    pub xfb_count: u32,
}
impl Default for GraphicsRuntimeState {
    fn default() -> Self {
        Self {
            base_instance: None,
            vertex_input_types: [AttributeType::Float; 32],
            input_topology: InputTopology::Points,
            point_size: 1.0,
            ndc_minus_one_to_one: false,
            force_early_z: false,
            y_negate: false,
            tess_clockwise: false,
            tess_primitive: TessPrimitive::Isolines,
            tess_spacing: TessSpacing::Equal,
            alpha_test_func: CompareFunction::Always,
            alpha_test_reference: 0.0,
            dual_source_blend: false,
            use_integer_fragment_outputs: false,
            color_output_types: [AttributeType::Float; 8],
            xfb_enabled: false,
            support_transform_feedback: false,
            xfb_varyings: [TransformFeedbackVarying::default(); 256],
            xfb_count: 0,
        }
    }
}
pub fn make_runtime_info(
    program: &Program,
    previous: Option<&Program>,
    has_guest_geometry: bool,
    state: &GraphicsRuntimeState,
) -> RuntimeInfo {
    let mut info = RuntimeInfo {
        input_topology: state.input_topology,
        force_early_z: state.force_early_z,
        y_negate: state.y_negate,
        ..Default::default()
    };
    if let Some(previous) = previous {
        info.previous_stage_stores = previous.info.stores.clone();
        info.previous_stage_legacy_stores_mapping = previous.info.legacy_stores_mapping.clone();
        if previous.execution.is_geometry_passthrough {
            for i in 0..8 {
                info.previous_stage_stores.mask[i] |= previous.info.passthrough.mask[i];
            }
        }
    } else {
        info.previous_stage_stores.mask = [u64::MAX; 8];
    }
    let capture = |info: &mut RuntimeInfo| {
        if state.xfb_enabled && state.support_transform_feedback {
            info.xfb_varyings = state.xfb_varyings;
            info.xfb_count = state.xfb_count;
        }
    };
    match program.stage {
        Stage::VertexB => {
            info.base_instance = state.base_instance;
            if !has_guest_geometry {
                if state.input_topology == InputTopology::Points {
                    info.fixed_state_point_size = Some(state.point_size);
                }
                capture(&mut info);
                info.convert_depth_mode = state.ndc_minus_one_to_one;
            }
            info.generic_input_types = state.vertex_input_types;
        }
        Stage::TessellationEval => {
            info.tess_clockwise = state.tess_clockwise;
            info.tess_primitive = state.tess_primitive;
            info.tess_spacing = state.tess_spacing;
        }
        Stage::Geometry => {
            if program.execution.output_topology == Some(OutputTopology::PointList) {
                info.fixed_state_point_size = Some(state.point_size);
            }
            capture(&mut info);
            info.convert_depth_mode = state.ndc_minus_one_to_one;
        }
        Stage::Fragment => {
            info.alpha_test_func = Some(state.alpha_test_func);
            info.alpha_test_reference = state.alpha_test_reference;
            info.dual_source_blend = state.dual_source_blend;
            if state.use_integer_fragment_outputs {
                info.color_output_types = state.color_output_types;
            }
        }
        _ => (),
    }
    info
}
