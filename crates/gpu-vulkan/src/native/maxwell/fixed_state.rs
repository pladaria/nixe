//! Feature-dependent packed pipeline identity, separate from current native values.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.h
use super::super::{GraphicsState, VulkanError};
use super::*;
use ash::vk;
use nixe_gpu::{FragmentDepthTransform, TessellationState};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(in crate::native) struct FixedPipelineState {
    shader: [u64; 6],
    raw1: u32,
    raw2: u32,
    colors: [u8; 8],
    alpha_test_ref: u32,
    point_size: u32,
    attribute_types: u128,
    // Unsupported dynamic groups alone contribute their static values.
    dynamic_state: [u32; 2],
    blending: Option<[u32; 8]>,
    vertex_input: VertexInputKey,
    // Required shader dependency for hosts restricted to normalized depth.
    fragment_depth: [u32; 4],
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum VertexInputKey {
    Dynamic,
    Static(Box<StaticVertexInput>),
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StaticVertexInput {
    attributes: [u32; 32],
    divisors: [u32; 32],
    instanced: u32,
    strides: Option<[u16; 32]>,
}
impl crate::native::pipeline_cache::PipelineContext {
    pub(in crate::native) fn refresh_fixed_pipeline_state(
        &mut self,
        profile: &crate::DeviceProfile,
        shaders: &super::GraphicsShaders,
        registers: &MaxwellThreeDState,
        draw: &MaxwellThreeDDrawState,
    ) -> Result<(), VulkanError> {
        let shader = shaders.hashes;
        let pixel_depth = shaders.pixel_depth;
        let flags = registers.graphics_dirty();
        if self.pixel_depth != Some(pixel_depth) {
            flags.mark(GraphicsDirty::VIEWPORT);
            self.pixel_depth = Some(pixel_depth);
        }
        let first = self.pipeline_state.is_none();
        let mut dirty = flags.bits(GraphicsDirty::PARAMETERS);
        if first {
            dirty |= GraphicsDirty::PARAMETERS;
        }
        let r = registers.register_words();
        registers::validate_draw(registers, draw)?;
        let f = &profile.dynamic_state;
        let topology = if f.base && f.base2 {
            match draw.topology {
                5..=9 => 4,
                11 => 10,
                13 => 12,
                value => value,
            }
        } else {
            draw.topology
        };
        // The supported-state contract is checked when its actual inputs
        // change. Shader/pipeline switches do not invalidate those inputs.
        // Native dynamic groups consume their own flags, as StateTracker.
        let validation = GraphicsDirty::PIPELINE_STATE
            | GraphicsDirty::TARGET_FORMAT
            | GraphicsDirty::RENDER_TARGET_CONTROL
            | GraphicsDirty::FRONT_FACE
            | GraphicsDirty::CULL_MODE
            | GraphicsDirty::LINE_WIDTH
            | GraphicsDirty::DEPTH_BIAS_ENABLE
            | GraphicsDirty::RASTERIZER_DISCARD;
        if first
            || flags.is_dirty(validation)
            || self
                .pipeline_state
                .as_ref()
                .is_some_and(|s| s.topology() != topology)
        {
            registers::validate_graphics(registers, draw)?;
        }
        // Pipeline identity reads format words only. The texture cache consumes
        // actual resource addresses/layouts and owns framebuffer dimensions.
        let mut colors = [0; 8];
        for (index, format) in colors.iter_mut().enumerate() {
            let raw = r[0x0810 / 4 + index * 0x10];
            *format = raw as u8;
        }
        let depth_enabled = r[0x1538 / 4] != 0;
        let depth_code = r[0x0fe8 / 4];
        let samples = r[0x15d0 / 4];
        let mut blend_state = None;
        if !profile.dynamic_state.blending() && dirty & GraphicsDirty::COLOR_OUTPUT != 0 {
            let mut slots = [0; 8];
            for (index, slot) in slots.iter_mut().enumerate() {
                *slot = blending::pack_attachment(r, index)?;
            }
            blend_state = Some(slots);
        }
        let fragment_depth_transform = if r[0x192c / 4] == 1 {
            registers::depth_range(r, (depth_enabled && r[0x12cc / 4] != 0) || pixel_depth)?.1
        } else {
            None
        };
        let tessellation = if draw.topology == 14 {
            nixe_gpu_maxwell::engines::threed::tessellation::patch_state(registers)
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?
        } else {
            if shaders.tessellation_enabled {
                return Err(registers::unsupported(
                    0x1618,
                    r[0x1618 / 4],
                    "tessellation stage enabled for a non-patch primitive",
                ));
            }
            None
        };
        let polygon = registers::pack_polygon(r)?;
        let mut raw1 = u32::from(f.base)
            | (u32::from(f.base2) << 1)
            | (u32::from(f.blending()) << 3)
            | (u32::from(f.vertex_input) << 5)
            | (u32::from(r[0x192c / 4] == 1 && r[0x0d7c / 4] == 0) << 7)
            | (polygon << 8)
            | (topology << 24)
            | (samples << 28);
        let alpha_func = if r[0x12ec / 4] == 1 {
            registers::pack_comparison(0x1314, r[0x1314 / 4])?
        } else {
            7
        };
        let mut raw2 = (alpha_func << 1)
            | (u32::from(r[0x0210 / 4] == 1) << 4)
            | (u32::from(depth_enabled) << 5)
            | ((depth_code & 31) << 6)
            | ((r[0x13ac / 4] & 1) << 11)
            | (u32::from(r[0x1684 / 4] != 0) << 12)
            | (u32::from(r[0x1570 / 4] != 0) << 14)
            | (u32::from(depth_enabled && matches!(depth_code, 0x14 | 0x16 | 0x19)) << 18)
            | (u32::from(fragment_depth_transform.is_some()) << 19);
        if let Some(t) = tessellation {
            raw1 |= ((t.mode.domain as u32) << 10)
                | ((t.mode.spacing as u32) << 12)
                | (u32::from(matches!(
                    t.mode.output,
                    TessellationOutput::Triangles(TessellationWinding::Clockwise)
                )) << 14)
                | (u32::from(t.input_control_points - 1) << 15);
            let output = match t.mode.output {
                TessellationOutput::Points => 0,
                TessellationOutput::Lines => 1,
                TessellationOutput::Triangles(_) => 2,
            };
            raw2 |= (u32::from(matches!(
                t.control,
                TessellationControl::DefaultLevels { .. }
            )) << 20)
                | (output << 21);
        }
        let mut dynamic_state = [0; 2];
        if !f.base {
            let stencil = registers::boolean(r, 0x1380)?;
            let faces = if stencil {
                registers::stencil_faces(r)?
            } else {
                [0; 2]
            };
            let depth_test = registers::boolean(r, 0x12cc)?;
            let depth_write = registers::boolean(r, 0x12e8)?;
            let compare = if depth_test {
                registers::pack_comparison(0x130c, r[0x130c / 4])?
            } else {
                7
            };
            dynamic_state[0] =
                registers::pack_cull_face(r)? | (u32::from(registers::boolean(r, 0x1918)?) << 2);
            dynamic_state[1] = u32::from(faces[0])
                | (u32::from(faces[1]) << 12)
                | (u32::from(stencil) << 24)
                | (u32::from(depth_write) << 25)
                | (u32::from(depth_test) << 27)
                | (registers::pack_front_face(r)? << 28)
                | (compare << 29);
        }
        if !f.base2 {
            let native_topology = registers::topology(draw.topology)?;
            dynamic_state[0] |= (u32::from(
                registers::boolean(r, 0x1644)?
                    && crate::cache::restart_supported(native_topology, profile.primitive_restart),
            ) << 3)
                | (u32::from(registers::depth_bias_enabled(r, draw.topology)?) << 4)
                | (u32::from(registers::boolean(r, 0x037c)?) << 5);
        }
        let fixed = self
            .pipeline_state
            .get_or_insert_with(|| FixedPipelineState {
                shader,
                raw1: 0,
                raw2: 0,
                colors: [0; 8],
                alpha_test_ref: 0,
                point_size: 0,
                attribute_types: 0,
                dynamic_state: [0; 2],
                blending: if f.blending() { None } else { Some([0; 8]) },
                vertex_input: if f.vertex_input {
                    VertexInputKey::Dynamic
                } else {
                    VertexInputKey::Static(Box::new(StaticVertexInput {
                        attributes: [0; 32],
                        divisors: [0; 32],
                        instanced: 0,
                        strides: (!f.base).then_some([0; 32]),
                    }))
                },
                fragment_depth: [0; 4],
            });
        fixed.shader = shader;
        fixed.raw1 = raw1;
        fixed.raw2 = raw2;
        fixed.dynamic_state = dynamic_state;
        fixed.alpha_test_ref = r[0x1310 / 4];
        fixed.point_size = r[0x1518 / 4];
        fixed.fragment_depth =
            fragment_depth_transform.map_or([0; 4], |t| t.coefficients().map(f32::to_bits));
        fixed.colors = colors;
        if let Some(blending) = blend_state {
            fixed.blending = Some(blending);
        }
        if dirty & GraphicsDirty::VERTEX_INPUT != 0 {
            fixed.attribute_types = super::vertex_input::attribute_types(registers)?;
            if let VertexInputKey::Static(input) = &mut fixed.vertex_input {
                input.refresh(r);
                flags.take(GraphicsDirty::VERTEX_INPUT);
            }
        }
        if let VertexInputKey::Static(input) = &mut fixed.vertex_input
            && let Some(strides) = &mut input.strides
        {
            for (binding, stride) in strides.iter_mut().enumerate() {
                *stride = (r[0x1c00 / 4 + binding * 4] & 0xfff) as u16;
            }
        }
        Ok(())
    }
}
impl StaticVertexInput {
    fn refresh(&mut self, r: &[u32]) {
        // Keep the actual guest fields in the fixed key. Native descriptions
        // and format support are consumed once the shader is compiled.
        for (attribute, &word) in self
            .attributes
            .iter_mut()
            .zip(&r[0x1160 / 4..0x1160 / 4 + 32])
        {
            *attribute = word & !((1 << 5) | (1 << 30));
        }
        self.instanced = 0;
        for (binding, divisor) in self.divisors.iter_mut().enumerate() {
            let instanced = r[0x1880 / 4 + binding] != 0;
            self.instanced |= u32::from(instanced) << binding;
            *divisor = if instanced {
                r[0x1c00 / 4 + binding * 4 + 3]
            } else {
                0
            };
        }
    }
}

impl FixedPipelineState {
    /// Disk pipeline state must select the same native feature paths as the
    /// current device, as Eden PipelineCache::LoadDiskResources checks.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp#L670-L679
    pub(in crate::native) fn matches_profile(&self, profile: &crate::DeviceProfile) -> bool {
        const DYNAMIC_FEATURES: u32 = 1 | (1 << 1) | (1 << 3) | (1 << 5);
        let state = &profile.dynamic_state;
        let features = u32::from(state.base)
            | (u32::from(state.base2) << 1)
            | (u32::from(state.blending()) << 3)
            | (u32::from(state.vertex_input) << 5);
        self.raw1 & DYNAMIC_FEATURES == features
    }

    pub(in crate::native) fn environment_key(&self) -> Vec<u8> {
        // Version three packs guest raster/blending fields. Superseded keys
        // used host blend values and combined raster interpretations.
        let mut bytes = vec![3];
        for hash in self.shader {
            bytes.extend_from_slice(&hash.to_le_bytes());
        }
        for word in [self.raw1, self.raw2, self.alpha_test_ref, self.point_size] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&self.colors);
        bytes.extend_from_slice(&self.attribute_types.to_le_bytes());
        for word in self.dynamic_state {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.push(u8::from(self.blending.is_some()));
        if let Some(blending) = self.blending {
            for word in blending {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes.push(u8::from(matches!(
            self.vertex_input,
            VertexInputKey::Static(_)
        )));
        if let VertexInputKey::Static(input) = &self.vertex_input {
            bytes.extend_from_slice(&input.instanced.to_le_bytes());
            for word in input.attributes.iter().chain(&input.divisors) {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            bytes.push(u8::from(input.strides.is_some()));
            if let Some(strides) = input.strides {
                for stride in strides {
                    bytes.extend_from_slice(&stride.to_le_bytes());
                }
            }
        }
        for word in self.fragment_depth {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
    pub(in crate::native) fn from_environment_key(mut bytes: &[u8]) -> Option<Self> {
        fn take<const N: usize>(bytes: &mut &[u8]) -> Option<[u8; N]> {
            let value = bytes.get(..N)?.try_into().ok()?;
            *bytes = &bytes[N..];
            Some(value)
        }
        if take::<1>(&mut bytes)? != [3] {
            return None;
        }
        let mut shader = [0; 6];
        for value in &mut shader {
            *value = u64::from_le_bytes(take(&mut bytes)?);
        }
        let raw1 = u32::from_le_bytes(take(&mut bytes)?);
        let raw2 = u32::from_le_bytes(take(&mut bytes)?);
        let alpha_test_ref = u32::from_le_bytes(take(&mut bytes)?);
        let point_size = u32::from_le_bytes(take(&mut bytes)?);
        let colors = take(&mut bytes)?;
        let attribute_types = u128::from_le_bytes(take(&mut bytes)?);
        let mut dynamic_state = [0; 2];
        for value in &mut dynamic_state {
            *value = u32::from_le_bytes(take(&mut bytes)?);
        }
        let blending = match take::<1>(&mut bytes)?[0] {
            0 => None,
            1 => {
                let mut values = [0; 8];
                for value in &mut values {
                    *value = u32::from_le_bytes(take(&mut bytes)?);
                }
                Some(values)
            }
            _ => return None,
        };
        let vertex_input = match take::<1>(&mut bytes)?[0] {
            0 => VertexInputKey::Dynamic,
            1 => {
                let mut input = Box::new(StaticVertexInput {
                    attributes: [0; 32],
                    divisors: [0; 32],
                    instanced: u32::from_le_bytes(take(&mut bytes)?),
                    strides: None,
                });
                for value in input.attributes.iter_mut().chain(&mut input.divisors) {
                    *value = u32::from_le_bytes(take(&mut bytes)?);
                }
                match take::<1>(&mut bytes)?[0] {
                    0 => (),
                    1 => {
                        let mut strides = [0; 32];
                        for value in &mut strides {
                            *value = u16::from_le_bytes(take(&mut bytes)?);
                        }
                        input.strides = Some(strides);
                    }
                    _ => return None,
                }
                VertexInputKey::Static(input)
            }
            _ => return None,
        };
        if matches!(vertex_input, VertexInputKey::Dynamic) != (raw1 & (1 << 5) != 0)
            || matches!(&vertex_input, VertexInputKey::Static(input) if input.strides.is_some() != (raw1 & 1 == 0))
        {
            return None;
        }
        let mut fragment_depth = [0; 4];
        for value in &mut fragment_depth {
            *value = u32::from_le_bytes(take(&mut bytes)?);
        }
        bytes.is_empty().then_some(Self {
            shader,
            raw1,
            raw2,
            colors,
            alpha_test_ref,
            point_size,
            attribute_types,
            dynamic_state,
            blending,
            vertex_input,
            fragment_depth,
        })
    }
    fn topology(&self) -> u32 {
        (self.raw1 >> 24) & 15
    }
    fn color_count(&self) -> usize {
        self.colors
            .iter()
            .rposition(|&format| format != 0)
            .map_or(0, |i| i + 1)
    }
    fn depth_format(&self) -> Result<vk::Format, VulkanError> {
        if self.raw2 & (1 << 5) == 0 {
            Ok(vk::Format::UNDEFINED)
        } else {
            registers::depth_attachment_format((self.raw2 >> 6) & 31)
        }
    }
    /// The actual native owner consumes the selected formats independently of
    /// shader code. Cold pipeline construction uses the same raw state.
    pub(in crate::native) fn update_attachments(
        &self,
        state: &mut GraphicsState,
    ) -> Result<(), VulkanError> {
        state.colors = self.colors[..self.color_count()]
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                Ok((
                    registers::color_attachment_format(f, i)?,
                    self.blending.map_or_else(
                        || Ok(super::super::BlendState::opaque()),
                        |b| blending::unpack(b[i]),
                    )?,
                ))
            })
            .collect::<Result<_, VulkanError>>()?;
        state.depth_format = self.depth_format()?;
        state.stencil_format = if self.raw2 & (1 << 18) != 0 {
            state.depth_format
        } else {
            vk::Format::UNDEFINED
        };
        state.samples = registers::attachment_samples(self.raw1 >> 28)?;
        Ok(())
    }
    pub(in crate::native) fn tessellation(&self) -> Option<TessellationState> {
        if self.topology() != 14 {
            return None;
        }
        Some(TessellationState {
            mode: TessellationMode {
                domain: match (self.raw1 >> 10) & 3 {
                    0 => TessellationDomain::Isolines,
                    1 => TessellationDomain::Triangles,
                    2 => TessellationDomain::Quads,
                    _ => unreachable!("packed tessellation domain"),
                },
                spacing: match (self.raw1 >> 12) & 3 {
                    0 => TessellationSpacing::Equal,
                    1 => TessellationSpacing::FractionalOdd,
                    2 => TessellationSpacing::FractionalEven,
                    _ => unreachable!("packed tessellation spacing"),
                },
                output: match (self.raw2 >> 21) & 3 {
                    0 => TessellationOutput::Points,
                    1 => TessellationOutput::Lines,
                    2 => TessellationOutput::Triangles(if self.raw1 & (1 << 14) != 0 {
                        TessellationWinding::Clockwise
                    } else {
                        TessellationWinding::CounterClockwise
                    }),
                    _ => unreachable!("packed tessellation output"),
                },
            },
            input_control_points: (((self.raw1 >> 15) & 31) + 1) as u8,
            control: if self.raw2 & (1 << 20) != 0 {
                TessellationControl::DefaultLevels {
                    outer: [0; 4],
                    inner: [0; 2],
                }
            } else {
                TessellationControl::Shader
            },
        })
    }
    pub(in crate::native) fn fragment_depth_transform(&self) -> Option<FragmentDepthTransform> {
        if self.raw2 & (1 << 19) == 0 {
            return None;
        }
        let [scale, offset, min, max] = self.fragment_depth.map(f32::from_bits);
        Some(
            FragmentDepthTransform::new(scale, offset, min, max)
                .expect("validated packed depth transform"),
        )
    }
    pub(in crate::native) fn graphics(
        &self,
        profile: &crate::DeviceProfile,
        vertex_attributes: u32,
    ) -> Result<GraphicsState, VulkanError> {
        let mut state = GraphicsState::color(vk::Format::UNDEFINED);
        self.update_attachments(&mut state)?;
        state.topology = registers::topology(self.topology())?;
        state.primitive_restart = self.dynamic_state[0] & (1 << 3) != 0;
        state.raster_discard = self.raw1 & (1 << 1) == 0 && self.dynamic_state[0] & (1 << 5) == 0;
        state.depth_bias = self.dynamic_state[0] & (1 << 4) != 0;
        state.patch_points = self
            .tessellation()
            .map_or(0, |t| u32::from(t.input_control_points));
        state.provoking = if self.raw2 & (1 << 12) == 0 {
            vk::ProvokingVertexModeEXT::FIRST_VERTEX
        } else {
            vk::ProvokingVertexModeEXT::LAST_VERTEX
        };
        state.cull = if self.dynamic_state[0] & 4 == 0 {
            vk::CullModeFlags::NONE
        } else {
            match self.dynamic_state[0] & 3 {
                0 => vk::CullModeFlags::FRONT,
                1 => vk::CullModeFlags::BACK,
                2 => vk::CullModeFlags::FRONT_AND_BACK,
                _ => return Err(VulkanError::InvalidResource("invalid packed cull face")),
            }
        };
        state.front_face = registers::unpack_front_face((self.dynamic_state[1] >> 28) & 1);
        state.polygon = match (self.raw1 >> 8) & 3 {
            0 => vk::PolygonMode::POINT,
            1 => vk::PolygonMode::LINE,
            2 => vk::PolygonMode::FILL,
            3 => vk::PolygonMode::FILL_RECTANGLE_NV,
            _ => unreachable!(),
        };
        state.line_mode = if self.raw2 & (1 << 14) != 0 && profile.smooth_lines {
            vk::LineRasterizationModeEXT::RECTANGULAR_SMOOTH
        } else {
            vk::LineRasterizationModeEXT::RECTANGULAR
        };
        state.depth_test = self.dynamic_state[1] & (1 << 27) != 0;
        state.depth_write = self.dynamic_state[1] & (1 << 25) != 0;
        state.depth_compare = vk::CompareOp::from_raw((self.dynamic_state[1] >> 29) as i32);
        state.stencil_test = self.dynamic_state[1] & (1 << 24) != 0;
        state.front = registers::stencil_face((self.dynamic_state[1] & 0xfff) as u16);
        state.back = registers::stencil_face(((self.dynamic_state[1] >> 12) & 0xfff) as u16);
        if let VertexInputKey::Static(input) = &self.vertex_input {
            let count = profile.limits.max_vertex_input_bindings.min(32);
            state.vertex_bindings = (0..count)
                .map(|binding| super::super::VertexBinding {
                    binding,
                    stride: input.strides.map_or(0, |s| u32::from(s[binding as usize])),
                    rate: if input.instanced & (1 << binding) != 0 {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    },
                })
                .collect();
            state.vertex_divisors = (0..count)
                .filter(|&binding| {
                    input.instanced & (1 << binding) != 0 && input.divisors[binding as usize] != 1
                })
                .map(|binding| (binding, input.divisors[binding as usize]))
                .collect();
            state.vertex_attributes = input
                .attributes
                .iter()
                .enumerate()
                .filter(|&(location, &word)| {
                    word & 64 == 0 && vertex_attributes & (1 << location) != 0
                })
                .map(|(location, &word)| {
                    Ok(super::super::VertexAttribute {
                        location: location as u32,
                        binding: word & 31,
                        format: super::vertex_input::native_format(word, location as u32)?,
                        offset: (word >> 7) & 0x3fff,
                    })
                })
                .collect::<Result<_, VulkanError>>()?;
        }
        Ok(state)
    }
    pub(in crate::native) fn runtime(
        &self,
    ) -> Result<nixe_shader_recompiler::passes::GraphicsRuntimeState, VulkanError> {
        use nixe_shader_recompiler::{
            passes::GraphicsRuntimeState,
            types::{
                AttributeType as A, CompareFunction as C, InputTopology as I, TessPrimitive as P,
                TessSpacing as S,
            },
        };
        let mut runtime = GraphicsRuntimeState {
            vertex_input_types: [A::Disabled; 32],
            input_topology: match self.topology() {
                0 => I::Points,
                1 | 3 => I::Lines,
                _ => I::Triangles,
            },
            ndc_minus_one_to_one: self.raw1 & (1 << 7) != 0,
            y_negate: self.raw2 & (1 << 11) != 0,
            force_early_z: self.raw2 & (1 << 4) != 0,
            ..Default::default()
        };
        if self.topology() == 0 {
            runtime.point_size = f32::from_bits(self.point_size);
            if !runtime.point_size.is_finite() || runtime.point_size <= 0. {
                return Err(VulkanError::InvalidResource(
                    "point rasterization requires a finite positive programmed size",
                ));
            }
        }
        for location in 0..32 {
            runtime.vertex_input_types[location] =
                match (self.attribute_types >> (location * 3)) & 7 {
                    0 => A::Disabled,
                    1 => A::Float,
                    2 => A::SignedInt,
                    3 => A::UnsignedInt,
                    4 => A::SignedScaled,
                    5 => A::UnsignedScaled,
                    _ => unreachable!("packed vertex type"),
                };
        }
        runtime.alpha_test_reference = f32::from_bits(self.alpha_test_ref);
        runtime.alpha_test_func = match (self.raw2 >> 1) & 7 {
            0 => C::Never,
            1 => C::Less,
            2 => C::Equal,
            3 => C::LessThanEqual,
            4 => C::Greater,
            5 => C::NotEqual,
            6 => C::GreaterThanEqual,
            7 => C::Always,
            _ => unreachable!(),
        };
        if let Some(t) = self.tessellation() {
            runtime.tess_primitive = match t.mode.domain {
                TessellationDomain::Isolines => P::Isolines,
                TessellationDomain::Triangles => P::Triangles,
                TessellationDomain::Quads => P::Quads,
            };
            runtime.tess_spacing = match t.mode.spacing {
                TessellationSpacing::Equal => S::Equal,
                TessellationSpacing::FractionalOdd => S::FractionalOdd,
                TessellationSpacing::FractionalEven => S::FractionalEven,
            };
            runtime.tess_clockwise = matches!(
                t.mode.output,
                TessellationOutput::Triangles(TessellationWinding::Clockwise)
            );
        }
        Ok(runtime)
    }
}
