//! Feature-dependent packed pipeline identity, separate from current native values.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.h
use super::super::{GraphicsState, VulkanError};
use super::*;
use ash::vk;
use nixe_gpu::TessellationState;

use super::vertex_input::Attribute;
use std::hash::{Hash, Hasher};

/// Integer-only packed state. Its feature-dependent prefix is both the equality
/// representation and CityHash input, as FixedPipelineState::Size/Hash in Eden.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::native) struct FixedPipelineState {
    pub(in crate::native) shader: [u64; 6],
    raw1: u32,
    raw2: u32,
    colors: [u8; 8],
    driver_id: u32,
    driver_version: u32,
    alpha_test_ref: u32,
    point_size: u32,
    viewport_swizzles: [u16; 16],
    // Dynamic input classes or the static enabled-divisor mask.
    attribute_types: u64,
    dynamic_state: [u32; 2],
    blending: [u32; 8],
    attributes: [Attribute; 32],
    divisors: [u32; 32],
    strides: [u16; 32],
}
// No padding may be observed by the byte comparisons or hashing below.
const _: () = assert!(std::mem::size_of::<FixedPipelineState>() == 480);
impl FixedPipelineState {
    pub(in crate::native) fn dynamic_vertex_input(&self) -> bool {
        self.raw1 & (1 << 5) != 0
    }

    pub(in crate::native) fn same_vertex_types(
        &self,
        registers: &MaxwellThreeDState,
    ) -> Result<bool, VulkanError> {
        Ok(self.attribute_types == super::vertex_input::attribute_types(registers)?)
    }

    fn key_bytes(&self) -> &[u8] {
        let size = if self.raw1 & (1 << 5) != 0 {
            if self.raw1 & (1 << 3) != 0 {
                std::mem::offset_of!(Self, blending)
            } else {
                std::mem::offset_of!(Self, attributes)
            }
        } else if self.raw1 & 1 != 0 {
            std::mem::offset_of!(Self, strides)
        } else {
            std::mem::size_of::<Self>()
        };
        // SAFETY: repr(C), the size assertion proves the sum of integer field
        // sizes has no padding, every field is initialized, and self is borrowed.
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(self).cast(), size) }
    }
}
impl PartialEq for FixedPipelineState {
    fn eq(&self, other: &Self) -> bool {
        self.key_bytes() == other.key_bytes()
    }
}
impl Eq for FixedPipelineState {}
impl Hash for FixedPipelineState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(nixe_gpu::shader_identity::cityhash::city_hash64(
            self.key_bytes(),
        ));
    }
}
impl crate::native::pipeline_cache::PipelineCache {
    pub(in crate::native) fn refresh_fixed_pipeline_state(
        &mut self,
        profile: &crate::DeviceProfile,
        shaders: &super::GraphicsShaders,
        registers: &MaxwellThreeDState,
        draw: &MaxwellThreeDDrawState,
    ) -> Result<(), VulkanError> {
        let shader = shaders.hashes;
        let flags = registers.graphics_dirty();
        let first = self.pipeline_state.is_none();
        let mut dirty = flags.bits(GraphicsDirty::PARAMETERS);
        if first {
            dirty |= GraphicsDirty::PARAMETERS;
        }
        let r = registers.register_words();
        registers::validate_draw(registers)?;
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
            | (u32::from(f.depth_clamp_enable) << 2)
            | (u32::from(f.blending()) << 3)
            | (u32::from(f.vertex_input) << 5)
            | (u32::from(r[0x0d7c / 4] == 0) << 7)
            | (polygon << 8)
            | ((r[0x1618 / 4] & 3) << 10)
            | (((r[0x1618 / 4] >> 4) & 3) << 12)
            | (u32::from((r[0x1618 / 4] >> 8) & 3 == 2) << 14)
            | (((r[0x0dcc / 4].wrapping_sub(1)) & 31) << 15)
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
            | (u32::from(depth_enabled && matches!(depth_code, 0x14 | 0x16 | 0x19)) << 18);
        if !f.depth_clamp_enable {
            raw2 |= u32::from(registers::depth_clamp_enabled(r)?);
        }
        if let Some(t) = tessellation {
            raw1 &= !(0x3ff << 10);
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
            .get_or_insert_with(FixedPipelineState::default);
        fixed.shader = shader;
        fixed.raw1 = raw1;
        fixed.raw2 = raw2;
        fixed.dynamic_state = dynamic_state;
        fixed.driver_id = profile.driver_id.as_raw() as u32;
        fixed.driver_version = profile.driver_version;
        fixed.alpha_test_ref = r[0x1310 / 4];
        fixed.point_size = r[0x1518 / 4];
        fixed.colors = colors;
        if let Some(blending) = blend_state {
            fixed.blending = blending;
        }
        if flags.take(GraphicsDirty::VIEWPORT_SWIZZLES) != 0 || first {
            for (index, swizzle) in fixed.viewport_swizzles.iter_mut().enumerate() {
                *swizzle = r[0x0a18 / 4 + index * 8] as u16;
            }
        }
        if dirty & GraphicsDirty::VERTEX_INPUT != 0 {
            if f.vertex_input {
                fixed.attribute_types = super::vertex_input::attribute_types(registers)?;
            } else {
                fixed.attribute_types = 0;
                for (location, attribute) in fixed.attributes.iter_mut().enumerate() {
                    let word = r[0x1160 / 4 + location];
                    super::vertex_input::reject_bgra(word, location as u32)?;
                    *attribute = Attribute::from_guest(word);
                }
                for (binding, divisor) in fixed.divisors.iter_mut().enumerate() {
                    let instanced = r[0x1880 / 4 + binding] != 0;
                    fixed.attribute_types |= u64::from(instanced) << binding;
                    *divisor = if instanced {
                        r[0x1c00 / 4 + binding * 4 + 3]
                    } else {
                        0
                    };
                }
                flags.take(GraphicsDirty::VERTEX_INPUT);
            }
        }
        if !f.vertex_input && !f.base {
            for (binding, stride) in fixed.strides.iter_mut().enumerate() {
                *stride = (r[0x1c00 / 4 + binding * 4] & 0xfff) as u16;
            }
        }
        Ok(())
    }
}

impl FixedPipelineState {
    /// Disk pipeline state must select the same native feature paths as the
    /// current device, as Eden PipelineCache::LoadDiskResources checks.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp#L670-L679
    pub(in crate::native) fn matches_profile(&self, profile: &crate::DeviceProfile) -> bool {
        const DYNAMIC_FEATURES: u32 = 1 | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5);
        let state = &profile.dynamic_state;
        let features = u32::from(state.base)
            | (u32::from(state.base2) << 1)
            | (u32::from(state.depth_clamp_enable) << 2)
            | (u32::from(state.blending()) << 3)
            | (u32::from(state.vertex_input) << 5);
        self.raw1 & DYNAMIC_FEATURES == features
    }

    pub(in crate::native) fn environment_key(&self) -> Vec<u8> {
        let mut bytes = vec![7];
        for hash in self.shader {
            bytes.extend_from_slice(&hash.to_le_bytes());
        }
        for word in [self.raw1, self.raw2] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&self.colors);
        for word in [
            self.driver_id,
            self.driver_version,
            self.alpha_test_ref,
            self.point_size,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for word in self.viewport_swizzles {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&self.attribute_types.to_le_bytes());
        for word in self.dynamic_state.iter().chain(&self.blending) {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for attribute in self.attributes {
            bytes.extend_from_slice(&attribute.0.to_le_bytes());
        }
        for word in self.divisors {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for word in self.strides {
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
        if take::<1>(&mut bytes)? != [7] {
            return None;
        }
        let mut key = Self::default();
        for value in &mut key.shader {
            *value = u64::from_le_bytes(take(&mut bytes)?);
        }
        key.raw1 = u32::from_le_bytes(take(&mut bytes)?);
        key.raw2 = u32::from_le_bytes(take(&mut bytes)?);
        key.colors = take(&mut bytes)?;
        key.driver_id = u32::from_le_bytes(take(&mut bytes)?);
        key.driver_version = u32::from_le_bytes(take(&mut bytes)?);
        key.alpha_test_ref = u32::from_le_bytes(take(&mut bytes)?);
        key.point_size = u32::from_le_bytes(take(&mut bytes)?);
        for value in &mut key.viewport_swizzles {
            *value = u16::from_le_bytes(take(&mut bytes)?);
        }
        key.attribute_types = u64::from_le_bytes(take(&mut bytes)?);
        for value in key.dynamic_state.iter_mut().chain(&mut key.blending) {
            *value = u32::from_le_bytes(take(&mut bytes)?);
        }
        for value in &mut key.attributes {
            *value = Attribute(u32::from_le_bytes(take(&mut bytes)?));
        }
        for value in &mut key.divisors {
            *value = u32::from_le_bytes(take(&mut bytes)?);
        }
        for value in &mut key.strides {
            *value = u16::from_le_bytes(take(&mut bytes)?);
        }
        bytes.is_empty().then_some(key)
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
                    if self.raw1 & (1 << 3) != 0 {
                        super::super::BlendState::opaque()
                    } else {
                        blending::unpack(self.blending[i])?
                    },
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
        state.depth_clamp = self.raw2 & 1 != 0;
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
        if self.raw1 & (1 << 5) == 0 {
            let count = profile.limits.max_vertex_input_bindings.min(32);
            state.vertex_bindings = (0..count)
                .map(|binding| super::super::VertexBinding {
                    binding,
                    stride: if self.raw1 & 1 == 0 {
                        u32::from(self.strides[binding as usize])
                    } else {
                        0
                    },
                    rate: if self.attribute_types & (1 << binding) != 0 {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    },
                })
                .collect();
            state.vertex_divisors = (0..count)
                .filter(|&binding| {
                    self.attribute_types & (1 << binding) != 0
                        && self.divisors[binding as usize] != 1
                })
                .map(|binding| (binding, self.divisors[binding as usize]))
                .collect();
            state.vertex_attributes = self
                .attributes
                .iter()
                .enumerate()
                .filter(|&(location, attribute)| {
                    attribute.enabled() && vertex_attributes & (1 << location) != 0
                })
                .map(|(location, &attribute)| {
                    Ok(super::super::VertexAttribute {
                        location: location as u32,
                        binding: attribute.buffer(),
                        format: attribute.format(profile, location as u32)?,
                        offset: attribute.offset(),
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
            runtime.vertex_input_types[location] = if self.raw1 & (1 << 5) != 0 {
                match (self.attribute_types >> (location * 2)) & 3 {
                    0 => A::Disabled,
                    1 => A::Float,
                    2 => A::SignedInt,
                    3 => A::UnsignedInt,
                    _ => unreachable!(),
                }
            } else {
                let attribute = self.attributes[location];
                if !attribute.enabled() {
                    A::Disabled
                } else {
                    match attribute.kind() {
                        1 | 2 | 7 => A::Float,
                        3 => A::SignedInt,
                        4 => A::UnsignedInt,
                        5 => A::UnsignedScaled,
                        6 => A::SignedScaled,
                        _ => {
                            return Err(VulkanError::InvalidResource(
                                "invalid static vertex attribute type",
                            ));
                        }
                    }
                }
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
