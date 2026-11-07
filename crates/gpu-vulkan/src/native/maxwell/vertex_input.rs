//! Current vertex inputs belong to the channel. Dynamic pipeline keys keep
//! only shader input types; the scheduler captures native descriptions.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L1882-L1940
use super::*;
use crate::{DeviceProfile, VulkanError, cache::GraphicsState};
use ash::vk;

#[derive(Default)]
pub(crate) struct VertexInput {
    pub(crate) bindings: smallvec::SmallVec<[vk::VertexInputBindingDescription2EXT<'static>; 32]>,
    pub(crate) attributes:
        smallvec::SmallVec<[vk::VertexInputAttributeDescription2EXT<'static>; 32]>,
}
impl VertexInput {
    pub(crate) fn host(state: &GraphicsState) -> Self {
        Self {
            bindings: state
                .vertex_bindings
                .iter()
                .map(|binding| {
                    vk::VertexInputBindingDescription2EXT::default()
                        .binding(binding.binding)
                        .stride(binding.stride)
                        .input_rate(binding.rate)
                        .divisor(
                            state
                                .vertex_divisors
                                .iter()
                                .find(|&&(index, _)| index == binding.binding)
                                .map_or(1, |&(_, divisor)| divisor),
                        )
                })
                .collect(),
            attributes: state
                .vertex_attributes
                .iter()
                .map(|attribute| {
                    vk::VertexInputAttributeDescription2EXT::default()
                        .location(attribute.location)
                        .binding(attribute.binding)
                        .format(attribute.format)
                        .offset(attribute.offset)
                })
                .collect(),
        }
    }
    pub(crate) fn validate(&self, profile: &DeviceProfile) -> Result<(), VulkanError> {
        let limits = &profile.limits;
        if self.bindings.len() > limits.max_vertex_input_bindings as usize
            || self.attributes.len() > limits.max_vertex_input_attributes as usize
            || self.bindings.iter().any(|binding| {
                binding.binding >= limits.max_vertex_input_bindings
                    || binding.binding >= 32
                    || binding.stride > limits.max_vertex_input_binding_stride
            })
            || self.attributes.iter().any(|attribute| {
                attribute.location >= limits.max_vertex_input_attributes
                    || attribute.offset > limits.max_vertex_input_attribute_offset
                    || !self
                        .bindings
                        .iter()
                        .any(|binding| binding.binding == attribute.binding)
            })
        {
            return Err(VulkanError::InvalidResource(
                "vertex input exceeds native limits or selects an absent binding",
            ));
        }
        let mut seen = 0u32;
        for binding in &self.bindings {
            if !matches!(
                binding.input_rate,
                vk::VertexInputRate::VERTEX | vk::VertexInputRate::INSTANCE
            ) {
                return Err(VulkanError::InvalidResource("invalid vertex input rate"));
            }
            let bit = 1 << binding.binding;
            if seen & bit != 0 {
                return Err(VulkanError::InvalidResource("duplicate vertex binding"));
            }
            seen |= bit;
            validate_divisor(
                binding.input_rate == vk::VertexInputRate::INSTANCE,
                binding.divisor,
                profile,
            )?;
        }
        let mut locations = 0u32;
        for attribute in &self.attributes {
            if attribute.location >= 32 || locations & (1 << attribute.location) != 0 {
                return Err(VulkanError::InvalidResource(
                    "duplicate or unsupported host vertex location",
                ));
            }
            locations |= 1 << attribute.location;
            if !usize::try_from(attribute.format.as_raw())
                .ok()
                .and_then(|index| profile.formats.get(index))
                .map(|p| {
                    p.buffer_features
                        .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
                })
                .unwrap_or(false)
            {
                return Err(VulkanError::Compilation(
                    format!(
                        "vertex attribute {}: format {:?} lacks native VERTEX_BUFFER support",
                        attribute.location, attribute.format
                    )
                    .into(),
                ));
            }
        }
        Ok(())
    }
    pub(in crate::native) fn refresh(
        &mut self,
        state: &MaxwellThreeDState,
        profile: &DeviceProfile,
    ) -> Result<(), VulkanError> {
        self.bindings.clear();
        self.attributes.clear();
        let r = state.register_words();
        let max_bindings = profile.limits.max_vertex_input_bindings.min(32);
        for location in 0..profile.limits.max_vertex_input_attributes.min(32) {
            let word = r[0x1160 / 4 + location as usize];
            if word & 64 != 0 || word & 31 >= max_bindings {
                continue;
            }
            let format = native_format(word, location)?;
            if !profile
                .formats
                .get(format.as_raw() as usize)
                .map(|p| {
                    p.buffer_features
                        .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
                })
                .unwrap_or(false)
            {
                return Err(VulkanError::Compilation(format!("vertex attribute {location}: format {format:?} lacks native vertex-buffer support").into()));
            }
            let offset = (word >> 7) & 0x3fff;
            if offset > profile.limits.max_vertex_input_attribute_offset {
                return Err(VulkanError::Compilation(
                    format!("vertex attribute {location}: offset {offset} exceeds host limit")
                        .into(),
                ));
            }
            self.attributes.push(
                vk::VertexInputAttributeDescription2EXT::default()
                    .location(location)
                    .binding(word & 31)
                    .format(format)
                    .offset((word >> 7) & 0x3fff),
            );
        }
        for binding in 0..max_bindings {
            let instanced = r[0x1880 / 4 + binding as usize] != 0;
            let base = 0x1c00 / 4 + binding as usize * 4;
            let stride = r[base] & 0xfff;
            let divisor = if instanced { r[base + 3] } else { 1 };
            if stride > profile.limits.max_vertex_input_binding_stride {
                return Err(VulkanError::Compilation(
                    format!("vertex binding {binding}: stride {stride} exceeds host limit").into(),
                ));
            }
            validate_divisor(instanced, divisor, profile)?;
            self.bindings.push(
                vk::VertexInputBindingDescription2EXT::default()
                    .binding(binding)
                    .stride(r[base] & 0xfff)
                    .input_rate(if instanced {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    })
                    .divisor(if instanced { r[base + 3] } else { 1 }),
            );
        }
        Ok(())
    }
}
/// Shader-affecting input classes use three bits per attribute to retain Nixe's
/// exact 32-bit scaled-to-integer SSA conversion, in addition to float/int/uint.
/// This is the same feature-dependent key projection as Eden attribute_types.
pub(in crate::native) fn attribute_types(state: &MaxwellThreeDState) -> Result<u128, VulkanError> {
    let mut types = 0;
    for (location, &word) in state.register_words()[0x1160 / 4..0x1160 / 4 + 32]
        .iter()
        .enumerate()
    {
        if word & 64 != 0 {
            continue;
        }
        let width = (word >> 21) & 63;
        let scaled32 = matches!(width, 0x12 | 0x04 | 0x02 | 0x01);
        let kind = match (word >> 27) & 7 {
            1 | 2 | 7 => 1,
            3 => 2,
            4 => 3,
            5 if scaled32 => 5,
            6 if scaled32 => 4,
            5 | 6 => 1,
            _ => {
                return Err(VulkanError::Compilation(
                    format!("vertex attribute {location}: unsupported numerical type {word:#x}")
                        .into(),
                ));
            }
        };
        types |= (kind as u128) << (location * 3);
    }
    Ok(types)
}

fn validate_divisor(
    instanced: bool,
    divisor: u32,
    profile: &DeviceProfile,
) -> Result<(), VulkanError> {
    if instanced
        && ((divisor != 1 && !profile.dynamic_state.vertex_divisor)
            || (divisor == 0 && !profile.dynamic_state.vertex_zero_divisor)
            || (divisor > 1 && divisor > profile.dynamic_state.max_vertex_divisor))
    {
        return Err(VulkanError::Compilation(
            format!("unsupported native vertex instance divisor {divisor}").into(),
        ));
    }
    Ok(())
}
// MaxwellToVK::VertexFormat is an ISA-to-native lookup, independent of buffer
// contents and programs. Scaled32 uses the same native integer fetch as the
// shader type projection above, with the exact SSA conversion in the shader.
pub(super) fn native_format(word: u32, location: u32) -> Result<vk::Format, VulkanError> {
    use vk::Format as F;
    let size = (word >> 21) & 63;
    let kind = (word >> 27) & 7;
    if word >> 31 != 0 {
        return Err(VulkanError::Compilation(
            format!("vertex attribute {location}: red/blue exchange is unsupported ({word:#x})")
                .into(),
        ));
    }
    if size == 0x30 {
        return match kind {
            1 => Ok(F::A2B10G10R10_SNORM_PACK32),
            2 => Ok(F::A2B10G10R10_UNORM_PACK32),
            _ => Err(VulkanError::Compilation(
                format!("vertex attribute {location}: unsupported packed numerical type {kind}")
                    .into(),
            )),
        };
    }
    let (width, count) = match size {
        0x1d => (8, 0),
        0x18 => (8, 1),
        0x13 => (8, 2),
        0x0a => (8, 3),
        0x1b => (16, 0),
        0x0f => (16, 1),
        0x05 => (16, 2),
        0x03 => (16, 3),
        0x12 => (32, 0),
        0x04 => (32, 1),
        0x02 => (32, 2),
        0x01 => (32, 3),
        _ => {
            return Err(VulkanError::Compilation(
                format!("vertex attribute {location}: unsupported component encoding {size:#x}")
                    .into(),
            ));
        }
    };
    let formats = match (width, kind) {
        (8, 1) => [
            F::R8_SNORM,
            F::R8G8_SNORM,
            F::R8G8B8_SNORM,
            F::R8G8B8A8_SNORM,
        ],
        (8, 2) => [
            F::R8_UNORM,
            F::R8G8_UNORM,
            F::R8G8B8_UNORM,
            F::R8G8B8A8_UNORM,
        ],
        (8, 3) => [F::R8_SINT, F::R8G8_SINT, F::R8G8B8_SINT, F::R8G8B8A8_SINT],
        (8, 4) => [F::R8_UINT, F::R8G8_UINT, F::R8G8B8_UINT, F::R8G8B8A8_UINT],
        (8, 5) => [
            F::R8_USCALED,
            F::R8G8_USCALED,
            F::R8G8B8_USCALED,
            F::R8G8B8A8_USCALED,
        ],
        (8, 6) => [
            F::R8_SSCALED,
            F::R8G8_SSCALED,
            F::R8G8B8_SSCALED,
            F::R8G8B8A8_SSCALED,
        ],
        (16, 1) => [
            F::R16_SNORM,
            F::R16G16_SNORM,
            F::R16G16B16_SNORM,
            F::R16G16B16A16_SNORM,
        ],
        (16, 2) => [
            F::R16_UNORM,
            F::R16G16_UNORM,
            F::R16G16B16_UNORM,
            F::R16G16B16A16_UNORM,
        ],
        (16, 3) => [
            F::R16_SINT,
            F::R16G16_SINT,
            F::R16G16B16_SINT,
            F::R16G16B16A16_SINT,
        ],
        (16, 4) => [
            F::R16_UINT,
            F::R16G16_UINT,
            F::R16G16B16_UINT,
            F::R16G16B16A16_UINT,
        ],
        (16, 5) => [
            F::R16_USCALED,
            F::R16G16_USCALED,
            F::R16G16B16_USCALED,
            F::R16G16B16A16_USCALED,
        ],
        (16, 6) => [
            F::R16_SSCALED,
            F::R16G16_SSCALED,
            F::R16G16B16_SSCALED,
            F::R16G16B16A16_SSCALED,
        ],
        (16, 7) => [
            F::R16_SFLOAT,
            F::R16G16_SFLOAT,
            F::R16G16B16_SFLOAT,
            F::R16G16B16A16_SFLOAT,
        ],
        (32, 3 | 6) => [
            F::R32_SINT,
            F::R32G32_SINT,
            F::R32G32B32_SINT,
            F::R32G32B32A32_SINT,
        ],
        (32, 4 | 5) => [
            F::R32_UINT,
            F::R32G32_UINT,
            F::R32G32B32_UINT,
            F::R32G32B32A32_UINT,
        ],
        (32, 7) => [
            F::R32_SFLOAT,
            F::R32G32_SFLOAT,
            F::R32G32B32_SFLOAT,
            F::R32G32B32A32_SFLOAT,
        ],
        _ => {
            return Err(VulkanError::Compilation(
                format!(
                    "vertex attribute {location}: unsupported {width}-bit numerical type {kind}"
                )
                .into(),
            ));
        }
    };
    Ok(formats[count])
}
