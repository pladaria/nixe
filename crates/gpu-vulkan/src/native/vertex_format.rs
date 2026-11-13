//! Shared NVIDIA vertex attribute encoding, defined by the B197 and C797 class headers.
use crate::{DeviceProfile, VulkanError};
use ash::vk;
// MaxwellToVK::VertexFormat is an ISA-to-native lookup, independent of buffer
// contents and programs. Scaled32 uses the same native integer fetch as the
// shader type projection above, with the exact SSA conversion in the shader.
/// The packed attribute used by Eden's fixed key. Reserved fields and BGRA do
/// not participate in identity; unsupported BGRA is rejected when consumed.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::native) struct Attribute(pub(in crate::native) u32);
impl Attribute {
    pub(in crate::native) fn from_guest(word: u32) -> Self {
        Self(
            u32::from(word & 64 == 0)
                | ((word & 31) << 1)
                | (((word >> 7) & 0x3fff) << 6)
                | (((word >> 27) & 7) << 20)
                | (((word >> 21) & 63) << 23),
        )
    }
    pub(in crate::native) fn enabled(self) -> bool {
        self.0 & 1 != 0
    }
    pub(in crate::native) fn buffer(self) -> u32 {
        (self.0 >> 1) & 31
    }
    pub(in crate::native) fn offset(self) -> u32 {
        (self.0 >> 6) & 0x3fff
    }
    pub(in crate::native) fn kind(self) -> u32 {
        (self.0 >> 20) & 7
    }
    pub(in crate::native) fn format(
        self,
        profile: &DeviceProfile,
        location: u32,
    ) -> Result<vk::Format, VulkanError> {
        let size = (self.0 >> 23) & 63;
        let mut kind = self.kind();
        if profile.emulate_scaled_formats {
            kind = match kind {
                5 => 4,
                6 => 3,
                other => other,
            };
        }
        let format = match (kind, size) {
            (2, 0x1d | 0x34) => vk::Format::R8_UNORM,
            (2, 0x18 | 0x32) => vk::Format::R8G8_UNORM,
            (2, 0x13) => vk::Format::R8G8B8_UNORM,
            (2, 0x0a | 0x33) => vk::Format::R8G8B8A8_UNORM,
            (2, 0x1b) => vk::Format::R16_UNORM,
            (2, 0x0f) => vk::Format::R16G16_UNORM,
            (2, 0x05) => vk::Format::R16G16B16_UNORM,
            (2, 0x03) => vk::Format::R16G16B16A16_UNORM,
            (2, 0x30) => vk::Format::A2B10G10R10_UNORM_PACK32,
            (1, 0x1d | 0x34) => vk::Format::R8_SNORM,
            (1, 0x18 | 0x32) => vk::Format::R8G8_SNORM,
            (1, 0x13) => vk::Format::R8G8B8_SNORM,
            (1, 0x0a | 0x33) => vk::Format::R8G8B8A8_SNORM,
            (1, 0x1b) => vk::Format::R16_SNORM,
            (1, 0x0f) => vk::Format::R16G16_SNORM,
            (1, 0x05) => vk::Format::R16G16B16_SNORM,
            (1, 0x03) => vk::Format::R16G16B16A16_SNORM,
            (1, 0x30) => vk::Format::A2B10G10R10_SNORM_PACK32,
            (5, 0x1d | 0x34) => vk::Format::R8_USCALED,
            (5, 0x18 | 0x32) => vk::Format::R8G8_USCALED,
            (5, 0x13) => vk::Format::R8G8B8_USCALED,
            (5, 0x0a | 0x33) => vk::Format::R8G8B8A8_USCALED,
            (5, 0x1b) => vk::Format::R16_USCALED,
            (5, 0x0f) => vk::Format::R16G16_USCALED,
            (5, 0x05) => vk::Format::R16G16B16_USCALED,
            (5, 0x03) => vk::Format::R16G16B16A16_USCALED,
            (5, 0x30) => vk::Format::A2B10G10R10_USCALED_PACK32,
            (6, 0x1d | 0x34) => vk::Format::R8_SSCALED,
            (6, 0x18 | 0x32) => vk::Format::R8G8_SSCALED,
            (6, 0x13) => vk::Format::R8G8B8_SSCALED,
            (6, 0x0a | 0x33) => vk::Format::R8G8B8A8_SSCALED,
            (6, 0x1b) => vk::Format::R16_SSCALED,
            (6, 0x0f) => vk::Format::R16G16_SSCALED,
            (6, 0x05) => vk::Format::R16G16B16_SSCALED,
            (6, 0x03) => vk::Format::R16G16B16A16_SSCALED,
            (6, 0x30) => vk::Format::A2B10G10R10_SSCALED_PACK32,
            (4, 0x1d | 0x34) => vk::Format::R8_UINT,
            (4, 0x18 | 0x32) => vk::Format::R8G8_UINT,
            (4, 0x13) => vk::Format::R8G8B8_UINT,
            (4, 0x0a | 0x33) => vk::Format::R8G8B8A8_UINT,
            (4, 0x1b) => vk::Format::R16_UINT,
            (4, 0x0f) => vk::Format::R16G16_UINT,
            (4, 0x05) => vk::Format::R16G16B16_UINT,
            (4, 0x03) => vk::Format::R16G16B16A16_UINT,
            (4, 0x12) => vk::Format::R32_UINT,
            (4, 0x04) => vk::Format::R32G32_UINT,
            (4, 0x02) => vk::Format::R32G32B32_UINT,
            (4, 0x01) => vk::Format::R32G32B32A32_UINT,
            (4, 0x30) => vk::Format::A2B10G10R10_UINT_PACK32,
            (3, 0x1d | 0x34) => vk::Format::R8_SINT,
            (3, 0x18 | 0x32) => vk::Format::R8G8_SINT,
            (3, 0x13) => vk::Format::R8G8B8_SINT,
            (3, 0x0a | 0x33) => vk::Format::R8G8B8A8_SINT,
            (3, 0x1b) => vk::Format::R16_SINT,
            (3, 0x0f) => vk::Format::R16G16_SINT,
            (3, 0x05) => vk::Format::R16G16B16_SINT,
            (3, 0x03) => vk::Format::R16G16B16A16_SINT,
            (3, 0x12) => vk::Format::R32_SINT,
            (3, 0x04) => vk::Format::R32G32_SINT,
            (3, 0x02) => vk::Format::R32G32B32_SINT,
            (3, 0x01) => vk::Format::R32G32B32A32_SINT,
            (3, 0x30) => vk::Format::A2B10G10R10_SINT_PACK32,
            (7, 0x1b) => vk::Format::R16_SFLOAT,
            (7, 0x0f) => vk::Format::R16G16_SFLOAT,
            (7, 0x05) => vk::Format::R16G16B16_SFLOAT,
            (7, 0x03) => vk::Format::R16G16B16A16_SFLOAT,
            (7, 0x12) => vk::Format::R32_SFLOAT,
            (7, 0x04) => vk::Format::R32G32_SFLOAT,
            (7, 0x02) => vk::Format::R32G32B32_SFLOAT,
            (7, 0x01) => vk::Format::R32G32B32A32_SFLOAT,
            (7, 0x31) => vk::Format::B10G11R11_UFLOAT_PACK32,
            _ => return Err(VulkanError::Compilation(format!("vertex attribute {location}: unsupported numerical type {kind} and size {size:#x}").into())),
        };
        Ok(format)
    }
}
