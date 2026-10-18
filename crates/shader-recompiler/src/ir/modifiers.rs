use crate::{
    CompileError, Result,
    diagnostics::FailureCategory,
    types::{ImageFormat, TextureType},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FmzMode {
    #[default]
    DontCare = 0,
    FTZ = 1,
    FMZ = 2,
    None = 3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FpRounding {
    #[default]
    DontCare = 0,
    RN = 1,
    RM = 2,
    RP = 3,
    RZ = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FpControl {
    pub no_contraction: bool,
    pub rounding: FpRounding,
    pub fmz_mode: FmzMode,
}

impl FpControl {
    pub const fn bits(self) -> u32 {
        self.no_contraction as u32 | ((self.rounding as u32) << 8) | ((self.fmz_mode as u32) << 16)
    }
    pub fn from_bits(bits: u32) -> Result<Self> {
        let invalid = || {
            CompileError::new(
                FailureCategory::Internal,
                "Invalid floating-point IR control flags",
            )
        };
        let no_contraction = match bits & 255 {
            0 => false,
            1 => true,
            _ => return Err(invalid()),
        };
        let rounding = match (bits >> 8) & 255 {
            0 => FpRounding::DontCare,
            1 => FpRounding::RN,
            2 => FpRounding::RM,
            3 => FpRounding::RP,
            4 => FpRounding::RZ,
            _ => return Err(invalid()),
        };
        let fmz_mode = match (bits >> 16) & 255 {
            0 => FmzMode::DontCare,
            1 => FmzMode::FTZ,
            2 => FmzMode::FMZ,
            3 => FmzMode::None,
            _ => return Err(invalid()),
        };
        Ok(Self {
            no_contraction,
            rounding,
            fmz_mode,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureInstInfo(pub u32);

impl TextureInstInfo {
    pub const fn descriptor_index(self) -> u32 {
        self.0 & 0xffff
    }
    pub fn texture_type(self) -> TextureType {
        TextureType::try_from((self.0 >> 16) & 7).unwrap()
    }
    pub const fn is_depth(self) -> bool {
        self.0 & (1 << 19) != 0
    }
    pub const fn has_bias(self) -> bool {
        self.0 & (1 << 20) != 0
    }
    pub const fn has_lod_clamp(self) -> bool {
        self.0 & (1 << 21) != 0
    }
    pub const fn relaxed_precision(self) -> bool {
        self.0 & (1 << 22) != 0
    }
    pub const fn gather_component(self) -> u32 {
        (self.0 >> 23) & 3
    }
    pub const fn num_derivatives(self) -> u32 {
        (self.0 >> 25) & 3
    }
    pub fn image_format(self) -> ImageFormat {
        ImageFormat::try_from((self.0 >> 27) & 7).unwrap()
    }
    pub const fn ndv_is_active(self) -> bool {
        self.0 & (1 << 30) != 0
    }
    pub const fn is_integer(self) -> bool {
        self.0 & (1 << 31) != 0
    }

    pub fn set_descriptor_index(&mut self, value: u32) {
        self.assign(0, 16, value);
    }
    pub fn set_texture_type(&mut self, value: TextureType) {
        self.assign(16, 3, value as u32);
    }
    pub fn set_is_depth(&mut self, value: bool) {
        self.assign(19, 1, value as u32);
    }
    pub fn set_has_bias(&mut self, value: bool) {
        self.assign(20, 1, value as u32);
    }
    pub fn set_has_lod_clamp(&mut self, value: bool) {
        self.assign(21, 1, value as u32);
    }
    pub fn set_relaxed_precision(&mut self, value: bool) {
        self.assign(22, 1, value as u32);
    }
    pub fn set_gather_component(&mut self, value: u32) {
        self.assign(23, 2, value);
    }
    pub fn set_num_derivatives(&mut self, value: u32) {
        self.assign(25, 2, value);
    }
    pub fn set_image_format(&mut self, value: ImageFormat) {
        self.assign(27, 3, value as u32);
    }
    pub fn set_ndv_is_active(&mut self, value: bool) {
        self.assign(30, 1, value as u32);
    }
    pub fn set_is_integer(&mut self, value: bool) {
        self.assign(31, 1, value as u32);
    }
    fn assign(&mut self, start: u32, width: u32, value: u32) {
        let mask = ((1 << width) - 1) << start;
        self.0 = (self.0 & !mask) | ((value << start) & mask);
    }
}
