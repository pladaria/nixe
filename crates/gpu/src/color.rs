//! Fixed-function color output, independent of guest register and host API encodings.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlphaSaturated,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BlendComponent {
    pub operation: BlendOperation,
    pub source: BlendFactor,
    pub destination: BlendFactor,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColorBlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

/// A component mask, with bit 0 red, bit 1 green, bit 2 blue and bit 3 alpha.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColorWriteMask(u8);
impl ColorWriteMask {
    pub const ALL: Self = Self(15);
    pub const NONE: Self = Self(0);

    #[must_use]
    pub const fn new(red: bool, green: bool, blue: bool, alpha: bool) -> Self {
        Self(red as u8 | (green as u8) << 1 | (blue as u8) << 2 | (alpha as u8) << 3)
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColorOutputState {
    pub blend: Option<ColorBlendState>,
    pub write_mask: ColorWriteMask,
}
impl ColorOutputState {
    pub const REPLACE: Self = Self {
        blend: None,
        write_mask: ColorWriteMask::ALL,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_mask_uses_rgba_bit_order_without_extra_bits() {
        for bits in 0..16 {
            assert_eq!(
                ColorWriteMask::new(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0)
                    .bits(),
                bits
            );
        }
        assert_eq!(ColorWriteMask::NONE.bits(), 0);
        assert_eq!(ColorWriteMask::ALL.bits(), 15);
    }
}
