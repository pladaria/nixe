//! Typed `MAXWELL_B` color-reduction configuration.
//!
//! NVIDIA exposes an explicit enable register independently from the
//! format-specific threshold registers. Retaining that distinction prevents a
//! programmed threshold from being mistaken for an active optimization.
//!
//! Register layouts and boolean encodings are pinned to NVIDIA's public class
//! header:
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L970-L973>
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1519-L1521>
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1535-L1537>
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1539-L1541>
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1547-L1549>
//! <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1551-L1553>

use crate::MaxwellMethodSource;

use super::MaxwellThreeDRegister;

/// One unsigned normalized eight-bit threshold component.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDUnorm8(u8);

impl MaxwellThreeDUnorm8 {
    #[must_use]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

/// The two UNORM8 color-reduction thresholds programmed as one atomic word.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorReductionThresholdsUnorm8 {
    all_covered_all_hit_once: MaxwellThreeDUnorm8,
    all_covered: MaxwellThreeDUnorm8,
}

impl MaxwellThreeDColorReductionThresholdsUnorm8 {
    #[must_use]
    pub const fn new(
        all_covered_all_hit_once: MaxwellThreeDUnorm8,
        all_covered: MaxwellThreeDUnorm8,
    ) -> Self {
        Self {
            all_covered_all_hit_once,
            all_covered,
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x00ff_00ff != 0 {
            return None;
        }
        Some(Self::new(
            MaxwellThreeDUnorm8::new(raw as u8),
            MaxwellThreeDUnorm8::new((raw >> 16) as u8),
        ))
    }

    #[must_use]
    pub const fn all_covered_all_hit_once(self) -> MaxwellThreeDUnorm8 {
        self.all_covered_all_hit_once
    }

    #[must_use]
    pub const fn all_covered(self) -> MaxwellThreeDUnorm8 {
        self.all_covered
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.all_covered_all_hit_once.raw() as u32 | ((self.all_covered.raw() as u32) << 16)
    }
}

/// The two thresholds selected for UNORM10 color surfaces.
///
/// Despite the surface name, NVIDIA encodes both components as eight-bit
/// unsigned normalized values. This remains a distinct semantic type and
/// register from the UNORM8 thresholds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorReductionThresholdsUnorm10 {
    all_covered_all_hit_once: MaxwellThreeDUnorm8,
    all_covered: MaxwellThreeDUnorm8,
}

impl MaxwellThreeDColorReductionThresholdsUnorm10 {
    #[must_use]
    pub const fn new(
        all_covered_all_hit_once: MaxwellThreeDUnorm8,
        all_covered: MaxwellThreeDUnorm8,
    ) -> Self {
        Self {
            all_covered_all_hit_once,
            all_covered,
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x00ff_00ff != 0 {
            return None;
        }
        Some(Self::new(
            MaxwellThreeDUnorm8::new(raw as u8),
            MaxwellThreeDUnorm8::new((raw >> 16) as u8),
        ))
    }

    #[must_use]
    pub const fn all_covered_all_hit_once(self) -> MaxwellThreeDUnorm8 {
        self.all_covered_all_hit_once
    }

    #[must_use]
    pub const fn all_covered(self) -> MaxwellThreeDUnorm8 {
        self.all_covered
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.all_covered_all_hit_once.raw() as u32 | ((self.all_covered.raw() as u32) << 16)
    }
}

/// The two thresholds selected for UNORM16 color surfaces.
///
/// NVIDIA encodes both components as eight-bit unsigned normalized values,
/// independently from the UNORM8 and UNORM10 threshold registers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorReductionThresholdsUnorm16 {
    all_covered_all_hit_once: MaxwellThreeDUnorm8,
    all_covered: MaxwellThreeDUnorm8,
}

impl MaxwellThreeDColorReductionThresholdsUnorm16 {
    #[must_use]
    pub const fn new(
        all_covered_all_hit_once: MaxwellThreeDUnorm8,
        all_covered: MaxwellThreeDUnorm8,
    ) -> Self {
        Self {
            all_covered_all_hit_once,
            all_covered,
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x00ff_00ff != 0 {
            return None;
        }
        Some(Self::new(
            MaxwellThreeDUnorm8::new(raw as u8),
            MaxwellThreeDUnorm8::new((raw >> 16) as u8),
        ))
    }

    #[must_use]
    pub const fn all_covered_all_hit_once(self) -> MaxwellThreeDUnorm8 {
        self.all_covered_all_hit_once
    }

    #[must_use]
    pub const fn all_covered(self) -> MaxwellThreeDUnorm8 {
        self.all_covered
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.all_covered_all_hit_once.raw() as u32 | ((self.all_covered.raw() as u32) << 16)
    }
}

/// One opaque eight-bit threshold component selected for FP16 color surfaces.
///
/// NVIDIA publishes the field width but not a numerical encoding, so this
/// type deliberately preserves the bits without interpreting them as an IEEE
/// floating-point value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDColorReductionFp16Threshold(u8);

impl MaxwellThreeDColorReductionFp16Threshold {
    #[must_use]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

/// The two opaque FP16 color-reduction thresholds programmed atomically.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorReductionThresholdsFp16 {
    all_covered_all_hit_once: MaxwellThreeDColorReductionFp16Threshold,
    all_covered: MaxwellThreeDColorReductionFp16Threshold,
}

impl MaxwellThreeDColorReductionThresholdsFp16 {
    #[must_use]
    pub const fn new(
        all_covered_all_hit_once: MaxwellThreeDColorReductionFp16Threshold,
        all_covered: MaxwellThreeDColorReductionFp16Threshold,
    ) -> Self {
        Self {
            all_covered_all_hit_once,
            all_covered,
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x00ff_00ff != 0 {
            return None;
        }
        Some(Self::new(
            MaxwellThreeDColorReductionFp16Threshold::new(raw as u8),
            MaxwellThreeDColorReductionFp16Threshold::new((raw >> 16) as u8),
        ))
    }

    #[must_use]
    pub const fn all_covered_all_hit_once(self) -> MaxwellThreeDColorReductionFp16Threshold {
        self.all_covered_all_hit_once
    }

    #[must_use]
    pub const fn all_covered(self) -> MaxwellThreeDColorReductionFp16Threshold {
        self.all_covered
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.all_covered_all_hit_once.raw() as u32 | ((self.all_covered.raw() as u32) << 16)
    }
}

/// One opaque eight-bit threshold component selected for SRGB8 color surfaces.
///
/// NVIDIA publishes the field width but not its transfer-function semantics,
/// so this type preserves the encoded byte without treating it as linear
/// UNORM data.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct MaxwellThreeDColorReductionSrgb8Threshold(u8);

impl MaxwellThreeDColorReductionSrgb8Threshold {
    #[must_use]
    pub const fn new(raw: u8) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

/// The two opaque SRGB8 color-reduction thresholds programmed atomically.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDColorReductionThresholdsSrgb8 {
    all_covered_all_hit_once: MaxwellThreeDColorReductionSrgb8Threshold,
    all_covered: MaxwellThreeDColorReductionSrgb8Threshold,
}

impl MaxwellThreeDColorReductionThresholdsSrgb8 {
    #[must_use]
    pub const fn new(
        all_covered_all_hit_once: MaxwellThreeDColorReductionSrgb8Threshold,
        all_covered: MaxwellThreeDColorReductionSrgb8Threshold,
    ) -> Self {
        Self {
            all_covered_all_hit_once,
            all_covered,
        }
    }

    pub(super) const fn parse(raw: u32) -> Option<Self> {
        if raw & !0x00ff_00ff != 0 {
            return None;
        }
        Some(Self::new(
            MaxwellThreeDColorReductionSrgb8Threshold::new(raw as u8),
            MaxwellThreeDColorReductionSrgb8Threshold::new((raw >> 16) as u8),
        ))
    }

    #[must_use]
    pub const fn all_covered_all_hit_once(self) -> MaxwellThreeDColorReductionSrgb8Threshold {
        self.all_covered_all_hit_once
    }

    #[must_use]
    pub const fn all_covered(self) -> MaxwellThreeDColorReductionSrgb8Threshold {
        self.all_covered
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.all_covered_all_hit_once.raw() as u32 | ((self.all_covered.raw() as u32) << 16)
    }
}

/// Whether later color output may consume the programmed reduction thresholds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u32)]
pub enum MaxwellThreeDColorReductionThresholdsEnable {
    Disabled = 0,
    Enabled = 1,
}

impl MaxwellThreeDColorReductionThresholdsEnable {
    pub(super) const fn parse(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Disabled),
            1 => Some(Self::Enabled),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        self as u32
    }
}

/// One validated color-reduction register transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellThreeDColorReductionStateWrite {
    Enable {
        value: MaxwellThreeDColorReductionThresholdsEnable,
        source: MaxwellMethodSource,
    },
    ThresholdsUnorm8 {
        value: MaxwellThreeDColorReductionThresholdsUnorm8,
        source: MaxwellMethodSource,
    },
    ThresholdsUnorm10 {
        value: MaxwellThreeDColorReductionThresholdsUnorm10,
        source: MaxwellMethodSource,
    },
    ThresholdsUnorm16 {
        value: MaxwellThreeDColorReductionThresholdsUnorm16,
        source: MaxwellMethodSource,
    },
    ThresholdsFp16 {
        value: MaxwellThreeDColorReductionThresholdsFp16,
        source: MaxwellMethodSource,
    },
    ThresholdsSrgb8 {
        value: MaxwellThreeDColorReductionThresholdsSrgb8,
        source: MaxwellMethodSource,
    },
}

/// Persistent color-reduction configuration for one `MAXWELL_B` channel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellThreeDColorReductionState {
    enable: MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsEnable>,
    thresholds_unorm8: MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsUnorm8>,
    thresholds_unorm10: MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsUnorm10>,
    thresholds_unorm16: MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsUnorm16>,
    thresholds_fp16: MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsFp16>,
    thresholds_srgb8: MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsSrgb8>,
}

impl MaxwellThreeDColorReductionState {
    #[must_use]
    pub const fn enable(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsEnable> {
        &self.enable
    }

    #[must_use]
    pub const fn thresholds_unorm8(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsUnorm8> {
        &self.thresholds_unorm8
    }

    #[must_use]
    pub const fn thresholds_unorm10(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsUnorm10> {
        &self.thresholds_unorm10
    }

    #[must_use]
    pub const fn thresholds_unorm16(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsUnorm16> {
        &self.thresholds_unorm16
    }

    #[must_use]
    pub const fn thresholds_fp16(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsFp16> {
        &self.thresholds_fp16
    }

    #[must_use]
    pub const fn thresholds_srgb8(
        &self,
    ) -> &MaxwellThreeDRegister<MaxwellThreeDColorReductionThresholdsSrgb8> {
        &self.thresholds_srgb8
    }

    pub(super) fn apply(&mut self, write: MaxwellThreeDColorReductionStateWrite) {
        match write {
            MaxwellThreeDColorReductionStateWrite::Enable { value, source } => {
                self.enable = MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm8 { value, source } => {
                self.thresholds_unorm8 =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm10 { value, source } => {
                self.thresholds_unorm10 =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm16 { value, source } => {
                self.thresholds_unorm16 =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDColorReductionStateWrite::ThresholdsFp16 { value, source } => {
                self.thresholds_fp16 =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
            MaxwellThreeDColorReductionStateWrite::ThresholdsSrgb8 { value, source } => {
                self.thresholds_srgb8 =
                    MaxwellThreeDRegister::programmed(value.raw(), value, source);
            }
        }
    }
}
