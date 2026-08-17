//! Shader Program Architecture version fields shared by Maxwell engines.

/// Shader Program Architecture version consumed by a Maxwell shader frontend.
///
/// NVIDIA publishes the same eight-bit minor and eight-bit major fields for
/// the 3D and compute classes in its pinned public class headers:
/// <https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L398-L400>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellSpaVersion {
    major: u8,
    minor: u8,
}

impl MaxwellSpaVersion {
    pub(crate) const fn parse(raw: u32) -> Option<Self> {
        if raw <= 0xffff {
            Some(Self {
                major: (raw >> 8) as u8,
                minor: raw as u8,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn major(self) -> u8 {
        self.major
    }

    #[must_use]
    pub const fn minor(self) -> u8 {
        self.minor
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        ((self.major as u32) << 8) | self.minor as u32
    }
}
