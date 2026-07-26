//! Stable, pointer-free vocabulary for graphics coverage and diagnostics.
//!
//! These types format guest and emulator identities only. They deliberately do
//! not accept host pointers or host graphics API objects.

use std::fmt::{Display, Formatter};

/// Semantic layer at which graphics execution first lacks faithful behavior.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphicsGapKind {
    ServiceCommand,
    DeviceOpen,
    Ioctl,
    GpuPacket,
    ClassMethod,
    ShaderInstruction,
    ResourceFormat,
    BackendCapability,
}

impl GraphicsGapKind {
    /// Stable machine-readable label used in diagnostics and coverage files.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ServiceCommand => "service-command",
            Self::DeviceOpen => "device-open",
            Self::Ioctl => "ioctl",
            Self::GpuPacket => "gpu-packet",
            Self::ClassMethod => "class-method",
            Self::ShaderInstruction => "shader-instruction",
            Self::ResourceFormat => "resource-format",
            Self::BackendCapability => "backend-capability",
        }
    }
}

impl Display for GraphicsGapKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

macro_rules! decimal_diagnostic_id {
    ($name:ident, $label:literal, $inner:ty) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, concat!($label, "={}"), self.0)
            }
        }
    };
}

macro_rules! hex_diagnostic_id {
    ($name:ident, $label:literal, $inner:ty, $width:literal) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, concat!($label, "=0x{:0", $width, "x}"), self.0)
            }
        }
    };
}

hex_diagnostic_id!(CpuVirtualAddress, "cpu-va", u64, 16);
hex_diagnostic_id!(GraphicsAllocationId, "allocation", u64, 16);
decimal_diagnostic_id!(GpuChannelId, "channel", u32);
decimal_diagnostic_id!(GpfifoEntryIndex, "gpfifo-entry", u32);
hex_diagnostic_id!(GpuClassId, "class", u32, 8);
hex_diagnostic_id!(GpuMethodId, "method", u32, 8);

/// One guest syncpoint threshold, kept distinct from host completion values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SyncpointValue {
    pub id: u32,
    pub value: u32,
}

impl Display for SyncpointValue {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "syncpoint={}:{}", self.id, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_vocabulary_has_stable_distinct_labels() {
        let cases = [
            (GraphicsGapKind::ServiceCommand, "service-command"),
            (GraphicsGapKind::DeviceOpen, "device-open"),
            (GraphicsGapKind::Ioctl, "ioctl"),
            (GraphicsGapKind::GpuPacket, "gpu-packet"),
            (GraphicsGapKind::ClassMethod, "class-method"),
            (GraphicsGapKind::ShaderInstruction, "shader-instruction"),
            (GraphicsGapKind::ResourceFormat, "resource-format"),
            (GraphicsGapKind::BackendCapability, "backend-capability"),
        ];

        for (kind, label) in cases {
            assert_eq!(kind.label(), label);
            assert_eq!(kind.to_string(), label);
        }
    }

    #[test]
    fn graphics_identifiers_have_stable_pointer_free_formatting() {
        assert_eq!(
            CpuVirtualAddress(0x1234_5678).to_string(),
            "cpu-va=0x0000000012345678"
        );
        assert_eq!(
            crate::GpuVirtualAddress::try_new(0xabcdef, 40)
                .unwrap()
                .to_string(),
            "gpu-va=0x0000000000abcdef"
        );
        assert_eq!(
            GraphicsAllocationId(0x42).to_string(),
            "allocation=0x0000000000000042"
        );
        assert_eq!(
            crate::MappingGeneration::new(17).to_string(),
            "mapping-generation=17"
        );
        assert_eq!(GpuChannelId(3).to_string(), "channel=3");
        assert_eq!(GpfifoEntryIndex(9).to_string(), "gpfifo-entry=9");
        assert_eq!(GpuClassId(0xb197).to_string(), "class=0x0000b197");
        assert_eq!(GpuMethodId(0x1234).to_string(), "method=0x00001234");
        assert_eq!(
            SyncpointValue { id: 5, value: 42 }.to_string(),
            "syncpoint=5:42"
        );
    }
}
