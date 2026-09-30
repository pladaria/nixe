//! Host-independent patch assembly and tessellator state. Levels retain IEEE
//! bit patterns; neither frontend nor host capability checks may invent clamps.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessellationDomain {
    Isolines,
    Triangles,
    Quads,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessellationSpacing {
    Equal,
    FractionalOdd,
    FractionalEven,
}

/// Orientation in a lower-left tessellation domain, before TES and viewport/
/// window transforms: counterclockwise has positive signed area in (u, v).
/// This matches GLSL domain winding, not Vulkan's default upper-left domain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessellationWinding {
    CounterClockwise,
    Clockwise,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessellationOutput {
    Points,
    Lines,
    Triangles(TessellationWinding),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TessellationMode {
    pub domain: TessellationDomain,
    pub spacing: TessellationSpacing,
    pub output: TessellationOutput,
}

/// Default levels are dynamic parameters, not part of a shader/pipeline key.
/// `Shader` means the active control program supplies the levels and output
/// patch size (the latter belongs to its IR metadata, not input patch assembly).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TessellationControl {
    Shader,
    DefaultLevels {
        outer: [u32; 4],
        inner: [u32; 2],
        /// Bits 0..3 identify programmed outer lanes; 4..5 identify inner
        /// lanes. Unset lanes are padding, never a guest-visible zero default.
        /// TES may read levels not consumed by the fixed-function domain.
        defined: u8,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TessellationState {
    pub mode: TessellationMode,
    pub input_control_points: u8,
    pub control: TessellationControl,
}
