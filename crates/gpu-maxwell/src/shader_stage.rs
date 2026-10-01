//! Maxwell execution stages shared by graphics and compute shader translation.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaxwellShaderStage {
    VertexCullBeforeFetch,
    Vertex,
    TessellationInit,
    Tessellation,
    Geometry,
    Pixel,
    Compute,
}
