//! Indexed arguments preserve GPU index fetches and shader-visible vertex IDs.
use super::*;
use crate::engines::threed::MaxwellThreeDIndexElementSize;
use nixe_gpu::IndexType;

pub(super) fn index_type(
    state: &MaxwellThreeDState,
) -> Result<IndexType, MaxwellThreeDLoweringError> {
    match state
        .vertex_input()
        .index()
        .element_size()
        .value()
        .copied()
        .ok_or(MaxwellThreeDLoweringError::IncompleteDraw(
            "SET_INDEX_BUFFER_E",
        ))? {
        MaxwellThreeDIndexElementSize::TwoBytes => Ok(IndexType::Uint16),
        MaxwellThreeDIndexElementSize::FourBytes => Ok(IndexType::Uint32),
        format => Err(MaxwellThreeDLoweringError::UnsupportedIndexFormat(format)),
    }
}

pub(super) fn draw_arguments(
    state: &MaxwellThreeDState,
    index_count: u32,
) -> Result<DrawArguments, MaxwellThreeDLoweringError> {
    if index_count == 0 {
        return Err(MaxwellThreeDLoweringError::EmptyDraw);
    }
    index_type(state)?;
    let input = state.vertex_input();
    let topology = primitive_topology(
        input
            .primitive()
            .active_begin()
            .copied()
            .ok_or(MaxwellThreeDLoweringError::IncompleteDraw("BEGIN"))?,
    )?;
    if !matches!(
        topology,
        PrimitiveTopology::Points
            | PrimitiveTopology::Lines
            | PrimitiveTopology::Triangles
            | PrimitiveTopology::Patches
    ) {
        // Host strips enable restart implicitly; indexed quads/fans need index
        // assembly conversion. Neither may silently use a triangle-list draw.
        return Err(MaxwellThreeDLoweringError::UnsupportedIndexedDraw(
            "topology requires index assembly conversion",
        ));
    }
    if input.index().count().value() != Some(&index_count) {
        return Err(MaxwellThreeDLoweringError::TriggerStateMismatch);
    }
    let first_index = input.index().first().value().copied().ok_or(
        MaxwellThreeDLoweringError::IncompleteDraw("SET_INDEX_BUFFER_F"),
    )?;
    first_index.checked_add(index_count).ok_or(
        MaxwellThreeDLoweringError::UnsupportedIndexedDraw("index range overflow"),
    )?;
    let assembly = input.assembly();
    let base_vertex = assembly
        .global_base_vertex_index()
        .value()
        .copied()
        .unwrap_or(0);
    let vertex_id_base = assembly.vertex_id_base().value().copied().unwrap_or(0);
    // deko3d sets both bases from its signed vertex-offset parameter. Host
    // baseVertex biases both attribute fetch and vertex_index; distinct guest
    // bases require an independent shader adjustment, not rebasing the buffer.
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/draw.mme
    if base_vertex != vertex_id_base {
        return Err(MaxwellThreeDLoweringError::UnsupportedIndexedDraw(
            "distinct vertex-fetch and shader-ID bases",
        ));
    }
    if input.primitive().restart_enabled().value() == Some(&true) {
        return Err(MaxwellThreeDLoweringError::UnsupportedIndexedDraw(
            "primitive restart",
        ));
    }
    let base_instance = assembly
        .global_base_instance_index()
        .value()
        .copied()
        .unwrap_or(0);
    Ok(DrawArguments::Indexed {
        first_index,
        index_count,
        vertex_offset: base_vertex as i32,
        first_instance: neutral_first_instance(base_instance, input.primitive().instance_index())?,
        instance_count: 1,
    })
}
